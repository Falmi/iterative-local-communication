"""Shared trainer with epoch-boundary resume and best-validation test evaluation."""
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from src.datasets.cifar import make_loader
from src.models.factory import create_model
from src.evaluation.evaluator import evaluate, synchronize
from src.utils.checkpoint import save_checkpoint
from src.utils.config import save_config
from src.utils.seed import seed_everything, resolve_device
from src.analysis.refinement_analysis import analyze_refinement
from src.experiments.metadata import write_json, environment, split_metadata, file_hash


def train(config: dict, smoke: bool = False, run_dir: Path | None = None,
          resume: bool = False, stop_after_epoch: int | None = None) -> Path:
    """Resume only this exact config, from a completed epoch (never mid-batch).

    stop_after_epoch provides an orderly pause for verification or manual use;
    it does not change the configured cosine schedule or full epoch count.
    Test evaluation is committed once. An interrupted test pass is marked for
    manual review rather than silently repeated.
    """
    seed_everything(config['seed'], config['deterministic'])
    device = resolve_device(config['device'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = Path(run_dir) if run_dir else Path(config['output_root']) / f'{config["experiment_name"]}-seed{config["seed"]}-{stamp}'
    if resume:
        saved_config = yaml.safe_load((run/'config.yaml').read_text())
        if saved_config != config:
            raise ValueError('Resume configuration differs from saved run')
        if (run/'summary.json').exists():
            return run
    else:
        run.mkdir(parents=True, exist_ok=False)
        save_config(config, run/'config.yaml')
        meta = environment(device, smoke)
        meta['timestamp_utc'] = stamp
        write_json(run/'environment.json', meta)
    training = make_loader(config,'train',smoke)
    validation = make_loader(config,'validation',smoke)
    split = split_metadata(training, validation, config['seed'], smoke)
    if resume and (run/'split.json').exists() and json.loads((run/'split.json').read_text()) != split:
        raise ValueError('Data split changed')
    write_json(run/'split.json', split)
    model = create_model(config['model']).to(device)
    t = config['training']
    optimizer = torch.optim.SGD(model.parameters(), lr=t['learning_rate'], momentum=t['momentum'], weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t['epochs'])
    best, best_epoch, history, elapsed, start_epoch = -1., 0, [], 0., 0
    if resume and (run/'last.pt').exists():
        saved = torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if not saved.get('training_state'):
            raise ValueError('Legacy checkpoint lacks resumable training history')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start_epoch = saved['epoch']+1
        best = saved['best_val_accuracy']
        state = saved['training_state']
        history, elapsed, best_epoch = state['history'], state['elapsed_seconds'], state['best_epoch']
        random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        training.generator.set_state(saved['rng']['loader'])
    write_json(run/'status.json',{'status':'training','completed_epochs':start_epoch})
    for epoch in range(start_epoch,t['epochs']):
        synchronize(device); started = time.perf_counter()
        model.train()
        loss_sum, correct, count = 0., 0, 0
        lr = optimizer.param_groups[0]['lr']
        for images,targets in training:
            images,targets=images.to(device),targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=F.cross_entropy(logits,targets)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(targets)
            correct += logits.detach().argmax(1).eq(targets).sum().item()
            count += len(targets)
        metrics=evaluate(model,validation,device)
        scheduler.step()
        improved=metrics['accuracy']>best
        if improved: best,best_epoch=metrics['accuracy'],epoch+1
        record={'epoch':epoch+1,'learning_rate':lr,'train_loss':loss_sum/count,
                'train_accuracy':correct/count,'validation':metrics}
        history.append(record)
        synchronize(device); elapsed += time.perf_counter()-started
        state={'history':history,'elapsed_seconds':elapsed,'best_epoch':best_epoch}
        # Best first: last checkpoint is the committed epoch boundary.
        if improved: save_checkpoint(run/'best.pt',model,optimizer,scheduler,epoch,best,config,training,state)
        save_checkpoint(run/'last.pt',model,optimizer,scheduler,epoch,best,config,training,state)
        temp=run/'metrics.jsonl.tmp'
        temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
        temp.replace(run/'metrics.jsonl')
        write_json(run/'status.json',{'status':'training','completed_epochs':epoch+1})
        print(json.dumps(record),flush=True)
        if stop_after_epoch is not None and epoch+1>=stop_after_epoch:
            write_json(run/'status.json',{'status':'paused','completed_epochs':epoch+1})
            return run
    # Recover a history file if interrupted after the last checkpoint commit.
    (run/'metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
    best_checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_checkpoint['model'])
    checkpoint_hash=file_hash(run/'best.pt')
    recurrent=config['model']['name']!='resnet18_cifar'
    guided=config['model']['name']=='uncertainty_cellular_resnet18'
    # Same final-only inference timing procedure and validation dataset for A-D.
    latency=evaluate(model,validation,device)['inference_latency_ms_per_sample']
    if guided:
        val_analysis=analyze_refinement(model,validation,device,feature_analysis=True,message_analysis=True)
        write_json(run/'validation_analysis.json',val_analysis)
    result_path=run/'test_results.json'
    journal=run/'test_evaluation_started.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result['checkpoint_sha256']!=checkpoint_hash: raise ValueError('Test results reference a different checkpoint')
    else:
        if journal.exists():
            raise RuntimeError('Interrupted test evaluation: manual review required; will not silently repeat test pass')
        write_json(journal,{'checkpoint_sha256':checkpoint_hash,'best_epoch':best_epoch})
        test_loader=make_loader(config,'test',smoke)
        if recurrent:
            analysis=analyze_refinement(model,test_loader,device,feature_analysis=True,
                       message_analysis=config['model']['name']!='iterative_resnet18',collect_samples=guided)
            test=analysis['final_metrics']
        else:
            analysis=None
            test=evaluate(model,test_loader,device)
        result={'checkpoint_sha256':checkpoint_hash,'best_epoch':best_epoch,'metrics':test,'analysis':analysis}
        write_json(result_path,result)
    if result['analysis'] is not None:
        write_json(run/'refinement_analysis.json',result['analysis'])
    m=config['model']; test=result['metrics']
    summary={'experiment_name':config['experiment_name'],'model_name':m['name'],'dataset':split['dataset'],
        'seed':config['seed'],'epochs':t['epochs'],'T':m.get('communication_iterations',m.get('refinement_iterations',0)),
        'residual_scale':m.get('residual_scale'),'parameters':sum(p.numel() for p in model.parameters()),'FLOPs':None,
        'best_val_accuracy':best,'best_val_epoch':best_epoch,'test_accuracy':test['accuracy'],
        'training_time_seconds':elapsed,'training_time_scope':'training and validation; excludes checkpoint IO and final evaluation',
        **{k:v for k,v in test.items() if k not in {'accuracy','inference_latency_ms_per_sample'}},
        'inference_latency_ms_per_sample':latency,'latency_dataset':'validation','latency_path':'final_only',
        'selection_metric':'validation_accuracy','test_checkpoint_sha256':checkpoint_hash,'test_evaluation_passes':1,
        'split':split, **{k:m.get(k) for k in ['communication_iterations','share_communication_weights',
            'share_refinement_weights','neighbourhood_kernel_size','communication_type','uncertainty_type',
            'gate_mode','detach_uncertainty_gate','normalize_uncertainty','fixed_gate_value']}}
    if recurrent: summary['refinement_analysis_file']='refinement_analysis.json'
    write_json(run/'summary.json',summary)
    write_json(run/'status.json',{'status':'completed','completed_epochs':t['epochs']})
    print(f'Run saved to {run}',flush=True)
    return run
