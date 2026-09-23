import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.final_experiments.data import CORRUPTIONS,CorruptionDataset
from src.models.factory import create_model
from src.models.uncertainty import predictive_entropy_from_logits
from src.analysis.refinement_analysis import analyze_refinement
from src.evaluation.metrics import ClassificationMetrics
from src.experiments.metadata import file_hash,write_json
from src.utils.seed import seed_everything


@torch.inference_mode()
def measure(model,loader,label):
    if label!='A':
        a=analyze_refinement(model,loader,torch.device('cuda'),feature_analysis=True,message_analysis=True)
        iterations=a['iterations']
        return {**a['final_metrics'],'predictive_entropy':iterations[-1]['mean_entropy'],
            'initial_predictive_entropy':iterations[0]['mean_entropy'],
            'prediction_change_rate':1-a['never_changed_count']/a['samples'],
            'mean_gate':sum(v['mean_gate'] for v in a['gate_evolution'])/len(a['gate_evolution']) if label=='D1' else None,
            'mean_relative_effective_update':sum(v['mean_relative_gated_update'] if label=='D1' else model.residual_scale*v['mean_relative_message_magnitude'] for v in iterations[1:])/(len(iterations)-1),
            'iterations':iterations}
    metrics=ClassificationMetrics();entropy=0.
    for images,targets in loader:
        logits=model(images.cuda())
        if not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite corruption logits')
        metrics.update(logits,targets);entropy+=predictive_entropy_from_logits(logits,normalized=False).double().sum().item()
    return {**metrics.compute(),'predictive_entropy':entropy/metrics.n,'initial_predictive_entropy':entropy/metrics.n,
            'prediction_change_rate':None,'mean_gate':None,'mean_relative_effective_update':None}


def execute(manifest,data_root):
    out=Path('outputs/final_cifar10c/cells');out.mkdir(parents=True,exist_ok=True)
    data_root=Path(data_root)
    missing=[c for c in [*CORRUPTIONS,'labels'] if not (data_root/(c+'.npy')).exists()]
    if missing:raise FileNotFoundError(f'Canonical CIFAR-10-C files missing: {missing}')
    provenance={c:file_hash(data_root/(c+'.npy')) for c in [*CORRUPTIONS,'labels']}
    path=out.parent/'dataset_hashes.json'
    if path.exists() and json.loads(path.read_text())!=provenance:raise ValueError('CIFAR-10-C content changed')
    write_json(path,provenance)
    for row in manifest['references']:
        if row['model'] not in ['A','C','D1']:continue
        seed_everything(row['seed'],True);checkpoint=Path(row['origin'])/'best.pt';digest=file_hash(checkpoint)
        model=create_model(row['config']['model']).cuda()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
        for corruption in CORRUPTIONS:
            for severity in range(1,6):
                dest=out/f"{row['model']}-seed{row['seed']}-{corruption}-{severity}.json"
                if dest.exists():
                    saved=json.loads(dest.read_text())
                    if saved['checkpoint_sha256']!=digest or saved['dataset_sha256']!=provenance[corruption]:raise ValueError('Robustness provenance mismatch')
                    if saved['status'] in ['completed','failed']:continue
                identity=dict(model=row['model'],seed=row['seed'],corruption=corruption,severity=severity,checkpoint_sha256=digest,
                    dataset_sha256=provenance[corruption],clean_accuracy=row['summary']['test_accuracy'])
                try:
                    loader=DataLoader(CorruptionDataset(data_root,corruption,severity),batch_size=128,shuffle=False,num_workers=0)
                    result=measure(model,loader,row['model'])
                    write_json(dest,{**identity,'status':'completed',**result})
                except Exception as error:
                    write_json(dest,{**identity,'status':'failed','error':repr(error)})
                    raise
                print(dest.name,result['accuracy'],flush=True)
        del model
