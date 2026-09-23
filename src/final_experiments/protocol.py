import copy
import json
from pathlib import Path
import torch
import yaml
from src.experiments.metadata import file_hash,source_hashes,write_json
from src.experiments.phase3 import compact,verify_frozen_sources
from src.experiments.aggregate import validate_run as validate_historical
from src.final_experiments.data import C100_MEAN,C100_STD

ROOT=Path('outputs/final_experiments')


def inventory():
    files=[f for i in range(1,5) for f in Path(f'outputs/experiment_phase{i}').rglob('*') if f.is_file()]
    files += [f for i in range(1,5) for f in Path('outputs').glob(f'experiment_phase{i}_*') if f.is_file()]
    return {str(f):[f.stat().st_size,f.stat().st_mtime_ns] for f in files}


def old_entry(label,seed):
    phase=(2 if label=='D1' else 1) if seed<=3 else 3
    parent=Path(f'outputs/experiment_phase{phase}')
    manifest=json.loads((parent/'manifest.json').read_text())
    return parent,next(e for e in manifest['runs'] if e['model']==label and e['seed']==seed),manifest


def configs():
    entries=[]
    for dataset,labels,steps in [('CIFAR-100',['A','C','D1'],[3]),('CIFAR-10',['C','D1'],[1,2,5])]:
        for label in labels:
            for t in steps:
                for seed in [1,2,3]:
                    parent,old,_=old_entry(label,1);cfg=copy.deepcopy(old['config'])
                    if label!='A':cfg['model']['communication_iterations']=t
                    cfg['model']['num_classes']=100 if dataset=='CIFAR-100' else 10
                    name=f'{dataset.lower()}-{label}-T{t if label!="A" else 0}-seed{seed}'
                    cfg.update(dataset=dataset,seed=seed,experiment_name=name,output_root=str(ROOT.resolve()/'runs'))
                    entries.append(dict(id=name,model=label,seed=seed,T=t if label!='A' else 0,dataset=dataset,config=cfg))
    return entries


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text());check(manifest);return manifest
    verify_frozen_sources()
    refs=[];pins={}
    for label in ['A','B','C','D1']:
        for seed in ([1,2,3] if label=='B' else [1,2,3,4,5]):
            if label=='C' and seed==4:continue
            parent,e,m=old_entry(label,seed);run=parent/e['id']
            row=compact(validate_historical(run,{**e,'model':'D' if label=='D1' else label},False));row['model']=label
            row.update(config=e['config'],origin=str(run),status='completed')
            if json.loads((run/'environment.json').read_text())['source_hashes']!=m['source_hashes']:raise ValueError('Historical source mismatch')
            refs.append(row)
            for f in ['best.pt','summary.json','config.yaml','split.json','environment.json','test_results.json']:
                pins[str(run/f)]=file_hash(run/f)
    for file in ['outputs/experiment_phase4_summary.json','outputs/experiment_phase3/cifar10-C-cellular-t3-seed4/diagnostic_run1.json','outputs/experiment_phase3/cifar10-C-cellular-t3-seed4/diagnostic_run2.json']:
        pins[file]=file_hash(Path(file))
    manuscript=Path('cellular_manuscript/manuscript.tex')
    if manuscript.exists():pins[str(manuscript)]=file_hash(manuscript)
    manifest=dict(version=1,runs=configs(),references=refs,source_hashes=source_hashes(),historical_inventory=inventory(),pinned_files=pins,
        failed_reference=dict(model='C',seed=4,status='failed',accuracy=None,reason='deterministic numerical failure, epoch 1 batch/step 73'),
        cifar100_normalization=dict(mean=C100_MEAN,std=C100_STD),optional_extension='Seeds 4,5 only if all initial CIFAR-100 models/seeds complete; explicit --extend-cifar100 resource decision',
        corruption_source='https://zenodo.org/records/2535967',corruption_count=19,severities=[1,2,3,4,5],one_shot_alias='C T=1',
        no_post_hoc_tuning=True)
    write_json(path,manifest)
    for file in manifest['source_hashes']:
        target=ROOT/'source_snapshot'/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(file).read_bytes())
    Path('outputs/final_experiments_protocol.md').write_text('''# Final manuscript experiment protocol

## Audit and preservation

Model factories, frozen trainer, phase matrix/aggregation code, Phase 1–4 configs,
dataset loaders, deterministic seeding and failure records were inspected. Source
SHA-256, reference checkpoint/result hashes and historical file size/mtime inventory
are captured in final_experiments/manifest.json before execution. No old file is
edited. New code reuses the frozen training loop through isolated dependency bindings.
The C seed-4 failure remains categorical; it has no accuracy or corruption score.

## Models and training

A: CIFAR ResNet-18. B: shared pointwise iterative refinement, historical clean
reference. C: shared depthwise 3x3 local messages with channel normalization,
pointwise transform and residual lambda=1. D1: the exact C modules with detached,
normalized linear predictive-entropy gates. D2/D3/D4 remain ablations.

Training: 200 epochs, batch 128, SGD lr 0.1, momentum 0.9, weight decay 5e-4,
cosine schedule, existing initialization and crop/flip augmentation. Seeded 45k/5k
split; first best validation accuracy chooses a checkpoint; official test is scored
once afterwards. No AMP, clipping, warmup or per-model tuning. Set
CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts; retain existing deterministic
flags. Source/manifest hashes freeze the new implementation before real execution.

CIFAR-100: A/C/D1 T=3 (A has no iterations), seeds 1–3; only normalization
(mean .5071,.4867,.4408; std .2675,.2565,.2761) and class count change. Optional
seeds 4–5 require all nine initial runs successful and an explicit resource decision.
Never replace a failed seed. CIFAR-10 T ablation: C/D1 T=1,2,5 seeds 1–3; reuse T=3.
C-one-shot is mathematically identical to C T=1 and creates no extra training runs.
There are 27 new initial training runs. Numerical failures are terminal outcomes;
infrastructure interruptions remain pending with errors/logs and epoch resume.

## Corruption benchmark

Use only the canonical CIFAR-10-C archive from https://zenodo.org/records/2535967,
linked by https://github.com/hendrycks/robustness. Download verifies the record's
archive checksum. Evaluate all 19 distributed corruption types, including the four
extra types, at all five severities; distinguish the original 15-type subset in
secondary summaries. No corruption training. A/D1 seeds 1–5, C seeds 1,2,3,5.
Means are unnormalized accuracy/error, never called mCE. Equal weight per corruption
and severity within each seed, then summarize independent seed means.

## Metrics and comparisons

Accuracy, NLL, 15-bin ECE, summed multiclass Brier, predictive entropy (natural-log
units), changes across any refinement iteration, per-iteration gates and relative
effective update. Reuse the established uncertainty quartile analysis for D1.
Report successful counts, failures and pending runs. For successful multi-seed
results: mean, sample SD, Student-t 95% CI, median and range. Paired seed differences,
unadjusted paired t-tests/CIs and all seed differences; p>0.05 is not equivalence.
Corruptions/severities are not treated as independent seeds. C conditional means
never become five-seed means. Primary robustness: D1-A, C-A, D1-C. CIFAR-100 same
comparisons. Ablations compare T values within model and one-shot C against A,
C T=3 and D1 T=3. No test-based selection of T or subsequent tuning.

## Compute and output

One consistent PyTorch Conv2d/Linear forward-hook counter at 1x3x32x32 counts dense
MACs including repeated invocations; two arithmetic FLOPs per MAC. Normalization,
activations, pooling, residual adds and entropy/gate operations are excluded and
explicitly listed; partial arithmetic counts are not total model FLOPs. Use existing
matched-protocol GPU validation latency, no CPU latency substitution.
Figures use matplotlib defaults (no manually selected colors), 300-dpi PNG and
vector PDF. Tables are generated from validated records in booktabs format.
The manuscript is revised only after required real evidence is complete and its
source is available. Do not remove the development warning while evidence is pending.
No post-hoc tuning, replacement of failed seeds, or automatic next phase.
''')
    return manifest


def check(manifest):
    if inventory()!=manifest['historical_inventory']:raise ValueError('Historical Phase 1–4 inventory changed')
    if source_hashes()!=manifest['source_hashes']:raise ValueError('Final experiment source changed after freeze')
    for name,h in manifest['pinned_files'].items():
        if file_hash(Path(name))!=h:raise ValueError(f'Historical file changed: {name}')



def validate_source_provenance(run,manifest):
    env=json.loads((run/'environment.json').read_text())
    legacy=manifest.get('legacy_run_provenance',{}).get(run.name)
    expected=manifest['source_hashes'] if legacy is None else legacy['source_hashes']
    if env['source_hashes']!=expected:raise ValueError('New source provenance mismatch')
    if legacy is not None and file_hash(run/'environment.json')!=legacy['environment_sha256']:
        raise ValueError('Original environment record changed')


def validate(run,entry):
    summary=json.loads((run/'summary.json').read_text());cfg=entry['config']
    if yaml.safe_load((run/'config.yaml').read_text())!=cfg:raise ValueError('config mismatch')
    if summary['dataset']!=entry['dataset'] or summary['seed']!=entry['seed'] or summary['T']!=entry['T']:raise ValueError('run identity mismatch')
    if summary['samples']!=10000 or summary['epochs']!=200 or summary['test_evaluation_passes']!=1:raise ValueError('incomplete protocol')
    history=[json.loads(v) for v in (run/'metrics.jsonl').read_text().splitlines()]
    if [v['epoch'] for v in history]!=list(range(1,201)):raise ValueError('incomplete history')
    selected=max(history,key=lambda r:r['validation']['accuracy'])
    if selected['epoch']!=summary['best_val_epoch']:raise ValueError('wrong selection')
    if file_hash(run/'best.pt')!=summary['test_checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
    best=torch.load(run/'best.pt',weights_only=False,map_location='cpu')
    if best['config']!=cfg or best['epoch']+1!=summary['best_val_epoch']:raise ValueError('checkpoint provenance')
    split=json.loads((run/'split.json').read_text())
    if split!=summary['split']:raise ValueError('split provenance')
    import hashlib
    indices=torch.randperm(50000,generator=torch.Generator().manual_seed(entry['seed'])).tolist()
    for key,part in [('train_indices_sha256',indices[5000:]),('validation_indices_sha256',indices[:5000])]:
        if split[key]!=hashlib.sha256(json.dumps(part).encode()).hexdigest():raise ValueError('split mismatch')
    test=json.loads((run/'test_results.json').read_text())
    if test['checkpoint_sha256']!=summary['test_checkpoint_sha256'] or test['metrics']['accuracy']!=summary['test_accuracy']:raise ValueError('test provenance')
    row=dict(id=entry['id'],model=entry['model'],seed=entry['seed'],T=entry['T'],dataset=entry['dataset'],summary=summary,config=cfg,status='completed',analysis=test['analysis'])
    if row['analysis']:row['analysis'].pop('sample_records',None)
    return row
