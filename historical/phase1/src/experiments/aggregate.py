"""Aggregate only validated completed runs; report missing evidence explicitly."""
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch
import yaml
from src.experiments.metadata import file_hash, write_json
from src.experiments.protocol import PARAMETERS


def assert_finite(value):
    if isinstance(value,float) and not math.isfinite(value): raise ValueError('Non-finite metric')
    if isinstance(value,dict):
        for item in value.values(): assert_finite(item)
    if isinstance(value,list):
        for item in value: assert_finite(item)


def validate_run(run: Path, entry: dict, smoke: bool) -> dict:
    summary=json.loads((run/'summary.json').read_text())
    assert_finite(summary)
    if yaml.safe_load((run/'config.yaml').read_text())!=entry['config']: raise ValueError('Config mismatch')
    if summary['seed']!=entry['seed'] or summary['model_name']!=entry['config']['model']['name']: raise ValueError('Model/seed mismatch')
    if summary['parameters']!=PARAMETERS[entry['model']]: raise ValueError('Parameter count mismatch')
    if summary['dataset']!=('synthetic_smoke' if smoke else 'CIFAR-10'): raise ValueError('Dataset mismatch')
    if summary['T']!=(0 if entry['model']=='A' else 3): raise ValueError('Iteration mismatch')
    if summary['epochs']!=entry['config']['training']['epochs']: raise ValueError('Epoch mismatch')
    if summary['selection_metric']!='validation_accuracy' or summary['test_evaluation_passes']!=1: raise ValueError('Selection/test protocol mismatch')
    if summary['test_checkpoint_sha256']!=file_hash(run/'best.pt'): raise ValueError('Best checkpoint checksum mismatch')
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert_finite(history)
    if [r['epoch'] for r in history]!=list(range(1,summary['epochs']+1)): raise ValueError('Incomplete history')
    best=max(history,key=lambda r:r['validation']['accuracy'])
    if best['epoch']!=summary['best_val_epoch'] or best['validation']['accuracy']!=summary['best_val_accuracy']: raise ValueError('Incorrect best selection')
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    if checkpoint['epoch']+1!=summary['best_val_epoch'] or checkpoint['config']!=entry['config']: raise ValueError('Wrong checkpoint provenance')
    if checkpoint['best_val_accuracy']!=summary['best_val_accuracy']: raise ValueError('Checkpoint best metric mismatch')
    split=json.loads((run/'split.json').read_text())
    if split!=summary['split']: raise ValueError('Split metadata mismatch')
    expected=(8,8) if smoke else (45000,5000)
    if (split['train_samples'],split['validation_samples'])!=expected: raise ValueError('Split sizes mismatch')
    if not smoke:
        indices=torch.randperm(50000,generator=torch.Generator().manual_seed(entry['seed'])).tolist()
        for key, subset in [('train_indices_sha256',indices[5000:]),('validation_indices_sha256',indices[:5000])]:
            if split[key]!=hashlib.sha256(json.dumps(subset).encode()).hexdigest(): raise ValueError('Unexpected deterministic split')
    analysis=None
    if entry['model']!='A':
        analysis=json.loads((run/'refinement_analysis.json').read_text());assert_finite(analysis)
        if len(analysis['iterations'])!=4 or analysis['iterations'][-1]['accuracy']!=summary['test_accuracy']: raise ValueError('Analysis mismatch')
        if sum(analysis['outcomes'].values())!=summary['samples']: raise ValueError('Outcome count mismatch')
    if entry['model']=='D':
        validation=json.loads((run/'validation_analysis.json').read_text());assert_finite(validation)
        if len(validation['gate_evolution'])!=3: raise ValueError('Validation gate diagnostics missing')
    result=json.loads((run/'test_results.json').read_text());assert_finite(result)
    if result['checkpoint_sha256']!=summary['test_checkpoint_sha256'] or result['metrics']['accuracy']!=summary['test_accuracy']: raise ValueError('Test provenance mismatch')
    if summary['samples']!=(8 if smoke else 10000): raise ValueError('Test sample count mismatch')
    return {'id':entry['id'],'model':entry['model'],'seed':entry['seed'],'run_dir':str(run),
            'summary':summary,'analysis':analysis}


def stats(values):
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
            'std':statistics.stdev(values) if len(values)>1 else None}


def aggregate(root: Path) -> dict:
    manifest=json.loads((root/'manifest.json').read_text())
    complete=[]; incomplete=[]; integrity=[]
    for entry in manifest['runs']:
        run=root/entry['id']
        if not (run/'summary.json').exists():
            status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'not_started'}
            incomplete.append({'id':entry['id'],**status});continue
        try:
            item=validate_run(run,entry,manifest['smoke'])
            if json.loads((run/'environment.json').read_text())['source_hashes']!=manifest['source_hashes']: raise ValueError('Source version mismatch')
            complete.append(item)
            integrity.append({'id':entry['id'],'passed':True})
        except (ValueError,KeyError,OSError,RuntimeError) as error:
            integrity.append({'id':entry['id'],'passed':False,'error':str(error)})
    # Splits are intentionally seed-specific, but must match across A-D for each seed.
    bad_seeds=set()
    for seed in manifest['seeds']:
        items=[r for r in complete if r['seed']==seed]
        if len({json.dumps(r['summary']['split'],sort_keys=True) for r in items})>1:
            bad_seeds.add(seed);integrity.append({'seed':seed,'passed':False,'error':'Cross-model split mismatch'})
    complete=[r for r in complete if r['seed'] not in bad_seeds]
    fields=['test_accuracy','NLL','ECE','Brier','inference_latency_ms_per_sample','training_time_seconds']
    aggregates={}
    for label in 'ABCD':
        rows=[r for r in complete if r['model']==label]
        if not rows: continue
        aggregates[label]={'parameters':PARAMETERS[label],**{k:stats([r['summary'][k] for r in rows]) for k in fields}}
        if label!='A':
            aggregates[label]['final_minus_initial_accuracy']=stats([r['analysis']['iterations'][-1]['accuracy']-r['analysis']['iterations'][0]['accuracy'] for r in rows])
        if label=='D':
            aggregates[label]['gate_statistics_by_iteration']=[{key:stats([r['analysis']['gate_evolution'][t][key] for r in rows])
                for key in rows[0]['analysis']['gate_evolution'][t] if key!='iteration'} for t in range(3)]
    pairs={}
    for left,right,meaning in [('B','A','iterative computation'),('C','B','local communication; parameter counts differ'),('D','C','uncertainty gating')]:
        differences=[]
        for seed in manifest['seeds']:
            x=next((r for r in complete if r['model']==left and r['seed']==seed),None)
            y=next((r for r in complete if r['model']==right and r['seed']==seed),None)
            if x and y: differences.append({'seed':seed,'accuracy_difference':x['summary']['test_accuracy']-y['summary']['test_accuracy']})
        pairs[f'{left}-{right}']={'interpretation':meaning,'per_seed':differences,**stats([d['accuracy_difference'] for d in differences])}
    result={'status':'complete' if len(complete)==len(manifest['runs']) else 'incomplete',
        'evidence_type':'synthetic_smoke' if manifest['smoke'] else 'CIFAR-10 full experiment',
        'protocol':manifest,'per_seed_results':complete,'aggregated_results':aggregates,'paired_comparisons':pairs,
        'gate_diagnostics':{r['id']:r['analysis']['gate_evolution'] for r in complete if r['model']=='D'},
        'uncertainty_quartile_analysis':{r['id']:r['analysis']['uncertainty_groups'] for r in complete if r['model']=='D'},
        'incomplete_runs':incomplete,'integrity_checks':integrity,'completed_runs':len(complete),'expected_runs':len(manifest['runs'])}
    prefix=root.parent/root.name
    write_json(Path(str(prefix)+'_summary.json'),result)
    write_report(Path(str(prefix)+'_report.md'),result)
    return result


def write_report(path: Path, result: dict):
    def cell(s,percentage=False):
        if s['mean'] is None:return 'pending'
        scale=100 if percentage else 1
        return f"{s['mean']*scale:.4f} ± {s['std']*scale:.4f}" if s['std'] is not None else f"{s['mean']*scale:.4f} (one seed; SD unavailable)"
    protocol=result['protocol'];config=protocol['runs'][0]['config']
    lines=['# Experiment Phase 1 report','',f"Status: **{result['status']}**; {result['completed_runs']}/{result['expected_runs']} validated runs.",
           f"Evidence: **{result['evidence_type']}**. Synthetic smoke runs are not CIFAR-10 evidence.",'',
           '## Experimental protocol','', protocol.get('resource_note',''), 'A–D × seeds 1, 2, 3. Shared splits across models within each seed; partitions vary by seed.',
           'Validation accuracy selects the first best checkpoint. One scored test pass per selected checkpoint; no test-based tuning.',
           '## Model definitions','', 'A: ResNet-18. B: pointwise refinement. C: fixed local messages. D: detached normalized entropy gates. B/C/D use T=3, residual scale 1.',
           '## Training configuration','', '```json',json.dumps(config['training'],indent=2),'```',
           'SGD, cosine annealing, deterministic seeds; same augmentation, normalization and loader settings across models.',
           '## Per-seed results','', '| Model | Seed | Test accuracy (%) | Best validation epoch |','|---|---:|---:|---:|']
    for r in result['per_seed_results']:lines.append(f"| {r['model']} | {r['seed']} | {r['summary']['test_accuracy']*100:.4f} | {r['summary']['best_val_epoch']} |")
    if not result['per_seed_results']:lines.append('No completed runs. No result values are inferred.')
    lines+=['','## Aggregated results','','Sample standard deviation across completed seeds; incomplete matrices must not be interpreted as final comparisons.',
            '| Model | Params | Accuracy % mean±SD | NLL | ECE | Brier | Latency ms/sample | Training seconds |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,a in result['aggregated_results'].items():
        lines.append(f"| {label} | {a['parameters']} | {cell(a['test_accuracy'],True)} | {cell(a['NLL'])} | {cell(a['ECE'])} | {cell(a['Brier'])} | {cell(a['inference_latency_ms_per_sample'])} | {cell(a['training_time_seconds'])} |")
    lines+=['','## Scientific comparison table','','| Comparison | Paired accuracy difference (percentage points), mean±SD | Matched seeds |',
            '|---|---:|---:|']
    for pair,p in result['paired_comparisons'].items():
        lines.append(f"| {pair} | {cell(p,True)} | {p['n']} |")
    for pair,title in [('B-A' ,'A vs B'),('C-B','B vs C'),('D-C','C vs D')]:
        p=result['paired_comparisons'][pair]
        lines+=['',f'## {title}','',f"{pair}: {cell(p,True)} percentage points; {p['n']} matching seeds. Interpretation: {p['interpretation']}.",
                'Paired differences are descriptive; three seeds do not justify strong significance claims.']
    lines+=['','## Calibration','','NLL, ECE and Brier are reported above (lower is better); conclusions await completed real-data results.',
            '## Iteration behaviour','','Per-run t0–t3 accuracy, confidence, entropy, change rates and update magnitudes:']
    for r in result['per_seed_results']:
        if r['analysis']:
            lines+=['',f"### {r['id']}",'','```json',json.dumps(r['analysis']['iterations'],indent=2),'```']
    lines+=['','## Model D gate behaviour','','Gate quantiles and threshold fractions are diagnostics, not success criteria; no mechanism is changed based on them.']
    for name,gates in result['gate_diagnostics'].items():lines+=['',name,'','```json',json.dumps(gates,indent=2),'```']
    lines+=['','## Uncertainty quartiles','','Initial-entropy rank quartiles; rates use all group samples. Higher/lower benefit can be read from accuracy_change; no benefit is assumed.']
    for name,groups in result['uncertainty_quartile_analysis'].items():lines+=['',name,'','```json',json.dumps(groups,indent=2),'```']
    lines+=['','## Computational cost','','C and D have 11,442,762 parameters; B has 11,700,298; A has 11,173,962. Latency uses the same validation-set final-only forward timing for A–D. Training time includes training and validation, excludes checkpoint IO and final evaluation.',
            '## Integrity and pending work','', '```json',json.dumps({'incomplete':result['incomplete_runs'],'checks':result['integrity_checks']},indent=2),'```',
            '## Limitations','','CIFAR-10 only; 3 seeds only; no corruption/OOD evaluation; no parameter-matched B/C control; no T sweep. No strong significance claim.',
            '## Recommended next experiment','','Complete and inspect this fixed real-data matrix first. No next experimental change is recommended from missing or synthetic evidence.']
    path.write_text('\n'.join(lines)+'\n')
