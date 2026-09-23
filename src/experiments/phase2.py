"""Gate-mapping ablation: reuse Phase 1 controls and the existing trainer/audit."""
import ast
import copy
import json
from pathlib import Path
from src.experiments.aggregate import validate_run as validate_existing_run, stats, assert_finite
from src.experiments.metadata import file_hash, write_json
from src.utils.config import load_config

REFERENCE_ROOT = Path('outputs/experiment_phase1')
MAPPINGS = {'D1':'linear','D2':'sqrt','D3':'cuberoot','D4':'floor_linear'}
FIELDS = ['test_accuracy','NLL','ECE','Brier','best_val_accuracy','best_val_epoch',
          'training_time_seconds','inference_latency_ms_per_sample']


def verify_scientific_sources() -> None:
    """Communication, entropy, backbone, training and evaluation remain fixed."""
    snapshot=REFERENCE_ROOT/'source_snapshot'
    names=['src/models/baseline.py','src/models/refinement.py','src/models/iterative_classifier.py',
           'src/models/cellular_communication.py','src/models/cellular_classifier.py',
           'src/datasets/cifar.py','src/training/trainer.py','src/utils/seed.py','src/evaluation/evaluator.py',
           'src/evaluation/metrics.py']
    for name in names:
        if file_hash(Path(name))!=file_hash(snapshot/name): raise ValueError(f'Changed scientific source: {name}')
    def entropy_ast(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predictive_entropy_from_logits'))
    name='src/models/uncertainty.py'
    if entropy_ast(Path(name))!=entropy_ast(snapshot/name): raise ValueError('Entropy calculation changed')


def matrix_configs(root: Path, smoke=False, seeds=(1,2,3)) -> list[dict]:
    if root.resolve()==REFERENCE_ROOT.resolve() or REFERENCE_ROOT.resolve() in root.resolve().parents:
        raise ValueError('Phase 2 cannot write into Phase 1')
    verify_scientific_sources()
    phase1=json.loads((REFERENCE_ROOT/'manifest.json').read_text())
    entries=[]
    for label,mode in MAPPINGS.items():
        template=load_config(f'configs/phase2_{mode}.yaml')
        for seed in seeds:
            reference=next(r['config'] for r in phase1['runs'] if r['model']=='D' and r['seed']==seed)
            for key in ('training','data','device','deterministic'):
                if template[key]!=reference[key]: raise ValueError(f'Phase 1 protocol changed: {key}')
            stripped={k:v for k,v in template['model'].items() if k not in {'gate_mapping','minimum_gate'}}
            if stripped!=reference['model']: raise ValueError('Only gate_mapping may differ from Phase 1 D')
            if template['model']['gate_mapping']!=mode or template['model']['minimum_gate']!=.1:
                raise ValueError('Incorrect mapping configuration')
            if not template['model']['detach_uncertainty_gate']: raise ValueError('Phase 2 requires detached gates')
            c=copy.deepcopy(template)
            name=f'cifar10-{label}-{"floor01" if label=="D4" else mode}-t3-seed{seed}'
            c.update(seed=seed,experiment_name=name,output_root=str(root))
            if smoke:
                c['training'].update(epochs=1,batch_size=4);c['device']='cpu';c['data']['workers']=0
            entries.append({'id':name,'model':label,'seed':seed,'gate_mapping':mode,'config':c})
    return entries


def manifest_metadata(smoke: bool) -> dict:
    """Validate controls once and pin files for future read-only reuse."""
    if smoke: return {'phase':2,'references':[],'reference_policy':'No real controls mixed into synthetic smoke'}
    phase1=json.loads((REFERENCE_ROOT/'manifest.json').read_text())
    references=[]
    for entry in phase1['runs']:
        if entry['model'] not in {'A','C'}: continue
        run=REFERENCE_ROOT/entry['id']
        row=validate_existing_run(run,entry,False)
        metadata=json.loads((run/'environment.json').read_text())
        if metadata['source_hashes']!=phase1['source_hashes']: raise ValueError('Phase 1 source provenance mismatch')
        paths=[run/name for name in ['summary.json','config.yaml','best.pt','split.json','environment.json']]
        if entry['model']=='C': paths.append(run/'refinement_analysis.json')
        row['pinned_files']={str(p.resolve()):file_hash(p) for p in paths}
        row['origin']='Phase 1 reused; no retraining or retesting'
        references.append(row)
    return {'phase':2,'references':references,'reference_policy':'Validated Phase 1 A context and C primary control, identical protocol'}


def validate_run(run: Path, entry: dict, smoke: bool) -> dict:
    result=validate_existing_run(run,{**entry,'model':'D'},smoke)
    result['model']=entry['model'];result['gate_mapping']=entry['gate_mapping']
    a=result['analysis'];m=entry['config']['model']
    if a.get('gate_mapping')!=entry['gate_mapping'] or not a['detach_uncertainty_gate']:
        raise ValueError('Gate mapping/detach mismatch')
    if m['communication_iterations']!=3 or m['residual_scale']!=1 or not m['share_communication_weights']:
        raise ValueError('Uncontrolled iteration/scale/sharing')
    for t in range(3):
        gates=a['gate_evolution'][t]
        if not 0<=gates['min_gate']<=gates['max_gate']<=1: raise ValueError('Invalid gate bounds')
        for key in ['fraction_below_0_10','fraction_above_0_25','q25_gate','q75_gate']:
            if key not in gates: raise ValueError('Missing gate diagnostic')
    if not smoke:
        parent=json.loads((REFERENCE_ROOT/f'cifar10-C-cellular-t3-seed{entry["seed"]}'/'split.json').read_text())
        if result['summary']['split']!=parent: raise ValueError('Phase 1/2 split mismatch')
    return result


def paired(rows, left, right):
    values=[]
    for seed in (1,2,3):
        a=next((r for r in rows if r['model']==left and r['seed']==seed),None)
        b=next((r for r in rows if r['model']==right and r['seed']==seed),None)
        if a and b: values.append({'seed':seed,'accuracy_difference':a['summary']['test_accuracy']-b['summary']['test_accuracy']})
    return {**stats([v['accuracy_difference'] for v in values]),'per_seed':values}


def aggregate(root: Path) -> dict:
    manifest=json.loads((root/'manifest.json').read_text())
    references=manifest['references'];checks=[];rows=[];pending=[]
    for reference in references:
        valid=all(file_hash(Path(p))==digest for p,digest in reference['pinned_files'].items())
        checks.append({'id':reference['id'],'passed':valid,'check':'Phase 1 pinned control unchanged'})
        if valid: rows.append(reference)
    completed=[]
    for entry in manifest['runs']:
        run=root/entry['id']
        if not (run/'summary.json').exists():
            status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'not_started'}
            pending.append({'id':entry['id'],**status});continue
        try:
            row=validate_run(run,entry,manifest['smoke'])
            if json.loads((run/'environment.json').read_text())['source_hashes']!=manifest['source_hashes']:
                raise ValueError('Phase 2 source fingerprint mismatch')
            completed.append(row);checks.append({'id':entry['id'],'passed':True})
        except (ValueError,KeyError,OSError,RuntimeError) as error:
            checks.append({'id':entry['id'],'passed':False,'error':str(error)})
    rows+=completed
    aggregates={}
    for label in ['A','C',*MAPPINGS]:
        group=[r for r in rows if r['model']==label]
        if not group: continue
        aggregates[label]={'parameters':group[0]['summary']['parameters'],
                           **{k:stats([r['summary'][k] for r in group]) for k in FIELDS}}
        if label in MAPPINGS:
            aggregates[label]['corrected_damaged_ratio']=stats([r['analysis']['corrected_damaged_ratio'] for r in group])
            aggregates[label]['gate_evolution']=[{key:stats([r['analysis']['gate_evolution'][t][key] for r in group])
                for key in group[0]['analysis']['gate_evolution'][t] if key!='iteration'} for t in range(3)]
    pairs={f'{left}-{right}':paired(rows,left,right) for left in ['D2','D3','D4'] for right in ['D1','C']}
    result={'phase':2,'status':'complete' if len(completed)==12 and all(c['passed'] for c in checks) else 'incomplete',
        'evidence_type':'synthetic_smoke' if manifest['smoke'] else 'real CIFAR-10',
        'completed_runs':len(completed),'expected_runs':12,'protocol':manifest,
        'per_seed_results':rows,'aggregated_results':aggregates,'paired_comparisons':pairs,
        'gate_diagnostics':{r['id']:r['analysis']['gate_evolution'] for r in completed},
        'uncertainty_quartiles':{r['id']:r['analysis']['uncertainty_groups'] for r in completed},
        'effective_updates':{r['id']:[{k:v for k,v in t.items() if 'update' in k or 'message' in k or k=='iteration'}
                                  for t in r['analysis']['iterations']] for r in completed},
        'outcome_diagnostics':{r['id']:{'counts':r['analysis']['outcomes'],'rates':r['analysis']['outcome_rates'],
                            'corrected_damaged_ratio':r['analysis']['corrected_damaged_ratio']} for r in completed},
        'integrity_checks':checks,'incomplete_runs':pending}
    assert_finite(result)
    write_json(root.parent/(root.name+'_summary.json'),result)
    write_report(root.parent/(root.name+'_report.md'),result)
    return result


def write_report(path: Path, result: dict):
    def fmt(s, scale=1):
        if s['mean'] is None: return 'pending'
        sd=f"{s['std']*scale:.4f}" if s['std'] is not None else 'SD unavailable'
        return f"{s['mean']*scale:.4f} ± {sd}"
    lines=['# Experiment Phase 2: uncertainty-to-gate mapping','',
        f"Status: **{result['status']}**, {result['completed_runs']}/12 Phase 2 runs. Evidence type: {result['evidence_type']}.",'',
        '## Motivation','',
        'Completed Phase 1 suggests uncertainty identifies difficult samples, but linear gates often suppress messages. Phase 2 tests only the uncertainty-to-strength mapping.',
        '', '## Gate mappings','', 'D1: g=u. D2: g=sqrt(u). D3: g=u^(1/3). D4: g=0.1+0.9u. C: g=1 (unchanged Phase 1 reference).',
        'All uncertainty is detached before mapping. Parameters: 11,442,762 for C/D1–D4. T=3, lambda=1. No learned gate or other architectural change.',
        '', '## Main results','',
        'Accuracy is percent; SD is sample SD across available seeds. Phase 1 controls are reused only after provenance/protocol validation. Partial results are not final evidence.',
        '', '| Gate | Seeds | Accuracy mean±SD | NLL | ECE | Brier | Latency ms/sample | Training s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ['C',*MAPPINGS]:
        a=result['aggregated_results'].get(label)
        if a: lines.append(f"| {label} | {a['test_accuracy']['n']} | {fmt(a['test_accuracy'],100)} | {fmt(a['NLL'])} | {fmt(a['ECE'])} | {fmt(a['Brier'])} | {fmt(a['inference_latency_ms_per_sample'])} | {fmt(a['training_time_seconds'])} |")
    if 'A' in result['aggregated_results']:
        lines+=['',f"A baseline context (Phase 1): {fmt(result['aggregated_results']['A']['test_accuracy'],100)}%."]
    lines+=['','## Per-seed results','','| Variant | Seed | Accuracy % | Best epoch |','|---|---:|---:|---:|']
    for row in result['per_seed_results']:
        lines.append(f"| {row['model']} | {row['seed']} | {row['summary']['test_accuracy']*100:.4f} | {row['summary']['best_val_epoch']} |")
    lines+=['','## Comparisons with D1 and fixed communication C','','Paired percentage-point differences; three seeds are exploratory, not strong significance evidence.','',
            '| Comparison | Matching seeds | Mean ± SD (pp) |','|---|---:|---:|']
    for name,pair in result['paired_comparisons'].items(): lines.append(f"| {name} | {pair['n']} | {fmt(pair,100)} |")
    for title,key in [('Gate distributions','gate_diagnostics'),('Effective message and update magnitudes','effective_updates'),
                      ('Uncertainty quartiles: Q1–Q3 damage versus Q4 corrections','uncertainty_quartiles'),('Corrected versus damaged','outcome_diagnostics')]:
        lines+=['',f'## {title}','']
        if not result[key]: lines.append('Pending Phase 2 measurements.')
        for name,value in result[key].items(): lines+=['',f'### {name}','','```json',json.dumps(value,indent=2),'```']
    lines+=['','## Calibration and computational cost','',
            'NLL, ECE and Brier (lower is better) must be considered alongside accuracy. Latency uses final-only validation inference, excluding detailed analysis. Training time excludes checkpoint IO. Gates do not skip computation.',
            '', '## Conclusion and recommended next experiment','']
    if result['status']!='complete' or result['evidence_type']=='synthetic_smoke':
        lines.append('No complete real Phase 2 evidence yet: nonlinear benefit, best mapping, improvement over C, and confident-sample damage are unresolved. Complete this frozen matrix before choosing a next experiment. Synthetic smoke is execution verification only.')
    else:
        best=max(MAPPINGS,key=lambda k:result['aggregated_results'][k]['test_accuracy']['mean'])
        lines.append(f"Highest mean Phase 2 accuracy: {best}, {fmt(result['aggregated_results'][best]['test_accuracy'],100)}%. This ranks observed accuracy only; inspect calibration and quartile trade-offs before selecting a mapping.")
        c=result['aggregated_results']['C']['test_accuracy']['mean']
        for label in MAPPINGS:
            group=[r for r in result['per_seed_results'] if r['model']==label]
            low_damage=stats([sum(r['analysis']['uncertainty_groups'][q]['damaged_count'] for q in ['q1_lowest','q2','q3']) for r in group])
            lines.append(f"{label}: mean accuracy minus C = {(result['aggregated_results'][label]['test_accuracy']['mean']-c)*100:+.4f} pp; mean Q1–Q3 damaged count = {fmt(low_damage)}.")
        lines.append('Recommended next experiment: confirm the observed accuracy/calibration/quartile trade-off with additional predeclared seeds before adding any mechanism. The causal explanation for Phase 1 cannot be established by gate amplification alone.')
    lines+=['','## Integrity checks','','```json',json.dumps(result['integrity_checks'],indent=2),'```',
            '', 'No Phase 1 files are written. Quartiles use the same rank/tie procedure as Phase 1, but membership is recomputed per trained model; these are not identical sample cohorts. All gates and diagnostic fractions use [0,1].']
    path.write_text('\n'.join(lines)+'\n')
