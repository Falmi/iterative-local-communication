"""Residual-scale experiment; failed reference seeds are categorical outcomes."""
import copy
import json
from pathlib import Path
from src.experiments.phase3 import describe, paired_comparison, compact, verify_frozen_sources, validate_run as validate_old
from src.experiments.aggregate import validate_run as validate_existing
from src.experiments.metadata import file_hash, write_json

LABELS={'C-lambda025':.25,'C-lambda050':.5}
HISTORY=[Path(f'outputs/experiment_phase{i}') for i in [1,2,3]]


def matrix_configs(root,smoke=False):
    verify_frozen_sources()
    if any(p.resolve()==root.resolve() or p.resolve() in root.resolve().parents for p in HISTORY):
        raise ValueError('Cannot write Phase 4 into historical outputs')
    manifest=json.loads((HISTORY[0]/'manifest.json').read_text())
    prototype=next(e['config'] for e in manifest['runs'] if e['model']=='C')
    entries=[]
    for label,scale in LABELS.items():
        for seed in range(1,6):
            cfg=copy.deepcopy(prototype)
            name=f'cifar10-{label}-t3-seed{seed}'
            cfg.update(seed=seed,experiment_name=name,output_root=str(root))
            cfg['model']['residual_scale']=scale
            if smoke:
                cfg['training'].update(epochs=1,batch_size=4);cfg['device']='cpu';cfg['data']['workers']=0
            entries.append(dict(id=name,model=label,seed=seed,config=cfg))
    return entries


def historical_inventory():
    files=[f for p in HISTORY for f in p.rglob('*') if f.is_file()]
    files += [f for f in Path('outputs').glob('experiment_phase[123]_*') if f.is_file()]
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in files}


def references():
    rows=[];pins={}
    for label in ['A','C','D1']:
        for seed in range(1,6):
            if label=='C' and seed==4:continue
            phase=(2 if label=='D1' else 1) if seed<4 else 3
            parent=HISTORY[phase-1];manifest=json.loads((parent/'manifest.json').read_text())
            entry=next(e for e in manifest['runs'] if e['model']==label and e['seed']==seed)
            run=parent/entry['id'];row=compact(validate_old(run,entry,False))
            if label=='C':row['model']='C-lambda100'
            row['origin']=f'Phase {phase} reused unchanged'
            rows.append(row)
            for name in ['config.yaml','summary.json','environment.json','split.json','best.pt','refinement_analysis.json']:
                if (run/name).exists():pins[str(run/name)]=file_hash(run/name)
    failed=HISTORY[2]/'cifar10-C-cellular-t3-seed4'
    a,b=[json.loads((failed/f'diagnostic_run{i}.json').read_text()) for i in [1,2]]
    if a['status']!='nonfinite_detected' or a['failure']!=b['failure']:raise ValueError('Reference failure not verified')
    for name in ['status.json','diagnostic_run1.json','diagnostic_run2.json']:
        pins[str(failed/name)]=file_hash(failed/name)
    return dict(rows=rows,pinned_files=pins,lambda100_failure={'seed':4,'status':'deterministic_numerical_failure',
        'location':{k:a['failure'][k] for k in ['epoch','batch','global_step','first_nonfinite']},'accuracy':None,
        'monitoring_coverage':'Original full per-step stability maxima unavailable; two diagnostic runs supply failure evidence.'})


def validate_run(run,entry,smoke):
    row=compact(validate_existing(run,{**entry,'model':'C'},smoke));row['model']=entry['model']
    if row['summary']['residual_scale']!=LABELS[entry['model']]:raise ValueError('Residual scale mismatch')
    stability=json.loads((run/'stability.json').read_text())
    samples=8 if smoke else 45000
    batch=entry['config']['training']['batch_size']
    expected=row['summary']['epochs']*((samples+batch-1)//batch)
    if not stability['completed_successfully'] or stability.get('failure') or stability['steps']!=expected:raise ValueError('Incomplete stability monitoring')
    row['stability']=stability
    return row


def aggregate(root):
    manifest=json.loads((root/'manifest.json').read_text());smoke=manifest['smoke']
    checks=[];refs=manifest['references'];rows=list(refs['rows']);new=[];failures=[];pending=[]
    for p,h in refs['pinned_files'].items():
        if not Path(p).exists() or file_hash(Path(p))!=h:checks.append({'passed':False,'file':p,'reason':'historical checksum changed'})
    if not smoke and historical_inventory()!=manifest['historical_inventory']:
        checks.append({'passed':False,'reason':'historical inventory changed'})
    for e in manifest['runs']:
        run=root/e['id']
        try:
            if (run/'summary.json').exists():
                row=validate_run(run,e,smoke)
                if json.loads((run/'environment.json').read_text())['source_hashes']!=manifest['source_hashes']:raise ValueError('Source provenance mismatch')
                new.append(row);rows.append(row)
            elif (run/'stability.json').exists() and json.loads((run/'stability.json').read_text()).get('failure'):
                failures.append(dict(id=e['id'],model=e['model'],seed=e['seed'],stability=json.loads((run/'stability.json').read_text())))
            else:
                pending.append(dict(id=e['id'],model=e['model'],status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'not_started'}))
        except (ValueError,KeyError,OSError,RuntimeError) as error:checks.append(dict(passed=False,id=e['id'],reason=str(error)))
    for seed in range(1,6):
        group=[r for r in rows if r['seed']==seed]
        if len({r['model'] for r in group})!=len(group):raise ValueError('Duplicate model seed')
        if len({json.dumps(r['summary']['split'],sort_keys=True) for r in group})>1:checks.append(dict(passed=False,seed=seed,reason='split mismatch'))
    stability={};results={};dynamics={};outcomes={};pairs={}
    for label in [*LABELS,'C-lambda100','A','D1']:
        group=sorted([r for r in rows if r['model']==label],key=lambda r:r['seed'])
        failed=sum(r['model']==label for r in failures)+(1 if label=='C-lambda100' and not smoke else 0)
        started=sum((root/e['id']/'environment.json').exists() for e in manifest['runs'] if e['model']==label) if label in LABELS else len(group)+failed
        stability[label]=dict(planned=5,attempted=started,completed=len(group),failed=failed,pending=5-len(group)-failed,
                              failure_rate=failed/started if started else None,
                              monitoring='per-step Phase 4' if label in LABELS else 'historical completion/failure; per-step maxima unavailable')
        if not group:continue
        results[label]=dict(seeds=[r['seed'] for r in group],conditional_on_success=len(group)!=5,
            accuracy_pp=describe([100*r['summary']['test_accuracy'] for r in group]),
            **{k:describe([r['summary'][k] for r in group]) for k in ['NLL','ECE','Brier','training_time_seconds','inference_latency_ms_per_sample']})
        if label=='A':continue
        records=[];out=[]
        for r in group:
            a=r['analysis'];scale=r['summary']['residual_scale'];its=a['iterations']
            records.append(dict(seed=r['seed'],iterations=[dict(iteration=t,state_norm=its[t-1]['mean_feature_norm'],
                message_norm=its[t]['mean_message_norm'],relative_message=its[t]['mean_relative_message_magnitude'],
                effective_update_norm=its[t]['mean_gated_update_norm'] if label=='D1' else scale*its[t]['mean_message_norm'],
                relative_effective_update=its[t]['mean_relative_gated_update'] if label=='D1' else scale*its[t]['mean_relative_message_magnitude']) for t in range(1,4)]))
            o=a['outcomes'];out.append(dict(seed=r['seed'],**o,corrected_damaged_ratio=o['corrected']/o['damaged'] if o['damaged'] else None,
                prediction_change_frequency=1-a['never_changed_count']/a['samples'],
                iteration_accuracy_pp=[100*v['accuracy'] for v in its]))
        dynamics[label]=dict(per_seed=records,across_seeds=[{key:describe([r['iterations'][t][key] for r in records])
            for key in ['state_norm','message_norm','relative_message','effective_update_norm','relative_effective_update']} for t in range(3)],
            scope='Selected checkpoint, official test set; per-image norms averaged within seed, then summarized across seeds')
        outcomes[label]=dict(per_seed=out,iteration_accuracy_pp=[describe([r['iteration_accuracy_pp'][t] for r in out]) for t in range(4)],
            across_seeds={k:describe([r[k] for r in out if r[k] is not None]) for k in ['corrected','damaged','stable_correct','stable_wrong','corrected_damaged_ratio','prediction_change_frequency']})
    for label in LABELS:
        for other in ['A','D1']:
            pairs[f'{label}-{other}']=paired_comparison(rows,label,other) if stability[label]['completed']==5 and not checks else {'status':'pending five successful seeds and integrity checks'}
    complete=len(new)+len(failures)==10 and not checks
    result=dict(phase=4,status='complete' if complete else 'incomplete',evidence_type='synthetic_smoke' if smoke else 'real CIFAR-10',
        completed_runs=len(new),expected_runs=10,failed_new_runs=len(failures),stability_table=stability,
        reference_failure=refs.get('lambda100_failure'),per_seed_results=rows,accuracy_and_calibration=results,
        communication_dynamics=dynamics,refinement_outcomes=outcomes,paired_comparisons=pairs,
        numerical_failures=failures,pending_runs=pending,integrity_failures=checks,
        statistics_note='Accuracy in percentage points. Sample SD and unadjusted Student-t CI. Partial or four-seed means are conditional on successful runs; failures have no numeric accuracy. No equivalence margin was predefined.',
        timing_note='Phase 4 training time includes monitoring overhead; do not compare directly with historical unmonitored training time.')
    write_json(root.parent/(root.name+'_summary.json'),result)
    report(root.parent/(root.name+'_report.md'),result)
    return result


def report(path,r):
    lines=['# Phase 4: fixed communication strength','',f"Status: {r['status']}; {r['completed_runs']}/10 successful new runs, {r['failed_new_runs']} numerical failures. Evidence: {r['evidence_type']}.",'',
        'The only scientific variable is residual_scale (0.25 or 0.50); lambda=1.0 is reused, including its failed seed. No further experiment is implemented.',
        '', '| Condition | Planned | Attempted | Completed | Failed | Failure rate |','|---|---:|---:|---:|---:|---:|']
    for label,s in r['stability_table'].items():
        rate='pending' if s['failure_rate'] is None else f"{100*s['failure_rate']:.1f}%"
        lines.append(f"| {label} | 5 | {s['attempted']} | {s['completed']} | {s['failed']} | {rate} |")
    lines+=['','Lambda = 1.0: **4 successful runs / 5 attempted seeds, 1 deterministic numerical failure** (real reference only). Its seed 4 has no accuracy value.',
        '',r['statistics_note'],'',r['timing_note'],'','## Accuracy and calibration','','```json',json.dumps(r['accuracy_and_calibration'],indent=2),'```',
        '', '## Successful per-seed accuracies','','| Condition | Seed | Accuracy (%) |','|---|---:|---:|']
    for row in r['per_seed_results']:lines.append(f"| {row['model']} | {row['seed']} | {100*row['summary']['test_accuracy']:.4f} |")
    for title,key in [('Paired comparisons','paired_comparisons'),('Communication dynamics','communication_dynamics'),('Iterations and corrected/damaged analysis','refinement_outcomes'),('Failures','numerical_failures'),('Integrity failures','integrity_failures')]:
        lines+=['',f'## {title}','','```json',json.dumps(r[key],indent=2),'```']
    lines+=['','## Scientific questions','']
    questions=['Does reducing lambda eliminate the observed instability?','Is lambda=0.5 stable across all five seeds?','Is lambda=0.25 stable across all five seeds?',
        'What happens to effective update magnitude?','What happens to accuracy?','Does a stable variant outperform or match A?','Does a stable variant outperform D1?',
        'Was lambda=1 instability plausibly caused by excessive residual strength?','Which fixed strength should be retained?']
    for i,q in enumerate(questions,1):
        answer='Pending the ten real runs; no conclusion from synthetic or incomplete evidence.'
        if r['status']=='complete' and r['evidence_type']=='real CIFAR-10':
            a=r['stability_table'];stable=[k for k in LABELS if a[k]['completed']==5]
            if i in [1,2,3]:
                k='C-lambda050' if i==2 else 'C-lambda025'
                answer=(f"Stable across the five observed seeds: {stable}. This does not prove zero failure probability." if i==1 else f"{a[k]['completed']}/5 completed; {a[k]['failed']}/5 numerical failures.")
            elif i in [4,5,6,7]:answer='Use the measured per-iteration distributions, per-seed values, calibration, and paired differences above. A CI including zero does not establish equivalence; no equivalence margin was predefined.'
            elif i==8:answer='Stable lower scales would support residual strength as a contributor, not uniquely localize a mechanism; seed count is small and the sweep was motivated by an observed failure.'
            else:answer='If both lower scales fail, stop residual tuning and investigate the block. Prefer 0.5 only if stable and accuracy is maintained; if neither stable variant benefits A, stop optimizing C solely for CIFAR-10 accuracy. Do not implement a next experiment automatically.'
        lines.append(f'{i}. {q} {answer}')
    lines+=['','Stability spike thresholds are descriptive flags only: loss >100, total gradient norm >1e4, per-image relative effective update >10. They do not alter training or define a numeric accuracy for failures. Full historical per-step monitoring is unavailable; compare historical/fresh selected-checkpoint test dynamics on the same measurement basis. Training monitor statistics are separate.']
    path.write_text('\n'.join(lines)+'\n')
