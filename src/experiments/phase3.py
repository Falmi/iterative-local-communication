"""Confirmatory seed extension; scientific modules and historical outputs stay frozen."""
import copy
import json
import math
import statistics
from pathlib import Path
from scipy import stats as scipy_stats
from src.experiments.aggregate import validate_run as validate_existing, assert_finite
from src.experiments.metadata import file_hash, write_json

ROOTS={1:Path('outputs/experiment_phase1'),2:Path('outputs/experiment_phase2')}
MODELS={'A':(1,'baseline'),'C':(1,'cellular-t3'),'D1':(2,'linear-t3'),'D2':(2,'sqrt-t3')}
PAIRS=[('C','A'),('D1','C'),('D2','C'),('D2','D1')]
PARAMETERS={'A':11173962,'C':11442762,'D1':11442762,'D2':11442762}


def verify_frozen_sources():
    manifest=json.loads((ROOTS[2]/'manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():
        if name.startswith(('src/models/','src/training/','src/datasets/','src/evaluation/','src/analysis/','src/utils/','configs/')):
            if file_hash(Path(name))!=digest: raise ValueError(f'Frozen scientific file changed: {name}')


def source_entries():
    manifests={p:json.loads((r/'manifest.json').read_text()) for p,r in ROOTS.items()}
    return {label:[r for r in manifests[phase]['runs'] if r['model']==label] for label,(phase,_) in MODELS.items()}


def matrix_configs(root: Path, smoke=False, seeds=(4,5)) -> list[dict]:
    if tuple(seeds)!=(4,5): raise ValueError('Phase 3 runs exactly seeds 4 and 5')
    for parent in ROOTS.values():
        if root.resolve()==parent.resolve() or parent.resolve() in root.resolve().parents:
            raise ValueError('Phase 3 cannot write into Phase 1/2')
    verify_frozen_sources()
    originals=source_entries()
    common=originals['A'][0]['config']
    result=[]
    for label,(_,suffix) in MODELS.items():
        prototype=originals[label][0]['config']
        for key in ['training','data','device','deterministic']:
            if prototype[key]!=common[key]: raise ValueError(f'Historical protocol differs: {label}/{key}')
        for old in originals[label]:
            if any(old['config'][k]!=prototype[k] for k in ['model','training','data','device','deterministic']):
                raise ValueError('Protocol varies across historical seeds')
        for seed in seeds:
            config=copy.deepcopy(prototype)
            name=f'cifar10-{label}-{suffix}-seed{seed}'
            config.update(seed=seed,experiment_name=name,output_root=str(root))
            if smoke:
                config['training'].update(epochs=1,batch_size=4);config['device']='cpu';config['data']['workers']=0
            result.append({'id':name,'model':label,'seed':seed,'config':config})
    return result


def validate_run(run: Path, entry: dict, smoke: bool) -> dict:
    label=entry['model']
    if label not in MODELS: raise ValueError('Unexpected Phase 3 model')
    row=validate_existing(run,{**entry,'model':'D' if label in {'D1','D2'} else label},smoke)
    row['model']=label
    m=entry['config']['model']
    if label!='A' and (m['communication_iterations']!=3 or m['residual_scale']!=1 or not m['share_communication_weights']):
        raise ValueError('T/scale/sharing changed')
    if label in {'D1','D2'}:
        mapping='linear' if label=='D1' else 'sqrt'
        if m['gate_mapping']!=mapping or not m['detach_uncertainty_gate'] or m['gate_mode']!='entropy' or not m['normalize_uncertainty']:
            raise ValueError('Gate definition changed')
        if row['analysis']['gate_mapping']!=mapping: raise ValueError('Reported gate mismatch')
    if row['summary']['parameters']!=PARAMETERS[label]: raise ValueError('Parameter count changed')
    return row


def compact(row):
    row=copy.deepcopy(row)
    if row['analysis']: row['analysis'].pop('sample_records',None)
    return row


def manifest_metadata(smoke: bool) -> dict:
    if smoke: return {'phase':3,'references':[],'reference_policy':'Synthetic only; no historical real results mixed in'}
    references=[]
    for label,entries in source_entries().items():
        phase=MODELS[label][0];parent=ROOTS[phase]
        manifest=json.loads((parent/'manifest.json').read_text())
        if {e['seed'] for e in entries}!={1,2,3}: raise ValueError('Need exactly historical seeds 1,2,3')
        for entry in entries:
            run=parent/entry['id'];row=compact(validate_run(run,entry,False))
            env=json.loads((run/'environment.json').read_text())
            if env['source_hashes']!=manifest['source_hashes']: raise ValueError('Historical source provenance differs')
            paths=[run/f for f in ['summary.json','config.yaml','best.pt','split.json','environment.json']]
            paths += [parent/'manifest.json']
            if label!='A':paths.append(run/'refinement_analysis.json')
            row.update(origin=f'Phase {phase}, reused unchanged',
                       pinned_files={str(p.resolve()):file_hash(p) for p in paths})
            references.append(row)
    return {'phase':3,'references':references,'reference_policy':'A/C seeds 1–3 from Phase 1; D1/D2 seeds 1–3 from Phase 2; no reruns'}


def describe(values):
    """Sample SD and two-sided Student-t CI for the seed mean; no pooling of images."""
    values=list(values);assert_finite(values)
    n=len(values)
    if not n:return dict(n=0,mean=None,std=None,ci95=None,median=None,minimum=None,maximum=None)
    mean=statistics.mean(values);sd=statistics.stdev(values) if n>1 else None
    half=float(scipy_stats.t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=mean,std=sd,ci95=[mean-half,mean+half] if half is not None else None,
                median=statistics.median(values),minimum=min(values),maximum=max(values))


def paired_comparison(rows,left,right,seeds=(1,2,3,4,5)):
    pairs=[]
    for seed in seeds:
        a=[r for r in rows if r['model']==left and r['seed']==seed]
        b=[r for r in rows if r['model']==right and r['seed']==seed]
        if len(a)>1 or len(b)>1: raise ValueError('Duplicate model/seed result')
        if a and b:
            pairs.append({'seed':seed,'difference_pp':100*(a[0]['summary']['test_accuracy']-b[0]['summary']['test_accuracy'])})
    values=[r['difference_pp'] for r in pairs];summary=describe(values)
    test={'statistic':None,'pvalue':None,'reason':'insufficient pairs or zero variance'}
    if len(values)>1 and summary['std']>0:
        # One-sample test of paired differences is exactly the paired t-test.
        t=scipy_stats.ttest_1samp(values,0.)
        test={'statistic':float(t.statistic),'pvalue':float(t.pvalue),'reason':None}
    return {**summary,'individual_differences':pairs,'paired_t_test':test,
            'positive_seeds':sum(v>0 for v in values),'negative_seeds':sum(v<0 for v in values),
            'zero_seeds':sum(v==0 for v in values),
            'standardized_paired_effect':summary['mean']/summary['std'] if summary['std'] else None}


def q4_record(row):
    q=row['analysis']['uncertainty_groups']['q4_highest']
    return {'seed':row['seed'],'initial_accuracy_pp':q['initial_accuracy']*100,
            'final_accuracy_pp':q['final_accuracy']*100,'accuracy_change_pp':q['accuracy_change']*100,
            **{k:q[k] for k in ['corrected_count','damaged_count','corrected_rate','damaged_rate','mean_gate']},
            'corrected_damaged_ratio':q['corrected_count']/max(q['damaged_count'],1),
            'mean_effective_gated_update':q['mean_effective_update_magnitude']}


def aggregate(root: Path) -> dict:
    manifest=json.loads((root/'manifest.json').read_text());checks=[];rows=[];pending=[];new=[]
    for old in manifest['references']:
        valid=all(Path(p).exists() and file_hash(Path(p))==digest for p,digest in old['pinned_files'].items())
        checks.append({'id':old['id'],'passed':valid,'check':'historical files unchanged'})
        if valid:rows.append(old)
    for entry in manifest['runs']:
        run=root/entry['id']
        if not (run/'summary.json').exists():
            status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'not_started'}
            pending.append({'id':entry['id'],**status});continue
        try:
            row=compact(validate_run(run,entry,manifest['smoke']))
            if json.loads((run/'environment.json').read_text())['source_hashes']!=manifest['source_hashes']:
                raise ValueError('Phase 3 source provenance mismatch')
            rows.append(row);new.append(row);checks.append({'id':entry['id'],'passed':True})
        except (ValueError,KeyError,OSError,RuntimeError) as error:
            checks.append({'id':entry['id'],'passed':False,'error':str(error)})
    bad_seeds=set()
    for seed in (1,2,3,4,5):
        group=[r for r in rows if r['seed']==seed]
        if len({r['model'] for r in group})!=len(group): raise ValueError('Duplicate seeds must never be pooled')
        if len({json.dumps(r['summary']['split'],sort_keys=True) for r in group})>1:
            checks.append({'seed':seed,'passed':False,'check':'cross-model split mismatch'});bad_seeds.add(seed)
    rows=[r for r in rows if r['seed'] not in bad_seeds];new=[r for r in new if r['seed'] not in bad_seeds]
    aggregates={};q4={};low={};gates={};iteration={}
    for label in MODELS:
        group=sorted([r for r in rows if r['model']==label],key=lambda r:r['seed'])
        if not group:continue
        aggregates[label]={'seeds':[r['seed'] for r in group],'parameters':PARAMETERS[label],
            'accuracy_pp':describe([r['summary']['test_accuracy']*100 for r in group]),
            **{k:describe([r['summary'][k] for r in group]) for k in ['NLL','ECE','Brier','training_time_seconds','inference_latency_ms_per_sample']}}
        if label!='A':
            iteration[label]=[{key:describe([r['analysis']['iterations'][t][key]*(100 if key=='accuracy' else 1) for r in group])
                 for key in ['accuracy','mean_entropy','mean_confidence']} for t in range(4)]
        if label in {'D1','D2'}:
            records=[q4_record(r) for r in group]
            q4[label]={'per_seed':records,'across_seeds':{k:describe([r[k] for r in records]) for k in records[0] if k!='seed'}}
            records=[]
            for r in group:
                groups=[r['analysis']['uncertainty_groups'][q] for q in ['q1_lowest','q2','q3']]
                records.append({'seed':r['seed'],'corrected_count':sum(q['corrected_count'] for q in groups),
                                'damaged_count':sum(q['damaged_count'] for q in groups)})
            low[label]={'per_seed':records,'total_corrected':sum(r['corrected_count'] for r in records),
                        'total_damaged':sum(r['damaged_count'] for r in records),
                        'across_seeds':{k:describe([r[k] for r in records]) for k in ['corrected_count','damaged_count']}}
            fields=['mean_gate','median_gate','std_gate','q25_gate','q75_gate','fraction_below_0_01',
                    'fraction_below_0_05','fraction_above_0_25','fraction_above_0_50']
            gates[label]={'per_seed':{str(r['seed']):r['analysis']['gate_evolution'] for r in group},
                'across_seeds_by_iteration':[{key:describe([r['analysis']['gate_evolution'][t][key] for r in group]) for key in fields} for t in range(3)]}
            for t in range(4):
                iteration[label][t]['mean_gate']=describe([r['analysis']['iterations'][t]['mean_gate'] for r in group]) if t<3 else None
                iteration[label][t]['mean_relative_gated_update']=describe([r['analysis']['iterations'][t]['mean_relative_gated_update'] for r in group]) if t>0 else None
    complete=len(new)==8 and all(c['passed'] for c in checks) and (manifest['smoke'] or len(rows)==20)
    result={'phase':3,'status':'complete' if complete else 'incomplete','completed_runs':len(new),'expected_runs':8,
        'reused_runs':len(rows)-len(new),'combined_runs':len(rows),'expected_combined_runs':8 if manifest['smoke'] else 20,
        'evidence_type':'synthetic_smoke' if manifest['smoke'] else 'real CIFAR-10',
        'protocol':manifest,'per_seed_results':rows,'aggregated_results':aggregates,
        'paired_comparisons':{f'{a}-{b}':paired_comparison(rows,a,b) for a,b in PAIRS},
        'new_seed_only_comparisons':{f'{a}-{b}':paired_comparison(rows,a,b,(4,5)) for a,b in PAIRS},
        'q4_analysis':q4,'q1_q3_analysis':low,'gate_distributions':gates,'iteration_behaviour':iteration,
        'integrity_checks':checks,'incomplete_runs':pending,
        'statistics_note':'Accuracy and differences in percentage points. Sample SD; unadjusted Student-t 95% CI over seeds. Gate summaries aggregate per-seed distribution statistics, not pooled-image quantiles.'}
    assert_finite(result)
    write_json(root.parent/(root.name+'_summary.json'),result)
    write_report(root.parent/(root.name+'_report.md'),result)
    return result


def write_report(path: Path, result: dict):
    def fmt(s):
        if s['mean'] is None:return 'pending'
        return f"{s['mean']:.4f} ± {s['std']:.4f}" if s['std'] is not None else f"{s['mean']:.4f} (SD unavailable)"
    def ci(s):return f"[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]" if s['ci95'] is not None else 'pending'
    lines=['# Phase 3 confirmatory seed extension','',
           f"Status: **{result['status']}**; {result['completed_runs']}/8 new runs, {result['reused_runs']} reused, {result['combined_runs']}/{result['expected_combined_runs']} combined. Evidence: {result['evidence_type']}.",'',
           'Only A, C, D1 and D2 seeds 4 and 5 are newly scheduled. Seeds 1–3 are reused unchanged. No scientific source or protocol changes.',
           '', '## Five-seed results','',
           'Until all new runs complete, these are partial available-seed summaries, not five-seed findings. Accuracy values use the percentage-point scale.',
           '', '| Model | Seeds | Accuracy mean ± sample SD | 95% CI | Median | Min | Max | Params |',
           '|---|---|---:|---|---:|---:|---:|---:|']
    for label,a in result['aggregated_results'].items():
        s=a['accuracy_pp'];lines.append(f"| {label} | {a['seeds']} | {fmt(s)} | {ci(s)} | {s['median']:.4f} | {s['minimum']:.4f} | {s['maximum']:.4f} | {a['parameters']} |")
    lines+=['','### Individual seed results','','| Model | Seed | Accuracy (%) | Origin |','|---|---:|---:|---|']
    for r in sorted(result['per_seed_results'],key=lambda r:(r['model'],r['seed'])):
        lines.append(f"| {r['model']} | {r['seed']} | {r['summary']['test_accuracy']*100:.4f} | {r.get('origin','Phase 3')} |")
    lines+=['','## Paired comparisons','','| Comparison | Pairs | Mean ± SD (pp) | 95% CI (pp) | Positive/negative seeds | Paired t-test p |','|---|---:|---:|---|---|---|']
    for name,p in result['paired_comparisons'].items():
        lines.append(f"| {name} | {p['n']} | {fmt(p)} | {ci(p)} | {p['positive_seeds']}/{p['negative_seeds']} | {p['paired_t_test']['pvalue']} |")
        lines+=[]
    lines+=['','```json',json.dumps({k:v['individual_differences'] for k,v in result['paired_comparisons'].items()},indent=2),'```','',
            'CIs/t-tests are unadjusted, assume approximately normal independent seed differences, and do not remove selection bias from choosing D2 after seeds 1–3. New-seed-only comparisons are also saved in JSON. n=5 remains small; consistency and effect size matter more than p-values.',
            '', '## Calibration and computational cost','','| Model | NLL | ECE | Brier | Latency ms/sample | Training seconds |','|---|---:|---:|---:|---:|---:|']
    for label,a in result['aggregated_results'].items():lines.append('| '+label+' | '+' | '.join(fmt(a[k]) for k in ['NLL','ECE','Brier','inference_latency_ms_per_sample','training_time_seconds'])+' |')
    for title,key in [('Q4 behaviour','q4_analysis'),('Q1–Q3 behaviour','q1_q3_analysis'),('Gate distributions','gate_distributions'),('Iterative behaviour','iteration_behaviour')]:
        lines+=['',f'## {title}','','```json',json.dumps(result[key],indent=2),'```']
    lines+=['','Quartiles use the unchanged within-model initial-uncertainty ranking; membership can differ between models. Gate SD/quantiles are summarized across seeds, not treated as pooled-image quantiles. Iteration accuracy is percent; other diagnostics retain their original units. Gates at t describe outgoing control, updates at t describe t−1→t.',
            '', '## Scientific conclusion','']
    if result['status']!='complete' or result['evidence_type']=='synthetic_smoke':
        lines+=['Five-seed confirmation is pending. None of the five requested scientific questions can yet receive a confirmatory answer. The historical three-seed evidence is not relabeled as five-seed evidence. Complete these eight frozen runs; no next mechanism or gate tuning is recommended from incomplete or synthetic results.']
    else:
        for name,question in [('C-A','Does local communication reliably outperform baseline?'),('D2-C','Does uncertainty guidance improve fixed communication?'),('D2-D1','Is sqrt consistently better than linear?')]:
            p=result['paired_comparisons'][name]
            lines.append(f"{question} {name} = {fmt(p)} pp, CI {ci(p)}, positive in {p['positive_seeds']}/5 seeds. " + ('Positive across all observed seeds.' if p['positive_seeds']==5 else 'Not uniformly positive across seeds; avoid a broad reliability claim.'))
        p=result['paired_comparisons']['D1-C'];lines.append(f"Direct linear guidance D1−C: {fmt(p)} pp, CI {ci(p)}.")
        lines.append('Is D2 reproducible? Compare the five per-seed values and the separately reported new-seed-only effects; a mean alone is insufficient to establish reproducibility.')
        a=result['aggregated_results'];delta={k:a['D2'][k]['mean']-a['C'][k]['mean'] for k in ['NLL','ECE','Brier']}
        lines.append('D2−C mean calibration differences (negative favors D2): '+json.dumps(delta)+'.')
        p=result['paired_comparisons']['D2-C']
        if p['positive_seeds']==5:
            decision='Continue uncertainty-guided communication research: D2 beats C in every observed seed, subject to small-sample and selection caveats.'
        elif p['mean']<0 and all(v>=0 for v in delta.values()):
            decision='Stop tuning gate mappings and retain C as primary: D2 remains below C with no mean calibration advantage. Robustness benefits have not been measured.'
        else:
            decision='Do not pre-commit to retaining uncertainty gates. Inspect Q4 benefit and Q1–Q3 damage alongside these effects; robustness evaluation may be considered if the trade-off warrants it. No equivalence margin was predefined, so an interval crossing zero is not proof of a tie.'
        lines.append('Should uncertainty gating remain? '+decision)
    lines+=['','## Integrity','','```json',json.dumps(result['integrity_checks'],indent=2),'```','',
            'Limitations: CIFAR-10 only, five seeds when complete, D2 selected after exploratory results, no corruption/OOD evidence, no T sweep. No next experiment is implemented automatically.']
    path.write_text('\n'.join(lines)+'\n')
