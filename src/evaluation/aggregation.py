"""Unchanged seed statistics and corruption aggregation from the full release."""
import csv
import json
import math
import statistics
from pathlib import Path
from scipy import stats as scipy_stats
from src.experiments.metadata import write_json
from src.final_experiments.data import CORRUPTIONS

def assert_finite(value):
    if isinstance(value,float) and not math.isfinite(value): raise ValueError('Non-finite metric')
    if isinstance(value,dict):
        for item in value.values(): assert_finite(item)
    if isinstance(value,list):
        for item in value: assert_finite(item)

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

def csvwrite(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ['status'])
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in fields})

def corruption_outputs(manifest):
    root=Path('outputs/final_cifar10c');root.mkdir(exist_ok=True)
    cells=[json.loads(p.read_text()) for p in (root/'cells').glob('*.json')]
    expected={(r['model'],r['seed'],c,v):r for r in manifest['references'] if r['model'] in ['A','C','D1'] for c in CORRUPTIONS for v in range(1,6)}
    seen=set()
    for cell in cells:
        identity=tuple(cell[k] for k in ['model','seed','corruption','severity'])
        if identity not in expected or identity in seen:raise ValueError('Invalid/duplicate corruption cell')
        seen.add(identity)
        ref=expected[identity]
        if cell['checkpoint_sha256']!=ref['summary']['test_checkpoint_sha256']:raise ValueError('Corruption checkpoint differs')
        if cell['status']=='completed' and cell['samples']!=10000:raise ValueError('Incomplete corruption cell')
    complete=[c for c in cells if c['status']=='completed'];seedrows=[];bytype=[];byseverity=[]
    keys=['accuracy','NLL','ECE','Brier','predictive_entropy','initial_predictive_entropy','prediction_change_rate','mean_gate','mean_relative_effective_update']
    def avg(g):return {k:statistics.mean([v[k] for v in g if v.get(k) is not None]) if any(v.get(k) is not None for v in g) else None for k in keys}
    for ref in manifest['references']:
        if ref['model'] not in ['A','C','D1']:continue
        g=[c for c in complete if c['model']==ref['model'] and c['seed']==ref['seed']]
        if len(g)==95:
            values=avg(g);values['clean_to_corrupted_drop_pp']=100*(ref['summary']['test_accuracy']-values['accuracy'])
            seedrows.append(dict(model=ref['model'],seed=ref['seed'],summary=dict(test_accuracy=values['accuracy']),metrics=values,
                                 primary15_metrics=avg([c for c in g if c['corruption'] in CORRUPTIONS[:15]])))
        for c in CORRUPTIONS:
            subset=[v for v in g if v['corruption']==c]
            if len(subset)==5:bytype.append(dict(model=ref['model'],seed=ref['seed'],corruption=c,**avg(subset)))
        for s in range(1,6):
            subset=[v for v in g if v['severity']==s]
            if len(subset)==19:byseverity.append(dict(model=ref['model'],seed=ref['seed'],severity=s,**avg(subset)))
    aggregated={}
    for label in ['A','C','D1']:
        group=[r for r in seedrows if r['model']==label]
        aggregated[label]={k:describe([r['metrics'][k] for r in group if r['metrics'].get(k) is not None]) for k in [*keys,'clean_to_corrupted_drop_pp']}
    comparisons={f'{a}-{b}':paired_comparison(seedrows,a,b) for a,b in [('D1','A'),('C','A'),('D1','C')]}
    result=dict(status='complete' if len(cells)==1330 and all(c['status']=='completed' for c in cells) else 'pending',
        expected_cells=1330,completed_cells=len(complete),failed_cells=[r for r in cells if r['status']=='failed'],
        per_seed=seedrows,aggregates=aggregated,paired_comparisons=comparisons,per_corruption=bytype,per_severity=byseverity,
        failed_training_reference=manifest['failed_reference'],note='19 types × 5 severities; not mCE. Primary-15 subset separately recorded per seed. Only complete cells/seed panels enter means. Entropy in nats; gates normalized. C conditional on seeds 1,2,3,5.')
    def grouped(records,axis):
        output=[]
        for label,value in sorted({(r['model'],r[axis]) for r in records}):
            g=[r for r in records if r['model']==label and r[axis]==value]
            output.append(dict(model=label,**{axis:value},**{k:describe([r[k] for r in g if r[k] is not None]) for k in keys}))
        return output
    result['per_corruption_across_seeds']=grouped(bytype,'corruption')
    result['per_severity_across_seeds']=grouped(byseverity,'severity')
    result['clean_D1_initial_entropy']=describe([r['analysis']['iterations'][0]['mean_entropy'] for r in manifest['references'] if r['model']=='D1'])
    write_json(root/'summary.json',result)
    csvwrite(root/'per_corruption.csv',result['per_corruption_across_seeds']);csvwrite(root/'per_severity.csv',result['per_severity_across_seeds'])
    csvwrite(root/'per_corruption_seed.csv',bytype);csvwrite(root/'per_severity_seed.csv',byseverity)
    csvwrite(root/'paired_comparisons.csv',[dict(comparison=k,**v) for k,v in comparisons.items()])
    csvwrite(root/'cells.csv',complete,fields=['model','seed','corruption','severity',*keys])
    (root/'report.md').write_text('# CIFAR-10-C\n\n'+result['note']+f"\n\nStatus: {result['status']}; {len(complete)}/1330 completed cells.\n\n```json\n"+json.dumps(dict(aggregates=aggregated,paired_comparisons=comparisons,failures=result['failed_cells']),indent=2)+'\n```\n')
    return result

