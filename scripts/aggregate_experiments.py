"""Aggregate newly reproduced runs separately from archived manuscript results."""
import argparse
import json
from pathlib import Path
from src.experiments.phase3 import describe,paired_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True,choices=['cifar10','cifar100','communication_strength','iteration_ablation','cifar10c']);p.add_argument('--runs-root',default='runs');a=p.parse_args()
    if a.suite=='cifar10c':
        from src.final_experiments.artifacts import corruption_outputs
        refs=json.loads(Path('results/cifar10_main/references.json').read_text())
        for r in refs:
            if r['model'] in ['A','C','D1']:
                r['origin']=str(Path(a.runs_root)/r['id']);r['summary']=json.loads((Path(r['origin'])/'summary.json').read_text())
        corruption_outputs({'references':refs,'failed_reference':json.loads(Path('results/cifar10_main/failed_reference.json').read_text())})
        return
    entries=[e for e in json.loads(Path('configs/index.json').read_text()) if e['group']==a.suite];rows=[];statuses=[]
    for e in entries:
        run=Path(a.runs_root)/e['id'];status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'pending'}
        label=e['model']
        cfg=json.loads(json.dumps(__import__('yaml').safe_load(Path(e['config']).read_text())))
        if a.suite=='iteration_ablation':label+=f"-T{cfg['model']['communication_iterations']}"
        statuses.append(dict(id=e['id'],model=label,seed=e['seed'],**status))
        if status['status']=='completed':
            s=json.loads((run/'summary.json').read_text());rows.append(dict(model=label,seed=e['seed'],summary=s))
    labels=sorted({r['model'] for r in rows});agg={k:{metric:describe([r['summary'][metric] for r in rows if r['model']==k]) for metric in ['test_accuracy','NLL','ECE','Brier','inference_latency_ms_per_sample']} for k in labels}
    pairs={f'{b}-{x}':paired_comparison(rows,b,x) for i,x in enumerate(labels) for b in labels[i+1:]}
    out=Path('outputs/reproduced');out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.suite}.json').write_text(json.dumps(dict(statuses=statuses,per_seed=rows,aggregates=agg,paired_comparisons=pairs,note='Accuracy fractions; paired differences percentage points. Successful runs only. No equivalence inference.'),indent=2,allow_nan=False))
if __name__=='__main__':main()
