import copy
import csv
import json
from pathlib import Path
import torch
from src.models.factory import create_model
from src.experiments.phase3 import describe
from src.experiments.metadata import write_json


def count(model):
    macs=0;handles=[]
    def hook(m,args,out):
        nonlocal macs
        if isinstance(m,torch.nn.Conv2d):macs+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,torch.nn.Linear):macs+=out.numel()*m.in_features
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():model(torch.zeros(1,3,32,32))
    for h in handles:h.remove()
    return dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),conv_linear_MACs=macs,
                conv_linear_arithmetic_FLOPs=2*macs,total_model_FLOPs=None)


def generate(manifest,trained):
    rows=[]
    for label,t in [('A',0),('B',3),('C',1),('C',3),('D1',3)]:
        ref=next(r for r in manifest['references'] if r['model']==label)
        cfg=copy.deepcopy(ref['config']['model'])
        if label in ['C','D1']:cfg['communication_iterations']=t
        row=dict(model=f'{label}-T{t}',**count(create_model(cfg)))
        source=[r for r in (trained if label=='C' and t==1 else manifest['references']) if r['model']==label and (r.get('T')==1 and r.get('dataset')=='CIFAR-10' if label=='C' and t==1 else True)]
        lat=describe([r['summary']['inference_latency_ms_per_sample'] for r in source])
        row.update(latency_ms=lat['mean'],latency_seed_n=lat['n'],latency_statistics=lat)
        rows.append(row)
    base=rows[0]
    for r in rows:
        r.update(relative_latency_vs_A=r['latency_ms']/base['latency_ms'] if r['latency_ms'] is not None else None,
                 parameter_increase_vs_A=r['parameters']/base['parameters']-1,
                 conv_linear_compute_increase_vs_A=r['conv_linear_MACs']/base['conv_linear_MACs']-1)
    write_json(Path('outputs/final_efficiency_summary.json'),dict(rows=rows,torch_version=torch.__version__,method='Conv2d/Linear hooks; executed invocation count; two arithmetic FLOPs per MAC; excludes bias additions, normalization, activation, pooling, residual/gate arithmetic, softmax/log/entropy and reductions. Counts are partial, not total FLOPs.'))
    with Path('outputs/final_efficiency_table.csv').open('w') as f:
        fields=[k for k in rows[0] if k!='latency_statistics'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
    Path('outputs/final_efficiency_report.md').write_text('# Efficiency accounting\n\nPyTorch '+torch.__version__+' Conv2d/Linear forward hooks count all repeated invocations at 1×3×32×32. One multiply-add = one MAC = two arithmetic FLOPs. No external counter installed.\n\nCounts exclude normalization, activation, pooling, residual/gate arithmetic, softmax/log/entropy, bias additions and reductions. **Total model FLOPs remain unavailable; partial arithmetic FLOPs are explicitly labeled.** D1 includes every classifier invocation, but functional gating is excluded.\n\nLatency is the frozen synchronized GPU validation forward timing, batch 128, historical successful seeds (C conditional); C T1 remains pending until trained. Monitoring/training overhead is not inference latency. Per-seed latency dispersion is in JSON.\n\n```json\n'+json.dumps(rows,indent=2)+'\n```\n')
    return rows
