"""Regenerate paper tables/figures from the included recorded measurements, without training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from src.final_experiments.artifacts import figures,generate_tables,savefig
from src.experiments.phase3 import describe
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/regenerated');a=p.parse_args()
    root=Path.cwd();out=(root/a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    read=lambda path:json.loads((root/path).read_text())
    m=dict(references=read('results/cifar10_main/references.json'),failed_reference=read('results/cifar10_main/failed_reference.json'))
    parts={old:read(f'results/{new}/summary.json') for old,new in [('final_cifar100','cifar100'),('final_iteration_ablation','iteration_ablation'),('final_spatial_control','one_shot')]}
    robust=read('results/cifar10c/summary.json');eff=read('results/efficiency/final_efficiency_summary.json')['rows']
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for name in ['final_iteration_ablation','final_cifar100','final_cifar10c']:Path('outputs',name).mkdir(parents=True)
            shutil.copy2(root/'results/communication_strength/phase4_summary.json','outputs/experiment_phase4_summary.json')
            generate_tables(m,parts,robust,eff);figures(m,parts,robust)
            shutil.copytree('outputs/manuscript_tables',out/'tables',dirs_exist_ok=True)
            shutil.copytree('outputs/manuscript_figures',out/'figures',dirs_exist_ok=True)
        finally:os.chdir(root)
    fig,ax=plt.subplots(figsize=(10,2.3));ax.axis('off')
    for i,text in enumerate(['A: ResNet-18\nFeed-forward','B: Pointwise\nShared iteration','C: Local 3×3\nShared messages','D1: Local + entropy\nDetached linear gate']):
        ax.text(.12+.25*i,.5,text,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#eaf1f8'))
        if i<3:ax.annotate('',xy=(.25+.25*i,.5),xytext=(.22+.25*i,.5),arrowprops=dict(arrowstyle='->'))
    savefig(fig,out/'figures/architecture_ladder')
    x=[];y=[];err=[]
    for severity in range(1,6):
        g=[r for r in robust['per_severity'] if r['severity']==severity];pairs=[]
        for seed in range(1,6):
            aa=next(r for r in g if r['model']=='A' and r['seed']==seed);dd=next(r for r in g if r['model']=='D1' and r['seed']==seed)
            pairs.append(100*(dd['accuracy']-aa['accuracy']))
        s=describe(pairs);x.append(severity);y.append(s['mean']);err.append(s['ci95'][1]-s['mean'])
    fig,ax=plt.subplots(figsize=(4,3));ax.errorbar(x,y,yerr=err,marker='o',capsize=3);ax.axhline(0,color='gray',linestyle=':');ax.set(xlabel='Severity',ylabel='D1 − A accuracy (percentage points)',title='Paired seed mean and 95% t interval')
    savefig(fig,out/'figures/D1_minus_A_severity')
    print(out)
if __name__=='__main__':main()
