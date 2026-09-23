import csv
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.experiments.phase3 import describe,paired_comparison
from src.experiments.metadata import write_json
from src.final_experiments.protocol import ROOT,validate,validate_source_provenance
from src.final_experiments.data import CORRUPTIONS
from src.final_experiments.efficiency import generate as efficiency


def csvwrite(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ['status'])
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in fields})


def tables(path,headers,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    def tex(value):return str(value).replace('_',r'\_').replace('%',r'\%')
    text='\\begin{tabular}{'+'l'*len(headers)+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(tex(v) for v in row)+r' \\' for row in rows)
    path.write_text(text+'\n\\bottomrule\n\\end{tabular}\n')


def fmt(s):
    if not s or s['mean'] is None:return 'pending'
    return f"{s['mean']:.3f} ± {s['std']:.3f}" if s['std'] is not None else f"{s['mean']:.3f} (n=1)"


def savefig(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def training_rows(manifest):
    rows=[];statuses=[]
    entries=list(manifest['runs'])
    extension=ROOT/'extension.json'
    if extension.exists():entries+=json.loads(extension.read_text())['runs']
    for e in entries:
        run=ROOT/'runs'/e['id'];status=dict(id=e['id'],model=e['model'],seed=e['seed'],T=e['T'],dataset=e['dataset'],status='pending')
        if (run/'summary.json').exists():
            row=validate(run,e)
            validate_source_provenance(run,manifest)
            rows.append(row);status['status']='completed'
        elif (run/'stability.json').exists() and json.loads((run/'stability.json').read_text()).get('failure'):
            status.update(status='failed',failure=json.loads((run/'stability.json').read_text())['failure'])
        elif (run/'status.json').exists():status['execution_detail']=json.loads((run/'status.json').read_text())
        statuses.append(status)
    return rows,statuses


def summarize(rows):
    out={}
    for name in sorted({r['model'] for r in rows}):
        g=[r for r in rows if r['model']==name]
        out[name]=dict(seeds=[r['seed'] for r in g],accuracy_pp=describe([100*r['summary']['test_accuracy'] for r in g]),
            **{k:describe([r['summary'][k] for r in g]) for k in ['NLL','ECE','Brier','inference_latency_ms_per_sample']})
    return out


def training_outputs(manifest,rows,statuses):
    c100=[r for r in rows if r['dataset']=='CIFAR-100'];s100=[s for s in statuses if s['dataset']=='CIFAR-100']
    iteration=[dict(r,model=f"{r['model']}-T{r['T']}") for r in rows if r['dataset']=='CIFAR-10']
    iteration += [dict(r,model=r['model']+'-T3',T=3) for r in manifest['references'] if r['model'] in ['C','D1'] and r['seed']<=3]
    spatial=[r for r in iteration if r['model'] in ['C-T1','C-T3','D1-T3']]+[r for r in manifest['references'] if r['model']=='A' and r['seed']<=3]
    groups=[('final_cifar100',c100,s100,[('C','A'),('D1','A'),('D1','C')]),
            ('final_iteration_ablation',iteration,[s for s in statuses if s['dataset']=='CIFAR-10'],[(f'{m}-T{t}',f'{m}-T3') for m in ['C','D1'] for t in [1,2,5]]),
            ('final_spatial_control',spatial,[s for s in statuses if s['dataset']=='CIFAR-10' and s['model']=='C' and s['T']==1],[('C-T1','A'),('C-T3','C-T1'),('D1-T3','C-T1')])]
    summaries={}
    for dirname,g,status,pairs in groups:
        root=Path('outputs')/dirname;root.mkdir(exist_ok=True)
        result=dict(status='complete' if all(s['status']!='pending' for s in status) else 'pending',run_statuses=status,
                    per_seed_results=g,aggregates=summarize(g),paired_comparisons={f'{a}-{b}':paired_comparison(g,a,b) for a,b in pairs},
                    note='Successful-run conditional summaries. Failure has no accuracy. C T1 is the one-shot control, no duplicate experiment. Small-seed unadjusted Student-t intervals; not equivalence tests.')
        for row in g:
            if row.get('analysis'):
                a=row['analysis'];o=a['outcomes']
                row['refinement']=dict(prediction_change_frequency=1-a['never_changed_count']/a['samples'],
                    corrected_damaged_ratio=o['corrected']/o['damaged'] if o['damaged'] else None,
                    outcomes=o,iterations=a['iterations'],uncertainty_quartiles=a.get('uncertainty_groups'))
        write_json(root/'summary.json',result)
        csvwrite(root/'per_seed.csv',[dict(model=r['model'],seed=r['seed'],status='completed',accuracy=100*r['summary']['test_accuracy'],NLL=r['summary']['NLL'],ECE=r['summary']['ECE'],Brier=r['summary']['Brier'],parameters=r['summary']['parameters'],latency_ms=r['summary']['inference_latency_ms_per_sample']) for r in g])
        csvwrite(root/'run_statuses.csv',status)
        csvwrite(root/'paired_comparisons.csv',[dict(comparison=k,**v) for k,v in result['paired_comparisons'].items()])
        (root/'report.md').write_text('# '+dirname+'\n\nStatus: '+result['status']+'\n\n'+result['note']+'\n\nFailures/pending:\n```json\n'+json.dumps(status,indent=2)+'\n```\n\nAggregates and paired comparisons:\n```json\n'+json.dumps({k:result[k] for k in ['aggregates','paired_comparisons']},indent=2)+'\n```\n')
        summaries[dirname]=result
    return summaries


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


def figures(manifest,parts,robust):
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    root=Path('outputs/manuscript_figures');root.mkdir(exist_ok=True)
    refs=manifest['references'];fig,ax=plt.subplots(figsize=(3.5,2.8));labels=[];means=[];sds=[]
    for label in ['A','B','C','D1']:
        g=[r for r in refs if r['model']==label];s=describe([100*r['summary']['test_accuracy'] for r in g])
        labels.append(f'{label}\n{len(g)}/{3 if label=="B" else 5}');means.append(s['mean']);sds.append(s['std'])
    ax.errorbar(range(4),means,yerr=sds,fmt='o',capsize=3);ax.set_xticks(range(4),labels);ax.set_ylabel('Clean accuracy (%)');ax.set_title('Mean ± sample SD; C conditional')
    savefig(fig,root/'clean_accuracy_stability')
    phase4=json.loads(Path('outputs/experiment_phase4_summary.json').read_text())
    fig,ax=plt.subplots(figsize=(3.5,2.8))
    for label,g in phase4['communication_dynamics'].items():
        ax.plot([1,2,3],[v['relative_effective_update']['mean'] for v in g['across_seeds']],marker='o',label=label)
    ax.set(xlabel='Communication iteration',ylabel='Relative effective update');ax.legend();savefig(fig,root/'communication_dynamics')
    c100=parts['final_cifar100']['aggregates']
    if c100:
        fig,ax=plt.subplots(figsize=(3.5,2.8));labels=list(c100)
        ax.errorbar(range(len(labels)),[c100[k]['accuracy_pp']['mean'] for k in labels],
                    yerr=[c100[k]['accuracy_pp']['std'] or 0 for k in labels],fmt='o',capsize=3)
        ax.set_xticks(range(len(labels)),labels);ax.set_ylabel('CIFAR-100 accuracy (%)')
        savefig(fig,Path('outputs/final_cifar100/figures/accuracy'))
    it=parts['final_iteration_ablation']['per_seed_results']
    # No fabricated curves: render ablation only when new T evidence exists.
    if any(r.get('T')!=3 for r in it):
        for name,key,ylabel in [('A1_accuracy','test_accuracy','Accuracy (%)'),('A2_latency','inference_latency_ms_per_sample','Latency (ms/sample)'),('A3_prediction_changes',None,'Prediction-change frequency')]:
            fig,ax=plt.subplots(figsize=(3.5,2.8))
            for label in ['C','D1']:
                xs=[];ys=[];err=[]
                for t in [1,2,3,5]:
                    g=[r for r in it if r['model']==f'{label}-T{t}']
                    if not g:continue
                    values=[r['refinement']['prediction_change_frequency'] if key is None else r['summary'][key]*(100 if key=='test_accuracy' else 1) for r in g]
                    s=describe(values);xs.append(t);ys.append(s['mean']);err.append(s['std'] or 0)
                ax.errorbar(xs,ys,yerr=err,marker='o',capsize=2,label=label)
            ax.set(xlabel='T',ylabel=ylabel);ax.legend();savefig(fig,root/name)
            target=Path('outputs/final_iteration_ablation/figures');target.mkdir(exist_ok=True)
            for ext in ['png','pdf']:(target/f'{name}.{ext}').write_bytes((root/f'{name}.{ext}').read_bytes())
    if robust['per_severity']:
        for name,key,ylabel,labels in [('R1_severity','accuracy','Corruption accuracy (%)',['A','C','D1']),('R3_gate','mean_gate','D1 mean gate',['D1']),('R3_entropy','initial_predictive_entropy','D1 initial entropy (nats)',['D1'])]:
            fig,ax=plt.subplots(figsize=(3.5,2.8))
            for label in labels:
                xs=[];ys=[];errors=[]
                for s in range(1,6):
                    values=[r[key]*(100 if key=='accuracy' else 1) for r in robust['per_severity'] if r['model']==label and r['severity']==s]
                    if values:
                        d=describe(values);xs.append(s);ys.append(d['mean']);errors.append(d['std'] or 0)
                ax.errorbar(xs,ys,yerr=errors,marker='o',capsize=2,label=label)
            ax.set(xlabel='Severity',ylabel=ylabel);ax.legend();savefig(fig,root/name)
        deltas=[]
        for c in CORRUPTIONS:
            g=[r for r in robust['per_corruption'] if r['corruption']==c];values=[]
            for seed in range(1,6):
                a=next((r for r in g if r['model']=='A' and r['seed']==seed),None);d=next((r for r in g if r['model']=='D1' and r['seed']==seed),None)
                if a and d:values.append(100*(d['accuracy']-a['accuracy']))
            if values:deltas.append((c,describe(values)))
        if deltas:
            fig,ax=plt.subplots(figsize=(4,5));ax.errorbar([s['mean'] for c,s in deltas],range(len(deltas)),xerr=[s['std'] or 0 for c,s in deltas],fmt='o',capsize=2)
            ax.set_yticks(range(len(deltas)),[c.replace('_',' ') for c,s in deltas]);ax.axvline(0,linestyle=':');ax.set_xlabel('D1 − A accuracy (pp), mean ± SD');savefig(fig,root/'R2_corruption_differences')
        target=Path('outputs/final_cifar10c/figures');target.mkdir(exist_ok=True)
        for f in root.glob('R*.*'):(target/f.name).write_bytes(f.read_bytes())


def generate_tables(manifest,parts,robust,eff):
    root=Path('outputs/manuscript_tables');root.mkdir(exist_ok=True)
    def accuracy_table(filename,aggregates,statuses=None):
        rows=[]
        for label,g in aggregates.items():
            n=g['accuracy_pp']['n'];failed=sum(s['status']=='failed' and (s['model']==label or f"{s['model']}-T{s.get('T')}"==label) for s in (statuses or []))
            rows.append([label,n,failed,fmt(g['accuracy_pp']),fmt(g['NLL']),fmt(g['ECE']),fmt(g['Brier'])])
        tables(root/filename,['Model','Successful $n$','Failed',r'Accuracy (\%)','NLL','ECE','Brier'],rows or [['pending']+['--']*6])
    clean=summarize(manifest['references']);accuracy_table('table_cifar10_main.tex',clean,[manifest['failed_reference']])
    phase4=json.loads(Path('outputs/experiment_phase4_summary.json').read_text())
    rows=[]
    for label,g in phase4['accuracy_and_calibration'].items():
        s=phase4['stability_table'][label];rows.append([label,f"{s['completed']}/{s['attempted']}",s['failed'],fmt(g['accuracy_pp'])])
    tables(root/'table_phase4_stability.tex',['Model','Completed/attempted','Failed',r'Accuracy (\%)'],rows)
    for dirname,filename in [('final_cifar100','table_cifar100.tex'),('final_iteration_ablation','table_iteration_ablation.tex'),('final_spatial_control','table_spatial_control.tex')]:
        accuracy_table(filename,parts[dirname]['aggregates'],parts[dirname]['run_statuses'])
    rows=[]
    for label,g in robust['aggregates'].items():
        a=g['accuracy'];pp={k:(v*100 if isinstance(v,(int,float)) and k not in ['n'] else v) for k,v in a.items()}
        rows.append([label,a['n'],fmt(pp),fmt(g['NLL']),fmt(g['ECE']),fmt(g['Brier'])])
    tables(root/'table_cifar10c.tex',['Model','Successful seeds',r'Mean corruption acc. (\%)','NLL','ECE','Brier'],rows)
    tables(root/'table_efficiency.tex',['Model','Parameters','Conv/linear MACs','Partial FLOPs','Latency (ms)'],[[r['model'],r['parameters'],r['conv_linear_MACs'],r['conv_linear_arithmetic_FLOPs'],f"{r['latency_ms']:.3f}" if r['latency_ms'] is not None else 'pending'] for r in eff])
    (root/'README.md').write_text('Booktabs input fragments generated from validated JSON. Accuracy mean ± sample SD; full Student-t CIs/median/range in JSON. n is successful seeds, not attempts. B has only three clean seeds; C clean has four successes of five attempted. No failed accuracy is imputed. CIFAR-10-C is the unnormalized 19-corruption average, not mCE. FLOPs are Conv2d/Linear arithmetic only, not total model FLOPs. Historical/current latency summaries may differ in seed count; no equivalence claim.\n')
    for p in root.glob('*.tex'):p.write_text(p.read_text().replace('±',r'$\pm$'))


def decision_report(parts,robust,eff):
    completed=robust['status']=='complete' and all(p['status']=='complete' for p in parts.values())
    questions=['Does D1 improve robustness over A?','Is benefit consistent across severity levels?','Which corruptions benefit most or least?',
        'Does uncertainty increase under corruption?','Does D1 communication strength increase with severity?',
        'Does D1 generalize better than A on CIFAR-100?','Does C reproduce instability on CIFAR-100?',
        'Does recurrent C outperform a one-shot block?','Is T=3 justified?','What is the accuracy/latency/compute trade-off?',
        'Does all evidence support the proposed title?','Strongest defensible contribution?','What claims must not be made?']
    answers=['Pending real CIFAR-10-C evaluation.']*5+['Pending CIFAR-100 training.']*2+['Pending iteration/spatial-control results.']*2
    answers += ['Measured Conv2d/Linear MACs and partial arithmetic FLOPs are available; total FLOPs exclude unsupported operations. T1 GPU latency and accuracy remain pending until training completes.',
        'The established clean CIFAR-10 evidence supports a study of adaptive local refinement and observed stability, not a universal stability guarantee. New robustness/generalization evidence remains pending.',
        'A controlled mechanism ladder and reproducible numerical-failure diagnosis, with evidence that nominal residual scaling and learned effective updates differ and D1 selectively modulates communication.',
        'Do not claim universal clean accuracy, robustness or generalization superiority; equivalence from nonsignificance; a population failure probability of 20%; or isolation of the backward NaN origin. Never hide the failed C seed.']
    if robust['per_seed']:
        p=robust['paired_comparisons']['D1-A'];answers[0]=f"Available matched-seed D1−A: {fmt(p)} pp; 95% CI {p['ci95']}, n={p['n']}. Full-panel completion: {robust['status']}."
        severity={};types={}
        for axis,records,labels in [('severity',robust['per_severity'],range(1,6)),('corruption',robust['per_corruption'],CORRUPTIONS)]:
            dest=severity if axis=='severity' else types
            for label in labels:
                diffs=[]
                for seed in range(1,6):
                    g=[r for r in records if r[axis]==label and r['seed']==seed]
                    a=next((r for r in g if r['model']=='A'),None);d=next((r for r in g if r['model']=='D1'),None)
                    if a and d:diffs.append(100*(d['accuracy']-a['accuracy']))
                if diffs:dest[str(label)]=describe(diffs)
        answers[1]='Matched-seed D1−A mean accuracy differences by severity (pp): '+json.dumps({k:v['mean'] for k,v in severity.items()})+'.'
        if types:
            ranked=sorted(types,key=lambda k:types[k]['mean']);answers[2]=f"Least: {ranked[0]} ({types[ranked[0]]['mean']:.4f} pp); most: {ranked[-1]} ({types[ranked[-1]]['mean']:.4f} pp). Full per-corruption effect sizes/seed values in CSV; no selection of favorable types."
        for i,key in [(3,'initial_predictive_entropy'),(4,'mean_gate')]:
            vals={s:describe([r[key] for r in robust['per_severity'] if r['model']=='D1' and r['severity']==s]) for s in range(1,6)}
            answers[i]='D1 severity summaries: '+json.dumps({k:v['mean'] for k,v in vals.items()})+'. These observational associations do not prove causal benefit from stronger gates.'
    c=parts['final_cifar100'];p=c['paired_comparisons']['D1-A']
    if p['n']:answers[5]=f"D1−A {fmt(p)} pp, 95% CI {p['ci95']}, n={p['n']}. Report completion counts and unadjusted paired p={p['paired_t_test']['pvalue']}; no equivalence inference."
    failures=[s for s in c['run_statuses'] if s['model']=='C' and s['status']=='failed']
    if failures or c['status']=='complete':answers[6]=f"C numerical failures: {len(failures)}; records: {json.dumps(failures)}. Do not replace seeds or introduce a fix."
    s=parts['final_spatial_control']['paired_comparisons']['C-T3-C-T1']
    if s['n']:answers[7]=f"C T3−T1: {fmt(s)} pp, CI {s['ci95']}, n={s['n']}. C T1 is exactly the one-shot spatial block."
    if parts['final_iteration_ablation']['status']=='complete':answers[8]='Use the paired T-vs-T3 differences, latency and change-rate figures; the ablation does not preregister an optimal T or establish equivalence. '+json.dumps(parts['final_iteration_ablation']['paired_comparisons'])
    if completed:
        answers[9]='Measured accuracy, latency and partial arithmetic cost appear in the efficiency and ablation tables; D1 incurs repeated classifier/gate evaluation. Total FLOPs remain explicitly unsupported.'
        answers[10]='The title describes the mechanism and tested observations; retain explicit dataset/seed scope and avoid interpreting stable observed runs as a universal guarantee.'
    text='# Final manuscript experiment decision report\n\n'+('\n\n'.join(f'{i+1}. **{q}** {a}' for i,(q,a) in enumerate(zip(questions,answers))))
    text+='\n\n'+('All scheduled real observations are available; manuscript revision and integrity review are still required.\n\n' if completed else 'New evidence is incomplete. No manuscript TBD is replaced with synthetic or invented results.\n\n')+'ADDITIONAL EVIDENCE REQUIRED\n'
    Path('outputs/final_manuscript_experiment_report.md').write_text(text)
    return completed


def revise_if_complete(complete):
    source=Path('cellular_manuscript/manuscript.tex')
    if not source.exists():return 'manuscript source unavailable'
    if not complete:return 'withheld until required experiments finish; original manuscript unchanged'
    text=source.read_text();start=text.index(r'\section{Robustness and Generalization Experiments to Complete Before Submission}')
    end=text.index(r'\section{Limitations}',start)
    section=r'''\section{Robustness, Generalization, and Recurrent Controls}
The following automatically generated tables include successful-run summaries;
failed seeds remain categorical and are not assigned accuracy. Complete per-seed
values, paired differences and unadjusted Student-t intervals accompany the source.
\subsection{CIFAR-10-C}
Evaluation covers all 19 distributed corruption types at severities 1--5 without
training on corruptions. Mean corruption accuracy is unnormalized, not mCE.
\input{../outputs/manuscript_tables/table_cifar10c.tex}
\subsection{CIFAR-100}
The training recipe is unchanged apart from 100 output classes and fixed dataset
normalization. Failures are retained rather than replaced.
\input{../outputs/manuscript_tables/table_cifar100.tex}
\subsection{Iteration and spatial control}
The one-shot control is exactly Model C at T=1. T=3 checkpoints are reused.
This is an ablation, not a test-selected claim of an optimal iteration count.
\input{../outputs/manuscript_tables/table_iteration_ablation.tex}
\input{../outputs/manuscript_tables/table_spatial_control.tex}
\subsection{Efficiency}
MACs count executed Conv2d and Linear operations for a single 32x32 RGB image.
Two arithmetic FLOPs per MAC are reported as partial counts, excluding normalization,
pooling, residual arithmetic and entropy/gating. They are not total model FLOPs.
\input{../outputs/manuscript_tables/table_efficiency.tex}
'''
    text=text[:start]+section+'\n'+text[end:]
    text=text.replace('The current experiments are restricted to CIFAR-10, so the conclusions concern mechanism behaviour in this setting rather than universal superiority.', 'The experiments cover CIFAR-10, CIFAR-10-C, and CIFAR-100; conclusions remain limited to these datasets, tested seeds, and the fixed training recipe rather than universal superiority.')
    # A separate draft: no unsupported automatic strengthening of abstract/conclusion.
    dest=source.with_name('manuscript_results_complete.tex');dest.write_text(text)
    return 'results-inserted draft saved; prose/LaTeX compilation review still required before submission'


def generate(manifest):
    rows,statuses=training_rows(manifest);parts=training_outputs(manifest,rows,statuses)
    robust=corruption_outputs(manifest);eff=efficiency(manifest,rows)
    generate_tables(manifest,parts,robust,eff);figures(manifest,parts,robust)
    complete=decision_report(parts,robust,eff);manuscript=revise_if_complete(complete)
    result=dict(status='completed' if complete else 'pending',training_statuses=statuses,robustness_completed_cells=robust['completed_cells'],manuscript=manuscript)
    write_json(ROOT/'status.json',result)
    return result
