"""Checks that the curated package stands alone and preserves recorded outcomes."""
import hashlib
import json
from pathlib import Path
import torch
import yaml


def test_recorded_partitions():
    for row in json.loads(Path('splits/hashes.json').read_text()):
        ids=torch.randperm(50000,generator=torch.Generator().manual_seed(row['seed'])).tolist()
        n=row['validation_samples']
        assert hashlib.sha256(json.dumps(ids[:n]).encode()).hexdigest()==row['validation_indices_sha256']
        assert hashlib.sha256(json.dumps(ids[n:]).encode()).hexdigest()==row['train_indices_sha256']


def test_all_indexed_configs():
    entries=json.loads(Path('configs/index.json').read_text())
    for entry in entries:
        cfg=yaml.safe_load(Path(entry['config']).read_text())
        assert cfg['seed']==entry['seed']
        assert cfg['data']['root']=='data'
        assert cfg['training']==dict(epochs=200,batch_size=128,learning_rate=.1,momentum=.9,weight_decay=.0005)
        assert not Path(cfg['output_root']).is_absolute()
    assert {e['seed'] for e in entries if e['group']=='cifar100'}=={1,2,3}


def test_failed_seeds_retained():
    import csv
    def rows(path):
        with Path(path).open() as stream:return list(csv.DictReader(stream))
    failed=rows('diagnostics/failure_summary.csv')
    assert len(failed)==6 and all(r['status']=='failed' for r in failed)
    assert {int(r['seed']) for r in failed if r['dataset']=='CIFAR-10' and r['T']=='5'}=={1,2,3}
    c100=next(r for r in rows('results/cifar100_summary.csv') if r['model']=='C')
    assert c100['successful_runs']=='1' and c100['attempted_runs']=='3' and c100['accuracy_std']==''
    ct5=next(r for r in rows('results/iteration_ablation.csv') if r['model']=='C' and r['T']=='5')
    assert ct5['successful_runs']=='0' and ct5['accuracy_mean']=='' and ct5['accuracy_std']==''


def test_corruption_references_need_only_surviving_files(tmp_path):
    from src.evaluation.run_records import corruption_references
    entries=json.loads(Path('configs/index.json').read_text())
    for e in entries:
        if e['group']!='cifar10' or e['model'] not in ['A','C','D1'] or (e['model']=='C' and e['seed']==4):continue
        run=tmp_path/e['id'];run.mkdir()
        (run/'summary.json').write_text(json.dumps({'test_accuracy':0.9}))
        if e['model']=='D1':(run/'refinement_analysis.json').write_text(json.dumps({'iterations':[{'mean_entropy':0.1}]}))
    refs=corruption_references(tmp_path)
    assert len(refs)==14
    assert not any(r['model']=='C' and r['seed']==4 for r in refs)
    assert all(r['config']['model']['num_classes']==10 for r in refs)


def test_paired_statistics_missing_seed_is_not_zero():
    from src.evaluation.aggregation import paired_comparison
    rows=[{'model':'A','seed':1,'summary':{'test_accuracy':.9}},
          {'model':'D1','seed':1,'summary':{'test_accuracy':.91}},
          {'model':'A','seed':2,'summary':{'test_accuracy':.8}}]
    result=paired_comparison(rows,'D1','A')
    assert result['n']==1 and abs(result['mean']-1)<1e-10
    assert result['std'] is None and result['ci95'] is None
