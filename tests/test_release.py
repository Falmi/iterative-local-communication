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


def test_scientific_source_identity():
    provenance=json.loads(Path('docs/source_provenance.json').read_text())
    for name,digest in provenance['original_source_hashes'].items():
        if name.startswith('src/'):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name


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
    it=json.loads(Path('results/iteration_ablation/summary.json').read_text())
    failed=[r for r in it['run_statuses'] if r['model']=='C' and r['T']==5]
    assert len(failed)==3 and all(r['status']=='failed' for r in failed)
    c100=json.loads(Path('results/cifar100/summary.json').read_text())
    assert {r['seed'] for r in c100['run_statuses'] if r['model']=='C' and r['status']=='failed'}=={1,2}
    assert json.loads(Path('results/cifar10_main/failed_reference.json').read_text())['seed']==4


def test_exact_phase1_snapshot():
    root=Path('historical/phase1')
    recorded=json.loads((root/'source_hashes.json').read_text())
    assert len(recorded)==38
    for name,digest in recorded.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
