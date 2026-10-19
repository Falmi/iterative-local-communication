"""Load reproduced successful reference runs without archived workspace records."""
import csv
import json
from pathlib import Path
import yaml


def corruption_references(runs_root):
    entries=json.loads(Path('configs/index.json').read_text())
    with Path('diagnostics/failure_summary.csv').open() as stream:
        failed={int(r['seed']) for r in csv.DictReader(stream)
                if r['model']=='C' and r['dataset']=='CIFAR-10' and r['T']=='3'}
    refs=[]
    for entry in entries:
        if entry['group']!='cifar10' or entry['model'] not in ['A','C','D1']:continue
        if entry['model']=='C' and entry['seed'] in failed:continue
        run=Path(runs_root)/entry['id']
        summary=json.loads((run/'summary.json').read_text())
        ref=dict(id=entry['id'],model=entry['model'],seed=entry['seed'],origin=str(run),
                 config=yaml.safe_load(Path(entry['config']).read_text()),summary=summary)
        if entry['model']=='D1':ref['analysis']=json.loads((run/'refinement_analysis.json').read_text())
        refs.append(ref)
    return refs


FAILED_REFERENCE=dict(model='C',seed=4,status='failed',accuracy=None,
                      reason='Recorded deterministic numerical failure; not replaced')
