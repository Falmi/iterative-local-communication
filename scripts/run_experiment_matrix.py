"""List or execute exact recorded configurations; failed seeds are never replaced."""
import argparse
import json
from pathlib import Path
from scripts.train import execute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True,choices=['cifar10','cifar100','communication_strength','iteration_ablation'])
    p.add_argument('--execute',action='store_true');p.add_argument('--download',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();entries=[e for e in json.loads(Path('configs/index.json').read_text()) if e['group']==a.suite]
    for entry in entries:
        print(entry['id'],entry['config'], 'historical reference' if 'reuse_id' in entry else '',flush=True)
        if a.execute:execute(entry,a.download,False,a.resume)
    print(len(entries),'configurations; no execution' if not a.execute else 'configurations handled')
if __name__=='__main__':main()
