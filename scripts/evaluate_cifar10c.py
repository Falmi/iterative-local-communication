"""Evaluate independently reproduced A/C/D1 checkpoints on all 19 corruptions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import torch
from src.final_experiments.data import download_corruptions
from src.final_experiments.robustness import execute


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true');p.add_argument('--data-root',default='data/CIFAR-10-C')
    p.add_argument('--checkpoints-root',default='runs');a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the recorded evaluation implementation')
    torch.set_num_threads(6);refs=json.loads(Path('results/cifar10_main/references.json').read_text())
    refs=[r for r in refs if r['model'] in ['A','C','D1']]
    for r in refs:
        r['origin']=str(Path(a.checkpoints_root)/r['id'])
        path=Path(r['origin'])/'summary.json'
        if not path.exists():raise FileNotFoundError(f'Reproduce or supply successful checkpoint and summary: {path}')
        r['summary']=json.loads(path.read_text())
    if a.download and not Path(a.data_root).exists():download_corruptions(Path(a.data_root).parent)
    execute({'references':refs},Path(a.data_root))
if __name__=='__main__':main()
