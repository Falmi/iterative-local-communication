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
    torch.set_num_threads(6)
    from src.evaluation.run_records import corruption_references
    refs=corruption_references(a.checkpoints_root)
    if a.download and not Path(a.data_root).exists():download_corruptions(Path(a.data_root).parent)
    execute({'references':refs},Path(a.data_root))
if __name__=='__main__':main()
