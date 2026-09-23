"""Count partial arithmetic cost; original training/evaluation records GPU latency."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from src.models.factory import create_model
from src.final_experiments.efficiency import count


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    torch.set_num_threads(2);cfg=yaml.safe_load(Path(a.config).read_text())
    print(json.dumps({**count(create_model(cfg['model'])),'note':'Conv2d/Linear only. Total FLOPs unavailable. Use evaluate for synchronized checkpoint latency.'},indent=2))
if __name__=='__main__':main()
