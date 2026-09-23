"""Validate completed runs and regenerate both JSON and Markdown reports."""
import argparse
from pathlib import Path
from src.experiments.aggregate import aggregate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('outputs/experiment_phase1'))
    args=parser.parse_args()
    result=aggregate(args.root.resolve())
    print(f"{result['completed_runs']}/{result['expected_runs']} validated; {result['status']}")


if __name__=='__main__': main()
