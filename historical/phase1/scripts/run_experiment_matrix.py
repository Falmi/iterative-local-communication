"""Prepare or execute the fixed A-D/three-seed protocol using the shared trainer."""
import argparse
import fcntl
import json
import shutil
from pathlib import Path
import torch
from src.experiments.protocol import matrix_configs
from src.experiments.metadata import write_json,source_hashes
from src.experiments.aggregate import aggregate,validate_run
from src.training.trainer import train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--execute',action='store_true',help='Without this flag, prepare and report only')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--download',action='store_true',help='Permit CIFAR download before training')
    parser.add_argument('--allow-cpu',action='store_true',help='Explicitly authorize the expensive full CPU matrix')
    parser.add_argument('--smoke',action='store_true',help='Separate synthetic matrix, never real-data evidence')
    args=parser.parse_args()
    root=args.root or Path('outputs/experiment_phase1_smoke' if args.smoke else 'outputs/experiment_phase1')
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    # OS lock is released even after a killed process; avoids concurrent writers.
    with (root/'matrix.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another process owns this matrix')
        entries=matrix_configs(root,args.smoke)
        proposed={'seeds':[1,2,3],'smoke':args.smoke,'runs':entries,'source_hashes':source_hashes(),
                  'split_rule':'seeded randperm of official training indices; first 5000 validation',
                  'selection':'first highest validation accuracy', 'test_rule':'one scored pass of best checkpoint',
                  'resource_note':'CUDA unavailable at preparation' if not torch.cuda.is_available() else 'CUDA available'}
        manifest_path=root/'manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            for key in ['seeds','smoke','runs','source_hashes']:
                if manifest[key]!=proposed[key]: raise SystemExit(f'Frozen matrix {key} differs. Preserve it and use a new root for a new protocol.')
        else:
            write_json(manifest_path,proposed)
            for filename in proposed['source_hashes']:
                target=root/'source_snapshot'/filename;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(filename,target)
        if not args.execute:
            result=aggregate(root);print(f"Prepared {root}: {result['completed_runs']}/12 completed; no training launched")
            return
        if not args.smoke and not torch.cuda.is_available() and not args.allow_cpu:
            write_json(root/'execution_status.json',{'status':'blocked','reason':'CUDA unavailable; full CPU matrix requires explicit --allow-cpu'})
            aggregate(root)
            raise SystemExit('CUDA unavailable. Matrix prepared; no full runs launched. Use a CUDA host or explicitly authorize CPU with --allow-cpu.')
        if args.smoke: torch.set_num_threads(2)
        if args.download and not args.smoke:
            from torchvision.datasets import CIFAR10
            data=entries[0]['config']['data']['root']
            CIFAR10(data,train=True,download=True);CIFAR10(data,train=False,download=True)
        failures=[]
        for entry in entries:
            run=root/entry['id']
            if (run/'summary.json').exists():
                validate_run(run,entry,args.smoke)
                print(f"Skipping validated completed run {entry['id']}",flush=True);continue
            if run.exists() and not args.resume:
                raise SystemExit(f"Incomplete {run}; use --resume to continue without overwriting")
            print(f"Starting {entry['id']}",flush=True)
            try:
                train(entry['config'],smoke=args.smoke,run_dir=run,resume=run.exists())
            except Exception as error:
                run.mkdir(exist_ok=True)
                write_json(run/'status.json',{'status':'failed','error':repr(error)})
                failures.append(entry['id']);print(f"Failed {entry['id']}: {error}",flush=True)
            except KeyboardInterrupt:
                if run.exists(): write_json(run/'status.json',{'status':'interrupted','resume':'last completed epoch'})
                aggregate(root);raise
            aggregate(root)
        result=aggregate(root)
        write_json(root/'execution_status.json',{'status':result['status'],'failed_runs':failures})
        print(f"Matrix {result['status']}: {result['completed_runs']}/12",flush=True)
        if failures: raise SystemExit(1)


if __name__=='__main__': main()
