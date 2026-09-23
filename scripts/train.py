"""Portable entry point selecting the original training engine for each run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import torch
import yaml


def execute(entry, download=False, smoke=False, resume=False):
    cfg=yaml.safe_load(Path(entry['config']).read_text())
    cfg['data']['download']=download
    torch.set_num_threads(2 if smoke else 6)
    if smoke:
        cfg['device']='cpu';cfg['training'].update(epochs=1,batch_size=4)
        cfg['experiment_name']+='-smoke';cfg['data']['workers']=0
    elif not torch.cuda.is_available():
        raise RuntimeError('Real experiments require CUDA; no CPU fallback')
    if entry['engine']=='historical':
        from src.training.trainer import train
    elif entry['engine']=='phase4':
        from src.experiments.phase4_training import train
    else:
        from src.final_experiments.training import train
    run=Path('runs')/(entry['id']+('-smoke' if smoke else ''))
    if run.exists() and (run/'status.json').exists():
        status=json.loads((run/'status.json').read_text()).get('status')
        if status in ('completed','failed'):
            print('Preserving terminal run:',run,status);return
    if run.exists() and not resume:raise FileExistsError(f'{run}: use --resume for an interrupted run')
    try:
        train(cfg,smoke=smoke,run_dir=run,resume=run.exists())
    except Exception as error:
        from src.experiments.stability import NumericalFailure as Phase4Failure
        from src.final_experiments.training import NumericalFailure as FinalFailure
        known=isinstance(error,(Phase4Failure,FinalFailure)) or str(error)=='Non-finite training loss'
        if known:
            run.mkdir(parents=True,exist_ok=True)
            (run/'status.json').write_text(json.dumps({'status':'failed','error':str(error),'replacement_allowed':False},indent=2))
            print('Retained numerical failure:',run)
        else:raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--download',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();entries=json.loads(Path('configs/index.json').read_text())
    entry=next((e for e in entries if Path(e['config'])==Path(a.config)),None)
    if entry is None:raise ValueError('Use a recorded configuration from configs/index.json')
    execute(entry,a.download,a.smoke,a.resume)
if __name__=='__main__':main()
