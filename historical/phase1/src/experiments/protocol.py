"""Frozen Phase 1 model/seed matrix; no model-specific optimization."""
import copy
from pathlib import Path
from src.utils.config import load_config

MODELS={'A':('baseline','baseline.yaml'),'B':('refinement-t3','refinement.yaml'),
        'C':('cellular-t3','cellular.yaml'),'D':('uncertainty-cellular-t3','uncertainty_cellular.yaml')}
PARAMETERS={'A':11173962,'B':11700298,'C':11442762,'D':11442762}


def matrix_configs(root: Path, smoke=False, seeds=(1,2,3)) -> list[dict]:
    baseline=load_config('configs/baseline.yaml')
    if baseline['training']['epochs'] != 200 or baseline['data']['validation_size'] != 5000:
        raise ValueError('Phase 1 requires the frozen 200-epoch, 45000/5000 protocol')
    entries=[]
    for label,(name,filename) in MODELS.items():
        template=load_config(str(Path('configs')/filename))
        for key in ['training','data','deterministic','device']:
            if template[key]!=baseline[key]: raise ValueError(f'Unfair {key} settings in {label}')
        m=template['model']
        if label!='A' and (m.get('communication_iterations',m.get('refinement_iterations'))!=3 or m['residual_scale']!=1.):
            raise ValueError('Phase 1 requires T=3 and residual_scale=1')
        if label in {'C','D'} and not m['share_communication_weights']: raise ValueError('Shared communication required')
        if label=='B' and not m['share_refinement_weights']: raise ValueError('Shared refinement required')
        if label=='D' and (m['gate_mode']!='entropy' or not m['detach_uncertainty_gate']): raise ValueError('Detached entropy required')
        for seed in seeds:
            config=copy.deepcopy(template)
            config.update(seed=seed,experiment_name=f'cifar10-{label}-{name}-seed{seed}',output_root=str(root))
            if smoke:
                config['training'].update(epochs=1,batch_size=4)
                config['device']='cpu'
                config['data']['workers']=0
            entries.append({'id':config['experiment_name'],'model':label,'seed':seed,'config':config})
    return entries
