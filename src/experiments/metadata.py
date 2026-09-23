"""Atomic records and auditable software/data fingerprints."""
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import torch


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict:
    paths = sorted([*Path('src').rglob('*.py'), *Path('scripts').rglob('*.py'),
                    *Path('configs').glob('*.yaml'), Path('requirements.txt')])
    return {str(p): file_hash(p) for p in paths}


def environment(device: torch.device, smoke: bool) -> dict:
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    return {'python':platform.python_version(), 'packages':{k:version(k) for k in ['torch','torchvision','numpy','PyYAML']},
            'device':str(device), 'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'cuda':torch.version.cuda, 'platform':platform.platform(), 'processor':platform.processor(),
            'cpu_count':os.cpu_count(), 'torch_threads':torch.get_num_threads(), 'git_commit':commit,
            'smoke':smoke, 'source_hashes':source_hashes()}


def split_metadata(training, validation, seed: int, smoke: bool) -> dict:
    def digest(dataset):
        indices = getattr(dataset, 'indices', list(range(len(dataset))))
        return hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    return {'seed':seed, 'dataset':'synthetic_smoke' if smoke else 'CIFAR-10',
            'train_samples':len(training.dataset), 'validation_samples':len(validation.dataset),
            'train_indices_sha256':digest(training.dataset), 'validation_indices_sha256':digest(validation.dataset)}
