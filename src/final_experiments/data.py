import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,Subset,TensorDataset,DataLoader
from torchvision import datasets,transforms
from src.datasets.cifar import make_loader as cifar10_loader,MEAN,STD
from src.utils.seed import seed_worker

C100_MEAN=(.5071,.4867,.4408)
C100_STD=(.2675,.2565,.2761)
CORRUPTIONS=('gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur',
    'zoom_blur','snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression',
    'speckle_noise','gaussian_blur','spatter','saturate')


def make_loader(config,split,smoke=False):
    if config.get('dataset','CIFAR-10')=='CIFAR-10':return cifar10_loader(config,split,smoke)
    seed=config['seed'];g=torch.Generator().manual_seed(seed)
    if smoke:
        rng=torch.Generator().manual_seed(seed+{'train':0,'validation':1,'test':2}[split])
        ds=TensorDataset(torch.randn(8,3,32,32,generator=rng),torch.randint(100,(8,),generator=rng))
    else:
        ops=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()] if split=='train' else []
        ds=datasets.CIFAR100(config['data']['root'],train=split!='test',download=config['data']['download'],
            transform=transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize(C100_MEAN,C100_STD)]))
        if split!='test':
            indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(seed)).tolist()
            n=config['data']['validation_size'];ds=Subset(ds,indices[n:] if split=='train' else indices[:n])
    return DataLoader(ds,batch_size=config['training']['batch_size'],shuffle=split=='train',num_workers=config['data']['workers'],worker_init_fn=seed_worker,generator=g)


def download_corruptions(root):
    """Canonical archive only, checksum from the Zenodo record, safe extraction."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/2535967') as response:record=json.load(response)
    item=next(f for f in record['files'] if f['key']=='CIFAR-10-C.tar')
    archive=root/'CIFAR-10-C.tar';tmp=archive.with_suffix('.partial')
    algo,expected=item['checksum'].split(':')
    def digest(p):
        h=hashlib.new(algo)
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    if not archive.exists() or digest(archive)!=expected:
        urllib.request.urlretrieve(item['links']['self'],tmp)
        if digest(tmp)!=expected:raise ValueError('Canonical archive checksum failed')
        tmp.replace(archive)
    with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
    (root/'cifar10c_provenance.json').write_text(json.dumps({'record':record['id'],'file':item,'verified_checksum':expected},indent=2))


class CorruptionDataset(Dataset):
    def __init__(self,root,corruption,severity):
        if corruption not in CORRUPTIONS or severity not in range(1,6):raise ValueError('Invalid corruption cell')
        root=Path(root);self.images=np.load(root/(corruption+'.npy'),mmap_mode='r')
        labels=np.load(root/'labels.npy',mmap_mode='r')
        if self.images.shape!=(50000,32,32,3) or self.images.dtype!=np.uint8:raise ValueError('Unexpected canonical image layout')
        if labels.shape not in [(10000,),(50000,)]:raise ValueError('Unexpected labels shape')
        self.offset=(severity-1)*10000
        self.labels=labels if len(labels)==10000 else labels[self.offset:self.offset+10000]
        if not np.issubdtype(labels.dtype,np.integer) or labels.min()<0 or labels.max()>9:raise ValueError('Invalid labels')
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self):return 10000
    def __getitem__(self,i):return self.transform(np.array(self.images[self.offset+i],copy=True)),int(self.labels[i])
