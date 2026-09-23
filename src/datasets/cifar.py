"""Disjoint seeded CIFAR-10 partitions with training-only augmentation."""
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from src.utils.seed import seed_worker

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def make_loader(config: dict, split: str, smoke: bool = False) -> DataLoader:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    seed = config["seed"]
    generator = torch.Generator().manual_seed(seed)
    if smoke:
        synthetic = torch.Generator().manual_seed(seed + {"train": 0, "validation": 1, "test": 2}[split])
        dataset = TensorDataset(torch.randn(8, 3, 32, 32, generator=synthetic),
                                torch.randint(10, (8,), generator=synthetic))
    else:
        ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if split == "train" else []
        transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        dataset = datasets.CIFAR10(config["data"]["root"], train=split != "test",
                                  download=config["data"]["download"], transform=transform)
        if split != "test":
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
            n = config["data"]["validation_size"]
            dataset = Subset(dataset, indices[n:] if split == "train" else indices[:n])
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=split == "train",
                      num_workers=config["data"]["workers"], worker_init_fn=seed_worker,
                      generator=generator)
