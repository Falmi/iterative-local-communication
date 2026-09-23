import torch
from src.models.baseline import build_model
from src.evaluation.metrics import ClassificationMetrics
from src.datasets.cifar import make_loader
from src.utils.config import load_config
from src.utils.seed import seed_everything


def test_forward_backward():
    torch.set_num_threads(2)
    seed_everything(1, True)
    model = build_model()
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics():
    metrics = ClassificationMetrics()
    metrics.update(torch.zeros(2, 10), torch.tensor([0, 1]))
    result = metrics.compute()
    assert abs(result["NLL"] - 2.302585) < 1e-5
    assert abs(result["Brier"] - 0.9) < 1e-5
    assert abs(result["ECE"] - 0.4) < 1e-5
    assert result["accuracy"] == 0.5


def test_seeded_smoke_loader():
    config = load_config("configs/baseline.yaml")
    a = next(iter(make_loader(config, "train", True)))
    b = next(iter(make_loader(config, "train", True)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    c = next(iter(make_loader(config, "test", True)))
    assert not torch.equal(a[0], c[0])


def test_cifar_split_and_transforms(monkeypatch):
    from torchvision import transforms
    from src.datasets import cifar

    class MockCIFAR:
        def __init__(self, root, train, download, transform):
            self.transform = transform
            self.train = train

        def __len__(self):
            return 50000 if self.train else 10000

    monkeypatch.setattr(cifar.datasets, "CIFAR10", MockCIFAR)
    config = load_config("configs/baseline.yaml")
    train = make_loader(config, "train").dataset
    validation = make_loader(config, "validation").dataset
    test = make_loader(config, "test").dataset
    assert len(train) == 45000 and len(validation) == 5000 and len(test) == 10000
    assert set(train.indices).isdisjoint(validation.indices)
    assert len(set(train.indices) | set(validation.indices)) == 50000
    assert train.indices == make_loader(config, "train").dataset.indices
    assert isinstance(train.dataset.transform.transforms[0], transforms.RandomCrop)
    assert isinstance(validation.dataset.transform.transforms[0], transforms.ToTensor)
    assert not test.train
