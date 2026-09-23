"""Sample-weighted classification and calibration metrics."""
import torch
import torch.nn.functional as F


class ClassificationMetrics:
    def __init__(self, bins: int = 15):
        self.n = 0
        self.loss = self.correct = self.brier = 0.0
        self.count = torch.zeros(bins, dtype=torch.float64)
        self.confidence = torch.zeros_like(self.count)
        self.accuracy = torch.zeros_like(self.count)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits, targets = logits.detach().float().cpu(), targets.detach().cpu()
        p = logits.softmax(1)
        confidence, prediction = p.max(1)
        correct = prediction.eq(targets)
        self.n += len(targets)
        self.loss += F.cross_entropy(logits, targets, reduction="sum").item()
        self.correct += correct.sum().item()
        self.brier += (p - F.one_hot(targets, logits.shape[1])).square().sum().item()
        indices = (confidence * len(self.count)).long().clamp(max=len(self.count) - 1)
        self.count += torch.bincount(indices, minlength=len(self.count))
        self.confidence.scatter_add_(0, indices, confidence.double())
        self.accuracy.scatter_add_(0, indices, correct.double())

    def compute(self) -> dict:
        if not self.n:
            raise ValueError("Cannot evaluate an empty loader")
        return {"accuracy": self.correct / self.n, "loss": self.loss / self.n,
                "NLL": self.loss / self.n, "Brier": self.brier / self.n,
                "ECE": (self.confidence - self.accuracy).abs().sum().item() / self.n,
                "samples": self.n}
