"""Evaluation; latency measures synchronized forward passes only."""
import time
import torch
from src.evaluation.metrics import ClassificationMetrics


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    metrics = ClassificationMetrics()
    elapsed = 0.0
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        if i == 0:
            model(images)  # unmeasured warm-up
        synchronize(device)
        start = time.perf_counter()
        logits = model(images)
        synchronize(device)
        elapsed += time.perf_counter() - start
        metrics.update(logits, targets)
    result = metrics.compute()
    result["inference_latency_ms_per_sample"] = 1000 * elapsed / result["samples"]
    return result
