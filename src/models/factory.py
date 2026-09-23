"""Configuration dispatch preserves the original baseline state_dict layout."""
from torch import nn
from src.models.uncertainty_cellular_classifier import UncertaintyGuidedCellularClassifier
from src.models.baseline import build_model
from src.models.iterative_classifier import IterativeClassifier
from src.models.cellular_classifier import CellularClassifier


def create_model(config: dict) -> nn.Module:
    options = dict(config)
    name = options.pop("name")
    if name == "resnet18_cifar":
        return build_model(**options)
    if name == "iterative_resnet18":
        return IterativeClassifier(**options)
    if name == "cellular_resnet18":
        return CellularClassifier(**options)
    if name == "uncertainty_cellular_resnet18":
        return UncertaintyGuidedCellularClassifier(**options)
    raise ValueError(f"Unknown model: {name}")
