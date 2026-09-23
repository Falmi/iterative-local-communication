"""YAML configuration and deliberately small validation surface."""
from pathlib import Path
import math
import yaml
from src.models.uncertainty_cellular_classifier import validate_uncertainty_options


def load_config(path: str) -> dict:
    with open(path) as stream:
        config = yaml.safe_load(stream)
    model = config["model"]
    if model["name"] not in {"resnet18_cifar", "iterative_resnet18", "cellular_resnet18", "uncertainty_cellular_resnet18"} or model["num_classes"] != 10:
        raise ValueError("Supported models: resnet18_cifar, iterative_resnet18, cellular_resnet18, uncertainty_cellular_resnet18; 10 classes")
    allowed = {"name", "num_classes"}
    if model["name"] == "iterative_resnet18":
        allowed |= {"refinement_iterations", "residual_scale", "share_refinement_weights"}
        model.setdefault("refinement_iterations", 3)
        model.setdefault("residual_scale", 1.0)
        model.setdefault("share_refinement_weights", True)
        if type(model["refinement_iterations"]) is not int or model["refinement_iterations"] < 0:
            raise ValueError("refinement_iterations must be a nonnegative integer")
        scale = model["residual_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        if type(model["share_refinement_weights"]) is not bool:
            raise ValueError("share_refinement_weights must be boolean")
    if model["name"] in {"cellular_resnet18", "uncertainty_cellular_resnet18"}:
        allowed |= {"communication_iterations", "residual_scale", "neighbourhood_kernel_size",
                    "share_communication_weights", "communication_type"}
        model.setdefault("communication_iterations", 3)
        model.setdefault("residual_scale", 1.0)
        model.setdefault("neighbourhood_kernel_size", 3)
        model.setdefault("share_communication_weights", True)
        model.setdefault("communication_type", "depthwise_local")
        if type(model["communication_iterations"]) is not int or model["communication_iterations"] < 0:
            raise ValueError("communication_iterations must be a nonnegative integer")
        scale = model["residual_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        if type(model["share_communication_weights"]) is not bool:
            raise ValueError("share_communication_weights must be boolean")
        if type(model["neighbourhood_kernel_size"]) is not int or model["neighbourhood_kernel_size"] != 3:
            raise ValueError("Milestone 3 supports only a 3x3 neighbourhood")
        if model["communication_type"] != "depthwise_local":
            raise ValueError("Only depthwise_local communication is implemented")
    if model["name"] == "uncertainty_cellular_resnet18":
        defaults = dict(uncertainty_type="predictive_entropy", normalize_uncertainty=True,
                        detach_uncertainty_gate=True, gate_mode="entropy", fixed_gate_value=1.0)
        allowed |= defaults.keys()
        for key, value in defaults.items():
            model.setdefault(key, value)
        validate_uncertainty_options(**{key: model[key] for key in defaults})
    if set(model) - allowed:
        raise ValueError(f"Unknown model options: {set(model) - allowed}")
    t = config["training"]
    if t["epochs"] < 1 or t["batch_size"] < 1 or t["learning_rate"] <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if not 0 < config["data"]["validation_size"] < 50000:
        raise ValueError("validation_size must lie between 1 and 49999")
    return config


def save_config(config: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False))
