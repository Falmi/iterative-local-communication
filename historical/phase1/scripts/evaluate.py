"""Evaluate a trusted local checkpoint on the held-out test partition."""
import argparse
import json
import torch
from src.models.factory import create_model
from src.datasets.cifar import make_loader
from src.evaluation.evaluator import evaluate
from src.utils.seed import seed_everything, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["data"]["download"] = args.download
    seed_everything(config["seed"], config["deterministic"])
    if args.smoke:
        torch.set_num_threads(2)
    model = create_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, make_loader(config, "test", args.smoke), device)
    result.update(uncertainty_type=config["model"].get("uncertainty_type", "none"),
                  gate_mode=config["model"].get("gate_mode"),
                  model_name=config["model"]["name"],
                  refinement_iterations=config["model"].get("communication_iterations", config["model"].get("refinement_iterations", 0)),
                  communication_iterations=config["model"].get("communication_iterations", 0),
                  communication_type=config["model"].get("communication_type", "none"),
                  parameters=sum(p.numel() for p in model.parameters()),
                  dataset="synthetic_smoke" if args.smoke else "CIFAR-10")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
