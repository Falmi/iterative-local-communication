"""Analyze trusted iterative or cellular checkpoints on ordered held-out test samples."""
import argparse
import json
from pathlib import Path
import torch
from src.analysis.refinement_analysis import analyze_refinement
from src.final_experiments.data import make_loader
from src.models.factory import create_model
from src.utils.seed import seed_everything, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--messages", action="store_true", help="Cellular message norms; automatic for Model D; implies --features")
    parser.add_argument("--samples", action="store_true", help="Include individual test-row records")
    parser.add_argument("--output", help="JSON path; defaults beside checkpoint")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(config["seed"], config["deterministic"])
    config["data"]["download"] = args.download
    if args.smoke:
        torch.set_num_threads(2)
    model = create_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    result = analyze_refinement(model, make_loader(config, "test", args.smoke), device,
                                feature_analysis=args.features, collect_samples=args.samples,
                                message_analysis=args.messages)
    result.update(dataset="synthetic_smoke" if args.smoke else config.get("dataset", "CIFAR-10"), seed=config["seed"],
                  checkpoint=str(Path(args.checkpoint).resolve()), model=config["model"],
                  parameters=sum(p.numel() for p in model.parameters()))
    destination = Path(args.output) if args.output else Path(args.checkpoint).with_name("refinement_analysis.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2, allow_nan=False))
    print(f"Analysis saved to {destination}")


if __name__ == "__main__":
    main()
