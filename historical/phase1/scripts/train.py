"""Run from the repository root using python -m scripts.train."""
import argparse
import torch
from src.utils.config import load_config
from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.download:
        config["data"]["download"] = True
    if args.smoke:
        config["training"].update(epochs=1, batch_size=4)
        config["data"]["workers"] = 0
        config["experiment_name"] += "-smoke"
        torch.set_num_threads(2)
    train(config, args.smoke)


if __name__ == "__main__":
    main()
