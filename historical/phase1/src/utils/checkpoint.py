"""Atomic checkpoints include training and random-generator state."""
import os
import random
import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, config, loader, training_state=None):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val_accuracy": best,
             "config": config, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "loader": loader.generator.get_state()}}
    state["training_state"] = training_state
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
