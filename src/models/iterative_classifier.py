"""Recurrent refinement of the final ResNet feature map."""
import math
import torch
from torch import nn
from src.models.baseline import build_model
from src.models.refinement import IterativeRefinementBlock


class IterativeClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, refinement_iterations: int = 3,
                 residual_scale: float = 1.0, share_refinement_weights: bool = True):
        super().__init__()
        if type(refinement_iterations) is not int or refinement_iterations < 0:
            raise ValueError("refinement_iterations must be a nonnegative integer")
        if type(share_refinement_weights) is not bool:
            raise ValueError("share_refinement_weights must be boolean")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        # Build before refinement to preserve baseline initialization for a seed.
        baseline = build_model(num_classes)
        self.backbone = nn.Sequential(*list(baseline.children())[:-2])
        self.pool = baseline.avgpool
        self.classifier = baseline.fc
        self.refinement_iterations = refinement_iterations
        self.share_refinement_weights = share_refinement_weights
        self.residual_scale = residual_scale
        # Keep one shared block even for T=0: shared parameter count is invariant
        # to T, though that block is unused (and has no gradient) at T=0.
        count = 1 if share_refinement_weights else refinement_iterations
        # Extra parameter initialization must not shift the random stream used
        # by training augmentation relative to the same-seed baseline.
        with torch.random.fork_rng(devices=[]):
            self.refinement_blocks = nn.ModuleList(
                IterativeRefinementBlock(512, residual_scale) for _ in range(count))

    def classify(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(state).flatten(1))

    def forward(self, x: torch.Tensor, return_intermediates: bool = False,
                return_features: bool = False) -> torch.Tensor | dict:
        """Detailed outputs retain graphs; use inference_mode for analysis.

        Features imply detailed output. Lists include S0/p0 through ST/pT.
        Normal training retains no diagnostic lists and supervises only pT.
        """
        detailed = return_intermediates or return_features
        state = self.backbone(x)
        predictions = [self.classify(state)] if detailed else None
        states = [state] if return_features else None
        for t in range(self.refinement_iterations):
            block = self.refinement_blocks[0 if self.share_refinement_weights else t]
            state = block(state)
            if detailed:
                predictions.append(self.classify(state))
            if return_features:
                states.append(state)
        if not detailed:
            return self.classify(state)
        return {"final_logits": predictions[-1], "initial_logits": predictions[0],
                "intermediate_logits": predictions, "feature_states": states}
