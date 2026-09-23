"""Pointwise residual refinement, with no new spatial mixing."""
import math
import torch
from torch import nn


class ChannelLayerNorm(nn.LayerNorm):
    """Normalize channels independently at each spatial location."""

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return super().forward(state.movedim(1, -1)).movedim(-1, 1)


class IterativeRefinementBlock(nn.Module):
    """S_next = S + residual_scale * F(S), using a pointwise F."""

    def __init__(self, channels: int = 512, residual_scale: float = 1.0):
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        self.residual_scale = residual_scale
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            ChannelLayerNorm(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            ChannelLayerNorm(channels),
        )
        # Small, nonzero output scale keeps initial updates near identity while
        # allowing gradients into both convolutions on the first backward pass.
        nn.init.constant_(self.transform[-1].weight, 0.01)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.residual_scale * self.transform(state)
