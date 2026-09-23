"""Local learned messages between spatial feature units."""
import math
import torch
from torch import nn
from src.models.refinement import ChannelLayerNorm


class LocalCellularCommunicationBlock(nn.Module):
    """Depthwise neighbourhood aggregation followed by pointwise transformation.

    A cell is one C-vector in an NCHW state. Zero padding retains H and W;
    the learned 3x3 kernel includes the central cell as well as its neighbours.
    Returned messages are unscaled: next_state = state + residual_scale * message.
    """

    def __init__(self, channels: int = 512, residual_scale: float = 1.0,
                 neighbourhood_kernel_size: int = 3):
        super().__init__()
        if type(neighbourhood_kernel_size) is not int or neighbourhood_kernel_size != 3:
            raise ValueError("Milestone 3 supports only a 3x3 neighbourhood")
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        self.residual_scale = residual_scale
        self.neighbourhood_aggregation = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.normalization = ChannelLayerNorm(channels)
        self.activation = nn.ReLU()
        self.message_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), ChannelLayerNorm(channels))
        # Match Model B: small nonzero residual messages, with gradients through
        # aggregation and transformation from the very first backward pass.
        nn.init.constant_(self.message_transform[-1].weight, 0.01)
        nn.init.zeros_(self.message_transform[-1].bias)

    def collect_message(self, state: torch.Tensor) -> torch.Tensor:
        neighbours = self.neighbourhood_aggregation(state)
        return self.message_transform(self.activation(self.normalization(neighbours)))

    def update_state(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return state + self.residual_scale * message

    def forward(self, state: torch.Tensor, return_message: bool = False):
        message = self.collect_message(state)
        next_state = self.update_state(state, message)
        return (next_state, message) if return_message else next_state
