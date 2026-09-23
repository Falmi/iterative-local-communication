"""Model C: recurrent local communication after the final ResNet stage."""
import math
import torch
from torch import nn
from src.models.baseline import build_model
from src.models.cellular_communication import LocalCellularCommunicationBlock


class CellularClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, communication_iterations: int = 3,
                 residual_scale: float = 1.0, neighbourhood_kernel_size: int = 3,
                 share_communication_weights: bool = True,
                 communication_type: str = "depthwise_local"):
        super().__init__()
        if type(communication_iterations) is not int or communication_iterations < 0:
            raise ValueError("communication_iterations must be a nonnegative integer")
        if type(share_communication_weights) is not bool:
            raise ValueError("share_communication_weights must be boolean")
        if communication_type != "depthwise_local":
            raise ValueError("Only depthwise_local communication is implemented")
        if type(neighbourhood_kernel_size) is not int or neighbourhood_kernel_size != 3:
            raise ValueError("Milestone 3 supports only a 3x3 neighbourhood")
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        baseline = build_model(num_classes)
        self.backbone = nn.Sequential(*list(baseline.children())[:-2])
        self.pool = baseline.avgpool
        self.classifier = baseline.fc
        self.communication_iterations = communication_iterations
        self.share_communication_weights = share_communication_weights
        self.communication_type = communication_type
        self.residual_scale = residual_scale
        count = 1 if share_communication_weights else communication_iterations
        # Preserve baseline/Model B initialization and subsequent CPU RNG draws.
        # As in B, shared T=0 retains one unused block for parameter invariance.
        with torch.random.fork_rng(devices=[]):
            self.communication_blocks = nn.ModuleList(
                LocalCellularCommunicationBlock(512, residual_scale, neighbourhood_kernel_size)
                for _ in range(count))

    def classify(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(state).flatten(1))

    def forward(self, x: torch.Tensor, return_intermediates: bool = False,
                return_features: bool = False, return_messages: bool = False) -> torch.Tensor | dict:
        """Opt-in lists retain graphs; analysis should use inference_mode.

        States/logits include t0 (T+1 entries); messages have T entries. Messages
        are unscaled. Normal final-only training does not create diagnostic lists.
        """
        detailed = return_intermediates or return_features or return_messages
        state = self.backbone(x)
        predictions = [self.classify(state)] if detailed else None
        states = [state] if return_features else None
        messages = [] if return_messages else None
        for t in range(self.communication_iterations):
            block = self.communication_blocks[0 if self.share_communication_weights else t]
            if return_messages:
                state, message = block(state, return_message=True)
                messages.append(message)
            else:
                state = block(state)
            if detailed:
                predictions.append(self.classify(state))
            if return_features:
                states.append(state)
        if not detailed:
            return self.classify(state)
        return {"initial_logits": predictions[0], "final_logits": predictions[-1],
                "intermediate_logits": predictions, "feature_states": states, "messages": messages}
