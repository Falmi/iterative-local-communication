"""Model D: Model C's exact modules with an entropy-derived update gate."""
import math
import torch
from src.models.cellular_classifier import CellularClassifier
from src.models.uncertainty import predictive_entropy_from_logits, communication_gate, gated_state_update


def validate_uncertainty_options(uncertainty_type='predictive_entropy', normalize_uncertainty=True,
                                 detach_uncertainty_gate=True, gate_mode='entropy', fixed_gate_value=1.0):
    if uncertainty_type != 'predictive_entropy' or normalize_uncertainty is not True:
        raise ValueError("Model D currently requires normalized predictive entropy")
    if type(detach_uncertainty_gate) is not bool:
        raise ValueError("detach_uncertainty_gate must be boolean")
    if gate_mode not in {'entropy', 'fixed'}:
        raise ValueError("gate_mode must be entropy or fixed")
    if isinstance(fixed_gate_value, bool) or not isinstance(fixed_gate_value, (int, float)) or not math.isfinite(fixed_gate_value) or not 0 <= fixed_gate_value <= 1:
        raise ValueError("fixed_gate_value must be finite and in [0,1]")


class UncertaintyGuidedCellularClassifier(CellularClassifier):
    def __init__(self, num_classes=10, communication_iterations=3, residual_scale=1.0,
                 neighbourhood_kernel_size=3, share_communication_weights=True,
                 communication_type='depthwise_local', uncertainty_type='predictive_entropy',
                 normalize_uncertainty=True, detach_uncertainty_gate=True,
                 gate_mode='entropy', fixed_gate_value=1.0):
        validate_uncertainty_options(uncertainty_type, normalize_uncertainty, detach_uncertainty_gate,
                                     gate_mode, fixed_gate_value)
        super().__init__(num_classes, communication_iterations, residual_scale,
                         neighbourhood_kernel_size, share_communication_weights, communication_type)
        self.uncertainty_type = uncertainty_type
        self.normalize_uncertainty = normalize_uncertainty
        self.detach_uncertainty_gate = detach_uncertainty_gate
        self.gate_mode = gate_mode
        self.fixed_gate_value = fixed_gate_value

    def forward(self, x: torch.Tensor, return_intermediates=False,
                return_features=False, return_messages=False) -> torch.Tensor | dict:
        """T+1 predictions, T controls; no diagnostic lists in normal training."""
        detailed = return_intermediates or return_features or return_messages
        state = self.backbone(x)
        logits = self.classify(state)
        predictions = [logits] if detailed else None
        states = [state] if return_features else None
        messages = [] if return_messages else None
        uncertainties, gates = ([], []) if detailed else (None, None)
        for t in range(self.communication_iterations):
            uncertainty = predictive_entropy_from_logits(logits)
            gate = communication_gate(uncertainty, self.detach_uncertainty_gate,
                                      self.gate_mode, self.fixed_gate_value)
            block = self.communication_blocks[0 if self.share_communication_weights else t]
            message = block.collect_message(state)
            state = gated_state_update(state, message, gate, self.residual_scale)
            logits = self.classify(state)
            if detailed:
                predictions.append(logits)
                uncertainties.append(uncertainty)
                gates.append(gate)
            if return_features:
                states.append(state)
            if return_messages:
                messages.append(message)
        if not detailed:
            return logits
        return {'initial_logits': predictions[0], 'final_logits': logits,
                'intermediate_logits': predictions, 'feature_states': states,
                'messages': messages, 'uncertainties': uncertainties, 'gates': gates}
