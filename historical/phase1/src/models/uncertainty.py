"""Parameter-free predictive entropy and sample-level control signals."""
import math
import torch


def predictive_entropy_from_logits(logits: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Entropy over the last dimension; output excludes the class dimension.

    Promote low precision inputs to float32. Clamp log-probabilities before
    multiplication to avoid 0 * -inf for extreme finite logits.
    """
    if logits.shape[-1] < 2:
        raise ValueError("Entropy requires at least two classes")
    values = logits if logits.dtype == torch.float64 else logits.float()
    log_p = values.log_softmax(-1)
    entropy = -(log_p.exp() * log_p.clamp_min(torch.finfo(values.dtype).min)).sum(-1)
    return (entropy / math.log(logits.shape[-1])).clamp(0, 1) if normalized else entropy


def communication_gate(uncertainty: torch.Tensor, detach: bool = True,
                       mode: str = 'entropy', fixed_value: float = 1.0) -> torch.Tensor:
    """Return one scalar per sample, broadcastable to NCHW."""
    if mode == 'entropy':
        gate = uncertainty.detach() if detach else uncertainty
    elif mode == 'fixed':
        if not math.isfinite(fixed_value) or not 0 <= fixed_value <= 1:
            raise ValueError("fixed gate must lie in [0,1]")
        gate = torch.full_like(uncertainty, fixed_value)
    else:
        raise ValueError("gate_mode must be entropy or fixed")
    return gate.reshape(-1, 1, 1, 1)


def gated_state_update(state: torch.Tensor, message: torch.Tensor,
                       gate: torch.Tensor, residual_scale: float) -> torch.Tensor:
    return state + residual_scale * gate * message
