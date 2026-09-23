"""Streaming iteration metrics; raw feature maps never leave the current batch."""
import torch
from src.evaluation.metrics import ClassificationMetrics
from src.models.uncertainty import predictive_entropy_from_logits
from src.models.uncertainty_cellular_classifier import UncertaintyGuidedCellularClassifier
from src.analysis.uncertainty_analysis import summarize_uncertainty
from src.models.iterative_classifier import IterativeClassifier
from src.models.cellular_classifier import CellularClassifier


def prediction_statistics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept [..., classes]; entropy is in natural-log units (nats)."""
    log_p = logits.float().log_softmax(-1)
    probabilities = log_p.exp()
    confidence, predictions = probabilities.max(-1)
    entropy = predictive_entropy_from_logits(logits, normalized=False)
    return predictions, confidence, entropy


def outcome_masks(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """Endpoint categories for predictions shaped [T+1, batch].

    Stable means endpoint correctness agrees, not that all steps are unchanged.
    """
    initial = predictions[0].eq(targets)
    final = predictions[-1].eq(targets)
    return {"corrected": ~initial & final, "damaged": initial & ~final,
            "stable_correct": initial & final, "stable_wrong": ~initial & ~final}


@torch.inference_mode()
def analyze_refinement(model: IterativeClassifier | CellularClassifier, loader, device: torch.device,
                       feature_analysis: bool = False, collect_samples: bool = False,
                       message_analysis: bool = False) -> dict:
    """Analyze a non-shuffled evaluation loader; sample_index is its row order.

    Accuracy, confidence, and change rates are fractions. Feature diagnostics
    average per-sample flattened L2 norms/ratios, using epsilon=1e-8.
    """
    if not isinstance(model, (IterativeClassifier, CellularClassifier)):
        raise ValueError("Refinement analysis requires an iterative or cellular model")
    cellular = isinstance(model, CellularClassifier)
    guided = isinstance(model, UncertaintyGuidedCellularClassifier)
    message_analysis = message_analysis or guided
    if message_analysis and not cellular:
        raise ValueError("Message analysis requires a cellular model")
    feature_analysis = feature_analysis or message_analysis
    final_metrics = ClassificationMetrics()
    previous_mode = model.training
    model.eval()
    steps = (model.communication_iterations if cellular else model.refinement_iterations) + 1
    correct_sum = torch.zeros(steps, dtype=torch.float64)
    confidence_sum = torch.zeros_like(correct_sum)
    entropy_sum = torch.zeros_like(correct_sum)
    change_sum = torch.zeros(steps - 1, dtype=torch.float64)
    feature_sum = torch.zeros_like(correct_sum)
    update_sum = torch.zeros_like(change_sum)
    relative_sum = torch.zeros_like(change_sum)
    effective_sum = torch.zeros_like(change_sum)
    relative_effective_sum = torch.zeros_like(change_sum)
    message_sum = torch.zeros_like(change_sum)
    relative_message_sum = torch.zeros_like(change_sum)
    outcomes = dict.fromkeys(["corrected", "damaged", "stable_correct", "stable_wrong"], 0)
    samples = []
    n = never_changed = 0
    try:
        for images, targets in loader:
            options = {"return_intermediates": True, "return_features": feature_analysis}
            if cellular:
                options["return_messages"] = message_analysis
            output = model(images.to(device), **options)
            logits = torch.stack(output["intermediate_logits"])
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite intermediate logits")
            predictions, confidence, entropy = [v.cpu() for v in prediction_statistics(logits)]
            final_metrics.update(logits[-1], targets)
            targets = targets.cpu()
            changes = predictions[1:] != predictions[:-1]
            masks = outcome_masks(predictions, targets)
            correct_sum += predictions.eq(targets.unsqueeze(0)).sum(1)
            confidence_sum += confidence.double().sum(1)
            entropy_sum += entropy.double().sum(1)
            change_sum += changes.sum(1)
            never_changed += (~changes.any(0)).sum().item()
            for name, mask in masks.items():
                outcomes[name] += mask.sum().item()
            if feature_analysis:
                states = output["feature_states"]
                if any(not torch.isfinite(s).all() for s in states):
                    raise ValueError("Non-finite feature state")
                norms = torch.stack([s.double().flatten(1).norm(dim=1) for s in states])
                updates = torch.stack([(b.double() - a.double()).flatten(1).norm(dim=1)
                                       for a, b in zip(states, states[1:])]) if steps > 1 else norms[:0]
                feature_sum += norms.sum(1).cpu()
                update_sum += updates.sum(1).cpu()
                relative_sum += (updates / (norms[:-1] + 1e-8)).sum(1).cpu()
            if message_analysis:
                messages = output["messages"]
                if any(not torch.isfinite(m).all() for m in messages):
                    raise ValueError("Non-finite communication message")
                message_norms = torch.stack([m.double().flatten(1).norm(dim=1) for m in messages]) if steps > 1 else norms[:0]
                message_sum += message_norms.sum(1).cpu()
                relative_message_sum += (message_norms / (norms[:-1] + 1e-8)).sum(1).cpu()
            if guided:
                uncertainty_values = torch.stack(output['uncertainties']).cpu() if steps > 1 else entropy[:0]
                gate_values = torch.stack(output['gates']).flatten(1).cpu() if steps > 1 else entropy[:0]
                effective_norms = message_norms.cpu() * model.residual_scale * gate_values.double()
                relative_effective = effective_norms / (norms[:-1].cpu() + 1e-8)
                effective_sum += effective_norms.sum(1)
                relative_effective_sum += relative_effective.sum(1)
                if not torch.isfinite(gate_values).all() or not torch.isfinite(uncertainty_values).all():
                    raise ValueError("Non-finite uncertainty or gate")
                initial_uncertainty = predictive_entropy_from_logits(logits[0]).cpu()
            if collect_samples or guided:
                for j in range(len(targets)):
                    samples.append({"sample_index": n + j, "target": targets[j].item(),
                        "predictions": predictions[:, j].tolist(), "confidence": confidence[:, j].tolist(),
                        "entropy": entropy[:, j].tolist(), "prediction_changes": changes[:, j].sum().item(),
                        "outcome": next(name for name, mask in masks.items() if mask[j])})
                    if guided:
                        samples[-1].update(initial_uncertainty=initial_uncertainty[j].item(),
                            uncertainties=uncertainty_values[:, j].tolist(), gates=gate_values[:, j].tolist(),
                            relative_message_magnitudes=(message_norms[:, j].cpu() / (norms[:-1, j].cpu() + 1e-8)).tolist(),
                            relative_gated_updates=relative_effective[:, j].tolist())
            n += len(targets)
    finally:
        model.train(previous_mode)
    if not n:
        raise ValueError("Cannot analyze an empty loader")
    iterations = []
    for t in range(steps):
        row = {"iteration": t, "accuracy": (correct_sum[t] / n).item(),
               "mean_confidence": (confidence_sum[t] / n).item(),
               "mean_entropy": (entropy_sum[t] / n).item(),
               "prediction_change_rate": None if t == 0 else (change_sum[t-1] / n).item()}
        if feature_analysis:
            row.update(mean_feature_norm=(feature_sum[t] / n).item(),
                       mean_update_norm=None if t == 0 else (update_sum[t-1] / n).item(),
                       mean_relative_feature_update=None if t == 0 else (relative_sum[t-1] / n).item())
        if message_analysis:
            row.update(mean_message_norm=None if t == 0 else (message_sum[t-1] / n).item(),
                       mean_relative_message_magnitude=None if t == 0 else (relative_message_sum[t-1] / n).item())
        if guided:
            row.update(mean_gated_update_norm=None if t == 0 else (effective_sum[t-1] / n).item(),
                       mean_relative_gated_update=None if t == 0 else (relative_effective_sum[t-1] / n).item())
        iterations.append(row)
    result = {"final_metrics": final_metrics.compute(),
              "outcome_rates": {key: value/n for key, value in outcomes.items()},
              "samples": n, "refinement_iterations": steps - 1,
              "iterations": iterations, "outcomes": outcomes, "never_changed_count": never_changed,
              "mean_prediction_changes": change_sum.sum().item() / n}
    if cellular:
        result.update(communication_iterations=steps - 1, communication_type=model.communication_type)
    if guided:
        result['corrected_damaged_ratio'] = outcomes['corrected'] / max(outcomes['damaged'], 1)
        result.update(summarize_uncertainty(samples, steps - 1))
        result.update(uncertainty_type=model.uncertainty_type, normalize_uncertainty=model.normalize_uncertainty,
                      detach_uncertainty_gate=model.detach_uncertainty_gate, gate_mode=model.gate_mode,
                      fixed_gate_value=model.fixed_gate_value, gate_mapping=model.gate_mapping, minimum_gate=model.minimum_gate)
        for t, row in enumerate(iterations):
            controls = result['gate_evolution'][t] if t < steps - 1 else dict.fromkeys(
                ['mean_uncertainty', 'mean_gate', 'std_gate', 'min_gate', 'max_gate'])
            row.update({key: value for key, value in controls.items() if key != 'iteration'})
    if collect_samples:
        result["sample_records"] = samples
    return result
