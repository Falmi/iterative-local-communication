"""Descriptive uncertainty groups from compact per-sample analysis records."""
import torch


def mean_or_none(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_uncertainty(records: list[dict], steps: int) -> dict:
    """Rank quartiles: ties resolved by test-row index; no causal interpretation.

    Rates use all samples in the group, not only initially wrong/correct samples.
    Gate means average samples and communication steps. Empty groups use null.
    """
    groups = {}
    ordered = sorted(records, key=lambda r: (r['initial_uncertainty'], r['sample_index']))
    for q, name in enumerate(('q1_lowest', 'q2', 'q3', 'q4_highest')):
        rows = ordered[len(ordered)*q//4:len(ordered)*(q+1)//4]
        for row in rows:
            row['uncertainty_group'] = name
        n = len(rows)
        initial = mean_or_none([float(r['predictions'][0] == r['target']) for r in rows])
        final = mean_or_none([float(r['predictions'][-1] == r['target']) for r in rows])
        corrected = sum(r['outcome'] == 'corrected' for r in rows)
        damaged = sum(r['outcome'] == 'damaged' for r in rows)
        groups[name] = {'samples': n, 'initial_accuracy': initial, 'final_accuracy': final,
            'accuracy_change': final - initial if n else None,
            'corrected_count': corrected, 'corrected_rate': corrected/n if n else None,
            'damaged_count': damaged, 'damaged_rate': damaged/n if n else None,
            'mean_initial_uncertainty': mean_or_none([r['initial_uncertainty'] for r in rows]),
            'mean_gate': mean_or_none([g for r in rows for g in r['gates']]),
            'mean_initial_gate': mean_or_none([r['gates'][0] for r in rows if r['gates']]),
            'mean_final_communication_gate': mean_or_none([r['gates'][-1] for r in rows if r['gates']]),
            'mean_prediction_change_frequency': mean_or_none([r['prediction_changes']/steps if steps else 0.0 for r in rows])}
    descriptions = {}
    for category in ('corrected', 'damaged', 'stable_correct', 'stable_wrong'):
        values = [r['initial_uncertainty'] for r in records if r['outcome'] == category]
        descriptions[category] = {'samples': len(values), 'mean_initial_uncertainty': mean_or_none(values)}
    evolution = []
    for t in range(steps):
        gates = torch.tensor([r['gates'][t] for r in records], dtype=torch.float64)
        evolution.append({'iteration': t, 'mean_uncertainty': mean_or_none([r['uncertainties'][t] for r in records]),
            'mean_gate': gates.mean().item(), 'std_gate': gates.std(unbiased=False).item(),
            'min_gate': gates.min().item(), 'max_gate': gates.max().item(),
            'median_gate': gates.quantile(.5).item(), 'q25_gate': gates.quantile(.25).item(),
            'q75_gate': gates.quantile(.75).item(),
            'fraction_below_0_01': (gates < .01).double().mean().item(),
            'fraction_below_0_05': (gates < .05).double().mean().item(),
            'fraction_above_0_50': (gates > .5).double().mean().item(),
            'fraction_above_0_90': (gates > .9).double().mean().item()})
    cases = {}
    for name, group, outcome in [('high_uncertainty_corrected', 'q4_highest', 'corrected'),
                                  ('high_uncertainty_still_wrong', 'q4_highest', 'stable_wrong'),
                                  ('low_uncertainty_stable_correct', 'q1_lowest', 'stable_correct'),
                                  ('low_uncertainty_damaged', 'q1_lowest', 'damaged')]:
        cases[name] = [r['sample_index'] for r in records if r['uncertainty_group'] == group and r['outcome'] == outcome]
    return {'uncertainty_groups': groups, 'uncertainty_by_outcome': descriptions,
            'gate_evolution': evolution, 'case_indices': cases,
            'uncertainty_group_method': 'rank quartiles; ties broken by sample_index; empty means are null'}
