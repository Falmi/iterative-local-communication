import pytest
import torch
from src.models.baseline import build_model
from src.models.iterative_classifier import IterativeClassifier
from src.models.refinement import IterativeRefinementBlock
from src.models.factory import create_model
from src.analysis.refinement_analysis import analyze_refinement, prediction_statistics, outcome_masks
from src.datasets.cifar import make_loader
from src.utils.config import load_config


def count(model):
    return sum(p.numel() for p in model.parameters())


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(17)


def test_outputs_and_gradients_across_every_iteration():
    model = IterativeClassifier()
    x = torch.randn(2, 3, 32, 32)
    output = model(x, return_features=True)
    assert output['final_logits'].shape == (2, 10)
    assert len(output['intermediate_logits']) == len(output['feature_states']) == 4
    assert output['initial_logits'] is output['intermediate_logits'][0]
    assert output['final_logits'] is output['intermediate_logits'][-1]
    assert not torch.equal(output['feature_states'][0], output['feature_states'][1])
    for value in output['intermediate_logits'] + output['feature_states']:
        assert torch.isfinite(value).all()
    for state in output['feature_states']:
        state.retain_grad()
    torch.nn.functional.cross_entropy(output['final_logits'], torch.tensor([1, 2])).backward()
    for module in (model.backbone, model.refinement_blocks, model.classifier):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in output['feature_states'])
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x, return_intermediates=True)['final_logits'])
        assert model(x, return_intermediates=True)['feature_states'] is None


def test_shared_parameter_count_and_reuse():
    assert count(IterativeClassifier(refinement_iterations=1)) == count(IterativeClassifier(refinement_iterations=5))
    model = IterativeClassifier(refinement_iterations=5)
    calls = []
    hook = model.refinement_blocks[0].register_forward_hook(lambda *args: calls.append(1))
    model(torch.randn(2, 3, 32, 32))
    hook.remove()
    assert len(model.refinement_blocks) == 1 and len(calls) == 5


def test_zero_iterations_matches_seeded_baseline():
    torch.manual_seed(5)
    baseline = build_model().eval()
    torch.manual_seed(5)
    model = IterativeClassifier(refinement_iterations=0).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(x, return_features=True)
        assert len(output['intermediate_logits']) == len(output['feature_states']) == 1
        assert torch.equal(baseline(x), output['final_logits'])


def test_unshared_and_checkpoint_roundtrip(tmp_path):
    model = IterativeClassifier(refinement_iterations=2, share_refinement_weights=False).eval()
    assert len(model.refinement_blocks) == 2
    assert model.refinement_blocks[0] is not model.refinement_blocks[1]
    x = torch.randn(2, 3, 32, 32)
    model(x).sum().backward()
    assert all(p.grad is not None for p in model.refinement_blocks.parameters())
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    clone = IterativeClassifier(refinement_iterations=2, share_refinement_weights=False).eval()
    clone.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(x), clone(x))


def test_refinement_is_pointwise_and_active_near_identity():
    block = IterativeRefinementBlock(channels=8).eval()
    x = torch.randn(2, 8, 4, 4)
    other = x.clone()
    other[:, :, 0, 0] += torch.randn(2, 8)
    with torch.no_grad():
        a, b = block(x), block(other)
    assert torch.equal(a[:, :, 1:, :], b[:, :, 1:, :])
    assert 0 < (a - x).norm() / x.norm() < 0.1
    assert torch.equal(IterativeRefinementBlock(8, residual_scale=0)(x), x)


def test_entropy_and_endpoint_categories():
    logits = torch.tensor([[10000., -10000.], [0., 0.]])
    _, _, entropy = prediction_statistics(logits)
    assert torch.isfinite(entropy).all()
    assert entropy[0] == 0 and torch.allclose(entropy[1], torch.tensor(2.).log())
    predictions = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 2]])
    masks = outcome_masks(predictions, torch.zeros(4, dtype=torch.long))
    assert all(mask.sum() == 1 for mask in masks.values())
    assert sum(mask.int() for mask in masks.values()).eq(1).all()


@pytest.mark.parametrize('iterations', [0, 3])
def test_analysis_consistency(iterations):
    config = load_config('configs/refinement.yaml')
    config['training']['batch_size'] = 3  # uneven last batch: test weighted means
    model = IterativeClassifier(refinement_iterations=iterations)
    result = analyze_refinement(model, make_loader(config, 'test', True), torch.device('cpu'), True, True)
    assert model.training  # prior mode restored
    assert len(result['iterations']) == iterations + 1
    assert sum(result['outcomes'].values()) == result['samples'] == 8
    assert len(result['sample_records']) == 8
    for t, row in enumerate(result['iterations']):
        records = result['sample_records']
        assert row['accuracy'] == sum(r['predictions'][t] == r['target'] for r in records) / 8
        assert row['mean_entropy'] == pytest.approx(sum(r['entropy'][t] for r in records) / 8)
        assert row['mean_feature_norm'] > 0
        if t:
            assert row['prediction_change_rate'] == sum(r['predictions'][t] != r['predictions'][t-1] for r in records) / 8
            assert row['mean_relative_feature_update'] > 0


def test_factory_preserves_baseline_and_config():
    baseline = create_model(load_config('configs/baseline.yaml')['model'])
    assert count(baseline) == 11173962
    assert 'conv1.weight' in baseline.state_dict()
    assert isinstance(create_model(load_config('configs/refinement.yaml')['model']), IterativeClassifier)


@pytest.mark.parametrize('iterations', [-1, 1.5, True])
def test_invalid_iterations(iterations):
    with pytest.raises(ValueError):
        IterativeClassifier(refinement_iterations=iterations)


def test_initial_prediction_and_rng_match_baseline():
    torch.manual_seed(9)
    baseline = build_model().eval()
    rng = torch.get_rng_state()
    torch.manual_seed(9)
    iterative = IterativeClassifier().eval()
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(baseline(x), iterative(x, return_intermediates=True)['initial_logits'])
