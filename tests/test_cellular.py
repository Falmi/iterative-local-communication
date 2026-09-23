import pytest
import torch
import yaml
from src.models.baseline import build_model
from src.models.cellular_classifier import CellularClassifier
from src.models.cellular_communication import LocalCellularCommunicationBlock
from src.models.refinement import IterativeRefinementBlock
from src.models.factory import create_model
from src.utils.config import load_config
from src.datasets.cifar import make_loader
from src.analysis.refinement_analysis import analyze_refinement


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(23)


def count(model):
    return sum(p.numel() for p in model.parameters())


def test_cellular_forward_backward_and_message_equation():
    model = CellularClassifier(residual_scale=0.5)
    x = torch.randn(2, 3, 32, 32)
    output = model(x, return_features=True, return_messages=True)
    assert output['final_logits'].shape == (2, 10)
    assert len(output['intermediate_logits']) == len(output['feature_states']) == 4
    assert len(output['messages']) == 3
    assert output['initial_logits'] is output['intermediate_logits'][0]
    assert output['final_logits'] is output['intermediate_logits'][-1]
    for value in output['feature_states'] + output['messages']:
        assert value.shape == (2, 512, 4, 4)
        assert torch.isfinite(value).all()
    for logits in output['intermediate_logits']:
        assert torch.isfinite(logits).all()
    for before, message, after in zip(output['feature_states'], output['messages'], output['feature_states'][1:]):
        assert torch.equal(after, before + 0.5 * message)
        assert not torch.equal(before, after)
    for state in output['feature_states']:
        state.retain_grad()
    torch.nn.functional.cross_entropy(output['final_logits'], torch.tensor([1, 2])).backward()
    for module in (model.backbone, model.communication_blocks, model.classifier):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert all(p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in output['feature_states'])
    model.eval()
    with torch.no_grad():
        normal = model(x)
        details = model(x, return_intermediates=True)
    assert torch.equal(normal, details['final_logits'])
    assert details['messages'] is None and details['feature_states'] is None


def test_neighbour_influence_distinguishes_pointwise_refinement():
    cellular = LocalCellularCommunicationBlock(channels=8).eval()
    pointwise = IterativeRefinementBlock(channels=8).eval()
    x = torch.randn(2, 8, 5, 5)
    changed = x.clone()
    changed[:, 0, 2, 3] += 10  # neighbour of target (2,2); target input unchanged
    with torch.no_grad():
        a, b = cellular(x), cellular(changed)
        pa, pb = pointwise(x), pointwise(changed)
    assert not torch.allclose(a[:, :, 2, 2], b[:, :, 2, 2], atol=1e-7, rtol=0)
    assert torch.equal(pa[:, :, 2, 2], pb[:, :, 2, 2])
    assert torch.equal(a[:, :, 0, 0], b[:, :, 0, 0])  # outside one-step neighbourhood
    assert cellular.neighbourhood_aggregation.groups == 8
    assert cellular.neighbourhood_aggregation.kernel_size == (3, 3)
    assert cellular.neighbourhood_aggregation.padding == (1, 1)


@pytest.mark.parametrize('shape', [(2, 8, 5, 7), (2, 8, 1, 1)])
def test_constant_spatial_input_and_padding(shape):
    block = LocalCellularCommunicationBlock(8)
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1).expand(shape).contiguous()
    result, message = block(x, return_message=True)
    assert result.shape == message.shape == x.shape
    assert torch.isfinite(result).all() and torch.isfinite(message).all()
    if shape[-1] > 1:
        assert torch.equal(result[:, :, 2, 2], result[:, :, 2, 3])  # same interior neighbourhood
    assert 0 < (result - x).norm() / x.norm() < 0.1


def test_shared_count_reuse_and_unshared_gradients():
    a = CellularClassifier(communication_iterations=1)
    b = CellularClassifier(communication_iterations=5)
    assert count(a) == count(b) == 11442762
    calls = []
    hook = b.communication_blocks[0].register_forward_hook(lambda *args: calls.append(1))
    b(torch.randn(2, 3, 32, 32))
    hook.remove()
    assert len(calls) == 5 and len(b.communication_blocks) == 1
    unshared = CellularClassifier(communication_iterations=2, share_communication_weights=False)
    assert len(unshared.communication_blocks) == 2
    assert count(unshared) == 11173962 + 2 * 268800
    assert unshared.communication_blocks[0] is not unshared.communication_blocks[1]
    unshared(torch.randn(2, 3, 32, 32)).sum().backward()
    assert all(p.grad is not None for p in unshared.communication_blocks.parameters())


@pytest.mark.parametrize('shared', [True, False])
def test_t0_and_baseline_initialization_rng(shared):
    torch.manual_seed(4)
    baseline = build_model().eval()
    rng = torch.get_rng_state()
    torch.manual_seed(4)
    model = CellularClassifier(communication_iterations=0, share_communication_weights=shared).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(x, return_features=True, return_messages=True)
        assert torch.equal(baseline(x), output['final_logits'])
    assert len(output['feature_states']) == len(output['intermediate_logits']) == 1
    assert output['messages'] == []


def test_t3_p0_matches_baseline():
    torch.manual_seed(4)
    baseline = build_model().eval()
    rng = torch.get_rng_state()
    torch.manual_seed(4)
    model = CellularClassifier().eval()
    assert torch.equal(torch.get_rng_state(), rng)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(baseline(x), model(x, return_intermediates=True)['initial_logits'])


@pytest.mark.parametrize('iterations', [0, 3])
def test_cellular_analysis_messages_and_final_metrics(iterations):
    config = load_config('configs/cellular.yaml')
    config['training']['batch_size'] = 3  # test incomplete batch weighting
    model = CellularClassifier(communication_iterations=iterations, residual_scale=0.5)
    result = analyze_refinement(model, make_loader(config, 'test', True), torch.device('cpu'),
                                collect_samples=True, message_analysis=True)
    assert model.training
    assert len(result['iterations']) == iterations + 1
    assert sum(result['outcomes'].values()) == result['samples'] == 8
    assert result['communication_type'] == 'depthwise_local'
    assert result['communication_iterations'] == iterations
    for t, row in enumerate(result['iterations']):
        assert row['accuracy'] == sum(r['predictions'][t] == r['target'] for r in result['sample_records']) / 8
        assert row['mean_entropy'] >= 0
        if t:
            assert row['mean_message_norm'] > 0
            assert row['mean_relative_feature_update'] == pytest.approx(0.5 * row['mean_relative_message_magnitude'], rel=1e-5)
            assert row['mean_update_norm'] == pytest.approx(0.5 * row['mean_message_norm'], rel=1e-5)
        else:
            assert row['mean_message_norm'] is None
    plain = analyze_refinement(model, make_loader(config, 'test', True), torch.device('cpu'))
    assert 'mean_message_norm' not in plain['iterations'][0]


def test_config_controls_and_checkpoint_roundtrip(tmp_path):
    config = load_config('configs/cellular.yaml')
    baseline_config = load_config('configs/baseline.yaml')
    assert all(config[k] == baseline_config[k] for k in ('seed', 'deterministic', 'data', 'training', 'device'))
    model = create_model(config['model']).eval()
    path = tmp_path / 'cellular.pt'
    torch.save({'model': model.state_dict(), 'config': config}, path)
    saved = torch.load(path, weights_only=True)
    clone = create_model(saved['config']['model']).eval()
    clone.load_state_dict(saved['model'])
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), clone(x))


@pytest.mark.parametrize('key,value', [('communication_iterations', -1), ('communication_iterations', 1.5),
    ('communication_iterations', True), ('share_communication_weights', 'yes'),
    ('neighbourhood_kernel_size', 1), ('communication_type', 'global'), ('residual_scale', float('nan'))])
def test_invalid_cellular_configuration(tmp_path, key, value):
    config = load_config('configs/cellular.yaml')
    config['model'][key] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(str(path))
    with pytest.raises(ValueError):
        create_model(config['model'])
