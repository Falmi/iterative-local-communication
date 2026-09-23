import pytest
import torch
import yaml
from src.models.uncertainty import predictive_entropy_from_logits, communication_gate, gated_state_update
from src.models.cellular_classifier import CellularClassifier
from src.models.uncertainty_cellular_classifier import UncertaintyGuidedCellularClassifier as Guided
from src.models.factory import create_model
from src.utils.config import load_config
from src.datasets.cifar import make_loader
from src.analysis.refinement_analysis import analyze_refinement
from src.analysis.uncertainty_analysis import summarize_uncertainty


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16])
def test_entropy_sanity_bounds_and_gate_order(dtype):
    logits = torch.tensor([[0.]*10, [1000.]+[-1000.]*9, [1.]+[0.]*9], dtype=dtype)
    u = predictive_entropy_from_logits(logits)
    assert torch.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()
    assert u[0].item() == pytest.approx(1., abs=1e-6)
    assert u[1] < 1e-6
    g = communication_gate(u).flatten()
    assert g[0] > g[2] > g[1]


def test_larger_gate_larger_update():
    state = torch.randn(2, 4, 3, 3)
    message = torch.ones_like(state)
    low = gated_state_update(state, message, torch.full((2,1,1,1), .1), .5)
    high = gated_state_update(state, message, torch.full((2,1,1,1), .9), .5)
    assert (high-state).norm() > (low-state).norm()


@pytest.mark.parametrize('detach', [True, False])
def test_detailed_counts_equation_and_gradients(detach):
    model = Guided(detach_uncertainty_gate=detach, residual_scale=.5)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, return_features=True, return_messages=True)
    assert out['final_logits'].shape == (2,10)
    assert len(out['intermediate_logits']) == len(out['feature_states']) == 4
    assert len(out['uncertainties']) == len(out['gates']) == len(out['messages']) == 3
    for t in range(3):
        gate = out['gates'][t]
        assert gate.shape == (2,1,1,1)
        assert gate.requires_grad is (not detach)
        assert out['uncertainties'][t].requires_grad
        assert torch.equal(out['uncertainties'][t], predictive_entropy_from_logits(out['intermediate_logits'][t]))
        assert torch.equal(out['feature_states'][t+1], out['feature_states'][t]+.5*gate*out['messages'][t])
        assert ((gate >= 0) & (gate <= 1)).all()
    for value in out['intermediate_logits'] + out['feature_states'] + out['messages'] + out['uncertainties']:
        assert torch.isfinite(value).all()
    # An intermediate prediction has a path to final loss only through the gate.
    out['initial_logits'].retain_grad()
    for state in out['feature_states']:
        state.retain_grad()
    torch.nn.functional.cross_entropy(out['final_logits'], torch.tensor([1,2])).backward()
    if detach:
        assert out['initial_logits'].grad is None
    else:
        assert out['initial_logits'].grad is not None and out['initial_logits'].grad.abs().sum() > 0
    for module in (model.backbone, model.communication_blocks, model.classifier):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in out['feature_states'])
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x, return_intermediates=True)['final_logits'])
        assert model(x, return_intermediates=True)['messages'] is None


@pytest.mark.parametrize('shared', [True, False])
def test_fixed_gate_equivalence_and_zero_gate(shared):
    c = CellularClassifier(share_communication_weights=shared).eval()
    d = Guided(share_communication_weights=shared, gate_mode='fixed').eval()
    d.load_state_dict(c.state_dict(), strict=True)
    assert sum(p.numel() for p in c.parameters()) == sum(p.numel() for p in d.parameters())
    x = torch.randn(2,3,32,32)
    with torch.no_grad():
        assert torch.allclose(c(x), d(x), atol=1e-6)
        d.fixed_gate_value = 0.
        out = d(x, return_features=True)
        assert all(torch.equal(out['feature_states'][0], s) for s in out['feature_states'])
        assert torch.equal(out['initial_logits'], out['final_logits'])


def test_parameter_invariance_and_initialization():
    a = Guided(communication_iterations=1)
    b = Guided(communication_iterations=5)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters()) == 11442762
    torch.manual_seed(5)
    c = CellularClassifier()
    rng = torch.get_rng_state()
    torch.manual_seed(5)
    d = Guided()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v,d.state_dict()[k]) for k,v in c.state_dict().items())


@pytest.mark.parametrize('steps', [0,3])
def test_analysis_groups_and_gate_stats(steps):
    config = load_config('configs/uncertainty_cellular.yaml')
    config['training']['batch_size'] = 3
    model = Guided(communication_iterations=steps, gate_mode='fixed', fixed_gate_value=.5)
    result = analyze_refinement(model,make_loader(config,'test',True),torch.device('cpu'),collect_samples=True)
    assert model.training
    assert len(result['iterations']) == steps+1
    assert len(result['gate_evolution']) == steps
    assert sum(g['samples'] for g in result['uncertainty_groups'].values()) == 8
    assert sum(g['corrected_count'] for g in result['uncertainty_groups'].values()) == result['outcomes']['corrected']
    assert sum(g['damaged_count'] for g in result['uncertainty_groups'].values()) == result['outcomes']['damaged']
    assert result['iterations'][-1]['mean_gate'] is None
    for t, row in enumerate(result['gate_evolution']):
        assert row['mean_gate'] == .5 and row['std_gate'] == 0
        update = result['iterations'][t+1]
        assert update['mean_relative_gated_update'] == pytest.approx(.5*update['mean_relative_message_magnitude'])
    for row in result['sample_records']:
        assert len(row['gates']) == steps
        assert row['initial_uncertainty'] >= 0
    if steps == 0:
        out = model(torch.randn(2,3,32,32),return_features=True,return_messages=True)
        assert len(out['intermediate_logits']) == 1
        assert out['gates'] == out['uncertainties'] == out['messages'] == []


def test_rank_groups_ties_empty_and_known_outcomes():
    records=[]
    for i, category in enumerate(['stable_correct','damaged','stable_wrong','corrected']):
        preds = {'stable_correct':[0,0], 'damaged':[0,1], 'stable_wrong':[1,1], 'corrected':[1,0]}[category]
        records.append(dict(sample_index=i,initial_uncertainty=i/4, predictions=preds,target=0,
            gates=[i/4],uncertainties=[i/4],prediction_changes=int(preds[0]!=preds[1]),outcome=category))
    result=summarize_uncertainty(records,1)
    assert result['case_indices']['high_uncertainty_corrected'] == [3]
    assert result['uncertainty_groups']['q4_highest']['accuracy_change'] == 1
    assert result['uncertainty_by_outcome']['damaged']['mean_initial_uncertainty'] == .25
    sparse=summarize_uncertainty(records[:1],1)
    assert sparse['uncertainty_groups']['q1_lowest']['samples'] == 0
    assert sparse['uncertainty_groups']['q1_lowest']['mean_gate'] is None


def test_checkpoint_and_config_controls(tmp_path):
    config=load_config('configs/uncertainty_cellular.yaml')
    c=load_config('configs/cellular.yaml')
    assert all(config[k]==c[k] for k in ['training','data','seed','deterministic'])
    model=create_model(config['model']).eval()
    path=tmp_path/'d.pt'
    torch.save({'config':config,'model':model.state_dict()},path)
    saved=torch.load(path,weights_only=True)
    clone=create_model(saved['config']['model']).eval()
    clone.load_state_dict(saved['model'])
    x=torch.randn(2,3,32,32)
    with torch.no_grad():
        assert torch.equal(model(x),clone(x))


@pytest.mark.parametrize('key,value',[('gate_mode','learned'),('detach_uncertainty_gate','true'),
    ('normalize_uncertainty',False),('uncertainty_type','margin'),('fixed_gate_value',-1),('fixed_gate_value',float('nan'))])
def test_invalid_options(tmp_path,key,value):
    config=load_config('configs/uncertainty_cellular.yaml'); config['model'][key]=value
    path=tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError): load_config(str(path))
    with pytest.raises(ValueError): create_model(config['model'])
