import pytest
import torch
from torch import nn

from src.training.config import OptimizerConfig
from src.training.optimizer import build_optimizer


def test_adamw_unique_trainable_parameters_and_state() -> None:
    shared = nn.Linear(3, 3)
    model = nn.Sequential(shared, shared, nn.Linear(3, 1))
    model[2].weight.requires_grad_(False)
    optimizer = build_optimizer(model, OptimizerConfig(weight_decay=0.03))
    parameters = optimizer.param_groups[0]["params"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    assert optimizer.param_groups[0]["weight_decay"] == 0.03
    before = shared.weight.detach().clone()
    model(torch.ones(1, 3)).square().sum().backward()
    optimizer.step()
    assert not torch.equal(before, shared.weight)
    restored = build_optimizer(model, OptimizerConfig())
    restored.load_state_dict(optimizer.state_dict())
    torch.testing.assert_close(restored.state[shared.weight]["exp_avg"], optimizer.state[shared.weight]["exp_avg"])


def test_empty_parameters_and_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="trainable"):
        build_optimizer(nn.Identity(), OptimizerConfig())
    with pytest.raises(ValueError, match="learning_rate"):
        OptimizerConfig(learning_rate=float("nan"))
