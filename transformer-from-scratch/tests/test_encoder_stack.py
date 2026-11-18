from dataclasses import replace

import pytest
import torch

from src.model.config import ModelConfig
from src.model.encoder import EncoderStack


def test_default_depth_and_independent_parameter_storage() -> None:
    stack = EncoderStack()
    assert len(stack.layers) == 4
    parameters = [parameter for layer in stack.layers for parameter in layer.parameters()]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert len({parameter.untyped_storage().data_ptr() for parameter in parameters}) == len(parameters)
    assert {id(parameter) for parameter in stack.parameters()} == {id(parameter) for parameter in parameters}


def test_sequential_outputs_mask_propagation_and_state_dict() -> None:
    config = ModelConfig(d_model=16, d_ff=32, num_encoder_layers=3)
    stack = EncoderStack(config).eval()
    x = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False]] * 2)
    mask = valid[:, None, None, :]
    entries, exits = [], []
    hooks = []
    for layer in stack.layers:
        hooks.append(layer.register_forward_pre_hook(lambda _, args: entries.append(args)))
        hooks.append(layer.register_forward_hook(lambda _, args, result: exits.append(result)))
    output, maps = stack(x, mask, valid, return_attention_weights=True)
    for hook in hooks:
        hook.remove()
    assert len(entries) == len(exits) == 3
    assert entries[0][0] is x
    assert entries[1][0] is exits[0][0]
    assert entries[2][0] is exits[1][0] and exits[2][0] is output
    assert all(entry[1] is mask and entry[2] is valid for entry in entries)
    assert all(torch.count_nonzero(result[0][~valid]) == 0 for result in exits)
    assert maps.shape == (2, 3, 4, 5, 5)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert torch.autograd.grad(output[valid, :][..., 0].sum(), x)[0].abs().sum() > 0

    restored = EncoderStack(config).eval()
    restored.load_state_dict(stack.state_dict())
    torch.testing.assert_close(restored(x, mask, valid), output)
    torch.testing.assert_close(stack(x, mask, valid), output)


@pytest.mark.parametrize("depth", [0, -1, True])
def test_invalid_depth(depth: int) -> None:
    with pytest.raises(ValueError, match="num_encoder_layers"):
        EncoderStack(replace(ModelConfig(), num_encoder_layers=depth))
