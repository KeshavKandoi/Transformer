import pytest
import torch

from src.model.config import ModelConfig
from src.model.decoder import DecoderStack


def test_default_depth_and_independent_parameters() -> None:
    stack = DecoderStack()
    assert len(stack.layers) == 4
    parameters = [parameter for layer in stack.layers for parameter in layer.parameters()]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert len({parameter.untyped_storage().data_ptr() for parameter in parameters}) == len(parameters)
    assert {id(parameter) for parameter in stack.parameters()} == {id(parameter) for parameter in parameters}


def test_stack_passes_states_memory_and_masks_and_serializes() -> None:
    torch.manual_seed(42)
    config = ModelConfig(d_model=16, d_ff=32, num_decoder_layers=3)
    stack = DecoderStack(config).eval()
    x, memory = torch.randn(2, 4, 16, requires_grad=True), torch.randn(2, 3, 16)
    valid = torch.tensor([[True, True, True, False]] * 2)
    self_mask = torch.ones(4, 4, dtype=torch.bool).tril() & valid[:, None, None, :]
    cross_mask = torch.ones(4, 3, dtype=torch.bool)
    inputs, outputs = [], []
    hooks = []
    for layer in stack.layers:
        hooks.append(layer.register_forward_pre_hook(lambda _, args: inputs.append(args)))
        hooks.append(layer.register_forward_hook(lambda _, args, result: outputs.append(result)))
    result, self_maps, cross_maps = stack(x, memory, self_mask, cross_mask, valid, return_attention_weights=True)
    for hook in hooks:
        hook.remove()
    assert len(inputs) == len(outputs) == 3
    assert inputs[0][0] is x
    assert all(entry[0] is previous[0] for entry, previous in zip(inputs[1:], outputs))
    assert all(
        entry[1] is memory and entry[2] is self_mask and entry[3] is cross_mask and entry[4] is valid
        for entry in inputs
    )
    assert all(torch.count_nonzero(layer_output[0][~valid]) == 0 for layer_output in outputs)
    assert self_maps.shape == (2, 3, 4, 4, 4)
    assert cross_maps.shape == (2, 3, 4, 4, 3)
    assert result.shape == x.shape and torch.isfinite(result).all()
    grad, = torch.autograd.grad(result[..., 0].square().sum(), x)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
    restored = DecoderStack(config).eval()
    restored.load_state_dict(stack.state_dict())
    torch.testing.assert_close(restored(x, memory, self_mask, cross_mask, valid), result)
    torch.testing.assert_close(stack(x, memory, self_mask, cross_mask, valid), result)


@pytest.mark.parametrize("depth", [0, -1, True])
def test_invalid_depth(depth: int) -> None:
    with pytest.raises(ValueError, match="num_decoder_layers"):
        DecoderStack(ModelConfig(num_decoder_layers=depth))


def test_invalid_head_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DecoderStack(ModelConfig(d_model=18, num_heads=4))
