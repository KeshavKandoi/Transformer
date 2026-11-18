import pytest
import torch

from src.model.feed_forward import PositionwiseFeedForward


@pytest.mark.parametrize("shape", [(1, 2, 256), (3, 7, 256)])
def test_default_dimensions_and_manual_formula(shape: tuple[int, ...]) -> None:
    torch.manual_seed(42)
    module = PositionwiseFeedForward().eval()
    inputs = torch.randn(shape)
    hidden = inputs @ module.linear1.weight.T + module.linear1.bias
    assert hidden.shape == (*shape[:-1], 1024)
    assert (hidden < 0).any()
    expected = hidden.clamp_min(0) @ module.linear2.weight.T + module.linear2.bias
    output = module(inputs)
    assert output.shape == shape and torch.isfinite(output).all()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(module(inputs), output, rtol=0, atol=0)


def test_token_independence_gradients_and_state_dict() -> None:
    torch.manual_seed(42)
    module = PositionwiseFeedForward(8, 16).eval()
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    output = module(inputs)
    changed = inputs.detach().clone()
    changed[1] = 30
    changed[0, 1:] = -30
    torch.testing.assert_close(module(changed)[0, 0], output[0, 0])
    output.square().mean().backward()
    for parameter in (*module.parameters(), inputs):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad)
    restored = PositionwiseFeedForward(8, 16).eval()
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(inputs), output)


def test_dropout_is_between_relu_and_second_linear() -> None:
    module = PositionwiseFeedForward(3, 4, dropout=1.0)
    with torch.no_grad():
        module.linear1.weight.fill_(1)
        module.linear1.bias.zero_()
        module.linear2.weight.fill_(1)
        module.linear2.bias.fill_(2)
    x = torch.ones(1, 2, 3)
    torch.testing.assert_close(module(x), torch.full_like(x, 2))
    module.eval()
    torch.testing.assert_close(module(x), torch.full_like(x, 14))


@pytest.mark.parametrize(("args", "message"), [
    ((0, 8, 0.1), "d_model"),
    ((8, -1, 0.1), "d_ff"),
    ((8, 16, 1.1), "dropout"),
])
def test_invalid_configuration(args: tuple[int, int, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PositionwiseFeedForward(*args)


def test_invalid_input_width() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        PositionwiseFeedForward(4, 8)(torch.zeros(2, 4, 3))
