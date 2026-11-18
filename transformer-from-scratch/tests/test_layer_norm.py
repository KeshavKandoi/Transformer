import pytest
import torch
from torch import nn

from src.model.layer_norm import LayerNormalization


def test_manual_example_constant_vector_and_initialization() -> None:
    module = LayerNormalization(3)
    assert module.gamma.requires_grad and module.beta.requires_grad
    torch.testing.assert_close(module.gamma, torch.ones(3))
    torch.testing.assert_close(module.beta, torch.zeros(3))
    output = module(torch.tensor([2., 4., 6.]))
    torch.testing.assert_close(output, torch.tensor([-1.2247, 0., 1.2247]), atol=5e-5, rtol=0)
    torch.testing.assert_close(module(torch.full((3,), 5.)), torch.zeros(3))


def test_last_dimension_and_token_independence() -> None:
    module = LayerNormalization(3)
    inputs = torch.tensor([[[2., 4., 6.], [8., 3., 0.]], [[-2., -4., 7.], [5., 3., 1.]]])
    output = module(inputs)
    assert output.shape == inputs.shape
    torch.testing.assert_close(output.mean(-1), torch.zeros(2, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.var(-1, unbiased=False), torch.ones(2, 2), atol=1e-5, rtol=0)
    changed = inputs.clone()
    changed[1] *= -2
    changed[0, 1] *= -3
    torch.testing.assert_close(module(changed)[0, 0], output[0, 0])


def test_reference_affine_forward_backward_and_state_dict() -> None:
    torch.manual_seed(42)
    module = LayerNormalization(8, 1e-4).double()
    reference = nn.LayerNorm(8, eps=1e-4).double()
    with torch.no_grad():
        module.gamma.copy_(torch.randn_like(module.gamma))
        module.beta.copy_(torch.randn_like(module.beta))
        reference.weight.copy_(module.gamma)
        reference.bias.copy_(module.beta)
    inputs = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(inputs)
    output = module(inputs)
    expected = reference(inputs)
    torch.testing.assert_close(output, expected)
    custom_grads = torch.autograd.grad((output - target).square().mean(), (inputs, module.gamma, module.beta))
    ref_grads = torch.autograd.grad((expected - target).square().mean(), (inputs, reference.weight, reference.bias))
    for gradient, expected_gradient in zip(custom_grads, ref_grads):
        assert torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(gradient, expected_gradient)
    restored = LayerNormalization(8, 1e-4).double()
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(inputs), output)


@pytest.mark.parametrize("epsilon", [0, -1e-5, float("nan"), float("inf")])
def test_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        LayerNormalization(3, epsilon)


def test_invalid_inputs_and_low_precision() -> None:
    with pytest.raises(ValueError, match="d_model"):
        LayerNormalization(0)
    module = LayerNormalization(3)
    with pytest.raises(ValueError, match="last dimension"):
        module(torch.zeros(2, 4))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.zeros(3, dtype=torch.long))
    inputs = torch.tensor([-500., 0., 500.], dtype=torch.float16)
    assert module(inputs).dtype == torch.float16
    torch.testing.assert_close(module(inputs), module(inputs.float()).half())


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_accelerator_forward_and_backward() -> None:
    device = "cuda" if torch.cuda.is_available() else "mps"
    module = LayerNormalization(3)
    inputs = torch.tensor([[2., 4., 6.]])
    expected = module(inputs).detach()
    output = module.to(device)(inputs.to(device))
    torch.testing.assert_close(output.cpu(), expected)
    output[:, 0].sum().backward()
    assert module.gamma.grad is not None and module.beta.grad is not None
