import pytest
import torch

from src.model.positional_encoding import SinusoidalPositionalEncoding


def test_formula_and_batch_consistency() -> None:
    module = SinusoidalPositionalEncoding(6, 8)
    output = module(torch.zeros(2, 8, 6))
    torch.testing.assert_close(output[0, 0], torch.tensor([0., 1., 0., 1., 0., 1.]))
    torch.testing.assert_close(
        output[0, 1],
        torch.tensor([0.8415, 0.5403, 0.0464, 0.9989, 0.0022, 1.0000]),
        atol=5e-5,
        rtol=0,
    )
    assert torch.unique(output[0], dim=0).size(0) == 8
    assert output.abs().max() <= 1
    torch.testing.assert_close(output[0], output[1])
    torch.testing.assert_close(output[:1, :3], module(torch.zeros(1, 3, 6)))


def test_buffer_persistence_dtype_and_gradient() -> None:
    module = SinusoidalPositionalEncoding(6, 8)
    assert dict(module.named_buffers())["encoding"] is module.encoding
    assert not list(module.parameters())
    assert "encoding" in module.state_dict()
    assert not module.encoding.requires_grad

    inputs = torch.zeros(1, 3, 6, dtype=torch.float64, requires_grad=True)
    output = module(inputs)
    assert output.dtype == inputs.dtype
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
    assert module.encoding.grad is None


def test_invalid_positional_configuration_and_input() -> None:
    with pytest.raises(ValueError, match="positive even"):
        SinusoidalPositionalEncoding(5, 8)
    with pytest.raises(ValueError, match="max_seq_len"):
        SinusoidalPositionalEncoding(6, 0)
    module = SinusoidalPositionalEncoding(6, 8)
    with pytest.raises(ValueError, match="shape"):
        module(torch.zeros(1, 3, 5))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.zeros(1, 3, 6, dtype=torch.long))
    with pytest.raises(ValueError, match="Sequence length 9 exceeds max_seq_len 8"):
        module(torch.zeros(1, 9, 6))


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available",
)
def test_buffer_moves_with_module() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    module = SinusoidalPositionalEncoding(6, 8).to(device)
    output = module(torch.zeros(1, 3, 6, device=device))
    assert module.encoding.device == output.device
    torch.testing.assert_close(output.cpu()[0], module.encoding.cpu()[:3])
