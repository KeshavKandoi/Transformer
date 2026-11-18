import pytest
import torch

from src.model.output_projection import VocabularyProjection


@pytest.mark.parametrize("shape", [(1, 1, 2), (3, 5, 2)])
def test_raw_logits_gradient_and_save_load(shape: tuple[int, ...]) -> None:
    module = VocabularyProjection(2, 3, bias=True)
    with torch.no_grad():
        module.projection.weight.copy_(torch.tensor([[1., 2.], [0., -3.], [4., 1.]]))
        module.projection.bias.copy_(torch.tensor([1., 0., -2.]))
    hidden = torch.tensor([2., 3.]).expand(shape).clone().requires_grad_()
    logits = module(hidden)
    torch.testing.assert_close(logits, torch.tensor([9., -9., 9.]).expand(*shape[:2], 3))
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(parameter.requires_grad and torch.count_nonzero(parameter.grad) for parameter in module.parameters())
    assert torch.count_nonzero(hidden.grad)
    restored = VocabularyProjection(2, 3, bias=True)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(hidden), logits)


@pytest.mark.parametrize(("width", "vocab"), [(0, 3), (2, 0), (True, 3)])
def test_invalid_dimensions(width: int, vocab: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        VocabularyProjection(width, vocab)


def test_invalid_hidden_shape_and_dtype() -> None:
    module = VocabularyProjection(2, 3)
    assert module.projection.bias is None
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        module(torch.zeros(2, 3, 4))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.zeros(2, 3, 2, dtype=torch.long))
