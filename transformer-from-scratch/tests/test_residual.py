import pytest
import torch

from src.model.residual import ResidualAddNorm


def test_manual_post_norm_and_both_gradient_paths() -> None:
    module = ResidualAddNorm(3, dropout=0.2).eval()
    x = torch.tensor([[[1., 3., 7.], [4., 0., 1.]]], requires_grad=True)
    sublayer = torch.tensor([[[4., 2., 1.], [0., 5., 7.]]], requires_grad=True)
    output = module(x, sublayer)
    summed = x + sublayer
    centered = summed - summed.mean(-1, keepdim=True)
    expected = centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    assert output.shape == x.shape
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(module(x, sublayer), output, rtol=0, atol=0)
    output[..., 0].sum().backward()
    assert torch.count_nonzero(x.grad) > 0
    torch.testing.assert_close(x.grad, sublayer.grad)


def test_training_dropout_is_on_sublayer_before_addition() -> None:
    module = ResidualAddNorm(3, dropout=1.0)
    x = torch.tensor([2., 4., 6.])
    sublayer = torch.tensor([0., 5., 0.])
    torch.testing.assert_close(module(x, sublayer), module.norm(x))
    module.eval()
    torch.testing.assert_close(module(x, sublayer), module.norm(x + sublayer))
    assert not torch.allclose(module(x, sublayer), module.norm(x))


def test_query_mask_zeros_padding_even_with_nonzero_beta() -> None:
    module = ResidualAddNorm(3).eval()
    with torch.no_grad():
        module.norm.beta.fill_(2)
    x = torch.tensor([[[2., 4., 6.], [0., 0., 0.]]])
    mask = torch.tensor([[True, False]])
    unmasked = module(x, x)
    output = module(x, x, mask)
    torch.testing.assert_close(output[mask], unmasked[mask])
    torch.testing.assert_close(unmasked[~mask], torch.full((1, 3), 2.))
    assert torch.count_nonzero(output[~mask]) == 0


def test_validations_and_independent_parameters() -> None:
    first, second = ResidualAddNorm(3), ResidualAddNorm(3)
    assert first.norm.gamma is not second.norm.gamma
    assert first.norm.beta is not second.norm.beta
    x = torch.zeros(2, 4, 3)
    with pytest.raises(ValueError, match="identical"):
        first(x, x[:1])
    with pytest.raises(ValueError, match="token_valid_mask must have shape"):
        first(x, x, torch.ones(2, 1, 4, dtype=torch.bool))
    with pytest.raises(TypeError, match="torch.bool"):
        first(x, x, torch.ones(2, 4))
    with pytest.raises(ValueError, match="dropout"):
        ResidualAddNorm(dropout=-0.1)
