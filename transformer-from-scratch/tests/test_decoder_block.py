import pytest
import torch

from src.data.masks import cross_attention_mask, decoder_attention_mask
from src.model.decoder_block import DecoderBlock
from src.model.residual import ResidualAddNorm


def test_block_matches_manual_post_norm_and_preserves_gradients() -> None:
    torch.manual_seed(42)
    block = DecoderBlock(16, 4, 32).eval()
    x = torch.randn(1, 4, 16, requires_grad=True)
    memory = torch.randn(1, 3, 16, requires_grad=True)
    target_ids = torch.tensor([[2, 5, 6, 0]])
    valid = target_ids.ne(0)
    self_mask = decoder_attention_mask(target_ids, 0)
    cross_mask = cross_attention_mask(torch.tensor([[5, 6, 0]]), 4, 0)
    a, expected_self = block.masked_self_attention(x, attention_mask=self_mask, return_attention_weights=True)
    y = block.self_attention_residual(x, a, valid)
    c, expected_cross = block.cross_attention(y, memory, cross_mask, return_attention_weights=True)
    z = block.cross_attention_residual(y, c, valid)
    expected = block.ffn_residual(z, block.feed_forward(z), valid)
    result, self_weights, cross_weights = block(x, memory, self_mask, cross_mask, valid, return_attention_weights=True)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(self_weights, expected_self)
    torch.testing.assert_close(cross_weights, expected_cross)
    assert result.shape == x.shape
    assert torch.count_nonzero(result[~valid]) == 0
    assert torch.count_nonzero(self_weights.triu(1)) == 0
    assert torch.count_nonzero(cross_weights[..., -1]) == 0
    residuals = [module for module in block.modules() if isinstance(module, ResidualAddNorm)]
    assert len(residuals) == 3
    assert len({id(parameter) for module in residuals for parameter in module.parameters()}) == 6
    assert not ({id(p) for p in block.cross_attention.parameters()}
                & {id(p) for p in block.masked_self_attention.parameters()})
    gradients = torch.autograd.grad(result[..., 0].square().sum(), (x, memory, *block.parameters()))
    assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in gradients)
    restored = DecoderBlock(16, 4, 32).eval()
    restored.load_state_dict(block.state_dict())
    torch.testing.assert_close(restored(x, memory, self_mask, cross_mask, valid), result)


def test_block_dropout_and_invalid_inputs() -> None:
    torch.manual_seed(42)
    block = DecoderBlock(16, 4, 32, dropout=0.5, attention_dropout=0.5)
    x, memory = torch.randn(2, 3, 16), torch.randn(2, 5, 16)
    self_mask = torch.ones(3, 3, dtype=torch.bool)
    cross_mask = torch.ones(3, 5, dtype=torch.bool)
    torch.manual_seed(1)
    first = block(x, memory, self_mask, cross_mask)
    torch.manual_seed(2)
    assert not torch.allclose(first, block(x, memory, self_mask, cross_mask))
    block.eval()
    first = block(x, memory, self_mask, cross_mask)
    torch.testing.assert_close(first, block(x, memory, self_mask, cross_mask), rtol=0, atol=0)
    with pytest.raises(ValueError, match="decoder_x must have shape"):
        block(x[0], memory, self_mask, cross_mask)
    with pytest.raises(ValueError, match="same batch size"):
        block(x, memory[:1], self_mask, cross_mask)
    with pytest.raises(ValueError, match="broadcast"):
        block(x, memory, cross_mask, self_mask)
    with pytest.raises(ValueError, match="broadcast"):
        block(x, memory, self_mask, torch.ones(3, 6, dtype=torch.bool))
