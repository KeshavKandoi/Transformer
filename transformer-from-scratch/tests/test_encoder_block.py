import pytest
import torch

from src.model.encoder_block import EncoderBlock


def test_block_matches_manual_post_norm_and_passes_mask() -> None:
    torch.manual_seed(42)
    block = EncoderBlock(16, 4, 32).eval()
    x = torch.randn(2, 4, 16)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    mask = valid[:, None, None, :]
    a, expected_maps = block.self_attention(x, attention_mask=mask, return_attention_weights=True)
    y = block.attention_residual(x, a, valid)
    expected = block.ffn_residual(y, block.feed_forward(y), valid)
    output, maps = block(x, mask, valid, return_attention_weights=True)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(maps, expected_maps)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert torch.count_nonzero(output[~valid]) == 0
    assert maps[0, :, 0, 2].gt(0).all()
    assert torch.count_nonzero(maps[0, :, :, 3]) == 0
    assert block.attention_residual.norm.gamma is not block.ffn_residual.norm.gamma
    assert block.attention_residual.norm.beta is not block.ffn_residual.norm.beta


def test_block_dropout_training_and_eval() -> None:
    torch.manual_seed(42)
    block = EncoderBlock(16, 4, 32, dropout=0.5, attention_dropout=0.5)
    x = torch.randn(2, 4, 16)
    torch.manual_seed(1)
    first = block(x)
    torch.manual_seed(2)
    assert not torch.allclose(first, block(x))
    block.eval()
    first = block(x)
    torch.testing.assert_close(first, block(x), rtol=0, atol=0)


def test_invalid_block_inputs_and_configuration() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EncoderBlock(15, 4)
    block = EncoderBlock(16, 4, 32)
    with pytest.raises(ValueError, match="shape"):
        block(torch.zeros(3, 16))
    with pytest.raises(ValueError, match="broadcast"):
        block(torch.zeros(2, 3, 16), torch.ones(4, 4, dtype=torch.bool))
