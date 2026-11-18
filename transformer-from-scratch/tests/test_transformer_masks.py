import pytest
import torch

from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer


def make_model(*, bias: bool = False) -> EncoderDecoderTransformer:
    torch.manual_seed(42)
    config = ModelConfig(d_model=16, d_ff=32, output_projection_bias=bias)
    return EncoderDecoderTransformer(32, 0, config).eval()


@pytest.mark.parametrize("prefix_length", [1, 2, 4])
def test_future_token_changes_do_not_leak_to_logits(prefix_length: int) -> None:
    model = make_model()
    source = torch.tensor([[4, 5, 6, 0]]).expand(2, -1)
    first = torch.tensor([2, 7, 8, 9, 10, 11, 12])
    second = first.clone()
    second[prefix_length:] += 9
    with torch.no_grad():
        result = model(source, torch.stack((first, second)), return_diagnostics=True)
    torch.testing.assert_close(result.logits[0, :prefix_length], result.logits[1, :prefix_length], atol=2e-6, rtol=1e-5)
    assert not torch.allclose(result.logits[0, prefix_length:], result.logits[1, prefix_length:])
    assert torch.count_nonzero(result.decoder_self_attention_maps.triu(1)) == 0


def test_padding_fully_masked_rows_and_bias() -> None:
    model = make_model(bias=True)
    with torch.no_grad():
        model.output_projection.projection.bias.fill_(0.2)
        source, target = torch.tensor([[4, 5, 6]]), torch.tensor([[2, 7]])
        base = model(source, target)
        source = torch.nn.functional.pad(source, (0, 2))
        target = torch.nn.functional.pad(target, (0, 2))
        permissive = torch.ones(4, 5, dtype=torch.bool)
        result = model(source, target, cross_attention_mask=permissive, return_diagnostics=True)
        torch.testing.assert_close(result.logits[:, :2], base, atol=2e-6, rtol=1e-5)
        for maps, key_count, query_count in (
            (result.encoder_attention_maps, 3, 5),
            (result.decoder_self_attention_maps, 2, 4),
            (result.decoder_cross_attention_maps, 3, 4),
        ):
            assert maps.shape[:4] == (1, 4, 4, query_count)
            assert torch.count_nonzero(maps[..., key_count:]) == 0
            torch.testing.assert_close(maps.sum(-1), torch.ones_like(maps[..., 0]))
        assert torch.count_nonzero(result.decoder_self_attention_maps.triu(1)) == 0
        assert torch.count_nonzero(result.decoder_hidden_states[:, 2:]) == 0
        torch.testing.assert_close(result.logits[:, 2:], torch.full((1, 2, 32), 0.2))
        permissive[0] = False
        masked = model(source, target, cross_attention_mask=permissive, return_diagnostics=True)
        assert torch.isfinite(masked.logits).all()
        assert torch.count_nonzero(masked.decoder_cross_attention_maps[..., 0, :]) == 0


def test_changing_valid_source_content_changes_logits() -> None:
    model = make_model()
    sources = torch.tensor([[4, 5, 6, 0], [17, 18, 19, 0]])
    target = torch.tensor([[2, 7, 8]]).expand(2, -1)
    with torch.no_grad():
        result = model(sources, target)
    assert (result[0] - result[1]).norm() > 0.1
