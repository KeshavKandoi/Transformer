import pytest
import torch

from src.model.config import ModelConfig
from src.model.decoder import TransformerDecoder
from src.model.encoder import TransformerEncoder
from src.model.output_projection import VocabularyProjection
from src.model.transformer import EncoderDecoderTransformer, TransformerDiagnostics


def test_wiring_matches_manual_forward_and_uses_independent_weights(tmp_path) -> None:
    torch.manual_seed(42)
    config = ModelConfig(d_model=16, d_ff=32, num_encoder_layers=2, num_decoder_layers=2)
    model = EncoderDecoderTransformer(29, 3, config).eval()
    assert isinstance(model.encoder, TransformerEncoder)
    assert isinstance(model.decoder, TransformerDecoder)
    assert isinstance(model.output_projection, VocabularyProjection)
    weights = (
        model.encoder.embedding.token_embedding.embedding.weight,
        model.decoder.embedding.token_embedding.embedding.weight,
        model.output_projection.projection.weight,
    )
    assert len({weight.data_ptr() for weight in weights}) == 3
    source = torch.tensor([[4, 5, 3], [6, 7, 8]], dtype=torch.int32)
    targets = torch.tensor([[2, 9], [2, 10]])
    inputs = []
    hooks = [model.decoder.register_forward_pre_hook(lambda _, args: inputs.append(args))]
    details = model(source, targets, return_diagnostics=True)
    for hook in hooks:
        hook.remove()
    assert isinstance(details, TransformerDiagnostics)
    assert inputs[0][0] is targets and inputs[0][1] is details.encoder_memory
    expected = model.output_projection(details.decoder_hidden_states)
    torch.testing.assert_close(details.logits, expected)
    torch.testing.assert_close(model(source, targets), expected)
    assert details.logits.shape == (2, 2, 29)
    keys = model.state_dict().keys()
    assert "encoder.embedding.token_embedding.embedding.weight" in keys
    assert "decoder.embedding.token_embedding.embedding.weight" in keys
    assert "output_projection.projection.weight" in keys
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = EncoderDecoderTransformer(29, 3, config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(source, targets), expected)


def test_train_and_eval_mode() -> None:
    torch.manual_seed(42)
    config = ModelConfig(d_model=16, d_ff=32, dropout=0.4, num_encoder_layers=1, num_decoder_layers=1)
    model = EncoderDecoderTransformer(29, 0, config)
    source, target = torch.tensor([[4, 5, 6]]), torch.tensor([[2, 7]])
    assert not torch.allclose(model(source, target), model(source, target))
    model.eval()
    torch.testing.assert_close(model(source, target), model(source, target), atol=0, rtol=0)


@pytest.mark.parametrize(("encoder_ids", "decoder_ids", "error", "message"), [
    ([2, 3], [[2]], ValueError, "encoder_input_ids must have shape"),
    ([[2, 3]], [2], ValueError, "decoder_input_ids must have shape"),
    ([[2, 3]], [[2], [2]], ValueError, "same batch size"),
    ([[31, 3]], [[2]], ValueError, "encoder vocabulary"),
    ([[2, 3]], [[-1]], ValueError, "decoder vocabulary"),
    ([[0, 0]], [[2]], ValueError, "non-padding"),
    ([[2, 3]], [[0]], ValueError, "non-padding"),
    ([[2., 3.]], [[2]], TypeError, "dtype"),
    ([[2, 3]], [[2.]], TypeError, "dtype"),
    ([[2, 3, 4, 5]], [[2]], ValueError, "exceeds max_seq_len"),
    ([[2, 3]], [[2, 3, 4, 5]], ValueError, "exceeds max_seq_len"),
])
def test_input_validation(encoder_ids, decoder_ids, error, message) -> None:
    config = ModelConfig(d_model=8, d_ff=16, max_seq_len=3, num_encoder_layers=1, num_decoder_layers=1)
    model = EncoderDecoderTransformer(29, 0, config)
    with pytest.raises(error, match=message):
        model(torch.tensor(encoder_ids), torch.tensor(decoder_ids))


@pytest.mark.parametrize("mask_name", ["encoder_attention_mask", "decoder_attention_mask", "cross_attention_mask"])
def test_invalid_mask_dimensions_and_dtype(mask_name: str) -> None:
    config = ModelConfig(d_model=8, d_ff=16, num_encoder_layers=1, num_decoder_layers=1)
    model = EncoderDecoderTransformer(29, 0, config)
    source, target = torch.tensor([[4, 5, 6]]), torch.tensor([[2, 7]])
    with pytest.raises(ValueError, match="broadcast"):
        model(source, target, **{mask_name: torch.ones(7, 8, dtype=torch.bool)})
    with pytest.raises(TypeError, match="torch.bool"):
        model(source, target, **{mask_name: torch.ones(1)})


def test_invalid_input_device() -> None:
    config = ModelConfig(d_model=8, d_ff=16, num_encoder_layers=1, num_decoder_layers=1)
    model = EncoderDecoderTransformer(29, 0, config)
    with pytest.raises(ValueError, match="same device"):
        model(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, device="meta", dtype=torch.long))
