from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from scripts.prepare_data import prepare
from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding
from src.model.feed_forward import PositionwiseFeedForward
from src.model.multi_head_attention import MultiHeadAttention
from src.model.residual import ResidualAddNorm


@pytest.fixture(scope="module")
def real_batch(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Tensor], BilingualBPETokenizer]:
    directory = tmp_path_factory.mktemp("phase5")
    data_path = ROOT / "configs/data_config.yaml"
    data = DataConfig.load(data_path)
    processed, artifacts = directory / "processed", directory / "tokenizer"
    prepare(Namespace(
        config=data_path, input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed, tokenizer_dir=artifacts,
    ))
    tokenizer = BilingualBPETokenizer.load(artifacts)
    pairs, _ = read_parallel_csv(processed / "train.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
    batch = next(iter(loader))
    assert batch["encoder_input_ids"].size(1) != batch["decoder_input_ids"].size(1)
    return batch, tokenizer


@pytest.mark.parametrize("side", ["encoder", "decoder"])
def test_real_batch_sublayer_pipeline(
    real_batch: tuple[dict[str, Tensor], BilingualBPETokenizer], side: str, tmp_path: Path
) -> None:
    torch.manual_seed(42)
    batch, tokenizer = real_batch
    model = ModelConfig.load(ROOT / "configs/model_config.yaml")
    model.validate_sublayer_settings()
    ids = batch[f"{side}_input_ids"]
    key_mask = batch[f"{side}_attention_mask"]
    valid = ids.ne(tokenizer.pad_id)
    assert valid.any() and not valid.all()

    embedding = TransformerEmbedding(
        tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id,
    )
    attention = MultiHeadAttention(model.d_model, model.num_heads, model.attention_dropout)
    first = ResidualAddNorm(model.d_model, model.dropout, model.layer_norm_eps)
    second = ResidualAddNorm(model.d_model, model.dropout, model.layer_norm_eps)
    ffn = PositionwiseFeedForward(model.d_model, model.d_ff, model.dropout)
    modules = nn.ModuleList([embedding, attention, first, second, ffn]).eval()
    assert first.norm.gamma is not second.norm.gamma
    assert first.norm.beta is not second.norm.beta

    def run() -> Tensor:
        x = embedding(ids)
        a = attention(x, attention_mask=key_mask)
        y = first(x, a, valid)
        f = ffn(y)
        z = second(y, f, valid)
        for tensor in (x, a, y, f, z):
            assert tensor.shape == (*ids.shape, model.d_model)
            assert torch.isfinite(tensor).all()
        for tensor in (x, y, z):
            assert torch.count_nonzero(tensor[~valid]) == 0
        return z

    output = run()
    torch.testing.assert_close(run(), output)
    before = [parameter.detach().clone() for parameter in modules.parameters()]
    optimizer = torch.optim.SGD(modules.parameters(), lr=0.01)
    (output[valid] - torch.randn_like(output[valid])).square().mean().backward()
    for parameter in modules.parameters():
        assert parameter.requires_grad and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad)
    weight = embedding.token_embedding.embedding.weight
    assert torch.count_nonzero(weight.grad[tokenizer.pad_id]) == 0
    optimizer.step()
    assert all(not torch.equal(parameter, old) for parameter, old in zip(modules.parameters(), before))
    assert torch.count_nonzero(weight[tokenizer.pad_id]) == 0

    path = tmp_path / "phase5.pt"
    torch.save(modules.state_dict(), path)
    with torch.no_grad():
        expected = run()
        second.norm.beta.add_(1)
        assert not torch.allclose(run(), expected)
        modules.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(run(), expected)


@pytest.mark.parametrize(("field", "value"), [("normalization_style", "pre_norm"), ("ffn_activation", "gelu")])
def test_config_rejects_unsupported_styles(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value}).validate_sublayer_settings()
