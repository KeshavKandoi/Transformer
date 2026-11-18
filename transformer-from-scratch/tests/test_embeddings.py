from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.prepare_data import prepare
from src.data.cleaning import read_parallel_csv
from src.data.collator import ParallelCollator
from src.data.config import ROOT, DataConfig
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.embeddings import TransformerEmbedding


def test_pipeline_padding_and_gradients() -> None:
    module = TransformerEmbedding(7, 6, 8, 0.0, pad_id=3)
    ids = torch.tensor([[1, 2, 3], [2, 3, 3]])
    output = module(ids)
    mask = ids.ne(3)
    expected = module.token_embedding(ids) + module.position_encoding.encoding[:ids.size(1)]
    torch.testing.assert_close(output[mask], expected[mask])
    assert torch.count_nonzero(output[~mask]) == 0
    assert output.requires_grad

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    weight = module.token_embedding.embedding.weight
    before = weight.detach().clone()
    output.sum().backward()
    assert weight.grad is not None
    assert torch.count_nonzero(weight.grad[1:3]) == 12
    assert torch.count_nonzero(weight.grad[3]) == 0
    assert module.position_encoding.encoding.grad is None
    optimizer.step()
    assert not torch.equal(weight[1:3], before[1:3])
    assert torch.count_nonzero(weight[3]) == 0


def test_dropout_mode_and_state_dict_round_trip(tmp_path: Path) -> None:
    module = TransformerEmbedding(7, 6, 8, 1.0, pad_id=3)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.count_nonzero(module(ids)) == 0

    module.eval()
    output = module(ids)
    assert torch.count_nonzero(output[0, :2]) > 0
    torch.testing.assert_close(output, module(ids), rtol=0, atol=0)
    path = tmp_path / "embedding.pt"
    torch.save(module.state_dict(), path)
    loaded = TransformerEmbedding(7, 6, 8, 1.0, pad_id=3).eval()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(loaded(ids), output, rtol=0, atol=0)


def test_incompatible_model_context() -> None:
    config = ModelConfig(max_seq_len=8)
    config.validate_sequence_lengths(8, 7)
    with pytest.raises(ValueError, match="must cover both"):
        config.validate_sequence_lengths(8, 9)


def test_real_phase_one_batch(tmp_path: Path) -> None:
    config_path = ROOT / "configs/data_config.yaml"
    processed = tmp_path / "processed"
    artifacts = tmp_path / "tokenizer"
    prepare(Namespace(
        config=config_path,
        input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=processed,
        tokenizer_dir=artifacts,
    ))
    data = DataConfig.load(config_path)
    model = ModelConfig.load(ROOT / "configs/model_config.yaml")
    model.validate_sequence_lengths(data.max_source_length, data.max_target_length)
    tokenizer = BilingualBPETokenizer.load(artifacts)
    pairs, _ = read_parallel_csv(processed / "train.csv")
    dataset = ParallelDataset(pairs, tokenizer, data.max_source_length, data.max_target_length)
    loader = DataLoader(dataset, batch_size=data.batch_size, collate_fn=ParallelCollator(tokenizer.pad_id))
    batch = next(iter(loader))
    original = {key: value.clone() for key, value in batch.items()}
    args = (tokenizer.vocab_size, model.d_model, model.max_seq_len, model.dropout, tokenizer.pad_id)
    encoder = TransformerEmbedding(*args).eval()
    decoder = TransformerEmbedding(*args).eval()
    assert encoder.token_embedding.embedding.weight is not decoder.token_embedding.embedding.weight

    encoder_ids = batch["encoder_input_ids"]
    decoder_ids = batch["decoder_input_ids"]
    assert encoder_ids.size(1) != decoder_ids.size(1)
    assert tokenizer.decode(decoder_ids[0].tolist()) == pairs[0].target
    for ids, module in ((encoder_ids, encoder), (decoder_ids, decoder)):
        output = module(ids)
        assert output.shape == (*ids.shape, model.d_model)
        assert module.token_embedding.embedding.num_embeddings == tokenizer.vocab_size
        assert module.token_embedding.embedding.padding_idx == tokenizer.pad_id
        assert torch.count_nonzero(output[ids.eq(tokenizer.pad_id)]) == 0
        boundary = ids.eq(tokenizer.bos_id) | ids.eq(tokenizer.eos_id)
        assert boundary.any()
        assert output[boundary].abs().sum(dim=-1).gt(0).all()

    for key, value in batch.items():
        assert torch.equal(value, original[key])
