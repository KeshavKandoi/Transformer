from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from scripts.overfit_single_batch import overfit
from scripts.prepare_data import prepare
from src.data.config import ROOT, DataConfig
from src.data.masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.checkpoint import CheckpointManager, load_checkpoint
from src.training.config import TrainingConfig
from src.training.loss import masked_cross_entropy
from src.training.trainer import TransformerTrainer
from src.training.utils import forward_batch, load_prepared_data, seed_everything


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("prepared-training")
    config_path = ROOT / "configs/data_config.yaml"
    prepare(Namespace(
        config=config_path, input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=directory / "splits", tokenizer_dir=directory / "tokenizer",
    ))
    return directory, DataConfig.load(config_path)


def test_default_model_trains_validates_checkpoints_and_resumes(prepared, tmp_path: Path) -> None:
    directory, data_config = prepared
    data = load_prepared_data(directory / "splits", directory / "tokenizer", data_config)
    config = ModelConfig.load(ROOT / "configs/model_config.yaml")
    config.validate_sequence_lengths(data_config.max_source_length, data_config.max_target_length)
    tokenizer = data.tokenizer
    assert data.tokenizer_metadata["vocab_size"] == tokenizer.vocab_size
    assert data.tokenizer_metadata["special_token_ids"]["<pad>"] == tokenizer.pad_id
    batch = next(iter(data.loader("train", 8, 42)))
    ids, targets = batch["decoder_input_ids"], batch["decoder_target_ids"]
    assert ids.size(0) == 8 and (ids[:, 0] == tokenizer.bos_id).all()
    valid = targets.ne(tokenizer.pad_id)
    assert not valid.all() and (targets.gather(1, valid.sum(-1, keepdim=True) - 1) == tokenizer.eos_id).all()
    assert torch.equal(ids[:, 1:][valid[:, 1:]], targets[:, :-1][valid[:, 1:]])
    assert torch.count_nonzero(batch["decoder_attention_mask"].triu(1)) == 0
    assert not batch["cross_attention_mask"][0, 0, 0, batch["encoder_input_ids"][0].eq(tokenizer.pad_id)].any()

    def factory(training: TrainingConfig):
        model = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, config)
        return TransformerTrainer(
            model, lambda epoch: data.loader("train", 8, 42 + epoch, shuffle=True),
            data.loader("val", 3, 42), training, torch.device("cpu"),
            pad_id=tokenizer.pad_id, model_config=asdict(config), tokenizer_metadata=data.tokenizer_metadata,
            data_identity=data.data_identity, checkpoint_manager=CheckpointManager(tmp_path),
        )

    seed_everything(42)
    training = TrainingConfig(epochs=1, gradient_accumulation_steps=3)
    trainer = factory(training)
    trainer.model.eval()
    with torch.no_grad():
        logits, actual_targets = forward_batch(trainer.model, batch, trainer.device)
        assert logits.shape == (*targets.shape, tokenizer.vocab_size) and torch.isfinite(logits).all()
        assert torch.equal(actual_targets, targets)
        first = batch["decoder_input_ids"].clone()
        first[0, 2] = tokenizer.eos_id
        changed, _ = forward_batch(trainer.model, {**batch, "decoder_input_ids": first}, trainer.device)
        torch.testing.assert_close(logits[0, :2], changed[0, :2])

    names = (
        "encoder.embedding.token_embedding.embedding.weight",
        "decoder.stack.layers.0.cross_attention.heads.0.Wq.weight",
        "output_projection.projection.weight",
    )
    parameters = dict(trainer.model.named_parameters())
    seen = {}
    for name in names:
        parameters[name].register_hook(lambda gradient, name=name: seen.update({name: gradient.detach().norm()}))
    before = {name: parameters[name].detach().clone() for name in names}
    history = trainer.fit()
    assert history[0]["train"]["updates"] == 2
    assert history[0]["validation"]["valid_tokens"] > 0
    assert all(torch.isfinite(seen[name]) and seen[name] > 0 for name in names)
    assert all(not torch.equal(before[name], parameters[name]) for name in names)
    payload = load_checkpoint(tmp_path / "latest.pt")
    resumed = factory(replace(training, epochs=2))
    resumed.restore(payload)
    assert resumed.epoch == 1 and resumed.global_step == 2
    assert resumed.validate() == trainer.validate()
    assert resumed.fit()[0]["global_step"] == 4


def test_real_fixed_batch_learns_and_mismatched_artifacts_are_rejected(prepared, tmp_path: Path) -> None:
    directory, config = prepared
    data = load_prepared_data(directory / "splits", directory / "tokenizer", config)
    result = overfit(data, "cpu", max_steps=50, curve=tmp_path / "curve.jsonl")
    assert result["final"]["loss"] < 0.35 and result["final"]["accuracy"] >= 0.95
    assert len((tmp_path / "curve.jsonl").read_text().splitlines()) == result["steps"] + 1
    with pytest.raises(ValueError, match="do not match"):
        load_prepared_data(directory / "splits", directory / "tokenizer", replace(config, seed=43))
    with pytest.raises(ValueError, match="Missing.*manifest"):
        load_prepared_data(tmp_path, directory / "tokenizer", config)


def _maximum_length_step(device: torch.device) -> None:
    config = ModelConfig(d_model=16, d_ff=32, num_encoder_layers=1, num_decoder_layers=1)
    model = EncoderDecoderTransformer(32, 0, config).to(device)
    source = torch.zeros(1, 128, dtype=torch.long, device=device)
    target = torch.zeros_like(source)
    source[0, :3] = torch.tensor([4, 5, 3], device=device)
    target[0, :3] = torch.tensor([2, 7, 8], device=device)
    labels = torch.zeros_like(target)
    labels[0, :3] = torch.tensor([7, 8, 3], device=device)
    logits = model(source, target, encoder_attention_mask(source, 0),
                   decoder_attention_mask(target, 0), cross_attention_mask(source, 128, 0))
    loss, count = masked_cross_entropy(logits, labels, 0)
    assert count == 3 and torch.isfinite(logits).all()
    (loss / count).backward()
    assert torch.isfinite(model.output_projection.projection.weight.grad).all()


def test_maximum_lengths_and_padding_heavy_batch_size_one() -> None:
    _maximum_length_step(torch.device("cpu"))


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available for the Phase 9 training smoke test",
)
def test_accelerator_training_smoke() -> None:
    _maximum_length_step(torch.device("cuda" if torch.cuda.is_available() else "mps"))
