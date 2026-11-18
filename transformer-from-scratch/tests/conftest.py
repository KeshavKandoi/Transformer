"""Shared fixtures for Phase 9 training tests."""

import pytest
import torch
from torch import nn

from src.training.trainer import TransformerTrainer


class TinyTokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(9, 5)
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(5, 9)
        self.calls = []

    def forward(self, encoder_input_ids, decoder_input_ids):
        self.calls.append((self.training, torch.is_grad_enabled(), decoder_input_ids.clone()))
        values = self.embedding(decoder_input_ids) + self.embedding(encoder_input_ids[:, :1])
        return self.head(self.dropout(values))


def example(source, inputs, labels):
    return {
        "encoder_input_ids": torch.tensor(source), "decoder_input_ids": torch.tensor(inputs),
        "decoder_target_ids": torch.tensor(labels),
    }


@pytest.fixture
def training_batches():
    return [
        example([[4, 5]], [[2, 6, 0]], [[6, 3, 0]]),
        example([[5, 6]], [[2, 7, 8]], [[7, 8, 3]]),
        example([[6, 7]], [[2, 0, 0]], [[3, 0, 0]]),
    ]


@pytest.fixture
def trainer_factory(training_batches):
    def make(config, *, dropout=0.25, **kwargs):
        model = TinyTokenModel()
        model.dropout.p = dropout

        def train_loader(epoch):
            order = torch.randperm(len(training_batches), generator=torch.Generator().manual_seed(config.seed + epoch))
            return [training_batches[index] for index in order.tolist()]

        return TransformerTrainer(
            model, train_loader, training_batches, config, torch.device("cpu"),
            pad_id=0, model_config={"width": 5},
            tokenizer_metadata={"vocab_size": 9, "special_token_ids": {"<pad>": 0}, "tokenizer_sha256": "example"},
            data_identity={"split": "fixture"}, **kwargs,
        )

    return make


@pytest.fixture(scope="session")
def translation_checkpoint(tmp_path_factory):
    from argparse import Namespace
    from dataclasses import asdict

    from scripts.prepare_data import prepare
    from src.data.config import ROOT, DataConfig
    from src.model.config import ModelConfig
    from src.model.transformer import EncoderDecoderTransformer
    from src.training.checkpoint import CheckpointManager
    from src.training.config import TrainingConfig
    from src.training.utils import load_prepared_data, seed_everything

    root = tmp_path_factory.mktemp("translation-checkpoint")
    config_path = ROOT / "configs/data_config.yaml"
    prepare(Namespace(
        config=config_path, input=ROOT / "data/raw/sample_parallel.csv",
        processed_dir=root / "splits", tokenizer_dir=root / "tokenizer",
    ))
    data = load_prepared_data(root / "splits", root / "tokenizer", DataConfig.load(config_path))
    config = ModelConfig(d_model=16, d_ff=32, num_encoder_layers=1, num_decoder_layers=1)
    seed_everything(42)
    model = EncoderDecoderTransformer(data.tokenizer.vocab_size, data.tokenizer.pad_id, config)
    training = next(iter(data.loader("train", 2, 42)))
    validation = next(iter(data.loader("val", 2, 42)))
    trainer = TransformerTrainer(
        model, lambda _: [training], [validation], TrainingConfig(epochs=1), torch.device("cpu"),
        pad_id=data.tokenizer.pad_id, model_config=asdict(config), tokenizer_metadata=data.tokenizer_metadata,
        data_identity=data.data_identity, checkpoint_manager=CheckpointManager(root),
    )
    trainer.fit()
    return root / "best.pt", root / "tokenizer", config
