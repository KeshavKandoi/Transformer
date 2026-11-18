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
