import json
import shutil
from dataclasses import replace

import pytest
import torch

from src.inference.checkpoint_loader import load_translation_model
from src.training.checkpoint import load_checkpoint


def test_checkpoint_restores_saved_weights_config_and_special_ids(translation_checkpoint) -> None:
    path, tokenizer_dir, config = translation_checkpoint
    model, tokenizer = load_translation_model(path, tokenizer_dir, torch.device("cpu"), config)
    payload = load_checkpoint(path)
    assert payload["global_step"] == 1 and model.config == config and not model.training
    assert model.vocab_size == tokenizer.vocab_size
    assert payload["tokenizer_metadata"]["special_token_ids"] == {
        "<pad>": tokenizer.pad_id, "<bos>": tokenizer.bos_id,
        "<eos>": tokenizer.eos_id, "<unk>": tokenizer.unk_id,
    }
    assert all(torch.equal(value, payload["model_state_dict"][name]) for name, value in model.state_dict().items())


def test_rejects_token_mapping_and_requested_architecture_mismatches(translation_checkpoint, tmp_path) -> None:
    path, tokenizer_dir, config = translation_checkpoint
    with pytest.raises(ValueError, match="architecture"):
        load_translation_model(path, tokenizer_dir, torch.device("cpu"), replace(config, d_model=32))
    bad_mapping = tmp_path / "tokenizer"
    shutil.copytree(tokenizer_dir, bad_mapping)
    tokenizer_json = bad_mapping / "tokenizer.json"
    tokenizer_json.write_text(tokenizer_json.read_text() + "\n")
    with pytest.raises(ValueError, match="token mapping"):
        load_translation_model(path, bad_mapping, torch.device("cpu"))
    metadata = bad_mapping / "metadata.json"
    values = json.loads(metadata.read_text())
    values["special_token_ids"]["<bos>"] = values["special_token_ids"]["<pad>"]
    metadata.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="metadata does not match"):
        load_translation_model(path, bad_mapping, torch.device("cpu"))


def test_missing_corrupt_and_untrained_checkpoints(translation_checkpoint, tmp_path) -> None:
    path, tokenizer_dir, _ = translation_checkpoint
    missing = tmp_path / "bad.pt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_translation_model(missing, tokenizer_dir, torch.device("cpu"))
    missing.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cannot read checkpoint"):
        load_translation_model(missing, tokenizer_dir, torch.device("cpu"))
    checkpoint = load_checkpoint(path)
    torch.save({**checkpoint, "global_step": 0}, missing)
    with pytest.raises(ValueError, match="optimizer update"):
        load_translation_model(missing, tokenizer_dir, torch.device("cpu"))
