"""Fingerprints for matching prepared splits and their tokenizer."""

import hashlib
import json
from pathlib import Path

from .config import DataConfig
from .tokenizer import BilingualBPETokenizer

SPLITS = ("train", "val", "test")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preparation_settings(config: DataConfig) -> dict:
    return {
        name: getattr(config, name)
        for name in ("seed", "vocab_size", "train_ratio", "val_ratio", "test_ratio")
    }


def prepared_fingerprints(processed_dir: Path, tokenizer_dir: Path) -> dict:
    return {
        "splits": {name: file_digest(processed_dir / f"{name}.csv") for name in SPLITS},
        "tokenizer_sha256": file_digest(tokenizer_dir / "tokenizer.json"),
    }


def write_preparation_manifest(
    processed_dir: Path, tokenizer_dir: Path, source_path: Path, config: DataConfig,
) -> None:
    manifest = {
        "version": 1,
        "input_sha256": file_digest(source_path),
        "preparation_settings": preparation_settings(config),
        **prepared_fingerprints(processed_dir, tokenizer_dir),
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_tokenizer_artifact(
    tokenizer_dir: Path, tokenizer_digest: str | None = None,
) -> tuple[BilingualBPETokenizer, dict]:
    """Load the tokenizer and check metadata against its actual token IDs."""
    tokenizer = BilingualBPETokenizer.load(tokenizer_dir)
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": {
            "<pad>": tokenizer.pad_id, "<unk>": tokenizer.unk_id,
            "<bos>": tokenizer.bos_id, "<eos>": tokenizer.eos_id,
        },
    }
    saved_metadata = json.loads((tokenizer_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata != saved_metadata:
        raise ValueError("Tokenizer metadata does not match tokenizer.json")
    metadata["tokenizer_sha256"] = tokenizer_digest or file_digest(tokenizer_dir / "tokenizer.json")
    return tokenizer, metadata
