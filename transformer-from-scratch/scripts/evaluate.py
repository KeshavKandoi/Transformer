"""Evaluate a checkpoint with teacher-forced token metrics."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.checkpoint import load_checkpoint, validate_checkpoint
from src.training.trainer import evaluate_model
from src.training.utils import add_data_arguments, load_prepared_data, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.set_defaults(model_config=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        data_config = DataConfig.load(args.data_config)
        model_config = (ModelConfig.load(args.model_config) if args.model_config is not None
                        else ModelConfig(**checkpoint["model_config"]))
        model_config.validate_sequence_lengths(data_config.max_source_length, data_config.max_target_length)
        data = load_prepared_data(args.processed_dir, args.tokenizer_dir, data_config, splits=(args.split,))
        validate_checkpoint(checkpoint, asdict(model_config), data.tokenizer_metadata, data.data_identity)
        device = select_device(args.device)
        model = EncoderDecoderTransformer(data.tokenizer.vocab_size, data.tokenizer.pad_id, model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        batch_size = args.batch_size if args.batch_size is not None else checkpoint["training_config"]["batch_size"]
        if batch_size <= 0:
            raise ValueError("batch-size must be positive")
        loader = data.loader(args.split, batch_size, checkpoint["training_config"]["seed"])
        metrics = evaluate_model(model, loader, device, data.tokenizer.pad_id)
        print(json.dumps({"split": args.split, "examples": len(loader.dataset), "device": str(device), **metrics}, allow_nan=False))
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
