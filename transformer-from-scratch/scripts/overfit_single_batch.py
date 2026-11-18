"""Verify that a small Transformer can fit a fixed real training batch."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.training.trainer import TransformerTrainer
from src.training.utils import PreparedData, add_data_arguments, load_prepared_data, seed_everything, select_device

LOGGER = logging.getLogger(__name__)


def overfit(
    data: PreparedData, device_name: str, *, seed: int = 42, batch_size: int = 2,
    max_steps: int = 100, learning_rate: float = 0.01, target_loss: float = 0.35,
    curve: Path | None = None,
) -> dict:
    if max_steps < 1 or batch_size < 1 or not 0 < target_loss < float("inf"):
        raise ValueError("Overfit budget, batch size and target loss must be positive")
    device = select_device(device_name)
    seed_everything(seed)
    batch = next(iter(data.loader("train", batch_size, seed)))
    tokenizer = data.tokenizer
    model_config = ModelConfig(
        d_model=32, num_heads=4, d_ff=64, dropout=0.0, attention_dropout=0.0,
        num_encoder_layers=1, num_decoder_layers=1,
        max_seq_len=max(batch["encoder_input_ids"].size(1), batch["decoder_input_ids"].size(1)),
    )
    model = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, model_config)
    config = TrainingConfig(
        batch_size=batch_size, seed=seed, device=device_name,
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        scheduler=SchedulerConfig(type="constant"),
    )
    trainer = TransformerTrainer(
        model, lambda _: [batch], [batch], config, device,
        pad_id=tokenizer.pad_id, model_config={}, tokenizer_metadata=data.tokenizer_metadata,
        data_identity=data.data_identity,
    )
    initial = trainer.validate()
    history = [{"step": 0, **initial}]
    LOGGER.info("Fixed real batch on %s, initial loss %.6f, %d valid labels", device, initial["loss"], initial["valid_tokens"])
    for step in range(1, max_steps + 1):
        trainer.train_epoch([batch])
        final = trainer.validate()
        history.append({"step": step, **final})
        if step % 10 == 0:
            LOGGER.info("overfit step=%d loss=%.6f accuracy=%.4f", step, final["loss"], final["accuracy"])
        if final["loss"] <= min(target_loss, initial["loss"] / 2) and final["accuracy"] >= 0.95:
            break
    if curve is not None:
        curve.parent.mkdir(parents=True, exist_ok=True)
        curve.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8")
    LOGGER.info("Fixed batch after %d updates: loss %.6f -> %.6f, token accuracy %.4f",
                step, initial["loss"], final["loss"], final["accuracy"])
    if final["loss"] > min(target_loss, initial["loss"] / 2) or final["accuracy"] < 0.95:
        raise ValueError("Fixed batch did not meet the configured loss target and 95% token accuracy")
    return {"initial": initial, "final": final, "steps": step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--target-loss", type=float, default=0.35)
    parser.add_argument("--curve", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        data = load_prepared_data(args.processed_dir, args.tokenizer_dir, DataConfig.load(args.data_config), splits=("train",))
        overfit(data, args.device, batch_size=args.batch_size, max_steps=args.max_steps,
                learning_rate=args.learning_rate, target_loss=args.target_loss, curve=args.curve)
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
