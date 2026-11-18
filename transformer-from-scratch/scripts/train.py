"""Train the custom English–Hindi Transformer or resume at an epoch boundary."""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.overfit_single_batch import overfit
from src.data.config import ROOT, DataConfig
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.checkpoint import CheckpointManager, load_checkpoint
from src.training.config import SchedulerConfig, TrainingConfig
from src.training.loss import masked_cross_entropy
from src.training.scheduler import resolve_schedule
from src.training.trainer import TransformerTrainer
from src.training.utils import add_data_arguments, forward_batch, load_prepared_data, seed_everything, select_device

LOGGER = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:
    resume = load_checkpoint(args.resume) if args.resume else None
    if args.config is not None:
        config = TrainingConfig.load(args.config)
    elif resume is not None:
        config = TrainingConfig.from_dict(resume["training_config"])
    else:
        config = TrainingConfig.load(ROOT / "configs/training_config.yaml")
    overrides = {key: getattr(args, key) for key in ("epochs", "batch_size", "device") if getattr(args, key) is not None}
    config = replace(config, **overrides)
    model_config = ModelConfig.load(args.model_config)
    data_config = DataConfig.load(args.data_config)
    model_config.validate_sequence_lengths(data_config.max_source_length, data_config.max_target_length)
    data = load_prepared_data(args.processed_dir, args.tokenizer_dir, data_config)
    if config.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    device = select_device(config.device)
    batches = math.ceil(len(data.datasets["train"]) / config.batch_size)
    updates_per_epoch = math.ceil(batches / config.gradient_accumulation_steps)
    scheduler = config.scheduler
    if resume is not None and scheduler.warmup_fraction is not None:
        saved = SchedulerConfig(**resume["training_config"]["scheduler"])
        if saved.warmup_fraction == scheduler.warmup_fraction and saved.type == scheduler.type:
            scheduler = saved
    planned_epochs = getattr(args, "planned_epochs", None)
    if planned_epochs is not None and (resume is not None or planned_epochs < config.epochs):
        raise ValueError("--planned-epochs is for a fresh run and must be at least --epochs")
    config = replace(config, scheduler=resolve_schedule(scheduler, updates_per_epoch * (planned_epochs or config.epochs)))
    LOGGER.info("Resolved settings: %s", json.dumps({"training": asdict(config), "model": asdict(model_config), "data": asdict(data_config)}))
    LOGGER.info("Device=%s vocabulary=%d pad_id=%d splits=%s", device, data.tokenizer.vocab_size,
                data.tokenizer.pad_id, {split: len(dataset) for split, dataset in data.datasets.items()})
    if args.overfit_one_batch:
        if resume is not None:
            raise ValueError("--overfit-one-batch cannot be combined with --resume")
        overfit(data, config.device, seed=config.seed, batch_size=min(2, config.batch_size))
        return

    seed_everything(config.seed)
    model = EncoderDecoderTransformer(data.tokenizer.vocab_size, data.tokenizer.pad_id, model_config).to(device)
    LOGGER.info("Trainable parameters: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def train_loader(epoch: int):
        return data.loader("train", config.batch_size, (config.seed + epoch) % 2**63, shuffle=True)

    LOGGER.info("Batches per epoch=%d optimizer updates per epoch=%d", batches, updates_per_epoch)
    if args.dry_run:
        if resume is not None:
            raise ValueError("--dry-run cannot be combined with --resume")
        model.train()
        logits, labels = forward_batch(model, next(iter(train_loader(1))), device)
        loss, count = masked_cross_entropy(logits, labels, data.tokenizer.pad_id, config.label_smoothing)
        (loss / count).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
        LOGGER.info("Dry-run forward/backward passed: logits=%s loss=%.6f tokens=%d preclip_norm=%.6f",
                    tuple(logits.shape), loss.item() / count, count, norm.item())
        return

    name = args.run_name
    if name is None:
        name = args.resume.parent.name if args.resume else datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("--run-name must be a single directory name")
    checkpoint_dir = args.checkpoint_dir / name
    if args.resume and args.run_name is None:
        checkpoint_dir = args.resume.parent
    manager = CheckpointManager(checkpoint_dir)
    metrics_path = args.runs_dir / name / "metrics.jsonl"
    latest = checkpoint_dir / "latest.pt"
    if resume is None and (latest.exists() or metrics_path.exists()):
        raise ValueError("Run already exists. Use --resume or choose another --run-name.")
    if resume is not None and latest.exists() and load_checkpoint(latest)["epoch"] > resume["epoch"]:
        raise ValueError("The run already has a newer checkpoint. Choose another --run-name to branch.")
    trainer = TransformerTrainer(
        model, train_loader, data.loader("val", config.batch_size, config.seed),
        config, device, pad_id=data.tokenizer.pad_id, model_config=asdict(model_config),
        tokenizer_metadata=data.tokenizer_metadata, data_identity=data.data_identity,
        checkpoint_manager=manager, metrics_path=metrics_path,
    )
    if resume is not None:
        trainer.restore(resume)
        LOGGER.info("Resumed through epoch %d, optimizer step %d, next LR %.8g",
                    trainer.epoch, trainer.global_step, trainer.optimizer.param_groups[0]["lr"])
    LOGGER.info("Checkpoints: %s | metrics: %s", checkpoint_dir, metrics_path)
    trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--epochs", type=int, help="Total target epoch count, including resumed epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--planned-epochs", type=int, help="Warmup planning horizon when stopping a fresh run early")
    parser.add_argument("--resume", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Resume from the last complete epoch checkpoint if available.")
        raise SystemExit(130) from None
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
