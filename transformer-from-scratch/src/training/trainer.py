"""Training, validation and epoch-boundary checkpoint orchestration."""

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .checkpoint import CheckpointManager, validate_checkpoint
from .config import EarlyStoppingConfig, TrainingConfig
from .loss import masked_cross_entropy
from .metrics import TokenMetrics
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .utils import capture_rng, forward_batch, restore_rng

LOGGER = logging.getLogger(__name__)
Batches = Iterable[dict[str, Tensor]]


@dataclass
class EarlyStopping:
    config: EarlyStoppingConfig
    reference: float | None = None
    bad_checks: int = 0

    @property
    def should_stop(self) -> bool:
        return self.config.enabled and self.bad_checks >= self.config.patience

    def update(self, validation_loss: float) -> bool:
        if self.reference is None or self.reference - validation_loss > self.config.min_delta:
            self.reference = validation_loss
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.should_stop


@torch.inference_mode()
def evaluate_model(model: nn.Module, batches: Batches, device: torch.device, pad_id: int) -> dict:
    """Use unsmoothed token cross-entropy and restore the caller's model mode."""
    was_training = model.training
    model.eval()
    metrics = TokenMetrics()
    try:
        for batch in batches:
            logits, targets = forward_batch(model, batch, device)
            loss, count = masked_cross_entropy(logits, targets, pad_id)
            metrics.add(loss, count, logits, targets, pad_id)
        return metrics.summary()
    finally:
        model.train(was_training)


class TransformerTrainer:
    def __init__(
        self, model: nn.Module, train_loader: Callable[[int], Batches],
        validation_loader: Batches, config: TrainingConfig, device: torch.device,
        *, pad_id: int, model_config: dict, tokenizer_metadata: dict, data_identity: dict,
        checkpoint_manager: CheckpointManager | None = None, metrics_path: Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = device
        self.pad_id = pad_id
        self.model_config = model_config
        self.tokenizer_metadata = tokenizer_metadata
        self.data_identity = data_identity
        self.checkpoint_manager = checkpoint_manager
        self.metrics_path = metrics_path
        if config.mixed_precision:
            LOGGER.warning("Phase 9 runs in float32. Mixed precision was requested and is disabled.")
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.scheduler)
        self.early_stopping = EarlyStopping(config.early_stopping)
        self.epoch = 0
        self.global_step = 0
        self.best_validation_loss: float | None = None

    def _update(self, window_tokens: int) -> tuple[float, float]:
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            raise RuntimeError("No trainable parameter received gradients")
        for parameter in parameters:
            parameter.grad.div_(window_tokens)
        try:
            norm = nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm, error_if_nonfinite=True)
        except RuntimeError as error:
            raise FloatingPointError(
                f"Gradient clipping failed before optimizer update {self.global_step + 1}: {error}"
            ) from error
        used_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm.item()), used_lr

    def train_epoch(self, batches: Batches) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = TokenMetrics()
        window_batches = window_tokens = 0
        last_norm = max_norm = used_lr = 0.0
        initial_step = self.global_step

        def update_window() -> tuple[float, float]:
            norm, rate = self._update(window_tokens)
            if self.global_step % self.config.logging.log_every_steps == 0:
                LOGGER.info("step=%d lr=%.8g grad_norm=%.6f window_tokens=%d",
                            self.global_step, rate, norm, window_tokens)
            return norm, rate

        try:
            for index, batch in enumerate(batches, start=1):
                logits, targets = forward_batch(self.model, batch, self.device)
                try:
                    loss, count = masked_cross_entropy(logits, targets, self.pad_id, self.config.label_smoothing)
                except (ValueError, FloatingPointError) as error:
                    raise type(error)(f"Epoch {self.epoch + 1}, batch {index}: {error}") from error
                loss.backward()
                metrics.add(loss, count, logits, targets, self.pad_id)
                window_tokens += count
                window_batches += 1
                if window_batches == self.config.gradient_accumulation_steps:
                    last_norm, used_lr = update_window()
                    max_norm = max(max_norm, last_norm)
                    window_batches = window_tokens = 0
            if window_batches:
                last_norm, used_lr = update_window()
                max_norm = max(max_norm, last_norm)
        finally:
            self.optimizer.zero_grad(set_to_none=True)

        summary = metrics.summary()
        # A smoothed objective is not standard perplexity.
        return {
            "loss": summary["loss"], "accuracy": summary["accuracy"], "valid_tokens": summary["valid_tokens"],
            "updates": self.global_step - initial_step, "learning_rate": used_lr,
            "gradient_norm": last_norm, "max_gradient_norm": max_norm,
        }

    def validate(self) -> dict:
        return evaluate_model(self.model, self.validation_loader, self.device, self.pad_id)

    def state_dict(self) -> dict:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch, "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "early_stopping_state": {
                "reference": self.early_stopping.reference, "bad_checks": self.early_stopping.bad_checks,
            },
            "training_config": asdict(self.config), "model_config": self.model_config,
            "tokenizer_metadata": self.tokenizer_metadata, "data_identity": self.data_identity,
            "rng_state": capture_rng(self.device),
        }

    def restore(self, payload: dict) -> None:
        validate_checkpoint(payload, self.model_config, self.tokenizer_metadata, self.data_identity)
        current, saved = asdict(self.config), payload["training_config"]
        for key in current.keys() - {"epochs", "device", "checkpoint_frequency", "logging"}:
            if current[key] != saved.get(key):
                raise ValueError(f"Checkpoint training configuration differs for {key}")
        epoch, global_step = payload["epoch"], payload["global_step"]
        if type(epoch) is not int or not 0 < epoch <= self.config.epochs:
            raise ValueError("Checkpoint epoch must be positive and no greater than the requested total epochs")
        if type(global_step) is not int or global_step < epoch or payload["scheduler_state_dict"]["last_epoch"] != global_step:
            raise ValueError("Checkpoint has inconsistent optimizer and scheduler step counters")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.epoch, self.global_step = epoch, global_step
        self.best_validation_loss = payload["best_validation_loss"]
        self.early_stopping = EarlyStopping(self.config.early_stopping, **payload["early_stopping_state"])
        restore_rng(payload["rng_state"], self.device)

    def fit(self) -> list[dict]:
        history = []
        while self.epoch < self.config.epochs and not self.early_stopping.should_stop:
            epoch = self.epoch + 1
            started = time.perf_counter()
            train = self.train_epoch(self.train_loader(epoch))
            validation = None
            improved = False
            if epoch == 1 or epoch % self.config.validation_frequency == 0:
                validation = self.validate()
                loss = validation["loss"]
                improved = self.best_validation_loss is None or loss < self.best_validation_loss
                if improved:
                    self.best_validation_loss = loss
                self.early_stopping.update(loss)
                if validation["perplexity_status"] == "overflow":
                    LOGGER.warning("Validation perplexity overflows at epoch %d", epoch)
            self.epoch = epoch
            entry = {
                "epoch": epoch, "global_step": self.global_step, "train": train, "validation": validation,
                "seconds": time.perf_counter() - started,
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "best_validation_loss": self.best_validation_loss,
            }
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_epoch(
                    self.state_dict(), best=improved, snapshot=epoch % self.config.checkpoint_frequency == 0,
                )
            if self.metrics_path is not None:
                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with self.metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
            LOGGER.info("epoch=%d step=%d train_loss=%.6f train_accuracy=%.4f train_tokens=%d validation=%s elapsed=%.2fs",
                        epoch, self.global_step, train["loss"], train["accuracy"], train["valid_tokens"],
                        validation, entry["seconds"])
            history.append(entry)
        if self.early_stopping.should_stop:
            LOGGER.info("Early stopping after %d non-improving validation checks", self.early_stopping.bad_checks)
        return history
