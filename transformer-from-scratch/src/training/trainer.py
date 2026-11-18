"""Training, validation and optimizer-boundary checkpoint orchestration."""

import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sized
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .checkpoint import CheckpointManager, experiment_fingerprints, validate_checkpoint
from .config import EarlyStoppingConfig, TrainingConfig
from .loss import masked_cross_entropy
from .metrics import TokenMetrics
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .utils import capture_rng, forward_batch, restore_rng

LOGGER = logging.getLogger(__name__)
Batches = Iterable[dict[str, Tensor]]


def peak_process_memory() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


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


@dataclass
class EpochProgress:
    epoch: int
    batch_count: int
    next_batch: int = 0
    updates: int = 0
    metrics: TokenMetrics = field(default_factory=TokenMetrics)
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    max_gradient_norm: float = 0.0


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
        resume_train_loader: Callable[[int, int], Batches] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.resume_train_loader = resume_train_loader
        self.progress: EpochProgress | None = None
        self._boundary_rng: dict | None = None
        self._update_in_flight = False
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
        """Train on a stand-alone batch stream. Use fit for resumable epochs."""
        return self._train_epoch(batches, None)

    def _train_epoch(self, batches: Batches, stop_requested: Callable[[], bool] | None) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress = self.progress
        metrics = copy(progress.metrics) if progress else TokenMetrics()
        window_batches = window_tokens = 0
        last_norm = progress.gradient_norm if progress else 0.0
        max_norm = progress.max_gradient_norm if progress else 0.0
        used_lr = progress.learning_rate if progress else 0.0
        updates = progress.updates if progress else 0
        index = progress.next_batch if progress else 0

        def check_stop() -> None:
            if stop_requested is not None and stop_requested():
                raise KeyboardInterrupt

        def update_window() -> None:
            nonlocal last_norm, used_lr, max_norm, updates
            # An actual exception inside optimizer.step or scheduler.step can leave
            # mixed state. Only disk checkpoints are safe while this flag is set.
            self._update_in_flight = True
            last_norm, used_lr = self._update(window_tokens)
            max_norm = max(max_norm, last_norm)
            updates += 1
            if progress is not None:
                progress.next_batch = index
                progress.updates = updates
                progress.metrics = copy(metrics)
                progress.learning_rate = used_lr
                progress.gradient_norm = last_norm
                progress.max_gradient_norm = max_norm
                self._boundary_rng = capture_rng(self.device)
            self._update_in_flight = False
            if self.global_step % self.config.logging.log_every_steps == 0:
                LOGGER.info("step=%d lr=%.8g grad_norm=%.6f window_tokens=%d",
                            self.global_step, used_lr, last_norm, window_tokens)
            frequency = self.config.save_every_updates
            if progress is not None and self.checkpoint_manager and frequency and self.global_step % frequency == 0:
                self._save_update()
            check_stop()

        try:
            check_stop()
            for index, batch in enumerate(batches, start=index + 1):
                check_stop()
                if progress is not None and index > progress.batch_count:
                    raise ValueError("Training loader exceeded the recorded epoch batch count")
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
                    update_window()
                    window_batches = window_tokens = 0
                else:
                    check_stop()
            if progress is not None and index != progress.batch_count:
                raise ValueError("Training loader length differs from the recorded epoch batch count")
            if window_batches:
                update_window()
        finally:
            self.optimizer.zero_grad(set_to_none=True)

        summary = metrics.summary()
        return {
            "loss": summary["loss"], "accuracy": summary["accuracy"], "valid_tokens": summary["valid_tokens"],
            "updates": updates, "learning_rate": used_lr, "gradient_norm": last_norm,
            "max_gradient_norm": max_norm,
        }

    def _save_update(self) -> None:
        if (self.checkpoint_manager is None or self.progress is None or not self.progress.next_batch
                or self._update_in_flight or self._boundary_rng is None
                or any(parameter.grad is not None for parameter in self.model.parameters())):
            raise RuntimeError("A mid-epoch checkpoint requires a committed optimizer boundary")
        path = self.checkpoint_manager.save_update(self.state_dict())
        LOGGER.info("Saved %s at epoch %d, next batch offset %d/%d, optimizer step %d",
                    path, self.progress.epoch, self.progress.next_batch, self.progress.batch_count, self.global_step)

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
            "rng_state": self._boundary_rng if self.progress is not None else capture_rng(self.device),
            **({"epoch_progress": asdict(self.progress)} if self.progress is not None else {}),
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "device": str(self.device)},
        }

    def restore(self, payload: dict) -> None:
        validate_checkpoint(payload, self.model_config, self.tokenizer_metadata, self.data_identity)
        current = asdict(self.config)
        saved = asdict(TrainingConfig.from_dict(payload["training_config"]))
        for key in current.keys() - {"epochs", "device", "checkpoint_frequency", "save_every_updates", "logging"}:
            if current[key] != saved.get(key):
                raise ValueError(f"Checkpoint training configuration differs for {key}")
        epoch, global_step = payload["epoch"], payload["global_step"]
        raw_progress = payload.get("epoch_progress")
        if type(epoch) is not int or not 0 <= epoch <= self.config.epochs or (epoch == 0 and raw_progress is None):
            raise ValueError("Checkpoint epoch is incompatible with the requested total epochs")
        if type(global_step) is not int or global_step < epoch or payload["scheduler_state_dict"]["last_epoch"] != global_step:
            raise ValueError("Checkpoint has inconsistent optimizer and scheduler step counters")
        progress = None
        if raw_progress is not None:
            if self.resume_train_loader is None:
                raise ValueError("Mid-epoch resume requires a cursor-aware training loader")
            try:
                progress = EpochProgress(**{**raw_progress, "metrics": TokenMetrics(**raw_progress["metrics"])})
            except (TypeError, KeyError) as error:
                raise ValueError("Invalid checkpoint epoch progress") from error
            accumulation = self.config.gradient_accumulation_steps
            cursor = progress.next_batch
            if (
                any(type(value) is not int for value in (progress.epoch, progress.batch_count, cursor, progress.updates))
                or progress.epoch != epoch + 1 or progress.epoch > self.config.epochs
                or not 0 < cursor <= progress.batch_count
                or (cursor != progress.batch_count and cursor % accumulation)
                or progress.updates != math.ceil(cursor / accumulation)
                or global_step < epoch + progress.updates
                or type(progress.metrics.token_count) is not int or progress.metrics.token_count < cursor
                or type(progress.metrics.correct) is not int
                or not 0 <= progress.metrics.correct <= progress.metrics.token_count
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                       for value in (progress.metrics.loss_sum, progress.learning_rate,
                                     progress.gradient_norm, progress.max_gradient_norm))
            ):
                raise ValueError("Invalid checkpoint optimizer-boundary cursor or epoch metrics")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.epoch, self.global_step = epoch, global_step
        self.best_validation_loss = payload["best_validation_loss"]
        self.early_stopping = EarlyStopping(self.config.early_stopping, **payload["early_stopping_state"])
        restore_rng(payload["rng_state"], self.device)
        self.progress = progress
        self._boundary_rng = capture_rng(self.device) if progress is not None else None
        if "epoch_metrics" in payload:
            self._write_epoch_metrics(payload["epoch_metrics"])

    def _write_epoch_metrics(self, entry: dict) -> None:
        """Publish an epoch row once, including after a checkpoint/log interruption."""
        path = self.metrics_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = previous.strip().splitlines()
        if lines and json.loads(lines[-1])["epoch"] >= entry["epoch"]:
            return
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(previous)
                if previous and not previous.endswith("\n"):
                    stream.write("\n")
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def fit(self, stop_requested: Callable[[], bool] | None = None) -> list[dict]:
        try:
            return self._fit(stop_requested)
        except KeyboardInterrupt:
            if (self.checkpoint_manager is not None and self.progress is not None
                    and self.progress.next_batch and not self._update_in_flight):
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    restore_rng(self._boundary_rng, self.device)
                    self._save_update()
                except (OSError, RuntimeError, KeyboardInterrupt) as error:
                    LOGGER.warning("Could not publish interruption checkpoint: %s", error)
            raise

    def _fit(self, stop_requested: Callable[[], bool] | None) -> list[dict]:
        history = []
        while self.epoch < self.config.epochs and not self.early_stopping.should_stop:
            epoch = self.epoch + 1
            started = time.perf_counter()
            if stop_requested is not None and stop_requested():
                raise KeyboardInterrupt
            if self.progress is not None:
                if self.resume_train_loader is None:
                    raise ValueError("Mid-epoch resume requires a cursor-aware training loader")
                batches = self.resume_train_loader(epoch, self.progress.next_batch)
                if not isinstance(batches, Sized) or len(batches) != self.progress.batch_count - self.progress.next_batch:
                    raise ValueError("Resumed loader length does not match the checkpoint batch cursor")
            else:
                batches = self.train_loader(epoch)
                if self.resume_train_loader is not None:
                    if not isinstance(batches, Sized):
                        raise ValueError("A cursor-aware training loader must report its batch count")
                    self.progress = EpochProgress(epoch, len(batches))
                    self._boundary_rng = capture_rng(self.device)
            train = self._train_epoch(batches, stop_requested)
            if stop_requested is not None and stop_requested():
                raise KeyboardInterrupt
            validation = None
            improved = False
            if epoch == 1 or epoch % self.config.validation_frequency == 0:
                validation = self.validate()
                if stop_requested is not None and stop_requested():
                    raise KeyboardInterrupt
                loss = validation["loss"]
                improved = self.best_validation_loss is None or loss < self.best_validation_loss
                if improved:
                    self.best_validation_loss = loss
                self.early_stopping.update(loss)
                if validation["perplexity_status"] == "overflow":
                    LOGGER.warning("Validation perplexity overflows at epoch %d", epoch)
            self.progress = None
            self._boundary_rng = None
            self.epoch = epoch
            entry = {
                "epoch": epoch, "global_step": self.global_step, "train": train, "validation": validation,
                "seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_process_memory(),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "best_validation_loss": self.best_validation_loss,
            }
            if self.checkpoint_manager is not None or self.metrics_path is not None:
                state = self.state_dict()
                entry["fingerprints"] = experiment_fingerprints(state)
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_epoch(
                    {**state, "epoch_metrics": entry}, best=improved,
                    snapshot=epoch % self.config.checkpoint_frequency == 0,
                )
            self._write_epoch_metrics(entry)
            LOGGER.info("epoch=%d step=%d train_loss=%.6f train_accuracy=%.4f train_tokens=%d validation=%s elapsed=%.2fs",
                        epoch, self.global_step, train["loss"], train["accuracy"], train["valid_tokens"],
                        validation, entry["seconds"])
            history.append(entry)
            if stop_requested is not None and stop_requested():
                raise KeyboardInterrupt
        if self.early_stopping.should_stop:
            LOGGER.info("Early stopping after %d non-improving validation checks", self.early_stopping.bad_checks)
        return history
