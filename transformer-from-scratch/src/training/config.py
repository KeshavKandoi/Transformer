"""Configuration for training and reproducible epoch-boundary resume."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _number(name: str, value: float, lower: float, *, strict: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < lower or (strict and value == lower):
        raise ValueError(f"{name} must be finite and {'greater than' if strict else 'at least'} {lower}")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-9
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if self.type != "adamw":
            raise ValueError("optimizer.type must be adamw")
        _number("learning_rate", self.learning_rate, 0, strict=True)
        _number("epsilon", self.epsilon, 0, strict=True)
        _number("weight_decay", self.weight_decay, 0)
        for name, beta in (("beta1", self.beta1), ("beta2", self.beta2)):
            _number(name, beta, 0)
            if beta >= 1:
                raise ValueError(f"{name} must be less than 1")


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "inverse_sqrt"
    warmup_steps: int = 100
    warmup_fraction: float | None = None
    planned_updates: int | None = None

    def __post_init__(self) -> None:
        if self.type not in ("constant", "inverse_sqrt"):
            raise ValueError("scheduler.type must be constant or inverse_sqrt")
        _positive_int("warmup_steps", self.warmup_steps)
        if self.warmup_fraction is not None:
            _number("warmup_fraction", self.warmup_fraction, 0, strict=True)
            if self.warmup_fraction > 1 or self.type != "inverse_sqrt":
                raise ValueError("warmup_fraction requires inverse_sqrt and must be at most one")
        if self.planned_updates is not None:
            _positive_int("planned_updates", self.planned_updates)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("early_stopping.enabled must be boolean")
        _positive_int("patience", self.patience)
        _number("min_delta", self.min_delta, 0)


@dataclass(frozen=True)
class LoggingConfig:
    log_every_steps: int = 10

    def __post_init__(self) -> None:
        _positive_int("log_every_steps", self.log_every_steps)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    seed: int = 42
    device: str = "auto"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    validation_frequency: int = 1
    checkpoint_frequency: int = 1
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mixed_precision: bool = False
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "validation_frequency", "checkpoint_frequency"):
            _positive_int(name, getattr(self, name))
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("training.seed must be a non-negative 63-bit integer")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("training.device must be auto, cpu, cuda or mps")
        _number("max_grad_norm", self.max_grad_norm, 0, strict=True)
        _number("label_smoothing", self.label_smoothing, 0)
        if self.label_smoothing > 1:
            raise ValueError("label_smoothing must be at most 1")
        for name in ("mixed_precision", "deterministic_algorithms"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    @classmethod
    def from_dict(cls, values: dict) -> "TrainingConfig":
        values = dict(values)
        for name, config_type in (
            ("optimizer", OptimizerConfig), ("scheduler", SchedulerConfig),
            ("early_stopping", EarlyStoppingConfig), ("logging", LoggingConfig),
        ):
            if name in values:
                values[name] = config_type(**values[name])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        try:
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict) or not isinstance(values.get("training"), dict):
                raise ValueError("Expected a training mapping")
            return cls.from_dict(values["training"])
        except (TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid training configuration in {path}: {error}") from error
