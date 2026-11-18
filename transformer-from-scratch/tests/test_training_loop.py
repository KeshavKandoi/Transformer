import copy
import math

import pytest
import torch

from src.training.config import EarlyStoppingConfig, SchedulerConfig, TrainingConfig
from src.training.loss import masked_cross_entropy
from src.training.trainer import EarlyStopping


def test_token_weighted_accumulation_matches_larger_batches_and_flushes_tail(trainer_factory, training_batches) -> None:
    torch.manual_seed(42)
    accumulated = trainer_factory(TrainingConfig(gradient_accumulation_steps=2, max_grad_norm=100), dropout=0)
    combined = trainer_factory(TrainingConfig(max_grad_norm=100), dropout=0)
    combined.model.load_state_dict(accumulated.model.state_dict())
    large = {key: torch.cat([training_batches[0][key], training_batches[1][key]]) for key in training_batches[0]}
    separate_metrics = accumulated.train_epoch(training_batches)
    combined_metrics = combined.train_epoch([large, training_batches[-1]])
    assert separate_metrics["valid_tokens"] == combined_metrics["valid_tokens"] == 6
    assert separate_metrics["loss"] == pytest.approx(combined_metrics["loss"], abs=1e-6)
    assert accumulated.global_step == combined.global_step == 2
    assert accumulated.scheduler.last_epoch == combined.scheduler.last_epoch == 2
    assert accumulated.optimizer.param_groups[0]["lr"] == pytest.approx(9e-6)
    for actual, expected in zip(accumulated.model.parameters(), combined.model.parameters()):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
    for call, batch in zip(accumulated.model.calls, training_batches):
        assert call[:2] == (True, True)
        assert torch.equal(call[2], batch["decoder_input_ids"])


def test_clipping_occurs_before_optimizer_and_reports_unclipped_norm(trainer_factory, training_batches) -> None:
    trainer = trainer_factory(TrainingConfig(max_grad_norm=0.02, scheduler=SchedulerConfig(type="constant")))
    measured = []

    def inspect_gradients(optimizer, args, kwargs):
        norm = torch.stack([p.grad.norm() for p in trainer.model.parameters()]).norm()
        measured.append(norm.item())

    handle = trainer.optimizer.register_step_pre_hook(inspect_gradients)
    actual = trainer.train_epoch(training_batches)
    handle.remove()
    assert len(measured) == 3
    assert all(norm <= 0.020001 for norm in measured)
    assert actual["gradient_norm"] > 0.02
    assert actual["learning_rate"] == trainer.config.optimizer.learning_rate


def test_validation_uses_unsmoothed_weighted_loss_and_does_not_update_state(trainer_factory, training_batches) -> None:
    trainer = trainer_factory(TrainingConfig(label_smoothing=0.2))
    trainer.train_epoch(training_batches)
    model_before = copy.deepcopy(trainer.model.state_dict())
    opt_before = copy.deepcopy(trainer.optimizer.state_dict())
    scheduler_before = copy.deepcopy(trainer.scheduler.state_dict())
    trainer.model.calls.clear()
    result = trainer.validate()
    assert trainer.model.training
    assert all(call[:2] == (False, False) for call in trainer.model.calls)
    assert trainer.scheduler.state_dict() == scheduler_before
    assert trainer.optimizer.state_dict()["param_groups"] == opt_before["param_groups"]
    for key, state in trainer.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, opt_before["state"][key][name], rtol=0, atol=0)
    for name, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter, model_before[name], rtol=0, atol=0)
    with torch.no_grad():
        trainer.model.eval()
        sums = []
        for batch in training_batches:
            logits = trainer.model(batch["encoder_input_ids"], batch["decoder_input_ids"])
            sums.append(masked_cross_entropy(logits, batch["decoder_target_ids"], 0)[0].item())
    assert result["valid_tokens"] == 6
    assert result["loss"] == pytest.approx(sum(sums) / 6)
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]))


@pytest.mark.parametrize("bad_loss", [True, False])
def test_nonfinite_loss_or_gradient_aborts_update(trainer_factory, training_batches, monkeypatch, bad_loss: bool) -> None:
    trainer = trainer_factory(TrainingConfig())
    parameter = next(trainer.model.parameters())
    old = parameter.detach().clone()
    if bad_loss:
        original = trainer.model.forward
        monkeypatch.setattr(trainer.model, "forward", lambda **inputs: original(**inputs) * float("nan"))
    else:
        parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    with pytest.raises(FloatingPointError, match="Non-finite cross-entropy|Gradient clipping failed"):
        trainer.train_epoch(training_batches)
    assert trainer.global_step == trainer.scheduler.last_epoch == 0
    assert torch.equal(parameter, old) and parameter.grad is None


def test_early_stopping_min_delta_and_disabled_mode() -> None:
    stopping = EarlyStopping(EarlyStoppingConfig(patience=2, min_delta=0.1))
    assert [stopping.update(loss) for loss in (3.0, 2.95, 2.8, 2.75, 2.74)] == [False, False, False, False, True]
    disabled = EarlyStopping(EarlyStoppingConfig(enabled=False, patience=1))
    assert not disabled.update(1.0) and not disabled.update(2.0)
