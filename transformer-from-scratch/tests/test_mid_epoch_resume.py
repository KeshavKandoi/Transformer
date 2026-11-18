import random
import signal
from argparse import Namespace
from dataclasses import asdict, replace

import pytest
import torch

from scripts.train import run
from src.data.config import ROOT, DataConfig
from src.data.dataset import ParallelDataset
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.checkpoint import CheckpointManager, load_checkpoint
from src.training.config import EarlyStoppingConfig, SchedulerConfig, TrainingConfig
from src.training.trainer import TransformerTrainer
from src.training.utils import load_prepared_data, seed_everything


@pytest.fixture
def factory(tmp_path):
    from scripts.prepare_data import prepare

    data_config = ROOT / "configs/data_config.yaml"
    prepare(Namespace(config=data_config, input=ROOT / "data/raw/sample_parallel.csv",
                      processed_dir=tmp_path / "processed", tokenizer_dir=tmp_path / "tokenizer"))
    configuration = DataConfig.load(data_config)
    data = load_prepared_data(tmp_path / "processed", tmp_path / "tokenizer", configuration)
    data.datasets["train"] = ParallelDataset(data.datasets["train"].pairs[:13], data.tokenizer, 128, 128)
    data.datasets["val"] = ParallelDataset(data.datasets["val"].pairs[:3], data.tokenizer, 128, 128)
    model_config = ModelConfig(d_model=16, d_ff=32, num_encoder_layers=1, num_decoder_layers=1,
                               dropout=0.2, attention_dropout=0.15)
    config = TrainingConfig(epochs=2, batch_size=2, gradient_accumulation_steps=2, save_every_updates=2,
                            scheduler=SchedulerConfig(warmup_steps=3), deterministic_algorithms=True,
                            early_stopping=EarlyStoppingConfig(enabled=False))
    original = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)

    def make(name, *, seed=42, overrides=None):
        seed_everything(seed)
        settings = replace(config, **(overrides or {}))
        model = EncoderDecoderTransformer(data.tokenizer.vocab_size, data.tokenizer.pad_id, model_config)
        trace = []
        model.register_forward_pre_hook(
            lambda module, args, kwargs: trace.append(kwargs["encoder_input_ids"].clone()) if module.training else None,
            with_kwargs=True,
        )

        def loader(epoch, start=0):
            return data.loader("train", settings.batch_size, settings.seed + epoch, shuffle=True, start_batch=start)

        directory = tmp_path / name
        trainer = TransformerTrainer(
            model, loader, data.loader("val", settings.batch_size, settings.seed), settings, torch.device("cpu"),
            pad_id=data.tokenizer.pad_id, model_config=asdict(model_config), tokenizer_metadata=data.tokenizer_metadata,
            data_identity={"fixture": data.data_identity, "train_prefix": 13, "val_prefix": 3},
            checkpoint_manager=CheckpointManager(directory), metrics_path=directory / "metrics.jsonl",
            resume_train_loader=loader,
        )
        return trainer, trace

    yield make, data
    torch.use_deterministic_algorithms(original)


def assert_equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_equal_tree(first, second)
    else:
        assert left == right


def test_exact_cpu_resume_after_repeated_interruptions_and_partial_windows(factory):
    make, _ = factory
    reference, reference_batches = make("uninterrupted")
    expected_history = reference.fit()
    expected_rng = (random.getstate(), torch.get_rng_state())

    first, first_batches = make("resumed")
    path = first.checkpoint_manager.directory / "latest.pt"
    with pytest.raises(KeyboardInterrupt):
        first.fit(lambda: len(first_batches) == 5)
    saved = load_checkpoint(path)
    assert saved["epoch"] == 0 and saved["global_step"] == 2
    assert saved["epoch_progress"]["next_batch"] == 4
    assert saved["epoch_progress"]["batch_count"] == 7
    assert saved["epoch_progress"]["metrics"]["token_count"] > 4
    assert all(p.grad is None for p in first.model.parameters())
    assert_equal_tree(first_batches, reference_batches[:5])

    second, second_batches = make("resumed", seed=999, overrides={"save_every_updates": 0})
    second.restore(saved)
    with pytest.raises(KeyboardInterrupt):
        second.fit(lambda: len(second_batches) == 2)
    saved_again = load_checkpoint(path)
    assert saved_again["global_step"] == 3 and saved_again["epoch_progress"]["next_batch"] == 6
    assert_equal_tree(second_batches, reference_batches[4:6])

    third, remaining = make("resumed", seed=123)
    third.restore(saved_again)
    actual_history = third.fit()
    assert_equal_tree(remaining, reference_batches[6:])
    assert third.global_step == reference.global_step == 8
    assert_equal_tree(third.model.state_dict(), reference.model.state_dict())
    assert_equal_tree(third.optimizer.state_dict(), reference.optimizer.state_dict())
    assert_equal_tree(third.scheduler.state_dict(), reference.scheduler.state_dict())
    for actual, expected in zip(actual_history, expected_history, strict=True):
        for field in ("epoch", "global_step", "train", "validation", "best_validation_loss"):
            assert actual[field] == expected[field]
    assert_equal_tree((random.getstate(), torch.get_rng_state()), expected_rng)
    assert "epoch_progress" not in load_checkpoint(path)
    assert actual_history[0]["train"]["updates"] == 4  # Includes the final one-batch update.


def test_cursor_skips_indices_without_fetching_old_examples(factory, monkeypatch):
    _, data = factory
    full = list(data.loader("train", 2, 43, shuffle=True))
    fetched = []
    original = ParallelDataset.__getitem__

    def record_getitem(dataset, index):
        fetched.append(index)
        return original(dataset, index)

    monkeypatch.setattr(ParallelDataset, "__getitem__", record_getitem)
    torch_rng = torch.get_rng_state()
    resumed = list(data.loader("train", 2, 43, shuffle=True, start_batch=4))
    assert len(fetched) == 5
    assert_equal_tree(resumed, full[4:])
    assert_equal_tree(torch.get_rng_state(), torch_rng)


def test_interrupt_inside_optimizer_cannot_publish_uncertain_update(factory):
    make, _ = factory
    trainer, _ = make("unsafe-update")
    optimizer = trainer.optimizer
    calls = 0

    def interrupt(_optimizer, _args, _kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt

    optimizer.register_step_post_hook(interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit()
    payload = load_checkpoint(trainer.checkpoint_manager.directory / "latest.pt")
    assert payload["global_step"] == payload["scheduler_state_dict"]["last_epoch"] == 2
    assert payload["epoch_progress"]["next_batch"] == 4
    assert not (trainer.checkpoint_manager.directory / "best.pt").exists()


def test_interrupt_after_last_update_resumes_validation_without_retraining(factory):
    make, _ = factory
    trainer, seen = make("last-update")
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(lambda: trainer.global_step == 4)
    checkpoint = load_checkpoint(trainer.checkpoint_manager.directory / "latest.pt")
    assert len(seen) == checkpoint["epoch_progress"]["next_batch"] == 7
    resumed, trace = make("last-update", overrides={"epochs": 1})
    resumed.restore(checkpoint)
    history = resumed.fit()
    assert not trace and len(history) == 1
    assert history[0]["train"]["updates"] == 4 and history[0]["validation"]


def test_checkpoint_compatibility_invalid_cursor_and_legacy_epoch(factory, tmp_path):
    make, _ = factory
    trainer, _ = make("compatibility")
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(lambda: trainer.global_step == 2)
    latest = trainer.checkpoint_manager.directory / "latest.pt"
    saved = load_checkpoint(latest)
    other, _ = make("other")
    for key, message in (("tokenizer_metadata", "token mapping"), ("data_identity", "prepared dataset")):
        with pytest.raises(ValueError, match=message):
            other.restore({**saved, key: {}})
    with pytest.raises(ValueError, match="cursor"):
        other.restore({**saved, "epoch_progress": {**saved["epoch_progress"], "next_batch": 5}})
    broken = tmp_path / "corrupt.pt"
    broken.write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="Cannot read checkpoint"):
        load_checkpoint(broken)

    reference, _ = make("epoch-boundary", overrides={"epochs": 1})
    reference.fit()
    legacy = load_checkpoint(reference.checkpoint_manager.directory / "latest.pt")
    legacy["version"] = 1
    legacy.pop("fingerprints")
    legacy["training_config"].pop("save_every_updates")
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    other.restore(load_checkpoint(path))
    assert other.epoch == 1 and other.global_step == 4 and other.progress is None


def test_atomic_save_and_mid_epoch_updates_preserve_best(factory, monkeypatch):
    make, _ = factory
    trainer, _ = make("protect-best")
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(lambda: trainer.global_step == 6)
    manager = trainer.checkpoint_manager
    best = manager.directory / "best.pt"
    original_best = best.read_bytes()
    latest = manager.directory / "latest.pt"
    assert load_checkpoint(best)["epoch"] == 1
    assert load_checkpoint(latest)["global_step"] == 6
    original_latest = latest.read_bytes()

    def fail_save(payload, stream):
        stream.write(b"incomplete")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(KeyboardInterrupt):
        manager.save_update(trainer.state_dict())
    assert latest.read_bytes() == original_latest
    assert best.read_bytes() == original_best
    assert not list(manager.directory.glob("tmp*"))


def test_cli_defers_real_sigint_until_update_is_fully_committed(factory, tmp_path, monkeypatch):
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: {d_model: 16, d_ff: 32, num_encoder_layers: 1, num_decoder_layers: 1}\n")
    args = Namespace(
        resume=None, config=None, model_config=model_config, data_config=ROOT / "configs/data_config.yaml",
        processed_dir=tmp_path / "processed", tokenizer_dir=tmp_path / "tokenizer",
        epochs=1, batch_size=4, device="cpu", save_every_updates=2,
        planned_epochs=None, dry_run=False, overfit_one_batch=False, run_name="cli",
        checkpoint_dir=tmp_path / "checkpoint-root", runs_dir=tmp_path / "logs",
    )
    original_step = torch.optim.AdamW.step

    def step_then_sigint(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        if not args and self.state[self.param_groups[0]["params"][0]]["step"] == 3:
            signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", step_then_sigint)
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        run(args)
    assert signal.getsignal(signal.SIGINT) is previous
    latest = args.checkpoint_dir / args.run_name / "latest.pt"
    saved = load_checkpoint(latest)
    assert saved["global_step"] == saved["scheduler_state_dict"]["last_epoch"] == 3
    assert saved["epoch_progress"]["next_batch"] == 3
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    args.resume = latest
    run(args)
    assert load_checkpoint(latest)["global_step"] == 8


def test_epoch_metrics_recovered_if_checkpoint_was_published_before_log(factory):
    make, _ = factory
    trainer, _ = make("missing-metrics", overrides={"epochs": 1})
    entry = trainer.fit()[0]
    checkpoint = load_checkpoint(trainer.checkpoint_manager.directory / "latest.pt")
    assert checkpoint["epoch_metrics"] == entry
    trainer.metrics_path.unlink()

    resumed, _ = make("missing-metrics")
    resumed.restore(checkpoint)
    assert resumed.metrics_path.read_text().count("\n") == 1
    resumed.restore(checkpoint)
    assert resumed.metrics_path.read_text().count("\n") == 1
    resumed.fit()
    assert resumed.metrics_path.read_text().count("\n") == 2
