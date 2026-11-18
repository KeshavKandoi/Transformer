import json
from argparse import Namespace
from dataclasses import replace
from itertools import chain
from pathlib import Path

import pytest
import torch

from scripts.audit_translation_experiment import audit
from scripts.download_iitb_dataset import DATASET, FILES, LICENSE, PUBLISHER, REVISION, WARNING
from scripts.evaluate_translations import evaluate, prediction_flags
from scripts.prepare_iitb_experiment import prepare, remove_leakage, train_bpe, verify_source
from scripts.train import run
from src.data.cleaning import TranslationPair, read_parallel_csv, write_parallel_csv
from src.data.config import ROOT, DataConfig
from src.data.corpus import CorpusPolicy
from src.data.prepared import file_digest
from src.data.tokenizer import BilingualBPETokenizer
from src.inference import InferenceConfig, TransformerTranslator
from src.training.checkpoint import load_checkpoint


def test_fixed_split_priority_includes_transitive_near_leaks() -> None:
    a = TranslationPair("The quick brown fox jumps over the lazy dog.", "यह पहला वाक्य है")
    b = TranslationPair("A quick brown fox jumps over the lazy dog.", "मुझे पानी चाहिए")
    c = TranslationPair("A different sentence about water.", "मुझे पानी चाहिए!")
    d = TranslationPair("Good morning", "सुप्रभात")
    splits, stats = remove_leakage({"train": [a, d], "val": [b], "test": [c]}, 0.8)
    assert splits == {"train": [d], "val": [], "test": [c]}
    assert stats["cross_split_groups"] == 1 and stats["removed"] == {"train": 1, "val": 1, "test": 0}


def test_overlength_quarantine_and_train_only_tokenizer(tmp_path: Path) -> None:
    long = TranslationPair("abcdefghijklmnopqrstuvwxyz" * 10, "अलंबा वाक्य")
    splits = {"train": [TranslationPair("Hi", "नमस्ते"), long],
              "val": [TranslationPair("Hello", "नमस्कार"), long],
              "test": [TranslationPair("A friend", "दोस्त")]}
    config = replace(DataConfig(), vocab_size=260, max_source_length=32, max_target_length=32)
    tokenizer, retained, removed, passes = train_bpe(splits, config)
    assert removed == {"train": 1, "val": 1, "test": 0}
    assert passes == 2 and long not in retained["train"] and long not in retained["val"]
    expected = BilingualBPETokenizer.train(chain.from_iterable((r.source, r.target) for r in retained["train"]), 260)
    expected.save(tmp_path / "ref")
    tokenizer.save(tmp_path / "actual")
    assert file_digest(tmp_path / "ref/tokenizer.json") == file_digest(tmp_path / "actual/tokenizer.json")


def fixture_source(directory: Path) -> Path:
    directory.mkdir()
    rows, _ = read_parallel_csv(ROOT / "data/raw/sample_parallel.csv")
    splits = {"train": [*rows[:12], rows[12]], "validation": rows[12:17], "test": rows[17:22]}
    for name, pairs in splits.items():
        write_parallel_csv(directory / FILES[name], pairs)
    files = {name: file_digest(directory / name) for name in FILES.values()}
    metadata = {"dataset": DATASET, "publisher": PUBLISHER, "source_url": f"https://huggingface.co/datasets/{DATASET}",
                "revision": REVISION, "license": LICENSE, "license_warning": WARNING,
                "selected_train_rows": 13, "selected_train_sha256": files[FILES["train"]], "files": files,
                "official_split_sizes": {"train": 100, "validation": 5, "test": 5}}
    (directory / "source_metadata.json").write_text(json.dumps(metadata))
    return directory


def test_fixed_split_full_training_resume_evaluation_audit_and_tamper(tmp_path: Path) -> None:
    source = fixture_source(tmp_path / "source")
    config_path = ROOT / "configs/experiment_data.yaml"
    experiment = tmp_path / "experiment"
    stats = prepare(source, experiment, DataConfig.load(config_path), CorpusPolicy())
    assert stats["leakage"]["removed"]["train"] == 1
    assert stats["splits"]["val"]["kept"] == stats["splits"]["test"]["kept"] == 5
    train = read_parallel_csv(experiment / "processed/train.csv")[0]
    val = read_parallel_csv(experiment / "processed/val.csv")[0]
    test = read_parallel_csv(experiment / "processed/test.csv")[0]
    assert not set(train).intersection(val + test)
    model = tmp_path / "model.yaml"
    model.write_text("model: {d_model: 16, d_ff: 32, num_encoder_layers: 1, num_decoder_layers: 1, output_projection_bias: true}\n")
    args = Namespace(
        config=ROOT / "configs/iitb_pilot_training.yaml", data_config=config_path, model_config=model, resume=None,
        epochs=1, planned_epochs=2, batch_size=4, device="cpu", processed_dir=experiment / "processed",
        tokenizer_dir=experiment / "tokenizer", run_name="fixture", checkpoint_dir=tmp_path / "checkpoints",
        runs_dir=tmp_path / "runs", dry_run=True, overfit_one_batch=False,
    )
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        run(args)
        args.dry_run = False
        run(args)
        latest = args.checkpoint_dir / "fixture/latest.pt"
        first = load_checkpoint(latest)
        args.resume, args.planned_epochs, args.epochs = latest, None, 2
        run(args)
        second = load_checkpoint(latest)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    assert first["global_step"] == 2 and second["global_step"] == 4
    assert second["scheduler_state_dict"]["last_epoch"] == second["global_step"]
    assert first["training_config"]["scheduler"] == second["training_config"]["scheduler"]
    key = "output_projection.projection.weight"
    assert not torch.equal(first["model_state_dict"][key], second["model_state_dict"][key])

    translator = TransformerTranslator.from_checkpoint(latest, args.tokenizer_dir, "cpu")
    with torch.no_grad():
        translator.model.output_projection.projection.weight.zero_()
        translator.model.output_projection.projection.bias.zero_()
        translator.model.output_projection.projection.bias[translator.tokenizer.eos_id] = 100
    assert translator.generate(["Hello"], InferenceConfig(max_new_tokens=2))[0].hypothesis.reached_eos
    altered = tmp_path / "altered"
    translator.tokenizer.save(altered)
    path = altered / "tokenizer.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="token mapping"):
        TransformerTranslator.from_checkpoint(latest, altered, "cpu")

    run_dir = args.runs_dir / "fixture"
    output = run_dir / "generated-validation"
    gen = evaluate(Namespace(
        checkpoint=latest, processed_dir=args.processed_dir, tokenizer_dir=args.tokenizer_dir,
        output=output, data_config=config_path, split="val", limit=3, stratified=True, batch_size=2,
        device="cpu", max_new_tokens=3, beam_width=2, length_penalty=0.6,
        training_metrics=run_dir / "metrics.jsonl",
    ))
    assert gen["dataset_revision"] == REVISION and gen["official_coverage"] == 0.6
    report = audit(Namespace(source_dir=source, experiment=experiment, checkpoint=latest, run_dir=run_dir,
                             generation=output, data_config=config_path))
    assert report["tokenizer_reproduced_from_train"] and (run_dir / "experiment_report.md").is_file()
    assert prediction_flags("ाााा\x07")["control"] and prediction_flags("ाााा")["repetition"]
    csv_path = source / FILES["train"]
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksums"):
        verify_source(source)
