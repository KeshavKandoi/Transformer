import json
import random
from argparse import Namespace
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest
import torch

from scripts.evaluate_translations import evaluate, score_translations
from scripts.import_corpus import DEMO, import_corpus
from scripts.train import run
from src.data.cleaning import TranslationPair, read_parallel_csv, split_groups, write_parallel_csv
from src.data.config import ROOT, DataConfig
from src.data.corpus import CorpusPolicy, _grams, comparison_key, filter_pairs, leakage_groups
from src.data.prepared import file_digest
from src.data.tokenizer import BilingualBPETokenizer
from src.inference import InferenceConfig, TransformerTranslator
from src.training.checkpoint import load_checkpoint
from src.training.config import SchedulerConfig
from src.training.scheduler import resolve_schedule
from src.training.utils import load_prepared_data


def test_quality_filters_and_normalized_duplicates() -> None:
    good = TranslationPair("  I am   happy. ", "मैं खुश हूँ।")
    duplicate = TranslationPair("i am happy!", "मैं खुश हूँ!")
    pairs, rejected = filter_pairs([
        good, duplicate,
        TranslationPair("मैं खुश हूँ", "I am happy"),
        TranslationPair("Hello", "नमस्ते\u202e"),
        TranslationPair("word " * 100, "नमस्ते"),
        TranslationPair("One two three four five six", "नमस्ते"),
        TranslationPair("Cafe\N{LATIN SMALL LETTER E WITH ACUTE}", "नमस्ते"),
    ], CorpusPolicy(max_words=10))
    assert rejected == {"length": 1, "length_ratio": 1, "normalized_duplicate": 1, "script": 1, "unicode_format": 1}
    assert len(pairs) == 2
    assert comparison_key("CAFE\u0301?") == comparison_key("café!")
    assert all("   " not in pair.source for pair in pairs)


def test_duplicate_components_keep_all_exact_and_near_pairs_together() -> None:
    pairs = [
        TranslationPair("The quick brown fox jumps over the lazy dog.", "यह पहला वाक्य है"),
        TranslationPair("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG!", "हमें घर जाना है"),
        TranslationPair("A quick brown fox jumps over the lazy dog.", "मुझे पानी चाहिए"),
        TranslationPair("I need water now.", "मुझे पानी चाहिए!"),
        TranslationPair("Good morning", "सुप्रभात"),
        TranslationPair("I am reading", "मैं पढ़ रहा हूँ"),
    ]
    groups = leakage_groups(pairs, 0.8)
    assert groups[0] == pairs[:4]
    splits = split_groups(groups, 42, (0.8, 0.1, 0.1))
    for left, right in combinations(splits.values(), 2):
        for first in left:
            for second in right:
                for side in ("source", "target"):
                    a, b = [_grams(comparison_key(getattr(pair, side))) for pair in (first, second)]
                    assert len(a & b) / len(a | b) < 0.8


def test_language_check_rejects_non_english_latin_text() -> None:
    pairs, rejected = filter_pairs([
        TranslationPair("Este es un ejemplo de una oración escrita completamente en español.", "यह एक उदाहरण है।"),
        TranslationPair("I would like to visit the library this morning.", "मैं आज सुबह पुस्तकालय जाना चाहता हूँ।"),
    ], CorpusPolicy())
    assert rejected == {"language": 1}
    assert len(pairs) == 1


def test_prefix_index_never_misses_bruteforce_matches() -> None:
    rng = random.Random(4)
    sources = ["the cat is sitting near the window"]
    for _ in range(80):
        source = list(rng.choice(sources))
        source[rng.randrange(len(source))] = rng.choice("abcdefgh")
        sources.append("".join(source))
    pairs = [TranslationPair(source, f"संदर्भ {index:03d}") for index, source in enumerate(sources)]
    threshold = 0.75
    groups = leakage_groups(pairs, threshold)
    assignments = {pair: index for index, group in enumerate(groups) for pair in group}
    for first, second in combinations(pairs, 2):
        a, b = (_grams(comparison_key(pair.source)) for pair in (first, second))
        if len(a & b) / len(a | b) >= threshold:
            assert assignments[first] == assignments[second]


def test_import_uses_only_training_rows_and_rejects_bad_provenance_or_limits(tmp_path: Path) -> None:
    config = replace(DataConfig(), vocab_size=620)
    kwargs = dict(name="fixture", license_id="Test fixture terms", provenance="pytest fixture")
    local = tmp_path / "local.csv"
    pairs, _ = read_parallel_csv(DEMO)
    write_parallel_csv(local, list(reversed(pairs)))
    first, second = tmp_path / "first", tmp_path / "second"
    stats = import_corpus(DEMO, first, config, CorpusPolicy(), **kwargs)
    import_corpus(local, second, config, CorpusPolicy(), **kwargs)
    for split in ("train", "val", "test"):
        assert (first / "processed" / f"{split}.csv").read_bytes() == (second / "processed" / f"{split}.csv").read_bytes()
    assert (first / "tokenizer/tokenizer.json").read_bytes() == (second / "tokenizer/tokenizer.json").read_bytes()
    train, _ = read_parallel_csv(first / "processed/train.csv")
    tokenizer = BilingualBPETokenizer.train(
        (text for row in train for text in (row.source, row.target)), config.vocab_size,
    )
    tokenizer.save(tmp_path / "expected")
    assert file_digest(tmp_path / "expected/tokenizer.json") == file_digest(first / "tokenizer/tokenizer.json")
    assert stats["license"] == kwargs["license_id"]
    with pytest.raises(ValueError, match="license"):
        import_corpus(local, tmp_path / "invalid", config, CorpusPolicy(), **{**kwargs, "license_id": None})
    with pytest.raises(ValueError, match="Tokenized rows exceed"):
        import_corpus(local, tmp_path / "short", replace(config, max_source_length=1), CorpusPolicy(), **kwargs)
    with pytest.raises(ValueError, match="already exists"):
        import_corpus(local, first, config, CorpusPolicy(), **kwargs)


def test_metric_controls_and_planned_warmup() -> None:
    references = ["हम अच्छे दोस्त हैं।", "कृपया मुझे पानी दीजिए।"]
    perfect = score_translations(references, references)
    wrong = score_translations(["xxxx", "yyyy"], references)
    for metric in ("chrf", "bleu"):
        assert perfect[metric]["score"] == pytest.approx(100)
        assert wrong[metric]["score"] == pytest.approx(0)
        assert "version:" in perfect[metric]["signature"]
    with pytest.raises(ValueError, match="equal"):
        score_translations([], references)
    scheduler = resolve_schedule(SchedulerConfig(warmup_fraction=0.1), 23)
    assert (scheduler.warmup_steps, scheduler.planned_updates) == (3, 23)
    assert resolve_schedule(scheduler, 1000) == scheduler
    assert resolve_schedule(SchedulerConfig(warmup_fraction=1.0), 2).warmup_steps == 1


def test_import_train_resume_generate_eos_and_fingerprints(tmp_path: Path) -> None:
    data_config = ROOT / "configs/experiment_data.yaml"
    config = DataConfig.load(data_config)
    corpus = tmp_path / "corpus"
    import_corpus(DEMO, corpus, config, CorpusPolicy(), name="smoke", license_id=None,
                  provenance="Bundled test smoke", demo=True)
    model = tmp_path / "model.yaml"
    model.write_text(
        "model: {d_model: 16, d_ff: 32, num_encoder_layers: 1, num_decoder_layers: 1, "
        "output_projection_bias: true}\n",
        encoding="utf-8",
    )
    args = Namespace(
        resume=None, config=ROOT / "configs/experiment_training.yaml", epochs=1, batch_size=8,
        device="cpu", model_config=model, data_config=data_config, processed_dir=corpus / "processed",
        tokenizer_dir=corpus / "tokenizer", overfit_one_batch=False, dry_run=False,
        run_name="test-experiment", checkpoint_dir=tmp_path / "checkpoints", runs_dir=tmp_path / "runs",
    )
    original_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        run(args)
        latest = args.checkpoint_dir / "test-experiment/latest.pt"
        first = load_checkpoint(latest)
        args.epochs, args.resume = 2, latest
        run(args)
        second = load_checkpoint(latest)
    finally:
        torch.use_deterministic_algorithms(original_determinism)
    assert first["global_step"] == 2 and second["global_step"] == 4
    assert first["training_config"]["scheduler"] == second["training_config"]["scheduler"]
    assert first["fingerprints"]["tokenizer_metadata"] == second["fingerprints"]["tokenizer_metadata"]
    assert set(second["fingerprints"]) == {"tokenizer_metadata", "model_config", "training_config", "data_identity"}
    assert second["data_identity"]["corpus"]["demo"]
    weight = "output_projection.projection.weight"
    assert not torch.equal(first["model_state_dict"][weight], second["model_state_dict"][weight])

    translator = TransformerTranslator.from_checkpoint(latest, args.tokenizer_dir, "cpu")
    with torch.no_grad():
        translator.model.output_projection.projection.weight.zero_()
        bias = translator.model.output_projection.projection.bias
        bias.zero_()
        bias[translator.tokenizer.eos_id] = 100
    for strategy in ("greedy", "beam"):
        result = translator.generate(["Hello"], InferenceConfig(strategy=strategy, max_new_tokens=1))[0]
        assert result.text == "" and result.hypothesis.token_ids == (translator.tokenizer.eos_id,)
        assert result.hypothesis.reached_eos
    out = tmp_path / "test-evaluation"
    report = evaluate(Namespace(
        output=out, checkpoint=latest, processed_dir=args.processed_dir, tokenizer_dir=args.tokenizer_dir,
        data_config=data_config, split="test", batch_size=2, limit=None, device="cpu",
        max_new_tokens=4, beam_width=2, length_penalty=0.6,
        training_metrics=args.runs_dir / "test-experiment/metrics.jsonl",
    ))
    assert report["full_split"] and report["examples"] == 4
    assert {path.name for path in out.iterdir()} == {"greedy.jsonl", "beam.jsonl", "metrics.json", "training_curves.csv"}
    assert len((out / "training_curves.csv").read_text().splitlines()) == 3
    for strategy in ("greedy", "beam"):
        rows = [json.loads(line) for line in (out / f"{strategy}.jsonl").read_text().splitlines()]
        assert len(rows) == 4 and all("reference" in row and "prediction" in row for row in rows)

    altered = {**second, "training_config": {**second["training_config"], "seed": 11}}
    broken = tmp_path / "broken.pt"
    torch.save(altered, broken)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_checkpoint(broken)
    path = args.processed_dir / "test.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="do not match"):
        load_prepared_data(args.processed_dir, args.tokenizer_dir, config)
