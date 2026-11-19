"""Inspect a saved checkpoint without rewriting its generated text or parameters."""

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_translations import prediction_flags
from src.data.config import DataConfig
from src.data.masks import encoder_attention_mask
from src.data.prepared import file_digest
from src.inference import InferenceConfig, TransformerTranslator
from src.inference.unicode_diagnostics import describe_ids
from src.training.checkpoint import load_checkpoint, validate_checkpoint
from src.training.utils import forward_batch, load_prepared_data

HINDI = ("नमस्ते", "हम दोस्त हैं।", "मेरा नाम राहुल है।", "आप कैसे हैं?", "मुझे पानी चाहिए।", "भारत एक देश है।",
         "क्षेत्र में विद्यार्थी हैं।", "क़िताब", "कि")
ENGLISH = ("We are friends", "Hello", "I am happy", "What time is it?", "Please sit here")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distribution(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    return {
        "mean": sum(ordered) / len(ordered), "max": ordered[-1],
        **{f"p{q}": ordered[math.ceil(len(ordered) * q / 100) - 1] for q in (50, 95, 99)},
    }


def conditioning(translator: TransformerTranslator) -> dict:
    model, tokenizer = translator.model, translator.tokenizer
    source_ids = [translator._source_ids(text, model.config.max_seq_len)
                  for text in ("We are friends", "The train is late")]
    source = torch.tensor([list(ids) + [tokenizer.pad_id] * (max(map(len, source_ids)) - len(ids))
                           for ids in source_ids], device=translator.device)
    target = torch.tensor([[tokenizer.bos_id, *tokenizer.encode("हम")]] * 2, device=translator.device)
    mask = encoder_attention_mask(source, tokenizer.pad_id)
    embeddings = model.encoder.embedding(source)
    memory = model.encoder(source, mask)
    base = translator._next_logits(target, source, memory)
    zero = translator._next_logits(target, source, torch.zeros_like(memory))
    padded = translator._next_logits(target, source, memory.masked_fill(~mask[:, 0, 0, :, None], 100.0))
    full = model(source, target, return_diagnostics=True)
    torch.testing.assert_close(base, full.logits[:, -1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(base, padded, atol=1e-5, rtol=1e-5)
    changed = target.clone()
    changed[:, -1] = tokenizer.eos_id
    torch.testing.assert_close(full.logits[:, :-1], model(source, changed)[:, :-1], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(full.decoder_self_attention_maps.triu(1)) == 0
    assert torch.count_nonzero(full.decoder_cross_attention_maps.masked_select(~mask.unsqueeze(1))) == 0
    probabilities = base.softmax(-1)
    tv = (probabilities[0] - probabilities[1]).abs().sum().item() / 2
    zero_tv = ((probabilities - zero.softmax(-1)).abs().sum(-1) / 2).tolist()
    assert tv > 0 and all(distance > 0 for distance in zero_tv)
    return {
        "sources": ["We are friends", "The train is late"], "shared_decoder_prefix": target[0].tolist(),
        "source_embeddings_norms": embeddings.flatten(1).norm(dim=1).tolist(),
        "encoder_memory_norms": memory.flatten(1).norm(dim=1).tolist(),
        "next_distribution_source_tv": tv, "next_distribution_zero_memory_tv": zero_tv,
        "padding_memory_perturbation_invariant": True, "future_target_change_invariant": True,
        "last_position_projection_matches": True, "eval_mode": all(not module.training for module in model.modules()),
        "cross_attention_blocks_source_padding": True, "causal_attention_blocks_future": True,
    }


def analyze_data(data, translator) -> dict:
    tokenizer = translator.tokenizer
    stats = {}
    for split in ("train", "val"):
        rows = data.datasets[split].pairs
        src = [tokenizer.encode(pair.source) for pair in rows]
        tgt = [tokenizer.encode(pair.target) for pair in rows]
        tokens = Counter(token for ids in tgt for token in ids)
        frequencies = tokens.most_common(20)
        stats[split] = {
            "pairs": len(rows), "source_tokens_including_eos": distribution([len(ids) + 1 for ids in src]),
            "target_tokens_including_eos": distribution([len(ids) + 1 for ids in tgt]),
            "target_over_64": sum(len(ids) + 1 > 64 for ids in tgt),
            "target_over_100": sum(len(ids) + 1 > 100 for ids in tgt),
            "target_tokens_per_whitespace_word": sum(map(len, tgt)) / sum(len(row.target.split()) for row in rows),
            "target_vocabulary_ids_seen_excluding_eos": len(tokens), "target_unknown_tokens": tokens[tokenizer.unk_id],
            "eos_labels": len(rows), "eos_fraction_of_labels": len(rows) / (sum(tokens.values()) + len(rows)),
            "common_target_ids": [{"id": token, "piece": tokenizer.id_to_token(token), "count": count}
                                  for token, count in frequencies],
        }

    bins = defaultdict(lambda: Counter())
    model = translator.model
    for batch in data.loader("val", 8, 42):
        logits, labels = forward_batch(model, batch, translator.device)
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=tokenizer.pad_id)
        log_probs = logits.log_softmax(-1)
        valid = labels.ne(tokenizer.pad_id)
        entropy = torch.special.entr(log_probs.exp()).sum(-1)
        for row, length in enumerate(valid.sum(-1).tolist()):
            bucket = next(name for threshold, name in ((32, "1–32"), (64, "33–64"), (96, "65–96"), (128, "97–128"))
                          if length <= threshold)
            totals = bins[bucket]
            totals["sentences"] += 1
            totals["tokens"] += length
            totals["loss"] += losses[row, valid[row]].sum().item()
            totals["correct"] += (logits[row].argmax(-1).eq(labels[row]) & valid[row]).sum().item()
            totals["entropy"] += entropy[row, valid[row]].sum().item()
            totals["gold_eos_probability"] += log_probs[row, length - 1, tokenizer.eos_id].exp().item()
    stats["teacher_forced_by_target_length"] = {
        bucket: {"sentences": totals["sentences"], "tokens": totals["tokens"], "loss": totals["loss"] / totals["tokens"],
                 "accuracy": totals["correct"] / totals["tokens"], "entropy_nats": totals["entropy"] / totals["tokens"],
                 "mean_probability_of_eos_at_gold_eos": totals["gold_eos_probability"] / totals["sentences"]}
        for bucket, totals in sorted(bins.items())
    }
    return stats


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    data = load_prepared_data(args.processed_dir, args.tokenizer_dir, DataConfig.load(args.data_config))
    translator = TransformerTranslator.from_checkpoint(args.checkpoint, args.tokenizer_dir, "cpu")
    tokenizer = translator.tokenizer
    checkpoint = load_checkpoint(args.checkpoint)
    validate_checkpoint(checkpoint, asdict(translator.model.config), data.tokenizer_metadata, data.data_identity)
    roundtrips = []
    for text in (*ENGLISH, *HINDI):
        details = describe_ids(tokenizer, tokenizer.encode(text))
        assert details["text"] == unicodedata.normalize("NFC", text)
        assert details["strict_utf8_valid"] and details["bytelevel_agrees_with_decoder"]
        assert not details["controls"] and not details["replacement_characters"]
        roundtrips.append({"original": text, **details})
    write_json(args.output / "tokenizer_roundtrip.json", roundtrips)
    fragment = next(describe_ids(tokenizer, [token]) for token in range(4, tokenizer.vocab_size)
                    if not describe_ids(tokenizer, [token])["strict_utf8_valid"])
    write_json(args.output / "byte_fragment_example.json", fragment)
    write_json(args.output / "conditioning.json", conditioning(translator))
    write_json(args.output / "tokenization_and_loss.json", analyze_data(data, translator))

    final = []
    with (args.output / "generation_trace.jsonl").open("w", encoding="utf-8") as stream:
        for source in ENGLISH:
            plain = translator.generate([source], InferenceConfig(max_new_tokens=64))[0]
            for strategy in ("greedy", "beam"):
                config = InferenceConfig(strategy=strategy, max_new_tokens=64)
                result = translator.generate([source], config, trace=True)[0]
                if strategy == "greedy":
                    assert result.hypothesis.token_ids == plain.hypothesis.token_ids
                if strategy == "beam":
                    width_one = translator.generate([source], InferenceConfig(strategy="beam", beam_width=1))[0]
                    assert width_one.hypothesis.token_ids == plain.hypothesis.token_ids
                hypothesis = result.hypothesis
                generated = list(hypothesis.token_ids)
                row = {
                    "source": source, "strategy": strategy, "source_ids": result.source_ids,
                    "source_pieces": [tokenizer.id_to_token(token) for token in result.source_ids],
                    "eos": hypothesis.reached_eos, "unicode": describe_ids(tokenizer, generated),
                    "flags": prediction_flags(result.text),
                }
                for index, step in enumerate(hypothesis.steps):
                    event = {
                        "source": source, "strategy": strategy, "step": index + 1,
                        "decoder_prefix": [tokenizer.bos_id, *generated[:index]],
                        "selected_id": step.token_id, "selected_piece": tokenizer.id_to_token(step.token_id),
                        "cumulative_text": tokenizer.decode(generated[:index + 1]),
                        "eos_probability": step.eos_probability, "entropy_nats": step.entropy_nats,
                        "candidates": [{"id": token, "piece": tokenizer.id_to_token(token),
                                        "logit": score, "model_probability": probability}
                                       for (token, score), probability in zip(step.candidates, step.candidate_probabilities)],
                    }
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                final.append(row)
                if source == ENGLISH[0] and strategy == "greedy":
                    path = args.output / "generated_hindi.txt"
                    path.write_text(result.text, encoding="utf-8")
                    assert path.read_text(encoding="utf-8") == result.text
    write_json(args.output / "unicode_analysis.json", {
        "checkpoint_sha256": file_digest(args.checkpoint), "stdout_encoding": sys.stdout.encoding,
        "model_config": checkpoint["model_config"], "tokenizer": data.tokenizer_metadata, "predictions": final,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("checkpoint", "tokenizer-dir", "processed-dir", "data-config", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
