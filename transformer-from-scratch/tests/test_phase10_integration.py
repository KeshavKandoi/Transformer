import pytest
import torch

from src.inference import InferenceConfig, TransformerTranslator


def test_real_checkpoint_translates_with_all_strategies(translation_checkpoint) -> None:
    path, tokenizer_dir, _ = translation_checkpoint
    translator = TransformerTranslator.from_checkpoint(path, tokenizer_dir, "cpu")
    names = ("We are friends", "Hello")
    before = [parameter.detach().clone() for parameter in translator.model.parameters()]
    calls = []
    hook = translator.model.encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        for strategy in ("greedy", "beam", "sampling"):
            config = InferenceConfig(strategy=strategy, max_new_tokens=8, top_k=5, seed=42)
            results = translator.generate(names, config, trace=True)
            assert results == translator.generate(names, config, trace=True)
            assert [result.initial_prefix for result in results] == [(translator.tokenizer.bos_id,)] * 2
            for name, result in zip(names, results):
                assert result.source_ids == (*translator.tokenizer.encode(name), translator.tokenizer.eos_id)
                assert result.text.encode("utf-8").decode("utf-8") == result.text
                hypothesis = result.hypothesis
                assert 1 <= len(hypothesis.token_ids) <= 8
                assert all(0 <= index < translator.tokenizer.vocab_size for index in hypothesis.token_ids)
                assert translator.tokenizer.pad_id not in hypothesis.token_ids
                assert [step.token_id for step in hypothesis.steps] == list(hypothesis.token_ids)
    finally:
        hook.remove()
    assert len(calls) == 6
    assert not translator.model.training
    assert all(parameter.grad is None and torch.equal(parameter, old)
               for parameter, old in zip(translator.model.parameters(), before))


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No CUDA or MPS device available for translation",
)
def test_real_checkpoint_accelerator_generation(translation_checkpoint) -> None:
    path, tokenizer_dir, _ = translation_checkpoint
    device = "cuda" if torch.cuda.is_available() else "mps"
    translator = TransformerTranslator.from_checkpoint(path, tokenizer_dir, device)
    assert next(translator.model.parameters()).device.type == device
    result = translator.generate(["Hello"], InferenceConfig(max_new_tokens=4))[0]
    assert 1 <= len(result.hypothesis.token_ids) <= 4
