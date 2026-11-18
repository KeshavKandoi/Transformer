from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.data.tokenizer import BilingualBPETokenizer
from src.inference import InferenceConfig, TransformerTranslator


class ScriptedModel(nn.Module):
    def __init__(self, tokenizer, generated):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.config = SimpleNamespace(max_seq_len=16)
        self.output_projection = nn.Identity()
        self.dropout = nn.Dropout(0.8)
        self.tokenizer = tokenizer
        self.generated = generated
        self.encoded = []
        self.decoded = []

    def encoder(self, ids, mask):
        assert not self.training and not self.dropout.training and torch.is_inference_mode_enabled()
        assert torch.equal(mask[:, 0, 0], ids.ne(self.pad_id))
        self.encoded.append(ids.clone())
        self.memory = ids.float().unsqueeze(-1)
        return self.memory

    def decoder(self, prefix, memory, self_mask, cross_mask):
        assert torch.is_inference_mode_enabled()
        self.decoded.append((prefix.clone(), memory))
        assert torch.equal(prefix[:, :1], torch.full_like(prefix[:, :1], self.tokenizer.bos_id))
        assert torch.equal(self_mask[:, 0], prefix.ne(self.pad_id)[:, None] & torch.ones(
            prefix.size(1), prefix.size(1), dtype=torch.bool,
        ).tril())
        assert torch.equal(cross_mask[:, 0], memory[..., 0].ne(self.pad_id)[:, None].expand_as(cross_mask[:, 0]))
        scores = torch.zeros(*prefix.shape, self.vocab_size)
        for row, source in enumerate(memory):
            planned = self.generated[int(source[0, 0])]
            token = planned[min(prefix.size(1) - 1, len(planned) - 1)]
            scores[row, -1, token] = 10
            scores[row, -1, self.pad_id] = 20
            scores[row, -1, self.tokenizer.bos_id] = 20
        return scores


def test_masks_growing_prefix_reused_memory_hindi_and_early_eos() -> None:
    tokenizer = BilingualBPETokenizer.train(["Hello", "We are friends", "हम दोस्त हैं"])
    hindi = tokenizer.encode("हम दोस्त हैं")
    model = ScriptedModel(tokenizer, {
        tokenizer.encode("Hello")[0]: [tokenizer.eos_id],
        tokenizer.encode("We are friends")[0]: [*hindi, tokenizer.eos_id],
    })
    translator = TransformerTranslator(model, tokenizer, torch.device("cpu"))
    model.train()
    config = InferenceConfig(max_new_tokens=len(hindi) + 1)
    early, normal = translator.generate(["Hello", "We are friends"], config, trace=True)
    assert early.text == "" and early.hypothesis.reached_eos
    assert normal.text == "हम दोस्त हैं" and normal.hypothesis.token_ids == (*hindi, tokenizer.eos_id)
    assert len(model.encoded) == 1 and len(model.decoded) == len(hindi) + 1
    assert all(memory is model.memory for _, memory in model.decoded)
    assert [prefix[1].tolist() for prefix, _ in model.decoded] == [
        [tokenizer.bos_id, *hindi[:index]] for index in range(len(hindi) + 1)
    ]
    assert all(prefix.grad_fn is None for prefix, _ in model.decoded)


def test_beams_stay_with_their_own_encoder_rows_and_unknown_is_visible() -> None:
    tokenizer = BilingualBPETokenizer.train(["Hello", "We are friends", "नमस्ते"])
    hindi = tokenizer.encode("नमस्ते")
    model = ScriptedModel(tokenizer, {
        tokenizer.encode("Hello")[0]: [tokenizer.unk_id, tokenizer.eos_id],
        tokenizer.encode("We are friends")[0]: [*hindi, tokenizer.eos_id],
    })
    translator = TransformerTranslator(model, tokenizer, torch.device("cpu"))
    config = InferenceConfig(strategy="beam", max_new_tokens=len(hindi) + 2, beam_width=2)
    results = translator.generate(["Hello", "We are friends"], config)
    assert [result.text for result in results] == ["<unk>", "नमस्ते"]
    assert len(model.encoded) == 1
    for _, memory in model.decoded:
        assert memory[:, 0, 0].unique().numel() == 1
    with pytest.raises(ValueError, match="nonempty"):
        translator.translate(" ")
    with pytest.raises(ValueError, match="positional limit"):
        translator.translate("Hello", InferenceConfig(max_new_tokens=17))
    with pytest.raises(ValueError, match="Source needs"):
        translator.translate("Hello " * 100, config)
