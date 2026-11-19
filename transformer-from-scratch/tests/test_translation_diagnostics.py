import unicodedata

import pytest
import torch

from scripts.diagnose_translation import ENGLISH, HINDI, conditioning
from scripts.translate import display_text
from src.data.tokenizer import BilingualBPETokenizer
from src.inference import TransformerTranslator
from src.inference.beam_search import beam_search_decode
from src.inference.greedy import greedy_decode
from src.inference.unicode_diagnostics import bytelevel_bytes, describe_ids, describe_text
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer


@pytest.fixture(scope="module")
def tokenizer():
    return BilingualBPETokenizer.train((*ENGLISH, *HINDI, "The train is late"), 700)


@pytest.mark.parametrize("sample", (*ENGLISH, *HINDI))
def test_saved_style_bytelevel_roundtrip(tokenizer, sample):
    normalized = unicodedata.normalize("NFC", sample)
    ids = tokenizer.encode(unicodedata.normalize("NFD", sample))
    details = describe_ids(tokenizer, [tokenizer.bos_id, *ids, tokenizer.eos_id, tokenizer.pad_id])
    assert details["text"] == normalized
    assert bytes.fromhex(details["bytes_hex"]) == normalized.encode("utf-8")
    assert details["strict_utf8_valid"] and not details["replacement_characters"] and not details["controls"]


def test_arbitrary_byte_fragment_and_legitimate_combining_marks(tokenizer):
    # Full alphabet coverage guarantees we can generate individual UTF-8 bytes.
    leading = next(token for token in range(4, tokenizer.vocab_size)
                   if bytelevel_bytes([tokenizer.id_to_token(token)]) == b"\xe0")
    fragment = describe_ids(tokenizer, [leading, tokenizer.eos_id])
    assert not fragment["strict_utf8_valid"]
    assert fragment["text"] == "\ufffd" and fragment["replacement_characters"] == 1
    assert describe_ids(tokenizer, [tokenizer.unk_id])["text"] == "<unk>"
    assert display_text("कि\x01") == "कि\\x01"
    assert any(cp["category"].startswith("M") for cp in describe_text("कि")["code_points"])
    assert describe_text("क्यार्यारार")["controls"] == []


def test_model_trace_probabilities_before_suppression_and_width_one_agree():
    raw = torch.tensor([7., 6., 2., 3.])
    logits = lambda prefixes: raw.expand(prefixes.size(0), -1)
    prefix = torch.tensor([[1]])
    options = dict(eos_id=2, pad_id=0, max_new_tokens=3, trace=True)
    greedy = greedy_decode(logits, prefix, **options)[0]
    beam = beam_search_decode(logits, prefix, beam_width=1, **options)
    assert greedy.token_ids == beam.token_ids == (3, 3, 3)
    assert greedy.steps == beam.steps
    expected = raw.softmax(-1)
    assert greedy.steps[0].eos_probability == expected[2].item()
    assert greedy.steps[0].candidate_probabilities == (expected[3].item(), expected[2].item())


@torch.inference_mode()
def test_source_and_memory_condition_masks_and_last_position(tokenizer):
    torch.manual_seed(42)
    model = EncoderDecoderTransformer(
        tokenizer.vocab_size, tokenizer.pad_id,
        ModelConfig(d_model=16, d_ff=32, num_encoder_layers=1, num_decoder_layers=1),
    )
    checks = conditioning(TransformerTranslator(model, tokenizer, torch.device("cpu")))
    assert checks["next_distribution_source_tv"] > 0
    assert checks["last_position_projection_matches"] and checks["future_target_change_invariant"]
    assert checks["cross_attention_blocks_source_padding"] and checks["eval_mode"]
