"""A learned end-to-end check from shifted labels to free-running EOS."""

import torch

from src.data.cleaning import TranslationPair
from src.data.collator import ParallelCollator
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.inference import InferenceConfig, TransformerTranslator
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.loss import masked_cross_entropy
from src.training.utils import forward_batch


def test_tiny_model_memorizes_full_token_sequences_and_eos():
    rows = [TranslationPair("water", "पानी"), TranslationPair("house", "घर"),
            TranslationPair("friend", "दोस्त")]
    torch.manual_seed(42)
    tokenizer = BilingualBPETokenizer.train([text for row in rows for text in (row.source, row.target)], 360)
    model = EncoderDecoderTransformer(
        tokenizer.vocab_size, tokenizer.pad_id, ModelConfig(
            d_model=16, d_ff=32, num_encoder_layers=1, num_decoder_layers=1,
            dropout=0, attention_dropout=0, max_seq_len=16,
        ),
    )
    device = torch.device("cpu")
    translator = TransformerTranslator(model, tokenizer, device)
    model.train()
    dataset = ParallelDataset(rows, tokenizer, 16, 16)
    batch = ParallelCollator(tokenizer.pad_id)([dataset[i] for i in range(len(rows))])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits, labels = forward_batch(model, batch, device)
        loss, count = masked_cross_entropy(logits, labels, tokenizer.pad_id)
        (loss / count).backward()
        optimizer.step()
    assert (loss / count).item() < 0.05

    # Different source order from training, with no labels supplied.
    selected = [rows[i] for i in (2, 0, 1)]
    greedy = translator.generate([row.source for row in selected], InferenceConfig(max_new_tokens=12))
    beam = translator.generate([row.source for row in selected],
                               InferenceConfig(strategy="beam", beam_width=1, max_new_tokens=12))
    for row, first, second in zip(selected, greedy, beam, strict=True):
        expected = tuple(tokenizer.encode(row.target) + [tokenizer.eos_id])
        assert first.hypothesis.token_ids == second.hypothesis.token_ids == expected
        assert first.text == row.target and first.hypothesis.reached_eos
