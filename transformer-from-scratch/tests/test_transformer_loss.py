import math

import pytest
import torch

from scripts.inspect_full_transformer import masked_cross_entropy
from src.data.cleaning import TranslationPair
from src.data.collator import ParallelCollator
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer


def test_manual_cross_entropy_uses_raw_logits_and_ignores_padding() -> None:
    logits = torch.tensor([[[0., math.log(2), 0.], [20., -20., 4.]], [[math.log(3), 0., 0.], [0., 1., 2.]]],
                          requires_grad=True)
    labels = torch.tensor([[1, 0], [2, 0]], dtype=torch.long)
    loss = masked_cross_entropy(logits, labels, pad_id=0)
    assert loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor((math.log(2) + math.log(5)) / 2))
    changed = logits.detach().clone()
    changed[labels.eq(0)] = torch.tensor([60., 40., 20.])
    torch.testing.assert_close(masked_cross_entropy(changed, labels, 0), loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[labels.eq(0)]) == 0
    assert logits.grad[labels.ne(0)].norm(dim=-1).gt(0).all()


def test_real_bpe_shift_bos_and_eos_with_unequal_padding() -> None:
    pairs = [TranslationPair("Hello", "नमस्ते"), TranslationPair("We are friends", "हम दोस्त हैं")]
    tokenizer = BilingualBPETokenizer.train([text for pair in pairs for text in (pair.source, pair.target)])
    dataset = ParallelDataset(pairs, tokenizer, 128, 128)
    batch = ParallelCollator(tokenizer.pad_id)([dataset[0], dataset[1]])
    ids, labels = batch["decoder_input_ids"], batch["decoder_target_ids"]
    valid = labels.ne(tokenizer.pad_id)
    assert valid.sum(-1).unique().numel() == 2
    assert labels.dtype == torch.long and ids.shape == labels.shape
    for row in range(len(pairs)):
        decoder_ids, decoder_labels = ids[row, valid[row]], labels[row, valid[row]]
        assert decoder_ids[0] == tokenizer.bos_id and decoder_labels[0] != tokenizer.bos_id
        assert decoder_labels[-1] == tokenizer.eos_id
        assert torch.equal(decoder_ids[1:], decoder_labels[:-1])
    logits = torch.zeros(*labels.shape, tokenizer.vocab_size, requires_grad=True)
    loss = masked_cross_entropy(logits, labels, tokenizer.pad_id)
    loss.backward()
    assert logits.grad[valid].norm(dim=-1).gt(0).all()
    assert torch.count_nonzero(logits.grad[~valid]) == 0


@pytest.mark.parametrize(("labels", "error", "message"), [
    (torch.zeros(2, 3, dtype=torch.long), ValueError, "at least one non-padding"),
    (torch.ones(2, 3), TypeError, "torch.long"),
    (torch.ones(2, 2, dtype=torch.long), ValueError, "matching labels"),
    (torch.full((2, 3), 4, dtype=torch.long), ValueError, "within the vocabulary"),
    (torch.full((2, 3), -1, dtype=torch.long), ValueError, "within the vocabulary"),
])
def test_invalid_loss_inputs(labels: torch.Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        masked_cross_entropy(torch.zeros(2, 3, 4), labels, 0)
