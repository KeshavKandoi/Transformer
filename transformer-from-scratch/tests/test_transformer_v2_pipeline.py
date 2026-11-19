from scripts.build_transformer_v2 import audit, train_tokenizer
from scripts.prepare_transformer_v2 import DuplicateIndex, quality, domain
from src.data.tokenizer import BilingualBPETokenizer


def test_unigram_unicode_special_ids_and_reload(tmp_path):
    rows = [
        {'source': 'I have two apples.', 'target': 'मेरे पास दो सेब हैं।'},
        {'source': 'I live in Delhi.', 'target': 'मैं दिल्ली में रहता हूँ।'},
        {'source': 'We have 12 books.', 'target': 'हमारे पास १२ किताबें हैं।'},
    ]
    tok = train_tokenizer(rows)
    tok.save(tmp_path)
    loaded = BilingualBPETokenizer.load(tmp_path)
    rare = 'नमस्ते 👩‍💻 — 𑀅'
    assert loaded.decode(loaded.encode(rare)) == rare
    assert loaded.unk_id not in loaded.encode(rare)
    assert len({loaded.pad_id, loaded.unk_id, loaded.bos_id, loaded.eos_id}) == 4
    assert all(record['roundtrip_nfc_exact'] == 3 and record['unknown_tokens'] == 0
               for record in audit(loaded, rows).values())


def test_quality_preserves_equal_digits_and_checks_bilingual_negation():
    src = 'I did not buy 12 apples.'
    assert quality(src, 'मैंने १२ सेब नहीं खरीदे।', domain(src)) is None
    assert quality(src, 'मैंने १३ सेब नहीं खरीदे।', domain(src)) == 'digit_mismatch_or_spelled_number'
    assert quality(src, 'मैंने १२ सेब खरीदे।', domain(src)) == 'negation_mismatch_or_paraphrase'


def test_bounded_near_duplicate_guard_checks_overlap():
    idx = DuplicateIndex()
    text = 'the library is open every morning from monday through friday for everyone in the city'
    idx.add(text)
    assert idx.overlaps(text) == 'exact'
    assert idx.overlaps(text.replace('monday','mondays')) == 'near'
    assert idx.overlaps('the library is open in a different building with a new entrance') is None
