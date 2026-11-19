from collections import Counter

from scripts.prepare_quality_recovery import classify, select_matched
from src.data.cleaning import TranslationPair


def test_quality_screen_flags_script_fragmentation_and_risk():
    source = 'We will not arrive before 1998.'
    target = 'हम 1988 से पहले पहुँचेंगे।'
    status, reasons = classify(TranslationPair(source, target), (12, 55))
    assert status == 'uncertain'
    assert {'digit_mismatch_or_spelled_number', 'negation_mismatch_or_paraphrase',
            'bpe_fragmentation_or_ratio'} <= set(reasons)
    assert classify(TranslationPair(source, 'We will arrive next year.'), (12, 12))[0] == 'rejected'
    assert classify(TranslationPair(source, target), (12, 129))[0] == 'rejected'


def test_matched_samples_exclude_risks_and_have_equal_strata():
    records = [
        dict(index=i, domain=('news_like' if i < 6 else 'general'), source_tokens=15, target_tokens=18,
             status=('high_confidence' if i % 2 else 'uncertain'), leakage=i in {0, 1},
             duplicate=i in {6, 7}, known_bad=i in {8, 9})
        for i in range(12)
    ]
    matched = select_matched(records, 3, 74)
    assert matched == select_matched(records, 3, 74)
    assert {row['index'] for match in matched for row in match} == {2, 3, 4, 5, 10, 11}
    assert Counter(a['domain'] for a, _ in matched) == Counter(b['domain'] for _, b in matched)
    assert all(a['status'] == 'uncertain' and b['status'] == 'high_confidence' for a, b in matched)
