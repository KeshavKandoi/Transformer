import random

from scripts.v2_alignment_gate import choose, independent, mismatches, number_flags


def test_alignment_flags_number_negation_and_named_entity():
    row = {'source':'I did not buy 12 books in Delhi in 1998.',
           'target':'मैंने १९८८ में मुंबई में १२ किताबें खरीदीं।'}
    assert {'digit_date_or_spelled_counterpart', 'negation_marker_or_paraphrase',
            'known_entity_alias_missing'} <= set(independent(row))
    assert number_flags('I have two dogs.', 'मेरे पास दो कुत्ते हैं।') == (False, False)
    assert number_flags('I have two dogs.', 'मेरे पास तीन कुत्ते हैं।') == (False, True)


def test_similarity_never_overrides_structural_risk():
    assert choose(.90, [], .2) == 'high_confidence'
    assert choose(.90, ['negation_marker_or_paraphrase'], .2) == 'uncertain'
    assert choose(.90, [], .2, known_bad=True) == 'rejected'
    assert choose(-.01, [], .2) == 'rejected'


def test_synthetic_negative_respects_domain_and_never_reuses_original_target():
    rows = [{'index': str(i), 'source': f'English {i}', 'target': f'हिंदी {i}', 'domain': 'general'}
            for i in range(5)]
    result = mismatches(rows, 5, random.Random(6))
    assert result == mismatches(rows, 5, random.Random(6))
    assert all(r['index'] != r['negative_target_index'] and
               r['target'] != rows[int(r['index'])]['target'] for r in result)
