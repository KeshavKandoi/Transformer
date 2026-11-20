from collections import defaultdict

import pytest

from scripts.interview_server import canonical_input, UnsupportedInput
from scripts.prepare_conversation_v2 import ALIASES, PROMPTS, generate_v2, sentence_key


def test_expanded_grammar_has_prompts_and_disjoint_groups():
    rows=generate_v2()
    assert rows==generate_v2()
    by_source={r['source']:r for r in rows}
    scope={sentence_key(r['source']):r['source'] for r in rows if r['split']!='structure'}
    assert all(canonical_input(prompt,scope,ALIASES) for prompt in PROMPTS)
    assert by_source['My name is Ravi.']['split']=='composition'
    assert by_source['I have two sisters.']['target']=='मेरी दो बहनें हैं।'
    assert by_source['Can she help me?']['target']=='क्या वह मेरी मदद कर सकती है?'
    for side in ('source','target'):
        assert len({sentence_key(r[side]) for r in rows})==len(rows)
    groups=defaultdict(set)
    for row in rows:
        groups[row['group_family'],tuple(row['group'])].add(row['split'])
    assert all(len(parts)==1 for parts in groups.values())
    assert {r['family'] for r in rows if r['split']=='structure'}.isdisjoint(
        {r['family'] for r in rows if r['split']=='train'})


def test_intentional_aliases_and_contraction_normalization():
    scope={'i love you':'I love you.', 'i do not like coffee':'I do not like coffee.'}
    assert canonical_input('LOVE YOU!',scope,ALIASES)=='I love you.'
    assert canonical_input('I don’t like coffee!',scope,ALIASES)=='I do not like coffee.'
    assert canonical_input('i DO  not like coffee',scope,ALIASES)=='I do not like coffee.'
    with pytest.raises(UnsupportedInput,match='Romanized Hindi'):
        canonical_input('kaise ho',scope,ALIASES)
    with pytest.raises(UnsupportedInput,match='outside'):
        canonical_input('love you',scope)


def test_v2_checkpoint_serves_alias_and_keeps_scope_enforcement(monkeypatch):
    from scripts.prepare_conversation_v2 import ROOT
    from scripts.interview_server import InterviewService

    checkpoint=ROOT/'checkpoints/scratch/best.pt'
    if not (ROOT/'test_evaluation.json').exists():
        pytest.skip('Local v2 training and evaluation artifacts required')
    import torch
    torch.set_num_threads(2)
    service=InterviewService(checkpoint,ROOT)
    assert len(service.info['gallery'])==20
    assert service.info['schema_version'] == 2
    assert service.info['grammar_version'] == 'v2'
    for fingerprint in ('checkpoint_sha256', 'dataset_sha256', 'tokenizer_sha256'):
        assert len(service.info[fingerprint]) == 64
    help_punctuated = service.predict('Can you help me?', 'beam')
    help_plain = service.predict('can you help me', 'beam')
    assert help_plain['canonical_source'] == help_punctuated['canonical_source'] == 'Can you help me?'
    for result in (help_punctuated, help_plain):
        assert result['outputs']['beam']['text'] == 'क्या तुम मेरी मदद कर सकते हो?'
        assert result['outputs']['beam']['eos']
    canonical=service.predict('I love you.')
    alias=service.predict('LOVE YOU!')
    assert alias['canonical_source']=='I love you.'
    assert alias['outputs']['greedy']['text']==canonical['outputs']['greedy']['text']
    assert alias['attention']['cross']==canonical['attention']['cross']
    assert service.predict('Where are you going?')['in_scope']

    def forbidden(*args,**kwargs):
        raise AssertionError('Out-of-domain input reached autoregressive generation')

    monkeypatch.setattr(service.translator,'generate',forbidden)
    for source in ('kaise ho','Why is the sky blue?','What is the weather in Paris?'):
        with pytest.raises(UnsupportedInput):
            service.predict(source)
