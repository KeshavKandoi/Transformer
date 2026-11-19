import json
from collections import defaultdict

import pytest
import torch

from scripts.evaluate_interview_model import CHECKPOINT, slot_checks
from scripts.interview_server import InterviewService
from scripts.prepare_interview_data import ROOT, generate
from src.data.corpus import comparison_key


def test_generator_examples_and_gender_number():
    rows = {row['source']:row for row in generate()}
    assert rows['I have two books.']['target'] == 'मेरे पास दो किताबें हैं।'
    assert rows['I do not like tea.']['target'] == 'मुझे चाय पसंद नहीं है।'
    assert rows['She is reading one book.']['target'] == 'वह एक किताब पढ़ रही है।'
    assert rows['He is reading one book.']['target'] == 'वह एक किताब पढ़ रहा है।'
    assert rows['Where are you going?']['target'] == 'तुम कहाँ जा रहे हो?'
    assert rows['Where are you going?']['split'] == 'structure'
    assert rows['Please bring two apples.']['target'] == 'कृपया दो सेब लाओ।'
    check = slot_checks(rows['She is reading one book.'],'वह दो किताबें पढ़ रहा है।')
    assert not check['number'] and not check['agreement'] and check['verb']


def test_deterministic_grouping_and_unseen_families():
    rows = generate()
    assert rows == generate()
    for side in ('source','target'):
        assert len({comparison_key(row[side]) for row in rows})==len(rows)
    assert len({row['pair_id'] for row in rows})==len(rows)
    groups, lexical = defaultdict(set), defaultdict(set)
    for row in rows:
        if row['split']!='structure':
            groups[row['group_family'],tuple(row['group'])].add(row['split'])
            if row['lexical_group'] is not None:
                lexical[tuple(row['lexical_group'])].add(row['split'])
    assert all(len(splits)==1 for splits in groups.values())
    assert all(splits=={'lexical'} for splits in lexical.values() if 'lexical' in splits)
    assert {row['family'] for row in rows if row['split']=='structure'}.isdisjoint(
        {row['family'] for row in rows if row['split']=='train'})


@pytest.mark.skipif(not CHECKPOINT.is_file() or not (ROOT/'validation_evaluation.json').is_file(),
                    reason='Local interview training artifacts required')
def test_service_runs_real_checkpoint_with_valid_attention():
    torch.set_num_threads(2)
    service=InterviewService()
    prediction=service.predict('I have two books.','compare')
    assert prediction['in_scope']
    assert prediction['outputs']['greedy']['text'] == 'मेरे पास दो किताबें हैं।'
    assert set(prediction['outputs']) == {'greedy','beam'}
    maps=prediction['attention']
    encoder=torch.tensor(maps['encoder'])
    decoder=torch.tensor(maps['decoder'])
    cross=torch.tensor(maps['cross'])
    assert encoder.shape[:2] == decoder.shape[:2] == cross.shape[:2] == (2,4)
    assert cross.shape[-2:] == (len(maps['target_input']),len(maps['source']))
    assert torch.allclose(encoder.sum(-1),torch.ones_like(encoder.sum(-1)),atol=1e-6)
    assert decoder.triu(1).count_nonzero()==0
    assert not service.predict('Where are you going?')['in_scope']
    with pytest.raises(ValueError):
        service.predict('   ')
    recorded=json.loads((ROOT/'validation_evaluation.json').read_text())
    assert service.info['metrics']['exact']==recorded['validation']['exact']
