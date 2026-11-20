"""Independently regenerate all v1 validation rows without changing their artifacts."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_interview_model import CHECKPOINT, evaluate, write, words
from scripts.prepare_interview_data import ROOT
from src.data.prepared import file_digest
from src.inference import TransformerTranslator

AGREEMENT = {
    'gender_number': 'रहा रही रहे गया गई गए थका थकी थके जाऊँगा जाओगे जाएगा जाएगी जाएँगे'.split(),
    'copula': 'है हैं हूँ हो था थी थे थीं'.split(),
}


def main():
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in (ROOT/'pairs.jsonl').read_text().splitlines()]
    chosen = [r for r in rows if r['split']=='validation']
    old = json.loads((ROOT/'validation_evaluation.json').read_text())
    assert old['checkpoint_sha256']==file_digest(CHECKPOINT)
    raw = list(__import__('csv').DictReader((ROOT/'processed/val.csv').open(encoding='utf-8')))
    assert [(r['source'],r['target']) for r in chosen]==[(r['source'],r['target']) for r in raw]
    translator = TransformerTranslator.from_checkpoint(CHECKPOINT,ROOT/'tokenizer','cpu')
    vocab=set().union(*(words(r['source']) for r in rows if r['split']=='train'))
    recomputed=evaluate(translator,chosen,True,vocab)
    comparisons={}
    for key,anchors in AGREEMENT.items():
        anchors=set(anchors)
        eligible=[r for r in recomputed['rows'] if words(r['target']) & anchors]
        comparisons[key]={'n':len(eligible),'correct':sum(words(r['prediction']) & anchors==
                              words(r['target']) & anchors for r in eligible)}
    named=[r for r in recomputed['rows'] if r['family']=='introduction']
    comparisons['names']={'n':len(named),'correct':sum(r['target'].removeprefix('मेरा नाम ').removesuffix(' है।')
                                                         in r['prediction'] for r in named)}
    recomputed['audit_markers']=comparisons
    recomputed['stored_prediction_agreements']=sum(
        a['pair_id']==b['pair_id'] and a['target']==b['target'] and a['prediction']==b['prediction']
        and a['eos']==b['eos'] for a,b in zip(recomputed['rows'],old['validation']['rows'],strict=True))
    recomputed['failures']=[r for r in recomputed['rows'] if not r['exact']]
    recomputed['reported_examples']=[r for r in recomputed['rows'] if r['source'] in
                                    ('My name is Ravi.','They went to the shop yesterday.')]
    write(Path('runs/interview-conversation-v2/v1_recomputed.json'),recomputed)
    print('v1 exact/agreement',recomputed['exact'],recomputed['stored_prediction_agreements'],len(chosen),flush=True)


if __name__=='__main__':
    main()
