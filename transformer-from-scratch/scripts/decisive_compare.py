"""Compare learned and structured scores on genuine errors and bilingual minimal pairs."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.clean_finetune import CHECKPOINT, TOKENIZER
from scripts.prepare_transformer_v2 import save_json
from scripts.v2_alignment_gate import BilingualRetriever, scores, independent
from scripts.validate_v2_alignment_gate import nll, rank_auc
from src.inference import TransformerTranslator

GATE = Path('runs/transformer-v2-alignment-gate')


def read(path):
    return json.loads(path.read_text())


def prepare(root):
    upstream = read(GATE/'final-v1/diagnostic_300.json')
    real = [{**r,'case_group':'previous_real'} for r in upstream
            if r['label'] in ('aligned','mismatch','mismatched','corrupt')]
    hard = []
    with (root/'hard_pairs.tsv').open(encoding='utf-8') as stream:
        for category,source,good,bad,required in csv.reader(stream,delimiter='\t'):
            hard.extend(dict(case_group='minimal_pair',contrast=category,source=source,target=target,
                             required=required,label=label) for label,target in (('anchor',good),('altered',bad)))
    sampler = read(root/'audit_base.json')+read(root/'b_screen_pool.json')
    assert all(r['origin']!='TED2020' for r in sampler)
    save_json(root/'scorer_input.json',sampler+real+hard)


def evaluate(root):
    rows = read(root/'labse_scores.json')['rows']
    torch.set_num_threads(2)
    audit = [r for r in rows if 'audit_id' in r]
    b = [r for r in rows if r.get('origin')=='IITB-B']
    for r in b:
        r['structural_flags']=independent(r)
    selected_b = sorted([r for r in b if r['labse_cosine']>=.85 and not r['structural_flags']],
                        key=lambda r:(-r['labse_cosine'],int(r['index'])))[:35]
    for i,r in enumerate(selected_b,166):
        r['audit_id']=i
    audit.extend(selected_b)
    save_json(root/'audit_200.json',audit)
    real = [r for r in rows if r.get('case_group')=='previous_real']
    hard = [r for r in rows if r.get('case_group')=='minimal_pair']
    retriever = BilingualRetriever()
    retriever.load_state_dict(torch.load(GATE/'data-v1/offline_retriever.pt',weights_only=True,map_location='cpu'))
    for r,score in zip(hard,scores(retriever.eval(),hard),strict=True):
        r['retrieval_cosine']=score
        r['required_concept_present']=r['required'] in r['target']
        r['structural_flags']=independent(r)
    translator = TransformerTranslator.from_checkpoint(CHECKPOINT,TOKENIZER,'cpu')
    for r,value in zip(hard,nll(translator.model,translator.tokenizer,hard),strict=True):
        r['old_checkpoint_nll']=value
    good = [r for r in real if r['label']=='aligned']
    bad = [r for r in real if r['label']!='aligned']
    metrics = {
        'natural_real_counts':[len(good),len(bad)],
        'natural_bad_types':Counter(r['label'] for r in bad),
        'natural_auc':{key:rank_auc([r[key] for r in good],[r[key] for r in bad])
                       for key in ('labse_cosine','score','old_checkpoint_source_margin')},
        'cosine_085':{'aligned_pass':sum(r['labse_cosine']>=.85 for r in good),
                      'real_errors_pass':sum(r['labse_cosine']>=.85 for r in bad)},
        'candidate_B_pool':len(b),'candidate_B_pass':sum(r['labse_cosine']>=.85 and not r['structural_flags'] for r in b),
        'audit_count':len(audit),
    }
    pairs = list(zip(hard[::2],hard[1::2],strict=True))
    metrics['minimal_pairs'] = {
        'n':len(pairs),'labse_prefers_anchor':sum(a['labse_cosine']>b['labse_cosine'] for a,b in pairs),
        'retriever_prefers_anchor':sum(a['retrieval_cosine']>b['retrieval_cosine'] for a,b in pairs),
        'old_nll_prefers_anchor':sum(a['old_checkpoint_nll']<b['old_checkpoint_nll'] for a,b in pairs),
        'fixed_labse_threshold_detects_altered':sum(b['labse_cosine']<.85 for _,b in pairs),
        'generic_flags_notice_altered':sum(bool(b['structural_flags']) for _,b in pairs),
        'explicit_concept_assertion_detects_altered':sum(not b['required_concept_present'] for _,b in pairs),
    }
    save_json(root/'scorer_metrics.json',metrics)
    save_json(root/'real_error_scores.json',real)
    save_json(root/'hard_negative_scores.json',hard)
    print(metrics,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','evaluate'))
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    (prepare if args.operation=='prepare' else evaluate)(args.root)
