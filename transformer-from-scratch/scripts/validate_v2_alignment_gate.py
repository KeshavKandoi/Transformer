"""Measure gate behavior on held-out diagnostics and emit fit-disjoint train tiers."""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_transformer_v2 import audit, train_tokenizer
from scripts.clean_finetune import CHECKPOINT, TOKENIZER, batch_for
from scripts.clean_finetune_data import HeldOutIndex
from scripts.prepare_transformer_v2 import BASE, read_rows, save_json
from scripts.v2_alignment_gate import BUCKETS, PREP, SEED, policy_data, independent
from src.data.cleaning import TranslationPair,write_parallel_csv
from src.data.config import DataConfig
from src.data.corpus import comparison_key
from src.data.prepared import file_digest,write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer
from src.inference import TransformerTranslator
from src.training.utils import forward_batch


def read(path):
    return json.loads(path.read_text())


@torch.inference_mode()
def nll(model, tokenizer, rows):
    out = []
    for start in range(0,len(rows),8):
        batch = batch_for(rows[start:start+8],tokenizer)
        logits, target = forward_batch(model,batch,torch.device('cpu'))
        values = F.cross_entropy(logits.transpose(1,2),target,ignore_index=tokenizer.pad_id,reduction='none')
        out.extend((values.sum(1)/target.ne(tokenizer.pad_id).sum(1)).tolist())
    return out


def rank_auc(left,right):
    return sum(float(a>b)+.5*(a==b) for a in left for b in right)/(len(left)*len(right))


def validate_diagnostic(root, dest):
    diagnostic = read(root/'diagnostic_300.json')
    translator = TransformerTranslator.from_checkpoint(CHECKPOINT,TOKENIZER,'cpu')
    tok = translator.tokenizer
    lengths = [len(tok.encode(r['source'])) for r in diagnostic]
    wrong = []
    for i,r in enumerate(diagnostic):
        j = min((j for j,q in enumerate(diagnostic) if q['source']!=r['source']),
                key=lambda j: (abs(lengths[i]-lengths[j]),j))
        wrong.append({**r,'source':diagnostic[j]['source']})
    observed, swapped = (nll(translator.model,tok,rows) for rows in (diagnostic,wrong))
    cutoff = read(root/'summary.json')['cutoff']
    for r,actual,null in zip(diagnostic,observed,swapped,strict=True):
        r['old_checkpoint_source_margin'] = null-actual
        r['signals_disagree'] = (r['score']>=cutoff) != (null>actual and not r['flags'])
    save_json(dest/'diagnostic_300.json',diagnostic)
    breakdown = {}
    for key in sorted({r['label'] for r in diagnostic}):
        group = [r for r in diagnostic if r['label']==key]
        breakdown[key] = {'n':len(group),'tier':Counter(r['policy_blind_to_label'] for r in group),
                          'disagreements':sum(r['signals_disagree'] for r in group)}
    positive = [r for r in diagnostic if r['label']=='aligned']
    negative = [r for r in diagnostic if r['label'] in ('synthetic_mismatch','mismatch','mismatched','corrupt')]
    save_json(dest/'diagnostic_metrics.json',{'by_label':breakdown,
        'auc_retriever_vs_ai_aligned_and_known_or_constructed_negatives':rank_auc(
            [r['score'] for r in positive],[r['score'] for r in negative]),
        'auc_old_margin_on_same_rows':rank_auc([r['old_checkpoint_source_margin'] for r in positive],
                    [r['old_checkpoint_source_margin'] for r in negative]),
        'independent_human_labels':0,'different_methods_same_underlying_corpus':True})


def finalize(root, dest):
    fit = set(read(root/'fit_indices.json'))
    provisional = read_rows(root/'high_confidence.csv')
    hard_excluded = {int(r['index']) for r in provisional if int(r['index']) in fit}
    chosen = [r for r in provisional if int(r['index']) not in hard_excluded]
    assert not {int(r['index']) for r in chosen} & fit
    fields = ['index','status','domain','score','reasons']
    counts, hints = Counter(), Counter()
    with (dest/'row_scores.csv').open('w',encoding='utf-8',newline='') as all_rows, (
            dest/'uncertain.csv').open('w',encoding='utf-8',newline='') as unsure, (
            dest/'rejected.csv').open('w',encoding='utf-8',newline='') as rejected, (
            root/'row_scores.csv').open(encoding='utf-8',newline='') as stream:
        writers = {name:csv.DictWriter(handle,fieldnames=fields)
                   for name,handle in [('all',all_rows),('uncertain',unsure),('rejected',rejected)]}
        for writer in writers.values():
            writer.writeheader()
        for r in csv.DictReader(stream):
            if int(r['index']) in hard_excluded:
                r.update(status='uncertain',reasons='trained_retriever_on_this_pair')
            counts[r['status']]+=1
            hints.update(r['reasons'].split('|') if r['reasons'] else ())
            writers['all'].writerow(r)
            if r['status'] in ('uncertain','rejected'):
                writers[r['status']].writerow(r)
    with (dest/'high_confidence.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=('index','source','target','domain','score'))
        writer.writeheader()
        writer.writerows(chosen)
    assert sum(counts.values())==1659083 and counts['high_confidence']==len(chosen)
    prior,_ = policy_data()
    forbidden = {comparison_key(r['source']) for r in prior}
    fresh_pool = [r for r in chosen if int(r['index'])%7 not in (0,3) and comparison_key(r['source']) not in forbidden]
    fresh = random.Random(311).sample(fresh_pool,30)
    save_json(dest/'fresh_30.json',fresh)
    baseline = BilingualBPETokenizer.load(PREP/'tokenizer')
    fit_rows = chosen[:min(30000,len(chosen)//2)]
    sample = random.Random(19).sample(chosen[len(fit_rows):],1000)
    save_json(dest/'tokenizer_refit_indices.json',[int(r['index']) for r in fit_rows])
    retrained = train_tokenizer(fit_rows)
    retrained.save(dest/'tokenizer_experimental')
    retrained = BilingualBPETokenizer.load(dest/'tokenizer_experimental')
    save_json(dest/'tokenizer_compare.json',{name:audit(tok,sample) for name,tok in [('retained',baseline),('refit',retrained)]})
    data = DataConfig.load(PREP/'data.yaml')
    sizes = [size for size in BUCKETS if size<=len(chosen)]
    # Exclusion verification is independent of the trained retrieval model.
    suites = read(PREP/'evaluation_suite.json')
    held = HeldOutIndex(read_rows(BASE/'source/validation_official.csv')+read_rows(BASE/'source/test_official.csv')
                        +suites['conversation_auxiliary'])
    for r in sample+fresh:
        assert not held.overlaps(r['source'],0) and not held.overlaps(r['target'],1)
    split_stats = {}
    for size in sizes:
        split = dest/f'prepared-{size}'
        split.mkdir()
        rows = chosen[:size]
        write_parallel_csv(split/'train.csv',[TranslationPair(r['source'],r['target']) for r in rows])
        for name in ('val','test'):
            (split/f'{name}.csv').write_bytes((PREP/f'prepared-64/{name}.csv').read_bytes())
        save_json(split/'indices.json',[int(r['index']) for r in rows])
        write_preparation_manifest(split,PREP/'tokenizer',dest/'high_confidence.csv',data,
           corpus={'offline_filter_seed':SEED,'cutoff':read(root/'summary.json')['cutoff'],
                   'selection_sha256':file_digest(split/'indices.json')})
        split_stats[str(size)] = {f:file_digest(split/f) for f in ('train.csv','indices.json','manifest.json')}
    save_json(dest/'summary.json',{'counts':counts,'reasons':hints,'demoted_scorer_fit_rows':len(hard_excluded),
        'sampled_holdout_checks':len(sample)+len(fresh),'sizes':split_stats,
        'domain_high':Counter(r['domain'] for r in chosen),
        'retained_tokenizer_sha256':file_digest(PREP/'tokenizer/tokenizer.json'),
        'experimental_tokenizer_sha256':file_digest(dest/'tokenizer_experimental/tokenizer.json'),
        'spaces_verified':'whole prepared rows NFC in comparison metrics',
        'artifacts':{f.name:file_digest(f) for f in dest.iterdir() if f.is_file()}})
    print('Final counts',dict(counts),'named sizes',sizes,flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('provisional',type=Path)
    cli.add_argument('destination',type=Path)
    args = cli.parse_args()
    args.destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(3)
    validate_diagnostic(args.provisional,args.destination)
    finalize(args.provisional,args.destination)


if __name__=='__main__':
    main()
