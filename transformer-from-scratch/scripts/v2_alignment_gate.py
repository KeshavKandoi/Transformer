"""Fit an offline bilingual retrieval scorer and screen the pinned IITB census."""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clean_finetune_data import digits, negated
from scripts.prepare_transformer_v2 import BASE, read_rows, save_json, domain
from src.data.cleaning import TranslationPair, _valid, write_parallel_csv
from src.data.config import DataConfig
from src.data.corpus import comparison_key, _language_identifier
from src.data.prepared import file_digest, write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer

PREP = Path('runs/transformer-v2-preparation')
RECOVERY = Path('runs/translation-quality-recovery-v1')
SEED = 209
FEATURES = 65536
WIDTH = 48
BUCKETS = (100000, 300000, 500000)

EN_NUM = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), range(13)))
HI_NUM = dict(zip('शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह'.split(), range(13)))
HI_NUM.update({'पांच':5, 'छः':6})
# Explicit aliases are narrow risk checks. OOV entities remain unverified.
ENTITIES = {
    'india':('भारत','इंडिया','हिंदुस्तान'), 'delhi':('दिल्ली','दिल्लि'),
    'mumbai':('मुंबई','मुम्बई','बंबई'), 'bombay':('मुंबई','मुम्बई','बंबई'),
    'pakistan':('पाकिस्तान','पाक'), 'china':('चीन','चाइना'),
    'gujarat':('गुजरात','गुजरात'), 'london':('लंदन','लण्डन'),
    'gandhi':('गांधी','गाँधी'), 'nepal':('नेपाल',), 'assam':('असम','आसाम'),
}


@lru_cache(maxsize=100000)
def feature(word):
    return 1 + zlib.crc32(word.encode()) % (FEATURES - 1)


def encode(rows, side):
    matrix = np.zeros((len(rows), WIDTH), dtype=np.int32)
    for i, row in enumerate(rows):
        values = [feature(word) for word in comparison_key(row[side]).split()[:WIDTH]]
        matrix[i, :len(values)] = values
    return torch.from_numpy(matrix)


class BilingualRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = nn.Embedding(FEATURES, 96, padding_idx=0)
        self.hi = nn.Embedding(FEATURES, 96, padding_idx=0)

    def forward(self, source, target):
        return F.normalize(self.en(source).sum(1)), F.normalize(self.hi(target).sum(1))


def fit(rows):
    torch.manual_seed(SEED)
    model = BilingualRetriever()
    en, hi = encode(rows,'source'), encode(rows,'target')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015, fused=False)
    history = []
    for _ in range(4):
        losses = []
        for ids in torch.randperm(len(rows)).split(256):
            a,b = model(en[ids],hi[ids])
            logits = a @ b.T / .09
            labels = torch.arange(len(ids))
            loss = (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels)) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return model.eval(), history


@torch.inference_mode()
def scores(model, rows):
    result = []
    for start in range(0,len(rows),512):
        batch = rows[start:start+512]
        left,right = model(encode(batch,'source'),encode(batch,'target'))
        result.extend((left*right).sum(1).tolist())
    return result


def number_flags(source,target):
    en_words = set(re.findall(r'\b[a-z]+\b',source.lower()))
    hi_words = set(comparison_key(target).split())
    hi_values = {HI_NUM[w] for w in hi_words & HI_NUM.keys()}
    hi_values.update(int(x) for x in digits(target) if x.isdecimal())
    # 'one' and 'zero' often function non-numerically; exact digit check still covers them.
    en_values = {EN_NUM[w] for w in en_words & EN_NUM.keys() if EN_NUM[w]>=2}
    return digits(source)!=digits(target), bool(en_values-hi_values)


def independent(row):
    source,target = row['source'],row['target']
    digits_differ, spell_risk = number_flags(source,target)
    flags = []
    if digits_differ:
        flags.append('digit_date_or_spelled_counterpart')
    if spell_risk:
        flags.append('small_quantity_or_idiom')
    if negated(source,'en') != negated(target,'hi'):
        flags.append('negation_marker_or_paraphrase')
    words = set(comparison_key(source).split())
    if any(not any(alias in target for alias in aliases) for name,aliases in ENTITIES.items() if name in words):
        flags.append('known_entity_alias_missing')
    if not .53 <= len(target.split())/max(1,len(source.split())) <= 2.2:
        flags.append('length_ratio')
    if re.search(r'&(?:amp|quot|nbsp|lt|gt);|[\ufffd\x00-\x1f]',source+' '+target):
        flags.append('markup_or_control')
    if re.match(r'^(?:of|with|from|because of|including)\b',source,re.I) and len(source.split())<9:
        flags.append('possible_incomplete_clause')
    return flags


def language_flags(row):
    model = _language_identifier()
    flags = []
    for side, expected in [('source','en'),('target','hi')]:
        if len(row[side])>=40:
            lang,conf = model.classify(row[side])
            if lang!=expected and conf>=.995:
                flags.append(f'{side}_langid_{lang}')
    return flags


def choose(score, flags, threshold, known_bad=False):
    if known_bad or 'markup_or_control' in flags or score < threshold-.17:
        return 'rejected'
    if flags or score < threshold:
        return 'uncertain'
    return 'high_confidence'


def mismatches(rows, count, rng):
    """Swap Hindi across distinct source rows within script/domain and approximate length."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['domain'], len(r['target'].split())//5].append(r)
    possible = [r for group in groups.values() if len(group)>1 for r in group]
    out = []
    for row in rng.sample(possible,count):
        group = groups[row['domain'],len(row['target'].split())//5]
        other = rng.choice([r for r in group if r['source']!=row['source'] and r['target']!=row['target']])
        out.append({**row,'target':other['target'],'negative_target_index':other['index'],'label':'synthetic_mismatch'})
    return out


def policy_data():
    prior = []
    paths = [RECOVERY/'review_180_machine_assessed.csv',
             PREP/'quality_sample_assessed.json',
             next(Path('runs/iitb-125k-wide-v1').glob('meaning-audit.*/audit_review.json'))]
    prior += [{**r,'label':r['assessment'].lower(),'label_source':'prior_AI_180'} for r in read_rows(paths[0])]
    for path,label_source,label_key in [(paths[1],'prior_AI_30','status'),(paths[2],'prior_AI_200','judgment')]:
        prior.extend({**r,'label':r[label_key],'label_source':label_source}
                     for r in json.loads(path.read_text())['rows'])
    unique = {(r['source'],r['target']):r for r in prior}
    return list(unique.values()),paths


def fingerprint(path):
    return file_digest(path)


def main(destination):
    destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(SEED)
    census = json.loads((PREP/'census.json').read_text())
    eligible_path = PREP/'eligible.csv'
    assert fingerprint(eligible_path)==census['eligible_sha256']
    prior, audit_paths = policy_data()
    forbidden = tuple({comparison_key(r[k]) for r in prior} for k in ('source','target'))
    known_bad = {(r['source'],r['target']) for r in prior if r['label'] in ('mismatch','mismatched','corrupt')}
    rows = read_rows(eligible_path)
    pool = [r for r in rows if int(r['index'])%7==0 and
            not any(comparison_key(r[k]) in keys for k,keys in zip(('source','target'),forbidden))]
    fit_rows = rng.sample(pool,min(72000,len(pool)))
    fit_ids = {r['index'] for r in fit_rows}
    model,history = fit(fit_rows)
    torch.save(model.state_dict(),destination/'offline_retriever.pt')
    save_json(destination/'fit_indices.json',sorted(map(int,fit_ids)))
    print('Offline retrieval loss',history,flush=True)

    calibration = [r for r in prior if int(r['index'])%2==0]
    audit = [r for r in prior if int(r['index'])%2!=0]
    candidate_pool = rng.sample([r for r in rows if int(r['index'])%7==3],10000)
    calibration_neg = mismatches(candidate_pool[:4000],100,rng)
    audit_neg = mismatches(candidate_pool[4000:8000],80,rng)
    cutoff = max(float(np.quantile(scores(model,calibration_neg),.95)),
                 float(np.quantile(scores(model,[r for r in calibration if r['label']=='aligned']),.20)))
    raw_scores = scores(model,rows)
    tokenizer = BilingualBPETokenizer.load(PREP/'tokenizer')
    accepted, distribution, flag_counts = [], Counter(), defaultdict(Counter)
    fieldnames = ['index','status','domain','score','reasons']
    with (destination/'row_scores.csv').open('w',encoding='utf-8',newline='') as report, (
            destination/'uncertain.csv').open('w',encoding='utf-8',newline='') as uncertain, (
            destination/'rejected.csv').open('w',encoding='utf-8',newline='') as rejected:
        writers = [csv.DictWriter(f,fieldnames=fieldnames) for f in (report,uncertain,rejected)]
        for w in writers:
            w.writeheader()
        with (PREP/'row_audit.csv').open(encoding='utf-8',newline='') as stream:
            for old in csv.DictReader(stream):
                if old['disposition']!='eligible':
                    item = dict(index=old['index'],status='rejected',domain=old['domain'],score='',
                                reasons='upstream:'+old['disposition'])
                    distribution['rejected']+=1
                    flag_counts['rejected'][item['reasons']]+=1
                    writers[0].writerow(item)
                    writers[2].writerow(item)
        for row,score in zip(rows,raw_scores,strict=True):
            flags = independent(row)
            if score>=cutoff and not flags:
                flags.extend(language_flags(row))
                if any(not _valid(row[k]) or len(tokenizer.encode(row[k]))>=192 for k in ('source','target')):
                    flags.append('invalid_or_over_192')
            bad = (row['source'],row['target']) in known_bad
            tier = choose(score,flags,cutoff,bad)
            reasons = flags+(['prior_AI_bad'] if bad else [])
            if score<cutoff:
                reasons.append('retrieval_below_threshold')
            item = dict(index=row['index'],status=tier,domain=row['domain'],score=round(score,7),reasons='|'.join(reasons))
            distribution[tier]+=1
            flag_counts[tier].update(reasons)
            writers[0].writerow(item)
            if tier=='high_confidence':
                accepted.append({**row,'score':score})
            else:
                writers[1 if tier=='uncertain' else 2].writerow(item)
    assert sum(distribution.values())==census['rows']
    rng.shuffle(accepted)
    fields = ['index','source','target','domain','score']
    with (destination/'high_confidence.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)

    prior_strata = defaultdict(list)
    for r,score in zip(audit,scores(model,audit),strict=True):
        prior_strata[(r['label_source'],r['label']=='aligned')].append(r)
    previous = []
    for group in prior_strata.values():
        rng.shuffle(group)
    while len(previous)<100 and any(prior_strata.values()):
        for group in prior_strata.values():
            if group and len(previous)<100:
                previous.append(group.pop())
    raw_candidates = candidate_pool[8000:]
    strata = defaultdict(list)
    for r,score in zip(raw_candidates,scores(model,raw_candidates),strict=True):
        band = 'high' if score>=cutoff else ('medium' if score>=cutoff-.17 else 'low')
        strata[r['domain'],len(r['source'].split())>=18,band].append(r)
    for group in strata.values():
        rng.shuffle(group)
    natural = []
    while len(natural)<120:
        for group in strata.values():
            if group and len(natural)<120:
                natural.append({**group.pop(),'label':'unlabeled_real'})
    diagnostic = previous+audit_neg+natural
    assert len(diagnostic)==300
    for row,score in zip(diagnostic,scores(model,diagnostic),strict=True):
        row.setdefault('domain',domain(row['source']))
        loose_flags = independent(row)+language_flags(row)
        row.update(score=score,flags=loose_flags,policy_blind_to_label=choose(score,loose_flags,cutoff),
                   independent_human_label=False)
    save_json(destination/'diagnostic_300.json',diagnostic)
    fresh = rng.sample([r for r in accepted if r['index'] not in fit_ids and
                        int(r['index'])%7 not in (0,3)],30)
    save_json(destination/'fresh_30.json',fresh)

    data_config = DataConfig.load(PREP/'data.yaml')
    datasets = {}
    for size in BUCKETS:
        if size>len(accepted):
            continue
        split = destination/f'prepared-{size}'
        split.mkdir()
        chosen = accepted[:size]
        write_parallel_csv(split/'train.csv',[TranslationPair(r['source'],r['target']) for r in chosen])
        for key in ('val','test'):
            (split/f'{key}.csv').write_bytes((PREP/f'prepared-64/{key}.csv').read_bytes())
        save_json(split/'indices.json',[int(r['index']) for r in chosen])
        write_preparation_manifest(split,PREP/'tokenizer',eligible_path,data_config,
                                   corpus={'selection_sha256':fingerprint(split/'indices.json'),
                                           'alignment_threshold':cutoff,'seed':SEED})
        datasets[str(size)] = {f:fingerprint(split/f) for f in ('train.csv','indices.json','manifest.json')}
    fit_index_old = set(json.loads((PREP/'tokenizer_fit_indices.json').read_text()))
    save_json(destination/'summary.json',{'distribution':distribution,'reasons':flag_counts,
        'seed':SEED,'cutoff':cutoff,'fit_rows':len(fit_rows),'fit_history':history,'datasets':datasets,
        'eligible_sha256':fingerprint(eligible_path),'census_sha256':fingerprint(PREP/'row_audit.csv'),
        'prior_audits':{str(p):fingerprint(p) for p in audit_paths},
        'outputs':{p.name:fingerprint(p) for p in destination.iterdir() if p.is_file()},
        'prior_tokenizer_fit_rows_retained':sum(int(r['index']) in fit_index_old for r in accepted),
        'domains_high':Counter(r['domain'] for r in accepted)})
    print(dict(distribution),'threshold',cutoff,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    main(parser.parse_args().destination)
