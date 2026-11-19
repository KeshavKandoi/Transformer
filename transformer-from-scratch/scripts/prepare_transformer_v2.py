"""Inspect the pinned local IITB train split and build fingerprinted v2 artifacts."""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_finetune_data import HeldOutIndex, digits, negated
from scripts.download_iitb_dataset import REVISION, parquet_rows
from src.data.corpus import comparison_key, _grams
from src.data.prepared import file_digest

BASE = Path('experiments/iitb-125k-wide-v1')
RAW = Path('experiments/.iitb-hub-cache/datasets--cfilt--iitb-english-hindi/snapshots') / REVISION / 'data'
TRAIN_SHA = '39f6095040a8b3a202b9f13c8a95183e00177621c06923b0a3941bfeef0905ae'
BAD_TEXT = re.compile(r'[\x00-\x1f\ufffd]|https?://|<\s*/?\w+\s*>|%\s*[sd]|\{\d+\}|\b(?:LRB|RRB)\b')
LATIN = re.compile(r'[A-Za-z\u00C0-\u024F]')
HINDI = re.compile(r'[\u0904-\u0939\u0958-\u0961\u0972-\u097f]')
DOMAINS = [(k,re.compile(p,re.I)) for k,p in [
    ('religious',r'\b(?:allah|prophet|quran|lord|believers|unbelievers|hereafter|worship|god)\b'),
    ('legal_admin',r'\b(?:court|section|government|shall|ministry|scheme|authority|act|tribunal)\b'),
    ('software_ui',r'\b(?:software|file|server|click|button|dialog|browser|device)\b'),
    ('conversational_proxy',r"\b(?:i|you|we|my|your|please|hello|thanks|sorry)\b"),
]]
SEED = 173


def read_rows(path, **kwargs):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream, **kwargs))


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')


def domain(source):
    if len(source.split()) <= 3:
        return 'glossary_or_fragment'
    for key, pattern in DOMAINS:
        if pattern.search(source) and (key != 'conversational_proxy' or len(source.split()) <= 20):
            return key
    return 'general'


def quality(source, target, kind):
    sw = source.split()
    tw = target.split()
    if not source or not target:
        return 'empty'
    if BAD_TEXT.search(source) or BAD_TEXT.search(target):
        return 'markup_or_corruption'
    if len(source)>440 or len(target)>460 or len(sw)>65 or len(tw)>68:
        return 'length_ceiling'
    if len(sw)<4 or len(tw)<3:
        return 'short_glossary_or_fragment'
    for text, pattern in ((source,LATIN),(target,HINDI)):
        counts = len(pattern.findall(text))
        if counts < 4 or counts < .8 * sum(c.isalpha() for c in text):
            return 'wrong_script_or_language_proxy'
    if not .43 <= len(tw) / len(sw) <= 2.55:
        return 'word_ratio_or_missing_content'
    if digits(source) != digits(target):
        return 'digit_mismatch_or_spelled_number'
    if negated(source,'en') != negated(target,'hi'):
        return 'negation_mismatch_or_paraphrase'
    if re.search(r'[,;:/–-]$',source) or re.search(r'[,;:/–-]$',target):
        return 'likely_truncation'
    if kind=='software_ui' and (not re.search(r'[.?!]$',source) or len(sw)<7):
        return 'software_ui_fragment'
    if any(a.casefold()==b.casefold()==c.casefold() for words in (sw,tw) for a,b,c in zip(words,words[1:],words[2:])):
        return 'repetition'
    return None


class DuplicateIndex:
    """Exact keys plus bounded prefix/suffix candidates verified by trigram Jaccard."""
    def __init__(self):
        self.exact = set()
        self.buckets = defaultdict(list)

    @staticmethod
    def signatures(key):
        return () if len(key)<34 else (key[:18],key[-18:])

    def overlaps(self, key):
        if key in self.exact:
            return 'exact'
        for signature in self.signatures(key):
            candidates = self.buckets.get(signature, ())
            for candidate in candidates:
                if min(len(candidate),len(key)) < .75*max(len(candidate),len(key)):
                    continue
                a,b = _grams(key),_grams(candidate)
                if len(a & b) >= .9*len(a | b):
                    return 'near'
        return None

    def add(self,key):
        self.exact.add(key)
        for signature in self.signatures(key):
            bucket = self.buckets[signature]
            if len(bucket)<8:
                bucket.append(key)


def census(dest):
    path = RAW/'train-00000-of-00001.parquet'
    assert file_digest(path)==TRAIN_SHA, 'Unexpected official train artifact'
    parquet = pq.ParquetFile(path)
    aux = json.loads(Path('runs/translation-quality-recovery-v1/evaluation_suite.json').read_text())
    held_rows = (read_rows(BASE/'source/validation_official.csv') + read_rows(BASE/'source/test_official.csv')
                 + aux['conversation_auxiliary'])
    held = HeldOutIndex(held_rows)
    duplicate = DuplicateIndex(), DuplicateIndex()
    tally, seen_domains, kept_domains = Counter(), Counter(), Counter()
    with (dest/'eligible.csv').open('x',encoding='utf-8',newline='') as good, (
            dest/'row_audit.csv').open('x',encoding='utf-8',newline='') as audit:
        accepted, history = csv.writer(good), csv.writer(audit)
        accepted.writerow(('index','source','target','domain'))
        history.writerow(('index','domain','disposition'))
        for index, pair in enumerate(parquet_rows(parquet)):
            en, hi = (unicodedata.normalize('NFC', text.strip()) for text in (pair.source,pair.target))
            kind = domain(en)
            seen_domains[kind] += 1
            reason = quality(en,hi,kind)
            if reason is None:
                keys = comparison_key(en),comparison_key(hi)
                for side, text in enumerate((en,hi)):
                    blocked = held.overlaps(text, side)
                    if blocked:
                        reason = f'heldout_{side}_{blocked}'
                        break
                    blocked = duplicate[side].overlaps(keys[side])
                    if blocked:
                        reason = f'duplicate_{side}_{blocked}'
                        break
                if reason is None:
                    for side, key in enumerate(keys):
                        duplicate[side].add(key)
                    accepted.writerow((index,en,hi,kind))
                    kept_domains[kind]+=1
            tally[reason or 'eligible']+=1
            history.writerow((index,kind,reason or 'eligible'))
    assert parquet.metadata.num_rows==sum(tally.values())
    save_json(dest/'census.json',{'revision':REVISION,'input_sha256':TRAIN_SHA,'seed':SEED,
                                 'rows':sum(tally.values()),'disposition':tally,'domains_all':seen_domains,
                                 'domains_eligible':kept_domains,'heldout_jaccard':0.85,
                                 'eligible_sha256':file_digest(dest/'eligible.csv')})
    print(dict(tally),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args = parser.parse_args()
    args.destination.mkdir(exist_ok=True,parents=True)
    census(args.destination)


if __name__=='__main__':
    main()
