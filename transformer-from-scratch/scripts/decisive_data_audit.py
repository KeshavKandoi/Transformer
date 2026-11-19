"""Build an independent source audit and a separate real/synthetic challenge."""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.clean_finetune_data import HeldOutIndex
from scripts.download_iitb_dataset import parquet_rows
from scripts.prepare_transformer_v2 import BASE, RAW, TRAIN_SHA, domain, read_rows, save_json
from scripts.v2_alignment_gate import independent
from src.data.corpus import comparison_key
from src.data.prepared import file_digest

GATE = Path('runs/transformer-v2-alignment-gate/final-v1')
SEED = 823
EXTERNAL = {'Tatoeba':55,'GlobalVoices':45,'tico-19':30}


def read_zip(path):
    with ZipFile(path) as archive:
        files = [[p for p in archive.namelist() if p.endswith('.'+language)] for language in ('en','hi')]
        assert all(len(fs)==1 for fs in files)
        left,right = [archive.read(fs[0]).decode('utf-8').splitlines() for fs in files]
    if len(left)!=len(right):
        raise ValueError(f'{path} has unequal language file lengths')
    return [{'index':i,'source':en.strip(),'target':hi.strip(),'origin':path.stem}
            for i,(en,hi) in enumerate(zip(left,right,strict=True),1)]


def stratum(row):
    source = row['source']
    return (domain(source),min(len(source.split())//8,3),
            bool(re.search(r'\d',source)), bool(re.search(r'\b[A-Z][a-z]{2,}\b',source[1:])))


def stratified(rows, count, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    sample = []
    while len(sample)<count:
        available = [key for key in keys if groups[key]]
        if not available:
            raise ValueError('Not enough rows to furnish the requested sample')
        for key in available[:count-len(sample)]:
            sample.append(groups[key].pop())
    return sample


def source_audit(root):
    rng = random.Random(SEED)
    external = {name:read_zip(root/f'{name}.zip') for name in EXTERNAL}
    external_keys = [{comparison_key(r[side]) for rows in external.values() for r in rows}
                     for side in ('source','target')]
    appeared = [set(),set()]
    raw = RAW/'train-00000-of-00001.parquet'
    assert file_digest(raw)==TRAIN_SHA
    for pair in parquet_rows(pq.ParquetFile(raw)):
        for i,text in enumerate((pair.source,pair.target)):
            key = comparison_key(text)
            if key in external_keys[i]:
                appeared[i].add(key)
    flores = [p.read_text().splitlines() for p in (root/'eng_Latn.devtest',root/'hin_Deva.devtest')]
    held = HeldOutIndex(read_rows(BASE/'source/validation_official.csv')+read_rows(BASE/'source/test_official.csv')
                        +[dict(source=en,target=hi) for en,hi in zip(*flores,strict=True)])
    picked, records = [], {}
    for name,rows in external.items():
        counts = Counter()
        usable = []
        seen = set()
        for row in rows:
            keys = tuple(comparison_key(row[k]) for k in ('source','target'))
            if not all(keys):
                counts['empty'] += 1
            elif keys in seen:
                counts['duplicate_pair'] += 1
            elif any(keys[i] in appeared[i] for i in range(2)):
                counts['exact_side_overlap_with_raw_iitb']+=1
            elif any(held.overlaps(row[side],i) for i,side in enumerate(('source','target'))):
                counts['official_or_flores_overlap']+=1
            else:
                usable.append(row)
            seen.add(keys)
        selected = stratified(usable,EXTERNAL[name],rng)
        picked.extend(selected)
        records[name] = {'actual_rows':len(rows),'exclusions':counts,'independence_screen_survivors':len(usable),
                         'zip_sha256':file_digest(root/f'{name}.zip')}
    audited_sources = set()
    for file in (GATE/'fresh_30.json',GATE/'diagnostic_300.json'):
        audited_sources.update(comparison_key(r['source']) for r in json.loads(file.read_text()))
    # Reservoir sampling first bounds memory. A separate strata pass balances the audit.
    reservoir = []
    with (GATE/'high_confidence.csv').open(encoding='utf-8',newline='') as stream:
        for n,row in enumerate(csv.DictReader(stream)):
            if n<2800:
                reservoir.append(row)
            elif (j:=rng.randrange(n+1))<2800:
                reservoir[j]=row
    pool = [r for r in reservoir if comparison_key(r['source']) not in audited_sources]
    a = stratified(pool,35,rng)
    ids = {r['index'] for r in a}
    b = stratified([r for r in pool if r['index'] not in ids],175,rng)
    main = picked+[{**r,'origin':'IITB-A'} for r in a]
    for i,row in enumerate(main,1):
        row.update(audit_id=i,structural_flags=independent(row),stratum=stratum(row))
    save_json(root/'audit_base.json',main)
    save_json(root/'b_screen_pool.json',[{**r,'origin':'IITB-B'} for r in b])
    save_json(root/'source_inventory.json',records)
    print(records,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    source_audit(parser.parse_args().root)
