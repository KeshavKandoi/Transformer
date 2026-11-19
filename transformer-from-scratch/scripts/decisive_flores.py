"""Freeze a small FLORES baseline suite with exact and indexed near-overlap checks."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.clean_finetune import TOKENIZER, derange, measure
from scripts.clean_finetune_data import HeldOutIndex
from scripts.download_iitb_dataset import parquet_rows
from scripts.prepare_transformer_v2 import RAW, TRAIN_SHA, BASE, read_rows, save_json
from src.data.corpus import comparison_key
from src.data.prepared import file_digest
from src.inference import InferenceConfig, TransformerTranslator


def freeze(root, tokenizer):
    paths = [root/f'{lang}.devtest' for lang in ('eng_Latn','hin_Deva')]
    sides = [p.read_text(encoding='utf-8').splitlines() for p in paths]
    assert [len(s) for s in sides] == [1012,1012]
    rows = [dict(row_id=i+1,source=en,target=hi) for i,(en,hi) in enumerate(zip(*sides,strict=True))
            if en.strip() and hi.strip()]
    by_token_budget = [r for r in rows if all(len(tokenizer.encode(r[k]))<128 for k in ('source','target'))]
    random.Random(423).shuffle(by_token_budget)
    shortlist = by_token_budget[:160]
    keys = [{comparison_key(r[k]): r['row_id'] for r in shortlist} for k in ('source','target')]
    excluded = set()
    parquet_path = RAW/'train-00000-of-00001.parquet'
    assert file_digest(parquet_path)==TRAIN_SHA
    for pair in parquet_rows(pq.ParquetFile(parquet_path)):
        for i,text in enumerate((pair.source,pair.target)):
            if (number:=keys[i].get(comparison_key(text))) is not None:
                excluded.add(number)
    ancestor = read_rows(BASE/'processed/train.csv')
    recovery = Path('runs/translation-quality-recovery-v1')
    for arm in ('control','clean'):
        ancestor.extend(read_rows(recovery/arm/'train.csv'))
    inherited = json.loads((recovery/'evaluation_suite.json').read_text())
    ancestor.extend(read_rows(BASE/'source/validation_official.csv'))
    ancestor.extend(inherited['conversation_auxiliary'])
    indexed = HeldOutIndex(ancestor,.85)
    exact_excluded = set(excluded)
    for r in shortlist:
        if any(indexed.overlaps(r[k],i) for i,k in enumerate(('source','target'))):
            excluded.add(r['row_id'])
    fixed = [r for r in shortlist if r['row_id'] not in excluded][:100]
    assert len(fixed)==100
    save_json(root/'evaluation_suite.json', {
        'flores_devtest_100':fixed,
        'iitb_official_prepared_494':read_rows(BASE/'processed/val.csv'),
        'conversation_auxiliary_20':inherited['conversation_auxiliary'],
    })
    save_json(root/'flores_freeze.json',{
        'source_files':{p.name:file_digest(p) for p in paths},'revision':'DGME/FLORES-200@3cb4c84',
        'license':'CC BY-SA 4.0 per original FLORES-200 publisher',
        'seed':423,'raw_rows':len(rows),'within_128':len(by_token_budget),'shortlist':len(shortlist),
        'raw_train_exact_excluded':sorted(exact_excluded),'additional_indexed_excluded':sorted(excluded-exact_excluded),
        'near_check_rows':len(ancestor),'suite_sha256':file_digest(root/'evaluation_suite.json')})
    return fixed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root
    torch.set_num_threads(2)
    checkpoints={
        'original':Path('checkpoints/iitb-125k-wide-v1/best.pt'),
        'recovery_control':Path('checkpoints/quality-recovery-control-v1/best.pt'),
        'recovery_clean':Path('checkpoints/quality-recovery-clean-v1/best.pt'),
    }
    first=TransformerTranslator.from_checkpoint(checkpoints['original'],TOKENIZER,'cpu')
    rows=freeze(root,first.tokenizer)
    wrong=derange(rows)
    config=InferenceConfig(device='cpu',strategy='greedy',max_new_tokens=80)
    for label,path in checkpoints.items():
        system=first if label=='original' else TransformerTranslator.from_checkpoint(path,TOKENIZER,'cpu')
        result=measure(system.model,system.tokenizer,rows,wrong,config)
        result['checkpoint_sha256']=file_digest(path)
        save_json(root/f'flores_{label}.json',result)
        print(label,result['chrf']['score'],result['bleu']['score'],flush=True)


if __name__=='__main__':
    main()
