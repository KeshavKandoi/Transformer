"""Verify v2 data, tokenizer, frozen suite and training checkpoint provenance."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_finetune_data import HeldOutIndex
from scripts.prepare_transformer_v2 import BASE, SEED, read_rows, save_json
from src.data.config import DataConfig
from src.data.prepared import file_digest, load_tokenizer_artifact, prepared_fingerprints
from src.training.checkpoint import load_checkpoint, validate_checkpoint
from src.training.utils import load_prepared_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args = parser.parse_args()
    dest = args.destination
    summary = json.loads((dest/'materialized.json').read_text())
    census = json.loads((dest/'census.json').read_text())
    assert file_digest(dest/'eligible.csv') == census['eligible_sha256']
    assert file_digest(dest/'tokenizer/tokenizer.json') == summary['tokenizer_sha256']
    labels = json.loads(Path('runs/translation-quality-recovery-v1/evaluation_suite.json').read_text())
    held = HeldOutIndex(read_rows(BASE/'source/validation_official.csv') + read_rows(BASE/'source/test_official.csv')
                        + labels['conversation_auxiliary'])
    last_indices = []
    checks = {}
    for size, entry in sorted(summary['artifacts'].items(),key=lambda x:int(x[0])):
        split = Path(entry['dataset'])
        identity = json.loads((split/'manifest.json').read_text())
        assert file_digest(split/'train_indices.json')==entry['selection_sha256']
        assert file_digest(split/'train.csv')==entry['train_sha256']
        assert prepared_fingerprints(split,dest/'tokenizer')['splits']==identity['splits']
        for held_split in ('val','test'):
            assert file_digest(split/f'{held_split}.csv')==file_digest(BASE/f'processed/{held_split}.csv')
        indices = json.loads((split/'train_indices.json').read_text())
        n = int(size)
        assert indices[:len(last_indices)]==last_indices
        assert len(indices)==len(set(indices))==n
        last_indices = indices
        selected = set(random.Random(SEED).sample(range(n),min(n,1000)))
        seen = 0
        with (split/'train.csv').open(encoding='utf-8',newline='') as stream:
            for i,row in enumerate(csv.DictReader(stream)):
                if i in selected:
                    assert not held.overlaps(row['source'],0) and not held.overlaps(row['target'],1)
                seen += 1
        assert seen == n
        checks[size] = {'exact_row_count':seen,'heldout_rows_rescreened':len(selected)}
    tokenizer, metadata = load_tokenizer_artifact(dest/'tokenizer')
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        data = load_prepared_data(dest/'prepared-64',dest/'tokenizer',DataConfig.load(dest/'data.yaml'))
        validate_checkpoint(checkpoint,checkpoint['model_config'],metadata,data.data_identity)
        assert checkpoint['global_step']==2 and checkpoint['epoch']==1
        assert checkpoint['tokenizer_metadata']['vocab_size']==tokenizer.vocab_size
    save_json(dest/'verification.json',{'checks':checks,'checkpoint':str(args.checkpoint) if args.checkpoint else None})
    print('Verified nested sizes, fingerprints, sampled overlap exclusion and checkpoint identity.')


if __name__=='__main__':
    main()
