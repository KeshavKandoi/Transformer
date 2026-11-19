"""Fit a 16K Unicode Unigram tokenizer and materialize nested training sets."""
import argparse
import json
import random
import shutil
import string
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import yaml
from tokenizers import SentencePieceUnigramTokenizer, decoders, models, normalizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_transformer_v2 import BASE, SEED, read_rows, save_json
from src.data.cleaning import TranslationPair, _valid, write_parallel_csv
from src.data.config import DataConfig
from src.data.prepared import file_digest, write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer, SPECIAL_TOKENS


def train_tokenizer(rows):
    model = SentencePieceUnigramTokenizer(add_prefix_space=False)
    model.normalizer = normalizers.NFC()
    alphabet = sorted(set(string.printable.strip()) | {chr(i) for i in range(0x0900,0x0980)})
    model.train_from_iterator((r[side] for r in rows for side in ('source','target')),
                              vocab_size=16000 - 256, show_progress=False, special_tokens=list(SPECIAL_TOKENS),
                              initial_alphabet=alphabet, unk_token='<unk>', length=2*len(rows))
    backend = model._tokenizer
    unigram = json.loads(backend.to_str())['model']
    vocab = [tuple(piece) for piece in unigram['vocab']] + [(f'<0x{byte:02X}>', -100.0) for byte in range(256)]
    backend.model = models.Unigram(vocab, unk_id=unigram['unk_id'], byte_fallback=True)
    backend.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Metaspace(replacement='▁', prepend_scheme='never'),
    ])
    backend.encode_special_tokens = True
    return BilingualBPETokenizer(backend)


def percentile(values, fraction):
    rank = int(fraction * (len(values)-1))
    return sorted(values)[rank]


def audit(tokenizer, rows):
    result = {}
    for side in ('source','target'):
        tokens, lengths, words, exact, unknown = [], [], 0, 0, 0
        for r in rows:
            text = r[side]
            ids = tokenizer.encode(text)
            tokens.extend(ids)
            lengths.append(len(ids)+1)
            words += len(text.split())
            unknown += ids.count(tokenizer.unk_id)
            exact += tokenizer.decode(ids)==unicodedata.normalize('NFC', text)
        counts = Counter(tokens)
        byte_fallback = {i for i in counts if len(tokenizer.id_to_token(i))==6
                         and tokenizer.id_to_token(i).startswith('<0x')
                         and tokenizer.id_to_token(i).endswith('>')}
        broken = {i for i in counts if '\ufffd' in tokenizer.decode([i])}
        result[side] = {'rows':len(rows),'tokens':len(tokens),'words':words,
                        'tokens_per_word':len(tokens)/words,
                        'unknown_tokens':unknown,'roundtrip_nfc_exact':exact,
                        'broken_single_piece_occurrences':sum(counts[i] for i in broken),
                        'broken_single_piece_types':len(broken),
                        'byte_fallback_tokens':sum(counts[i] for i in byte_fallback),
                        'singleton_piece_occurrences':sum(n==1 for n in counts.values()),
                        'mean_with_eos':mean(lengths),'p95_with_eos':percentile(lengths,.95),
                        'p99_with_eos':percentile(lengths,.99),
                        'over_128':sum(n>128 for n in lengths),'over_192':sum(n>192 for n in lengths)}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('destination',type=Path)
    root = cli.parse_args().destination
    eligible = root/'eligible.csv'
    assert file_digest(eligible)==json.loads((root/'census.json').read_text())['eligible_sha256']
    rows = read_rows(eligible)
    audit_path = Path('runs/translation-quality-recovery-v1/review_180_machine_assessed.csv')
    known_bad = {(r['source'], r['target']) for r in read_rows(audit_path)
                 if r['assessment'] in ('MISMATCHED', 'CORRUPT')}
    previous_audit = next(Path('runs/iitb-125k-wide-v1').glob('meaning-audit.*/audit_review.json'))
    known_bad.update((r['source'], r['target']) for r in json.loads(previous_audit.read_text())['rows']
                     if r['judgment'] == 'mismatch')
    quarantined = [int(r['index']) for r in rows if (r['source'], r['target']) in known_bad]
    rows = [r for r in rows if (r['source'], r['target']) not in known_bad]
    save_json(root/'prior_audit_quarantine.json', {'indices':quarantined, 'sources':{
        str(path):file_digest(path) for path in (audit_path,previous_audit)}})
    random.Random(SEED).shuffle(rows)
    fit, frozen = rows[:60000], rows[-2048:]
    assert len(fit)==60000 and not {r['index'] for r in fit} & {r['index'] for r in frozen}
    tokenizer = train_tokenizer(fit)
    tokenizer.save(root/'tokenizer')
    tokenizer = BilingualBPETokenizer.load(root/'tokenizer')
    save_json(root/'tokenizer_fit_indices.json',[int(r['index']) for r in fit])
    save_json(root/'tokenizer_audit_train.json',frozen)
    validation = read_rows(BASE/'processed/val.csv')
    assert len(validation)==494
    baseline = BilingualBPETokenizer.load(BASE/'tokenizer')
    save_json(root/'tokenizer_comparison.json',{label:{'vocab_size':tok.vocab_size,
              'training_sample':audit(tok,frozen),'official_validation':audit(tok,validation)}
              for label,tok in [('byte_bpe_8k',baseline),('sentencepiece_style_unigram_16k',tokenizer)]})
    print('Tokenizer comparison written',flush=True)
    data = DataConfig(seed=SEED,vocab_size=tokenizer.vocab_size,max_source_length=192,max_target_length=192)
    (root/'data.yaml').write_text(yaml.safe_dump(asdict(data)))
    caps = (100000,300000,500000)
    kept, exclusions = [], Counter()
    fit_ids = {r['index'] for r in fit}
    for row in rows:
        encodings = [tokenizer.encode(row[key]) for key in ('source','target')]
        if not all(_valid(row[key]) for key in ('source','target')):
            exclusions['invalid_text'] += 1
            continue
        if any(len(ids)>=192 for ids in encodings):
            exclusions['over_192_positions'] += 1
            continue
        if any(tokenizer.unk_id in ids for ids in encodings):
            exclusions['unknown_token'] += 1
            continue
        kept.append(row)
    sizes = [n for n in caps if n <= len(kept)]
    if sizes and sizes[-1] != len(kept):
        sizes.append(len(kept))
    if not sizes:
        sizes = [len(kept)]
    manifest = {}
    for size in sorted(set((64,*sizes))):
        split = root/f'prepared-{size}'
        split.mkdir()
        selected = kept[:size]
        write_parallel_csv(split/'train.csv',[TranslationPair(r['source'],r['target']) for r in selected])
        shutil.copyfile(BASE/'processed/val.csv',split/'val.csv')
        # The existing training manifest requires a test fingerprint. The trainer uses train/val only.
        shutil.copyfile(BASE/'processed/test.csv',split/'test.csv')
        save_json(split/'train_indices.json',[int(r['index']) for r in selected])
        write_preparation_manifest(split,root/'tokenizer',eligible,data,corpus={
            'license':'CC BY-NC 4.0; source component licenses may apply',
            'selection_seed':SEED,'training_indices_sha256':file_digest(split/'train_indices.json'),
            'fitted_tokenizer_indices_sha256':file_digest(root/'tokenizer_fit_indices.json')})
        manifest[str(size)] = {'dataset':str(split),'domains':dict(Counter(r['domain'] for r in selected)),
                               'train_sha256':file_digest(split/'train.csv'),'selection_sha256':file_digest(split/'train_indices.json'),
                               'tokenizer_fit_rows_present':sum(r['index'] in fit_ids for r in selected)}
    save_json(root/'materialized.json', {'total_eligible_after_tokenization':len(kept),
               'after_tokenization_exclusions':dict(exclusions),'artifacts':manifest,
               'tokenizer_sha256':file_digest(root/'tokenizer/tokenizer.json')})
    print('Prepared sizes',list(manifest),flush=True)


if __name__=='__main__':
    main()
