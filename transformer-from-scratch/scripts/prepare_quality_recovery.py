"""Materialize audited quality tiers and deterministic matched training selections."""
import argparse
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_finetune import BASE, CHECKPOINT, TOKENIZER, read, write
from scripts.clean_finetune_data import HeldOutIndex, domain, screen
from src.data.cleaning import write_parallel_csv
from src.data.config import DataConfig
from src.data.corpus import _dominant_script, comparison_key, leakage_groups
from src.data.prepared import file_digest, prepared_fingerprints
from src.model.config import ModelConfig
from src.training.checkpoint import load_checkpoint, validate_checkpoint
from src.training.utils import load_prepared_data

TIERS = {
    'high_confidence_candidate': 'high_confidence',
    'uncertain': 'uncertain',
    'rejected': 'rejected',
}
CORRECTIONS = {
    33365: 'देशभर में अच्छी फसल के त्योहार मनाए जाते हैं, जैसे असम में रोंगाली बिहू, बंगाल में नबान्न, पंजाब में बैसाखी, केरल में ओणम और तमिलनाडु में पोंगल।',
    11015: 'क्या आप वाकई अमान्य पतों वाले संदेश को भेजना चाहते हैं?',
    72984: '8 सितंबर 1998 से मंत्रालय पशु कल्याण का काम भी देख रहा है।',
    42111: 'एक ऐतिहासिक कार्यक्रम में प्रधानमंत्री 11 अक्टूबर को स्वामित्व योजना के तहत संपत्ति कार्डों के भौतिक वितरण की शुरुआत करेंगे।',
    82929: 'खाड़ियों को बंद करके हासिल की गई ज़मीन टोका भूधारण व्यवस्था के तहत कुनबी किसानों को किराए पर दी गई थी। इसमें उपज, मुख्यतः चावल, ज़मीन की मालिक कंपनी और किसानों के बीच बाँटी जाती थी।',
}


def csv_rows(path, **kwargs):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream, **kwargs))


def review(queue, notes, output):
    with notes.open(encoding='utf-8') as stream:
        judgment = {int(n): (label, issue) for n, label, issue in csv.reader(stream, dialect='excel-tab')}
    output_rows = []
    for n, record in enumerate(csv_rows(queue, dialect='excel-tab'), 1):
        label, issue = judgment.get(n, ('ALIGNED', 'Main meaning and applicable details are preserved.'))
        index = int(record['index'])
        output_rows.append({'index': index, 'source': record['source'], 'target': record['target'],
                            'assessment': label, 'issue': issue, 'confidence': 'moderate' if label == 'UNCERTAIN' else 'high',
                            'suggested_hindi': CORRECTIONS.get(index, ''),
                            'explanation': 'AI comparison of both texts in context; glossary and UI may be valid as fragments.',
                            'human_verification_missing': True})
    assert len(output_rows) == len({r['index'] for r in output_rows}) == 180
    assert all(r['assessment'] != 'ALIGNED' for r in output_rows if r['index'] in CORRECTIONS)
    with (output/'review_180_machine_assessed.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0])
        writer.writeheader()
        writer.writerows(output_rows)
    write(output/'review_180_workbook_input.json', output_rows)
    lines = ['# AI-assisted alignment issues\n', 'No independent bilingual review. Suggested corrections were excluded from training.\n']
    for row in output_rows:
        if row['assessment'] != 'ALIGNED':
            lines.append(f"- {row['index']} · {row['assessment']} · {row['issue']}")
    (output/'review_180_issues.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return output_rows


def classify(pair, lengths):
    status, reasons = screen(pair.source, pair.target)
    if any(not _dominant_script(text, script) for text, script in
           ((pair.source, 'LATIN'), (pair.target, 'DEVANAGARI'))):
        return 'rejected', reasons + ['script_mismatch']
    if any(unicodedata.category(c) == 'Cc' or c == '\ufffd' for c in pair.source + pair.target):
        return 'rejected', reasons + ['broken_unicode_or_control']
    if max(lengths)>128 or not min(lengths):
        return 'rejected', reasons + ['bpe_truncation_or_empty']
    if status == 'rejected':
        return status, reasons
    left, right = lengths
    if right > 4.5 * max(1, len(pair.target.split())) or not 0.5 <= right / left <= 3.5:
        reasons.append('bpe_fragmentation_or_ratio')
    if re.search(r'external website that opens|all rights reserved|\{\d+\}', pair.source, re.I):
        reasons.append('template_or_placeholder')
    if any(a == b == c for a,b,c in zip(pair.source.split(), pair.source.split()[1:], pair.source.split()[2:])):
        reasons.append('repeated_english')
    return ('uncertain' if reasons else 'high_confidence_candidate'), reasons


def select_matched(records, sample_size, seed):
    clean, control = defaultdict(list), defaultdict(list)
    for record in records:
        if record['leakage'] or record['duplicate'] or record['known_bad']:
            continue
        key = (record['domain'], record['source_tokens'] // 8, record['target_tokens'] // 16)
        if record['status'] == 'high_confidence':
            clean[key].append(record)
        elif record['status'] == 'uncertain':
            control[key].append(record)
    rng = random.Random(seed)
    matches = []
    for key in sorted(clean.keys() & control.keys()):
        rng.shuffle(clean[key])
        rng.shuffle(control[key])
        matches.extend(zip(control[key], clean[key]))
    if len(matches) < sample_size:
        raise ValueError(f'Only {len(matches)} matched pairs are available')
    return rng.sample(matches, sample_size)


def prepare(args):
    data = load_prepared_data(BASE/'processed', TOKENIZER, DataConfig.load(Path('configs/experiment_data.yaml')))
    base = load_checkpoint(CHECKPOINT)
    validate_checkpoint(base, asdict(ModelConfig.load(Path('configs/iitb_full_model.yaml'))),
                        data.tokenizer_metadata, data.data_identity)
    reviewed = review(Path('runs/iitb-clean-finetune-matched-v1/bilingual_review_queue.tsv'),
                      args.output/'machine_notes.tsv', args.output)
    notes = {r['index']: r for r in reviewed}
    aux = read(Path('runs/iitb-clean-finetune-matched-v1/evaluation_suite.json'))
    write(args.output/'evaluation_suite.json', aux)
    held = HeldOutIndex([asdict(p) for split in ('val','test') for p in data.datasets[split].pairs]
                        + aux['conversation_auxiliary'])
    pairs = data.datasets['train'].pairs
    records = []
    clean_candidates = []
    exact_seen = [set(), set()]
    for index, pair in enumerate(pairs):
        lengths = tuple(len(data.tokenizer.encode(text)) + 1 for text in (pair.source, pair.target))
        label, reasons = classify(pair, lengths)
        audit_label = notes.get(index, {}).get('assessment')
        known_bad = audit_label in ('PARTIAL','MISMATCHED','CORRUPT')
        status = TIERS[label]
        keys = (comparison_key(pair.source), comparison_key(pair.target))
        duplicate = any(key in seen for key, seen in zip(keys, exact_seen))
        overlap = [f'held_out_{side}_{kind}' for i, side in enumerate(('source','target'))
                   if status != 'rejected' and (kind := held.overlaps(getattr(pair, side), i))]
        if duplicate:
            reasons += ['within_train_exact_side']
        if overlap or duplicate or audit_label in ('MISMATCHED','CORRUPT'):
            status = 'rejected'
        elif known_bad:
            status = 'uncertain'
        if known_bad:
            reasons += ['AI_review_' + audit_label.lower()]
        record = {'index': index, **asdict(pair), 'status': status, 'pre_review_status': TIERS[label],
                  'reasons': reasons + overlap, 'source_tokens': lengths[0], 'target_tokens': lengths[1],
                  'domain': domain(pair.source), 'leakage': bool(overlap), 'duplicate': duplicate, 'known_bad': known_bad}
        if status != 'rejected':
            for key, seen in zip(keys, exact_seen):
                seen.add(key)
        records.append(record)
        if status == 'high_confidence':
            clean_candidates.append(pair)
    index_by_pair = {pair: i for i, pair in enumerate(pairs)}
    for group in leakage_groups(clean_candidates, .90):
        for pair in group[1:]:
            row = records[index_by_pair[pair]]
            row.update(status='rejected', duplicate=True)
            row['reasons'].append('within_high_confidence_near_side')
    args.dataset.mkdir(parents=True, exist_ok=False)
    counts = Counter(r['status'] for r in records)
    reasons = Counter(reason for row in records if row['status']=='rejected' for reason in row['reasons'])
    fields = list(records[0])
    with ExitStack() as stack:
        writers = {}
        for tier in (*TIERS.values(), 'alignment_audit'):
            writer = csv.DictWriter(stack.enter_context((args.dataset/f'{tier}.csv').open(
                'w', encoding='utf-8', newline='')), fieldnames=fields)
            writer.writeheader()
            writers[tier] = writer
        for row in records:
            entry = {**row, 'reasons': '|'.join(row['reasons'])}
            writers[row['status']].writerow(entry)
            writers['alignment_audit'].writerow(entry)
    write(args.dataset/'rejection_reasons.json', dict(reasons))
    calibration = defaultdict(Counter)
    for row in reviewed:
        calibration[records[row['index']]['pre_review_status']][row['assessment']] += 1
    write(args.output/'review_180_summary.json', {'labels': dict(Counter(r['assessment'] for r in reviewed)),
                                                'surface_calibration': calibration, 'human_verified': 0})
    matched = select_matched(records, args.rows, args.seed)
    train_config = {'training': {'epochs': 1, 'batch_size': 8, 'seed': args.seed, 'device': 'cpu',
                                 'deterministic_algorithms': True, 'save_every_updates': 64,
                                 'optimizer': {'learning_rate': 3e-5},
                                 'scheduler': {'type': 'constant'}, 'early_stopping': {'enabled': False}}}
    (args.output/'training.yaml').write_text(yaml.safe_dump(train_config), encoding='utf-8')
    identity = read(BASE/'processed/manifest.json')
    selections = {}
    for arm, offset in [('control',0),('clean',1)]:
        destination = args.output/arm
        destination.mkdir()
        selected = [match[offset] for match in matched]
        for row in selected:
            assert not any(held.overlaps(row[side], i) for i, side in enumerate(('source','target')))
        selections[arm] = [r['index'] for r in selected]
        write_parallel_csv(destination/'train.csv', [pairs[i] for i in selections[arm]])
        for split in ('val','test'):
            shutil.copyfile(BASE/f'processed/{split}.csv', destination/f'{split}.csv')
        write(destination/'manifest.json', {**identity, **prepared_fingerprints(destination, TOKENIZER),
                                            'recovery_arm': arm, 'seed': args.seed})
    write(args.output/'selection.json', selections)
    write(args.dataset/'cleaning_manifest.json', {'rows': len(records), 'counts': dict(counts), 'seed': args.seed,
           'checkpoint_sha256': file_digest(CHECKPOINT), 'original_data_identity': identity,
           'selection_sha256': file_digest(args.output/'selection.json'),
           'files': {p.name: file_digest(p) for p in args.dataset.glob('*.csv')},
           'held_out_trigram_jaccard': 0.85, 'high_pool_near_duplicate_jaccard': 0.90})
    print(dict(counts), 'matched rows per arm:', args.rows, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=Path('runs/translation-quality-recovery-v1'))
    cli.add_argument('--dataset', type=Path, default=Path('data/quality_recovery_v1'))
    cli.add_argument('--rows', type=int, default=2048)
    cli.add_argument('--seed', type=int, default=74)
    prepare(cli.parse_args())


if __name__ == '__main__':
    main()
