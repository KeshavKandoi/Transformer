"""Screen local IITB candidates and run a bounded warm-start comparison."""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clean_finetune_data import HeldOutIndex, digits, domain, negated, screen
from scripts.evaluate_translations import prediction_flags, score_translations
from src.data.cleaning import TranslationPair
from src.data.collator import ParallelCollator
from src.data.config import DataConfig
from src.data.dataset import ParallelDataset
from src.data.prepared import file_digest
from src.inference import InferenceConfig, TransformerTranslator
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.checkpoint import CheckpointManager, load_checkpoint, validate_checkpoint
from src.training.config import SchedulerConfig
from src.training.scheduler import build_scheduler
from src.training.utils import capture_rng, forward_batch, load_prepared_data

CPU = torch.device('cpu')
BASE = ROOT / 'experiments/iitb-125k-wide-v1'
TOKENIZER = BASE / 'tokenizer'
CHECKPOINT = ROOT / 'checkpoints/iitb-125k-wide-v1/best.pt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rows_for_pairs(pairs):
    return [asdict(pair) for pair in pairs]


def screen_corpus(data, held, destination):
    counts = Counter()
    reasons_by_status = {}
    with destination.open('w', encoding='utf-8') as stream:
        for index, pair in enumerate(data.datasets['train'].pairs):
            status, reasons = screen(pair.source, pair.target)
            normalized_overlap = []
            for side, value in enumerate((pair.source, pair.target)):
                if status != 'rejected':
                    overlap = held.overlaps(value, side)
                    if overlap:
                        normalized_overlap.append(f'held_out_{overlap}_{"source" if side == 0 else "target"}')
            if normalized_overlap:
                status, reasons = 'rejected', reasons + normalized_overlap
            row_domain = domain(pair.source)
            counts[status, row_domain] += 1
            reasons_by_status.setdefault(status, Counter()).update(reasons)
            record = {'index': index, **asdict(pair), 'status': status, 'reasons': reasons,
                      'domain_proxy': row_domain, 'bilingual_verified': False}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return {'counts': dict(Counter({status: sum(n for (label, _), n in counts.items() if label == status)
                                    for status, _ in counts})),
            'by_domain': {status: {key: n for (label, key), n in counts.items() if label == status}
                          for status, _ in counts},
            'reasons': reasons_by_status, 'independently_verified': 0}


def smoke_arms(data, prior, held, fraction):
    original, edited = (read(prior / 'ab' / f'{name}.json') for name in ('control', 'corrected'))
    patch = []
    shared = []
    for before, after in zip(original, edited, strict=True):
        index = before['index']
        assert before['source'] == after['source'] and index == after['index']
        assert asdict(data.datasets['train'].pairs[index]) == {key: before[key] for key in ('source', 'target')}
        if any(held.overlaps(row[side], i) for row in (before, after)
               for i, side in enumerate(('source', 'target'))):
            continue
        (shared if before['target'] == after['target'] else patch).append((before, after))
    count = round(4 * fraction)
    if not patch or (count and not shared):
        raise ValueError('Smoke data has no eligible corrected or shared pairs')
    return patch, shared, count


def make_batches(patch, shared, shared_count, updates, seed):
    rng = random.Random(seed)
    batches = []
    for _ in range(updates):
        rows = rng.sample(patch, 4 - shared_count) + rng.sample(shared, shared_count)
        rng.shuffle(rows)
        batches.append(rows)
    return batches


def batch_for(rows, tokenizer, max_length=128):
    pairs = [TranslationPair(row['source'], row['target']) for row in rows]
    dataset = ParallelDataset(pairs, tokenizer, max_length, max_length)
    return ParallelCollator(tokenizer.pad_id)([dataset[i] for i in range(len(dataset))])


def model_from(checkpoint, tokenizer):
    config = ModelConfig(**checkpoint['model_config'])
    model = EncoderDecoderTransformer(tokenizer.vocab_size, tokenizer.pad_id, config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert all(torch.equal(value, checkpoint['model_state_dict'][name]) for name, value in model.state_dict().items())
    return model


@torch.inference_mode()
def nll(model, tokenizer, rows):
    model.eval()
    losses = 0.0
    tokens = 0
    for start in range(0, len(rows), 8):
        logits, target = forward_batch(model, batch_for(rows[start:start + 8], tokenizer, model.config.max_seq_len), CPU)
        losses += F.cross_entropy(logits.transpose(1, 2), target, ignore_index=tokenizer.pad_id, reduction='sum').item()
        tokens += int(target.ne(tokenizer.pad_id).sum())
    return losses / tokens


def train(model, tokenizer, batches, lr, seed):
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = build_scheduler(optimizer, SchedulerConfig(type='constant'))
    history = []
    for rows in batches:
        optimizer.zero_grad(set_to_none=True)
        logits, targets = forward_batch(model, batch_for(rows, tokenizer), CPU)
        loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=tokenizer.pad_id)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite fine-tuning loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        history.append({'nll': loss.item(), 'gradient_norm': norm.item(), 'lr': optimizer.param_groups[0]['lr']})
        optimizer.step()
        scheduler.step()
    as_steps = {int(state['step'].item()) for state in optimizer.state.values()}
    assert as_steps == {len(batches)} and scheduler.last_epoch == len(batches)
    return optimizer, scheduler, history


def measure(model, tokenizer, rows, mismatch, config):
    shuffled = [{**row, 'source': wrong} for row, wrong in zip(rows, mismatch, strict=True)]
    correct_nll = nll(model, tokenizer, rows)
    shuffled_nll = nll(model, tokenizer, shuffled)
    translator = TransformerTranslator(model, tokenizer, CPU)
    output = []
    for start in range(0, len(rows), 12):
        chunk = rows[start:start + 12]
        predictions = translator.generate([row['source'] for row in chunk], config)
        for row, prediction in zip(chunk, predictions, strict=True):
            reference = row['target']
            text = prediction.text
            output.append({**row, 'prediction': text, 'eos': prediction.hypothesis.reached_eos,
                           'flags': prediction_flags(text),
                           'digits_preserved': digits(text) == digits(reference) if digits(reference) else None,
                           'negation_retained': negated(text, 'hi') if negated(reference, 'hi') else None})
    return {**score_translations([r['prediction'] for r in output], [r['target'] for r in output]),
            'nll': correct_nll, 'shuffled_nll': shuffled_nll, 'source_advantage': shuffled_nll - correct_nll,
            'counts': {'rows': len(output), 'eos': sum(r['eos'] for r in output),
                       **{key: sum(r['flags'][key] for r in output) for key in ('empty', 'repetition')}},
            **{key: {'correct': sum(r[key] is True for r in output),
                     'eligible': sum(r[key] is not None for r in output)}
               for key in ('digits_preserved', 'negation_retained')},
            'rows': output}


def derange(rows):
    order = [row['source'] for row in rows]
    return next(order[i:] + order[:i] for i in range(1, len(rows))
                if all(a != b for a, b in zip(order, order[i:] + order[:i])))


def save_checkpoint(out, name, source, model, optimizer, scheduler, history, settings, manifest, val_loss):
    payload = {key: source[key] for key in ('model_config', 'tokenizer_metadata')}
    payload.update(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                   scheduler_state_dict=scheduler.state_dict(), epoch=0, global_step=len(history),
                   best_validation_loss=val_loss, early_stopping_state={}, rng_state=capture_rng(CPU),
                   training_config=settings, data_identity=manifest, epoch_metrics={'updates': history})
    manager = CheckpointManager(out / name)
    path = manager.save('terminal.pt', payload)
    saved = load_checkpoint(path)
    validate_checkpoint(saved, payload['model_config'], payload['tokenizer_metadata'], manifest)
    return str(path)


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = load_prepared_data(BASE / 'processed', TOKENIZER, DataConfig.load(ROOT / 'configs/experiment_data.yaml'))
    base = load_checkpoint(CHECKPOINT)
    tokenizer = data.tokenizer
    validate_checkpoint(base, asdict(ModelConfig.load(ROOT / 'configs/iitb_full_model.yaml')),
                        data.tokenizer_metadata, data.data_identity)
    args.output.mkdir(parents=True, exist_ok=False)
    backup = args.output / 'original-best.backup.pt'
    shutil.copy2(CHECKPOINT, backup)
    parent_hash = file_digest(CHECKPOINT)
    assert file_digest(backup) == parent_hash
    prior = args.prior
    fixed = read(prior / 'ab/fixed_validation.json')
    assert len(fixed) == 100 and len({(r['source'], r['target']) for r in fixed}) == 100
    assert set(TranslationPair(r['source'], r['target']) for r in fixed) <= set(data.datasets['val'].pairs)
    conversational = read(prior / 'ab/fixed_conversation.json')
    literal = [{**r, 'source': 'We are friends'} for r in conversational if r['source'] == 'We are friends.']
    assert len(literal) == 1
    splits = {'official_fixed': fixed, 'conversation_auxiliary': conversational}
    write(args.output / 'evaluation_suite.json', splits)
    held = HeldOutIndex(rows_for_pairs(data.datasets['val'].pairs)
                        + rows_for_pairs(data.datasets['test'].pairs) + conversational)
    screening = screen_corpus(data, held, args.output / 'candidate_review.jsonl')
    write(args.output / 'screening.json', screening)
    print('Screening complete:', screening['counts'], flush=True)
    patch, shared, shared_count = smoke_arms(data, prior, held, args.shared_fraction)
    batches = make_batches(patch, shared, shared_count, args.updates, args.seed)
    arm_rows = {name: [[pair[i] for pair in batch] for batch in batches]
                for i, name in enumerate(('a_matched', 'b_corrected'))}
    assert all([r['source'] for r in a] == [r['source'] for r in b]
               for a, b in zip(*arm_rows.values(), strict=True))
    write(args.output / 'smoke_batches.json', arm_rows)
    manifest = {'parent_checkpoint_sha256': parent_hash,
                'smoke_batches_sha256': file_digest(args.output / 'smoke_batches.json'),
                'evaluation_sha256': file_digest(args.output / 'evaluation_suite.json'),
                'screening_sha256': file_digest(args.output / 'candidate_review.jsonl'),
                'eligible_target_corrections': len(patch), 'shared_rows_per_batch': shared_count}
    write(args.output / 'manifest.json', manifest)
    diagnostic = fixed[:8]
    trials = {}
    starting_model = model_from(base, tokenizer)
    starting_nll = nll(starting_model, tokenizer, diagnostic)
    for lr in (1e-5, 3e-5):
        trial_model = model_from(base, tokenizer)
        _, _, history = train(trial_model, tokenizer, arm_rows['a_matched'][:2], lr, args.seed)
        trials[str(lr)] = {'updates': history, 'validation_nll': nll(trial_model, tokenizer, diagnostic)}
    lr = 3e-5 if trials['3e-05']['validation_nll'] <= starting_nll + 0.05 else 1e-5
    write(args.output / 'stability.json', {'before': starting_nll, 'trials': trials, 'selected_lr': lr})
    config = InferenceConfig(device='cpu', strategy='greedy', max_new_tokens=80)
    settings = {'seed': args.seed, 'optimizer': 'AdamW (0.9, 0.98), eps 1e-9', 'lr': lr,
                'scheduler': 'constant', 'updates': args.updates, 'shared_fraction': args.shared_fraction,
                'max_grad_norm': 1.0, 'generation': asdict(config)}
    old_shuffled = read(prior / 'conditioning.json')['shuffled_sources']
    assert [(r['source'], r['target']) for r in fixed] == list(map(tuple, read(prior / 'conditioning.json')['pairs']))
    for name in ('original', *arm_rows):
        model = model_from(base, tokenizer)
        history = []
        if name in arm_rows:
            optimizer, scheduler, history = train(model, tokenizer, arm_rows[name], lr, args.seed)
            assert any(not torch.equal(value, base['model_state_dict'][key])
                       for key, value in model.named_parameters())
        scores = {key: measure(model, tokenizer, rows, old_shuffled if key == 'official_fixed' else derange(rows), config)
                  for key, rows in splits.items()}
        translation = TransformerTranslator(model, tokenizer, CPU).generate(['We are friends'], config)[0]
        result = {'scores': scores, 'history': history, 'literal_friends': translation.text,
                  'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if history:
            result['checkpoint'] = save_checkpoint(args.output, name, base, model, optimizer, scheduler, history,
                                                   settings, {**manifest, 'arm': name}, scores['official_fixed']['nll'])
        write(args.output / f'{name}.json', result)
        print('Evaluated:', name, flush=True)
    assert file_digest(CHECKPOINT) == file_digest(backup) == parent_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True, help='Prior local 200-pair audit and fixed suites')
    parser.add_argument('--output', type=Path, required=True, help='Fresh local artifact directory')
    parser.add_argument('--shared-fraction', type=float, default=0.25, help='Shared examples per four-pair batch')
    parser.add_argument('--updates', type=int, default=12, choices=range(2, 33), metavar='2..32')
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.shared_fraction not in (0, 0.25, 0.5, 0.75) or args.threads < 1:
        parser.error('Use 0, .25, .5 or .75 shared fraction and at least one thread')
    run(args)


if __name__ == '__main__':
    main()
