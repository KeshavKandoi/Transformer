"""Evaluate a recovery checkpoint against frozen official and conversational rows."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_finetune import TOKENIZER, derange, measure, nll, read, write
from src.inference import InferenceConfig, TransformerTranslator

PROBES = [
    'We are friends.', 'I am happy.', 'I have two sisters.', 'Can you help me?',
    'Where are you going?', 'I will come tomorrow.', "I don't like this.",
    'She is reading a book.', 'There are five students.', 'What is your name?',
]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--tokenizer-dir', type=Path, default=TOKENIZER)
    cli.add_argument('--max-new-tokens', type=int, default=80)
    cli.add_argument('--full', action='store_true', help='Also generate for all 494 official rows')
    args = cli.parse_args()
    torch.set_num_threads(2)
    translator = TransformerTranslator.from_checkpoint(args.checkpoint, args.tokenizer_dir, 'cpu')
    config = InferenceConfig(device='cpu', strategy='greedy', max_new_tokens=args.max_new_tokens)
    prior = args.output.parent / 'evaluation_suite.json'
    if not prior.exists():
        prior = Path('runs/iitb-clean-finetune-matched-v1/evaluation_suite.json')
    conditioning = next(Path('runs/iitb-125k-wide-v1').glob('meaning-audit.*/conditioning.json'))
    fixed = read(prior)
    fixed['official_full'] = read(args.output.parent / 'official_full.json')
    result = {}
    for name, rows in fixed.items():
        if name == 'official_full' and not args.full:
            result[name] = {'nll': nll(translator.model, translator.tokenizer, rows), 'rows': len(rows)}
            continue
        wrong = read(conditioning)['shuffled_sources'] if name == 'official_fixed' else derange(rows)
        result[name] = measure(translator.model, translator.tokenizer, rows, wrong, config)
    entity_path = args.output.parent / 'frozen_entity_mentions.json'
    if entity_path.exists():
        mentions = read(entity_path)
        fixed_rows = result['official_fixed']['rows']
        assert all(fixed_rows[m['row']]['source'] == m['source'] for m in mentions)
        result['official_fixed']['entity_surface'] = {
            'eligible':len(mentions),
            'matched':sum(any(alias in fixed_rows[m['row']]['prediction'] for alias in m['aliases'])
                          for m in mentions),
        }
    result['probes'] = [{'source': source, 'prediction': item.text, 'eos': item.hypothesis.reached_eos}
                        for source, item in zip(PROBES, translator.generate(PROBES, config), strict=True)]
    write(args.output, result)


if __name__ == '__main__':
    main()
