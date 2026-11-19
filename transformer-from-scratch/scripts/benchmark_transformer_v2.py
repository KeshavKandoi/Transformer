"""Measure real forward/backward/AdamW throughput with the repository model."""
import argparse
import json
import random
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_transformer_v2 import BASE, read_rows
from src.data.cleaning import TranslationPair
from src.data.collator import ParallelCollator
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.loss import masked_cross_entropy
from src.training.utils import forward_batch


def measure(rows, tokenizer, device, positions, threads, steps):
    torch.set_num_threads(threads)
    torch.manual_seed(173)
    config = replace(ModelConfig.load(Path('configs/iitb_full_model.yaml')),max_seq_len=positions)
    model = EncoderDecoderTransformer(tokenizer.vocab_size,tokenizer.pad_id,config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0003)
    pairs = [TranslationPair(r['source'],r['target']) for r in rows[:(steps+2)*8]]
    samples = ParallelDataset(pairs,tokenizer,positions,positions)
    collate = ParallelCollator(tokenizer.pad_id)
    measurements = []
    for iteration in range(steps+2):
        started = time.perf_counter()
        batch = collate([samples[i] for i in range(iteration*8,(iteration+1)*8)])
        optimizer.zero_grad(set_to_none=True)
        logits, targets = forward_batch(model,batch,device)
        loss, count = masked_cross_entropy(logits,targets,tokenizer.pad_id)
        (loss/count).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if device.type=='mps':
            torch.mps.synchronize()
        if iteration>=2:
            measurements.append({'seconds':time.perf_counter()-started,'target_tokens':int(count),
                                 'grad_norm':norm.item(),'loss':loss.item()/count})
    seconds = sum(row['seconds'] for row in measurements)
    return {'device':str(device),'threads':threads,'positions':positions,'batch_size':8,'updates':steps,
            'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024),
            'seconds':seconds,'rows_per_second':steps*8/seconds,
            'target_tokens_per_second':sum(r['target_tokens'] for r in measurements)/seconds,
            'steps':measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokenizer')
    parser.add_argument('--sample',type=Path,default=BASE/'processed/train.csv')
    parser.add_argument('--positions',type=int,default=128)
    parser.add_argument('--threads',type=int,nargs='+',default=[2,4])
    parser.add_argument('--steps',type=int,default=12)
    args = parser.parse_args()
    if args.sample.suffix=='.json':
        rows = json.loads(args.sample.read_text())
    else:
        rows = read_rows(args.sample)
    random.Random(173).shuffle(rows)
    tokenizer = BilingualBPETokenizer.load(args.tokenizer)
    devices = [torch.device('cpu')]
    if torch.backends.mps.is_available():
        devices.append(torch.device('mps'))
    results = []
    for device in devices:
        for threads in args.threads if device.type=='cpu' else [args.threads[0]]:
            result = measure(rows,tokenizer,device,args.positions,threads,args.steps)
            results.append(result)
            print(result['device'],threads,result['rows_per_second'],flush=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
