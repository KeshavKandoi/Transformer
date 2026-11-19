"""Compare domain tokenizers and actual scratch-model CPU training throughput."""
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_transformer_v2 import audit
from scripts.prepare_interview_data import ROOT
from src.data.cleaning import TranslationPair
from src.data.collator import ParallelCollator
from src.data.dataset import ParallelDataset
from src.data.tokenizer import BilingualBPETokenizer
from src.model.config import ModelConfig
from src.model.transformer import EncoderDecoderTransformer
from src.training.loss import masked_cross_entropy
from src.training.utils import forward_batch


def main():
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in (ROOT/'pairs.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['split']=='train'][:640]
    modes = [('domain',ROOT/'tokenizer'),('unigram16k',Path('runs/transformer-v2-preparation/tokenizer'))]
    result = {'tokenizers':{},'model_shapes':{}}
    config = ModelConfig.load(Path('configs/interview_model.yaml'))
    for name,path in modes:
        tok = BilingualBPETokenizer.load(path)
        result['tokenizers'][name] = {'size':tok.vocab_size, **audit(tok,rows)}
        dataset = ParallelDataset([TranslationPair(r['source'],r['target']) for r in rows],tok,64,64)
        collate = ParallelCollator(tok.pad_id)
        for width in ((128,192) if name=='domain' else (128,)):
            torch.manual_seed(91)
            model = EncoderDecoderTransformer(tok.vocab_size,tok.pad_id,replace(config,d_model=width,d_ff=3*width))
            optimizer = torch.optim.AdamW(model.parameters(),lr=.0015)
            measured = 0
            for step in range(10):
                start=time.perf_counter()
                batch=collate([dataset[i] for i in range(step*48,(step+1)*48)])
                optimizer.zero_grad(set_to_none=True)
                logits,labels=forward_batch(model,batch,torch.device('cpu'))
                loss,count=masked_cross_entropy(logits,labels,tok.pad_id)
                (loss/count).backward()
                optimizer.step()
                if step>=2:
                    measured+=time.perf_counter()-start
            result['model_shapes'][f'{name}_{width}']={'parameters':sum(p.numel() for p in model.parameters()),
                'pairs_per_second':384/measured,'threads':2,
                'cumulative_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (ROOT/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['model_shapes'])


if __name__=='__main__':
    main()
