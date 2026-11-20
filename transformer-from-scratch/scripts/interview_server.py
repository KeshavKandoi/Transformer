"""Serve the scratch model and its PyTorch attention tensors on localhost."""
import argparse
import json
import re
from collections import Counter
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.evaluate_interview_model import CHECKPOINT
from scripts.prepare_interview_data import ROOT
from scripts.prepare_conversation_v2 import ROOT as CONVERSATION_ROOT, ALIASES, SEED as CONVERSATION_SEED
from src.data.corpus import comparison_key
from src.data.prepared import file_digest
from src.data.masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask
from src.inference import InferenceConfig, TransformerTranslator
from src.training.checkpoint import load_checkpoint


class UnsupportedInput(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validation_gallery(rows):
    """Expose errors deliberately while keeping gallery selection off the test split."""
    wrong = [r for r in rows if not r['exact']]
    counts = Counter()
    wins = []
    for row in rows:
        if row['exact'] and counts[row['family']] < 3:
            wins.append(row)
            counts[row['family']] += 1
    chosen = wrong[:10] + wins[:20 - min(10, len(wrong))]
    return [{k: row[k] for k in ('source', 'target', 'prediction', 'family', 'exact')} for row in chosen]


def canonical_input(text, scope, aliases=None):
    if not isinstance(text, str) or not text.strip() or len(text) > 300:
        raise ValueError('Enter an English sentence between 1 and 300 characters.')
    text = text.strip()
    normalized = text
    if aliases is not None:
        normalized = normalized.replace('’', "'")
        expansions = {"don't":'do not', "doesn't":'does not', "i'm":'I am',
                      "you're":'you are', "we're":'we are', "they're":'they are',
                      "what's":'what is'}
        normalized = re.sub(r"\b[a-z]+['’][a-z]+\b",
                            lambda m: expansions.get(m.group().lower(), m.group()),
                            normalized, flags=re.IGNORECASE)
    key = comparison_key(normalized)
    if re.fullmatch(r"[A-Za-z\s,]+[.?!]*", normalized):
        alias = aliases.get(key) if aliases is not None else None
        match = comparison_key(alias) if alias else key
        if match in scope:
            return scope[match]
    if re.search(r'[\u0900-\u097f]', text):
        raise UnsupportedInput('input_language', 'Hindi input is not supported. Please enter English.')
    words = set(key.split())
    markers = {'kaise', 'kaisi', 'kese', 'kya', 'kyun', 'kahan', 'kahaan', 'mera', 'meri', 'mujhe',
               'namaste', 'aap', 'tum', 'main', 'ho', 'hai', 'hain', 'hoon', 'theek', 'thik', 'haal'}
    if words and words <= markers and (len(words) >= 2 or words == {'namaste'}):
        raise UnsupportedInput('romanized_hindi', 'Romanized Hindi input is not supported. Please enter English.')
    raise UnsupportedInput('out_of_domain', "This sentence is outside the model's validated translation domain.")


class InterviewService:
    def __init__(self,checkpoint=CHECKPOINT,root=ROOT):
        self.translator=TransformerTranslator.from_checkpoint(checkpoint,root/'tokenizer','cpu')
        checkpoint_data=load_checkpoint(checkpoint)
        self.lock=threading.Lock()
        inventory = root/'pairs.jsonl'
        if file_digest(inventory) != checkpoint_data['data_identity']['input_sha256']:
            raise ValueError('The grammar inventory does not match the checkpoint dataset.')
        grammar = [json.loads(line) for line in inventory.read_text(encoding='utf-8').splitlines()]
        grammar = [row for row in grammar if row['split'] != 'structure']
        self.aliases = ALIASES if root.resolve()==CONVERSATION_ROOT.resolve() else None
        self.scope = {comparison_key(row['source']):row['source'] for row in grammar}
        validation=json.loads((root/'validation_evaluation.json').read_text())
        checkpoint_sha256 = file_digest(checkpoint)
        if validation['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError('Validation results belong to a different checkpoint.')
        summary=validation['validation']
        gallery_rows=summary['rows']
        if self.aliases is not None:
            gallery_rows=sorted(gallery_rows,key=lambda row: row.get('seed')!=CONVERSATION_SEED)
        gallery=validation_gallery(gallery_rows)
        held_out=json.loads((root/'test_evaluation.json').read_text())
        if held_out['checkpoint_sha256'] != validation['checkpoint_sha256']:
            raise ValueError('Held-out results belong to a different checkpoint.')
        composition=held_out['composition']
        info_path=root/'training/scratch/metrics.jsonl'
        curve=[json.loads(line) for line in info_path.read_text().splitlines()]
        self.info={'name':str(checkpoint.relative_to(REPO) if checkpoint.is_absolute() and checkpoint.is_relative_to(REPO) else checkpoint),
                   'schema_version':2,'grammar_version':'v2' if self.aliases is not None else 'v1',
                   'checkpoint_sha256':checkpoint_sha256,
                   'tokenizer_sha256':file_digest(root/'tokenizer/tokenizer.json'),
                   'dataset_sha256':checkpoint_data['data_identity']['input_sha256'],
                   'epoch':checkpoint_data['epoch'],'updates':checkpoint_data['global_step'],
                   'parameters':sum(p.numel() for p in self.translator.model.parameters()),
                   'vocab':self.translator.tokenizer.vocab_size,'gallery':gallery,
                   'aliases':self.aliases or {},
                   'families':sorted({r['family'] for r in grammar if r['split']=='train'}),
                   'composition':{'n':composition['count'],'exact':composition['exact']},
                   'metrics':{'n':summary['count'],'exact':summary['exact'],'eos':summary['eos'],
                              'chrf':summary['chrf']['score'],'bleu':summary['bleu']['score']},
                   'curve':[{'epoch':row['epoch'],'train':row['train']['loss'],'validation':row['validation']['loss']}
                            for row in curve]}

    @torch.inference_mode()
    def predict(self,text,strategy='greedy'):
        if strategy not in ('greedy','beam','compare'):
            raise ValueError('Choose greedy, beam, or compare.')
        canonical=canonical_input(text,self.scope,self.aliases)
        tok=self.translator.tokenizer
        if len(tok.encode(canonical))+1>self.translator.model.config.max_seq_len:
            raise ValueError('This input exceeds the checkpoint positional limit.')
        models=('greedy','beam') if strategy=='compare' else (strategy,)
        outputs={}
        with self.lock:
            for method in models:
                start=time.perf_counter()
                result=self.translator.generate([canonical],InferenceConfig(strategy=method,beam_width=4,
                                                    max_new_tokens=32,device='cpu'))[0]
                outputs[method]={'text':result.text,'eos':result.hypothesis.reached_eos,
                                 'milliseconds':round((time.perf_counter()-start)*1000,1)}
                if method==models[0]:
                    attention=self.attention(result)
        return {'source':text.strip(),'canonical_source':canonical,'in_scope':True,
                'outputs':outputs,'attention_for':models[0],'attention':attention}

    def attention(self,result):
        tok=self.translator.tokenizer
        source=torch.tensor([result.source_ids])
        decoder=torch.tensor([[tok.bos_id,*result.hypothesis.token_ids[:-1]]])
        source_mask=encoder_attention_mask(source,tok.pad_id)
        decoder_mask=decoder_attention_mask(decoder,tok.pad_id)
        cross_mask=cross_attention_mask(source,decoder.size(1),tok.pad_id)
        maps=self.translator.model(source,decoder,source_mask,decoder_mask,cross_mask,return_diagnostics=True)
        def labels(ids):
            return [tok.id_to_token(i) for i in ids]
        return {'source':labels(result.source_ids),'target_input':labels(decoder[0].tolist()),
                'target_emitted':labels(result.hypothesis.token_ids),
                'encoder':maps.encoder_attention_maps[0].tolist(),'decoder':maps.decoder_self_attention_maps[0].tolist(),
                'cross':maps.decoder_cross_attention_maps[0].tolist(),
                'padding_mask':source_mask[0,0,0].tolist(),'causal_mask':decoder_mask[0,0].tolist(),
                'cross_mask':cross_mask[0,0].tolist()}


def handler(service):
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,content_type,value):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(value)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(value)

        def json(self,status,value):
            self.send(status,'application/json; charset=utf-8',json.dumps(value,ensure_ascii=False).encode())

        def do_GET(self):
            files={'/':('index.html','text/html'),'/style.css':('style.css','text/css'),
                   '/app.js':('app.js','text/javascript')}
            if self.path=='/api/info':
                self.json(200,service.info)
            elif self.path=='/api/guide':
                guide=(REPO/'docs/INTERVIEW_GUIDE.md').read_bytes()
                self.send(200,'text/plain; charset=utf-8',guide)
            elif self.path in files:
                file,mime=files[self.path]
                self.send(200,mime+'; charset=utf-8',(REPO/'interview_app'/file).read_bytes())
            else:
                self.json(404,{'error':'Route not found.'})

        def do_POST(self):
            if self.path!='/api/translate':
                self.json(404,{'error':'Route not found.'})
                return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:
                    raise ValueError('Request must contain a small JSON body.')
                value=json.loads(self.rfile.read(length))
                if not isinstance(value,dict):
                    raise ValueError('Expected a JSON object.')
                self.json(200,service.predict(value.get('text'),value.get('strategy','beam')))
            except UnsupportedInput as e:
                self.json(422,{'error':str(e),'code':e.code})
            except (ValueError,UnicodeError) as e:
                self.json(400,{'error':str(e)})

    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--version',choices=('v1','v2'),default='v2')
    args=parser.parse_args()
    torch.set_num_threads(3)
    root=CONVERSATION_ROOT if args.version=='v2' else ROOT
    service=InterviewService(root/'checkpoints/scratch/best.pt',root)
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler(service)) as server:
        print(f'Interview Transformer at http://127.0.0.1:{server.server_port}',flush=True)
        server.serve_forever()


if __name__=='__main__':
    main()
