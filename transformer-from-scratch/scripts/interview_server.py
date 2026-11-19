"""Serve the scratch model and its PyTorch attention tensors on localhost."""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.evaluate_interview_model import CHECKPOINT
from scripts.prepare_interview_data import ROOT, generate
from src.data.corpus import comparison_key
from src.data.prepared import file_digest
from src.data.masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask
from src.inference import InferenceConfig, TransformerTranslator
from src.training.checkpoint import load_checkpoint


class InterviewService:
    def __init__(self,checkpoint=CHECKPOINT,root=ROOT):
        self.translator=TransformerTranslator.from_checkpoint(checkpoint,root/'tokenizer','cpu')
        checkpoint_data=load_checkpoint(checkpoint)
        self.lock=threading.Lock()
        self.scope={comparison_key(row['source']) for row in generate() if row['split']!='structure'}
        validation=json.loads((root/'validation_evaluation.json').read_text())
        if validation['checkpoint_sha256'] != file_digest(checkpoint):
            raise ValueError('Validation results belong to a different checkpoint.')
        summary=validation['validation']
        gallery=json.loads((root/'gallery.json').read_text())
        info_path=root/'training/scratch/metrics.jsonl'
        curve=[json.loads(line) for line in info_path.read_text().splitlines()]
        self.info={'name':str(checkpoint.relative_to(REPO) if checkpoint.is_absolute() and checkpoint.is_relative_to(REPO) else checkpoint),
                   'epoch':checkpoint_data['epoch'],'updates':checkpoint_data['global_step'],
                   'parameters':sum(p.numel() for p in self.translator.model.parameters()),
                   'vocab':self.translator.tokenizer.vocab_size,'gallery':gallery,
                   'metrics':{'n':summary['count'],'exact':summary['exact'],'eos':summary['eos'],
                              'chrf':summary['chrf']['score'],'bleu':summary['bleu']['score']},
                   'curve':[{'epoch':row['epoch'],'train':row['train']['loss'],'validation':row['validation']['loss']}
                            for row in curve]}

    @torch.inference_mode()
    def predict(self,text,strategy='greedy'):
        if not isinstance(text,str) or not text.strip() or len(text)>300 or not any(c.isascii() and c.isalpha() for c in text):
            raise ValueError('Enter an English sentence between 1 and 300 characters.')
        if strategy not in ('greedy','beam','compare'):
            raise ValueError('Choose greedy, beam, or compare.')
        text=text.strip()
        tok=self.translator.tokenizer
        if len(tok.encode(text))+1>self.translator.model.config.max_seq_len:
            raise ValueError('This input exceeds the checkpoint positional limit.')
        models=('greedy','beam') if strategy=='compare' else (strategy,)
        outputs={}
        with self.lock:
            for method in models:
                start=time.perf_counter()
                result=self.translator.generate([text],InferenceConfig(strategy=method,beam_width=4,
                                                    max_new_tokens=32,device='cpu'))[0]
                outputs[method]={'text':result.text,'eos':result.hypothesis.reached_eos,
                                 'milliseconds':round((time.perf_counter()-start)*1000,1)}
                if method==models[0]:
                    attention=self.attention(result)
        return {'source':text,'in_scope':comparison_key(text) in self.scope,
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
                self.json(200,service.predict(value.get('text'),value.get('strategy','greedy')))
            except (ValueError,UnicodeError) as e:
                self.json(400,{'error':str(e)})

    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    torch.set_num_threads(3)
    service=InterviewService()
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler(service)) as server:
        print(f'Interview Transformer at http://127.0.0.1:{server.server_port}',flush=True)
        server.serve_forever()


if __name__=='__main__':
    main()
