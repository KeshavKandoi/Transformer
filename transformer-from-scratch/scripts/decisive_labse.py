"""Offline LaBSE cosine scoring using the publisher's ONNX and Dense module."""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def read_dense(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q',stream.read(8))[0]
        header = json.loads(stream.read(size))
        data = stream.read()
    state = {}
    for name,info in header.items():
        if name=='__metadata__':
            continue
        assert info['dtype']=='F32'
        left,right = info['data_offsets']
        state[name] = np.frombuffer(data[left:right],dtype='<f4').reshape(info['shape'])
    return state['linear.weight'],state['linear.bias']


def score(model, texts):
    ops = ort.SessionOptions()
    ops.intra_op_num_threads = 4
    ops.inter_op_num_threads = 1
    ops.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(model/'onnx/model.onnx'),sess_options=ops,providers=['CPUExecutionProvider'])
    output_names = [item.name for item in session.get_outputs()]
    assert 'last_hidden_state' in output_names, output_names
    inputs = {item.name for item in session.get_inputs()}
    assert inputs >= {'input_ids','attention_mask'}
    tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]'),pad_token='[PAD]')
    weight,bias = read_dense(model/'2_Dense/model.safetensors')
    output = {}
    started = time.perf_counter()
    for start in range(0,len(texts),16):
        batch = texts[start:start+16]
        ids = tokenizer.encode_batch(batch)
        available = {
            'input_ids':np.asarray([e.ids for e in ids],dtype=np.int64),
            'attention_mask':np.asarray([e.attention_mask for e in ids],dtype=np.int64),
            'token_type_ids':np.asarray([e.type_ids for e in ids],dtype=np.int64),
        }
        hidden = session.run(['last_hidden_state'],{k:available[k] for k in inputs})[0][:,0,:]
        projected = np.tanh(hidden@weight.T+bias)
        normalized = projected/np.maximum(np.linalg.norm(projected,axis=1,keepdims=True),1e-12)
        output.update(zip(batch,normalized,strict=True))
    return output,time.perf_counter()-started


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    rows=json.loads(args.input.read_text())
    texts=list(dict.fromkeys(r[k] for r in rows for k in ('source','target')))
    vectors,elapsed=score(args.model,texts)
    for row in rows:
        row['labse_cosine']=float(np.dot(vectors[row['source']],vectors[row['target']]))
    args.output.write_text(json.dumps({'rows':rows,'distinct_texts':len(texts),'seconds':elapsed},
                                      ensure_ascii=False,indent=2)+'\n')
    print('LaBSE processed',len(texts),'texts in',elapsed,'seconds',flush=True)


if __name__=='__main__':
    main()
