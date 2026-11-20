"""Evaluate frozen conversational splits and preserve actual predictions."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.clean_finetune import nll
from scripts.evaluate_translations import prediction_flags, score_translations
from scripts.prepare_interview_data import ROOT, LIKES, NOUNS, PLACES, QUANTITIES, SUBJECTS, TRAITS, VERBS
from src.data.corpus import comparison_key
from src.data.prepared import file_digest
from src.inference import InferenceConfig, TransformerTranslator

CHECKPOINT = ROOT/'checkpoints/scratch/best.pt'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def words(text):
    return set(comparison_key(text).split())


def slot_checks(row,text):
    metadata=row['slots']
    family=row['family']
    tokens=words(text)
    expected=words(row['target'])
    checks={}
    if 'number' in metadata:
        quantities=set(QUANTITIES[1])
        checks['number']=tokens&quantities==expected&quantities
    if family in ('action-now','action-negative','state','state-negative','preference','preference-negative','affection-love','affection-miss'):
        checks['negation']=('नहीं' in tokens)==('नहीं' in expected)
    if family.startswith('action') or family.startswith('movement-now') or family=='location-question':
        checked={'रहा','रही','रहे'}
        checks['agreement']=tokens&checked==expected&checked
    if family in ('affection-love','modal-request'):
        markers={'करता','करती','करते','सकता','सकती','सकते'}
        checks['agreement']=tokens&markers==expected&markers
    if family.startswith('arrival-'):
        markers={'आऊँगा','आओगे','आएगा','आएगी','आएँगे','आया','आई','आए'}
        checks['tense']=bool(tokens&expected&markers)
    if family.startswith('movement-'):
        anchors={'movement-past':{'गया','गई','गए'},'movement-now':{'रहा','रही','रहे'},
                 'movement-future':{'जाऊँगा','जाओगे','जाएगा','जाएगी','जाएँगे'}}[family]
        checks['tense']=bool(tokens & expected & anchors)
    if family in ('action-now','action-negative','action-question','request'):
        stem=next(v[2] for v in VERBS if v[0]==metadata['verb'])
        if family=='request':
            stem='लाओ'
        checks['verb']=all(w in tokens for w in stem.split())
    subject=next((r for r in SUBJECTS if r[0]==metadata.get('subject')),None)
    if subject is not None and family not in ('request','family-count'):
        marker=subject[4] if family=='possession' else subject[5] if (
            family.startswith('preference') or family=='affection-miss') else subject[1]
        checks['subject']=marker in tokens
    if 'object' in metadata and 'number' in metadata:
        noun=next(n for n in NOUNS if n[0]==metadata['object'])
        checks['object']=noun[2 if metadata['number']==1 else 3] in tokens
    elif family.startswith('preference'):
        checks['object']=next(h for e,h in LIKES if e==metadata['object']) in tokens
    if metadata.get('place'):
        checks['place']=next(h for e,h in PLACES if e==metadata['place']) in tokens
    if 'name' in metadata:
        hname=metadata.get('hindi_name')
        if hname or family=='introduction':
            hname=hname or row['target'].removeprefix('मेरा नाम ').removesuffix(' है।')
            checks['name']=hname in tokens
    if family=='family-count':
        relatives={'बहन','बहनें','भाई','दोस्त','बच्चा','बच्चे'}
        checks['relationship']=bool(tokens&expected&relatives)
        checks['agreement']=('हैं' in tokens)==('हैं' in expected) and ('है' in tokens)==('है' in expected)
    if family.endswith('question'):
        checks['question']='?' in text
    if family.startswith('state'):
        adj=next(h for e,h in TRAITS if e==metadata['adjective'])
        if isinstance(adj,tuple):
            desired=expected & set(adj)
            checks['adjective']=bool(desired & tokens)
            checks['agreement']=bool(desired & tokens)
        else:
            checks['adjective']=adj in tokens
    return checks


def evaluate(translator,rows,domain,known_vocab):
    result=[]
    for start in range(0,len(rows),48):
        batch=rows[start:start+48]
        predictions=translator.generate([r['source'] for r in batch],InferenceConfig(max_new_tokens=32,device='cpu'))
        for row,prediction in zip(batch,predictions,strict=True):
            text=prediction.text
            result.append({**row,'prediction':text,'eos':prediction.hypothesis.reached_eos,
                'exact':text.strip()==row['target'],'repetition':prediction_flags(text)['repetition'],
                'english_word_unseen':sorted(words(row['source'])-known_vocab),
                'checks':slot_checks(row,text) if domain else {}})
    counts={}
    for key in sorted({k for r in result for k in r['checks']}):
        values=[r['checks'][key] for r in result if key in r['checks']]
        counts[key]={'correct':sum(values),'n':len(values),'rate':sum(values)/len(values)}
    mean=lambda key:sum(r[key] for r in result)/len(result)
    lexical_groups={}
    for unseen in (False,True):
        chosen=[r for r in result if bool(r['english_word_unseen'])==unseen]
        lexical_groups['unseen_forms' if unseen else 'seen_forms']={
            'count':len(chosen),'exact':sum(r['exact'] for r in chosen)}
    return {**score_translations([r['prediction'] for r in result],[r['target'] for r in result]),
            'count':len(result),'exact':mean('exact'),'eos':mean('eos'),'repetition':mean('repetition'),
            'word_form_groups':lexical_groups,'slots':counts,
            'all_measured_slots':sum(all(r['checks'].values()) for r in result if r['checks'])/len(result),
            'rows':result}


def source_probe(translator,rows):
    rng=random.Random(194)
    chosen=rng.sample(rows,min(len(rows),128))
    rotated=chosen[1:]+chosen[:1]
    wrong=[{**row,'source':foil['source']} for row,foil in zip(chosen,rotated,strict=True)]
    proper=nll(translator.model,translator.tokenizer,chosen)
    swapped=nll(translator.model,translator.tokenizer,wrong)
    return {'correct_nll':proper,'wrong_nll':swapped,'advantage':swapped-proper,
            'count':len(chosen),'pair_ids':[r['pair_id'] for r in chosen]}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--validation-only',action='store_true')
    cli.add_argument('--experiment',type=Path,default=ROOT)
    cli.add_argument('--output-dir',type=Path,help='Write evaluation artifacts outside the experiment directory')
    cli.add_argument('--domain-only',action='store_true',help='Skip optional local third-party probes')
    args=cli.parse_args()
    output=args.output_dir or args.experiment
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    root=args.experiment
    checkpoint=root/'checkpoints/scratch/best.pt'
    translator=TransformerTranslator.from_checkpoint(checkpoint,root/'tokenizer','cpu')
    rows=[json.loads(line) for line in (root/'pairs.jsonl').read_text().splitlines()]
    train_vocab=set().union(*(words(r['source']) for r in rows if r['split']=='train'))
    names=['validation'] if args.validation_only else ['composition','lexical','structure']
    result={}
    for name in names:
        selected=[r for r in rows if r['split']==name]
        result[name]=evaluate(translator,selected,True,train_vocab)
        result[name]['source_probe']=source_probe(translator,selected)
        print(name,result[name]['exact'],result[name]['chrf']['score'],flush=True)
    if not args.validation_only:
        suite=Path('runs/transformer-v2-decisive-data-v1/evaluation_suite.json')
        if not args.domain_only and suite.exists():
            for name,choices in json.loads(suite.read_text()).items():
                general=[r for r in choices if len(translator.tokenizer.encode(r['source']))<64]
                if general:
                    result[name]=evaluate(translator,general[:24],False,train_vocab)
    else:
        seen=Counter()
        gallery=[]
        for row in result['validation']['rows']:
            family=row['family']
            if seen[family]<1:
                gallery.append({k:row[k] for k in ('source','target','prediction','family','exact')})
                seen[family]+=1
        write(output/'gallery.json',gallery)
    result['checkpoint_sha256']=file_digest(checkpoint)
    write(output/('validation_evaluation.json' if args.validation_only else 'test_evaluation.json'),result)


if __name__=='__main__':
    main()
