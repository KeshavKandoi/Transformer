"""Generate and fingerprint a constrained English–Hindi experiment."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cleaning import TranslationPair, write_parallel_csv
from src.data.config import DataConfig
from src.data.corpus import comparison_key
from src.data.prepared import write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer

ROOT = Path('experiments/interview-transformer-v1')
SEED = 91
# English, Hindi, grammatical agreement, copula, possessive, genitive/dative experiencer.
SUBJECTS = [
    ('I','मैं','m','हूँ','मेरे','मुझे'),
    ('you','तुम','p','हो','तुम्हारे','तुम्हें'),
    ('he','वह','m','है','उसके','उसे'),
    ('she','वह','f','है','उसके','उसे'),
    ('we','हम','p','हैं','हमारे','हमें'),
    ('they','वे','p','हैं','उनके','उन्हें'),
]
NOUNS = [
    ('book','books','किताब','किताबें','f'), ('pen','pens','कलम','कलमें','f'),
    ('apple','apples','सेब','सेब','m'), ('banana','bananas','केला','केले','m'),
    ('bottle','bottles','बोतल','बोतलें','f'), ('key','keys','चाबी','चाबियाँ','f'),
    ('bag','bags','बैग','बैग','m'), ('notebook','notebooks','कॉपी','कॉपियाँ','f'),
    ('cup','cups','कप','कप','m'), ('ticket','tickets','टिकट','टिकट','m'),
    ('picture','pictures','तस्वीर','तस्वीरें','f'), ('ball','balls','गेंद','गेंदें','f'),
    ('toy','toys','खिलौना','खिलौने','m'), ('chair','chairs','कुर्सी','कुर्सियाँ','f'),
]
QUANTITIES = ('one two three four five six seven eight nine ten eleven twelve'.split(),
              'एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह'.split())
TRAITS = [('happy','खुश'),('sad','उदास'),('ready','तैयार'),('busy','व्यस्त'),
          ('quiet','शांत'),('safe','सुरक्षित'),('tired',('थका','थकी','थके'))]
LIKES = [('tea','चाय'),('coffee','कॉफ़ी'),('music','संगीत'),('cricket','क्रिकेट'),
         ('rice','चावल'),('milk','दूध'),('juice','रस'),('football','फ़ुटबॉल'),
         ('yoga','योग'),('hindi','हिंदी'),('chocolate','चॉकलेट'),('soup','सूप')]
PLACES = [('home','घर'),('to school','स्कूल'),('to the market','बाज़ार'),
          ('to the park','पार्क'),('to the office','दफ़्तर'),('to the hospital','अस्पताल'),
          ('to the station','स्टेशन'),('to the library','पुस्तकालय'),('to the shop','दुकान')]
VERBS = [
    ('buy','buying','खरीद',set(range(14))),
    ('find','finding','ढूँढ़',set(range(14))),
    ('carry','carrying','ले जा',set(range(14))),
    ('see','seeing','देख',set(range(14))),
    ('bring','bringing','ला',set(range(14))),
    ('read','reading','पढ़',{0,7}),
    ('eat','eating','खा',{2,3}),
]


def bucket(*parts):
    text = json.dumps((SEED,*parts),ensure_ascii=False)
    return int(hashlib.sha256(text.encode()).hexdigest()[:12],16)%100


def generate(subjects=SUBJECTS, nouns=NOUNS, verbs=VERBS, traits=TRAITS):
    records = []
    sources, targets = set(), set()

    def add(en, hi, family, group, lexical=None, ood=False, **slots):
        en = en[0].upper()+en[1:]
        hi += '?' if en.endswith('?') else '।'
        src, tgt = comparison_key(en), comparison_key(hi)
        if src in sources or tgt in targets:
            return
        sources.add(src)
        targets.add(tgt)
        group_family = {'state-negative':'state','state-question':'state',
                        'preference-negative':'preference','preference-question':'preference',
                        'action-negative':'action-now','movement-past':'movement-now',
                        'movement-future':'movement-now'}.get(family,family)
        code = bucket(group_family,group)
        split = ('structure' if ood else 'lexical' if lexical is not None and bucket(*lexical)<14
                 else 'composition' if code<15 else 'validation' if code<25 else 'train')
        records.append(dict(source=en,target=hi,family=family,group=group,group_family=group_family,
                            lexical_group=lexical,split=split,slots=slots,
                            origin='hand-authored grammar; programmatic; no independent review',
                            seed=SEED,pair_id=hashlib.sha256((en+'\n'+hi).encode()).hexdigest()[:20]))

    for en,hi,gender,copula,owner,experiencer in subjects:
        be = 'am' if en=='I' else 'is' if en in ('he','she') else 'are'
        third = en in ('he','she')
        index = 'mfp'.index(gender)
        progress = ('रहा','रही','रहे')[index]
        future = {'I':'ऊँगा','you':'ओगे','he':'ेगा','she':'ेगी','we':'ेंगे','they':'ेंगे'}[en]
        for state,word in traits:
            adjective = word[index] if isinstance(word,tuple) else word
            add(f'{en} {be} {state}.',f'{hi} {adjective} {copula}','state',[en,state],subject=en,adjective=state)
            add(f'{en} {be} not {state}.',f'{hi} {adjective} नहीं {copula}','state-negative',
                [en,state],subject=en,adjective=state,negative=True)
            add(f'{be} {en if en=="I" else en.lower()} {state}?',f'क्या {hi} {adjective} {copula}',
                'state-question',[en,state],subject=en,adjective=state,question=True)
        for eng,hindi in LIKES:
            like = 'likes' if third else 'like'
            add(f'{en} {like} {eng}.',f'{experiencer} {hindi} पसंद है','preference',[en,eng],
                subject=en,object=eng)
            neg = 'does' if third else 'do'
            add(f'{en} {neg} not like {eng}.',f'{experiencer} {hindi} पसंद नहीं है','preference-negative',
                [en,eng],subject=en,object=eng,negative=True)
            add(f'{neg} {en} like {eng}?',f'क्या {experiencer} {hindi} पसंद है','preference-question',
                [en,eng],subject=en,object=eng,question=True)
        for eng,hindi in PLACES:
            past = ('गया','गई','गए')[index]
            if en=='you':
                past += ' थे'
            yesterday = 'कल'
            add(f'{en} {be} going {eng}.',f'{hi} {hindi} जा {progress} {copula}','movement-now',
                [en,eng],subject=en,place=eng,tense='progressive')
            add(f'{en} went {eng} yesterday.', f'{hi} {yesterday} {hindi} {past}','movement-past',
                [en,eng],subject=en,place=eng,tense='past')
            stem = 'जाऊँगा' if en=='I' else {'you':'जाओगे','he':'जाएगा','she':'जाएगी','we':'जाएँगे','they':'जाएँगे'}[en]
            add(f'{en} will go {eng} tomorrow.',f'{hi} कल {hindi} {stem}','movement-future',
                [en,eng],subject=en,place=eng,tense='future')
        for n,(sing,plural,hs,hp,ng) in enumerate(nouns):
            for q,(eq,hq) in enumerate(zip(*QUANTITIES),1):
                object_en = sing if q==1 else plural
                object_hi = hs if q==1 else hp
                have = 'has' if third else 'have'
                amount = f'{eq} {object_en}'
                quant = f'{hq} {object_hi}'
                add(f'{en} {have} {amount}.',f'{owner} पास {quant} '+('है' if q==1 else 'हैं'),
                    'possession',[en,sing,q],('quantity',sing,q),subject=en,object=sing,number=q)
                for verb,ing,stem,compatibility in verbs:
                    if n not in compatibility:
                        continue
                    metadata = dict(subject=en,object=sing,number=q,verb=verb,agreement=gender)
                    base = [en,verb,sing,q]
                    lex = ('verb-object',verb,sing)
                    add(f'{en} {be} {ing} {amount}.',f'{hi} {quant} {stem} {progress} {copula}',
                        'action-now',base,lex,**metadata)
                    add(f'{en} {be} not {ing} {amount}.',f'{hi} {quant} नहीं {stem} {progress} {copula}',
                        'action-negative',base,lex,negative=True,**metadata)
                    # Questions with this word order are an entirely withheld family.
                    if q in (2,5) and verb in ('buy','bring'):
                        add(f'{be} {en} {ing} {amount}?',f'क्या {hi} {quant} {stem} {progress} {copula}',
                            'action-question',base,ood=True,question=True,**metadata)
    for sing,plural,hs,hp,_ in nouns:
        for q,(eq,hq) in enumerate(zip(*QUANTITIES),1):
            add(f'please bring {eq} {sing if q==1 else plural}.',
                f'कृपया {hq} {hs if q==1 else hp} लाओ','request',[sing,q],
                subject='you',object=sing,verb='bring',number=q)
    for name,hindi in [('Anita','अनीता'),('Ravi','रवि'),('Meera','मीरा'),('Asha','आशा'),
                       ('Amit','अमित'),('Neha','नेहा'),('Riya','रिया'),('Priya','प्रिया'),
                       ('Kavita','कविता'),('Rahul','राहुल'),('Kiran','किरण'),('Tara','तारा')]:
        add(f'my name is {name}.',f'मेरा नाम {hindi} है','introduction',[name],name=name)
    for label,head in [('brother','भाई'),('sister','बहन'),('friend','दोस्त')]:
        en = 'she' if label=='sister' else 'he'
        possessive = 'मेरी' if label=='sister' else 'मेरा'
        add(f'{en} is my {label}.',f'वह {possessive} {head} है','relationship',[label],object=label)
    add('we are friends.','हम दोस्त हैं','relationship',['friends'],object='friends')
    # These standalone questions are a second held structure and never demo defaults.
    for en,hi,gender,copula,*_ in subjects:
        be = 'am' if en=='I' else 'is' if en in ('he','she') else 'are'
        add(f'where {be} {en} going?',f'{hi} कहाँ जा '+('रहा','रही','रहे')['mfp'.index(gender)]+f' {copula}',
            'location-question',[en],ood=True,subject=en,question=True)
    return records


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    rows = generate()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    counts = Counter(r['split'] for r in rows)
    training = [r for r in rows if r['split']=='train']
    tokenizer = BilingualBPETokenizer.train((r[k] for r in training for k in ('source','target')),2000)
    tokenizer.save(root/'tokenizer')
    maximum = max(len(tokenizer.encode(r[k]))+1 for r in rows for k in ('source','target'))
    assert maximum<=64
    data = DataConfig(seed=SEED,vocab_size=2000,max_source_length=64,max_target_length=64)
    (root/'data.yaml').write_text(yaml.safe_dump(asdict(data)))
    manifest_input = root/'pairs.jsonl'
    manifest_input.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for standard,split in [('train','train'),('val','validation'),('test','composition')]:
        write_parallel_csv(root/'processed'/f'{standard}.csv',
                           [TranslationPair(r['source'],r['target']) for r in rows if r['split']==split])
    write_preparation_manifest(root/'processed',root/'tokenizer',manifest_input,data,
                                corpus={'origin':'programmatic bilingual grammar', 'seed':SEED})
    (root/'acceptance.json').write_text(json.dumps({'composition_exact':.80,'composition_eos':.95,
        'composition_number':.90,'composition_negation':.90},indent=2)+'\n')
    (root/'dataset_summary.json').write_text(json.dumps({'splits':counts,'longest_tokens':maximum,
        'vocabulary':tokenizer.vocab_size,'families':sorted({r['family'] for r in rows})},indent=2)+'\n')
    print(dict(counts),tokenizer.vocab_size,maximum)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    prepare(parser.parse_args().output)
