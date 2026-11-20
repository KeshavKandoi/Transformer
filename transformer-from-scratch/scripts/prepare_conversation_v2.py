"""Generate the scoped conversational extension. No translation is served by this grammar."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_interview_data import SUBJECTS, PLACES, QUANTITIES, bucket, generate
from src.data.cleaning import TranslationPair, write_parallel_csv
from src.data.config import DataConfig
from src.data.corpus import comparison_key
from src.data.prepared import write_preparation_manifest
from src.data.tokenizer import BilingualBPETokenizer

ROOT = Path('experiments/interview-conversation-v2')
SEED = 187
PROMPTS = (
    'I love you.', 'I miss you.', 'How are you?', 'What is your name?', 'My name is Ravi.',
    'We are friends.', 'You are my friend.', 'I am happy.', 'I am sad.', 'I have two sisters.',
    'I like tea.', "I don't like coffee.", 'Can you help me?', 'Where are you going?',
    'I am going home.', 'I will come tomorrow.', 'I went home yesterday.', 'Thank you.',
    'Good morning.', 'See you tomorrow.',
)
ALIASES = {'love you':'I love you.', 'miss you':'I miss you.'}
NAMES = [
    ('Anita','अनीता'),('Ravi','रवि'),('Meera','मीरा'),('Asha','आशा'),('Amit','अमित'),
    ('Neha','नेहा'),('Riya','रिया'),('Priya','प्रिया'),('Kavita','कविता'),('Rahul','राहुल'),
    ('Kiran','किरण'),('Tara','तारा'),('Mohan','मोहन'),('Rohan','रोहन'),('Suman','सुमन'),
    ('Nisha','निशा'),('Arjun','अर्जुन'),('Sita','सीता'),('Gita','गीता'),('Dev','देव'),
    ('Maya','माया'),('Sonia','सोनिया'),('Vijay','विजय'),('Vikram','विक्रम'),
]
# Representative family compositions reserved in advance; formulaic greetings are trained.
RESERVED = {'My name is Ravi.', 'I have two sisters.', 'I will come tomorrow.', 'Can you help me?'}
FORMULAS = {'I love you.','I miss you.','How are you?','What is your name?','Where are you going?',
            'You are my friend.','Thank you.','Good morning.','See you tomorrow.'}


def sentence_key(text):
    return comparison_key(text)


def generate_v2():
    reserved = {sentence_key(x) for x in RESERVED}
    rows = [r for r in generate() if r['family']!='location-question'
            and sentence_key(r['source']) not in reserved]
    used = [set(sentence_key(r[side]) for r in rows) for side in ('source','target')]

    def add(en,hi,family,group,*,force=None,**slots):
        en=en[0].upper()+en[1:]
        hi+= '?' if en.endswith('?') else '।'
        keys = [sentence_key(x) for x in (en,hi)]
        if any(key in keys_seen for key,keys_seen in zip(keys,used,strict=True)):
            return
        for key,keys_seen in zip(keys,used,strict=True):
            keys_seen.add(key)
        index=bucket('conversation-v2',family,group)
        partition=('composition' if keys[0] in reserved else 'train' if en in FORMULAS else force or
                   ('composition' if index<15 else 'validation' if index<29 else 'train'))
        rows.append({'source':en,'target':hi,'family':family,'group_family':family,'group':group,
                     'lexical_group':None,'slots':slots,'split':partition,'seed':SEED,
                     'origin':'hand-authored grammar; programmatic; no independent review',
                     'pair_id':hashlib.sha256((en+'\n'+hi).encode()).hexdigest()[:20]})

    for name,hindi in NAMES:
        for person,possessive in [('my','मेरा'),('your','तुम्हारा'),('his','उसका')]:
            group=[person,name]
            add(f'{person} name is {name}.',f'{possessive} नाम {hindi} है','extended-introduction',group,
                name=name,hindi_name=hindi)
        for en,hi in [('Good morning','सुप्रभात'),('Thank you','धन्यवाद'),('See you tomorrow','कल मिलते हैं')]:
            # Comma varies in user input. The canonical Hindi uses a vocative.
            add(f'{en}, {name}.',f'{hi}, {hindi}','named-greeting',[en,name],name=name,hindi_name=hindi)
        add(f'How are you, {name}?',f'{hindi}, तुम कैसे हो','greeting-question',[name],name=name,hindi_name=hindi)
    add('Thank you.','धन्यवाद','courtesy',['thanks'])
    add('Good morning.','सुप्रभात','courtesy',['morning'])
    add('See you tomorrow.','कल मिलते हैं','courtesy',['farewell'])
    add('What is your name?','तुम्हारा नाम क्या है','name-question',['your'])
    add("What is his name?",'उसका नाम क्या है','name-question',['his'])
    add("What is their name?",'उनका नाम क्या है','name-question',['their'])

    recipients = [('you','तुमसे','तुम्हारी'),('me','मुझसे','मेरी'),('him','उससे','उसकी'),
                  ('us','हमसे','हमारी'),('them','उनसे','उनकी')]+[
                      (name,f'{hi} से',f'{hi} की') for name,hi in NAMES]
    for en,hi,gender,copula,_,experience in SUBJECTS:
        third=en in ('he','she')
        hi_verb = {'m':'करता','f':'करती','p':'करते'}[gender]
        for eng,love,miss in recipients:
            if eng=={'I':'me','you':'you','he':'him','she':'him','we':'us','they':'them'}[en]:
                continue
            group = [en,eng]
            for neg in (False,True):
                qualifier=(' does not' if third else ' do not') if neg else ''
                loves='loves' if third and not neg else 'love'
                misses='misses' if third and not neg else 'miss'
                negative='नहीं ' if neg else ''
                add(f'{en}{qualifier} {loves} {eng}.',f'{hi} {love} प्यार {negative}{hi_verb} {copula}',
                    'affection-love',group,subject=en,recipient=eng,negative=neg)
                add(f'{en}{qualifier} {misses} {eng}.',f'{experience} {miss} याद {negative}आती है',
                    'affection-miss',group,subject=en,recipient=eng,negative=neg)
        inflect={'m':'सकता','f':'सकती','p':'सकते'}[gender]
        if en!='I':
            for recipient,poss,acc in [('me','मेरी','मुझे'),('us','हमारी','हमें'),('them','उनकी','उन्हें'),
                                      *[(a,f'{b} की',f'{b} को') for a,b in NAMES]]:
                for verb,action in [('help',f'{poss} मदद कर'),('call',f'{acc} फ़ोन कर')]:
                    add(f'Can {en} {verb} {recipient}?',f'क्या {hi} {action} {inflect} {copula}',
                        'modal-request',[en,verb,recipient],subject=en,verb=verb,recipient=recipient)
        be = 'am' if en=='I' else 'is' if third else 'are'
        stem={'m':'रहा','f':'रही','p':'रहे'}[gender]
        feeling={'m':'कैसा','f':'कैसी','p':'कैसे'}[gender]
        add(f'How {be} {en}?',f'{hi} {feeling} {copula}','greeting-question',[en],subject=en)
        for move,hmove in [('going','जा'),('coming','आ')]:
            for when,hwhen in [('', ''),(' now',' अभी'),(' today',' आज')]:
                add(f'Where {be} {en} {move}{when}?',f'{hi}{hwhen} कहाँ {hmove} {stem} {copula}',
                    'location-question',[en,move,when],subject=en,time=when)
        come={'I':'आऊँगा','you':'आओगे','he':'आएगा','she':'आएगी','we':'आएँगे','they':'आएँगे'}[en]
        verb_past={'m':'आया','f':'आई','p':'आए'}[gender] + (' थे' if en=='you' else '')
        for place,hplace in [('', ''),*PLACES]:
            place=f' {place}' if place else ''
            where=f' {hplace}' if hplace else ''
            for when,hwhen in [('today','आज'),('tomorrow','कल'),('in the morning','सुबह'),
                               ('in the evening','शाम को')]:
                add(f'{en} will come{place} {when}.',f'{hi} {hwhen}{where} {come}',
                    'arrival-future',[en,place,when],subject=en,place=place.strip(),tense='future')
            add(f'{en} came{place} yesterday.',f'{hi} कल{where} {verb_past}',
                'arrival-past',[en,place],subject=en,place=place.strip(),tense='past')
        have='has' if third else 'have'
        possessive={'I':('मेरी','मेरे'),'you':('तुम्हारी','तुम्हारे'),'he':('उसकी','उसके'),
                    'she':('उसकी','उसके'),'we':('हमारी','हमारे'),'they':('उनकी','उनके')}[en]
        for single,plural,h_single,h_plural,kind in [('sister','sisters','बहन','बहनें',0),
            ('brother','brothers','भाई','भाई',1),('friend','friends','दोस्त','दोस्त',1),
            ('child','children','बच्चा','बच्चे',1)]:
            for count,(number,hnum) in enumerate(zip(*QUANTITIES,strict=True),1):
                h_owner=possessive[kind]
                if count==1 and kind==1:
                    h_owner={'I':'मेरा','you':'तुम्हारा','he':'उसका','she':'उसका','we':'हमारा','they':'उनका'}[en]
                add(f'{en} {have} {number} {single if count==1 else plural}.',
                    f'{h_owner} {hnum} {h_single if count==1 else h_plural} '+('है' if count==1 else 'हैं'),
                    'family-count',[en,single,count],subject=en,number=count,relationship=single)
    add('You are my friend.','तुम मेरे दोस्त हो','friendship',['you'])
    add('Are you and he friends?','क्या तुम और वह दोस्त हो','friendship-extra',['he'],force='structure')
    for when,hindi in [('today','आज'),('tomorrow','कल'),('in the morning','सुबह'),('in the evening','शाम को')]:
        for name,hname in NAMES:
            ending = 'आएगी' if name in {'Anita','Meera','Asha','Neha','Riya','Priya','Kavita','Tara',
                                     'Nisha','Sita','Gita','Maya','Sonia'} else 'आएगा'
            add(f'When will {name} come {when}?',f'{hname} {hindi} कब {ending}','withheld-when',[name,when],
                force='structure',name=name,hindi_name=hname)
    return rows


def prepare(root):
    rows=generate_v2()
    random.Random(SEED).shuffle(rows)
    train=[r for r in rows if r['split']=='train']
    ids={sentence_key(r['source']) for r in rows}
    assert all(sentence_key(p.replace("don't","do not")) in ids for p in PROMPTS)
    root.mkdir(parents=True,exist_ok=False)
    tok=BilingualBPETokenizer.train((r[side] for r in train for side in ('source','target')),2000)
    tok.save(root/'tokenizer')
    maximum=max(len(tok.encode(r[k]))+1 for r in rows for k in ('source','target'))
    assert maximum<=64
    data=DataConfig(seed=SEED,vocab_size=2000,max_source_length=64,max_target_length=64)
    (root/'data.yaml').write_text(yaml.safe_dump(asdict(data)))
    frozen=root/'pairs.jsonl'
    frozen.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for filename,partition in [('train','train'),('val','validation'),('test','composition')]:
        write_parallel_csv(root/'processed'/f'{filename}.csv',
                           [TranslationPair(r['source'],r['target']) for r in rows if r['split']==partition])
    write_preparation_manifest(root/'processed',root/'tokenizer',frozen,data,
                               corpus={'recipe':'conversation-v2', 'seed':SEED})
    (root/'dataset_summary.json').write_text(json.dumps({'counts':Counter(r['split'] for r in rows),
                            'vocabulary':tok.vocab_size,'max_length':maximum},indent=2)+'\n')
    print((root/'dataset_summary.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    prepare(parser.parse_args().output)
