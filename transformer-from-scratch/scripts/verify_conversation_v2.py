"""Exercise the deployed checkpoint on the fixed conversational prompt list."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.interview_server import InterviewService, UnsupportedInput
from scripts.prepare_conversation_v2 import ROOT, PROMPTS
from src.data.corpus import comparison_key
from src.data.prepared import file_digest


def main():
    torch.set_num_threads(2)
    checkpoint = ROOT/'checkpoints/scratch/best.pt'
    service=InterviewService(checkpoint,ROOT)
    rows=[json.loads(line) for line in (ROOT/'pairs.jsonl').read_text().splitlines()]
    lookup={comparison_key(row['source']):row for row in rows}
    actual=[]
    for prompt in (*PROMPTS,'love you','LOVE YOU!',"I don’t like coffee!","They went to the shop yesterday."):
        result=service.predict(prompt)
        row=lookup[comparison_key(result['canonical_source'])]
        output=result['outputs']['greedy']
        actual.append({'input':prompt,'canonical':result['canonical_source'],
                       'prediction':output['text'],'reference':row['target'],'partition':row['split'],
                       'exact':output['text']==row['target'],'latency_ms':output['milliseconds'],'eos':output['eos']})
        print(prompt,row['split'],output['text'],output['text']==row['target'],flush=True)
    rejections={}
    for text in ('kaise ho','Do you know the weather in Paris?','What caused the First World War?'):
        try:
            service.predict(text)
        except UnsupportedInput as err:
            rejections[text]=err.code
        else:
            raise AssertionError(f'{text} escaped the scope check')
    result={'checkpoint_sha256':file_digest(checkpoint),'prompts':actual[:20],'extra':actual[20:],
            'rejections':rejections,'gallery':service.info['gallery']}
    (ROOT/'conversation_probes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
