import json, pathlib, sys, time, urllib.request
label=sys.argv[1];port=int(sys.argv[2])
out=pathlib.Path('build-gtx1650/tuning3')/(label+'.followup.json')
if out.exists():raise SystemExit('Refusing overwrite')
url=f'http://127.0.0.1:{port}'
with urllib.request.urlopen(url+'/health',timeout=5) as r:health=json.load(r)
messages=[{'role':'user','content':'The session code is amber-29. Remember it. Reply only OK.'}]
rows=[]
for turn in range(2):
 body={'model':health['model'],'messages':list(messages),'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 started=time.monotonic();ttft=None;content='';events=[]
 with urllib.request.urlopen(req,timeout=120) as response:
  for line in response:
   text=line.decode().strip()
   if not text.startswith('data:'):continue
   raw=text[5:].strip()
   if raw=='[DONE]':break
   chunk=json.loads(raw);events.append(chunk)
   for choice in chunk.get('choices',[]):
    token=choice.get('delta',{}).get('content') or ''
    if token:
     if ttft is None:ttft=time.monotonic()-started
     content+=token
 elapsed=time.monotonic()-started
 row={'turn':turn+1,'request':body,'content':content,'ttft_s':ttft,'wall_s':elapsed,'events':events,'passed':('OK' in content if turn==0 else 'amber-29' in content)}
 rows.append(row);print(label,turn+1,row['passed'],'TTFT',round(ttft or 0,3),'total',round(elapsed,3),repr(content),flush=True)
 messages.extend([{'role':'assistant','content':content},{'role':'user','content':'What is the session code? Reply only with the code.'}])
out.write_text(json.dumps({'label':label,'health':health,'turns':rows},ensure_ascii=False,indent=2))
