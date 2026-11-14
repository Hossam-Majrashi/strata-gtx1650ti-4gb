"""Controlled HTTP fixture: isolate omission of unused batched prompt buffers, not batched/window arithmetic."""
import hashlib, json, os, pathlib, signal, subprocess, sys, time, urllib.request
ROOT=pathlib.Path.cwd().resolve();WORK=ROOT/'build-gtx1650'/'tuning3'
sys.path.insert(0,str(ROOT/'tools'));from bench_gtx1650 import request, CASES
PYTHON=sys.executable
rows=[]
for name,engine,enabled in [('parity_old','strata-before-prefill-pool',False),('parity_disabled','strata-residency-final',False),('parity_enabled','strata-residency-final',True)]:
 cfg=json.loads((ROOT/'build-gtx1650/gtx1650-hc-q8.json').read_text())
 cfg['exe']=str(ROOT/'build-gtx1650'/engine);cfg['port']=18080;cfg['log']=str(WORK/(name+'.engine.log'))
 cfg['env'].update({'STRATA_PREFILL_CPU_SHARE':'0','STRATA_LOGPOS':str(WORK/(name+'.logprobs')),'STRATA_WEIGHT_CACHE_MIB':'96'})
 if enabled:cfg['env']['STRATA_DECODE_ONLY_PROMPT']='1'
 args=cfg['args']
 for flag,value in {'--prompt-cache':'0','--short-read':'8192','--pcie-frac':'0.5','--adapt-every':'0','--adapt-swaps':'0'}.items():
  if flag in args:args[args.index(flag)+1]=value
  else:args.extend([flag,value])
 path=WORK/(name+'.json')
 if path.exists():raise RuntimeError('Refusing overwrite')
 path.write_text(json.dumps(cfg,indent=2)+'\n')
 env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
 proc=None
 try:
  with (WORK/(name+'.server.log')).open('w') as out:proc=subprocess.Popen([PYTHON,'-m','serve.server','--engine','strata','--config',str(path),'--host','127.0.0.1','--port','18080'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(240):
   if proc.poll() is not None:raise RuntimeError('startup exit '+str(proc.returncode))
   try:
    h=request('http://127.0.0.1:18080/health',timeout=1)
    if h.get('loaded'):break
   except Exception:pass
   time.sleep(1)
  else:raise RuntimeError('timeout')
  body={'model':h['model'],'messages':[{'role':'user','content':CASES[0][1]}],'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
  r=request('http://127.0.0.1:18080/v1/chat/completions',body)
  row={'label':name,'engine_sha256':hashlib.sha256(pathlib.Path(cfg['exe']).read_bytes()).hexdigest(),'request':body,'response':r}
  rows.append(row);(WORK/(name+'.response.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
  print(name,r['choices'][0]['message'],flush=True)
 finally:
  if proc:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  time.sleep(2)
texts=[r['response']['choices'][0]['message'] for r in rows]
probs=[(WORK/(r['label']+'.logprobs')).read_bytes() for r in rows]
result={'fixture':'one 44-token English prompt, 64-token greedy continuation; identical window-only reads in every arm','head':'host in all arms, to isolate buffer allocation','adaptation':'off; fixed PCIe 0.5; prompt reuse off; CPU prefill share off','output_messages_identical':texts[0]==texts[1]==texts[2],'prompt_logprob_bytes_identical':probs[0]==probs[1]==probs[2],'prompt_logprob_rows':[len(p.splitlines()) for p in probs],'runs':[{'label':r['label'],'engine_sha256':r['engine_sha256']} for r in rows]}
(WORK/'parity-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if not (result['output_messages_identical'] and result['prompt_logprob_bytes_identical']):raise SystemExit(1)
