import json, os, pathlib, signal, subprocess, sys, time
ROOT=pathlib.Path.cwd().resolve();WORK=ROOT/'build-gtx1650'/'tuning3'
sys.path.insert(0,str(ROOT/'tools'));from bench_gtx1650 import request
PYTHON=sys.executable
base=json.loads((ROOT/'build-gtx1650/gtx1650-hc-q8-decode.json').read_text())
base['exe']=str(ROOT/'build-gtx1650/strata-residency-final');base['port']=18080;base['log']=str(WORK/'decode_live.engine.log')
args=base['args']
if '--prompt-cache' in args:
 idx=args.index('--prompt-cache');del args[idx:idx+2]
path=WORK/'decode_live.json'
if path.exists():raise RuntimeError('Refusing overwrite')
path.write_text(json.dumps(base,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
proc=None
try:
 with (WORK/'decode_live.server.log').open('w') as out:proc=subprocess.Popen([PYTHON,'-m','serve.server','--engine','strata','--config',str(path),'--host','127.0.0.1','--port','18080'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(240):
  if proc.poll() is not None:raise RuntimeError('Startup exit '+str(proc.returncode))
  try:
   h=request('http://127.0.0.1:18080/health',timeout=1)
   if h.get('loaded'):break
  except Exception:pass
  time.sleep(1)
 else:raise RuntimeError('Startup timeout')
 subprocess.run([PYTHON,str(WORK/'followup.py'),'decode_live','18080'],cwd=ROOT,check=True,timeout=180)
 d=json.loads((WORK/'decode_live.followup.json').read_text())
 assert all(t['passed'] for t in d['turns'])
 caches=[e.get('timings',{}).get('cache_n',0) for e in d['turns'][1]['events']]
 assert max(caches)>0, caches
 print('Streaming 2/2 passed; followup reused',max(caches),'tokens',flush=True)
finally:
 if proc:
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
# Invalid short-read limit: reject at startup rather than using an absent batched path.
base['args'][base['args'].index('--short-read')+1]='0';base['log']=str(WORK/'guard_shortread.engine.log')
p=WORK/'guard_shortread.json';p.write_text(json.dumps(base,indent=2)+'\n')
with (WORK/'guard_shortread.server.log').open('w') as out:
 r=subprocess.run([PYTHON,'-m','serve.server','--engine','strata','--config',str(p),'--host','127.0.0.1','--port','18080'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240)
log=(WORK/'guard_shortread.engine.log').read_text()
assert r.returncode!=0 and 'STRATA_DECODE_ONLY_PROMPT=1 needs one discrete GPU' in log
print('Invalid short-read startup rejected as expected',flush=True)
