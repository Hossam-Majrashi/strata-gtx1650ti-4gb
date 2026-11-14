import json, os, pathlib, signal, subprocess, sys, time, urllib.request
ROOT = pathlib.Path.cwd().resolve()
WORK = ROOT/'build-gtx1650'/'tuning3'
PYTHON = sys.executable
sys.path.insert(0, str(ROOT/'tools'))
import bench_gtx1650 as bench

def run(item):
    name = item['name']
    path = WORK/(name+'.json')
    if path.exists(): raise RuntimeError('Already exists: '+str(path))
    base=json.loads((ROOT/'build-gtx1650'/('gtx1650-'+item.get('profile','hc-q8')+'.json')).read_text())
    base['exe']=str(ROOT/'build-gtx1650'/item.get('engine','strata'))
    base['env'].update(item.get('env',{}))
    base['env']={k:str(v) for k,v in base['env'].items() if v is not None}
    args=base['args']
    for flag,value in {'--prompt-cache':'0',**item.get('flags',{})}.items():
        if flag in args:
            idx=args.index(flag)
            if isinstance(value,bool):
                if not value: args.pop(idx)
            elif value is None: del args[idx:idx+2]
            else: args[idx+1]=str(value)
        elif value is not None and value is not False:
            args.append(flag)
            if value is not True: args.append(str(value))
    base['port']=18080; base['log']=str(WORK/(name+'.engine.log'))
    path.write_text(json.dumps(base,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    proc=None
    try:
        with (WORK/(name+'.server.log')).open('w') as out:
            proc=subprocess.Popen([PYTHON,'-m','serve.server','--engine','strata','--config',str(path),'--host','127.0.0.1','--port','18080'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(240):
            if proc.poll() is not None: raise RuntimeError('Startup exit '+str(proc.returncode))
            try:
                health=bench.request('http://127.0.0.1:18080/health',timeout=1)
                if health.get('loaded'): break
            except Exception: pass
            time.sleep(1)
        else: raise RuntimeError('Startup timeout')
        print('READY',name,flush=True)
        with (WORK/(name+'.bench.log')).open('w') as out:
            ret=subprocess.run([PYTHON,str(ROOT/'tools/bench_gtx1650.py'),'--url','http://127.0.0.1:18080','--label',name,'--rounds',str(item.get('rounds',1)),'--out',str(WORK/(name+'.results.jsonl'))],stdout=out,stderr=subprocess.STDOUT,timeout=1200)
        print(name,'EXIT',ret.returncode,(WORK/(name+'.bench.log')).read_text(),flush=True)
        if item.get('sanity'):
            with (WORK/(name+'.sanity.log')).open('w') as out:
                ret=subprocess.run([PYTHON,str(WORK/'sanity.py'),name,'--long'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=1200)
            print(name,'SANITY',ret.returncode,(WORK/(name+'.sanity.log')).read_text(),flush=True)
    except Exception as e:
        print('FAILED',name,repr(e),flush=True)
    finally:
        if proc:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        time.sleep(2)

if __name__=='__main__':
    for item in json.loads(pathlib.Path(sys.argv[1]).read_text()): run(item)
