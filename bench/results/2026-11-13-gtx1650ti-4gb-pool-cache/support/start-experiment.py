import json, os, pathlib, signal, subprocess, sys, time, urllib.request
root=pathlib.Path('$REPO')
work=root/'build-gtx1650'/'tuning2';work.mkdir(exist_ok=True)
name=sys.argv[1]
envs=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
flags=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
pidfile=work/'server.pid'
if pidfile.exists():
 pid=int(pidfile.read_text())
 cmd=pathlib.Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and 'serve.server' in cmd.read_bytes().replace(b'\0',b' ').decode():
  os.kill(pid,signal.SIGTERM)
  for _ in range(100):
   p=pathlib.Path(f'/proc/{pid}/stat')
   if not p.exists() or p.read_text().split()[2]=='Z':break
   time.sleep(.2)
  else:raise SystemExit('Previous server did not stop')
profile=os.environ.get('TUNE_PROFILE','fast')
base=json.loads((root/'build-gtx1650'/f'gtx1650-{profile}.json').read_text())
if os.environ.get('TUNE_ENGINE'):base['exe']=os.environ['TUNE_ENGINE']
args=base['args']
for flag,value in {'--prompt-cache':'0',**flags}.items():
 if flag in args:
  idx=args.index(flag)
  if isinstance(value,bool):
   if not value:args.pop(idx)
  elif value is None:del args[idx:idx+2]
  else:args[idx+1]=str(value)
 elif value is not None and value is not False:
  args.append(flag)
  if value is not True:args.append(str(value))
for key,value in envs.items():
 if value is None:base['env'].pop(key,None)
 else:base['env'][key]=str(value)
base['port']=18080;base['log']=str(work/(name+'.engine.log'))
config=work/(name+'.json')
if config.exists():raise SystemExit('Refusing overwrite: '+str(config))
config.write_text(json.dumps(base,indent=2)+'\n')
with (work/(name+'.server.log')).open('w') as out:
 process=subprocess.Popen(['$CONTROL/python/bin/python3','-m','serve.server','--engine','strata','--config',str(config),'--host','127.0.0.1','--port','18080'],cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(process.pid));print('PID',process.pid,'CONFIG',config,flush=True)
for _ in range(130):
 if process.poll() is not None:
  print('FAILED',process.returncode);print((work/(name+'.server.log')).read_text()[-2400:]);raise SystemExit(1)
 try:
  with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=1) as response:h=json.load(response)
  if h.get('loaded'):
   print('READY',h);break
 except Exception:pass
 time.sleep(1)
else:process.terminate();raise SystemExit('Startup timeout')
