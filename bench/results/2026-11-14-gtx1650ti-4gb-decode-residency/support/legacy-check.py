import json, os, pathlib, subprocess, sys, tempfile
root=pathlib.Path.cwd().resolve()
script=root/'tools/run_gtx1650.py'
old=subprocess.check_output(['git','show','e65fa94f:tools/run_gtx1650.py'],cwd=root,text=True)
meta=[]
with tempfile.TemporaryDirectory(prefix='strata legacy check ') as td:
 p=pathlib.Path(td)
 for name in ['strata','packs/iq3_xxs/dense.bin','packs/iq3_xxs/index.txt','packs/iq3_xxs/tokenizer/tokenizer.json']+[f'models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{i}-of-00002.gguf' for i in (1,2)]:
  f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.touch()
 env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
 for profile in ['stable','fast','cached','hc-q8']:
  for no_cache in [False,True]:
   flags=['--data-dir',td,'--engine',str(p/'strata'),'--profile',profile,'--write-config-only']+(['--no-prompt-cache'] if no_cache else [])
   path=p/('gtx1650.json' if profile=='stable' else f'gtx1650-{profile}.json')
   bootstrap='__file__='+repr(str(script))+'\n'+old
   a=subprocess.run([sys.executable,'-c',bootstrap,*flags],env=env,cwd=root,capture_output=True,check=True)
   old_bytes=path.read_bytes()
   b=subprocess.run([sys.executable,str(script),*flags],env=env,cwd=root,capture_output=True,check=True)
   meta.append({'profile':profile,'no_prompt_cache':no_cache,'config_identical':old_bytes==path.read_bytes(),'stdout_identical':a.stdout==b.stdout,'stderr_identical':a.stderr==b.stderr})
assert all(r['config_identical'] and r['stdout_identical'] and r['stderr_identical'] for r in meta)
path=root/'build-gtx1650/tuning3/legacy-check.json';path.write_text(json.dumps(meta,indent=2)+'\n')
print('8/8 existing profile/cache combinations: config, stdout and stderr byte-identical')
