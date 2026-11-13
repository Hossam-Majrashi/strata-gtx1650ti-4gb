import json, os, pathlib, subprocess, sys
sys.path.insert(0,'tools')
from strata_tokenizer import Tokenizer
root=pathlib.Path.cwd();work=root/'build-gtx1650/tuning2'
tok=Tokenizer.from_gguf('$DATA/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf')
prompt='<|im_start|>user\nWrite a short list of three colors.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
continuation='Red, green, and blue are common colors. This is a short deterministic continuation.'
(work/'hash-prompt.tokens').write_text(','.join(map(str,tok.encode(prompt,parse_special=True))))
(work/'hash-follow.tokens').write_text(','.join(map(str,tok.encode(continuation))))
base=json.loads((root/'build-gtx1650/gtx1650-fast.json').read_text())
meta={'prompt':prompt,'continuation':continuation,'runs':[]}
for label,exe,extra in [('hash_old',root/'build-gtx1650/strata-before-pool',{}),('hash_new',root/'build-gtx1650/strata',{}),('hash_cached',root/'build-gtx1650/strata',{'STRATA_NATIVE_POOL':'1','STRATA_WEIGHT_CACHE_MIB':'160'})]:
 env=dict(os.environ)
 for key in ['STRATA_HEAD_HOST','STRATA_HC_Q8','STRATA_HC_REQ8','STRATA_NATIVE_POOL','STRATA_WEIGHT_CACHE_MIB']:
  env.pop(key,None)
 env.update(base['env']);env.update(extra);env['STRATA_PREFILL_CPU_SHARE']='0';env['CUDA_MODULE_LOADING']='EAGER'
 args=base['args']+['--tokens-file',str(work/'hash-prompt.tokens'),'--spec-follow',str(work/'hash-follow.tokens'),'--window-hashes',str(work/(label+'.hashes')),'--pcie-frac','0.4','--adapt-every','0','--adapt-swaps','0','--max-new','16','--temperature','0']
 print('START',label,flush=True)
 with (work/(label+'.log')).open('w') as out:
  r=subprocess.run([str(exe),*args],env=env,stdout=out,stderr=subprocess.STDOUT,timeout=180)
 row={'label':label,'exe':str(exe),'env':{**base['env'],**extra,'STRATA_PREFILL_CPU_SHARE':'0','CUDA_MODULE_LOADING':'EAGER'},'args':args,'returncode':r.returncode};meta['runs'].append(row)
 (work/'hash-check.json').write_text(json.dumps(meta,indent=2))
 print('EXIT',label,r.returncode,flush=True)
 if r.returncode:print((work/(label+'.log')).read_text()[-3000:]);raise SystemExit(1)
files=[(work/(name+'.hashes')).read_bytes() for name in ['hash_old','hash_new','hash_cached']]
print('Old vs new disabled hash equality:',files[0]==files[1],flush=True)
print('Old vs cached hash equality:',files[0]==files[2],flush=True)
meta['disabled_matches_old']=files[0]==files[1];meta['cached_matches_old']=files[0]==files[2]
(work/'hash-check.json').write_text(json.dumps(meta,indent=2))
