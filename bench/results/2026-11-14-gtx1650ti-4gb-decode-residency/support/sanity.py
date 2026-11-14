import ast, json, pathlib, re, sys, time, urllib.request
label=sys.argv[1];out=pathlib.Path('build-gtx1650/tuning3')/(label+'.sanity.json')
if out.exists():raise SystemExit('Refusing overwrite')
url='http://127.0.0.1:18080'
with urllib.request.urlopen(url+'/health',timeout=3) as r:health=json.load(r)
rows=[]
def call(name,body,expected,check):
 body={'model':health['model'],'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False},**body}
 start=time.monotonic()
 try:
  req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=480) as response:r=json.load(response)
  text=r['choices'][0]['message'].get('content') or ''
  row={'case':name,'request':body,'expected':expected,'passed':bool(check(r,text)),'wall_s':round(time.monotonic()-start,3),'response':r}
 except Exception as e:row={'case':name,'request':body,'expected':expected,'passed':False,'error':str(e),'wall_s':round(time.monotonic()-start,3)}
 rows.append(row);out.write_text(json.dumps({'label':label,'health':health,'checks':rows},ensure_ascii=False,indent=2))
 print(name,row['passed'],row['wall_s'],(row.get('response',{}).get('choices',[{'message':{'content':row.get('error')}}])[0]['message']),flush=True)
 return row

def prompt(text):return {'messages':[{'role':'user','content':text}]}
for name,text,expected in [('arithmetic','What is 17 + 25? Reply with the number only.','42'),('multiplication','What is 13 times 7? Reply with the number only.','91'),('arabic','ما عاصمة السعودية؟ أجب بالعربية بجملة قصيرة.','الرياض'),('capital','What is the capital of France? Reply with the city only.','Paris')]:
 call(name,prompt(text),expected,lambda r,t,e=expected:e in t)
call('json',prompt('Return only a JSON object with keys name and count. Set name to apple and count to 3. No markdown.'),{'name':'apple','count':3},lambda r,t:json.loads(t)=={'name':'apple','count':3})
call('sort',prompt('Sort [3, 1, 2] in ascending order. Reply with only the JSON array.'),[1,2,3],lambda r,t:json.loads(t)==[1,2,3])
def codecheck(r,text):
 text=re.sub(r'^```(?:python)?\s*|\s*```$','',text.strip())
 tree=ast.parse(text)
 allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.BinOp,ast.Mult,ast.Pow,ast.Name,ast.Load,ast.Constant,ast.Expr)
 if any(not isinstance(node,allowed) for node in ast.walk(tree)):return False
 if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='square' or tree.body[0].decorator_list:return False
 scope={};exec(compile(tree,'<restricted-square-smoke>','exec'),{'__builtins__':{}},scope)
 return [scope['square'](n) for n in [-3,0,2,7]]==[9,0,4,49]
call('python_square',prompt('Write only Python code, no markdown. Define square(n) that returns n multiplied by itself.'),{'square(-3)':9,'square(0)':0,'square(2)':4,'square(7)':49},codecheck)
call('hi',prompt('hi'),'greeting',lambda r,t:'hi' in t.lower() or 'hello' in t.lower())
tool={'type':'function','function':{'name':'get_weather','description':'Get current weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}
body={**prompt('Use get_weather to check the weather in Riyadh, Saudi Arabia.'),'tools':[tool],'max_tokens':96}
def toolcheck(r,t):
 calls=r['choices'][0]['message'].get('tool_calls') or []
 return bool(calls and calls[0]['function']['name']=='get_weather' and 'riyadh' in json.loads(calls[0]['function']['arguments']).get('city','').lower())
row=call('tool_call',body,'get_weather for Riyadh',toolcheck)
if row['passed']:
 message=row['response']['choices'][0]['message'];calls=message['tool_calls']
 follow={**body,'messages':body['messages']+[message,{'role':'tool','tool_call_id':calls[0]['id'],'content':'{"city":"Riyadh","temperature_c":35,"condition":"sunny"}'}],'max_tokens':64}
 call('tool_result',follow,'35 degrees C',lambda r,t:'35' in t)
if '--long' in sys.argv:
 text='Remember this exact access code: violet-742.\n'+'The small laptop runs a local language model. '*903+'\nWhat access code did I ask you to remember? Reply with the code only.'
 call('near_8k_recall',{**prompt(text),'max_tokens':16},'violet-742',lambda r,t:'violet-742' in t)
print('CHECKS',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
