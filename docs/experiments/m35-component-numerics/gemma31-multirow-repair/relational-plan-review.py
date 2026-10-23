# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Independent retained-metadata comparator; no model execution.
# Derived from the reviewed D/A relational comparator, with explicit identity
# ledger membership replacing equality across the changed module fingerprint.
import pathlib,json,collections,hashlib,sys,argparse,copy

def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def records(plan):
 return plan['recipes']+[r for r in plan['logical_records'] if r.get('kind')=='selection']
def check_identities(plans,ledger,side):
 field='reconstructed_old_identity' if side=='A' else 'current_identity'
 pairs=set()
 for plan in plans['plans']:
  for record in records(plan):
   name=record['implementation']; identity=record['identity']
   if name not in ledger or identity!=ledger[name][field]:
    raise ValueError(f'unknown {side} implementation identity: {name} {identity}')
   pairs.add((name,identity))
 return len(pairs)
def compare_pair(ap,dp,ledger):
 check_identities(ap,ledger,'A');check_identities(dp,ledger,'R')
 assert len(ap['plans'])==len(dp['plans'])==1
 results=[]
 for A,D in zip(ap['plans'],dp['plans'],strict=True):
  for k in ['stage','phase','rows','segments']:assert A['header'][k]==D['header'][k]
  ad={v['id']:v for v in A['descriptors']};dd={v['id']:v for v in D['descriptors']}
  for r in D['recipes']:
   for n in r['operated_nodes']:
    assert n['output']==dd[n['output']['id']];assert n['sources']==[dd[s] if s is not None else None for s in n['output']['sources']]
  prep={r['operated_nodes'][0]['output']['id']:r['operated_nodes'][0]['sources'][0]['id'] for r in D['recipes'] if r['implementation']=='llmp.q8_1'}
  # Collapse incumbent separate gate/up + GEGLU by actual dependency IDs, never by recipe count or spelling alone.
  removed=set(); glu_inputs={}
  for r in A['recipes']:
   if r['implementation']=='ggml.geglu':
    g=r['operated_nodes'][0]['output'];gate,up=[ad[i] for i in g['sources']];assert gate['op']==up['op']=='MUL_MAT';glu_inputs[g['id']]=(gate,up);removed.update(g['sources'])
  ar=[r for r in A['recipes'] if not (len(r['operated_nodes'])==1 and r['operated_nodes'][0]['output']['id'] in removed)]
  dr=[r for r in D['recipes'] if r['implementation']!='llmp.q8_1'];assert len(ar)==len(dr)
  sels={r['step']:r for r in D['logical_records'] if r['kind']=='selection'};asel={r['step']:r for r in A['logical_records'] if r['kind']=='selection'}
  special={};dense=collections.Counter();glu=collections.Counter();geometry=collections.Counter();unaffected=collections.Counter()
  for a,d in zip(ar,dr,strict=True):
   if a['implementation']=='ggml.mul_mat.mmvq':
    assert d['implementation'] in ['llmp.vecq','llmp.mmvq.prepared'];ao=a['operated_nodes'];do=d['operated_nodes'];assert len(ao)==len(do)==1;ax=ao[0]['output'];dx=do[0]['output'];assert dx['sources'][1] in prep
    if d['implementation']=='llmp.mmvq.prepared':assert dx['sources'][2]==prep[dx['sources'][1]]
    else:assert len(dx['sources'])==2
    special[(ax['id'],dx['id'])]=[(ax['sources'][0],dx['sources'][0]),(ax['sources'][1],prep[dx['sources'][1]])]
    wt=dd[dx['sources'][0]];dense[(sels[d['step']]['implementation'],wt['ir_format']['encoding'])]+=1;geometry[(wt['ir_format']['encoding'],tuple(wt['shape']))]+=1
   elif a['implementation'] in ['ggml.geglu','ggml.mul_mat_geglu.mmvq_fused']:
    assert d['implementation']=='llmp.vecq' and sels[d['step']]['implementation']=='llmp.vecq.glu';assert len(d['operated_nodes'])==1
    out=a['operated_nodes'][-1]['output'];dx=d['operated_nodes'][0]['output']
    if a['implementation']=='ggml.geglu':gate,up=glu_inputs[out['id']]
    else:gate,up,_=[v['output'] for v in a['operated_nodes']];removed.update([gate['id'],up['id']])
    assert gate['sources'][1]==up['sources'][1] and out['sources']==[gate['id'],up['id']];assert all(v['shape']==out['shape'] and v['strides']==out['strides'] and v['dtype']=='f32' for v in [gate,up]);assert dx['sources'][1] in prep and len(dx['sources'])==3
    special[(out['id'],dx['id'])]=[(gate['sources'][0],dx['sources'][2]),(up['sources'][0],dx['sources'][0]),(gate['sources'][1],prep[dx['sources'][1]])]
    for n in [dx['sources'][0],dx['sources'][2]]:glu[dd[n]['ir_format']['encoding']]+=1;geometry[(dd[n]['ir_format']['encoding'],tuple(dd[n]['shape']))]+=1
   else:
    for k in ['implementation','op','lane','ordered_nodes','prologue']:assert a[k]==d[k],(A['identity'],k,a['implementation'],d['implementation'])
    unaffected[a['implementation']]+=1
  maps={k:{} for k in ['id','ir_node','tensor','root','storage']};rev={k:{} for k in maps};seen=set()
  def bind(k,a,d):
   if a is None or d is None:assert a is d;return
   assert maps[k].get(a,d)==d and rev[k].get(d,a)==a,(A['identity'],k,a,d,maps[k].get(a),rev[k].get(d));maps[k][a]=d;rev[k][d]=a
  def pair(ai,di):
   if ai is None or di is None:assert ai is di;return
   if (ai,di) in seen:return
   seen.add((ai,di));a=ad[ai];d=dd[di]
   for k in maps:bind(k,a[k],d[k])
   for k in ['kind','dtype','shape','strides','ir_format','part','view_offset']:assert a[k]==d[k],(A['identity'],ai,di,k,a[k],d[k])
   if (ai,di) in special:assert a['op'] in ['MUL_MAT','GLU'] and d['op']=='CUSTOM';edges=special[(ai,di)]
   else:assert a['op']==d['op'],(ai,di,a['op'],d['op']);assert len(a['sources'])==len(d['sources']);edges=zip(a['sources'],d['sources'])
   if a['view_source'] is not None:assert d['view_source'] is not None;pair(a['view_source'],d['view_source'])
   else:assert d['view_source'] is None
   for u,v in edges:pair(u,v)
  for a,d in zip(ar,dr,strict=True):
   if a['implementation'] in ['ggml.geglu','ggml.mul_mat_geglu.mmvq_fused']:pair(a['operated_nodes'][-1]['output']['id'],d['operated_nodes'][0]['output']['id'])
   else:
    assert len(a['operated_nodes'])==len(d['operated_nodes'])
    for av,dv in zip(a['operated_nodes'],d['operated_nodes']):pair(av['output']['id'],dv['output']['id'])
  left=set(ad)-set(maps['id'])-removed;right=set(dd)-set(rev['id'])-set(prep)
  for ai in sorted(left):
   a=ad[ai];cs=[di for di in right if dd[di]['ir_node']==maps['ir_node'].get(a['ir_node']) and dd[di]['op']==a['op'] and dd[di]['shape']==a['shape']];assert len(cs)==1,(A['identity'],ai,a,cs);di=cs[0];pair(ai,di);right.remove(di)
  assert not right,(A['identity'],right)
  for pi,src in prep.items():
   q=dd[pi];v=dd[src];assert q['op']=='CUSTOM' and q['dtype']=='i8' and q['ir_format']['encoding']=='q8_1' and q['ir_format']['layout']=='ggml_blocks';assert q['sources']==[src] and v['dtype']=='f32' and q['ir_format']['shape']==v['shape'];K,rows=v['shape'][:2];n=((K+511)//512*512)//32*36*rows;assert q['shape']==[n,1,1,1] and q['strides']==[1,n,n,n]
  # Unchanged logical recipes preserve component ownership and exact graph relationships under the proven maps.
  for a,d in zip(ar,dr,strict=True):
   if a['implementation'] in ['ggml.mul_mat.mmvq','ggml.geglu','ggml.mul_mat_geglu.mmvq_fused']:continue
   av=asel[a['step']];dv=sels[d['step']]
   for k in ['implementation','lane','rung','expansion_rule','group']:assert av[k]==dv[k]
   assert len(av['nodes'])==len(dv['nodes'])
   for an,dn in zip(av['nodes'],dv['nodes']):
    na=dict(an)
    na['id']=maps['ir_node'].get(na['id'],na['id'])
    for k in ['sources','outputs']:na[k]=[maps['ir_node'].get(v,v) for v in na[k]]
    assert na==dn,(A['identity'],a['implementation'],na,dn)
  results.append({'domain':{k:A['header'][k] for k in ['phase','stage','rows','segments']},'a_identity':A['identity'],'d_identity':D['identity'],'uses':len(D['completions']),'paired_descriptors':len(maps['id']),'a_descriptors':len(ad),'d_descriptors':len(dd),'removed_glu_temporaries':len(removed),'added_q8_1_preparations':len(prep),'dense':{str(k):v for k,v in dense.items()},'glu_weights':dict(glu),'weight_geometries':{str(k):v for k,v in geometry.items()},'unaffected_exact':dict(unaffected)})
 return results

def compare(ap,dp,ledger):
 check_identities(ap,ledger,'A');check_identities(dp,ledger,'R')
 assert ap['plans'] and dp['plans']
 keys=['stage','phase','rows','segments']
 def domain(p):return tuple(p['header'][k] for k in keys)
 score=collections.Counter();joined=0
 for p in dp['plans']:
  assert p['completions']
  if p['header']['phase']==4:
   assert p['header']['stage']==0 and p['header']['segments']==1 and p['header']['rows'] in (1,2)
   score[p['header']['rows']]+=len(p['completions'])
  if domain(p)==(0,1,2,2):joined+=len(p['completions'])
 assert score=={2:535,1:1},('registered score geometry/coverage',score)
 assert joined>0,'no actual pure two-owner joined decode'
 results=[]
 for d in dp['plans']:
  candidates=[a for a in ap['plans'] if domain(a)==domain(d)]
  failures=[]
  for a in candidates:
   try:result=compare_pair({'plans':[a]},{'plans':[d]},ledger)
   except (AssertionError,KeyError,ValueError,IndexError) as error:
    failures.append(str(error)[:300]);continue
   results.extend(result);break
  else:raise AssertionError(('unmatched actual candidate semantic plan',d['identity'],domain(d),failures))
 return results

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--baseline',required=True);parser.add_argument('--candidate',required=True)
 parser.add_argument('--identity-map',required=True);parser.add_argument('--output',required=True)
 parser.add_argument('--candidate-log');parser.add_argument('--self-test',action='store_true')
 parser.add_argument('--parser-directory',default='/tmp/llmp-u4-gemma31-multirow-materialized-r1/reg/docs/experiments/m35-component-numerics')
 args=parser.parse_args()
 ap=json.loads(pathlib.Path(args.baseline).read_text());dp=json.loads(pathlib.Path(args.candidate).read_text())
 mapped=json.loads(pathlib.Path(args.identity_map).read_text());ledger={r['name']:r for r in mapped['implementations']}
 assert len(ledger)==len(mapped['implementations'])==168
 assert all(r['module_substitutions']==1 for r in ledger.values())
 synthetic=args.self_test
 if synthetic:
  # D is retained evidence in the old runtime namespace. Change identities
  # only in memory to exercise the prospective R comparison mechanism.
  check_identities(dp,ledger,'A')
  for plan in dp['plans']:
   for r in records(plan):r['identity']=ledger[r['implementation']]['current_identity']
 else:
  if not args.candidate_log:raise ValueError('actual candidate requires raw runtime-log reparse')
  sys.path.insert(0,args.parser_directory);import plan_reports
  assert json.loads(json.dumps(plan_reports.parse_log(pathlib.Path(args.candidate_log))))==dp
 results=compare(ap,dp,ledger)
 negatives=[]
 if synthetic:
  for domain in ['recipes','logical_records']:
   r=next(r for r in dp['plans'][0][domain] if 'implementation' in r)
   old=r['identity'];r['identity']='0'*64
   try:check_identities(dp,ledger,'R')
   except ValueError:negatives.append(domain+' unknown hash refused')
   else:raise AssertionError('unknown identity was accepted')
   r['identity']=old
  r=dp['plans'][0]['recipes'][0];old=r['implementation'];r['implementation']='unknown.implementation'
  try:check_identities(dp,ledger,'R')
  except ValueError:negatives.append('unknown name refused')
  else:raise AssertionError('unknown name was accepted')
  r['implementation']=old
  # Scheduling variance is allowed prospectively: one fewer scalar chat plan
  # and fewer joined completions; fixed teacher-score coverage remains exact.
  varied={'plans':list(dp['plans'])}
  at=next(i for i,p in enumerate(varied['plans']) if p['header']['phase']==1 and p['header']['rows']==1)
  varied['plans'].pop(at)
  at=next(i for i,p in enumerate(varied['plans']) if p['header']['phase']==1 and p['header']['rows']==2)
  varied['plans'][at]=dict(varied['plans'][at]);varied['plans'][at]['completions']=varied['plans'][at]['completions'][:-1]
  assert len(compare(ap,varied,ledger))==len(varied['plans'])
  negatives.append('synthetic chat scheduling variance accepted with all semantic checks')
  at=next(i for i,p in enumerate(varied['plans']) if p['header']['phase']==4 and p['header']['rows']==2)
  varied['plans'][at]=dict(varied['plans'][at]);varied['plans'][at]['completions']=varied['plans'][at]['completions'][:-1]
  try:compare(ap,varied,ledger)
  except AssertionError:negatives.append('missing registered score completion refused')
  else:raise AssertionError('short score coverage was accepted')
 out={'status':'PASS','synthetic_metadata_self_test':synthetic,'actual_R_review':not synthetic,
      'baseline_sha256':digest(args.baseline),'candidate_input_sha256':digest(args.candidate),
      'identity_map_sha256':digest(args.identity_map),'script_sha256':digest(__file__),
      'baseline_identity_pairs':check_identities(ap,ledger,'A'),
      'candidate_identity_pairs':check_identities(dp,ledger,'R'),
      'negative_controls':negatives,'per_plan_results':results}
 pathlib.Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='per_plan_results'},indent=2))
if __name__=='__main__':main()
