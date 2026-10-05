#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

import array,hashlib,json,pathlib,math,datetime
s=pathlib.Path('/home/pmeenan/.local/share/jitllm/gemma-assistant-native')
def ident(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
pre=json.loads((s/'native-pre3.json').read_text()); own=json.loads((s/'native-own3-frozen.json').read_text()); m=json.loads((s/'posthoc-c1-owner0-manifest.json').read_text())
assert hashlib.sha256((s/'native-own3-frozen.json').read_bytes()).hexdigest()==m['native_endogenous_freeze_sha256']
assert ident(pathlib.Path('/home/pmeenan/src/jitLLM-wt/m3fixb/build/spark-native/benchmarks/jitllm_gemma_assistant_fixture'))==pre['binary']
for n,r in m['files'].items():assert ident(s/'posthoc-c1'/n)==r
job=pathlib.Path('/home/pmeenan/.local/share/jitllm/jobs/m35-assistant-execution-posthoc3')
assert json.loads((job/'final.json').read_text())['rc']==0
assert all(r['rc']==0 for r in json.loads((job/'steps.json').read_text())['steps'])
log=(job/'log').read_text();cache=[x.split('=')[-1] for x in log.splitlines() if 'immutable_cache_' in x];assert len(cache)==5 and len(set(cache))==1 and cache[0]==own['borrowed_cache_sha256']
rows=[]
for mode in ('endogenous3','posthoc3'):
 for role,stock,size in [('head','heads.f32',262144*4),('projection','postprojection.f32',2816*4)]:
  original=(s/'posthoc-c1'/stock).read_bytes();assert len(original)==3*size
  for repeat in (0,1):
   parts=[(s/mode/f'chain3-repeat{repeat}-step{step}-{role}.f32').read_bytes() for step in range(3)]
   assert all(len(p)==size for p in parts)
   joined=b''.join(parts);assert joined==original, (mode,role,repeat)
   rows.append({'mode':mode,'role':role,'repeat':repeat,'complete_bytes':len(joined),'sha256':hashlib.sha256(joined).hexdigest(),'whole_file_byte_exact_to_original':True})
  for repeat in (0,1):assert (s/mode/f'chain1-repeat{repeat}-step0-{role}.f32').read_bytes()==original[:size]
anchors=array.array('i');anchors.frombytes((s/'posthoc-c1/incoming-anchor.i32').read_bytes());features=(s/'posthoc-c1/incoming-feature.f32').read_bytes();assert features[:2816*4]==(s/'stage0-owner0/feature.f32').read_bytes()
assert features[2816*4:]==(s/'posthoc-c1/postprojection.f32').read_bytes()[:2*2816*4]
result={'schema':1,'frozen_utc':datetime.datetime.now(datetime.UTC).isoformat(),'native_own_receipt':ident(s/'native-own3-frozen.json'),'posthoc_manifest':ident(s/'posthoc-c1-owner0-manifest.json'),'binary':pre['binary'],'constant_P':64,'owners':1,'steps':3,'endogenous_and_posthoc_complete_byte_exact_to_original':True,'argmax_mismatches':0,'max_raw_head_delta':0,'max_raw_projection_delta':0,'distribution_delta':0,'own_repeat_movement':0,'cache_unchanged_sha256':cache[0],'comparisons':rows,'official_posthoc_retirement':{n:ident(job/n) for n in ('log','final.json','steps.json')},'limits':'One frozen C1 original-input fixture only; no native-target chain/reference comparison, optimized C2, performance, serving or speculation qualification.'}
with (s/'comparison-c1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print('C1_COMPLETE_MATCH steps3/fullheads/projections; endogenous and posthoc exact; one-step exact; cache unchanged')
print('comparison_sha256',ident(s/'comparison-c1.json')['sha256'])
