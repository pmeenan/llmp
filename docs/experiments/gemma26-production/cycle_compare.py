# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib,json,math,pathlib,re,statistics,sys
p=pathlib.Path(sys.argv[1]); owners=int(sys.argv[2]); profile=sys.argv[3]
assert owners in (1,4) and profile in ('gemma26','gemma31')
part=sys.argv[4] if len(sys.argv)==5 else 'all'
assert part in ('all','first')
result={};owned={}
for arm in (('N1','F1') if part=='first' else ('N1','F1','F2','N2')):
    log=(p/arm/'run.log').read_text()
    assert log.count('SERVING_PROOF_RETIRED')==1
    assert log.count('FRONTIER_CONSTRUCTOR_CONTROL full_final_ffn='+str(int(arm.startswith('F')))+' slots='+str(owners)+' head_capacity='+str(owners))==1
    cycles={}
    for line in log.splitlines():
        if not line.startswith('SERVING_CYCLE '): continue
        row=dict(re.findall(r'(\w+)=([^ ]+)',line));assert row['owners']==str(owners) and row['emitted']==str(129*owners) and row['waves']=='128' and row['cursor']=='8191'
        cycles[row['phase']]={k:float(row[k+'_seconds']) for k in ('cycle','prefill','decode')}
        assert all(math.isfinite(x) and x>0 for x in cycles[row['phase']].values()), 'invalid timing'
    assert set(cycles)=={'warm','second','third'}
    token_bytes=[];history_bytes=[];token_shas=[]
    for owner in range(owners):
        tokens=[];history=[]
        for phase in ('warm','second','third'):
            directory=p/arm/'output'/phase
            t=(directory/('tokens-'+str(owner)+'.i32')).read_bytes()
            h=(directory/('history-'+str(owner)+'.i32')).read_bytes()
            assert len(t)==129*4 and len(h)==8191*4
            tokens.append(t);history.append(h)
        assert tokens[0]==tokens[1]==tokens[2] and history[0]==history[1]==history[2], 'within-arm ownrepeat differs'
        token_sha=hashlib.sha256(tokens[0]).hexdigest()
        if arm.startswith('N') and profile=='gemma26' and owners==1:
            assert token_sha=='840682720df471dedb501c3e2d26edceb2f48f206857b5d503453605e9cd3c9d', 'retained C1 baseline tokenidentity changed'
        token_bytes.append(tokens[0]);history_bytes.append(history[0]);token_shas.append(token_sha)
    owned[arm]=(token_bytes,history_bytes)
    result[arm]={'cycles':cycles,'token_sha256_by_owner':token_shas,'own_repeat_exact':True}
if part=='first':
    (p/'first-own-repeat.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FIRST_HALF_OWN_REPEAT_FROZEN arms=2 allowners=exact phases=3')
    sys.exit(0)
assert owned['N1']==owned['N2'] and owned['F1']==owned['F2'], 'across-process samepolicy repeat differs'
summary={'arms':result,'order':['N1','F1','F2','N2'],'scope':profile+' C'+str(owners)+' actual serving8063+128, token-only warm/second/third, no added head publication', 'cross_policy_tokens_equal':owned['N1'][0]==owned['F1'][0]}
summary['mean_seconds']={mode:{phase:{metric:statistics.mean(result[a]['cycles'][phase][metric] for a in arms) for metric in ('cycle','prefill','decode')} for phase in ('second','third')} for mode,arms in (('narrow',('N1','N2')),('full',('F1','F2')))}
(p/'timing-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
