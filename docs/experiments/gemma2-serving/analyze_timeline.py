#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze the authenticated Gemma2 short C2 timeline pair.

Usage: analyze_timeline.py PRIVATE_CAPTURE_DIRECTORY
Inputs: capture.json and native/stock.sqlite from the reviewed one-pair capture.
Writes a new attribution.json. Raw traces/outputs stay outside the repository.
"""
import collections
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
capture = json.loads((root / 'capture.json').read_text())
assert capture['capture_fidelity_passed']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def union_ns(intervals):
    total = 0
    reach = None
    for start, end in sorted(intervals):
        assert end >= start
        total += max(0, end - max(start, reach if reach is not None else start))
        reach = max(end, reach if reach is not None else end)
    return total

def aggregate(kernels, copies):
    keys = collections.defaultdict(lambda: [0, 0])
    for k in kernels:
        key = f"{k['name']} grid={k['gridX']},{k['gridY']},{k['gridZ']}"
        keys[key][0] += 1
        keys[key][1] += (k['end'] - k['start']) / 1e6
    start, end = kernels[0]['start'], max(k['end'] for k in kernels)
    intervals = [(k['start'], k['end']) for k in kernels]
    holes = []
    reach = kernels[0]['end']; before = kernels[0]['name']
    for k in kernels[1:]:
        if k['start'] > reach:
            holes.append(dict(us=(k['start'] - reach) / 1000, before=before, after=k['name']))
        if k['end'] > reach: reach, before = k['end'], k['name']
    copy_keys = collections.defaultdict(lambda: [0, 0, 0])
    for c in copies:
        key = f"kind={c['copyKind']} bytes={c['bytes']}"
        copy_keys[key][0] += 1
        copy_keys[key][1] += (c['end'] - c['start']) / 1e6
        copy_keys[key][2] += c['bytes']
    return dict(kernel_count=len(kernels), first_kernel_ns=start, last_kernel_ns=end,
                device_span_ms=(end-start)/1e6, kernel_union_ms=union_ns(intervals)/1e6,
                summed_kernel_ms=sum(k['end']-k['start'] for k in kernels)/1e6,
                idle_without_kernels_ms=((end-start)-union_ns(intervals))/1e6,
                keys=dict(keys), copies=dict(copy_keys), longest_kernel_free_holes_including_inter_step=sorted(holes,key=lambda h:-h['us'])[:12])

result = {'scope': 'one pair of profiled whole processes; not a new performance qualification',
          'base': '424990d', 'capture_sha256': sha(root/'capture.json'),
          'script_sha256': sha(Path(__file__)), 'arms': {},
          'boundary_method': 'Exactly40 grid128000 Q8 vocabulary projections per engine:8 warm+32 paid by source loops. Final32 each have26 core C2 attention calls, complete final-logit/sampling tail and no other vocabulary projection. Native32 and stock31 correlate to full cudaGraphLaunch node groups; stock first paid unit is eager and is bounded after preceding scalar terminal argmax and through its two terminal sampling argmax calls. No idle-gap/span heuristic.',
          'limits': 'Kernel-free gaps may contain copies or host work, not idle CPU. Kernel durations overlap and are instrumented. Kernel sums are not wall time; copy kernels do not imply host/device memcpy. GPU spans omit host work before first kernel and after terminal output. Counts/structural differences support a causal test but do not prove a removable latency benefit.'}
for arm in ('native','stock'):
    assert sha(root/(arm+'.sqlite')) == capture['files'][arm+'.sqlite']
    with sqlite3.connect('file:'+str(root/(arm+'.sqlite'))+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        kernels=[dict(x) for x in db.execute('select k.*,s.value name,d.value demangled from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName join StringIds d on d.id=k.demangledName order by k.start')]
        calls=[dict(x) for x in db.execute("select a.*,s.value name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId order by a.start")]
        copies=[dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY order by start')]
    launches={a['correlationId']:a for a in calls if a['name']=='cudaGraphLaunch_v10000'}
    bycorr=collections.defaultdict(list)
    for k in kernels: bycorr[k['correlationId']].append(k)
    heads=[i for i,k in enumerate(kernels) if k['name']=='mul_mat_vec_q' and k['gridX']==128000]
    assert len(heads)==40
    assert all(kernels[i]['gridY']==kernels[i]['gridZ']==1 and 'mul_mat_vec_q<(ggml_type)8, (int)2,' in kernels[i]['demangled'] for i in heads)
    paid=[]; records=[]
    for step, index in enumerate(heads[-32:]):
        head=kernels[index];corr=head['correlationId']
        if corr in launches:
            group=bycorr[corr]
            assert all(k['graphNodeId'] is not None for k in group)
            graph=launches[corr]
            mode='graph'
        else:
            assert arm=='stock' and step==0
            # Previous head is a scalar (one owner), with softcap/pad/one argmax.
            previous=max(j for j in range(index) if kernels[j]['name']=='mul_mat_vec_q' and kernels[j]['gridX']>=64000)
            assert kernels[previous]['gridX']==256000
            assert [k['name'] for k in kernels[previous+1:previous+4]]==['softcap_f32','pad_f32','argmax_f32']
            begin=previous+4
            assert kernels[begin]['name']=='scale_f32' and kernels[begin]['gridX']==18
            assert [k['name'] for k in kernels[index+1:index+5]]==['softcap_f32','pad_f32','argmax_f32','argmax_f32']
            group=kernels[begin:index+5]
            assert all(k['graphNodeId'] is None for k in group)
            graph=None;mode='eager'
        assert sum(k['name']=='mul_mat_vec_q' and k['gridX']==128000 for k in group)==1
        core='flash_attn_owner_f16' if arm=='native' else 'flash_attn_ext_f16'
        attention=[k for k in group if k['name']==core]
        assert len(attention)==26
        assert all((k['gridX'],k['gridY'],k['gridZ'])==(48,1,1) for k in attention)
        assert all('<(int)256, (int)256, (int)4, (int)2, (bool)1,' in k['demangled'] for k in attention)
        if arm=='native':
            assert sum(k['name']=='ArgmaxKernel' for k in group)==(step!=31)
            assert sum(k['name']=='concat_cont' and k['gridX']==4096 for k in group)==52
        else:
            assert sum(k['name']=='argmax_f32' for k in group)==2
        paid.extend(group)
        records.append(dict(step=step,mode=mode,projection_correlation=corr,projection_ns=[head['start'],head['end']],
                            first_kernel_ns=group[0]['start'],last_kernel_ns=max(k['end'] for k in group),
                            kernels=len(group),core_attention=26,
                            graph_api_ns=[graph['start'],graph['end']] if graph else None))
    assert len(paid)==len({(k['start'], k['end'], k['streamId'], k['correlationId'], k['graphNodeId']) for k in paid})
    assert all(a['last_kernel_ns']<b['first_kernel_ns'] for a,b in zip(records,records[1:]))
    # Last warm unit's complete tail, then seven prefill waves ending at the
    # second one-row vocabulary publication. Remaining six scalar heads are off-clock.
    warm_last=heads[7]
    warm_tail=3 if arm=='native' else 4
    expected_tail=['scale_f32','unary_op_kernel','scale_f32'] if arm=='native' else ['softcap_f32','pad_f32','argmax_f32','argmax_f32']
    assert [k['name'] for k in kernels[warm_last+1:warm_last+1+warm_tail]]==expected_tail
    scalar_heads=[i for i in range(warm_last+1,heads[8]) if kernels[i]['name']=='mul_mat_vec_q' and kernels[i]['gridX']==256000]
    assert len(scalar_heads)==8
    prefill_end=scalar_heads[1]+(4 if arm=='native' else 3)
    prefill=kernels[warm_last+1+warm_tail:prefill_end+1]
    assert sum(k['name']=='flash_attn_ext_f16' for k in prefill)==(177 if arm=='native' else 182)
    assert not any(k['name']=='flash_attn_owner_f16' for k in prefill)
    assert sum(k['name']=='mul_mat_vec_q' and k['gridX']==256000 for k in prefill)==2
    # Copies are selected by independent phase bounds; graph-correlated copies
    # and output traffic after the last kernel are reported separately.
    start,end=paid[0]['start'],max(k['end'] for k in paid)
    phase_copies=[c for c in copies if start<=c['start']<=end]
    post=[c for c in copies if c['start']>end]
    graph_copies=[c for c in copies if c['correlationId'] in {r['projection_correlation'] for r in records if r['mode']=='graph'}]
    summary=aggregate(paid,phase_copies)
    summary['steps']=records
    summary['per_step_device_span_ms']=[(r['last_kernel_ns']-r['first_kernel_ns'])/1e6 for r in records]
    summary['inter_step_gap_ms']=[(b['first_kernel_ns']-a['last_kernel_ns'])/1e6 for a,b in zip(records,records[1:])]
    summary['graph_api_sum_ms']=sum((r['graph_api_ns'][1]-r['graph_api_ns'][0])/1e6 for r in records if r['graph_api_ns'])
    summary['graph_correlated_copies']=dict(count=len(graph_copies),bytes=sum(c['bytes'] for c in graph_copies))
    if arm=='native':
        final_publication=[c for c in post if c['correlationId']==records[-1]['projection_correlation']]
        assert len(final_publication)==1 and final_publication[0]['copyKind']==2 and final_publication[0]['bytes']==2048000
        excluded=[c for c in post if c not in final_publication]
        assert all(c['bytes']==2097152 and c['start']>final_publication[-1]['end'] for c in excluded)
    else:
        final_publication=post
        assert collections.Counter((c['copyKind'],c['bytes']) for c in final_publication)=={(2,4):2,(2,1024000):2}
        copy_calls={a['correlationId']:a for a in calls if 'MemcpyAsync' in a['name']}
        assert all(c['correlationId'] in copy_calls for c in final_publication)
        excluded=[]
    summary['final_publication_copies']=[{k:c[k] for k in ['start','end','bytes','copyKind','correlationId']} for c in final_publication]
    summary['excluded_after_final_publication']=dict(count=len(excluded),bytes=sum(c['bytes'] for c in excluded),scope='later teardown/state-spill traffic; excluded from paid decode')
    summary['kernel_templates']={key:sorted({k['demangled'] for k in paid if f"{k['name']} grid={k['gridX']},{k['gridY']},{k['gridZ']}"==key}) for key in summary['keys']}

    summary['prefill']=aggregate(prefill,[c for c in copies if prefill[0]['start']<=c['start']<=prefill[-1]['end']])
    summary['prefill']['scope_note']='seven prompt waves, two frontier heads; five state-only waves skip native final attention (tail predicate), stock computes182 vs native177 attention calls. Bounds are GPU kernels only, excluding first-input and final publication host work.'
    result['arms'][arm]=summary
with (root/'attribution.json').open('x') as f:json.dump(result,f,indent=2)
for arm,a in result['arms'].items():
 print(arm, 'paid32',[(r['mode'],r['kernels']) for r in (a['steps'][0],a['steps'][-1])],
       'span_ms',a['device_span_ms'],'kernel_union_ms',a['kernel_union_ms'],'sum_ms',a['summed_kernel_ms'],
       'prefill_span_ms',a['prefill']['device_span_ms'])
for key in sorted(set(result['arms']['native']['keys'])|set(result['arms']['stock']['keys']),
                  key=lambda key:-abs(result['arms']['native']['keys'].get(key,[0,0])[1]-result['arms']['stock']['keys'].get(key,[0,0])[1])):
 n=result['arms']['native']['keys'].get(key,[0,0]);s=result['arms']['stock']['keys'].get(key,[0,0])
 print(key,n,s,'diff_ms',n[1]-s[1])
