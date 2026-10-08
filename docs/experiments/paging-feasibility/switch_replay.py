#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Matched multi-model offline replay; timing outputs are explicit service scenarios."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from collections import OrderedDict
import json
import hashlib
import re
import math
from pathlib import Path
from run import IMAGE, REVISION, sha
from prediction_checks import validate_prediction_pair, validate_session_predictions

GIB=1<<30
EXTENT=2<<20
READ_BPS=14.962131e9
WRITE_BPS=11.641e9
OVERHEAD=3*GIB+8*(1<<20)  # scratch1GiB, overhead/headroom2GiB, four destination slots


def mark_spill_recompute(models, names):
    byid={m['id']:m for m in models}
    if set(names)-byid.keys():raise ValueError('Unknown spill-recompute model')
    for name in names:byid[name]['spill_requires_recompute']=True


def state_allocation(model, normal=False):
    return model['normal_kv_bytes'] if normal else model['kv_bytes']


def read_capture_log(root, run):
    path=Path(root)/(run['name']+'.log')
    if 'log_sha256' in run and sha(path)!=run['log_sha256']:
        raise ValueError('Log hash mismatch')
    return path.read_text(errors='replace')


def log_allocations(log):
    # Distinct KV partitions and recurrent/DSV4 state are additive.
    states=re.findall(r'(?:KV|RS|DSV4 \w+ state) buffer size\s*=\s*([0-9.]+) MiB',log)
    if not states:raise ValueError('Missing state allocation')
    buffers={}
    for device,kind,size in re.findall(r'(CUDA\d+|CUDA_Host|CPU)\s+(compute|output) buffer size\s*=\s*([0-9.]+) MiB',log):
        buffers[device,kind]=max(buffers.get((device,kind),0),float(size)*(1<<20))
    if not buffers:raise ValueError('Missing workspace accounting')
    return (math.ceil(sum(map(float,states))*(1<<20)/EXTENT)*EXTENT,
            math.ceil(sum(buffers.values())/EXTENT)*EXTENT)


def quantiles(values):
    values=sorted(values)
    if not values:return dict(count=0,mean=0,p50=0,p95=0,max=0)
    return dict(count=len(values),mean=sum(values)/len(values),p50=values[(len(values)-1)//2],
                p95=values[int((len(values)-1)*.95)],max=values[-1])


def route_shape_matches(record, tokens, phase, layers):
    flag=record.get('output_only',False)
    if type(flag) is not bool or record['phase']!=phase:return False
    if flag:
        return phase=='prefill' and tokens>1 and record['layer']==layers-1 and len(record['routes'])==1
    return len(record['routes'])==tokens


def load_events(path,run,model):
    if sha(path)!=run['events_sha256']:raise ValueError('Event hash mismatch')
    requests=[];active=None;layers=[];step=0;steps=0;count=0
    with path.open() as stream:
        for line in stream:
            if len(line)>512*1024:raise ValueError('Oversized record')
            r=json.loads(line);event=r['event']
            if event=='request':
                if active is not None or r['request']!=len(requests):raise ValueError('Request ordering')
                if not 0<=r['reused']<r['prompt'] or r['prompt']+r['decode']>model['context']:raise ValueError('Request bounds')
                active=dict(r,groups=[],step_us=[],state_bytes=None);step=0;count=0;steps=0
            elif event=='routes':
                if active is None or r['request']!=active['request'] or r['step']!=step or r['layer']!=len(layers):raise ValueError('Route order')
                if not 0<=r['layer']<model['layers']:raise ValueError('Layer out of range')
                ids=r['routes']
                if not ids or len(ids)>run['batch']:raise ValueError('Token count')
                for selected in ids:
                    if len(selected)!=model['topk'] or len(set(selected))!=len(selected) or any(type(n)is not int or not 0<=n<model['experts'] for n in selected):raise ValueError('Expert range/uniqueness')
                if layers and not route_shape_matches(r,len(layers[0]['routes']),layers[0]['phase'],model['layers']):raise ValueError('Inconsistent layer group')
                layers.append(r)
            elif event=='step':
                if active is None or r['request']!=active['request'] or r['step']!=step:raise ValueError('Step order')
                if run['trace'] and len(layers)!=model['layers']:raise ValueError('Missing routed layers')
                expected='prefill' if count<active['prompt']-active['reused'] else 'decode'
                remaining=active['prompt']-active['reused']-count
                n=min(run['batch'],remaining) if expected=='prefill' else 1
                if r['phase']!=expected or r['tokens']!=n or r['us']<0:raise ValueError('Step shape')
                if any(not route_shape_matches(r,n,expected,model['layers']) for r in layers):raise ValueError('Step/route mismatch')
                active['groups'].extend(layers);active['step_us'].append(dict(phase=expected,tokens=n,us=r['us']))
                layers=[];step+=1;count+=n;steps+=1
            elif event=='end':
                if active is None or r['request']!=active['request'] or layers or count!=active['prompt']-active['reused']+active['decode']:raise ValueError('Incomplete request')
                if r['history']!=active['prompt']+active['decode'] or r['state_bytes']<=0:raise ValueError('State receipt')
                active['state_bytes']=r['state_bytes'];active['history']=r['history'];requests.append(active);active=None
            else:raise ValueError('Unknown event')
    if active is not None or not requests:raise ValueError('Incomplete session')
    return requests


class Layout:
    def __init__(self,models,packing):
        self.packing=packing;self.models={m['id']:m for m in models};self.sizes={};self.closures={};self.weights={}
        for m in models:
            name=m['id'];base=(name,'base',0)
            self.sizes[base]=math.ceil(m['nonexpert_bytes']/EXTENT)*EXTENT
            allkeys={base};offset=0
            for layer,logical in enumerate(m['expert_bytes']):
                for expert in range(m['experts']):
                    start=offset
                    end=start+logical
                    keys={(name,'expert',n) for n in range(start//EXTENT,(end+EXTENT-1)//EXTENT)}
                    self.closures[name,layer,expert]=keys
                    for key in keys:self.sizes[key]=EXTENT
                    allkeys.update(keys)
                    offset=((end+EXTENT-1)//EXTENT)*EXTENT if packing=='isolated' else end
                offset=((offset+EXTENT-1)//EXTENT)*EXTENT
            self.weights[name]=allkeys

    def group(self,name,record):
        cache_key='_demands_'+name+'_'+self.packing
        if cache_key in record:return record[cache_key]
        result={(name,'base',0)}
        for ids in record['routes']:
            for expert in ids:result.update(self.closures[name,record['layer'],expert])
        record[cache_key]=frozenset(result)
        return record[cache_key]


class Cache:
    def __init__(self,layout,budget):self.layout=layout;self.budget=budget;self.cache=OrderedDict();self.used=0;self.peak=0
    def discard(self,keys):
        for key in keys:
            if key in self.cache:self.used-=self.cache.pop(key)
    def missing_bytes(self,required):
        return sum(self.layout.sizes[k]-self.cache.get(k,0) for k in required)
    def ensure(self,required,fixed):
        need=sum(self.layout.sizes[k] for k in required)
        if fixed+need>self.budget:raise ValueError('Execution envelope does not fit')
        incoming=self.missing_bytes(required)
        while self.used+incoming+fixed>self.budget:
            key=next(iter(self.cache))
            if key in required:
                self.cache.move_to_end(key)
                continue
            # Base weights share an access time, but retain extent-level eligibility.
            # A partially reclaimed base keeps its old age until actually accessed.
            deficit=self.used+incoming+fixed-self.budget
            reclaim=min(self.cache[key],((deficit+EXTENT-1)//EXTENT)*EXTENT)
            self.cache[key]-=reclaim;self.used-=reclaim
            if not self.cache[key]:del self.cache[key]
        for key in sorted(required):
            self.used+=self.layout.sizes[key]-self.cache.get(key,0)
            self.cache[key]=self.layout.sizes[key]
            self.cache.move_to_end(key)
        self.peak=max(self.peak,self.used+fixed)
        return incoming


def simulate(models,captures,timeline,budget,policy,state,packing='isolated',normal=False):
    layout=Layout(models,packing);cache=Cache(layout,budget)
    allocated=set();saved={};saved_history={};disk={};previous=None;results=[];peak_spill=0
    try:
        first=timeline[0]['model'];initial=layout.models[first]
        # Matched starting point: full first-model weights before the first turn.
        initial_fixed=OVERHEAD+max(0,initial.get('workspace_bytes',0)-GIB)+state_allocation(initial,normal)
        warm_start=initial_fixed+sum(layout.sizes[k] for k in layout.weights[first])<=budget
        if warm_start:cache.ensure(layout.weights[first],initial_fixed)
        for event in timeline:
            name=event['model'];turn=event['turn'];m=layout.models[name]
            outgoing=previous if previous!=name else None
            written=0;restored=0
            if state=='spill' and outgoing is not None:
                written=saved[outgoing];disk[outgoing]=written;allocated.discard(outgoing)
                peak_spill=max(peak_spill,sum(disk.values()))
                if peak_spill>8*GIB:raise ValueError('8 GiB spill limit exceeded')
            allocated.add(name)
            if state=='spill' and name in disk:
                restored=disk.pop(name)
            fixed=OVERHEAD+max(0,m.get('workspace_bytes',0)-GIB)+sum(state_allocation(layout.models[key],normal) for key in allocated)
            variant='normal_retained' if normal else 'retained'
            req=captures[name,variant][turn]
            fallback=None
            if name=='A' and normal and not all('coverage_reset' in r for r in captures[name,'normal_retained']):
                # Legacy normal captures lacked the SWA coverage guard; never use
                # their apparent reuse as evidence that old window cells exist.
                variant='normal_recompute';fallback='legacy normal capture lacks SWA coverage validation'
            if name=='A' and restored and req['reused']<saved_history[name]-1:
                variant='normal_recompute' if normal else 'recompute'
                fallback='sequence snapshot lacks SWA cells needed after prefix rollback'
            if restored and m.get('spill_requires_recompute',False):
                variant='normal_recompute' if normal else 'recompute'
                fallback='explicit conservative policy: restored continuation equivalence unvalidated'
            req=captures[name,variant][turn]
            if state=='recompute':
                allocated={name};fixed=OVERHEAD+max(0,m.get('workspace_bytes',0)-GIB)+state_allocation(m,normal)
                variant='normal_recompute' if normal else 'recompute';req=captures[name,variant][turn]
            preloaded=0
            if policy in ('whole','retained_eager'):
                required=layout.weights[name]
                # Preserve both complete models when they fit; otherwise evict whole inactive models.
                while policy=='whole' and cache.used+cache.missing_bytes(required)+fixed>budget:
                    candidates=[key for key in cache.cache if key[0]!=name]
                    if not candidates:break
                    victim=candidates[0][0];cache.discard(layout.weights[victim])
                preloaded=cache.ensure(required,fixed)
            else:cache.ensure(set(),fixed)
            reads=preloaded;prefill=preloaded;decode={};first_bytes=preloaded
            for r in req['groups']:
                amount=cache.ensure(layout.group(name,r),fixed);reads+=amount
                if r['phase']=='prefill':prefill+=amount;first_bytes+=amount
                else:decode[r['step']]=decode.get(r['step'],0)+amount
            # Pinned native sequence-file header: three uint32s, then token IDs and state.
            saved[name]=req['state_bytes']+12+4*(req['prompt']+req['decode']);saved_history[name]=req['prompt']+req['decode'];previous=name
            bytes_per_token=list(decode.values())
            transfer=first_bytes+restored
            write_seconds=written/WRITE_BPS+(.003838 if written else 0)
            row=dict(model=name,turn=turn,arrival_s=event['arrival_s'],capture_variant=variant,state_reuse_fallback=fallback,reused=req['reused'],prefill_tokens=req['prompt']-req['reused'],
                     weight_read_bytes=reads,first_token_weight_bytes=first_bytes,prefill_weight_bytes=prefill,
                     state_written_bytes=written,state_restored_bytes=restored,state_snapshot_bytes=saved[name],state_payload_bytes=req['state_bytes'],
                     spill_occupancy_bytes=sum(disk.values()),decode_miss_bytes=quantiles(bytes_per_token),
                     first_token_storage_seconds=transfer/READ_BPS+write_seconds,
                     first_token_serial_storage_seconds=(math.ceil(first_bytes/EXTENT)*.000177+restored/READ_BPS+write_seconds),
                     first_token_slow_storage_seconds=transfer/1e9+written/1e9+(.003838 if written else 0),
                     decode_no_overlap_ms=quantiles([x/READ_BPS*1000 for x in bytes_per_token]),
                     decode_half_overlap_ms=quantiles([x/READ_BPS*500 for x in bytes_per_token]))
            results.append(row)
        return dict(feasible=True,budget_bytes=budget,policy=policy,state=state,packing=packing,normal=normal,
                    warm_start=warm_start,peak_occupancy_bytes=cache.peak,peak_spill_bytes=peak_spill,requests=results)
    except ValueError as e:
        return dict(feasible=False,budget_bytes=budget,policy=policy,state=state,packing=packing,normal=normal,reason=str(e))


WORK=None

def simulate_job(args):
    return simulate(*WORK,*args)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('capture',type=Path);p.add_argument('sessions',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--allow-prediction-drift',action='append',default=[],metavar='MODEL_ID',help='Explicitly accept recorded failed equivalence for conditional same-trajectory replay')
    p.add_argument('--spill-recompute',action='append',default=[],metavar='MODEL_ID',help='Conservatively recompute flagged model prompts after every state restore')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--extra-capture',action='append',type=Path,default=[])
    a=p.parse_args();receipt=json.loads((a.capture/'receipt.json').read_text());spec=receipt['spec'];models=spec['models']
    if not receipt.get('cuda_disable_fusion') or not receipt.get('cuda_disable_graphs') or receipt['image']!=IMAGE or receipt['revision']!=REVISION or sha(a.sessions)!=spec['trace_sha256']:raise ValueError('Provenance mismatch')
    for r in receipt['runs']:r['root']=str(a.capture)
    for extra in a.extra_capture:
        added=json.loads((extra/'receipt.json').read_text())
        if not added.get('cuda_disable_fusion') or not added.get('cuda_disable_graphs') or added['image']!=IMAGE or added['revision']!=REVISION or added['spec']['models']!=models or added['spec']['trace_sha256']!=spec['trace_sha256']:raise ValueError('Extra capture provenance mismatch')
        for r in added['runs']:r['root']=str(extra)
        receipt['runs'].extend(added['runs']);spec['configurations'].extend(added['spec']['configurations'])
    mark_spill_recompute(models,a.spill_recompute)
    if set(a.allow_prediction_drift)-{m['id'] for m in models}:raise ValueError('Unknown prediction-drift model')
    prediction_checks=[]
    sessions=json.loads(a.sessions.read_text());captures={};byid={m['id']:m for m in models};summaries=[]
    for m in models:
        requests=[r for r in sessions['requests'] if r['model']==m['id']]
        text=str(len(requests))+'\n'
        for r in requests:
            text+=f"{len(r['prompt'])} {len(r['expected_tokens'])-1}\n"+' '.join(map(str,r['prompt']+r['expected_tokens'][:-1]))+'\n'
        if hashlib.sha256(text.encode()).hexdigest()!=m['tokens_sha256']:raise ValueError('Token file/content mismatch')
    expected={(m['id'],b,reuse,swa,trace) for m in models for b,reuse,swa in spec['configurations'] for trace in (False,True)}
    found={(r['model'],r['batch'],r['reuse'],r['swa'],r['trace']) for r in receipt['runs']}
    if len(receipt['runs'])!=len(expected) or found!=expected:raise ValueError('Incomplete/duplicate capture matrix')
    paired={(r['model'],r['batch'],r['reuse'],r['swa'],r['trace']):r for r in receipt['runs']}
    for key,r in paired.items():
        if key[-1]:
            control=paired[(*key[:-1],False)]
            comparison=validate_prediction_pair(control,r,Path(control['root'])/(control['name']+'.predictions'),Path(r['root'])/(r['name']+'.predictions'),r['model'] in a.allow_prediction_drift)
            prediction_checks.append(dict(model=r['model'],run=r['name'],**comparison))
    for m in models:
        m['prediction_equivalence']='failed_conditional_replay' if any(x['model']==m['id'] and x['different_records'] for x in prediction_checks) else 'exact_top1_match'
        m['prediction_drift_explicitly_allowed']=m['id'] in a.allow_prediction_drift
    for r in receipt['runs']:
        events=load_events(Path(r['root'])/(r['name']+'.jsonl'),r,byid[r['model']])
        if sha(Path(r['root'])/(r['name']+'.predictions'))!=r['predictions_sha256']:raise ValueError('Predictions mismatch')
        expected_requests=[x for x in sessions['requests'] if x['model']==r['model']]
        if len(events)!=len(expected_requests) or any(e['prompt']!=len(x['prompt']) or e['decode']!=len(x['expected_tokens'])-1 for e,x in zip(events,expected_requests)):raise ValueError('Capture/input request mismatch')
        validate_session_predictions(Path(r['root'])/(r['name']+'.predictions'),events)
        summaries.append(dict(name=r['name'],requests=[{k:v for k,v in req.items() if k not in ('groups','step_us')} for req in events],
                              decode_us=quantiles([s['us'] for req in events for s in req['step_us'] if s['phase']=='decode'])))
        if r['trace'] and r['batch']==512:
            variant=('normal_retained' if r['reuse'] else 'normal_recompute') if not r['swa'] else ('retained' if r['reuse'] else 'recompute')
            captures[r['model'],variant]=events
    if set(captures)!={(m['id'],v) for m in models for v in ('normal_retained','normal_recompute','retained','recompute')}:raise ValueError('Missing capture variants')
    timeline=[dict(model=r['model'],turn=r['turn'],arrival_s=r['arrival_s']) for r in sessions['requests']]
    for m in models:
        maximum=0;state_by_swa={False:0,True:0}
        m['input_spec_kv_bytes']=m['kv_bytes']
        for r in receipt['runs']:
            if r['model']!=m['id']:continue
            log=read_capture_log(r['root'],r)
            state_bytes,workspace_bytes=log_allocations(log)
            state_by_swa[bool(r['swa'])]=max(state_by_swa[bool(r['swa'])],state_bytes)
            maximum=max(maximum,workspace_bytes)
        if not all(state_by_swa.values()):raise ValueError('Missing full/default SWA allocation')
        m['kv_bytes']=state_by_swa[True];m['normal_kv_bytes']=state_by_swa[False]
        m['workspace_bytes']=maximum
        m['allocation_basis']='Maximum logged native allocations by SWA mode, rounded up to 2 MiB; input spec KV retained separately'
    budgets=[28,32,36,40,41.6877,44,48,64] if set(byid)=={'A','B'} else [96,112,121.6877,144,192,256]
    jobs=[(int(b*GIB),policy,state,packing,normal) for b in budgets
          for policy in ('partial','retained_eager','whole') for state in ('resident','spill','recompute')
          for packing in ('isolated','packed') for normal in (False,True)]
    global WORK
    WORK=(models,captures,timeline)
    if a.workers<1 or a.workers>8:raise ValueError('Worker bound')
    if a.workers==1:results=[simulate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('fork')) as pool:
            results=list(pool.map(simulate_job,jobs))
    with a.output.open('x') as f:json.dump(dict(models=models,prediction_checks=prediction_checks,captures=summaries,simulations=results),f,indent=2);f.write('\n')


if __name__=='__main__':main()
