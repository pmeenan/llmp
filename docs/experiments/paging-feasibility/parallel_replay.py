#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate four-sequence capture and replay its actual union of dependencies."""
import argparse
import json
import math
import re
from pathlib import Path
from run import IMAGE, REVISION, sha
from prediction_checks import validate_prediction_records
from switch_replay import Cache, Layout, EXTENT, GIB, OVERHEAD, READ_BPS, quantiles, log_allocations, read_capture_log, route_shape_matches


def input_shape(path, model):
    if sha(path)!=model['tokens_sha256']:raise ValueError('Input hash mismatch')
    tokens=iter(path.read_text().split())
    try:
        if int(next(tokens))!=4:raise ValueError('Need four requests')
        counts=[];rounds=[]
        for _ in range(4):
            n,d=int(next(tokens)),int(next(tokens))
            if n<=0 or d<=0 or n+d>model['context']//4:raise ValueError('Input bounds')
            for _ in range(n+d):int(next(tokens))
            counts.extend([('prefill',min(512,n-pos)) for pos in range(0,n,512)])
            rounds.append(d)
        if next(tokens,None) is not None:raise ValueError('Trailing input')
        return counts+[('decode',4)]*min(rounds)
    except StopIteration as e:raise ValueError('Truncated input') from e


def load(path, run, model, expected=None):
    if sha(path)!=run['events_sha256']:raise ValueError('Event hash mismatch')
    batches=[];layers=[];step=0;ended=False;decode=0
    with path.open() as stream:
        for line in stream:
            if ended:raise ValueError('Trailing events')
            if len(line)>512*1024:raise ValueError('Oversized event')
            r=json.loads(line)
            if r['event']=='routes':
                if r['step']!=step or r['layer']!=len(layers):raise ValueError('Route ordering')
                if len(layers)>=model['layers']:raise ValueError('Extra layer')
                for ids in r['routes']:
                    if len(ids)!=model['topk'] or len(set(ids))!=len(ids) or any(type(e)is not int or not 0<=e<model['experts'] for e in ids):raise ValueError('Invalid expert')
                layers.append(r)
            elif r['event']=='batch':
                if r['step']!=step or not 1<=r['tokens']<=512:raise ValueError('Batch ordering/size')
                if expected is not None and (step>=len(expected) or (r['phase'],r['tokens'])!=expected[step]):raise ValueError('Batch/input mismatch')
                if r['phase'] not in ('prefill','decode') or (r['phase']=='decode' and r['tokens']!=4):raise ValueError('Batch shape')
                if run['trace'] and len(layers)!=model['layers']:raise ValueError('Missing layers')
                if any(not route_shape_matches(x,r['tokens'],r['phase'],model['layers']) for x in layers):raise ValueError('Route batch shape')
                if r['phase']=='decode':decode+=1
                elif decode:raise ValueError('Prefill after decode')
                batches.append(dict(r,groups=layers));layers=[];step+=1
            elif r['event']=='end':
                if layers or r['decode_batches']!=decode or r['batch_size']!=4 or not decode:raise ValueError('Incomplete capture')
                ended=True
            else:raise ValueError('Unknown event')
    if not ended:raise ValueError('Missing end')
    if expected is not None and len(batches)!=len(expected):raise ValueError('Incomplete input trace')
    return batches


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('capture',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--inputs',type=Path,required=True)
    a=p.parse_args();receipt=json.loads((a.capture/'receipt.json').read_text())
    if not receipt.get('parallel') or not receipt.get('cuda_disable_fusion') or not receipt.get('cuda_disable_graphs') or receipt['image']!=IMAGE or receipt['revision']!=REVISION:raise ValueError('Provenance mismatch')
    if len({m['id'] for m in receipt['spec']['models']})!=len(receipt['spec']['models']):raise ValueError('Duplicate model')
    if len(receipt['runs'])!=2*len(receipt['spec']['models']):raise ValueError('Unexpected capture count')
    out=[]
    for m in receipt['spec']['models']:
        expected=input_shape(a.inputs/m['tokens'],m)
        runs=[r for r in receipt['runs'] if r['model']==m['id']]
        if len(runs)!=2 or {r['trace'] for r in runs}!={False,True}:raise ValueError('Missing/extra pair')
        if runs[0]['predictions_sha256']!=runs[1]['predictions_sha256']:raise ValueError('Changed predictions')
        for r in runs:
            if sha(a.capture/(r['name']+'.predictions'))!=r['predictions_sha256']:raise ValueError('Prediction hash')
            if (r['batch'],r['reuse'],r['swa'])!=(512,0,1):raise ValueError('Parallel configuration mismatch')
            b=load(a.capture/(r['name']+'.jsonl'),r,m,expected)
            rounds=sum(phase=='decode' for phase,_ in expected)
            validate_prediction_records(a.capture/(r['name']+'.predictions'),
                                        ((step,seq) for step in range(rounds) for seq in range(4)))
            if r['trace']:batches=b;traced=r
        log=read_capture_log(a.capture,traced)
        state,workspace=log_allocations(log)
        fixed=OVERHEAD+state+max(0,workspace-GIB)
        unions=quantiles([len({e for ids in r['routes'] for e in ids}) for b in batches if b['phase']=='decode' for r in b['groups']])
        curves=[]
        for packing in ('isolated','packed'):
            layout=Layout([m],packing)
            for budget in (16,20,24,28,32,40):
                cache=Cache(layout,budget*GIB);miss=[];prefill=0
                try:
                    for b in batches:
                        amount=sum(cache.ensure(layout.group(m['id'],r),fixed) for r in b['groups'])
                        if b['phase']=='decode':miss.append(amount)
                        else:prefill+=amount
                    curves.append(dict(budget_gib=budget,packing=packing,feasible=True,prefill_read_bytes=prefill,
                                       decode_batch_miss_bytes=quantiles(miss),decode_batch_service_ms=quantiles([n/READ_BPS*1000 for n in miss]),peak_bytes=cache.peak))
                except ValueError as e:curves.append(dict(budget_gib=budget,packing=packing,feasible=False,reason=str(e)))
        out.append(dict(model=m['id'],state_allocation_bytes=state,fixed_bytes=fixed,decode_batches=len([b for b in batches if b['phase']=='decode']),union_experts=unions,curves=curves))
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')


if __name__=='__main__':main()
