#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-layer locality and held-out next-layer prediction; no prefetch speedup claim."""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
from switch_replay import load_events, quantiles, Layout, READ_BPS
from prediction_checks import validate_prediction_pair, validate_session_predictions


def analyze(requests,model):
    layers=model['layers'];experts=model['experts'];topk=model['topk']
    seen=[[-1]*experts for _ in range(layers)];clock=[0]*layers
    histories={p:[Counter() for _ in range(layers)] for p in ('prefill','decode')}
    cold={p:[0]*layers for p in histories};union={p:[] for p in histories}
    train_pairs=[[Counter() for _ in range(experts)] for _ in range(layers)]
    popularity=[Counter() for _ in range(layers)]
    split=max(1,len(requests)//2);hits=static=last_hits=total=0;prediction_bytes=useful_bytes=0
    byte_per_expert=[((n+(2<<20)-1)//(2<<20))*(2<<20) for n in model['expert_bytes']]
    service=[]
    for request_index,req in enumerate(requests):
        bystep={};last=[set() for _ in range(layers)]
        for r in req['groups']:
            layer=r['layer'];phase=r['phase'];union[phase].append(len({e for ids in r['routes'] for e in ids}))
            for ids in r['routes']:
                times=seen[layer];ordered=sorted(times)
                for n in ids:
                    if times[n]<0:cold[phase][layer]+=1
                    else:histories[phase][layer][experts-bisect_right(ordered,times[n])]+=1
                for n in ids:times[n]=clock[layer]
                clock[layer]+=1
            if phase!='decode':
                last[layer]=set(r['routes'][-1])
                continue
            if len(r['routes'])!=1:raise ValueError('This predictor needs single-sequence decode')
            actual=set(r['routes'][0]);bystep[layer]=actual
            if request_index<split:
                popularity[layer].update(actual)
                if layer:
                    for e in bystep[layer-1]:train_pairs[layer][e].update(actual)
            elif layer:
                score=Counter()
                for e in bystep[layer-1]:score.update(train_pairs[layer][e])
                predicted=set(sorted(range(experts),key=lambda e:(-score[e],e))[:topk])
                baseline=set(sorted(range(experts),key=lambda e:(-popularity[layer][e],e))[:topk])
                hit=len(predicted&actual);hits+=hit;static+=len(baseline&actual)
                last_hits+=len(last[layer]&actual);total+=topk
                n=byte_per_expert[layer]
                prediction_bytes+=len(predicted)*n;useful_bytes+=hit*n
                # Cold-target service requirement, not cache-aware prefetch traffic.
                service.append(len(predicted)*n/READ_BPS*1000)
            last[layer]=actual
    perlayer=[]
    for layer in range(layers):
        row={'layer':layer}
        for phase in histories:
            h=histories[phase][layer];count=sum(h.values())
            def q(frac):
                target=int((count-1)*frac)
                for value,n in sorted(h.items()):
                    if target<n:return value
                    target-=n
                return None
            row[phase]=dict(cold=cold[phase][layer],reused=count,p50=q(.5),p95=q(.95))
        perlayer.append(row)
    return dict(per_layer=perlayer,selected_union={p:quantiles(v) for p,v in union.items()},
                predictor=dict(training_requests=split,test_requests=len(requests)-split,
                               selected_contributions=total,next_layer_recall=hits/total if total else None,
                               static_recall=static/total if total else None,previous_token_recall=last_hits/total if total else None,
                               cold_prediction_bytes=prediction_bytes,useful_selected_bytes=useful_bytes,
                               cold_prediction_service_ms=quantiles(service),
                               lead_time_coverage={str(ms):sum(t<=ms for t in service)/len(service) if service else None for ms in (.1,.5,1.0)},
                               caveat='Cold-target isolated-closure scenario; excludes cache hits, evictions, contention, measured compute lead time'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('capture',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--allow-prediction-drift',action='append',default=[],metavar='MODEL_ID')
    a=p.parse_args();r=json.loads((a.capture/'receipt.json').read_text());models={m['id']:m for m in r['spec']['models']};out=[]
    if set(a.allow_prediction_drift)-models.keys():raise ValueError('Unknown prediction-drift model')
    pairs={(run['model'],run['batch'],run['reuse'],run['swa'],run['trace']):run for run in r['runs']}
    if len(pairs)!=len(r['runs']):raise ValueError('Duplicate capture configuration')
    for run in r['runs']:
        if run['trace'] and run['reuse']==1 and run['swa']==1:
            control=pairs[(run['model'],run['batch'],run['reuse'],run['swa'],False)]
            comparison=validate_prediction_pair(control,run,a.capture/(control['name']+'.predictions'),a.capture/(run['name']+'.predictions'),run['model'] in a.allow_prediction_drift)
            m=models[run['model']];requests=load_events(a.capture/(run['name']+'.jsonl'),run,m)
            for item in (control,run):
                validate_session_predictions(a.capture/(item['name']+'.predictions'),requests)
            out.append(dict(model=m['id'],batch=run['batch'],prediction_comparison=comparison,**analyze(requests,m)))
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')


if __name__=='__main__':main()
