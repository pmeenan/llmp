#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate private, template-correct synthetic conversations with pinned reference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('reference_tools',type=Path)
    p.add_argument('models',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--large-pins',type=Path)
    p.add_argument('--port',type=int,default=18921)
    a=p.parse_args(); os.umask(0o077)
    def interrupted(signum,frame): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    spec=importlib.util.spec_from_file_location('aba',a.reference_tools/'experiment.py')
    aba=importlib.util.module_from_spec(spec); spec.loader.exec_module(aba)
    pins=json.loads((a.reference_tools/'artifacts.json').read_text())
    names={'A':'A','B':'B'}
    if a.large_pins:
        large=json.loads(a.large_pins.read_text())['large_models']
        for alias,m in zip(('A','B'),large):
            for f in m['files']:
                if aba.sha(a.models/m['id']/f['path'])!=f['sha256']:raise ValueError('Large model mismatch')
            pins['models'][alias]={'filename':m['id']+'/'+m['files'][0]['path'],**m}
            names[alias]=m['id']
    else:
        for m in pins['models'].values():
            if aba.sha(a.models/m['filename'])!=m['sha256']: raise ValueError('Model mismatch')
    a.output=a.output.resolve(); a.output.mkdir(mode=0o700)
    router=aba.Router(SimpleNamespace(models=a.models,port=a.port,device='nvme0n1'),pins,a.output,maximum=1,normal=bool(a.large_pins))
    if a.large_pins:
        ini=a.output/'models.ini'
        ini.write_text(ini.read_text().replace('ctx-size = 32768','ctx-size = 8192'))
    trace={'version':1,'models':pins['models'],'requests':[],'sampling':{'temperature':0,'seed':42}}
    # Authored synthetic tasks: stateful planning, coding, failure analysis, synthesis.
    histories={}; saved={}
    questions=[
        'Using the notebook, design a detailed implementation and validation plan for a reliable request retry service. Explain tradeoffs and failure cases in about 700 words.',
        'Now write Python code for bounded retries with idempotency keys and exponential backoff. Explain cancellation and include meaningful test cases. Use the notebook constraints.',
        'Review that design for races, lost replies, duplicate execution, and crashes after durable writes. Give concrete failure timelines and fixes in detail.',
        'Summarize the final design as a technical handoff. Compare three retention policies, their memory costs, and how to verify correctness under pressure.'
    ]
    try:
        router.start()
        for turn,question in enumerate(questions[:3] if a.large_pins else questions):
            for model in ('A','B'):
                switch_start=trace['requests'][-1]['save_started_ns'] if trace['requests'] else time.monotonic_ns()
                router.load(model)
                restore=router.restore(model,saved[model]) if a.large_pins and model in saved else None
                if model not in histories:
                    n=64 if a.large_pins else (256 if model=='A' else 64)
                    histories[model]=[{'role':'user','content':aba.records(0,n)+'\n'+question}]
                else: histories[model].append({'role':'user','content':question})
                tokens=router.template_messages(model,histories[model])
                result=router.complete(model,tokens,predict=512 if a.large_pins else 768)
                save_started=time.monotonic_ns()
                state=router.save(model); saved[model]=state
                record={'model':names[model],'turn':turn,'arrival_s':[0,120,1200,5400][turn]+(60 if model=='B' else 0),
                        'prompt':tokens,'expected_tokens':result['tokens'],'state_receipt':state,
                        'reference_result':result,'save_started_ns':save_started,'restore_receipt':restore,
                        'switch_first_token_s':(result['first_token_ns']-switch_start)/1e9}
                trace['requests'].append(record)
                histories[model].append({'role':'assistant','content':result['content']})
                (a.output/'sessions.json').write_text(json.dumps(trace,indent=2)+'\n')
                print(model,turn,len(tokens),len(result['tokens']),flush=True)
                router.unload(model)
    finally: router.stop()
    for model in ('A','B'):
        reqs=[r for r in trace['requests'] if r['model']==names[model]]
        with (a.output/(names[model]+'.tokens')).open('x') as f:
            f.write(str(len(reqs))+'\n')
            for r in reqs:
                decode=r['expected_tokens'][:-1]
                f.write(f"{len(r['prompt'])} {len(decode)}\n")
                f.write(' '.join(map(str,r['prompt']+decode))+'\n')
    print('Trace SHA256',aba.sha(a.output/'sessions.json'),flush=True)


if __name__=='__main__':main()
