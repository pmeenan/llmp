#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the generalized session capture against explicit external input/model pins."""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
from run import IMAGE, REVISION, container, sha
from prediction_checks import capture_comparison

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('spec',type=Path);p.add_argument('models',type=Path);p.add_argument('source',type=Path)
    p.add_argument('inputs',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--record-prediction-drift',action='append',default=[],metavar='MODEL_ID',help='Record failed top-1 equivalence explicitly instead of aborting; no tolerance is assumed')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--parallel',action='store_true')
    mode.add_argument('--restore-probe',action='store_true')
    a=p.parse_args();os.umask(0o077)
    for k in ['spec','models','source','inputs','output']:setattr(a,k,getattr(a,k).resolve())
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    spec=json.loads(a.spec.read_text());pins=json.loads((HERE/'pins.json').read_text())
    if set(a.record_prediction_drift)-{m['id'] for m in spec['models']}:raise ValueError('Unknown prediction-drift model')
    if a.record_prediction_drift and (a.parallel or a.restore_probe):raise ValueError('Drift recording is supported only for session capture')
    for name,digest in pins['headers'].items():
        if sha(a.source/name)!=digest:raise ValueError('Header mismatch')
    for m in spec['models']:
        for f in m['files']:
            if sha(a.models/f['path'])!=f['sha256']:raise ValueError('Model mismatch')
        if sha(a.inputs/m['tokens'])!=m['tokens_sha256']:raise ValueError('Input mismatch')
    a.output.mkdir(mode=0o700)
    docker=shlex.split(os.environ.get('DOCKER','docker'))
    base=docker+['run','--rm','--network','none','--read-only','--user',f'{os.getuid()}:{os.getgid()}',
                 '--tmpfs','/tmp:rw,size=1g']
    for path,dest,readonly in [(HERE,'/harness',True),(a.source,'/source',True),(a.inputs,'/inputs',True),
                               (a.models,'/models',True),(a.output,'/output',False)]:
        base+=['--mount',f'type=bind,src={path},dst={dest}'+(',readonly' if readonly else '')]
    build=['g++','-std=c++23','-O2','-march=armv8-a','-Wall','-Wextra','-Werror',
           '-I/source/include','-I/source/ggml/include','/harness/'+('parallel_capture.cc' if a.parallel else ('restore_probe.cc' if a.restore_probe else 'session_capture.cc')),'-L/app',
           '-Wl,-rpath,/app','-lllama','-lggml','-lggml-base','-o','/output/capture']
    container(docker,base,['--entrypoint',build[0],IMAGE,*build[1:]])
    receipt=dict(parallel=a.parallel,restore_probe=a.restore_probe,cuda_disable_fusion=True,cuda_disable_graphs=True,image=IMAGE,revision=REVISION,spec=spec,source_sha256=sha(HERE/('parallel_capture.cc' if a.parallel else ('restore_probe.cc' if a.restore_probe else 'session_capture.cc'))),
                 executable_sha256=sha(a.output/'capture'),build=build,runs=[],host=os.uname().nodename,
                 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())
    for m in spec['models']:
        for batch,reuse,swa in spec['configurations']:
            control=None
            for trace in ((False,) if a.restore_probe else (False,True)):
                name=f"{m['id']}-b{batch}-r{reuse}-s{swa}-t{int(trace)}"
                tail=['--device','nvidia.com/gpu=all','--env','CUDA_DISABLE_PTX_JIT=1','--env','GGML_CUDA_DISABLE_FUSION=1','--env','GGML_CUDA_DISABLE_GRAPHS=1','--entrypoint','/output/capture',IMAGE,
                      '/models/'+m['files'][0]['path'],'/inputs/'+m['tokens'],'/output/'+name+'.jsonl',
                      '/output/'+name+'.predictions',str(m['layers']),str(m['experts']),str(m['topk']),
                      str(m['context']),str(batch),str(int(trace)),str(reuse),str(swa)]
                if a.parallel:
                    tail=tail[:-4]+[str(int(trace))]
                with (a.output/(name+'.log')).open('x') as log:
                    container(docker,base,tail,stdout=log,stderr=subprocess.STDOUT)
                log_text=(a.output/(name+'.log')).read_text(errors='replace')
                offload=re.search(r'offloaded (\d+)/(\d+) layers to GPU',log_text)
                if not offload or offload[1]!=offload[2] or int(offload[1])<m['layers']:raise ValueError('Full GPU offload not verified')
                digest=sha(a.output/(name+'.predictions'))
                comparison=capture_comparison(control,a.output/(name+'.predictions'),m['id'] in a.record_prediction_drift) if trace else None
                control=a.output/(name+'.predictions')
                r=dict(name=name,model=m['id'],batch=batch,reuse=reuse,swa=swa,trace=trace,
                       log_sha256=sha(a.output/(name+'.log')),predictions_sha256=digest,events_sha256=sha(a.output/(name+'.jsonl')))
                if comparison is not None:
                    r['prediction_comparison']=comparison
                    r['prediction_drift_recorded']=bool(comparison['different_records'])
                receipt['runs'].append(r)
                (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print(name+(' recorded; numerical equivalence FAILED' if comparison and comparison['different_records'] else ' passed'),flush=True)


if __name__=='__main__':main()
