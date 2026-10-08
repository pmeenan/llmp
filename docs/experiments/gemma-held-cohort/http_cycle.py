#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Native HTTP old/new bookend for the unchanged held cohort selection guard."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

PROFILES = {
    'gemma26': '4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3',
    'gemma31': '32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','installed','baseline','candidate','binding','http-controls','library-path'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--baseline-sha256',required=True)
    args = parser.parse_args()
    model=args.profile
    artifact=PROFILES[model]
    binding = json.loads(args.binding.read_text())
    assert digest(args.baseline) == args.baseline_sha256
    assert digest(args.candidate) == binding['paths']['build/spark-native/src/runtime/llmp-runtime']['sha256']
    for name in ('libcublas.so.13', 'libcublasLt.so.13'):
        assert digest(args.library_path/name) == binding['paths']['build/spark-native/lib/llmp/'+name]['sha256']
    environment = {**os.environ, 'LD_LIBRARY_PATH': str(args.library_path.resolve())}
    controls = json.loads(args.http_controls.read_text())
    assert len(controls['cases']) == 4 and all(c['passed'] for c in controls['cases'])
    assert controls['runtime_exit']==0 and controls['profile']==model
    assert controls['runtime_sha256']==digest(args.candidate)
    args.output.mkdir(mode=0o700)
    arms = []
    for index,kind in enumerate(('old','new','new','old')):
        root = args.output / f'{index}-{kind}'
        root.mkdir(mode=0o700)
        for name in ('spill','state','checkpoints'):
            (root/name).mkdir(mode=0o700)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        base=f'http://127.0.0.1:{port}'
        config=root/'node.toml'
        config.write_text('schema_version=2\n[storage]\n'
            f'installed="{args.installed.resolve()}"\n'+''.join(
                f'{name}="{root.resolve()}/{name}"\n' for name in ('spill','state','checkpoints'))+
            f'[models.{model}]\nartifact="{artifact}"\ncontext=8192\nmax_slots=4\n'
            f'[client]\nbind="127.0.0.1:{port}"\n[memory]\nkeep_across_restart=false\n')
        config.chmod(0o600)
        binary=args.baseline if kind=='old' else args.candidate
        log=(root/'runtime.log').open('x')
        process=subprocess.Popen([str(binary.resolve()),'--config',str(config.resolve()),
            '--anchor',str((root/'enrollment').resolve())],stdout=log,stderr=log,env=environment)
        record=dict(arm=kind,index=index,runtime_sha256=digest(binary),samples=[],warm_controls=[])
        try:
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                assert process.poll() is None,'runtime startup exited'
                try:
                    with urllib.request.urlopen(base+'/v1/models',timeout=1) as response:
                        assert [r['id'] for r in json.load(response)['data']]==[model]
                    break
                except (OSError,urllib.error.URLError):
                    time.sleep(.1)
            else:
                raise AssertionError('startup deadline')

            def request(body):
                with urllib.request.urlopen(urllib.request.Request(base+'/v1/chat/completions',
                    json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120) as response:
                    result=json.load(response)
                assert result['usage']['completion_tokens']==32
                assert result['choices'][0]['finish_reason']=='length'
                return dict(choices=result['choices'],usage=result['usage'])

            for slots in (1,4):
                bodies=[dict(model=model,messages=[dict(role='user',content=
                    f'Continue this numbered list with many entries, one per line: 1. benchmark {slots} owner {owner}')],
                    temperature=0,max_tokens=32) for owner in range(slots)]
                def run():
                    if slots==1:
                        return [request(bodies[0])]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
                        return list(pool.map(request,bodies))
                # Ordinary two warm requests per geometry exercise the same
                # prefix checkpoint/replay machinery as the paid request.
                initial=run()
                warm=run()
                repeated=run()
                record['warm_controls'].append(dict(slots=slots,cold=initial,cached=warm,repeated=repeated,
                    cold_cached_equal=[a['choices']==b['choices'] for a,b in zip(initial,warm)],
                    cold_scope='descriptive; cold and cached/cohort geometry may differ'))
                assert all(r['usage']['prompt_tokens_details']['cached_tokens']>0 for r in warm+repeated), 'cached warm reuse missing'
                assert repeated==warm,'same-arm cached warm repeat'
                start=time.monotonic()
                paid=run()
                seconds=time.monotonic()-start
                assert math.isfinite(seconds) and seconds>0,'invalid paid duration'
                assert all(r['usage']['prompt_tokens_details']['cached_tokens']>0 for r in paid), 'paid cached reuse missing'
                assert paid==warm,'same-arm paid response and usage repeat'
                record['samples'].append(dict(slots=slots,seconds=seconds,responses=paid,
                    request_sha256=hashlib.sha256(json.dumps(bodies,sort_keys=True).encode()).hexdigest()))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                record['exit']=process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
                record['exit']='retirement-timeout-killed'
            log.close()
            (root/'arm.json').write_text(json.dumps(record,indent=2)+'\n')
        assert record['exit']==0
        arms.append(record)
        print('GEMMA4_HTTP_BOOKEND_ARM',json.dumps(record),flush=True)
    ratios=[]
    for offset in range(2):
        ref=arms[0]['samples'][offset]
        for arm in arms:
            row=arm['samples'][offset]
            assert row['responses']==ref['responses'] and row['request_sha256']==ref['request_sha256']
        old=sum(a['samples'][offset]['seconds'] for a in arms if a['arm']=='old')/2
        new=sum(a['samples'][offset]['seconds'] for a in arms if a['arm']=='new')/2
        ratios.append(dict(slots=ref['slots'],old_seconds=old,new_seconds=new,new_over_old=new/old-1))
    result=dict(schema=1,profile=model,scope='native HTTP old/new bookend, cached chat, 32 generated tokens per owner; client concurrency1/4, no backend C4 shape or stock parity claim',
        changed_behavior='unchanged held selection avoids redundant closure refresh',
        harness_sha256=digest(Path(__file__)),binding_sha256=digest(args.binding),arms=arms,ratios=ratios)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GEMMA4_HTTP_BOOKEND_PASS',json.dumps(ratios),flush=True)


if __name__=='__main__':
    main()
