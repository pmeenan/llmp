#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Native HTTP old/new bookend for compatible cold literal prefill."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import struct
import threading
import time
import urllib.error
import urllib.request

ARTIFACT = '8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','installed','baseline','candidate','binding','http-controls','library-path','context-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    assert digest(args.baseline) == 'e133403cabb507cb0c26e9e210f9a60c6d5f8191aafe6719bdf676ee3f0f55b2'
    assert digest(args.candidate) == binding['paths']['build/spark-native/src/runtime/llmp-runtime']['sha256']
    for name in ('libcublas.so.13', 'libcublasLt.so.13'):
        assert digest(args.library_path/name) == binding['paths']['build/spark-native/lib/llmp/'+name]['sha256']
    environment = {**os.environ, 'LD_LIBRARY_PATH': str(args.library_path.resolve())}
    controls = json.loads(args.http_controls.read_text())
    assert len(controls['cases']) == 5 and all(c['passed'] for c in controls['cases'])
    assert len(controls['epochs']) == 2 and all(e['exit']==0 for e in controls['epochs'])
    prompts = []
    input_record = json.loads((args.context_root/'ring/inputs.json').read_text())
    for slot in range(2):
        path=args.context_root/f'ring/input{slot}/ids.i32'
        assert digest(path)==input_record[str(slot)]['ids_sha256']
        data=path.read_bytes()
        ids=list(struct.unpack(f'<{len(data)//4}i',data))
        prompts.append(ids[:input_record[str(slot)]['prefix']])
    assert [len(ids) for ids in prompts]==[1280,1536]
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
            f'[models.gemma3]\nartifact="{ARTIFACT}"\ncontext=4096\nprefill_chunk=128\nmax_slots=2\n'
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
                        assert [r['id'] for r in json.load(response)['data']]==['gemma3']
                    break
                except (OSError,urllib.error.URLError):
                    time.sleep(.1)
            else:
                raise AssertionError('startup deadline')

            def request(body):
                with urllib.request.urlopen(urllib.request.Request(base+'/v1/completions',
                    json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120) as response:
                    result=json.load(response)
                assert result['usage']['completion_tokens']==32
                assert result['choices'][0]['finish_reason']=='length'
                row=result['choices'][0]
                scores=row['logprobs']
                tokens=[int(token.removeprefix('token_id:')) for token in scores['tokens']]
                assert len(tokens)==32 and all(0<=token<262208 for token in tokens)
                assert len(scores['token_logprobs']) == len(scores['top_logprobs']) == 32
                assert all(math.isfinite(n) for n in scores['token_logprobs'])
                assert all(len(row)==1 and all(math.isfinite(n) for n in row.values()) for row in scores['top_logprobs'])
                assert result['usage']['prompt_tokens_details']['cached_tokens']==0,'cold literal unexpectedly reused a prefix'
                return dict(generated_ids=tokens,finish_reason=row['finish_reason'],usage=result['usage'])

            for slots in (1,2):
                bodies=[dict(model='gemma3',prompt=prompts[owner],temperature=0,max_tokens=32,
                             logprobs=1,return_tokens_as_token_ids=True) for owner in range(slots)]
                def run():
                    if slots==1:
                        return [request(bodies[0])]
                    gate=threading.Barrier(2)
                    def synchronized(body):
                        gate.wait(timeout=30)
                        return request(body)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        return list(pool.map(synchronized,bodies))
                warm=run()
                repeated=run()
                record['warm_controls'].append(dict(slots=slots,warm=warm,repeated=repeated,
                    scope='both cold literal requests recompute the full prompt; exact generated IDs and usage'))
                assert repeated==warm,'same-arm cold literal generated history repeat'
                start=time.monotonic()
                paid=run()
                seconds=time.monotonic()-start
                assert math.isfinite(seconds) and seconds>0,'paid HTTP duration must be finite and positive'
                assert paid==warm,'same-arm paid generated history and usage repeat'
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
        matches=re.findall(r'Gemma3 serving final gemma3: (\{[^\n]*\})',(root/'runtime.log').read_text())
        assert len(matches)==1
        record['final']=json.loads(matches[0])
        assert record['final']['joined_groups']>0
        assert record['final']['joined_units']==2*record['final']['joined_groups']
        assert record['final']['gpu_greedy_tokens']==0  # all timed requests publish score-bearing full rows
        if kind=='new':
            assert record['final']['max_wave_rows']==256
            assert record['final']['joined_prefill_groups']>0 and record['final']['joined_prefill_rows']>=256
            assert record['final']['bound_packed_prefill_attention']>0
        arms.append(record)
        print('GEMMA3_HTTP_JOINT_PREFILL_BOOKEND_ARM',json.dumps(record),flush=True)
    ratios=[]
    for offset in range(2):
        ref=arms[0]['samples'][offset]
        for arm in arms:
            row=arm['samples'][offset]
            assert row['responses']==ref['responses'] and row['request_sha256']==ref['request_sha256']
        old=sum(a['samples'][offset]['seconds'] for a in arms if a['arm']=='old')/2
        new=sum(a['samples'][offset]['seconds'] for a in arms if a['arm']=='new')/2
        ratios.append(dict(slots=ref['slots'],old_seconds=old,new_seconds=new,new_over_old=new/old-1))
    result=dict(schema=1,scope='native HTTP old/new bookend, cold literal C1/C2 ring prompts,32 generated tokens per owner; no stock parity claim',
        changed_behavior='compatible plain prefill may share two owners; explicit totalwave256/perowner128',
        harness_sha256=digest(Path(__file__)),binding_sha256=digest(args.binding),arms=arms,ratios=ratios)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GEMMA3_HTTP_JOINT_PREFILL_BOOKEND_PASS',json.dumps(ratios),flush=True)


if __name__=='__main__':
    main()
