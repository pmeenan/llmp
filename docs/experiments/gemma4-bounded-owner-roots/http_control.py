#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Ordinary bounded Gemma4 C2 common-root HTTP adoption; no timing claim."""
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
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

ARTIFACTS = {
    '26': '4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3',
    '31': '32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'installed', 'runtime', 'binding', 'context-root', 'library-path'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profile', choices=ARTIFACTS, required=True)
    args = parser.parse_args()
    profile, model = args.profile, 'gemma'+args.profile
    artifact, chunk = ARTIFACTS[profile], 1024 if profile == '26' else 256
    binding = json.loads(args.binding.read_text())
    runtime_sha = binding['paths']['build/spark-native/src/runtime/jitllm-runtime']['sha256']
    assert sha(args.runtime) == runtime_sha and binding['receipt']['official'] and binding['receipt']['cuda']
    for name in ('libcublas.so.13', 'libcublasLt.so.13'):
        assert sha(args.library_path/name) == binding['paths']['build/spark-native/lib/jitllm/'+name]['sha256']
    environment = {**os.environ, 'LD_LIBRARY_PATH': str(args.library_path.resolve())}
    assert json.loads((args.context_root/'ring-pass.json').read_text())['passed']
    own = json.loads((args.context_root/'own-control.json').read_text())
    assert own['passed'] and own['state']['restore_equal'] and own['prefixes'] == [2560,3072]
    record = json.loads((args.context_root/'inputs.json').read_text())
    ring = []
    for slot, prefix in enumerate((2560,3072)):
        p = args.context_root/f'input{slot}.i32';raw = p.read_bytes()
        assert sha(p) == record[str(slot)]['sha256'] and len(raw) == (prefix+35)*4
        ids = list(struct.unpack(f'<{len(raw)//4}i',raw))
        assert ids[0] == 2 and all(0 <= n < 262144 for n in ids)
        ring.append(ids[:prefix])
    args.output.mkdir(mode=0o700)
    for name in ('spill','state','checkpoints'):
        (args.output/name).mkdir(mode=0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    config = args.output/'node.toml'
    config.write_text('schema_version=2\n[storage]\n'+f'installed="{args.installed.resolve()}"\n'+
        ''.join(f'{name}="{args.output.resolve()}/{name}"\n' for name in ('spill','state','checkpoints'))+
        f'[models.{model}]\nartifact="{artifact}"\ncontext=4096\nprefill_chunk={chunk}\nmax_slots=2\n'+
        f'[client]\nbind="127.0.0.1:{port}"\n[memory]\nkeep_across_restart=true\n')
    config.chmod(0o600)
    anchor = args.output/'enrollment'
    result = dict(schema=1,profile=profile,artifact=artifact,runtime_sha256=runtime_sha,
        binding_sha256=sha(args.binding),harness_sha256=sha(Path(__file__)),
        context=4096,max_slots=2,prefill_chunk=chunk,cases=[],epochs=[],
        scope='ordinary C2 unequal-root adoption, physical-ring literals and matched cached scalar restart; no endpoint timing')
    process = log = None
    epoch = 1

    def post(path,body):
        with urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(body).encode(),
                headers={'Content-Type':'application/json'}),timeout=180) as response:
            return json.load(response)

    def start(epoch):
        nonlocal process,log
        log = (args.output/f'runtime{epoch}.log').open('x')
        process = subprocess.Popen([str(args.runtime.resolve()),'--config',str(config.resolve()),
            '--anchor',str(anchor.resolve())],stdout=log,stderr=log,env=environment)
        deadline = time.monotonic()+120
        while time.monotonic() < deadline:
            assert process.poll() is None,'runtime exited during startup'
            try:
                with urllib.request.urlopen(base+'/v1/models',timeout=1) as response:
                    models = json.load(response)
                assert [x['id'] for x in models['data']] == [model]
                return
            except (OSError,urllib.error.URLError):
                time.sleep(.1)
        raise AssertionError('runtime startup deadline')

    def stop(epoch):
        nonlocal process,log
        if process is None:return
        if process.poll() is None:process.send_signal(signal.SIGTERM)
        try:
            rc = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()
            raise AssertionError('runtime retirement deadline; process killed')
        finally:
            log.close();process = None
        assert rc == 0,rc
        text = (args.output/f'runtime{epoch}.log').read_text()
        rows = re.findall(r'Gemma4 serving final '+model+r': (\{[^\n]*\})',text)
        assert len(rows) == 1,'completed HTTP-drain metadata missing'
        extra = json.loads(rows[0])
        assert extra['architecture'] == 'gemma4' and extra['max_rows'] == chunk
        assert extra['common_owner_reads_requested'] and extra['bounded_owner_roots_requested']
        result['epochs'].append(dict(epoch=epoch,exit=rc,final=extra,
            log_sha256=sha(args.output/f'runtime{epoch}.log')))

    def case(name,fn):
        then = time.monotonic()
        try:entry = dict(name=name,passed=True,detail=fn())
        except Exception as error:entry = dict(name=name,passed=False,error=f'{type(error).__name__}: {error}')
        entry['seconds'] = time.monotonic()-then
        result['cases'].append(entry)
        print(json.dumps(entry),flush=True)

    def chat(text,count=16):
        return dict(model=model,messages=[dict(role='user',content=text)],max_tokens=count,temperature=0)

    def literal(ids,count=8):
        return dict(model=model,prompt=ids,max_tokens=count,temperature=0,
            logprobs=2,return_tokens_as_token_ids=True)

    def choice(response,count=None):
        row = response['choices'][0]
        assert row['finish_reason'] in ('length','stop')
        if count is not None:
            assert response['usage']['completion_tokens'] == count and row['finish_reason'] == 'length'
        return row

    def likelihood(response,count):
        row = choice(response,count);scores = row['logprobs']
        tokens = [int(n.removeprefix('token_id:')) for n in scores['tokens']]
        assert len(tokens) == len(scores['token_logprobs']) == len(scores['top_logprobs']) == count
        assert all(0 <= n < 262144 for n in tokens)
        assert all(math.isfinite(n) for n in scores['token_logprobs'])
        assert all(len(d) == 2 and all(math.isfinite(n) for n in d.values()) for d in scores['top_logprobs'])
        return tokens

    def stream(body,first=None,disconnect=False,release=None):
        text,done,reason = '',False,None
        with urllib.request.urlopen(urllib.request.Request(base+'/v1/chat/completions',
                json.dumps({**body,'stream':True}).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: '):continue
                if line[6:] == '[DONE]':done = True;break
                event = json.loads(line[6:]);assert 'error' not in event,event
                for item in event.get('choices',[]):
                    piece = item.get('delta',{}).get('content','');text += piece
                    reason = item.get('finish_reason') or reason
                    if piece and first is not None:first.set()
                    if piece and release is not None:assert release.wait(60),'peer coordination deadline'
                    if piece and disconnect:return dict(disconnected=True,text=text)
        assert done and reason in ('stop','length')
        return dict(done=done,text=text,finish_reason=reason)

    try:
        inspected = subprocess.run([str(args.runtime.resolve()),'--config',str(config.resolve()),
            '--anchor',str(anchor.resolve()),'settings','--json'],check=True,capture_output=True,text=True,
            timeout=60,env=environment)
        settings = json.loads(inspected.stdout)['models'][0]['settings']
        for name,value in (('context',4096),('prefill_chunk',chunk),('max_slots',2)):
            assert settings[name]['value'] == value and settings[name]['source'] == 'override'
        assert settings['speculation']['value'] is False
        result['settings'] = json.loads(inspected.stdout)
        start(epoch)

        def unequal():
            gate = threading.Barrier(2)
            prompts = [ring[slot][:prefix] for slot,prefix in enumerate((256,768))]
            def request(ids):gate.wait(timeout=30);return post('/v1/completions',literal(ids,32))
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                answers = list(pool.map(request,prompts))
            detail = []
            for slot,answer in enumerate(answers):
                tokens = likelihood(answer,32)
                assert answer['usage']['prompt_tokens'] == len(prompts[slot])
                detail.append(dict(slot=slot,prompt_rows=len(prompts[slot]),generated_ids=tokens))
            return dict(owners=detail,scope='simultaneous actual HTTP owners; successful joined waves and selected bounded plans checked after drain')
        case('unequal-two-owner-literal-likelihoods',unequal)

        def literals():
            detail = []
            for slot,ids in enumerate(ring):
                a,b = post('/v1/completions',literal(ids)),post('/v1/completions',literal(ids))
                tokens = likelihood(a,8);likelihood(b,8)
                assert choice(a,8) == choice(b,8),'same-geometry ring literal repeat differs'
                body = literal(ids+tokens[:4],4)
                c,d = post('/v1/completions',body),post('/v1/completions',body)
                likelihood(c,4);likelihood(d,4)
                assert choice(c,4) == choice(d,4),'same-geometry continuation repeat differs'
                detail.append(dict(slot=slot,prompt_rows=len(ids),generated_ids=tokens,repeat_exact=True,continuation_repeat_exact=True))
            try:post('/v1/completions',literal([2]*4097,1));raise AssertionError('over-context request admitted')
            except urllib.error.HTTPError as error:assert 400 <= error.code < 500;error.read()
            likelihood(post('/v1/completions',literal(ring[0],1)),1)
            return dict(owners=detail,physical_ring_capacity=own['physical_ring_capacity'],bounds_refused=True,followup_healthy=True)
        case('physical-ring-literal-continuations-and-bounds',literals)

        def chats():
            body = chat('Continue this numbered list with many entries, one per line: 1. alpha',16)
            post('/v1/chat/completions',body)  # Descriptive cold primer.
            a,b = post('/v1/chat/completions',body),post('/v1/chat/completions',body)
            assert choice(a,16) == choice(b,16),'matched cached chat repeat differs'
            assert all(x['usage']['prompt_tokens_details']['cached_tokens'] > 0 for x in (a,b))
            body = chat('Reply with one short word.',16)
            post('/v1/chat/completions',body)
            plain = choice(post('/v1/chat/completions',body));content = plain['message']['content'];assert content
            actual = stream(body)
            assert actual['text'] == content and actual['finish_reason'] == plain['finish_reason']
            delimiter = content[:min(3,len(content))]
            stopped = choice(post('/v1/chat/completions',{**body,'stop':delimiter}))
            assert stopped['message']['content'] == '' and stopped['finish_reason'] == 'stop'
            actual = stream({**body,'stop':delimiter})
            assert actual['text'] == '' and actual['finish_reason'] == 'stop'
            return dict(cached_repeat_exact=True,SSE_exact=True,stop_suppressed=True,cold_cached_scope='cold primer descriptive only')
        case('chat-cache-SSE-and-stop',chats)

        def peers():
            first = [threading.Event(),threading.Event()];release = threading.Event()
            bodies = [chat(f'Continue this numbered list with many entries, one per line: 1. peer {i}',32) for i in range(3)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                runs = [pool.submit(stream,bodies[i],first[i],i==0,release) for i in range(2)]
                try:
                    assert all(event.wait(60) for event in first),'both slots must publish'
                    queued = pool.submit(stream,bodies[2])
                finally:release.set()
                answers = [run.result(timeout=180) for run in runs]+[queued.result(timeout=180)]
            assert answers[0]['disconnected'] and all(x['done'] and x['text'] for x in answers[1:])
            choice(post('/v1/chat/completions',chat('Reply with one short word.')))
            return dict(requests=3,slots=2,disconnected=1,completed_peers=2,
                scope='client departure/queued request progress; reads may be buffered, joined proof checked after drain')
        case('two-slots-queued-third-and-partial-departure',peers)

        def restart():
            nonlocal epoch
            requests = [chat(f'Continue this numbered list with many entries, one per line: 1. durable {i}',16) for i in range(2)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                cold = list(pool.map(lambda body:post('/v1/chat/completions',body),requests))
            for r in cold:choice(r,16)
            warm = [post('/v1/chat/completions',body) for body in requests]
            before = [post('/v1/chat/completions',body) for body in requests]
            for a,b in zip(warm,before):
                assert choice(a,16) == choice(b,16)
                assert b['usage']['prompt_tokens_details']['cached_tokens'] > 0
            (args.output/'restart-before.json').write_text(json.dumps(dict(cold=cold,warm=warm,before=before),indent=2)+'\n')
            stop(epoch)
            extra = result['epochs'][0]['final']
            assert extra['joined_groups'] > 0 and extra['joined_units'] == 2*extra['joined_groups']
            assert extra['max_bound_bounded_owner_attention'] > 0
            records = sorted((args.output/'spill/conversations'/artifact).glob('slot-*.record'))
            assert len(records) == 2
            kept = []
            for path in records:
                r = json.loads(path.read_text());state = path.with_name(r['file']);st = state.stat()
                assert r['artifact'] == artifact and r['layout'].startswith(f'gemma{profile}-f16-kv-scalar-device-v1:4096:{chunk}:')
                assert r['tokens'] and 0 < r['cursor'] <= len(r['tokens'])
                assert r['file'] == f"slot-{r['slot']}.state"
                assert (r['inode'],r['device'],r['file_bytes']) == (st.st_ino,st.st_dev,st.st_size)
                assert r['extents'] and all(len(x[2]) == 64 for x in r['extents'])
                assert path.stat().st_mode & 0o777 == state.stat().st_mode & 0o777 == 0o600
                kept.append(dict(slot=r['slot'],cursor=r['cursor'],record_sha256=sha(path)))
            epoch = 2;start(epoch)
            assert all(sha(path) == row['record_sha256'] for path,row in zip(records,kept))
            for slot in (0,1):assert f'slot {slot} adopted a kept conversation' in (args.output/'runtime2.log').read_text()
            after = [post('/v1/chat/completions',body) for body in requests]
            (args.output/'restart-after.json').write_text(json.dumps(after,indent=2)+'\n')
            for a,b in zip(before,after):
                assert choice(a,16) == choice(b,16),'matched cached scalar restart differs'
                assert b['usage']['prompt_tokens_details']['cached_tokens'] > 0
            return dict(kept=kept,cached_scalar_restart_exact=True,record_bytes_preserved_on_adoption=True,
                cold_cached_scope='cold versus cached and scalar versus joined comparisons are not equality gates')
        case('two-slot-clean-restart-and-exact-cached-scalar-replay',restart)
    finally:
        try:stop(epoch)
        finally:(args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    assert len(result['cases']) == 5 and len(result['epochs']) == 2
    assert all(row['passed'] for row in result['cases']),'one or more HTTP cases failed'
    print('GEMMA4_BOUNDED_ROOT_HTTP_PASS',json.dumps(result['epochs']),flush=True)


if __name__ == '__main__':main()
