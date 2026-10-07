#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Gemma3 ordinary HTTP, two-slot progress and restart adoption controls."""
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

ARTIFACT = '8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb'
TEMPLATE = '7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--context-root', type=Path, required=True)
    parser.add_argument('--library-path', type=Path, required=True)
    parser.add_argument('--bounded-roots', action='store_true',
                        help='Require fresh unequal/ring copy-free gates and actual bounded plans')
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    runtime_sha = binding['paths']['build/spark-native/src/runtime/jitllm-runtime']['sha256']
    assert sha(args.runtime) == runtime_sha
    assert binding['receipt']['official'] and binding['receipt']['cuda']
    for name in ('libcublas.so.13', 'libcublasLt.so.13'):
        assert sha(args.library_path / name) == binding['paths']['build/spark-native/lib/jitllm/' + name]['sha256']
    environment = {**os.environ, 'LD_LIBRARY_PATH': str(args.library_path.resolve())}
    # The original screen keeps all three gates; adoption requires its two fresh
    # same-policy own/reference gates and an actual unequal-root selection below.
    scenarios = ('unequal', 'ring') if args.bounded_roots else ('unequal', 'ring', 'short')
    for scenario in scenarios:
        assert json.loads((args.context_root / scenario / 'quality.json').read_text())['passed']
        assert json.loads((args.context_root / scenario / 'own-control.json').read_text())['state']['restore_equal']
    record = json.loads((args.context_root / 'ring/inputs.json').read_text())
    ring = []
    for slot in range(2):
        path = args.context_root / f'ring/input{slot}/ids.i32'
        assert sha(path) == record[str(slot)]['ids_sha256']
        raw = path.read_bytes()
        ids = list(struct.unpack(f'<{len(raw)//4}i', raw))
        assert ids[0] == 2 and all(0 <= n < 262208 for n in ids)
        ring.append(ids[:record[str(slot)]['prefix']])
    assert len(ring[0]) == 1280 and len(ring[1]) == 1536
    args.output.mkdir(mode=0o700)
    for name in ('spill', 'state', 'checkpoints'):
        (args.output / name).mkdir(mode=0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    config = args.output / 'node.toml'
    config.write_text('schema_version=2\n[storage]\n'
                      f'installed="{args.installed.resolve()}"\n' + ''.join(
                          f'{name}="{args.output.resolve()}/{name}"\n'
                          for name in ('spill', 'state', 'checkpoints')) +
                      f'[models.gemma3]\nartifact="{ARTIFACT}"\ncontext=4096\n'
                      'prefill_chunk=128\nmax_slots=2\n'
                      f'[client]\nbind="127.0.0.1:{port}"\n'
                      '[memory]\nkeep_across_restart=true\n')
    config.chmod(0o600)
    anchor = args.output / 'enrollment'
    result = dict(schema=1, runtime_sha256=runtime_sha, binding_sha256=sha(args.binding),
                  harness_sha256=sha(Path(__file__)), artifact=ARTIFACT,
                  template_sha256=TEMPLATE, context=4096, max_slots=2, prefill_chunk=128,
                  ordinary_production=True, bounded_roots=args.bounded_roots, cases=[], epochs=[],
                  scope='HTTP lifecycle and actual successful two-owner groups; no endpoint throughput claim')
    process = None
    log = None

    def post(path, body):
        with urllib.request.urlopen(urllib.request.Request(base+path, json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'}), timeout=120) as response:
            return json.load(response)

    def start(epoch):
        nonlocal process, log
        path = args.output / f'runtime{epoch}.log'
        log = path.open('x')
        process = subprocess.Popen([str(args.runtime.resolve()), '--config', str(config.resolve()),
                                    '--anchor', str(anchor.resolve())], stdout=log, stderr=log, env=environment)
        deadline = time.monotonic()+90
        while time.monotonic() < deadline:
            assert process.poll() is None, 'runtime exited during startup'
            try:
                with urllib.request.urlopen(base+'/v1/models', timeout=1) as response:
                    models = json.load(response)
                assert [x['id'] for x in models['data']] == ['gemma3']
                return
            except (OSError, urllib.error.URLError):
                time.sleep(.1)
        raise AssertionError('runtime startup deadline')

    def stop(epoch):
        nonlocal process, log
        if process is None:
            return
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            rc = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise AssertionError('runtime retirement deadline; killed process')
        finally:
            log.close()
            process = None
        assert rc == 0, rc
        text = (args.output / f'runtime{epoch}.log').read_text()
        rows = re.findall(r'Gemma3 serving final gemma3: (\{[^\n]*\})', text)
        assert len(rows) == 1, 'completed HTTP-drain metadata missing'
        extra = json.loads(rows[0])
        assert extra['architecture'] == 'gemma3' and extra['recipe'] == 'bounded-4096-two-owner'
        assert extra['max_rows'] == 128
        result['epochs'].append(dict(epoch=epoch, exit=rc, final=extra, log_sha256=sha(args.output/f'runtime{epoch}.log')))

    def case(name, fn):
        then = time.monotonic()
        try:
            detail = fn()
            entry = dict(name=name, passed=True, detail=detail)
        except Exception as error:
            entry = dict(name=name, passed=False, error=f'{type(error).__name__}: {error}')
        entry['seconds'] = time.monotonic()-then
        result['cases'].append(entry)
        print(json.dumps(entry), flush=True)

    def chat(text, count=16):
        return dict(model='gemma3', messages=[dict(role='user', content=text)],
                    max_tokens=count, temperature=0)

    def literal(ids, count=8):
        return dict(model='gemma3', prompt=ids, max_tokens=count, temperature=0,
                    logprobs=2, return_tokens_as_token_ids=True)

    def choice(response, count=None):
        row = response['choices'][0]
        assert row['finish_reason'] in ('length', 'stop')
        if count is not None:
            assert response['usage']['completion_tokens'] == count and row['finish_reason'] == 'length'
        return row

    def stream(body, first=None, disconnect=False, release=None):
        text, done, reason = '', False, None
        with urllib.request.urlopen(urllib.request.Request(base+'/v1/chat/completions',
                json.dumps({**body, 'stream': True}).encode(),
                headers={'Content-Type':'application/json'}), timeout=120) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    done = True
                    break
                event = json.loads(line[6:])
                assert 'error' not in event, event
                for item in event.get('choices', []):
                    piece = item.get('delta', {}).get('content', '')
                    text += piece
                    reason = item.get('finish_reason') or reason
                    if piece and first is not None:
                        first.set()
                    if piece and release is not None:
                        assert release.wait(60), 'peer coordination deadline'
                    if piece and disconnect:
                        return dict(disconnected=True, text=text)
        assert done and reason in ('stop', 'length')
        return dict(done=done, text=text, finish_reason=reason)

    epoch = 1
    try:
        inspected = subprocess.run([str(args.runtime.resolve()), '--config', str(config.resolve()),
                '--anchor', str(anchor.resolve()), 'settings', '--json'], check=True,
                capture_output=True, text=True, timeout=30, env=environment)
        settings = json.loads(inspected.stdout)['models'][0]['settings']
        for name, value in [('context',4096), ('prefill_chunk',128), ('max_slots',2)]:
            assert settings[name]['value'] == value and settings[name]['source'] == 'override'
        assert settings['speculation']['value'] is False
        result['settings'] = json.loads(inspected.stdout)
        start(epoch)

        def literals():
            details = []
            for slot, ids in enumerate(ring):
                a = post('/v1/completions', literal(ids))
                b = post('/v1/completions', literal(ids))
                row = choice(a, 8)
                assert choice(b, 8) == row, 'same-geometry literal response/likelihood repeat'
                scores = row['logprobs']
                tokens = [int(n.removeprefix('token_id:')) for n in scores['tokens']]
                assert len(tokens) == 8 and all(0 <= n < 262208 for n in tokens)
                assert all(math.isfinite(n) for n in scores['token_logprobs'])
                assert all(len(d) == 2 and all(math.isfinite(n) for n in d.values()) for d in scores['top_logprobs'])
                split = literal(ids+tokens[:4], 4)
                c, d = post('/v1/completions', split), post('/v1/completions', split)
                assert choice(c,4) == choice(d,4), 'fresh continuation repeat'
                details.append(dict(slot=slot, prompt_rows=len(ids), generated_ids=tokens,
                    repeat_exact=True, continuation_repeat_exact=True,
                    cross_geometry_matches=sum(x==y for x,y in zip(tokens[4:],
                        [int(n.removeprefix('token_id:')) for n in c['choices'][0]['logprobs']['tokens']]))))
            try:
                post('/v1/completions', literal([2]*4097, 1))
                raise AssertionError('over-context request admitted')
            except urllib.error.HTTPError as error:
                assert 400 <= error.code < 500
                error.read()
            choice(post('/v1/completions', literal(ring[0],1)),1)
            unequal = []
            if args.bounded_roots:
                # The serial ring checks above cannot witness unequal joined
                # attention. Start two actual literal owners in different KV
                # buckets; the drained bound-plan counter below proves selection.
                barrier = threading.Barrier(2)
                def joined(slot):
                    ids = ring[slot][:256 if slot == 0 else 768]
                    barrier.wait(timeout=60)
                    response = post('/v1/completions', literal(ids,32))
                    row = choice(response,32)
                    scores = row['logprobs']
                    tokens = [int(n.removeprefix('token_id:')) for n in scores['tokens']]
                    assert response['usage']['prompt_tokens'] == len(ids)
                    assert len(tokens) == 32 and all(0 <= n < 262208 for n in tokens)
                    assert len(scores['token_logprobs']) == len(scores['top_logprobs']) == 32
                    assert all(math.isfinite(n) for n in scores['token_logprobs'])
                    assert all(len(d) == 2 and all(math.isfinite(n) for n in d.values())
                               for d in scores['top_logprobs'])
                    return dict(slot=slot, prompt_rows=len(ids), generated_tokens=len(tokens),
                                likelihoods_finite=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    unequal = list(pool.map(joined,range(2)))
            return dict(owners=details, unequal_joined_requests=unequal,
                        over_context_refused=True, followup_healthy=True,
                        cross_geometry_scope='descriptive; separate fresh prefill geometry')
        case('literal-ring-continuations-and-bounds', literals)

        def chat_checks():
            body = chat('Continue this numbered list with many entries: 1. checkpoint replay entry',16)
            a = post('/v1/chat/completions',body)
            short = post('/v1/chat/completions',{**body,'max_tokens':4})
            b = post('/v1/chat/completions',body)
            assert choice(a,16) == choice(b,16)
            assert short['usage']['completion_tokens'] == 4
            cached = [x['usage']['prompt_tokens_details']['cached_tokens'] for x in (short,b)]
            assert all(n>0 for n in cached)
            body = chat('Reply with one short word.',16)
            plain = choice(post('/v1/chat/completions',body))
            content = plain['message']['content']
            assert content
            actual = stream(body)
            assert actual['text'] == content and actual['finish_reason'] == plain['finish_reason']
            delimiter = content[:min(3,len(content))]
            stopped = choice(post('/v1/chat/completions',{**body,'stop':delimiter}))
            assert stopped['message']['content'] == '' and stopped['finish_reason'] == 'stop'
            streamed = stream({**body,'stop':delimiter})
            assert streamed['text'] == '' and streamed['finish_reason'] == 'stop'
            return dict(cached_tokens=cached, replay_exact=True, SSE_exact=True, stop_suppressed=True)
        case('chat-template-cache-SSE-and-stop',chat_checks)

        def peers():
            first = [threading.Event(),threading.Event()]
            release = threading.Event()
            bodies = [chat(f'Continue this numbered list with many entries: 1. peer {i}',128) for i in range(3)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                runs = [pool.submit(stream,bodies[i],first[i],i==0,release) for i in range(2)]
                try:
                    assert all(event.wait(60) for event in first), 'both actual slots must publish'
                    queued = pool.submit(stream,bodies[2])
                finally:
                    release.set()
                answers = [run.result(timeout=120) for run in runs]+[queued.result(timeout=120)]
            assert answers[0]['disconnected']
            assert all(x['done'] and x['text'] for x in answers[1:])
            choice(post('/v1/chat/completions',chat('Reply with one short word.')))
            return dict(requests=3, slots=2, disconnected=1, completed_peers=2,
                        queue_scope='third request submitted after both peers published; client reads may be buffered',
                        progress_scope='client completion; successful joined groups checked after drain')
        case('two-slots-queued-third-and-partial-departure',peers)

        prime = [chat(f'Continue this numbered list with many entries: 1. durable conversation {i}',16) for i in range(2)]
        diagnostic = {'requests': prime}
        def remember(label, responses):
            diagnostic[label] = responses
            (args.output/'restart-responses.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
            for response in responses:
                choice(response,16)
            return responses
        def parallel(label):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                return remember(label,list(pool.map(lambda body:post('/v1/chat/completions',body),prime)))
        before = parallel('cold_parallel')
        parallel('same_process_parallel1')
        parallel('same_process_parallel2')
        remember('same_process_scalar1',[post('/v1/chat/completions',body) for body in prime])
        scalar_before = remember('same_process_scalar2',[post('/v1/chat/completions',body) for body in prime])
        stop(epoch)
        assert result['epochs'][0]['final']['joined_groups'] > 0
        assert result['epochs'][0]['final']['joined_units'] == 2*result['epochs'][0]['final']['joined_groups']
        assert result['epochs'][0]['final']['bound_owner_attention'] > 0
        if args.bounded_roots:
            assert result['epochs'][0]['final']['bound_bounded_owner_attention'] > 0
        records = sorted((args.output/'spill/conversations'/ARTIFACT).glob('slot-*.record'))
        assert len(records) == 2
        kept = []
        for path in records:
            data = json.loads(path.read_text())
            assert data['artifact'] == ARTIFACT and data['layout'].startswith('gemma3-4b-f16-kv-device-v1:4096:128:')
            assert data['tokens'] and 0 < data['cursor'] <= len(data['tokens'])
            assert data['file'] == f"slot-{data['slot']}.state"
            state = path.with_name(data['file'])
            stat = state.stat()
            assert data['inode'] == stat.st_ino and data['device'] == stat.st_dev and data['file_bytes'] == stat.st_size
            assert data['generation'] is None or isinstance(data['generation'],int)
            assert data['extents'] and all(len(extent[2]) == 64 for extent in data['extents'])
            assert path.stat().st_mode & 0o777 == 0o600 and state.stat().st_mode & 0o777 == 0o600
            kept.append(dict(slot=data['slot'],cursor=data['cursor'],tokens=len(data['tokens']),
                             generation=data['generation'],record_sha256=sha(path),extents=len(data['extents'])))
        diagnostic['records'] = [json.loads(path.read_text()) for path in records]
        diagnostic['kept'] = kept
        (args.output/'restart-responses.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
        epoch = 2
        start(epoch)
        assert all(sha(path)==saved['record_sha256'] for path,saved in zip(records,kept)), 'adoption mutated persisted records before replay'
        for slot in (0,1):
            assert f'slot {slot} adopted a kept conversation' in (args.output/'runtime2.log').read_text()
        scalar_after = remember('restart_scalar',[post('/v1/chat/completions',body) for body in prime])
        after = parallel('restart_parallel1')
        parallel('restart_parallel2')
        def equality(left,right):
            return [choice(a,16)==choice(b,16) for a,b in zip(diagnostic[left],diagnostic[right])]
        comparisons = {f'{a}__{b}':equality(a,b) for a,b in (
            ('cold_parallel','same_process_parallel1'),
            ('same_process_parallel1','same_process_parallel2'),
            ('same_process_scalar1','same_process_scalar2'),
            ('same_process_scalar2','restart_scalar'),
            ('restart_parallel1','restart_parallel2'),
            ('same_process_parallel2','restart_parallel1'),
            ('cold_parallel','restart_parallel1'))}
        diagnostic['comparisons'] = comparisons
        diagnostic['kept'] = kept
        (args.output/'restart-responses.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
        print('GEMMA3_HTTP_RESTART_DIAGNOSTIC',json.dumps(comparisons),flush=True)
        assert all(choice(a,16)==choice(b,16) for a,b in zip(scalar_before,scalar_after)), 'scalar restart replay differs'
        for label in ('same_process_parallel1__same_process_parallel2',
                      'same_process_scalar1__same_process_scalar2',
                      'same_process_scalar2__restart_scalar',
                      'restart_parallel1__restart_parallel2',
                      'same_process_parallel2__restart_parallel1'):
            assert all(comparisons[label]), f'matched cached replay differs: {label}'
        for a,b in zip(diagnostic['same_process_parallel2'],after):
            assert choice(a,16) == choice(b,16)
            assert b['usage']['prompt_tokens_details']['cached_tokens'] > 0
        case('two-slot-clean-restart-adoption-and-exact-checkpoint-replay',lambda:dict(
            kept=kept, scalar_restart_exact=True, cached_parallel_restart_exact=True,
            cold_cached_equal=comparisons['cold_parallel__same_process_parallel1'],
            cross_geometry_scope='cold/cached and scalar/joined responses may differ; cached comparisons remain exact',
            cached_tokens=[r['usage']['prompt_tokens_details']['cached_tokens'] for r in after]))
    finally:
        try:
            stop(epoch)
        finally:
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    assert len(result['cases']) == 4 and len(result['epochs']) == 2
    assert all(case['passed'] for case in result['cases']), 'one or more HTTP cases failed'
    print('GEMMA3_HTTP_CONTROLS_PASS',json.dumps(result['epochs']),flush=True)


if __name__ == '__main__':
    main()
