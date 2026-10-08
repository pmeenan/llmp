#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit scalar context8448 HTTP boundary; repeated input is not corpus quality."""
import argparse
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
import time
import urllib.error
import urllib.request

ARTIFACT = '8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'binding', 'installed', 'library-path', 'depth-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    assert sha(args.runtime) == binding['paths']['build/spark-native/src/runtime/llmp-runtime']['sha256']
    assert binding['receipt']['official'] and binding['receipt']['cuda']
    for name in ('libcublas.so.13', 'libcublasLt.so.13'):
        assert sha(args.library_path / name) == binding['paths']['build/spark-native/lib/llmp/' + name]['sha256']
    env = {**os.environ, 'LD_LIBRARY_PATH': str(args.library_path.resolve())}
    quality = json.loads((args.depth_root / 'quality.json').read_text())
    cycle = json.loads((args.depth_root / 'cycle.json').read_text())
    assert sha(args.depth_root / 'quality.json') == binding['depth']['quality_sha256']
    assert sha(args.depth_root / 'cycle.json') == binding['depth']['cycle_sha256']
    assert quality['passed'] and quality['positive_margin_differences'] == 0
    raw = (args.depth_root / 'input/ids.i32').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == quality['ids_sha256'] and len(raw) == 8256 * 4
    ids = struct.unpack('<8256i', raw)
    chosen_path = args.depth_root / 'native-cycle1/chosen.i32'
    assert len(set(cycle['chosen_sha256'])) == 1 and len(cycle['chosen_sha256']) == 4
    assert sha(chosen_path) == cycle['chosen_sha256'][1] == binding['depth']['natural_choices_sha256']
    expected = list(struct.unpack('<64i', chosen_path.read_bytes()))
    assert all(0 <= token < 262208 for token in expected)
    args.output.mkdir(mode=0o700)
    for name in ('spill', 'state', 'checkpoints'):
        (args.output / name).mkdir(mode=0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    config = args.output / 'node.toml'
    config.write_text('schema_version=2\n[storage]\n' +
                      f'installed="{args.installed.resolve()}"\n' + ''.join(
                          f'{name}="{args.output.resolve()}/{name}"\n'
                          for name in ('spill', 'state', 'checkpoints')) +
                      f'[models.gemma3]\nartifact="{ARTIFACT}"\ncontext=8448\n'
                      'prefill_chunk=128\nmax_slots=1\n' +
                      f'[client]\nbind="127.0.0.1:{port}"\n')
    config.chmod(0o600)
    command = [str(args.runtime.resolve()), '--config', str(config.resolve()),
               '--anchor', str((args.output / 'enrollment').resolve())]
    inspected = subprocess.run([*command, 'settings', '--json'], check=True, env=env,
                               capture_output=True, text=True, timeout=30)
    settings = json.loads(inspected.stdout)
    for name, value in [('context', 8448), ('prefill_chunk', 128), ('max_slots', 1)]:
        setting = settings['models'][0]['settings'][name]
        assert setting['value'] == value and setting['source'] == 'override'
    result = dict(scope='one scalar repeated-text boundary; no throughput/corpus-quality claim',
                  runtime_sha256=sha(args.runtime), binding_sha256=sha(args.binding),
                  harness_sha256=sha(Path(__file__)), settings=settings, cases=[])
    process = None
    log = (args.output / 'runtime.log').open('x')
    base = f'http://127.0.0.1:{port}'

    def post(body):
        request = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def body(prompt, count=64):
        return dict(model='gemma3', prompt=prompt, max_tokens=count, temperature=0,
                    logprobs=2, return_tokens_as_token_ids=True)

    def check(response, count):
        row = response['choices'][0]
        assert row['finish_reason'] == 'length' and response['usage']['completion_tokens'] == count
        scores = row['logprobs']
        selected = [int(n.removeprefix('token_id:')) for n in scores['tokens']]
        assert len(selected) == count and all(0 <= n < 262208 for n in selected)
        assert len(scores['token_logprobs']) == len(scores['top_logprobs']) == count
        assert all(math.isfinite(n) for n in scores['token_logprobs'])
        assert all(len(d) == 2 and all(math.isfinite(n) for n in d.values())
                   for d in scores['top_logprobs'])
        return selected

    def case(name, fn):
        try:
            entry = dict(name=name, passed=True, detail=fn())
        except Exception as error:
            entry = dict(name=name, passed=False, error=f'{type(error).__name__}: {error}')
        result['cases'].append(entry)
        print(json.dumps(entry), flush=True)

    try:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        deadline = time.monotonic() + 90
        while True:
            assert process.poll() is None, 'runtime exited at startup'
            try:
                with urllib.request.urlopen(base + '/v1/models', timeout=1) as response:
                    assert [m['id'] for m in json.load(response)['data']] == ['gemma3']
                break
            except (OSError, urllib.error.URLError):
                assert time.monotonic() < deadline, 'startup deadline'
                time.sleep(.1)

        def boundary():
            responses = []
            for name in ('first', 'repeat1', 'repeat2'):
                response = post(body(list(ids[:8192])))
                (args.output / (name + '.json')).write_text(json.dumps(response, indent=2) + '\n')
                assert check(response, 64) == expected, 'public IDs differ from frozen own control'
                assert response['usage']['prompt_tokens'] == 8192
                responses.append(response)
            # Literal completions publish cached_tokens=0 and do not use the
            # chat checkpoint boundary. Compare identical full-prefill requests.
            for response in responses:
                assert response['choices'] == responses[0]['choices']
                assert response['usage'] == responses[0]['usage']
                assert response['usage']['prompt_tokens_details']['cached_tokens'] == 0
            return dict(prefix=8192, continuation=64, own_choices_exact=True,
                        literal_full_prefill_repeats_exact=True, cached_tokens=0)
        case('scalar-8k-boundary-and-literal-repeat', boundary)

        def refusal():
            try:
                post(body([2] * 8449, 1))
                raise AssertionError('over-context input admitted')
            except urllib.error.HTTPError as error:
                assert 400 <= error.code < 500
                error.read()
            check(post(body([2, 105], 1)), 1)
            return dict(over_context_refused=True, healthy_after_refusal=True)
        case('capacity-refusal-preserves-serving', refusal)
    finally:
        retirement_error = None
        try:
            if process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                try:
                    rc = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise AssertionError('runtime retirement deadline')
                finally:
                    log.close()
                result['exit'] = rc
                assert rc == 0, rc
                rows = re.findall(r'Gemma3 serving final gemma3: (\{[^\n]*\})',
                                  (args.output / 'runtime.log').read_text())
                assert len(rows) == 1, 'drained metadata absent'
                extra = json.loads(rows[0])
                assert extra['recipe'] == 'bounded-serving' and extra['context'] == 8448
                assert extra['configured_slots'] == 1 and extra['max_rows'] == 128
                assert extra['joined_groups'] == extra['joined_prefill_groups'] == 0
                result['final'] = extra
            else:
                log.close()
        except Exception as error:
            retirement_error = error
            result['retirement_error'] = f'{type(error).__name__}: {error}'
        finally:
            (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        if retirement_error is not None:
            raise retirement_error
    assert len(result['cases']) == 2 and all(case['passed'] for case in result['cases'])


if __name__ == '__main__':
    main()
