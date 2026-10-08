#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Four fixed fresh natural turns via the existing runtime/server callers."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path.home() / '.local/share/llmp/gemma-production-natural'
INPUT = Path.home() / '.local/share/llmp/gemma-input-preparation'
IMAGE = 'ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7'
STOPS = ['<turn|>', '<|tool_response>', '<|channel>', '<|tool_call>']
STOP_IDS = [106, 50, 100, 48]
NAMES = ['explanation', 'arithmetic', 'instruction', 'retrieval']


def guard(profile):
    subprocess.run(['sha256sum', '--check', str(ROOT / f'run{profile}.sha256')], check=True)
    facts = json.loads((ROOT / 'facts1.json').read_text())
    assert facts['native_binary_sha256'] == 'fe15cdb9abda6b11aad468f4c264aec500fe40538141ed5b3ab2ee8f4598dd2a'
    prompts = json.loads((ROOT / 'natural-prompts.json').read_text())['cases']
    assert [p['name'] for p in prompts] == NAMES
    ids = []
    for case in facts['profiles'][profile]['cases']:
        raw = (INPUT / f'inputs-{profile}' / case['name'] / 'ids.i32').read_bytes()
        assert len(raw) == case['rows'] * 4 and hashlib.sha256(raw).hexdigest() == case['input_sha256']
        tokens = list(struct.unpack(f'<{case["rows"]}i', raw))
        assert tokens[0] == 2 and tokens.count(2) == 1 and all(0 <= t < 262144 for t in tokens)
        ids.append(tokens)
    return facts['profiles'][profile], prompts, ids


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def native(profile):
    facts, prompts, ids = guard(profile)
    out = ROOT / profile
    out.mkdir(mode=0o700)
    config = out / 'native.toml'
    config.write_bytes((ROOT / f'native-{profile}.toml').read_bytes())
    config.chmod(0o600)
    assert not (out / 'enrollment').exists() and not (out / 'enrollment.lock').exists()
    argv = [str(ROOT / 'llmp-runtime-candidate'), '--config', str(config),
            '--anchor', str(out / 'enrollment'), 'chat', '--fresh', '--plain',
            '--max-tokens', '128', '--report', str(out / 'native-report.json')]
    for p in prompts:
        argv += ['--turn', f'gemma{profile}', p['user']]
    env = dict(os.environ, LD_LIBRARY_PATH='/home/pmeenan/src/llmp-wt/m3fixb/build/spark-native/cublas')
    with (out / 'native.log').open('x') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    # Runtime return0 includes Server.TearDown success; a report alone is insufficient.
    report = json.loads((out / 'native-report.json').read_text())
    assert report['command'] == 'chat' and len(report['turns']) == 4
    for case, turn, supplied in zip(prompts, report['turns'], ids):
        assert turn['model'] == f'gemma{profile}' and turn['prompt_ids'] == supplied
        assert turn['prefill_chunk'] == facts['max_rows'] and turn['reused'] == 0
        assert turn['speculative'] is False and 1 <= turn['generated'] <= 128
        assert turn['generated'] == len(turn['tokens']) and isinstance(turn['stopped'], bool)
        assert all(isinstance(t, int) and 0 <= t < 262144 for t in turn['tokens'])
        assert isinstance(turn['text'], str)
        if turn['stopped']:
            assert turn['tokens'][-1] in STOP_IDS
    guard(profile)
    print(f'NATURAL_NATIVE_RETIRED profile={profile} turns=4 full_prompt_ids_exact=1 runtime_return0=1', flush=True)


def public(profile):
    facts, prompts, ids = guard(profile)
    model = Path.home() / '.local/share/llmp/reference-models' / facts['raw_gguf']['name']
    assert model.is_file() and model.stat().st_size == facts['raw_gguf']['bytes']
    out = ROOT / f'public-{profile}'
    out.mkdir(mode=0o700)
    name = f'task71-natural-reference{profile}-1'
    cid = out / f'{name}.cid'
    container = 'llmp-gemma26-late-moe-' + name
    key = uuid.uuid4().hex
    port = 18171
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def docker(*args, **kwargs):
        return subprocess.run(['sudo', '-n', 'docker', *args], check=True, **kwargs)

    def request(path, body=None, timeout=180):
        raw = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}' + path, data=raw,
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    try:
        docker('create', '--name', container, '--cidfile', str(cid), '--pull', 'never',
               '--label', 'llmp.observer=gemma26-late-moe', '--device', 'nvidia.com/gpu=all',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--user', f'{os.getuid()}:{os.getgid()}', '--pids-limit', '512',
               '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m', '--publish', f'127.0.0.1:{port}:8080',
               '--mount', f'type=bind,src={model},dst=/model.gguf,readonly',
               '--env', 'CUDA_DISABLE_PTX_JIT=1', '--entrypoint', '/app/llama-server', IMAGE,
               '--model', '/model.gguf', '--alias', 'reference-gemma4', '--host', '0.0.0.0',
               '--port', '8080', '--n-gpu-layers', '999', '--ctx-size', '4096', '--parallel', '1',
               '--threads', '8', '--threads-batch', '8', '--batch-size', str(facts['max_rows']),
               '--ubatch-size', str(facts['max_rows']), '--cache-type-k', 'f16', '--cache-type-v', 'f16',
               '--flash-attn', 'on', '--cache-ram', '0', '--no-webui', '--verbosity', '4', '--special',
               '--api-key', key, stdout=subprocess.DEVNULL)
        docker('start', container, stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert docker('inspect', '--format', '{{.State.Running}}', container,
                          capture_output=True, text=True).stdout.strip() == 'true'
            try:
                if request('/health', timeout=2).get('status') == 'ok':
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError('reference server health timeout')
        props = request('/props')
        save(out / 'props.json', props)
        assert props['total_slots'] == 1 and props['default_generation_settings']['n_ctx'] == 4096
        startup = docker('logs', container, capture_output=True, text=True)
        text = startup.stdout + startup.stderr
        (out / 'startup.log').write_text(text)
        caches = re.findall(r'llama_kv_cache: size =.*?\(\s*(\d+) cells,\s*(\d+) layers,\s*1/1 seqs\), K \(f16\):.*?V \(f16\):', text)
        expected = [('4096', '10'), ('1280', '50')] if profile == '31' else [('4096', '5'), ('2048', '25')]
        assert sorted(caches) == sorted(expected)
        assert re.search(rf'n_batch\s*=\s*{facts["max_rows"]}\b', text) and re.search(rf'n_ubatch\s*=\s*{facts["max_rows"]}\b', text)
        assert f'offloaded {facts["layers"] + 1}/{facts["layers"] + 1} layers to GPU' in text
        # Use already qualified full chat IDs, not another template render/tokenization.
        for case, supplied in zip(prompts, ids):
            result = request('/completion', dict(prompt=supplied, n_predict=128, temperature=0,
                             seed=0, cache_prompt=False, id_slot=0, return_tokens=True, stop=STOPS,
                             repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0))
            save(out / (case['name'] + '.json'), result)
            assert isinstance(result['content'], str) and 1 <= len(result['tokens']) <= 128
            assert result['tokens_predicted'] == len(result['tokens'])
            assert all(isinstance(t, int) and 0 <= t < 262144 for t in result['tokens'])
            assert result['stop_type'] in ('eos', 'word', 'limit', 'none')
        docker('stop', '--time', '30', container, stdout=subprocess.DEVNULL)
    finally:
        if cid.exists():
            try:
                with (out / 'server.log').open('x') as log:
                    docker('logs', container, stdout=log, stderr=subprocess.STDOUT)
            finally:
                subprocess.run(['python3', '-B', str(INPUT / 'container_retire.py'), str(out), name], check=True)
    proof = json.loads((out / (name + '-container-retired.json')).read_text())
    assert proof['container_absent_after_checked_docker_query'] is True
    guard(profile)
    print(f'NATURAL_REFERENCE_RETIRED profile={profile} responses=4 container_absent=1', flush=True)


def assess(profile):
    facts, prompts, ids = guard(profile)
    native_report = json.loads((ROOT / profile / 'native-report.json').read_text())
    results = []
    for case, native_turn, supplied in zip(prompts, native_report['turns'], ids):
        public_turn = json.loads((ROOT / f'public-{profile}' / (case['name'] + '.json')).read_text())
        assert native_turn['prompt_ids'] == supplied
        nstop = native_turn['tokens'][-1] if native_turn['stopped'] else None
        pstop = public_turn['tokens'][-1] if public_turn['stop_type'] in ('eos', 'word') else None
        complete_native = native_turn['stopped'] and nstop not in (48, 100)
        complete_public = public_turn['stop_type'] in ('eos', 'word') and not public_turn['truncated'] and pstop not in (48, 100)
        row = dict(name=case['name'], native_text=native_turn['text'], public_text=public_turn['content'],
                   native_tokens=native_turn['tokens'], public_tokens=public_turn['tokens'],
                   native_stopped=native_turn['stopped'], public_stop_type=public_turn['stop_type'],
                   public_stopping_word=public_turn['stopping_word'], native_terminal_stop_id=nstop,
                   public_terminal_stop_id=pstop, complete_native=complete_native, complete_public=complete_public,
                   text_byte_equal=native_turn['text'].encode() == public_turn['content'].encode(),
                   token_ids_equal=native_turn['tokens'] == public_turn['tokens'])
        if 'expected' in case:
            row.update(expected=case['expected'], native_exact_outer_whitespace_only=complete_native and native_turn['text'].strip() == case['expected'],
                       public_exact_outer_whitespace_only=complete_public and public_turn['content'].strip() == case['expected'])
        else:
            row['explanation_semantic_two_sentence_assessment'] = 'pending full-answer review; no automated semantic pass'
        results.append(row)
    save(ROOT / f'assessment-{profile}.json', dict(profile=profile, cases=results,
         limits_never_complete_pass=True, explanation_semantic_review_pending=True,
         short_retrieval_not_depth=True, no_corpus_or_batch_quality_claim=True))
    guard(profile)


if __name__ == '__main__':
    os.umask(0o077)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if len(sys.argv) != 3 or sys.argv[1] not in ('native', 'public', 'assess') or sys.argv[2] not in ('26', '31'):
        raise SystemExit('natural.py native|public|assess 26|31')
    globals()[sys.argv[1]](sys.argv[2])
