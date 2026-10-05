#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive Gemma26 source and fresh native two-head/state boundary before ring stock."""
import datetime
import hashlib
import importlib.util
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

old = load('large_prefill', ROOT/'docs/experiments/gemma-prefill-large/compare.py')
pinned = load('c1_pins', ROOT/'docs/experiments/gemma-dense31-c1-norm/own_freeze.py')
BASE = '437d59f3530ec8d7b560a4127745ef5fd23b1938'
PREFIX = 'docs/experiments/gemma26-swa-ring-transfer/'
SOURCES = ['benchmarks/gemma_prefill.cc', 'docs/experiments/gemma-prefill-large/compare.py',
           'docs/experiments/gemma-dense31-c1-norm/own_freeze.py'] + [PREFIX+n for n in
           ['llama_prefill_ring.cc', 'reference.sh', 'own_freeze.py', 'compare.py', 'PROTOCOL.md']]
IDS = '6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b'
LAYOUT = 'gemma26-f16-kv-scalar-device-v1:16384:1024:16384:2048\n'
STATE_BYTES = 592445440
ARTIFACT = '4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3'
INDEX = 'e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171'
LIBRARIES = {'/app/libllama.so': '36f5cb5b9361fd7ce4c9c38ad7276367a39e753a2f4764d9aca952b29c5aba55',
             '/app/libggml.so': 'f09875d0593d950f0b1f4f57803235cb06209f774c524c005fb192000d5d7418',
             '/app/libggml-base.so': 'ed0a6a3f65472059789159569080d08e7f1dcc8bdd7a35d127efbd1b3f4e273e',
             '/app/libggml-cuda.so': '5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e'}


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': old.digest(path) if path.stat().st_size < 8388608 else stream_digest(path)}


def stream_digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while data := source.read(8*1024*1024):
            result.update(data)
    return result.hexdigest()


def retired(job):
    root = pathlib.Path.home()/'.local/share/jitllm/jobs'/job
    final = json.loads((root/'final.json').read_text())
    assert final['state'] == 'done' and final['rc'] == 0
    return {n: identity(root/n) for n in ['final.json', 'job.json', 'log']}


def environment(root, scratch):
    sources = {n: identity(root/n) for n in SOURCES}
    assert sources['benchmarks/gemma_prefill.cc']['sha256'] == '8fc6e4b377f5fa120d1dcc82f44531c2b201eb0462905fbe9ff1c68e45b772a2'
    ancestry = json.loads((scratch/'source-ancestry.json').read_text())
    assert ancestry['base'] == BASE and ancestry['tracked_production_diff'] == []
    assert ancestry['source_files'] == sources
    assert set(ancestry['dirty_paths']).issubset(SOURCES)
    ids = identity(scratch/'ids.i32')
    assert ids == {'bytes': 8227*4, 'sha256': IDS}
    headers = {n: identity(scratch/'headers'/n) for n in pinned.HEADERS}
    assert {n:v['sha256'] for n,v in headers.items()} == pinned.HEADERS
    artifact = pathlib.Path.home()/'.local/share/jitllm/m3-artifacts'/ARTIFACT
    manifest, index = identity(artifact/'manifest.json'), identity(artifact/'index.json')
    assert manifest['sha256'] == ARTIFACT and index['sha256'] == INDEX
    libraries = dict(line.split()[::-1] for line in (scratch/'libraries.sha256').read_text().splitlines())
    assert libraries == LIBRARIES
    return {'base': ancestry['base'], 'sources': sources, 'ancestry': ancestry,
            'headers': headers, 'libraries': libraries, 'ids': ids, 'manifest': manifest, 'index': index,
            'production_source_manifest': identity(scratch/'production-source-manifest.json'),
            'build_receipt': identity(root/'build/spark-native/jitllm-receipt.json'),
            'native_binary': identity(root/'build/spark-native/benchmarks/jitllm_gemma_prefill'),
            'original_binary': identity(scratch/'llama_prefill_ring')}


def record(path):
    heads = {}
    for n in ['prefill.f32', 'final.f32']:
        values, sha = old.head(path/n)
        heads[n] = {'sha256': sha, 'bytes': len(values)*4}
    state = path/'initialized-state.bin'
    assert state.stat().st_size == STATE_BYTES
    assert (path/'state-layout.txt').read_text() == LAYOUT
    return {'heads': heads, 'state': identity(state), 'layout': LAYOUT.strip()}


def main(mode, root, scratch, job, expected=None):
    env = environment(root, scratch)
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'official_retirement': retired(job)}
    if mode == 'source':
        result['environment'] = env
        destination = scratch/'source-identities.json'
    else:
        assert mode == 'own' and identity(scratch/'source-identities.json')['sha256'] == expected
        assert json.loads((scratch/'source-identities.json').read_text())['environment'] == env
        records = {n: record(scratch/n) for n in ['native-first', 'native-repeat']}
        assert records['native-first'] == records['native-repeat']
        log = pathlib.Path.home()/'.local/share/jitllm/jobs'/job/'log'
        runs = old.runs(log, 'all', '26')
        assert len(runs) == 2 and all(r['engine'] == 'PREFILL_NATIVE' for r in runs)
        assert all(r['metrics']['chunk'] == '1024' for r in runs)
        assert runs[0]['tokens'] == runs[1]['tokens']
        assert all(len(r['tokens']) == 32 for r in runs)
        assert log.read_text().count('PREFILL_STATE positions=8227 bytes=592445440 ranges=60 layout='+LAYOUT.strip()) == 2
        result.update({'source_identity': identity(scratch/'source-identities.json'), 'records': records,
                       'native_runs': runs, 'two_head_state_repeat_exact': True,
                       'all_32_head_vectors_retained': False, 'margin_noise_calibrated': False})
        destination = scratch/'native-frozen.json'
    with destination.open('x') as file:
        json.dump(result, file, indent=2, sort_keys=True); file.write('\n')
    destination.chmod(0o444)
    print(json.dumps({'identity': identity(destination), **result}, indent=2, sort_keys=True))


if __name__ == '__main__':
    assert len(sys.argv) in (5, 6)
    main(sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4],
         sys.argv[5] if len(sys.argv) == 6 else None)
