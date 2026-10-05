#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Create exclusive inputs, then authenticate the compiled checkout before native work."""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

HOME = pathlib.Path.home()
ROOT = HOME / '.local/share/jitllm/gemma26-compound-packed-c4'
REPO = HOME / 'src/jitLLM-wt/m3fixb'
OLD = HOME / '.local/share/jitllm/gemma26-packed-attention-c4'
ID_SHA = 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'
CLIENT_SHA = '01889d8c4a281110f611cedcab43235a9ac80bcc21abf094d982f14dd2112f61'
IMAGE = 'ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7'
LIBS = {
    'libllama.so': '36f5cb5b9361fd7ce4c9c38ad7276367a39e753a2f4764d9aca952b29c5aba55',
    'libggml.so': 'f09875d0593d950f0b1f4f57803235cb06209f774c524c005fb192000d5d7418',
    'libggml-base.so': 'ed0a6a3f65472059789159569080d08e7f1dcc8bdd7a35d127efbd1b3f4e273e',
    'libggml-cuda.so': '5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e',
}


def record(path, limit=128 << 20):
    size = path.stat().st_size
    if size > limit:
        raise ValueError('unbounded identity file')
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': size, 'sha256': digest.hexdigest()}


def exclusive(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    path.chmod(0o400)


def inputs(manifest):
    ROOT.mkdir(mode=0o700, exist_ok=False)
    old = json.loads((OLD / 'pre-native.json').read_text())
    for item in old['reference_headers']:
        if record(pathlib.Path(item['path'])) != item:
            raise ValueError('original headers changed')
    if record(OLD / 'ids.i32')['sha256'] != ID_SHA:
        raise ValueError('actual input changed')
    shutil.copyfile(OLD / 'ids.i32', ROOT / 'ids.i32')
    shutil.copytree(OLD / 'headers', ROOT / 'headers')
    shutil.copyfile(manifest, ROOT / 'expected-source.json')
    print('Exclusive immutable source-local inputs prepared; no model payload scan')


def prefreeze():
    expected = json.loads((ROOT / 'expected-source.json').read_text())
    if not isinstance(expected, list) or len(expected) > 5000:
        raise ValueError('unbounded checkout inventory')
    sources = []
    seen = set()
    for entry in expected:
        name = entry['relative_path']
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or name in seen:
            raise ValueError('invalid checkout inventory path')
        seen.add(name)
        item = record(REPO / name)
        if item['bytes'] != entry['bytes'] or item['sha256'] != entry['sha256']:
            raise ValueError('source checksum sync differs: ' + name)
        if name.startswith(('src/', 'benchmarks/', 'third_party/', 'toolchains/',
                            'docs/experiments/gemma26-compound-packed-c4/')) or name in (
                                'CMakeLists.txt', 'CMakePresets.json',
                                'docs/experiments/gemma26-packed-attention-c4/compare.py'):
            sources.append(item)
    if sum(item['bytes'] for item in sources) > 256 << 20:
        raise ValueError('unbounded source bytes')
    client = record(ROOT / 'llama_joined')
    ids = record(ROOT / 'ids.i32')
    if client['sha256'] != CLIENT_SHA or ids['sha256'] != ID_SHA or ids['bytes'] != 4096:
        raise ValueError('original client or actual input differs')
    output = subprocess.check_output(['sudo', '-n', 'docker', 'run', '--rm', '--network', 'none',
                                     '--read-only', '--entrypoint', 'sha256sum', IMAGE,
                                     *['/app/' + name for name in LIBS]], text=True)
    actual = {pathlib.Path(line.split()[1]).name: line.split()[0] for line in output.splitlines()}
    if actual != LIBS:
        raise ValueError('original image libraries changed')
    artifact = HOME / '.local/share/jitllm/m3-artifacts/4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3'
    metadata = [record(artifact / name) for name in ('manifest.json', 'index.json')]
    if [item['sha256'] for item in metadata] != [
            '4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3',
            'e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171']:
        raise ValueError('prepared artifact metadata changed')
    raw = HOME / '.local/share/jitllm/reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf'
    if raw.stat().st_size != 16947541728:
        raise ValueError('approved raw model size changed')
    receipt = record(REPO / 'build/spark-native/jitllm-receipt.json')
    result = {
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'production_base': 'c3095754e5820acb302a26abd4eb4593465c99fa',
        'source_files': sources,
        'complete_source_manifest_sha256': record(ROOT / 'expected-source.json')['sha256'],
        'native_binary': record(REPO / 'build/spark-native/benchmarks/jitllm_gemma26_attention_compound', 256 << 20),
        'metadata_binary': record(REPO / 'build/spark-native/benchmarks/jitllm_gemma26_attention_packed_metadata', 256 << 20),
        'reference_binary': client,
        'reference_client': record(REPO / 'docs/experiments/gemma-joined-serving/llama_joined.cc'),
        'reference_headers': [record(p) for p in sorted((ROOT / 'headers').rglob('*')) if p.is_file()],
        'image': IMAGE, 'image_libraries': actual, 'build_receipt': receipt,
        'input': ids, 'artifact_metadata': metadata, 'raw_model_bytes': raw.stat().st_size,
        'approved_raw_sha256': 'f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f',
        'payload_qualification': 'Inherits previously verified immutable approved assets; no payload rescan',
    }
    exclusive(ROOT / 'pre-native.json', result)
    print(json.dumps({'pre_native_sha256': record(ROOT / 'pre-native.json')['sha256'],
                      'native_binary': result['native_binary'], 'build_receipt': receipt,
                      'authenticated_checkout_files': len(expected)}))


if __name__ == '__main__':
    os.umask(0o077)
    if len(sys.argv) == 3 and sys.argv[1] == 'inputs':
        inputs(pathlib.Path(sys.argv[2]))
    elif sys.argv[1:] == ['prefreeze']:
        prefreeze()
    else:
        raise SystemExit('prepare.py inputs EXPECTED_CHECKOUT_JSON | prefreeze')
