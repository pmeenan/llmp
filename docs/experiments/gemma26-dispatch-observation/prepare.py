#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive pre-load identity receipt, without rehashing model payloads."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess

import observe_controller

IMAGE = 'ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7'
LIBS = {
    'libllama.so': '36f5cb5b9361fd7ce4c9c38ad7276367a39e753a2f4764d9aca952b29c5aba55',
    'libggml.so': 'f09875d0593d950f0b1f4f57803235cb06209f774c524c005fb192000d5d7418',
    'libggml-base.so': 'ed0a6a3f65472059789159569080d08e7f1dcc8bdd7a35d127efbd1b3f4e273e',
    'libggml-cuda.so': '5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e',
}


def record(path):
    if path.stat().st_size > 128 << 20:
        raise ValueError('identity file exceeds bounded source size')
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    os.umask(0o077)
    home = pathlib.Path.home()
    root = home / '.local/share/llmp/gemma26-dispatch-observation'
    source = home / 'src/llmp-wt/m3fixb'
    previous = home / '.local/share/llmp/gemma26-packed-attention-c4'
    old = json.loads((previous / 'pre-native.json').read_text())
    generated = (root / 'controller.cu').read_bytes()
    original = (root / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_bytes()
    if observe_controller.transform(original) != generated:
        raise ValueError('generated observer differs from current exact source')
    if observe_controller.original(generated.decode()).encode() != original:
        raise ValueError('original reconstruction failed')
    for item in old['reference_headers']:
        if record(pathlib.Path(item['path'])) != item:
            raise ValueError('original client header changed')
    reference = record(root / 'llama_joined')
    if reference['sha256'] != old['reference_binary']['sha256']:
        raise ValueError('original client changed')
    inputs = record(root / 'ids.i32')
    if inputs['sha256'] != old['input']['sha256'] or inputs['bytes'] != 4096:
        raise ValueError('actual IDs changed')
    output = subprocess.check_output(['sudo', '-n', 'docker', 'run', '--rm', '--network',
                                     'none', '--read-only', '--entrypoint', 'sha256sum', IMAGE,
                                     *['/app/' + name for name in LIBS]], text=True)
    actual = {pathlib.Path(line.split()[1]).name: line.split()[0] for line in output.splitlines()}
    if actual != LIBS:
        raise ValueError('original image math changed')
    artifact = home / '.local/share/llmp/m3-artifacts/4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3'
    metadata = [record(artifact / name) for name in ('manifest.json', 'index.json')]
    if [m['sha256'] for m in metadata] != [
            '4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3',
            'e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171']:
        raise ValueError('prepared artifact metadata changed')
    raw = home / '.local/share/llmp/reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf'
    if raw.stat().st_size != 16947541728:
        raise ValueError('approved raw artifact size changed')
    files = sorted(p for p in (root / 'ggml').rglob('*') if p.is_file())
    if len(files) > 5000 or sum(p.stat().st_size for p in files) > 256 << 20:
        raise ValueError('unbounded source closure')
    tools = sorted((source / 'docs/experiments/gemma26-dispatch-observation').glob('*'))
    if any(not p.is_file() for p in tools):
        raise ValueError('unexpected diagnostic source entry')
    result = {
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'base': '2eb057476dea83d89a9bbc2453f799b0a759f609',
        'diagnostic_sources': [record(p) for p in tools],
        'original_source': record(root / 'ggml/src/ggml-cuda/ggml-cuda.cu'),
        'generated_source': record(root / 'controller.cu'),
        'controller_library': record(root / 'controller.so'),
        'original_client': reference,
        'original_client_source': record(source / 'docs/experiments/gemma-joined-serving/llama_joined.cc'),
        'original_client_headers': old['reference_headers'],
        'controller_source_closure': [record(p) for p in files],
        'sdk_receipt': record(home / '.local/share/llmp/sdk/aarch64-c09daba6ac31edee/sdk.json'),
        'original_image': IMAGE, 'original_image_libraries': actual,
        'actual_input': inputs, 'artifact_metadata': metadata,
        'raw_size': raw.stat().st_size,
        'approved_raw_sha256': 'f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f',
        'payload_qualification': 'payload identity inherits existing verified immutable approved assets; no rescan',
    }
    path = root / 'pre-observation.json'
    with path.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    path.chmod(0o400)
    print('pre_observation_sha256=' + record(path)['sha256'])
    print('compiled_controller_sha256=' + result['controller_library']['sha256'])
    print('original_reconstructed_sha256=' + observe_controller.PIN)


if __name__ == '__main__':
    main()
