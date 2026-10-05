#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate actual compiled identities without rescanning model payloads."""
import datetime
import json
import os
import pathlib
import runpy
import re
import sys

tool = pathlib.Path(__file__).resolve().parent
capture = runpy.run_path(str(tool / 'capture_controller.py'))
snapshot = runpy.run_path(str(tool / 'source_snapshot.py'))
record = snapshot['record']
LIBS = {
    'libllama.so': '36f5cb5b9361fd7ce4c9c38ad7276367a39e753a2f4764d9aca952b29c5aba55',
    'libggml.so': 'f09875d0593d950f0b1f4f57803235cb06209f774c524c005fb192000d5d7418',
    'libggml-base.so': 'ed0a6a3f65472059789159569080d08e7f1dcc8bdd7a35d127efbd1b3f4e273e',
    'libggml-cuda.so': '5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e',
}


def main():
    if len(sys.argv) != 3:
        raise ValueError('SOURCE_SNAPSHOT NEW_PRE_CAPTURE_RECEIPT')
    source_manifest, output = map(pathlib.Path, sys.argv[1:])
    home = pathlib.Path.home()
    source = home / 'src/jitLLM-wt/m3fixb'
    root = home / '.local/share/jitllm/gemma26-late-moe'
    declared = json.loads(source_manifest.read_text())
    if declared['base'] != snapshot['BASE'] or len(declared['files']) > 5000:
        raise ValueError('wrong source declaration')
    for path, item in declared['files'].items():
        pure = pathlib.PurePosixPath(path)
        if pure.is_absolute() or '..' in pure.parts or record(source / path, 32 << 20) != item:
            raise ValueError('compiled checkout differs from source snapshot: ' + path)
    original = (root / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_bytes()
    generated = (root / 'controller.cu').read_bytes()
    if capture['transform'](original) != generated or capture['original'](generated.decode()).encode() != original:
        raise ValueError('capture edits original defining controller')
    if record(root / 'ggml/src/ggml-cuda/ggml-cuda.cu') != {
            'bytes': 236684, 'sha256': '523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634'}:
        raise ValueError('original controller pin differs')
    for name in ('capture.h', 'capture_main.cc'):
        if record(root / name) != record(tool / name):
            raise ValueError('compiled capture client/header differs')
    if record(root / 'client.cc')['sha256'] != 'c80b73c85453ece170eda57bc0725cdef5b4e0af3d7cacf3d1a8bee0d1cfa19e' or \
            record(root / 'untouched-client')['sha256'] != '01889d8c4a281110f611cedcab43235a9ac80bcc21abf094d982f14dd2112f61':
        raise ValueError('original recipe client differs')
    if record(root / 'ids.i32') != {'bytes': 4096, 'sha256': 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'}:
        raise ValueError('canonical input IDs differ')
    libs = {pathlib.Path(line.split()[1]).name: line.split()[0]
            for line in (root / 'libraries.sha256').read_text().splitlines()}
    if libs != LIBS:
        raise ValueError('original floating libraries differ')
    retired = json.loads((root / 'build-container-retired.json').read_text())
    cid = (root / 'build.cid').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', cid) or retired != {
            'cid': cid, 'name': 'jitllm-gemma26-late-moe-build', 'owner_label': 'gemma26-late-moe',
            'container_absent_after_checked_docker_query': True}:
        raise ValueError('build container not retired')
    sdk = home / '.local/share/jitllm/sdk/aarch64-c09daba6ac31edee/sdk.json'
    artifact = home / '.local/share/jitllm/m3-artifacts/4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3'
    metadata = {name: record(artifact / name) for name in ('manifest.json', 'index.json')}
    if metadata['manifest.json']['sha256'] != artifact.name or metadata['index.json']['sha256'] != \
            'e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171':
        raise ValueError('approved artifact metadata differs')
    raw = home / '.local/share/jitllm/reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf'
    if raw.stat().st_size != 16947541728:
        raise ValueError('approved raw size differs')
    files = sorted(p for p in (root / 'ggml').rglob('*') if p.is_file())
    if len(files) > 5000 or sum(p.stat().st_size for p in files) > (256 << 20):
        raise ValueError('unbounded original header/source closure')
    previous = home / '.local/share/jitllm/gemma26-packed-attention-c4'
    old = json.loads((tool.parent / 'gemma26-packed-attention-c4/provenance.json').read_text())
    header_records = {}
    for expected in old['original_public_headers']:
        path = pathlib.Path(expected['path'])
        relative = path.relative_to(previous / 'headers')
        current = record(root / 'headers' / relative)
        if current != {'bytes': expected['bytes'], 'sha256': expected['sha256']}:
            raise ValueError('copied C API header pin differs')
        header_records[str(relative)] = current
    header_files = sorted(p for p in (root / 'headers').rglob('*') if p.is_file())
    actual_headers = {str(p.relative_to(root / 'headers')): record(p) for p in header_files}
    if actual_headers != header_records or len(actual_headers) != 8:
        raise ValueError('copied C API header closure differs')
    source_flags = source / 'build/spark-native/compile_commands.json'
    entries = json.loads(source_flags.read_text())
    required = {'gemma_moe.cu', 'topk-moe.cu', 'moe-weighted-reduction.cu', 'softmax.cu',
                'argsort.cu', 'getrows.cu', 'sumrows.cu', 'clamp.cu', 'binbcast.cu'}
    defining = {}
    for entry in entries:
        name = pathlib.Path(entry['file']).name
        if name in required:
            if name in defining:
                raise ValueError('ambiguous native defining translation unit')
            path = pathlib.Path(entry['file'])
            if not path.is_absolute():
                path = pathlib.Path(entry['directory']) / path
            defining[name] = {'source': record(path), 'actual_compile_entry': entry}
    if set(defining) != required:
        raise ValueError('missing actual native defining translation unit')
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'base': snapshot['BASE'], 'source_snapshot': record(source_manifest),
              'source_files': declared['files'], 'generated': record(root / 'controller.cu'),
              'controller_library': record(root / 'controller.so'),
              'capture_client': record(root / 'capture-client'), 'original_operator_client': record(root / 'oracle'),
              'native_operator_client': record(source / 'build/spark-native/benchmarks/jitllm_gemma26_late_moe'),
              'native_build_receipt': record(source / 'build/spark-native/jitllm-receipt.json'),
              'native_compile_commands': record(source_flags),
              'native_defining_translation_units': defining, 'actual_C_API_headers': actual_headers,
              'sdk_receipt': record(sdk), 'original_libraries': libs,
              'source_and_headers': {str(p.relative_to(root)): record(p) for p in files},
              'canonical_input': record(root / 'ids.i32'), 'artifact_metadata': metadata,
              'raw_size': raw.stat().st_size, 'approved_raw_sha256': 'f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f',
              'payload_identity': 'inherited verified immutable approved payload; no historical rescan',
              'build_container_retirement': retired,
              'compiler_qualification': 'rebuilt controller/integer pointer kernel and thin host clients; original floating operator library unchanged'}
    os.umask(0o077)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    output.chmod(0o400)


if __name__ == '__main__':
    main()
