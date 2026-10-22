#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One frozen literal teacher-score request under the established native Service owner."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import urllib.error
import urllib.request

import control


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'runtime', 'checkpoint', 'store', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    registration = control.registration(args.registration)
    control.inputs()
    control.checked(args.runtime, control.RUNTIME_SHA)
    control.checked(args.repo / 'tools/speed_gate.py', control.SERVICE_SHA)
    raw = control.raw_reference()
    # Hash metadata only, never rescan weight payloads. The staged source and
    # published artifact already have their own deep-validation receipts.
    source_metadata = {name: control.checked(args.checkpoint / name, digest)
                       for name, digest in raw.METADATA_HASHES.items()}
    source_metadata['chat_template.jinja'] = control.checked(
        args.checkpoint / 'chat_template.jinja', control.TEMPLATE_SHA)
    artifact_metadata = {name: control.checked(args.store / control.ARTIFACT / name, digest)
                         for name, digest in control.ARTIFACT_METADATA.items()}
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.out.resolve()
    config_path = out / 'native.toml'
    config_path.write_text(f'''schema_version = 2
[storage]
data_dir = {json.dumps(str(out / 'state'))}
installed = {json.dumps(str(args.store.resolve()))}
[models.gemma26]
artifact = "{control.ARTIFACT}"
tokenizer = {json.dumps(str((args.checkpoint / 'tokenizer.json').resolve()))}
chat_template = {json.dumps(str((args.checkpoint / 'chat_template.jinja').resolve()))}
context = 4096
max_slots = 1
prefill_chunk = 512
kv_cache_encoding = "f16"
temperature = 0.0
top_p = 1.0
top_k = 0
[client]
bind = "127.0.0.1"
port = {control.PORT}
''')
    here = Path(__file__).resolve().parent
    helper_paths = [here / name for name in ('native.py', 'control.py', 'ids.json',
                                            'source-hashes.json', 'pair-prereg.json', 'PROTOCOL.md')]
    helper_paths += [here.parent / 'reference.py', here.parent / 'paris/control.py',
                     here.parent.parent / 'qwen27/contract.py', args.repo / 'tools/speed_gate.py']
    before = {str(p): control.sha(p) for p in helper_paths}
    request_bytes = control.encoded(control.body())
    (out / 'request.json').write_bytes(request_bytes)
    receipt = dict(format='llmp-gemma26-short-native-launch/1', registration_commit=registration,
                   input_sha256=control.INPUT_SHA, runtime_sha256=control.RUNTIME_SHA,
                   runtime_source_manifest_sha256=control.RUNTIME_SOURCE_SHA,
                   protocol_sha256=control.sha(here / 'PROTOCOL.md'),
                   config_sha256=control.sha(config_path), helper_hashes_before=before,
                   source_metadata=source_metadata, artifact_metadata=artifact_metadata,
                   request_sha256=control.sha(out / 'request.json'), collection_completed=False)
    (out / 'launch.json').write_bytes(control.encoded(receipt))
    dependencies = subprocess.run(['ldd', str(args.runtime.resolve())], check=True,
                                  capture_output=True, timeout=10)
    (out / 'dependencies.txt').write_bytes(dependencies.stdout + dependencies.stderr)
    control.require(b'not found' not in dependencies.stdout + dependencies.stderr, 'missing runtime library')
    sys.path.insert(0, str(args.repo / 'tools'))
    from speed_gate import Service
    service = None
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        service = Service(args.runtime.resolve(), config_path, out, control.PORT, {})
        receipt['ready_seconds'] = service.wait_ready(timeout=420)
        request = urllib.request.Request(f'http://127.0.0.1:{control.PORT}/v1/completions',
                                         data=request_bytes, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_bytes = response.read()
                receipt['http_status'] = response.status
        except urllib.error.HTTPError as error:
            response_bytes = error.read()
            (out / 'response.json').write_bytes(response_bytes)
            receipt['http_status'] = error.code
            raise
        (out / 'response.json').write_bytes(response_bytes)
        rows = control.native_output(json.loads(response_bytes))
        result = control.write_losses(out, rows)
        result.update(control='gemma26-short-likelihood', input_sha256=control.INPUT_SHA,
                      registration_commit=registration, response_sha256=control.sha(out / 'response.json'))
        (out / 'result.json').write_bytes(control.encoded(result))
        receipt['collection_completed'] = True
    except BaseException as error:
        receipt['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        try:
            if service is not None:
                receipt['retirement'] = service.stop(timeout=20)
        finally:
            receipt['runtime_sha256_after'] = control.sha(args.runtime)
            receipt['config_sha256_after'] = control.sha(config_path)
            receipt['helper_hashes_after'] = {str(p): control.sha(p) for p in helper_paths}
            receipt['source_metadata_after'] = {
                name: control.sha(args.checkpoint / name) for name in source_metadata}
            receipt['artifact_metadata_after'] = {
                name: control.sha(args.store / control.ARTIFACT / name) for name in artifact_metadata}
            (out / 'launch.json').write_bytes(control.encoded(receipt))
            for sig, handler in previous.items(): signal.signal(sig, handler)
        control.require(receipt.get('retirement') == {'signal': 'TERM', 'returncode': 0},
                        'native retirement unconfirmed')
        control.require(receipt['runtime_sha256_after'] == control.RUNTIME_SHA and
                        receipt['config_sha256_after'] == receipt['config_sha256'] and
                        receipt['helper_hashes_after'] == before and
                        receipt['source_metadata_after'] == source_metadata and
                        receipt['artifact_metadata_after'] == artifact_metadata,
                        'native source/runtime/config changed')


if __name__ == '__main__':
    main()
