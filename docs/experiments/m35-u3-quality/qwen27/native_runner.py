#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One supervised native process and fresh state directory for one Qwen collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from contract import encoded, frozen, metadata, select_cases, sha
from lifecycle import restore_signals, signals, stop
from reference import require_validation, validation_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'runtime', 'checkpoint', 'store', 'protocol', 'inputs', 'validation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prereg-commit', required=True)
    parser.add_argument('--phase', choices=['short', 'depth', 'behavior'], required=True)
    parser.add_argument('--repeat', type=int, choices=[1, 2], default=1)
    parser.add_argument('--case')
    args = parser.parse_args()
    spec, _, behavior, protocol_sha = frozen(args.protocol, args.inputs, args.prereg_commit)
    select_cases(args.phase, args.case, behavior)
    require_validation(args.validation, validation_identity(protocol_sha, args.prereg_commit))
    metadata(args.checkpoint)
    with args.runtime.open('rb') as f:
        binary = hashlib.file_digest(f, 'sha256').hexdigest()
    if binary != spec['native_runtime_sha256']:
        raise ValueError('native runtime differs from the frozen build')
    artifact = args.store / spec['native_artifact']
    for name, key in [('manifest.json', 'native_manifest_sha256'), ('spec.json', 'native_spec_sha256')]:
        if sha((artifact / name).read_bytes()) != spec[key]:
            raise ValueError('native artifact metadata differs from the frozen import')
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    config = f'''schema_version = 2
[storage]
data_dir = {json.dumps(str(args.out.resolve() / 'state'))}
installed = {json.dumps(str(args.store.resolve()))}
[models.qwen27]
artifact = "{spec['native_artifact']}"
tokenizer = {json.dumps(str(args.checkpoint.resolve() / 'tokenizer.json'))}
chat_template = {json.dumps(str(args.checkpoint.resolve() / 'chat_template.jinja'))}
context = 12288
max_slots = 1
prefill_chunk = 512
kv_cache_encoding = "f16"
temperature = 0.0
top_p = 1.0
top_k = 0
[client]
bind = "127.0.0.1"
port = 18974
'''
    config_path = args.out / 'native.toml'
    config_path.write_text(config)
    receipt = {'runtime_sha256': binary,
        'artifact': spec['native_artifact'], 'config_sha256': sha(config.encode()),
        'protocol_sha256': protocol_sha, 'fresh_state': True, 'completed': False,
        'validation_sha256': sha(args.validation.read_bytes()),
        'service_helper_sha256': sha((args.repo / 'tools' / 'speed_gate.py').read_bytes())}
    (args.out / 'launch.json').write_bytes(encoded(receipt))
    sys.path.insert(0, str(args.repo / 'tools'))
    from speed_gate import Service
    previous = signals(stop)
    service = None
    try:
        service = Service(args.runtime.resolve(), config_path.resolve(), args.out, 18974, {})
        service.wait_ready(timeout=420)
        command = [sys.executable, str(Path(__file__).with_name('native.py')),
                   '--url', 'http://127.0.0.1:18974', '--protocol', str(args.protocol),
                   '--inputs', str(args.inputs), '--prereg-commit', args.prereg_commit,
                   '--phase', args.phase, '--repeat', str(args.repeat), '--out', str(args.out / 'collected')]
        if args.case is not None:
            command += ['--case', args.case]
        subprocess.run(command, check=True)
        receipt['completed'] = True
    except BaseException as error:
        receipt['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signals(signal.SIG_IGN)
        retirement = {'started': service is not None, 'retired': service is None}
        try:
            if service is not None:
                retirement.update(service.stop(timeout=5))
                retirement['retired'] = service.proc.poll() is not None
            with args.runtime.open('rb') as f:
                receipt['runtime_after_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
            if receipt['runtime_after_sha256'] != binary:
                raise ValueError('native executable changed during collection')
            if not retirement['retired'] or (service is not None and retirement.get('returncode') != 0):
                raise RuntimeError('native retirement was not successful')
        except BaseException as error:
            receipt['completed'] = False
            retirement['failure'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            try:
                (args.out / 'retirement.json').write_bytes(encoded(retirement))
                (args.out / 'launch.json').write_bytes(encoded(receipt))
            finally:
                restore_signals(previous)


if __name__ == '__main__':
    main()
