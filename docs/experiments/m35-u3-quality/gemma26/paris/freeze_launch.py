#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run once inside installed spark-job --shared --timeout 180."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import uuid

from freeze import IMAGE, METADATA, encoded, require, sha

DOCKER = ['sudo', '-n', 'docker']


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def retire(name):
    result = dict(name=name, retired=False)
    try:
        removed = subprocess.run(DOCKER + ['rm', '-f', name], capture_output=True,
                                 text=True, timeout=4, check=False)
        result.update(remove_returncode=removed.returncode, remove_stderr=removed.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result['remove_error'] = str(error)
    try:
        check = subprocess.run(DOCKER + ['ps', '-a', '--filter', f'name=^/{name}$',
                                       '--format', '{{.ID}}'], capture_output=True,
                               text=True, timeout=1, check=False)
        result.update(check_returncode=check.returncode, remaining_ids=check.stdout.strip())
        result['retired'] = check.returncode == 0 and not check.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result['check_error'] = str(error)
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.out.resolve()
    metadata, helpers = out / 'metadata', out / 'helpers'
    metadata.mkdir(); helpers.mkdir()
    # Copy only six authenticated small metadata files: no checkpoint directory
    # or weight path is exposed to the container, and no shards are opened.
    for name, digest in METADATA.items():
        require(sha(args.checkpoint / name) == digest, 'source metadata changed: ' + name)
        shutil.copyfile(args.checkpoint / name, metadata / name)
        require(sha(metadata / name) == digest, 'metadata copy changed: ' + name)
    helper = Path(__file__).with_name('freeze.py')
    shutil.copyfile(helper, helpers / helper.name)
    before = {p.name: sha(p) for p in (Path(__file__), helper)}
    require(sha(helpers / helper.name) == before[helper.name], 'helper snapshot changed')
    name = 'llmp-gemma26-paris-freeze-' + uuid.uuid4().hex
    create = DOCKER + ['create', '--name', name, '--runtime', 'runc', '--network', 'none',
        '--cpus', '2', '--memory', '3g', '--memory-swap', '3g', '--entrypoint', 'python3',
        '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'HF_HUB_OFFLINE=1',
        '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
        '-v', f'{metadata}:/metadata:ro', '-v', f'{helpers}:/helpers:ro',
        '-v', f'{out}:/output', IMAGE, '/helpers/freeze.py']
    receipt = dict(format='llmp-gemma26-paris-freeze-launch/1', image=IMAGE,
                   argv=create, helper_hashes_before=before, source_metadata=METADATA,
                   input_preparation_complete=False, model_constructed=False)
    (out / 'launch.json').write_bytes(encoded(receipt))
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        subprocess.run(create, check=True, capture_output=True, timeout=20)
        inspected = subprocess.run(DOCKER + ['inspect', name], check=True, capture_output=True, timeout=5)
        (out / 'container.json').write_bytes(inspected.stdout)
        info = json.loads(inspected.stdout)[0]
        require(info['Image'] == IMAGE and info['HostConfig']['Runtime'] == 'runc' and
                info['HostConfig']['NetworkMode'] == 'none' and not info['HostConfig'].get('Devices') and
                not info['HostConfig'].get('DeviceRequests'), 'unexpected container exposure')
        with (out / 'container.log').open('wb') as log:
            subprocess.run(DOCKER + ['start', '--attach', name], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=120)
        result = json.loads((out / 'result.json').read_text())
        require(result['status'] == 'frozen-inputs-only' and
                result['input_sha256'] == sha(out / 'input.json'), 'invalid preparation receipt')
        receipt['input_preparation_complete'] = True
    except BaseException as error:
        receipt['failure'] = f'{type(error).__name__}: {error}'
        for field in ('stdout', 'stderr'):
            value = getattr(error, field, None)
            if isinstance(value, bytes): value = value.decode(errors='replace')
            if value is not None: receipt['failed_' + field] = value
        raise
    finally:
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        receipt['retirement'] = retire(name)
        receipt['helper_hashes_after'] = {p.name: sha(p) for p in (Path(__file__), helper)}
        receipt['helper_snapshot_sha256_after'] = sha(helpers / helper.name)
        receipt['copied_metadata_sha256_after'] = {name: sha(metadata / name) for name in METADATA}
        (out / 'launch.json').write_bytes(encoded(receipt))
        for sig, handler in previous.items(): signal.signal(sig, handler)
        require(receipt['retirement']['retired'], 'container retirement unconfirmed')
        require(receipt['helper_hashes_after'] == before and
                receipt['helper_snapshot_sha256_after'] == before[helper.name], 'helper changed')
        require(receipt['copied_metadata_sha256_after'] == METADATA, 'copied metadata changed')


if __name__ == '__main__':
    main()
