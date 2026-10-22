#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One bounded source diagnostic, inside an installed spark-job --gpu job."""
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

from reference import IMAGE, SERIALIZATION_ENV, checkpoint_check, encoded, require, sha

DOCKER = ['sudo', '-n', 'docker']


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def retire(name):
    result = {'name': name, 'retired': False}
    try:
        removed = subprocess.run(DOCKER + ['rm', '-f', name], capture_output=True,
                                 text=True, timeout=4, check=False)
        result.update(remove_returncode=removed.returncode, remove_stderr=removed.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result['remove_error'] = str(error)
    # A failed inspect is not proof of absence: require a successful exact-name list.
    try:
        check = subprocess.run(DOCKER + ['ps', '-a', '--filter', f'name=^/{name}$',
                                       '--format', '{{.ID}}'], capture_output=True,
                               text=True, timeout=1, check=False)
        result.update(check_returncode=check.returncode, remaining_ids=check.stdout.strip())
        result['retired'] = check.returncode == 0 and not check.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result['check_error'] = str(error)
    return result


def execute(create, name, out, receipt):
    previous = {sig: signal.signal(sig, stop)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        try:
            subprocess.run(create, check=True, capture_output=True, timeout=30)
            inspected = subprocess.run(DOCKER + ['inspect', name], check=True,
                                       capture_output=True, timeout=5)
            (out / 'container.json').write_bytes(inspected.stdout)
            info = json.loads(inspected.stdout)[0]
            require(info['Image'] == IMAGE, 'Docker created the wrong immutable image')
            with (out / 'container.log').open('wb') as log:
                # Foreground attachment is bounded independently of the outer 600s
                # supervisor; the finally block owns detached-container retirement.
                subprocess.run(DOCKER + ['start', '--attach', name], stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=510)
            record = json.loads((out / 'result.json').read_text())
            require(record['status'] == 'collected-not-quality-qualified',
                    'container exited without a validated result')
            receipt['collection_completed'] = True
        except BaseException as error:
            receipt['failure'] = f'{type(error).__name__}: {error}'
            if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                # Creation/inspection happen before container.log exists. Preserve
                # their captured daemon diagnostics, not just an exit-status string.
                receipt['failed_command'] = error.cmd
                receipt['failed_returncode'] = getattr(error, 'returncode', None)
                for field in ('stdout', 'stderr'):
                    value = getattr(error, field, None)
                    if isinstance(value, bytes):
                        value = value.decode(errors='replace')
                    receipt['failed_' + field] = value
            raise
        finally:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            receipt['retirement'] = retire(name)
            # Retirement happens even if the log cannot be opened or receipts cannot
            # be written. Never infer successful cleanup from docker run's exit.
            (out / 'launch.json').write_bytes(encoded(receipt))
            require(receipt['retirement']['retired'], 'container retirement unconfirmed')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--staging-evidence', type=Path, required=True,
                        help='official stage-model successful job receipt/log, retained externally')
    parser.add_argument('--staging-sha256', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.out.resolve()
    checkpoint = args.checkpoint.resolve()
    require(sha(args.staging_evidence) == args.staging_sha256, 'staging evidence changed')
    source = checkpoint_check(checkpoint)
    shutil.copyfile(args.staging_evidence, out / 'staging-evidence')
    shutil.copyfile(checkpoint / 'llmp-archive.json', out / 'source-archive.json')
    pin_path = args.repo / 'docs/experiments/fast-swap/mia-launch.json'
    require(sha(pin_path) == '61732d66793a85ca140cb090a2c8aecad23da141eb4980f3cb8767953f3127a1',
            'loader manifest differs from the registered pin')
    pins = json.loads(pin_path.read_text())
    require(pins['image'] == IMAGE and pins['loader_version'] == '0.2.0', 'loader pin changed')
    pylib = Path.home() / '.local/share/mia-load-study/pylib'
    # Copy only authenticated loader entries; do not expose unpinned extra modules
    # from the broad installed PYTHONPATH to this reference process.
    loader = out / 'loader'
    for relative, digest in pins['loader_files'].items():
        require(sha(pylib / relative) == digest, f'loader payload changed: {relative}')
        dest = loader / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pylib / relative, dest)
        require(sha(dest) == digest, f'loader copy changed: {relative}')
    helper = Path(__file__).with_name('reference.py')
    helpers = out / 'helpers'
    helpers.mkdir()
    shutil.copyfile(helper, helpers / helper.name)
    before = {str(p): sha(p) for p in (Path(__file__), helper)}
    require(sha(helpers / helper.name) == before[str(helper)], 'helper snapshot changed')
    cache = Path.home() / '.cache/vllm/triton-u3-gemma26'
    cache.mkdir(parents=True, exist_ok=True)
    name = 'llmp-u3-gemma26-' + uuid.uuid4().hex
    create = DOCKER + [
        'create', '--name', name, '--device', 'nvidia.com/gpu=all',
        '--network', 'none', '--ipc', 'host', '--entrypoint', 'python3',
        '-v', f'{checkpoint}:/model:ro', '-v', f'{helpers}:/helpers:ro',
        '-v', f'{loader}:/opt/instanttensor:ro', '-v', f'{cache}:/triton',
        '-v', f'{out}:/output', '-e', 'PYTHONPATH=/opt/instanttensor',
        '-e', 'TRITON_CACHE_DIR=/triton', '-e', 'VLLM_USE_V2_MODEL_RUNNER=1',
        '-e', f'{SERIALIZATION_ENV}=1',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
        IMAGE, '/helpers/reference.py', 'run', '--out', '/output',
    ]
    receipt = dict(format='llmp-gemma26-reference-launch/1', image=IMAGE,
                   argv=create, checkpoint=str(checkpoint), source=source,
                   staging_evidence_sha256=args.staging_sha256,
                   staging_evidence_semantics='Official stage-model job success must be reviewed before launch',
                   loader_pin_sha256=sha(pin_path), loader_files=pins['loader_files'],
                   helper_hashes_before=before, collection_completed=False)
    (out / 'launch.json').write_bytes(encoded(receipt))
    try:
        execute(create, name, out, receipt)
    finally:
        after = {str(p): sha(p) for p in (Path(__file__), helper)}
        receipt['helper_hashes_after'] = after
        receipt['helper_snapshot_sha256_after'] = sha(helpers / helper.name)
        (out / 'launch.json').write_bytes(encoded(receipt))
        require(after == before and sha(helpers / helper.name) == before[str(helper)],
                'diagnostic helpers changed during launch')


if __name__ == '__main__':
    main()
