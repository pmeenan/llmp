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

from prepare import IMAGE, CORPUS, PARENTS, encoded, require, sha, parent

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
    parser.add_argument('--corpus', type=Path, required=True)
    args = parser.parse_args()
    family = Path(__file__).resolve().parent.parent
    sources = {'paris_freeze.py': family / 'paris/freeze.py',
               'raw_reference.py': family / 'reference.py'}
    freeze = parent(sources['paris_freeze.py'], 'paris_freeze.py')
    METADATA = freeze.METADATA
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
    require(args.corpus.stat().st_size == 817275 and sha(args.corpus) == CORPUS, 'corpus differs')
    shutil.copyfile(args.corpus, out / 'corpus.txt')
    require(sha(out / 'corpus.txt') == CORPUS, 'corpus copy differs')
    for filename, source in sources.items():
        require(sha(source) == PARENTS[filename], 'parent helper differs')
        shutil.copyfile(source, helpers / filename)
    helper = Path(__file__).with_name('prepare.py')
    shutil.copyfile(helper, helpers / helper.name)
    before = {p.name: sha(p) for p in (Path(__file__), helper)}
    snapshots = {p.name: sha(p) for p in helpers.iterdir()}
    require(sha(helpers / helper.name) == before[helper.name], 'helper snapshot changed')
    name = 'llmp-gemma26-short-prep-' + uuid.uuid4().hex
    create = DOCKER + ['create', '--name', name, '--runtime', 'runc', '--network', 'none',
        '--cpus', '2', '--memory', '3g', '--memory-swap', '3g', '--entrypoint', 'python3',
        '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'HF_HUB_OFFLINE=1',
        '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
        '-v', f'{metadata}:/metadata:ro', '-v', f'{helpers}:/helpers:ro',
        '-v', f'{out / "corpus.txt"}:/corpus.txt:ro',
        '-v', f'{out}:/output', IMAGE, '/helpers/prepare.py']
    receipt = dict(format='llmp-gemma26-short-prep-launch/1', image=IMAGE,
                   argv=create, helper_hashes_before=before, source_metadata=METADATA,
                   input_preparation_complete=False, model_constructed=False,
                   corpus_sha256=CORPUS, helper_snapshots_before=snapshots, parent_hashes=PARENTS)
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
        require(result['status'] == 'frozen-inputs-only-source-review-pending' and
                result['ids_sha256'] == sha(out / 'ids.json') and
                result['source_evidence_sha256'] == sha(out / 'source-evidence.json'), 'invalid preparation receipt')
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
        receipt['helper_snapshots_after'] = {p.name: sha(p) for p in helpers.iterdir()}
        receipt['copied_corpus_sha256_after'] = sha(out / 'corpus.txt')
        receipt['parent_hashes_after'] = {name: sha(path) for name, path in sources.items()}
        receipt['copied_metadata_sha256_after'] = {name: sha(metadata / name) for name in METADATA}
        (out / 'launch.json').write_bytes(encoded(receipt))
        for sig, handler in previous.items(): signal.signal(sig, handler)
        require(receipt['retirement']['retired'], 'container retirement unconfirmed')
        require(receipt['helper_hashes_after'] == before and
                receipt['helper_snapshot_sha256_after'] == before[helper.name], 'helper changed')
        require(receipt['copied_metadata_sha256_after'] == METADATA, 'copied metadata changed')
        require(receipt['helper_snapshots_after'] == snapshots and
                receipt['parent_hashes_after'] == PARENTS, 'parent/snapshot helper changed')
        require(receipt['copied_corpus_sha256_after'] == CORPUS, 'copied corpus changed')


if __name__ == '__main__':
    main()
