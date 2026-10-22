#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Inputs-only container: immutable image, metadata-only mounts, no GPU or network."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from contract import IMAGE, MANIFEST, METADATA, encoded, metadata, sha
from lifecycle import container_name, run_container


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'corpus', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    metadata(args.checkpoint)
    if sha((args.checkpoint / 'llmp-archive.json').read_bytes()) != MANIFEST:
        raise ValueError('staged archive manifest differs')
    inspect = subprocess.check_output(['sudo', '-n', 'docker', 'image', 'inspect', IMAGE])
    details = json.loads(inspect)
    if len(details) != 1 or details[0]['Id'] != IMAGE or details[0]['Architecture'] != 'arm64':
        raise ValueError('reference image identity or architecture differs')
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'image-inspect.json').write_bytes(inspect)
    root = Path(__file__).resolve().parent
    helpers = {p.name: sha(p.read_bytes()) for p in sorted(root.glob('*.py'))}
    helpers['tensorfold-source-hashes.txt'] = sha((root / 'tensorfold-source-hashes.txt').read_bytes())
    name = container_name('inputs')
    command = ['sudo', '-n', 'docker', 'run', '--rm', '--name', name, '--runtime=runc',
               '--network=none', '--user', f'{os.getuid()}:{os.getgid()}',
               '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'CUDA_VISIBLE_DEVICES=',
               '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'HOME=/tmp',
               '--mount', f'type=bind,src={root},dst=/helpers,readonly',
               '--mount', f'type=bind,src={args.corpus.resolve()},dst=/corpus.txt,readonly',
               '--mount', f'type=bind,src={args.out.resolve()},dst=/out']
    for filename in METADATA:
        command += ['--mount', f'type=bind,src={args.checkpoint.resolve()/filename},dst=/checkpoint/{filename},readonly']
    command += ['--entrypoint', 'python3', IMAGE, '/helpers/prepare_inputs.py',
                '--checkpoint', '/checkpoint', '--corpus', '/corpus.txt', '--out', '/out/frozen']
    receipt = {'image': IMAGE, 'archive_manifest_sha256': MANIFEST, 'helpers': helpers,
               'command': command, 'weight_mounts': 0, 'gpu_devices_requested': 0, 'completed': False}
    run_container(command, args.out, receipt, name)


if __name__ == '__main__':
    main()
