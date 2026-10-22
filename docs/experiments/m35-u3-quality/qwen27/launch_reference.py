#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned source container lifecycle; call only inside an authorized exclusive job."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from contract import IMAGE, MANIFEST, frozen, metadata, select_cases, sha
from lifecycle import container_name, run_container
from reference import require_validation, validation_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'inputs', 'protocol', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--validation', type=Path)
    p.add_argument('--prereg-commit', required=True)
    p.add_argument('--phase', choices=['validate', 'short', 'depth', 'behavior'], required=True)
    p.add_argument('--repeat', type=int, choices=[1, 2], default=1)
    p.add_argument('--case')
    a = p.parse_args()
    _, _, behavior, protocol_sha = frozen(a.protocol, a.inputs, a.prereg_commit)
    select_cases(a.phase, a.case, behavior)
    if a.phase != 'validate':
        require_validation(a.validation, validation_identity(protocol_sha, a.prereg_commit))
    metadata(a.checkpoint)
    if sha((a.checkpoint / 'llmp-archive.json').read_bytes()) != MANIFEST:
        raise ValueError('archive manifest differs')
    inspection = subprocess.check_output(['sudo', '-n', 'docker', 'image', 'inspect', IMAGE])
    info = json.loads(inspection)
    if len(info) != 1 or info[0]['Id'] != IMAGE or info[0]['Architecture'] != 'arm64':
        raise ValueError('immutable source image differs')
    os.umask(0o077)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'image-inspect.json').write_bytes(inspection)
    root = Path(__file__).resolve().parent
    name = container_name('reference')
    command = ['sudo', '-n', 'docker', 'run', '--rm', '--name', name,
               '--device=nvidia.com/gpu=all', '--network=none', '--ipc=host', '--ulimit=memlock=-1',
               '--user', f'{os.getuid()}:{os.getgid()}', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--env', 'HOME=/out', '--env', 'TRITON_CACHE_DIR=/out/triton',
               '--env', 'TORCH_EXTENSIONS_DIR=/out/torch-extensions',
               '--mount', f'type=bind,src={root},dst=/helpers,readonly',
               '--mount', f'type=bind,src={a.checkpoint.resolve()},dst=/checkpoint,readonly',
               '--mount', f'type=bind,src={a.inputs.resolve()},dst=/inputs,readonly',
               '--mount', f'type=bind,src={a.protocol.resolve()},dst=/protocol.json,readonly',
               '--mount', f'type=bind,src={a.out.resolve()},dst=/out']
    if a.validation is not None:
        command += ['--mount', f'type=bind,src={a.validation.resolve()},dst=/validation.json,readonly']
    command += ['--entrypoint', 'python3', IMAGE, '/helpers/reference.py', '--checkpoint', '/checkpoint',
                '--inputs', '/inputs', '--protocol', '/protocol.json', '--prereg-commit', a.prereg_commit,
                '--phase', a.phase, '--repeat', str(a.repeat), '--out', '/out/collected']
    if a.validation is not None:
        command += ['--validation', '/validation.json']
    if a.case is not None:
        command += ['--case', a.case]
    receipt = {'command': command, 'image': IMAGE, 'completed': False}
    run_container(command, a.out, receipt, name)


if __name__ == '__main__':
    main()
