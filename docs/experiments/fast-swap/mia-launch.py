#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Launch the pinned fast-start Mia reference; run under spark-job --gpu."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, why):
    if not value:
        raise RuntimeError(why)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--pylib', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(all(re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path)) for path in (args.recipe, args.pylib)),
            'absolute paths without shell metacharacters required')
    pin = json.loads(Path(__file__).with_name('mia-launch.json').read_text())
    require(subprocess.check_output(['git', '-C', str(args.recipe), 'rev-parse', 'HEAD'], text=True).strip()
            == pin['recipe_revision'], 'pinned recipe revision required')
    changed = subprocess.check_output(['git', '-C', str(args.recipe), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
    require(changed == ['files/patch_ple_offload.py'], 'only the pinned PLE loader override may change the recipe')
    require(sha(args.recipe / 'files/patch_ple_offload.py') == pin['ple_generator_sha256'],
            'mandatory PLE safetensors override differs')
    require(not subprocess.check_output(['git', '-C', str(args.recipe), 'ls-files', '--others',
                                         '--exclude-standard'], text=True).strip(),
            'untracked recipe sources refused; ignored generated caches are recorded by each experiment')
    require(not args.pylib.is_symlink(), 'loader directory must not be a symlink')
    inventory = list(args.pylib.rglob('*'))
    require(not any(p.is_symlink() for p in inventory), 'loader symlinks refused')
    allowed_dirs = {str(parent) for name in pin['loader_files'] for parent in Path(name).parents if str(parent) != '.'}
    require({str(p.relative_to(args.pylib)) for p in inventory if p.is_file()} == set(pin['loader_files'])
            and {str(p.relative_to(args.pylib)) for p in inventory if p.is_dir()} == allowed_dirs,
            'loader payload must match the complete pinned inventory')
    for name, expected in pin['loader_files'].items():
        require(sha(args.pylib / name) == expected, 'pinned instanttensor payload differs: ' + name)
    env = os.environ.copy()
    require(env.get('IMAGE', pin['image']) == pin['image'], 'pinned engine image required')
    env['IMAGE'] = pin['image']
    extra = env.get('EXTRA_DOCKER_ARGS', '')
    require('PYTHONPATH' not in extra and 'TRITON_CACHE_DIR' not in extra and 'VLLM_USE_V2_MODEL_RUNNER' not in extra,
            'launcher owns loader path, persistent Triton cache and V2 runner')
    env['EXTRA_DOCKER_ARGS'] = (extra + ' --mount type=bind,src=' + str(args.pylib)
        + ',dst=/opt/llmp-mia-pylib,readonly -e PYTHONPATH=/opt/llmp-mia-pylib'
        + ' -e TRITON_CACHE_DIR=/root/.cache/vllm/triton-issue -e VLLM_USE_V2_MODEL_RUNNER=1')
    require('--load-format' not in env.get('EXTRA_VLLM_ARGS', ''), 'launcher owns the load format')
    env['EXTRA_VLLM_ARGS'] = env.get('EXTRA_VLLM_ARGS', '') + ' --load-format instanttensor'
    argv = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    os.chdir(args.recipe)
    os.execvpe('./start.sh', ['./start.sh', *argv], env)


if __name__ == '__main__':
    main()
