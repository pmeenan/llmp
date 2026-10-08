#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture the exact image closure or check three pre-context usage refusals."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


root = pathlib.Path('/scratch')
mode = sys.argv[1]
if mode == 'inspect':
    # No model paths are mounted. These commands terminate without creating contexts.
    versions = {}
    for name in ['llama-cli', 'llama-server']:
        run = subprocess.run(['/app/' + name, '--version'], capture_output=True, text=True,
                             timeout=30, check=True)
        text = run.stdout + run.stderr
        if '11429' not in text or 'd812350' not in text:
            raise ValueError('executable does not report frozen release source: ' + name)
        versions[name] = text
    out = root / 'image-closure1'
    out.mkdir(mode=0o700)
    lib = out / 'lib'
    lib.mkdir(mode=0o700)
    files = {}
    for p in sorted(pathlib.Path('/app').iterdir()):
        if p.is_file() and ('.so' in p.name or p.name in versions):
            files[p.name] = {'bytes': p.stat().st_size, 'sha256': digest(p),
                             'symlink_target': str(p.resolve()) if p.is_symlink() else None}
            if '.so' in p.name:
                shutil.copyfile(p, lib / p.name)
    for name in ['libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cuda.so']:
        if name not in files:
            raise ValueError('required reference library absent: ' + name)
    dependencies = {}
    for name in [*versions, 'libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cuda.so']:
        run = subprocess.run(['ldd', '/app/' + name], capture_output=True, text=True,
                             timeout=30, check=True)
        if 'not found' in run.stdout:
            raise ValueError('unresolved image dependency: ' + name)
        dependencies[name] = run.stdout
    # Retain actual CUDA and C++ runtime files loaded by these executables/libraries.
    external = {}
    resolved_hashes = {}
    for lines in dependencies.values():
        for line in lines.splitlines():
            fields = line.split()
            candidate = fields[2] if len(fields) >= 3 and fields[1] == '=>' else fields[0] if fields else ''
            if candidate.startswith('/') and pathlib.Path(candidate).is_file():
                p = pathlib.Path(candidate)
                if not candidate.startswith('/app/') and candidate not in external:
                    resolved = str(p.resolve())
                    if resolved not in resolved_hashes:
                        resolved_hashes[resolved] = digest(p)
                    external[candidate] = {'resolved': resolved, 'bytes': p.stat().st_size,
                                           'sha256': resolved_hashes[resolved]}
    save(out / 'receipt.json', {'versions': versions, 'app_files': files,
                               'ldd': dependencies, 'runtime_dependencies': external,
                               'model_contexts_created': False})
elif mode == 'usage':
    result = {}
    for name, marker in [('joined', 'MODEL'), ('quality', 'MODEL'), ('assistant', 'TARGET ASSISTANT')]:
        run = subprocess.run(['/scratch/compiled1/' + name], capture_output=True, text=True, timeout=30)
        text = run.stdout + run.stderr
        if run.returncode == 0 or marker not in text:
            raise ValueError('expected pre-context argument refusal absent: ' + name)
        result[name] = {'returncode': run.returncode, 'message': text,
                        'sha256': digest(root / 'compiled1' / name)}
    save(root / 'compiled1' / 'usage.json', {'controls': result, 'model_contexts_created': False})
else:
    raise ValueError('inspect|usage only')
