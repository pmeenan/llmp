#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run on a Spark. All inputs and outputs other than the harness are external."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import uuid
import subprocess

HERE = Path(__file__).resolve().parent
IMAGE = 'ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7'
REVISION = 'b29c606e28a01b1bc8c1351026a0fa6e616bf6c4'
MODEL_SHA = 'f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f'
TRACE_SHA = '7625929fd143bafa31f5d3ebfe03e4dcf96452129b0d080a67434940c85e2383'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def container(docker, base, tail, **kwargs):
    name = 'llmp-routes-' + uuid.uuid4().hex[:12]
    try:
        subprocess.run(base + ['--name', name] + tail, check=True, **kwargs)
    finally:
        # Also attempted on SIGTERM/Ctrl-C; --rm handles ordinary completion.
        result = subprocess.run(docker + ['rm', '-f', name], capture_output=True, text=True)
        if result.returncode and 'No such container' not in result.stderr:
            raise RuntimeError('Container cleanup failed: ' + result.stderr)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('trace', type=Path)
    p.add_argument('source', type=Path, help='Exact upstream source checkout/archive, verified against pins.json')
    p.add_argument('output', type=Path, help='New external directory')
    a = p.parse_args()
    os.umask(0o077)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    for name in ('model', 'trace', 'source', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    if sha(a.model) != MODEL_SHA or sha(a.trace) != TRACE_SHA:
        raise ValueError('Model or input trace identity mismatch')
    pins = json.loads((HERE / 'pins.json').read_text())
    for name, digest in pins['headers'].items():
        if sha(a.source / name) != digest:
            raise ValueError('Header identity mismatch: ' + name)
    a.output.mkdir(mode=0o700)
    trace = json.loads(a.trace.read_text())
    prompt, decode = trace['A_continuation'], trace['A_expected_continuation'][:-1]
    tokens = [*prompt, *decode]
    (a.output / 'tokens.txt').write_text(f'{len(prompt)} {len(decode)}\n' + '\n'.join(map(str, tokens)) + '\n')
    docker = shlex.split(os.environ.get('DOCKER', 'docker'))
    base = docker + ['run', '--rm', '--network', 'none', '--read-only',
                     '--user', f'{os.getuid()}:{os.getgid()}', '--tmpfs', '/tmp:rw,size=1g',
                     '--mount', f'type=bind,src={a.source},dst=/source,readonly',
                     '--mount', f'type=bind,src={HERE},dst=/harness,readonly',
                     '--mount', f'type=bind,src={a.output},dst=/output',
                     '--mount', f'type=bind,src={a.model},dst=/model.gguf,readonly']
    command = ['g++', '-std=c++23', '-O2', '-march=armv8-a', '-Wall', '-Wextra', '-Werror',
               '-I/source/include', '-I/source/ggml/include', '/harness/capture.cc',
               '-L/app', '-Wl,-rpath,/app', '-lllama', '-lggml', '-lggml-base', '-o', '/output/capture']
    container(docker, base, ['--entrypoint', command[0], IMAGE, *command[1:]])
    metadata = dict(image=IMAGE, engine_revision=REVISION, model_sha256=MODEL_SHA,
                    trace_sha256=TRACE_SHA, prompt_tokens=len(prompt), decode_tokens=len(decode),
                    source_sha256=sha(HERE / 'capture.cc'), executable_sha256=sha(a.output / 'capture'),
                    host=os.uname().nodename, kernel=os.uname().release, build_command=command,
                    tokens_sha256=sha(a.output / 'tokens.txt'), runs=[])
    metadata['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                              '--format=csv,noheader'], text=True).strip()
    for batch, tracing in [(512, False), (512, True), (64, False), (64, True)]:
        name = f'b{batch}-' + ('routes' if tracing else 'control')
        routes = '/output/' + name + '.jsonl' if tracing else '-'
        cmd = ['--device', 'nvidia.com/gpu=all', '--env', 'CUDA_DISABLE_PTX_JIT=1',
                      '--entrypoint', '/output/capture', IMAGE, '/model.gguf', '/output/tokens.txt',
                      str(batch), routes, '/output/' + name + '.predictions', '32768']
        with (a.output / (name + '.log')).open('w') as log:
            container(docker, base, cmd, stdout=log, stderr=subprocess.STDOUT)
        log = (a.output / (name + '.log')).read_text()
        if 'offloaded 31/31 layers to GPU' not in log:
            raise ValueError('GPU offload not verified')
        record = dict(name=name, batch=batch, tracing=tracing,
                      predictions_sha256=sha(a.output / (name + '.predictions')))
        if tracing:
            record['routes_sha256'] = sha(a.output / (name + '.jsonl'))
            if record['predictions_sha256'] != metadata['runs'][-1]['predictions_sha256']:
                raise ValueError('Callback changed predictions')
        metadata['runs'].append(record)
        (a.output / 'capture.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(name + ' passed', flush=True)


if __name__ == '__main__':
    main()
