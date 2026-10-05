#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate unchanged reference identities or paid native full outputs."""
import json
import pathlib
import subprocess
import sys

import own_freeze
import prepare

if len(sys.argv) != 3 or sys.argv[1] not in ('reference', 'paid'):
    raise SystemExit('verify.py reference|paid EXPECTED_NATIVE_FREEZE_SHA')
root = prepare.ROOT
freeze = root / 'native-own-frozen.json'
if own_freeze.digest(freeze) != sys.argv[2]:
    raise ValueError('native own freeze changed')
own = json.loads(freeze.read_text())
if own_freeze.digest(root / 'pre-native.json') != own['pre_native_sha256']:
    raise ValueError('compiled source prefreeze changed')
pre = json.loads((root / 'pre-native.json').read_text())
if sys.argv[1] == 'paid':
    paid_input = prepare.record(root / 'candidate-paid' / 'inputs.i32')
    if paid_input['bytes'] != 4096 or paid_input['sha256'] != own_freeze.INPUT_SHA:
        raise ValueError('actual paid input differs from canonical frozen IDs')
    for name, expected in own['policies']['candidate'].items():
        path = root / 'candidate-paid' / name
        if path.stat().st_size != expected['bytes'] or own_freeze.digest(path) != expected['sha256']:
            raise ValueError('paid full head/state differs from own candidate: ' + name)
    print('PAID_NATIVE full128heads/all4initializedstates EXACT owncandidate')
else:
    identities = pre['source_files'] + pre['reference_headers'] + pre['artifact_metadata']
    identities += [pre[k] for k in ('native_binary', 'reference_binary', 'reference_client',
                                   'build_receipt', 'input')]
    for expected in identities:
        actual = prepare.record(pathlib.Path(expected['path']), 256 << 20)
        if actual != expected:
            raise ValueError('source/binary/input identity changed: ' + expected['path'])
    output = subprocess.check_output(['sudo', '-n', 'docker', 'run', '--rm', '--network', 'none',
                                     '--read-only', '--entrypoint', 'sha256sum', prepare.IMAGE,
                                     *['/app/' + name for name in prepare.LIBS]], text=True)
    actual = {pathlib.Path(line.split()[1]).name: line.split()[0] for line in output.splitlines()}
    if actual != pre['image_libraries'] or actual != prepare.LIBS:
        raise ValueError('original math libraries changed')
    raw = prepare.HOME / '.local/share/jitllm/reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf'
    if raw.stat().st_size != pre['raw_model_bytes']:
        raise ValueError('approved raw model size changed')
    print('REFERENCE_PRECHECK frozen source/binaries/headers/libraries/IDs/artifact metadata PASS')
