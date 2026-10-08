#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Small installed-supervisor success proof; app markers remain separate."""
import json
import pathlib
import re
import runpy

record = runpy.run_path(str(pathlib.Path(__file__).with_name('source_snapshot.py')))['record']


def success(name, expected_steps=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('outside owned supervisor job name')
    root = pathlib.Path.home() / '.local/share/llmp/jobs' / name
    final = json.loads((root / 'final.json').read_text())
    steps = json.loads((root / 'steps.json').read_text())['steps']
    if final['state'] != 'done' or final['rc'] != 0 or not steps or \
            any(s['state'] != 'done' or s['rc'] != 0 or s.get('signal') for s in steps) or \
            expected_steps is not None and len(steps) != expected_steps:
        raise ValueError('official supervisor acquisition not successfully retired')
    return {'name': name, 'final': record(root / 'final.json', 1 << 20),
            'steps': record(root / 'steps.json', 1 << 20), 'log': record(root / 'log', 64 << 20)}, steps
