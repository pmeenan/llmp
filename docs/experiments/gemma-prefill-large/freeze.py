#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze same-shape heads/state before cross-engine analysis; no quality gate."""
import array
import hashlib
import json
import math
from pathlib import Path
import sys

VOCAB = 262144
ID_SHA = '6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b'
ID_BYTES = 8227 * 4
PROFILES = {'26': (1024, 2048, 592445440), '31': (256, 1280, 1740636160)}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while data := source.read(8 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


def head(path):
    assert path.stat().st_size == VOCAB * 4
    values = array.array('f')
    with path.open('rb') as source:
        values.fromfile(source, VOCAB)
    if sys.byteorder != 'little':
        values.byteswap()
    assert all(math.isfinite(value) for value in values)
    return {'sha256': digest(path), 'argmax': max(range(VOCAB), key=values.__getitem__)}


def main():
    root, ids, source = map(Path, sys.argv[1:4])
    assert 4 <= len(sys.argv) <= 6
    policy = sys.argv[4] if len(sys.argv) >= 5 else 'ordinary'
    variant = sys.argv[5] if len(sys.argv) == 6 else '26'
    assert variant in ['26', '31']
    assert policy in (['ordinary', 'all'] if variant == '26' else ['ordinary', 'both'])
    rows, cells, state_bytes = PROFILES[variant]
    assert ids.stat().st_size == ID_BYTES and digest(ids) == ID_SHA
    arms = {}
    for name in [f'{policy}{rows}-first', f'{policy}{rows}-repeat']:
        path = root / name
        layout = (path / 'state-layout.txt').read_text()
        assert layout == f'gemma{variant}-f16-kv-scalar-device-v1:16384:{rows}:16384:{cells}\n'
        state = path / 'initialized-state.bin'
        assert state.stat().st_size == state_bytes
        arms[name] = {'heads': {file: head(path / file) for file in ['prefill.f32', 'final.f32']},
                      'state_sha256': digest(state), 'state_bytes': state.stat().st_size,
                      'layout': layout.strip()}
    assert arms[f'{policy}{rows}-first'] == arms[f'{policy}{rows}-repeat']
    receipt = {'kind': 'same-shape bounded two-head/state repeat, no inherited quality allowance',
               'input_sha256': ID_SHA, 'native_base': 'c2d2147', 'variant': int(variant),
               'policy': policy, 'max_rows': rows, 'context': 16384, 'completed_positions': 8227,
               'source_log_sha256': digest(source), 'arms': arms}
    output = root / f'{policy}{rows}-own-repeat-frozen.json'
    with output.open('x') as file:
        json.dump(receipt, file, indent=2)
        file.write('\n')
    print(output.name, digest(output), flush=True)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
