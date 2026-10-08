#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze finite exact Gemma2 native own results before reference reads."""
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1])
vocab, rows = 256000, 33
ids_raw = (root / 'input/ids.i32').read_bytes()
assert len(ids_raw) == 291 * 4
ids = struct.unpack('<291i', ids_raw)
assert ids[0] == 2 and all(0 <= value < vocab for value in ids)
result = {'scope': 'bounded own controls only',
          'input_sha256': hashlib.sha256(ids_raw).hexdigest(),
          'heads': {}, 'chosen': {}, 'final': {}}
arms = ('native-eager', 'native-graph1', 'native-graph2', 'native-lifetime',
        'native-greedy-own', 'native-primitive1', 'native-primitive2')
for arm in arms:
    for key, filename, size in (('chosen', 'chosen.i32', 32 * 4),
                                ('final', 'final.f32', vocab * 4)):
        data = (root / arm / filename).read_bytes()
        assert len(data) == size, 'incomplete ' + filename
        if key == 'chosen':
            assert all(0 <= value < vocab for value in struct.unpack('<32i', data))
        else:
            values = array.array('f')
            values.frombytes(data)
            assert all(math.isfinite(value) for value in values), 'nonfinite final row'
        result[key][arm] = hashlib.sha256(data).hexdigest()
    if arm == 'native-lifetime':
        continue
    data = (root / arm / 'heads.f32').read_bytes()
    assert len(data) == rows * vocab * 4, 'incomplete teacher rows'
    values = array.array('f')
    values.frombytes(data)
    assert all(math.isfinite(value) for value in values), 'nonfinite teacher rows'
    result['heads'][arm] = hashlib.sha256(data).hexdigest()
for key in ('heads', 'chosen', 'final'):
    optimized = {arm: digest for arm, digest in result[key].items()
                 if not arm.startswith('native-primitive')}
    assert len(set(optimized.values())) == 1, 'optimized own repeat/state/greedy changed ' + key
    assert result[key]['native-primitive1'] == result[key]['native-primitive2'], \
        'primitive own repeat changed ' + key
with (root / 'own-control.json').open('x') as output:
    json.dump(result, output, indent=2)
    output.write('\n')
print(json.dumps(result), flush=True)
