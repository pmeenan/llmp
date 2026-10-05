#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-acquisition owner/step aggregates; complete row records stay external."""
import hashlib
import json
import pathlib
import runpy
import sys

import own_freeze

if len(sys.argv) != 3:
    raise SystemExit('breakdown.py ROOT EXPECTED_NATIVE_FREEZE_SHA')
root = pathlib.Path(sys.argv[1])
frozen_path = root / 'native-own-frozen.json'
if own_freeze.digest(frozen_path) != sys.argv[2]:
    raise ValueError('own freeze changed')
frozen = json.loads(frozen_path.read_text())
comparison_path = root / 'comparison-candidate.json'
comparison = json.loads(comparison_path.read_text())
if comparison['native_own_freeze_sha256'] != sys.argv[2]:
    raise ValueError('comparison uses a different native freeze')
original = runpy.run_path(str(pathlib.Path(__file__).resolve().parent.parent /
                             'gemma26-packed-attention-c4/compare.py'))
ids = original['checked_input'](root / 'ids.i32', own_freeze.INPUT_SHA, 4096)
if hashlib.sha256(ids).hexdigest() != comparison['input_sha256']:
    raise ValueError('comparison input differs')
paths = [root / 'candidate-first/heads.f32', root / 'reference-a/heads.f32',
         root / 'candidate-repeat/heads.f32', root / 'reference-b/heads.f32']
expected = [frozen['policies']['candidate']['heads.f32']['sha256'], comparison['reference_sha256']]
expected += expected
for path, digest in zip(paths, expected):
    if path.stat().st_size != own_freeze.HEAD_BYTES or own_freeze.digest(path) != digest:
        raise ValueError('complete acquired head identity differs')
row_bytes = 262144 * 4
records = []
with paths[0].open('rb') as native, paths[1].open('rb') as reference:
    for step in range(32):
        for owner in range(4):
            a, b = native.read(row_bytes), reference.read(row_bytes)
            r = original['row'](a, b, 0, False)
            if r['strict']:
                av, bv = original['decode'](a), original['decode'](b)
                r['native_id'] = max(range(262144), key=av.__getitem__)
                r['reference_id'] = max(range(262144), key=bv.__getitem__)
            records.append({'step': step, 'owner': owner, 'position': 67 + owner + step, **r})
    if native.read(1) or reference.read(1):
        raise ValueError('extra head data')
if sum(r['byte_exact'] for r in records) != comparison['byte_exact_rows'] or \
   sum(r['strict'] for r in records) != comparison['strict_argmax_differences']:
    raise ValueError('full row breakdown disagrees with official comparison')
result = {'native_own_freeze_sha256': sys.argv[2],
          'comparison_sha256': own_freeze.digest(comparison_path),
          'analyzer_sha256': own_freeze.digest(pathlib.Path(__file__)),
          'complete_rows': 128, 'owners': [], 'steps': [], 'strict_misses': []}
for owner in range(4):
    rows = [r for r in records if r['owner'] == owner]
    first = next((r for r in rows if not r['byte_exact']), None)
    result['owners'].append({'owner': owner, 'rows': 32,
                             'byte_exact_rows': sum(r['byte_exact'] for r in rows),
                             'strict_differences': sum(r['strict'] for r in rows),
                             'first_nonexact_step': None if first is None else first['step'],
                             'first_nonexact_position': None if first is None else first['position']})
for step in range(32):
    rows = [r for r in records if r['step'] == step]
    result['steps'].append({'step': step, 'rows': 4,
                           'byte_exact_rows': sum(r['byte_exact'] for r in rows),
                           'strict_differences': sum(r['strict'] for r in rows)})
for r in records:
    if r['strict']:
        result['strict_misses'].append({k: r[k] for k in
                                       ('step', 'owner', 'position', 'native_id', 'reference_id',
                                        'reference_winner_over_native_choice',
                                        'positive_reference_winner_margin', 'max_raw')})
for name, data in [('candidate-breakdown.json', result), ('candidate-row-records.json', records)]:
    with (root / name).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
print(json.dumps(result))
