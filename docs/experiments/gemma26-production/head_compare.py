# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import array
import hashlib
import heapq
import json
import math
import pathlib
import sys

assert len(sys.argv) in (4,5)
native, stock, output = map(pathlib.Path, sys.argv[1:4])
profile = sys.argv[4] if len(sys.argv)==5 else 'gemma26'
assert profile in ('gemma26','gemma31')
names = ('prefix128', 'scalar132', 'fresh132') if profile=='gemma31' else ('prefix256', 'scalar260', 'fresh260')
owned = {}
for name in names:
    a, b = ((native / (str(i) + '-' + name + '.f32')).read_bytes() for i in range(2))
    assert len(a) == len(b) == 262144 * 4 and a == b, 'native owned repeat must freeze first'
    owned[name] = hashlib.sha256(a).hexdigest()

def read(path):
    row = array.array('f')
    row.frombytes(path.read_bytes())
    assert len(row) == 262144 and all(math.isfinite(x) for x in row)
    return row

def summary(row):
    top = heapq.nlargest(2, range(len(row)), key=row.__getitem__)
    maximum = row[top[0]]
    logz = maximum + math.log(math.fsum(math.exp(x - maximum) for x in row))
    return {'top2': top, 'top2_logits': [row[i] for i in top],
            'top2_logprobs': [row[i] - logz for i in top],
            'margin': row[top[0]] - row[top[1]]}

result = {'native_owned_repeat': owned, 'comparison_scope': 'diagnostic; no new quality allowance', 'heads': {}}
for name in names:
    a, b = read(native / ('0-' + name + '.f32')), read(stock / (name + '.f32'))
    differences = [float(x) - float(y) for x, y in zip(a, b)]
    result['heads'][name] = {
        'native': summary(a), 'stock': summary(b),
        'byte_exact': a.tobytes() == b.tobytes(),
        'max_abs': max(abs(x) for x in differences),
        'rms': math.sqrt(math.fsum(x*x for x in differences)/len(differences)),
    }
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
