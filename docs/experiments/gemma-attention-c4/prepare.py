#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate native-origin capture/control fidelity; construct four-stream roots."""
import array
import hashlib
import json
import math
import pathlib
import struct
import sys

base = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
freeze = json.loads((base / 'source-freeze-capture1.json').read_text())
def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
for path, expected in freeze['sources'].items():
    # Replay-only target additions are outside the acquired capture executable.
    if path != 'benchmarks/CMakeLists.txt':
        assert identity((root / path).read_bytes()) == expected, path
assert identity((root / 'build/spark-native/benchmarks/llmp_gemma_attention_capture').read_bytes()) == freeze['binary']
assert freeze['policy'] == 'dense31 natural C4 context256 max_rows128 normBOTH row_invariantOFF otherflagsOFF'
heads = (base / 'control-first/heads.f32').read_bytes()
assert len(heads) == 4 * 262144 * 4
values = array.array('f'); values.frombytes(heads)
assert all(math.isfinite(v) for v in values)
for mode in ['control-repeat', 'capture-first', 'capture-repeat']:
    assert (base / mode / 'heads.f32').read_bytes() == heads, mode
first = json.loads((base / 'capture-first/operands.json').read_text())
assert first == json.loads((base / 'capture-repeat/operands.json').read_text())
assert first['origin'] == 'native dense31 first-local natural C4' and len(first['owners']) == 4
packed = base / 'packed'; packed.mkdir(mode=0o700)
roles = {'q': (0, 32768, [256, 1, 32, 1], [4, 32768, 1024, 32768]),
         'k': (1, 2097152, [256, 256, 16, 1], [2, 8192, 512, 2097152]),
         'v': (1, 2097152, [256, 256, 16, 1], [2, 8192, 512, 2097152]),
         'mask': (1, 16384, [256, 32, 1, 1], [2, 512, 16384, 16384])}
ids = {}
for role, (kind, size, ne, nb) in roles.items():
    chunks = []
    for i, owner in enumerate(first['owners']):
        assert owner['owner'] == i and owner['position'] == 67 + i
        t = owner['tensors'][role]
        assert (t['type'], t['bytes'], t['ne'], t['nb']) == (kind, size, ne, nb)
        assert t['relative_offset'] + size <= t['root_bytes'] and t['view_depth'] <= 32
        b = (base / 'capture-first' / f'owner-{i}-{role}.bin').read_bytes()
        assert len(b) == size and (base / 'capture-repeat' / f'owner-{i}-{role}.bin').read_bytes() == b
        if role == 'q':
            assert t['relative_offset'] == i*32768 and t['root_bytes'] == 131072
            assert t['root_ne'] == [256,32,4,1] and t['root_nb'] == [4,1024,32768,131072]
            a = array.array('f'); a.frombytes(b); assert all(math.isfinite(v) for v in a)
        elif role in ['k', 'v']:
            assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<e', b))
            # Cold/reset context: current target query has already written KV.
            assert b[(68+i)*8192:] == bytes((256-68-i)*8192), (i, role, 'uninitialized padding')
        else:
            a = list(struct.unpack('<8192H', b))
            assert a[:68+i] == [0]*(68+i) and a[68+i:256] == [0xfc00]*(256-68-i)
            assert a[256:] == [0xfc00]*(8192-256), (i, 'padded query mask')
        chunks.append(b)
    data = b''.join(chunks)
    with (packed / f'{role}.bin').open('xb') as f: f.write(data)
    ids[role] = identity(data)
parameters = first['owners'][0]['op_params']
assert parameters == [1065353216, 0, 0, 10] + [0]*12
assert all(o['op_params'] == parameters for o in first['owners'])
data = struct.pack('<16i', *parameters)
with (packed / 'params.bin').open('xb') as f: f.write(data)
ids['params'] = identity(data)
receipt = {'schema': 1, 'source_freeze': identity((base/'source-freeze-capture1.json').read_bytes()),
           'complete_model_heads_capture_control_byte_exact': identity(heads),
           'native_origin_operands_own_repeat_byte_exact': True,
           'captured_metadata': identity((base/'capture-first/operands.json').read_bytes()),
           'packed': ids, 'origin': 'native only; reconstructed four-stream roots, not original-model captured tensors',
           'query_positions': [67,68,69,70], 'initialized_endpoints': [68,69,70,71],
           'source_policy': freeze['policy']}
with (base / 'capture-own-freeze1.json').open('x') as f: json.dump(receipt,f,indent=2);f.write('\n')
print('CAPTURE_FIDELITY complete4heads exact; 16 operands repeated; cold padded cells zero; query masks exact')
print('CAPTURE_OWN_FREEZE',identity((base/'capture-own-freeze1.json').read_bytes()))
