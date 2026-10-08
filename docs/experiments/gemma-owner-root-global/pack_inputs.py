#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate native-origin capture/control fidelity; construct four-stream roots."""
import array
import os
import hashlib
import json
import math
import pathlib
import struct
import sys

if len(sys.argv) != 3:
    raise SystemExit("pack_inputs.py CAPTURE_CONTROL_BASE NEW_PACKED_DIR")
os.umask(0o077)
base = pathlib.Path(sys.argv[1])
packed = pathlib.Path(sys.argv[2])
def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
canonical_input_sha256 = 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'
input_bytes = None
for mode in ['control-first', 'control-repeat', 'capture-first', 'capture-repeat']:
    p = base / mode / 'inputs.i32'
    assert p.stat().st_size == 4096, (mode, 'input size')
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == canonical_input_sha256, (mode, 'canonical IDs')
    if input_bytes is None:
        input_bytes = data
    assert data == input_bytes, (mode, 'input identity')
assert (base / 'control-first/heads.f32').stat().st_size == 4 * 262144 * 4
heads = (base / 'control-first/heads.f32').read_bytes()
assert len(heads) == 4 * 262144 * 4
values = array.array('f'); values.frombytes(heads)
assert all(math.isfinite(v) for v in values)
for mode in ['control-repeat', 'capture-first', 'capture-repeat']:
    assert (base / mode / 'heads.f32').stat().st_size == len(heads), mode
    assert (base / mode / 'heads.f32').read_bytes() == heads, mode
assert (base / 'capture-first/operands.json').stat().st_size < 65536
assert (base / 'capture-repeat/operands.json').stat().st_size < 65536
first = json.loads((base / 'capture-first/operands.json').read_text())
assert first == json.loads((base / 'capture-repeat/operands.json').read_text())
assert first['origin'] == 'native dense31 first-global natural C4' and first['layer'] == 5 and len(first['owners']) == 4
packed.mkdir(mode=0o700)
roles = {'q': (0, 65536, [512, 1, 32, 1], [4, 65536, 2048, 65536]),
         'k': (1, 1048576, [512, 256, 4, 1], [2, 4096, 1024, 1048576]),
         'v': (1, 1048576, [512, 256, 4, 1], [2, 4096, 1024, 1048576]),
         'mask': (1, 16384, [256, 32, 1, 1], [2, 512, 16384, 16384])}
ids = {}
for role, (kind, size, ne, nb) in roles.items():
    chunks = []
    for i, owner in enumerate(first['owners']):
        assert owner['owner'] == i and owner['position'] == 67 + i
        t = owner['tensors'][role]
        assert (t['type'], t['bytes'], t['ne'], t['nb']) == (kind, size, ne, nb)
        assert t['relative_offset'] + size <= t['root_bytes'] and t['view_depth'] <= 32
        assert (base / 'capture-first' / f'owner-{i}-{role}.bin').stat().st_size == size
        assert (base / 'capture-repeat' / f'owner-{i}-{role}.bin').stat().st_size == size
        b = (base / 'capture-first' / f'owner-{i}-{role}.bin').read_bytes()
        assert len(b) == size and (base / 'capture-repeat' / f'owner-{i}-{role}.bin').read_bytes() == b
        if role == 'q':
            assert t['relative_offset'] == i*65536 and t['root_bytes'] == 262144
            assert t['root_ne'] == [512,32,4,1] and t['root_nb'] == [4,2048,65536,262144]
            a = array.array('f'); a.frombytes(b); assert all(math.isfinite(v) for v in a)
        elif role in ['k', 'v']:
            assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<e', b))
            # Cold/reset context: current target query has already written KV.
            assert b[(68+i)*4096:] == bytes((256-68-i)*4096), (i, role, 'uninitialized padding')
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
receipt = {'schema': 1, 'layer': 5, 'canonical_input': identity(input_bytes),
           'complete_model_heads_capture_control_byte_exact': identity(heads),
           'native_origin_operands_own_repeat_byte_exact': True,
           'captured_metadata': identity((base/'capture-first/operands.json').read_bytes()),
           'packed': ids, 'origin': 'native-origin reconstructed four-stream D512 roots',
           'query_positions': [67,68,69,70], 'initialized_endpoints': [68,69,70,71]}
with (packed / 'carrier.json').open('x') as f:
    json.dump(receipt,f,indent=2);f.write('\n')
print('GLOBAL_CAPTURE_FIDELITY complete4heads exact; 16 operands repeated; cold padded cells zero; query masks exact')
print('GLOBAL_CARRIER',identity((packed/'carrier.json').read_bytes()))
