#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Full common-input operator bytes; never a whole-chain/PPL comparison."""
import hashlib
import json
import math
import os
import pathlib
import runpy
import struct
import sys

snapshot = runpy.run_path(str(pathlib.Path(__file__).with_name('source_snapshot.py')))
identity, admitted_receipts = snapshot['record'], snapshot['admitted_receipts']
valid = runpy.run_path(str(pathlib.Path(__file__).with_name('own_freeze.py')))['valid']


def metric(a, b, route):
    valid(a, route); valid(b, route)
    id_mismatches = None
    if route:
        ai, bi = list(struct.iter_unpack('<i', a[:2048])), list(struct.iter_unpack('<i', b[:2048]))
        id_mismatches = sum(x != y for x, y in zip(ai, bi))
        a, b = a[2048:], b[2048:]
    av, bv = [v[0] for v in struct.iter_unpack('<f', a)], [v[0] for v in struct.iter_unpack('<f', b)]
    return {'float_values': len(av), 'float_bytes_exact': a == b,
            'max_absolute_float_delta': max(abs(x-y) for x, y in zip(av, bv)),
            'selected_id_mismatches': id_mismatches,
            'candidate_float_sha256': hashlib.sha256(a).hexdigest(),
            'reference_float_sha256': hashlib.sha256(b).hexdigest(),
            'normalized_sum_range_candidate': [min(math.fsum(av[i:i+8]) for i in range(0, len(av), 8)),
                                                max(math.fsum(av[i:i+8]) for i in range(0, len(av), 8))] if route else None}


def read(path, route):
    if identity(path)['bytes'] != (4096 if route else 720896):
        raise ValueError('wrong full output size')
    data = path.read_bytes()
    valid(data, route)
    return data


def main():
    if len(sys.argv) != 16:
        raise ValueError('PRE OWN_FREEZE CAPTURE_RECEIPT ORIGINAL_PROOF CAPTURE_ROOT NATIVE_PRIMITIVE NATIVE_FUSED ORIGINAL_PRIMITIVE ORIGINAL_FUSED kind NEW_AGGREGATE EXPECTED_PRE_SHA EXPECTED_OWN_SHA EXPECTED_CAPTURE_SHA EXPECTED_ORIGINAL_SHA')
    pre_path, own_path, capture_path, original_path, captured, native_p, native_f, original_p, original_f = map(pathlib.Path, sys.argv[1:10])
    kind, out = sys.argv[10], pathlib.Path(sys.argv[11])
    if kind not in ('route', 'reduce'):
        raise ValueError('closed kind')
    receipts, admissions = admitted_receipts(dict(zip(
        ('pre', 'own', 'capture', 'original'),
        zip((pre_path, own_path, capture_path, original_path), sys.argv[12:16]))))
    own, original, capture = (receipts[name] for name in ('own', 'original', 'capture'))
    if own['schema'] != 'gemma26-late-moe-native-own-v1' or own['kind'] != kind or \
            own['pre_sha256'] != identity(pre_path)['sha256'] or \
            own['capture_receipt_sha256'] != identity(capture_path)['sha256'] or \
            original['schema'] != 'gemma26-late-moe-original-v1' or original['kind'] != kind or \
            original['pre_sha256'] != own['pre_sha256'] or \
            original['native_own_sha256'] != identity(own_path)['sha256'] or \
            original['capture_receipt_sha256'] != own['capture_receipt_sha256'] or \
            original['input_manifest'] != own['input_manifest']:
        raise ValueError('missing native blind own freeze')
    result = {'kind': kind, 'rows_per_layer': 64, 'layers': {}, 'native_own_freeze': admissions['own'],
              'prior_gate_receipt_identities': admissions,
              'claim': 'common fixed operands only; no product/input-history/whole-head attribution or adoption'}
    route = kind == 'route'
    for layer in (28, 29):
        data = {}
        for name, root in [('native_primitive', native_p), ('native_fused', native_f),
                            ('original_primitive', original_p), ('original_fused', original_f)]:
            first, repeat = (root / f'layer-{layer}-{which}.bin' for which in ('first', 'repeat'))
            data[name] = read(first, route)
            if read(repeat, route) != data[name]:
                raise ValueError('own eager/captured/repeated output changed')
            if name.startswith('native'):
                key = f"{name.removeprefix('native_')}/{layer}"
                if identity(first) != own['own_output_proofs'][key][0]:
                    raise ValueError('native output differs from pre-exposure calibration')
            else:
                key = f"{name.removeprefix('original_')}/{layer}"
                if identity(first) != original['files'][key]['first'] or \
                        identity(repeat) != original['files'][key]['repeat']:
                    raise ValueError('original output differs from admitted counterpart acquisition')
        roles = ('ids', 'route-weights') if route else ('sum',)
        for role in roles:
            name = f'layer-{layer}-{role}.bin'
            if identity(captured / name) != capture['files'][name]:
                raise ValueError('captured stock operator output changed')
        if route:
            ids = (captured / f'layer-{layer}-ids.bin').read_bytes()
            weights = (captured / f'layer-{layer}-route-weights.bin').read_bytes()
            if len(ids) != 2048 or len(weights) != 2048:
                raise ValueError('incomplete captured first-eight routing output')
            data['captured_stock'] = ids + weights
        else:
            data['captured_stock'] = read(captured / f'layer-{layer}-sum.bin', False)
        valid(data['captured_stock'], route)
        comparisons = {}
        for a, b in [('native_primitive', 'original_primitive'), ('native_fused', 'original_fused'),
                     ('original_fused', 'original_primitive'), ('original_primitive', 'captured_stock'),
                     ('original_fused', 'captured_stock')]:
            comparisons[a + '_vs_' + b] = metric(data[a], data[b], route)
            comparisons[a + '_vs_' + b]['whole_bytes_exact'] = data[a] == data[b]
        result['layers'][str(layer)] = comparisons
    os.umask(0o077)
    with out.open('x') as file:
        json.dump(result, file, indent=2); file.write('\n'); file.flush(); os.fsync(file.fileno())


if __name__ == '__main__':
    main()
