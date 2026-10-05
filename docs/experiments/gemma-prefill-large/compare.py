#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate paid counts and two complete heads, without a quality/peak gate."""
import array
import hashlib
import json
import math
from pathlib import Path
import sys

VOCAB = 262144


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head(path):
    raw = path.read_bytes()
    assert len(raw) == VOCAB * 4
    values = array.array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    assert all(math.isfinite(value) for value in values)
    return values, hashlib.sha256(raw).hexdigest()


def compare(a, b):
    def distribution(values):
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        total = math.fsum(weights)
        return weights, total, maximum + math.log(total)
    aw, at, al = distribution(a)
    bw, bt, bl = distribution(b)
    ai = max(range(len(a)), key=a.__getitem__)
    bi = max(range(len(b)), key=b.__getitem__)
    return {'max_raw_delta': max(abs(x - y) for x, y in zip(a, b)),
            'full_softmax_tv': .5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)),
            'reference_argmax_nll_delta': abs((al - a[bi]) - (bl - b[bi])),
            'argmax': ai, 'reference_argmax': bi}


def runs(path, policy='ordinary', variant='26', mixed=False):
    assert (variant, policy) in [('26', 'ordinary'), ('26', 'all'), ('31', 'both')]
    result, prefixes, current = [], [], None
    for line in path.read_text().splitlines():
        if line.startswith('PREFILL_TIMED_PREFIX '):
            fields = dict(item.split('=', 1) for item in line.split()[1:])
            prefixes.append((int(fields['appended']), int(fields['past'])))
        elif line.startswith(('PREFILL_NATIVE ', 'PREFILL_REFERENCE ')):
            assert prefixes == [(236761, 8193), (108, 8194), (236913, 8195)]
            prefixes = []
            current = {'engine': line.split()[0],
                       'metrics': dict(item.split('=', 1) for item in line.split()[1:]),
                       'tokens': []}
            m = current['metrics']
            assert (m['prefill_rows'], m['decode_chunks'], m['timed_start'],
                    m['completed'], m['context']) == ('8192', '32', '8195', '8227', '16384')
            assert all(math.isfinite(float(m[key])) and float(m[key]) > 0
                       for key in ['prefill_seconds', 'decode_seconds'])
            if current['engine'] == 'PREFILL_NATIVE':
                assert m['masks'] == 'device'
                assert all(m[key] == '0' for key in ['shared_vecq', 'row_products', 'norm_fused',
                           'rope_store'])
                expected = {'ordinary': (0, 0, 0, 0), 'all': (60, 90, 30, 30),
                            'both': (120, 120, 0, 0)}[policy]
                if mixed and m['norm_rope'] == '0':
                    expected = (0, 0, 0, 0)
                assert tuple(int(m[key]) for key in ['norm_rope', 'norm_add', 'gemma_route',
                                                     'gemma_reduce']) == expected
                assert tuple(int(m[key]) for key in ['prefill_norm_rope',
                                                     'prefill_norm_add']) == expected[:2]
                if 'prefill_gemma_route' in m:
                    assert tuple(int(m[key]) for key in ['prefill_gemma_route',
                                                         'prefill_gemma_reduce']) == expected[2:]
                assert int(m['prefill_chunks']) == 8192 // int(m['chunk'])
                assert int(m['intermediate_heads']) == int(m['prefill_chunks']) - 1
            else:
                assert (m['batch'], m['ubatch'], m['fusion'], m['graphs'],
                        m['intermediate_heads']) == ('8192', '1024' if variant == '26' else '256',
                                                    'enabled', 'allowed', '0')
            result.append(current)
        elif line.startswith('PREFILL_TOKEN '):
            assert current is not None
            fields = dict(item.split('=', 1) for item in line.split()[1:])
            assert int(fields['step']) == len(current['tokens'])
            current['tokens'].append((int(fields['argmax']), int(fields['forced'])))
    assert all(len(run['tokens']) == 32 for run in result)
    return result


def main():
    assert 3 <= len(sys.argv) <= 5
    root, expected_freeze = Path(sys.argv[1]), sys.argv[2]
    policy = sys.argv[3] if len(sys.argv) >= 4 else 'ordinary'
    variant = sys.argv[4] if len(sys.argv) == 5 else '26'
    assert (variant, policy) in [('26', 'ordinary'), ('26', 'all'), ('31', 'both')]
    rows = 1024 if variant == '26' else 256
    freeze_file = root / f'{policy}{rows}-own-repeat-frozen.json'
    assert digest(freeze_file) == expected_freeze
    own = json.loads(freeze_file.read_text())
    # The authenticated initial ordinary26 receipt predates the policy field.
    assert (own['variant'], own.get('policy', 'ordinary'), own['max_rows'], own['context'],
            own['completed_positions']) == (int(variant), policy, rows, 16384, 8227)
    source = ('native1.log' if policy == 'ordinary' else 'all1.log') if variant == '26' else 'native31.log'
    bookend_log = ('bookend1.log' if policy == 'ordinary' else 'allbookend1.log') if variant == '26' else 'bookend31.log'
    assert digest(root / source) == own['source_log_sha256']
    native = runs(root / source, policy, variant, mixed=variant == '31')
    bookends = runs(root / bookend_log, policy, variant)
    assert len(native) == ((3 if policy == 'ordinary' else 2) if variant == '26' else 5) and len(bookends) == 3
    assert [run['metrics']['chunk'] for run in native] == (
        (['128', '1024', '1024'] if policy == 'ordinary' else ['1024', '1024'])
        if variant == '26' else ['128', '256', '256', '256', '256'])
    assert [run['engine'] for run in bookends] == ['PREFILL_REFERENCE', 'PREFILL_NATIVE',
                                                 'PREFILL_REFERENCE']
    assert native[-2]['tokens'] == native[-1]['tokens'] == bookends[1]['tokens']
    assert bookends[0]['tokens'] == bookends[2]['tokens']
    gold = [token[1] for token in native[0]['tokens']]
    assert all([token[1] for token in run['tokens']] == gold for run in native + bookends)
    reference = ('reference-large1024' if policy == 'ordinary' else 'reference-largeall1024') if variant == '26' else 'reference-large256'
    cases = [f'{policy}{rows}-first', f'{policy}{rows}-repeat', f'{policy}{rows}-bookend',
             reference + '-a', reference + '-b']
    if policy == 'ordinary':
        cases.insert(0, 'ordinary128')
    if variant == '31':
        cases = ['ordinary128', 'ordinary256-first', 'ordinary256-repeat'] + cases
        ordinary = json.loads((root / 'ordinary256-own-repeat-frozen.json').read_text())
        assert ordinary['source_log_sha256'] == digest(root / source)
        assert (ordinary['variant'], ordinary['policy'], ordinary['max_rows']) == (31, 'ordinary', 256)
    result = {'scope': 'two retained heads only; no representative quality, memory or support gate',
              'policy': policy, 'variant': variant, 'freeze_sha256': expected_freeze,
              'native_screen': native, 'bookends': bookends,
              'heads': {}}
    if variant == '31':
        result['ordinary_freeze_sha256'] = digest(root / 'ordinary256-own-repeat-frozen.json')
    for name in ['prefill.f32', 'final.f32']:
        values, hashes = {}, {}
        for case in cases:
            values[case], hashes[case] = head(root / ('heads' if variant == '26' else 'heads31') / case / name)
        baseline = values[reference + '-a']
        assert hashes[reference + '-a'] == hashes[reference + '-b']
        candidate = own['arms'][f'{policy}{rows}-first']['heads'][name]['sha256']
        assert all(hashes[case] == candidate for case in cases if case.startswith(f'{policy}{rows}'))
        if variant == '31':
            assert hashes['ordinary256-first'] == hashes['ordinary256-repeat'] == ordinary['arms']['ordinary256-first']['heads'][name]['sha256']
        result['heads'][name] = {'sha256': hashes,
            f'relative_to_reference{rows}': {case: compare(value, baseline)
                                          for case, value in values.items()}}
    result['strict_argmax_differences_32'] = sum(a[0] != b[0] for a, b in
                                               zip(bookends[1]['tokens'], bookends[0]['tokens']))
    for stage in ['prefill', 'decode']:
        key = stage + '_seconds'
        mean = sum(float(run['metrics'][key]) for run in [bookends[0], bookends[2]]) / 2
        result[stage + '_native_relative_to_reference_percent'] = 100 * (
            float(bookends[1]['metrics'][key]) / mean - 1)
    output = root / (('comparison.json' if policy == 'ordinary' else 'comparison-all.json') if variant == '26' else 'comparison31.json')
    with output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
