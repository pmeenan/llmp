#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete attention-output comparisons, with exact bytes and explicit ties."""
import array
import hashlib
import json
import math
import pathlib
import re
import struct
import sys


def metrics(reference_bytes, candidate_bytes, owners=4):
    assert len(reference_bytes) == len(candidate_bytes) and len(reference_bytes) % (4*owners) == 0
    a = array.array('f'); a.frombytes(reference_bytes)
    b = array.array('f'); b.frombytes(candidate_bytes)
    assert all(math.isfinite(v) for v in a) and all(math.isfinite(v) for v in b)
    denominator = sum(float(v)*float(v) for v in a)
    delta = [float(x)-float(y) for x, y in zip(a, b)]
    strict = positive_margin = ties = 0
    width = len(a)//owners
    for owner in range(owners):
        ar, br = a[owner*width:(owner+1)*width], b[owner*width:(owner+1)*width]
        ai, bi = max(range(width), key=ar.__getitem__), max(range(width), key=br.__getitem__)
        if ai != bi:
            strict += 1
            margin = float(ar[ai])-float(ar[bi])
            assert margin >= 0
            positive_margin += margin > 0
            ties += margin == 0
    return {'complete_values': len(a), 'whole_file_byte_exact': reference_bytes == candidate_bytes,
            'max_raw_delta': max(map(abs, delta), default=0),
            'nmse_reference_energy': sum(d*d for d in delta)/denominator if denominator else None,
            'strict_attention_argmax_mismatches': strict,
            'positive_reference_margin_mismatches': positive_margin,
            'exact_reference_tie_mismatches': ties,
            'limits': 'Attention-coordinate choices only, not target-vocabulary choices or model quality.'}


def selftest():
    def packed(v): return struct.pack('<'+str(len(v))+'f', *v)
    m = metrics(packed([0.0, 1.0]), packed([-0.0, 1.0]), 1)
    assert not m['whole_file_byte_exact'] and m['max_raw_delta'] == 0
    m = metrics(packed([1.0, 1.0]), packed([0.0, 1.0]), 1)
    assert m['strict_attention_argmax_mismatches'] == 1 and m['exact_reference_tie_mismatches'] == 1 and m['positive_reference_margin_mismatches'] == 0
    for values in [[float('nan'), 1], [float('inf'), 1]]:
        try: metrics(packed(values), packed([1, 1]), 1)
        except AssertionError: pass
        else: raise AssertionError('nonfinite accepted')
    print('signedzero/tie/nonfinite metrics controls PASS')


def identify(path):
    b = path.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def compare(base, log, output):
    pre = json.loads((base/'source-freeze-replay1.json').read_text())
    own = json.loads((base/'native-own-freeze1.json').read_text())
    assert own['source_freeze'] == identify(base/'source-freeze-replay1.json')
    for name, expected in pre['packed'].items():
        assert identify(base/'packed'/(name+'.bin')) == expected
    reference = (base/'original-first/first.f32').read_bytes()
    assert len(reference) == 131072
    for directory in ['original-first', 'original-repeat']:
        assert (base/directory/'first.f32').read_bytes() == reference
        assert (base/directory/'repeat.f32').read_bytes() == reference
        assert json.loads((base/directory/'completion.json').read_text()) == {'phase': 'completed_after_backend_release', 'owners': 4, 'paid_waves': 32, 'completed_units': 128}
    result = {'schema': 1, 'native_source_freeze': identify(base/'source-freeze-replay1.json'),
              'native_own_freeze': identify(base/'native-own-freeze1.json'),
              'origin': 'Native target first-local inputs; reconstructed four-stream roots; no original-model operand capture.',
              'reference': identify(base/'original-first/first.f32'), 'comparisons': {}}
    for arm in ['B', 'C', 'D']:
        first = (base/('native-paid-'+arm)/'first.f32').read_bytes()
        assert identify(base/('native-own-'+arm)/'first.f32') == own['outputs'][arm]['first']
        assert first == (base/('native-own-'+arm)/'first.f32').read_bytes()
        assert first == (base/('native-paid-'+arm)/'repeat.f32').read_bytes()
        assert (base/('native-paid-'+arm)/'fresh.f32').read_bytes() == (base/('native-own-'+arm)/'fresh.f32').read_bytes()
        result['comparisons'][arm] = metrics(reference, first)
        result['comparisons'][arm]['native_result'] = identify(base/('native-paid-'+arm)/'first.f32')
    result['segmented_vs_fourstream_mma'] = metrics((base/'native-paid-D/first.f32').read_bytes(), (base/'native-paid-C/first.f32').read_bytes())
    times = re.findall(r'(ORIGINAL_ATTENTION_REPLAY|ATTENTION_REPLAY) seconds=([0-9.e+-]+)', log.read_text())
    assert [x[0] for x in times] == ['ORIGINAL_ATTENTION_REPLAY','ATTENTION_REPLAY','ATTENTION_REPLAY','ATTENTION_REPLAY','ORIGINAL_ATTENTION_REPLAY']
    t = [float(x[1]) for x in times]
    mean = (t[0]+t[-1])/2
    result['paid_bookend'] = {'waves_per_arm': 32, 'completed_owner_units_per_arm': 128,
                             'output_values_per_wave': 32768, 'original_seconds': [t[0],t[-1]],
                             'native_seconds': dict(zip(['B','C','D'],t[1:4])),
                             'native_latency_delta_percent_vs_reference_mean': {a: (v/mean-1)*100 for a,v in zip(['B','C','D'],t[1:4])},
                             'limits': 'One-op host wall time includes submission/completion, full D2H and output scans; foreign/native scheduling differ; no full model gain or adoption.'}
    with output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']: selftest()
    else: compare(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
