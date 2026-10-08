#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare complete real-operand FFN arrays; coordinate scans are not quality."""
import hashlib
import json
import math
import pathlib
import re
import struct
import sys

COUNTS = (21504, 5376)


def identify(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def compare(reference, candidate):
    assert len(reference) == len(candidate) and len(reference) % 4 == 0
    n = len(reference) // 4
    assert n > 0
    a, b = (struct.unpack('<' + str(n) + 'f', x) for x in (reference, candidate))
    assert all(math.isfinite(x) for x in a + b)
    delta = [float(x) - float(y) for x, y in zip(a, b)]
    energy = math.fsum(float(x) * float(x) for x in a)
    error = math.fsum(x * x for x in delta)
    ia = max(range(n), key=a.__getitem__)
    ib = max(range(n), key=b.__getitem__)
    return {
        "values": n, "byte_exact": reference == candidate,
        "different_f32_bit_patterns": sum(reference[4*i:4*i+4] != candidate[4*i:4*i+4] for i in range(n)),
        "maximum_raw_delta": max(map(abs, delta)),
        "reference_energy_nmse": error / energy if energy else (0.0 if not error else None),
        "coordinate_argmax_matches": ia == ib,
        "reference_winner_over_candidate_coordinate_margin": float(a[ia]) - float(a[ib]),
        "reference_energy_zero": energy == 0,
    }


def selftest():
    pack = lambda xs: struct.pack('<' + str(len(xs)) + 'f', *xs)
    assert compare(pack([1, 2]), pack([1, 2]))['byte_exact']
    z = compare(pack([0.0]), pack([-0.0]))
    assert not z['byte_exact'] and z['maximum_raw_delta'] == 0
    tie = compare(pack([2, 2, 0]), pack([1, 2, 0]))
    assert not tie['coordinate_argmax_matches'] and tie['reference_winner_over_candidate_coordinate_margin'] == 0
    changed = compare(pack([2, 1]), pack([0, 1]))
    assert changed['reference_winner_over_candidate_coordinate_margin'] == 1
    assert compare(pack([0]), pack([1]))['reference_energy_nmse'] is None
    for ref, cand in [(pack([float('nan')]), pack([0])), (pack([0]), pack([float('inf')])), (b'\0', b'\0')]:
        try:
            compare(ref, cand)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    print('ANALYZER_CONTROLS PASS byte/signed-zero/tie/positive-margin/zero-energy/nonfinite/length')


def main(base, output, expected_native_freeze_sha):
    assert re.fullmatch('[0-9a-f]{64}', expected_native_freeze_sha)
    freeze_bytes = (base/'native-frozen.json').read_bytes()
    freeze_id = identify(freeze_bytes)
    assert freeze_id['sha256'] == expected_native_freeze_sha
    frozen = json.loads(freeze_bytes)
    assert frozen['mode'] == 'native' and frozen['actual_capture_reconstruction_exact'] is True
    source_names = ['source-frozen.json', 'source-replay-frozen.json', 'source-replay2-frozen.json']
    source_name = next((name for name in source_names if (base/name).is_file()
                        and identify((base/name).read_bytes()) == frozen['source_freeze']), None)
    assert source_name is not None
    linked = {}
    for key, name in [('source_freeze', source_name),
                      ('capture_source_freeze', 'source-frozen.json'),
                      ('capture_freeze', 'capture-frozen.json')]:
        linked[key] = identify((base/name).read_bytes())
        assert linked[key] == frozen[key]
    capture_receipt = json.loads((base/'capture-frozen.json').read_bytes())
    assert capture_receipt['source_freeze'] == linked['capture_source_freeze']
    names = ['original-first', 'native-bookend', 'original-unfused', 'original-repeat']
    arrays, controls = {}, {}
    for name in names:
        data = (base/name/'first.f32').read_bytes()
        assert len(data) == 4 * sum(COUNTS)
        compare(data, data)
        assert data == (base/name/'repeat.f32').read_bytes()
        fresh = (base/name/'fresh.f32').read_bytes()
        assert len(fresh) == len(data) and fresh != data
        compare(fresh, fresh)
        arrays[name] = data
        controls[name] = {'first': identify(data), 'fresh': identify(fresh), 'repeat_byte_exact': True, 'fresh_changed': True}
        if name != 'native-bookend':
            assert json.loads((base/name/'completion.json').read_text()) == {'phase': 'completed_after_backend_release', 'paid_chains': 32}
    assert arrays['original-first'] == arrays['original-repeat']
    assert identify(arrays['native-bookend']) == frozen['full_outputs'][0]
    assert identify((base/'native-bookend/fresh.f32').read_bytes()) == frozen['fresh_outputs'][0]
    comparisons = {}
    for name in ['native-bookend', 'original-unfused']:
        comparisons[name] = {}
        start = 0
        for role, count in zip(['activation', 'down'], COUNTS):
            end = start + count * 4
            comparisons[name][role] = compare(arrays['original-first'][start:end], arrays[name][start:end])
            start = end
    log = (base/'bookend1.log').read_text()
    original = re.findall(r'ORIGINAL_FFN_REPLAY arm=(A0|A) seconds=([0-9.]+) completed_chains=32 api_groups=([0-9]+)', log)
    assert [x[0] for x in original] == ['A', 'A0', 'A']
    native = re.findall(r'^FFN_REPLAY seconds=([0-9.]+) completed_chains=32 api_groups=32 primitive_operations=128', log, re.M)
    assert len(native) == 1
    times = {'A_first': float(original[0][1]), 'B': float(native[0]), 'A0': float(original[1][1]), 'A_repeat': float(original[2][1])}
    mean = (times['A_first'] + times['A_repeat']) / 2
    result = {'schema': 1, 'origin': 'Dense31 scalar layer0 native input; full 21,504-value activation and 5,376-value down output.',
              'authenticated_native_own_freeze': freeze_id, 'authenticated_linked_receipts': linked,
              'controls': controls, 'comparisons_to_original_fused': comparisons,
              'paid_seconds_32_chains': times, 'original_bookend_mean_seconds': mean,
              'native_latency_delta_percent_vs_original_mean': 100*(times['B']/mean-1),
              'separate_original_latency_delta_percent_vs_original_mean': 100*(times['A0']/mean-1),
              'original_bookend_span_percent_of_mean': 100*abs(times['A_repeat']-times['A_first'])/mean,
              'limits': 'One isolated input/layer. Coordinate argmax is not vocabulary quality. No quality or performance adoption gate; timings are untraced and include full activation/down publication.'}
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        selftest()
    else:
        assert len(sys.argv) == 4
        main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
