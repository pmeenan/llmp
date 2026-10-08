#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Closed all32-head C1 screen; candidate own bound does not calibrate control."""
import importlib.util
import json
import math
import pathlib
import re
import sys

sys.dont_write_bytecode = True
source = pathlib.Path(__file__).resolve().parents[1]/'gemma-joined-serving/analyze.py'
spec = importlib.util.spec_from_file_location('historical_joined_analysis', source)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def compare(root, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected) or a.digest(root/'native-frozen.json') != expected:
        raise ValueError('native own freeze identity changed')
    own = json.loads((root/'native-frozen.json').read_text())
    assert own['owners'] == 1 and own['steps'] == 32 and own['candidate_own_repeat_byte_exact'] is True
    assert a.digest(root/'source-identities.json') == own['source_identity']['sha256']
    ids = a.read_inputs(root/'inputs.i32')
    pre = json.loads((root/'source-identities.json').read_text())
    assert a.digest(root/'inputs.i32') == pre['environment']['inputs']['sha256']
    for name, r in own['records'].items():
        assert a.digest(root/name/'heads.f32') == r['sha256']
        assert a.digest(root/name/'inputs.i32') == pre['environment']['inputs']['sha256']
    paths = ['reference-first', 'reference-repeat', 'candidate-bookend']
    for name in paths:
        assert a.digest(root/name/'inputs.i32') == pre['environment']['inputs']['sha256']
    reference = a.record(root/'reference-first/heads.f32', 1)
    assert a.record(root/'reference-repeat/heads.f32', 1) == reference
    assert a.record(root/'candidate-bookend/heads.f32', 1) == own['records']['candidate-first']
    result = {'native_freeze_sha256': expected, 'source_identity': own['source_identity'],
              'reference_sha256': reference['sha256'], 'reference_repeat_byte_exact': True,
              'candidate_bookend_byte_exact': True, 'complete_rows': 32, 'comparisons': {},
              'corpus_ppl_claim': False, 'control_own_noise_measured': False}
    for name, calibrated in [('candidate-first', True), ('ordinary-control', False)]:
        ownrecord = own['records'][name]
        count = maximum = outside = ties = strict = 0
        selected = []
        with (root/name/'heads.f32').open('rb') as x, (root/'reference-first/heads.f32').open('rb') as y:
            for s in range(32):
                ah, bh = x.read(a.VOCAB*4), y.read(a.VOCAB*4)
                ar, br, m = a.compared_row(ah, bh, ownrecord['argmax'][s], reference['argmax'][s])
                count += m['byte_exact']; strict += m['strict']; ties += m['reference_tie']
                outside += m['outside_zero_margin']; maximum = max(maximum, m['max_raw'])
                if s in (0, 7, 15, 30):
                    ap, az, am = a.distribution(ar); bp, bz, bm = a.distribution(br)
                    target = ids[68+s]
                    selected.append({'step': s, 'owner': 0, 'target_position': 68+s,
                                     'tv': .5*math.fsum(abs(v/az-w/bz) for v,w in zip(ap,bp,strict=True)),
                                     'native_nll': math.log(az)+am-ar[target],
                                     'reference_nll': math.log(bz)+bm-br[target]})
        result['comparisons'][name] = {'byte_exact_rows': count, 'strict_argmax_differences': strict,
                                      'strict_differences_at_reference_tie': ties, 'all_head_max_raw_delta': maximum,
                                      'outside_frozen_zero_candidate_margin_noise': outside if calibrated else None,
                                      'margin_noise_basis': 'candidate full-head repeat byte-exact' if calibrated else 'control own repeat not acquired',
                                      'selected_likelihood_rows': selected}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        a.selftest()
    else:
        assert len(sys.argv) == 3
        compare(pathlib.Path(sys.argv[1]), sys.argv[2])
