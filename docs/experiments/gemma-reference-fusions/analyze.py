#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare fixed reference norm allowlists with immutable native/production rows.

Run under installed spark-job --gpu; retained vectors never enter Git.
Usage: ANALYZER FUSION_ROOT QUALITY_ROOT
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys

CALIBRATION = '2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901'
BASELINE_SHA = {
    'off': '1c66029f1f088e04a7a4001fbdd0e24afbb97ad3b9e85f20fb73c58292e4b340',
    'on': 'c07c711bfc732ce498182917c0cc3cfd74f21374975e9fa69c90ec88461cc256',
}


def main():
    assert len(sys.argv) in (3, 4)
    root, quality = map(pathlib.Path, sys.argv[1:3])
    combined = len(sys.argv) == 4
    assert not combined or sys.argv[3] == "both"
    arms = ("both",) if combined else ("norm_rope", "norm_add")
    module = pathlib.Path(__file__).resolve().parent.parent / 'gemma-quality' / 'analyze.py'
    spec = importlib.util.spec_from_file_location('quality_analysis', module)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    freeze_bytes = (quality / 'native-noise-128-frozen.json').read_bytes()
    assert hashlib.sha256(freeze_bytes).hexdigest() == CALIBRATION
    freeze = json.loads(freeze_bytes)
    raw_ids = (quality / 'input1' / 'ids.i32').read_bytes()
    assert hashlib.sha256(raw_ids).hexdigest() == common.ID_SHA
    ids = struct.unpack('<1024i', raw_ids)
    paths = {
        'off': quality / 'native-ordinary128' / 'logits.f32',
        'on': quality / 'reference128-first' / 'logits.f32',
    }
    directories = dict(norm_rope='norm-rope', norm_add='norm-add', both='both')
    for arm in arms:
        paths[arm] = root / (directories[arm] + '-first') / 'logits.f32'
        paths[arm + '_repeat'] = root / (directories[arm] + '-repeat') / 'logits.f32'
    for path in paths.values():
        assert path.stat().st_size == common.ROWS * common.VOCAB * 4
    files = {name: path.open('rb') for name, path in paths.items()}
    digests = {name: hashlib.sha256() for name in paths}
    rows = {arm: [] for arm in arms}
    nll = {name: [] for name in ('off', 'on') + arms}
    try:
        for index in range(common.ROWS):
            values, hashes = {}, {}
            for name, file in files.items():
                raw = file.read(common.VOCAB * 4)
                assert len(raw) == common.VOCAB * 4
                digests[name].update(raw)
                hashes[name] = hashlib.sha256(raw).hexdigest()
                if not name.endswith('_repeat'):
                    # The existing bounded helper validates all finite F32 cells.
                    import io
                    values[name], checked_hash = common.read_row(io.BytesIO(raw))
                    assert checked_hash == hashes[name]
            assert hashes['off'] == freeze['hashes'][index][0]
            for arm in arms:
                assert hashes[arm] == hashes[arm + '_repeat'], 'allowlist own repeat changed'
            distributions = {name: common.distribution(row) for name, row in values.items()}
            winners = {name: max(range(common.VOCAB), key=row.__getitem__)
                       for name, row in values.items()}
            assert winners['off'] == freeze['argmax_ids'][index]
            if index + 1 < common.ROWS:
                target = ids[index + 1]
                for name, row in values.items():
                    nll[name].append(distributions[name][2] - row[target])
            for arm in arms:
                item = dict(row=index, argmax=winners[arm], repeat_byte_exact=True)
                aw, at, _ = distributions[arm]
                for baseline in ('off', 'on'):
                    bw, bt, _ = distributions[baseline]
                    item[baseline] = dict(
                        byte_exact=hashes[arm] == hashes[baseline],
                        max_raw_delta=max(abs(a-b) for a, b in zip(values[arm], values[baseline])),
                        full_softmax_tv=.5*math.fsum(abs(a/at-b/bt) for a, b in zip(aw, bw)),
                        argmax_mismatch=winners[arm] != winners[baseline],
                        baseline_margin=values[baseline][winners[baseline]] -
                                        values[baseline][winners[arm]])
                rows[arm].append(item)
            if index % 128 == 0:
                print('ALLOWLIST_ANALYSIS completed_rows=' + str(index + 1), flush=True)
    finally:
        for file in files.values():
            file.close()
    file_hashes = {name: digest.hexdigest() for name, digest in digests.items()}
    for name, expected in BASELINE_SHA.items():
        assert file_hashes[name] == expected
    means = {name: math.fsum(scores)/len(scores) for name, scores in nll.items()}
    summaries = {}
    for arm in arms:
        summaries[arm] = dict(mean_nll=means[arm], ppl=math.exp(means[arm]),
                              repeat_byte_exact_rows=common.ROWS)
        for baseline in ('off', 'on'):
            comparisons = [row[baseline] for row in rows[arm]]
            summaries[arm][baseline] = dict(
                byte_exact_rows=sum(row['byte_exact'] for row in comparisons),
                strict_argmax_mismatches=sum(row['argmax_mismatch'] for row in comparisons),
                outside_frozen_noise=sum(row['argmax_mismatch'] and
                                         row['baseline_margin'] > freeze['p99_top2_margin_movement']
                                         for row in comparisons),
                max_raw_delta=max(row['max_raw_delta'] for row in comparisons),
                max_full_softmax_tv=max(row['full_softmax_tv'] for row in comparisons),
                ppl_relative_percent=100*math.expm1(means[arm]-means[baseline]))
    result = dict(input_sha256=common.ID_SHA, rows=common.ROWS, scored_targets=common.ROWS-1,
                  frozen_native_calibration_sha256=CALIBRATION,
                  frozen_native_p99=freeze['p99_top2_margin_movement'],
                  complete_file_sha256=file_hashes, mean_nll=means,
                  summaries=summaries, comparisons=rows,
                  scope='same gold-prefix 128-row reference allowlists; no native policy adoption or qualification')
    with (root / ('comparison-both.json' if combined else 'comparison.json')).open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'comparisons'}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
