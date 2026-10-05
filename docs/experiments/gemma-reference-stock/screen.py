#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete-row byte/argmax screen plus explicitly selected likelihood rows.

ROOT QUALITY POLICY. Supervised heavy analysis; no sampled quality gate.
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys

CALIBRATION = '2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901'
BASELINE_SHA = dict(native='1c66029f1f088e04a7a4001fbdd0e24afbb97ad3b9e85f20fb73c58292e4b340',
                    stock='c07c711bfc732ce498182917c0cc3cfd74f21374975e9fa69c90ec88461cc256')
POLICIES = dict(no_routing='no-routing', no_reduction='no-reduction',
                no_norm_mul='no-norm-mul', no_softcap='no-softcap')


def main():
    assert len(sys.argv) == 4
    root, quality = map(pathlib.Path, sys.argv[1:3])
    policy = sys.argv[3]
    assert policy in POLICIES
    module = pathlib.Path(__file__).resolve().parent.parent / 'gemma-quality' / 'analyze.py'
    spec = importlib.util.spec_from_file_location('quality_analysis', module)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    calibration_bytes = (quality / 'native-noise-128-frozen.json').read_bytes()
    assert hashlib.sha256(calibration_bytes).hexdigest() == CALIBRATION
    calibration = json.loads(calibration_bytes)
    raw_ids = (quality / 'input1' / 'ids.i32').read_bytes()
    assert hashlib.sha256(raw_ids).hexdigest() == common.ID_SHA
    ids = struct.unpack('<1024i', raw_ids)
    paths = dict(native=quality / 'native-ordinary128' / 'logits.f32',
                 stock=quality / 'reference128-first' / 'logits.f32',
                 candidate=root / (POLICIES[policy] + '-first') / 'logits.f32',
                 repeat=root / (POLICIES[policy] + '-repeat') / 'logits.f32')
    for path in paths.values():
        assert path.stat().st_size == common.ROWS * common.VOCAB * 4
    files = {name: path.open('rb') for name, path in paths.items()}
    hashes = {name: hashlib.sha256() for name in paths}
    selected = {0, 127, 128, 512, 1023}
    first_changed = first_changed_argmax = None
    rows, selected_rows = [], []
    try:
        for index in range(common.ROWS):
            values, row_hashes = {}, {}
            for name, file in files.items():
                raw = file.read(common.VOCAB * 4)
                assert len(raw) == common.VOCAB * 4
                hashes[name].update(raw)
                row_hashes[name] = hashlib.sha256(raw).hexdigest()
                if name != 'repeat':
                    import io
                    values[name], checked_hash = common.read_row(io.BytesIO(raw))
                    assert checked_hash == row_hashes[name]
            assert row_hashes['native'] == calibration['hashes'][index][0]
            assert row_hashes['candidate'] == row_hashes['repeat'], 'own repeat changed'
            winners = {name: max(range(common.VOCAB), key=row.__getitem__)
                       for name, row in values.items()}
            assert winners['native'] == calibration['argmax_ids'][index]
            item = dict(row=index, winners=winners, repeat_byte_exact=True)
            for baseline in ('native', 'stock'):
                item[baseline] = dict(byte_exact=row_hashes['candidate'] == row_hashes[baseline],
                                      argmax_mismatch=winners['candidate'] != winners[baseline],
                                      max_raw_delta=max(abs(a-b) for a,b in
                                                        zip(values['candidate'], values[baseline])))
            if first_changed is None and not item['stock']['byte_exact']:
                first_changed = index
                selected.add(index)
            if first_changed_argmax is None and item['stock']['argmax_mismatch']:
                first_changed_argmax = index
                selected.add(index)
            if index in selected:
                distributions = {name: common.distribution(row) for name,row in values.items()}
                detail = dict(row=index, reason='fixed boundary or first stock byte/argmax change')
                aw, at, _ = distributions['candidate']
                for baseline in ('native', 'stock'):
                    bw, bt, _ = distributions[baseline]
                    detail[baseline] = dict(full_softmax_tv=.5*math.fsum(
                        abs(a/at-b/bt) for a,b in zip(aw,bw)))
                if index+1 < common.ROWS:
                    target = ids[index+1]
                    detail['target'] = target
                    detail['target_nll'] = {
                        name: distributions[name][2]-row[target] for name,row in values.items()}
                selected_rows.append(detail)
            rows.append(item)
            if index % 128 == 0:
                print('STOCK_SCREEN completed_rows=' + str(index+1), flush=True)
    finally:
        for file in files.values():
            file.close()
    digests = {name: digest.hexdigest() for name,digest in hashes.items()}
    for name, expected in BASELINE_SHA.items():
        assert digests[name] == expected
    summaries = {}
    for baseline in ('native', 'stock'):
        summaries[baseline] = dict(
            byte_exact_rows=sum(row[baseline]['byte_exact'] for row in rows),
            strict_argmax_mismatches=sum(row[baseline]['argmax_mismatch'] for row in rows),
            max_raw_delta=max(row[baseline]['max_raw_delta'] for row in rows))
    result = dict(policy=policy, input_sha256=common.ID_SHA, rows=common.ROWS,
                  calibration_sha256=CALIBRATION, calibration_p99=calibration['p99_top2_margin_movement'],
                  complete_file_sha256=digests, repeat_byte_exact_rows=common.ROWS,
                  summaries=summaries, first_stock_byte_change=first_changed,
                  first_stock_argmax_change=first_changed_argmax,
                  explicitly_selected_distribution_rows=selected_rows, comparisons=rows,
                  scope='all heads for bytes/argmax/raw delta; selected likelihood rows only, no PPL or quality gate')
    with (root / (POLICIES[policy] + '-screen.json')).open('x') as file:
        json.dump(result,file,indent=2)
        file.write('\n')
    print(json.dumps({name:value for name,value in result.items() if name!='comparisons'},indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
