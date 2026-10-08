#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity native schedule calibration; reuses the full-row quality judge.
freeze ROOT SCHEDULE SOURCE SOURCE_SHA MANIFEST MANIFEST_SHA;
calibrate ROOT1024 FREEZE_SHA ROOT128 FREEZE_SHA NEW_BOUND;
oracle ROOT OWN_SHA MANIFEST MANIFEST_SHA BOUND BOUND_SHA.
No oracle inputs are read by freeze/calibrate. All supplied receipt SHAs are
external prior gates, authenticated before JSON parse or output access.
"""
import hashlib
import heapq
import importlib.util
import json
import math
import pathlib
import sys

spec = importlib.util.spec_from_file_location(
    'quality', pathlib.Path(__file__).resolve().parent.parent / 'gemma-quality/analyze.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
ROWS, VOCAB = quality.ROWS, quality.VOCAB
METHOD = 'abs((A[top1]-A[top2])-(B[top1]-B[top2])); A=1024, B=128; lower-index A top2 ties; nearest-rank p99 over1024'


def authenticated(path, expected_sha):
    assert len(expected_sha) == 64 and all(c in '0123456789abcdef' for c in expected_sha)
    assert path.is_file() and path.stat().st_size <= 4 << 20
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha, 'prior gate identity changed'
    return json.loads(raw)


def recipe(schedule):
    assert schedule in (128, 1024)
    return dict(context=4096, teacher_chunk=schedule, max_rows=1024, variant='26',
                policy='all', normmul='normmul-on', all_outputs=True,
                max_head_rows=0, slots=1, local_cells=2048, global_cells=4096)


def manifest(path, expected_sha):
    result = authenticated(path, expected_sha)
    quality.current_input(path, expected_sha, '26')
    assert result['history_index'] in (13, 14) and result['mode'] == 'literal'
    return result


def own(root, expected_sha, schedule):
    frozen = authenticated(root / 'current-native-repeat-frozen.json', expected_sha)
    assert frozen['native_recipe'] == recipe(schedule)
    assert frozen['rows'] == ROWS and frozen['vocab'] == VOCAB
    assert frozen['byte_exact_rows'] == ROWS and frozen['p99_top2_margin_movement'] == 0
    assert frozen['whole_file_sha256'][0] == frozen['whole_file_sha256'][1]
    source = authenticated(pathlib.Path(frozen['source_identities_path']),
                           frozen['source_identities_sha256'])
    assert source == frozen['source'] and source['native_recipe'] == recipe(schedule)
    return frozen


def nearest_p99(moves):
    assert len(moves) == ROWS and all(math.isfinite(x) and x >= 0 for x in moves)
    return sorted(moves)[math.ceil(.99 * ROWS) - 1]


def calibrate(a_root, a_sha, b_root, b_sha, output):
    assert not any((root / arm).exists() for root in (a_root, b_root)
                   for arm in ('reference-first', 'reference-repeat')), 'oracle released before calibration'
    a, b = own(a_root, a_sha, 1024), own(b_root, b_sha, 128)
    # Each source receipt includes the externally frozen history manifest.
    assert a['input_manifest_sha256'] == b['input_manifest_sha256']
    assert a['input_sha256'] == b['input_sha256']
    aa, bb = dict(a['source']), dict(b['source'])
    aa.pop('native_recipe'); bb.pop('native_recipe')
    assert aa == bb, 'only teacher schedule may change'
    history_path = pathlib.Path(a['source']['input_manifest_path'])
    history = manifest(history_path, a['input_manifest_sha256'])
    assert history['history_index'] == 13
    assert history['input_sha256'] == a['input_sha256']
    quality.current_ids(a_root, a['input_sha256'], True)
    quality.current_ids(b_root, b['input_sha256'], True)
    paths = [root / 'native-first/logits.f32' for root in (a_root, b_root)]
    assert all(p.stat().st_size == ROWS * VOCAB * 4 for p in paths)
    moves, hashes = [], [hashlib.sha256(), hashlib.sha256()]
    choices_differ = exact = 0
    with paths[0].open('rb') as af, paths[1].open('rb') as bf:
        for index in range(ROWS):
            av, ah = quality.read_row(af); bv, bh = quality.read_row(bf)
            assert ah == a['row_sha256'][index] and bh == b['row_sha256'][index]
            for digest, values in zip(hashes, (av, bv)):
                if sys.byteorder != 'little':
                    values.byteswap()
                digest.update(values.tobytes())
                if sys.byteorder != 'little':
                    values.byteswap()
            first, second = heapq.nlargest(2, range(VOCAB), key=av.__getitem__)
            moves.append(abs((av[first]-av[second])-(bv[first]-bv[second])))
            choices_differ += first != max(range(VOCAB), key=bv.__getitem__)
            exact += ah == bh
            if index % 128 == 0:
                print(f'SCHEDULE_CALIBRATION completed_rows={index+1}', flush=True)
        assert not af.read(1) and not bf.read(1)
    assert [h.hexdigest() for h in hashes] == [a['whole_file_sha256'][0], b['whole_file_sha256'][0]]
    result = dict(schema=1, method=METHOD, calibration_history=13, heldout_history=14,
                  profile='26', rows=ROWS, input_sha256=a['input_sha256'],
                  input_manifest_sha256=a['input_manifest_sha256'],
                  text_manifest_sha256=history['text_manifest_sha256'],
                  production_recipe=recipe(1024), supported_recipe=recipe(128),
                  production_own_sha256=a_sha, supported_own_sha256=b_sha,
                  source_common=aa, whole_file_sha256=[h.hexdigest() for h in hashes],
                  byte_exact_rows=exact, native_schedule_choice_differences=choices_differ,
                  p99_top2_margin_movement=nearest_p99(moves),
                  max_top2_margin_movement=max(moves),
                  oracle_read=False, own_repeat_allowance=0,
                  transfer='unchanged history13 bound on EVERY heldout14 strict oracle difference; no exceptions')
    with output.open('x') as file:
        json.dump(result, file, indent=2); file.write('\n')
    print('bound_sha256=' + hashlib.sha256(output.read_bytes()).hexdigest())


def oracle(root, own_sha, history_path, history_sha, bound_path, bound_sha):
    bound = authenticated(bound_path, bound_sha)
    assert bound['schema'] == 1 and bound['method'] == METHOD and bound['oracle_read'] is False
    assert bound['calibration_history'] == 13 and bound['heldout_history'] == 14
    assert bound['profile'] == '26' and bound['rows'] == ROWS
    assert bound['production_recipe'] == recipe(1024) and bound['supported_recipe'] == recipe(128)
    p99 = bound['p99_top2_margin_movement']
    assert math.isfinite(p99) and p99 >= 0
    history = manifest(history_path, history_sha)
    assert history['history_index'] == 14
    assert history['text_manifest_sha256'] == bound['text_manifest_sha256']
    frozen = own(root, own_sha, 1024)
    assert frozen['input_manifest_sha256'] == history_sha
    assert frozen['input_sha256'] == history['input_sha256'] != bound['input_sha256']
    common = dict(frozen['source']); common.pop('native_recipe')
    # Input identities/paths differ on the heldout; all code/options/artifacts
    # must remain exactly those used for calibration.
    for key in ('input_manifest_path', 'input_manifest_sha256', 'input_sha256'):
        common.pop(key)
    expected = dict(bound['source_common'])
    for key in ('input_manifest_path', 'input_manifest_sha256', 'input_sha256'):
        expected.pop(key)
    assert common == expected
    quality.current_oracle(root, own_sha, history_path, history_sha,
                           dict(bound_sha256=bound_sha, calibration_history=13,
                                heldout_history=14, method=METHOD,
                                p99_top2_margin_movement=p99))


def main():
    args = sys.argv[1:]
    if len(args) == 7 and args[0] == 'freeze':
        root, schedule = pathlib.Path(args[1]), int(args[2])
        history = manifest(pathlib.Path(args[5]), args[6])
        assert history['history_index'] == 13 or schedule == 1024
        source = authenticated(pathlib.Path(args[3]), args[4])
        assert source['input_manifest_path'] == str(pathlib.Path(args[5]).resolve())
        assert source['input_manifest_sha256'] == args[6]
        assert source['input_sha256'] == history['input_sha256']
        quality.current_freeze(root, recipe(schedule), pathlib.Path(args[3]), args[4],
                               pathlib.Path(args[5]), args[6])
    elif len(args) == 6 and args[0] == 'calibrate':
        calibrate(pathlib.Path(args[1]), args[2], pathlib.Path(args[3]), args[4], pathlib.Path(args[5]))
    elif len(args) == 7 and args[0] == 'oracle':
        oracle(pathlib.Path(args[1]), args[2], pathlib.Path(args[3]), args[4],
               pathlib.Path(args[5]), args[6])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
