#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One real-corpus C1 trained-maximum teacher screen; no retrieval or full PPL claim."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

VOCAB, PREFIX, STEPS, ROWS = 262208, 131008, 64, 65
spec = importlib.util.spec_from_file_location(
    'gemma_quality_core', Path(__file__).resolve().parents[1] / 'gemma-quality/analyze.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.VOCAB = VOCAB


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(root, name, value):
    with (root / name).open('x') as target:
        json.dump(value, target, indent=2)
        target.write('\n')
    print(json.dumps(value), flush=True)


def choices(path):
    raw = path.read_bytes()
    assert len(raw) == STEPS * 4
    values = struct.unpack('<64i', raw)
    assert all(0 <= value < VOCAB for value in values)
    return values


def check(path):
    assert (path / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
    selected = choices(path / 'chosen.i32')
    with (path / 'heads.f32').open('rb') as source:
        for row in range(ROWS):
            values, raw_hash = core.read_row(source)
            if row < STEPS:
                assert max(range(VOCAB), key=values.__getitem__) == selected[row]
        assert not source.read(1)
    assert digest(path / 'final.f32') == raw_hash
    return {name: digest(path / name) for name in ('heads.f32', 'chosen.i32', 'final.f32')}


def own(root):
    lineage = json.loads((root / 'input.json').read_text())
    values = [check(root / arm) for arm in ('native-own1', 'native-own2')]
    assert values[0] == values[1], 'same-binary maximum own repeat differs'
    for arm in ('native-own1', 'native-own2'):
        record = json.loads((root / arm / 'result.json').read_text())
        for key, value in dict(context=131072, prefix=PREFIX, teacher_writes=STEPS,
                               full_heads=ROWS, scored_transitions=STEPS,
                               initialized_positions=131072, natural_emitted_tokens=0,
                               initialized_history_sha256=lineage['ids_sha256'],
                               prefill_chunks=1024).items():
            assert record[key] == value, (arm, key)
        assert record['graphs']['captured'] > 0 and record['graphs']['replayed'] > 0
        assert record['graphs']['refused'] == 0
        assert record['model']['context'] == 131072 and record['model']['configured_slots'] == 1
        assert record['model']['max_rows'] == 128 and record['model']['gpu_greedy_tokens'] == 0
    save(root, 'own-control.json', dict(
        scope='same-binary finite full-head/choice/history repeat; initialized cursor, not KV-byte snapshot',
        full_heads=ROWS, teacher_writes=STEPS, initialized_positions=131072,
        ids_sha256=lineage['ids_sha256'], payload_hashes=values[0]))


def quality(root):
    own_record = json.loads((root / 'own-control.json').read_text())
    lineage = json.loads((root / 'input.json').read_text())
    assert digest(root / 'input/ids.i32') == lineage['ids_sha256'] == own_record['ids_sha256']
    raw = (root / 'input/ids.i32').read_bytes()
    assert len(raw) == 131072 * 4
    ids = struct.unpack('<131072i', raw)
    paths = [root / arm for arm in ('native-own1', 'stock-teacher1', 'stock-teacher2')]
    hashes = [check(path) for path in paths]
    assert hashes[0] == own_record['payload_hashes'] and hashes[1] == hashes[2]
    selected = [choices(path / 'chosen.i32') for path in paths]
    nll_a, nll_b, tv, misses = [], [], [], []
    exact, maximum, final_choice_difference = 0, 0.0, False
    with (paths[0] / 'heads.f32').open('rb') as native, (paths[1] / 'heads.f32').open('rb') as stock:
        for row in range(ROWS):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            exact += ah == bh
            if row < STEPS:
                assert ni == selected[0][row] and ri == selected[1][row] == selected[2][row]
            else:
                final_choice_difference = ni != ri
            if ni != ri:
                misses.append(dict(row=row, native=ni, stock=ri, stock_margin=margin,
                                   scored=row < STEPS))
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            tv.append(.5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)))
            maximum = max(maximum, max(abs(x - y) for x, y in zip(a, b)))
            if row < STEPS:
                target = ids[PREFIX + row]
                nll_a.append(al - a[target])
                nll_b.append(bl - b[target])
    delta = math.fsum(a - b for a, b in zip(nll_a, nll_b)) / STEPS
    positive = sum(m['stock_margin'] > 0 for m in misses)
    relative = math.expm1(delta)
    result = dict(scope='one real-corpus tail at trained maximum; no retrieval, full PPL or natural-generation claim',
                  context=131072, prefix_rows=PREFIX, scored_targets=STEPS, full_head_rows=ROWS,
                  initialized_positions=131072, final_row_scored=False,
                  ids_sha256=lineage['ids_sha256'], payload_hashes=hashes,
                  exact_head_rows=exact, strict_greedy_differences=len(misses),
                  scored_greedy_differences=sum(m['scored'] for m in misses),
                  positive_margin_differences=positive,
                  exact_tie_differences=sum(m['stock_margin'] == 0 for m in misses), misses=misses,
                  final_unscored_choice_difference=final_choice_difference,
                  native_mean_target_nll=math.fsum(nll_a) / STEPS,
                  stock_mean_target_nll=math.fsum(nll_b) / STEPS,
                  mean_target_nll_delta=delta, relative_conditional_loss=relative,
                  mean_total_variation=math.fsum(tv) / ROWS, max_total_variation=max(tv),
                  max_raw_logit_delta=maximum, passed=len(misses) == 0 and relative <= .03)
    save(root, 'quality.json', result)
    assert result['passed'], 'strict trained-maximum quality gate failed'


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] in ('own', 'quality')
    globals()[sys.argv[1]](Path(sys.argv[2]))
