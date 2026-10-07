#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed C1 context8448/prefix8192/64-target controls; preserves historical screens."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB, PREFIX, STEPS, ROWS = 262208, 8192, 64, 65
spec = importlib.util.spec_from_file_location(
    'gemma_quality_core', Path(__file__).resolve().parents[1] / 'gemma-quality/analyze.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.VOCAB = VOCAB


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(root, name, result):
    with (root / name).open('x') as target:
        json.dump(result, target, indent=2)
        target.write('\n')
    print(json.dumps(result), flush=True)


def choices(path):
    raw = path.read_bytes()
    assert len(raw) == STEPS * 4
    result = struct.unpack('<64i', raw)
    assert all(0 <= value < VOCAB for value in result)
    return result


def finite(path, rows):
    assert path.stat().st_size == rows * VOCAB * 4
    with path.open('rb') as source:
        for _ in range(rows):
            core.read_row(source)
        assert not source.read(1)
    return digest(path)


def own(root):
    records = [json.loads((root / arm / 'state.json').read_text())
               for arm in ('native-own1', 'native-own2')]
    for record in records:
        assert record['reference'] == record['device'] and record['choices_checked'] == STEPS
        assert record['refusals_unchanged'] and record['allocated_slots'] == 2
        assert record['executed_slots'] == 1 and record['kept_extents'] > 0
    assert records[0] == records[1]
    result = dict(scope='C1 executed; two allocated own-control slots; no C2 depth claim',
                  heads={}, chosen={}, final={}, final_state_sha256=records[0]['device'])
    for arm in ('native-own1', 'native-own2'):
        path = root / arm
        result['heads'][arm] = finite(path / 'heads.f32', ROWS)
        result['final'][arm] = finite(path / 'final.f32', 1)
        choices(path / 'chosen.i32')
        result['chosen'][arm] = digest(path / 'chosen.i32')
    for key in ('heads', 'chosen', 'final'):
        assert len(set(result[key].values())) == 1, 'own repeat differs: ' + key
    lifetime = root / 'native-lifetime'
    record = json.loads((lifetime / 'lifetime.json').read_text())
    final_state = json.loads((lifetime / 'final-state.json').read_text())
    assert record['positions'] == PREFIX and record['state_only_equal']
    assert record['restored_equal'] and record['refusals_unchanged']
    assert final_state['positions'] == PREFIX + STEPS
    assert final_state['initialized_state_sha256'] == records[0]['reference']
    assert finite(lifetime / 'final.f32', 1) == result['final']['native-own1']
    assert digest(lifetime / 'chosen.i32') == result['chosen']['native-own1']
    result['prefix_state_sha256'] = record['initialized_state_sha256']
    result['lifetime'] = record
    save(root, 'own-control.json', result)


def quality(root):
    own_record = json.loads((root / 'own-control.json').read_text())
    lineage = json.loads((root / 'input.json').read_text())
    raw = (root / 'input/ids.i32').read_bytes()
    assert len(raw) == (PREFIX + STEPS) * 4 and digest(root / 'input/ids.i32') == lineage['ids_sha256']
    ids = struct.unpack('<8256i', raw)
    paths = [root / arm / 'heads.f32' for arm in
             ('native-own1', 'stock-teacher1', 'stock-teacher2')]
    hashes = [finite(path, ROWS) for path in paths]
    assert hashes[0] == own_record['heads']['native-own1'] and hashes[1] == hashes[2]
    selected = [choices(path.parent / 'chosen.i32') for path in paths]
    assert digest(paths[1].parent / 'final.f32') == digest(paths[2].parent / 'final.f32')
    nll_a, nll_b, tv, misses = [], [], [], []
    exact, maximum = 0, 0.0
    with paths[0].open('rb') as native, paths[1].open('rb') as stock:
        for row in range(ROWS):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            if row < STEPS:
                assert ni == selected[0][row] and ri == selected[1][row] == selected[2][row]
            exact += ah == bh
            if ni != ri:
                misses.append(dict(row=row, native=ni, stock=ri, stock_margin=margin))
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
    result = dict(scope='fixed scalar depth screen; no numerical allowance or full PPL claim',
                  context=8448, prefix_rows=PREFIX, scored_targets=STEPS, full_head_rows=ROWS,
                  final_row_scored=False, ids_sha256=lineage['ids_sha256'], full_head_sha256=hashes,
                  exact_head_rows=exact, strict_greedy_differences=len(misses),
                  positive_margin_differences=positive,
                  exact_tie_differences=sum(m['stock_margin'] == 0 for m in misses), misses=misses,
                  native_mean_target_nll=math.fsum(nll_a) / STEPS,
                  stock_mean_target_nll=math.fsum(nll_b) / STEPS,
                  mean_target_nll_delta=delta, relative_conditional_loss=relative,
                  mean_total_variation=math.fsum(tv) / ROWS, max_total_variation=max(tv),
                  max_raw_logit_delta=maximum, passed=positive == 0 and relative <= .03)
    save(root, 'quality.json', result)
    assert result['passed'], 'strict depth quality gate failed; timing stays blocked'


def cycle(root):
    assert json.loads((root / 'quality.json').read_text())['passed']
    arms = ('stock-cycle1', 'native-cycle1', 'native-cycle2', 'stock-cycle2')
    selected, finals, measurements, memory = [], [], [], []
    for arm in arms:
        path = root / arm
        choices(path / 'chosen.i32')
        selected.append(digest(path / 'chosen.i32'))
        finals.append(finite(path / 'final.f32', 1))
        log = (root / (arm + '.log')).read_text()
        kind = 'STOCK' if arm.startswith('stock') else 'PROBE'
        match = re.findall(r'^GEMMA3_' + kind + r' mode=greedy-cycle ([^\n]+)', log, re.M)
        assert len(match) == 1
        fields = dict(item.split('=', 1) for item in match[0].split())
        assert all(fields[key] == value for key, value in
                   dict(context='8448', slots='1', chunk='128', prompt_rows='8192',
                        untimed_rows='0', decode_rows='64', past='8256').items())
        if kind == 'STOCK':
            assert fields['backend_tokens'] == '74' and fields['final_head_paid'] == '1'
            assert fields['sampled_logits_min'] == fields['sampled_logits_max'] == str(VOCAB)
        else:
            assert fields['gpu_tokens'] == '70' and int(fields['captured']) > 0
            assert int(fields['replayed']) > 0 and fields['coverage_violations'] == '0'
        prefill, decode = (float(fields[k]) for k in ('prefill_seconds', 'decode_seconds'))
        assert all(math.isfinite(value) and value > 0 for value in (prefill, decode))
        measurements.append(dict(arm=arm, prefill_seconds=prefill, decode_seconds=decode,
                                 paid_seconds=prefill + decode))
        memory.append(json.loads((root / (arm + '-memory.json')).read_text()))
    assert len(set(selected)) == 1, 'natural histories diverged'
    assert len(set(finals)) == 1, 'natural final heads differ'
    result = dict(scope='short C1 8192+64 RNNR; original stock sampled-logit transfers retained; '
                        'one final full head paid in both arms',
                  observations_per_engine=2, measurements=measurements, memory=memory,
                  chosen_sha256=selected, final_sha256=finals)
    for phase in ('prefill_seconds', 'decode_seconds', 'paid_seconds'):
        stock = (measurements[0][phase] + measurements[3][phase]) / 2
        native = (measurements[1][phase] + measurements[2][phase]) / 2
        result[phase] = dict(native=native, stock=stock, native_over_stock=native / stock - 1)
    save(root, 'cycle.json', result)


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] in ('own', 'quality', 'cycle')
    globals()[sys.argv[1]](Path(sys.argv[2]))
