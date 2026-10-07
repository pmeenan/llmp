#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict compatible-prefill/ring C2 screen, including partial departure; all output is external."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB, ROWS, STEPS = 256000, 74, 72
spec = importlib.util.spec_from_file_location(
    'gemma_quality_core', Path(__file__).resolve().parents[1] / 'gemma-quality/analyze.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.VOCAB = VOCAB


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(root, name, result):
    with (root / name).open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result), flush=True)


def inputs(root):
    record = json.loads((root / 'inputs.json').read_text())
    result = []
    for slot in range(2):
        path = root / f'input{slot}/ids.i32'
        raw = path.read_bytes()
        assert len(raw) == record[str(slot)]['rows'] * 4 and digest(path) == record[str(slot)]['ids_sha256']
        assert digest(root / f'prompt{slot}.txt') == record[str(slot)]['text_sha256']
        ids = struct.unpack(f'<{len(raw) // 4}i', raw)
        assert ids[0] == 2 and all(0 <= n < VOCAB for n in ids)
        result.append(ids)
    assert result[0] != result[1]
    return result


def row_map(ids):
    prefix = [len(row) - 39 for row in ids]
    rows = [(s, prefix[s] + 3 + i) for i in range(33) for s in range(2)]
    rows += [(s, prefix[s] + 36 + i) for s in range(2) for i in range(4)]
    choices = {i: i for i in range(64)}
    choices.update({64: 64, 65: 68, 66: 65, 67: 66, 68: 67, 70: 69, 71: 70, 72: 71})
    assert len(rows) == ROWS and len(choices) == STEPS
    return rows, choices


def prefill_geometry(ids):
    prefix = [len(row) - 39 for row in ids]
    past = [0, 0]
    groups = rows = 0
    while past != prefix:
        active = [slot for slot in range(2) if past[slot] < prefix[slot]]
        first = min(active, key=lambda slot: (prefix[slot] - past[slot], slot))
        head = past[first] + min(128, prefix[first] - past[first]) == prefix[first]
        joined = [slot for slot in active if (past[slot] + min(128, prefix[slot] - past[slot]) == prefix[slot]) == head]
        wave_rows = sum(min(128, prefix[slot] - past[slot]) for slot in joined)
        if len(joined) == 2:
            groups += 1
            rows += wave_rows
        for slot in joined:
            past[slot] += min(128, prefix[slot] - past[slot])
    return groups, rows


def own(root):
    ids = inputs(root)
    mapping, decisions = row_map(ids)
    arms = ['native-own1', 'native-own2']
    hashes = {}
    for name in ('prefill.f32', 'heads.f32', 'chosen.i32', 'final.f32', 'state.json'):
        paths = [root / arm / name for arm in arms]
        assert digest(paths[0]) == digest(paths[1]), f'native repeat changed {name}'
        hashes[name] = digest(paths[0])
    path = root / arms[0]
    assert (path / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
    assert (path / 'prefill.f32').stat().st_size == 2 * VOCAB * 4
    with (path / 'prefill.f32').open('rb') as frontier:
        for _ in range(2): core.read_row(frontier)
        assert not frontier.read(1)
    chosen = struct.unpack('<72i', (path / 'chosen.i32').read_bytes())
    with (path / 'heads.f32').open('rb') as heads:
        last = {}
        for index in range(ROWS):
            row, raw_hash = core.read_row(heads)
            if index in decisions:
                assert max(range(VOCAB), key=row.__getitem__) == chosen[decisions[index]]
            last[mapping[index][0]] = row
        assert not heads.read(1)
    assert struct.pack(f'<{VOCAB * 2}f', *(last[0] + last[1])) == (path / 'final.f32').read_bytes()
    state = json.loads((path / 'state.json').read_text())
    assert all(state[k] is True for k in ('gpu_equal', 'restore_equal', 'refusals_unchanged'))
    assert state['slot0'] != state['slot1'] and all(len(state[k]) == 64 for k in ('slot0', 'slot1'))
    groups, joined_rows = prefill_geometry(ids)
    assert state['joined_prefill_groups'] == groups * 2 and state['joined_prefill_rows'] == joined_rows * 2
    assert groups > 0
    save(root, 'own-control.json', dict(scope='two initialized slots, compatible prefill, joined decode and partial departure', hashes=hashes,
                                      rows=ROWS, choices_checked=STEPS, state=state))


def quality(root):
    ids = inputs(root)
    mapping, decisions = row_map(ids)
    own_record = json.loads((root / 'own-control.json').read_text())
    paths = [root / arm for arm in ('native-own1', 'stock-teacher1', 'stock-teacher2')]
    assert digest(paths[0] / 'heads.f32') == own_record['hashes']['heads.f32']
    for name in ('prefill.f32', 'heads.f32', 'chosen.i32', 'final.f32'):
        assert digest(paths[1] / name) == digest(paths[2] / name), f'stock repeat changed {name}'
    assert all((p / 'heads.f32').stat().st_size == ROWS * VOCAB * 4 for p in paths)
    chosen = [struct.unpack('<72i', (p / 'chosen.i32').read_bytes()) for p in paths]
    misses, deltas, tvs = [], [], []
    exact, max_raw = 0, 0.0
    with (paths[0] / 'heads.f32').open('rb') as native, (paths[1] / 'heads.f32').open('rb') as stock:
        for index in range(ROWS):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            exact += ah == bh
            if index in decisions:
                at = decisions[index]
                assert ni == chosen[0][at] and ri == chosen[1][at] == chosen[2][at]
            if ni != ri:
                misses.append(dict(row=index, position=mapping[index][1], slot=mapping[index][0],
                                   native=ni, stock=ri, stock_margin=margin))
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            tvs.append(.5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)))
            max_raw = max(max_raw, max(abs(x - y) for x, y in zip(a, b)))
            if index in decisions:
                slot, position = mapping[index]
                target = ids[slot][position]
                deltas.append((al - a[target]) - (bl - b[target]))
        assert not native.read(1) and not stock.read(1)
    frontier_misses, frontier_deltas, frontier_exact = [], [], 0
    assert digest(paths[0] / 'prefill.f32') == own_record['hashes']['prefill.f32']
    assert all((p / 'prefill.f32').stat().st_size == 2 * VOCAB * 4 for p in paths)
    with (paths[0] / 'prefill.f32').open('rb') as native, (paths[1] / 'prefill.f32').open('rb') as stock:
        for slot in range(2):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            frontier_exact += ah == bh
            if ni != ri:
                frontier_misses.append(dict(slot=slot, native=ni, stock=ri, stock_margin=margin))
            _, _, al = core.distribution(a)
            _, _, bl = core.distribution(b)
            target = ids[slot][len(ids[slot]) - 39]
            frontier_deltas.append((al - a[target]) - (bl - b[target]))
        assert not native.read(1) and not stock.read(1)
    frontier_relative = math.expm1(math.fsum(frontier_deltas) / 2)
    positive = sum(m['stock_margin'] > 0 for m in misses)
    frontier_positive = sum(m['stock_margin'] > 0 for m in frontier_misses)
    nll_delta = math.fsum(deltas) / STEPS
    relative = math.expm1(nll_delta)
    result = dict(scope='compatible prefill, 32 joined steps then four departure steps per slot', rows=ROWS,
                  scored_targets=STEPS, final_rows_scored=False, exact_head_rows=exact,
                  heads_sha256=[digest(p / 'heads.f32') for p in paths],
                  strict_greedy_differences=len(misses), positive_margin_differences=positive,
                  exact_tie_differences=sum(m['stock_margin'] == 0 for m in misses), misses=misses,
                  mean_target_nll_delta=nll_delta, relative_conditional_loss_delta=relative,
                  mean_total_variation=math.fsum(tvs) / ROWS, max_raw_logit_delta=max_raw,
                  prefill_heads=2, prefill_scored_targets=2, prefill_exact_heads=frontier_exact,
                  prefill_misses=frontier_misses, prefill_relative_conditional_loss_delta=frontier_relative,
                  prefill_positive_margin_differences=frontier_positive,
                  passed=positive == 0 and frontier_positive == 0 and relative <= .03 and frontier_relative <= .03)
    save(root, 'quality.json', result)
    assert result['passed'], 'strict C2 quality gate failed; timing must not run'


def cycle(root, log_path):
    ids = inputs(root)
    prefix = [len(row) - 39 for row in ids]
    assert json.loads((root / 'quality.json').read_text())['passed']
    text = log_path.read_text()
    order = re.findall(r'step \d+/\d+ start: .*?(stock-cycle[12]|native-cycle[12])', text)
    assert order == ['stock-cycle1', 'native-cycle1', 'native-cycle2', 'stock-cycle2'], order
    result = []
    for arm in order:
        stock = arm.startswith('stock')
        pattern = r'GEMMA2_JOINT_PREFILL_STOCK mode=cycle (.*)' if stock else r'GEMMA2_JOINT_PREFILL mode=cycle (.*)'
        # Per-arm log is bounded by the installed queue's step boundary.
        sections = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
        matching = [s for s in sections[1:] if arm in s.split('\n', 1)[0]]
        assert len(matching) == 1, (arm, len(matching))
        section = matching[0]
        found = re.search(pattern, section)
        assert found, arm
        fields = dict(re.findall(r'(\w+)=([^\s]+)', found.group(1)))
        for k, v in dict(slots='2', context_per_slot='8192', chunk='128', compatible_prefill='1', max_wave_rows='256',
                         prompt_rows0=str(prefix[0]), prompt_rows1=str(prefix[1]), untimed_rows_per_slot='3', decode_steps='32',
                         departure_steps='0', paid_generated_tokens='64', past0=str(prefix[0]+35), past1=str(prefix[1]+35)).items():
            assert fields[k] == v, (arm, k)
        groups, joined_rows = prefill_geometry(ids)
        assert int(fields['joined_prefill_groups']) == groups * 2
        assert int(fields['joined_prefill_rows']) == joined_rows * 2
        if stock:
            assert fields['backend_tokens'] == '96' and fields['observer'] == '0'
            assert fields['sampled_logits_min'] == fields['sampled_logits_max'] == str(VOCAB)
            assert fields['final_head_paid'] == '1' and fields['tokenized_equal'] == '2'
        else:
            assert fields['gpu_tokens'] == '88'
            if prefix[0] == prefix[1]: assert int(fields['selected_owner']) > 0
            assert int(fields['captured']) > 0 and int(fields['replayed']) > 0
        prefill, decode = float(fields['prefill_seconds']), float(fields['decode_seconds'])
        assert math.isfinite(prefill + decode) and prefill > 0 and decode > 0
        path = root / arm
        chosen = struct.unpack('<64i', (path / 'chosen.i32').read_bytes())
        assert all(0 <= n < VOCAB for n in chosen)
        with (path / 'final.f32').open('rb') as f:
            for _ in range(2): core.read_row(f)
            assert not f.read(1)
        result.append(dict(arm=arm, prefill_seconds=prefill, decode_seconds=decode,
                           paid_seconds=prefill + decode, choices=digest(path / 'chosen.i32'),
                           final=digest(path / 'final.f32'), fields=fields))
    assert len({a['choices'] for a in result}) == len({a['final'] for a in result}) == 1
    native = math.fsum(a['paid_seconds'] for a in result if a['arm'].startswith('native')) / 2
    stock = math.fsum(a['paid_seconds'] for a in result if a['arm'].startswith('stock')) / 2
    save(root, 'cycle.json', dict(scope='short compatibly-prefilled C2 RNNR; final heads paid',
                                 arms=result, native_paid_seconds=native, stock_paid_seconds=stock,
                                 native_latency_over_stock=native / stock - 1,
                                 native_generated_tokens_per_paid_second=64 / native,
                                 stock_generated_tokens_per_paid_second=64 / stock))


if __name__ == '__main__':
    root = Path(sys.argv[2])
    if sys.argv[1] == 'own': own(root)
    elif sys.argv[1] == 'quality': quality(root)
    elif sys.argv[1] == 'cycle': cycle(root, Path(sys.argv[3]))
    else: raise ValueError('unknown analysis mode')
