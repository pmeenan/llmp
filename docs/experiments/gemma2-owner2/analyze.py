#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict two-owner, independently-prefilled C2 screen; all output is external."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB, ROWS, STEPS = 256000, 66, 64
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
        assert len(raw) == 4387 * 4 and digest(path) == record[str(slot)]['ids_sha256']
        assert digest(root / f'prompt{slot}.txt') == record[str(slot)]['text_sha256']
        ids = struct.unpack('<4387i', raw)
        assert ids[0] == 2 and all(0 <= n < VOCAB for n in ids)
        result.append(ids)
    assert result[0] != result[1]
    return result


def own(root, ring=False):
    inputs(root)
    arms = ['native-ring1', 'native-ring2'] if ring else ['native-own1', 'native-own2']
    hashes = {}
    for name in ('heads.f32', 'chosen.i32', 'final.f32', 'state.json'):
        paths = [root / arm / name for arm in arms]
        assert digest(paths[0]) == digest(paths[1]), f'native repeat changed {name}'
        hashes[name] = digest(paths[0])
    path = root / arms[0]
    assert (path / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
    chosen = struct.unpack('<64i', (path / 'chosen.i32').read_bytes())
    with (path / 'heads.f32').open('rb') as heads:
        last = []
        for index in range(ROWS):
            row, raw_hash = core.read_row(heads)
            if index < STEPS:
                assert max(range(VOCAB), key=row.__getitem__) == chosen[index]
            else:
                last.extend(row)
        assert not heads.read(1)
    assert struct.pack(f'<{VOCAB * 2}f', *last) == (path / 'final.f32').read_bytes()
    state = json.loads((path / 'state.json').read_text())
    assert all(state[k] is True for k in ('gpu_equal', 'restore_equal', 'refusals_unchanged'))
    assert state['slot0'] != state['slot1'] and all(len(state[k]) == 64 for k in ('slot0', 'slot1'))
    if not ring:
        for name in ('heads.f32', 'chosen.i32', 'final.f32', 'state.json'):
            assert digest(root/'native-eager'/name) == hashes[name], f'eager/capture changed {name}'
    fallback = None
    if not ring:
        p = root/'native-ordinary'
        fallback_choices=struct.unpack('<64i',(p/'chosen.i32').read_bytes())
        assert all(0 <= n < VOCAB for n in fallback_choices)
        with (p/'heads.f32').open('rb') as f:
            last=[]
            for index in range(ROWS):
                row,_=core.read_row(f)
                if index < STEPS:
                    assert max(range(VOCAB),key=row.__getitem__)==fallback_choices[index]
                else: last.extend(row)
            assert not f.read(1)
        assert struct.pack(f'<{VOCAB*2}f',*last)==(p/'final.f32').read_bytes()
        fallback_state=json.loads((p/'state.json').read_text())
        assert all(fallback_state[k] is True for k in ('gpu_equal','restore_equal','refusals_unchanged'))
        fallback=dict(hashes={name:digest(p/name) for name in ('heads.f32','chosen.i32','final.f32','state.json')},
                      scope='ordinary attention own controls; no cross-policy equality gate')
    save(root, 'ring-own-control.json' if ring else 'own-control.json', dict(fallback=fallback,scope='two initialized slots, joined C2 decode', hashes=hashes,
                                      rows=ROWS, choices_checked=STEPS, state=state))


def quality(root, ring=False):
    ids = inputs(root)
    own_record = json.loads((root / ('ring-own-control.json' if ring else 'own-control.json')).read_text())
    names = ('native-ring1', 'stock-ring1', 'stock-ring2') if ring else ('native-own1', 'stock-teacher1', 'stock-teacher2')
    paths = [root / arm for arm in names]
    assert digest(paths[0] / 'heads.f32') == own_record['hashes']['heads.f32']
    for name in ('heads.f32', 'chosen.i32', 'final.f32'):
        assert digest(paths[1] / name) == digest(paths[2] / name), f'stock repeat changed {name}'
    for arm in names[1:]:
        retired=json.loads((root/f'{arm}-container-retired.json').read_text())
        assert retired['container_absent_after_checked_docker_query'] is True
    assert all((p / 'heads.f32').stat().st_size == ROWS * VOCAB * 4 for p in paths)
    chosen = [struct.unpack('<64i', (p / 'chosen.i32').read_bytes()) for p in paths]
    misses, deltas, tvs = [], [], []
    exact, max_raw = 0, 0.0
    with (paths[0] / 'heads.f32').open('rb') as native, (paths[1] / 'heads.f32').open('rb') as stock:
        for index in range(ROWS):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            exact += ah == bh
            if index < STEPS:
                assert ni == chosen[0][index] and ri == chosen[1][index] == chosen[2][index]
            if ni != ri:
                misses.append(dict(row=index, step=index // 2, slot=index % 2,
                                   native=ni, stock=ri, stock_margin=margin))
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            tvs.append(.5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)))
            max_raw = max(max_raw, max(abs(x - y) for x, y in zip(a, b)))
            if index < STEPS:
                target = ids[index % 2][(4355 if ring else 259) + index // 2]
                deltas.append((al - a[target]) - (bl - b[target]))
        assert not native.read(1) and not stock.read(1)
    positive = sum(m['stock_margin'] > 0 for m in misses)
    nll_delta = math.fsum(deltas) / STEPS
    relative = math.expm1(nll_delta)
    result = dict(scope='independently-prefilled C2 decode, 32 steps per slot', rows=ROWS,
                  scored_targets=STEPS, final_rows_scored=False, exact_head_rows=exact,
                  heads_sha256=[digest(p / 'heads.f32') for p in paths],
                  strict_greedy_differences=len(misses), positive_margin_differences=positive,
                  exact_tie_differences=sum(m['stock_margin'] == 0 for m in misses), misses=misses,
                  mean_target_nll_delta=nll_delta, relative_conditional_loss_delta=relative,
                  mean_total_variation=math.fsum(tvs) / ROWS, max_raw_logit_delta=max_raw,
                  passed=positive == 0 and relative <= .03,
                  stock_sampled_logits='full F32 vocabulary rows also exported by original backend sampler')
    result['prefix_rows_per_owner'] = 4352 if ring else 256
    save(root, 'ring-quality.json' if ring else 'quality.json', result)
    assert result['passed'], 'strict C2 quality gate failed; timing must not run'


def cycle(root, log_path):
    inputs(root)
    assert json.loads((root / 'quality.json').read_text())['passed']
    assert json.loads((root / 'ring-quality.json').read_text())['passed']
    text = log_path.read_text()
    order = re.findall(r'step \d+/\d+ start: .*?(stock-cycle[12]|native-cycle[12])', text)
    assert order == ['stock-cycle1', 'native-cycle1', 'native-cycle2', 'stock-cycle2'], order
    result = []
    for arm in order:
        stock = arm.startswith('stock')
        pattern = r'GEMMA2_C2_STOCK mode=cycle (.*)' if stock else r'GEMMA2_C2 mode=cycle (.*)'
        # Per-arm log is bounded by the installed queue's step boundary.
        sections = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
        matching = [s for s in sections[1:] if arm in s.split('\n', 1)[0]]
        assert len(matching) == 1, (arm, len(matching))
        section = matching[0]
        found = re.search(pattern, section)
        assert found, arm
        fields = dict(re.findall(r'(\w+)=([^\s]+)', found.group(1)))
        for k, v in dict(slots='2', context_per_slot='8192', chunk='128', independent_prefill='1',
                         prompt_rows_per_slot='256', untimed_rows_per_slot='3', decode_steps='32',
                         paid_generated_tokens='64', past0='291', past1='291').items():
            assert fields[k] == v, (arm, k)
        if stock:
            retired=json.loads((root/f'{arm}-container-retired.json').read_text())
            assert retired['container_absent_after_checked_docker_query'] is True
            assert 'GEMMA2_C2_STOCK_RETIRED' in section
            assert fields['backend_tokens'] == '96' and fields['observer'] == '0'
            assert fields['sampled_logits_min'] == fields['sampled_logits_max'] == str(VOCAB)
            assert fields['final_head_paid'] == '1' and fields['tokenized_equal'] == '2'
        else:
            assert 'GEMMA2_C2_RETIRED' in section
            assert fields['gpu_tokens'] == '88' and int(fields['selected_owner']) > 0
            assert int(fields['captured']) > 0 and int(fields['replayed']) > 0
        prefill, decode = float(fields['prefill_seconds']), float(fields['decode_seconds'])
        assert all(math.isfinite(v) and v > 0 for v in (prefill,decode,prefill+decode))
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
    save(root, 'cycle.json', dict(scope='short independently-prefilled C2 RNNR; final heads paid',
                                 arms=result, native_paid_seconds=native, stock_paid_seconds=stock,
                                 native_latency_over_stock=native / stock - 1,
                                 native_generated_tokens_per_paid_second=64 / native,
                                 stock_generated_tokens_per_paid_second=64 / stock))


if __name__ == '__main__':
    root = Path(sys.argv[2])
    if sys.argv[1] == 'own': own(root)
    elif sys.argv[1] == 'ring-own': own(root, True)
    elif sys.argv[1] == 'ring-quality': quality(root, True)
    elif sys.argv[1] == 'quality': quality(root)
    elif sys.argv[1] == 'cycle': cycle(root, Path(sys.argv[3]))
    else: raise ValueError('unknown analysis mode')
