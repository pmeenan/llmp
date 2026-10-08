#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-root C4 screen; raw inputs, heads and state remain external."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB, OWNERS, ROWS, STEPS = 262208, 4, 148, 144
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
        assert len(raw) == record[str(slot)]['rows'] * 4 == 295 * 4
        assert digest(path) == record[str(slot)]['ids_sha256']
        assert digest(root / f'prompt{slot}.txt') == record[str(slot)]['text_sha256']
        ids = struct.unpack('<295i', raw)
        assert ids[0] == 2 and all(0 <= n < VOCAB for n in ids)
        result.append(ids)
    assert result[0] != result[1]
    return result * 2  # Four real owners, two explicitly duplicated input identities.


def row_map():
    rows = [(s, 259 + i) for i in range(33) for s in range(OWNERS)]
    rows += [(s, 292 + i) for s in range(OWNERS) for i in range(4)]
    choices = {i: i for i in range(128)}
    for s in range(OWNERS):
        choices[128 + s] = 128 + s * 4
        for i in range(3):
            choices[132 + s * 4 + i] = 128 + s * 4 + i + 1
    assert len(rows) == ROWS and len(choices) == STEPS
    return rows, choices


def validate(path, state=False):
    mapping, decisions = row_map()
    assert (path / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
    chosen = struct.unpack('<144i', (path / 'chosen.i32').read_bytes())
    last = {}
    with (path / 'heads.f32').open('rb') as heads:
        for index in range(ROWS):
            row, _ = core.read_row(heads)
            if index in decisions:
                assert max(range(VOCAB), key=row.__getitem__) == chosen[decisions[index]]
            last[mapping[index][0]] = row
        assert not heads.read(1)
    assert b''.join(struct.pack(f'<{VOCAB}f', *last[s]) for s in range(OWNERS)) == (path / 'final.f32').read_bytes()
    if state:
        record = json.loads((path / 'state.json').read_text())
        assert all(record[k] is True for k in ('gpu_equal', 'restore_equal', 'refusals_unchanged'))
        assert all(len(record[f'slot{s}']) == 64 for s in range(OWNERS))
        assert record['slot0'] != record['slot1']
        assert record['joined_prefill_groups'] == 8 and record['joined_prefill_rows'] == 2048
        return record


def compare(root, left, right):
    ids = inputs(root)
    mapping, decisions = row_map()
    a_path, b_path = (root / name for name in (left, right))
    exact, misses, deltas, tvs, max_raw = 0, [], [], [], 0.0
    with (a_path / 'heads.f32').open('rb') as fa, (b_path / 'heads.f32').open('rb') as fb:
        for index in range(ROWS):
            a, ah = core.read_row(fa)
            b, bh = core.read_row(fb)
            ni, ri, margin = core.current_choices(a, b)
            exact += ah == bh
            if ni != ri:
                misses.append(dict(row=index, position=mapping[index][1], slot=mapping[index][0], right_margin=margin))
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            tvs.append(.5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)))
            max_raw = max(max_raw, max(abs(x - y) for x, y in zip(a, b)))
            if index in decisions:
                slot, position = mapping[index]
                target = ids[slot][position]
                deltas.append((al - a[target]) - (bl - b[target]))
        assert not fa.read(1) and not fb.read(1)
    nll = math.fsum(deltas) / STEPS
    relative = math.expm1(nll)
    return dict(left=left, right=right, rows=ROWS, scored_targets=STEPS, final_rows_scored=False,
                exact_head_rows=exact, strict_greedy_differences=len(misses),
                positive_margin_differences=sum(m['right_margin'] > 0 for m in misses),
                exact_tie_differences=sum(m['right_margin'] == 0 for m in misses), differences=misses,
                mean_target_nll_delta=nll, relative_conditional_loss_delta=relative,
                mean_total_variation=math.fsum(tvs) / ROWS, max_raw_logit_delta=max_raw,
                heads_sha256=[digest(p / 'heads.f32') for p in (a_path, b_path)],
                passed=len(misses) == 0 and relative <= .03)


def footer(text, arm, stock=False, mode='own'):
    sections = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
    matching = [s for s in sections[1:] if arm in s.split('\n', 1)[0]]
    assert len(matching) == 1, (arm, len(matching))
    prefix = 'GEMMA3_C4_BATCH_STOCK' if stock else 'GEMMA3_C4_BATCH'
    found = re.search(prefix + ' mode=' + mode + r' (.*)', matching[0])
    assert found, arm
    assert prefix + '_RETIRED' in matching[0]
    return dict(re.findall(r'(\w+)=([^\s]+)', found.group(1)))


def own(root, log_path):
    inputs(root)
    hashes = {}
    for name in ('heads.f32', 'chosen.i32', 'final.f32', 'state.json'):
        a, b = [root / arm / name for arm in ('native-own1', 'native-own2')]
        assert digest(a) == digest(b), f'native repeat changed {name}'
        hashes[name] = digest(a)
    for name in ('heads.f32', 'chosen.i32', 'final.f32', 'state.json'):
        assert digest(root / 'native-eager' / name) == hashes[name], f'eager changed {name}'
    text = log_path.read_text()
    for arm in ('native-own1', 'native-own2', 'native-eager', 'native-solo'):
        f = footer(text, arm)
        assert all(f[k] == v for k, v in dict(slots='4', context_per_slot='4096', chunk='128',
                   input_identities='2', max_wave_rows='256', prompt_rows0='256', prompt_rows1='256',
                   gpu_tokens='156', selected_bounded_owner='0', departure_steps='4').items())
        assert all(f[f'past{s}'] == '295' for s in range(OWNERS))
        if arm == 'native-solo':
            assert f['policy'] == 'solo' and f['selected_owner'] == '0'
        else:
            assert f['policy'] == ('eager' if arm == 'native-eager' else 'joined')
            assert int(f['selected_owner']) > 0
        if arm != 'native-eager':
            assert int(f['captured']) > 0 and int(f['replayed']) > 0
        assert int(f['selected_packed_prefill']) > 0
    state = validate(root / 'native-own1', state=True)
    solo_state = validate(root / 'native-solo', state=True)
    solo = compare(root, 'native-own1', 'native-solo')
    solo['state_equal'] = [state[f'slot{s}'] == solo_state[f'slot{s}'] for s in range(OWNERS)]
    # Scalar versus joined geometry is a descriptive control, never an equality allowance.
    save(root, 'own-control.json', dict(scope='four initialized independent roots, two input identities; paired prefill and C4 decode',
                                      hashes=hashes, rows=ROWS, choices_checked=STEPS, state=state,
                                      solo_state=solo_state, joined_vs_solo=solo))


def quality(root):
    own_record = json.loads((root / 'own-control.json').read_text())
    assert digest(root / 'native-own1/heads.f32') == own_record['hashes']['heads.f32']
    for first, second in [('stock-teacher1', 'stock-teacher2'), ('stock-solo1', 'stock-solo2')]:
        for name in ('heads.f32', 'chosen.i32', 'final.f32'):
            assert digest(root / first / name) == digest(root / second / name), f'stock repeat changed {name}'
        validate(root / first)
        retired = [json.loads((root / f'{arm}-container-retired.json').read_text()) for arm in (first, second)]
        assert all(r['container_absent_after_checked_docker_query'] for r in retired)
    joined = compare(root, 'native-own1', 'stock-teacher1')
    solo = compare(root, 'native-solo', 'stock-solo1')
    geometry = compare(root, 'stock-teacher1', 'stock-solo1')
    result = dict(scope='one equal-width four-root model screen; 32 joined and four departure steps per owner',
                  joined=joined, solo=solo, stock_joined_vs_solo=geometry, passed=joined['passed'])
    save(root, 'quality.json', result)
    assert result['passed'], 'strict C4 quality gate failed; timing must not run'


def departure_own(root, log_path):
    inputs(root)
    arms = ('native-departure1', 'native-departure2')
    hashes = {}
    for name in ('heads.f32', 'chosen.i32', 'final.f32', 'state.json'):
        a, b = [root / arm / name for arm in arms]
        assert digest(a) == digest(b), f'departure repeat changed {name}'
        hashes[name] = digest(a)
    states = []
    for arm in arms:
        f = footer(log_path.read_text(), arm)
        expected = dict(slots='4', context_per_slot='4096', chunk='128', input_identities='2',
                        compatible_prefill='1', max_wave_rows='256', prompt_rows0='256', prompt_rows1='256',
                        gpu_tokens='156', policy='departure', graphs='1', selected_bounded_owner='0',
                        departure_steps='4', decode_steps='32', c3_waves='16', catchup_rows='16',
                        rejoined_waves='32', departure_passes='2', joined_prefill_groups='8',
                        joined_prefill_rows='2048')
        expected.update({f'past{s}': '295' for s in range(OWNERS)})
        for k, v in expected.items(): assert f[k] == v, (arm, k, f[k], v)
        assert all(int(f[k]) > 0 for k in ('selected_owner', 'selected_packed_prefill', 'captured', 'replayed'))
        assert int(f['c3_selected_owner']) >= 68 and int(f['c3_selected_owner']) % 34 == 0
        state = validate(root / arm, state=True)
        assert all(state[k] is True for k in ('paused_peer_equal', 'catchup_peers_equal', 'rejoin_equal'))
        states.append(state)
    save(root, 'own-control.json', dict(scope='four real roots, two input identities; actual C4-to-C3 departure, private scalar catchup and C4 rejoin',
                                      hashes=hashes, rows=ROWS, choices_checked=STEPS, state=states[0],
                                      schedule=dict(initial_c4=8, actual_c3=8, paused_owner=3, catchup_rows=8,
                                                    rejoined_c4=16, scalar_tail_per_owner=4),
                                      concurrency_acceptance=False, passed=True))


def departure_quality(root, log_path):
    own_record = json.loads((root / 'own-control.json').read_text())
    assert own_record['passed'] and own_record['concurrency_acceptance'] is False
    for name, sha in own_record['hashes'].items():
        assert digest(root / 'native-departure1' / name) == sha
    arms = ('stock-departure1', 'stock-departure2')
    text = log_path.read_text()
    for name in ('heads.f32', 'chosen.i32', 'final.f32'):
        assert digest(root / arms[0] / name) == digest(root / arms[1] / name), f'stock departure repeat changed {name}'
    for arm in arms:
        validate(root / arm)
        assert json.loads((root / f'{arm}-container-retired.json').read_text())['container_absent_after_checked_docker_query']
        f = footer(text, arm, stock=True, mode='departure')
        expected = dict(slots='4', context_per_slot='4096', chunk='128', input_identities='2',
                        compatible_prefill='1', max_wave_rows='256', prompt_rows0='256', prompt_rows1='256',
                        backend_tokens='160', observer='1', sampled_logits_min=str(VOCAB),
                        sampled_logits_max=str(VOCAB), final_head_paid='0', tokenized_equal='4',
                        departure_steps='4', decode_steps='32', c3_waves='8', catchup_rows='8',
                        rejoined_waves='16', joined_prefill_groups='4', joined_prefill_rows='1024')
        expected.update({f'past{s}': '295' for s in range(OWNERS)})
        for k, v in expected.items(): assert f[k] == v, (arm, k, f[k], v)
        sections = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
        section, = [part for part in sections[1:] if arm in part.split('\n', 1)[0]]
        shapes = re.findall(r'GEMMA3_C4_BATCH_STOCK_FLASH count=(\d+) ([^\n]+)', section)
        observed = set()
        for count, raw in shapes:
            key = tuple(map(int, raw.split()))
            assert len(key) == 12 and int(count) > 0
            if (key[0:3] == (256, 1, 8) and key[4] == 256 and key[5] >= 256 and
                    key[6] == 4 and key[8] == key[5] and key[3] == key[7] == key[11]):
                observed.add(key[3])
        assert {3, 4} <= observed, (arm, observed)
    joined = compare(root, 'native-departure1', 'stock-departure1')
    result = dict(scope='same-schedule C4/C3/catchup/C4 reference screen; four real roots, two input identities',
                  joined=joined, paused_native_state_exact=True, native_gpu_replay_exact=True,
                  concurrency_acceptance=False, passed=joined['passed'])
    save(root, 'quality.json', result)
    assert result['passed'], 'strict same-schedule departure quality failed'


def cycle(root, log_path):
    inputs(root)
    assert json.loads((root / 'quality.json').read_text())['passed']
    text = log_path.read_text()
    order = re.findall(r'step \d+/\d+ start: .*?(stock-cycle[12]|native-cycle[12])', text)
    assert order == ['stock-cycle1', 'native-cycle1', 'native-cycle2', 'stock-cycle2'], order
    result = []
    for arm in order:
        stock = arm.startswith('stock')
        pattern = r'GEMMA3_C4_BATCH_STOCK mode=cycle (.*)' if stock else r'GEMMA3_C4_BATCH mode=cycle (.*)'
        sections = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
        matching = [s for s in sections[1:] if arm in s.split('\n', 1)[0]]
        assert len(matching) == 1
        found = re.search(pattern, matching[0])
        assert found, arm
        f = dict(re.findall(r'(\w+)=([^\s]+)', found.group(1)))
        expected = dict(slots='4', context_per_slot='4096', chunk='128', input_identities='2',
                        compatible_prefill='1', max_wave_rows='256', prompt_rows0='256', prompt_rows1='256',
                        untimed_rows_per_slot='3', decode_steps='32', departure_steps='0', paid_generated_tokens='128',
                        joined_prefill_groups='8', joined_prefill_rows='2048')
        expected.update({f'past{s}': '291' for s in range(OWNERS)})
        if stock:
            expected.update(backend_tokens='192', observer='0', sampled_logits_min=str(VOCAB),
                            sampled_logits_max=str(VOCAB), final_head_paid='1', tokenized_equal='4')
            assert json.loads((root / f'{arm}-container-retired.json').read_text())['container_absent_after_checked_docker_query']
        else:
            expected.update(gpu_tokens='176', policy='joined', graphs='1', selected_bounded_owner='0')
            assert all(int(f[k]) > 0 for k in ('selected_owner', 'selected_packed_prefill', 'captured', 'replayed'))
        for k, v in expected.items(): assert f[k] == v, (arm, k, f[k], v)
        prefill, decode = float(f['prefill_seconds']), float(f['decode_seconds'])
        assert math.isfinite(prefill) and math.isfinite(decode) and prefill > 0 and decode > 0
        path = root / arm
        chosen = struct.unpack('<128i', (path / 'chosen.i32').read_bytes())
        assert all(0 <= n < VOCAB for n in chosen)
        with (path / 'final.f32').open('rb') as head:
            for _ in range(OWNERS): core.read_row(head)
            assert not head.read(1)
        result.append(dict(arm=arm, prefill_seconds=prefill, decode_seconds=decode, paid_seconds=prefill+decode,
                           choices=digest(path / 'chosen.i32'), final=digest(path / 'final.f32'), fields=f))
    assert len({a['choices'] for a in result}) == len({a['final'] for a in result}) == 1
    native = math.fsum(a['paid_seconds'] for a in result if a['arm'].startswith('native')) / 2
    stock = math.fsum(a['paid_seconds'] for a in result if a['arm'].startswith('stock')) / 2
    save(root, 'cycle.json', dict(scope='short C4 RNNR, two fresh processes per engine, final heads paid', arms=result,
                                 native_paid_seconds=native, stock_paid_seconds=stock,
                                 native_latency_over_stock=native/stock-1,
                                 native_generated_tokens_per_paid_second=128/native,
                                 stock_generated_tokens_per_paid_second=128/stock))


if __name__ == '__main__':
    root = Path(sys.argv[2])
    if sys.argv[1] == 'inputs': inputs(root)
    elif sys.argv[1] == 'own': own(root, Path(sys.argv[3]))
    elif sys.argv[1] == 'quality': quality(root)
    elif sys.argv[1] == 'departure-own': departure_own(root, Path(sys.argv[3]))
    elif sys.argv[1] == 'departure-quality': departure_quality(root, Path(sys.argv[3]))
    elif sys.argv[1] == 'cycle': cycle(root, Path(sys.argv[3]))
    else: raise ValueError('unknown analysis mode')
