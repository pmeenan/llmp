#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One continuous grouped-owner screen; raw heads, IDs and state remain external."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB, OWNERS, ROWS = 262208, 12, 472
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
    two = []
    for source, count in enumerate((295, 807)):
        path = root / f'input{source}/ids.i32'
        raw = path.read_bytes()
        assert len(raw) == count * 4 == record[str(source)]['rows'] * 4
        assert digest(path) == record[str(source)]['ids_sha256']
        assert digest(root / f'prompt{source}.txt') == record[str(source)]['text_sha256']
        ids = struct.unpack(f'<{count}i', raw)
        assert ids[0] == 2 and all(0 <= n < VOCAB for n in ids)
        two.append(ids)
    return [two[1 if s in (2, 3, 8, 9, 10, 11) else 0] for s in range(OWNERS)]


def mapping(root):
    """Independent target/cursor derivation, not acceptance of caller metadata."""
    ids = inputs(root)
    prefix = [len(i) - 39 for i in ids]
    past, rows = [0] * OWNERS, []

    def emit(phase, kind, first, count, n, head=True):
        for s in range(first, first + count):
            past[s] += n
            assert past[s] <= len(ids[s])
            if head:
                rows.append((phase, kind, s, past[s], ids[s][past[s]] if past[s] < len(ids[s]) else -1))

    for at in range(0, 768, 128):
        for first in range(0, OWNERS, 2):
            if at < prefix[first]:
                emit(0, 0, first, 2, 128, at + 128 == prefix[first])
    for s in range(OWNERS):
        for _ in range(3):
            emit(1, 1, s, 1, 1)
    phase = 2
    for count in (12, 11, 10, 9, 8, 7, 6, 5, 4):
        for _ in range(4 if count in (12, 8, 4) else 2):
            emit(phase, 2, 0, count, 1)
        phase += 1
    for s in range(4, OWNERS):
        while past[s] < prefix[s] + 27:
            emit(phase, 1, s, 1, 1)
    phase += 1
    for _ in range(8):
        emit(phase, 2, 0, OWNERS, 1)
    phase += 1
    past[10] = past[11] = 0
    for at in range(0, prefix[10], 128):
        emit(phase, 0, 10, 2, 128, at + 128 == prefix[10])
        if at < 256:
            emit(phase, 2, 0, 4, 1)
    for s in (10, 11):
        for _ in range(3):
            emit(phase, 1, s, 1, 1)
    phase += 1
    for s in range(OWNERS):
        for _ in range(2):
            emit(phase, 1, s, 1, 1)
    assert len(rows) == ROWS and sum(row[-1] >= 0 for row in rows) == 468
    assert past == [prefix[s] + (39 if s < 4 else 37 if s < 10 else 5) for s in range(OWNERS)]
    return rows, past


def validate(root, arm, state=False):
    path = root / arm
    expected, past = mapping(root)
    actual = [tuple(map(int, line.split())) for line in (path / 'rows.tsv').read_text().splitlines()]
    assert actual == expected
    assert (path / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
    assert (path / 'chosen.i32').stat().st_size == ROWS * 4
    chosen = struct.unpack(f'<{ROWS}i', (path / 'chosen.i32').read_bytes())
    last = {}
    with (path / 'heads.f32').open('rb') as heads:
        for index, item in enumerate(expected):
            row, _ = core.read_row(heads)
            assert max(range(VOCAB), key=row.__getitem__) == chosen[index]
            last[item[2]] = struct.pack(f'<{VOCAB}f', *row)
        assert not heads.read(1)
    assert b''.join(last[s] for s in range(OWNERS)) == (path / 'final.f32').read_bytes()
    if state:
        lines = [line.split() for line in (path / 'state.tsv').read_text().splitlines()]
        assert len(lines) == OWNERS
        assert all(int(item[0]) == s and int(item[1]) == past[s] and re.fullmatch('[0-9a-f]{64}', item[2])
                   for s, item in enumerate(lines))
    return past


def footer(text, arm, stock=False):
    pieces = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
    matching = [s for s in pieces[1:] if arm in s.split('\n', 1)[0]]
    assert len(matching) == 1, (arm, len(matching))
    prefix = 'GEMMA3_WIDE_STOCK' if stock else 'GEMMA3_WIDE'
    assert prefix + '_RETIRED' in matching[0]
    line = re.search(prefix + r'_PASS (.*)', matching[0])
    assert line
    fields = dict(re.findall(r'(\w+)=([^\s]+)', line.group(1)))
    assert fields['rows'] == '472' and fields['slots'] == '12' and fields['context'] == '4096'
    assert fields['input_identities'] == '2' and fields['per_owner_rows'] == '128' and fields['wave_rows'] == '256'
    phases = re.findall(prefix + r'_PHASE logical=(\d+) actual_waves=(\d+) (?:selected_owner_plans|flash_shapes)=(\d+)', matching[0])
    expected_waves = {4: 6, 8: 4, 12: 12, **{n: 2 for n in (5, 6, 7, 9, 10, 11)}}
    assert len(phases) == 9
    for n, waves, selected in map(lambda t: tuple(map(int, t)), phases):
        assert waves == expected_waves[n] * (1 if stock else 2) and selected > 0
    if stock:
        assert fields['backend_tokens'] == '472'
        assert fields['sampled_logits_min'] == fields['sampled_logits_max'] == str(VOCAB)
        assert fields['prefill_groups'] == '30' and fields['prefill_rows'] == '7680'
    else:
        assert fields['gpu_equal'] == fields['state_equal'] == fields['restore_equal'] == '1'
        assert fields['gpu_tokens'] == '460' and fields['peer_checks'] == '36'
        assert fields['head_rows'] == '12' and fields['prefill_groups'] == '60' and fields['prefill_rows'] == '15360'
        assert int(fields['captured']) > 0 and int(fields['replayed']) > 0
        assert int(fields['cache_keys']) == int(fields['actual_plans']) > 0
        assert 0 < int(fields['plan_graph_bytes']) <= int(fields['cache_budget']) <= 64 * 2**30
        assert fields['host_overcharges'] == '0'
        assert int(fields['host_counted']) == int(fields['plan_graph_bytes'])
        assert 0 <= int(fields['host_charged']) <= int(fields['cache_budget'])
    return fields


def own(root, log):
    hashes = {}
    for name in ('heads.f32', 'rows.tsv', 'chosen.i32', 'final.f32', 'state.tsv'):
        first, second = (root / arm / name for arm in ('native-own1', 'native-own2'))
        assert digest(first) == digest(second), name
        hashes[name] = digest(first)
    past = validate(root, 'native-own1', state=True)
    footers = {arm: footer(log.read_text(), arm) for arm in ('native-own1', 'native-own2')}
    save(root, 'own-control.json', dict(passed=True, hashes=hashes, rows=ROWS, scored_targets=468,
         owners=OWNERS, input_identities=2, final_cursors=past, footers=footers,
         scope='same continuous unequal schedule only; no solo/batch numerical allowance'))


def quality(root, log):
    own_record = json.loads((root / 'own-control.json').read_text())
    assert own_record['passed']
    for name, expected in own_record['hashes'].items():
        assert digest(root / 'native-own1' / name) == expected
    for name in ('heads.f32', 'rows.tsv', 'chosen.i32', 'final.f32'):
        assert digest(root / 'stock-teacher1' / name) == digest(root / 'stock-teacher2' / name), name
    validate(root, 'stock-teacher1')
    for arm in ('stock-teacher1', 'stock-teacher2'):
        footer(log.read_text(), arm, stock=True)
        retired = json.loads((root / f'{arm}-container-retired.json').read_text())
        assert retired['container_absent_after_checked_docker_query']
    expected, _ = mapping(root)
    exact, misses, losses, tvs, max_raw = 0, [], [], [], 0.0
    with (root / 'native-own1/heads.f32').open('rb') as fa, (root / 'stock-teacher1/heads.f32').open('rb') as fb:
        for index, item in enumerate(expected):
            a, ah = core.read_row(fa)
            b, bh = core.read_row(fb)
            ni, ri, margin = core.current_choices(a, b)
            exact += ah == bh
            if ni != ri:
                misses.append(dict(row=index, phase=item[0], slot=item[2], position=item[3], reference_margin=margin))
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            if item[4] >= 0:
                losses.append((al - a[item[4]]) - (bl - b[item[4]]))
            tvs.append(.5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)))
            max_raw = max(max_raw, max(abs(x - y) for x, y in zip(a, b)))
        assert not fa.read(1) and not fb.read(1)
    assert len(losses) == 468
    nll = math.fsum(losses) / len(losses)
    relative = math.expm1(nll)
    result = dict(scope='same-schedule native versus original stock; concurrency across geometry remains open',
        rows=ROWS, scored_targets=len(losses), exact_head_rows=exact, strict_greedy_differences=len(misses),
        positive_margin_differences=sum(m['reference_margin'] > 0 for m in misses),
        exact_tie_differences=sum(m['reference_margin'] == 0 for m in misses), differences=misses,
        mean_target_nll_delta=nll, relative_conditional_loss_delta=relative,
        mean_total_variation=math.fsum(tvs) / ROWS, max_raw_logit_delta=max_raw,
        heads_sha256=[digest(root / arm / 'heads.f32') for arm in ('native-own1', 'stock-teacher1')],
        passed=not misses and relative <= .03)
    save(root, 'quality.json', result)
    assert result['passed'], 'strict same-schedule wide quality failed; timing must not run'


if __name__ == '__main__':
    assert len(sys.argv) == 4 and sys.argv[1] in ('own', 'quality')
    {'own': own, 'quality': quality}[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3]))
