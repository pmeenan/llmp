#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze own results, then compare current bounded Gemma serving evidence."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

V, ROWS, PREFIX, INPUT = 262144, 129, 8063, 8192
spec = importlib.util.spec_from_file_location('gemma_quality',
    Path(__file__).resolve().parents[1] / 'gemma-quality/analyze.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def fields(log, prefix):
    return [dict(item.split('=', 1) for item in line.split()[1:])
            for line in log.splitlines() if line.startswith(prefix)]


def save(path, result):
    with path.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps(result, allow_nan=False), flush=True)


def outputs(directory, owners, heads=False, state=False):
    result = {}
    for owner in range(owners):
        for name, size in [('tokens', ROWS * 4), ('history', 8191 * 4)]:
            p = directory / f'{name}-{owner}.i32'
            assert p.stat().st_size == size
            values = struct.unpack('<' + str(size // 4) + 'i', p.read_bytes())
            assert all(0 <= value < V for value in values)
            result[p.name] = sha(p)
        if heads:
            p = directory / f'heads-{owner}.f32'
            assert p.stat().st_size == ROWS * V * 4
            result[p.name] = sha(p)
        if state:
            p = directory / f'state-{owner}.bin'
            assert p.stat().st_size > 0
            result[p.name] = sha(p)
    if state:
        result['layout.txt'] = sha(directory / 'layout.txt')
    return result


def native_freeze(root, owners):
    log = (root / 'native-quality/run.log').read_text()
    assert log.count('SERVING_PROOF_RETIRED') == 1
    assert log.count(f'FRONTIER_CONSTRUCTOR_CONTROL full_final_ffn=1 slots={owners} head_capacity={owners}') == 1
    rows = fields(log, 'SERVING_CYCLE ')
    assert [r['phase'] for r in rows] == ['first', 'repeat']
    for row in rows:
        assert row['owners'] == str(owners) and row['emitted'] == str(owners * ROWS)
        assert row['waves'] == '128' and row['cursor'] == '8191'
        assert row['rows'] == row['segments'] == str(owners)
        assert int(row['norm_rope']) > 0 and int(row['norm_add']) > 0
        if owners == 4:
            assert int(row['owner']) > 0
    first = root / 'native-quality/output/first'
    repeat = root / 'native-quality/output/repeat'
    a, b = [outputs(p, owners, heads=True, state=True) for p in (first, repeat)]
    assert a == b, 'native own repeat changed'
    input_raw = (root / 'input.i32').read_bytes()
    assert len(input_raw) == 4 * INPUT * 4
    for owner in range(owners):
        history = (first / f'history-{owner}.i32').read_bytes()
        tokens = (first / f'tokens-{owner}.i32').read_bytes()
        assert history[:PREFIX * 4] == input_raw[owner * INPUT * 4:(owner * INPUT + PREFIX) * 4]
        assert history[PREFIX * 4:] == tokens[:128 * 4]
    save(root / 'native-own.json', dict(owners=owners, files=a, input_sha256=sha(root / 'input.i32'),
                                      policy=rows, own_repeat_exact=True))


def quality(root, owners):
    own = json.loads((root / 'native-own.json').read_text())
    native = root / 'native-quality/output/first'
    assert outputs(native, owners, heads=True, state=True) == own['files']
    assert sha(root / 'input.i32') == own['input_sha256']
    reference = root / 'stock-quality/output/first'
    a, b = [outputs(root / 'stock-quality/output' / phase, owners, heads=True)
            for phase in ('first', 'repeat')]
    assert a == b, 'stock own repeat changed'
    retired = json.loads((root / 'stock-quality-container-retired.json').read_text())
    assert retired['container_absent_after_checked_docker_query'] is True
    log = (root / 'stock-quality/run.log').read_text()
    assert log.count('GEMMA_STOCK_RETIRED') == 1
    sample_rows = fields(log, 'GEMMA_STOCK_CYCLE ')
    assert [r['phase'] for r in sample_rows] == ['first', 'repeat']
    for row in sample_rows:
        assert row['backend_tokens'] == str(owners * ROWS)
        assert row['sampled_logit_bytes'] == str(owners * ROWS * V * 4)
        assert row['full_head_oracle'] == row['supplied_teacher'] == '1'
    exact = strict = ties = differences = 0
    first_differences = []
    losses_native, losses_stock = [], []
    head_shas = []
    for owner in range(owners):
        assert (reference / f'history-{owner}.i32').read_bytes() == (native / f'history-{owner}.i32').read_bytes()
        generated = struct.unpack('<129i', (native / f'tokens-{owner}.i32').read_bytes())
        stock_tokens = struct.unpack('<129i', (reference / f'tokens-{owner}.i32').read_bytes())
        head_shas.append([sha(p / f'heads-{owner}.f32') for p in (native, reference)])
        with (native / f'heads-{owner}.f32').open('rb') as nf, (reference / f'heads-{owner}.f32').open('rb') as rf:
            for index in range(ROWS):
                n, nb = core.read_row(nf)
                r, rb = core.read_row(rf)
                ni, ri, margin = core.current_choices(n, r)
                assert ni == generated[index] and ri == stock_tokens[index]
                if ni != ri and len(first_differences) < 8:
                    first_differences.append(dict(owner=owner, row=index, stock_margin=margin))
                exact += nb == rb
                differences += ni != ri
                strict += margin > 0
                ties += ni != ri and margin == 0
                if index < ROWS - 1:
                    _, _, nl = core.distribution(n)
                    _, _, rl = core.distribution(r)
                    losses_native.append(nl - n[generated[index]])
                    losses_stock.append(rl - r[generated[index]])
            assert not nf.read(1) and not rf.read(1)
    means = [math.fsum(values) / len(values) for values in (losses_native, losses_stock)]
    delta = means[0] - means[1]
    relative = math.expm1(delta) if delta < 700 else None
    passed = strict == 0 and differences == 0 and relative is not None and relative <= .03
    result = dict(owners=owners, rows=ROWS * owners, scored_targets=128 * owners,
                  final_row_scored=False, conditional_targets='frozen native generated 0 through127',
                  full_head_sha256_by_owner=head_shas, exact_head_rows=exact,
                  positive_margin_differences=strict, tie_differences=ties,
                  argmax_differences=differences, first_differences=first_differences,
                  stock_natural_lineage=differences == 0,
                  mean_native_target_nll=means[0], mean_stock_target_nll=means[1],
                  mean_target_nll_delta=delta, relative_loss=relative,
                  native_own_sha256=sha(root / 'native-own.json'), pass_all=passed)
    save(root / 'quality.json', result)
    assert passed, 'current stock strict quality or generated-history loss failed'


def timing(root, owners, single=None):
    q = json.loads((root / 'quality.json').read_text())
    assert q['pass_all'] and q['native_own_sha256'] == sha(root / 'native-own.json')
    own = json.loads((root / 'native-own.json').read_text())
    records = []
    for arm in ([single] if single else ('R1', 'N1', 'N2', 'R2')):
        native = arm.startswith('N')
        log = (root / arm / 'run.log').read_text()
        sentinel = 'SERVING_PROOF_RETIRED' if native else 'GEMMA_STOCK_RETIRED'
        assert log.count(sentinel) == 1
        if native:
            assert log.count(f'FRONTIER_CONSTRUCTOR_CONTROL full_final_ffn=1 slots={owners} head_capacity={owners}') == 1
        else:
            assert json.loads((root / (arm + '-container-retired.json')).read_text())['container_absent_after_checked_docker_query'] is True
        cycles = fields(log, 'SERVING_CYCLE ' if native else 'GEMMA_STOCK_CYCLE ')
        assert [r['phase'] for r in cycles] == ['warm', 'second', 'third']
        for row in cycles:
            assert row['owners'] == str(owners) and row['emitted'] == str(owners * ROWS)
            assert row['waves'] == '128' and row['cursor'] == '8191'
            if native:
                assert row['timed'] == '0' and row['rows'] == row['segments'] == str(owners)
                if owners == 4: assert int(row['owner']) > 0
            else:
                assert row['backend_tokens'] == str(owners * ROWS)
                assert row['sampled_logit_bytes'] == str(owners * ROWS * V * 4)
                assert row['final_head_paid'] == row['full_head_oracle'] == '0'
            directory = root / arm / 'output' / row['phase']
            current = outputs(directory, owners)
            assert current == {k:v for k,v in own['files'].items() if k.startswith(('tokens-', 'history-'))}
            assert not list(directory.glob('*.f32')), 'timing published an extra host head'
            times = {key:float(row[key + '_seconds']) for key in ('cycle', 'prefill', 'decode')}
            assert all(math.isfinite(value) and value > 0 for value in times.values())
            records.append(dict(arm=arm, phase=row['phase'], **times))
    if single:
        save(root / ('arm-' + single + '.json'), dict(records=records, frozen_history_exact=True))
        return
    means = {engine:{phase:{metric:math.fsum(r[metric] for r in records if r['arm'].startswith(engine) and r['phase'] == phase) / 2
                            for metric in ('cycle','prefill','decode')} for phase in ('second','third')}
             for engine in ('R','N')}
    save(root / 'timing.json', dict(scope='current bounded serving 8063+128 RNNR; n=2 processes per engine; no paid final-head request',
         stock_transfer='ordinary greedy chain exports full sampled logits each head', records=records, means=means,
         native_over_stock={phase:{metric:means['N'][phase][metric]/means['R'][phase][metric]-1
                                  for metric in ('cycle','prefill','decode')} for phase in ('second','third')}))


if __name__ == '__main__':
    action, path, count, *extra = sys.argv[1:]
    assert action in ('native', 'quality', 'timing', 'arm') and count in ('1', '4')
    if action == 'arm':
        assert len(extra) == 1 and extra[0] in ('R1', 'N1', 'N2', 'R2')
        timing(Path(path), int(count), extra[0])
    else:
        assert not extra
        {'native':native_freeze, 'quality':quality, 'timing':timing}[action](Path(path), int(count))
