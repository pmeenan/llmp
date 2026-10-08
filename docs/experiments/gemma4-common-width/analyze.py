#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded same-geometry C2 common-read screen; predecode diagnostics stay separate."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB, ROWS, TARGETS = 262144, 66, 64
# Final installed report resides in docs/experiments/gemma4-common-width/.
spec = importlib.util.spec_from_file_location('gemma_quality_core',
    Path(__file__).resolve().parents[1] / 'gemma-quality/analyze.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.VOCAB = VOCAB


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(root, name, result):
    with (root / name).open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result), flush=True)


def inputs(root):
    record = json.loads((root / 'inputs.json').read_text())
    assert record['source_sha256'] == '90bf82fd26b7040c513e7370eca2cd283ac8fd15ce89a2e8d05d6f7e670b5ba2'
    result = []
    prefixes = tuple(record[str(owner)]['rows'] - 35 for owner in range(2))
    assert prefixes in ((256, 768), (2560, 3072))
    for owner, prefix in enumerate(prefixes):
        path = root / f'input{owner}.i32'
        raw = path.read_bytes()
        assert len(raw) == (prefix + 35) * 4 and digest(path) == record[str(owner)]['sha256']
        ids = struct.unpack(f'<{prefix+35}i', raw)
        assert ids[0] == 2 and all(0 <= token < VOCAB for token in ids)
        result.append(ids)
    assert result[0] != result[1]
    return result


def own(root, policy):
    ids = inputs(root)
    arms = [root / f'native-{policy}-own{i}' for i in (1, 2)]
    hashes = {}
    for name in ('predecode.f32', 'predecode-state.json', 'heads.f32', 'chosen.i32',
                 'final.f32', 'state.json', 'history0.i32', 'history1.i32'):
        assert digest(arms[0] / name) == digest(arms[1] / name), f'own repeat changed {name}'
        hashes[name] = digest(arms[0] / name)
    assert (arms[0] / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
    choices = struct.unpack('<64i', (arms[0] / 'chosen.i32').read_bytes())
    last = {}
    with (arms[0] / 'heads.f32').open('rb') as file:
        for row_index in range(ROWS):
            row, _ = core.read_row(file)
            if row_index < TARGETS:
                assert max(range(VOCAB), key=row.__getitem__) == choices[row_index]
            last[row_index % 2] = row
        assert not file.read(1)
    assert struct.pack(f'<{VOCAB*2}f', *(last[0] + last[1])) == (arms[0] / 'final.f32').read_bytes()
    with (arms[0] / 'predecode.f32').open('rb') as file:
        for _ in range(2): core.read_row(file)
        assert not file.read(1)
    for owner in range(2):
        raw = (arms[0] / f'history{owner}.i32').read_bytes()
        assert raw == struct.pack(f'<{len(ids[owner])}i', *ids[owner])
    state = json.loads((arms[0] / 'state.json').read_text())
    assert all(state[k] is True for k in ('gpu_equal', 'restore_equal', 'refusals_unchanged'))
    assert state['slot0'] != state['slot1'] and all(len(state[k]) == 64 for k in ('slot0','slot1'))
    save(root, f'own-{policy}.json', dict(policy=policy, rows=ROWS, choices=TARGETS,
         hashes=hashes, state=state, initial_predecode_rows=2))


def quality(root, policy):
    ids = inputs(root)
    owned = json.loads((root / f'own-{policy}.json').read_text())
    arms = [root / f'native-{policy}-own1'] + [root / f'stock-teacher{i}' for i in (1,2)]
    assert digest(arms[0] / 'heads.f32') == owned['hashes']['heads.f32']
    for name in ('predecode.f32','heads.f32','chosen.i32','final.f32','history0.i32','history1.i32'):
        assert digest(arms[1] / name) == digest(arms[2] / name), f'stock repeat changed {name}'
    choices = [struct.unpack('<64i', (arm / 'chosen.i32').read_bytes()) for arm in arms]
    misses, deltas, tvs = [], [], []
    exact, max_raw = 0, 0.0
    for arm in arms:
        assert (arm / 'heads.f32').stat().st_size == ROWS * VOCAB * 4
        for owner in range(2):
            assert (arm / f'history{owner}.i32').read_bytes() == struct.pack(f'<{len(ids[owner])}i', *ids[owner])
    with (arms[0] / 'heads.f32').open('rb') as native, (arms[1] / 'heads.f32').open('rb') as stock:
        for index in range(ROWS):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            exact += ah == bh
            if index < TARGETS:
                assert ni == choices[0][index] and ri == choices[1][index] == choices[2][index]
            if ni != ri:
                misses.append(dict(row=index, slot=index%2, position=len(ids[index%2])-32+index//2,
                                   native=ni, stock=ri, stock_margin=margin))
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            tvs.append(.5 * math.fsum(abs(x/at-y/bt) for x,y in zip(aw,bw)))
            max_raw = max(max_raw, max(abs(x-y) for x,y in zip(a,b)))
            if index < TARGETS:
                target = ids[index%2][len(ids[index%2])-32+index//2]
                deltas.append((al-a[target])-(bl-b[target]))
        assert not native.read(1) and not stock.read(1)
    initial = dict(rows=2, exact_head_rows=0, positive_margin_differences=0, max_raw_logit_delta=0.0)
    with (arms[0] / 'predecode.f32').open('rb') as native, (arms[1] / 'predecode.f32').open('rb') as stock:
        for _ in range(2):
            a, ah = core.read_row(native); b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a,b)
            initial['exact_head_rows'] += ah == bh
            initial['positive_margin_differences'] += ni != ri and margin > 0
            initial['max_raw_logit_delta'] = max(initial['max_raw_logit_delta'], max(abs(x-y) for x,y in zip(a,b)))
        assert not native.read(1) and not stock.read(1)
    if policy == 'candidate':
        for name in ('predecode.f32','predecode-state.json'):
            assert digest(arms[0] / name) == digest(root / 'native-baseline-own1' / name), f'flag changed independent prefill {name}'
    positive = sum(m['stock_margin'] > 0 for m in misses)
    relative = math.expm1(math.fsum(deltas) / TARGETS)
    result = dict(policy=policy, scope=f'independent{len(ids[0])-35}/{len(ids[1])-35} prefill, three scalar warm rows,32 joined C2 steps',
        rows=ROWS, scored_targets=TARGETS, final_rows_scored=False, initial_predecode=initial,
        exact_head_rows=exact, strict_greedy_differences=len(misses), positive_margin_differences=positive,
        exact_tie_differences=sum(m['stock_margin']==0 for m in misses), misses=misses,
        mean_target_nll_delta=math.fsum(deltas)/TARGETS, relative_conditional_loss_delta=relative,
        mean_total_variation=math.fsum(tvs)/ROWS, max_raw_logit_delta=max_raw,
        passed=positive==0 and initial['positive_margin_differences']==0 and relative<=.03)
    save(root, f'quality-{policy}.json', result)
    if policy == 'candidate': assert result['passed'], 'strict candidate quality failed; no timing/adoption'


def cycle(root, log):
    ids = inputs(root)
    assert json.loads((root / 'quality-candidate.json').read_text())['passed']
    text = log.read_text()
    profile = root.name
    pattern_order = r'step \d+/\d+ start: [^\n]*?(stock-cycle[12]|native-cycle[12])'
    order = re.findall(pattern_order, text)
    assert order == ['stock-cycle1','native-cycle1','native-cycle2','stock-cycle2'], order
    sections = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
    result = []
    for arm in order:
        matching = [s for s in sections[1:] if arm in s.split('\n',1)[0]]
        assert len(matching)==1
        stock = arm.startswith('stock')
        marker = 'GEMMA4_CONTEXT_STOCK' if stock else 'GEMMA4_CONTEXT'
        found = re.search(marker+r' mode=cycle (.*)', matching[0]); assert found
        fields = dict(re.findall(r'(\w+)=([^\s]+)', found.group(1)))
        expected = dict(slots='2', context_per_slot='4096', independent_prefill='1', prompt_rows0='256', prompt_rows1='768',
                        untimed_rows_per_slot='3', decode_steps='32', departure_steps='0', paid_generated_tokens='64', past0='291', past1='803')
        expected['profile'] = '26' if profile=='gemma26' else '31'
        expected['chunk' if stock else 'actual_chunk'] = '1024' if profile=='gemma26' else '256'
        for k,v in expected.items(): assert fields[k]==v,(arm,k,fields.get(k),v)
        if stock:
            assert fields['backend_tokens']=='96' and fields['observer']=='0' and fields['final_head_paid']=='1'
            assert fields['sampled_logits_min']==fields['sampled_logits_max']==str(VOCAB)
            assert fields['sampled_logit_transfer']=='full'
        else:
            assert fields['policy']=='candidate' and fields['gpu_tokens']=='88'
            assert int(fields['selected_owner']) == (30 if profile=='gemma26' else 60)
            assert int(fields['captured'])>0 and int(fields['replayed'])>0
        prefill,decode = float(fields['prefill_seconds']),float(fields['decode_seconds'])
        assert prefill>0 and decode>0 and math.isfinite(prefill+decode)
        path=root/arm
        chosen=struct.unpack('<64i',(path/'chosen.i32').read_bytes()); assert all(0<=x<VOCAB for x in chosen)
        for owner in range(2):
            expected_history=list(ids[owner][:-32])+list(chosen[owner::2])
            assert (path/f'history{owner}.i32').read_bytes()==struct.pack(f'<{len(expected_history)}i',*expected_history)
        with (path/'final.f32').open('rb') as file:
            for _ in range(2): core.read_row(file)
            assert not file.read(1)
        result.append(dict(arm=arm,fields=fields,prefill_seconds=prefill,decode_seconds=decode,
            paid_seconds=prefill+decode,choices=digest(path/'chosen.i32'),final=digest(path/'final.f32'),
            history=[digest(path/f'history{i}.i32') for i in range(2)]))
    assert len({a['choices'] for a in result})==len({a['final'] for a in result})==1
    assert len({tuple(a['history']) for a in result})==1
    native=math.fsum(a['paid_seconds'] for a in result if a['arm'].startswith('native'))/2
    stock=math.fsum(a['paid_seconds'] for a in result if a['arm'].startswith('stock'))/2
    save(root,'cycle.json',dict(scope='short independent-prefill unequal C2 RNNR, final heads paid; stock full sampled logits transfer',
        arms=result,native_paid_seconds=native,stock_paid_seconds=stock,native_latency_over_stock=native/stock-1,
        native_generated_tokens_per_paid_second=64/native,stock_generated_tokens_per_paid_second=64/stock))


if __name__=='__main__':
    mode,root=sys.argv[1],Path(sys.argv[2])
    if mode=='own': own(root,sys.argv[3])
    elif mode=='quality': quality(root,sys.argv[3])
    elif mode=='cycle': cycle(root,Path(sys.argv[3]))
    else: raise ValueError('unknown mode')
