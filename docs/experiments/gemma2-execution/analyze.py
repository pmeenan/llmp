#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare bounded Gemma2 teacher heads after the native own control froze.

ROOT contains native-graph1 and stock-teacher1/stock-teacher2. Output stays
external. Reuses the established full-row probability and winner arithmetic.
Run under installed spark-job, or hostlock shared on the workstation.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

VOCAB, ROWS, STEPS = 256000, 33, 32
core_path = Path(__file__).resolve().parents[1] / 'gemma-quality/analyze.py'
spec = importlib.util.spec_from_file_location('gemma_quality_core', core_path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.VOCAB = VOCAB


def run(root):
    ids_raw = (root / 'input/ids.i32').read_bytes()
    input_sha = hashlib.sha256(ids_raw).hexdigest()
    assert len(ids_raw) == 291 * 4
    ids = struct.unpack('<291i', ids_raw)
    own_raw = (root / 'own-control.json').read_bytes()
    own = json.loads(own_raw)
    assert input_sha == own['input_sha256']
    assert len({digest for arm, digest in own['heads'].items()
                if not arm.startswith('native-primitive')}) == 1
    paths = [root / arm / 'heads.f32' for arm in
             ('native-graph1', 'stock-teacher1', 'stock-teacher2')]
    assert all(p.stat().st_size == ROWS * VOCAB * 4 for p in paths)
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert digests[0] == own['heads']['native-graph1'], 'native changed after own control'
    assert digests[1] == digests[2], 'stock own repeat changed'
    choices = [struct.unpack('<32i', (p.parent / 'chosen.i32').read_bytes()) for p in paths]
    nll_native, nll_stock, tvs, misses, margins = [], [], [], [], []
    max_raw = max_chosen = 0.0
    exact = 0
    with paths[0].open('rb') as native, paths[1].open('rb') as stock:
        for index in range(ROWS):
            a, ah = core.read_row(native)
            b, bh = core.read_row(stock)
            ni, ri, margin = core.current_choices(a, b)
            if index < STEPS:
                assert ni == choices[0][index] and ri == choices[1][index] == choices[2][index]
            exact += ah == bh
            if ni != ri:
                misses.append(dict(row=index, native=ni, stock=ri, stock_margin=margin))
            winner2 = max(value for j, value in enumerate(b) if j != ri)
            margins.append(b[ri] - winner2)
            aw, at, al = core.distribution(a)
            bw, bt, bl = core.distribution(b)
            tvs.append(.5 * math.fsum(abs(x / at - y / bt) for x, y in zip(aw, bw)))
            max_raw = max(max_raw, max(abs(x - y) for x, y in zip(a, b)))
            max_chosen = max(max_chosen, abs((al - a[ri]) - (bl - b[ri])))
            if index < STEPS:
                target = ids[259 + index]
                nll_native.append(al - a[target])
                nll_stock.append(bl - b[target])
        assert not native.read(1) and not stock.read(1)
    result = dict(scope='representative bounded C1 first screen; no inherited numerical allowance',
                  input_sha256=input_sha, rows=ROWS, scored_targets=STEPS,
                  final_row_scored=False, vocab=VOCAB, full_head_sha256=digests,
                  own_control_sha256=hashlib.sha256(own_raw).hexdigest(),
                  exact_head_rows=exact, strict_greedy_differences=len(misses),
                  positive_margin_differences=sum(m['stock_margin'] > 0 for m in misses),
                  exact_tie_differences=sum(m['stock_margin'] == 0 for m in misses),
                  misses=misses, min_stock_top2_margin=min(margins),
                  max_raw_logit_delta=max_raw, max_stock_chosen_nll_delta=max_chosen,
                  mean_total_variation=math.fsum(tvs) / ROWS, max_total_variation=max(tvs),
                  native_mean_target_nll=math.fsum(nll_native) / STEPS,
                  stock_mean_target_nll=math.fsum(nll_stock) / STEPS,
                  mean_target_nll_delta=math.fsum(a - b for a, b in zip(nll_native, nll_stock)) / STEPS)
    with (root / 'quality.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
