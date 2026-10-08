# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Describe same-conditioned generated-history rows; this is not held-out PPL."""
import argparse, array, hashlib, heapq, json, math, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('native', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('history', type=Path)
p.add_argument('--history-key')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
history = json.loads(a.history.read_text())
history = history[a.history_key] if a.history_key else history
if len(history) != 32 or any((type(t) != int or not 0 <= t < 248320 for t in history)):
    raise RuntimeError('invalid shared conditioning targets')

def read(path):
    data = path.read_bytes()
    if len(data) != 32 * 248320 * 4:
        raise RuntimeError('incomplete vocabulary rows')
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all((math.isfinite(x) for x in values)):
        raise RuntimeError('nonfinite vocabulary rows')
    return (values, hashlib.sha256(data).hexdigest())
native, nh = read(a.native)
reference, rh = read(a.reference)
rows = []
for i, target in enumerate(history):
    n = native[i * 248320:(i + 1) * 248320]
    r = reference[i * 248320:(i + 1) * 248320]
    ntop = heapq.nlargest(5, range(len(n)), key=n.__getitem__)
    rtop = heapq.nlargest(5, range(len(r)), key=r.__getitem__)
    nm = n[ntop[0]]
    rm = r[rtop[0]]
    nz = nm + math.log(math.fsum((math.exp(x - nm) for x in n)))
    rz = rm + math.log(math.fsum((math.exp(x - rm) for x in r)))
    shared = sorted(set(ntop + rtop))
    deltas = [n[t] - nz - (r[t] - rz) for t in shared]
    rows.append({'row': i, 'target': target, 'native_argmax': ntop[0], 'reference_argmax': rtop[0], 'agree': ntop[0] == rtop[0], 'native_top2_margin': nm - n[ntop[1]], 'reference_top2_margin': rm - r[rtop[1]], 'native_loss_for_reference_argmax': nm - n[rtop[0]], 'reference_loss_for_native_argmax': rm - r[ntop[0]], 'native_nll': nz - n[target], 'reference_nll': rz - r[target], 'native_top5': ntop, 'reference_top5': rtop, 'union_top5_logprob_max_abs_delta': max(map(abs, deltas)), 'union_top5_logprob_mean_abs_delta': sum(map(abs, deltas)) / len(deltas)})
nn = math.fsum((row['native_nll'] for row in rows)) / 32
rn = math.fsum((row['reference_nll'] for row in rows)) / 32
out = {'scope': '32 supplied generated-history targets, identical conditioning; no held-out PPL qualification', 'native_sha256': nh, 'reference_sha256': rh, 'history': history, 'argmax_agreement': sum((row['agree'] for row in rows)), 'prior_near_tie_margin': 1.0, 'mismatches_outside_prior_margin': [row['row'] for row in rows if not row['agree'] and row['reference_top2_margin'] > 1.0], 'native_mean_nll': nn, 'reference_mean_nll': rn, 'mean_nll_delta': nn - rn, 'generated_history_exp_mean_nll_ratio': math.exp(nn - rn), 'rows': rows}
a.out.write_text(json.dumps(out, indent=2) + '\n')
