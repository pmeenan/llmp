# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarizes session.sh results: GB/s per variant (the first load or pass of
each process dropped), and per-extent latency (p50 and p99, microseconds).

  python3 -B summarize.py OUT/results.txt
"""
import sys, collections, statistics
rows = collections.defaultdict(list)
lat = collections.defaultdict(list)
conds = collections.defaultdict(list)
for line in open(sys.argv[1]):
    f = line.strip().split(',')
    if len(f) < 4 or f[0] in ('done',) or f[0].startswith('FAIL'):
        if f[0].startswith('FAIL'): print('FAILED', line.strip())
        continue
    tag, rnd = f[0], f[1]
    if f[2] == 'restore':      # probe: restore,mode,depth,pass,gib,gbps,p50,p99,max,cpu,bad,segs,cond
        if int(f[5]) == 0: continue
        rows[tag].append(float(f[7])); lat[tag].append((float(f[8]), float(f[9])))
        conds[tag].append(f[-1])
    elif f[2] == 'load':
        n = int(f[3])
        if n == 1: continue
        rows[tag].append(float(f[4]))
        if len(f) > 8 and f[6].replace('.','').isdigit():
            lat[tag].append((float(f[6]), float(f[7])))
        conds[tag].append(f[-1])
order = sorted(rows)
print(f"{'variant':34s} {'n':>3s} {'median':>7s} {'min':>7s} {'max':>7s}  lat p50 (median, range) / p99 median")
for t in order:
    xs = rows[t]
    l = lat.get(t) or []
    ls = f"{statistics.median([a for a,b in l]):.0f} ({min(a for a,b in l):.0f}-{max(a for a,b in l):.0f}) / {statistics.median([b for a,b in l]):.0f}" if l else ''
    print(f"{t:34s} {len(xs):3d} {statistics.median(xs):7.3f} {min(xs):7.3f} {max(xs):7.3f}  {ls}")
