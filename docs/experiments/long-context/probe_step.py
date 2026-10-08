#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares DeepSeek's fast plan and reference mode at one forced step (README.md, "Step 249").

    probe_step.py FAST_RUN EXACT_RUN ORACLE_TOKEN OTHER_TOKEN

Each RUN is a `llmp_dsv4_exec --probe-step N` output directory (the
fast plan's run, and the reference mode's with --exact on): RUN/probe/fast
and RUN/probe/exact hold that step run from the run's own state in each
plan, every named intermediate as F32 (index.json: name -> count; I32
tensors as their values in F32). So there are four variants: F/F and E/F
on the fast run's state, E/E and F/E on the reference's.

Prints, for each variant, the step's lead of ORACLE_TOKEN over OTHER_TOKEN
(the oracle's greedy token and the one llmpalooza chose instead) and its top
token; and for pairs of variants: each layer's relative L2 difference of the
attention output and the residual streams, the first layer that differs by
more than 1%, the layers whose routed experts differ (with the reference's
gap between its 6th and 7th selection scores there), and the CSA layers
whose indexer selections differ (and by how many rows). Standard library
only (runs with the Spark's system Python).
"""
import json
import math
import struct
import sys
from pathlib import Path

LAYERS = 43


def load(directory):
    d = Path(directory)
    index = json.loads((d / "index.json").read_text())
    out = {}
    for name, count in index.items():
        raw = (d / f"{name}.f32").read_bytes()
        out[name] = struct.unpack("<%df" % count, raw)
    raw = (d / "logits.f32").read_bytes()
    out["logits"] = struct.unpack("<%df" % (len(raw) // 4), raw)
    return out


def rel(a, b):
    num = den = 0.0
    for x, y in zip(a, b):
        if math.isinf(x) or math.isinf(y) or math.isnan(x) or math.isnan(y):
            continue
        num += (x - y) ** 2
        den += y * y
    return math.sqrt(num / den) if den > 0 else 0.0


def experts(v, il):
    """The layer's selected experts, as a set (fast: the route node's ids; reference: top-k)."""
    if f"ffn_moe_route-{il}" in v:
        route = v[f"ffn_moe_route-{il}"]
        return set(int(x) for x in route[:len(route) // 2])
    if f"ffn_moe_topk-{il}" in v:
        return set(int(x) for x in v[f"ffn_moe_topk-{il}"])
    return None


def selection(v, il):
    """The indexer's selected rows, as a set (-1 padding dropped)."""
    t = v.get(f"lid_topk-{il}")
    return None if t is None else set(int(x) for x in t if x >= 0)


def gap(v, il):
    """The reference's 6th minus 7th selection score, where it has them."""
    s = v.get(f"ffn_moe_selection-{il}")
    if s is None:
        return None
    top = sorted(s, reverse=True)
    return top[5] - top[6]


def main():
    fast_run, exact_run = sys.argv[1], sys.argv[2]
    oracle, other = int(sys.argv[3]), int(sys.argv[4])
    v = {"F/F": load(Path(fast_run) / "probe" / "fast"),
         "E/F": load(Path(fast_run) / "probe" / "exact"),
         "E/E": load(Path(exact_run) / "probe" / "exact"),
         "F/E": load(Path(exact_run) / "probe" / "fast")}
    report = {"variants": {}, "pairs": {}}
    for name, t in v.items():
        logits = t["logits"]
        best = max(range(len(logits)), key=lambda i: logits[i])
        report["variants"][name] = {"lead": logits[oracle] - logits[other], "argmax": best,
                                    "argmax_is_oracle": best == oracle}
    reference_gaps = {il: gap(v["E/F"], il) for il in range(LAYERS)}
    for a, b in [("F/F", "E/F"), ("F/E", "E/E"), ("E/F", "E/E"), ("F/F", "F/E")]:
        va, vb = v[a], v[b]
        attn = [rel(va.get(f"attn_out-{il}", ()), vb.get(f"attn_out-{il}", ()))
                for il in range(LAYERS)]
        streams = [rel(va.get(f"l_last-{il}", ()), vb.get(f"l_last-{il}", ()))
                   for il in range(LAYERS)]
        first = next((il for il in range(LAYERS) if streams[il] > 0.01), None)
        flips = []
        for il in range(LAYERS):
            ea, eb = experts(va, il), experts(vb, il)
            if ea is not None and eb is not None and ea != eb:
                g = reference_gaps.get(il)
                flips.append({"layer": il, "differ": len(ea ^ eb) // 2,
                              "reference_gap": None if g is None else round(g, 5)})
        picks = []
        for il in range(LAYERS):
            sa, sb = selection(va, il), selection(vb, il)
            if sa is not None and sb is not None and sa != sb:
                picks.append({"layer": il, "rows": len(sa), "differ": len(sa ^ sb) // 2})
        report["pairs"][f"{a} vs {b}"] = {
            "logits_rel": rel(va["logits"], vb["logits"]),
            "attn_rel_by_layer": [round(x, 5) for x in attn],
            "streams_rel_by_layer": [round(x, 5) for x in streams],
            "first_layer_over_1pct": first,
            "routing_flips": flips,
            "indexer_selections_differing": picks}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
