#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares the variants of a probed step (README.md, "Step 93, diagnosed").

  probe.py DIR
      DIR is llmp_spec_runner --check probe's --out DIR/probe: each
      variant's named intermediates (VARIANT/index.json, dump.bin) and the
      target's states before the probed chunk (state-*.bin, state.json).

Prints, for pairs of variants: each layer's relative L2 difference of the
attention output, the MoE output and the residual streams, and the first
tensor that differs; each routed layer's selected experts, where they differ,
and the gap between the 6th and 7th selection scores; whether the attention
masks and the indexer's selections of the verify's row and the one-row decode
match; and the states cell by cell (per kind, and the window cache per
position with the verify row that wrote it). Standard library only (runs with
the Spark's system Python).
"""
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
FORMATS = {"f32": ("f", 4), "i32": ("i", 4), "f16": ("e", 2)}
VARIANTS = ["fast-verify", "fast-decode", "exact-verify", "exact-decode", "fast-teacher",
            "exact-teacher"]
PAIRS = [("fast-verify", "fast-decode"), ("fast-decode", "exact-decode"),
         ("exact-verify", "exact-decode"), ("fast-decode", "fast-teacher"),
         ("exact-decode", "exact-teacher")]
LAYERS = 43


def load(variant):
    d = ROOT / variant
    index = json.loads((d / "index.json").read_text())
    blob = (d / "dump.bin").read_bytes()
    return index, {t["name"]: (t, blob[t["at"]:t["at"] + t["bytes"]]) for t in index["tensors"]}


variants = {v: load(v) for v in VARIANTS if (ROOT / v / "index.json").exists()}


def get(variant, name):
    """The probed row's values of a named tensor (rows are its outermost dimension)."""
    index, tensors = variants[variant]
    if name not in tensors or tensors[name][0]["type"] not in FORMATS:
        return None
    t, raw = tensors[name]
    rows = index["rows"]
    ne = t["ne"]
    n = ne[0] * ne[1] * ne[2] * ne[3]
    outer = max([i for i in range(4) if ne[i] != 1] or [0])
    if rows > 1 and ne[outer] != rows:
        return None
    per = n // rows if rows > 1 else n
    row = index["row"] if rows > 1 else 0
    fmt, size = FORMATS[t["type"]]
    return list(struct.unpack_from("<%d%s" % (per, fmt), raw, row * per * size))


def rel(a, b):
    num = den = 0.0
    for x, y in zip(a, b):
        if math.isinf(x) or math.isinf(y):
            if x != y:
                return float("inf")
            continue
        num += (x - y) ** 2
        den += y * y
    return math.sqrt(num / den) if den > 0 else (0.0 if num == 0 else float("inf"))


def softplus(x):
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def layers(a, b):
    print("== %s against %s: relative L2 per layer" % (a, b))
    first = None
    for t in variants[a][0]["tensors"]:
        x, y = get(a, t["name"]), get(b, t["name"])
        if x is not None and y is not None and len(x) == len(y) and rel(x, y) > 0:
            first = "%s (%.2e)" % (t["name"], rel(x, y))
            break
    print("first tensor that differs (graph order): %s" % first)
    print("layer attn_out  ffn_out   l_last")
    for il in range(LAYERS):
        cells = []
        for n in ["attn_out", "ffn_out", "l_last"]:
            x, y = get(a, "%s-%d" % (n, il)), get(b, "%s-%d" % (n, il))
            cells.append("%.2e" % rel(x, y) if x and y and len(x) == len(y) else "-")
        print("%5d %s" % (il, "  ".join(cells)))
    for n in ["result_norm", "result_output"]:
        x, y = get(a, n), get(b, n)
        if x and y:
            print("%s %.3e" % (n, rel(x, y)))


def selected(variant, il, bias):
    """The routed layer's top-6 by the selection score, the 6th-7th gap."""
    logits = get(variant, "ffn_moe_logits-%d" % il)
    if logits is None or bias is None:
        return None, None
    score = [math.sqrt(softplus(x)) + c for x, c in zip(logits, bias)]
    order = sorted(range(len(score)), key=lambda e: (-score[e], e))
    return frozenset(order[:6]), score[order[5]] - score[order[6]]


def routing():
    print("== routing (the selection score's bias is the reference's selection minus its probs)")
    near = {v: 0 for v in variants}
    flips = {pair: [] for pair in PAIRS}
    for il in range(LAYERS):
        sel = get("exact-decode", "ffn_moe_selection-%d" % il)
        probs = get("exact-decode", "ffn_moe_probs-%d" % il)
        bias = [s - p for s, p in zip(sel, probs)] if sel and probs else None
        if bias is None:
            continue  # a hash-routed layer: the token's table
        sets = {}
        line = []
        for v in variants:
            s, gap = selected(v, il, bias)
            if s is None:
                continue
            sets[v] = s
            near[v] += 1 if gap < 0.01 else 0
            line.append("%s %.4f" % (v, gap))
        for a, b in PAIRS:
            if a in sets and b in sets and sets[a] != sets[b]:
                flips[(a, b)].append("%d (+%s -%s)" % (il, sorted(sets[a] - sets[b]),
                                                       sorted(sets[b] - sets[a])))
        print("%2d 6th-7th gap: %s" % (il, "; ".join(line)))
    for v, n in near.items():
        print("%s: %d routed layers with a 6th-7th gap below 0.01" % (v, n))
    for pair, where in flips.items():
        print("%s against %s: %d layers select other experts: %s" % (pair + (len(where),
                                                                             ", ".join(where))))


def masks():
    differ = 0
    for il in range(LAYERS):
        for n in ["kq_mask", "lid_topk"]:
            a, b = get("fast-verify", "%s-%d" % (n, il)), get("fast-decode", "%s-%d" % (n, il))
            if a is None or b is None:
                continue
            same = ([x == float("-inf") for x in a] == [x == float("-inf") for x in b]
                    if n == "kq_mask" else sorted(a) == sorted(b))
            differ += 0 if same else 1
    print("== the verify's row against one-row decode: %d masks or indexer selections differ" %
          differ)


def states():
    meta = json.loads((ROOT / "state.json").read_text())
    written = {}
    for pos, rows, kept, _ in meta["steps"]:
        for i in range(kept):
            written[pos + i] = i
    data = {v: (ROOT / ("state-%s.bin" % v)).read_bytes()
            for v in ["spec", "fast-teacher", "exact-teacher"]
            if (ROOT / ("state-%s.bin" % v)).exists()}
    kinds = ["raw_k", "csa_k", "csa_state_kv", "csa_state_score", "lid_k", "lid_state_kv",
             "lid_state_score", "hca_k", "hca_state_kv", "hca_state_score"]
    positions = meta["prompt_tokens"] + meta["probe_step"] - 1
    for a, b in [("spec", "fast-teacher"), ("fast-teacher", "exact-teacher")]:
        if a not in data or b not in data:
            continue
        per_kind = {}
        per_position = {}
        for t in meta["tensors"]:
            kind = kinds[t["kind"]]
            rows = {"raw_k": positions, "csa_k": positions // 4, "lid_k": positions // 4,
                    "hca_k": positions // 128}.get(kind, t["ne1"])
            fmt, size = ("e", 2) if t["f16"] else ("f", 4)
            for r in range(rows):
                at = t["offset"] + r * t["ne0"] * size
                d = rel(struct.unpack_from("<%d%s" % (t["ne0"], fmt), data[a], at),
                        struct.unpack_from("<%d%s" % (t["ne0"], fmt), data[b], at))
                per_kind.setdefault(kind, []).append(d)
                if kind == "raw_k":
                    per_position.setdefault(r, []).append(d)
        print("== state %s against %s (positions 0..%d): per row, median / p99 / max" %
              (a, b, positions - 1))
        for kind, v in per_kind.items():
            v.sort()
            print("  %-16s %.2e / %.2e / %.2e" % (kind, v[len(v) // 2], v[int(len(v) * 0.99)],
                                                   v[-1]))
        by_writer = {}
        for pos, v in per_position.items():
            v.sort()
            by_writer.setdefault(written.get(pos, "prompt"), []).append(v[len(v) // 2])
        for writer, v in sorted(by_writer.items(), key=lambda kv: str(kv[0])):
            v.sort()
            print("  raw_k at the positions the run's verify row %s wrote: %d positions, their "
                  "median over layers median %.2e max %.2e" % (writer, len(v), v[len(v) // 2],
                                                                v[-1]))


for a, b in PAIRS:
    if a in variants and b in variants:
        layers(a, b)
routing()
masks()
states()
