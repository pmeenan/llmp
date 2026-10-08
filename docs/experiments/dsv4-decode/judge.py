#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Judges DeepSeek V4 Flash's fast plan coarsely (README.md, "Correctness").

  judge.py noise EXACT FAST
      The near-tie bound from llmpalooza's own kernel-to-kernel noise: two
      llmp_dsv4_exec runs forced on the same tokens, the reference mode
      (--exact on) and the fast plan. At each step, the change in the
      reference's top-two margin (the fast run's logit difference between
      the reference's two best tokens against the reference's); the bound is
      twice the largest change over every step.
  judge.py greedy ORACLE LLMP --bound B
      llmpalooza (the fast plan) forced on an oracle arm's generated tokens: its
      argmax equals the oracle's token at every step, except where the
      oracle's margin between its token and llmpalooza's argmax is below B (a
      near-tie). Every exception is listed.
  judge.py ppl ORACLE LLMP [--within F]
      Perplexity within F (relative, default 0.03, D-085's note) of the
      oracle arm's.

ORACLE is oracle.cc's output directory (dsv4-native), EXACT, FAST and LLMP
llmp_dsv4_exec's (--prompts ... --generate N --force TOKENS, or --ppl).
Prints one JSON document; exits 1 if a bound fails. Runs in the pinned
llama.cpp image, whose Python has NumPy.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

VOCAB = 129280


def lines(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            name, _, ids = line.partition("\t")
            out[name] = [int(x) for x in ids.split()]
    return out


def logits(directory, name, steps):
    a = np.fromfile(Path(directory) / f"{name}.logits.f32", dtype=np.float32)
    if a.size != steps * VOCAB:
        raise SystemExit(f"{directory}/{name}: {a.size} logits, not {steps} x {VOCAB}")
    if not np.all(np.isfinite(a)):
        raise SystemExit(f"{directory}/{name}: non-finite logits")
    return a.reshape(steps, VOCAB).astype(np.float64)


def summary(directory):
    return json.loads((Path(directory) / "summary.json").read_text())


def noise(exact, fast):
    e_sum = summary(exact)
    deltas = []
    max_abs = 0.0
    rms = []
    for entry in e_sum["prompts"]:
        name = entry["name"]
        steps = len(entry["argmax"])
        e = logits(exact, name, steps)
        f = logits(fast, name, steps)
        for k in range(steps):
            order = np.argsort(e[k])
            a, b = int(order[-1]), int(order[-2])
            deltas.append(abs((f[k, a] - f[k, b]) - (e[k, a] - e[k, b])))
        d = np.abs(e - f)
        max_abs = max(max_abs, float(d.max()))
        rms.append(float(math.sqrt(float(np.mean((e - f) ** 2)))))
    deltas = np.array(deltas)
    # "bound" is the rule this slice recorded (twice the largest change);
    # README.md's "The bound, going forward" explains why later slices take
    # a percentile of noise between two of llmpalooza's own paths instead.
    report = {"steps": int(deltas.size), "margin_change_max": float(deltas.max()),
              "margin_change_p99": float(np.percentile(deltas, 99)),
              "margin_change_p95": float(np.percentile(deltas, 95)),
              "margin_change_median": float(np.median(deltas)),
              "logit_max_abs": max_abs, "logit_rms_max": max(rms),
              "bound": 2.0 * float(deltas.max())}
    print(json.dumps(report, indent=1))
    return 0


def greedy(oracle, jit, bound):
    generated = lines(Path(oracle) / "generated.tokens")
    report = {"bound": bound, "steps": 0, "equal": 0, "near_ties": [], "violations": [], "ok": True}
    for entry in summary(jit)["prompts"]:
        name = entry["name"]
        want = generated[name]
        steps = len(want)
        o = logits(oracle, name, steps)
        got = entry["argmax"]
        for k in range(steps):
            report["steps"] += 1
            if got[k] == want[k]:
                report["equal"] += 1
                continue
            margin = float(o[k, want[k]] - o[k, got[k]])
            item = {"prompt": name, "step": k, "oracle": want[k], "llmp": got[k],
                    "oracle_margin": margin}
            if margin < bound:
                report["near_ties"].append(item)
            else:
                report["violations"].append(item)
                report["ok"] = False
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


def ppl(oracle, jit, within):
    text = (Path(oracle) / "summary.json").read_text()
    start = text.index('"ppl":') + len('"ppl":')
    ref = json.loads(text[start:text.index("}", start) + 1])["ppl"]
    got = summary(jit)["ppl"]["ppl"]
    rel = abs(got - ref) / ref
    report = {"oracle": ref, "llmp": got, "relative": rel, "within": within, "ok": rel <= within}
    a = np.fromfile(Path(jit) / "ppl.nll.f64", dtype=np.float64)
    b = np.fromfile(Path(oracle) / "ppl.nll.f64", dtype=np.float64)
    if a.size == b.size:
        report["nll_max_abs_diff"] = float(np.max(np.abs(a - b)))
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


def main(argv):
    if len(argv) >= 4 and argv[1] == "noise":
        return noise(argv[2], argv[3])
    if len(argv) >= 6 and argv[1] == "greedy" and argv[4] == "--bound":
        return greedy(argv[2], argv[3], float(argv[5]))
    if len(argv) >= 4 and argv[1] == "ppl":
        within = float(argv[argv.index("--within") + 1]) if "--within" in argv else 0.03
        return ppl(argv[2], argv[3], within)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
