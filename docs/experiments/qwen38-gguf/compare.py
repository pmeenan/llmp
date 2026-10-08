#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Judges llmpalooza's Qwen3.8 GGUF run against the llama.cpp oracle (README.md).

  compare.py ORACLE_UNFUSED ORACLE_FUSED LLMP [--free LLMP_FREE] [--ppl LLMP_PPL]

ORACLE_* are oracle.cc's output directories, LLMP llmp_qwen38_exec's run
forced on the unfused oracle's tokens (--prompts prompts.tsv --force
ORACLE_UNFUSED/generated.tokens --generate 32), LLMP_PPL its --ppl run.
Prints one JSON document: per prompt, the logit differences against each
oracle arm and between the arms, the teacher-forced argmax check with its
near-tie rule (a step whose oracle top-1 to top-2 margin is below twice
the largest absolute logit difference between the engines at that step),
and the perplexities. Exits 1 if a bound fails (README.md). Runs in the
llama.cpp image, whose Python has NumPy.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

VOCAB = 248320
PPL_BOUND = 0.02  # relative to the unfused oracle's


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


def diff(a, b):
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "rms": float(math.sqrt(float(np.mean((a - b) ** 2))))}


def top_agreement(a, b, k=5):
    """Steps whose top-k token sets agree."""
    return int(sum(set(np.argsort(a[i])[-k:]) == set(np.argsort(b[i])[-k:]) for i in range(a.shape[0])))


def main(argv):
    unfused, fused, jit = argv[1:4]
    generated = lines(Path(unfused) / "generated.tokens")
    jsummary = json.loads((Path(jit) / "summary.json").read_text())
    report = {"prompts": [], "exceptions": [], "ok": True}
    steps_total = matched = 0
    for entry in jsummary["prompts"]:
        name = entry["name"]
        steps = len(generated[name])
        o = logits(unfused, name, steps)
        f = logits(fused, name, steps)
        j = logits(jit, name, steps)
        argmax = entry["argmax"]
        row = {"name": name, "steps": steps, "vs_unfused": diff(j, o), "vs_fused": diff(j, f),
               "fused_vs_unfused": diff(f, o), "argmax_equal": 0,
               "top5_sets_equal_vs_unfused": top_agreement(j, o),
               "top5_sets_equal_fused_vs_unfused": top_agreement(f, o)}
        for k in range(steps):
            want = generated[name][k]
            steps_total += 1
            if argmax[k] == want:
                matched += 1
                row["argmax_equal"] += 1
                continue
            top = np.sort(o[k])[-2:]
            margin = float(top[1] - top[0])
            step_diff = float(np.max(np.abs(j[k] - o[k])))
            tie = margin < 2 * step_diff
            report["exceptions"].append({"prompt": name, "step": k, "oracle": want, "llmp": argmax[k],
                                         "oracle_margin": margin, "max_abs_diff": step_diff,
                                         "near_tie": tie})
            if not tie:
                report["ok"] = False
        report["prompts"].append(row)
    report["argmax"] = {"steps": steps_total, "equal": matched}
    for arm, path in (("unfused", unfused), ("fused", fused)):
        summary = json.loads((Path(path) / "summary.json").read_text())
        report[f"ppl_{arm}"] = summary.get("ppl")
    if "--ppl" in argv:
        ppl_dir = Path(argv[argv.index("--ppl") + 1])
        jppl = json.loads((ppl_dir / "summary.json").read_text())["ppl"]
        report["ppl_llmp"] = jppl
        ref = report["ppl_unfused"]["ppl"]
        rel = (jppl["ppl"] - ref) / ref
        report["ppl_relative_to_unfused"] = rel
        if abs(rel) > PPL_BOUND:
            report["ok"] = False
        a = np.fromfile(ppl_dir / "ppl.nll.f64", dtype=np.float64)
        b = np.fromfile(Path(unfused) / "ppl.nll.f64", dtype=np.float64)
        if a.size == b.size:
            report["ppl_nll_max_abs_diff"] = float(np.max(np.abs(a - b)))
            report["ppl_nll_mean_abs_diff"] = float(np.mean(np.abs(a - b)))
    if "--free" in argv:
        # A free-running greedy run (no --force): its argmax is its own
        # continuation; reported, not gated.
        free = json.loads((Path(argv[argv.index("--free") + 1]) / "summary.json").read_text())
        report["free_running"] = []
        for entry in free["prompts"]:
            want = generated[entry["name"]]
            got = entry["argmax"]
            agree = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), len(want))
            report["free_running"].append({"name": entry["name"], "identical": got == want,
                                           "agreeing_prefix": agree})
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
