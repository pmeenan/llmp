#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Judges llmpalooza's DeepSeek V4 run against the llama.cpp oracle (README.md).

  compare.py ORACLE_UNFUSED ORACLE_FUSED LLMP [--dump ORACLE_DUMP] [--free LLMP_FREE]

ORACLE_* are oracle.cc's output directories, LLMP llmp_dsv4_exec's run
forced on the unfused oracle's tokens (--force generated.tokens). Prints one
JSON document: per prompt, the logit differences against each oracle arm and
between the arms, the teacher-forced argmax check with its near-tie rule,
the perplexities, and (with --dump) each named tensor's difference. Exits 1
if a pre-registered bound fails. Runs in the pinned llama.cpp image, whose
Python has NumPy.
"""
import json
import math
import re
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


def diff(a, b):
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "rms": float(math.sqrt(float(np.mean((a - b) ** 2))))}


def main(argv):
    unfused, fused, jit = argv[1:4]
    dump = argv[argv.index("--dump") + 1] if "--dump" in argv else None
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
               "fused_vs_unfused": diff(f, o), "argmax_equal": 0}
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
        # The oracle's first build wrote its generated IDs space-separated in
        # summary.json (they are in generated.tokens); read its perplexity
        # object alone.
        text = (Path(path) / "summary.json").read_text()
        found = re.search(r'"ppl":(\{[^}]*\})', text)
        report[f"ppl_{arm}"] = json.loads(found.group(1)) if found else None
    if "ppl" in jsummary:
        report["ppl_llmp"] = jsummary["ppl"]
        ref = report["ppl_unfused"]["ppl"]
        rel = abs(jsummary["ppl"]["ppl"] - ref) / ref
        report["ppl_relative_to_unfused"] = rel
        if rel > 0.02:
            report["ok"] = False
        a = np.fromfile(Path(jit) / "ppl.nll.f64", dtype=np.float64)
        b = np.fromfile(Path(unfused) / "ppl.nll.f64", dtype=np.float64)
        if a.size == b.size:
            report["ppl_nll_max_abs_diff"] = float(np.max(np.abs(a - b)))
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
    if dump:
        tensors = {}
        for p in sorted((Path(jit) / "dump").glob("*.f32")):
            q = Path(dump) / "dump" / p.name
            if not q.exists():
                tensors[p.stem] = "missing in the oracle's dump"
                continue
            x = np.fromfile(p, dtype=np.float32).astype(np.float64)
            y = np.fromfile(q, dtype=np.float32).astype(np.float64)
            if x.size != y.size:
                tensors[p.stem] = f"sizes differ: {x.size} vs {y.size}"
                continue
            scale = float(np.max(np.abs(y))) or 1.0
            tensors[p.stem] = {"max_abs": float(np.max(np.abs(x - y))), "max_abs_ref": scale,
                               "nmse": float(np.sum((x - y) ** 2) / max(np.sum(y ** 2), 1e-30))}
        report["dump"] = tensors
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
