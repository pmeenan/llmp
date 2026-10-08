#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares the native per-linear sweep with upstream's (BP-N5, Tier E).

  compare.py REFERENCE.json NATIVE.jsonl [--repeat REPEAT.json] [--summary OUT.json]

REFERENCE is linear_reference.py's run with --profile (and REPEAT a second run of it), NATIVE the
output of benchmarks/exl3_linear_sweep.cc at the plans native_plan.py wrote from REFERENCE. It
checks, and exits 1 unless all hold:
- the reference is self-consistent: every stepwise case reproduces upstream's module call
  (module_equal) and repeats itself (repeat_equal), REPEAT equals it stage for stage, and its
  tuning cache was unchanged by the run;
- the weights native loaded from the v0 artifact are upstream's, byte for byte (trellis, suh, svh,
  bias), and native's full reconstructed weights, rotated and fused, equal upstream's;
- every case's input is the same, and every stage native hashed equals upstream's: the
  transformed input, the product, the output transform, the bias, and each reconstruction slice;
- native's composed implementation (implementations.h) reproduces its steps;
- for EXL3-O, llmpalooza's copy of upstream's GEMV choice (upstream_gemv.h) picks, for every packed
  case, what upstream launched: the GEMV at its configuration and grid, or the GEMM.
Reference-only experiment tooling; the summary (per path and row count) is what the report
records.
"""

import argparse
import collections
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    problems = []
    if ref["tune_cache"]["before"] != ref["tune_cache"]["after"]:
        problems.append("the reference run changed its tuning cache")
    for case in ref["cases"]:
        if case["module_equal"] is False or not case["repeat_equal"]:
            problems.append(f"reference {case['id']}: module_equal {case['module_equal']}, "
                            f"repeat_equal {case['repeat_equal']}")
    if args.repeat:
        rep = json.loads(args.repeat.read_text())
        if [c["stages"] for c in rep["cases"]] != [c["stages"] for c in ref["cases"]]:
            problems.append("the repeat run differs from the reference")
        if rep["weights"] != ref["weights"]:
            problems.append("the repeat run's weights differ")

    native_weights, native_cases = {}, {}
    for line in args.native.read_text().splitlines():
        record = json.loads(line)
        if "linear" in record:
            native_weights[record["linear"]] = record
        else:
            native_cases[record["case"]] = record
    weights_equal = 0
    for name, w in ref["weights"].items():
        n = native_weights.get(name)
        if n is None:
            problems.append(f"native has no weights for {name}")
            continue
        parts = ["trellis", "suh", "svh", "W", "W_fused"] + (["bias"] if w["bias"] else [])
        bad = [p for p in parts if n.get(p) != w[p]]
        if w["bias"] is None and "bias" in n:
            bad.append("bias (native has one)")
        if bad:
            problems.append(f"{name}: weights differ in {bad}")
        else:
            weights_equal += 1

    table = collections.defaultdict(lambda: {"cases": 0, "exact": 0, "stages": 0, "composed": 0})
    for case in ref["cases"]:
        n = native_cases.get(case["id"])
        path = case["path"]
        if path == "packed":
            path = "gemv" if case.get("tag") == 90 else "gemm"
        row = table[(path, case["rows"])]
        row["cases"] += 1
        if n is None:
            problems.append(f"native did not run {case['id']}")
            continue
        if n["path"] != path:
            problems.append(f"{case['id']}: native ran {n['path']}, upstream {path}")
        if n["x"] != case["x"]:
            problems.append(f"{case['id']}: the inputs differ")
        mismatched = [s for s, h in case["stages"].items() if n["stages"].get(s) != h]
        extra = [s for s in n["stages"] if s not in case["stages"]]
        row["stages"] += len(case["stages"])
        if mismatched or extra:
            problems.append(f"{case['id']}: stages differ {mismatched}, native only {extra}")
        else:
            row["exact"] += 1
        # Llmpalooza's copy of upstream's GEMV choice (upstream_gemv.h) must pick what EXL3-O picked:
        # the recorded configuration and grid where upstream launched the GEMV, and the GEMM
        # where it did not.
        if case["path"] == "packed" and ref["arm"] == "O":
            launch = next((x for x in case.get("launches") or [] if x["name"].startswith("void exl3_gemv_kernel")),
                          None)
            if launch is not None:
                config = int(launch["name"].split("<", 1)[1].split(",")[4])
                expected = [config, launch["grid"][0]]
            else:
                expected = None
            if n.get("upstream_gemv") != expected:
                problems.append(f"{case['id']}: llmpalooza's GEMV choice {n.get('upstream_gemv')}, upstream's {expected}")
            else:
                row["gemv_choice"] = row.get("gemv_choice", 0) + 1
        if n["composed_equal"]:
            row["composed"] += 1
        else:
            problems.append(f"{case['id']}: native's composed run differs from its steps")
    extra_cases = set(native_cases) - {c["id"] for c in ref["cases"]}
    if extra_cases:
        problems.append(f"native ran {len(extra_cases)} cases upstream did not")

    summary = {
        "fixture": ref["fixture"], "arm": ref["arm"], "extension": ref["extension"],
        "cublaslt_versions": ref["cublaslt_versions"], "tune_cache_sha256": ref["tune_cache"]["after"],
        "linears": len(ref["weights"]), "weights_equal": weights_equal,
        "cases": [{"path": p, "rows": r, **v} for (p, r), v in sorted(table.items(), key=lambda i: (i[0][1], i[0][0]))],
        "problems": len(problems),
    }
    text = json.dumps(summary, indent=1) + "\n"
    if args.summary:
        args.summary.write_text(text)
    print(text)
    for problem in problems[:30]:
        print("  ", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
