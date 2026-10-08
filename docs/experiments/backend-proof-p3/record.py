#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Assembles results.json, the per-linear sweep's record in Git, from the runs outside it.

  record.py DIR --out results.json

DIR holds, per fixture (40, 45) and arm (G, O): the reference run ref-F-A.json (linear_reference.py
--profile), its repeat repeat-F-A.json, the forced plans plan-F-A.txt (native_plan.py), the
comparison summaries sum-F-A-P.json (compare.py) for each placement P, and launches-F-A.txt
(launches_compare.py's verdict on an nsys trace of the sweep); and sass.json (sass_compare.py). The record keeps what a later native run needs to be judged without the raw
runs: per fixture, every linear's weight and reconstructed-weight SHA-256s (the same in both arms);
per arm, its environment, libraries and final tuning cache (bytes and SHA-256), every case's plan
and final-output SHA-256, and the summaries. Inputs follow from cases.py; stage hashes stay in the
raw runs.
"""

import argparse
import json
from pathlib import Path

FIXTURES = {"40": "4.0bpw", "45": "4.5bpw"}
PLACEMENTS = ("malloc", "minimal", "flush-end", "flush-start")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs, weights = {}, {}
    for short, fixture in FIXTURES.items():
        for arm in ("G", "O"):
            ref = json.loads((args.dir / f"ref-{short}-{arm}.json").read_text())
            plans = {}
            for line in (args.dir / f"plan-{short}-{arm}.txt").read_text().splitlines():
                words = line.split()
                if words[0] == "case":
                    # The path and its numbers; the linears are the case's own.
                    plans[words[1]] = " ".join(w for w in words[3:] if not w.startswith(("model.", "lm_head")))
            runs[f"{fixture} EXL3-{arm}"] = {
                "environment": ref["environment"],
                "extension": ref["extension"],
                "cublaslt_versions": ref["cublaslt_versions"],
                "library_sha256": ref["library_sha256"],
                "torch": ref["torch"],
                "tune_cache": {"sha256": ref["tune_cache"]["after"], "unchanged": ref["tune_cache"]["before"] ==
                               ref["tune_cache"]["after"], "base64": ref["tune_cache"]["bytes_base64"]},
                "cases": {c["id"]: [plans[c["id"]], c["stages"]["out"]] for c in ref["cases"]},
                "comparisons": {p: json.loads((args.dir / f"sum-{short}-{arm}-{p}.json").read_text())
                                for p in PLACEMENTS},
                "launches": (args.dir / f"launches-{short}-{arm}.txt").read_text().strip().splitlines(),
            }
            if weights.setdefault(fixture, ref["weights"]) != ref["weights"]:
                raise SystemExit(f"{fixture}: the arms loaded different weights")
    record = {
        "recorded_on": "2026-09-27",
        "host": "spark-b",
        "tool": "linear_reference.py (reference), benchmarks/exl3_linear_sweep.cc (native), native_plan.py, "
                "compare.py, launches_compare.py, sass_compare.py; assembled by record.py",
        "interpretation": "BP-N5, the per-linear sweep: every real projection of both EXL3 fixtures at rows 1, 8, 9, "
                          "16, 32, 33, 144, 145, 1,023 and 1,024, native against upstream at the same forced plan and "
                          "inputs, in EXL3-G and EXL3-O. `cases` maps each case (linear@rows) to its plan, "
                          "native_plan.py's line without the linears' names (path, then shape and grid, GEMV configuration and grid, or the "
                          "pinned cuBLASLt algorithms per slice), and the SHA-256 of upstream's final output (F16 or F32 bytes, row-major; both matrices for "
                          "the fused gate/up). The comparisons are compare.py's summaries per operand placement.",
        "weights": weights,
        "runs": runs,
        "sass": json.loads((args.dir / "sass.json").read_text()),
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
