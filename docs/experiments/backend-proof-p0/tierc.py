#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Calibrate and check the EXL3 Tier C rule from oracle_compare.py scores.

Input: oracle_compare.py output over one fixture's legitimate arms and its
injected faults (runs whose name contains "-f_"). Every statistic (per
prefix and part of the trajectory, and per layer) is divided by the median
of the legitimate arms' distinct values: arms whose value is bit-identical
(for example the retunings at the reconstruction-path prefixes, which share
the frozen arm's plan there) count once. For a legitimate arm the median
leaves out its own value (leave one out); for a fault it uses them all. Statistics
fall into two families:

- averaged: the logits' RMS error, and each block's row-averaged relative
  RMS error;
- extreme: the worst row's logit RMS error and the largest absolute logit
  error, each block's worst-row relative error, and K's and V's worst
  position (relative), over the prefix and over the single-token steps.

For each family the report gives the largest ratio any legitimate arm
reaches (L) and, per fault, the largest ratio it reaches (F). A threshold T
per family separates them when L < T < F for every fault. With --thresholds
AVERAGED EXTREME it checks a proposed pair: every legitimate arm must pass
(leave one out), and every fault must fail. With --bounds it writes the
absolute bounds a native run is checked against: T times the median of all
the legitimate arms, per statistic.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

LOGIT = {"rms": "averaged", "row_rms_max": "extreme", "max_abs": "extreme"}
LAYER = {"block_relative_rms": "averaged", "block_row_relative_rms_max": "extreme",
         "k_relative_position_max": "extreme", "v_relative_position_max": "extreme",
         "k_relative_suffix_max": "extreme", "v_relative_suffix_max": "extreme"}
PREFIXES = {32, 144, 145, 1023, 1024}
LAYERS = 24


def flatten(run):
    """{statistic key: (family, value)} for one run."""
    out = {}
    prefixes = [p["prefix"] for p in run["prefixes"]]
    if len(prefixes) != len(PREFIXES) or set(prefixes) != PREFIXES:
        raise ValueError("Tier C needs every held-out prefix exactly once")
    for p in run["prefixes"]:
        for part in ("prefill", "suffix"):
            if p[part].get("finite") is not True:
                raise ValueError("Tier C needs finite logits")
            if p[part]["rows"] != (p["prefix"] if part == "prefill" else 16):
                raise ValueError("Tier C trajectory row count mismatch")
            for name, family in LOGIT.items():
                out[f"{p['prefix']}/{part}/{name}"] = (family, p[part][name])
        for name, family in LAYER.items():
            values = p.get("layers", {}).get(name, [])
            if len(values) != LAYERS:
                raise ValueError(f"Tier C needs {name} for every layer")
            for layer, value in enumerate(values):
                out[f"{p['prefix']}/layer{layer}/{name}"] = (family, value)
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
           for _, v in out.values()):
        raise ValueError("Tier C errors must be finite and nonnegative")
    return out


def distinct_median(values):
    return statistics.median(sorted(set(values)))


def ratios(value, others):
    median = distinct_median(others) if others else value
    return value / median if median > 0 else (1.0 if value == 0 else float("inf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--thresholds", type=float, nargs=2, metavar=("AVERAGED", "EXTREME"))
    parser.add_argument("--bounds", type=Path, help="write absolute bounds for native runs")
    args = parser.parse_args()
    if args.bounds and not args.thresholds:
        parser.error("--bounds needs --thresholds")
    if args.thresholds and any(not math.isfinite(t) or t <= 0 for t in args.thresholds):
        parser.error("thresholds must be finite and positive")
    records = json.loads(args.scores.read_text())["runs"]
    runs = {r["run"]: flatten(r) for r in records}
    if len(runs) != len(records):
        raise ValueError("duplicate run names")
    legit = [n for n in runs if "-f_" not in n]
    faults = [n for n in runs if "-f_" in n]
    if len(legit) < 2:
        raise ValueError("leave-one-out calibration needs at least two legitimate arms")
    keys = set(runs[legit[0]])
    report = {"legitimate": legit, "faults": faults, "statistics": len(keys), "families": {}}
    worst = {}
    for family in ("averaged", "extreme"):
        fam = [k for k in keys if runs[legit[0]][k][0] == family]
        legit_max = {}
        for n in legit:
            r = {}
            for k in fam:
                value = runs[n][k][1]
                others = [runs[m][k][1] for m in legit if runs[m][k][1] != value]
                r[k] = ratios(value, others)
            key = max(r, key=r.get)
            legit_max[n] = (r[key], key)
        fault_max = {}
        for n in faults:
            r = {k: ratios(runs[n][k][1], [runs[m][k][1] for m in legit]) for k in fam if k in runs[n]}
            key = max(r, key=r.get)
            fault_max[n] = (r[key], key)
        worst[family] = (legit_max, fault_max)
        report["families"][family] = {
            "statistics": len(fam),
            "legitimate_max_ratio": max(v[0] for v in legit_max.values()),
            "legitimate": {n: {"max_ratio": v[0], "at": v[1]} for n, v in legit_max.items()},
            "faults": {n: {"max_ratio": v[0], "at": v[1]} for n, v in fault_max.items()},
        }
    if args.thresholds:
        t = dict(zip(("averaged", "extreme"), args.thresholds))
        legit_pass = {n: all(worst[f][0][n][0] <= t[f] for f in t) for n in legit}
        fault_fail = {n: any(worst[f][1][n][0] > t[f] for f in t) for n in faults}
        report["check"] = {"thresholds": t, "legitimate_pass": legit_pass, "faults_fail": fault_fail,
                           "separates": bool(faults) and all(legit_pass.values()) and all(fault_fail.values())}
        if args.bounds:
            bounds = {k: {"family": runs[legit[0]][k][0],
                          "legitimate_median": distinct_median([runs[n][k][1] for n in legit]),
                          "bound": t[runs[legit[0]][k][0]] * distinct_median([runs[n][k][1] for n in legit])}
                      for k in sorted(keys)}
            args.bounds.write_text(json.dumps({"source": str(args.scores), "thresholds": t,
                                               "legitimate_arms": legit, "bounds": bounds}, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
