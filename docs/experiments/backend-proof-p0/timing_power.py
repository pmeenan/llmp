#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Detection power of the approved kernel-timing rule for slowdowns of case subsets.

timing_protocol.py (the approved rule, unchanged) estimates power for one
slowed case at a time. This companion slows a whole subset of cases in both
sessions of a pair and reports whether the rule fails the stage, which is
what the block-level aggregate test is for:

  timing_power.py CALIBRATION PRIMARY CONFIRMATION

Subsets: every case of 145 rows or more (the reconstruction paths), the
fused-reconstruction cases (1,024 rows), the noisiest quarter by
calibrated sigma, and all cases; each at 0.5, 1, 2 and 3%.
"""

import argparse
import json
from pathlib import Path

import timing_protocol as rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("confirmation", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    sessions = [json.loads(p.read_text()) for p in (args.primary, args.confirmation)]
    keys = list(calibration["sigma"])
    rows = {k: int(k.rsplit("|", 1)[1]) for k in keys}
    by_sigma = sorted(keys, key=calibration["sigma"].get)
    subsets = {
        "reconstruction (145+ rows)": [k for k in keys if rows[k] >= 145],
        "fused reconstruction (1,024 rows)": [k for k in keys if rows[k] == 1024],
        "noisiest quarter": by_sigma[-len(keys) // 4:],
        "all cases": keys,
    }
    out = {}
    for name, subset in subsets.items():
        out[name] = {"cases": len(subset)}
        for pct in (0.5, 1, 2, 3):
            r = rule.apply(calibration, sessions, {k: 1 + pct / 100 for k in subset})
            out[name][f"{pct}%"] = {"stage_fails": not r["stage_passes"],
                                    "aggregate_t": [round(a["t"], 2) for a in r["aggregate"]],
                                    "cases_failed": len(r["failed"] or [])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
