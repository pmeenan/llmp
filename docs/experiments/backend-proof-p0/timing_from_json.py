#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Rebuild timing_stats.py-shaped session summaries from timing.json.

  timing_from_json.py TIMING_JSON SESSION OUT.json

timing.json keeps each session's cases as columns (the key names once,
then one row per case): set, name, rows, path, ratio, same_kernels and the
reference's and candidate's block medians (microseconds). The output has
the fields timing_protocol.py reads: the session name, the arms' block
counts, the sign test and, per case, the ratio and each block's median.
Ratios and medians are rounded as timing.json stores them, so a statistic
recomputed from them can differ from the original in its last digits.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing", type=Path)
    parser.add_argument("session")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    session = json.loads(args.timing.read_text())["sessions"][args.session]
    columns = session["columns"]
    cases = []
    for row in session["cases"]:
        c = dict(zip(columns, row))
        cases.append({"set": c["set"], "name": c["name"], "rows": c["rows"], "path": c["path"], "ratio": c["ratio"],
                      "same_kernels": c["same_kernels"],
                      "reference": {"blocks": [{"median": m} for m in c["reference_block_medians_us"]]},
                      "candidate": {"blocks": [{"median": m} for m in c["candidate_block_medians_us"]]}})
    args.out.write_text(json.dumps({"session": args.session, "summary": session["summary"], "cases": cases}) + "\n")


if __name__ == "__main__":
    main()
