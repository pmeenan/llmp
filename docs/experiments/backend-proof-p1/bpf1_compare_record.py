#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Condense BP-F1's gated comparison into the record kept in Git.

  bpf1_compare_record.py --protocol timing_protocol.py --calibration bpf1-calibration.json
                         --primary P.json [--confirmation M.json] > bpf1-comparison.json

P.json and M.json are bpf1_stats.py summaries of the primary and the
mirrored confirmation sessions (run with --registry). The record holds the
rule's identity, its outcome (timing_protocol.py's apply, unchanged), each
case's d in each session, and each session condensed as bpf1_record.py
does: manifest and per-case block medians, graph and stream-launched.
"""

import argparse
import hashlib
import json
from pathlib import Path

from bpf1_record import condense, load_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    args = parser.parse_args()
    rule = load_protocol(args.protocol)
    calibration = json.loads(args.calibration.read_text())
    sessions = [json.loads(p.read_text()) for p in [args.primary] + ([args.confirmation] if args.confirmation else [])]
    d = {}
    for s in sessions:
        cases, _ = rule.session_outcome(calibration, s)
        d[s["session"]] = {k: [round(v["d"], 3), v["verdict"]] for k, v in cases.items()}
    record = {
        "rule": {"timing_protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                 "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                 "z_case": calibration["z_case"], "t_aggregate": calibration["t_aggregate"],
                 "cases": len(calibration["sigma"])},
        "outcome": rule.apply(calibration, sessions),
        "d": d,
        "sessions": {s["session"]: condense(s) for s in sessions},
    }
    print(json.dumps(record))


if __name__ == "__main__":
    main()
