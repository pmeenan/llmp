#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Condense BP-F1's calibration and holdout sessions into the record kept in Git.

  bpf1_record.py --protocol timing_protocol.py --calibration CAL.json
                 --sessions S.json... --holdout H1.json H2.json > bpf1-timing.json

S.json and H*.json are bpf1_stats.py summaries; CAL.json is --calibrate's
output of the protocol over the calibration sessions. --protocol is the
rule's implementation: the approved one is timing_protocol.py at
c05fd2dd..., whose hash is recorded. The record holds, per session, its
manifest (host, driver, GPU settings, clocks and temperatures at block
boundaries, identities) and per case the block medians of both arms, graph
and stream-launched, in columns; then the rule's outcome for every ordered
pairing of calibration sessions (in sample), the holdout declared before it
ran (each holdout session as the primary, the other as its confirmation),
and the rule's power on the holdout pair and on the calibration pairs: for
one slowed case at a time (timing_protocol.py --power), and for whole
subsets slowed together (as ../backend-proof-p0/timing_power.py).
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

COLUMNS = ["set", "name", "rows", "sets", "set_bytes", "ratio", "reference_block_medians_us",
           "candidate_block_medians_us", "stream_ratio", "stream_reference_block_medians_us",
           "stream_candidate_block_medians_us"]


def load_protocol(path):
    spec = importlib.util.spec_from_file_location("timing_protocol_rule", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subset_power(rule, calibration, pair):
    """Whether the stage fails when a whole subset is slowed in both sessions (as timing_power.py)."""
    keys = list(calibration["sigma"])
    by_sigma = sorted(keys, key=calibration["sigma"].get)
    subsets = {
        "all cases": keys,
        "matrix products (linear, attention)": [k for k in keys if "|linear." in k or "|attn." in k],
        "elementwise and norms": [k for k in keys if "|linear." not in k and "|attn." not in k],
        "512 rows": [k for k in keys if k.endswith("|512")],
        "noisiest quarter": by_sigma[-len(keys) // 4:],
    }
    out = {}
    for name, subset in subsets.items():
        out[name] = {"cases": len(subset)}
        for pct in (0.5, 1, 2, 3):
            r = rule.apply(calibration, pair, {k: 1 + pct / 100 for k in subset})
            out[name][f"{pct}%"] = {"stage_fails": not r["stage_passes"],
                                    "aggregate_t": [round(a["t"], 2) for a in r["aggregate"]],
                                    "cases_failed": len(r["failed"] or [])}
    return out


def condense(summary):
    rows = []
    for c in summary["cases"]:
        rows.append([c["set"], c["name"], c["rows"], c["sets"], c["set_bytes"], round(c["ratio"], 6),
                     [round(b["median"], 4) for b in c["reference"]["blocks"]],
                     [round(b["median"], 4) for b in c["candidate"]["blocks"]],
                     round(c["stream"]["ratio"], 6),
                     [round(b["median"], 4) for b in c["stream"]["reference"]["blocks"]],
                     [round(b["median"], 4) for b in c["stream"]["candidate"]["blocks"]]])
    return {"manifest": summary["manifest"], "summary": summary["summary"], "columns": COLUMNS, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, nargs=2, required=True)
    args = parser.parse_args()
    rule = load_protocol(args.protocol)
    calibration = json.loads(args.calibration.read_text())
    sessions = [json.loads(p.read_text()) for p in args.sessions]
    holdout = [json.loads(p.read_text()) for p in args.holdout]
    if [s["session"] for s in sessions] != calibration["sessions"]:
        raise SystemExit("the calibration was not made from these sessions")
    in_sample = {}
    for first in sessions:
        for second in sessions:
            if first is not second:
                in_sample[f"{first['session']}+{second['session']}"] = rule.apply(calibration, [first, second])
    record = {
        "rule": {"timing_protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                 "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                 "z_case": calibration["z_case"], "t_aggregate": calibration["t_aggregate"],
                 "cases": len(calibration["sigma"])},
        "holdout_declaration": ("declared before h1 and h2 ran: the rule is rejected if either holdout "
                                "session, taken as the primary with the other as its confirmation, fails "
                                "the stage"),
        "holdout": {f"{a['session']}+{b['session']}": rule.apply(calibration, [a, b])
                    for a, b in (holdout, holdout[::-1])},
        "in_sample": in_sample,
        "power": {f"{a['session']}+{b['session']}": rule.power(calibration, [a, b])
                  for a, b in [holdout] + list(zip(sessions[::2], sessions[1::2]))},
        "subset_power": {f"{a['session']}+{b['session']}": subset_power(rule, calibration, [a, b])
                         for a, b in [holdout] + list(zip(sessions[::2], sessions[1::2]))},
        "sessions": {s["session"]: condense(s) for s in sessions + holdout},
    }
    print(json.dumps(record))


if __name__ == "__main__":
    main()
