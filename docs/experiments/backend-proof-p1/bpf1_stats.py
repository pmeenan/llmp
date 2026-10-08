#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize a BP-F1 session (bpf1_session.py) for the kernel-timing rule.

  bpf1_stats.py SESSION_DIR [--registry backend-proof.md] > SESSION.json

The output is what ../backend-proof-p0/timing_protocol.py (the approved
rule, at c05fd2dd...) reads, as timing_stats.py writes it for BP-F2: the
session name, the arms' block numbers, and per case (set, name, rows) the
ratio of the candidate's median to the reference's, each over all 124
samples of the arm (31 per block, four blocks), and each block's median.
The stream-launched arm is summarized beside it (`stream`), reported and
not gated.

Before any statistic it checks the raw blocks: eight timed blocks, four per
arm, each with the manifest's memory kind and the same cases in the same
order; 31 finite, positive samples per case and arm; every case's launches
verified; and one output hash per case, equal in every block of both arms
(VMM and cudaMalloc give identical results, BP-N3). A host- or device-VMM
block must carry the calibration SHA-256 pre-registered in --registry
(docs/backend-proof.md) for its memory kind's rule (v1 for host VMM, v2
for device VMM), which is then required, and its session must have run
that rule's pre-registered harness binary and case file.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

SAMPLES = 31
BLOCKS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpf1_session import RULES, registered  # noqa: E402


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_session(directory, registry_text=None):
    """The validated blocks of a session: {block label: block JSON}, and its manifest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if "failed" in manifest or "ended" not in manifest:
        raise ValueError(f"{directory.name}: the session did not complete")
    recorded = [b["block"] for b in manifest["blocks"]]
    if sorted(recorded) != sorted(BLOCKS) or recorded != manifest["order"].split():
        raise ValueError(f"{directory.name}: the blocks are not the declared order")
    blocks, keys = {}, None
    for label in BLOCKS:
        block = json.loads((directory / f"{label}.json").read_text())
        memory = manifest["arms"][label[0]]
        if block.get("memory") != memory:
            raise ValueError(f"{label}: memory {block.get('memory')}, not the manifest's {memory}")
        if memory in RULES:
            rule = RULES[memory]
            calibration = registered(registry_text, "calibration", rule) if registry_text is not None else None
            if calibration is None:
                raise ValueError(f"a {memory} block needs the registry that pre-registers its calibration")
            if block.get("calibration_sha256") != calibration or manifest.get("calibration_sha256") != calibration:
                raise ValueError(f"{label}: not run under the pre-registered rule {rule} calibration")
            identities = manifest.get("identities", {})
            for what in ("harness", "cases"):
                expected = registered(registry_text, what, rule)
                if expected is None or identities.get(f"{what}_sha256") != expected:
                    raise ValueError(f"{directory.name}: not run with the rule {rule} pre-registered {what}")
        cases = block["cases"]
        these = [(block["set"], c["name"], c["rows"]) for c in cases]
        if not these or len(set(these)) != len(these):
            raise ValueError(f"{label}: needs nonempty, unique cases")
        if keys is None:
            keys = these
        elif these != keys:
            raise ValueError(f"{label}: the cases differ from A1's")
        for c in cases:
            for field in ("graph_replay_us", "stream_us"):
                samples = c[field]
                if not isinstance(samples, list) or len(samples) != SAMPLES or not all(map(positive, samples)):
                    raise ValueError(f"{label} {c['name']}@{c['rows']}: {field} needs 31 finite, positive samples")
            if c["eager_output_fnv1a64"] != c["final_output_fnv1a64"]:
                raise ValueError(f"{label} {c['name']}@{c['rows']}: the output changed during the block")
            if not isinstance(c["graph_launches_verified"], int) or c["graph_launches_verified"] <= 0:
                raise ValueError(f"{label} {c['name']}@{c['rows']}: no verified launches")
        blocks[label] = block
    for i, key in enumerate(keys):
        hashes = {blocks[label]["cases"][i]["eager_output_fnv1a64"] for label in BLOCKS}
        if len(hashes) != 1:
            raise ValueError(f"{key}: the output differs between blocks")
    return blocks, manifest, keys


def arm(blocks, labels, i, field):
    samples = [blocks[label]["cases"][i][field] for label in labels]
    return {"median": statistics.median([s for block in samples for s in block]),
            "blocks": [{"median": statistics.median(block)} for block in samples]}


def sign_test(greater, less):
    n = greater + less
    if n == 0:
        return 1.0
    k = min(greater, less)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(directory, registry_text=None):
    blocks, manifest, keys = load_session(directory, registry_text)
    reference, candidate = ("A1", "A2", "A3", "A4"), ("B1", "B2", "B3", "B4")
    cases = []
    for i, (case_set, name, rows) in enumerate(keys):
        a, b = arm(blocks, reference, i, "graph_replay_us"), arm(blocks, candidate, i, "graph_replay_us")
        sa, sb = arm(blocks, reference, i, "stream_us"), arm(blocks, candidate, i, "stream_us")
        cases.append({"set": case_set, "name": name, "rows": rows, "ratio": b["median"] / a["median"],
                      "reference": a, "candidate": b,
                      "stream": {"ratio": sb["median"] / sa["median"], "reference": sa, "candidate": sb},
                      "sets": blocks["A1"]["cases"][i]["sets"],
                      "set_bytes": blocks["A1"]["cases"][i]["set_bytes"]})
    ratios = sorted(c["ratio"] for c in cases)
    greater = sum(r > 1 for r in ratios)
    less = sum(r < 1 for r in ratios)
    summary = {
        "cases": len(cases),
        "blocks": {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]},
        "memory": manifest["arms"],
        "ratio_quantiles": {"0": ratios[0], "0.5": statistics.median(ratios), "1": ratios[-1]},
        "sign_test": {"candidate_slower": greater, "candidate_faster": less,
                      "p_two_sided": sign_test(greater, less)},
    }
    return {"session": directory.name, "reference": "A", "candidate": "B", "summary": summary,
            "manifest": manifest, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    registry = args.registry.read_text() if args.registry else None
    print(json.dumps(summarize(args.session, registry), indent=1))


if __name__ == "__main__":
    main()
