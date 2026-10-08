#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize a timing_session.sh session under the P0 timing rules.

For each kernel case, arm A is the reference and arm B the candidate. Each
arm has two or more blocks (separate processes) of 31 samples
(per-invocation microseconds). Per arm: the median of all its samples, each
block's median with a seeded 10,000-resample bootstrap 95% interval, and
two allowances:

- pair: the larger of the relative difference between two block medians and
  the sum of their intervals' half-widths divided by their mean (the M0
  rule; with more than two blocks, the first and last);
- range: the larger of the relative range of all block medians and the mean
  interval half-width divided by the mean block median.

An arm whose allowance exceeds 10% is unstable. B passes a case under a rule
when both arms are stable under it and B's median is at most A's median
times (1 + A's allowance). The summary gives each rule's pass count, a
two-sided sign test over the cases' median ratios, and whether the arms
launched the same kernels, from the profiler's symbols.
"""

import argparse
import json
import math
from pathlib import Path

SETS = ("real-40", "real-45", "synthetic")
SEED = 20260925
RESAMPLES = 10000
SAMPLES = 31


def block(samples, rng):
    import numpy as np

    samples = np.asarray(samples, dtype=np.float64)
    boots = np.median(rng.choice(samples, size=(RESAMPLES, samples.size), replace=True), axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"median": float(np.median(samples)), "half_width": float((hi - lo) / 2)}


def arm(blocks, rng):
    import numpy as np

    b = [block(s, rng) for s in blocks]
    first, last = b[0], b[-1]
    pair = max(abs(first["median"] / last["median"] - 1),
               (first["half_width"] + last["half_width"]) / ((first["median"] + last["median"]) / 2))
    medians = [x["median"] for x in b]
    mean = sum(medians) / len(medians)
    spread = max((max(medians) - min(medians)) / mean, sum(x["half_width"] for x in b) / len(b) / mean)
    return {"median": float(np.median(np.concatenate(blocks))), "blocks": b,
            "allowance": {"pair": pair, "range": spread}}


def blocks_of(session, name, case_set="real-40"):
    numbers = {p.name[len(name) + 1:].split("-")[0] for p in session.glob(f"{name}-*-{case_set}") if p.is_dir()}
    return sorted(int(n) for n in numbers if n.isdigit())


def load(session, name, number, case_set):
    rows = json.loads((session / f"{name}-{number}-{case_set}" / "kernels.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("a timing block needs nonempty case records")
    cases = {}
    for row in rows:
        key = (case_set, row["name"], row["rows"])
        if key in cases:
            raise ValueError("duplicate case in a timing block")
        samples = row["graph_replay_us"]
        if not isinstance(samples, list) or len(samples) != SAMPLES:
            raise ValueError("each timing case needs 31 samples per block")
        if any(not isinstance(s, (int, float)) or isinstance(s, bool)
               or not math.isfinite(s) or s <= 0 for s in samples):
            raise ValueError("timing samples must be finite and positive")
        cases[key] = row
    return cases


def load_session(session, arm_a, arm_b):
    """Validate all raw blocks before any medians can hide bad samples."""
    if arm_a == arm_b:
        raise ValueError("a comparison needs two distinct arms")
    numbers = blocks_of(session, arm_a)
    if len(numbers) < 2:
        raise ValueError("a timing session needs at least two blocks per arm")
    data, expected = {}, {}
    for name in (arm_a, arm_b):
        for case_set in SETS:
            if blocks_of(session, name, case_set) != numbers:
                raise ValueError("timing arms and case sets need matching blocks")
            for number in numbers:
                cases = load(session, name, number, case_set)
                keys = set(cases)
                if keys != expected.setdefault(case_set, keys):
                    raise ValueError("timing case sets differ between blocks or arms")
                for key, row in cases.items():
                    data.setdefault(key, {}).setdefault(name, []).append(row)
    return data


def sign_test(greater, less):
    n = greater + less
    if n == 0:
        return 1.0
    k = min(greater, less)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("arm_a")
    parser.add_argument("arm_b")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    data = load_session(args.session, args.arm_a, args.arm_b)
    cases = []
    for key in sorted(data):
        runs = data[key]
        a = arm([r["graph_replay_us"] for r in runs[args.arm_a]], rng)
        b = arm([r["graph_replay_us"] for r in runs[args.arm_b]], rng)
        symbols = {name: sorted({s for r in rs for s in r["profile"]["events"]}) for name, rs in runs.items()}
        ratio = b["median"] / a["median"]
        passes = {rule: a["allowance"][rule] <= 0.10 and b["allowance"][rule] <= 0.10
                  and ratio <= 1 + a["allowance"][rule] for rule in ("pair", "range")}
        cases.append({"set": key[0], "name": key[1], "rows": key[2], "path": runs[args.arm_a][0]["path"],
                      "reference": a, "candidate": b, "ratio": ratio, "passes": passes,
                      "same_kernels": symbols[args.arm_a] == symbols[args.arm_b],
                      "kernels": symbols})
    greater = sum(c["ratio"] > 1 for c in cases)
    less = sum(c["ratio"] < 1 for c in cases)
    summary = {
        "cases": len(cases),
        "blocks": {name: blocks_of(args.session, name) for name in (args.arm_a, args.arm_b)},
        "passed": {rule: sum(c["passes"][rule] for c in cases) for rule in ("pair", "range")},
        "reference_unstable": {rule: sum(c["reference"]["allowance"][rule] > 0.10 for c in cases)
                               for rule in ("pair", "range")},
        "candidate_unstable": {rule: sum(c["candidate"]["allowance"][rule] > 0.10 for c in cases)
                               for rule in ("pair", "range")},
        "same_kernels": sum(c["same_kernels"] for c in cases),
        "ratio_quantiles": {q: float(np.quantile([c["ratio"] for c in cases], float(q)))
                            for q in ("0", "0.5", "1")},
        "sign_test": {"candidate_slower": greater, "candidate_faster": less, "p_two_sided": sign_test(greater, less)},
        "allowance_max": {side: {rule: max(c[side]["allowance"][rule] for c in cases) for rule in ("pair", "range")}
                          for side in ("reference", "candidate")},
    }
    observations = (args.session / "observations.txt").read_text().splitlines()
    if any("tuning cache changed" in line for line in observations):
        raise ValueError("the session changed a frozen tuning cache")
    print(json.dumps({"session": args.session.name, "reference": args.arm_a, "candidate": args.arm_b,
                      "summary": summary, "observations": observations, "cases": cases}, indent=1))


if __name__ == "__main__":
    main()
