#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The proposed P0 kernel-timing rule: calibrate it, apply it, estimate power.

Sessions are timing_stats.py summaries with four blocks (separate processes)
per arm, ORDER "A1 B1 B2 A2 B3 A3 A4 B4" after a warm-up process; A is the
reference, B the candidate.

--calibrate A/A-SESSION...: every block median of an A/A session measures
the same plan, so each case's process-to-process noise sigma (relative
standard deviation of block medians within a session, pooled over the
sessions) is estimated from all of them. Writes the calibration.

Applying the rule (CALIBRATION PRIMARY [CONFIRMATION]), per case: the
standardized difference d = (candidate median / reference median - 1) /
(sigma * sqrt(1/2)) (each median averages four processes). z is the
one-sided normal quantile for a family-wise false-failure rate of ALPHA (1%)
over all the calibrated cases. A case fails a session when d > z.
Aggregate: cases are not independent, because a whole process (block) can
run fast or slow. Each block's session-wide shift is the median, over the
cases, of its block median divided by that case's mean over all eight
blocks. The candidate's four shifts are compared with the reference's four
by a one-sided two-sample t-test (6 degrees of freedom); the aggregate fails
when t exceeds the quantile for ALPHA (3.143). If any case fails, or the
aggregate fails, a confirmation session repeats every case in the mirrored
order (B1 A1 A2 B2 A3 B3 B4 A4); a case fails the stage only when it fails
both sessions, and the aggregate must pass in the confirmation too.

An earlier version also failed a case whose candidate block medians spread
more than 3 sigma (and made it inconclusive for such a reference). Its first
holdout A/A pair (h1, h2) failed on single outlier processes (one block
1.6% off in an 11-microsecond kernel), so that check was removed before the
second holdout pair ran.

--power CALIBRATION SESSION SESSION: for each case in turn, slows the
candidate by 2, 3, 5 and 10% in both sessions and reports the share of
cases whose slowdown the rule detects.
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from statistics import NormalDist

ALPHA = 0.01
T_AGGREGATE = 3.143  # one-sided t quantile, 6 degrees of freedom, ALPHA


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_session(session, expected=None):
    """Incomplete or nonfinite measurements are not passing evidence."""
    cases = session["cases"]
    keys = [key(c) for c in cases]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("a session needs nonempty, unique cases")
    if expected is not None and set(keys) != set(expected):
        raise ValueError("session cases differ from the calibration")
    for c in cases:
        if not positive(c["ratio"]):
            raise ValueError("case ratios must be finite and positive")
        for side in ("reference", "candidate"):
            blocks = c[side]["blocks"]
            if len(blocks) != 4 or any(not positive(b["median"]) for b in blocks):
                raise ValueError("each case needs four finite, positive block medians per arm")
    return keys


def key(case):
    return f"{case['set']}|{case['name']}|{case['rows']}"


def relative_sd(medians):
    mean = statistics.mean(medians)
    return statistics.stdev([m / mean for m in medians])


def calibrate(sessions):
    if not sessions:
        raise ValueError("calibration needs at least one session")
    keys = validate_session(sessions[0])
    for session in sessions[1:]:
        validate_session(session, keys)
    sigma = {}
    for k in keys:
        variances = []
        for s in sessions:
            c = next(c for c in s["cases"] if key(c) == k)
            medians = [b["median"] for b in c["reference"]["blocks"] + c["candidate"]["blocks"]]
            variances.append(relative_sd(medians) ** 2)
        sigma[k] = math.sqrt(statistics.mean(variances))
    if any(not positive(value) for value in sigma.values()):
        raise ValueError("calibration needs measurable, positive noise for every case")
    return {"sessions": [s["session"] for s in sessions], "alpha": ALPHA,
            "z_case": NormalDist().inv_cdf(1 - ALPHA / len(keys)), "t_aggregate": T_AGGREGATE,
            "sigma": sigma}


def session_outcome(calibration, session, slow=None):
    sigma, z = calibration["sigma"], calibration["z_case"]
    if not sigma or any(not positive(v) for v in sigma.values()) or not positive(z):
        raise ValueError("invalid timing calibration")
    if not positive(calibration.get("t_aggregate", T_AGGREGATE)):
        raise ValueError("invalid aggregate threshold")
    validate_session(session, sigma)
    if slow is not None and (set(slow) - set(sigma) or any(not positive(v) for v in slow.values())):
        raise ValueError("invalid injected slowdown")
    cases = {}
    reference_shifts, candidate_shifts = [[] for _ in range(4)], [[] for _ in range(4)]
    for c in session["cases"]:
        k = key(c)
        factor = slow.get(k, 1.0) if slow else 1.0
        ratio = c["ratio"] * factor
        d = (ratio - 1) / (sigma[k] * math.sqrt(0.5))
        cases[k] = {"d": d, "verdict": "fail" if d > z else "pass"}
        reference = [b["median"] for b in c["reference"]["blocks"]]
        candidate = [b["median"] * factor for b in c["candidate"]["blocks"]]
        mean = statistics.mean(reference + candidate)
        for i in range(4):
            reference_shifts[i].append(reference[i] / mean - 1)
            candidate_shifts[i].append(candidate[i] / mean - 1)
    a = [statistics.median(s) for s in reference_shifts]
    b = [statistics.median(s) for s in candidate_shifts]
    pooled = (statistics.variance(a) + statistics.variance(b)) / 2
    shift = statistics.mean(b) - statistics.mean(a)
    # Constant blocks with a positive shift are evidence of a slowdown,
    # not t=0. Equal constant arms still pass.
    t = shift / math.sqrt(pooled * 0.5) if pooled > 0 else (math.copysign(math.inf, shift) if shift else 0.0)
    return cases, {"t": t, "reference_block_shifts": a, "candidate_block_shifts": b,
                   "passes": t <= calibration.get("t_aggregate", T_AGGREGATE)}


def apply(calibration, sessions, slow=None):
    if not 1 <= len(sessions) <= 2:
        raise ValueError("apply needs a primary and at most one confirmation session")
    for session in sessions:
        validate_session(session, calibration["sigma"])
    first, first_aggregate = session_outcome(calibration, sessions[0], slow)
    needs_confirmation = (not first_aggregate["passes"]
                          or any(v["verdict"] != "pass" for v in first.values()))
    result = {"sessions": [s["session"] for s in sessions], "aggregate": [first_aggregate],
              "needs_confirmation": needs_confirmation}
    if not needs_confirmation:
        result.update(failed=[], stage_passes=True)
        return result
    if len(sessions) < 2:
        result.update(failed=None, stage_passes=None)
        return result
    second, second_aggregate = session_outcome(calibration, sessions[1], slow)
    failed = [k for k in first if first[k]["verdict"] != "pass" and second[k]["verdict"] != "pass"]
    result["aggregate"].append(second_aggregate)
    result["first_session_not_passing"] = [[k, v["verdict"], round(v["d"], 2)] for k, v in first.items()
                                           if v["verdict"] != "pass"]
    result["failed"] = [[k, first[k]["verdict"], second[k]["verdict"]] for k in failed]
    result["stage_passes"] = not failed and second_aggregate["passes"]
    return result


def power(calibration, sessions):
    if len(sessions) != 2:
        raise ValueError("power needs a primary and a confirmation session")
    out = {}
    keys = list(calibration["sigma"])
    for pct in (2, 3, 5, 10):
        detected = 0
        for k in keys:
            r = apply(calibration, sessions, {k: 1 + pct / 100})
            detected += any(f[0] == k for f in r["failed"] or [])
        out[f"{pct}%"] = {"cases": len(keys), "detected": detected, "share": detected / len(keys)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--power", action="store_true")
    args = parser.parse_args()
    loaded = [json.loads(p.read_text()) for p in args.files]
    if args.calibrate:
        print(json.dumps(calibrate(loaded), indent=1))
        return
    calibration, sessions = loaded[0], loaded[1:]
    for s in sessions:
        if any(len(v) != 4 for v in s["summary"]["blocks"].values()):
            raise ValueError(f"{s['session']}: the rule needs four blocks per arm")
    print(json.dumps(power(calibration, sessions) if args.power else apply(calibration, sessions), indent=1))


if __name__ == "__main__":
    main()
