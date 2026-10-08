#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Tier C: a full-model EXL3-G run against the approved bounds (reference only).

Reference-side experiment tooling for the backend proof's P3 (docs/backend-proof.md, "Tier C: EXL3
full model against an FP64 oracle"); it shares no code with llmpalooza. A run directory in the reference
layout (pack_run.py packs native's output into it) is scored against the FP64 oracle with P0's
oracle_compare.score and reduced to its 750 statistics with P0's tierc.flatten (both unchanged).
The run passes only when every statistic is present and within the approved bound for its fixture
(../backend-proof-p0/tierc.json: fixtures[F].bounds, 2 or 8 times the legitimate arms' median) and
every logit (and every captured error) is finite. The report gives, mechanically: pass or fail; every failing statistic with its
value, bound and value/bound; the first (lowest) layer with a statistic over its bound; per family the
largest ratio to the legitimate median (the calibration's ratio, against the thresholds); and top-1
agreement with the oracle (reported, not gated).

  tierc_check.py --fixture 4.0bpw|4.5bpw --oracle ORACLE.npz [--bounds tierc.json] [--json OUT]
      (--artifact ARTIFACT_DIR | --reference) RUN_DIR...

The oracle must be --fixture's by its recorded SHA-256 (fixture_identity.py). A native run must carry
pack_run.py's native.json, of the --artifact it names, whose fixture (the checkpoint it was prepared
from, never a label) must be --fixture (the bounds' and the oracle's), and its captured logits must
equal the uninstrumented run's (RE-010), or the run fails. --reference judges reference runs (P0's
arms), which have no native.json.

Exit 0 when every run passes, 1 when any fails, 2 when any is malformed or incomplete.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P0 = HERE.parent / "backend-proof-p0"
sys.path.insert(0, str(P0))
import oracle_compare  # noqa: E402
import tierc  # noqa: E402

sys.path.insert(0, str(HERE))
from fixture_identity import artifact_fixture, oracle_fixture  # noqa: E402


def judge(run, oracle, fixture_bounds, fixture, reference=False, artifact=None):
    """(exit code, report) for one run directory; artifact (fixture, id) for a native run."""
    report = {"run": str(run)}
    if not reference:
        path = run / "native.json"
        if not path.exists():
            return 2, {**report, "malformed": f"missing {path} (pack_run.py --uninstrumented)"}
        native = json.loads(path.read_text())
        report["native"] = native
        if artifact is None or native.get("artifact") != artifact[1]:
            return 2, {**report, "malformed": f"the run is of artifact {native.get('artifact')}, not --artifact's"}
        if native.get("fixture") != fixture or artifact[0] != fixture:
            return 2, {**report, "malformed": f"the run is {native.get('fixture')}, its artifact {artifact[0]}'s, "
                                              f"the bounds {fixture}"}
        if native.get("logits_equal_uninstrumented") is not True:
            differing = native.get("logits_differing")
            return 1, {**report, "passed": False,
                       "re010": f"captured logits differ from the uninstrumented run's: {differing}"}
    try:
        scores = oracle_compare.score(run, oracle)
    except (OSError, KeyError, ValueError) as error:
        return 2, {**report, "malformed": f"{type(error).__name__}: {error}"}
    nonfinite = [f"{p['prefix']}/{part}" for p in scores["prefixes"] for part in ("prefill", "suffix")
                 if p[part].get("finite") is False]
    report["top1_equal_oracle"] = {f"{p['prefix']}/{part}": [p[part]["top1_equal_oracle"], p[part]["rows"]]
                                   for p in scores["prefixes"] for part in ("prefill", "suffix")}
    report["top1_total"] = [sum(v[0] for v in report["top1_equal_oracle"].values()),
                            sum(v[1] for v in report["top1_equal_oracle"].values())]
    nonfinite += [f"{p['prefix']}/{key}" for p in scores["prefixes"] for key, values in p.get("layers", {}).items()
                  if not key.endswith("_by_position") and not np.all(np.isfinite(values))]
    if nonfinite:
        return 1, {**report, "passed": False, "nonfinite": nonfinite}
    try:
        stats = tierc.flatten(scores)
    except ValueError as error:
        return 2, {**report, "malformed": str(error)}
    bounds = fixture_bounds["bounds"]
    if set(stats) != set(bounds):
        missing, extra = sorted(set(bounds) - set(stats)), sorted(set(stats) - set(bounds))
        return 2, {**report, "malformed": f"statistics missing {missing[:5]} extra {extra[:5]}"}
    failing, worst = [], {}
    for key in sorted(stats):
        family, value = stats[key]
        b = bounds[key]
        if family != b["family"]:
            return 2, {**report, "malformed": f"{key}: family {family}, the bounds' {b['family']}"}
        ratio = value / b["legitimate_median"] if b["legitimate_median"] > 0 else (0.0 if value == 0 else float("inf"))
        if ratio > worst.get(family, (-1.0, None))[0]:
            worst[family] = (ratio, key)
        if value > b["bound"]:
            failing.append({"statistic": key, "family": family, "value": value, "bound": b["bound"],
                            "value_over_bound": value / b["bound"] if b["bound"] > 0 else float("inf")})
    layers = sorted({int(m[1]) for f in failing if (m := re.search(r"/layer(\d+)/", f["statistic"]))})
    report.update(
        passed=not failing, statistics=len(stats), failing=failing,
        first_layer_over_bound=layers[0] if layers else None,
        logit_statistics_over_bound=[f["statistic"] for f in failing if "/layer" not in f["statistic"]],
        max_ratio_to_legitimate_median={f: {"ratio": r, "at": k, "threshold": fixture_bounds["thresholds"][f]}
                                        for f, (r, k) in worst.items()})
    return (0 if not failing else 1), report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--fixture", choices=("4.0bpw", "4.5bpw"), required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--bounds", type=Path, default=P0 / "tierc.json")
    parser.add_argument("--json", type=Path, help="write the reports as JSON")
    parser.add_argument("--reference", action="store_true", help="reference runs (no native.json)")
    parser.add_argument("--artifact", type=Path, help="the installed artifact the native runs loaded (required "
                                                      "for native runs: its fixture must be --fixture)")
    args = parser.parse_args()
    try:
        if oracle_fixture(args.oracle) != args.fixture:
            print(f"MALFORMED OR INCOMPLETE: {args.oracle} is not the {args.fixture} oracle")
            return 2
        artifact = None if args.reference else artifact_fixture(args.artifact) if args.artifact else None
    except (OSError, KeyError, ValueError) as error:   # Unidentified included
        print("MALFORMED OR INCOMPLETE:", error)
        return 2
    if not args.reference and artifact is None:
        parser.error("native runs need --artifact")
    fixture_bounds = json.loads(args.bounds.read_text())["fixtures"][args.fixture]
    oracle = dict(np.load(args.oracle))
    codes, reports = [], []
    for run in args.runs:
        code, report = judge(run, oracle, fixture_bounds, args.fixture, args.reference, artifact)
        codes.append(code)
        reports.append(report)
        verdict = {0: "PASS", 1: "FAIL", 2: "MALFORMED OR INCOMPLETE"}[code]
        print(f"{verdict} {run}")
        if "malformed" in report:
            print("  ", report["malformed"])
        if "re010" in report:
            print("  ", report["re010"])
        if "nonfinite" in report:
            print("   nonfinite logits or captured errors:", report["nonfinite"])
        if "statistics" in report:
            print(f"   {len(report['failing'])} of {report['statistics']} statistics over their bound; "
                  f"first layer over bound: {report['first_layer_over_bound']}; "
                  f"logit statistics over bound: {report['logit_statistics_over_bound']}")
            for family, w in report["max_ratio_to_legitimate_median"].items():
                print(f"   {family}: max ratio to the legitimate median {w['ratio']:.3f} at {w['at']} "
                      f"(threshold {w['threshold']})")
            for f in report["failing"]:
                print(f"   over: {f['statistic']} ({f['family']}) value {f['value']:.6g} bound {f['bound']:.6g} "
                      f"ratio {f['value_over_bound']:.3f}")
        if "top1_total" in report:
            print(f"   top-1 agreement with the oracle (reported, not gated): {report['top1_total'][0]} of "
                  f"{report['top1_total'][1]} rows")
    if args.json:
        args.json.write_text(json.dumps({"fixture": args.fixture, "bounds": str(args.bounds), "runs": reports},
                                        indent=1) + "\n")
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
