#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate a complete run and emit aggregates; raw inputs stay outside Git."""
import argparse
import json
from pathlib import Path
from statistics import median

from experiment import CASES, check

GIB = 1 << 30


def span(values):
    values = list(values)
    return {"median": median(values), "min": min(values), "max": max(values)}


def summarize(root):
    manifest = json.loads((root / "run.json").read_text())
    order = json.loads((root / "order.json").read_text())
    expected = [f"{r+1:02d}-{c}" for r in range(manifest["repeats"]) for c in manifest["cases"]]
    check(order == expected, "Incomplete or reordered run")
    check(set(manifest["cases"]) == set(CASES), "The report requires every protocol arm")
    trials = [json.loads((root / name / "result.json").read_text()) for name in order]
    for name, trial in zip(order, trials):
        check(trial["passed"] and name[3:] == trial["case"], "Failed or mislabeled trial")
        check(trial["trace_sha256"] == manifest["pins"]["trace_sha256"], "Mixed traces")
        check(trial["image"] == manifest["pins"]["image"], "Mixed engine images")
        check(trial["swap_allowance_bytes"] == manifest["pins"]["swap_allowance_bytes"], "Mixed swap protocols")
        actual = tuple(trial[k] for k in ("mode", "cache_condition", "ballast_gib", "normal_reference"))
        check(actual == CASES[trial["case"]], "Case configuration does not match the fixed protocol")
        check(trial["physical_memory_bytes"] == manifest["physical_memory_bytes"], "Mixed physical budgets")
        if trial["mode"] != "resident":
            caches = [trial[d]["weight_cache_before_switch"] for d in ("A_to_B", "B_to_A")]
            if trial["mode"] == "restore":
                caches.append(trial["B_to_A"]["state_cache_before_switch"])
            for cache in caches:
                valid = cache["resident_fraction"] >= .999 if trial["cache_condition"] == "warm" else cache["resident_pages"] == 0
                check(valid, "Recorded file-cache precondition does not match the case")
    result = {"provenance": manifest, "trial_count": len(trials), "cases": {}}
    for case in manifest["cases"]:
        rows = [t for t in trials if t["case"] == case]
        check(len(rows) == manifest["repeats"], "Unequal repeat counts")
        group = {"n": len(rows), "mode": rows[0]["mode"], "ballast_gib": rows[0]["ballast_gib"],
                 "cache_condition": rows[0]["cache_condition"],
                 "normal_reference": rows[0]["normal_reference"],
                 "peak_spill_logical_bytes": span(t["peak_spill_logical_bytes"] for t in rows),
                 "peak_spill_allocated_bytes": span(t["peak_spill_allocated_bytes"] for t in rows),
                 "sampled_used_gib": span(t["memory"]["max_total_minus_available_bytes"]/GIB for t in rows),
                 "min_available_gib": span(t["memory"]["min_available_bytes"]/GIB for t in rows),
                 "trial_swap_bytes": span(t["trial_swap_traffic_bytes"] for t in rows),
                 "A_matching_outputs": sum(t["A_continuation_matches_trace"] for t in rows),
                 "B_matching_outputs": sum(t["B_output_matches_trace"] for t in rows)}
        for direction in ("A_to_B", "B_to_A"):
            switches = [t[direction] for t in rows]
            metrics = {k: span(s[k] for s in switches) for k in
                       ("first_token_s", "first_text_s", "eviction_and_load_s", "restore_api_s", "request_first_token_s")}
            metrics["save_durable_s"] = span(s.get("save_outgoing", {}).get("durable_elapsed_s", 0) for s in switches)
            for key in ("prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_per_second"):
                metrics[key] = span(s["completion"]["final"]["timings"][key] for s in switches)
            for key in ("read_bytes", "write_bytes", "pswpin", "pswpout", "oom_kill"):
                metrics["first_token_" + key] = span(s["io_through_first_token"][key] for s in switches)
            group[direction] = metrics
        result["cases"][case] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path, help="Aggregate JSON; no raw samples")
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("| Arm (n per arm) | A→B first token, s | B→A first token, s | A cached / prefilled tokens |")
    print("| --- | ---: | ---: | ---: |")
    def shown(s):
        return f"{s['median']:.3f} [{s['min']:.3f}–{s['max']:.3f}]"
    for name, row in result["cases"].items():
        a = row["B_to_A"]
        print(f"| {name} ({row['n']}) | {shown(row['A_to_B']['first_token_s'])} | "
              f"{shown(a['first_token_s'])} | {a['cache_n']['median']:g} / {a['prompt_n']['median']:g} |")


if __name__ == "__main__":
    main()
