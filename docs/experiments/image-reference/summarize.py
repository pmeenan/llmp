# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Reduce external measurement output to the report's aggregate evidence."""
import argparse
import json
from pathlib import Path


def summarize_case(result):
    steps = result["steps"]
    calls = result["calls"]
    components = {}
    for name in sorted({call["component"] for call in calls}):
        selected = [call for call in calls if call["component"] == name]
        components[name] = {"calls": len(selected), "seconds": sum(c["seconds"] for c in selected),
                            "cumulative_request_peak_bytes": max(c["cuda_peak_allocated"] for c in selected)}
    kv = [c["prefix_kv_bytes"] for c in calls if "prefix_kv_bytes" in c]
    block_calls = result["transformer_block_calls"]
    if steps and not all(step["finite"] for step in steps):
        raise ValueError("Nonfinite latent in source evidence")
    if not result["arguments"]["plain"] and (len(block_calls) != 32 or any(n != len(steps) for n in block_calls)):
        raise ValueError("Unexpected denoiser block coverage")
    return {"configuration": {k: v for k, v in result["arguments"].items() if k not in ("model", "output")},
            "load_seconds": result["load_seconds"], "run_seconds": result["run_seconds"],
            "cpu_load_seconds": result.get("cpu_load_seconds"),
            "transfer_seconds": result.get("transfer_seconds"),
            "pipeline_reused": result.get("pipeline_reused", False),
            "pipeline_released": result.get("pipeline_released", True),
            "after_load": result["after_load"], "after_run": result["after_run"],
            "after_cleanup": result.get("after_cleanup", result.get("after_release")),
            "component_calls": components, "completed_steps": len(steps) if steps else None,
            "latent_bytes": max((s["latent_bytes"] for s in steps), default=None),
            "prefix_kv_bytes": max(kv, default=None),
            "block_call_count_range": [min(block_calls), max(block_calls)],
            "output": result["output"], "input_rgb_sha256": result["input_rgb_sha256"],
            "phase_releases": result.get("phase_releases", []),
            "cancellation_return_seconds": result.get("cancellation_return_seconds"),
            "cancellation_cleanup_seconds": result.get("cancellation_cleanup_seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("switch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recompute", type=Path)
    parser.add_argument("--initial-recompute-mismatch", action="store_true")
    args = parser.parse_args()
    checks = json.loads((args.suite / "checks.json").read_text())
    if not checks or not all(checks.values()):
        raise ValueError("Suite checks did not pass")
    results = {p.parent.name: json.loads(p.read_text()) for p in sorted(args.suite.glob("*/result.json"))}
    if len(results) != 12:
        raise ValueError("Incomplete matrix")
    first = next(iter(results.values()))
    switch = json.loads((args.switch / "results.json").read_text())
    if args.recompute:
        if not args.initial_recompute_mismatch:
            raise ValueError("Explicitly identify the strict comparison failure before replacing its diagnostic")
        switch = [r for r in switch if r["mode"] != "recompute"]
        replacement = json.loads((args.recompute / "results.json").read_text())
        if len(replacement) != 1 or replacement[0]["mode"] != "recompute":
            raise ValueError("Expected one recompute diagnostic")
        controls = json.loads((args.recompute / "recompute" / "controls.json").read_text())
        actual = json.loads((args.recompute / "recompute" / "actual.json").read_text())
        resident = controls["resident"]["tokens"]
        matched = controls["matched_configuration"]["tokens"]
        if actual["tokens"] != matched:
            raise ValueError("Recompute diagnostic does not match its control")
        differences = [i for i in range(max(len(resident), len(matched)))
                       if i >= len(resident) or i >= len(matched) or resident[i] != matched[i]]
        replacement[0]["resident_token_mismatch_count"] = len(differences)
        replacement[0]["first_resident_token_mismatch_index"] = differences[0] if differences else None
        replacement[0]["generated_token_count"] = len(matched)
        switch += replacement
    if [r["mode"] for r in switch] != ["resident", "restore", "recompute"]:
        raise ValueError("Incomplete switching evidence")
    switches = []
    for row in switch:
        if not row["text_matches"] or not row["image_matches"]:
            raise ValueError("Switch equivalence failed")
        if row["image"]["output"]["pixels_sha256"] != results["plain-512-4"]["output"]["pixels_sha256"]:
            raise ValueError("Switch image differs from standalone control")
        switches.append({**{k: v for k, v in row.items() if k not in ("image", "text_matches")},
                         "text_matches_matched_configuration": row["text_matches"],
                         "text_matches_resident": row.get("text_matches_resident", row["text_matches"]),
                         "image": summarize_case(row["image"])})
    aggregate = {"host": "spark-c4e2", "date": "2026-09-22", "samples_per_case": 1,
                 "model_revision": first["model_revision"], "torch": first["torch"],
                 "cuda": first["cuda"], "device": first["device"], "components": first["components"],
                 "checks": checks, "suite_release": json.loads((args.suite / "release.json").read_text()),
                 "initial_recompute_resident_comparison_failed": args.initial_recompute_mismatch,
                 "recompute_is_separate_diagnostic": args.recompute is not None,
                 "cross_mode_text_equivalence_passed": all(r["text_matches_resident"] for r in switches),
                 "cases": {name: summarize_case(r) for name, r in results.items()}, "switch": switches}
    with args.output.open("x") as stream:
        json.dump(aggregate, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
