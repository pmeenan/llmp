#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Reduce a suite directory to aggregates.json; raw images and logs stay external.

Run in the pinned image-reference container (numpy, Pillow) with the suite and
the BF16 diffusers study outputs mounted read-only.
Usage: summarize.py SUITE_DIR DIFFUSERS_SUITE_DIR [CROSS_NODE_DIR] > aggregates.json
"""

import json
from pathlib import Path
import sys

import compare

# Diffusers BF16 baseline case directories, keyed by (size, steps).
DIFFUSERS = {(512, 4): "plain-512-4", (1024, 4): "t2i-1024-4", (1024, 40): "t2i-1024-40", (2048, 40): "t2i-2048-40"}
BF16_CONTROL = {(512, 4): "bf16-512-4", (1024, 4): "bf16-1024-4-a", (1024, 40): "bf16-1024-40",
                (2048, 40): "bf16-2048-40"}
PAIRS = [  # same-configuration identity/repeat controls
    ("q4-1024-4-a", "q4-1024-4-cold"), ("q4-1024-4-b", "q4-1024-4-a"), ("bf16-1024-4-b", "bf16-1024-4-a"),
    ("q4-1024-40-phase", "q4-1024-40"), ("q4-1024-4-disk", "q4-1024-4-a"),
    ("q4-1024-4-disk-3g", "q4-1024-4-a"), ("q4-1024-4-disk-3g-cold", "q4-1024-4-a"),
    ("q4-1024-4-disk-2g", "q4-1024-4-a"), ("q4-1024-4-vae-tiled", "q4-1024-4-a"),
    ("q4-1024-4-disk-3g", "q4-1024-4-vae-tiled"),
]
# Second-node runs (patched rebuild and upstream-GGML build) against the main suite.
CROSS_NODE = {"patched-q4-1024-4": "q4-1024-4-a", "up-q4-512-4": "q4-512-4", "up-q4-1024-4": "q4-1024-4-a",
              "up-q4-1024-40": "q4-1024-40", "up-q4-2048-40": "q4-2048-40"}
SEGMENT_WORDS = ("segment", "evict", "prefetch", "budget", "reload", "release")


def reduce_case(directory):
    result = json.loads((directory / "result.json").read_text())
    t = result["timings"]
    one = lambda key: t[key][0][0] if len(t[key]) == 1 else [row[0] for row in t[key]]
    log = (directory / "sd-cli.log").read_text(errors="replace")
    params = [l.split("] ", 1)[-1] for l in log.splitlines() if "total params memory size" in l]
    buffers = sorted({l.split("] ", 1)[-1].split(" - ", 1)[-1] for l in log.splitlines() if "compute buffer size" in l})
    segment_lines = [l.split("] ", 1)[-1] for l in log.splitlines()
                     if any(w in l.lower() for w in SEGMENT_WORDS) and "available_uma_memory" not in l]
    return {
        "case": result["case"],
        "exit_code": result["exit_code"],
        "image_sha256": result["image_sha256"],
        "outward_seconds": result["outward_seconds"],
        "load_seconds": [row[0] for row in t["load"]],
        "load_read_seconds": [row[1] for row in t["load"]],
        "condition_seconds": one("condition") if t["condition"] else None,
        "sampling_seconds": one("sampling") if t["sampling"] else None,
        "decode_seconds": one("decode") if t["decode"] else None,
        "generate_seconds": one("generate") if t["generate"] else None,
        "params_line": params[0] if params else None,
        "compute_buffers": buffers,
        "segment_log_lines": len(segment_lines),
        "segment_log_sample": segment_lines[:12],
        "page_cache_before": result["page_cache_before"],
        "cgroup_read_bytes": result["cgroup_read_bytes"],
        "cgroup_memory_peak": result["cgroup_memory_peak"],
        "max_host_used_delta": result["max_host_used_delta"],
        "vmstat_delta": result["vmstat_delta"],
        "sd_cli_args": result["sd_cli_args"],
    }


def main():
    suite, diffusers = Path(sys.argv[1]), Path(sys.argv[2])
    cases = {d.name: reduce_case(d) for d in sorted(suite.iterdir()) if (d / "result.json").is_file()}
    comparisons = []

    def add(kind, candidate, reference, cand_path, ref_path):
        if cand_path.is_file() and ref_path.is_file():
            row = compare.compare(str(cand_path), str(ref_path))
            row.update(kind=kind, candidate=candidate, reference=reference)
            comparisons.append(row)

    for name, case in cases.items():
        key = (case["case"]["size"], case["case"]["steps"])
        if case["case"]["placement"] != "resident" or name.endswith(("-b", "-cold", "-tiled")):
            continue
        if case["case"]["denoiser"] != "bf16" and key in BF16_CONTROL:
            add("quantized-vs-sdcpp-bf16", name, BF16_CONTROL[key], suite / name / "image.png",
                suite / BF16_CONTROL[key] / "image.png")
        if key in DIFFUSERS:
            add("sdcpp-vs-diffusers-bf16", name, f"diffusers/{DIFFUSERS[key]}", suite / name / "image.png",
                diffusers / DIFFUSERS[key] / "image.png")
    for candidate, reference in PAIRS:
        add("same-configuration-control", candidate, reference, suite / candidate / "image.png",
            suite / reference / "image.png")
    cross = {}
    if len(sys.argv) > 3:
        other = Path(sys.argv[3])
        for name, reference in CROSS_NODE.items():
            if (other / name / "result.json").is_file():
                cross[name] = reduce_case(other / name)
                add("cross-node-control", f"{other.name}/{name}", reference, other / name / "image.png",
                    suite / reference / "image.png")
    json.dump({"cases": cases, "cross_node_cases": cross, "comparisons": comparisons}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
