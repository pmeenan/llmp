#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Reduce one copied run directory to an aggregate; raw logs stay external.

Keeps the engine's own load/memory/pool/transport lines verbatim (without
timestamps) so the report can cite them, plus the node guard summaries and
the workload's measured cases without generated text beyond the smoke answers.
Usage: summarize.py RUN_DIR > aggregate.json
"""

import json
from pathlib import Path
import re
import sys

KEEP = [
    re.compile(p) for p in (
        r"Load weight (begin|end)", r"KV Cache is allocated", r"Memory pool end", r"max_total_num_tokens=",
        r"SWA", r"Capture cuda graph", r"NET/IB : Using", r"NCCL version", r"via NET/", r"Using network",
        r"init torch distributed", r"The server is fired up", r"flashinfer_mxfp4|mxfp4|MXFP4",
        r"Traceback|Error|error:|OOM|out of memory",
    )
]
STAMP = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[^\]]*\]?\s*")


def engine_lines(path, limit=80):
    """Distinct kept lines in first-seen order, with repeat counts."""
    first, counts = {}, {}
    for raw in path.read_text(errors="replace").splitlines():
        line = STAMP.sub("", raw).strip()[:400]
        if not any(p.search(line) for p in KEEP):
            continue
        # Per-channel NCCL and per-layer expert-preparation lines collapse to one shape.
        key = re.sub(r"\d+", "N", line) if ("via NET/" in line or "(layer:" in line) else line
        first.setdefault(key, line)
        counts[key] = counts.get(key, 0) + 1
    return [f"{first[k]} [x{n}]" if n > 1 else first[k] for k, n in counts.items()][:limit]


def main():
    run = Path(sys.argv[1])
    record = json.loads((run / "run.json").read_text())
    out = {"events": record["events"], "error": record.get("error"), "released": record.get("released"),
           "cleanup_errors": record.get("cleanup_errors", []),
           "not_released": record.get("not_released"), "preflight": record.get("preflight"),
           "post_boot_state": record.get("post_boot_state"), "server_args_head": record["server_args_head"],
           "nodes": {}}
    for node in sorted(p for p in run.iterdir() if p.is_dir()):
        entry = {}
        guard = node / "guard.json"
        if guard.is_file():
            g = json.loads(guard.read_text())
            g.pop("per_second_min_available", None)
            entry["guard"] = g
        for log in node.glob("llmp-mimo-rank*.log"):
            entry[log.stem] = engine_lines(log)
        workload = node / "workload.json"
        if workload.is_file():
            w = json.loads(workload.read_text())
            for case in w["cases"]:
                if not case["case"].startswith("smoke"):
                    case["text_chars"] = len(case.pop("text"))
            out["workload"] = w
        out["nodes"][node.name] = entry
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
