#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""sudo python3 trace.py start|stop; stop emits evidence JSON to stdout."""
import argparse
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument("action", choices=["start", "stop"])
a = p.parse_args()
root = Path("/sys/kernel/tracing/instances/llmp_io_20260921")
if a.action == "start":
    root.mkdir()
    (root / "buffer_size_kb").write_text("256")
    (root / "events/swiotlb/swiotlb_bounced/enable").write_text("1")
    (root / "tracing_on").write_text("1")
    (root / "trace_marker").write_text("llmp IO comparison trace begin")
else:
    if "llmp IO comparison trace begin" not in (root / "trace").read_text():
        raise RuntimeError("Missing owned trace start marker; do not remove instance")
    enabled = (root / "events/swiotlb/swiotlb_bounced/enable").read_text().strip()
    tracing = (root / "tracing_on").read_text().strip()
    (root / "trace_marker").write_text("llmp IO comparison trace end")
    (root / "tracing_on").write_text("0")
    trace = (root / "trace").read_text()
    stats = {cpu.name: (cpu / "stats").read_text() for cpu in (root / "per_cpu").glob("cpu*")}
    evidence = {"captured_at": time.time(), "event_enabled": enabled, "tracing_on": tracing,
                "trace": trace, "cpu_stats": stats,
                "bounce_events": sum("swiotlb_bounced:" in line for line in trace.splitlines())}
    print(json.dumps(evidence, indent=2))
    (root / "events/swiotlb/swiotlb_bounced/enable").write_text("0")
    root.rmdir()
