#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate scratch output, then retain lossless CSV blocks in one file per phase."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("destination", type=Path)
a = p.parse_args()
subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), str(a.source)], check=True)
a.destination.mkdir(parents=True, exist_ok=False)
for phase in sorted(a.source.iterdir()):
    if not phase.is_dir() or not (phase / "jobs.json").exists():
        continue
    out = a.destination / phase.name
    out.mkdir()
    for name in ["jobs.json", "telemetry.jsonl", "host-counters.jsonl"]:
        if (phase / name).exists():
            shutil.copyfile(phase / name, out / name)
    with (out / "measurements.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for source in sorted(phase.glob("*.csv")):
            assert not source.with_suffix(".stderr").read_text().strip()
            writer.writerow(["job", source.stem])
            writer.writerows(csv.reader(source.open()))
    (out / "capture.json").write_text(json.dumps({"stderr": "all source stderr files checked empty",
        "packing": "CSV rows preserved; job marker precedes each original output", "jobs": len(json.loads((out / "jobs.json").read_text()))}, indent=2) + "\n")
for name in ["dma-trace.json", "coalescing-feature.json"]:
    if (a.source / name).exists():
        shutil.copyfile(a.source / name, a.destination / name)
subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), str(a.destination)], check=True)
