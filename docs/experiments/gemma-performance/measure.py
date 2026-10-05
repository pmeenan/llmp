#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one comparator and sample unified-memory availability every 50 ms.

OUTDIR COMMAND...; raw process logs and samples stay in the external OUTDIR.
This observer is outside the model runtime. MemAvailable includes reclaimable
page-cache behavior and unrelated system activity; it is not exact occupancy.
"""
import json
import pathlib
import subprocess
import sys
import time


def available():
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable absent")


def clocks():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,power.draw",
        "--format=csv,noheader", ], text=True).strip()


def main():
    if len(sys.argv) < 3:
        return 2
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    before_clocks = clocks()
    baseline = available()
    samples = [(0.0, baseline)]
    started = time.monotonic()
    with (out / "process.log").open("wb") as log:
        with subprocess.Popen(sys.argv[2:], stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                while process.poll() is None:
                    samples.append((time.monotonic() - started, available()))
                    time.sleep(0.05)
                result = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    samples.append((time.monotonic() - started, available()))
    summary = {
        "command": sys.argv[2:], "exit_code": result,
        "sample_period_seconds": 0.05, "mem_available_before": baseline,
        "sampled_mem_available_min": min(v for _, v in samples),
        "sampled_mem_available_drop": baseline - min(v for _, v in samples),
        "whole_process_seconds": time.monotonic() - started,
        "gpu_before": before_clocks, "gpu_after": clocks(),
    }
    (out / "samples.json").write_text(json.dumps(samples) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    print((out / "process.log").read_text(), end="", flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
