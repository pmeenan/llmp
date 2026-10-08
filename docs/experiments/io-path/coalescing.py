#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, temporary controller-feature A/B test; restore the observed value."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("scratch", type=Path)
a = p.parse_args()
root = a.scratch.resolve()
env = dict(os.environ, CUDA_DISABLE_PTX_JIT="1", LC_ALL="C")
def get_feature():
    text = subprocess.check_output(["sudo", "-n", "nvme", "get-feature", "/dev/nvme0", "-f", "8", "-H"], text=True, env=env, timeout=20)
    found = re.search(r"Current value:(?:0x)?([0-9a-fA-F]+)", text)
    if not found:
        raise RuntimeError(text)
    return int(found[1], 16), text

original, original_text = get_feature()
if original == 0:
    raise RuntimeError("Controller already has coalescing disabled; no A/B baseline")
shapes = [("uring_hostvmm", 4, 1), ("uring_hostvmm", 64, 1),
          ("uring_hostvmm", 2048, 1), ("uring_hostvmm", 2048, 4),
          ("uring_hostvmm", 8192, 4), ("buffered", 2048, 4)]
outputs = {}
for value in [original, 0]:
    out = root / "results" / f"coalescing-{value}"
    out.mkdir(exist_ok=False)
    jobs = [(mode, kib, q, 2, "cold", 0, "none", rep)
            for rep in range(1, 4) for mode, kib, q in shapes]
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    outputs[value] = out
audit = root / "results/coalescing-feature.json"
record = {"original": original, "original_text": original_text, "changes": [], "restored": False}
with audit.open("x") as f:
    f.write(json.dumps(record, indent=2) + "\n")

def set_feature(value):
    result = subprocess.check_output(["sudo", "-n", "nvme", "set-feature", "/dev/nvme0", "--feature-id=8", f"--value={value}"], text=True, env=env, timeout=20)
    observed, description = get_feature()
    record["changes"].append({"time": time.time(), "requested": value, "observed": observed, "set_output": result, "get_output": description})
    audit.write_text(json.dumps(record, indent=2) + "\n")
    if observed != value:
        raise RuntimeError("Feature readback mismatch")

def interrupted(_sig, _frame):
    raise KeyboardInterrupt()
termination_signals = [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]
for sig in termination_signals:
    signal.signal(sig, interrupted)

def counters():
    irq = 0
    for line in Path("/proc/interrupts").read_text().splitlines():
        if "nvme0q" in line:
            for value in line.split()[1:]:
                if not value.isdigit():
                    break
                irq += int(value)
    cpu = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return {"time": time.time(), "nvme_interrupts": irq, "cpu_ticks": cpu}

try:
    for rep in range(1, 4):
        # Alternate baseline/tuned within each pair; restore even if a probe fails.
        for value in [original, 0]:
            set_feature(value)
            out = outputs[value]
            for mode, kib, q in shapes:
                name = f"{mode}-{kib}k-q{q}-cold-p0-none-r{rep}"
                print(f"feature={value} {name}", flush=True)
                before = counters()
                cmd = [str(root / "build/io-path"), str(root / "build/kernel.cubin"), str(root / "data.bin"),
                       mode, str(kib), str(q), "2", "cold", "0", "none"]
                with (out / f"{name}.csv").open("x") as stdout, (out / f"{name}.stderr").open("x") as stderr:
                    result = subprocess.run(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr, timeout=60)
                after = counters()
                with (out / "host-counters.jsonl").open("a") as f:
                    f.write(json.dumps({"job": name, "before": before, "after": after}) + "\n")
                if result.returncode or not (out / f"{name}.csv").read_text().endswith("complete,PASS\n"):
                    raise RuntimeError(f"failed {name}")
finally:
    for sig in termination_signals:
        signal.signal(sig, signal.SIG_IGN)
    set_feature(original)
    record["restored"] = get_feature()[0] == original
    audit.write_text(json.dumps(record, indent=2) + "\n")
print("coalescing comparison complete; original value restored", flush=True)
