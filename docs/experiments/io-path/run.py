#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run sequentially on the Spark; outputs and the synthetic file stay in scratch."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("scratch", type=Path)
p.add_argument("phase", choices=["matrix", "contention", "sustained", "writes", "cache-pressure"])
a = p.parse_args()
root = a.scratch.resolve()
out = root / "results" / a.phase
out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, CUDA_DISABLE_PTX_JIT="1")
jobs = []
def add(mode, kib=2048, depth=4, seconds=3, cache="cold", pressure=0, load="none", repeat=1):
    for rep in range(1, repeat + 1):
        jobs.append((mode, kib, depth, seconds, cache, pressure, load, rep))

if a.phase == "matrix":
    for mode in ["buffered", "direct", "uring", "inplace", "hostvmm", "uring_hostvmm", "cufile"]:
        for q in [1, 4, 8]:
            add(mode, depth=q, repeat=3)
    add("buffered", cache="warm", repeat=3)
    for mode in ["direct", "uring_hostvmm"]:
        for kib in [64, 8192]:
            for q in [1, 4, 8]:
                add(mode, kib=kib, depth=q, seconds=2, repeat=3)
    for mode in ["direct", "inplace", "hostvmm"]:
        add(mode, kib=262144, depth=1, cache="resident", repeat=3)
    for mode in ["direct", "uring_hostvmm"]:
        for kib in [4, 16]:
            add(mode, kib=kib, depth=1, seconds=2, repeat=3)
        add(mode, depth=2, repeat=3)
    for mode in ["buffered", "direct", "hostvmm"]:
        add(mode, kib=64, depth=1, cache="sparse", repeat=3)
elif a.phase == "contention":
    for pressure, load in [(100, "none"), (0, "compute"), (0, "memory"), (100, "memory")]:
        for mode in ["buffered", "direct", "uring_hostvmm", "cufile"]:
            add(mode, pressure=pressure, load=load, repeat=2)
    for pressure, load in [(0, "compute"), (0, "memory"), (100, "memory")]:
        add("direct", depth=1, cache="control", pressure=pressure, load=load, repeat=2)
elif a.phase == "sustained":
    add("uring_hostvmm", seconds=180)
elif a.phase == "cache-pressure":
    add("buffered", seconds=60, pressure=100, repeat=2)
else:
    for mode in ["write", "hostvmm_write"]:
        add(mode, seconds=30, repeat=3)
random.Random(20260921).shuffle(jobs)
(out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
current = "setup"
stop = threading.Event()
def monitor():
    with (out / "telemetry.jsonl").open("w") as f:
        while not stop.is_set():
            row = {"time": time.time(), "job": current}
            row["disk_stat"] = Path("/sys/block/nvme0n1/stat").read_text().split()
            row["meminfo"] = {line.split(":")[0]: line.split(":")[1].strip()
                              for line in Path("/proc/meminfo").read_text().splitlines()}
            row["vmstat"] = {line.split()[0]: int(line.split()[1])
                             for line in Path("/proc/vmstat").read_text().splitlines()
                             if line.startswith(("pgscan_", "pgsteal_", "pswp"))}
            row["nvme_temperatures_mC"] = {}
            for hw in Path("/sys/class/hwmon").glob("hwmon*"):
                if (hw / "name").read_text().strip() == "nvme":
                    row["nvme_temperatures_mC"] = {x.name: int(x.read_text()) for x in hw.glob("temp*_input")}
            f.write(json.dumps(row) + "\n"); f.flush()
            stop.wait(1)
thread = threading.Thread(target=monitor)
thread.start()
try:
    for index, job in enumerate(jobs, 1):
        mode, kib, depth, seconds, cache, pressure, load, rep = job
        current = f"{mode}-{kib}k-q{depth}-{cache}-p{pressure}-{load}-r{rep}"
        file = root / (f"{current}.bin" if mode.endswith("write") else "data.bin")
        cmd = [str(root / "build/io-path"), str(root / "build/kernel.cubin"), str(file),
               mode, str(kib), str(depth), str(seconds), cache, str(pressure), load]
        print(f"{index}/{len(jobs)} {current}", flush=True)
        with (out / f"{current}.csv").open("x") as f, (out / f"{current}.stderr").open("x") as err:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=f, stderr=err, timeout=seconds + 180)
        if result.returncode:
            raise RuntimeError(f"failed {current}: {(out / f'{current}.stderr').read_text()}")
        if not (out / f"{current}.csv").read_text().endswith("complete,PASS\n"):
            raise RuntimeError(f"incomplete output {current}")
        if mode.endswith("write"):
            file.unlink()  # Only the uniquely named file created exclusively for this run.
finally:
    stop.set(); thread.join()
print("phase complete", flush=True)
