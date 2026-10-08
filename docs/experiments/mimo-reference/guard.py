#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-node memory sampler and userspace guard for the sharded reference.

Spark's CUDA allocations come from the same physical memory as the host and
are not bounded by a container cgroup. This samples MemAvailable and kills the
named container if it falls below the guard, before the node can wedge. It is
a reference-harness safety net, not a llmpalooza budget mechanism.
"""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0]) * 1024
    return values


def vmstat(keys=("pswpin", "pswpout", "oom_kill")):
    values = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {k: int(values.get(k, 0)) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--guard-gib", type=float, default=6.0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    guard = int(args.guard_gib * (1 << 30))
    start = meminfo()
    before = vmstat()
    minimum = start["MemAvailable"]
    minimum_at = 0.0
    trace = []  # one-second minima, for phase attribution
    bucket_min = None
    bucket = 0
    triggered = None
    killed = False
    last_attempt = None
    began = time.monotonic()
    while not stop:
        now = time.monotonic() - began
        available = meminfo()["MemAvailable"]
        if available < minimum:
            minimum, minimum_at = available, now
        if int(now) != bucket:
            if bucket_min is not None:
                trace.append([bucket, bucket_min])
            bucket, bucket_min = int(now), available
        else:
            bucket_min = available if bucket_min is None else min(bucket_min, available)
        # Retry (at most once a second) until a kill succeeds: a single failed
        # attempt under memory pressure must not disarm the guard.
        if available < guard and not killed and (last_attempt is None or now - last_attempt >= 1.0):
            last_attempt = now
            if triggered is None:
                triggered = {"seconds": round(now, 3), "mem_available": available, "kill_attempts": 0}
            triggered["kill_attempts"] += 1
            try:
                killed = subprocess.run(["sudo", "-n", "docker", "kill", args.container],
                                        check=False, timeout=10).returncode == 0
            except subprocess.TimeoutExpired:
                pass
            triggered["killed"] = killed
        time.sleep(args.interval)
    if bucket_min is not None:
        trace.append([bucket, bucket_min])
    after = vmstat()
    args.output.write_text(json.dumps({
        "mem_total": start["MemTotal"],
        "mem_available_start": start["MemAvailable"],
        "min_mem_available": minimum,
        "min_at_seconds": round(minimum_at, 3),
        "guard_bytes": guard,
        "guard_triggered": triggered,
        "vmstat_delta": {k: after[k] - before[k] for k in after},
        "per_second_min_available": trace,
    }) + "\n")


if __name__ == "__main__":
    main()
