# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate complete probe output and summarize per-run, untrimmed samples."""
import csv
import math
from pathlib import Path
import statistics
import sys


def summarize(path):
    rows = list(csv.reader(path.read_text().splitlines()))
    if not rows or rows[-1] != ["PASS"]:
        raise ValueError(f"{path}: incomplete/failed run")
    metadata = rows[0]
    if metadata[0] != "device":
        raise ValueError("missing device metadata")
    minimum = int(metadata[metadata.index("minimum") + 1])
    sizes = sorted({(s + minimum - 1) // minimum * minimum
                    for s in [minimum, 2 << 20, 8 << 20, 32 << 20, 128 << 20]})
    groups = {}
    for row in rows:
        if row[0] == "sample":
            _, mode, size, iteration, operation, host_us, kernel_ms, pending = row
            key = mode, int(size), operation
            group = groups.setdefault(key, {})
            if int(iteration) in group:
                raise ValueError("duplicate sample")
            values = float(host_us), float(kernel_ms), int(pending)
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("invalid measurement")
            if values[2] not in (0, 1):
                raise ValueError("invalid pending flag")
            group[int(iteration)] = values
    expected = {(mode, size, op)
                for mode in ("idle", "one_block", "sm_count_blocks")
                for size in sizes
                for op in ("create", "map", "access", "unmap", "release")}
    if groups.keys() != expected:
        raise ValueError("missing/unexpected sample groups")
    print(f"\n{path.name}: minimum={minimum}, recommended={metadata[-1]}")
    print("mode,MiB,operation,n,median_us,p95_us,max_us,median_kernel_ms,pending_after_count")
    for (mode, size, operation), group in groups.items():
        count = 100 if mode == "idle" else 30
        if set(group) != set(range(count)):
            raise ValueError("missing/unexpected iteration")
        times = sorted(v[0] for v in group.values())
        kernel = [v[1] for v in group.values()]
        if mode == "idle" and any(v[1] or v[2] for v in group.values()):
            raise ValueError("idle sample reports kernel activity")
        if mode != "idle" and not all(t >= 9 for t in kernel):
            raise ValueError("unexpectedly short background kernel")
        print(f"{mode},{size / (1 << 20):g},{operation},{count},"
              f"{statistics.median(times):.3f},{times[math.ceil(.95 * count)-1]:.3f},"
              f"{max(times):.3f},{statistics.median(kernel):.3f},"
              f"{sum(v[2] for v in group.values())}")
    phases = [r for r in rows if r[0] == "memory"]
    if [r[1] for r in phases] != ["baseline", "address_reserved", "created",
            "mapped_touched", "unmapped_handles_retained", "remapped_verified",
            "handles_released", "address_freed"]:
        raise ValueError("missing/unexpected memory phases")
    if ["verification_negative_control", "PASS"] not in rows:
        raise ValueError("missing verifier negative control")
    if ["pool_verified_bytes", str(16 * ((64 * (1 << 20) + minimum - 1) // minimum * minimum))] not in rows:
        raise ValueError("missing pool verification")
    baseline = int(phases[0][2])
    print("memory_phase,cuda_free_delta_MiB,MemAvailable_delta_MiB")
    for row in phases:
        print(f"{row[1]},{(int(row[2]) - baseline) / (1 << 20):.3f},"
              f"{(int(row[4]) - int(phases[0][4])) / (1 << 20):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: summarize.py RUN.csv [RUN.csv ...]")
    for argument in sys.argv[1:]:
        summarize(Path(argument))
