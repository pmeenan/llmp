#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize a host-VMM diagnosis session (README.md in this directory).

  summarize.py SESSION_DIR [--ncu NCU_DIR] [--json OUT.json] > tables.md

Reads the CSVs and manifest.json that diag_session.py wrote and prints
Markdown tables: the microkernels' bandwidth per memory arm, copy bandwidth
and 2 MiB copy latency, and each GGML product's time per buffer placement
with its ratio to all-cudaMalloc. Every figure is the median over all
samples of all processes; the range is that of the per-process medians. With
--ncu it adds the L2 counters ncu_counters.sh collected; with --json it also
writes the aggregates. Stops if a GGML case's output hash
differs between placements or processes.
"""

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path


def rows(path):
    with open(path) as f:
        return list(csv.reader(line for line in f if line.strip() and not line.startswith("#")))


def aggregate(groups):
    """{key: {process: [values]}} -> {key: (median, low, high, n, processes)}"""
    out = {}
    for key, by_process in groups.items():
        values = [v for vs in by_process.values() for v in vs]
        per_process = [statistics.median(vs) for vs in by_process.values()]
        out[key] = (statistics.median(values), min(per_process), max(per_process), len(values),
                    len(by_process))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.session / "manifest.json").read_text())
    result = {"manifest": {k: v for k, v in manifest.items() if k != "processes"},
              "processes": len(manifest["processes"]),
              "conditions": {}}
    states = [p[w] for p in manifest["processes"] for w in ("before", "after")]
    sm = [int(s["clocks.sm"].split()[0]) for s in states]
    temp = [int(s["temperature.gpu"]) for s in states]
    throttle = sorted({s["clocks_throttle_reasons.active"] for s in states})
    loads = [p["before"]["load"] for p in manifest["processes"]]
    result["conditions"] = {"sm_mhz": [min(sm), max(sm)], "gpu_c": [min(temp), max(temp)],
                            "throttle_reasons": throttle, "load_before": [min(loads), max(loads)],
                            "first": manifest["processes"][0]["before"]["at"],
                            "last": manifest["processes"][-1]["after"]["at"]}
    print(f"Conditions: {manifest['host']}, driver {manifest['gpu']['driver_version']}, "
          f"{len(manifest['processes'])} processes {result['conditions']['first']} to "
          f"{result['conditions']['last']}; SM {min(sm)}-{max(sm)} MHz at process boundaries, "
          f"GPU {min(temp)}-{max(temp)} C, throttle reasons {throttle}, load before each "
          f"{min(loads):.2f}-{max(loads):.2f}.\n")

    info = args.session / "info.csv"
    if info.exists():
        result["info"] = [",".join(r[1:]) for r in rows(info)]

    micro = collections.defaultdict(lambda: collections.defaultdict(list))
    moved = {}
    arms = []
    for path in sorted(args.session.glob("micro-*.csv")):
        for r in rows(path):
            _, arm, test, nbytes, _, _, us = r
            micro[(arm, test)][path.stem].append(int(nbytes) / float(us) / 1e3)
            moved[test] = int(nbytes)
            if arm not in arms:
                arms.append(arm)
    if micro:
        agg = aggregate(micro)
        tests = list(dict.fromkeys(t for _, t in micro))
        result["micro_gbps"] = {f"{a}|{t}": agg[(a, t)] for a, t in agg}
        print("Microkernels, GB/s (median of all samples; range of per-process medians):\n")
        print("| Arm | " + " | ".join(f"`{t}`" for t in tests) + " |")
        print("| --- |" + " ---: |" * len(tests))
        for arm in sorted(arms, key=ARM_ORDER.index):
            cells = []
            for t in tests:
                m, lo, hi, _, _ = agg[(arm, t)]
                cells.append(f"{m:.0f} ({lo:.0f}-{hi:.0f})")
            print(f"| `{arm}` | " + " | ".join(cells) + " |")
        n = {agg[k][3] for k in agg}
        p = {agg[k][4] for k in agg}
        print(f"\nSamples per cell: {sorted(n)} from {sorted(p)} processes.\n")

    copies = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(args.session.glob("copy-*.csv")):
        for r in rows(path):
            _, source, destination, method, nbytes, _, _, us = r
            copies[(source, destination, method, int(nbytes))][path.stem].append(float(us))
    if copies:
        agg = aggregate(copies)
        result["copy_us"] = {"|".join(map(str, k)): v for k, v in agg.items()}
        print("Copies (median; range of per-process medians):\n")
        print("| From | To | Method | 2 MiB, µs | 2 MiB, GB/s | 64 MiB, GB/s | 1 GiB, GB/s |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        pairs = list(dict.fromkeys((s, d, m) for s, d, m, _ in copies))
        for s, d, m in pairs:
            cells = []
            small = agg[(s, d, m, 2 << 20)]
            cells.append(f"{small[0]:.1f} ({small[1]:.1f}-{small[2]:.1f})")
            for size in (2 << 20, 64 << 20, 1 << 30):
                med, lo, hi, _, _ = agg[(s, d, m, size)]
                cells.append(f"{size / med / 1e3:.0f} ({size / hi / 1e3:.0f}-{size / lo / 1e3:.0f})")
            print(f"| `{s}` | `{d}` | {m} | " + " | ".join(cells) + " |")
        n = {agg[k][3] for k in agg}
        p = {agg[k][4] for k in agg}
        print(f"\nSamples per cell: {sorted(n)} from {sorted(p)} processes.\n")

    restores = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(args.session.glob("restore-*.csv")):
        for r in rows(path):
            _, landing, destination, method, depth, _, gbps, p50, p99, _ = r
            key = (method, landing, destination, int(depth))
            restores[key + ("gbps",)][path.stem].append(float(gbps))
            restores[key + ("p50",)][path.stem].append(float(p50))
            restores[key + ("p99",)][path.stem].append(float(p99))
    if restores:
        agg = aggregate(restores)
        result["restore"] = {"|".join(map(str, k)): v for k, v in agg.items()}
        print("Restores of an 8 GiB file in 2 MiB direct reads through llmpalooza's io_uring "
              "provider (median of passes; range of per-process medians; latency from a read's "
              "submission to its extent being usable in the destination):\n")
        print("| Path | In flight | GB/s | p50 µs | p99 µs |")
        print("| --- | ---: | ---: | ---: | ---: |")
        names = {"none": "in place into `hvmm`", "memcpy": "`hvmm` landing, copy engine to `dvmm`",
                 "kernel": "`hvmm` landing, SM copy kernel to `dvmm`"}
        keys = sorted({k[:4] for k in restores}, key=lambda k: (k[3], list(names).index(k[0])))
        for k in keys:
            g, p50, p99 = (agg[k + (m,)] for m in ("gbps", "p50", "p99"))
            print(f"| {names[k[0]]} | {k[3]} | {g[0]:.3f} ({g[1]:.3f}-{g[2]:.3f}) | "
                  f"{p50[0]:.0f} ({p50[1]:.0f}-{p50[2]:.0f}) | {p99[0]:.0f} ({p99[1]:.0f}-{p99[2]:.0f}) |")
        n = {agg[k][3] for k in agg}
        p = {agg[k][4] for k in agg}
        print(f"\nPasses per row: {sorted(n)} from {sorted(p)} processes.\n")

    ggml = collections.defaultdict(lambda: collections.defaultdict(list))
    hashes = collections.defaultdict(set)
    placements = []
    for path in sorted(args.session.glob("ggml-*.csv")):
        for r in rows(path):
            _, placement, case, nrows, _, _, _, fnv, us = r
            key = f"{case}@{nrows}"
            ggml[(placement, key)][path.stem].append(float(us))
            hashes[key].add(fnv)
            if placement not in placements:
                placements.append(placement)
    if ggml:
        bad = {k: v for k, v in hashes.items() if len(v) != 1}
        if bad:
            sys.exit(f"output hashes differ: {bad}")
        agg = aggregate(ggml)
        result["ggml_us"] = {f"{p}|{c}": v for (p, c), v in agg.items()}
        cases = list(dict.fromkeys(c for _, c in ggml))
        base = PLACEMENT_ORDER[0]
        order = sorted(placements, key=lambda p: PLACEMENT_ORDER.index(p)
                       if p in PLACEMENT_ORDER else len(PLACEMENT_ORDER))
        print("GGML products: all-cudaMalloc median µs, and each placement's median as a ratio "
              "to it (W weights or KV cache, A activations, S GGML scratch, K cuBLAS "
              "workspace):\n")
        short = [p for p in order if p != base]
        print("| Case | all `malloc`, µs | " + " | ".join(f"`{p}`" for p in short) + " |")
        print("| --- | ---: |" + " ---: |" * len(short))
        for c in cases:
            ref = agg[(base, c)][0]
            cells = [f"{agg[(p, c)][0] / ref:.2f}" for p in short]
            print(f"| `{c}` | {ref:.1f} | " + " | ".join(cells) + " |")
        n = {agg[k][3] for k in agg}
        p = {agg[k][4] for k in agg}
        spread = max((agg[k][2] - agg[k][1]) / agg[k][0] for k in agg)
        print(f"\nSamples per cell: {sorted(n)} from {sorted(p)} processes; the largest range of "
              f"per-process medians is {100 * spread:.1f}% of its median. Every case's output hash "
              f"is the same under every placement.\n")
        per_token(agg, base)
    if args.ncu:
        result["ncu"] = ncu(args.ncu)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")


def per_token(agg, base):
    """Derived, not measured: the covered kernels' medians summed per token."""
    layers = 24
    projections = (("q_o", 2), ("k_v", 2), ("gate_up", 2), ("down", 1))
    attention = {1: "kv768", 16: "kv256", 512: "kv768"}
    device = "W=dvmm/A=dvmm/S=dvmm/K=dvmm"
    rows = {
        "all `malloc`": (base, base),
        "all `dvmm`": (device, device),
        "all `hvmm`": ("W=hvmm/A=hvmm/S=hvmm/K=hvmm",) * 2,
        "weights and KV cache `hvmm`, the rest `dvmm`": ("W=hvmm/A=dvmm/S=dvmm/K=dvmm",) * 2,
        "weights `hvmm`, KV cache and the rest `dvmm`": ("W=hvmm/A=dvmm/S=dvmm/K=dvmm", device),
        "weights `registered-thp`, the rest `dvmm`": ("W=registered-thp/A=dvmm/S=dvmm/K=dvmm",) * 2,
    }

    def total(linear, attn, n):
        us = sum(layers * count * agg[(linear, f"linear.{name}@{n}")][0]
                 for name, count in projections)
        us += layers * sum(agg[(attn, f"attn.{op}.{attention[n]}@{n}")][0] for op in ("kq", "kqv"))
        return us + agg[(linear, f"linear.lm_head@{n}")][0]

    print("Per token, DERIVED (sums of the medians above, not a measured token): per layer two "
          "q_o, two k_v, two gate_up, one down, one KQ and one KQV (KV 768 cells at 1 and 512 "
          "rows, 256 at 16), times 24 layers, plus one output head:\n")
    print("| Placement | 1 row | 16 rows | 512 rows |")
    print("| --- | ---: | ---: | ---: |")
    reference = {n: total(base, base, n) for n in attention}
    for label, (linear, attn) in rows.items():
        cells = [f"{total(linear, attn, n) / reference[n]:.3f}×" for n in attention]
        if linear == base:
            cells = [f"{reference[n]:,.0f} µs" for n in attention]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()


def ncu(directory):
    """Per profiled kernel: L2 read sectors, lookup hits and duration."""
    out = {}
    print("L2 counters (Nsight Compute; each row one profiled kernel launch, or the listed "
          "launches' totals):\n")
    print("| Run | Kernel | Launches | Read sectors | L2 hits | Hit rate | Duration µs |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for path in sorted(directory.glob("*.csv")):
        with open(path) as f:
            lines = [line for line in f if line.startswith('"')]
        launches = collections.defaultdict(dict)
        names = {}
        for r in csv.DictReader(lines):
            value = float(r["Metric Value"].replace(",", ""))
            launches[r["ID"]][r["Metric Name"]] = value
            names[r["ID"]] = r["Kernel Name"].split("(")[0].replace("void ", "")[:48]
        by_kernel = collections.defaultdict(list)
        for launch_id, metrics in launches.items():
            by_kernel[names[launch_id]].append(metrics)
        for kernel, ms in by_kernel.items():
            read = sum(m["lts__t_sectors_srcunit_tex_op_read.sum"] for m in ms)
            hits = sum(m["lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_hit.sum"] +
                       m["lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum"]
                       for m in ms)
            sysmem = sum(m["lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum"] for m in ms)
            duration = statistics.median(m["gpu__time_duration.sum"] for m in ms) / 1e3
            out[f"{path.stem}|{kernel}"] = {"launches": len(ms), "read_sectors": read,
                                            "sysmem_read_sectors": sysmem, "hits": hits,
                                            "median_us": duration}
            rate = f"{100 * hits / read:.1f}%" if read else "-"
            print(f"| {path.stem} | `{kernel}` | {len(ms)} | {read:,.0f} | {hits:,.0f} | {rate} | "
                  f"{duration:.1f} |")
    print()
    return out


ARM_ORDER = ["malloc", "dvmm", "hvmm", "hvmm-jit", "hvmm-gpu", "hvmm-2m", "hvmm-host", "pinned",
             "registered", "pageable", "pageable-thp", "registered-thp", "managed",
             "managed-prefetch"]

PLACEMENT_ORDER = [
    "W=malloc/A=malloc/S=malloc/K=malloc",
    "W=dvmm/A=dvmm/S=dvmm/K=dvmm",
    "W=hvmm/A=hvmm/S=hvmm/K=hvmm",
    "W=hvmm-jit/A=hvmm-jit/S=hvmm-jit/K=hvmm-jit",
    "W=hvmm/A=malloc/S=malloc/K=malloc",
    "W=hvmm/A=dvmm/S=dvmm/K=dvmm",
    "W=hvmm-2m/A=dvmm/S=dvmm/K=dvmm",
    "W=malloc/A=hvmm/S=malloc/K=malloc",
    "W=malloc/A=hvmm/S=malloc/K=malloc/O=malloc",
    "W=malloc/A=malloc/S=malloc/K=malloc/O=hvmm",
    "W=malloc/A=malloc/S=hvmm/K=hvmm",
    "W=registered-thp/A=dvmm/S=dvmm/K=dvmm",
]

if __name__ == "__main__":
    main()
