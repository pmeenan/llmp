# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize a dmabuf-direct session directory (session.sh's OUTDIR).

Prints the report's tables and optionally writes the aggregates as JSON.
Run with `python3 -B`.
"""

import argparse
import collections
import json
import pathlib
import statistics


def per_process_micro(raw):
    """(arm, test) -> list of per-process (median, min, max, us_median)."""
    cells = collections.defaultdict(list)
    content = collections.defaultdict(list)
    coherence = []
    for path in sorted(raw.glob("micro-4k-r*.txt")):
        for line in path.read_text().splitlines():
            parts = line.split(",")
            if line.startswith("micro,") and "content_bad=" in line:
                content[parts[1]].append(int(parts[2].split("=")[1]))
            elif line.startswith("micro,"):
                arm, test = parts[1], parts[2]
                cells[(arm, test)].append(
                    {"samples": int(parts[3]), "gbps": float(parts[4]), "min": float(parts[5]),
                     "max": float(parts[6]), "us": float(parts[7])})
            elif line.startswith("coherence,"):
                fields = dict(p.split("=") for p in parts[2:])
                coherence.append({"arm": parts[1], **{k: int(v) for k, v in fields.items()}})
    return cells, content, coherence


def restores(raw):
    """(mode, depth) -> {"passes": [...], "processes": [[...], ...], "negative": [...]}"""
    rows = collections.defaultdict(lambda: {"passes": [], "processes": [], "negative": []})
    for path in sorted(raw.glob("restore-*.txt")):
        mine = []
        negative = None  # stays None if the process never reached its control
        for line in path.read_text().splitlines():
            parts = line.split(",")
            if not line.startswith("restore,"):
                continue
            if "negative_control_bad=" in line:
                negative = int(parts[2].split("=")[1])
                continue
            if len(parts) < 12:
                raise SystemExit(f"{path}: {line}")
            mode, depth, index = parts[1], int(parts[2]), int(parts[3])
            if index == 0:  # warm-up pass
                continue
            mine.append({"gbps": float(parts[5]), "p50": float(parts[6]), "p99": float(parts[7]),
                         "max": float(parts[8]), "cpu_s": float(parts[9]), "bad": int(parts[10])})
        rows[(mode, depth)]["passes"] += mine
        rows[(mode, depth)]["processes"].append(mine)
        rows[(mode, depth)]["negative"].append(negative)
    return rows


def fmt_range(values, digits):
    return f"{min(values):.{digits}f}-{max(values):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    out = {}

    cells, content, coherence = per_process_micro(args.raw)
    arms = ["dvmm", "malloc", "hvmm", "udmabuf-4k"]
    tests = []
    for _, test in cells:
        if test not in tests:
            tests.append(test)
    print("## Microkernels, GB/s: median of per-process medians (range of per-process medians)\n")
    print("| Arm | " + " | ".join(f"`{t}`" for t in tests) + " |")
    print("| --- |" + " ---: |" * len(tests))
    out["micro"] = {}
    for arm in arms:
        row = []
        for test in tests:
            procs = cells[(arm, test)]
            meds = [p["gbps"] for p in procs]
            digits = 1 if max(meds) < 20 else 0
            row.append(f"{statistics.median(meds):.{digits}f} ({fmt_range(meds, digits)})")
            out["micro"].setdefault(arm, {})[test] = {
                "gbps_median_of_process_medians": statistics.median(meds),
                "process_medians": meds,
                "us_median_of_process_medians": statistics.median(p["us"] for p in procs),
                "samples_per_process": procs[0]["samples"], "processes": len(procs)}
        print(f"| `{arm}` | " + " | ".join(row) + " |")
    print()
    for arm in arms:
        us = statistics.median(p["us"] for p in cells[(arm, "write-per-block")])
        print(f"write-per-block median us, {arm}: {us:.0f}")
    print("content bad words:", dict(content))
    out["micro_content_bad"] = dict(content)
    agg = collections.defaultdict(lambda: collections.Counter())
    for c in coherence:
        agg[c["arm"]].update({k: v for k, v in c.items() if k not in ("arm", "bytes")})
        agg[c["arm"]]["bytes_per_round"] = c["bytes"]
    print("coherence totals:", {k: dict(v) for k, v in agg.items()})
    out["coherence"] = {k: dict(v) for k, v in agg.items()}

    rows = restores(args.raw)
    print("\n## Restores, 8 GiB in 2 MiB direct reads (passes 1-3 of each process)\n")
    print("| Path | In flight | GB/s | p50 µs | p99 µs | passes | bad words | negative control |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    out["restore"] = {}
    for (mode, depth) in sorted(rows, key=lambda k: (k[1], ["hvmm-inplace", "land-ce", "land-sm", "udmabuf-4k"].index(k[0]))):
        r = rows[(mode, depth)]
        p = r["passes"]
        def med(key):
            return statistics.median(x[key] for x in p)
        def rng(key, digits):
            return fmt_range([statistics.median(x[key] for x in proc) for proc in r["processes"]], digits)
        bad = sum(x["bad"] for x in p)
        print(f"| `{mode}` | {depth} | {med('gbps'):.3f} ({rng('gbps', 3)}) | {med('p50'):.0f} ({rng('p50', 0)}) "
              f"| {med('p99'):.0f} ({rng('p99', 0)}) | {len(p)} | {bad} | {r['negative']} |")
        out["restore"][f"{mode}@{depth}"] = {
            "gbps_median": med("gbps"), "p50_us_median": med("p50"), "p99_us_median": med("p99"),
            "cpu_s_median": med("cpu_s"),
            "gbps_process_medians": [statistics.median(x["gbps"] for x in proc) for proc in r["processes"]],
            "p50_process_medians": [statistics.median(x["p50"] for x in proc) for proc in r["processes"]],
            "p99_process_medians": [statistics.median(x["p99"] for x in proc) for proc in r["processes"]],
            "passes": len(p), "bad_words": bad, "negative_control_bad": r["negative"]}

    for name in ["feasibility", "udmabuf-4k", "cost-4k", "info"]:
        path = args.raw / f"{name}.txt"
        if path.exists():
            print(f"\n## {name}\n")
            print(path.read_text().rstrip())
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
