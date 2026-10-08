#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate retained measurements and print grouped ranges; never trim outliers."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

def read_result(path, rows=None):
    if rows is None:
        rows = list(csv.reader(path.open()))
    assert rows, f"{path}: empty"
    assert rows[-1] == ["complete", "PASS"], f"{path}: incomplete"
    assert rows.count(["preflight", "PASS"]) == 1 and rows.count(["verification", "PASS"]) == 1
    headers = [r[1:] for r in rows if r and r[0] == "result_header"]
    results = [r[1:] for r in rows if r and r[0] == "result"]
    assert len(headers) == len(results) == 1 and len(headers[0]) == len(results[0])
    row = dict(zip(headers[0], results[0]))
    for key in row.keys() - {"mode", "cache", "load"}:
        row[key] = float(row[key])
        assert math.isfinite(row[key]) and row[key] >= 0, (path, key)
    for key in ["requests", "bytes", "chunk_bytes", "depth", "pressure_gib", "cache_before", "cache_after", "disk_bytes"]:
        assert row[key].is_integer(), (path, key)
    assert row["seconds"] > 0 and row["chunk_bytes"] > 0 and row["depth"] > 0
    assert row["bytes"] == row["requests"] * row["chunk_bytes"]
    assert row["cache"] == "control" or row["requests"] > 0, f"{path}: no transfers"
    assert abs(row["GBps"] - row["bytes"] / row["seconds"] / 1e9) < 0.0001
    assert 0 <= row["p50_us"] <= row["p95_us"] <= row["p99_us"] <= row["max_us"]
    hist = [r for r in rows if r and r[0] == "hist_us"]
    assert len({r[1] for r in hist}) == len(hist), f"{path}: duplicate bucket"
    assert all(len(r) == 3 and int(r[1]) % 10 == 0 and 0 <= int(r[1]) <= 100000 and int(r[2]) > 0 for r in hist)
    assert sum(int(r[2]) for r in hist) == row["requests"]
    if row["requests"]:
        ordered = sorted((int(r[1]), int(r[2])) for r in hist)
        for name, p in [("p50_us", .5), ("p95_us", .95), ("p99_us", .99), ("max_us", 1)]:
            rank = math.floor(p * (row["requests"] - 1))
            count = 0
            for lower, n in ordered:
                count += n
                if count > rank:
                    assert row[name] >= lower - .001, (path, name, "below histogram bucket")
                    assert lower == 100000 or row[name] <= lower + 10 + .001, (path, name, "above histogram bucket")
                    break
    if row["cache"] in {"cold", "sparse"}:
        assert row["cache_before"] == 0, f"{path}: not cold"
    if row["mode"] != "buffered" and row["cache"] not in {"warm"}:
        assert row["cache_after"] == 0, f"{path}: unexpected page cache"
    if row["cache"] == "resident":
        assert row["disk_bytes"] < 1024 * 1024, f"{path}: residency did storage I/O"
    row["file"] = str(path)
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = []
    for phase in sorted(args.results.iterdir()):
        if not phase.is_dir() or not (phase / "jobs.json").exists():
            continue
        jobs = json.loads((phase / "jobs.json").read_text())
        bundle = None
        if (phase / "measurements.csv").exists():
            bundle, current = {}, None
            for line in csv.reader((phase / "measurements.csv").open()):
                if line and line[0] == "job":
                    assert len(line) == 2 and line[1] not in bundle, "duplicate/malformed bundled job"
                    current = line[1]
                    bundle[current] = []
                else:
                    assert current is not None, "result before bundled job"
                    bundle[current].append(line)
        expected = set()
        for mode, kib, depth, duration, cache, pressure, load, rep in jobs:
            name = f"{mode}-{kib}k-q{depth}-{cache}-p{pressure}-{load}-r{rep}"
            assert name not in expected, f"duplicate job {name}"
            expected.add(name)
            row = read_result(phase / f"{name}.csv", None if bundle is None else bundle[name])
            assert row["mode"] == mode and row["chunk_bytes"] == kib * 1024 and row["depth"] == depth
            assert row["cache"] == cache and row["pressure_gib"] == pressure and row["load"] == load
            eof_limited = mode.endswith("write") or cache == "sparse" or (mode == "buffered" and cache == "cold")
            assert eof_limited or row["seconds"] >= duration - .000001, f"{name}: shorter than requested duration"
            if bundle is None:
                assert not (phase / f"{name}.stderr").read_text().strip(), f"{name}: stderr"
            row["phase"] = phase.name
            rows.append(row)
        assert (set(bundle) if bundle is not None else {f.stem for f in phase.glob("*.csv")}) == expected
    assert rows, "no result phases"
    trace_path = args.results / "dma-trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        assert trace["event_enabled"] == trace["tracing_on"] == "1"
        assert "llmp IO comparison trace begin" in trace["trace"] and "llmp IO comparison trace end" in trace["trace"]
        assert trace["bounce_events"] == sum("swiotlb_bounced:" in line for line in trace["trace"].splitlines())
        for stats in trace["cpu_stats"].values():
            for line in stats.splitlines():
                if "overrun:" in line or "dropped events:" in line:
                    assert int(line.split(":", 1)[1].strip()) == 0, "trace lost events"
    feature_path = args.results / "coalescing-feature.json"
    if feature_path.exists():
        feature = json.loads(feature_path.read_text())
        assert feature["restored"] is True
        assert all(c["requested"] == c["observed"] for c in feature["changes"])
        assert feature["changes"][-1]["observed"] == feature["original"]
    groups = {}
    for r in rows:
        key = (r["phase"], r["mode"], int(r["chunk_bytes"]) // 1024, int(r["depth"]), r["cache"], int(r["pressure_gib"]), r["load"])
        groups.setdefault(key, []).append(r)
    for key, group in sorted(groups.items()):
        def span(field):
            return f"{min(r[field] for r in group):.3f}–{max(r[field] for r in group):.3f}"
        print(*key, f"n={len(group)}", f"GB/s={span('GBps')}", f"p50_us={span('p50_us')}",
              f"p99_us={span('p99_us')}", f"cpu_s={span('cpu_seconds')}",
              f"cache_GiB={statistics.mean(r['cache_after'] for r in group) / 2**30:.3f}",
              f"bg/s={span('background_per_second')}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Validated {len(rows)} measurements, {sum(int(r['requests']) for r in rows):,} transfers.")

if __name__ == "__main__":
    main()
