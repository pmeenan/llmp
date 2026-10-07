#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare repeated GPU steps (decode tokens, prefill chunks) between two nsys traces.

Each trace is an `nsys export --type sqlite` of a run, ideally captured with
`--cuda-graph-trace=node` so graph replays show their kernels. Kernels are split into
steps wherever no kernel runs for longer than --gap (a copy alone longer than that
also splits a step, and is then counted in neither half); the steps whose span lies in
--span (milliseconds) are compared, the last --last of each trace. The report is per
step, averaged:

  - span (first kernel start to last kernel end) and the idle gap before the next step;
  - kernel time and count by key: name, the first template argument (a quantized
    product's weight type) and the grid, so the same launch is matched across engines;
  - device copies and memsets inside steps (graph copy nodes are easy to miss: they
    show in no kernel table);
  - the longest idle holes inside a step, with the kernels on either side.

Kernel time is summed, so kernels that overlap (concurrent streams, programmatic
launch) count twice: compare the span for the step's cost, and the kernel rows for
where it goes. This tool reads the traces only; it runs nothing.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import sqlite3
import statistics
import sys


@dataclasses.dataclass
class Step:
    kernels: list  # (start, end, key, stream)
    copies: list  # (start, end, bytes, kind)
    end: int = 0  # the latest kernel end

    @property
    def start(self) -> int:
        return self.kernels[0][0]


def kernel_key(name: str, demangled: str, grid: int) -> str:
    """The kernel's name, its first template argument (if any) and its grid's x."""
    first = ""
    if "<" in demangled:
        inner = demangled[demangled.find("<") + 1:]
        depth, end = 0, len(inner)
        for i, ch in enumerate(inner):
            if ch == "<":
                depth += 1
            elif ch == ">" and depth == 0:
                end = i
                break
            elif ch == ">":
                depth -= 1
            elif ch == "," and depth == 0:
                end = i
                break
        first = inner[:end].strip()
    return f"{name}<{first}> g{grid}" if first else f"{name} g{grid}"


def load(path: str, gap_ns: int) -> list[Step]:
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = [(s, e, kernel_key(names.get(n, str(n)), names.get(d, ""), g), st)
               for s, e, n, d, g, st in db.execute(
                   "SELECT start, end, shortName, demangledName, gridX, streamId "
                   "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    copies = []
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        copies += [(s, e, b, f"copy{k}") for s, e, b, k in db.execute(
            "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        copies += [(s, e, b, "memset") for s, e, b in db.execute(
            "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET")]
    copies.sort()
    steps: list[Step] = []
    for k in kernels:
        if steps and k[0] - steps[-1].end <= gap_ns:
            steps[-1].kernels.append(k)
            steps[-1].end = max(steps[-1].end, k[1])
        else:
            steps.append(Step([k], [], k[1]))
    ci = 0
    for step in steps:
        while ci < len(copies) and copies[ci][0] < step.start:
            ci += 1
        j = ci
        while j < len(copies) and copies[j][0] <= step.end:
            step.copies.append(copies[j])
            j += 1
    return steps


def select(steps: list[Step], low_ms: float, high_ms: float, last: int) -> list[Step]:
    chosen = [s for s in steps if low_ms * 1e6 <= s.end - s.start <= high_ms * 1e6]
    return chosen[-last:] if last > 0 else chosen


@dataclasses.dataclass
class Summary:
    steps: int
    span_ms: float
    gap_ms: float
    kernels: dict  # key -> (ms per step, count per step)
    copies: dict  # kind -> (ms per step, count per step, bytes per step)
    holes: list  # (µs, before, after)


def summarize(all_steps: list[Step], chosen: list[Step], hole_us: float) -> Summary:
    n = len(chosen)
    span = statistics.median((s.end - s.start) / 1e6 for s in chosen) if n else 0.0
    order = {id(s): i for i, s in enumerate(all_steps)}
    gaps = []
    for s in chosen:
        i = order[id(s)]
        if i + 1 < len(all_steps):
            gaps.append((all_steps[i + 1].start - s.end) / 1e6)
    gap = statistics.median(gaps) if gaps else 0.0
    time = collections.Counter()
    count = collections.Counter()
    for s in chosen:
        for start, end, key, _ in s.kernels:
            time[key] += end - start
            count[key] += 1
    copy_time = collections.Counter()
    copy_count = collections.Counter()
    copy_bytes = collections.Counter()
    for s in chosen:
        for start, end, size, kind in s.copies:
            copy_time[kind] += end - start
            copy_count[kind] += 1
            copy_bytes[kind] += size
    holes = []
    if chosen:
        s = chosen[len(chosen) // 2]
        reach = s.kernels[0][1]
        for prev, nxt in zip(s.kernels, s.kernels[1:]):
            reach = max(reach, prev[1])
            idle = (nxt[0] - reach) / 1e3
            if idle >= hole_us:
                holes.append((idle, prev[2], nxt[2]))
        holes.sort(reverse=True)
    return Summary(
        steps=n, span_ms=span, gap_ms=gap,
        kernels={k: (time[k] / 1e6 / n, count[k] / n) for k in time} if n else {},
        copies={k: (copy_time[k] / 1e6 / n, copy_count[k] / n, copy_bytes[k] / n)
                for k in copy_time} if n else {},
        holes=holes[:10])


def report(a: Summary, b: Summary, labels: tuple[str, str], top: int, out) -> None:
    la, lb = labels
    print(f"steps: {la} {a.steps}, {lb} {b.steps}", file=out)
    print(f"span ms: {la} {a.span_ms:.3f}, {lb} {b.span_ms:.3f}, diff {b.span_ms - a.span_ms:+.3f}",
          file=out)
    print(f"idle gap after a step ms: {la} {a.gap_ms:.3f}, {lb} {b.gap_ms:.3f}", file=out)
    keys = sorted(set(a.kernels) | set(b.kernels),
                  key=lambda k: -abs(b.kernels.get(k, (0, 0))[0] - a.kernels.get(k, (0, 0))[0]))
    print(f"\n{'kernel':60} {la + ' ms':>10} {'n':>6} {lb + ' ms':>10} {'n':>6} {'diff':>8}", file=out)
    for k in keys[:top]:
        ta, na = a.kernels.get(k, (0.0, 0.0))
        tb, nb = b.kernels.get(k, (0.0, 0.0))
        print(f"{k[:60]:60} {ta:10.3f} {na:6.1f} {tb:10.3f} {nb:6.1f} {tb - ta:+8.3f}", file=out)
    total_a = sum(v[0] for v in a.kernels.values())
    total_b = sum(v[0] for v in b.kernels.values())
    print(f"{'kernel time (overlap counts twice)':60} {total_a:10.3f} {'':6} {total_b:10.3f} {'':6} "
          f"{total_b - total_a:+8.3f}", file=out)
    for label, s in ((la, a), (lb, b)):
        if s.copies:
            print(f"\ncopies and memsets inside {label}'s steps:", file=out)
            for kind, (ms, cnt, size) in sorted(s.copies.items()):
                print(f"  {kind:8} {ms:8.3f} ms {cnt:7.1f} per step {size / 1024:10.1f} KiB", file=out)
        if s.holes:
            print(f"\nlongest idle holes in a middle {label} step:", file=out)
            for idle, before, after in s.holes:
                print(f"  {idle:8.1f} µs  after {before[:40]}  before {after[:40]}", file=out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", help="the reference engine's nsys sqlite export")
    parser.add_argument("native", help="jitLLM's nsys sqlite export")
    parser.add_argument("--span", nargs=2, type=float, metavar=("LOW_MS", "HIGH_MS"), required=True,
                        help="the steps compared: those whose span lies in this range")
    parser.add_argument("--last", type=int, default=127, help="the last N such steps (0: all)")
    parser.add_argument("--gap", type=float, default=250.0,
                        help="idle µs that separates steps (default 250)")
    parser.add_argument("--hole", type=float, default=5.0, help="report idle holes of at least µs")
    parser.add_argument("--top", type=int, default=30, help="kernel rows, by absolute difference")
    args = parser.parse_args(argv)
    sides = []
    for path in (args.reference, args.native):
        steps = load(path, int(args.gap * 1e3))
        chosen = select(steps, args.span[0], args.span[1], args.last)
        if not chosen:
            print(f"{path}: no step spans {args.span[0]}-{args.span[1]} ms", file=sys.stderr)
            return 1
        sides.append(summarize(steps, chosen, args.hole))
    report(sides[0], sides[1], ("reference", "native"), args.top, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
