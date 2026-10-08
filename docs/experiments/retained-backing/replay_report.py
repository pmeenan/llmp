#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Applies the frozen deterministic criteria (D-079) to the replay's output.

`llmp_rb_replay` (benchmarks/retained_backing/) prints one JSON line per
design and budget file. This checks that the run is whole and not void (every
design at every budget, both replays agreeing exactly, the recorded trace
files), then applies backend-proof.md's deterministic criteria mechanically,
per budget, against D-033's replay of the same file:

- waste, mean over ticks and peak, no more than the baseline's;
- useful content lost no more than the baseline's;
- no more refusals than the baseline.

"No worse" means <=, and a tie passes. A replay that stopped at a refusal
covers only a prefix of the trace, so its other metrics are not comparable:
the stricter reading makes it ineligible at that budget whatever the
baseline did. A design that meets every criterion at a budget is eligible
for that budget's timed sessions. Stdlib only.

    python3 replay_report.py RESULTS.jsonl
"""
import json
import sys
from fractions import Fraction

GIB = 1 << 30
DESIGNS = ["d033"] + [f"slab{size}-{policy}" for size in ("32m", "256m", "1g")
                      for policy in ("run", "class", "hybrid", "compact")]
BASELINE = "d033"
# The budget files in the criteria's order, with their recorded SHA-256
# (README.md#identity); the confirmation seed's are listed for its winner.
FILES = {
    "primary": {
        "trace-r5-4.jsonl": "01611f43c464dd42b954224b00f21f8aeaef0476e75b9021e9d827a4991c692d",
        "trace-r3-2.jsonl": "040ea829349324c8dfd7e9b5320fb4bde3ddf2796863edbf38d037bd5f70ff24",
        "trace-r2-1.jsonl": "95255d38fcf005ba0e6b53b0de8e14a4f767bca8914d7c512e6b6c2d378c37dc",
    },
    "confirmation": {
        "trace-r5-4.jsonl": "36ce0dfeb71a643f791cd2f2e96a275fd89fcfa66436b16ce28c49bb0855e9aa",
        "trace-r3-2.jsonl": "6ea108f714cff887a6c8b81c875fd422a31c3d86f6bbfb9f34027dc5785bd543",
        "trace-r2-1.jsonl": "f75a69cec906f5071678dbb70b3f81547855c61ff006a2a0c8c2dc3e5cd4c3d6",
    },
}
CALLS = ("reserve", "free", "create", "release", "map", "set_access", "unmap", "register", "unregister")


class VoidRun(ValueError):
    """The run cannot be judged: incomplete, disagreeing or on another trace."""


def load(lines, designs=None):
    """{file: {design: record}} after checking the run is whole and agreed."""
    records, role = {}, None
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        r = json.loads(line)
        role = role or r["role"]
        if r["role"] != role or role not in FILES:
            raise VoidRun(f"line {number}: mixed or unknown role")
        if FILES[role].get(r["file"]) != r["trace_sha256"]:
            raise VoidRun(f"line {number}: {r['file']} is not the recorded {role} trace")
        if r["replays_agree"] is not True:
            raise VoidRun(f"line {number}: {r['design']}'s two replays disagree, so the run is void")
        if r["design"] not in DESIGNS:
            raise VoidRun(f"line {number}: {r['design']} is not one of the criteria's designs")
        if r["content_lost_bytes"] != r["extra_restored_bytes"] + r["shrink_extra_evicted_bytes"]:
            raise VoidRun(f"line {number}: {r['design']}'s content lost is not its two parts")
        if r["design"] in records.setdefault(r["file"], {}):
            raise VoidRun(f"line {number}: {r['design']} repeated for {r['file']}")
        records[r["file"]][r["design"]] = r
    wanted = designs or DESIGNS
    for name in FILES[role or "primary"]:
        missing = [d for d in wanted if d not in records.get(name, {})]
        if missing:
            raise VoidRun(f"{name}: no result for {', '.join(missing)}")
    return role, records


def waste_mean(r):
    return Fraction(r["waste_sum"], r["ticks"]) if r["ticks"] else Fraction(0)


def judge(base, r):
    """The deterministic criteria for one design at one budget."""
    stopped = r["refusals"] > 0
    checks = {
        "waste_mean": not stopped and waste_mean(r) <= waste_mean(base),
        "waste_peak": not stopped and r["waste_peak"] <= base["waste_peak"],
        "content_lost": not stopped and r["content_lost_bytes"] <= base["content_lost_bytes"],
        "refusals": r["refusals"] <= base["refusals"] and not stopped,
    }
    checks["eligible"] = all(checks.values())
    return checks


def gib(value):
    return f"{value / GIB:.3f}"


def mark(ok):
    return "yes" if ok else "**no**"


def report(role, records):
    out = [f"Role: {role}.", ""]
    eligible = {d: [] for d in DESIGNS if d != BASELINE}
    for name in FILES[role]:
        rows = records[name]
        base = rows[BASELINE]
        budget = base["budget_bytes"] // GIB
        out += [f"### {budget} GiB (`{name}`)", "",
                "| Design | Waste mean GiB | Waste peak GiB | Content lost GiB | Extra restored GiB "
                "| Shrink extra GiB | Refusals | Eligible |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for d in DESIGNS:
            r = rows[d]
            refusals = str(r["refusals"]) + (f" (event {r['refused_at']})" if r["refusals"] else "")
            if d == BASELINE:
                verdict = "baseline"
            else:
                checks = judge(base, r)
                if checks["eligible"]:
                    eligible[d].append(budget)
                failed = [k for k, ok in checks.items() if k != "eligible" and not ok]
                verdict = "yes" if checks["eligible"] else "no: " + ", ".join(failed)
            out.append(f"| `{d}` | {float(waste_mean(r)) / GIB:.3f} | {gib(r['waste_peak'])} "
                       f"| {gib(r['content_lost_bytes'])} | {gib(r['extra_restored_bytes'])} "
                       f"| {gib(r['shrink_extra_evicted_bytes'])} | {refusals} | {verdict} |")
        out.append("")
        out += ["Reported, not gated (calls per restored GiB; io_uring registrations and "
                "unregistrations; relocation):", "",
                "| Design | Restored GiB | Create | Release | Map | Set access | Unmap | Reserve "
                "| Reg / unreg | Delayed admissions | Relocated GiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for d in DESIGNS:
            r = rows[d]
            restored = r["restored_bytes"] / GIB
            per = {k: (r["calls"][k] / restored if restored else 0.0) for k in CALLS}
            out.append(f"| `{d}` | {restored:.2f} | {per['create']:.2f} | {per['release']:.2f} "
                       f"| {per['map']:.2f} | {per['set_access']:.2f} | {per['unmap']:.2f} "
                       f"| {r['calls']['reserve']} | {r['calls']['register']} / "
                       f"{r['calls']['unregister']} | {r['delayed_admissions']} "
                       f"| {gib(r['relocated_bytes'])} |")
        out.append("")
    out += ["### Exact counts", "",
            "What a timed run's own calls must equal (backend-proof.md), per budget and design:", "",
            "| Budget | Design | Create | Release | Map | Set access | Unmap | Reserve | Register "
            "| Unregister | Restored bytes | Relocated bytes |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in FILES[role]:
        for d in DESIGNS:
            r = records[name][d]
            c = r["calls"]
            out.append(f"| {r['budget_bytes'] // GIB} | `{d}` | {c['create']} | {c['release']} | {c['map']} "
                       f"| {c['set_access']} | {c['unmap']} | {c['reserve']} | {c['register']} "
                       f"| {c['unregister']} | {r['restored_bytes']} | {r['relocated_bytes']} |")
    out.append("")
    budgets = [records[name][BASELINE]["budget_bytes"] // GIB for name in FILES[role]]
    out += ["### Eligibility for timed sessions", "",
            "| Design | " + " | ".join(f"{b} GiB" for b in budgets) + " | Every budget |",
            "| --- |" + " --- |" * (len(budgets) + 1)]
    for d, passed in eligible.items():
        out.append(f"| `{d}` | " + " | ".join(mark(b in passed) for b in budgets)
                   + f" | {mark(len(passed) == len(budgets))} |")
    return "\n".join(out) + "\n", eligible


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    with open(argv[1]) as f:
        role, records = load(f)
    text, _ = report(role, records)
    sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except VoidRun as e:
        raise SystemExit(f"void run: {e}") from None
