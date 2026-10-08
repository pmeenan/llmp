# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel time of one prefill chunk and one decode step at depth (README.md, "Where the time goes").

    profile.py TRACE.sqlite VOCAB [--top N] [--decode-steps D]
    profile.py TRACE.sqlite --graphs [--top N]

With --graphs (a trace of llmp-runtime serving, whose decode steps, and
speculation's drafts and verifies, are CUDA graphs): each kind of graph
launch, its count and its mean kernel time per launch, by kernel.

TRACE is `nsys profile --trace=cuda --sample=none --cpuctxsw=none
--export=sqlite` of one resident harness run (llmp_dsv4_exec or
llmp_qwen38_exec --prompts P --generate G): the prompt in chunks, then
D = G - 1 decode steps (default 2). Each chunk ends with its logits copied
to the host (a device-to-host copy of a multiple of VOCAB floats), which
splits the kernels into chunks. Reports the prompt's last chunk and the
last decode step: each kernel's total time and launches, grouped by its
short name, and the chunk's device span.

Runs on a Spark with the system Python (sqlite3); prints one JSON document.
"""
import json
import sqlite3
import sys


def graphs(db, names, top):
    """The runtime's decode steps are CUDA graphs: every kernel of one graph
    launch shares the launch's correlation ID. Launches are grouped by their
    shape (kernel count and first kernel); for each, the mean per launch."""
    rows = db.execute(
        "SELECT correlationId, start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE graphNodeId IS NOT NULL AND graphNodeId != 0 ORDER BY start").fetchall()
    launches = {}
    for corr, start, end, short in rows:
        launches.setdefault(corr, []).append((start, end, names.get(short, str(short))))
    kinds = {}
    for members in launches.values():
        key = f"{len(members)} kernels, first {members[0][2]}"
        kinds.setdefault(key, []).append(members)
    out = {}
    for key, runs in kinds.items():
        by = {}
        for members in runs:
            for start, end, name in members:
                by[name] = by.get(name, 0) + (end - start)
        busy = sum(by.values())
        out[key] = {"launches": len(runs), "busy_ms_each": busy / len(runs) / 1e6,
                    "span_ms_each": sum(m[-1][1] - m[0][0] for m in runs) / len(runs) / 1e6,
                    "top": [{"kernel": k, "ms_each": v / len(runs) / 1e6, "share": v / busy}
                            for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:top]]}
    return out


def main():
    trace = sys.argv[1]
    if "--graphs" in sys.argv:
        db = sqlite3.connect(trace)
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
        print(json.dumps(graphs(db, names, top), indent=1))
        return
    vocab = int(sys.argv[2])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
    db = sqlite3.connect(trace)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = db.execute(
        "SELECT start, end, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "ORDER BY start").fetchall()
    # Device-to-host copies (copyKind 2) of whole logits rows end a chunk.
    ends = [row[0] for row in db.execute(
        "SELECT start FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 2 AND bytes >= ? "
        "AND bytes % ? = 0 ORDER BY start", (vocab * 4, vocab * 4))]
    chunks, at = [], 0
    for end in ends:
        members = []
        while at < len(kernels) and kernels[at][1] <= end:
            members.append(kernels[at])
            at += 1
        if members:
            chunks.append(members)
    if len(chunks) < 3:
        raise SystemExit(f"only {len(chunks)} chunks in the trace")
    # --generate G gives G - 1 decode steps after the prompt's last chunk.
    steps = int(sys.argv[sys.argv.index("--decode-steps") + 1]) if "--decode-steps" in sys.argv \
        else 2
    decode = chunks[-1]
    prefill = chunks[-(steps + 1)]

    def summary(members):
        by = {}
        for start, end, short, _ in members:
            name = names.get(short, str(short))
            entry = by.setdefault(name, [0, 0])
            entry[0] += end - start
            entry[1] += 1
        busy = sum(end - start for start, end, _, _ in members)
        rows = sorted(by.items(), key=lambda kv: -kv[1][0])
        return {"kernels": len(members), "busy_ms": busy / 1e6,
                "span_ms": (members[-1][1] - members[0][0]) / 1e6,
                "top": [{"kernel": k, "ms": v[0] / 1e6, "launches": v[1],
                         "share": v[0] / busy} for k, v in rows[:top]]}

    print(json.dumps({"chunks": len(chunks), "prefill_chunk": summary(prefill),
                      "decode_step": summary(decode)}, indent=1))


if __name__ == "__main__":
    main()
