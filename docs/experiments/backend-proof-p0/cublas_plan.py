#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Reduce cuBLASLt trace logs to the distinct GEMMs and their resolved algorithms.

External reference tooling for the backend proof's P0. Input: logs written
with CUBLASLT_LOG_LEVEL=5 (and optionally the cuBLAS API log, for the
handle's workspace and math-mode calls). Each "[Trace][cublasLt...Matmul]"
line names the matrix layouts, the compute descriptor and the algorithm
cuBLASLt resolved (algoId, tile, stages, split-K, reduction, swizzle,
custom option). Pointers and timestamps are dropped; identical calls are
counted. The heuristic queries' preferences (workspace limit, alignments)
are reduced the same way.
"""

import argparse
import collections
import json
import re
from pathlib import Path

POINTER = re.compile(r"\b(?:A|B|C|D|workSpace)=0X[0-9A-F]+ ?")


def reduce(path):
    calls, prefs = collections.Counter(), collections.Counter()
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\[[^]]*\]\[cublasLt\]\[\d+\]\[(Trace|Api)\]\[(cublasLt\w+)\] (.*)", line)
        if not m:
            continue
        kind, function, rest = m.groups()
        if kind == "Trace" and function.endswith("Matmul"):
            calls[(function, POINTER.sub("", rest).strip())] += 1
        elif kind == "Api" and function.endswith("GetHeuristicForStream"):
            pref = re.search(r"preference=\[[^]]*\]", rest)
            prefs[pref.group(0) if pref else ""] += 1
    return ([{"function": f, "call": c, "count": n} for (f, c), n in sorted(calls.items())],
            [{"preference": p, "count": n} for p, n in sorted(prefs.items())])


def api(path):
    """The handle calls that shape the heuristic: workspace sizes and math modes."""
    text = path.read_text(errors="replace")
    return {"workspace_sizes": sorted(set(int(x) for x in re.findall(r"workspaceSizeInBytes: .*?val=(\d+)", text))),
            "math_modes": sorted(set(re.findall(r"MathMode=(\w+)", text)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lt", type=Path, required=True, help="CUBLASLT_LOG_FILE output")
    parser.add_argument("--api", type=Path, help="CUBLAS_LOGDEST_DBG output")
    args = parser.parse_args()
    calls, prefs = reduce(args.lt)
    out = {"calls": calls, "heuristic_preferences": prefs}
    if args.api:
        out["handle"] = api(args.api)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
