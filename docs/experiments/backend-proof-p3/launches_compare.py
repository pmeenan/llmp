#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares every kernel native's per-linear sweep launched with the launches upstream recorded.

  launches_compare.py REFERENCE.json NATIVE.sqlite

REFERENCE is linear_reference.py's run with --profile, whose record holds, per case, the kernels
its product launched (name, grid, block). NATIVE.sqlite is `nsys export --type sqlite` of an nsys
trace (`-t cuda`) of benchmarks/exl3_linear_sweep.cc run at REFERENCE's plans. The sweep first
reconstructs every linear's full weights (two launches per linear), then runs each case twice
(its steps, then the registry's implementation), so its launches, in time order, after those
reconstructions, must be each case's recorded launches, twice, case after case in the reference's
order: the same kernel (name without its parameter list, which nsys and PyTorch's profiler spell
differently for CUTLASS kernels, and with nsys's `(int)4, (bool)0` template arguments read as the
profiler's `4, false`), grid and block. The one declared difference is the
reconstruction path's bias add: upstream uses PyTorch's element-wise add there, and the approved
operation plan ExLlamaV3's add_kernel_hhh, launched as upstream's add_gr launches it (1,024
threads per block over the output's elements). The approved reconstruction-path Tier E item
requires native's cuBLAS kernel names and grids to equal the arm's; this checks them with every
other launch. Exits 1 on the first difference. Reference-only experiment tooling.
"""

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path


def kernel(name):
    """The kernel's name without its parameter list, and with nsys's spelling of template arguments
    ((int)4, (bool)0) read as the profiler's (4, false)."""
    name = re.sub(r"\((?:[^()]|\([^()]*\))*\)$", "", name)
    name = name.replace("(bool)0", "false").replace("(bool)1", "true")
    return re.sub(r"\((?:int|unsigned int|long|unsigned long)\)(-?\d+)", r"\1", name)


def cublas(name):
    return name.startswith(("nvjet", "void cutlass", "cutlass"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    expected = []
    for case in ref["cases"]:
        weights = ref["weights"][case["linears"][0]]
        launches = []
        for launch in case["launches"]:
            name = launch["name"]
            if name.startswith("void at::native::"):
                # The reconstruction path's bias add (declared above).
                if case["path"] not in ("recon", "fused") or weights["bias"] is None:
                    raise SystemExit(f"{case['id']}: an unexpected PyTorch kernel {name}")
                elements = case["rows"] * weights["n"]
                launches.append(("add_kernel_hhh", (-(-elements // 1024), 1, 1), (1024, 1, 1)))
            else:
                launches.append((kernel(name), tuple(launch["grid"]), tuple(launch["block"])))
        expected += [(case["id"], launch) for launch in launches * 2]

    db = sqlite3.connect(args.native)
    rows = db.execute(
        "SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start").fetchall()
    native = [(kernel(name), (gx, gy, gz), (bx, by, bz)) for name, gx, gy, gz, bx, by, bz in rows]
    # The full-weight reconstructions come first: two per linear.
    loading = 2 * len(ref["weights"])
    for name, _, _ in native[:loading]:
        if not name.startswith(("void reconstruct_kernel", "void reconstruct_had_kernel")):
            print(f"an unexpected kernel while loading: {name}")
            return 1
    native = native[loading:]
    print(f"{len(expected)} expected launches, {len(native)} in the native trace after {loading} reconstructions")
    for i, ((case, want), got) in enumerate(zip(expected, native)):
        if want != got:
            print(f"launch {i} ({case}): upstream {want}, native {got}")
            return 1
    if len(expected) != len(native):
        print("the counts differ")
        return 1
    tally = collections.Counter(("cuBLAS" if cublas(launch[0]) else "ExLlamaV3") for _, launch in expected)
    print(f"all equal: {dict(tally)} launches, {len({launch[0] for _, launch in expected})} distinct kernels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
