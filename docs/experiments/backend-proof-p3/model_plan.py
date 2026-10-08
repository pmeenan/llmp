#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The native full model's forced linear plans, one per fixture and arm (reference only).

Reference-side experiment tooling for the backend proof's P3; it shares no code with llmpalooza's
planner. The native full-model run reads its linears' plans in the per-linear sweep's format
(native_plan.py plan; benchmarks/exl3_linear_sweep.cc ReadPlan). This writes that plan restricted to
the row counts the held-out trajectories run (1, 32, 144, 145, 1,023 and 1,024), regenerated with
native_plan.py's logic from the per-linear sweep's reference run of that fixture and arm (ref-F-A.json,
P3 part 1), and checks that
  (a) every tuning record a packed case uses (GEMM and multi-GEMM keys; a GEMV case's key if it has a
      record) is present and equal in the arm's frozen P0 cache (tune-40-gemvoff / tune-45-gemvoff for
      EXL3-G, tune-40 / tune-45 for EXL3-O; ../backend-proof-p0/results.json), so none of the sweep's
      added 8-row records is used;
  (b) every case the model needs is present exactly once with the path upstream takes: every
      projection of every layer and lm_head at each row count, gate and up as one multi case at 1 and
      32 rows and as separate cases from 144, packed to 144 rows, reconstructed at 145 and 1,023,
      fused at 1,024;
  (c) with --record (the operation plan record of the arm, exl3-op-plan-g.json or -o.json), each
      packed case's grid is the one the record's probe launched for that linear and phase kind
      (1 row: every single-token kind; 32 and 144 rows: the prefills).
Header lines, which native ignores: "# fixture F arm A", "# cache SHA256" (the frozen cache's) and
"# reference ref-F-A.json SHA256".

  model_plan.py --fixture 4.0bpw --arm G --reference ref-40-G.json [--record exl3-op-plan-g.json] --out plan.txt

Exits 1 on any failed check (nothing is written).
"""

import argparse
import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_plan  # noqa: E402

ROWS = (1, 32, 144, 145, 1023, 1024)
FROZEN = {("4.0bpw", "G"): "tune-40-gemvoff", ("4.5bpw", "G"): "tune-45-gemvoff",
          ("4.0bpw", "O"): "tune-40", ("4.5bpw", "O"): "tune-45"}
KERNELS = {"gemm": "exl3_gemm_kernel", "gemv": "exl3_gemv_kernel", "multi": "exl3_mgemm_kernel"}


def frozen_cache(name):
    entry = json.loads((native_plan.P0 / "results.json").read_text())["exl3"]["tuning_caches"][name]
    data = base64.b64decode(entry["base64"])
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise SystemExit(f"{name}: bytes differ from the recorded SHA-256")
    return data, entry["sha256"]


def recorded_grids(record, fixture):
    """{(op, layer or None for lm_head, rows): {grids}} of the packed launches in the record."""
    names = {k["id"]: k["name"] for k in record["kernels"]}
    out = {}
    for pk in record["phase_kinds"]:
        part = pk["fixtures"][fixture]
        groups = [(op, g["layers"], g["launches"]) for op, gs in part["linear_launches"].items() for g in gs]
        groups += [("lm_head", [None], o["launches"]) for o in part["lm_head_launches"]]
        for op, layers, launches in groups:
            grid = next((ln[1] for ln in launches if isinstance(ln[0], int)
                         and any(k in names[ln[0]] for k in KERNELS.values())), None)
            if grid is not None:
                for layer in layers:
                    out.setdefault((op, layer, pk["rows"]), set()).add(json.dumps(grid))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", choices=("4.0bpw", "4.5bpw"), required=True)
    parser.add_argument("--arm", choices=("G", "O"), required=True)
    parser.add_argument("--reference", type=Path, required=True, help="the sweep's reference run, ref-F-A.json")
    parser.add_argument("--record", type=Path, help="the arm's operation plan record, to check the grids")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    if (ref["fixture"], ref["arm"]) != (args.fixture, args.arm):
        raise SystemExit(f"{args.reference} is {ref['fixture']} {ref['arm']}")
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "plan.txt"
        if native_plan.plan(args.reference, full, native_plan.P0 / "exl3-recon-pin.json") != 0:
            raise SystemExit("native_plan.py plan failed")
        lines = full.read_text().splitlines()
    cases = {line.split()[1]: line for line in lines if line.startswith("case ") and int(line.split()[2]) in ROWS}
    problems = []

    # (b) every case the model needs, once, on upstream's path.
    want = {c["id"]: c for c in native_plan.sweep.cases() if c["rows"] in ROWS}
    ids = [line.split()[1] for line in lines if line.startswith("case ") and int(line.split()[2]) in ROWS]
    if len(ids) != len(set(ids)) or set(ids) != set(want):
        problems.append(f"cases: {len(set(want) - set(ids))} missing, {len(set(ids) - set(want))} unexpected, "
                        f"{len(ids) - len(set(ids))} repeated")
    paths = {"packed": ("gemm", "gemv"), "multi": ("multi",), "recon": ("recon",), "fused": ("fused",)}
    for cid, line in cases.items():
        if cid in want and line.split()[3] not in paths[want[cid]["path"]]:
            problems.append(f"{cid}: path {line.split()[3]}, upstream's {want[cid]['path']}")

    # (a) the tuning records used are the frozen cache's.
    frozen_bytes, frozen_sha = frozen_cache(FROZEN[(args.fixture, args.arm)])
    frozen = native_plan.records(frozen_bytes)
    final = native_plan.records(base64.b64decode(ref["tune_cache"]["bytes_base64"]))
    weights = ref["weights"]
    used = 0
    for cid, line in cases.items():
        kind, rows, first = line.split()[3], int(line.split()[2]), weights[line.split()[4]]
        if kind not in KERNELS:
            continue
        keyer = native_plan.mgemm_key if kind == "multi" else native_plan.gemm_key
        key = keyer(rows, first["k"], first["n"], first["K"], first["out"] == "F32")
        if kind == "gemv" and key not in final and key not in frozen:
            continue    # the GEMV heuristic, no tuning record
        used += 1
        if key not in frozen or final.get(key) != frozen[key]:
            problems.append(f"{cid}: record {final.get(key)} is not the frozen cache's {frozen.get(key)}")

    # (c) the grids the operation plan record launched.
    checked = 0
    if args.record:
        grids = recorded_grids(json.loads(args.record.read_text()), args.fixture)
        for cid, line in cases.items():
            f = line.split()
            if f[3] not in KERNELS:
                continue
            name, rows = cid.split("@")[0], int(f[2])
            layer = None if name == "lm_head" else int(name.split(".")[2])
            op = "lm_head" if name == "lm_head" else name.rsplit(".", 1)[1]
            grid = [int(f[7]), 1, int(f[8])] if f[3] == "multi" else [int(f[6]), 1, 1]
            seen = grids.get((op, layer, rows))
            if seen is None:
                continue    # a row count the record does not launch packed (none for 1, 32, 144)
            checked += 1
            if seen != {json.dumps(grid)}:
                problems.append(f"{cid}: grid {grid}, the record's {sorted(seen)}")
        if checked != sum(1 for line in cases.values() if line.split()[3] in KERNELS):
            problems.append(f"record: only {checked} packed cases have a recorded launch")

    print(f"{args.fixture} {args.arm}: {len(cases)} cases, {used} tuning records used (all frozen), "
          f"{checked} grids checked against the record, {len(problems)} problems")
    for problem in problems[:20]:
        print("  ", problem)
    if problems:
        return 1
    header = [f"# fixture {args.fixture} arm {args.arm}", f"# cache {frozen_sha}",
              f"# reference {args.reference.name} {hashlib.sha256(args.reference.read_bytes()).hexdigest()}"]
    body = [line for line in lines if line.startswith("linear ")] + [cases[c["id"]] for c in native_plan.sweep.cases()
                                                                     if c["id"] in cases]
    args.out.write_text("\n".join(header + body) + "\n")
    print("wrote", args.out, hashlib.sha256(args.out.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
