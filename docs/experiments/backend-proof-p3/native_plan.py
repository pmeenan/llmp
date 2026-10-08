#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Forced plans for the per-linear sweep (BP-N5), from upstream's frozen tuning caches.

Reference-side experiment tooling for the backend proof's P3; it shares no code with llmpalooza's
planner or launchers.

  native_plan.py caches OUTDIR
      writes P0's four frozen model caches (tune-40-gemvoff and tune-45-gemvoff for EXL3-G, tune-40
      and tune-45 for EXL3-O) from ../backend-proof-p0/results.json, each checked against its
      recorded SHA-256: the starting caches of the reference runs.

  native_plan.py plan REFERENCE.json --out PLAN.txt
      decodes the tuning cache a reference run (linear_reference.py) ended with and writes each
      case's forced plan for the native harness (benchmarks/exl3_linear_sweep.cc):
        linear NAME K N BITS F16|F32 BIAS
        case ID ROWS gemm LINEAR SHAPE BLOCKS
        case ID ROWS gemv LINEAR CONFIG BLOCKS
        case ID ROWS multi GATE UP SHAPE BLOCKS CONCURRENCY
        case ID ROWS recon|fused LINEAR SLICES (per slice: the pinned GEMM's KIND M K N LDC, then its
            9 algorithm attributes)...
      A packed case's plan is the cache record for its key (../backend-proof-p0/decode_tuning.py),
      or, where the reference launched the GEMV kernel, the configuration and grid it launched. The
      reconstruction paths take each slice's pinned algorithm (../backend-proof-p0/
      exl3-recon-pin.json). Every plan is checked against the launches the reference recorded
      (--profile): kernel, grid and block. Exits 1 on any case without a plan or disagreeing with
      the reference's launches.
"""

import argparse
import base64
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P0 = HERE.parent / "backend-proof-p0"
sys.path.insert(0, str(P0))
sys.path.insert(0, str(HERE))
import cases as sweep  # noqa: E402
import decode_tuning  # noqa: E402

FROZEN = {"tune-40-gemvoff": "4.0bpw G", "tune-45-gemvoff": "4.5bpw G", "tune-40": "4.0bpw O", "tune-45": "4.5bpw O"}
BLOCKDIM = (0, 256, 512, 512, 256)
SMS, DEVICE, CC_BLACKWELL, MCG = 48, 0, 5, 1


def caches(outdir):
    record = json.loads((P0 / "results.json").read_text())["exl3"]["tuning_caches"]
    outdir.mkdir(parents=True, exist_ok=True)
    for name in FROZEN:
        data = base64.b64decode(record[name]["base64"])
        if hashlib.sha256(data).hexdigest() != record[name]["sha256"]:
            raise SystemExit(f"{name}: bytes differ from the recorded SHA-256")
        (outdir / f"{name}.bin").write_bytes(data)
        print(name, record[name]["sha256"][:16], len(data))


def roundup_pow2(x):
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def records(cache_bytes):
    magic, fmt, size = struct.unpack_from("<8sII", cache_bytes, 0)
    if magic != b"EX3ATUNE" or fmt != 1 or size != 32:
        raise SystemExit("not a version-1 coop_autotune cache")
    out = {}
    for offset in range(16, len(cache_bytes), size):
        h, tag, block, sms, concurrency, _, _ = struct.unpack_from("<QiiiiII", cache_bytes, offset)
        out[h] = (tag, block, sms, concurrency)
    return out


def gemm_key(rows, k, n, bits, fp32):
    bucket = min(roundup_pow2(max(rows, 2)), 16)
    return decode_tuning.salt(decode_tuning.gemm_hash(bucket, k, n, bits, fp32, DEVICE, CC_BLACKWELL, SMS, MCG, 0))


def mgemm_key(rows, k, n, bits, fp32):
    bucket = min(roundup_pow2(rows), 16)
    h = decode_tuning.gemm_hash(bucket, k, n, bits, fp32, DEVICE, CC_BLACKWELL, SMS, MCG, 0)
    return decode_tuning.salt(decode_tuning.mix(decode_tuning.mix(h, 1), 2))


def first_launch(case, prefix):
    for launch in case.get("launches") or []:
        if launch["name"].startswith(prefix):
            return launch
    return None


def plan(reference, out, pins_path):
    data = json.loads(reference.read_text())
    table = records(base64.b64decode(data["tune_cache"]["bytes_base64"]))
    pins = {}
    for g in json.loads(pins_path.read_text())["gemms"]:
        c = g["config"]
        # The GEMM the pin was recorded for, as the table has it (the native side refuses a pin
        # for any other GEMM), then its configuration.
        pins[(g["kind"], g["n"], g["k"], g["m"], g["lda"], g["ldb"], g["ldc"])] = [
            g["kind"], g["m"], g["k"], g["n"], g["ldc"],
            c["algo_id"], c["tile"], c["splitk"], c["reduction"], c["swizzle"], c["custom"], c["stages"],
            c["inner_shape"], c["cluster_shape"]]
    weights = data["weights"]
    lines, problems = [], []
    for name in sweep.linears():
        w = weights[name]
        lines.append(f"linear {name} {w['k']} {w['n']} {w['K']} {w['out']} {int(w['bias'] is not None)}")
    for case in data["cases"]:
        rows, first = case["rows"], weights[case["linears"][0]]
        fp32 = first["out"] == "F32"
        head = f"case {case['id']} {rows}"
        if case["path"] == "packed":
            gemv = first_launch(case, "void exl3_gemv_kernel")
            if case["tag"] == 90 or gemv is not None:
                if gemv is None:
                    problems.append(f"{case['id']}: upstream took the GEMV but no launch was recorded")
                    continue
                args = re.match(r"void exl3_gemv_kernel<(\d+), (true|false), (\d+), (\d+), (\d+),", gemv["name"])
                config, blocks = int(args[5]), gemv["grid"][0]
                if int(args[4]) != (0 if rows == 1 else 1) or gemv["block"][0] != (512 if config == 0 else 256):
                    problems.append(f"{case['id']}: an unexpected GEMV launch {gemv}")
                lines.append(f"{head} gemv {case['linears'][0]} {config} {blocks}")
                continue
            record = table.get(gemm_key(rows, first["k"], first["n"], first["K"], fp32))
            if record is None:
                problems.append(f"{case['id']}: no cache record")
                continue
            shape, block, sms, concurrency = record
            launch = first_launch(case, "void exl3_gemm_kernel")
            if concurrency != 1 or block != BLOCKDIM[shape] or case["tag"] != shape or (
                    launch is not None and (launch["grid"] != [sms, 1, 1] or launch["block"] != [block, 1, 1])):
                problems.append(f"{case['id']}: record {record} against tag {case['tag']} and launch {launch}")
            lines.append(f"{head} gemm {case['linears'][0]} {shape} {sms}")
        elif case["path"] == "multi":
            record = table.get(mgemm_key(rows, first["k"], first["n"], first["K"], fp32))
            if record is None:
                problems.append(f"{case['id']}: no cache record")
                continue
            shape, block, sms, concurrency = record
            launch = first_launch(case, "void exl3_mgemm_kernel")
            if block != BLOCKDIM[shape] or case["tag"] != shape or (
                    launch is not None and (launch["grid"] != [sms, 1, concurrency] or launch["block"] != [block, 1, 1])):
                problems.append(f"{case['id']}: record {record} against tag {case['tag']} and launch {launch}")
            lines.append(f"{head} multi {case['linears'][0]} {case['linears'][1]} {shape} {sms} {concurrency}")
        else:
            kind = "HSS" if fp32 else "HSH"
            widths = sweep.slices(first["n"])
            algorithms = []
            for width in widths:
                pin = pins.get((kind, width, first["k"], rows, width, first["k"], first["n"]))
                if pin is None:
                    problems.append(f"{case['id']}: no pinned algorithm for {kind} {width} {first['k']} {rows}")
                    break
                algorithms.append(" ".join(map(str, pin)))
            else:
                lines.append(f"{head} {case['path']} {case['linears'][0]} {len(widths)} " + " ".join(algorithms))
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(data['cases'])} cases, {len(problems)} problems")
    for problem in problems[:20]:
        print("  ", problem)
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("caches")
    c.add_argument("outdir", type=Path)
    p = sub.add_parser("plan")
    p.add_argument("reference", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pins", type=Path, default=P0 / "exl3-recon-pin.json")
    args = parser.parse_args()
    if args.command == "caches":
        caches(args.outdir)
        return 0
    return plan(args.reference, args.out, args.pins)


if __name__ == "__main__":
    sys.exit(main())
