#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Decode ExLlamaV3's autotuning cache (coop_autotune_v1.bin) into cases.

Each record keys a launch choice by a hash of the problem (ExLlamaV3
6b84a21b, exllamav3_ext/quant/exl3_gemm.cu and coop_autotune.cu): FNV-1a over
half_k, min(next power of two of max(m, 2), 16), k, n, K, the FP32-output
flag, the device, the compute-capability class, the SM count and the
codebook; the fused gate/up kernel (mgemm) hashes the rows as given rather than
max(m, 2), also mixes min(bszm_in, 24) and
min(bszm_out, 24), and XORs a constant for sliced launches; the stored hash is
then salted with the tuner's version (4). This rebuilds those hashes for the
Qwen2.5-0.5B projection shapes and prints each record's case and choice:
the tile shape index (tag), block size, SMs and concurrency. A record that
matches no case is reported, not guessed.
"""

import argparse
import itertools
import json
import struct
from pathlib import Path

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK = (1 << 64) - 1
VERSION = 4
SLICED = 0x9E3779B97F4A7C15

# (name, k, n) for Qwen2.5-0.5B: hidden 896, 2 KV heads of 64, MLP 4864.
SHAPES = [("q/o", 896, 896), ("k/v", 896, 128), ("gate/up", 896, 4864), ("down", 4864, 896),
          ("head", 896, 151936)]


def mix(h, v):
    return ((h ^ (v & MASK)) * FNV_PRIME) & MASK


def gemm_hash(bucket, k, n, rate, fp32, device, cc, sms, cb, half_k):
    """The hash for rows in `bucket`: min(next power of two of rows, 16)."""
    h = FNV_OFFSET
    for v in (int(half_k), bucket, k, n, rate, int(fp32), device, cc, sms, cb):
        h = mix(h, v)
    return h


def salt(h):
    return mix(h, VERSION)


def records(path):
    data = Path(path).read_bytes()
    magic, fmt, size = struct.unpack_from("<8sII", data, 0)
    if magic != b"EX3ATUNE" or fmt != 1 or size != 32:
        raise ValueError("not a version-1 coop_autotune cache")
    for offset in range(16, len(data), size):
        h, tag, block, sms, concurrency, _, _ = struct.unpack_from("<QiiiiII", data, offset)
        yield {"hash": f"{h:016x}", "tag": tag, "block_dim": block, "num_sms": sms, "concurrency": concurrency}


def table(sms, cb, device):
    cases = {}
    # The plain GEMM hashes max(rows, 2), so its smallest bucket is 2; the
    # fused kernel hashes the rows as given, so a single row is bucket 1.
    for (name, k, n), m, rate, fp32, cc in itertools.product(SHAPES, (1, 2, 4, 8, 16), (4, 5, 6, 8), (0, 1),
                                                             range(16)):
        base = gemm_hash(m, k, n, rate, fp32, device, cc, sms, cb, 0)
        case = {"kernel": "gemm", "shape": name, "k": k, "n": n, "rows_bucket": m, "K": rate, "fp32_out": bool(fp32),
                "cc_class": cc}
        if m >= 2:
            cases[salt(base)] = case
        for bin_, bout in itertools.product(range(1, 25), repeat=2):
            h = mix(mix(base, bin_), bout)
            for sliced in (False, True):
                key = salt(h ^ SLICED if sliced else h)
                cases[key] = {**case, "kernel": "mgemm", "bszm_in": bin_, "bszm_out": bout, "sliced": sliced}
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--sms", type=int, default=48, help="GB10: 48")
    parser.add_argument("--codebook", type=int, default=1, help="mcg: 1")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    cases = table(args.sms, args.codebook, args.device)
    out = []
    for record in records(args.cache):
        out.append({**record, "case": cases.get(int(record["hash"], 16))})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
