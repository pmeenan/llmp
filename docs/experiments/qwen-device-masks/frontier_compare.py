# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare two complete Qwen frontier heads on one authenticated conditioning."""

import argparse
import array
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

VOCAB = 248320


def read(path):
    data = path.read_bytes()
    if len(data) != VOCAB * 4:
        raise ValueError("incomplete frontier: " + str(path))
    values = array.array("f")
    if values.itemsize != 4:
        raise ValueError("host float width differs")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError("nonfinite frontier: " + str(path))
    return values, hashlib.sha256(data).hexdigest()


def compare(left, right, target):
    n, nh = read(left)
    r, rh = read(right)
    nt = heapq.nlargest(5, range(VOCAB), key=n.__getitem__)
    rt = heapq.nlargest(5, range(VOCAB), key=r.__getitem__)
    nz = n[nt[0]] + math.log(math.fsum(math.exp(x - n[nt[0]]) for x in n))
    rz = r[rt[0]] + math.log(math.fsum(math.exp(x - r[rt[0]]) for x in r))
    deltas = [abs(a - b) for a, b in zip(n, r)]
    union = sorted(set(nt + rt))
    logprob_deltas = [abs(n[t] - nz - r[t] + rz) for t in union]
    return {
        "left_sha256": nh, "right_sha256": rh, "byte_exact": nh == rh,
        "differing_values": sum(a != b for a, b in zip(n, r)),
        "max_abs_logit_delta": max(deltas),
        "mean_abs_logit_delta": math.fsum(deltas) / VOCAB,
        "left_argmax": nt[0], "right_argmax": rt[0],
        "argmax_equal": nt[0] == rt[0],
        "left_top2_margin": n[nt[0]] - n[nt[1]],
        "right_top2_margin": r[rt[0]] - r[rt[1]],
        "left_loss_for_right_argmax": n[nt[0]] - n[rt[0]],
        "right_loss_for_left_argmax": r[rt[0]] - r[nt[0]],
        "left_top5": nt, "right_top5": rt,
        "union_top5_logprob_max_abs_delta": max(logprob_deltas),
        "union_top5_logprob_mean_abs_delta": math.fsum(logprob_deltas) / len(union),
        "target": target, "left_target_nll": nz - n[target],
        "right_target_nll": rz - r[target], "prior_near_tie_margin": 1.0,
        "right_mismatch_outside_prior_margin": (
            nt[0] != rt[0] and r[rt[0]] - r[rt[1]] > 1.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.target < VOCAB:
        parser.error("target outside Qwen vocabulary")
    result = compare(args.left, args.right, args.target)
    with args.out.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
