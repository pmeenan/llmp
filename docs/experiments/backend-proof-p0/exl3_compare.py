#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare two exl3_heldout.py runs of one fixture, prefix by prefix.

For the prefill logits (every prefix in the second harness version; prefixes
through 145 in the first) and the 16-step suffix of each prefix: whether they are bit-identical, the largest absolute and RMS logit
difference, the per-row maximum's quantiles, top-1 agreement and the margins
at disagreeing rows. It infers no tolerance.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def margin(row):
    top = np.partition(row, -2)[-2:]
    return float(top[1] - top[0])


def diff(a, b):
    identical = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    a = a.reshape(-1, a.shape[-1]).astype(np.float64)
    b = b.reshape(-1, b.shape[-1]).astype(np.float64)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("shape mismatch or nonfinite logits")
    delta = np.abs(a - b)
    per_row = delta.max(axis=1)
    top_a, top_b = a.argmax(axis=1), b.argmax(axis=1)
    return {
        "bit_identical": identical,
        "max_absolute_logit_difference": float(delta.max()),
        "root_mean_square_logit_difference": float(np.sqrt(np.mean(delta * delta))),
        "row_max_quantiles": {q: float(np.quantile(per_row, float(q))) for q in ("0.5", "0.9", "1.0")},
        "rows": int(a.shape[0]),
        "top1_equal_rows": int(np.count_nonzero(top_a == top_b)),
        "top1_disagreements": [{"row": int(i), "left_margin": margin(a[i]), "right_margin": margin(b[i])}
                               for i in np.flatnonzero(top_a != top_b)],
    }


def compare(left, right):
    out = {"left": str(left), "right": str(right), "prefixes": []}
    for path in sorted(left.glob("logits-*.npz"), key=lambda p: int(p.stem.split("-")[1])):
        a, b = np.load(path), np.load(right / path.name)
        if not np.array_equal(a["ids"], b["ids"]):
            raise ValueError("different teacher-forced IDs")
        row = {"prefix": int(path.stem.split("-")[1]), "suffix": diff(a["suffix"], b["suffix"])}
        if "prefill" in a:
            row["prefill"] = diff(a["prefill"], b["prefill"])
        out["prefixes"].append(row)
    out["interpretation"] = "Observed difference between two runs; not an acceptance tolerance"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))
