#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare two runs of fp16_reference on the same trajectory.

Reports identities (logit and token hashes) and the differences between the
runs: the largest absolute logit difference, overall and per row, the RMS
difference, top-1 agreement, and for each disagreeing row the two runs'
top-1/top-2 margins. It infers no tolerance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VOCABULARY = 151936


def read_run(directory):
    record = json.loads((directory / "summary.json").read_text())
    rows, columns = record["tokens"], record["vocabulary"]
    if not (0 < rows <= 1040 and columns == VOCABULARY):
        raise ValueError("Unexpected logit dimensions")
    if record["logit_values"] != rows * columns:
        raise ValueError("Inconsistent logit count")
    if record["repeat_bit_differences"] or record["restore_bit_differences"]:
        raise ValueError("Repeat or restore failed")
    tokens = (directory / "tokens.txt").read_bytes()
    ids = [int(token) for token in tokens.splitlines()]
    if len(ids) != rows or any(token < 0 or token >= columns for token in ids):
        raise ValueError("Invalid token stream")
    path = directory / "logits.f32le"
    if path.stat().st_size != rows * columns * 4:
        raise ValueError("Wrong logit byte count")
    logits = np.memmap(path, dtype="<f4", mode="r", shape=(rows, columns))
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    with path.open("rb") as source:
        record["logits_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    record["tokens_sha256"] = hashlib.sha256(tokens).hexdigest()
    return record, ids, logits


def margin(row):
    top = np.partition(row, -2)[-2:]
    return float(top[1] - top[0])


def compare(left, right):
    a, ta, la = read_run(left)
    b, tb, lb = read_run(right)
    if ta != tb or la.shape != lb.shape or a["chunks"] != b["chunks"]:
        raise ValueError("Different teacher-forced trajectories")
    delta = np.abs(la.astype(np.float64) - lb.astype(np.float64))
    per_row = delta.max(axis=1)
    top_a, top_b = np.argmax(la, axis=1), np.argmax(lb, axis=1)
    disagreements = [
        {"row": int(i), "left_margin": margin(la[i]), "right_margin": margin(lb[i]),
         "row_max_absolute_difference": float(per_row[i])}
        for i in np.flatnonzero(top_a != top_b)
    ]
    return {
        "left": a,
        "right": b,
        "bit_identical": a["logits_sha256"] == b["logits_sha256"],
        "max_absolute_logit_difference": float(delta.max()),
        "root_mean_square_logit_difference": float(np.sqrt(np.mean(delta * delta))),
        "row_max_absolute_difference_quantiles": {
            q: float(np.quantile(per_row, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")
        },
        "top1_equal_rows": int(np.count_nonzero(top_a == top_b)),
        "rows": len(ta),
        "top1_disagreements": disagreements,
        "interpretation": "Observed difference between two runs; not an acceptance tolerance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))
