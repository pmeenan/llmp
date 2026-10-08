#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare finite, identically tokenized reference logits; no tolerance is inferred."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_run(directory):
    record = json.loads((directory / "summary.json").read_text())
    rows, columns = record["tokens"], record["vocabulary"]
    if not (0 < rows <= 512 and columns == 151936):
        raise ValueError("Unexpected logit dimensions")
    if record["logit_values"] != rows * columns:
        raise ValueError("Inconsistent logit count")
    if record["repeat_bit_differences"] or record["restore_bit_differences"]:
        raise ValueError("Reference repeat/restore failed")
    tokens = (directory / "tokens.txt").read_bytes()
    ids = [int(token) for token in tokens.splitlines()]
    if len(ids) != rows or any(token < 0 or token >= columns for token in ids):
        raise ValueError("Invalid token stream")
    path = directory / "logits.f32le"
    if path.stat().st_size != rows * columns * 4:
        raise ValueError("Wrong logit byte count")
    logits = np.memmap(path, dtype="<f4", mode="r", shape=(rows, columns))
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite reference logits")
    with path.open("rb") as source:
        record["logits_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    record["tokens_sha256"] = hashlib.sha256(tokens).hexdigest()
    return record, ids, logits


def compare(left, right):
    a, ta, la = read_run(left)
    b, tb, lb = read_run(right)
    if ta != tb or la.shape != lb.shape:
        raise ValueError("Different teacher-forced token streams")
    delta = la.astype(np.float64) - lb.astype(np.float64)
    return {
        "left": a,
        "right": b,
        "max_absolute_logit_difference": float(np.abs(delta).max()),
        "root_mean_square_logit_difference": float(np.sqrt(np.mean(delta * delta))),
        "top1_equal_rows": int(np.count_nonzero(np.argmax(la, axis=1) == np.argmax(lb, axis=1))),
        "rows": len(ta),
        "interpretation": "Observed backend difference; not a llmpalooza acceptance tolerance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))
