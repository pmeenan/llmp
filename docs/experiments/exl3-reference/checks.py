# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure numerical checks and statistics for the external reference."""

import numpy as np


def summary(values, clusters=None):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("invalid timing samples")
    rng = np.random.default_rng(20260922)
    if clusters is None:
        medians = np.median(rng.choice(a, (2000, a.size), replace=True), axis=1)
    else:
        groups = np.asarray(clusters, dtype=np.float64)
        if groups.ndim != 2 or groups.size != a.size:
            raise ValueError("invalid request clusters")
        sampled = groups[rng.integers(0, len(groups), size=(2000, len(groups)))]
        medians = np.median(sampled.reshape(2000, -1), axis=1)
    return {"count": int(a.size), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)),
            "min": float(a.min()), "max": float(a.max()),
            "median_ci95": np.quantile(medians, [.025, .975]).tolist(),
            "bootstrap_unit": "request" if clusters is not None else "sample"}


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("logit shape/dtype mismatch")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"finite": False, "exact": False}
    d = a.astype(np.float64) - b.astype(np.float64)
    return {"finite": True, "exact": bool(np.array_equal(a.view(np.uint8), b.view(np.uint8))),
            "elements": int(a.size), "max_abs": float(np.abs(d).max()),
            "rms": float(np.sqrt(np.mean(d*d))),
            "top1_equal": int(np.sum(a.argmax(-1) == b.argmax(-1))),
            "positions": int(a.size // a.shape[-1])}


