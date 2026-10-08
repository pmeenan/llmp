#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute helper-only means/ranges/deltas from the durable CPU samples."""
import json
import math
import statistics
import sys
from pathlib import Path


def compare(path):
    data = json.loads(Path(path).read_text())
    samples = data["batch_timings"]
    formulas = ["prefix-fill", "compressed-fill", "ring-fill", "prefix-check", "ring-check", "qsa-fill"]
    expected = [(formula, arm) for formula in formulas for arm in ["O1", "A1", "A2", "O2"]]
    assert [(row["formula"], row["arm"]) for row in samples] == expected
    result = {}
    for formula in formulas:
        rows = [row for row in samples if row["formula"] == formula]
        assert all(row["calls"] == 16 and row["rows"] == 512 and row["width"] == 4608
                   and math.isfinite(row["seconds"]) and row["seconds"] > 0 for row in rows)
        old = [row["seconds"] * 1000 / row["calls"] for row in rows if row["arm"].startswith("O")]
        new = [row["seconds"] * 1000 / row["calls"] for row in rows if row["arm"].startswith("A")]
        result[formula] = {"old_ms_mean": statistics.mean(old), "old_ms_range": [min(old), max(old)],
                           "new_ms_mean": statistics.mean(new), "new_ms_range": [min(new), max(new)],
                           "new_delta_percent": (statistics.mean(new) / statistics.mean(old) - 1) * 100,
                           "old_bookend_percent": (old[1] / old[0] - 1) * 100}
    return result


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("results.json")),
                     indent=2))
