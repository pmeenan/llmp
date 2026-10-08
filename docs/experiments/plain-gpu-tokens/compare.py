# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute the bounded endpoint means/deltas from retained aggregates."""
import json
import math
import statistics
from pathlib import Path

results = json.loads(Path(__file__).with_name("results.json").read_text())
assert results["order"] == ["O1", "A1", "A2", "O2"]
for name, factor in results["factors"].items():
    for endpoint, metric in factor["metrics"].items():
        old, new = metric["off"], metric["on"]
        assert len(old) == len(new) == 2
        assert all(math.isfinite(v) and v > 0 for v in old + new)
        mean_old, mean_new = statistics.mean(old), statistics.mean(new)
        delta = 100 * (mean_new / mean_old - 1)
        movement = 100 * (old[1] / old[0] - 1)
        assert math.isclose(delta, metric["delta_percent"], abs_tol=1e-10)
        assert math.isclose(movement, metric["old_bookend_percent"], abs_tol=1e-10)
        assert metric["off_range"] == [min(old), max(old)]
        assert metric["on_range"] == [min(new), max(new)]
        print(f"{name} {endpoint}: off={mean_old:.9f}s on={mean_new:.9f}s "
              f"delta={delta:+.6f}% old_movement={movement:+.6f}%")
