#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute complete endpoint means, ranges and bookend movement."""
import json
import math
import statistics
import sys
from pathlib import Path


def compare(path):
    data = json.loads(Path(path).read_text())
    result = {}
    for name, profile in data["profiles"].items():
        samples = profile["samples"]
        assert [row["arm"] for row in samples] == ["O1", "A1", "A2", "O2"]
        assert [row["group_stores"] for row in samples] == [False, True, True, False]
        result[name] = {}
        for field in ["prefill_seconds", "decode_seconds", "paid_seconds"]:
            values = [row[field] for row in samples]
            assert all(math.isfinite(value) and value > 0 for value in values)
            off, on = [values[0], values[3]], values[1:3]
            result[name][field] = {
                "off_mean": statistics.mean(off), "on_mean": statistics.mean(on),
                "off_range": [min(off), max(off)], "on_range": [min(on), max(on)],
                "delta_percent": 100 * (statistics.mean(on) / statistics.mean(off) - 1),
                "off_bookend_percent": 100 * (off[1] / off[0] - 1),
                "on_pair_percent": 100 * (on[1] / on[0] - 1),
            }
    return result


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1] if len(sys.argv) > 1 else
                             Path(__file__).with_name("results.json")), indent=2))
