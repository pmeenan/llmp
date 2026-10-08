#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute the bounded state-preparation means, ranges and drift."""
import json
import math
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).with_name("results.json")
data = json.loads(path.read_text())
output = {}
for name, factor in data["factors"].items():
    arms = factor["arms"]
    assert [arm["name"] for arm in arms] == ["O1", "A1", "A2", "O2"]
    assert [arm["prepare_state"] for arm in arms] == [False, True, True, False]
    metrics = {}
    for key in ["prefill_seconds", "decode_seconds", "paid_prefill_decode_seconds"]:
        values = [arm[key] for arm in arms]
        assert all(math.isfinite(value) and value > 0 for value in values)
        off, on = [values[0], values[3]], values[1:3]
        computed = {
            "off_mean": statistics.mean(off), "on_mean": statistics.mean(on),
            "delta_percent": 100 * (statistics.mean(on) / statistics.mean(off) - 1),
            "off_range": [min(off), max(off)], "on_range": [min(on), max(on)],
            "off_bookend_percent": 100 * (off[1] / off[0] - 1),
            "on_pair_percent": 100 * (on[1] / on[0] - 1),
        }
        assert computed == factor["metrics"][key], (name, key)
        metrics[key] = computed
    output[name] = metrics
print(json.dumps(output, indent=2))
