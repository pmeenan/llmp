#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Aggregation regressions using synthetic receipts outside the repository."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import summarize as aggregator


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmp-summary-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pins = {"trace_sha256": "trace", "image": "image", "swap_allowance_bytes": 256 << 20}
        self.manifest = {"repeats": 3, "cases": list(aggregator.CASES), "pins": self.pins,
                         "physical_memory_bytes": 128 << 30}
        self.order = []
        for repeat, latency in enumerate((9.0, 1.0, 5.0), 1):
            for case, (mode, cache, ballast, normal) in aggregator.CASES.items():
                name = f"{repeat:02d}-{case}"
                self.order.append(name)
                direction = {"first_token_s": latency, "first_text_s": latency + .01,
                             "eviction_and_load_s": latency / 2, "restore_api_s": .1,
                             "request_first_token_s": .2,
                             "completion": {"final": {"timings": {
                                 "prompt_n": 42, "cache_n": 18297, "prompt_ms": 100,
                                 "predicted_n": 118, "predicted_per_second": 40}}},
                             "io_through_first_token": {"read_bytes": repeat * 512,
                                 "write_bytes": 0, "pswpin": 0, "pswpout": 0, "oom_kill": 0}}
                state = {"resident_fraction": 1.0 if cache == "warm" else 0.0,
                         "resident_pages": 1000 if cache == "warm" else 0, "pages": 1000}
                if mode != "resident":
                    direction["weight_cache_before_switch"] = copy.deepcopy(state)
                trial = {"passed": True, "case": case, **self.pins, "mode": mode,
                         "cache_condition": cache, "ballast_gib": ballast, "normal_reference": normal,
                         "physical_memory_bytes": 128 << 30,
                         "peak_spill_logical_bytes": 100, "peak_spill_allocated_bytes": 4096,
                         "memory": {"max_total_minus_available_bytes": 100 << 30,
                                    "min_available_bytes": 28 << 30},
                         "trial_swap_traffic_bytes": 0, "A_continuation_matches_trace": True,
                         "B_output_matches_trace": True,
                         "A_to_B": copy.deepcopy(direction), "B_to_A": copy.deepcopy(direction)}
                if mode == "restore":
                    trial["B_to_A"]["state_cache_before_switch"] = copy.deepcopy(state)
                (self.root / name).mkdir()
                self.write(self.root / name / "result.json", trial)
        self.write(self.root / "run.json", self.manifest)
        self.write(self.root / "order.json", self.order)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def change_trial(self, name, mutation):
        path = self.root / name / "result.json"
        trial = json.loads(path.read_text())
        mutation(trial)
        self.write(path, trial)

    def test_three_samples_produce_median_and_observed_range(self):
        result = aggregator.summarize(self.root)
        self.assertEqual(result["trial_count"], 27)
        for group in result["cases"].values():
            self.assertEqual(group["n"], 3)
            self.assertEqual(group["A_to_B"]["first_token_s"], {"median": 5.0, "min": 1.0, "max": 9.0})
            self.assertEqual(group["B_to_A"]["first_token_read_bytes"], {"median": 1024, "min": 512, "max": 1536})

    def test_incomplete_order_is_rejected(self):
        self.write(self.root / "order.json", self.order[:-1])
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            aggregator.summarize(self.root)

    def test_failed_receipt_is_rejected(self):
        self.change_trial("02-pressure_restore", lambda t: t.update(passed=False))
        with self.assertRaisesRegex(RuntimeError, "Failed"):
            aggregator.summarize(self.root)

    def test_mixed_trace_image_and_swap_protocol_are_rejected(self):
        for key in ("trace_sha256", "image", "swap_allowance_bytes"):
            with self.subTest(key=key):
                self.change_trial("02-pressure_restore", lambda t: t.update({key: "different"}))
                with self.assertRaises(RuntimeError):
                    aggregator.summarize(self.root)
                self.change_trial("02-pressure_restore", lambda t: t.update({key: self.pins[key]}))

    def test_mixed_case_settings_and_physical_budget_are_rejected(self):
        originals = {"mode": "restore", "ballast_gib": 80, "cache_condition": "cold",
                     "normal_reference": False, "physical_memory_bytes": 128 << 30}
        changed = {"mode": "recompute", "ballast_gib": 0, "cache_condition": "warm",
                   "normal_reference": True, "physical_memory_bytes": 64 << 30}
        for key, value in changed.items():
            with self.subTest(key=key):
                self.change_trial("02-pressure_restore", lambda t: t.update({key: value}))
                with self.assertRaises(RuntimeError):
                    aggregator.summarize(self.root)
                self.change_trial("02-pressure_restore", lambda t: t.update({key: originals[key]}))

    def test_noncompliant_cache_receipts_are_rejected(self):
        for case, field, bad in [
                ("warm_restore", "state_cache_before_switch", {"resident_fraction": .998, "resident_pages": 998}),
                ("warm_restore", "weight_cache_before_switch", {"resident_fraction": .998, "resident_pages": 998}),
                ("cold_restore", "state_cache_before_switch", {"resident_fraction": .001, "resident_pages": 1}),
                ("cold_recompute", "weight_cache_before_switch", {"resident_fraction": .001, "resident_pages": 1})]:
            with self.subTest(case=case, field=field):
                path = self.root / ("02-" + case) / "result.json"
                original = path.read_text()
                self.change_trial("02-" + case, lambda t: t["B_to_A"][field].update(bad))
                with self.assertRaises(RuntimeError):
                    aggregator.summarize(self.root)
                path.write_text(original)


if __name__ == "__main__":
    unittest.main()
