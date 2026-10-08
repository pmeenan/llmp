# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Raw timing evidence is validated without NumPy or a GPU."""

import copy
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
PROOF = pathlib.Path(__file__).resolve().parents[2] / "docs/experiments/backend-proof-p0"
SPEC = importlib.util.spec_from_file_location("timing_stats", PROOF / "timing_stats.py")
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


class RawTimingEvidence(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.session = pathlib.Path(self.scratch.name)
        self.rows = [{"name": name, "rows": 1, "graph_replay_us": [10.0] * 31}
                     for name in ("a", "b")]
        for arm in ("reference", "candidate"):
            for number in (1, 2):
                for case_set in timing.SETS:
                    self.write(arm, number, case_set, self.rows)

    def write(self, arm, number, case_set, rows):
        directory = self.session / f"{arm}-{number}-{case_set}"
        directory.mkdir(exist_ok=True)
        (directory / "kernels.json").write_text(json.dumps(rows))

    def load(self):
        return timing.load_session(self.session, "reference", "candidate")

    def test_two_historical_blocks_and_four_protocol_blocks_are_supported(self):
        data = self.load()
        self.assertEqual(len(data), 6)
        self.assertTrue(all(len(arms["candidate"]) == 2 for arms in data.values()))
        for arm in ("reference", "candidate"):
            for number in (3, 4):
                for case_set in timing.SETS:
                    self.write(arm, number, case_set, self.rows)
        self.assertTrue(all(len(arms["reference"]) == 4 for arms in self.load().values()))

    def test_duplicate_or_empty_raw_cases_are_rejected(self):
        for rows in ([], [*self.rows, self.rows[0]]):
            self.write("candidate", 1, "real-40", rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load()

    def test_missing_case_in_one_block_is_rejected(self):
        self.write("candidate", 2, "real-45", self.rows[:1])
        with self.assertRaises(ValueError):
            self.load()

    def test_an_extra_block_in_only_one_case_set_is_rejected(self):
        self.write("candidate", 3, "synthetic", self.rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_one_sample_cannot_stand_in_for_31(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["graph_replay_us"] = [10.0]
        self.write("candidate", 1, "real-40", rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_invalid_sample_cannot_hide_behind_a_finite_median(self):
        for value in (math.nan, math.inf, -1.0, 0.0, True, "10"):
            rows = copy.deepcopy(self.rows)
            rows[0]["graph_replay_us"][0] = value
            self.write("candidate", 1, "real-40", rows)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()

    def test_same_arm_cannot_be_compared_with_itself(self):
        with self.assertRaises(ValueError):
            timing.load_session(self.session, "reference", "reference")
