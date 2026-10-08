# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed checks for P0's numerical and timing evidence (no GPU needed)."""

import copy
import contextlib
import importlib.util
import json
import io
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
PROOF = pathlib.Path(__file__).resolve().parents[2] / "docs/experiments/backend-proof-p0"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PROOF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = load("timing_protocol")
tierc = load("tierc")


def session():
    return {"session": "synthetic", "cases": [
        {"set": "synthetic", "name": name, "rows": 1, "ratio": 1.0,
         "reference": {"blocks": [{"median": m} for m in (10, 11, 10, 11)]},
         "candidate": {"blocks": [{"median": m} for m in (10, 11, 10, 11)]}}
        for name in ("a", "b")]}


class TimingEvidence(unittest.TestCase):
    def setUp(self):
        self.session = session()
        self.calibration = timing.calibrate([self.session])

    def test_missing_extra_duplicate_and_empty_cases_are_rejected(self):
        for cases in ([], self.session["cases"][:1], self.session["cases"] * 2,
                      [*self.session["cases"], {**self.session["cases"][0], "name": "extra"}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                timing.apply(self.calibration, [{**self.session, "cases": cases}])

    def test_confirmation_is_validated_even_when_primary_passes(self):
        with self.assertRaises(ValueError):
            timing.apply(self.calibration, [self.session, {**self.session, "cases": []}])

    def test_bad_ratio_or_block_cannot_pass(self):
        for value in (math.nan, math.inf, -1.0, 0.0):
            for field in ("ratio", "block"):
                bad = copy.deepcopy(self.session)
                if field == "ratio":
                    bad["cases"][0]["ratio"] = value
                else:
                    bad["cases"][0]["candidate"]["blocks"][0]["median"] = value
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    timing.apply(self.calibration, [bad])

    def test_each_case_requires_four_blocks(self):
        self.session["cases"][0]["candidate"]["blocks"].pop()
        with self.assertRaises(ValueError):
            timing.apply(self.calibration, [self.session])

    def test_calibration_cannot_silently_drop_a_case(self):
        with self.assertRaises(ValueError):
            timing.calibrate([self.session, {**self.session, "cases": self.session["cases"][:1]}])

    def test_zero_variance_with_systematic_slowdown_fails_aggregate(self):
        for case in self.session["cases"]:
            case["ratio"] = 1.01
            case["reference"]["blocks"] = [{"median": 100.0}] * 4
            case["candidate"]["blocks"] = [{"median": 101.0}] * 4
        result = timing.apply(self.calibration, [self.session, self.session])
        self.assertFalse(result["stage_passes"])
        self.assertEqual(result["aggregate"][0]["t"], math.inf)

    def test_no_noise_cannot_calibrate_a_threshold(self):
        for case in self.session["cases"]:
            for side in ("reference", "candidate"):
                case[side]["blocks"] = [{"median": 100.0}] * 4
        with self.assertRaises(ValueError):
            timing.calibrate([self.session])

    def test_archived_holdout_still_passes(self):
        archive = json.loads((PROOF / "timing.json").read_text())
        calibration = json.loads((PROOF / "timing-calibration.json").read_text())
        for name, expected_t in (("h7", 0.77), ("h8", -2.31)):
            record = archive["sessions"][name]
            cases = []
            for row in record["cases"]:
                case = dict(zip(record["columns"], row))
                for side in ("reference", "candidate"):
                    case[side] = {"blocks": [{"median": m} for m in case[f"{side}_block_medians_us"]]}
                cases.append(case)
            result = timing.apply(calibration, [{"session": name, "cases": cases}])
            self.assertTrue(result["stage_passes"])
            self.assertFalse(result["needs_confirmation"])
            self.assertAlmostEqual(result["aggregate"][0]["t"], expected_t, delta=0.01)


class NumericalEvidence(unittest.TestCase):
    def setUp(self):
        self.run = {"run": "synthetic", "prefixes": [
            {"prefix": p,
             **{part: {"finite": True, "rows": p if part == "prefill" else 16,
                       **dict.fromkeys(tierc.LOGIT, 0.01)} for part in ("prefill", "suffix")},
             "layers": {name: [0.01] * 24 for name in tierc.LAYER}}
            for p in sorted(tierc.PREFIXES)]}

    def test_complete_trajectory_has_all_750_statistics(self):
        self.assertEqual(len(tierc.flatten(self.run)), 750)

    def test_missing_or_duplicate_prefix_is_rejected(self):
        self.run["prefixes"].pop()
        with self.assertRaises(ValueError):
            tierc.flatten(self.run)
        self.run["prefixes"].append(copy.deepcopy(self.run["prefixes"][0]))
        with self.assertRaises(ValueError):
            tierc.flatten(self.run)

    def test_missing_layer_cannot_shrink_the_gate(self):
        self.run["prefixes"][0]["layers"]["k_relative_suffix_max"].pop()
        with self.assertRaises(ValueError):
            tierc.flatten(self.run)

    def test_nonfinite_or_negative_errors_cannot_be_hidden_by_max(self):
        for value in (math.nan, math.inf, -0.01):
            self.run["prefixes"][0]["layers"]["block_relative_rms"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                tierc.flatten(self.run)

    def test_wrong_trajectory_length_is_rejected(self):
        self.run["prefixes"][0]["suffix"]["rows"] = 1
        with self.assertRaises(ValueError):
            tierc.flatten(self.run)

    def test_nonfinite_logits_are_rejected(self):
        self.run["prefixes"][0]["prefill"]["finite"] = False
        with self.assertRaises(ValueError):
            tierc.flatten(self.run)

    def test_no_faults_cannot_claim_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = pathlib.Path(directory) / "scores.json"
            scores.write_text(json.dumps({"runs": [self.run, {**self.run, "run": "second"}]}))
            output = io.StringIO()
            with mock.patch("sys.argv", ["tierc.py", str(scores), "--thresholds", "2", "8"]), \
                    contextlib.redirect_stdout(output):
                tierc.main()
            self.assertFalse(json.loads(output.getvalue())["check"]["separates"])
