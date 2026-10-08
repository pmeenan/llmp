# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic capture regressions; run beside judge.py in the NumPy image."""

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


JUDGE = Path(__file__).with_name("judge.py")


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.oracle = self.root / "oracle.json"
        self.oracle.write_text(json.dumps({"steps": [
            {"id": 0, "logprob": -0.1, "top": [[0, -0.1], [1, -3.0]]}
        ] * 2}))
        self.a = self.logits("a", [[3, 0], [3, 0]])
        self.b = self.logits("b", [[3, 0], [3, 0]])

    def binary(self, name, values, code):
        path = self.root / name
        path.write_bytes(struct.pack(f"={len(values)}{code}", *values))
        return path

    def logits(self, name, rows):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        self.binary(f"{name}/probe.logits.f32", [v for row in rows for v in row], "f")
        return directory

    def judge(self, *args, ok=True):
        result = subprocess.run([sys.executable, "-B", str(JUDGE), *map(str, args)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if ok else 1, result.stdout + result.stderr)
        return json.loads(result.stdout) if result.stdout else None

    def greedy(self, directory, ok=True):
        return self.judge("greedy", self.oracle, directory, "probe", "--vocab", 2,
                          "--bound", 0.5, ok=ok)

    def test_greedy_checks_every_oracle_step(self):
        self.assertEqual(self.greedy(self.a)["agree"], 2)
        self.logits("a", [[3, 0], [0, 3]])
        self.assertEqual(self.greedy(self.a, ok=False)["violations"][0]["step"], 1)
        for rows in ([], [[3, 0]], [[3, 0]] * 3):
            with self.subTest(rows=len(rows)):
                self.greedy(self.logits("a", rows), ok=False)
        self.oracle.write_text(json.dumps({"steps": []}))
        self.greedy(self.b, ok=False)

    # The oracle prefers token 0 by 2.9 nats at every step (outside a 1.0
    # bound). [3, 0] agrees; [2.9, 3.0] is a tie flip (llmpalooza's own margin
    # 0.1, its NLL of token 0 0.644 nats over the oracle's).
    AGREE, FLIP = [3, 0], [2.9, 3.0]

    def tie_aware(self, rows, reference=None, ok=True):
        self.oracle.write_text(json.dumps({"steps": [
            {"id": 0, "logprob": -0.1, "top": [[0, -0.1], [1, -3.0]]}
        ] * len(rows)}))
        self.logits("a", rows)
        extra = []
        if reference is not None:
            self.logits("b", reference)
            extra = ["--reference", self.b]
        return self.judge("greedy", self.oracle, self.a, "probe", "--vocab", 2, "--bound", 1.0,
                          *extra, ok=ok)

    def test_tie_aware_rule_needs_a_reference(self):
        rows = [self.AGREE] * 299 + [self.FLIP]
        # Without the model's reference run, only the strict rule.
        result = self.tie_aware(rows, ok=False)
        self.assertFalse(result["strict_pass"])
        result = self.tie_aware(rows, [self.AGREE] * 300)
        self.assertFalse(result["strict_pass"])
        self.assertEqual([f["step"] for f in result["tie_flips"]], [299])

    def test_tie_aware_per_step_tolerance(self):
        clean = [self.AGREE] * 300
        # Llmpalooza decisive (margin 3): a violation.
        rows = [self.AGREE] * 299 + [[0, 3]]
        self.assertEqual(self.tie_aware(rows, clean, ok=False)["violations"][0]["step"], 299)
        # Margin 0.9, but its NLL 1.14 nats over the oracle's: a violation.
        rows = [self.AGREE] * 299 + [[2.0, 2.9]]
        self.assertEqual(self.tie_aware(rows, clean, ok=False)["violations"][0]["step"], 299)

    def test_tie_flips_are_capped(self):
        rows = [self.AGREE] * 2997 + [self.FLIP] * 3
        # Three flips against a reference with none: past max(2, 0).
        result = self.tie_aware(rows, [self.AGREE] * 3000, ok=False)
        self.assertEqual((result["violations"], result["flip_cap"]), ([], 2))
        # Two pass, against a reference with none or with two.
        two = [self.AGREE] * 2998 + [self.FLIP] * 2
        self.tie_aware(two, [self.AGREE] * 3000)
        self.assertEqual(self.tie_aware(two, two)["flip_cap"], 2)

    def test_a_reference_past_the_rule_is_refused(self):
        # A reference with three outside steps (past the default cap), or
        # one past the per-step tolerance, needs the owner's decision.
        rows = [self.AGREE] * 2997 + [self.FLIP] * 3
        self.assertIsNone(self.tie_aware(rows, rows, ok=False))
        decisive = [self.AGREE] * 299 + [[0, 3]]
        self.assertIsNone(self.tie_aware([self.AGREE] * 300, decisive, ok=False))

    def test_a_missing_reference_is_a_clean_error(self):
        self.oracle.write_text(json.dumps({"steps": [
            {"id": 0, "logprob": -0.1, "top": [[0, -0.1], [1, -3.0]]}] * 2}))
        empty = self.root / "empty"
        empty.mkdir()
        for args in (["--reference", empty], ["--reference", self.root / "absent"],
                     ["--reference"]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-B", str(JUDGE), "greedy", str(self.oracle), str(self.a),
                     "probe", "--vocab", "2", "--bound", "1.0", *map(str, args)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertTrue(result.stderr.strip())

    def test_continuation_is_relative_to_the_reference(self):
        # Every step agrees, but each holds the oracle's token less firmly:
        # the continuation ratio rises far past the reference's.
        result = self.tie_aware([[1, 0]] * 300, [self.AGREE] * 300, ok=False)
        self.assertGreater(result["continuation_ratio"],
                           result["reference"]["continuation_ratio"] + 0.005)
        # A model whose accepted run sits well above the oracle (as Qwen3.8's
        # does, about 2%): the same continuation, and one flip, pass.
        reference = [[1, 0]] * 300
        result = self.tie_aware(reference, reference)
        self.assertGreater(result["continuation_ratio"], 1.02)
        self.tie_aware([[1, 0]] * 299 + [self.FLIP], reference)

    def test_repeat_and_noise_require_matching_nonempty_captures(self):
        for command in ("repeat", "noise"):
            with self.subTest(command=command):
                self.assertEqual(self.judge(command, self.a, self.b, "probe", "--vocab", 2)
                                 ["steps"], 2)
                for rows in ([], [[3, 0]], [[3, 0]] * 3):
                    self.logits("b", rows)
                    self.judge(command, self.a, self.b, "probe", "--vocab", 2, ok=False)
                self.logits("a", [])
                self.logits("b", [])
                self.judge(command, self.a, self.b, "probe", "--vocab", 2, ok=False)
                self.logits("a", [[3, 0], [3, 0]])
                self.logits("b", [[3, 0], [3, 0]])

    def test_repeat_mismatch_fails_the_command(self):
        result = self.judge("repeat", self.a, self.b, "probe", "--vocab", 2)
        self.assertTrue(result["identical"])
        self.logits("b", [[3, 0], [0, 3]])
        result = self.judge("repeat", self.a, self.b, "probe", "--vocab", 2, ok=False)
        self.assertFalse(result["identical"])
        self.assertEqual(result["first_differing_step"], 1)

    def test_partial_values_rows_and_nonfinite_logits_are_refused(self):
        self.binary("a/probe.logits.f32", [3, 0, 3], "f")
        self.greedy(self.a, ok=False)
        self.logits("a", [[3, 0], [3, 0]])
        path = self.a / "probe.logits.f32"
        path.write_bytes(path.read_bytes() + b"\x00")
        self.greedy(self.a, ok=False)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.greedy(self.logits("a", [[3, 0], [value, 0]]), ok=False)

    def test_invalid_vocab_is_refused(self):
        for vocab in (0, -1):
            self.judge("greedy", self.oracle, self.a, "probe", "--vocab", vocab, ok=False)
        self.judge("noise", self.a, self.b, "probe", "--vocab", 1, ok=False)

    def test_perplexity_checks_the_complete_window(self):
        # The first half differs, so this also checks the scored interval.
        values = [math.log(9)] * 4 + [math.log(2)] * 3
        nll = self.binary("ppl.nll.f64", values, "d")
        oracle = self.root / "nll.json"
        oracle.write_text(json.dumps(values))
        for reference in ("2", oracle):
            result = self.judge("ppl", reference, nll, "--ctx", 8)
            self.assertEqual(result["scored"], 3)
            self.assertAlmostEqual(result["ratio"], 1)
        self.assertFalse(self.judge("ppl", "4", nll, "--ctx", 8, ok=False)["pass"])
        for size in (0, 5, 6, 8):
            with self.subTest(size=size):
                self.binary("ppl.nll.f64", [math.log(2)] * size, "d")
                self.judge("ppl", "2", nll, "--ctx", 8, ok=False)
        self.binary("ppl.nll.f64", values, "d")
        for size in (0, 5, 6, 8):
            oracle.write_text(json.dumps([math.log(2)] * size))
            self.judge("ppl", oracle, nll, "--ctx", 8, ok=False)

    def test_perplexity_refuses_partial_values_and_nonfinite_data(self):
        nll = self.binary("ppl.nll.f64", [math.log(2)] * 7, "d")
        nll.write_bytes(nll.read_bytes() + b"\x00")
        self.judge("ppl", "2", nll, "--ctx", 8, ok=False)
        for value in (math.nan, math.inf, -math.inf):
            self.binary("ppl.nll.f64", [value] + [math.log(2)] * 6, "d")
            self.judge("ppl", "2", nll, "--ctx", 8, ok=False)
        self.binary("ppl.nll.f64", [math.log(2)] * 7, "d")
        oracle = self.root / "nll.json"
        for values in ([math.nan] + [0] * 6, [[0]] * 7):
            oracle.write_text(json.dumps(values))
            self.judge("ppl", oracle, nll, "--ctx", 8, ok=False)
        for scalar in ("nan", "inf", "0", "-1"):
            self.judge("ppl", scalar, nll, "--ctx", 8, ok=False)

    def test_empty_scored_window_is_refused(self):
        nll = self.binary("ppl.nll.f64", [0], "d")
        for ctx in (0, 1, 2):
            self.judge("ppl", "1", nll, "--ctx", ctx, ok=False)


if __name__ == "__main__":
    unittest.main()
