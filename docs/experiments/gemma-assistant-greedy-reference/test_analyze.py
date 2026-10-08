#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny operand-independent controls for the diagnostic judge/score semantics."""
import math
import unittest

import analyze


class Semantics(unittest.TestCase):
    def test_anchor_and_pending_are_not_extra_committed_rows(self):
        proposal = [0, 1, 2, 3]
        for keep in range(1, 5):
            rows = []
            for row in range(4):
                chosen = proposal[row+1] if row < keep-1 else 4
                rows.extend([float(i == chosen) for i in range(5)])
            self.assertEqual(analyze.judge(proposal, rows, 5), (keep, 4))

    def test_reference_ties_do_not_become_positive_margin_failures(self):
        tied = analyze.row_compare([0, 2, 1], [2, 2, 0])
        self.assertTrue(tied["exact_tie_difference"])
        self.assertFalse(tied["strict_positive_difference"])
        strict = analyze.row_compare([0, 3, 1], [3, 2, 0])
        self.assertTrue(strict["strict_positive_difference"])
        self.assertEqual(strict["reference_margin"], 1)

    def test_conditional_score_uses_full_vocabulary_stably(self):
        self.assertAlmostEqual(analyze.nll([1000, 1000, 1000, 1000], 2), math.log(4))
        self.assertAlmostEqual(analyze.nll([-1000, -1000, -1000, -1000], 2), math.log(4))


if __name__ == "__main__":
    unittest.main()
