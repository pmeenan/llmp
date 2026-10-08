#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent-fact scoring regressions for the fixed recall probe."""

import unittest

from recall import score


CORRECT = ("Record 0063: svc-12; shard 8; timeout 83 ms.\n"
           "Record 0612: svc-0; shard 7; timeout 72 ms.")


class RecallScoreTests(unittest.TestCase):
    def test_correct_records_are_scored_independently(self):
        result = score(CORRECT)
        self.assertTrue(result["correct"])
        self.assertEqual(result["observed"], {"0063": ("svc-12", 8, 83), "0612": ("svc-0", 7, 72)})
        self.assertFalse(score(CORRECT.replace("shard 8", "shard 9"))["correct"])
        self.assertFalse(score(CORRECT.replace("timeout 72", "timeout 73"))["correct"])

    def test_case_and_record_zero_padding_do_not_change_facts(self):
        text = CORRECT.upper().replace("0063", "63").replace("0612", "000612")
        self.assertTrue(score(text)["correct"])

    def test_labels_cannot_be_swapped_while_numbers_stay_in_order(self):
        text = ("Record 0063: svc-12; timeout 8; shard 83 ms.\n"
                "Record 0612: svc-0; timeout 7; shard 72 ms.")
        self.assertFalse(score(text)["correct"])

    def test_negative_and_fractional_fields_are_not_truncated(self):
        for text in (CORRECT.replace("shard 8", "shard -8"),
                     CORRECT.replace("timeout 83", "timeout -83"),
                     CORRECT.replace("timeout 83", "timeout 83.5"),
                     CORRECT.replace("shard 8", "shard 8.5")):
            with self.subTest(text=text):
                self.assertFalse(score(text)["correct"])

    def test_missing_duplicate_or_wrong_records_are_rejected(self):
        for text in (CORRECT.splitlines()[0],
                     CORRECT + "\n" + CORRECT.splitlines()[0],
                     CORRECT.replace("0612", "0613"),
                     CORRECT.replace("0063", "9063")):
            with self.subTest(text=text):
                self.assertFalse(score(text)["correct"])


if __name__ == "__main__":
    unittest.main()
