#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare import compare, read_run


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.left = Path(self.temporary.name) / "left"
        self.right = Path(self.temporary.name) / "right"
        for directory in (self.left, self.right):
            directory.mkdir()
            (directory / "summary.json").write_text(json.dumps({
                "tokens": 2, "vocabulary": 151936, "logit_values": 303872,
                "repeat_bit_differences": 0, "restore_bit_differences": 0,
            }))
            (directory / "tokens.txt").write_text("1\n2\n")
            logits = np.zeros((2, 151936), dtype="<f4")
            logits[:, 3] = 1
            logits.tofile(directory / "logits.f32le")

    def test_exact_and_changed_predictions(self):
        self.assertEqual(compare(self.left, self.right)["max_absolute_logit_difference"], 0)
        logits = np.memmap(self.right / "logits.f32le", dtype="<f4", mode="r+")
        logits[4] = 2
        logits.flush()
        result = compare(self.left, self.right)
        self.assertEqual(result["top1_equal_rows"], 1)
        self.assertEqual(result["max_absolute_logit_difference"], 2)

    def test_reject_changed_tokens(self):
        (self.right / "tokens.txt").write_text("1\n3\n")
        with self.assertRaisesRegex(ValueError, "Different teacher-forced"):
            compare(self.left, self.right)

    def test_reject_short_logits(self):
        (self.right / "logits.f32le").write_bytes(b"\0" * 4)
        with self.assertRaisesRegex(ValueError, "Wrong logit byte count"):
            read_run(self.right)

    def test_reject_nonfinite_logits(self):
        logits = np.memmap(self.right / "logits.f32le", dtype="<f4", mode="r+")
        logits[0] = np.nan
        logits.flush()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            read_run(self.right)

    def test_reject_failed_restore(self):
        path = self.right / "summary.json"
        record = json.loads(path.read_text())
        record["restore_bit_differences"] = 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Reference repeat/restore failed"):
            read_run(self.right)


if __name__ == "__main__":
    unittest.main()
