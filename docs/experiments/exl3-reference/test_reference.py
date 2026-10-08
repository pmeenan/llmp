# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from checks import compare, summary
from fetch import fetch_file


class Response(io.BytesIO):
    status = 200


class ReferenceChecks(unittest.TestCase):
    def test_exact_logits_and_signed_zero(self):
        a = np.array([[0.0, 1.0]], dtype=np.float32)
        self.assertTrue(compare(a, a.copy())["exact"])
        b = a.copy(); b[0, 0] = -0.0
        self.assertFalse(compare(a, b)["exact"])

    def test_nonfinite_and_shape_mismatch(self):
        a = np.array([[1.0, np.nan]], dtype=np.float32)
        self.assertFalse(compare(a, a)["finite"])
        with self.assertRaises(ValueError):
            compare(a, a.reshape(2, 1))

    def test_same_argmax_does_not_hide_logit_error(self):
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[1.0, 2.25]], dtype=np.float32)
        result = compare(a, b)
        self.assertEqual(result["top1_equal"], 1)
        self.assertEqual(result["max_abs"], .25)
        self.assertFalse(result["exact"])

    def test_cluster_bootstrap_keeps_requests_together(self):
        groups = np.array([[1.0]*64, [10.0]*64])
        result = summary(groups.ravel(), groups)
        self.assertEqual(result["bootstrap_unit"], "request")
        self.assertEqual(result["median_ci95"], [1.0, 10.0])

    def test_nonfinite_timings_fail(self):
        with self.assertRaises(ValueError): summary([1.0, float("nan")])

    def test_verified_atomic_download(self):
        data = b"verified fixture"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"model"
            with patch("urllib.request.urlopen", return_value=Response(data)):
                fetch_file("https://example.invalid/file", target, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_truncation_overflow_and_bad_hash_leave_no_payload(self):
        for data, size, checksum in [(b"abc", 4, "0"*64), (b"abc", 2, "0"*64), (b"abc", 3, "0"*64)]:
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                target = Path(directory)/"model"
                with patch("urllib.request.urlopen", return_value=Response(data)):
                    with self.assertRaises(ValueError): fetch_file("https://example.invalid/file", target, size, checksum)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_concurrent_destination_is_not_replaced(self):
        data = b"download"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"model"
            def opening(*args, **kwargs):
                target.write_bytes(b"other writer")
                return Response(data)
            with patch("urllib.request.urlopen", side_effect=opening):
                with self.assertRaises(FileExistsError):
                    fetch_file("https://example.invalid/file", target, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(), b"other writer")

    def test_existing_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"model"
            target.symlink_to(Path(directory)/"missing")
            with self.assertRaises(ValueError): fetch_file("unused", target, 0, "0"*64)


if __name__ == "__main__":
    unittest.main()
