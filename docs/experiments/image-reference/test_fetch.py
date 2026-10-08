# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        self.payload = b"verified"
        self.path = self.root / "weights.safetensors"
        self.path.write_bytes(self.payload)
        self.pins = {"model": "test/model", "revision": "a" * 40, "files": [{
            "path": self.path.name, "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest()}]}

    def test_valid_and_same_size_corruption(self):
        fetch.verify_directory(self.root, self.pins)
        self.path.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            fetch.verify_directory(self.root, self.pins)

    def test_unpinned_loader_override(self):
        (self.root / "model.safetensors").write_bytes(b"override")
        with self.assertRaises(ValueError):
            fetch.verify_directory(self.root, self.pins)

    def test_symlink_rejected(self):
        other = Path(self.temp.name) / "outside"
        self.path.rename(other)
        self.path.symlink_to(other)
        with self.assertRaises(ValueError):
            fetch.verify_directory(self.root, self.pins)

    def test_partial_symlink_is_not_followed(self):
        self.path.unlink()
        other = Path(self.temp.name) / "outside"
        other.write_bytes(b"untouched")
        self.path.with_name(self.path.name + ".partial").symlink_to(other)
        def download(command, stdout, check):
            stdout.write(self.payload)
        with patch.object(fetch.json, "loads", return_value=self.pins), \
             patch.object(fetch.subprocess, "run", side_effect=download), \
             patch.object(fetch.sys, "argv", ["fetch", str(self.root)]):
            fetch.main()
        self.assertEqual(other.read_bytes(), b"untouched")
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_bad_download_preserves_existing_file(self):
        self.path.write_bytes(b"old")
        def download(command, stdout, check):
            stdout.write(b"bad")
        with patch.object(fetch.json, "loads", return_value=self.pins), \
             patch.object(fetch.subprocess, "run", side_effect=download), \
             patch.object(fetch.sys, "argv", ["fetch", str(self.root)]):
            with self.assertRaises(ValueError):
                fetch.main()
        self.assertEqual(self.path.read_bytes(), b"old")
        self.assertFalse(list(self.root.glob(".fetch-*")))


if __name__ == "__main__":
    unittest.main()
