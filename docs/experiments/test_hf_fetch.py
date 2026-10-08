# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hf_fetch


def row(path, payload):
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "snapshot"
        self.root.mkdir()

    def test_unsafe_paths_rejected(self):
        for bad in ("../escape", "/abs", "a/../../b", "a//b", "./a", ""):
            with self.assertRaises(ValueError, msg=bad):
                hf_fetch.destination(self.root, bad)

    def test_symlinked_directory_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (self.root / "sub").symlink_to(outside)
        with self.assertRaises(ValueError):
            hf_fetch.destination(self.root, "sub/file.bin")

    def test_fetch_verifies_and_publishes(self):
        payload = b"weights"
        pinned = row("sub/w.bin", payload)
        target = hf_fetch.destination(self.root, pinned["path"])
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            self.assertEqual(hf_fetch.fetch("https://example.invalid/w", target, pinned), "fetched")
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(hf_fetch.fetch("https://example.invalid/w", target, pinned), "present")
        self.assertEqual([p.name for p in target.parent.iterdir()], ["w.bin"])

    def test_corrupt_or_oversized_download_leaves_nothing(self):
        pinned = row("w.bin", b"weights")
        target = self.root / "w.bin"
        for body in (b"wrongly", b"weights+extra"):
            with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
                with self.assertRaises(ValueError):
                    hf_fetch.fetch("https://example.invalid/w", target, pinned)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_tampered_file_is_an_error(self):
        pinned = row("w.bin", b"weights")
        (self.root / "w.bin").write_bytes(b"WEIGHTS")
        with self.assertRaises(ValueError):
            hf_fetch.verified(self.root / "w.bin", pinned)

    def test_tree_must_match_pins_exactly(self):
        pinned = [row("a.bin", b"a")]
        (self.root / "a.bin").write_bytes(b"a")
        hf_fetch.verify_tree(self.root, pinned)
        (self.root / "extra.py").write_bytes(b"print()")
        with self.assertRaises(ValueError):
            hf_fetch.verify_tree(self.root, pinned)


if __name__ == "__main__":
    unittest.main()
