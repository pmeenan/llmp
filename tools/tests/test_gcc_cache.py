# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""GCC runtime cache corruption must fail before it reaches a new SDK."""

import importlib.machinery
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import llmp_sdk as sdklib  # noqa: E402

loader = importlib.machinery.SourceFileLoader("setup_toolchain_cache_tests", str(TOOLS / "setup-toolchain"))
spec = importlib.util.spec_from_loader(loader.name, loader)
setup = importlib.util.module_from_spec(spec)
loader.exec_module(setup)


class GccCache(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = pathlib.Path(tmp)
        self.cache = self.root / "cached-runtime"
        self.cache.mkdir()
        self.header = self.cache / "include/c++/16/unused-header"
        self.header.parent.mkdir(parents=True)
        self.header.write_text("original header\n")
        self.inputs = {"recipe": 2, "target": "x86_64-linux-gnu", "sources": {"source/gcc": "locked"}}
        (self.cache / "build-inputs.json").write_text(json.dumps(self.inputs))
        self.sdk = mock.Mock(triple="x86_64-linux-gnu")
        self.comp = {"dest": "gcc/{triple}"}
        self.staging = self.root / "staging"
        self.dest = self.staging / "gcc/x86_64-linux-gnu"
        self.enterContext(mock.patch.object(setup, "gcc_cache_dir", return_value=(self.cache, self.inputs)))
        self.enterContext(mock.patch.object(setup, "artifact_keys_for_gcc", return_value=["source/gcc"]))
        self.build = self.enterContext(mock.patch.object(setup, "build_gcc"))

    def install(self):
        return setup.install_gcc(self.sdk, self.comp, self.staging, jobs=1, keep=False)

    def assert_rejected(self, reason):
        with self.assertRaisesRegex(sdklib.SdkError, reason) as cm:
            self.install()
        self.assertIn("remove this cached build and rerun setup", str(cm.exception))
        self.assertIn(str(self.cache), str(cm.exception))
        self.assertFalse(self.dest.exists())
        self.build.assert_not_called()

    def test_valid_cache_reused_without_copying_integrity_records(self):
        setup.seal_gcc_cache(self.cache)
        result = self.install()
        self.build.assert_not_called()
        self.assertEqual((self.dest / "include/c++/16/unused-header").read_text(), "original header\n")
        self.assertEqual(result["inputs"], self.inputs)
        self.assertFalse((self.dest / "build-inputs.json").exists())
        self.assertFalse((self.dest / setup.GCC_CACHE_RECEIPT).exists())

    def test_modified_header_rejected_before_new_sdk_can_seal_it(self):
        setup.seal_gcc_cache(self.cache)
        self.header.write_text("corrupted header\n")
        self.assert_rejected("modified after its build")

    def test_missing_header_rejected(self):
        setup.seal_gcc_cache(self.cache)
        self.header.unlink()
        self.assert_rejected("modified after its build")

    def test_added_output_rejected(self):
        setup.seal_gcc_cache(self.cache)
        (self.cache / "unexpected-library.a").write_bytes(b"extra archive")
        self.assert_rejected("modified after its build")

    def test_changed_output_mode_rejected(self):
        self.header.chmod(0o644)
        setup.seal_gcc_cache(self.cache)
        self.header.chmod(0o755)
        self.assert_rejected("modified after its build")

    def test_old_cache_without_integrity_record_needs_explicit_repair(self):
        self.assert_rejected("no readable integrity record")

    def test_wrong_build_inputs_rejected_even_when_content_digest_matches(self):
        (self.cache / "build-inputs.json").write_text(json.dumps({"recipe": "wrong"}))
        setup.seal_gcc_cache(self.cache)
        self.assert_rejected("different build inputs")

    def test_unreadable_receipt_rejected(self):
        for data in (b"{", b"\xff"):
            with self.subTest(data=data):
                (self.cache / setup.GCC_CACHE_RECEIPT).write_bytes(data)
                self.assert_rejected("no readable integrity record")

    def test_invalid_receipt_shape_rejected(self):
        for receipt in ([], {}, {"schema": 2}):
            with self.subTest(receipt=receipt):
                (self.cache / setup.GCC_CACHE_RECEIPT).write_text(json.dumps(receipt))
                self.assert_rejected("invalid receipt")


if __name__ == "__main__":
    unittest.main()
