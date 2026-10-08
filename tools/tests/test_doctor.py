# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Failed shared-library checks must not become successful doctor results."""

import dataclasses
import importlib.machinery
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import llmp_sdk as sdklib  # noqa: E402

loader = importlib.machinery.SourceFileLoader("doctor_failure_tests", str(TOOLS / "check-toolchain"))
spec = importlib.util.spec_from_loader(loader.name, loader)
check = importlib.util.module_from_spec(spec)
loader.exec_module(check)


class ToolLibraryChecks(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        sdk = dataclasses.replace(sdklib.load(), home=pathlib.Path(tmp), identity="test")
        self.sdk = dataclasses.replace(sdk, manifest={**sdk.manifest, "bin": {"llvm-strip": "llvm/bin/llvm-strip"}})
        tool = self.sdk.root / "bin/llvm-strip"
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b"\x7fELF" + b"\x00" * 20)
        tool.chmod(0o755)
        nvcc = self.sdk.root / "cuda/bin/nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.write_text("#!/bin/sh\n")
        nvcc.chmod(0o755)
        self.enterContext(mock.patch.object(check, "VERSION_PATTERNS", {}))

    def report(self, results):
        report = check.Report()
        with mock.patch.object(check, "capture", side_effect=results):
            check.check_tools(self.sdk, report)
        return report

    def test_failed_ldd_does_not_accept_a_broken_elf(self):
        report = self.report([
            subprocess.CompletedProcess([], 1, stdout="", stderr="not a dynamic executable"),
            subprocess.CompletedProcess([], 1, stdout="", stderr="invalid ELF header"),
        ])
        self.assertEqual(len(report.problems), 1)
        self.assertIn("invalid ELF header", report.problems[0])

    def test_failed_ldd_is_reported_for_a_dynamic_executable(self):
        report = self.report([
            subprocess.CompletedProcess([], 1, stdout="", stderr="dynamic loader failed"),
            subprocess.CompletedProcess([], 0, stdout="(NEEDED) Shared library: [libc.so.6]\n", stderr=""),
        ])
        self.assertEqual(len(report.problems), 1)
        self.assertIn("ldd failed", report.problems[0])

    def test_legitimate_static_executable_does_not_require_ldd_success(self):
        report = self.report([
            subprocess.CompletedProcess([], 1, stdout="", stderr="not a dynamic executable"),
            subprocess.CompletedProcess([], 0, stdout="There is no dynamic section in this file.\n", stderr=""),
        ])
        self.assertEqual(report.problems, [])

    def test_missing_tool_after_valid_tool_is_reported_once(self):
        self.sdk.manifest["bin"]["llvm-profdata"] = "llvm/bin/llvm-profdata"
        report = self.report([subprocess.CompletedProcess([], 0, stdout="libc.so.6 => /lib/libc.so.6", stderr="")])
        self.assertEqual(len(report.problems), 1)
        self.assertIn("bin/llvm-profdata: missing", report.problems[0])


if __name__ == "__main__":
    unittest.main()
