# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""The package's documents and inventory (tools/jitllm_package.py, D-074)."""

import io
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import jitllm_package as package  # noqa: E402
import jitllm_sdk as sdklib  # noqa: E402
import jitllm_sources as srclib  # noqa: E402


class Extract(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.dir.name)
        (self.root / "header.h").write_text(
            "/*\n * Copyright (c) 1994\n * Someone\n *\n * Permission granted.\n * As is.\n */\nint x;\n")

    def tearDown(self):
        self.dir.cleanup()

    def test_from_to_plus_and_comment_prefix(self):
        spec = {"file": "sdk:header.h", "from": "Copyright", "to": "Permission", "plus": 1}
        self.assertEqual(package.extract(spec, self.root), "Copyright (c) 1994\nSomeone\n\nPermission granted.\nAs is.\n")

    def test_whole_file(self):
        self.assertTrue(package.extract({"file": "sdk:header.h"}, self.root).startswith("/*\n"))

    def test_missing_markers_fail(self):
        with self.assertRaises(package.PackageError):
            package.extract({"file": "sdk:header.h", "from": "nowhere"}, self.root)
        with self.assertRaises(package.PackageError):
            package.extract({"file": "sdk:header.h", "from": "Copyright", "to": "nowhere"}, self.root)
        with self.assertRaises(package.PackageError):
            package.extract({"file": "sdk:header.h", "from": "As is", "plus": 5}, self.root)
        with self.assertRaises(package.PackageError):
            package.extract({"file": "elsewhere:header.h"}, self.root)

    def test_component_notice_lines(self):
        (self.root / "u.hpp").write_text("a\n\t\t// Copyright (c) 2008 Someone <x@y>\nb\n")
        self.assertEqual(package.component_notice(self.root, "u.hpp:2-2"), "Copyright (c) 2008 Someone <x@y>\n")
        self.assertEqual(srclib.notice_range("LICENSE"), ("LICENSE", None))
        self.assertEqual(srclib.notice_range("a/b.h:3-9"), ("a/b.h", (3, 9)))
        self.assertEqual(srclib.notice_range("a/b.h:9-3"), ("a/b.h:9-3", None))


class ReadDeb(unittest.TestCase):
    def deb(self, members: dict[str, bytes]) -> pathlib.Path:
        out = io.BytesIO()
        out.write(b"!<arch>\n")
        for name, data in members.items():
            out.write(f"{name:<16}{0:<12}{0:<6}{0:<6}{100644:<8}{len(data):<10}`\n".encode())
            out.write(data + (b"\n" if len(data) % 2 else b""))
        fd, name = tempfile.mkstemp(suffix=".deb")
        os.close(fd)
        path = pathlib.Path(name)
        path.write_bytes(out.getvalue())
        self.addCleanup(path.unlink)
        return path

    @staticmethod
    def tar(files: dict[str, bytes]) -> bytes:
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return out.getvalue()

    def test_members(self):
        path = self.deb({"debian-binary": b"2.0\n", "control.tar.gz": self.tar({"./control": b"Package: x\n"}),
                         "data.tar.gz": self.tar({"./usr/bin/x": b"binary"})})
        parts = package.read_deb(path)
        self.assertEqual(parts["control"]["control"][1], b"Package: x\n")
        self.assertEqual(parts["data"]["usr/bin/x"][1], b"binary")

    def test_not_a_package(self):
        with self.assertRaises(package.PackageError):
            package.read_deb(self.deb({"debian-binary": b"3.0\n"}))


class Binaries(unittest.TestCase):
    """What a packaged executable may need and where it may look (D-060, D-076)."""

    RUNTIME = "usr/libexec/jitllm/jitllm-runtime"

    def facts(self, needed, runpath=()):
        return {"needed": list(needed), "glibc": (2, 38), "runpath": list(runpath)}

    def test_plain_executables_need_no_run_path(self):
        self.assertIsNone(package.binary_problem("usr/bin/jitllm", self.facts(["libc.so.6", "libcuda.so.1"])))
        self.assertIn("run path", package.binary_problem(
            "usr/bin/jitllm", self.facts(["libc.so.6"], ["$ORIGIN/../../lib/jitllm"])))
        self.assertIn("no dependency provides", package.binary_problem(
            "usr/bin/jitllm", self.facts(["libstdc++.so.6"])))

    def test_only_the_runtime_finds_cublas_and_only_privately(self):
        cublas = ["libc.so.6", "libcuda.so.1", "libcublas.so.13", "libcublasLt.so.13"]
        self.assertIsNone(package.binary_problem(self.RUNTIME, self.facts(cublas, ["$ORIGIN/../../lib/jitllm"])))
        self.assertIn("no run path", package.binary_problem(self.RUNTIME, self.facts(cublas)))
        self.assertIn("other than", package.binary_problem(self.RUNTIME, self.facts(cublas, ["/usr/local/cuda/lib64"])))
        self.assertIn("other than", package.binary_problem(
            self.RUNTIME, self.facts(cublas, ["$ORIGIN/../../lib/jitllm", "/tmp"])))
        self.assertIn("only the runtime", package.binary_problem(
            "usr/bin/jitllm", self.facts(cublas, ["$ORIGIN/../../lib/jitllm"])))
        # A run path with nothing private to find is refused too.
        self.assertIn("run path", package.binary_problem(
            self.RUNTIME, self.facts(["libc.so.6"], ["$ORIGIN/../../lib/jitllm"])))

    def test_cuda_packages_list_cublas(self):
        self.assertNotIn("usr/lib/jitllm/libcublas.so.13", package.expected_files(False))
        cuda = package.expected_files(True)
        self.assertEqual(cuda["usr/lib/jitllm/libcublas.so.13"], 0o644)
        self.assertEqual(cuda["usr/lib/jitllm/libcublasLt.so.13"], 0o644)
        self.assertIn("cublas", package.CUDA_UNITS)
        units = dict(package.shipped_units(tomllib.loads(package.PROVENANCE.read_text()), True))
        self.assertIn("nvidia-cuda-eula", units["cublas"]["notices"])
        self.assertNotIn("cublas", dict(package.shipped_units(tomllib.loads(package.PROVENANCE.read_text()), False)))


class InTreeUnits(unittest.TestCase):
    """jitLLM's files with third-party data are listed exactly when an executable is built from them (D-088)."""

    def test_listed_only_when_built_from(self):
        tables = (package.REPO / "src/tokenizer/unicode_data.cc").resolve()
        other = (package.REPO / "src/tokenizer/unicode.cc").resolve()
        self.assertEqual([name for name, _ in package.in_tree_units({tables, other})], ["unicode-data"])
        self.assertEqual(package.in_tree_units({other}), [])
        self.assertEqual(package.in_tree_units(set()), [])

    def test_records_match_the_files_and_notices(self):
        notices = tomllib.loads(package.PROVENANCE.read_text())["notices"]
        license_tag = "SPDX-" + "License-Identifier:"  # split, so REUSE does not read it as this file's tag
        for name, unit in package.IN_TREE_UNITS.items():
            with self.subTest(unit=name):
                self.assertIn(unit["notice"], notices)
                self.assertTrue((package.REPO / "LICENSES" / f"{unit['license']}.txt").is_file())
                for path in unit["files"]:
                    header = (package.REPO / path).read_text(encoding="utf-8").splitlines()[:5]
                    declared = next(line.split(":", 1)[1].strip() for line in header if license_tag in line)
                    self.assertIn(unit["license"], declared.split(" AND "))

    def test_license_pointer_names_unicode(self):
        self.assertIn("THIRD-PARTY-NOTICES", package.license_pointer("Unicode-3.0"))
        self.assertNotIn("Unicode", package.license_pointer("MIT"))

    def test_repository_files_are_all_listed(self):
        package.check_in_tree_units()

    def tree(self, files: dict[str, str]) -> pathlib.Path:
        """A synthetic repository root holding files, each with a header declaring its license."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = pathlib.Path(scratch.name)
        license_tag = "SPDX-" + "License-Identifier:"  # split, so REUSE does not read it as this file's tag
        for path, expression in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(f"// {license_tag} {expression}\nint x;\n" if expression else "int x;\n")
        return root

    def test_unlisted_or_missing_data_fails_loudly(self):
        listed = next(iter(package.IN_TREE_UNITS.values()))["files"][0]
        mixed = "Apache-2.0 AND Unicode-3.0"
        package.check_in_tree_units(self.tree({listed: mixed, "src/a/b.cc": "Apache-2.0", "src/a/c.h": "Apache-2.0",
                                               "src/a/adapted.cu": "MIT AND Apache-2.0"}))  # a lock component's
        for files in ({listed: mixed, "src/tokenizer/renamed_data.cc": mixed},  # moved or copied tables
                      {listed: mixed, "src/a/other.cc": "Apache-2.0 AND CC-BY-4.0"},  # any other data license
                      # Admitted by D-091, but no lock component or shipped unit records them yet, so no notice
                      # would ship: a CUB-derived file waits for the cccl unit's record.
                      {listed: mixed, "src/a/sort.cu": "Apache-2.0 AND BSD-3-Clause AND BSL-1.0"},
                      {listed: mixed, "src/a/hash.cc": "Apache-2.0 AND Zlib"},
                      {listed: mixed, "src/a/gpl.cc": "Apache-2.0 AND GPL-2.0-only"},  # copyleft (D-080)
                      {listed: mixed, "src/a/lgpl.cc": "Apache-2.0 AND LGPL-2.1-only"},
                      {listed: mixed, "src/a/nc.cc": "Apache-2.0 AND CC-BY-NC-4.0"},  # non-commercial
                      {listed: mixed, "src/a/dsl.cc": "Apache-2.0 AND LicenseRef-NVIDIA-cutlass-dsl"},  # proprietary
                      {listed: mixed, "src/a/ref.cc": "MIT AND LicenseRef-unknown"},  # unknown terms
                      {listed: mixed, "src/tokenizer/unicode_data.inc": mixed},  # tables in an included file
                      {listed: mixed, "src/a/no_header.cc": None},
                      {"src/tokenizer/renamed_data.cc": mixed}):  # the listed file is gone
            with self.subTest(files=sorted(files)), self.assertRaises(package.PackageError):
                package.check_in_tree_units(self.tree(files))

    def sidecar_tree(self, expression="Apache-2.0"):
        listed = next(iter(package.IN_TREE_UNITS.values()))["files"][0]
        root = self.tree({listed: "Apache-2.0 AND Unicode-3.0"})
        data = root / "src/metadata.json"
        data.write_text('{"schema":1}\n')
        sidecar = pathlib.Path(str(data) + ".license")
        tag = "SPDX-" + "License-Identifier:"
        copyright_tag = "SPDX-" + "FileCopyrightText:"
        sidecar.write_text(f"{copyright_tag} 2026 jitLLM contributors\n{tag} {expression}\n")
        return root, data, sidecar

    def test_json_uses_its_sidecar_and_remains_in_the_inventory(self):
        root, data, sidecar = self.sidecar_tree()
        self.assertEqual(package.declared_license(data), "Apache-2.0")
        self.assertEqual(package.declared_license(sidecar), "Apache-2.0")
        package.check_in_tree_units(root)
        sidecar.unlink()
        self.assertIsNone(package.declared_license(data))
        with self.assertRaises(package.PackageError):
            package.check_in_tree_units(root)

    def test_malformed_and_unrecorded_sidecars_are_refused(self):
        for expression in ("", "Apache-2.0 garbage", "LicenseRef-unknown"):
            with self.subTest(expression=expression):
                root, _, _ = self.sidecar_tree(expression)
                with self.assertRaises(package.PackageError):
                    package.check_in_tree_units(root)
        root, _, sidecar = self.sidecar_tree()
        sidecar.write_text("copyright only\n")
        with self.assertRaises(package.PackageError):
            package.check_in_tree_units(root)
        sidecar.write_bytes(b"\xff\n")
        with self.assertRaises(package.PackageError):
            package.check_in_tree_units(root)

    def test_conflicting_sidecar_declarations_are_refused(self):
        for padding in ("", "\n" * 12):
            with self.subTest(padding=len(padding)):
                root, _, sidecar = self.sidecar_tree()
                tag = "SPDX-" + "License-Identifier:"
                sidecar.write_text(sidecar.read_text() + padding + f"{tag} MIT\n")
                with self.assertRaisesRegex(package.PackageError, "conflicting"):
                    package.check_in_tree_units(root)
        root, data, _ = self.sidecar_tree()
        tag = "SPDX-" + "License-Identifier:"
        data.write_text(f"// {tag} MIT\n{{}}\n")
        with self.assertRaisesRegex(package.PackageError, "conflicting"):
            package.check_in_tree_units(root)

    def test_a_sidecar_cannot_rescue_code_without_an_embedded_header(self):
        root, data, sidecar = self.sidecar_tree()
        code = data.with_suffix(".cc")
        data.rename(code)
        sidecar.rename(pathlib.Path(str(code) + ".license"))
        with self.assertRaisesRegex(package.PackageError, "can hold a comment"):
            package.check_in_tree_units(root)

    def test_built_from_reads_ninja_and_fails_loudly(self):
        root = self.tree({})
        build = root / "build"
        build.mkdir()
        ninja = root / "ninja"
        ninja.write_text("#!/bin/sh\n[ \"$3 $4 $5 $6\" = \"-t inputs -0 -E\" ] || exit 2\n"
                         "printf '%s\\0' /abs/src/x.cc src/lib.a\n")
        ninja.chmod(0o755)
        self.assertEqual(package.built_from(build, ninja),
                         {pathlib.Path("/abs/src/x.cc"), (build / "src/lib.a").resolve()})
        ninja.write_text("#!/bin/sh\necho \"ninja: error: loading 'build.ninja'\" >&2\nexit 1\n")
        with self.assertRaisesRegex(package.PackageError, "build.ninja"):
            package.built_from(build, ninja)


class Provenance(unittest.TestCase):
    """Every shipped unit's notices can be extracted from this SDK and repository."""

    def test_every_notice_extracts(self):
        try:
            sdk = sdklib.load()
        except sdklib.SdkError as e:
            self.skipTest(str(e))
        if not (sdk.root / "sysroot").is_dir():
            self.skipTest(f"no SDK at {sdk.root}; run `mise run setup`")
        records = tomllib.loads(package.PROVENANCE.read_text())
        for name, notice in records["notices"].items():
            with self.subTest(notice=name):
                self.assertGreater(len(package.extract(notice["extract"], sdk.root)), 100)


class Purge(unittest.TestCase):
    """Run the real purge script with scratch paths and credential-command shims."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.data = self.root / "data"
        self.spill = self.data / "spill"
        (self.spill / "conversations" / "a").mkdir(parents=True)
        (self.spill / "conversations" / "a" / "record").write_text("tokens")
        (self.data / "models").mkdir()
        (self.data / "models" / "keep").write_text("model")
        self.victim = self.root / "victim"
        (self.victim / "conversations").mkdir(parents=True)
        (self.victim / "conversations" / "keep").write_text("protected")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.marker = self.root / "credentials"
        runuser = self.bin / "runuser"
        runuser.write_text(f"#!{sys.executable}\n"
                           "import os, pathlib, subprocess, sys\n"
                           "assert sys.argv[1:5] == ['-u', 'jitllm', '--', 'rm']\n"
                           "pathlib.Path(os.environ['PURGE_CREDENTIALS']).write_text('jitllm')\n"
                           "env = dict(os.environ, PURGE_DROPPED='1')\n"
                           "sys.exit(subprocess.run(sys.argv[4:], env=env).returncode)\n")
        runuser.chmod(0o755)
        rm = self.bin / "rm"
        rm.write_text(f"#!{sys.executable}\n"
                      "import os, pathlib, subprocess, sys\n"
                      "if os.environ.get('PURGE_RACE') == '1':\n"
                      "    spill = pathlib.Path(os.environ['PURGE_SPILL'])\n"
                      "    spill.rename(spill.with_name('oldspill'))\n"
                      "    spill.symlink_to(os.environ['PURGE_VICTIM'], target_is_directory=True)\n"
                      "    if os.environ.get('PURGE_DROPPED') == '1':\n"
                      "        sys.exit(1)  # simulate the protected directory refusing this user\n"
                      "sys.exit(subprocess.run(['/bin/rm', *sys.argv[1:]]).returncode)\n")
        rm.chmod(0o755)
        source = (TOOLS.parent / "packaging" / "debian" / "postrm").read_text()
        source = source.replace("/var/lib/jitllm", str(self.data))
        source = source.replace("/usr/sbin/runuser", str(runuser))
        source = source.replace("/run/systemd/system", str(self.root / "no-systemd"))
        source = source.replace("/usr/bin/deb-systemd-helper", str(self.root / "no-helper"))
        self.script = self.root / "postrm"
        self.script.write_text(source)
        self.env = dict(os.environ, DPKG_ROOT="", PATH=f"{self.bin}:{os.environ['PATH']}",
                        PURGE_CREDENTIALS=str(self.marker), PURGE_SPILL=str(self.spill),
                        PURGE_VICTIM=str(self.victim))

    def purge(self, race=False):
        return subprocess.run(["sh", str(self.script), "purge"],
                              env=dict(self.env, PURGE_RACE="1" if race else "0"),
                              capture_output=True, text=True, timeout=10)

    def test_default_conversations_deleted_as_service_user_only(self):
        result = self.purge()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.marker.read_text(), "jitllm")
        self.assertFalse((self.spill / "conversations").exists())
        self.assertTrue((self.data / "models" / "keep").exists())
        self.assertTrue((self.victim / "conversations" / "keep").exists())

    def test_parent_link_swap_cannot_gain_root_deletion(self):
        # rm's shim swaps the owner-controlled parent after postrm's link checks.
        # A privileged rm deletes the victim; the credential shim refuses it.
        result = self.purge(race=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.marker.read_text(), "jitllm")
        self.assertTrue((self.victim / "conversations" / "keep").exists())


if __name__ == "__main__":
    unittest.main()
