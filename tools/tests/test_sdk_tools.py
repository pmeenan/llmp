# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SDK tools' pure logic (no downloads, builds or SDK needed).

Run: python3 -m unittest discover -s tools/tests
"""

import copy
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
import zipfile
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import jitllm_sdk as sdklib  # noqa: E402


def load_script(name: str):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


setup = load_script("setup-toolchain")
check = load_script("check-toolchain")


class ManifestAndLock(unittest.TestCase):
    def test_every_component_artifact_is_locked(self):
        for arch in ("x86_64", "aarch64"):
            sdk = sdklib.load(arch)
            keys = setup.artifact_keys(sdk)  # raises SdkError on a key missing from the lock
            self.assertTrue(keys)
            for key in keys:
                entry = sdk.artifact(key)
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(entry["size"], 0)
                self.assertTrue(entry["urls"])

    def test_identity_depends_on_host_and_inputs(self):
        x86, arm = sdklib.load("x86_64"), sdklib.load("aarch64")
        self.assertTrue(x86.identity.startswith("x86_64-"))
        self.assertEqual(x86.identity.split("-", 1)[1], arm.identity.split("-", 1)[1])
        self.assertEqual(set(x86.inputs), {str(p.relative_to(sdklib.REPO)) for p in sdklib.IDENTITY_INPUTS})

    def test_bin_links_avoid_nvcc(self):
        # NVCC finds nvcc.profile beside its invocation path, so a bin/ link breaks it.
        self.assertNotIn("nvcc", sdklib.load("x86_64").manifest["bin"])

    def test_python_tools_are_pinned_for_the_check_gates_host(self):
        sdk = sdklib.load("x86_64")
        wheels = {name: comp for name, comp in sdk.components() if comp["kind"] == "wheels"}
        self.assertEqual(list(wheels), ["reuse"])  # the check gate runs on x86-64 hosts only (D-061)
        for name, comp in wheels.items():
            self.assertIn(name, sdk.manifest["versions"])
            self.assertEqual(sdk.artifact(f"wheel/{name}")["version"], sdk.manifest["versions"][name])
            self.assertEqual(len(comp["packages"]), len(set(comp["packages"])))
            self.assertTrue(comp["module"])
        self.assertIn("libmagic1t64", [n for n, _, _ in sdklib.prerequisites(sdk)])  # python-magic


class CrossRuntime(unittest.TestCase):
    def test_runtime_closure_is_locked_and_assembled_after_gcc(self):
        sdk = sdklib.load("x86_64")
        names = [name for name, _ in sdk.components()]
        runtime = sdk.manifest["components"]["sysroot-aarch64-runtime"]
        self.assertGreater(names.index("sysroot-aarch64-runtime"), names.index("gcc-runtime-aarch64"))
        self.assertNotIn("sysroot-aarch64-runtime", [n for n, _ in sdklib.load("aarch64").components()])
        self.assertEqual(runtime["kind"], "debs")
        self.assertEqual(runtime["arch"], "arm64")
        self.assertEqual(runtime["dest"], sdk.manifest["components"]["sysroot-aarch64"]["dest"])
        self.assertEqual(runtime["packages"], ["gcc-14-base", "libgcc-s1"])
        base = sdk.artifact("deb/gcc-14-base/arm64")
        lib = sdk.artifact("deb/libgcc-s1/arm64")
        self.assertEqual(base["version"], lib["version"])
        self.assertEqual(base["source"], lib["source"])
        # libgcc-s1's authenticated dependency is libc6 >= 2.35.
        self.assertGreaterEqual(tuple(map(int, sdk.artifact("deb/libc6/arm64")["version"].split("-")[0].split("."))),
                                (2, 35))

    def test_runtime_pin_changes_do_not_rebuild_static_gcc(self):
        sdk = sdklib.load("x86_64")
        manifest = copy.deepcopy(sdk.manifest)
        manifest["hosts"]["x86_64"]["components"].remove("sysroot-aarch64-runtime")
        del manifest["components"]["sysroot-aarch64-runtime"]
        lock = copy.deepcopy(sdk.lock)
        del lock["artifacts"]["deb/gcc-14-base/arm64"]
        del lock["artifacts"]["deb/libgcc-s1/arm64"]
        before = dataclasses.replace(sdk, manifest=manifest, lock=lock)
        with mock.patch.object(setup, "tool_version", return_value="same host tool"):
            for name in ("gcc-runtime", "gcc-runtime-aarch64"):
                comp = sdk.manifest["components"][name]
                self.assertEqual(setup.gcc_cache_dir(before, comp), setup.gcc_cache_dir(sdk, comp))

    def test_runtime_merge_keeps_static_outputs_and_copyright_link(self):
        sdk = sdklib.load("x86_64")
        comp = sdk.manifest["components"]["sysroot-aarch64-runtime"]
        with tempfile.TemporaryDirectory() as tmp:
            staging = pathlib.Path(tmp)
            dest = staging / comp["dest"]
            static = dest / "opt/gcc/lib/gcc/aarch64-linux-gnu/16/libgcc.a"
            static.parent.mkdir(parents=True)
            static.write_bytes(b"unchanged static GCC 16")
            libc = dest / "usr/lib/aarch64-linux-gnu/libc.so.6"
            libc.parent.mkdir(parents=True)
            libc.write_bytes(b"unchanged glibc")
            setup.relative_link(dest / "lib", dest / "usr/lib")
            def extract(deb, root):
                if deb.name == "gcc-14-base":
                    doc = root / "usr/share/doc/gcc-14-base"
                    doc.mkdir(parents=True)
                    (doc / "copyright").write_bytes(b"GCC Runtime Library Exception")
                else:
                    lib = root / "usr/lib/aarch64-linux-gnu/libgcc_s.so.1"
                    lib.parent.mkdir(parents=True)
                    lib.write_bytes(b"separate shared runtime")
                    (root / "usr/share/doc/libgcc-s1").symlink_to("gcc-14-base")
            with (mock.patch.object(setup, "fetch", side_effect=lambda _, key: pathlib.Path(key.split("/")[1])),
                  mock.patch.object(setup, "extract_deb", side_effect=extract)):
                receipt = setup.install_debs(sdk, comp, staging)
                self.assertEqual(receipt["artifacts"], ["deb/gcc-14-base/arm64", "deb/libgcc-s1/arm64"])
                self.assertEqual(static.read_bytes(), b"unchanged static GCC 16")
                self.assertEqual(libc.read_bytes(), b"unchanged glibc")
                self.assertEqual((dest / "lib/aarch64-linux-gnu/libgcc_s.so.1").read_bytes(), b"separate shared runtime")
                self.assertEqual((dest / "usr/share/doc/libgcc-s1/copyright").read_bytes(), b"GCC Runtime Library Exception")
                # A second merge cannot overwrite a runtime already present.
                with self.assertRaisesRegex(sdklib.SdkError, "already exists"):
                    setup.install_debs(sdk, comp, staging)


class Provenance(unittest.TestCase):
    """Every SDK component and host prerequisite has a D-017 provenance record (D-071)."""

    def setUp(self):
        self.records = tomllib.loads((sdklib.TOOLCHAINS / "provenance.toml").read_text())
        self.units = self.records["units"]

    def recorded(self, field: str) -> set[str]:
        return {item for unit in self.units.values() for item in unit.get(field, [])}

    def test_every_artifact_prerequisite_and_mise_tool_is_recorded(self):
        def unarched(key: str) -> str:  # deb/<package>/<arch> and archive/<name>/<arch>
            return key.rsplit("/", 1)[0] if key.startswith(("deb/", "archive/")) else key
        hosts = sdklib.load("x86_64").manifest["hosts"]
        artifacts = {unarched(k) for arch in hosts for k in setup.artifact_keys(sdklib.load(arch))}
        packages = {name for arch in hosts for name, _, _ in sdklib.prerequisites(sdklib.load(arch))}
        tools = {"mise", *tomllib.loads((sdklib.REPO / "mise.toml").read_text())["tools"]}
        self.assertEqual(self.recorded("artifacts"), artifacts)
        self.assertEqual(self.recorded("packages"), packages)
        self.assertEqual(self.recorded("tools"), tools)

    def test_records_are_complete(self):
        self.assertEqual(self.records["schema"], 1)
        for name, unit in self.units.items():
            with self.subTest(unit=name):
                self.assertTrue(unit.keys() & {"artifacts", "packages", "tools"})
                self.assertIn(unit["category"], ("tool", "platform"))
                self.assertTrue(unit["license"] and unit["enters"])
                self.assertIsInstance(unit["ships"], bool)
                self.assertLessEqual(set(unit["notices"]), set(self.records["notices"]))
                if not unit["ships"]:
                    self.assertEqual(unit["notices"], [])
        referenced = {n for unit in self.units.values() for n in unit["notices"]}
        self.assertEqual(referenced, set(self.records["notices"]))
        for name, notice in self.records["notices"].items():
            with self.subTest(notice=name):
                self.assertEqual(set(notice), {"text", "source", "when", "extract"})
                self.assertLessEqual(set(notice["extract"]), {"file", "from", "to", "plus"})
                self.assertRegex(notice["extract"]["file"], r"^(sdk|repo):[^/]")


class Roots(unittest.TestCase):
    SPEC = {"env": "JITLLM_TEST_HOME", "xdg": "JITLLM_TEST_XDG", "default": "~/.local/share", "suffix": "jitllm/sdk"}

    def test_override_then_xdg_then_default(self):
        with mock.patch.dict(os.environ, {"JITLLM_TEST_HOME": "/opt/sdk", "JITLLM_TEST_XDG": "/x"}):
            self.assertEqual(sdklib.resolve_root(self.SPEC), pathlib.Path("/opt/sdk"))
        with mock.patch.dict(os.environ, {"JITLLM_TEST_XDG": "/x"}):
            os.environ.pop("JITLLM_TEST_HOME", None)
            self.assertEqual(sdklib.resolve_root(self.SPEC), pathlib.Path("/x/jitllm/sdk"))
        with mock.patch.dict(os.environ, {"HOME": "/home/u"}):
            os.environ.pop("JITLLM_TEST_HOME", None)
            os.environ.pop("JITLLM_TEST_XDG", None)
            self.assertEqual(sdklib.resolve_root(self.SPEC), pathlib.Path("/home/u/.local/share/jitllm/sdk"))

    def test_relative_override_is_rejected(self):
        with mock.patch.dict(os.environ, {"JITLLM_TEST_HOME": "relative/dir"}):
            with self.assertRaises(sdklib.SdkError):
                sdklib.resolve_root(self.SPEC)


class Prerequisites(unittest.TestCase):
    def test_lists_parse(self):
        for arch in ("x86_64", "aarch64"):
            reqs = sdklib.prerequisites(sdklib.load(arch))
            names = [name for name, _, _ in reqs]
            self.assertIn("libc6-dev", names)
            self.assertEqual(len(names), len(set(names)))
        x86 = dict((n, (op, v)) for n, op, v in sdklib.prerequisites(sdklib.load("x86_64")))
        self.assertEqual(x86["libc6-dev"], (">=", "2.39"))
        self.assertIn("binutils-aarch64-linux-gnu", x86)
        self.assertIn("qemu-user-static", x86)
        for arch in ("x86_64", "aarch64"):  # llvm-22's debuginfod client (llvm-symbolizer)
            self.assertIn("libcurl4t64", [n for n, _, _ in sdklib.prerequisites(sdklib.load(arch))])

    def test_installed_means_installed_state_and_native_arch(self):
        out = ("held\tamd64\thi \t1.0\n"        # held but installed: counts
               "removed\tamd64\trc \t1.0\n"     # config files only: missing
               "foreign\tarm64\tii \t1.0\n"     # another architecture: missing
               "arch-all\tall\tii \t2.0\n")
        def fake_run(cmd, **kwargs):
            stdout = "amd64\n" if cmd[:2] == ["dpkg", "--print-architecture"] else out
            return mock.Mock(stdout=stdout, returncode=0)
        with mock.patch.object(sdklib.subprocess, "run", fake_run):
            versions = sdklib.installed_versions(["held", "removed", "foreign", "arch-all"])
        self.assertEqual(versions, {"held": "1.0", "arch-all": "2.0"})


class Trees(unittest.TestCase):
    def test_absolute_links_become_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "usr/lib/aarch64-linux-gnu").mkdir(parents=True)
            (root / "usr/lib/aarch64-linux-gnu/libm.so.6").write_text("")
            link = root / "usr/lib/aarch64-linux-gnu/libm.so"
            link.symlink_to("/usr/lib/aarch64-linux-gnu/libm.so.6")
            self.assertEqual(setup.relativize_absolute_links(root), 1)
            self.assertEqual(os.readlink(link), "libm.so.6")
            self.assertTrue(link.resolve().is_relative_to(root.resolve()))

    def test_links_leaving_the_sysroot_are_refused(self):
        for target in ("/../../etc/passwd", "../../../../etc/passwd"):
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp) / "sysroot"
                (root / "usr/lib").mkdir(parents=True)
                (root / "usr/lib/escape").symlink_to(target)
                with self.assertRaises(sdklib.SdkError):
                    setup.relativize_absolute_links(root)

    def test_unreadable_receipt_is_no_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.assertIsNone(sdklib.read_receipt(root))
            (root / sdklib.RECEIPT).write_text("")  # a crash before the data reached disk
            self.assertIsNone(sdklib.read_receipt(root))
            for content in (b'[]', b'"text"', b'null', b'\xff'):
                (root / sdklib.RECEIPT).write_bytes(content)
                self.assertIsNone(sdklib.read_receipt(root))

    def test_tree_digest_tracks_content_mode_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "a").write_text("x")
            (root / "l").symlink_to("a")
            (root / sdklib.RECEIPT).write_text("ignored")
            base = sdklib.tree_digest(root)
            (root / sdklib.RECEIPT).write_text("changed")
            self.assertEqual(sdklib.tree_digest(root), base)
            (root / "a").chmod(0o755)
            moded = sdklib.tree_digest(root)
            self.assertNotEqual(moded, base)
            (root / "a").write_text("y")
            self.assertNotEqual(sdklib.tree_digest(root), moded)
            (root / "l").unlink()
            (root / "l").symlink_to("b")
            self.assertNotEqual(sdklib.tree_digest(root), moded)


class Fetch(unittest.TestCase):
    PAYLOAD = b"pinned bytes"

    def fetch(self, served: bytes) -> tuple[bytes | None, str]:
        """Fetches a locked artifact from a file:// URL serving `served`."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "a.bin").write_bytes(served)
            entry = {"urls": [(tmp / "a.bin").as_uri()], "size": len(self.PAYLOAD),
                     "sha256": hashlib.sha256(self.PAYLOAD).hexdigest()}
            sdk = sdklib.Sdk(manifest={}, lock={"artifacts": {"archive/a/x86_64": entry}}, arch="x86_64",
                             identity="x86_64-test", inputs={}, home=tmp / "home", cache=tmp / "cache")
            try:
                path = setup.fetch(sdk, "archive/a/x86_64")
                return path.read_bytes(), ""
            except sdklib.SdkError as e:
                leftovers = list((tmp / "cache").rglob(".part-*"))
                self.assertEqual(leftovers, [])
                return None, str(e)

    def test_verified_bytes_are_cached(self):
        self.assertEqual(self.fetch(self.PAYLOAD), (self.PAYLOAD, ""))

    def test_truncated_transfer_is_retried(self):
        data, error = self.fetch(self.PAYLOAD[:-1])
        self.assertIsNone(data)
        self.assertIn("attempt 3", error)

    def test_wrong_bytes_are_not_retried(self):
        for served in (b"x" * len(self.PAYLOAD), self.PAYLOAD + b"!"):
            data, error = self.fetch(served)
            self.assertIsNone(data)
            self.assertIn("attempt 1", error)
            self.assertNotIn("attempt 2", error)


class Wheels(unittest.TestCase):
    """Wheels unpack into one importable directory; anything else a wheel can carry is refused."""

    def setUp(self):
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.lock = {"artifacts": {}}
        self.sdk = sdklib.Sdk(manifest={}, lock=self.lock, arch="x86_64", identity="x86_64-test", inputs={},
                              home=self.tmp / "home", cache=self.tmp / "cache")

    def wheel(self, name: str, members: dict[str, bytes], links: tuple[str, ...] = ()) -> None:
        path = self.tmp / f"{name}-1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as z:
            for member, data in members.items():
                z.writestr(member, data)
            for member in links:
                info = zipfile.ZipInfo(member)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, "elsewhere")
        data = path.read_bytes()
        self.lock["artifacts"][f"wheel/{name}"] = {"urls": [path.as_uri()], "size": len(data),
                                                   "sha256": hashlib.sha256(data).hexdigest()}

    def install(self, *names: str) -> pathlib.Path:
        staging = self.tmp / "staging"
        staging.mkdir(exist_ok=True)
        comp = {"kind": "wheels", "dest": "python/t", "module": "t", "packages": list(names)}
        self.assertEqual(setup.install_wheels(self.sdk, comp, staging), {"artifacts": [f"wheel/{n}" for n in names]})
        return staging / "python/t"

    def test_wheels_unpack_into_one_directory_with_fixed_modes(self):
        self.wheel("t", {"t/__init__.py": b"", "t/__main__.py": b"print(1)", "t-1.0.dist-info/RECORD": b""})
        self.wheel("dep", {"dep.py": b"X = 1"})
        dest = self.install("t", "dep")
        self.assertEqual(sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()),
                         ["dep.py", "t-1.0.dist-info/RECORD", "t/__init__.py", "t/__main__.py"])
        self.assertEqual({p.stat().st_mode & 0o777 for p in dest.rglob("*") if p.is_file()}, {0o644})

    def test_unsafe_members_are_refused(self):
        for members, links, reason in (({"../evil.py": b""}, (), "unsafe path"),
                                       ({"/abs.py": b""}, (), "unsafe path"),
                                       ({"t//x.py": b""}, (), "unsafe path"),
                                       ({"t/./x.py": b""}, (), "unsafe path"),
                                       ({"C:/x.py": b""}, (), "unsafe path"),
                                       ({"t\\x.py": b""}, (), "unsafe path"),
                                       ({"t-1.0.data/scripts/t": b""}, (), ".data tree"),
                                       ({"t/__pycache__/x.cpython-314.pyc": b""}, (), "bytecode"),
                                       ({"t/x.pyc": b""}, (), "bytecode"),
                                       ({"t/x": b"", "t/x/y.py": b""}, (), "collides"),
                                       ({}, ("t/link.py",), "symbolic link")):
            with self.subTest(reason=reason, members=list(members)):
                self.wheel("t", members, links)
                with self.assertRaisesRegex(sdklib.SdkError, reason):
                    self.install("t")

    def test_two_wheels_may_not_provide_one_file(self):
        self.wheel("a", {"shared/x.py": b"a"})
        self.wheel("b", {"shared/x.py": b"b"})
        with self.assertRaisesRegex(sdklib.SdkError, "shared/x.py is also in wheel/a"):
            self.install("a", "b")


class PythonTools(unittest.TestCase):
    """A wheels component's tool runs with only the standard library and its own directory importable."""

    MAIN = ("import sys, json, importlib.util, shadowed\n"
            "print(json.dumps([shadowed.WHERE, sys.argv, importlib.util.find_spec('outside') is not None,\n"
            "                  sys.flags.isolated, sys.flags.no_site, sys.flags.dont_write_bytecode]))")

    def test_only_the_standard_library_and_the_component_are_importable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            manifest = {"components": {"t": {"kind": "wheels", "dest": "python/t", "module": "probe_tool"}}}
            sdk = dataclasses.replace(sdklib.load(), manifest=manifest, home=tmp)
            tool = sdk.root / "python/t"
            (tool / "probe_tool").mkdir(parents=True)
            (tool / "probe_tool/__main__.py").write_text(self.MAIN)
            (tool / "shadowed.py").write_text("WHERE = 'sdk'")
            (tmp / "elsewhere").mkdir()
            (tmp / "elsewhere/shadowed.py").write_text("WHERE = 'PYTHONPATH'")
            (tmp / "elsewhere/outside.py").write_text("")  # on PYTHONPATH and in the working directory only
            env = {**os.environ, "PYTHONPATH": str(tmp / "elsewhere")}
            out = subprocess.run([*sdk.python_tool("t"), "lint", "--x"], capture_output=True, text=True, env=env,
                                 cwd=tmp / "elsewhere", check=True)
            where, argv, outside, isolated, no_site, no_bytecode = json.loads(out.stdout)
            self.assertEqual((where, argv[1:]), ("sdk", ["lint", "--x"]))
            self.assertEqual(argv[0], str(tool / "probe_tool/__main__.py"))  # as `python -m` sets it
            self.assertEqual((outside, isolated, no_site, no_bytecode), (False, 1, 1, True))
            # Bytecode in the SDK would change the tree digest that `doctor --deep` checks.
            self.assertEqual(list(tool.rglob("__pycache__")), [])

    def test_doctor_checks_each_python_tools_version(self):
        manifest = {"versions": {"t": "1.0"}, "hosts": {"x86_64": {"components": ["t"]}},
                    "components": {"t": {"kind": "wheels", "dest": "python/t", "module": "t"}}}
        sdk = dataclasses.replace(sdklib.load("x86_64"), manifest=manifest)
        for stdout, problems in (("t, version 1.0\n", 0), ("t, version 2.0\n", 1), ("", 1)):
            r = check.Report()
            done = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
            with mock.patch.object(check, "capture", return_value=done):
                check.check_python_tools(sdk, r)
            self.assertEqual(len(r.problems), problems, stdout)


class Binfmt(unittest.TestCase):
    def report(self, entries: dict[str, str]) -> "check.Report":
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in entries.items():
                (pathlib.Path(tmp) / name).write_text(text)
            r = check.Report()
            with (mock.patch.object(check, "BINFMT_MISC", pathlib.Path(tmp)),
                  mock.patch.object(sdklib, "check_prerequisites", return_value=[])):
                check.check_prerequisites(sdklib.load("x86_64"), r)
            return r

    def test_unmounted_binfmt_misc_defers_to_the_cross_probe(self):
        self.assertEqual(self.report({}).problems, [])

    def test_missing_or_unfixed_handler_is_a_problem(self):
        self.assertEqual(len(self.report({"status": "enabled"}).problems), 1)
        r = self.report({"status": "enabled", "qemu-aarch64": "enabled\ninterpreter /x\nflags: PO\n"})
        self.assertEqual(len(r.problems), 1)
        r = self.report({"status": "enabled", "qemu-aarch64": "enabled\ninterpreter /x\nflags: POF\n"})
        self.assertEqual(r.problems, [])


class Pruning(unittest.TestCase):
    def test_dry_run_prune_preserves_sdk_and_build_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = dataclasses.replace(sdklib.load(), home=pathlib.Path(tmp) / "home",
                                      cache=pathlib.Path(tmp) / "cache")
            old = sdk.home / "old-sdk"
            work = sdk.cache / "builds" / "gcc.work"
            old.mkdir(parents=True)
            work.mkdir(parents=True)
            (old / sdklib.RECEIPT).write_text('{"identity": "old-sdk"}')
            (work / "build.log").write_text("build evidence")
            before = sorted(p.relative_to(tmp) for p in pathlib.Path(tmp).rglob("*"))
            with (mock.patch.object(sdklib, "load", return_value=sdk),
                  mock.patch.object(sys, "argv", ["setup-toolchain", "--dry-run", "--prune"]),
                  mock.patch.object(setup, "log")):
                self.assertEqual(setup.main(), 0)
            self.assertEqual(sorted(p.relative_to(tmp) for p in pathlib.Path(tmp).rglob("*")), before)
            with mock.patch.object(setup, "log"):
                setup.prune(sdk)
            self.assertFalse(old.exists())
            self.assertFalse(work.exists())

    def test_preview_with_missing_roots_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = dataclasses.replace(sdklib.load(), home=pathlib.Path(tmp) / "home",
                                      cache=pathlib.Path(tmp) / "cache")
            setup.prune(sdk, dry_run=True)
            self.assertEqual(list(pathlib.Path(tmp).iterdir()), [])


class InstalledSdk(unittest.TestCase):
    def test_only_a_receipt_matching_all_inputs_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = dataclasses.replace(sdklib.load(), home=pathlib.Path(tmp))
            self.assertFalse(setup.sdk_is_ready(sdk))
            sdk.root.mkdir()
            valid = {"schema": 1, "identity": sdk.identity, "inputs": sdk.inputs}
            for receipt in ({}, {**valid, "schema": 2}, {**valid, "inputs": {}},
                            {**valid, "identity": "different-sdk"}):
                (sdk.root / sdklib.RECEIPT).write_text(json.dumps(receipt))
                with self.assertRaisesRegex(sdklib.SdkError, "remove it and rerun setup"):
                    setup.sdk_is_ready(sdk)
            (sdk.root / sdklib.RECEIPT).write_text(json.dumps(valid))
            self.assertTrue(setup.sdk_is_ready(sdk))


class Doctor(unittest.TestCase):
    def test_missing_tool_without_a_version_probe_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = dataclasses.replace(sdklib.load(), home=pathlib.Path(tmp))
            paths = [*(f"bin/{name}" for name in sdk.manifest["bin"]), "cuda/bin/nvcc"]
            for path in paths:
                tool = sdk.root / path
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.write_text("#!/bin/sh\nexit 0\n")
                tool.chmod(0o755)
            (sdk.root / "bin/llvm-strip").unlink()
            r = check.Report()
            with mock.patch.object(check, "VERSION_PATTERNS", {}):
                check.check_tools(sdk, r)
            self.assertEqual(len(r.problems), 1)
            self.assertIn("bin/llvm-strip", r.problems[0])

    def test_failed_library_inspection_is_not_an_empty_dependency_list(self):
        out = subprocess.CompletedProcess([], 1, stdout="", stderr="invalid ELF file")
        with mock.patch.object(check, "capture", return_value=out):
            with self.assertRaisesRegex(sdklib.SdkError, "invalid ELF file"):
                check.needed(sdklib.load(), pathlib.Path("probe"))

    def test_missing_executable_is_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(sdklib.SdkError, "could not run"):
                check.capture([pathlib.Path(tmp) / "missing"])


class Signals(unittest.TestCase):
    def setUp(self):
        saved = {s: signal.getsignal(s) for s in setup.STOP_SIGNALS}
        self.addCleanup(lambda: [signal.signal(s, h) for s, h in saved.items()])
        for s in setup.STOP_SIGNALS:
            signal.signal(s, setup._stop)

    def test_first_stop_signal_unwinds_and_the_rest_are_ignored(self):
        with self.assertRaises(SystemExit) as cm:
            setup._stop(signal.SIGTERM, None)
        self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)
        for s in setup.STOP_SIGNALS:
            self.assertIs(signal.getsignal(s), signal.SIG_IGN)

    def test_a_repeated_sigterm_still_kills_the_build_group(self):
        # mise delivers a group signal twice. The copy that lands as cleanup
        # starts must not stop run_group from killing the build's group.
        real_killpg = os.killpg
        def killpg_after_a_second_signal(pgid, sig):
            signal.raise_signal(signal.SIGTERM)
            real_killpg(pgid, sig)
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = pathlib.Path(tmp) / "pid"
            finished = threading.Event()
            timed_out = threading.Event()
            def terminate():
                # Wait until the child has exec'd sleep: a signal between
                # fork and exec could otherwise hit its inherited shell trap.
                deadline = time.monotonic() + 5
                group = None
                while not finished.wait(0.01):
                    try:
                        group, child = map(int, pidfile.read_text().split())
                        if pathlib.Path(f"/proc/{child}/comm").read_text().strip() == "sleep":
                            break
                    except (FileNotFoundError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        timed_out.set()
                        break
                else:
                    return
                os.kill(os.getpid(), signal.SIGTERM)
                # If run_group only stops the leader, its TERM trap waits
                # for the live child. Fail promptly and clean up that group.
                if not finished.wait(5):
                    timed_out.set()
                    if group is not None:
                        try:
                            real_killpg(group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            worker = threading.Thread(target=terminate, daemon=True)
            worker.start()
            try:
                with (mock.patch.object(setup.os, "killpg", killpg_after_a_second_signal),
                      self.assertRaises(SystemExit)):
                    # Reap the child before the shell exits, so the assertion
                    # also works when container PID 1 does not reap orphans.
                    # This trap deliberately does not signal the child itself.
                    setup.run_group(["sh", "-c", "trap 'wait; exit 0' TERM; "
                                     'sleep 30 & echo "$$ $!" > "$1"; wait',
                                     "sdk-signal-test", str(pidfile)])
            finally:
                finished.set()
                worker.join(timeout=1)
            self.assertFalse(timed_out.is_set(), "the build's live child did not stop promptly")
            group = int(pidfile.read_text().split()[0])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.killpg(group, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail("the build's process group outlived the interrupt")


class CrossConfigure(unittest.TestCase):
    def test_defines_land_in_the_linux_cross_branch_only_once(self):
        text = "native branch\n" + setup.LIBSTDCXX_CROSS_ANCHOR + "rest\n"
        with tempfile.TemporaryDirectory() as tmp:
            configure = pathlib.Path(tmp) / "configure"
            configure.write_text(text)
            setup.record_cross_results(configure, ["HAVE_ICONV 1", "ICONV_CONST"])
            out = configure.read_text()
            self.assertIn('$as_echo "#define HAVE_ICONV 1" >>confdefs.h\n'
                          '$as_echo "#define ICONV_CONST" >>confdefs.h\n    ;;\n  *-mingw32*)\n', out)
            with self.assertRaises(sdklib.SdkError):  # the anchor is gone now
                setup.record_cross_results(configure, ["HAVE_ICONV 1"])

    def test_missing_anchor_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure = pathlib.Path(tmp) / "configure"
            configure.write_text("some other GCC version\n")
            with self.assertRaises(sdklib.SdkError):
                setup.record_cross_results(configure, ["HAVE_ICONV 1"])


if __name__ == "__main__":
    unittest.main()
