# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/build, tools/run-target and the presets they rely on (no SDK needed)."""

import contextlib
import importlib.machinery
import io
import importlib.util
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
REPO = TOOLS.parent


def load_script(name: str):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


build = load_script("build")
run_target = load_script("run-target")
PRESETS = json.loads((REPO / "CMakePresets.json").read_text())


def inherited(kind: str, name: str, key: str, default=None):
    """A preset's value for key, following `inherits` as CMake does (single inheritance here)."""
    presets = {p["name"]: p for p in PRESETS[kind]}
    preset = presets[name]
    while key not in preset and "inherits" in preset:
        preset = presets[preset["inherits"]]
    return preset.get(key, default)


def cache_variable(name: str, variable: str):
    """A configure preset's value for one cache variable: CMake merges cacheVariables per variable."""
    presets = {p["name"]: p for p in PRESETS["configurePresets"]}
    preset = presets[name]
    while variable not in preset.get("cacheVariables", {}) and "inherits" in preset:
        preset = presets[preset["inherits"]]
    return preset.get("cacheVariables", {}).get(variable)


def environment(test_preset: str) -> dict:
    return inherited("testPresets", test_preset, "environment", {})


class Presets(unittest.TestCase):
    def test_driver_names_existing_presets(self):
        configure = {p["name"] for p in PRESETS["configurePresets"]}
        tests = {p["name"]: p for p in PRESETS["testPresets"]}
        self.assertEqual(set(build.configure_presets()),
                         {"native", "cpu", "cross", "spark-native", "cpu-asan", "cross-asan", "cross-tsan"})
        self.assertLessEqual(set(build.DEFAULT_PRESET.values()), configure)
        for preset, remote in build.REMOTE_TEST_PRESET.items():
            self.assertIn(preset, configure)
            self.assertEqual(tests[remote]["configurePreset"], preset)

    def test_every_visible_preset_has_a_toolchain_file_and_build_and_test_presets(self):
        builds = {p["name"] for p in PRESETS["buildPresets"]}
        tests = set(build.test_presets())
        for preset in PRESETS["configurePresets"]:
            if preset.get("hidden"):
                continue
            name = preset["name"]
            toolchain = inherited("configurePresets", name, "toolchainFile").replace("${sourceDir}", str(REPO))
            self.assertTrue(pathlib.Path(toolchain).is_file(), toolchain)
            self.assertIn(name, builds)
            if name == "cross-tsan":  # ThreadSanitizer is untested under qemu-user (D-061): Spark only
                self.assertIn(name, build.REMOTE_TEST_PRESET)
                self.assertNotIn(name, tests)
            else:
                self.assertIn(name, tests)

    def test_sanitizer_presets_and_their_run_time_options(self):
        sanitize = lambda name: cache_variable(name, "LLMP_SANITIZE")  # noqa: E731
        self.assertEqual(sanitize("cpu-asan"), "address;undefined")
        self.assertEqual(sanitize("cross-asan"), "address;undefined")
        self.assertEqual(sanitize("cross-tsan"), "thread")
        for name in ("native", "cpu", "cross", "spark-native"):
            self.assertIsNone(sanitize(name))
        for name in ("cpu-asan", "cross-asan", "cross-tsan"):
            self.assertEqual(inherited("configurePresets", name, "binaryDir"), "${sourceDir}/build/${presetName}")
        # Leak detection is on natively and on a Spark, off only under qemu-user (RE-014).
        self.assertEqual(environment("cpu-asan")["ASAN_OPTIONS"], "detect_leaks=1")
        self.assertEqual(environment("cross-asan")["ASAN_OPTIONS"], "detect_leaks=0")
        self.assertEqual(environment("cross-asan-remote")["ASAN_OPTIONS"], "detect_leaks=1")
        self.assertIn("halt_on_error=1", environment("cross-tsan-remote")["TSAN_OPTIONS"])
        for name in ("cpu-asan", "cross-asan", "cross-asan-remote"):
            self.assertEqual(environment(name)["UBSAN_OPTIONS"], "print_stacktrace=1")
        # Only the options run-target forwards reach a Spark.
        for name in ("cross-asan-remote", "cross-tsan-remote"):
            self.assertLessEqual(set(environment(name)), set(run_target.FORWARD_NAMES))

    def test_libstdcxx_assertions_in_the_test_presets_and_not_the_package(self):
        # D-083: the test and development presets, spark-native (D-084's per-slice
        # test build) included, check libstdc++'s preconditions; only the package's build (cross) does not.
        visible = {p["name"] for p in PRESETS["configurePresets"] if not p.get("hidden")}
        on = {n for n in visible if cache_variable(n, "LLMP_LIBSTDCXX_ASSERTIONS") == "ON"}
        self.assertEqual(on, {"native", "cpu", "spark-native", "cpu-asan", "cross-asan", "cross-tsan"})
        self.assertEqual(visible - on, {"cross"})
        for name in visible - on:
            self.assertIsNone(cache_variable(name, "LLMP_LIBSTDCXX_ASSERTIONS"), name)

    def test_build_dir_matches_the_presets_binary_dir(self):
        base = next(p for p in PRESETS["configurePresets"] if p["name"] == "base")
        self.assertEqual(base["binaryDir"], "${sourceDir}/build/${presetName}")
        self.assertEqual(build.build_dir("cross"), REPO / "build" / "cross")

    def test_only_presets_with_a_gb10_run_gpu_tests(self):
        tests = {p["name"]: p for p in PRESETS["testPresets"]}

        def excludes_gpu(name):
            preset = tests[name]
            while True:
                if preset.get("filter", {}).get("exclude", {}).get("label") == "gpu":
                    return True
                if "inherits" not in preset:
                    return False
                preset = tests[preset["inherits"]]

        self.assertEqual({n for n in tests if not tests[n].get("hidden") and not excludes_gpu(n)},
                         {"cross-remote", "cross-asan-remote", "cross-tsan-remote", "spark-native",
                          "native-gpu"})
        # native-gpu runs only the discrete GPU's tests, with the host's
        # driver rather than the stub (D-082).
        gpu = tests["native-gpu"]
        self.assertEqual(gpu["configurePreset"], "native")
        self.assertEqual(gpu["filter"], {"include": {"label": "^gpu-discrete$"}})
        self.assertNotIn("environment", gpu)
        self.assertEqual(gpu["inherits"], "base")


class ConfiguredSdk(unittest.TestCase):
    def test_reads_the_cached_sdk_and_tolerates_a_missing_cache(self):
        tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        original = build.build_dir
        self.addCleanup(setattr, build, "build_dir", original)
        build.build_dir = lambda preset: tmp / preset
        self.assertIsNone(build.configured_sdk("native"))
        (tmp / "native").mkdir()
        (tmp / "native" / "CMakeCache.txt").write_text("X:BOOL=ON\nLLMP_SDK:PATH=/sdk/a\n")
        self.assertEqual(build.configured_sdk("native"), "/sdk/a")


class CtestArgs(unittest.TestCase):
    def test_arguments_after_the_separator_go_to_ctest(self):
        self.assertEqual(build.split_ctest_args(["test", "cross", "--", "-R", "x"]),
                         (["test", "cross"], ["-R", "x"]))
        self.assertEqual(build.split_ctest_args(["test"]), (["test"], []))


class ConfigureArgs(unittest.TestCase):
    SDK = types.SimpleNamespace(root=pathlib.Path("/sdk"))

    def test_locked_selects_the_core_profile_from_locked_sources(self):
        self.assertEqual(build.configure_args(self.SDK, "cpu", False, True),
                         ["--preset", "cpu", "-DLLMP_SDK=/sdk", "-DLLMP_REQUIRE_LOCKED_SOURCES=ON",
                          "-DLLMP_MODULES="])

    def test_unlocked_clears_a_previous_checks_setting_and_keeps_modules(self):
        args = build.configure_args(self.SDK, "cpu", True, False)
        self.assertEqual(args, ["--preset", "cpu", "-DLLMP_SDK=/sdk", "-DLLMP_REQUIRE_LOCKED_SOURCES=OFF",
                                "--fresh"])


class TestDriverEnvironment(unittest.TestCase):
    def run_driver(self, *args, preset="cross"):
        sdk = types.SimpleNamespace(root=pathlib.Path("/sdk"), arch="x86_64")
        ambient = {"LLMP_TARGET_HOST": "stale-host", "LLMP_TARGET_DIR": "/stale/build",
                   "LLMP_TARGET_SSH_CONTROL": "/stale/socket", "GTEST_FILTER": "Example.*"}
        with mock.patch.dict(os.environ, ambient), \
                mock.patch.object(sys, "argv", ["build", "test", preset, *args]), \
                mock.patch.object(build, "ready_sdk", return_value=sdk), \
                mock.patch.object(build, "configured_sdk", return_value=None), \
                mock.patch.object(build, "deploy", return_value="/new/build") as deploy, \
                mock.patch.object(build, "ssh_control_path", return_value=None), \
                mock.patch.object(build, "run") as run:
            self.assertEqual(build.main(), 0)
        return run.call_args_list, deploy

    def test_local_run_ignores_inherited_remote_controls(self):
        calls, deploy = self.run_driver()
        deploy.assert_not_called()
        self.assertEqual(calls[-1].args[0][1:3], ["--preset", "cross"])
        for call in calls:
            env = call.args[1]
            self.assertFalse(any(k.startswith("LLMP_TARGET_") for k in env))
            self.assertEqual(env["GTEST_FILTER"], "Example.*")

    def test_explicit_host_uses_only_the_new_deployment(self):
        calls, deploy = self.run_driver("--host", "new-host")
        self.assertEqual(deploy.call_args.args[2:], ("new-host", None))
        self.assertEqual(calls[-1].args[0][1:3], ["--preset", "cross-remote"])
        env = calls[-1].args[1]
        self.assertEqual(env["LLMP_TARGET_HOST"], "new-host")
        self.assertEqual(env["LLMP_TARGET_DIR"], "/new/build")
        self.assertNotIn("LLMP_TARGET_SSH_CONTROL", env)
        for call in calls[:-1]:
            self.assertNotIn("LLMP_TARGET_HOST", call.args[1])

    def test_sanitizer_cross_builds_use_their_remote_presets(self):
        for preset in ("cross-asan", "cross-tsan"):
            calls, _ = self.run_driver("--host", "new-host", "--locked", preset=preset)
            self.assertIn("-DLLMP_REQUIRE_LOCKED_SOURCES=ON", calls[0].args[0])
            self.assertEqual(calls[-1].args[0][1:3], ["--preset", f"{preset}-remote"])

    def test_gpu_runs_the_native_builds_discrete_gpu_tests_one_at_a_time(self):
        calls, deploy = self.run_driver("--gpu", preset="native")
        deploy.assert_not_called()
        self.assertEqual(calls[-1].args[0][1:5], ["--preset", "native-gpu", "--parallel", 1])

    def test_gpu_needs_a_preset_with_discrete_gpu_tests_and_no_host(self):
        for argv, message in ((["cross", "--gpu"], "cross has no discrete GPU tests"),
                              (["native", "--gpu", "--host", "h"], "not allowed with argument")):
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", ["build", "test", *argv]), \
                    mock.patch.object(build, "ready_sdk", return_value=types.SimpleNamespace(arch="x86_64")), \
                    mock.patch.object(build, "run") as run, contextlib.redirect_stderr(stderr), \
                    self.assertRaises(SystemExit):
                build.main()
            run.assert_not_called()
            self.assertIn(message, stderr.getvalue())

    def test_thread_sanitizer_tests_need_a_spark(self):
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["build", "test", "cross-tsan"]), \
                mock.patch.object(build, "ready_sdk", return_value=types.SimpleNamespace(arch="x86_64")), \
                mock.patch.object(build, "run") as run, contextlib.redirect_stderr(stderr), \
                self.assertRaises(SystemExit):
            build.main()
        run.assert_not_called()
        self.assertIn("cross-tsan's tests run only on a Spark", stderr.getvalue())


class PathMapping(unittest.TestCase):
    B, R = "/work/build/cross", "/home/u/.cache/llmp/deploy/cross-1"

    def test_prefixes_map_and_lookalikes_do_not(self):
        m = lambda v: run_target.map_path(v, self.B, self.R)  # noqa: E731
        self.assertEqual(m(self.B), self.R)
        self.assertEqual(m(self.B + "/tests/t"), self.R + "/tests/t")
        self.assertEqual(m("--gtest_output=xml:" + self.B + "/out.xml"), "--gtest_output=xml:" + self.R + "/out.xml")
        self.assertEqual(m("/work/build/cross2/t"), "/work/build/cross2/t")
        self.assertEqual(m("/other/work/build/cross/t"), "/other/work/build/cross/t")
        self.assertEqual(m("-x"), "-x")

    def test_remote_command_maps_cwd_and_forwards_test_variables_only(self):
        env = {"GTEST_FILTER": "A.*", "ASAN_OPTIONS": "a=1", "HOME": "/home/me", "LLMP_TEST_X": "it's"}
        cmd = run_target.remote_command(self.B, self.R, self.B + "/tests", [self.B + "/tests/t", "--flag"], env)
        self.assertEqual(shlex.split(cmd), [
            "cd", self.R + "/tests", "&&", "exec", "env", "ASAN_OPTIONS=a=1", "GTEST_FILTER=A.*",
            "LLMP_TEST_X=it's", self.R + "/tests/t", "--flag"])

    def test_a_cwd_outside_the_build_dir_starts_at_the_remote_root(self):
        cmd = run_target.remote_command(self.B, self.R, "/tmp", [self.B + "/t"], {})
        self.assertTrue(cmd.startswith(f"cd {self.R} && "))

    def test_paths_in_forwarded_environment_values_map_to_the_deployment(self):
        env = {"GTEST_OUTPUT": f"xml:{self.B}/results.xml",
               "ASAN_OPTIONS": f"suppressions={self.B}/suppressions:detect_leaks=1",
               "LLMP_TEST_DATA": f"{self.B}/data with spaces"}
        command = shlex.split(run_target.remote_command(self.B, self.R, self.B, [self.B + "/t"], env))
        self.assertIn(f"GTEST_OUTPUT=xml:{self.R}/results.xml", command)
        self.assertIn(f"ASAN_OPTIONS=suppressions={self.R}/suppressions:detect_leaks=1", command)
        self.assertIn(f"LLMP_TEST_DATA={self.R}/data with spaces", command)

    def test_ssh_shares_a_connection_only_when_asked(self):
        self.assertEqual(run_target.ssh_command("h", "c", None), ["ssh", "-o", "BatchMode=yes", "--", "h", "c"])
        self.assertIn("ControlPath=/run/x-%C", run_target.ssh_command("h", "c", "/run/x-%C"))


class RunTargetProcess(unittest.TestCase):
    """Runs tools/run-target with stand-ins for qemu and ssh on PATH."""

    def setUp(self):
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.log = self.tmp / "log.json"
        for name, status in (("fake-qemu", 0), ("ssh", 7)):
            tool = self.bin / name
            tool.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                            f"json.dump({{'argv': sys.argv[1:], 'ld': os.environ.get('QEMU_LD_PREFIX')}}, "
                            f"open({str(self.log)!r}, 'w'))\nsys.exit({status})\n")
            tool.chmod(0o755)
        self.build_dir = self.tmp / "build"
        (self.build_dir / "tests").mkdir(parents=True)

    def run_target(self, env: dict, *argv: str) -> subprocess.CompletedProcess:
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", **env}
        return subprocess.run([sys.executable, TOOLS / "run-target", "--build-dir", self.build_dir, "--sysroot",
                               "/sys/root", "--qemu", "fake-qemu", "--", *argv],
                              env=env, cwd=self.build_dir / "tests", capture_output=True, text=True)

    def test_local_runs_under_qemu_with_the_sysroot_loader(self):
        result = self.run_target({}, str(self.build_dir / "tests" / "t"), "-a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.log.read_text()),
                         {"argv": [str(self.build_dir / "tests" / "t"), "-a"], "ld": "/sys/root"})

    def test_remote_needs_the_deployed_directory(self):
        result = self.run_target({"LLMP_TARGET_HOST": "spark"}, "t")
        self.assertEqual(result.returncode, 2)
        self.assertIn("LLMP_TARGET_DIR", result.stderr)
        self.assertFalse(self.log.exists())

    def test_remote_runs_over_ssh_and_returns_its_status(self):
        result = self.run_target({"LLMP_TARGET_HOST": "spark", "LLMP_TARGET_DIR": "/r/x"},
                                 str(self.build_dir / "tests" / "t"))
        self.assertEqual(result.returncode, 7)
        argv = json.loads(self.log.read_text())["argv"]
        self.assertEqual(argv[:4], ["-o", "BatchMode=yes", "--", "spark"])
        self.assertEqual(argv[4], "cd /r/x/tests && exec env /r/x/tests/t")


if __name__ == "__main__":
    unittest.main()
