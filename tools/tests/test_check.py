# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/check, the local check gate's driver (no SDK, builds or docker needed)."""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import tomllib
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import llmp_sources as srclib  # noqa: E402


def load_script(name: str):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


check = load_script("check")


class Tiers(unittest.TestCase):
    def names(self, tier, host=None):
        return [step.name for step in check.plan(tier, None, host)]

    def test_check_formats_builds_every_workstation_profile_and_lints(self):
        self.assertEqual(self.names("check"),
                         ["format", "reuse", "headers", "boundaries", "tools", "native", "cpu", "cross",
                          "tidy"])

    def test_full_adds_the_sanitizers_the_reference_build_the_package_and_jobs(self):
        self.assertEqual(self.names("full"),
                         [*self.names("check"), "cpu-asan", "cross-asan", "reference", "package", "jobs"])

    def test_spark_runs_the_cross_builds_on_the_named_host_only(self):
        self.assertEqual(self.names("spark", "s1"), ["cross@s1", "cross-asan@s1", "cross-tsan@s1"])
        with mock.patch.object(check, "run", return_value=0) as run:
            for step in check.plan("spark", None, "s1"):
                self.assertTrue(step.run())
        for call, preset in zip(run.call_args_list, ("cross", "cross-asan", "cross-tsan"), strict=True):
            self.assertEqual([str(a) for a in call.args[0][1:]],
                             [str(check.BUILD), "test", "--locked", "--fresh", preset, "--host", "s1"])

    def test_workstation_builds_are_locked_fresh_and_local(self):
        with mock.patch.object(check, "run", return_value=1) as run:
            self.assertFalse(check.build_and_test("cpu-asan"))
        self.assertEqual([str(a) for a in run.call_args.args[0][1:]],
                         [str(check.BUILD), "test", "--locked", "--fresh", "cpu-asan"])

    def test_mise_tasks(self):
        tasks = tomllib.loads((REPO / "mise.toml").read_text())["tasks"]
        self.assertEqual(tasks["check"]["run"], "python3 tools/check")
        self.assertEqual(tasks["check:full"]["run"], "python3 tools/check full")
        self.assertEqual(tasks["check:spark"]["run"], "python3 tools/check spark")


class Licensing(unittest.TestCase):
    def test_reuse_lints_the_checkout_with_the_sdks_isolated_tool(self):
        sdk = mock.Mock()
        sdk.python_tool.return_value = ["python3", "-I", "-S", "-c", "...", "/sdk/python/reuse", "reuse"]
        with mock.patch.object(check, "run", return_value=1) as run:
            self.assertFalse(check.check_reuse(sdk))
        sdk.python_tool.assert_called_once_with("reuse")
        self.assertEqual([str(a) for a in run.call_args.args[0]],
                         [*sdk.python_tool.return_value, "--root", str(check.REPO), "lint"])

    def test_headers_checks_the_working_tree_against_reuses_report(self):
        sdk = mock.Mock()
        sdk.python_tool.return_value = ["python3", "reuse"]
        report = {"files": [{"path": "a.py"}]}
        done = types.SimpleNamespace(returncode=1, stdout=json.dumps(report), stderr="")
        with (mock.patch.object(check, "worktree_files", return_value=["a.py"]),
              mock.patch.object(check.subprocess, "run", return_value=done) as run,
              mock.patch.object(check.headers, "check", return_value=(1, ["a.py: no tag"])) as headers,
              mock.patch.object(check, "log") as log):
            self.assertFalse(check.check_headers(sdk))
        self.assertEqual(run.call_args.args[0], ["python3", "reuse", "--root", str(check.REPO), "lint", "--json"])
        headers.assert_called_once_with(check.REPO, ["a.py"], report)
        self.assertIn(mock.call("a.py: no tag"), log.call_args_list)

    def test_boundaries_checks_the_working_tree(self):
        with (mock.patch.object(check, "worktree_files", return_value=["src/a.cc"]),
              mock.patch.object(check.boundaries, "check", return_value=(1, ["src/a.cc:1: uses epoll_wait"])) as run,
              mock.patch.object(check, "log") as log):
            self.assertFalse(check.check_boundaries())
        run.assert_called_once_with(check.REPO, ["src/a.cc"])
        self.assertIn(mock.call("src/a.cc:1: uses epoll_wait"), log.call_args_list)
        with (mock.patch.object(check, "worktree_files", return_value=[]),
              mock.patch.object(check.boundaries, "check", return_value=(0, [])),
              mock.patch.object(check, "log")):
            self.assertTrue(check.check_boundaries())

    def test_headers_fails_without_a_report(self):
        sdk = mock.Mock()
        sdk.python_tool.return_value = ["python3", "reuse"]
        done = types.SimpleNamespace(returncode=2, stdout="", stderr="crashed")
        with (mock.patch.object(check.subprocess, "run", return_value=done),
              mock.patch.object(check.headers, "check") as headers, mock.patch.object(check, "log")):
            self.assertFalse(check.check_headers(sdk))
        headers.assert_not_called()


class Arguments(unittest.TestCase):
    def main(self, *argv):
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["check", *argv]), contextlib.redirect_stderr(stderr), \
                mock.patch.object(check, "ready_sdk", side_effect=AssertionError("must not run")), \
                self.assertRaises(SystemExit) as raised:
            check.main()
        return raised.exception.code, stderr.getvalue()

    def test_spark_needs_a_host_and_only_spark_takes_one(self):
        self.assertIn("`spark` needs --host", self.main("spark")[1])
        self.assertIn("--host is for the `spark` tier", self.main("full", "--host", "s1")[1])
        self.assertIn("is not a host name", self.main("spark", "--host=-oProxyCommand=x")[1])

    def test_list_runs_nothing(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["check", "full", "--list"]), contextlib.redirect_stdout(stdout), \
                mock.patch.object(check, "ready_sdk", side_effect=AssertionError("must not run")):
            self.assertEqual(check.main(), 0)
        self.assertIn("reference", stdout.getvalue())

    def test_every_step_runs_and_any_failure_fails_the_tier(self):
        ran = []
        steps = [check.Step(name, "", lambda n=name, ok=ok: ran.append(n) or ok)
                 for name, ok in (("a", True), ("b", False), ("c", True))]
        sdk = types.SimpleNamespace(arch="x86_64", identity="x86_64-0")
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["check"]), contextlib.redirect_stdout(stdout), \
                mock.patch.object(check, "ready_sdk", return_value=sdk), \
                mock.patch.object(check, "plan", return_value=steps), \
                mock.patch.object(check, "git", return_value="abc\n"), \
                mock.patch.object(check, "isolate_environment", return_value=["CXXFLAGS"]) as isolate, \
                mock.patch.object(check.signal, "getsignal", return_value=check.signal.SIG_DFL), \
                mock.patch.object(check.signal, "signal") as handler:
            self.assertEqual(check.main(), 1)
        self.assertEqual(ran, ["a", "b", "c"])
        self.assertIn("2 of 3 steps passed; failed: b", stdout.getvalue())
        self.assertIn("commit abc with uncommitted changes, SDK x86_64-0", stdout.getvalue())
        isolate.assert_called_once_with()
        self.assertIn("the steps ignore CXXFLAGS", stdout.getvalue())
        self.assertEqual({c.args[0] for c in handler.call_args_list}, {check.signal.SIGTERM, check.signal.SIGHUP})
        with self.assertRaises(KeyboardInterrupt):
            handler.call_args.args[1](check.signal.SIGTERM, None)

    def test_a_signal_the_caller_ignores_stays_ignored(self):
        sdk = types.SimpleNamespace(arch="x86_64", identity="x86_64-0")
        with mock.patch.object(sys, "argv", ["check"]), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(check, "ready_sdk", return_value=sdk), \
                mock.patch.object(check, "plan", return_value=[]), \
                mock.patch.object(check, "git", return_value="abc\n"), \
                mock.patch.object(check, "isolate_environment", return_value=[]), \
                mock.patch.object(check.signal, "getsignal", return_value=check.signal.SIG_IGN), \
                mock.patch.object(check.signal, "signal") as handler:
            self.assertEqual(check.main(), 0)
        handler.assert_not_called()


class Environment(unittest.TestCase):
    def test_the_callers_build_test_and_sanitizer_settings_never_reach_a_step(self):
        environ = {name: "x" for name in (
            "CXXFLAGS", "LDFLAGS", "CUDAFLAGS", "CPLUS_INCLUDE_PATH", "CCC_OVERRIDE_OPTIONS", "NVCC_APPEND_FLAGS",
            "LD_PRELOAD", "ASAN_OPTIONS", "LSAN_OPTIONS", "TSAN_OPTIONS", "UBSAN_OPTIONS",
            "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER_LAUNCHER", "CTEST_PARALLEL_LEVEL", "GTEST_FILTER",
            "LLMP_TEST_X", "QEMU_SET_ENV")}
        kept = {name: "y" for name in (
            "PATH", "HOME", "SSH_AUTH_SOCK", "XDG_RUNTIME_DIR", "LLMP_SDK_HOME", "LLMP_CACHE_HOME",
            "DOCKER_HOST", "LANG")}
        environ |= kept
        removed = check.isolate_environment(environ)
        self.assertEqual(environ, kept)
        self.assertEqual(removed, sorted(set(removed)))
        self.assertIn("GTEST_FILTER", removed)
        self.assertIn("CXXFLAGS", removed)


class Sources(unittest.TestCase):
    def test_ours_excludes_experiment_evidence_and_third_party_code(self):
        files = ["src/a.cc", "src/a.h", "k/b.cu", "k/b.cuh", "docs/experiments/x/main.cc", "third_party/g/g.cc",
                 "tools/build", "src/c.py", "src/d.cc.orig"]
        self.assertEqual(check.own_sources(files), ["src/a.cc", "src/a.h", "k/b.cu", "k/b.cuh"])
        self.assertEqual(check.own_sources(files, check.TIDY_SUFFIXES), ["src/a.cc"])

    def test_a_c_or_cpp_file_is_checked_not_skipped(self):
        files = ["src/e.cpp", "src/e.hpp", "src/f.cxx", "src/g.c", "src/h.hh", "src/i.inl"]
        self.assertEqual(check.own_sources(files), files)
        self.assertEqual(check.own_sources(files, check.TIDY_SUFFIXES), ["src/e.cpp", "src/f.cxx", "src/g.c"])

    def test_worktree_files_include_new_files_and_skip_deleted_ones(self):
        listing = "tracked.cc\0new.cc\0deleted.cc\0"
        with mock.patch.object(check, "git", return_value=listing) as git, \
                mock.patch.object(check.os.path, "lexists", side_effect=lambda p: not str(p).endswith("deleted.cc")):
            self.assertEqual(check.worktree_files(), ["new.cc", "tracked.cc"])
        self.assertEqual(git.call_args.args, ("ls-files", "-z", "--cached", "--others", "--exclude-standard"))


class TidyUnits(unittest.TestCase):
    def setUp(self):
        self.repo = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(mock.patch.object(check, "REPO", self.repo))

    def database(self, preset, *files, directory=None):
        build = self.repo / "build" / preset
        build.mkdir(parents=True)
        entries = [{"directory": str(directory or build), "file": f, "command": "clang++ -c " + f} for f in files]
        (build / "compile_commands.json").write_text(json.dumps(entries))

    def test_each_unit_once_per_architecture_and_only_ours(self):
        self.database("native", str(self.repo / "a.cc"), str(self.repo / "k.cu"), "/elsewhere/g.cc",
                      str(self.repo / "build/sources/gtest/gtest-all.cc"))
        self.database("cross", "../../a.cc", str(self.repo / "b.cc"))
        self.database("cpu", str(self.repo / "a.cc"), str(self.repo / "c.cc"))
        units, problems = check.tidy_units({"a.cc", "b.cc", "c.cc"})
        self.assertEqual(problems, [])
        self.assertEqual(units, [("native", "a.cc"), ("cross", "a.cc"), ("cross", "b.cc"), ("cpu", "c.cc")])

    def test_a_missing_database_or_an_unbuilt_source_is_a_problem(self):
        self.database("native", str(self.repo / "a.cc"))
        self.database("cpu", str(self.repo / "a.cc"))
        units, problems = check.tidy_units({"a.cc", "orphan.cc"})
        self.assertEqual(units, [("native", "a.cc")])
        self.assertEqual(len(problems), 2)
        self.assertIn("build/cross/compile_commands.json", problems[0])
        self.assertIn("orphan.cc is in no checked compile database", problems[1])


class PreparedClosure(unittest.TestCase):
    """The empty-cache preparation holds the core closure of the checked-in lock and nothing else."""

    def setUp(self):
        tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache, self.sources = tmp / "cache", tmp / "sources"
        lock = srclib.load_lock()
        self.core = [(lock["components"][c], c) for c in srclib.select(lock, [])]
        self.assertTrue(self.core)

    def prepare(self):
        for comp, cid in self.core:
            (self.cache / "downloads" / comp["archive"]["sha256"]).mkdir(parents=True)
            srclib.prepared_dir(self.sources, cid, comp).mkdir(parents=True)
        (self.sources / ".prepare.lock").touch()

    def problems(self):
        return check.prepared_problems(srclib.LOCK, self.cache, self.sources)

    def test_exactly_the_core_closure_passes(self):
        self.prepare()
        self.assertEqual(self.problems(), [])

    def test_anything_else_fetched_or_unpacked_fails(self):
        self.prepare()
        (self.cache / "downloads" / ("f" * 64)).mkdir()
        (self.sources / "optional-0123456789abcdef").mkdir()
        problems = self.problems()
        self.assertEqual(len(problems), 2)
        self.assertIn("f" * 64, problems[0])
        self.assertIn("optional-0123456789abcdef", problems[1])

    def test_a_missing_component_fails(self):
        self.assertEqual(len(self.problems()), 2 * len(self.core))


class ReferenceContainer(unittest.TestCase):
    def test_every_container_is_amd64_with_a_read_only_sdk_by_default(self):
        ref = check.Reference()
        cmd = ref.docker(["true"])
        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertIn("--platform", cmd)
        self.assertEqual(cmd[cmd.index("--platform") + 1], "linux/amd64")
        self.assertIn(check.SDK_VOLUME + ":ro", cmd)
        self.assertNotIn("--network", cmd)
        self.assertNotIn(check.SDK_CACHE_VOLUME, cmd)
        self.assertEqual(cmd[-2:], [check.IMAGE, "true"])
        self.assertIn(f"{ref.volume}:/workspaces", cmd)

    def test_setup_may_write_the_sdk_and_the_build_has_no_network(self):
        ref = check.Reference()
        setup = ref.docker(["setup"], sdk_writable=True, sdk_cache=True)
        self.assertIn(check.SDK_VOLUME, setup)
        self.assertIn(check.SDK_CACHE_VOLUME, setup)
        offline = ref.docker(["build"], network=False)
        self.assertEqual(offline[offline.index("--network") + 1], "none")
        self.assertEqual(len(set(ref.containers)), 2)

    def test_the_offline_build_first_proves_it_has_no_network(self):
        self.assertIn("/sys/class/net", check.NO_NETWORK)
        self.assertIn("exit 1", check.NO_NETWORK)

    def test_container_steps_stop_at_the_first_failure_and_always_clean_up(self):
        ref = check.Reference()
        commands = []
        with mock.patch.object(check.shutil, "which", return_value="/usr/bin/docker"), \
                mock.patch.object(check, "run", side_effect=lambda cmd, **kw: commands.append(cmd) or 0), \
                mock.patch.object(ref, "copy_worktree", return_value=False), \
                mock.patch.object(check.subprocess, "run", return_value=mock.Mock(returncode=1)) as cleanup, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertFalse(ref())
        self.assertEqual([c[1] for c in commands], ["build", "volume"])
        self.assertEqual(cleanup.call_args.args[0], ["docker", "volume", "rm", "-f", ref.volume])
        self.assertIn(f"could not remove the docker volume {ref.volume}", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
