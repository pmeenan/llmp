# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Local orchestration checks; no SSH connections or containers are started."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("mimo_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunTests(unittest.TestCase):
    def test_failed_gpu_query_is_not_reported_as_an_idle_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            query = Path(temporary) / "nvidia-smi"
            query.write_text("#!/bin/sh\nexit 1\n")
            query.chmod(0o700)
            env = dict(os.environ, PATH=temporary + os.pathsep + os.environ["PATH"])

            def local_script(_host, command, **kwargs):
                return subprocess.run(runner.shlex.split(command), env=env,
                                      check=True, capture_output=True, text=True, **kwargs)

            with patch.object(runner, "ssh", side_effect=local_script):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.node_state("unused")

    def run_case(self, fail_guard=False, disconnect=False, cleanup_failure=None,
                 copy_failure=False, busy=False, workload_returncode=0):
        config = json.loads((runner.HERE / "config.json").read_text())
        commands = []
        copies = []
        guard_failed = False
        cleanup_started = False
        clock = 0.0

        def monotonic():
            nonlocal clock
            clock += 1
            return clock

        def sleep(seconds):
            nonlocal clock
            clock += seconds

        def fail(host, phase, timeout):
            self.assertIsNotNone(timeout, f"{phase} must have a time limit")
            raise subprocess.TimeoutExpired(["ssh", host, phase], timeout)

        def ssh(host, command, check=True, timeout=None):
            nonlocal guard_failed, cleanup_started
            commands.append((host, command))
            if fail_guard and host == config["worker"] and "nohup python3 guard.py" in command:
                guard_failed = True
                raise subprocess.CalledProcessError(255, ["ssh", host])
            if "docker stop" in command:
                cleanup_started = True
                self.assertIsNotNone(timeout)
                if host == config["head"] and cleanup_failure == "container":
                    fail(host, "container", timeout)
            if "kill -TERM" in command:
                self.assertIsNotNone(timeout)
                if host == config["head"] and cleanup_failure == "guard":
                    fail(host, "guard", timeout)
            if disconnect and guard_failed and host == config["worker"]:
                fail(host, "disconnected", timeout)
            if "python3 workload.py" in command:
                return subprocess.CompletedProcess([], workload_returncode, stdout="", stderr="")
            stdout = "running\n" if "docker inspect" in command else ""
            return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

        def node_state(host, timeout=10):
            if disconnect and guard_failed and host == config["worker"]:
                fail(host, "release observation", timeout)
            if cleanup_started and host == config["head"] and cleanup_failure == "release observation":
                raise ValueError("invalid state response")
            apps = ["123"] if busy and cleanup_started and host == config["worker"] else []
            return {"mem_available": 200 << 30, "compute_apps": apps}

        def subprocess_run(command, **kwargs):
            if command[0] == "rsync":
                copies.append(command[-2].split(":", 1)[0])
                self.assertIsNotNone(kwargs.get("timeout"))
                if copy_failure and copies[-1] == config["head"]:
                    raise subprocess.CalledProcessError(23, command)
                if disconnect and copies[-1] == config["worker"]:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run.py", "--name", "test", "--results", temporary, "--no-workload"]
            if workload_returncode:
                argv.remove("--no-workload")
            with (
                patch.object(runner.sys, "argv", argv),
                patch.object(runner, "ssh", side_effect=ssh),
                patch.object(runner, "node_state", side_effect=node_state),
                patch.object(runner.subprocess, "run", side_effect=subprocess_run),
                patch.object(runner.time, "sleep", side_effect=sleep),
                patch.object(runner.time, "monotonic", side_effect=monotonic),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = runner.main()
            record = json.loads((Path(temporary) / "test" / "run.json").read_text())
        self.assertEqual(copies, [config["head"], config["worker"]])
        for host in (config["head"], config["worker"]):
            self.assertTrue(any(node == host and "kill -TERM" in command
                                for node, command in commands))
        return status, record, commands, config

    def test_second_guard_failure_is_unsuccessful_after_teardown(self):
        status, record, _, _ = self.run_case(fail_guard=True)
        self.assertEqual(status, 1)
        self.assertIn("CalledProcessError", record["error"])
        self.assertNotIn("containers started", record["events"])

    def test_successful_boot_returns_success_after_teardown(self):
        status, record, _, _ = self.run_case(fail_guard=False)
        self.assertEqual(status, 0)
        self.assertNotIn("error", record)
        self.assertIn("healthy", record["events"])
        self.assertIn("guard shutdown attempted", record["events"])
        self.assertNotIn("cleanup_errors", record)
        self.assertEqual(record["not_released"], [])

    def test_disconnected_worker_does_not_skip_healthy_guard_or_local_result(self):
        status, record, _, config = self.run_case(fail_guard=True, disconnect=True)
        self.assertEqual(status, 1)
        self.assertIn("CalledProcessError", record["error"])
        self.assertIn(config["head"], record["released"])
        self.assertEqual(record["not_released"], [config["worker"]])
        self.assertEqual({error["phase"] for error in record["cleanup_errors"]},
                         {"release observation", "guard", "results"})

    def test_each_cleanup_failure_is_recorded_without_skipping_other_nodes(self):
        for phase in ("container", "release observation", "guard"):
            with self.subTest(phase=phase):
                status, record, _, config = self.run_case(cleanup_failure=phase)
                self.assertEqual(status, 1)
                self.assertEqual(record["cleanup_errors"][0]["phase"], phase)
                self.assertIn(config["worker"], record["released"])

    def test_failed_result_copy_is_saved_and_does_not_skip_the_other_node(self):
        status, record, _, _ = self.run_case(copy_failure=True)
        self.assertEqual(status, 1)
        self.assertEqual(record["cleanup_errors"][0]["phase"], "results")

    def test_release_deadline_reports_unknown_and_still_stops_guards(self):
        status, record, _, config = self.run_case(busy=True)
        self.assertEqual(status, 1)
        self.assertEqual(record["not_released"], [config["worker"]])
        self.assertIn(config["head"], record["released"])

    def test_workload_failure_is_unsuccessful_after_teardown(self):
        status, record, _, _ = self.run_case(workload_returncode=1)
        self.assertEqual(status, 1)
        self.assertEqual(record["workload_returncode"], 1)
        self.assertIn("CalledProcessError", record["error"])

    def test_summary_keeps_cleanup_failures_when_remote_results_are_missing(self):
        failures = [{"host": "unreachable", "phase": "guard", "error": "timeout"}]
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "run.json").write_text(json.dumps({
                "events": {}, "server_args_head": [], "cleanup_errors": failures,
                "error": "teardown incomplete", "not_released": ["unreachable"],
            }))
            result = subprocess.run([
                runner.sys.executable, str(runner.HERE / "summarize.py"), temporary,
            ], check=True, capture_output=True, text=True, timeout=10)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["cleanup_errors"], failures)
        self.assertEqual(summary["not_released"], ["unreachable"])
        self.assertEqual(summary["nodes"], {})


if __name__ == "__main__":
    unittest.main()
