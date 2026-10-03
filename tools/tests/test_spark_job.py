# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of tools/spark-job: real supervisors and process groups, with short sleeps."""

import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
TOOL = pathlib.Path(__file__).resolve().parent.parent / "spark-job"


def pid_alive(pid: int) -> bool:
    try:
        state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state not in ("Z", "X")


@unittest.skipUnless(sys.platform.startswith("linux"), "spark-job reads /proc")
class SparkJobTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.env = dict(os.environ, JITLLM_JOBS_DIR=str(self.tmp / "jobs"))
        self.started = []

    def tearDown(self):
        for name in self.started:  # nothing a test started outlives it
            self.run_tool("kill", name)
        self._tmp.cleanup()

    def run_tool(self, *args, timeout=60) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-B", str(TOOL), *args], env=self.env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=timeout)

    def start(self, name, *args) -> None:
        result = self.run_tool("start", "--name", name, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.started.append(name)

    def state(self, name) -> dict:
        result = self.run_tool("status", name, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def wait(self, name, timeout=30) -> tuple[subprocess.CompletedProcess, float]:
        began = time.monotonic()
        result = self.run_tool("wait", name, "--timeout", str(timeout), timeout=timeout + 30)
        return result, time.monotonic() - began

    def test_normal_exit(self):
        self.start("ok", "--", "bash", "-c", "echo hello; echo oops >&2")
        result, _ = self.wait("ok")
        self.assertEqual(result.returncode, 0, result.stdout)
        info = self.state("ok")
        self.assertEqual((info["state"], info["rc"]), ("done", 0))
        log = pathlib.Path(info["log"]).read_text()
        self.assertIn("hello", log)
        self.assertIn("oops", log)
        self.assertIn("hello", self.run_tool("tail", "ok").stdout)
        self.assertIn("done (rc 0)", self.run_tool("status").stdout)

    def test_steps_run_with_umask_at_least_022(self):
        # A caller's group-writable umask (ssh sessions on the Sparks use 0002) must not reach the
        # steps: jitllm-runtime refuses a process lock under a group- or world-writable directory.
        previous = os.umask(0o002)
        try:
            self.start("mask", "--", "bash", "-c", "umask; mkdir made")
        finally:
            os.umask(previous)
        result, _ = self.wait("mask")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("0022", pathlib.Path(self.state("mask")["log"]).read_text())
        self.assertEqual((self.tmp / "made").stat().st_mode & 0o777, 0o755)

    def test_nonzero_exit(self):
        self.start("bad", "--", "bash", "-c", "echo Traceback: HTTP 400; exit 3")
        result, _ = self.wait("bad")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed (rc 3)", result.stdout)
        self.assertIn("HTTP 400", result.stdout)  # the log's tail comes with a failure
        self.assertEqual(self.state("bad")["rc"], 3)

    def test_crash_by_signal(self):
        self.start("segv", "--", "bash", "-c", "ulimit -c 0; kill -SEGV $$")
        result, _ = self.wait("segv")
        self.assertNotEqual(result.returncode, 0)
        info = self.state("segv")
        self.assertEqual((info["state"], info["signal"]), ("failed", "SIGSEGV"))

    def test_timeout_kills_the_group_with_its_children(self):
        pids = self.tmp / "pids"
        # The children ignore SIGTERM, so the group needs the SIGKILL after the grace period.
        self.start("slow", "--timeout", "1", "--grace", "1", "--", "bash", "-c",
                   f"trap '' TERM; sleep 60 & echo $! >> {pids}; sleep 60 & echo $! >> {pids}; wait")
        result, took = self.wait("slow")
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(took, 15)
        self.assertEqual(self.state("slow")["state"], "timed-out")
        children = [int(p) for p in pids.read_text().split()]
        self.assertEqual(len(children), 2)
        time.sleep(0.5)  # init reaps the killed orphans
        self.assertFalse([p for p in children if pid_alive(p)])

    def test_supervisor_killed_is_lost(self):
        self.start("orphan", "--", "sleep", "60")
        info = self.state("orphan")
        self.assertEqual(info["state"], "running")
        child = pathlib.Path(info["dir"]) / "child.json"
        for _ in range(100):  # until the supervisor has started the sleep
            if child.exists():
                break
            time.sleep(0.05)
        os.kill(info["supervisor"]["pid"], signal.SIGKILL)
        result, took = self.wait("orphan")
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(took, 10)
        info = self.state("orphan")
        self.assertEqual(info["state"], "lost")
        self.assertTrue(info["orphans"])  # the sleep outlived its supervisor
        self.assertEqual(self.run_tool("kill", "orphan").returncode, 0)
        self.assertFalse(self.state("orphan")["orphans"])

    def test_queue_continues_past_a_failed_step(self):
        steps = self.tmp / "steps"
        steps.write_text("# a comment\necho one\n\nexit 4\necho three\n")
        self.start("queue", "--steps", str(steps))
        result, _ = self.wait("queue")
        self.assertNotEqual(result.returncode, 0)
        info = self.state("queue")
        self.assertEqual(info["state"], "failed")
        self.assertEqual([(s["state"], s["rc"]) for s in info["steps"]],
                         [("done", 0), ("failed", 4), ("done", 0)])
        self.assertIn("three", pathlib.Path(info["log"]).read_text())

    def test_queue_stop_on_fail(self):
        steps = self.tmp / "steps"
        steps.write_text("echo one\nfalse\necho three\n")
        self.start("strict", "--steps", str(steps), "--stop-on-fail")
        result, _ = self.wait("strict")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([s["state"] for s in self.state("strict")["steps"]], ["done", "failed", "skipped"])

    def test_queue_step_timeout_moves_on(self):
        steps = self.tmp / "steps"
        steps.write_text("sleep 60\necho after\n")
        self.start("stall", "--steps", str(steps), "--step-timeout", "1", "--grace", "1")
        result, took = self.wait("stall")
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(took, 15)
        self.assertEqual([s["state"] for s in self.state("stall")["steps"]], ["timed-out", "done"])

    def test_duplicate_name_refused_while_running(self):
        self.start("dup", "--", "sleep", "60")
        again = self.run_tool("start", "--name", "dup", "--", "true")
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("running", again.stderr)
        killed = self.run_tool("kill", "dup")
        self.assertEqual(killed.returncode, 0, killed.stdout)
        self.assertEqual(self.state("dup")["state"], "killed")
        self.start("dup", "--", "true")  # a finished name can be reused
        self.assertEqual(self.wait("dup")[0].returncode, 0)

    def test_busy_and_gc(self):
        self.start("gpu", "--gpu", "--", "sleep", "60")
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 1)
        self.assertIn("gpu job gpu", busy.stdout)
        self.assertIn(": busy (1 --gpu job(s)", busy.stdout)
        self.run_tool("kill", "gpu")
        self.assertIn("(0 --gpu job(s)", self.run_tool("busy").stdout)  # other agents may hold the GPU
        self.assertEqual(self.run_tool("gc", "--older-than", "1").stdout, "")  # too recent
        self.run_tool("gc", "--older-than", "0")
        self.assertFalse((self.tmp / "jobs" / "gpu").exists())

    def wait_for(self, predicate, timeout=20) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "condition never held")
            time.sleep(0.2)

    def test_gpu_jobs_run_one_at_a_time(self):
        marks = self.tmp / "marks"
        self.start("first", "--gpu", "--", "bash", "-c", f"echo first-start >> {marks}; sleep 3; "
                   f"echo first-end >> {marks}")
        self.wait_for(lambda: marks.exists())
        self.start("second", "--gpu", "--", "bash", "-c", f"echo second-start >> {marks}")
        self.wait_for(lambda: self.state("second").get("gpu_waiting"))
        self.assertIn("waiting for the GPU", self.run_tool("status").stdout)
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 1)
        self.assertIn("gpu job second waiting for the GPU", busy.stdout)
        self.assertIn("1 --gpu job(s), 1 waiting", busy.stdout)
        self.assertEqual(self.wait("second")[0].returncode, 0)
        self.assertEqual(marks.read_text().split(), ["first-start", "first-end", "second-start"])
        self.assertIn("waiting for the GPU: held by first", pathlib.Path(self.state("second")["log"]).read_text())

    def test_gpu_no_wait_refuses_while_held(self):
        self.start("holder", "--gpu", "--", "sleep", "60")
        self.wait_for(lambda: (self.tmp / "jobs" / "holder" / "gpu.json").exists())
        refused = self.run_tool("start", "--name", "eager", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("held by holder", refused.stderr)
        self.assertFalse((self.tmp / "jobs" / "eager").exists())
        self.run_tool("kill", "holder")
        self.start("eager", "--gpu", "--no-wait", "--", "true")  # the lock passes to its supervisor
        self.assertEqual(self.wait("eager")[0].returncode, 0)
        self.assertNotEqual(self.run_tool("start", "--name", "x", "--no-wait", "--", "true").returncode, 0)

    def test_no_wait_lock_outlives_start(self):
        self.start("held", "--gpu", "--no-wait", "--", "sleep", "60")  # `start` has exited here
        refused = self.run_tool("start", "--name", "next", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)

    def test_kill_ends_a_wait_and_timeout_counts_from_the_gpu(self):
        self.start("long", "--gpu", "--", "sleep", "4")
        self.wait_for(lambda: (self.tmp / "jobs" / "long" / "gpu.json").exists())
        self.start("queued", "--gpu", "--", "true")
        self.start("patient", "--gpu", "--timeout", "3", "--", "sleep", "1")
        self.wait_for(lambda: self.state("queued").get("gpu_waiting"))
        killed = self.run_tool("kill", "queued")
        self.assertEqual(killed.returncode, 0, killed.stdout)
        self.assertEqual(self.state("queued")["state"], "killed")
        self.assertEqual(self.state("long")["state"], "running")
        result, _ = self.wait("patient")  # waited ~4 s for the GPU, longer than its --timeout of 3
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_gpu_lock_freed_when_the_supervisor_dies(self):
        self.start("doomed", "--gpu", "--", "sleep", "60")
        self.wait_for(lambda: (self.tmp / "jobs" / "doomed" / "gpu.json").exists())
        os.kill(self.state("doomed")["supervisor"]["pid"], signal.SIGKILL)
        self.wait_for(lambda: self.state("doomed")["state"] == "lost")
        self.start("after", "--gpu", "--", "true")
        self.assertEqual(self.wait("after")[0].returncode, 0)


if __name__ == "__main__":
    unittest.main()
