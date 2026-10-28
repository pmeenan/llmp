# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of tools/spark-job: real supervisors and process groups, with short sleeps."""

import fcntl
import json
import os
import pathlib
import runpy
import signal
import socket
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
        # A GPU with no compute processes, so `busy` does not see this host's real GPU work.
        fake_bin = self.tmp / "bin"
        fake_bin.mkdir()
        (fake_bin / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
        (fake_bin / "nvidia-smi").chmod(0o755)
        self.env = dict(os.environ, LLMP_JOBS_DIR=str(self.tmp / "jobs"),
                        PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
        self.env.pop("LLMP_JOBS_SHARED", None)
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
        # steps: llmp-runtime refuses a process lock under a group- or world-writable directory.
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
        self.wait_for(lambda: self.state("gpu").get("admitted"))
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 1)
        self.assertIn("exclusive job gpu running", busy.stdout)
        self.assertIn(": busy (exclusive: 1 running, 0 waiting; shared: 0 of 2 running", busy.stdout)
        self.run_tool("kill", "gpu")
        free = self.run_tool("busy")
        self.assertEqual(free.returncode, 0, free.stdout)
        self.assertIn(": free (exclusive: 0 running, 0 waiting; shared: 0 of 2 running", free.stdout)
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
        self.assertIn("exclusive job second waiting for the GPU", busy.stdout)
        self.assertIn("exclusive: 1 running, 1 waiting", busy.stdout)
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
        self.start("long", "--gpu", "--", "sleep", "60")
        self.wait_for(lambda: (self.tmp / "jobs" / "long" / "gpu.json").exists())
        self.start("queued", "--gpu", "--", "true")
        self.start("patient", "--gpu", "--timeout", "3", "--", "sleep", "1")
        queued_at = time.monotonic()
        self.wait_for(lambda: self.state("queued").get("gpu_waiting"))
        killed = self.run_tool("kill", "queued")
        self.assertEqual(killed.returncode, 0, killed.stdout)
        self.assertEqual(self.state("queued")["state"], "killed")
        self.assertEqual(self.state("long")["state"], "running")
        # Hold admission longer than patient's timeout, independently of how
        # long the status/kill subprocesses take on a busy build host.
        time.sleep(max(0, 4 - (time.monotonic() - queued_at)))
        self.assertEqual(self.run_tool("kill", "long").returncode, 0)
        result, _ = self.wait("patient")  # waited >=4 s for the GPU, longer than its --timeout of 3
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_gpu_lock_outlives_a_lost_supervisor_until_its_child_exits(self):
        self.start("doomed", "--gpu", "--", "sleep", "60")
        self.wait_for(lambda: (self.tmp / "jobs" / "doomed" / "child.json").exists())
        child = json.loads((self.tmp / "jobs" / "doomed" / "child.json").read_text())["pid"]
        os.kill(self.state("doomed")["supervisor"]["pid"], signal.SIGKILL)
        self.wait_for(lambda: self.state("doomed")["state"] == "lost")
        self.assertTrue(pid_alive(child))
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 1, busy.stdout)
        self.assertIn("lost supervisor and a running child", busy.stdout)
        collected = self.run_tool("gc", "--older-than", "0")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertTrue((self.tmp / "jobs" / "doomed").is_dir())
        self.assertTrue(self.state("doomed")["orphans"])
        refused = self.run_tool("start", "--name", "eager", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        self.start("after", "--gpu", "--", "true")
        self.wait_for(lambda: self.state("after").get("gpu_waiting"))
        self.assertTrue(pid_alive(child))
        self.assertEqual(self.run_tool("kill", "doomed").returncode, 0)
        self.assertFalse(pid_alive(child))
        self.assertEqual(self.wait("after")[0].returncode, 0)

    def test_gpu_admission_waits_for_a_lost_wrappers_descendants(self):
        # Python's default close_fds removes the inherited GPU descriptor
        # from the worker. Once its wrapper exits, only the group guard
        # can keep a second GPU job from overlapping the worker.
        worker_file = self.tmp / "worker"
        release = self.tmp / "release"
        wrapper = ("import pathlib,subprocess,time; "
                   "worker=subprocess.Popen(['sleep','60'],close_fds=True); "
                   f"pathlib.Path({str(worker_file)!r}).write_text(str(worker.pid)); "
                   f"release=pathlib.Path({str(release)!r}); "
                   "\nwhile not release.exists(): time.sleep(.01)")
        self.start("wrapper", "--gpu", "--", sys.executable, "-B", "-c", wrapper)
        self.wait_for(worker_file.exists)
        child = json.loads((self.tmp / "jobs" / "wrapper" / "child.json").read_text())["pid"]
        worker = int(worker_file.read_text())
        os.kill(self.state("wrapper")["supervisor"]["pid"], signal.SIGKILL)
        self.wait_for(lambda: self.state("wrapper")["state"] == "lost")
        release.touch()
        self.wait_for(lambda: not pid_alive(child))
        self.assertTrue(pid_alive(worker))
        self.assertTrue(self.state("wrapper")["orphans"])
        refused = self.run_tool("start", "--name", "eager", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        self.assertIn("process group still runs", refused.stderr)
        self.start("after", "--gpu", "--", "true")
        self.wait_for(lambda: self.state("after").get("gpu_waiting"))
        self.assertTrue(pid_alive(worker))
        self.assertEqual(self.run_tool("kill", "wrapper").returncode, 0)
        self.assertFalse(pid_alive(worker))
        self.assertEqual(self.wait("after")[0].returncode, 0)

    def test_gpu_orphan_scan_tolerates_concurrent_completed_job_removal(self):
        helper = runpy.run_path(str(TOOL))["lock_orphans"]
        root = self.tmp / "jobs"
        directory = root / "completed"
        directory.mkdir(parents=True)
        original = helper.__globals__["job_state"]

        def removed_before_read(path):
            self.assertEqual(path, directory)
            directory.rmdir()
            return original(path)  # its initial directory stat now fails

        helper.__globals__["job_state"] = removed_before_read
        self.assertEqual(helper(root), [])

    # ------------------------------------------------------------ shared jobs

    def marks(self) -> list[str]:
        path = self.tmp / "marks"
        return path.read_text().split() if path.exists() else []

    def mark(self, word: str) -> str:
        return f"echo {word} >> {self.tmp / 'marks'}"

    def until(self, path: pathlib.Path) -> str:
        """A shell loop that ends once path exists (at most 30 s, then fails)."""
        return f"for i in $(seq 300); do [ -e {path} ] && break; sleep 0.1; done; [ -e {path} ]"

    def test_flags_are_checked(self):
        both = self.run_tool("start", "--name", "x", "--gpu", "--shared", "--", "true")
        self.assertNotEqual(both.returncode, 0)
        self.assertIn("not both", both.stderr)
        bad = dict(self.env, LLMP_JOBS_SHARED="0")
        refused = subprocess.run([sys.executable, "-B", str(TOOL), "start", "--name", "x", "--shared",
                                  "--", "true"], env=bad, cwd=self.tmp, capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("LLMP_JOBS_SHARED", refused.stderr)
        self.assertFalse((self.tmp / "jobs" / "x").exists())

    def test_shared_jobs_overlap(self):
        # Each finishes only once the other has started: they must run together.
        for me, other in (("a", "b"), ("b", "a")):
            self.start(f"shared-{me}", "--shared", "--", "bash", "-c",
                       f"touch {self.tmp / me}; {self.until(self.tmp / other)}")
        self.assertEqual(self.wait("shared-a")[0].returncode, 0)
        self.assertEqual(self.wait("shared-b")[0].returncode, 0)
        self.assertEqual(self.state("shared-a")["class"], "shared")
        self.assertEqual({self.state("shared-a")["slot"], self.state("shared-b")["slot"]}, {0, 1})
        self.assertNotIn("waiting", pathlib.Path(self.state("shared-b")["log"]).read_text())

    def test_a_shared_job_past_the_slots_waits(self):
        release = self.tmp / "release"
        for name in ("one", "two"):
            self.start(name, "--shared", "--", "bash", "-c", self.until(release))
            self.wait_for(lambda n=name: self.state(n).get("admitted"))
        self.start("three", "--shared", "--", "bash", "-c", self.mark("three"))
        self.wait_for(lambda: self.state("three").get("gpu_waiting"))
        self.assertIn("waiting for a shared slot", self.run_tool("status").stdout)
        refused = self.run_tool("start", "--name", "eager", "--shared", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        self.assertIn("no shared slot", refused.stderr)
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 3, busy.stdout)
        self.assertIn(": shared (exclusive: 0 running, 0 waiting; shared: 2 of 2 running, 1 waiting",
                      busy.stdout)
        self.assertIn("shared job three waiting for a shared slot", busy.stdout)
        self.assertEqual(self.marks(), [])
        release.touch()
        self.assertEqual(self.wait("three")[0].returncode, 0)
        self.assertIn("held by one (shared", pathlib.Path(self.state("three")["log"]).read_text())

    def test_exclusive_waits_for_shared_jobs_and_keeps_new_ones_out(self):
        release = self.tmp / "release"
        self.start("build", "--shared", "--", "bash", "-c",
                   f"{self.mark('build-start')}; {self.until(release)}; {self.mark('build-end')}")
        self.wait_for(lambda: self.marks() == ["build-start"])
        self.start("measure", "--gpu", "--timeout", "20", "--", "bash", "-c",
                   f"{self.mark('measure-start')}; sleep 1; {self.mark('measure-end')}")
        self.wait_for(lambda: self.state("measure").get("gpu_waiting"))
        # While it waits, no shared job starts, though a slot is free.
        refused = self.run_tool("start", "--name", "eager", "--shared", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        self.assertIn("exclusive job(s) waiting: measure", refused.stderr)
        self.start("later", "--shared", "--", "bash", "-c", self.mark("later"))
        self.wait_for(lambda: self.state("later").get("gpu_waiting"))
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 1, busy.stdout)
        self.assertIn(": busy (exclusive: 0 running, 1 waiting; shared: 1 of 2 running, 1 waiting",
                      busy.stdout)
        time.sleep(1.5)  # a few admission rounds: neither may start
        self.assertEqual(self.marks(), ["build-start"])
        release.touch()
        self.assertEqual(self.wait("measure")[0].returncode, 0)
        self.assertEqual(self.wait("later")[0].returncode, 0)
        self.assertEqual(self.marks(), ["build-start", "build-end", "measure-start", "measure-end", "later"])
        self.assertIn("waiting for the GPU: held by build (shared",
                      pathlib.Path(self.state("measure")["log"]).read_text())

    def test_shared_waits_for_a_running_exclusive_job_and_its_timeout_counts_from_admission(self):
        self.start("measure", "--gpu", "--", "bash", "-c", f"{self.mark('measure')}; sleep 3")
        self.wait_for(lambda: self.state("measure").get("admitted"))
        refused = self.run_tool("start", "--name", "eager", "--shared", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        self.assertIn("measure (exclusive, running for", refused.stderr)
        # Waits ~3 s for its slot, longer than its --timeout of 2.
        self.start("build", "--shared", "--timeout", "2", "--", "bash", "-c",
                   f"sleep 1; {self.mark('build')}")
        self.wait_for(lambda: self.state("build").get("gpu_waiting"))
        result, _ = self.wait("build")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.marks(), ["measure", "build"])
        self.assertEqual(self.state("measure")["state"], "done")

    def test_shared_no_wait_lock_outlives_start(self):
        self.env["LLMP_JOBS_SHARED"] = "1"
        self.start("held", "--shared", "--no-wait", "--", "sleep", "60")  # `start` has exited here
        for flag in ("--shared", "--gpu"):
            refused = self.run_tool("start", "--name", "next", flag, "--no-wait", "--", "true")
            self.assertEqual(refused.returncode, 1, refused.stdout)
        self.assertEqual(self.state("held")["slot"], 0)

    def lose_supervisor(self, name: str) -> None:
        os.kill(self.state(name)["supervisor"]["pid"], signal.SIGKILL)
        self.wait_for(lambda: self.state(name)["state"] == "lost")

    def test_a_lost_shared_jobs_child_keeps_its_share(self):
        self.env["LLMP_JOBS_SHARED"] = "1"
        self.start("doomed", "--shared", "--", "sleep", "60")
        self.wait_for(lambda: (self.tmp / "jobs" / "doomed" / "child.json").exists())
        child = json.loads((self.tmp / "jobs" / "doomed" / "child.json").read_text())["pid"]
        self.lose_supervisor("doomed")
        self.assertTrue(pid_alive(child))
        for flag in ("--shared", "--gpu"):
            refused = self.run_tool("start", "--name", "eager", flag, "--no-wait", "--", "true")
            self.assertEqual(refused.returncode, 1, refused.stdout)
            self.assertIn("process group still runs", refused.stderr)
        busy = self.run_tool("busy")
        self.assertEqual(busy.returncode, 3, busy.stdout)
        self.assertIn("shared job doomed has a lost supervisor and a running child", busy.stdout)
        self.assertIn("shared: 1 of 1 running", busy.stdout)
        self.start("after", "--shared", "--", "true")
        self.wait_for(lambda: self.state("after").get("gpu_waiting"))
        self.assertEqual(self.run_tool("kill", "doomed").returncode, 0)
        self.assertFalse(pid_alive(child))
        self.assertEqual(self.wait("after")[0].returncode, 0)

    def test_a_lost_shared_wrappers_descendants_keep_its_slot(self):
        # The worker closes the inherited descriptors (Python's close_fds), so
        # once its wrapper exits only the recorded slot keeps it counted.
        self.env["LLMP_JOBS_SHARED"] = "2"
        worker_file = self.tmp / "worker"
        release = self.tmp / "release"
        wrapper = ("import pathlib,subprocess,time; "
                   "worker=subprocess.Popen(['sleep','60'],close_fds=True); "
                   f"pathlib.Path({str(worker_file)!r}).write_text(str(worker.pid)); "
                   f"release=pathlib.Path({str(release)!r}); "
                   "\nwhile not release.exists(): time.sleep(.01)")
        self.start("wrapper", "--shared", "--", sys.executable, "-B", "-c", wrapper)
        self.wait_for(worker_file.exists)
        child = json.loads((self.tmp / "jobs" / "wrapper" / "child.json").read_text())["pid"]
        worker = int(worker_file.read_text())
        slot = self.state("wrapper")["slot"]
        self.lose_supervisor("wrapper")
        release.touch()
        self.wait_for(lambda: not pid_alive(child))
        self.assertTrue(pid_alive(worker))
        # Its slot is free as a lock but still taken: a new shared job gets the other one.
        self.start("other", "--shared", "--no-wait", "--", "sleep", "60")
        self.assertNotEqual(self.state("other")["slot"], slot)
        refused = self.run_tool("start", "--name", "third", "--shared", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        refused = self.run_tool("start", "--name", "measure", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)
        self.run_tool("kill", "other")
        refused = self.run_tool("start", "--name", "measure", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(refused.returncode, 1, refused.stdout)  # the worker still runs
        self.assertIn("process group still runs", refused.stderr)
        self.assertEqual(self.run_tool("kill", "wrapper").returncode, 0)
        self.assertFalse(pid_alive(worker))
        self.start("measure", "--gpu", "--no-wait", "--", "true")
        self.assertEqual(self.wait("measure")[0].returncode, 0)

    def test_waiting_shared_jobs_never_exceed_the_slots(self):
        # Six supervisors race for two slots; each records how many run beside it.
        running = self.tmp / "running"
        running.mkdir()
        counts = self.tmp / "counts"
        names = [f"queue{i}" for i in range(6)]
        for name in names:
            self.start(name, "--shared", "--", "bash", "-c",
                       f"touch {running / name}; ls {running} | wc -l >> {counts}; sleep 0.5; "
                       f"rm {running / name}")
        for name in names:
            self.assertEqual(self.wait(name)[0].returncode, 0)
        seen = [int(n) for n in counts.read_text().split()]
        self.assertEqual(len(seen), 6)
        self.assertLessEqual(max(seen), 2)

    def test_racing_starts_never_exceed_the_slots(self):
        # Eight shared --no-wait starts at once: exactly two are admitted.
        procs = [subprocess.Popen([sys.executable, "-B", str(TOOL), "start", "--name", f"race{i}", "--shared",
                                   "--no-wait", "--", "sleep", "60"], env=self.env, cwd=self.tmp,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for i in range(8)]
        codes = [p.wait(timeout=60) for p in procs]
        for p in procs:
            p.stdout.close()
            p.stderr.close()
        self.started += [f"race{i}" for i, code in enumerate(codes) if code == 0]
        self.assertEqual(sorted(codes), [0, 0] + [1] * 6)
        slots = sorted(self.state(f"race{i}")["slot"] for i, code in enumerate(codes) if code == 0)
        self.assertEqual(slots, [0, 1])

    # ------------------------------------------------------------ interleavings

    def admission(self):
        """The tool's Admission class over this test's state directory, driven in this process."""
        root = self.tmp / "jobs"
        root.mkdir(exist_ok=True)
        cls = runpy.run_path(str(TOOL))["Admission"]
        made = []

        def make(kind, slots=2):
            made.append(cls(root, kind, slots))
            return made[-1]

        self.addCleanup(lambda: [a.close() for a in made])
        return make, root

    def test_admission_interleavings_keep_an_exclusive_job_alone(self):
        make, _ = self.admission()
        build = make("shared")
        self.assertTrue(build.try_admit())
        build.release_gate()
        measure = make("exclusive")
        self.assertFalse(measure.try_admit())  # the build runs; it keeps the gate while it waits
        self.assertTrue(measure.gate_held)
        later = make("shared")
        self.assertFalse(later.try_admit())  # a slot is free, but an exclusive job waits
        self.assertIsNone(later.slot_fd)
        second = make("exclusive")
        self.assertFalse(second.try_admit())  # the first waiting exclusive job holds the gate
        self.assertFalse(second.gate_held)
        build.close()
        self.assertTrue(measure.try_admit())
        self.assertFalse(measure.gate_held)
        self.assertFalse(later.try_admit())  # the exclusive job runs
        self.assertFalse(second.try_admit())  # it now waits holding the gate
        self.assertTrue(second.gate_held)
        measure.close()
        self.assertFalse(later.try_admit())  # the second exclusive job goes first
        self.assertTrue(second.try_admit())
        second.close()
        self.assertTrue(later.try_admit())

    def test_a_shared_job_mid_admission_only_defers_an_exclusive_one(self):
        # A shared job holds the gate shared only while it takes the class lock:
        # an exclusive job that finds it so gives up this round holding nothing.
        make, root = self.admission()
        gate = os.open(root / ".gate.lock", os.O_RDWR | os.O_CREAT, 0o644)
        self.addCleanup(os.close, gate)
        fcntl.flock(gate, fcntl.LOCK_SH)
        measure = make("exclusive")
        self.assertFalse(measure.try_admit())
        self.assertFalse(measure.gate_held)
        probe = os.open(root / ".gpu.lock", os.O_RDWR)
        self.addCleanup(os.close, probe)
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # it did not keep the class lock either
        fcntl.flock(probe, fcntl.LOCK_UN)
        fcntl.flock(gate, fcntl.LOCK_UN)
        self.assertTrue(measure.try_admit())

    def test_a_shared_job_waiting_for_a_slot_holds_no_lock(self):
        make, root = self.admission()
        first = make("shared", 1)
        self.assertTrue(first.try_admit())
        waiting = make("shared", 1)
        self.assertFalse(waiting.try_admit())
        self.assertIsNone(waiting.slot_fd)
        first.close()
        probe = os.open(root / ".gpu.lock", os.O_RDWR)
        self.addCleanup(os.close, probe)
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the waiting job keeps nothing shared
        fcntl.flock(probe, fcntl.LOCK_UN)
        self.assertTrue(waiting.try_admit())

    def overlap_script(self, name: str, hold: float) -> str:
        """A child that lists who runs beside it as it starts and before it ends.

        Each child marks itself before it lists, so of two that overlap at
        least one sees the other."""
        running, seen = self.tmp / "running", self.tmp / "seen"
        return (f"touch {running / name}; ls {running} > {seen / name}.start; sleep {hold}; "
                f"ls {running} > {seen / name}.end; rm {running / name}")

    def check_overlaps(self, names: list[str]) -> None:
        for name in names:
            for when in ("start", "end"):
                beside = set((self.tmp / "seen" / f"{name}.{when}").read_text().split()) - {name}
                if name.startswith("excl"):
                    self.assertEqual(beside, set(), f"{name} ran beside {beside} at its {when}")
                else:
                    self.assertFalse({b for b in beside if b.startswith("excl")},
                                     f"{name} ran beside an exclusive job at its {when}")
                    self.assertLessEqual(len(beside), 1, f"{name} ran beside {beside} at its {when}")

    def test_mixed_waiting_jobs_never_overlap_an_exclusive_one(self):
        # Exclusive and shared supervisors started together, all waiting for one another.
        (self.tmp / "running").mkdir()
        (self.tmp / "seen").mkdir()
        names = [f"excl{i}" for i in range(3)] + [f"shrd{i}" for i in range(6)]
        names = [names[i] for i in (3, 0, 4, 5, 1, 6, 7, 2, 8)]  # interleaved start order
        procs = [subprocess.Popen([sys.executable, "-B", str(TOOL), "start", "--name", name,
                                   "--gpu" if name.startswith("excl") else "--shared", "--",
                                   "bash", "-c", self.overlap_script(name, 0.3)],
                                  env=self.env, cwd=self.tmp, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) for name in names]
        self.assertEqual([p.wait(timeout=60) for p in procs], [0] * len(names))
        self.started += names
        for name in names:
            self.assertEqual(self.wait(name, timeout=60)[0].returncode, 0, name)
        self.check_overlaps(names)

    def test_racing_no_wait_starts_of_both_classes_admit_a_valid_set(self):
        # Exclusive and shared --no-wait starts at once: one exclusive job, or
        # up to two shared ones, never a mix; and never nothing.
        for round_ in range(3):
            names = [f"excl{round_}x{i}" if i % 3 == 0 else f"shrd{round_}x{i}" for i in range(9)]
            procs = [subprocess.Popen([sys.executable, "-B", str(TOOL), "start", "--name", name,
                                       "--gpu" if name.startswith("excl") else "--shared", "--no-wait",
                                       "--", "sleep", "60"], env=self.env, cwd=self.tmp,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                     for name in names]
            admitted = [n for n, p in zip(names, procs) if p.wait(timeout=60) == 0]
            self.started += admitted
            exclusive = [n for n in admitted if n.startswith("excl")]
            self.assertTrue(admitted)
            self.assertTrue(admitted == exclusive[:1] or (not exclusive and len(admitted) <= 2), admitted)
            for name in admitted:
                self.assertEqual(self.run_tool("kill", name).returncode, 0)

    def test_a_no_wait_start_from_before_shared_jobs_keeps_its_lock(self):
        # A `start` already running as the tool is replaced passes `gpu_fd`, the
        # older name of `lock_fds`, to a supervisor from the new file.
        root = self.tmp / "jobs"
        job_dir = root / "old"
        job_dir.mkdir(parents=True)
        fd = os.open(root / ".gpu.lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = {"name": "old", "host": socket.gethostname(), "cwd": str(self.tmp),
               "log": str(job_dir / "log"), "command": ["sleep", "60"], "steps_file": None,
               "gpu": True, "timeout": None, "step_timeout": None, "grace": 1.0,
               "stop_on_fail": False, "start": time.time(), "umask": 0o022, "gpu_fd": fd}
        (job_dir / "job.json").write_text(json.dumps(job))
        with open(job_dir / "log", "ab") as out:
            sup = subprocess.Popen([sys.executable, "-B", str(TOOL), "_supervise", str(job_dir)],
                                   env=self.env, cwd=self.tmp, stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=out, start_new_session=True, pass_fds=(fd,))
        os.close(fd)
        job["supervisor"] = {"pid": sup.pid, "start_ticks": int(pathlib.Path(f"/proc/{sup.pid}/stat")
                                                              .read_text().rsplit(")", 1)[1].split()[19])}
        (job_dir / "job.json").write_text(json.dumps(job))
        self.started.append("old")
        self.wait_for(lambda: (job_dir / "child.json").exists())  # never waits on its own lock
        for flag in ("--gpu", "--shared"):
            refused = self.run_tool("start", "--name", "next", flag, "--no-wait", "--", "true")
            self.assertEqual(refused.returncode, 1, refused.stdout)
        self.assertEqual(self.run_tool("kill", "old").returncode, 0)
        sup.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
