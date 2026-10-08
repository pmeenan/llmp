#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Measurement and failure-path regressions; no Docker, sudo, or GPU required."""

import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "reference_aba_experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def event(**fields):
    return b"data: " + json.dumps(fields).encode() + b"\n\n"


def counters():
    return dict(read_bytes=0, write_bytes=0, pswpin=0, pswpout=0, oom_kill=0)


class StreamTests(unittest.TestCase):
    def router(self, stream):
        router = experiment.Router.__new__(experiment.Router)
        router.args = SimpleNamespace(port=18081, device="unused")
        router.key = "unused"
        router.opener = mock.Mock()
        router.opener.open.return_value = io.BytesIO(stream)
        return router

    def test_first_token_ignores_pings_and_empty_events(self):
        stream = (b": keepalive\n\n" + event(tokens=[], content="", stop=False)
                  + event(tokens=[9], content="", stop=False)
                  + event(tokens=[10], content="visible", stop=False)
                  + event(tokens=[], content="", stop=True, truncated=False))
        with mock.patch.object(experiment.time, "monotonic_ns",
                               side_effect=[100, 200, 300, 400, 500, 600]), \
                mock.patch.object(experiment, "system_counters", return_value=counters()) as sampled:
            result = self.router(stream).complete("A", [1, 2])
        self.assertEqual(result["request_start_ns"], 100)
        self.assertEqual(result["first_token_ns"], 300)
        self.assertEqual(result["first_text_ns"], 400)
        self.assertEqual(result["tokens"], [9, 10])
        self.assertEqual(result["content"], "visible")
        sampled.assert_called_once_with("unused")

    def test_incomplete_error_and_truncated_streams_are_rejected(self):
        streams = [event(tokens=[9], stop=False), event(tokens=[], stop=True),
                   event(error={"message": "failed"}),
                   event(tokens=[9], stop=True, truncated=True)]
        for stream in streams:
            with self.subTest(stream=stream), \
                    mock.patch.object(experiment, "system_counters", return_value=counters()), \
                    self.assertRaises(RuntimeError):
                self.router(stream).complete("A", [1, 2])


class StateTests(unittest.TestCase):
    def test_reuse_requires_both_suffix_work_and_cached_coverage(self):
        for prompt_n, cache_n, accepted in [
                (3, 10, True), (4, 9, True), (13, 0, False),
                (2, 10, False), (3, 8, False), (3, 11, False),
                (3, 9, False), (4, 10, False)]:
            completion = {"final": {"timings": {"prompt_n": prompt_n, "cache_n": cache_n}}}
            with self.subTest(prompt_n=prompt_n, cache_n=cache_n):
                if accepted:
                    experiment.verify_reuse(completion, {"n_saved": 10}, range(13))
                else:
                    with self.assertRaises(RuntimeError):
                        experiment.verify_reuse(completion, {"n_saved": 10}, range(13))
        with self.assertRaises(RuntimeError):
            experiment.verify_reuse(completion, {"n_saved": 13}, range(13))

    def test_reuse_accounts_for_the_canonical_templates_rewritten_tail(self):
        completion = {"final": {"timings": {"prompt_n": 6, "cache_n": 7}}}
        experiment.verify_reuse(completion, {"n_saved": 10}, range(13), cached_prefix=7)
        for prefix in (0, 11, 14):
            with self.subTest(prefix=prefix), self.assertRaises(RuntimeError):
                experiment.verify_reuse(completion, {"n_saved": 10}, range(13), cached_prefix=prefix)
        trace = {"A_turns": [{"prompt": [1, 2, 3, 4], "expected_tokens": [5, 6]}],
                 "A_continuation": [1, 2, 3, 9, 8]}
        self.assertEqual(experiment.a_reusable_prefix(trace, 6), 3)
        self.assertEqual(experiment.a_reusable_prefix(trace, 2), 2)

    def test_save_syncs_the_state_file_and_directory(self):
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            router = experiment.Router.__new__(experiment.Router)
            router.states = Path(temp)
            def save_response(*unused):
                (router.states / "A.bin").write_bytes(b"state")
                return {"n_saved": 10, "n_written": 5}
            router.request = save_response
            synced = []
            def record_sync(fd):
                synced.append("directory" if stat.S_ISDIR(experiment.os.fstat(fd).st_mode) else "file")
            with mock.patch.object(experiment.os, "fsync", side_effect=record_sync):
                result = router.save("A")
            self.assertEqual(synced, ["file", "directory"])
            self.assertGreaterEqual(result["durable_elapsed_s"], result["api_elapsed_s"])

    def test_restore_rejects_token_byte_and_file_size_mismatches(self):
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            router = experiment.Router.__new__(experiment.Router)
            router.states = Path(temp)
            (router.states / "A.bin").write_bytes(b"state")
            saved = {"n_saved": 10, "n_written": 5}
            for response in [{"n_restored": 9, "n_read": 5},
                             {"n_restored": 10, "n_read": 4}]:
                with self.subTest(response=response):
                    router.request = mock.Mock(return_value=response)
                    with self.assertRaises(RuntimeError):
                        router.restore("A", saved)
            router.request = mock.Mock(return_value={"n_restored": 10, "n_read": 5})
            (router.states / "A.bin").write_bytes(b"truncated")
            with self.assertRaises(RuntimeError):
                router.restore("A", saved)


class TemplateTests(unittest.TestCase):
    def test_template_tokens_come_from_the_server_without_extra_special_tokens(self):
        router = experiment.Router.__new__(experiment.Router)
        messages = [{"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "follow-up"}]
        rendered = "<arbitrary-server-format>already-has-bos-and-turn-markers"
        router.request = mock.Mock(side_effect=[{"prompt": rendered}, {"tokens": [91, 92, 93]}])
        self.assertEqual(router.template_messages("A", messages), [91, 92, 93])
        self.assertEqual(router.request.call_args_list, [
            mock.call("/apply-template", {"model": "A", "messages": messages,
                                          "chat_template_kwargs": {"enable_thinking": False}}),
            mock.call("/tokenize", {"model": "A", "content": rendered,
                                   "add_special": False, "parse_special": True})])

    def test_prepare_forwards_history_and_accepts_short_template_tail_rewrites(self):
        class StopAfterTwoTurns(RuntimeError):
            pass
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            router = mock.Mock()
            histories = []
            def apply_template(model, messages):
                histories.append(json.loads(json.dumps(messages)))
                if len(histories) == 1:
                    return list(range(100))
                if len(histories) == 2:
                    return list(range(97)) + [301, 302, 303]
                raise StopAfterTwoTurns()
            router.template_messages.side_effect = apply_template
            router.complete.return_value = {"tokens": [7], "content": "opaque assistant reply"}
            with mock.patch.object(experiment, "Router", return_value=router), \
                    self.assertRaises(StopAfterTwoTurns):
                experiment.prepare(SimpleNamespace(output=Path(temp) / "prepare"), {})
            self.assertEqual([len(history) for history in histories], [1, 3, 5])
            self.assertEqual(histories[1][1], {"role": "assistant", "content": "opaque assistant reply"})
            self.assertEqual(histories[2][3], histories[1][1])
            router.complete.assert_any_call("A", list(range(97)) + [301, 302, 303])
            router.tokens.assert_not_called()
            router.stop.assert_called_once_with()

    def test_prepare_rejects_early_history_rewrites_before_freezing_trace(self):
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            output = Path(temp) / "prepare"
            router = mock.Mock()
            router.template_messages.side_effect = [list(range(100)), list(range(20)) + [301, 302]]
            router.complete.return_value = {"tokens": [7], "content": "assistant reply"}
            with mock.patch.object(experiment, "Router", return_value=router), \
                    self.assertRaisesRegex(RuntimeError, "rewrote.*tail"):
                experiment.prepare(SimpleNamespace(output=output), {})
            self.assertEqual(router.complete.call_count, 1)
            router.save.assert_not_called()
            router.stop.assert_called_once_with()
            self.assertTrue((output / "prepare-results.json").is_file())
            self.assertFalse((output / "trace.json").exists())


class PressureTests(unittest.TestCase):
    def test_unpinned_helper_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            root = Path(temp)
            program = root / "helper"
            program.write_bytes(b"wrong executable")
            pressure = experiment.Pressure(program, 1, root, "0" * 64)
            with mock.patch.object(experiment.subprocess, "Popen") as spawn, \
                    self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                pressure.start()
            spawn.assert_not_called()

    def test_pressure_must_be_alive_and_still_locked(self):
        pressure = experiment.Pressure(Path("unused"), 1, Path("unused"), "unused")
        pressure.receipt = {"pid": 123}
        pressure.process = mock.Mock()
        pressure.process.poll.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "exited"):
            pressure.verify()
        pressure.process.poll.return_value = None
        with mock.patch.object(Path, "read_text", return_value="VmLck:\t0 kB\n"), \
                self.assertRaisesRegex(RuntimeError, "unlocked"):
            pressure.verify()
        with mock.patch.object(Path, "read_text", return_value="VmLck:\t1048576 kB\n"):
            pressure.verify()

    def test_monitor_records_loss_and_releases_low_memory_pressure(self):
        released = mock.Mock()
        monitor = experiment.Telemetry(Path("unused"), released, lambda: False)
        monitor.done = mock.Mock()
        monitor.done.is_set.side_effect = [False, True]
        with mock.patch.object(experiment, "memory", return_value={"MemAvailable": experiment.GIB}):
            monitor._run()
        self.assertTrue(monitor.low_memory)
        self.assertTrue(monitor.pressure_lost)
        released.assert_called_once_with()


class SwapTests(unittest.TestCase):
    def test_allowance_counts_both_directions_and_includes_boundary(self):
        allowance = 1 << 20
        for page_size in (4096, 65536):
            pages = allowance // page_size
            with self.subTest(page_size=page_size), \
                    mock.patch.object(experiment.mmap, "PAGESIZE", page_size):
                self.assertEqual(experiment.verify_swap(
                    {"pswpin": 0, "pswpout": 0, "oom_kill": 0}, allowance), 0)
                self.assertEqual(experiment.verify_swap(
                    {"pswpin": pages // 2, "pswpout": pages // 2, "oom_kill": 0},
                    allowance), allowance)
                with self.assertRaisesRegex(RuntimeError, "exceeded"):
                    experiment.verify_swap(
                        {"pswpin": pages // 2, "pswpout": pages // 2 + 1, "oom_kill": 0},
                        allowance)

    def test_oom_and_backwards_counters_are_always_rejected(self):
        for deltas in [{"pswpin": 0, "pswpout": 0, "oom_kill": 1},
                       {"pswpin": -1, "pswpout": 0, "oom_kill": 0},
                       {"pswpin": 0, "pswpout": -1, "oom_kill": 0},
                       {"pswpin": 0, "pswpout": 0, "oom_kill": -1}]:
            with self.subTest(deltas=deltas), self.assertRaises(RuntimeError):
                experiment.verify_swap(deltas, 1 << 20)


class CleanupTests(unittest.TestCase):
    def test_cleanup_failures_still_stop_router_monitor_and_write_receipt(self):
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            root = Path(temp)
            trace = root / "trace.json"
            trace.write_text("{}")
            args = SimpleNamespace(trace=trace, ballast=Path("unused"), device="unused")
            router, pressure, monitor = mock.Mock(), mock.Mock(), mock.Mock()
            pressure.receipt = {}
            router.start.side_effect = RuntimeError("startup failed")
            pressure.stop.side_effect = RuntimeError("pressure cleanup failed")
            router.stop.side_effect = RuntimeError("router cleanup failed")
            monitor.thread.ident = 1
            monitor.__enter__ = mock.Mock()
            monitor.__exit__ = mock.Mock()
            with mock.patch.object(experiment, "Router", return_value=router), \
                    mock.patch.object(experiment, "Pressure", return_value=pressure), \
                    mock.patch.object(experiment, "Telemetry", return_value=monitor), \
                    mock.patch.object(experiment, "memory", return_value={"MemTotal": 128 * experiment.GIB}), \
                    mock.patch.object(experiment, "system_counters", return_value=counters()), \
                    self.assertRaises(RuntimeError):
                experiment.run_trial(args, {"image": "unused", "ballast_sha256": "unused",
                                            "swap_allowance_bytes": 1 << 20},
                                     {}, "pressure_restore", root / "trial")
            pressure.stop.assert_called_once_with()
            router.stop.assert_called_once_with()
            monitor.__exit__.assert_called_once_with()
            self.assertFalse(json.loads((root / "trial" / "result.json").read_text())["passed"])

    def test_completed_trial_is_not_marked_passed_when_cleanup_fails(self):
        with tempfile.TemporaryDirectory(prefix="llmp-aba-test-") as temp:
            root = Path(temp)
            trace_path = root / "trace.json"
            trace_path.write_text("{}")
            args = SimpleNamespace(trace=trace_path, ballast=Path("unused"),
                                   models=root, device="unused")
            trace = {"A_turns": [{"prompt": [1, 2], "expected_tokens": [9]}],
                     "B_prompt": [3], "B_expected_tokens": [10],
                     "A_continuation": [1, 2, 9, 4], "A_expected_continuation": [11]}
            router, pressure, monitor = mock.Mock(), mock.Mock(), mock.Mock()
            pressure.receipt = {}
            pressure.stop.side_effect = RuntimeError("pressure cleanup failed")
            monitor.low_memory = monitor.pressure_lost = False
            monitor.thread.ident = 1
            monitor.__enter__ = mock.Mock()
            monitor.__exit__ = mock.Mock()
            monitor.summary.return_value = {}
            router.load.return_value = {"value": "loaded"}
            def create_router(*unused, **kwargs):
                router.states = root / "trial" / "states"
                router.states.mkdir()
                return router
            def completion(model, prompt, **kwargs):
                now = experiment.time.monotonic_ns()
                tokens = [10] if model == "B" else [9] if len(prompt) == 2 else [11]
                return {"request_start_ns": now, "first_token_ns": now + 1,
                        "first_text_ns": now + 1, "end_ns": now + 2, "tokens": tokens,
                        "io_at_first_token": counters(),
                        "final": {"timings": {"prompt_n": 1, "cache_n": 3}}}
            router.complete.side_effect = completion
            router.stop.side_effect = lambda: (root / "trial" / "server.log").write_text(
                "offloaded 31/31 layers to GPU\noffloaded 42/42 layers to GPU\n")
            pins = {"image": "unused", "ballast_sha256": "unused", "models": {},
                    "swap_allowance_bytes": 1 << 20}
            with mock.patch.object(experiment, "Router", side_effect=create_router), \
                    mock.patch.object(experiment, "Pressure", return_value=pressure), \
                    mock.patch.object(experiment, "Telemetry", return_value=monitor), \
                    mock.patch.object(experiment, "memory", return_value={"MemTotal": 128 * experiment.GIB}), \
                    mock.patch.object(experiment, "system_counters", return_value=counters()), \
                    self.assertRaisesRegex(RuntimeError, "pressure cleanup failed"):
                experiment.run_trial(args, pins, trace, "resident", root / "trial")
            monitor.__exit__.assert_called_once_with()
            self.assertFalse(json.loads((root / "trial" / "result.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
