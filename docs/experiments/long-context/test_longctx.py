# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic long-context protocol checks; run on a Spark, never locally."""
import contextlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("longctx", HERE / "longctx.py")
LONGCTX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LONGCTX)
BASELINE = sys.modules["baseline"]


class Memory:
    available = staticmethod(lambda: 128 << 30)

    def start(self):
        pass

    def minimum(self, start, end=None):
        return 120 << 30


def sse(chunks, done=True):
    data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if done:
        data += "data: [DONE]\n\n"
    return io.BytesIO(data.encode())


def completed_chunks(text, finish="stop", count=3, reasoning=""):
    return [
        {"choices": [{"delta": {"content": text, "reasoning_content": reasoning},
                      "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": finish}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": count}},
    ]


class ProtocolTest(unittest.TestCase):
    def read(self, response, path="/v1/chat/completions", requested=32):
        payload = {"max_tokens": requested} if path.startswith("/v1/") else {
            "n_predict": requested}
        with mock.patch.object(BASELINE.OPENER, "open", return_value=response):
            record = LONGCTX.stream(BASELINE.Client("http://unused"), path, payload, Memory())
        usage = record.get("usage") or {}
        LONGCTX.summarize(record, usage.get("prompt_tokens", 100),
                          usage.get("completion_tokens", (record.get("timings") or {}).get(
                              "predicted_n")))
        return record

    def reported(self, record):
        prompt = {"needles": [{"id": letter, "passphrase": letter + "-fact"}
                              for letter in ("A", "B", "C")]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            short = LONGCTX.report(Path(directory), "synthetic", prompt, record)
            saved = json.loads((Path(directory) / "synthetic.json").read_text())
        return short, saved

    def test_failed_process_queries_prevent_starting_the_runtime(self):
        for failed_command in ("nvidia-smi", LONGCTX.DOCKER[0]):
            with self.subTest(command=failed_command), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.toml"
                config.write_text("# no engine may start\n")
                args = SimpleNamespace(out=str(root / "run"), config=str(config),
                                       min_gib=105, runtime="unused")

                def query(command, *, check=False, **kwargs):
                    result = subprocess.CompletedProcess(command,
                                                         int(command[0] == failed_command),
                                                         stdout="", stderr="query unavailable")
                    if check:
                        result.check_returncode()
                    return result

                with (
                    mock.patch.object(LONGCTX, "Memory", Memory),
                    mock.patch.object(LONGCTX.subprocess, "run", side_effect=query),
                    mock.patch.object(LONGCTX.subprocess, "Popen") as started,
                ):
                    with self.assertRaises(subprocess.CalledProcessError):
                        LONGCTX.cmd_llmp(args)
                started.assert_not_called()

    def test_reasoning_does_not_count_as_a_visible_retrieval_answer(self):
        record = self.read(sse(completed_chunks("A-fact", reasoning="B-fact C-fact")))
        short, saved = self.reported(record)
        self.assertEqual(saved["needles_found"], {"A": True, "B": True, "C": True})
        self.assertEqual(saved["visible_needles_found"], {"A": True, "B": False, "C": False})
        self.assertFalse(short["visible_retrieval_pass"])
        self.assertTrue(short["early_stop"])
        self.assertEqual(short["completion_status"], "early_stop")
        self.assertEqual((short["completion_tokens"], short["requested_tokens"]), (3, 32))

    def test_visible_completed_stop_passes_even_when_eos_ends_before_the_budget(self):
        record = self.read(sse(completed_chunks("A-fact B-fact C-fact")))
        short, _ = self.reported(record)
        self.assertTrue(short["seen_done"])
        self.assertTrue(short["stream_complete"])
        self.assertTrue(short["completed_stop"])
        self.assertTrue(short["visible_retrieval_pass"])
        self.assertFalse(short["full_budget"])
        self.assertTrue(short["early_stop"])

    def test_complete_early_stop_retains_actual_count_and_valid_throughput(self):
        chunks = completed_chunks("first", count=3)
        chunks.insert(1, {"choices": [{"delta": {"content": " last"}}]})
        with mock.patch.object(LONGCTX.time, "monotonic", side_effect=range(10, 30)):
            record = self.read(sse(chunks))
        self.assertEqual(record["prefill_tok_s"], 100)
        self.assertEqual(record["decode_tok_s"], 2)
        self.assertEqual(record["completion_tokens"], 3)
        self.assertEqual(record["requested_tokens"], 32)
        self.assertEqual(record["completion_status"], "early_stop")

    def test_length_finish_is_a_full_budget_timing_but_not_completed_retrieval(self):
        record = self.read(sse(completed_chunks("A-fact B-fact C-fact", "length", 32)))
        short, _ = self.reported(record)
        self.assertTrue(short["full_budget"])
        self.assertFalse(short["early_stop"])
        self.assertEqual(short["completion_status"], "full_budget")
        self.assertFalse(short["visible_retrieval_pass"])

    def test_missing_done_and_missing_finish_reason_are_incomplete_errors(self):
        cases = [sse(completed_chunks("A-fact B-fact C-fact", count=32), done=False),
                 sse(completed_chunks("A-fact B-fact C-fact", finish=None)),
                 sse(completed_chunks("A-fact B-fact C-fact", finish="")),
                 sse(completed_chunks("A-fact B-fact C-fact", finish=42))]
        for response in cases:
            with self.subTest():
                record = self.read(response)
                short, _ = self.reported(record)
                self.assertFalse(short["stream_complete"])
                self.assertFalse(short["visible_retrieval_pass"])
                self.assertEqual(short["error"]["code"], "incomplete_stream")
                self.assertEqual(short["completion_status"], "error")
                self.assertIsNone(short["prefill_tok_s"])
                self.assertIsNone(short["decode_tok_s"])

    def test_truncated_json_and_server_error_preserve_partial_text(self):
        first = sse(completed_chunks("partial")[:1], done=False).getvalue()
        for suffix, code in [(b'data: {"choices":', "stream_read_failed"),
                             (b'data: {"error":{"code":"failed"}}\n\n', "failed")]:
            with self.subTest(code=code):
                record = self.read(io.BytesIO(first + suffix))
                self.assertEqual(record["text"], "partial")
                self.assertEqual(record["error"]["code"], code)
                self.assertFalse(record["stream_complete"])

    def test_http_truncation_preserves_partial_text_and_suppresses_throughput(self):
        first = sse(completed_chunks("partial")[:1], done=False).getvalue()

        class Interrupted(io.BytesIO):
            def __iter__(self):
                yield first
                raise http.client.IncompleteRead(b"data: incomplete")

        record = self.read(Interrupted())
        short, _ = self.reported(record)
        self.assertEqual(record["text"], "partial")
        self.assertEqual(short["error"]["code"], "stream_read_failed")
        self.assertFalse(short["stream_complete"])
        self.assertFalse(short["visible_retrieval_pass"])
        self.assertIsNone(short["prefill_tok_s"])
        self.assertIsNone(short["decode_tok_s"])

    def test_native_llama_terminal_envelope_is_retained_without_done(self):
        for stop_type, finish, count in [("eos", "stop", 3), ("limit", "length", 32)]:
            with self.subTest(stop_type=stop_type):
                record = self.read(sse([{"content": "answer", "stop": False},
                                       {"stop": True, "stop_type": stop_type,
                                        "timings": {"predicted_n": count}}], done=False),
                                   path="/completion")
                self.assertTrue(record["stream_complete"])
                self.assertFalse(record["seen_done"])
                self.assertEqual(record["finish_reason"], finish)
                self.assertEqual(record["llama_stop_details"]["stop_type"], stop_type)
                self.assertEqual(record["completion_tokens"], count)

    def test_short_length_is_not_mislabeled_as_early_natural_eos(self):
        record = self.read(sse(completed_chunks("partial", "length", 3)))
        self.assertEqual(record["completion_status"], "short_length")
        self.assertFalse(record["full_budget"])
        self.assertFalse(record["early_stop"])

    def test_open_error_is_an_incomplete_error_record(self):
        with mock.patch.object(BASELINE.OPENER, "open", side_effect=urllib.error.URLError("lost")):
            record = LONGCTX.stream(BASELINE.Client("http://unused"), "/v1/chat/completions",
                                     {"max_tokens": 32}, Memory())
        self.assertFalse(record["stream_complete"])
        self.assertEqual(record["error"]["code"], "stream_open_failed")
        self.assertEqual(record["requested_tokens"], 32)
        self.assertIn("min_available_bytes", record)

    def test_failed_cold_warmup_is_saved_and_prevents_any_timed_prompt(self):
        for failure in ("server_error", "incomplete", "http_error"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.toml"
                config.write_text("# synthetic, no engine is started\n")
                out = root / "run"
                process = mock.Mock()
                process.poll.return_value = None

                def started(*args, **kwargs):
                    (out / "service.log").write_text("llmp-runtime: ready\n")
                    return process

                response = (sse([{"error": {"code": "capacity_refused"}}], done=False)
                            if failure == "server_error" else
                            sse(completed_chunks("partial"), done=False))
                open_error = (urllib.error.HTTPError("http://unused", 503, "refused", {},
                                                     io.BytesIO(b"capacity_refused"))
                              if failure == "http_error" else None)
                args = SimpleNamespace(out=str(out), config=str(config), min_gib=105,
                                       runtime="unused", port=18140, model="synthetic",
                                       prompts=["must-never-be-read"], session=False,
                                       reasoning="drop", retries=0, tokens=32)
                with (
                    mock.patch.object(LONGCTX, "gate"),
                    mock.patch.object(LONGCTX, "Memory", Memory),
                    mock.patch.object(LONGCTX.time, "sleep"),
                    mock.patch.object(LONGCTX.subprocess, "Popen", side_effect=started),
                    mock.patch.object(LONGCTX, "prompt_files") as prompt_files,
                    mock.patch.object(BASELINE.OPENER, "open", return_value=response,
                                      side_effect=open_error) as opened,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaisesRegex(RuntimeError, "no timed requests"):
                        LONGCTX.cmd_llmp(args)
                prompt_files.assert_not_called()
                self.assertEqual(opened.call_count, 1)
                saved = json.loads((out / "run.json").read_text())
                self.assertEqual(saved["prompts"], {})
                self.assertIn("warmup", saved["failures"])
                self.assertFalse(saved["warmup"]["stream_complete"])
                self.assertEqual(json.loads((out / "warmup.json").read_text())["requested_tokens"], 8)
                process.send_signal.assert_called_once()


if __name__ == "__main__":
    unittest.main()
