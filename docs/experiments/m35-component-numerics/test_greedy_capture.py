# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import greedy_capture as collector


def chunk(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def usage():
    return {"choices": [], "usage": {"prompt_tokens": 66, "completion_tokens": 128,
                                    "total_tokens": 194,
                                    "prompt_tokens_details": {"cached_tokens": 0}}}


def events():
    return [chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning": "ré", "content": "A"}),
            chunk({"reasoning": "B", "content": "猫"}), chunk({}, "length"), usage()]


def encode(items, done=True):
    raw = b": heartbeat\r\n\r\n" + b"".join(
        b"data: " + json.dumps(e, ensure_ascii=False).encode() + b"\r\n\r\n" for e in items)
    return raw + (b"data: [DONE]\r\n\r\n" if done else b"")


def parse(items, done=True):
    return collector.parse_sse([encode(items, done)], io.BytesIO())


class GreedyCaptureTest(unittest.TestCase):
    def test_chunk_boundaries_utf8_crlf_and_original_digest_contract(self):
        raw = encode(events())
        for width in (1, 2, 7, 31, len(raw)):
            with self.subTest(width=width):
                sink = io.BytesIO()
                reply = collector.parse_sse((raw[i:i + width] for i in range(0, len(raw), width)), sink)
                self.assertEqual(reply["stream_text"], "réAB猫")
                self.assertEqual(reply["reasoning"], "réB")
                self.assertEqual(reply["content"], "A猫")
                self.assertEqual(reply["reply_sha256"], hashlib.sha256("réAB猫".encode()).hexdigest())
                self.assertTrue(raw.startswith(sink.getvalue()))
                self.assertIn(b"data: [DONE]", sink.getvalue())

    def test_done_without_trailing_newline_and_null_channel(self):
        sample = events()
        sample[1] = chunk({"reasoning": None, "content": "hello"})
        reply = collector.parse_sse([encode(sample).rstrip()], io.BytesIO())
        self.assertEqual(reply["stream_text"], "helloB猫")

    def test_missing_done_finish_usage_or_text_refused(self):
        with self.assertRaises(ValueError):
            parse(events(), done=False)
        for sample in (events()[:3] + [usage()], events()[:-1], [chunk({}, "stop"), usage()]):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                parse(sample)

    def test_invalid_json_error_event_and_unknown_sse_line_refused(self):
        for raw in (b"data: {broken}\n", b"data: []\n", b"data: {\"error\":{}}\n",
                    b"event: unexpected\n", b"data: \xff\n"):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeError)):
                collector.parse_sse([raw], io.BytesIO())

    def test_wrong_choice_or_delta_schema_refused(self):
        variants = [{"choices": {}}, {"choices": []},
                    {"choices": [chunk({})["choices"][0]] * 2},
                    chunk({"content": 42}), chunk({"tool_calls": []})]
        for index in (None, True, 1):
            variant = chunk({"content": "text"})
            variant["choices"][0]["index"] = index
            variants.append(variant)
        variant = chunk({})
        variant["choices"][0]["delta"] = None
        variants.append(variant)
        for item in variants:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse([item, *events()])

    def test_duplicate_finish_usage_and_post_finish_text_refused(self):
        for tail in (chunk({}, "length"), chunk({"content": "late"}), usage()):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                parse([*events(), tail])

    def test_usage_integrity_and_request_limit(self):
        for field, value in (("prompt_tokens", 0), ("prompt_tokens", True),
                             ("completion_tokens", 0), ("completion_tokens", 129),
                             ("total_tokens", 193)):
            sample = events()
            sample[-1]["usage"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse(sample)
        sample = events()
        sample[-1]["usage"] = None
        with self.assertRaises(ValueError):
            parse(sample)

    def test_digest_finish_and_full_usage_must_match_prior_capture(self):
        reply = parse(events())
        original = {k: copy.deepcopy(reply[k]) for k in ("reply_sha256", "finish", "usage")}
        original["error"] = None
        collector.match_reply(reply, original)
        for field in ("reply_sha256", "finish", "usage", "error"):
            changed = copy.deepcopy(original)
            changed[field] = "mismatch"
            with self.subTest(field=field), self.assertRaises(ValueError):
                collector.match_reply(reply, changed)

    def test_fresh_private_output_reserved_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            out = root / "fresh"
            argv = ["greedy_capture.py", "--family", "qwen38", "--arm", "A",
                    "--runtime", str(root / "absent-runtime"), "--quality-receipt", "absent.json",
                    "--repo", str(root / "absent-repo"), "--out", str(out)]
            old_umask = os.umask(0o022)
            try:
                with mock.patch("sys.argv", argv), self.assertRaises(FileNotFoundError):
                    collector.main()
                self.assertEqual(out.stat().st_mode & 0o777, 0o700)
                private = out / "probe"
                private.write_text("private")
                self.assertEqual(private.stat().st_mode & 0o777, 0o600)
                with mock.patch("sys.argv", argv), self.assertRaises(FileExistsError):
                    collector.main()
                self.assertEqual(private.read_text(), "private")
            finally:
                os.umask(old_umask)


if __name__ == "__main__":
    unittest.main()
