# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/k1_quality.py."""

import contextlib
import importlib.util
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("k1_quality", ROOT / "tools/k1_quality.py")
k1 = importlib.util.module_from_spec(spec)
sys.modules["k1_quality"] = k1
spec.loader.exec_module(k1)


class TestK1Quality(unittest.TestCase):
    def test_families_coverage(self):
        expected = {
            "gemma2",
            "gemma3",
            "gemma26",
            "gemma26-assistant",
            "gemma31",
            "gemma31-assistant",
            "qwen38",
            "qwen38-gguf",
            "deepseek",
        }
        self.assertEqual(set(k1.FAMILIES.keys()), expected)
        for name in k1.STANDARD_LLM_FAMILIES:
            self.assertIn(name, k1.FAMILIES)

    def test_make_config_encodings(self):
        share = pathlib.Path("/share")
        data = pathlib.Path("/data")
        port = 18114
        family = k1.FAMILIES["gemma2"]

        cfg_f16 = k1.make_config(family, "f16", share, data, port)
        self.assertIn('kv_cache_encoding = "f16"', cfg_f16)
        self.assertIn('installed = "/share/m3-artifacts"', cfg_f16)
        self.assertIn('data_dir = "/data"', cfg_f16)
        self.assertIn("port = 18114", cfg_f16)

        cfg_q8 = k1.make_config(family, "q8_0", share, data, port)
        self.assertIn('kv_cache_encoding = "q8_0"', cfg_q8)

    def test_assistant_families(self):
        fam26 = k1.FAMILIES["gemma26-assistant"]
        self.assertIn(k1.GEMMA26_ASSISTANT, fam26.models_toml)
        self.assertIn("speculation = true", fam26.models_toml)

        fam31 = k1.FAMILIES["gemma31-assistant"]
        self.assertIn(k1.GEMMA31_ASSISTANT, fam31.models_toml)
        self.assertIn("speculation = true", fam31.models_toml)

    def test_passage(self):
        corpus = "line 1\nline 2 with some words\nline 3 ends here.\n"
        # Small corpus edge case
        p1 = k1.passage(corpus, 0, 10)
        self.assertEqual(len(p1), 10)
        # Offset passage
        p2 = k1.passage(corpus, 100003, 15)
        self.assertEqual(len(p2), 15)
        # A shorter passage is a prefix of a longer one from the same start.
        big = "".join(f"line {i} of the corpus\n" for i in range(20000))
        long = k1.passage(big, 100003, 5000)
        self.assertEqual(k1.passage(big, 100003, 700), long[:700])
        self.assertTrue(big[big.index(long) - 1] == "\n")
        with self.assertRaises(ValueError):
            k1.passage(big, 1, k1.MAX_PASSAGE_CHARS + 1)

    def test_gemma4_scored_in_its_turn_format(self):
        for name in ("gemma26", "gemma26-assistant", "gemma31", "gemma31-assistant"):
            self.assertEqual(k1.FAMILIES[name].prefix, k1.GEMMA4_TURN)
            self.assertTrue(k1.GEMMA4_TURN.endswith("<channel|>"))
        for name in ("gemma2", "gemma3", "qwen38", "qwen38-gguf", "deepseek"):
            self.assertEqual(k1.FAMILIES[name].prefix, "")

    def test_score_rows_skip_the_prefix(self):
        rows = [None] + [{str(10 + i): {"logprob": -0.5 * i}} for i in range(1, 6)]
        ids, scores = k1.score_rows(rows, 1)
        self.assertEqual(ids, [11, 12, 13, 14, 15])
        ids, scores = k1.score_rows(rows, 3)
        self.assertEqual(ids, [13, 14, 15])
        self.assertEqual(scores, [-1.5, -2.0, -2.5])
        with self.assertRaises(RuntimeError):
            k1.score_rows([None, {"1": {"logprob": 0}, "2": {"logprob": -1}}], 1)

    def test_combine_refuses_different_passages(self):
        arm = {"prompt_sha256": "a", "ids_sha256": "b"}
        with self.assertRaises(RuntimeError):
            k1.combine(k1.FAMILIES["gemma2"], pathlib.Path("/x"), arm,
                       {"prompt_sha256": "a", "ids_sha256": "c"})

    def test_compute_metrics(self):
        # 3000 tokens: overall, early (first 1024), late (last 2048)
        scores = [-1.0] * 1024 + [-2.0] * 1976
        self.assertEqual(len(scores), 3000)

        metrics = k1.compute_metrics(scores)
        self.assertEqual(metrics["tokens"], 3000)
        self.assertEqual(metrics["early_tokens"], 1024)
        self.assertEqual(metrics["late_tokens"], 2048)

        # Early: mean of 1024 entries with -1.0 => nll = 1.0, ppl = exp(1.0)
        self.assertAlmostEqual(metrics["early_nll"], 1.0)
        self.assertAlmostEqual(metrics["early_ppl"], math.exp(1.0))

        # Late: last 2048 entries: 72 entries of -1.0, 1976 entries of -2.0
        # sum = 72 * (-1.0) + 1976 * (-2.0) = -72 - 3952 = -4024
        # late_nll = 4024 / 2048 = 1.96484375
        expected_late_nll = (72 * 1.0 + 1976 * 2.0) / 2048
        self.assertAlmostEqual(metrics["late_nll"], expected_late_nll)
        self.assertAlmostEqual(metrics["late_ppl"], math.exp(expected_late_nll))

        self.assertEqual([b["tokens"] for b in metrics["buckets"]], [2048, 952])
        self.assertAlmostEqual(metrics["buckets"][1]["nll"], 2.0)

        # Overall: sum = 1024 * (-1.0) + 1976 * (-2.0) = -1024 - 3952 = -4976
        expected_overall_nll = (1024 * 1.0 + 1976 * 2.0) / 3000
        self.assertAlmostEqual(metrics["overall_nll"], expected_overall_nll)
        self.assertAlmostEqual(metrics["overall_ppl"], math.exp(expected_overall_nll))

    def test_evaluate_family_mocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "out"
            out.mkdir()
            share = pathlib.Path(tmpdir) / "share"
            share.mkdir()
            corpus = "Mock corpus text for testing long context scoring.\n" * 100

            fake_runtime = pathlib.Path(tmpdir) / "llmp-runtime"
            fake_runtime.write_text("#!/bin/sh\nexit 0\n")

            family = k1.FAMILIES["gemma2"]

            # Mock Service, completions HTTP, and chat HTTP
            with mock.patch.object(k1, "Service") as mock_service_cls, \
                 mock.patch("http.client.HTTPConnection") as mock_conn_cls:

                mock_service = mock.MagicMock()
                mock_service.wait_ready.return_value = 0.5
                mock_service.stop.return_value = {"signal": "TERM", "returncode": 0}
                mock_service_cls.return_value = mock_service

                # Build mock completions responses for f16 and q8_0
                def make_conn(*args, **kwargs):
                    mock_conn = mock.MagicMock()
                    mock_resp = mock.MagicMock()
                    mock_resp.status = 200

                    # 1500 tokens
                    logprobs = [None] + [{"7": {"logprob": -1.5}} for _ in range(1500)]
                    resp_data = json.dumps({"choices": [{"prompt_logprobs": logprobs}],
                                            "usage": {"prompt_tokens": 1501}}).encode()
                    mock_resp.read.return_value = resp_data
                    mock_conn.getresponse.return_value = mock_resp
                    return mock_conn

                mock_conn_cls.side_effect = make_conn

                with mock.patch.object(k1, "stream_chat") as mock_chat:
                    mock_chat.return_value = {
                        "error": None,
                        "reply_sha256": "fake_reply_digest",
                        "usage": {"completion_tokens": 42},
                        "chunks": 42,
                    }

                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        record = k1.evaluate_family(
                            family=family,
                            runtime=fake_runtime,
                            share=share,
                            corpus=corpus,
                            out=out,
                            port=18114,
                            runtime_env={},
                            timeout=10.0,
                            keep_data=False,
                        )

                    self.assertIsNone(record["error"])
                    self.assertEqual(record["tokens"], 1500)
                    self.assertEqual(record["prefix_tokens"], 1)
                    self.assertEqual(record["delta"]["max_abs_logprob_diff"], 0.0)
                    self.assertTrue((out / "gemma2-k1-f16.json").exists())
                    self.assertTrue((out / "gemma2-k1-q8_0.json").exists())
                    self.assertAlmostEqual(record["f16"]["overall_nll"], 1.5)
                    self.assertAlmostEqual(record["q8_0"]["overall_nll"], 1.5)
                    self.assertAlmostEqual(record["delta"]["overall_ppl_change_pct"], 0.0)

                    out_file = out / "gemma2-k1.json"
                    self.assertTrue(out_file.exists())
                    saved = json.loads(out_file.read_text())
                    self.assertEqual(saved["family"], "gemma2")
                    self.assertEqual(saved["tokens"], 1500)

                    output = stdout.getvalue()
                    self.assertIn('"family": "gemma2"', output)
                    self.assertIn("| gemma2 | 1500 |", output)

                    # The q8_0 arm alone reuses the saved f16 arm's passage.
                    (out / "gemma2-k1-q8_0.json").unlink()
                    with contextlib.redirect_stdout(io.StringIO()):
                        again = k1.evaluate_family(
                            family=family, runtime=fake_runtime, share=share, corpus=corpus,
                            out=out, port=18114, runtime_env={}, timeout=10.0,
                            keep_data=False, arms=("q8_0",))
                    self.assertIsNone(again["error"])
                    self.assertEqual(again["prompt_sha256"], record["prompt_sha256"])


if __name__ == "__main__":
    unittest.main()
