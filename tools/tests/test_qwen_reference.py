# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Saved native rows bind exact reference inputs without another model call."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwen_reference", ROOT / "docs/experiments/m36-numerical-audit/collect-qwen-reference.py")
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


class NativeQualityInputs(unittest.TestCase):
    def collect(self, mutate=lambda body: None):
        with tempfile.TemporaryDirectory() as scratch:
            parent = Path(scratch)
            body = {"choices": [{"logprobs": {"tokens": ["token_id:11", "token_id:22", "token_id:33"],
                                                "token_logprobs": [None, -1.0, -2.0]},
                                  "prompt_logprobs": [None, {"22": {"logprob": -1.0}},
                                                      {"33": {"logprob": -2.0}}]}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 0}}
            mutate(body)
            record = {"family": "qwen38", "error": None,
                      "score": {"tokens": 2, "nll": 1.5,
                                "scores_sha256": hashlib.sha256(b"[-1.0, -2.0]").hexdigest()}}
            for key, data in (("score_prompt", b"example"),
                              ("score_response", json.dumps(body).encode())):
                (parent / key).write_bytes(data)
                record[key] = {"file": key, "sha256": hashlib.sha256(data).hexdigest()}
            (parent / "quality.json").write_text(json.dumps(record))
            COLLECT.native_inputs(parent / "quality.json", parent)
            return json.loads((parent / "inputs.json").read_text())

    def test_keeps_unscored_first_token_and_all_target_coordinates(self):
        result = self.collect()
        self.assertEqual(result["documents"][0]["tokens"], [11, 22, 33])
        self.assertTrue(result["complete_tokenization"])
        self.assertTrue(result["add_special_tokens"])

    def test_rejects_missing_first_id(self):
        with self.assertRaises(ValueError):
            self.collect(lambda b: b["choices"][0]["logprobs"]["tokens"].pop(0))

    def test_rejects_shifted_target(self):
        with self.assertRaises(ValueError):
            self.collect(lambda b: b["choices"][0]["logprobs"]["tokens"].__setitem__(1, "token_id:23"))

    def test_rejects_nonfinite_and_positive_loss(self):
        for value in (float("nan"), float("inf"), 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.collect(lambda b: b["choices"][0]["prompt_logprobs"][1]["22"].update(logprob=value))

    def test_rejects_echo_coordinate_disagreement(self):
        with self.assertRaises(ValueError):
            self.collect(lambda b: b["choices"][0]["logprobs"]["token_logprobs"].__setitem__(1, -2.0))


if __name__ == "__main__":
    unittest.main()
