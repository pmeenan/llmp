# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

from capture import HERE, load_quality_pair, verify_score

PINS = json.loads((HERE / "llm-capture-pins.json").read_text())
QP = load_quality_pair(HERE.parents[2], PINS["quality_pair_sha256"])
PROMPT = b"Frozen score passage.\n" * 200


def response(count):
    return {"choices": [{"prompt_logprobs": [None] + [
        {str(i): {"logprob": -0.125}} for i in range(1, count + 1)],
        "logprobs": {"tokens": [f"token_id:{i}" for i in range(count + 1)]}}],
        "usage": {"prompt_tokens": count + 1, "completion_tokens": 0}}


class ScoreCaptureTest(unittest.TestCase):
    def validate(self, payload, family="qwen38", prompt=PROMPT, alter_receipt=None):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / f"{family}-quality-old.json"
            prompt_name, response_name = path.stem + "-score.txt", path.stem + "-score.json"
            raw = json.dumps(payload).encode()
            (path.parent / prompt_name).write_bytes(prompt)
            (path.parent / response_name).write_bytes(raw)
            receipt = {"error": None,
                       "score_prompt": {"file": prompt_name,
                                        "sha256": hashlib.sha256(prompt).hexdigest()},
                       "score_response": {"file": response_name,
                                          "sha256": hashlib.sha256(raw).hexdigest()},
                       "score": {"tokens": 966 if family == "deepseek" else 996}}
            if alter_receipt:
                alter_receipt(receipt)
            return verify_score(path, receipt, PROMPT, family, QP)

    def test_complete_registered_targets_pass(self):
        for family, count in (("qwen38", 996), ("qwen38-gguf", 996), ("deepseek", 966)):
            with self.subTest(family=family):
                result = self.validate(response(count), family)
                self.assertEqual(result["targets"], count)
                self.assertEqual(result["ordered_target_ids_sha256"],
                                 QP.ids_digest(list(range(1, count + 1))))

    def test_partial_and_interior_null_refused(self):
        for family, count in (("qwen38", 996), ("qwen38-gguf", 996), ("deepseek", 966)):
            partial = response(count - 1)
            interior_null = response(count)
            interior_null["choices"][0]["prompt_logprobs"][32] = None
            for payload in (partial, interior_null):
                with self.subTest(family=family), self.assertRaises((ValueError, QP.Refused)):
                    self.validate(payload, family)

    def test_nonfinite_positive_and_missing_actual_target_refused(self):
        variants = []
        for value in (float("nan"), float("inf"), float("-inf"), 0.01):
            payload = response(996)
            payload["choices"][0]["prompt_logprobs"][1]["1"]["logprob"] = value
            variants.append(payload)
        for row in ({}, {"01": {"logprob": -0.1}}, {"1": {}},
                    {"2": {"logprob": -0.1}}, {"1": {"logprob": -0.1}, "2": {"logprob": -0.2}}):
            payload = response(996)
            payload["choices"][0]["prompt_logprobs"][1] = row
            variants.append(payload)
        for payload in variants:
            with self.subTest(row=payload["choices"][0]["prompt_logprobs"][1]), \
                    self.assertRaises((ValueError, QP.Refused)):
                self.validate(payload)

    def test_echo_identity_and_exact_length_required(self):
        for mode in ("wrong", "short", "long", "absent", "bad_first"):
            payload = response(996)
            echo = payload["choices"][0]["logprobs"]["tokens"]
            if mode == "wrong":
                echo[1] = "token_id:2"
            elif mode == "short":
                echo.pop()
            elif mode == "long":
                echo.append("token_id:997")
            elif mode == "absent":
                del payload["choices"][0]["logprobs"]
            else:
                echo[0] = "token_id:00"
            with self.subTest(mode=mode), self.assertRaises((ValueError, QP.Refused)):
                self.validate(payload)

    def test_single_choice_usage_and_bos_required(self):
        variants = []
        payload = response(996)
        payload["choices"].append(copy.deepcopy(payload["choices"][0]))
        variants.append(payload)
        payload = response(996)
        payload["choices"][0]["prompt_logprobs"][0] = {"0": {"logprob": -0.1}}
        variants.append(payload)
        for usage in ({}, {"prompt_tokens": 996, "completion_tokens": 0},
                      {"prompt_tokens": 997, "completion_tokens": 1}):
            payload = response(996)
            payload["usage"] = usage
            variants.append(payload)
        for payload in variants:
            with self.assertRaises((ValueError, QP.Refused)):
                self.validate(payload)

    def test_prompt_bytes_hash_response_hash_and_exact_names_required(self):
        with self.assertRaises(ValueError):
            self.validate(response(996), prompt=b"wrong passage")
        for field, key, wrong in (("score_prompt", "sha256", "wrong"),
                                  ("score_response", "sha256", "wrong"),
                                  ("score_prompt", "file", "other.txt"),
                                  ("score_response", "file", "other.json"),
                                  ("score", "tokens", 1)):
            with self.subTest(field=field, key=key), self.assertRaises((ValueError, QP.Refused)):
                self.validate(response(996), alter_receipt=lambda r: r[field].update({key: wrong}))


if __name__ == "__main__":
    unittest.main()
