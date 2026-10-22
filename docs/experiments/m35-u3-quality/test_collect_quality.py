#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent response-contract controls; no model or server required."""
import unittest
from collect_quality import extract


class ActualTargetContract(unittest.TestCase):
    def fixture(self, count):
        ids = [128000] + [17] * (count - 1)
        scores = [None] + [{"42": {"logprob": -0.1}, "17": {"logprob": -3.0}}
                           for _ in range(count - 1)]
        return ids, {"choices": [{"prompt_logprobs": scores}],
                     "usage": {"prompt_tokens": count}}

    def test_actual_target_not_top_one(self):
        ids, response = self.fixture(1024)
        rows = extract(response, ids, "short")
        self.assertEqual(len(rows), 1023)
        self.assertEqual(rows[0], {"document": 0, "position": 1, "target": 17, "nll": 3.0})

    def test_depth_uses_registered_coordinates(self):
        ids, response = self.fixture(9216)
        rows = extract(response, ids, "depth")
        self.assertEqual(len(rows), 1024)
        self.assertEqual((rows[0]["position"], rows[-1]["position"]), (8192, 9215))
        self.assertEqual(rows[0]["document"], 1)

    def test_missing_actual_target_refused(self):
        ids, response = self.fixture(1024)
        del response["choices"][0]["prompt_logprobs"][1]["17"]
        with self.assertRaises(ValueError):
            extract(response, ids, "short")

    def test_nonfinite_and_positive_refused(self):
        for value in [float("nan"), float("inf"), 0.1, None, True]:
            ids, response = self.fixture(1024)
            response["choices"][0]["prompt_logprobs"][1]["17"]["logprob"] = value
            with self.assertRaises(ValueError):
                extract(response, ids, "short")

    def test_prompt_count_or_ids_refused(self):
        ids, response = self.fixture(1024)
        response["usage"]["prompt_tokens"] = 1023
        with self.assertRaises(ValueError):
            extract(response, ids, "short")
        response["usage"]["prompt_tokens"] = 1024
        response["choices"][0]["prompt_token_ids"] = [0] * 1024
        with self.assertRaises(ValueError):
            extract(response, ids, "short")


if __name__ == "__main__":
    unittest.main()
