# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest

from compare_routes import compare_outputs, compare_records, compare_routes, read_lines, read_routes


def route(step, layer, rows, request=0, phase="decode", output_only=False):
    return {"event": "routes", "request": request, "step": step, "phase": phase,
            "layer": layer, "output_only": output_only, "routes": rows}


class Comparisons(unittest.TestCase):
    def arm(self, root, name, predictions, hashes):
        directory = Path(root) / name
        directory.mkdir()
        (directory / "predictions").write_text("".join(f"{r} {t}\n" for r, t in predictions))
        (directory / "logits").write_text("".join(f"{r} {s} {h}\n" for r, s, h in hashes))
        return directory

    def test_identical_arms_and_counted_differences(self):
        with tempfile.TemporaryDirectory() as root:
            a = self.arm(root, "a", [(0, 5), (0, 7)], [(0, 3, "aa"), (0, 4, "bb")])
            b = self.arm(root, "b", [(0, 5), (0, 7)], [(0, 3, "aa"), (0, 4, "bb")])
            c = self.arm(root, "c", [(0, 5), (0, 8)], [(0, 3, "aa"), (0, 4, "cc")])
            same = compare_outputs(a, b)
            self.assertEqual(same["predictions"]["different_records"], 0)
            self.assertEqual(same["logit_hashes"]["different_records"], 0)
            changed = compare_outputs(a, c)
            self.assertEqual(changed["predictions"]["different_records"], 1)
            self.assertEqual(changed["logit_hashes"]["first_different_record"], 1)

    def test_same_prediction_with_different_logits_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            a = self.arm(root, "a", [(0, 5)], [(0, 3, "aa")])
            b = self.arm(root, "b", [(0, 5)], [(0, 3, "ab")])
            result = compare_outputs(a, b)
            self.assertEqual(result["predictions"]["different_records"], 0)
            self.assertEqual(result["logit_hashes"]["different_records"], 1)

    def test_misaligned_or_malformed_records_fail(self):
        with self.assertRaises(ValueError):
            compare_records([("0", "1")], [("0", "1"), ("0", "2")])
        with tempfile.TemporaryDirectory() as root:
            a = self.arm(root, "a", [(0, 5)], [(0, 3, "aa")])
            b = self.arm(root, "b", [(0, 5)], [(0, 4, "aa")])
            with self.assertRaises(ValueError):
                compare_outputs(a, b)
            (Path(root) / "bad").write_text("0 1 2\n")
            with self.assertRaises(ValueError):
                read_lines(Path(root) / "bad", 2)
            (Path(root) / "empty").write_text("")
            with self.assertRaises(ValueError):
                read_lines(Path(root) / "empty", 2)

    def test_route_rows_sets_and_identity(self):
        left = [route(0, 0, [[1, 2], [3, 4]]), route(0, 1, [[5, 6]])]
        self.assertEqual(compare_routes(left, left)["different_ordered_rows"], 0)
        reordered = [route(0, 0, [[2, 1], [3, 4]]), route(0, 1, [[5, 6]])]
        result = compare_routes(left, reordered)
        self.assertEqual((result["different_ordered_rows"], result["different_expert_sets"]), (1, 0))
        changed = [route(0, 0, [[1, 7], [3, 4]]), route(0, 1, [[5, 6]])]
        self.assertEqual(compare_routes(left, changed)["different_expert_sets"], 1)
        for other in ([route(0, 0, [[1, 2], [3, 4]])],
                      [route(0, 0, [[1, 2], [3, 4]]), route(0, 2, [[5, 6]])],
                      [route(0, 0, [[1, 2]]), route(0, 1, [[5, 6]])]):
            with self.subTest(other=other), self.assertRaises(ValueError):
                compare_routes(left, other)

    def test_read_routes_skips_other_events_and_rejects_invalid_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            path.write_text(json.dumps({"event": "step"}) + "\n" + json.dumps(route(0, 0, [[1, 2]])) + "\n")
            self.assertEqual(len(read_routes(path)), 1)
            path.write_text(json.dumps(route(0, 0, [[1, 1]])) + "\n")
            with self.assertRaises(ValueError):
                read_routes(path)
            path.write_text(json.dumps({"event": "step"}) + "\n")
            with self.assertRaises(ValueError):
                read_routes(path)


if __name__ == "__main__":
    unittest.main()
