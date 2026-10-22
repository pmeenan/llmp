# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Declarative fusion measurements cannot inject code or silently lose fields."""
import copy
import json
import pathlib
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import fusion_rank_data as ranks  # noqa: E402


class FusionRankData(unittest.TestCase):
    def setUp(self):
        # Synthetic syntax fixtures only; no actual backend measurements.
        self.row = {
            "shape_class": "synthetic", "query": "12" * 32,
            "selection_environment": "78" * 32,
            "fusion": {"name": "test.fused", "identity": "34" * 32},
            "decomposition": [{"name": "test.primitive", "identity": "56" * 32}],
            "fused_ns": 100, "decomposition_ns": 200, "samples": 9,
            "provenance": "synthetic test fixture",
        }

    def compile(self, row=None):
        return ranks.compile_data({"version": 1, "comparisons": [row or self.row]})

    def test_all_semantic_fields_are_compiled(self):
        output = self.compile()
        for text in ("test.fused", "test.primitive", "synthetic test fixture", "100ULL", "200ULL", "9U"):
            self.assertIn(text, output)
        self.assertIn("return {};", ranks.compile_data({"version": 1, "comparisons": []}))

    def test_unknown_fields_invalid_numbers_and_unreviewed_rows_fail(self):
        for key, value in (("extra", 0), ("fused_ns", 0), ("samples", True),
                           ("samples", 2**32), ("query", "0" * 64),
                           ("decomposition", []), ("provenance", "UNREVIEWED")):
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(self.row)
                row[key] = value
                with self.assertRaises(ValueError):
                    self.compile(row)

    def test_duplicate_keys_and_comparisons_fail(self):
        with self.assertRaises(ValueError):
            json.loads('{"version":1,"version":2}', object_pairs_hook=ranks.unique_object)
        with self.assertRaises(ValueError):
            ranks.compile_data({"version": 1, "comparisons": [self.row, self.row]})

    def test_text_is_escaped_and_never_inserted_as_source(self):
        row = copy.deepcopy(self.row)
        row["provenance"] = 'quote"\\\n}; injected(); //'
        output = self.compile(row)
        self.assertIn(json.dumps(row["provenance"]), output)
        self.assertNotIn('\n}; injected(); //', output)
        row["fusion"]["name"] = 'x"; injected();'
        with self.assertRaises(ValueError):
            self.compile(row)

    def test_fusion_cannot_compare_against_itself(self):
        self.row["decomposition"] = [self.row["fusion"]]
        with self.assertRaises(ValueError):
            self.compile()

    def test_version_is_integer_and_text_is_valid_cpp(self):
        for version in (True, 1.0, "1", 2):
            with self.subTest(version=version), self.assertRaises(ValueError):
                ranks.compile_data({"version": version, "comparisons": []})
        for value in ("nul\0suffix", "control\x01", "escape\x1b"):
            row = copy.deepcopy(self.row)
            row["provenance"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compile(row)


if __name__ == "__main__":
    unittest.main()
