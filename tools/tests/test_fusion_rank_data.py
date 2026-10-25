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
        self.recipe = {
            "key": {
                "version": 1, "region_canonicalizer": 2,
                "discovery_operation": "rms_norm", "mode": "full",
                "subject": "11" * 32, "boundary": "22" * 32,
                "policy": "33" * 32, "device": "44" * 32,
                "selection_environment": "55" * 32, "preparation_domain": "66" * 32,
                "incumbent": {
                    "planned_region": "77" * 32, "physical_recipe": "88" * 32,
                    "backend_recipe": "99" * 32, "backend_build": "aa" * 32,
                    "backend_selection": "bb" * 32,
                },
                "alternative": {
                    "planned_region": "cc" * 32, "physical_recipe": "dd" * 32,
                    "backend_recipe": "ee" * 32, "backend_build": "ff" * 32,
                    "backend_selection": "10" * 32,
                },
            },
            "incumbent_ns": 100, "alternative_ns": 80, "samples": 7,
            "protocol": "12" * 32, "receipt": "34" * 32,
            "provenance": "synthetic protocol receipt",
            "shape_class": "synthetic region",
        }

    def compile(self, row=None):
        return ranks.compile_data({"version": 1, "comparisons": [row or self.row]})

    def compile_v2(self, recipes=None, rows=None):
        return ranks.compile_data({"version": 2, "comparisons": [rows or self.row],
                                   "recipe_comparisons": [recipes or self.recipe]})

    def test_all_semantic_fields_are_compiled(self):
        output = self.compile()
        for text in ("test.fused", "test.primitive", "synthetic test fixture", "100ULL", "200ULL", "9U"):
            self.assertIn(text, output)
        self.assertIn("return {};", ranks.compile_data({"version": 1, "comparisons": []}))

    def test_v2_emits_typed_recipe_rows_and_v1_remains_readable(self):
        output = self.compile_v2()
        for text in ("MeasuredFusionComparisons", "MeasuredRecipeComparisons",
                     "ir::Primitive(ir::Op::kRmsNorm)", "RecipeRunMode::kFull",
                     "incumbent_ns = 100ULL", "alternative_ns = 80ULL",
                     'shape_class = "synthetic region"'):
            self.assertIn(text, output)
        self.assertIn("{.key = {", output)
        self.assertIn(".shape_class = \"synthetic region\"}", output)
        self.assertNotIn(",,", output)
        legacy = ranks.compile_data({"version": 1, "comparisons": []})
        self.assertIn("return {};", legacy.split("MeasuredRecipeComparisons", 1)[1])

    def test_quality_disposition_has_no_latency_and_keeps_exact_key_schema(self):
        row = copy.deepcopy(self.recipe)
        row.update(disposition="reject_incumbent_quality", incumbent_ns=0,
                   alternative_ns=0, samples=0)
        output = self.compile_v2(recipes=row)
        self.assertIn("RecipeComparisonKind::kRejectIncumbentQuality", output)
        self.assertIn("incumbent_ns = 0ULL", output)
        for field, value in (("disposition", "unknown"), ("incumbent_ns", 1),
                             ("alternative_ns", 1), ("samples", 1),
                             ("samples", False), ("receipt", "0" * 64)):
            bad = copy.deepcopy(row)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compile_v2(recipes=bad)

    def test_recipe_schema_rejects_unknown_enums_fields_and_invalid_values(self):
        for field, value in (("extra", 0), ("incumbent_ns", True),
                             ("incumbent_ns", 0), ("alternative_ns", 2**64),
                             ("samples", False), ("samples", 2**32),
                             ("protocol", "0" * 64), ("receipt", "A" * 64),
                             ("provenance", "UNREVIEWED"), ("shape_class", "control\x01"),
                             ("shape_class", "control\x7f")):
            with self.subTest(field=field, value=value):
                row = copy.deepcopy(self.recipe)
                row[field] = value
                with self.assertRaises(ValueError):
                    self.compile_v2(recipes=row)
        for field, value in (("version", True), ("version", 2),
                           ("region_canonicalizer", 0), ("region_canonicalizer", 1),
                             ("discovery_operation", "unknown.operation"),
                             ("discovery_operation", "fused"),
                             ("discovery_operation", "fused.none"),
                             ("mode", "warm-ish"), ("subject", "0" * 64)):
            with self.subTest(key_field=field, value=value):
                row = copy.deepcopy(self.recipe)
                row["key"][field] = value
                with self.assertRaises(ValueError):
                    self.compile_v2(recipes=row)

    def test_recipe_duplicate_complete_keys_fail_even_if_evidence_differs(self):
        row = copy.deepcopy(self.recipe)
        changed = copy.deepcopy(row)
        changed["provenance"] = "different receipt text"
        with self.assertRaises(ValueError):
            ranks.compile_data({"version": 2, "comparisons": [],
                                "recipe_comparisons": [row, changed]})
        identical_arms = copy.deepcopy(self.recipe)
        identical_arms["key"]["alternative"] = copy.deepcopy(identical_arms["key"]["incumbent"])
        with self.assertRaises(ValueError):
            self.compile_v2(recipes=identical_arms)
        with self.assertRaises(ValueError):
            ranks.compile_data({"version": 2, "comparisons": []})
        with self.assertRaises(ValueError):
            ranks.compile_data({"version": 2, "comparisons": [],
                                "recipe_comparisons": [], "extra": 1})

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
