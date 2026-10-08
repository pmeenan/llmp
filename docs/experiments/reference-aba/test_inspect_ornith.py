#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Closure-accounting regressions using synthetic GGUF reader records."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "reference_ornith_inspector", Path(__file__).with_name("inspect_ornith.py"))
inspector = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"gguf": SimpleNamespace(GGUFReader=None)}):
    SPEC.loader.exec_module(inspector)


def tensor(name, byte_count, shape):
    return SimpleNamespace(name=name, n_bytes=byte_count, shape=shape,
                           tensor_type=SimpleNamespace(name="Q4_K"), n_elements=byte_count * 2)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        metadata = {"general.architecture": "qwen35moe", "qwen35moe.block_count": 41,
                    "qwen35moe.expert_count": 256, "qwen35moe.nextn_predict_layers": 1}
        fields = {name: SimpleNamespace(contents=lambda value=value: value)
                  for name, value in metadata.items()}
        tensors = []
        for layer in range(41):
            for projection in ("gate", "up", "down"):
                tensors.append(tensor(f"blk.{layer}.ffn_{projection}_exps.weight", 4096, [2, 4, 256]))
                tensors.append(tensor(f"blk.{layer}.ffn_{projection}_shexp.weight", 16, [2, 4]))
            tensors.append(tensor(f"blk.{layer}.ffn_gate_inp_shexp.weight", 16, [4]))
        self.reader = SimpleNamespace(fields=fields, tensors=tensors)

    def summarize(self):
        with mock.patch.object(inspector, "GGUFReader", return_value=self.reader):
            return inspector.summarize("unused")

    def test_primary_mtp_routed_and_shared_accounting_are_disjoint(self):
        result = self.summarize()
        self.assertEqual(result["routed_expert_payload_bytes"], 41 * 3 * 4096)
        self.assertEqual(result["primary_routed_expert_bytes"], 40 * 3 * 4096)
        self.assertEqual(result["shared_projection_bytes"], 41 * 3 * 16)
        self.assertEqual(result["non_routed_tensor_bytes"], 41 * 4 * 16)
        self.assertEqual(result["mtp_tensor_payload_bytes"], 3 * 4096 + 4 * 16)
        self.assertEqual(result["primary_tensor_payload_bytes"], 40 * (3 * 4096 + 4 * 16))
        self.assertEqual(set(result["expert_bytes_per_layer"].values()), {48})

    def test_missing_routed_or_shared_projection_is_rejected(self):
        original = list(self.reader.tensors)
        for name in ("blk.3.ffn_down_exps.weight", "blk.3.ffn_up_shexp.weight"):
            with self.subTest(name=name):
                self.reader.tensors = [t for t in original if t.name != name]
                with self.assertRaisesRegex(ValueError, "Incomplete expert closure"):
                    self.summarize()

    def test_unexpected_routed_projection_is_rejected(self):
        self.reader.tensors.append(tensor("blk.0.ffn_down_exps.scale", 1024, [256]))
        with self.assertRaisesRegex(ValueError, "Unaccounted expert tensor"):
            self.summarize()

    def test_expert_axis_and_divisibility_are_checked(self):
        target = self.reader.tensors[0]
        for byte_count, shape in ((4096, [256, 2, 4]), (4097, [2, 4, 256])):
            with self.subTest(byte_count=byte_count, shape=shape):
                target.n_bytes, target.shape = byte_count, shape
                with self.assertRaisesRegex(ValueError, "Unexpected expert tensor axis"):
                    self.summarize()

    def test_unexpected_primary_mtp_layout_is_rejected(self):
        self.reader.fields["qwen35moe.nextn_predict_layers"] = SimpleNamespace(contents=lambda: 0)
        with self.assertRaisesRegex(ValueError, "Unexpected primary/MTP/expert layout"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
