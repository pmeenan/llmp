# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Approved Gemma2 preflight controls; no checkpoint payload required."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

import import_m3
import test_layout


class Gemma2ImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.contract = json.loads((import_m3.HERE.parents[2] /
                                    "tests/unit/data/gemma2/gemma2_2b_q8.json").read_text())
        identity = self.contract["source"]
        self.path = Path(self.tmp.name) / identity["file"]
        # Sparse logical length: no weight bytes or large allocation.
        with self.path.open("wb") as file:
            file.truncate(identity["full_bytes"])
        self.expected = {self.path.name: identity["full_sha256"]}
        meta = dict(self.contract["metadata"])
        meta["tokenizer.ggml.tokens"] = {"array_len": meta.pop("vocab_count")}
        tensors = [dict(name=t["name"], family="ggml", dtype=t["type"], ne=t["ne"],
                        nbytes=t["bytes"], offset=self.contract["data_offset"] + t["relative_offset"],
                        path=str(self.path)) for t in self.contract["tensors"]]
        part = dict(kind="gguf", path=str(self.path),
                    header_len=self.contract["complete_header_bytes"],
                    header_sha256=self.contract["complete_header_sha256"])
        self.src = dict(meta=meta, tensors=tensors, parts=[part])

    def check(self):
        import_m3.check_gemma2_source(self.src, self.expected)

    def test_approved_contract_and_recipe(self):
        self.check()
        pins = import_m3.HERE.parent / "gemma2-execution/pins.json"
        self.assertEqual(import_m3.pinned_sources(pins, "gemma2-2b-q8_0", [self.path]),
                         self.expected)
        # The source table has separate V and a tied head (no output tensor).
        names = {t["name"] for t in self.src["tensors"]}
        self.assertIn("blk.0.attn_v.weight", names)
        self.assertNotIn("output.weight", names)

    def test_every_profile_metadata_field_and_type_is_checked(self):
        original = self.src["meta"]
        for key in self.contract["metadata"]:
            if key == "vocab_count":
                continue
            for value in (None, True, str(original[key]) + "changed"):
                with self.subTest(key=key, value=value):
                    self.src["meta"] = dict(original, **{key: value})
                    with self.assertRaisesRegex(ValueError, "metadata differs"):
                        self.check()
        for tokens in (None, [], {"array_len": 8}, {"array_len": True},
                       {"array_len": "256000"}, {"array_len": 256000, "extra": 0}):
            self.src["meta"] = dict(original)
            self.src["meta"]["tokenizer.ggml.tokens"] = tokens
            with self.assertRaisesRegex(ValueError, "vocab_count"):
                self.check()

    def test_large_vocab_uses_the_actual_parser_compact_descriptor(self):
        key = b"tokenizer.ggml.tokens"
        count = self.contract["metadata"]["vocab_count"]
        data = (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) +
                struct.pack("<Q", len(key)) + key + struct.pack("<IIQ", 9, 8, count) +
                bytes(count * 8))  # 256000 empty strings, not materialized by the parser
        path = Path(self.tmp.name) / "vocab.gguf"
        path.write_bytes(data)
        parsed = import_m3.load_layout().read_gguf(path)
        self.assertEqual(parsed["meta"][key.decode()], {"array_len": count})
        self.src["meta"][key.decode()] = parsed["meta"][key.decode()]
        self.check()

    def test_source_pin_length_header_and_shard_refusals(self):
        original = copy.deepcopy(self.src)
        for change in (lambda: self.expected.update({self.path.name: "f" * 64}),
                       lambda: self.src["parts"].append(dict(self.src["parts"][0])),
                       lambda: self.src["parts"][0].update(header_sha256="f" * 64),
                       lambda: self.src["parts"][0].update(header_len=24),
                       lambda: self.src["parts"][0].update(kind="safetensors")):
            self.src = copy.deepcopy(original)
            self.expected = {self.path.name: self.contract["source"]["full_sha256"]}
            change()
            with self.assertRaises(ValueError):
                self.check()
        self.src = original
        with self.path.open("r+b") as file:
            file.truncate(self.contract["source"]["full_bytes"] - 1)
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.check()

    def test_tensor_shape_type_bounds_identity_and_tied_head(self):
        original = copy.deepcopy(self.src["tensors"])
        for field, value in (("family", "plain"), ("dtype", "F16"), ("ne", [1]),
                             ("nbytes", 1), ("offset", 0), ("path", "/other"),
                             ("name", "output.weight")):
            with self.subTest(field=field):
                self.src["tensors"] = copy.deepcopy(original)
                self.src["tensors"][0][field] = value
                with self.assertRaisesRegex(ValueError, "tensor differs"):
                    self.check()
        self.src["tensors"] = copy.deepcopy(original)
        self.src["tensors"][1] = dict(self.src["tensors"][0])
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            self.check()
        self.src["tensors"] = original[:-1]
        with self.assertRaisesRegex(ValueError, "tensor count"):
            self.check()

    def test_generic_build_rejects_unapproved_gemma2_before_output(self):
        data = test_layout.gguf_bytes({"general.architecture": (8, "gemma2")},
                                     [("token_embd.weight", [8, 4], 0, bytes(128))])
        self.path.write_bytes(data)
        pins = Path(self.tmp.name) / "pins.json"
        pins.write_text(json.dumps({"models": [{"id": "edited", "files": [
            {"path": self.path.name, "bytes": len(data), "sha256": "f" * 64}]}]}))
        out = Path(self.tmp.name) / "store"
        with self.assertRaisesRegex(ValueError, "approved Q8_0 source pin"):
            import_m3.main(["import_m3.py", "build", str(out), str(pins), "edited", str(self.path)])
        self.assertFalse(out.exists())

    def test_other_architectures_keep_the_generic_path(self):
        import_m3.check_gemma2_source(dict(meta={"general.architecture": "llama"}), {})


if __name__ == "__main__":
    unittest.main()
