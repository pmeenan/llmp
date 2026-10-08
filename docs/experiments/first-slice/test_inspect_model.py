# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import inspect_model

METADATA = {"qwen2.block_count": 24, "qwen2.embedding_length": 896,
            "qwen2.feed_forward_length": 4864, "qwen2.attention.head_count": 14,
            "qwen2.attention.head_count_kv": 2, "qwen2.rope.freq_base": 1000000.0,
            "qwen2.attention.layer_norm_rms_epsilon": float(np.float32(1e-06)),
            "qwen2.context_length": 8192}
CONFIG = {"num_hidden_layers": 24, "hidden_size": 896, "intermediate_size": 4864,
          "num_attention_heads": 14, "num_key_value_heads": 2, "rope_theta": 1000000.0,
          "rms_norm_eps": 1e-06, "vocab_size": 151936, "max_position_embeddings": 32768,
          "tie_word_embeddings": True}


class BaseConfigCrossCheck(unittest.TestCase):
    def test_matching_shapes_and_reported_context_difference(self):
        result = inspect_model.base_config_cross_check(METADATA, CONFIG, 151936)
        self.assertTrue(result["all_shapes_match"])
        self.assertEqual(result["context"], {"base_max_position_embeddings": 32768,
                                             "gguf_context_length": 8192, "match": False})

    def test_any_shape_difference_is_reported(self):
        for key, gguf_key in inspect_model.SHAPE_KEYS.items():
            with self.subTest(key=key):
                metadata = {**METADATA, gguf_key: METADATA[gguf_key] + 1}
                result = inspect_model.base_config_cross_check(metadata, CONFIG, 151936)
                self.assertFalse(result["shapes"][key]["match"])
                self.assertFalse(result["all_shapes_match"])
        self.assertFalse(inspect_model.base_config_cross_check(METADATA, CONFIG, 151935)["all_shapes_match"])
        metadata = {**METADATA, "qwen2.attention.layer_norm_rms_epsilon": 1e-05}
        self.assertFalse(inspect_model.base_config_cross_check(metadata, CONFIG, 151936)["all_shapes_match"])

    def test_base_files_must_match_pins(self):
        data = json.dumps(CONFIG).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                inspect_model.read_pinned(path, "config.json")
            pinned = {"config.json": (len(data), hashlib.sha256(data).hexdigest())}
            with patch.dict(inspect_model.BASE_FILES, pinned):
                self.assertEqual(inspect_model.read_pinned(path, "config.json"), CONFIG)
                path.write_bytes(data + b" ")
                with self.assertRaises(ValueError):
                    inspect_model.read_pinned(path, "config.json")


if __name__ == "__main__":
    unittest.main()
