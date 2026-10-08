#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Inspect only the hash-verified first-slice GGUF; not an untrusted importer."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_SHA = "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc"
MODEL_SIZE = 1266425696
# Pinned base revision 7ae557604adf67be50417f59c2c2f167def9a775 (pins.json).
BASE_FILES = {
    "tokenizer_config.json": (7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "config.json": (659, "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"),
}
# Base config key -> GGUF metadata key that must hold the same value.
SHAPE_KEYS = {
    "num_hidden_layers": "qwen2.block_count",
    "hidden_size": "qwen2.embedding_length",
    "intermediate_size": "qwen2.feed_forward_length",
    "num_attention_heads": "qwen2.attention.head_count",
    "num_key_value_heads": "qwen2.attention.head_count_kv",
    "rope_theta": "qwen2.rope.freq_base",
}


def read_pinned(path, name):
    data = path.read_bytes()
    size, digest = BASE_FILES[name]
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected base {name} identity")
    return json.loads(data)


def base_config_cross_check(metadata, config, vocabulary_entries):
    """Compare the GGUF's declared shape with the pinned base config; report, never repair."""
    shapes = {key: {"base": config[key], "gguf": metadata[gguf_key],
                    "match": config[key] == metadata[gguf_key]}
              for key, gguf_key in SHAPE_KEYS.items()}
    # GGUF stores epsilon as float32; compare in that precision.
    eps = float(np.float32(config["rms_norm_eps"]))
    shapes["rms_norm_eps"] = {"base": config["rms_norm_eps"],
                              "gguf": metadata["qwen2.attention.layer_norm_rms_epsilon"],
                              "match": eps == metadata["qwen2.attention.layer_norm_rms_epsilon"]}
    shapes["vocab_size"] = {"base": config["vocab_size"], "gguf": vocabulary_entries,
                            "match": config["vocab_size"] == vocabulary_entries}
    return {"source_sha256": BASE_FILES["config.json"][1], "shapes": shapes,
            "all_shapes_match": all(row["match"] for row in shapes.values()),
            "context": {"base_max_position_embeddings": config["max_position_embeddings"],
                        "gguf_context_length": metadata["qwen2.context_length"],
                        "match": config["max_position_embeddings"] == metadata["qwen2.context_length"]},
            "base_tie_word_embeddings": config["tie_word_embeddings"]}


def summarize(path, tokenizer_config, base_config):
    from gguf import GGUFReader

    base = read_pinned(tokenizer_config, "tokenizer_config.json")
    config = read_pinned(base_config, "config.json")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != MODEL_SIZE or digest != MODEL_SHA:
        raise ValueError("Unexpected checkpoint identity")
    reader = GGUFReader(path, mode="r")
    metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if name.startswith(("general.", "qwen2."))
    }
    if metadata.get("general.architecture") != "qwen2":
        raise ValueError("Unexpected architecture")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("Duplicate tensor name")
    for tensor in reader.tensors:
        if tensor.tensor_type.name not in {"F16", "F32"}:
            raise ValueError("Unexpected quantized tensor")
        if tensor.data_offset < reader.data_offset or tensor.data_offset + tensor.n_bytes > MODEL_SIZE:
            raise ValueError("Invalid tensor range")
        if not np.isfinite(tensor.data).all():
            raise ValueError("Nonfinite weights")
    embedding, output = tensors["token_embd.weight"], tensors["output.weight"]
    tied_equal = (
        embedding.tensor_type == output.tensor_type
        and np.array_equal(embedding.shape, output.shape)
        and embedding.data.tobytes() == output.data.tobytes()
    )
    template = reader.fields["tokenizer.chat_template"].contents()
    vocabulary_entries = len(reader.fields["tokenizer.ggml.tokens"].data)
    token_metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if name.startswith("tokenizer.")
        and name not in {"tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                         "tokenizer.ggml.token_type", "tokenizer.chat_template"}
    }
    return {
        "model_sha256": digest,
        "metadata": metadata,
        "tensor_count": len(tensors),
        "tensor_types": dict(Counter(t.tensor_type.name for t in reader.tensors)),
        "stored_elements": sum(t.n_elements for t in reader.tensors),
        "tensor_payload_bytes": sum(t.n_bytes for t in reader.tensors),
        "embedding_bytes": embedding.n_bytes,
        "output_embedding_byte_identical": tied_equal,
        "unique_elements_if_tied": sum(t.n_elements for t in reader.tensors) -
        (output.n_elements if tied_equal else 0),
        "tokenizer": token_metadata,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "chat_template_matches_pinned_base": template == base["chat_template"],
        "vocabulary_entries": vocabulary_entries,
        "merge_entries": len(reader.fields["tokenizer.ggml.merges"].data),
        "base_config_cross_check": base_config_cross_check(metadata, config, vocabulary_entries),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer_config", type=Path)
    parser.add_argument("base_config", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.model, args.tokenizer_config, args.base_config), indent=2))
