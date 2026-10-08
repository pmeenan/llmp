#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Checks the compiled-in DeepSeek V4 Flash profile (src/model/dsv4.cc,
Dsv4Flash) against a GGUF's deepseek4.* key/values, read with the pinned M0
prototype's bounded header reader (headers only, no payload).

  python3 gguf_profile_check.py SHARD_1_OF_N.gguf

Prints each key, the GGUF's value and the profile's; exits 1 on a mismatch.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "artifact-layout"))
import import_m3  # noqa: E402

LAYERS = 43
# The profile, as src/model/dsv4.cc writes it. Floats are compared as float32.
PROFILE = {
    "deepseek4.block_count": LAYERS,
    "deepseek4.embedding_length": 4096,
    "deepseek4.attention.head_count": 64,
    "deepseek4.attention.head_count_kv": 1,
    "deepseek4.attention.key_length": 512,
    "deepseek4.attention.value_length": 512,
    "deepseek4.rope.dimension_count": 64,
    "deepseek4.attention.q_lora_rank": 1024,
    "deepseek4.attention.output_lora_rank": 1024,
    "deepseek4.attention.output_group_count": 8,
    "deepseek4.attention.sliding_window": 128,
    "deepseek4.expert_count": 256,
    "deepseek4.expert_used_count": 6,
    "deepseek4.expert_feed_forward_length": 2048,
    "deepseek4.expert_shared_count": 1,
    "deepseek4.hash_layer_count": 3,
    "deepseek4.hyper_connection.count": 4,
    "deepseek4.hyper_connection.sinkhorn_iterations": 20,
    "deepseek4.attention.indexer.head_count": 64,
    "deepseek4.attention.indexer.key_length": 128,
    "deepseek4.attention.indexer.top_k": 512,
    "deepseek4.rope.scaling.original_context_length": 65536,
    "deepseek4.attention.layer_norm_rms_epsilon": 1e-6,
    "deepseek4.hyper_connection.epsilon": 1e-6,
    "deepseek4.rope.freq_base": 10000.0,
    "deepseek4.attention.compress_rope_freq_base": 160000.0,
    "deepseek4.rope.scaling.factor": 16.0,
    "deepseek4.rope.scaling.type": "yarn",
    "deepseek4.rope.scaling.yarn_beta_fast": 32.0,
    "deepseek4.rope.scaling.yarn_beta_slow": 1.0,
    "deepseek4.expert_weights_scale": 1.5,
    "deepseek4.expert_weights_norm": True,
    "deepseek4.expert_gating_func": 4,  # LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS
    "deepseek4.swiglu_clamp_exp": [10.0] * LAYERS,
    "deepseek4.swiglu_clamp_shexp": [10.0] * LAYERS,
    "deepseek4.attention.compress_ratios": [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, LAYERS)],
    "tokenizer.ggml.tokens": {"array_len": 129280},
}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def same(got, want):
    if isinstance(want, float):
        return isinstance(got, float) and f32(got) == f32(want)
    if isinstance(want, list):
        # compress_ratios may carry entries past block_count (the MTP slots).
        return isinstance(got, list) and len(got) >= len(want) and all(
            same(g, w) for g, w in zip(got, want))
    return got == want


def main(argv):
    meta = import_m3.load_layout().read_gguf(argv[1])["meta"]
    ok = True
    for key, want in PROFILE.items():
        got = meta.get(key)
        good = same(got, want)
        ok = ok and good
        print(f"{'ok ' if good else 'BAD'} {key}: gguf {got!r}; profile {want!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
