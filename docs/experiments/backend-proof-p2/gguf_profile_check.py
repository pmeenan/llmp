#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Checks llmpalooza's compiled-in Qwen2 profile against the checkpoint's GGUF.

M2 has no GGUF reader, so the Qwen2.5-0.5B profile the native runs use is
compiled in (src/model/qwen2.cc). This reference-side check reads the FP16
GGUF's key/values and tensor table with the M0 prototype's GGUF reader
(../artifact-layout/layout.py read_gguf: pure Python, bounded, the pinned
gguf.cpp's key rules) and compares every scalar the profile holds, as
llama.cpp b29c606e2 derives it from the same keys (llama-model.cpp): block
count, embedding and feed-forward lengths, head counts, head and RoPE
dimensions, RoPE base and original context, the RMS norm epsilon (as
float32), the vocabulary (token list and table), the tensors' types and
the output head's shape, and that the checkpoint asks for no RoPE scaling.

  gguf_profile_check.py GGUF [--profile src/model/qwen2.cc]

Exit 0 when every value agrees, 1 at the first that does not.
"""
import argparse
import importlib.util
import pathlib
import re
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_layout():
    spec = importlib.util.spec_from_file_location("layout", HERE.parent / "artifact-layout" / "layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compiled_profile(source):
    """The designated initializers of Qwen25Instruct05B's profile."""
    text = pathlib.Path(source).read_text()
    body = text[text.index("Qwen25Instruct05B() {"):]
    body = body[:body.index("};")]
    fields = dict(re.findall(r"\.(\w+)\s*=\s*([^,}]+)", body))
    out = {}
    for key, value in fields.items():
        value = value.strip()
        if value.startswith('"'):
            out[key] = value.strip('"')
        elif value.endswith("f"):
            out[key] = float(value[:-1])
        else:
            out[key] = int(value)
    return out


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--profile", default=str(ROOT / "src" / "model" / "qwen2.cc"))
    a = ap.parse_args(argv)
    layout = load_layout()
    g = layout.read_gguf(a.gguf)
    meta = g["meta"]
    tensors = {t["name"]: t for t in g["tensors"]}
    p = compiled_profile(a.profile)
    arch = meta["general.architecture"]
    heads = meta[f"{arch}.attention.head_count"]
    width = meta[f"{arch}.embedding_length"]
    head_dim = meta.get(f"{arch}.attention.key_length", width // heads)
    checks = [
        ("architecture", arch, "qwen2"),
        ("layers", meta[f"{arch}.block_count"], p["layers"]),
        ("width", width, p["width"]),
        ("heads", heads, p["heads"]),
        ("kv_heads", meta[f"{arch}.attention.head_count_kv"], p["kv_heads"]),
        ("head_dim", head_dim, p["head_dim"]),
        ("head_dim (value length)", meta.get(f"{arch}.attention.value_length", width // heads), p["head_dim"]),
        ("rope dimensions (n_rot)", meta.get(f"{arch}.rope.dimension_count", width // heads), p["head_dim"]),
        ("ffn", meta[f"{arch}.feed_forward_length"], p["ffn"]),
        ("train_context (n_ctx_orig)", meta[f"{arch}.context_length"], p["train_context"]),
        ("rms_eps (float32)", f32(meta[f"{arch}.attention.layer_norm_rms_epsilon"]), f32(p["rms_eps"])),
        ("rope_base (float32)", f32(meta[f"{arch}.rope.freq_base"]), f32(p["rope_base"])),
        ("rope scaling", meta.get(f"{arch}.rope.scaling.type", "none"), "none"),
        ("vocab (tokens)", meta["tokenizer.ggml.tokens"]["array_len"], p["vocab"]),
        ("vocab (token table)", tensors["token_embd.weight"]["ne"], [p["width"], p["vocab"]]),
    ]
    # The output head: its own tensor here, which the artifact's importer
    # found identical to the token table and bound to it as an alias.
    head = tensors.get("output.weight", tensors["token_embd.weight"])
    checks.append(("output head", [head["dtype"], head["ne"]], [p["weight_type"], [p["width"], p["vocab"]]]))
    for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        checks.append((f"blk.0.{name}.weight type", tensors[f"blk.0.{name}.weight"]["dtype"], p["weight_type"]))
    checks.append(("token_embd.weight type", tensors["token_embd.weight"]["dtype"], p["weight_type"]))
    failed = False
    for name, gguf, profile in checks:
        ok = gguf == profile
        failed |= not ok
        print(f"{'ok  ' if ok else 'DIFF'} {name}: GGUF {gguf!r}, profile {profile!r}")
    print(f"GGUF header SHA-256 {g['header_sha256']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
