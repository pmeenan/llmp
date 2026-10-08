#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""FP64 teacher-forced forward pass of an EXL3 fixture: the accuracy oracle.

External reference tooling for the backend proof's P0; it does not implement
llmpalooza inference. Qwen2.5-0.5B's computation (config.json: 24 layers, hidden
896, 14 query and 2 KV heads of 64, MLP 4,864, RMSNorm eps 1e-6, NEOX RoPE
with theta 1e6, SiLU-gated MLP, biases on Q/K/V) over the exported weights
(exl3_export.py), entirely in FP64 on the CPU. Each linear's original-basis
weight is rebuilt from its decoded rotated-basis trellis as upstream's
get_weight_tensor() does, but in FP64: a 128-point Hadamard on the input
side, suh, a 128-point Hadamard on the output side, svh, each Hadamard
scaled by 1/sqrt(128).

The pass covers the first T held-out IDs once. Causal attention makes row i
depend only on IDs 0..i, so these rows are the exact counterparts of every
prefix's prefill and single-token steps. Output: the logits, the residual
stream after every block, the final norm's output, and each layer's K
(after RoPE) and V.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 24, 896, 14, 2, 64
EPS, THETA = 1e-6, 1e6
HELD_OUT_SHA256 = "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def weight(w, key, hadamard):
    inner = torch.from_numpy(w[f"{key}.inner"]).double()
    suh = torch.from_numpy(w[f"{key}.suh"]).double()
    svh = torch.from_numpy(w[f"{key}.svh"]).double()
    k, n = inner.shape
    x = (hadamard @ inner.view(-1, 128, n)).view(k, n)
    x = x * suh[:, None]
    x = (x.view(k, -1, 128) @ hadamard).view(k, n)
    return x * svh[None, :]


def norm(x, w):
    return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + EPS) * w


def rope(x, positions):
    inv = THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    angles = positions[:, None].double() * inv[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[:, None, :]
    half = HEAD_DIM // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1040)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.ids) != HELD_OUT_SHA256:
        raise ValueError("held-out ID identity mismatch")
    torch.set_num_threads(16)
    ids = torch.from_numpy(np.frombuffer(args.ids.read_bytes(), dtype="<i8")[:args.tokens].copy())
    w = np.load(args.weights)
    hadamard = torch.from_numpy(w["hadamard128"]).double() / math.sqrt(128)
    t = len(ids)
    positions = torch.arange(t)
    mask = torch.full((t, t), float("-inf"), dtype=torch.float64).triu(1)
    x = torch.from_numpy(w["model.embed_tokens.weight"]).double()[ids]
    blocks, keys, values = [], [], []
    with torch.inference_mode():
        for layer in range(LAYERS):
            p = f"model.layers.{layer}"
            h = norm(x, torch.from_numpy(w[f"{p}.input_layernorm.weight"]).double())
            proj = {}
            for name in ("q", "k", "v"):
                key = f"{p}.self_attn.{name}_proj"
                proj[name] = h @ weight(w, key, hadamard) + torch.from_numpy(w[f"{key}.bias"]).double()
            q = rope(proj["q"].view(t, HEADS, HEAD_DIM), positions)
            k = rope(proj["k"].view(t, KV_HEADS, HEAD_DIM), positions)
            v = proj["v"].view(t, KV_HEADS, HEAD_DIM)
            keys.append(k.numpy().copy())
            values.append(v.numpy().copy())
            group = HEADS // KV_HEADS
            kk = k.repeat_interleave(group, dim=1).transpose(0, 1)
            vv = v.repeat_interleave(group, dim=1).transpose(0, 1)
            scores = q.transpose(0, 1) @ kk.transpose(1, 2) / math.sqrt(HEAD_DIM) + mask
            attn = torch.softmax(scores, dim=-1) @ vv
            x = x + attn.transpose(0, 1).reshape(t, HIDDEN) @ weight(w, f"{p}.self_attn.o_proj", hadamard)
            h = norm(x, torch.from_numpy(w[f"{p}.post_attention_layernorm.weight"]).double())
            gate = h @ weight(w, f"{p}.mlp.gate_proj", hadamard)
            up = h @ weight(w, f"{p}.mlp.up_proj", hadamard)
            x = x + (torch.nn.functional.silu(gate) * up) @ weight(w, f"{p}.mlp.down_proj", hadamard)
            blocks.append(x.numpy().copy())
        final = norm(x, torch.from_numpy(w["model.norm.weight"]).double())
        logits = final @ weight(w, "lm_head", hadamard)
    np.savez(args.output, logits=logits.numpy(), blocks=np.stack(blocks), final_norm=final.numpy(),
             k=np.stack(keys), v=np.stack(values))
    print(json.dumps({"tokens": t, "weights_sha256": digest(args.weights), "output_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
