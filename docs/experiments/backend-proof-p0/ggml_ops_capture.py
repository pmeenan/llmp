#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture EXL3-G's non-linear operations on the held-out trajectory.

External reference harness for the backend proof's P0; it does not implement
llmpalooza inference. It runs the frozen EXL3-G profile (EXL3_GEMV=0,
EXL3_HGEMM_F16ACC=0, a copy of the frozen GEMM-only tuning cache) with
EXL3_BC_ATTN=0, which takes attention, RoPE and the norms through the Python
paths and is bit-identical to the optimized profile (README, third pass).
For each prefix of exl3_heldout.py it runs the prefill and the 16
single-token steps once, and records, for the chosen layers, the inputs and
outputs of each non-linear operation, exactly as upstream computed them:

- RMSNorm: the input (FP32 residual stream), the weight, and the FP16
  output, for input_layernorm, post_attention_layernorm (whose fused
  residual add is recorded too: residual before, block output, residual
  after) and the final norm;
- RoPE: q and k before and after, and the position;
- attention: q as passed (after RoPE), the new k and v rows, the F16 cache's
  K and V over every position the call attends to (after the call wrote the
  new rows), the positions, the softmax scale and the output; plus the
  PyTorch allocation peak of the call above its start;
- SiLU gating: gate and up (FP32) and the gated product (FP16), for prefill
  calls (single-token steps run the graphed C++ MLP, whose activation is
  internal);
- the embedding lookup's IDs and output.

Recorded phases: the prefill and single-token steps 0 and 15. The run's
logits must equal the frozen reference run's (--reference, a v3-*-g
directory) bit for bit, prefill and steps, or the capture is rejected: the
hooks only copy. --check-mlp-bc also reruns two prefixes with the graphed
bsz-1 MLP disabled (GatedMLP.bc = None, the Python path with upstream's own
activation) and records whether the logits are unchanged; exl3_heldout.py's
ggml_ops variant relies on that to route the single-token SiLU gating.

Output: capture-PREFIX.npz per prefix (keys PHASE.LAYER.OP.NAME) and
manifest.json.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

PREFIXES = (32, 144, 145, 1023, 1024)
SUFFIX = 16
RECORDED_STEPS = (0, 15)
CAPACITY = 4096
HELD_OUT_SHA256 = "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def host(t):
    t = t.detach()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.cpu().numpy().copy()


class State:
    tag = None      # the phase being recorded, or None
    layer = None    # the decoder layer currently running
    records = {}


def install_hooks(model, cache, layers, state):
    from exllamav3.modules import attn as attn_module
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.mlp import GatedMLP
    from exllamav3.modules.rmsnorm import RMSNorm
    from exllamav3.util.rope import RoPE

    def rec(key, **arrays):
        for name, value in arrays.items():
            if value is not None:
                state.records[f"{state.tag}.{key}.{name}"] = value if isinstance(value, np.ndarray) else host(value)

    norms = {"model.norm": "final.norm"}
    for layer in layers:
        norms[f"model.layers.{layer}.input_layernorm"] = f"L{layer}.attn_norm"
        norms[f"model.layers.{layer}.post_attention_layernorm"] = f"L{layer}.mlp_norm"

    norm_forward = RMSNorm.forward

    def norm_hook(self, x, params, out_dtype=None, residual=None, residual_in=None):
        key = norms.get(self.key)
        if state.tag is None or key is None:
            return norm_forward(self, x, params, out_dtype, residual, residual_in)
        x0 = x.clone()
        r0 = residual_in.clone() if residual_in is not None else None
        y = norm_forward(self, x, params, out_dtype, residual, residual_in)
        rec(key, x=x0, residual_before=r0, residual_after=residual_in, y=y, weight=self.weight,
            eps=np.array(self.rms_norm_eps, dtype=np.float64))
        return y
    RMSNorm.forward = norm_hook

    attn_forward = Attention.decode_flash_attn

    def attn_hook(self, x, bsz, seqlen, params):
        state.layer = self.layer_idx
        try:
            return attn_forward(self, x, bsz, seqlen, params)
        finally:
            state.layer = None
    Attention.decode_flash_attn = attn_hook

    rope_apply = RoPE.apply

    def rope_hook(self, q, k=None, position=0, positions=None, position_ids=None, in_place=False, *args, **kwargs):
        if state.tag is None or state.layer not in layers:
            return rope_apply(self, q, k, position, positions, position_ids, in_place, *args, **kwargs)
        assert positions is None and position_ids is None
        q0, k0 = q.clone(), k.clone()
        out = rope_apply(self, q, k, position, positions, position_ids, in_place, *args, **kwargs)
        rec(f"L{state.layer}.rope", q_before=q0, k_before=k0, q_after=out[0], k_after=out[1],
            position=np.array(position, dtype=np.int64))
        return out
    RoPE.apply = rope_hook

    dispatch = attn_module.attn_dispatch

    def dispatch_hook(q, k, v, cache=None, cache_idx=None, cache_instance=None, block_table=None,
                      cache_seqlens=None, sm_scale=None, **kwargs):
        if state.tag is None or state.layer not in layers:
            return dispatch(q, k, v, cache=cache, cache_idx=cache_idx, cache_instance=cache_instance,
                            block_table=block_table, cache_seqlens=cache_seqlens, sm_scale=sm_scale, **kwargs)
        q0, k0, v0 = q.clone(), k.clone(), v.clone()
        past = int(cache_seqlens[0])
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        o = dispatch(q, k, v, cache=cache, cache_idx=cache_idx, cache_instance=cache_instance,
                     block_table=block_table, cache_seqlens=cache_seqlens, sm_scale=sm_scale, **kwargs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        layer = cache.layers[cache_idx, cache_instance or 0]
        pages = block_table.shape[-1]
        assert torch.equal(block_table.cpu().reshape(-1)[:pages], torch.arange(pages, dtype=block_table.dtype))
        n = past + q.shape[1]
        kc = layer.k.view(-1, *layer.k.shape[2:])[:n]
        vc = layer.v.view(-1, *layer.v.shape[2:])[:n]
        rec(f"L{state.layer}.attn", q=q0[0], k_new=k0[0], v_new=v0[0], k_cache=kc, v_cache=vc, o=o[0],
            past=np.array(past, dtype=np.int64), scale=np.array(sm_scale, dtype=np.float64),
            torch_peak_bytes=np.array(peak, dtype=np.int64), output_bytes=np.array(o.numel() * o.element_size()))
        return o
    attn_module.attn_dispatch = dispatch_hook

    for module in model.modules:
        if type(module).__name__ == "TransformerBlock" and module.layer_idx in layers:
            mlp = module.mlp
            assert isinstance(mlp, GatedMLP)
            act = mlp.activation_fn_call

            def act_hook(g, u, a, act_limit, _act=act, _layer=module.layer_idx):
                if state.tag is None:
                    return _act(g, u, a, act_limit)
                g0, u0 = g.clone(), u.clone()
                out = _act(g, u, a, act_limit)
                rec(f"L{_layer}.swiglu", gate=g0, up=u0, out=a, act_limit=np.array(act_limit, dtype=np.float64))
                return out
            mlp.activation_fn_call = act_hook
        if type(module).__name__ == "Embedding":
            emb_forward = module.forward

            def emb_hook(x, params, *args, _forward=emb_forward, _module=module, **kwargs):
                y = _forward(x, params, *args, **kwargs)
                if state.tag == "prefill":
                    rec("embed", ids=x, out=y, dtype=np.array(str(_module.embedding.weight.dtype)))
                return y
            module.forward = emb_hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="the frozen EXL3-G run (v3-*-g)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,6,12,18,23")
    parser.add_argument("--check-mlp-bc", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inherited = [k for k in os.environ if k.startswith("EXL3_")]
    if inherited:
        raise ValueError(f"undeclared inherited EXL3 overrides: {inherited}")
    os.environ.update({"EXL3_BC_ATTN": "0", "EXL3_GEMV": "0", "EXL3_HGEMM_F16ACC": "0"})
    tune_cache = os.environ["EXLLAMAV3_TUNE_CACHE"]
    tune_before = digest(tune_cache)
    if digest(args.ids) != HELD_OUT_SHA256:
        raise ValueError("held-out ID identity mismatch")
    layers = tuple(int(x) for x in args.layers.split(","))
    ids_all = torch.from_numpy(np.frombuffer(args.ids.read_bytes(), dtype="<i8").copy()).reshape(1, -1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from exllamav3 import Cache, Config, Model
    fixture = next(f for f in json.loads(args.pins.read_text())["fixtures"]
                   if f["repository"].endswith(args.model.name))
    for row in fixture["metadata_files_verified"] + [{"filename": fixture["filename"],
                                                      "sha256": fixture["published_sha256"]}]:
        if digest(args.model / row["filename"]) != row["sha256"]:
            raise ValueError("fixture hash mismatch")
    config = Config.from_directory(str(args.model))
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CAPACITY, max_batch_size=1)
    model.load(device="cuda:0")
    state = State()
    install_hooks(model, cache, layers, state)
    torch.cuda.synchronize()

    def params(p):
        return {"attn_mode": "flash_attn", "cache": cache, "past_len": p, "batch_shape": (1, CAPACITY),
                "pinned_staging": bool(p)}

    def zero():
        for tensor in cache.get_all_tensors():
            tensor.zero_()
        torch.cuda.synchronize()

    def trajectory(prefix, record):
        ids = ids_all[:, :prefix + SUFFIX]
        zero()
        state.tag = "prefill" if record else None
        pre = model.forward(ids[:, :prefix], params(0)).float().cpu().numpy()
        steps = []
        for i in range(SUFFIX):
            state.tag = f"step{i}" if record and i in RECORDED_STEPS else None
            steps.append(model.forward(ids[:, prefix + i:prefix + i + 1], params(prefix + i)).float().cpu().numpy())
        state.tag = None
        return pre, np.concatenate(steps, axis=1)

    def same(a, b):
        return a.shape == b.shape and a.tobytes() == b.tobytes()

    results = []
    with torch.inference_mode():
        for prefix in PREFIXES:
            state.records = {}
            pre, suf = trajectory(prefix, True)
            ref = np.load(args.reference / f"logits-{prefix}.npz")
            row = {"prefix": prefix, "prefill_equal_reference": same(pre, ref["prefill"]),
                   "suffix_equal_reference": same(suf, ref["suffix"]), "arrays": len(state.records)}
            np.savez(args.output / f"capture-{prefix}.npz", **state.records)
            results.append(row)
            print("capture", row, flush=True)
        mlp_bc = None
        if args.check_mlp_bc:
            mlps = [m.mlp for m in model.modules if type(m).__name__ == "TransformerBlock"]
            saved = [m.bc for m in mlps]
            for m in mlps:
                m.bc = None
            mlp_bc = []
            for prefix in (32, 145):
                pre, suf = trajectory(prefix, False)
                ref = np.load(args.reference / f"logits-{prefix}.npz")
                mlp_bc.append({"prefix": prefix, "prefill_equal_reference": same(pre, ref["prefill"]),
                               "suffix_equal_reference": same(suf, ref["suffix"])})
            for m, bc in zip(mlps, saved):
                m.bc = bc
            print("mlp_bc", mlp_bc, flush=True)
    manifest = {"fixture": fixture["repository"], "layers": layers, "recorded_steps": RECORDED_STEPS,
                "reference": str(args.reference), "results": results, "mlp_bc_none": mlp_bc,
                "harness_sha256": digest(Path(__file__)), "tune_cache_before_sha256": tune_before,
                "tune_cache_after_sha256": digest(tune_cache), "torch": torch.__version__,
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMAV3_"))},
                "passed": all(r["prefill_equal_reference"] and r["suffix_equal_reference"] for r in results)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["passed"]:
        raise SystemExit("capture logits differ from the reference run")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
