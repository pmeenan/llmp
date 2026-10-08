#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The declared EXL3 held-out trajectory on the pinned ExLlamaV3 reference.

External reference harness for the backend proof's P0 controls; it does not
implement llmpalooza inference. It follows the M0 reference's numerical mode
(../exl3-reference/measure.py): for each prefix, a forward pass over the
prefix (recording every row's logits), then 16 teacher-forced single-token
steps, repeated three times; then the cache is snapshotted, poisoned,
restored and the suffix run again. All repeats and the restored suffix must
be bit-identical. The IDs are the declared held-out trajectory (1,040
little-endian int64, SHA-256 6dd8da89...), prefix i using its first i + 16.

The manifest records the cuBLAS and cuBLASLt libraries mapped into the
process and whether ExLlamaV3's fp16-accumulating reconstruction GEMM is
active (a timing probe decides it unless EXL3_HGEMM_F16ACC is set).

With --capture, a further instrumented pass per prefix records the residual
stream after every decoder block, the final norm's output, and the cache's
K and V over the prefix and the 16 single-token steps. Its logits must
equal the uninstrumented run's bit for bit, or the run fails. With
--memory, the manifest records each phase's peak PyTorch allocation above
its starting point.

--variant changes one thing in upstream, for the Tier C calibration:
legitimate variants swap in upstream's own alternative code for the
non-linear operations (PyTorch RMSNorm or RoPE, general Triton or PyTorch
SDPA attention); faults inject a bug so the gates can be shown to trip.
Variants are harness monkeypatches, recorded in the manifest.

The autotuner's disk cache is named explicitly (EXLLAMAV3_TUNE_CACHE): a run
with an empty cache tunes and records its choices; later runs given a copy
reuse them, so their kernel shapes and grids are the same. Every logit is
saved for cross-arm comparison (exl3_compare.py).
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

PREFIXES = (32, 144, 145, 1023, 1024)
SUFFIX = 16
REPEATS = 3
CAPACITY = 4096
HELD_OUT_SHA256 = "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def exact(a, b):
    return {"exact": a.shape == b.shape and a.tobytes() == b.tobytes(),
            "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())}


def blas_libraries():
    """The cuBLAS/cuBLASLt files mapped into this process, with cuBLASLt's version."""
    paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libcublas" in line and line.split()[-1].startswith("/")})
    versions = {}
    for path in paths:
        if "libcublasLt" in path:
            versions[path] = ctypes.CDLL(path).cublasLtGetVersion()
    return {"mapped": paths, "cublasLt_version": versions}


class _RopeProxy:
    """One layer's view of a shared RoPE object with a replaced apply()."""

    def __init__(self, rope, apply):
        self._rope, self.apply = rope, apply

    def __getattr__(self, name):
        return getattr(self._rope, name)


def _layer_rope(model, layer, change):
    attn = model.find_module(f"model.layers.{layer}.self_attn")
    rope = attn.rope

    def apply(q, k=None, position=0, positions=None, position_ids=None, in_place=False, *args, **kwargs):
        position, q_out, k_out = change(q, position)
        out = rope.apply(q, k, position, positions, position_ids, in_place, *args, **kwargs)
        return k_out(out) if k_out else out
    attn.rope = _RopeProxy(rope, apply)


VARIANTS = ("rmsnorm_torch", "rope_torch", "attn_general", "attn_sdpa", "all_torch", "f_rope_offset_l10",
            "f_one_key_l10", "f_decode_rope_l12", "f_norm_eps", "f_softmax_scale_l8", "f_bf16_mlp_l16", "ggml_ops",
            "f_q_rope_offset_l10")


def apply_variant(name, model):
    """Monkeypatch upstream for one --variant. Every variant runs with
    EXL3_BC_ATTN=0, set before import (the attention_eager profile, bit-identical
    to the optimized one), so attention, RoPE and norms go through the patched
    Python paths rather than the graph-captured C++ attention step."""
    import exllamav3.modules.attention_fn.dispatch as dispatch
    from exllamav3.modules.attention_fn import torch as torch_attn
    from exllamav3.modules.rmsnorm import RMSNorm
    from exllamav3.util.rope import RoPE

    def rmsnorm_torch():
        def forward(self, x, params, out_dtype=None, residual=None, residual_in=None):
            if residual is not None:
                raise NotImplementedError("rmsnorm_torch: residual output")
            if residual_in is not None:
                residual_in += x
                x = residual_in
            return self.forward_torch(x, params, out_dtype)
        RMSNorm.forward = forward

    def rope_torch():
        def apply(self, q, k=None, position=0, positions=None, position_ids=None, in_place=False,
                  q_norm=None, k_norm=None, norm_eps=1e-6, norm_constant_bias=0.0, inv_freq=None):
            if q_norm is not None or k_norm is not None or inv_freq is not None:
                raise NotImplementedError("rope_torch: fused norms or a custom inv_freq")
            return self.apply_torch(q, k, position, positions, position_ids, in_place)
        RoPE.apply = apply

    def attn_general():
        dispatch.attn_fns = [f for f in dispatch.attn_fns if f not in dispatch._fns_triton_fast]

    def attn_sdpa():
        def sdpa(args):
            if args.is_varlen() or not args.has_kv_cache() or args.non_causal_spans or args.sinks is not None:
                return None
            return torch_attn._torch_bighead_fallback(
                q=args.q, k=args.k, v=args.v, k_cache=args.k_cache, v_cache=args.v_cache,
                block_table=args.block_table, cache_seqlens=args.cache_seqlens, causal=args.causal,
                softmax_scale=args.sm_scale, window_size=args.get_window_size(), softcap=args.softcap)
        dispatch.attn_fns = [sdpa]

    def f_rope_offset_l10():
        _layer_rope(model, 10, lambda q, position: (position + 1, None, None))

    def f_one_key_l10():
        def change(q, position):
            def fix(out):
                q2, k2 = out
                if k2 is not None and k2.shape[-3] > 21:
                    k2[..., 20, :, :] = k2[..., 21, :, :]
                return q2, k2
            return position, None, fix
        _layer_rope(model, 10, change)

    def f_q_rope_offset_l10():
        # Unlike f_rope_offset_l10 (q and k shifted together, which RoPE's
        # relative form nearly cancels), only q moves: a real position error.
        attn = model.find_module("model.layers.10.self_attn")
        rope = attn.rope

        def apply(q, k=None, position=0, positions=None, position_ids=None, in_place=False, *args, **kwargs):
            q2, _ = rope.apply(q, None, position + 1, positions, position_ids, in_place, *args, **kwargs)
            k2, _ = rope.apply(k, None, position, positions, position_ids, in_place, *args, **kwargs)
            return q2, k2
        attn.rope = _RopeProxy(rope, apply)

    def f_decode_rope_l12():
        _layer_rope(model, 12, lambda q, position: (position + 1 if q.shape[-3] == 1 else position, None, None))

    def f_norm_eps():
        for module in model:
            if isinstance(module, RMSNorm):
                module.rms_norm_eps = 1e-5

    def f_softmax_scale_l8():
        model.find_module("model.layers.8.self_attn").sm_scale *= 1.01

    def f_bf16_mlp_l16():
        mlp = model.find_module("model.layers.16.mlp")
        forward = mlp.forward

        def rounded(x, params, *args, **kwargs):
            y = forward(x, params, *args, **kwargs)
            return y.to(torch.bfloat16).to(y.dtype)
        mlp.forward = rounded

    def ggml_ops_plan():
        """Native-like: GGML's CUDA kernels (the reference shim, ggml_ops.py and
        ggml_shim/, library named by GGML_OPS_LIB) for the non-linear operations,
        with the paths ggml-ops.json recommends; ExLlamaV3's linears unchanged.
        - RMSNorm: rms_norm and mul in F32 on the FP32 residual stream, output
          converted to F16 by GGML's copy; the fused pre-norm residual add is
          GGML's F32 add, in place.
        - RoPE: NEOX in F32 on the F16 projections; q stays F32, k is rounded to
          F16 (the cache's type).
        - Attention: the new K/V rows are copied into the F16 cache (exact), then
          ggml_flash_attn_ext's vector kernel (forced launcher) over the cache
          padded to 256 positions, q F32, output converted to F16 for o_proj.
        - SiLU gating: GGML's swiglu on the FP32 gate and up, output F16; the
          graphed bsz-1 MLP is disabled so single-token steps take the same
          Python path (ggml_ops_capture.py --check-mlp-bc: logits unchanged).
        The embedding lookup and the MLP's residual add stay upstream's; GGML's
        are bit-identical to them (ggml-ops.json). The library's identity is
        recorded in the manifest's environment (EXL3_VARIANT_GGML_OPS_*)."""
        import glob
        from ggml_ops import GGMLOps
        from exllamav3.modules.mlp import GatedMLP
        from exllamav3.util.rope import RopeStyle

        library = os.environ["GGML_OPS_LIB"]
        ops = GGMLOps(library)
        build = sorted(glob.glob(os.path.join(os.path.dirname(library), "**", "*.so.*"), recursive=True) + [library])
        os.environ["EXL3_VARIANT_GGML_OPS_LIB"] = library
        os.environ["EXL3_VARIANT_GGML_OPS_BUILD"] = json.dumps(ops.build_info, sort_keys=True)
        os.environ["EXL3_VARIANT_GGML_OPS_SHA256"] = json.dumps(
            {os.path.relpath(p, os.path.dirname(library)): digest(p) for p in build if not os.path.islink(p)},
            sort_keys=True)
        weights = {}

        def norm_forward(self, x, params, out_dtype=None, residual=None, residual_in=None):
            if (residual is not None or self.span_heads or self.groups != 1 or self.unweighted
                    or self.constant_bias != 0.0 or self.constant_scale != 1.0):
                raise NotImplementedError("ggml_ops: RMSNorm configuration")
            if residual_in is not None:
                ops.add(residual_in, x, out=residual_in)
                x = residual_in
            w = weights.get(id(self))
            if w is None:
                w = weights[id(self)] = self.weight.float().contiguous()
            y = ops.rms_norm(x, w, self.rms_norm_eps, out_dtype or self.out_dtype or x.dtype)
            return y.view(*x.shape[:-1], y.shape[-1])
        RMSNorm.forward = norm_forward

        def rope_apply(self, q, k=None, position=0, positions=None, position_ids=None, in_place=False,
                       q_norm=None, k_norm=None, norm_eps=1e-6, norm_constant_bias=0.0, inv_freq=None):
            if (k is None or positions is not None or position_ids is not None or q_norm is not None
                    or k_norm is not None or inv_freq is not None or q.dim() != 4 or q.shape[0] != 1
                    or self.rope_settings.rope_style != RopeStyle.NEOX or self.attn_factor != 1.0
                    or self.inv_freq.numel() * 2 != q.shape[-1]):
                raise NotImplementedError("ggml_ops: RoPE configuration")
            theta = self.rope_settings.rope_theta
            q_out = ops.rope_neox(q[0], position, theta, torch.float32, torch.float32)
            k_out = ops.rope_neox(k[0], position, theta, torch.float32, torch.float16)
            return q_out[None], k_out[None]
        RoPE.apply = rope_apply

        def attention(args):
            if (args.is_varlen() or not args.has_kv_cache() or args.non_causal_spans or args.sinks is not None
                    or args.bsz != 1 or args.softcap or args.is_swa() or not args.causal or args.q_cache is not None):
                return None
            pages = args.block_table.shape[-1]
            if not torch.equal(args.block_table.reshape(-1)[:pages].cpu(), torch.arange(pages, dtype=args.block_table.dtype)):
                raise NotImplementedError("ggml_ops: non-contiguous pages")
            past = int(args.cache_seqlens[0])
            k_cache = args.k_cache.view(-1, *args.k_cache.shape[2:])
            v_cache = args.v_cache.view(-1, *args.v_cache.shape[2:])
            k_cache[past:past + args.q_len].copy_(args.k[0])
            v_cache[past:past + args.q_len].copy_(args.v[0])
            o = ops.attention(args.q[0], k_cache, v_cache, past + args.q_len, past, args.sm_scale, "fa_vec",
                              torch.float16)
            return o[None]
        dispatch.attn_fns = [attention]

        def swiglu(g, u, a, act_limit):
            if act_limit != 0.0 or a.dtype != torch.float16:
                raise NotImplementedError("ggml_ops: activation configuration")
            a.copy_(ops.swiglu(g, u, torch.float16).view_as(a))
        for module in model:
            if isinstance(module, GatedMLP):
                if module.activation_fn != "silu":
                    raise NotImplementedError("ggml_ops: activation")
                module.bc = None
                module.activation_fn_call = swiglu

    variants = {
        "ggml_ops": [ggml_ops_plan],
        "rmsnorm_torch": [rmsnorm_torch],
        "rope_torch": [rope_torch],
        "attn_general": [attn_general],
        "attn_sdpa": [attn_sdpa],
        "all_torch": [rmsnorm_torch, rope_torch, attn_sdpa],
        "f_rope_offset_l10": [f_rope_offset_l10],
        "f_one_key_l10": [f_one_key_l10],
        "f_decode_rope_l12": [f_decode_rope_l12],
        "f_norm_eps": [f_norm_eps],
        "f_softmax_scale_l8": [f_softmax_scale_l8],
        "f_bf16_mlp_l16": [f_bf16_mlp_l16],
        "f_q_rope_offset_l10": [f_q_rope_offset_l10],
    }
    for patch in variants[name]:
        patch()


def run(model, cache, ids_all, output, capture_layers, memory):
    def params(p):
        return {"attn_mode": "flash_attn", "cache": cache, "past_len": p, "batch_shape": (1, CAPACITY),
                "pinned_staging": bool(p)}

    def zero():
        for tensor in cache.get_all_tensors():
            tensor.zero_()
        torch.cuda.synchronize()

    def cache_prefix(p):
        out = []
        for tensor in cache.get_all_tensors():
            flat = tensor.view(-1, *tensor.shape[2:])
            out.append(flat[:p].float().cpu().numpy().copy())
        return np.stack(out)

    def capture(ids, prefix, reference, suffix_reference):
        """One instrumented forward over the prefix: block outputs, final norm, K/V."""
        zero()
        blocks, originals = [], []
        for module in model.modules:
            if type(module).__name__ == "TransformerBlock" or module.key == "model.norm":
                forward = module.forward

                def wrapped(x, params, *args, _forward=forward, **kwargs):
                    y = _forward(x, params, *args, **kwargs)
                    blocks.append(y.float().cpu().numpy().copy())
                    return y
                module.forward = wrapped
                originals.append((module, forward))
        try:
            logits = model.forward(ids[:, :prefix], params(0)).float().cpu().numpy().copy()
        finally:
            for module, forward in originals:
                module.forward = forward
        torch.cuda.synchronize()
        after_prefill = cache_prefix(prefix)
        steps = suffix(ids, prefix)
        np.savez(output / f"capture-{prefix}.npz", blocks=np.stack(blocks[:-1]), final_norm=blocks[-1],
                 kv=after_prefill, kv_after_suffix=cache_prefix(prefix + SUFFIX))
        return [{"kind": "capture_logits_equal_prefill", **exact(reference, logits)},
                {"kind": "capture_suffix_equal", **exact(suffix_reference, steps)}]

    def suffix(ids, p):
        out = []
        for i in range(SUFFIX):
            logits = model.forward(ids[:, p + i:p + i + 1], params(p + i))
            out.append(logits.float().cpu().numpy().copy())
        return np.concatenate(out, axis=1)

    results = []
    memory_record = {}
    for prefix in PREFIXES:
        ids = ids_all[:, :prefix + SUFFIX]

        def begin():
            zero()
            return model.forward(ids[:, :prefix], params(0)).float().cpu().numpy().copy()

        pre = begin()
        tensors = cache.get_all_tensors()
        snapshot = [t.cpu().clone() for t in tensors]
        hashes = [hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for t in snapshot]
        ref = suffix(ids, prefix)
        np.savez(output / f"logits-{prefix}.npz", ids=ids.numpy(), suffix=ref, prefill=pre)
        comparisons = []
        for _ in range(REPEATS):
            repeated_pre = begin()
            comparisons.append({"kind": "prefill_repeat", **exact(pre, repeated_pre)})
            comparisons.append({"kind": "suffix_repeat", **exact(ref, suffix(ids, prefix))})
        for t in tensors:
            t.fill_(float("nan"))
        torch.cuda.synchronize()
        poisoned = all(bool(torch.isnan(t).all()) for t in tensors)
        for target, saved in zip(tensors, snapshot, strict=True):
            target.copy_(saved)
        torch.cuda.synchronize()
        restored = [hashlib.sha256(t.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for t in tensors]
        comparisons.append({"kind": "suffix_restore", **exact(ref, suffix(ids, prefix))})
        if capture_layers:
            comparisons += capture(ids, prefix, pre, ref)
        if memory:
            zero()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            model.forward(ids[:, :prefix], params(0))
            torch.cuda.synchronize()
            phase = {"prefill_peak_above_start": torch.cuda.max_memory_allocated() - base, "steps": []}
            for i in range(SUFFIX):
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                model.forward(ids[:, prefix + i:prefix + i + 1], params(prefix + i))
                torch.cuda.synchronize()
                phase["steps"].append(torch.cuda.max_memory_allocated() - base)
            memory_record[prefix] = phase
        record = {"prefix": prefix, "suffix_tokens": SUFFIX, "poison_verified": poisoned,
                  "restored_storage_exact": restored == hashes,
                  "suffix_sha256": hashlib.sha256(ref.tobytes()).hexdigest(),
                  "prefill_sha256": hashlib.sha256(pre.tobytes()).hexdigest(),
                  "comparisons": comparisons}
        record["passed"] = poisoned and restored == hashes and all(c["exact"] and c["finite"] for c in comparisons)
        results.append(record)
        if memory:
            record["memory"] = memory_record[prefix]
        print("heldout", prefix, record["passed"], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True, help="../exl3-reference/pins.json")
    parser.add_argument("--profile", choices=("optimized", "attention_eager"), default="optimized")
    parser.add_argument("--gemv", choices=("default", "off"), default="default")
    parser.add_argument("--hgemm-f16acc", choices=("probe", "0", "1"), default="probe",
                        help="ExLlamaV3's reconstruction GEMM: its timing probe decides, or forced")
    parser.add_argument("--capture", action="store_true", help="also record per-layer state")
    parser.add_argument("--memory", action="store_true", help="also record per-phase peak allocations")
    parser.add_argument("--variant", choices=VARIANTS, help="a legitimate variant or an injected fault (apply_variant)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inherited = [k for k in os.environ if k.startswith("EXL3_")]
    if inherited:
        raise ValueError(f"undeclared inherited EXL3 overrides: {inherited}")
    if args.profile == "attention_eager" or args.variant:
        os.environ["EXL3_BC_ATTN"] = "0"
    if args.gemv == "off":
        os.environ["EXL3_GEMV"] = "0"
    if args.hgemm_f16acc != "probe":
        os.environ["EXL3_HGEMM_F16ACC"] = args.hgemm_f16acc
    tune_cache = os.environ.get("EXLLAMAV3_TUNE_CACHE")
    if not tune_cache:
        raise ValueError("name the tuning cache with EXLLAMAV3_TUNE_CACHE")
    tune_before = digest(tune_cache) if Path(tune_cache).exists() else None
    if digest(args.ids) != HELD_OUT_SHA256:
        raise ValueError("held-out ID identity mismatch")
    ids = torch.from_numpy(np.frombuffer(args.ids.read_bytes(), dtype="<i8").copy()).reshape(1, -1)
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
    if args.variant:  # after load, which creates each layer's RoPE object
        apply_variant(args.variant, model)
    torch.cuda.synchronize()
    with torch.inference_mode():
        results = run(model, cache, ids, args.output, args.capture, args.memory)
    from exllamav3.ext import exllamav3_ext as ext
    manifest = {"profile": args.profile, "gemv": args.gemv, "hgemm_f16acc_request": args.hgemm_f16acc,
                "variant": args.variant,
                "hgemm_f16acc_active": ext.hgemm_f16acc_status(0), "capture": args.capture,
                "blas": blas_libraries(), "fixture": fixture["repository"],
                "revision": fixture["revision"], "weight_sha256": fixture["published_sha256"],
                "harness_sha256": digest(Path(__file__)), "ids_sha256": HELD_OUT_SHA256,
                "tune_cache_before_sha256": tune_before, "tune_cache_after_sha256": digest(tune_cache),
                "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMAV3_", "CUDA_"))},
                "results": results, "passed": all(r["passed"] for r in results)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["passed"]:
        raise SystemExit("FAILED repeat/restore controls; diagnostics preserved")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
