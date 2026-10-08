#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The native EXL3 operation plan record (reference only): exl3-op-plan.json.

External reference tooling for the backend proof's P0; it does not implement
llmpalooza inference. It has three steps.

probe (in the reference container, with the GPU): runs the declared plan
inside upstream ExLlamaV3 and records what executed. It starts from
exl3_heldout.py's ggml_ops variant (GGML's CUDA kernels through the shim,
ggml_ops.py and ggml_shim/, library named by GGML_OPS_LIB; ExLlamaV3's
linears) and moves the three operations that arm leaves to upstream onto
their plan owners:
  - the embedding lookup: GGML get_rows on the device BF16 table, F32 out
    (upstream looks up on the CPU and converts in PyTorch);
  - the MLP residual add: GGML add in F32, in place (upstream: PyTorch);
  - the q/k/v bias add on the reconstruction path: ExLlamaV3's own add
    kernel (ext.add, add_kernel_hhh on F16), as its packed path already does
    inside the linear (upstream: PyTorch's elementwise add).
Before profiling it runs each phase once with checks: each moved operation
is compared bit for bit with upstream's on the same inputs, and the logits
with the reference ggml_ops arm's (--reference). Then it profiles one run of
each phase kind (prefill of 32, 144, 145, 1,023 and 1,024 rows from an
empty cache; single-token steps at positions 32 and 1,024, whose padded K
lengths are 256 and 1,280, the only two the trajectories reach) with the
PyTorch profiler (CUPTI) and records every kernel, memcpy and memset with
its launch configuration, attributed to the innermost labelled operation
through the launching runtime call. --bias-exhaustive also runs ExLlamaV3's
add_kernel_hhh and PyTorch's F16 add over all 2^32 F16 input pairs.

In the reference container through ggml_ops_run.sh, with the cuBLAS 13.8.0.4
bind mounts (README, "Second pass") and the NVCC 13.4 build of the shim:

  TUNE=<copy of tune-40-gemvoff.bin> DOCKER_EXTRA="-e GGML_OPS_LIB=/p0/ggmlops/cuda134/libggml_shim.so <mounts>" \\
    ggml_ops_run.sh exl3_op_plan.py probe --model /experiment/models/4.0bpw --ids /p0/heldout-ids.i64le \\
    --pins /p0/exl3/pins.json --reference /p0/exl3/v3-40-ggml_ops_cublas138b --output probe-40.json \\
    [--bias-exhaustive]

sass (on the host, no GPU): per-function SASS hashes of the same build's
libggml-cuda, with fp16_plan.py's hashing, and its resource usage:

  cuobjdump -sass libggml-cuda.so.0.24.0 | fp16_plan.py sass-hash --label cuda134 > sass.jsonl
  cuobjdump -res-usage libggml-cuda.so.0.24.0 > res.txt

build (Python and c++filt): reduces the probes, the SASS hashes and
fp16-plan.json (for SASS shared with the FP16 bridge) to the record:

  exl3_op_plan.py build --probe 4.0bpw=probe-40.json 4.5bpw=probe-45.json --sass sass.jsonl \\
      --res-usage res.txt --fp16-plan fp16-plan.json --out exl3-op-plan.json
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HELD_OUT_SHA256 = "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"
CAPACITY = 4096
PHASES = (("prefill", 32), ("prefill", 144), ("prefill", 145), ("prefill", 1023), ("prefill", 1024),
          ("step", 32), ("step", 1024))


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


# --------------------------------------------------------------------------- probe


class Recorder:
    """Labelled operations: op#SEQ record_function ranges, with details."""

    def __init__(self):
        self.calls = []
        self.stack = []

    def reset(self):
        self.calls, self.stack = [], []

    @contextlib.contextmanager
    def op(self, category, **details):
        import torch
        seq = len(self.calls)
        call = {"seq": seq, "category": category, "parent": self.stack[-1] if self.stack else None, **details}
        self.calls.append(call)
        self.stack.append(seq)
        try:
            with torch.profiler.record_function(f"op#{seq}"):
                yield call
        finally:
            self.stack.pop()


def tdesc(t):
    return {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}


def bits_equal(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}[a.dtype]
    return bool(torch.equal(a.contiguous().view(view).cpu(), b.contiguous().view(view).cpu()))


def install(model, rec, checks, state):
    """The plan's patches on top of exl3_heldout.py's ggml_ops variant, with labels."""
    import torch
    import ggml_ops
    import exl3_heldout
    import exllamav3.modules.attention_fn.dispatch as dispatch
    import exllamav3.modules.mlp as mlp_module
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.embedding import Embedding
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.mlp import GatedMLP
    from exllamav3.modules.quant.exl3 import AUTO_RECONSTRUCT_THRESHOLD, MAX_RECONSTRUCT_SLICE_N, LinearEXL3
    from exllamav3.modules.transformer import TransformerBlock
    from exllamav3.util.tensor import to2

    def count(name, equal):
        c = checks.setdefault(name, {"cases": 0, "equal": 0})
        c["cases"] += 1
        c["equal"] += int(bool(equal))

    # Labels on every GGML call (class level, so the variant's instance has them).
    for name in ("rms_norm", "add", "rope_neox", "swiglu", "get_rows", "attention"):
        original = getattr(ggml_ops.GGMLOps, name)

        def wrapped(self, *args, _original=original, _name=name, **kwargs):
            tensors = [tdesc(a) for a in args if isinstance(a, torch.Tensor)]
            scalars = [a if not isinstance(a, torch.dtype) else str(a).replace("torch.", "")
                       for a in args if not isinstance(a, torch.Tensor)]
            kw = {k: (str(v).replace("torch.", "") if isinstance(v, torch.dtype) else v)
                  for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
            with rec.op(f"ggml.{_name}", inputs=tensors, scalars=scalars, kwargs=kw) as call:
                out = _original(self, *args, **kwargs)
                torch.cuda.synchronize()
            call["output"] = tdesc(out)
            call["scratch_pool_peak_bytes"], call["scratch_intermediate_bytes"] = self.last_scratch()
            return out
        setattr(ggml_ops.GGMLOps, name, wrapped)

    exl3_heldout.apply_variant("ggml_ops", model)
    ops = ggml_ops.GGMLOps(os.environ["GGML_OPS_LIB"])

    # ExLlamaV3 linears: path and output dtype per call.
    linear_forward = Linear.forward

    def linear_hook(self, x, params, out_dtype=None):
        inner = self.inner
        rows = x.numel() // x.shape[-1]
        detail = {"key": self.key, "input": tdesc(x), "has_bias": getattr(inner, "bias", None) is not None}
        if isinstance(inner, LinearEXL3):
            if rows <= AUTO_RECONSTRUCT_THRESHOLD:
                detail["path"] = "packed"
            else:
                fused = (inner.in_features % 128 == 0 and inner.out_features % 128 == 0
                         and os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") == "0" and rows >= 1024)
                detail["path"] = "fused-reconstruct" if fused else "reconstruct"
                detail["reconstruct_slices"] = -(-inner.out_features // MAX_RECONSTRUCT_SLICE_N)
            detail["K"] = int(inner.K)
        else:
            detail["path"] = type(inner).__name__
        with rec.op("linear", **detail) as call:
            y = linear_forward(self, x, params, out_dtype)
            torch.cuda.synchronize()
        call["output"] = tdesc(y)
        return y
    Linear.forward = linear_hook

    # The reconstruction path's bias add: ExLlamaV3's add kernel instead of PyTorch's.
    recon = LinearEXL3.reconstruct_hgemm

    def recon_hook(self, x, out_dtype):
        bias = self.bias
        if bias is None:
            return recon(self, x, out_dtype)
        self.bias = None
        try:
            y = recon(self, x, out_dtype)
        finally:
            self.bias = bias
        reference = None
        if state["check"]:
            reference = y.clone()
            reference += bias
        with rec.op("bias_add", owner="exllamav3 add_gr", input=tdesc(y), bias=tdesc(bias)) as call:
            ext.add(y, bias, y)
            torch.cuda.synchronize()
        call["output"] = tdesc(y)
        if reference is not None:
            count("reconstruction-path bias add: ExLlamaV3 add kernel equals PyTorch's", bits_equal(y, reference))
        return y
    LinearEXL3.reconstruct_hgemm = recon_hook

    # The fused gate/up multi-linear (rows <= 32), called directly by GatedMLP.
    class ExtProxy:
        def __getattr__(self, name):
            return getattr(ext, name)

        @staticmethod
        def exl3_mgemm(*args):
            with rec.op("multilinear", input=tdesc(args[0]), output=tdesc(args[2]), path="packed (exl3_mgemm)"):
                out = ext.exl3_mgemm(*args)
                torch.cuda.synchronize()
            return out
    mlp_module.ext = ExtProxy()

    # Attention as the variant runs it (KV write, then GGML's vector kernel).
    attend = dispatch.attn_fns[0]

    def attention_hook(args):
        detail = {"q": tdesc(args.q), "k_new": tdesc(args.k), "v_new": tdesc(args.v),
                  "past": int(args.cache_seqlens[0]), "q_len": int(args.q_len), "sm_scale": float(args.sm_scale),
                  "k_cache": tdesc(args.k_cache)}
        with rec.op("attention_block", **detail) as call:
            o = attend(args)
            torch.cuda.synchronize()
        call["output"] = tdesc(o) if o is not None else None
        return o
    dispatch.attn_fns = [attention_hook]

    mlp_forward = GatedMLP.forward

    def mlp_hook(self, x, params, out_dtype=None):
        with rec.op("mlp", key=self.key):
            return mlp_forward(self, x, params, out_dtype)
    GatedMLP.forward = mlp_hook

    # Embedding: GGML get_rows on the device table.
    emb_forward = Embedding.forward
    tables = {}

    def emb_hook(self, x, params, out_dtype=None):
        if "input_ids" not in params:
            params["input_ids"] = x
        if params.get("indexed_embeddings") or self.multiplier != 1.0 or self.normalize:
            raise NotImplementedError("plan: embedding configuration")
        if (out_dtype or self.out_dtype) != torch.float:
            raise NotImplementedError("plan: embedding output type")
        table = tables.get(id(self))
        if table is None:
            table = tables[id(self)] = self.embedding.weight.detach().to("cuda:0").contiguous()
        with rec.op("embedding", table=tdesc(table), host_table_device=str(self.embedding.weight.device)):
            y = ops.get_rows(table, x)
        if state["check"]:
            reference = emb_forward(self, x, params, out_dtype)
            count("embedding: GGML get_rows equals upstream's lookup", bits_equal(y.cpu(), reference.reshape(y.shape).cpu()))
        return y.view(*x.shape, y.shape[-1])
    Embedding.forward = emb_hook

    # The decoder block: upstream's order for this configuration, with the MLP
    # residual add in GGML.
    def block_hook(self, x, params, out_dtype=None):
        if (self.attn_hc or self.mlp_hc or self.attn_post_norm or self.mlp_post_norm or not self.attn_norm
                or not self.mlp_norm or self.attn_resid_scalar is not None or self.mlp_resid_scalar is not None
                or self.layer_scalar_f is not None or params.get("export_state_layers") or params.get("prefill")):
            raise NotImplementedError("plan: block configuration")
        with rec.op("block", layer=self.layer_idx):
            y = self.attn_norm.forward(x, params, out_dtype=torch.half)
            y = self.attn.forward(y, params)
            if not self.mlp_norm.can_fuse_residual(x, y):
                raise NotImplementedError("plan: residual fusion")
            params["residual"] = x
            y = self.mlp_norm.forward(y, params, out_dtype=torch.half, residual_in=x)
            y = self.mlp.forward(y, params)
            reference = x + y if state["check"] else None
            with rec.op("residual_add"):
                ops.add(x, y, out=x)
            if reference is not None:
                count("MLP residual add: GGML add equals upstream's", bits_equal(x, reference))
        return to2(x, out_dtype, self.out_dtype)
    TransformerBlock.forward = block_hook
    return ops


def trace_events(prof):
    """Every GPU activity in start order, with the innermost op# range of its launch."""
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        prof.export_chrome_trace(handle.name)
        events = json.loads(Path(handle.name).read_text())["traceEvents"]
    ranges = sorted(((e["ts"], e["ts"] + e["dur"], int(e["name"][3:])) for e in events
                     if e.get("cat") == "user_annotation" and e.get("name", "").startswith("op#")),
                    key=lambda r: r[1] - r[0])
    runtime = {e["args"]["correlation"]: e for e in events
               if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    out = []
    for e in sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                    key=lambda e: e["ts"]):
        args = e.get("args", {})
        call = runtime.get(args.get("correlation"))
        owner = None
        if call is not None:
            owner = next((seq for lo, hi, seq in ranges if lo <= call["ts"] <= hi), None)
        row = {"kind": e["cat"], "name": e["name"], "op": owner, "stream": args.get("stream"),
               "api": call["name"] if call is not None else None}
        if e["cat"] == "kernel":
            row.update(grid=args.get("grid"), block=args.get("block"), shared=args.get("shared memory"),
                       registers=args.get("registers per thread"))
        else:
            row["bytes"] = args.get("bytes")
        out.append(row)
    return out


def bias_exhaustive():
    """ExLlamaV3's add_kernel_hhh against PyTorch's F16 add on every F16 pair."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    b = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.float16)
    result = {"pairs": 0, "nan_result_pairs": 0, "bit_identical_non_nan": 0, "differing_non_nan": 0,
              "nan_disagreement": 0}
    rows = 2048
    for start in range(-32768, 32768, rows):
        a = torch.arange(start, start + rows, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.float16)
        x = a[:, None].expand(rows, 65536).contiguous()
        z = torch.empty_like(x)
        ext.add(x, b, z)
        ref = x + b[None, :]
        zn, rn = torch.isnan(z), torch.isnan(ref)
        both = zn & rn
        same = (z.view(torch.int16) == ref.view(torch.int16)) & ~rn & ~zn
        result["pairs"] += x.numel()
        result["nan_result_pairs"] += int(both.sum())
        result["nan_disagreement"] += int((zn ^ rn).sum())
        result["bit_identical_non_nan"] += int(same.sum())
        result["differing_non_nan"] += int((~zn & ~rn).sum()) - int(same.sum())
    torch.cuda.synchronize()
    return result


def probe(args):
    import numpy as np
    inherited = [k for k in os.environ if k.startswith("EXL3_")]
    if inherited:
        raise ValueError(f"undeclared inherited EXL3 overrides: {inherited}")
    os.environ.update({"EXL3_BC_ATTN": "0", "EXL3_GEMV": "0", "EXL3_HGEMM_F16ACC": "0"})
    tune_cache = os.environ["EXLLAMAV3_TUNE_CACHE"]
    tune_before = digest(tune_cache)
    if digest(args.ids) != HELD_OUT_SHA256:
        raise ValueError("held-out ID identity mismatch")
    import torch
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
    rec, checks, state = Recorder(), {}, {"check": False}
    ops = install(model, rec, checks, state)
    torch.cuda.synchronize()

    def params(p):
        return {"attn_mode": "flash_attn", "cache": cache, "past_len": p, "batch_shape": (1, CAPACITY),
                "pinned_staging": bool(p)}

    def zero():
        for tensor in cache.get_all_tensors():
            tensor.zero_()
        torch.cuda.synchronize()

    def run_phase(kind, p, profiled):
        """Returns (logits of the phase, trace events or None)."""
        zero()
        if kind == "step":
            model.forward(ids_all[:, :p], params(0))
            torch.cuda.synchronize()
            ids, par = ids_all[:, p:p + 1], params(p)
        else:
            ids, par = ids_all[:, :p], params(0)
        rec.reset()
        if not profiled:
            return model.forward(ids, par).float().cpu().numpy(), None
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with rec.op("phase", kind=kind, p=p):
                logits = model.forward(ids, par)
            torch.cuda.synchronize()
        return logits.float().cpu().numpy(), trace_events(prof)

    def reference(kind, p):
        ref = np.load(args.reference / f"logits-{p}.npz")
        return ref["prefill"] if kind == "prefill" else ref["suffix"][:, :1]

    def same(a, b):
        return a.shape == b.shape and a.tobytes() == b.tobytes()

    phases = []
    with torch.inference_mode():
        state["check"] = True
        checked = [same(run_phase(kind, p, False)[0], reference(kind, p)) for kind, p in PHASES]
        state["check"] = False
        for (kind, p), equal in zip(PHASES, checked):
            logits, events = run_phase(kind, p, True)
            n_kv = p + (1 if kind == "step" else 0)
            phases.append({"kind": kind, "p": p, "rows": 1 if kind == "step" else p, "past": p if kind == "step" else 0,
                           "n_kv": n_kv, "checked_run_logits_equal_reference": equal,
                           "profiled_logits_equal_reference": same(logits, reference(kind, p)),
                           "calls": rec.calls, "events": events})
            print("phase", kind, p, equal, phases[-1]["profiled_logits_equal_reference"], len(events), flush=True)
        exhaustive = bias_exhaustive() if args.bias_exhaustive else None
    library = os.environ["GGML_OPS_LIB"]
    lib_dir = os.path.dirname(library)
    import glob
    build = sorted(glob.glob(os.path.join(lib_dir, "**", "*.so.*"), recursive=True) + [library])
    record = {"fixture": fixture["repository"], "weight_sha256": fixture["published_sha256"],
              "reference": str(args.reference), "library": library, "library_build": ops.build_info,
              "library_sha256": {os.path.relpath(p, lib_dir): digest(p) for p in build if not os.path.islink(p)},
              "blas": __import__("exl3_heldout").blas_libraries(), "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
              "environment": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMAV3_", "CUDA_"))},
              "harness_sha256": {n: digest(Path(__file__).parent / n) for n in
                                 ("exl3_op_plan.py", "exl3_heldout.py", "ggml_ops.py")},
              "tune_cache_before_sha256": tune_before, "tune_cache_after_sha256": digest(tune_cache),
              "config": {"hidden": config.hidden_size, "heads": config.num_q_heads, "kv_heads": config.num_kv_heads,
                         "head_dim": config.head_dim, "intermediate": config.intermediate_size,
                         "vocab": config.vocab_size, "layers": config.num_hidden_layers,
                         "rms_norm_eps": config.rms_norm_eps,
                         "rope_theta": config.rope_settings.rope_theta},
              "checks": checks, "bias_add_exhaustive": exhaustive, "phases": phases}
    record["passed"] = (tune_before == record["tune_cache_after_sha256"]
                        and all(p["checked_run_logits_equal_reference"] and p["profiled_logits_equal_reference"]
                                for p in phases)
                        and all(c["cases"] == c["equal"] for c in checks.values()))
    args.output.write_text(json.dumps(record) + "\n")
    print("DONE", args.output, record["passed"], flush=True)


# --------------------------------------------------------------------------- build

# The plan's tensors between operations, for one decoder layer (n rows, P the
# first position, N = P + n attended positions, Npad = N rounded up to 256),
# and the embedding and the output once. Shapes are row-major, last dimension
# contiguous; the dtype is the element format in memory.
TENSORS = {
    "ids": ("int32", "[n]", "token IDs, host-built"),
    "positions": ("int32", "[n]", "P .. P+n-1, host-built (RoPE's position input)"),
    "mask": ("float16", "[n, Npad]", "row i, column j: 0 if j <= P+i, else -inf; host-built; padded cells masked"),
    "embed_table": ("bfloat16", "[151936, 896]", "the artifact's embedding, on the device"),
    "embed.out": ("float32", "[n, 896]", "the residual stream entering layer 0"),
    "resid.in": ("float32", "[n, 896]", "the layer's input: embed.out, or the previous layer's resid.out"),
    "attn_norm.w": ("float32", "[896]", "norm weight, widened exactly to F32"),
    "attn_norm.f32": ("float32", "[n, 896]", ""),
    "attn_norm.out": ("float16", "[n, 896]", "input of q/k/v_proj"),
    "q_proj.gemm": ("float16", "[n, 896]", "linear output before its bias"),
    "q_proj.bias": ("float16", "[896]", "artifact bias"),
    "q": ("float16", "[n, 896]", "viewed [n, 14, 64]"),
    "k_proj.gemm": ("float16", "[n, 128]", ""),
    "k_proj.bias": ("float16", "[128]", ""),
    "k": ("float16", "[n, 128]", "viewed [n, 2, 64]"),
    "v_proj.gemm": ("float16", "[n, 128]", ""),
    "v_proj.bias": ("float16", "[128]", ""),
    "v": ("float16", "[n, 128]", "viewed [n, 2, 64]; written to the V cache as is"),
    "rope_q.in": ("float32", "[n, 14, 64]", "q widened exactly"),
    "q_rope": ("float32", "[n, 14, 64]", "attention's Q; never rounded to F16"),
    "rope_k.in": ("float32", "[n, 2, 64]", "k widened exactly"),
    "k_rope.f32": ("float32", "[n, 2, 64]", ""),
    "k_rope": ("float16", "[n, 2, 64]", "written to the K cache"),
    "k_cache": ("float16", "[4096, 2, 64] per layer", "cells [P, P+n) written by kv_write.k; attention reads [0, Npad)"),
    "v_cache": ("float16", "[4096, 2, 64] per layer", "cells [P, P+n) written by kv_write.v; attention reads [0, Npad)"),
    "attn.f32": ("float32", "[n, 14, 64]", "flash_attn_ext output (GGML layout [64, 14, n] is the same memory)"),
    "attn.out": ("float16", "[n, 896]", "input of o_proj"),
    "o_proj.out": ("float32", "[n, 896]", ""),
    "resid.mid": ("float32", "[n, 896]", "resid.in + o_proj.out, in place"),
    "mlp_norm.w": ("float32", "[896]", ""),
    "mlp_norm.f32": ("float32", "[n, 896]", ""),
    "mlp_norm.out": ("float16", "[n, 896]", "input of gate/up"),
    "gate": ("float32", "[n, 4864]", "n <= 32: plane 0 of the multi-linear's [2, n, 4864] output"),
    "up": ("float32", "[n, 4864]", "n <= 32: plane 1"),
    "swiglu.f32": ("float32", "[n, 4864]", ""),
    "swiglu.out": ("float16", "[n, 4864]", "input of down_proj"),
    "down_proj.out": ("float32", "[n, 896]", ""),
    "resid.out": ("float32", "[n, 896]", "resid.mid + down_proj.out, in place"),
    "final_norm.w": ("float32", "[896]", ""),
    "final_norm.f32": ("float32", "[n, 896]", ""),
    "final_norm.out": ("float16", "[n, 896]", "input of lm_head"),
    "logits": ("float16", "[n, 151936]", "the plan's output: lm_head's F16 output; widening to F32 is exact and outside the plan"),
}

# The plan's operations in order. owner: ggml (the pinned GGML CUDA kernels,
# gated at operation level), exllamav3 (the linears and their bias add, gated
# with the linears) or device-copy. "when" restricts an operation to some
# phases.
GGML = "ggml"
OPERATIONS = {
    "embedding": [
        {"op": "embed", "owner": GGML, "ggml_op": "GGML_OP_GET_ROWS", "inputs": ["embed_table", "ids"],
         "outputs": ["embed.out"], "params": {"table": "BF16", "output": "F32 (exact widening)"}},
    ],
    "layer": [
        {"op": "attn_norm", "owner": GGML, "ggml_op": "GGML_OP_RMS_NORM + GGML_OP_MUL, fused by ggml-cuda",
         "inputs": ["resid.in", "attn_norm.w"], "outputs": ["attn_norm.f32"], "params": {"eps": "rms_norm_eps"}},
        {"op": "attn_norm.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F32 -> F16 (round to nearest even)",
         "inputs": ["attn_norm.f32"], "outputs": ["attn_norm.out"]},
        {"op": "q_proj", "owner": "exllamav3", "linear": "q_proj", "inputs": ["attn_norm.out"], "outputs": ["q_proj.gemm"]},
        {"op": "q_proj.bias_add", "owner": "exllamav3", "kernel": "add_kernel_hhh (add_gr)",
         "inputs": ["q_proj.gemm", "q_proj.bias"], "outputs": ["q"], "params": {"precision": "F16 + F16 -> F16, one round to nearest even"}},
        {"op": "k_proj", "owner": "exllamav3", "linear": "k_proj", "inputs": ["attn_norm.out"], "outputs": ["k_proj.gemm"]},
        {"op": "k_proj.bias_add", "owner": "exllamav3", "kernel": "add_kernel_hhh (add_gr)",
         "inputs": ["k_proj.gemm", "k_proj.bias"], "outputs": ["k"], "params": {"precision": "F16 + F16 -> F16, one round to nearest even"}},
        {"op": "v_proj", "owner": "exllamav3", "linear": "v_proj", "inputs": ["attn_norm.out"], "outputs": ["v_proj.gemm"]},
        {"op": "v_proj.bias_add", "owner": "exllamav3", "kernel": "add_kernel_hhh (add_gr)",
         "inputs": ["v_proj.gemm", "v_proj.bias"], "outputs": ["v"], "params": {"precision": "F16 + F16 -> F16, one round to nearest even"}},
        {"op": "rope_q.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F16 -> F32 (exact)", "inputs": ["q"], "outputs": ["rope_q.in"]},
        {"op": "rope_q", "owner": GGML, "ggml_op": "GGML_OP_ROPE (NEOX)", "inputs": ["rope_q.in", "positions"],
         "outputs": ["q_rope"], "params": "rope"},
        {"op": "rope_k.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F16 -> F32 (exact)", "inputs": ["k"], "outputs": ["rope_k.in"]},
        {"op": "rope_k", "owner": GGML, "ggml_op": "GGML_OP_ROPE (NEOX)", "inputs": ["rope_k.in", "positions"],
         "outputs": ["k_rope.f32"], "params": "rope"},
        {"op": "rope_k.cast_out", "owner": GGML, "ggml_op": "GGML_OP_CPY F32 -> F16 (round to nearest even)",
         "inputs": ["k_rope.f32"], "outputs": ["k_rope"]},
        {"op": "kv_write.k", "owner": "device-copy", "copy": "cudaMemcpyAsync device to device, n x 256 bytes, byte-exact",
         "inputs": ["k_rope"], "outputs": ["k_cache"], "params": {"cells": "[P, P+n)"}},
        {"op": "kv_write.v", "owner": "device-copy", "copy": "cudaMemcpyAsync device to device, n x 256 bytes, byte-exact",
         "inputs": ["v"], "outputs": ["v_cache"], "params": {"cells": "[P, P+n)"}},
        {"op": "attention.kv_max", "owner": GGML, "ggml_op": "GGML_OP_FLASH_ATTN_EXT (mask pre-pass inside launch_fattn)",
         "inputs": ["mask"], "outputs": ["(pool) KV_max"], "when": "n >= 1024 (fattn-common.cuh:1108; Npad is a multiple of 256)"},
        {"op": "attention", "owner": GGML, "ggml_op": "GGML_OP_FLASH_ATTN_EXT, vector kernel forced",
         "inputs": ["q_rope", "k_cache", "v_cache", "mask"], "outputs": ["attn.f32"], "params": "attention"},
        {"op": "attention.combine", "owner": GGML, "ggml_op": "GGML_OP_FLASH_ATTN_EXT (launch_fattn's combine of parallel blocks)",
         "inputs": ["(pool) partial results"], "outputs": ["attn.f32"], "when": "parallel_blocks > 1 (every recorded phase)"},
        {"op": "attention.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F32 -> F16 (round to nearest even)",
         "inputs": ["attn.f32"], "outputs": ["attn.out"]},
        {"op": "o_proj", "owner": "exllamav3", "linear": "o_proj", "inputs": ["attn.out"], "outputs": ["o_proj.out"]},
        {"op": "attn_residual_add", "owner": GGML, "ggml_op": "GGML_OP_ADD F32", "inputs": ["resid.in", "o_proj.out"],
         "outputs": ["resid.mid"]},
        {"op": "mlp_norm", "owner": GGML, "ggml_op": "GGML_OP_RMS_NORM + GGML_OP_MUL, fused by ggml-cuda",
         "inputs": ["resid.mid", "mlp_norm.w"], "outputs": ["mlp_norm.f32"], "params": {"eps": "rms_norm_eps"}},
        {"op": "mlp_norm.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F32 -> F16 (round to nearest even)",
         "inputs": ["mlp_norm.f32"], "outputs": ["mlp_norm.out"]},
        {"op": "gate_up", "owner": "exllamav3", "linear": "gate_proj + up_proj as one multi-linear (exl3_mgemm)",
         "inputs": ["mlp_norm.out"], "outputs": ["gate", "up"], "when": "n <= 32"},
        {"op": "gate_proj", "owner": "exllamav3", "linear": "gate_proj", "inputs": ["mlp_norm.out"], "outputs": ["gate"],
         "when": "n > 32"},
        {"op": "up_proj", "owner": "exllamav3", "linear": "up_proj", "inputs": ["mlp_norm.out"], "outputs": ["up"],
         "when": "n > 32"},
        {"op": "swiglu", "owner": GGML, "ggml_op": "GGML_OP_GLU (GGML_GLU_OP_SWIGLU, split inputs) F32",
         "inputs": ["gate", "up"], "outputs": ["swiglu.f32"]},
        {"op": "swiglu.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F32 -> F16 (round to nearest even)",
         "inputs": ["swiglu.f32"], "outputs": ["swiglu.out"]},
        {"op": "down_proj", "owner": "exllamav3", "linear": "down_proj", "inputs": ["swiglu.out"], "outputs": ["down_proj.out"]},
        {"op": "mlp_residual_add", "owner": GGML, "ggml_op": "GGML_OP_ADD F32", "inputs": ["resid.mid", "down_proj.out"],
         "outputs": ["resid.out"]},
    ],
    "output": [
        {"op": "final_norm", "owner": GGML, "ggml_op": "GGML_OP_RMS_NORM + GGML_OP_MUL, fused by ggml-cuda",
         "inputs": ["resid.out (layer 23)", "final_norm.w"], "outputs": ["final_norm.f32"], "params": {"eps": "rms_norm_eps"}},
        {"op": "final_norm.cast", "owner": GGML, "ggml_op": "GGML_OP_CPY F32 -> F16 (round to nearest even)",
         "inputs": ["final_norm.f32"], "outputs": ["final_norm.out"]},
        {"op": "lm_head", "owner": "exllamav3", "linear": "lm_head", "inputs": ["final_norm.out"], "outputs": ["logits"]},
    ],
}

# Which plan operation each kernel of a labelled call is, in launch order.
GGML_KERNEL_OPS = {
    "ggml.get_rows": {"k_get_rows_float": "embed"},
    "ggml.rope_neox": {"cpy_scalar_contiguous<__half, float>": "rope_{p}.cast", "rope_neox": "rope_{p}",
                       "cpy_scalar_contiguous<float, __half>": "rope_{p}.cast_out"},
    "ggml.attention": {"flash_attn_mask_to_KV_max": "attention.kv_max", "flash_attn_ext_vec": "attention",
                       "flash_attn_combine_results": "attention.combine",
                       "cpy_scalar_contiguous<float, __half>": "attention.cast"},
    "ggml.swiglu": {"unary_gated_op_kernel": "swiglu", "cpy_scalar_contiguous<float, __half>": "swiglu.cast"},
}


def internal(mangled):
    """A mangled name with NVCC's per-translation-unit _INTERNAL_<hash> prefix normalized."""
    return re.sub(r"_INTERNAL_[0-9a-f]{8}_", "_INTERNAL_x_", mangled)


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
    return dict(zip(names, out.splitlines()))


def kernel_family(name):
    """The kernel's name without its leading 'void ' and its parameter list."""
    base = name[5:] if name.startswith("void ") else name
    depth = 0
    for i, ch in enumerate(base):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return base[:i]
    return base


def build(args):
    probes = {}
    for item in args.probe:
        fixture, path = item.split("=", 1)
        probes[fixture] = json.loads(Path(path).read_text())
    sass_rows = [json.loads(line) for line in args.sass.read_text().splitlines() if line.strip()]
    mangled_by_name = {}
    for mangled, name in demangle([r["function"] for r in sass_rows]).items():
        mangled_by_name.setdefault(name, []).append(mangled)
    sass_by_mangled = {r["function"]: r for r in sass_rows}
    fp16 = json.loads(args.fp16_plan.read_text())
    fp16_by_mangled = {internal(k["mangled"]): k for k in fp16["kernels"]}

    probe_config = next(iter(probes.values()))["config"]
    kernels, kernel_ids = [], {}

    def kernel_id(name):
        if name in kernel_ids:
            return kernel_ids[name]
        entry = {"id": len(kernels), "name": name}
        family = kernel_family(name)
        mangled = mangled_by_name.get(name, [])
        if mangled:
            if len(mangled) != 1:
                raise ValueError(f"ambiguous SASS for {name}")
            row = sass_by_mangled[mangled[0]]
            entry.update(owner=GGML, mangled=mangled[0], sass_text_sha256=row["text_sha256"],
                         sass_encoding_sha256=row["encoding_sha256"], sass_instructions=row["instructions"])
            entry["mangled_normalized"] = internal(mangled[0])
            bridge = fp16_by_mangled.get(internal(mangled[0]))
            entry["fp16_bridge_sass"] = ("absent" if bridge is None else
                                         "identical" if bridge.get("sass_encoding_sha256") == row["encoding_sha256"]
                                         else "different")
        elif family.startswith(("nvjet_", "cutlass::")) or "cublasLt" in family:
            entry["owner"] = "cublas"
        else:
            entry["owner"] = "exllamav3"
        kernels.append(entry)
        kernel_ids[name] = entry["id"]
        return entry["id"]

    def launch(e):
        return [kernel_id(e["name"]), e["grid"], e["block"], e["shared"]]

    def reduce_phase(fixture, phase):
        """Plan operations with their launches, per segment; excluded transfers."""
        calls = {c["seq"]: c for c in phase["calls"]}

        def chain(seq):
            out = []
            while seq is not None:
                out.append(calls[seq])
                seq = calls[seq]["parent"]
            return out
        segments = {"embedding": [], "layers": {}, "output": []}
        excluded = {}
        norm_seen = {}
        observed = {}
        for e in phase["events"]:
            if e["op"] is None:
                raise ValueError(f"{fixture} {phase['kind']} {phase['p']}: unattributed {e['name']}")
            ch = chain(e["op"])
            cats = [c["category"] for c in ch]
            block = next((c for c in ch if c["category"] == "block"), None)
            if block is not None:
                layer = block["layer"]
                target = segments["layers"].setdefault(layer, [])
            elif "embedding" in cats:
                layer, target = None, segments["embedding"]
            else:
                layer, target = "out", segments["output"]
            inner = ch[0]
            if e["kind"] != "kernel":
                is_kv = (inner["category"] == "attention_block" and e["api"] == "cudaMemcpyAsync"
                         and e["name"].startswith("Memcpy DtoD"))
                if is_kv:
                    n = sum(1 for o in target if o[0].startswith("kv_write"))
                    target.append(("kv_write.k" if n == 0 else "kv_write.v", ["memcpy", "DtoD", e["bytes"]], inner))
                    continue
                where = inner["category"]
                key = f"{e['name']} [{where}]"
                excluded[key] = excluded.get(key, 0) + 1
                continue
            cat = inner["category"]
            if cat.startswith("ggml."):
                if cat == "ggml.rms_norm":
                    if (layer, inner["seq"]) not in norm_seen:
                        norm_seen[(layer, inner["seq"])] = len([k for k in norm_seen if k[0] == layer])
                    idx = norm_seen[(layer, inner["seq"])]
                    base = "final_norm" if layer == "out" else ("attn_norm", "mlp_norm")[idx]
                    op = base if e["name"].startswith("void rms_norm_f32") else f"{base}.cast"
                elif cat == "ggml.add":
                    op = "mlp_residual_add" if any(c["category"] == "residual_add" for c in ch) else "attn_residual_add"
                elif cat == "ggml.rope_neox":
                    part = "q" if inner["inputs"][0]["shape"][1] == probe_config["heads"] else "k"
                    op = next(v for k, v in GGML_KERNEL_OPS[cat].items() if k in e["name"]).format(p=part)
                else:
                    op = next(v for k, v in GGML_KERNEL_OPS[cat].items() if k in e["name"])
                observed.setdefault(op, inner)
            elif cat == "bias_add" or (cat == "linear" and e["name"].startswith("add_kernel_hhh")):
                key = inner["key"] if cat == "linear" else calls[inner["parent"]]["key"]
                op = key.rsplit(".", 1)[1] + ".bias_add"
            elif cat == "linear":
                op = "lm_head" if inner["key"] == "lm_head" else inner["key"].rsplit(".", 1)[1]
                observed.setdefault(op, inner)
            elif cat == "multilinear":
                op = "gate_up"
                observed.setdefault(op, inner)
            else:
                raise ValueError(f"unexpected kernel {e['name']} in {cat}")
            target.append((op, launch(e), inner))
        return segments, excluded, observed

    def group(entries):
        """[(op, launch)] -> ordered [{op, launches}] with consecutive launches merged per op."""
        out = []
        for op, ln, _ in entries:
            if out and out[-1]["op"] == op:
                out[-1]["launches"].append(ln)
            else:
                out.append({"op": op, "launches": [ln]})
        return out

    linear_ops = {o["op"] for seg in OPERATIONS.values() for o in seg if o["owner"] == "exllamav3" and "linear" in o}
    phase_kinds = []
    evidence = {}
    for idx, (kind, p) in enumerate(PHASES):
        entry = None
        for fixture, rec in probes.items():
            phase = rec["phases"][idx]
            assert (phase["kind"], phase["p"]) == (kind, p)
            segments, excluded, observed = reduce_phase(fixture, phase)
            layers = [group(segments["layers"][layer]) for layer in sorted(segments["layers"])]
            if len(layers) != rec["config"]["layers"]:
                raise ValueError("layer count")
            order = [o["op"] for o in layers[0]]
            for lay in layers:
                if [o["op"] for o in lay] != order:
                    raise ValueError(f"{fixture} {kind} {p}: layer operation order differs")
            n, past = phase["rows"], phase["past"]
            n_kv = past + n
            n_kv_pad = -(-n_kv // 256) * 256
            fixed = {}
            for i, op in enumerate(order):
                if op in linear_ops:
                    continue
                variants = {json.dumps(lay[i]["launches"]) for lay in layers}
                if len(variants) != 1:
                    raise ValueError(f"{fixture} {kind} {p}: {op} launches differ between layers")
                fixed[op] = layers[0][i]["launches"]
            per_layer = {}
            for i, op in enumerate(order):
                if op not in linear_ops:
                    continue
                by = {}
                for layer, lay in enumerate(layers):
                    by.setdefault(json.dumps(lay[i]["launches"]), []).append(layer)
                per_layer[op] = [{"layers": ls, "launches": json.loads(k)} for k, ls in by.items()]
            emb = group(segments["embedding"])
            out = group(segments["output"])
            linear_info = {op: {"path": c.get("path"), "output_dtype": c["output"]["dtype"],
                                "output_shape": c["output"]["shape"], "reconstruct_slices": c.get("reconstruct_slices")}
                           for op, c in observed.items() if op in linear_ops}
            fixture_part = {"linear_launches": per_layer, "lm_head_launches": [o for o in out if o["op"] == "lm_head"],
                            "linear_paths": linear_info,
                            "K_by_linear": {op: ks for op in sorted(linear_ops) if (ks := sorted(
                                {c["K"] for c in phase["calls"] if c["category"] == "linear"
                                 and c["key"].rsplit(".", 1)[-1] == op}))}}
            ggml_part = {"embedding": emb, "layer_order": order,
                         "layer_launches": {op: v for op, v in fixed.items()},
                         "output": [o for o in out if o["op"] != "lm_head"],
                         "output_order": [o["op"] for o in out]}
            attn_call = observed["attention"]
            params = {"n": n, "P": past, "N": n_kv, "Npad": n_kv_pad,
                      "vec_cols_per_block": 1 if n == 1 else 2,
                      "attention_pool_scratch_bytes": attn_call["scratch_pool_peak_bytes"]}
            vec = ggml_part["layer_launches"]["attention"][0]
            params["parallel_blocks"] = vec[1][1]
            if entry is None:
                entry = {"phase_kind": f"{kind} {p}", "kind": "prefill" if kind == "prefill" else "single-token step",
                         "rows": n, "first_position": past, "attended": n_kv, "padded_kv_length": n_kv_pad,
                         "shape_parameters": params, **ggml_part, "fixtures": {}}
            else:
                for key in ("embedding", "layer_order", "layer_launches", "output", "output_order"):
                    if entry[key] != ggml_part[key]:
                        raise ValueError(f"{kind} {p}: {key} differs between fixtures")
            entry["fixtures"][fixture] = fixture_part
            evidence.setdefault(fixture, {})[f"{kind} {p}"] = {
                "checked_run_logits_equal_reference_arm": phase["checked_run_logits_equal_reference"],
                "profiled_logits_equal_reference_arm": phase["profiled_logits_equal_reference"],
                "excluded_transfers": excluded}
        phase_kinds.append(entry)

    # Dtype chain: the probes' observed operation inputs and outputs against TENSORS.
    chain_checks = {"cases": 0, "equal": 0, "mismatches": []}

    def expect(what, got, dtype):
        chain_checks["cases"] += 1
        if got == dtype:
            chain_checks["equal"] += 1
        else:
            chain_checks["mismatches"].append(f"{what}: {got} != {dtype}")
    for fixture, rec in probes.items():
        for phase in rec["phases"]:
            for c in phase["calls"]:
                cat = c["category"]
                if cat == "linear":
                    op = c["key"].rsplit(".", 1)[-1]
                    expect(f"{op} input", c["input"]["dtype"], "float16")
                    want = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o_proj.out", "gate_proj": "gate",
                            "up_proj": "up", "down_proj": "down_proj.out", "lm_head": "logits"}[op]
                    expect(f"{op} output", c["output"]["dtype"], TENSORS[want][0])
                elif cat == "multilinear":
                    expect("gate_up input", c["input"]["dtype"], "float16")
                    expect("gate_up output", c["output"]["dtype"], TENSORS["gate"][0])
                elif cat == "bias_add":
                    expect("bias_add", (c["input"]["dtype"], c["bias"]["dtype"], c["output"]["dtype"]),
                           ("float16",) * 3)
                elif cat == "ggml.rms_norm":
                    expect("rms_norm", (c["inputs"][0]["dtype"], c["inputs"][1]["dtype"], c["output"]["dtype"]),
                           ("float32", "float32", "float16"))
                elif cat == "ggml.add":
                    expect("add", tuple(t["dtype"] for t in c["inputs"]) + (c["output"]["dtype"],), ("float32",) * 3)
                elif cat == "ggml.rope_neox":
                    q = c["inputs"][0]["shape"][1] == 14
                    expect("rope", (c["inputs"][0]["dtype"], c["scalars"][2], c["output"]["dtype"]),
                           ("float16", "float32", TENSORS["q_rope" if q else "k_rope"][0]))
                elif cat == "ggml.attention":
                    expect("attention", tuple(t["dtype"] for t in c["inputs"]) + (c["scalars"][3], c["output"]["dtype"]),
                           ("float32", "float16", "float16", "fa_vec", "float16"))
                elif cat == "ggml.swiglu":
                    expect("swiglu", tuple(t["dtype"] for t in c["inputs"]) + (c["output"]["dtype"],),
                           ("float32", "float32", "float16"))
                elif cat == "ggml.get_rows":
                    expect("get_rows", (c["inputs"][0]["dtype"], c["output"]["dtype"]), ("bfloat16", "float32"))
    if chain_checks["mismatches"]:
        raise ValueError(chain_checks["mismatches"][:5])
    del chain_checks["mismatches"]

    first = next(iter(probes.values()))
    record = {
        "recorded_on": "2026-09-26",
        "tool": "exl3_op_plan.py probe and build (this file's generator); fp16_plan.py sass-hash",
        "interpretation": (
            "The native EXL3-G operation plan, as a record: per phase kind, every operation of the embedding, one "
            "decoder layer (the same for all 24) and the output, in order, with its owner, its GGML operation, the "
            "tensors it reads and writes, and the kernels it launches (grid, block, shared memory). It was run, not "
            "only read: a probe executed exactly this plan inside upstream ExLlamaV3 (the ggml_ops arm plus GGML's "
            "embedding and MLP residual add and ExLlamaV3's bias add on the reconstruction path) on both fixtures, "
            "and its logits equal the ggml_ops arm's (cuBLAS 13.8.0.4) bit for bit in every phase kind. Reference "
            "only."),
        "matching_rule": [
            "Before any operation-level or model comparison, native's recorded executed plan must equal this record "
            "for every phase kind it runs: the same operations in the same order, each launching the same kernels in "
            "the same order with the same grid, block and shared memory; GGML kernels matched by mangled name (with "
            "NVCC's per-translation-unit _INTERNAL_<hash> prefix normalized, as in mangled_normalized: it differs "
            "between builds of the same source) and SASS encoding hash, ExLlamaV3 and cuBLAS kernels by demangled name (their builds are gated by "
            "exl3-launch.json and exl3-recon-pin.json). A phase is matched to the kind with its row count; a "
            "single-token step to the kind with its padded K length.",
            "Every tensor between operations has the dtype and shape in `tensors`. Each operation's recorded input must "
            "equal, byte for byte, the recorded output of the operation that produced it (`inputs` name the producer's "
            "tensor): a stale or mis-wired buffer fails even when every operation is individually exact.",
            "Only stream identity, device addresses, the launch API and the PDL attribute may differ. The probe's "
            "harness transfers (excluded_transfers: the shim's copies in and out of GGML buffers, its K/V staging and "
            "memsets, host uploads of IDs, positions and mask, PyTorch's bookkeeping copies) are not plan operations; "
            "native produces the declared input contents however it likes and records them.",
            "A change to any entry (kernel, launch, dtype, owner, order) needs a new record and approval."],
        "pins": {
            "ggml": "llama.cpp b29c606e28a01b1bc8c1351026a0fa6e616bf6c4, ggml/ subtree unmodified",
            "exllamav3": "6b84a21b (M0 reference image llmp-exl3-reference:20260922)",
            "device": f"{first['device']}, {first['sm_count']} SMs",
            "profile": "EXL3-G: EXL3_GEMV=0, EXL3_HGEMM_F16ACC=0, EXL3_BC_ATTN=0, frozen tune-40-gemvoff / tune-45-gemvoff (copies, unchanged)",
            "cublas": "13.8.0.4, bind-mounted over PyTorch's (cuBLASLt 130800, the only one mapped)",
            "ggml_build": "ggml_shim/build.sh cuda134: NVCC 13.4.92, sm_121a, -O3 -use_fast_math, GCC 13.3 host compiler, "
                          "shared libraries, GGML_CUDA_GRAPHS=OFF; libggml-cuda.so.0.24.0 SHA-256 "
                          + first["library_sha256"]["ggml/src/ggml-cuda/libggml-cuda.so.0.24.0"],
        },
        "model": first["config"],
        "provenance": {
            "kernel_launches": "PyTorch profiler (Kineto/CUPTI) around one run of each phase kind; each GPU activity is "
                               "attributed to the innermost labelled operation through the correlation id of the "
                               "runtime call that launched it. `shared` is the profiler's shared memory per block: "
                               "static plus dynamic. For GGML kernels `static_shared` comes from cuobjdump -res-usage "
                               "(its SHARED minus the 1 KiB it reports as reserved when any is used; flash_attn_ext_vec"
                               "<64, 2>: 5,632 - 1,024 = 4,608, the profiler's figure, with no dynamic shared memory, "
                               "as launch_fattn passes 0).",
            "sass": "cuobjdump 13.0.85 -sass on the cuda134 build's libggml-cuda.so.0.24.0, hashed as fp16_plan.py "
                    "sass-hash does (text without offsets and encodings; the 128-bit encodings in order), names "
                    "demangled with c++filt and matched to the profiler's. fp16_bridge_sass compares the encoding hash "
                    "with fp16-plan.json's kernel of the same normalized mangled name (the FP16 bridge: the SDK's Clang "
                    "host compiler, static libraries); absent means the bridge does not launch that kernel.",
            "probes": {fx: {k: rec[k] for k in ("fixture", "weight_sha256", "reference", "library_build", "library_sha256",
                                                "blas", "torch", "harness_sha256", "tune_cache_before_sha256",
                                                "tune_cache_after_sha256", "passed")}
                       for fx, rec in probes.items()},
        },
        "bias_add": {
            "owner": "ExLlamaV3's add_gr (add_kernel_hhh on F16) for q/k/v_proj on every path, as part of the linear and "
                     "gated with it (Tier E, EXL3 linears).",
            "precision": "linear output F16 (packed: exl3_gemm_kernel's F16 output; reconstruction: cuBLASLt HSH), bias "
                         "F16, sum rounded once to nearest even, F16 out.",
            "why": "Upstream already adds the bias this way inside its packed linear (BC_LinearEXL3::run -> add_gr), in "
                   "F16. Its reconstruction path adds it with PyTorch's elementwise kernel (y += bias: F16 operands, "
                   "F32 arithmetic, one rounding). The two are the same function: ExLlamaV3's kernel equals PyTorch's "
                   "on all 2^32 F16 input pairs (every non-NaN result bit-identical, NaN exactly where PyTorch gives "
                   "NaN), and on every reconstruction-path bias add of both probes. Keeping ExLlamaV3's kernel gives "
                   "the bias add one owner and one precision on every path, changes no bit of upstream's result, and "
                   "keeps it with the linear it belongs to. A GGML F32 add would need F32 linear outputs, a different "
                   "dtype chain from upstream's and a separate accuracy case; it was not adopted.",
            "exhaustive": first["bias_add_exhaustive"],
            "probe_checks": {fx: rec["checks"] for fx, rec in probes.items()},
        },
        "attention": {
            "kernel": "the vector kernel for every phase, launched through ggml_cuda_flash_attn_ext_vec_case<64, "
                      "GGML_TYPE_F16, GGML_TYPE_F16> (fattn-vec.cuh:545): flash_attn_ext_vec<64, cols_per_block, F16, "
                      "F16, false> with cols_per_block 1 for one query row and 2 otherwise, 128 threads (4 warps), "
                      "nbatch_fa = 64, parallel_blocks from launch_fattn's occupancy search over the 48 SMs "
                      "(fattn-common.cuh:1127-1195), then flash_attn_combine_results<64> when parallel_blocks > 1. "
                      "For n >= 1024 launch_fattn first runs flash_attn_mask_to_KV_max<2>, which finds for each "
                      "pair of query rows the last 256-position KV tile with an unmasked cell; the vector kernel "
                      "stops there. GGML's own choice would be the MMA kernel for every "
                      "prefill; the plan never takes it.",
            "parameters": {"scale": "head_dim^-1/2 = 0.125", "max_bias": 0.0, "logit_softcap": 0.0,
                           "precision_flag": "GGML_PREC_F32 set as llama.cpp does; no CUDA kernel reads it",
                           "Q": "q_rope, F32, GGML view [64, n, 14] (nb1 = 14 x 64 x 4 B, nb2 = 64 x 4 B)",
                           "K, V": "the layer's F16 cache cells [0, Npad), GGML view [64, Npad, 2] (nb1 = 256 B per "
                                   "position, nb2 = 128 B per KV head); GQA 7",
                           "mask": "F16 [n, Npad], row stride Npad x 2 B"},
            "padding": "Every phase, prefill and single-token alike, attends K and V padded to Npad = N rounded up to a "
                       "multiple of 256 (FATTN_KQ_STRIDE; llama.cpp pads its cache the same way). The recorded phase "
                       "kinds cover N = 32, 144, 145 (Npad 256), 1,023 and 1,024 (Npad 1,024), and single-token steps "
                       "with N 33-161 (Npad 256) and 1,025-1,040 (Npad 1,280), all the trajectories reach.",
            "padded_cells": "Declared content: zero, what the reference holds (the shim zero-fills its padded K/V; "
                            "ExLlamaV3's cache, zeroed before each trajectory, holds zero past the written cells). Finite "
                            "is the hard requirement: a padded cell's score is masked to exactly -inf, so a finite K "
                            "and V there are weighted by exactly 0, while an Inf or NaN would poison the row. Native "
                            "records the padded cells it passes as part of the attention's input; the operation-level "
                            "recomputation uses those bytes.",
            "mask_construction": "F16, n rows by Npad columns: 0 where column j <= P + i (row i is position P + i), "
                                 "-inf elsewhere, including every padded column. The reference builds it on the host "
                                 "(ggml_shim.cu); native may build it anywhere, but its bytes are these.",
            "scratch": "partial results in GGML's pool: parallel_blocks x the output's F32 elements plus metadata; per "
                       "phase in shape_parameters.attention_pool_scratch_bytes (the shim's counting pool). Scores are "
                       "never materialized.",
        },
        "kv_write": "After RoPE, layer l's K (k_rope, F16) and V (v, F16) rows are copied byte for byte into cells "
                    "[P, P+n) of layer l's K and V cache (ExLlamaV3's paged layout, 256-position pages, contiguous "
                    "pages 0..15, so a cell is [2, 64] F16 = 256 B), K first; then attention reads cells [0, Npad).",
        "rope": {"mode": "GGML_ROPE_TYPE_NEOX", "n_dims": 64, "freq_base": "rope_theta = 1e6", "freq_scale": 1.0,
                 "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0, "n_ctx_orig": 32768,
                 "positions": "P .. P+n-1"},
        "tensors": {k: {"dtype": v[0], "shape": v[1], **({"note": v[2]} if v[2] else {})} for k, v in TENSORS.items()},
        "operations": OPERATIONS,
        "kernels": kernels,
        "launch_format": "[kernel id, grid, block, shared bytes per block (static + dynamic)]; kv_write: "
                         "[\"memcpy\", \"DtoD\", bytes]",
        "phase_kinds": phase_kinds,
        "evidence": evidence,
        "dtype_chain_checks": chain_checks,
    }
    for k in kernels:
        if k.get("owner") == GGML:
            res = RES.get(k["mangled"]) if RES else None
            if res is not None:
                k["registers"], shared = res
                k["static_shared"] = shared - 1024 if shared else 0
    from fp16_plan import dump
    args.out.write_text(dump(record) + "\n")
    print("wrote", args.out, len(kernels), "kernels")


RES = {}


def load_res_usage(path):
    """cuobjdump -res-usage: {mangled: (registers, SHARED)}."""
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("Function ") and line.endswith(":"):
            current = line[len("Function "):-1]
        elif current and line.startswith("REG:"):
            fields = dict(f.split(":", 1) for f in line.split() if ":" in f)
            RES[current] = (int(fields["REG"]), int(fields["SHARED"]))
            current = None


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--pins", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="the ggml_ops arm run to compare logits with")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bias-exhaustive", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--probe", nargs="+", required=True, help="FIXTURE=probe.json")
    b.add_argument("--sass", type=Path, required=True)
    b.add_argument("--fp16-plan", type=Path, required=True)
    b.add_argument("--res-usage", type=Path, required=True, help="cuobjdump -res-usage of the same library")
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "probe":
        probe(args)
    else:
        load_res_usage(args.res_usage)
        build(args)


if __name__ == "__main__":
    main()
