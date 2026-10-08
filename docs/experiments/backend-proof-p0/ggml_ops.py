#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper over the reference-only GGML operation shim (ggml_shim/).

External reference tooling for the backend proof's P0; it does not implement
llmpalooza inference. Each method runs one GGML CUDA operation on PyTorch CUDA
tensors and returns a new tensor. The shim synchronizes the device on entry
and exit, so no stream coordination is needed. Layouts follow ExLlamaV3's:
rows are tokens, the last dimension is contiguous; attention takes the F16
cache as [positions, KV heads, head dim].

Attention paths (ggml_shim.cu, attn_path):
  fa_auto            ggml_flash_attn_ext with K/V padded to 256 positions, as
                     llama.cpp pads its cache; GGML selects the kernel
  fa_auto_unpadded   the same without padding
  fa_vec, fa_tile    the vector or tile kernel's launcher, forced (padded)
  nonfa_llama        llama.cpp's non-flash graph: KQ at GGML_PREC_F32,
                     soft_max_ext, KQV at default precision; the cuBLAS
                     handle as GGML sets it up (CUBLAS_TF32_TENSOR_OP_MATH)
  nonfa_f32          both products at GGML_PREC_F32
  nonfa_f32_notf32, nonfa_llama_notf32
                     the same with the handle's math mode CUBLAS_DEFAULT_MATH
"""

import ctypes
import json

import torch

F32, F16, BF16 = 0, 1, 30
TYPES = {torch.float32: F32, torch.float16: F16, torch.bfloat16: BF16}
PATHS = {"fa_auto": 0, "fa_auto_unpadded": 1, "fa_vec": 2, "fa_tile": 3, "nonfa_llama": 4, "nonfa_f32": 5,
         "nonfa_f32_notf32": 6, "nonfa_llama_notf32": 7}
ROUTES = {1: "mmvf", 2: "mmf", 3: "cublas"}


class GGMLOps:
    def __init__(self, library, device=0):
        lib = self.lib = ctypes.CDLL(str(library))
        p, i64, i32, f32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float
        lib.ggml_shim_last_error.restype = ctypes.c_char_p
        lib.ggml_shim_build_info.restype = ctypes.c_char_p
        lib.ggml_shim_init.argtypes = [i32]
        lib.ggml_shim_rms_norm.argtypes = [p, p, p, i32, i64, i64, f32, i32]
        lib.ggml_shim_add.argtypes = [p, p, p, i64]
        lib.ggml_shim_rope_neox.argtypes = [p, i32, i32, p, i32, i64, i64, i64, i64, f32, i32]
        lib.ggml_shim_swiglu.argtypes = [p, p, i32, p, i32, i64, i64]
        lib.ggml_shim_get_rows.argtypes = [p, i32, i64, i64, p, i64, p]
        lib.ggml_shim_attention.argtypes = [p, p, p, p, i32, i64, i64, i64, i64, i64, i64, f32, i32]
        lib.ggml_shim_nonfa_route.argtypes = [i64, i64, i64, i64, i32]
        lib.ggml_shim_last_scratch.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        lib.ggml_shim_last_scratch.restype = None
        torch.cuda.init()
        self._check(lib.ggml_shim_init(device))
        self.build_info = json.loads(lib.ggml_shim_build_info().decode())

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError(f"ggml_shim: {self.lib.ggml_shim_last_error().decode()}")

    @staticmethod
    def _ptr(t):
        return ctypes.c_void_p(t.data_ptr())

    def last_scratch(self):
        """(pool peak bytes, materialized intermediate bytes) of the last call."""
        pool, inter = ctypes.c_size_t(), ctypes.c_size_t()
        self.lib.ggml_shim_last_scratch(ctypes.byref(pool), ctypes.byref(inter))
        return pool.value, inter.value

    def rms_norm(self, x, w, eps, out_dtype=torch.float16, fused=True):
        x = x.reshape(-1, x.shape[-1]).float().contiguous()
        w = w.float().contiguous()
        y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        self._check(self.lib.ggml_shim_rms_norm(self._ptr(x), self._ptr(w), self._ptr(y), TYPES[out_dtype],
                                                x.shape[0], x.shape[1], eps, int(fused)))
        return y

    def add(self, a, b, out=None):
        a, b = a.float().contiguous(), b.float().contiguous()
        out = torch.empty_like(a) if out is None else out
        assert out.is_contiguous() and out.dtype == torch.float32 and out.numel() == a.numel() == b.numel()
        self._check(self.lib.ggml_shim_add(self._ptr(a), self._ptr(b), self._ptr(out), a.numel()))
        return out

    def rope_neox(self, x, pos0, freq_base, compute_dtype, out_dtype, n_ctx_orig=32768):
        """x [n_tok, n_head, dim]; positions pos0 + i."""
        x = x.contiguous()
        n_tok, n_head, dim = x.shape[-3:]
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        self._check(self.lib.ggml_shim_rope_neox(self._ptr(x), TYPES[x.dtype], TYPES[compute_dtype], self._ptr(out),
                                                 TYPES[out_dtype], n_tok, n_head, dim, pos0, freq_base, n_ctx_orig))
        return out

    def swiglu(self, g, u, out_dtype):
        g, u = g.contiguous(), u.to(g.dtype).contiguous()
        rows, cols = g.numel() // g.shape[-1], g.shape[-1]
        out = torch.empty(g.shape, dtype=out_dtype, device=g.device)
        self._check(self.lib.ggml_shim_swiglu(self._ptr(g), self._ptr(u), TYPES[g.dtype], self._ptr(out),
                                              TYPES[out_dtype], rows, cols))
        return out

    def get_rows(self, table, ids):
        table = table.contiguous()
        ids32 = ids.reshape(-1).to(torch.int32).cpu().contiguous()
        out = torch.empty((ids32.numel(), table.shape[1]), dtype=torch.float32, device=table.device)
        self._check(self.lib.ggml_shim_get_rows(self._ptr(table), TYPES[table.dtype], table.shape[0], table.shape[1],
                                                ctypes.c_void_p(ids32.data_ptr()), ids32.numel(), self._ptr(out)))
        return out

    def cast(self, x, out_dtype):
        """x converted to out_dtype by GGML's cpy (F32 to F16, or F16 to F32), a new tensor.

        The symbol is looked up here, not in __init__, so libraries built before it (cuda134)
        still load."""
        fn = self.lib.ggml_shim_cpy
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]
        x = x.contiguous()
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        self._check(fn(self._ptr(x), TYPES[x.dtype], self._ptr(out), TYPES[out_dtype], x.numel()))
        return out

    def attention(self, q, k, v, n_kv, q_pos0, scale, path, out_dtype=torch.float32):
        """q [n_q, n_head, dim] (converted to F32 exactly); k, v [>= n_kv, n_head_kv, dim] F16."""
        q = q.float().contiguous()
        assert k.dtype == v.dtype == torch.float16 and k.is_contiguous() and v.is_contiguous()
        n_q, n_head, dim = q.shape
        out = torch.empty((n_q, n_head, dim), dtype=out_dtype, device=q.device)
        self._check(self.lib.ggml_shim_attention(self._ptr(q), self._ptr(k), self._ptr(v), self._ptr(out),
                                                 TYPES[out_dtype], n_q, n_kv, n_head, k.shape[-2], dim, q_pos0,
                                                 scale, PATHS[path]))
        return out

    def nonfa_route(self, n_q, n_kv, n_head_kv, dim):
        """The matrix-multiply routes GGML takes for the non-flash KQ and KQV products."""
        return {which: ROUTES[self.lib.ggml_shim_nonfa_route(n_q, n_kv, n_head_kv, dim, i)]
                for i, which in enumerate(("kq", "kqv"))}
