// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
//
// Reference-only C API over the pinned GGML CUDA backend (../ggml-ops.json,
// ../ggml_ops.py). Each call runs one GGML operation on device buffers owned
// by the caller (PyTorch): the inputs are copied device-to-device into
// tensors of a GGML CUDA buffer, the operation runs on GGML's backend stream,
// and the result is copied back. Every call synchronizes the device on entry
// and on exit, so it needs no stream coordination with the caller. It is an
// external measurement tool, not llmpalooza's dispatch (D-053): operations run
// either through ggml_backend_graph_compute on a one-operation graph, which
// takes GGML's own launch selection, or, where a path is forced, by calling
// the selected launcher directly with the backend's context.
//
// Errors inside GGML abort, as they do upstream; the shim's own argument
// checks return a negative code with ggml_shim_last_error() set.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "common.cuh"

// Launchers that are external in ggml-cuda but not declared in a public
// header (fattn-tile.cuh, fattn-vec.cuh, mmvf.cuh, mmf.cuh).
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
extern template void ggml_cuda_flash_attn_ext_vec_case<64, GGML_TYPE_F16, GGML_TYPE_F16>(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);
bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne,
                              const size_t * src0_nb, const int src1_ncols, bool mul_mat_id);

namespace {

ggml_backend_t g_backend = nullptr;
ggml_gallocr_t g_galloc = nullptr;
std::string g_error;

// Operation scratch: a ggml_cuda_pool installed in place of GGML's VMM pool
// (the context's pool member is public and created only when absent). It
// counts the bytes live at once during a call; allocations are cudaMalloc'd
// and freed only after the call's closing device synchronization, since GGML
// returns pool memory as the host launcher returns.
struct counting_pool : ggml_cuda_pool {
    size_t live = 0;
    size_t peak = 0;
    std::vector<void *> retired;
    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        *actual_size = size;
        live += size;
        peak = std::max(peak, live);
        return ptr;
    }
    void free(void * ptr, size_t size) override {
        live -= size;
        retired.push_back(ptr);
    }
    void release() {
        for (void * ptr : retired) CUDA_CHECK(cudaFree(ptr));
        retired.clear();
    }
};
counting_pool * g_pool = nullptr;
size_t g_intermediate_bytes = 0;

// Attention paths (ggml_shim_attention).
enum attn_path : int {
    FA_AUTO_PADDED = 0,    // ggml_flash_attn_ext, K/V padded to 256 as llama.cpp does; GGML selects the kernel
    FA_AUTO_UNPADDED = 1,  // the same without padding (selects the MMA kernel on GB10)
    FA_VEC = 2,            // the vector kernel's launcher, forced (padded)
    FA_TILE = 3,           // the tile kernel's launcher, forced (padded)
    NONFA_LLAMA = 4,       // mul_mat(K, Q) at GGML_PREC_F32, soft_max_ext, mul_mat(V^T, P) at default precision
    NONFA_F32 = 5,         // both products at GGML_PREC_F32
    NONFA_F32_NO_TF32 = 6, // as NONFA_F32 with the cuBLAS handle's math mode set to CUBLAS_DEFAULT_MATH
    NONFA_LLAMA_NO_TF32 = 7,
};

int fail(const std::string & message) {
    g_error = message;
    return -1;
}

int check_cuda(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        return fail(std::string(what) + ": " + cudaGetErrorString(status));
    }
    return 0;
}

bool is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}

ggml_backend_cuda_context & cuda_ctx() {
    return *static_cast<ggml_backend_cuda_context *>(g_backend->context);
}

// A ggml context for one call's tensor and graph metadata.
struct call_ctx {
    ggml_context * ctx = nullptr;
    call_ctx() {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
    }
    ~call_ctx() { ggml_free(ctx); }
};

ggml_tensor * input(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_input(t);
    return t;
}

// Allocates the graph's tensors and records the bytes of its materialized
// intermediates: every non-view node except the last (the output).
int alloc(ggml_cgraph * graph) {
    if (!ggml_gallocr_alloc_graph(g_galloc, graph)) {
        return fail("ggml_gallocr_alloc_graph failed");
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
    g_intermediate_bytes = 0;
    for (int i = 0; i + 1 < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (!is_view(node)) g_intermediate_bytes += ggml_backend_buft_get_alloc_size(buft, const_cast<ggml_tensor *>(node));
    }
    return 0;
}

// Device-to-device cudaMemcpy is asynchronous with respect to the host and
// runs on the legacy default stream, which GGML's non-blocking streams do not
// wait for; compute() and the forced launchers therefore synchronize the
// device before launching.
int copy_in(ggml_tensor * t, const void * src) {
    return check_cuda(cudaMemcpy(t->data, src, ggml_nbytes(t), cudaMemcpyDeviceToDevice), "copy in");
}

int copy_out(void * dst, const ggml_tensor * t) {
    return check_cuda(cudaMemcpy(dst, t->data, ggml_nbytes(t), cudaMemcpyDeviceToDevice), "copy out");
}

int compute(ggml_cgraph * graph) {
    if (check_cuda(cudaDeviceSynchronize(), "inputs")) return -1;
    if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS) {
        return fail("ggml_backend_graph_compute failed");
    }
    return 0;
}

int sync_all(const char * what) {
    const int rc = check_cuda(cudaDeviceSynchronize(), what);
    if (rc == 0 && g_pool != nullptr) {
        g_pool->release();
        if (std::strstr(what, "entry") != nullptr) {
            g_pool->peak = g_pool->live;
            g_intermediate_bytes = 0;
        }
    }
    return rc;
}


// Result tensor converted to the requested output type (a GGML cpy, round to
// nearest for F32 to F16) or returned as is.
ggml_tensor * as_type(ggml_context * ctx, ggml_tensor * t, int type) {
    if (t->type == (ggml_type) type) {
        return t;
    }
    ggml_tensor * out = ggml_new_tensor(ctx, (ggml_type) type, GGML_MAX_DIMS, t->ne);
    return ggml_cpy(ctx, t, out);
}

}  // namespace

extern "C" {

const char * ggml_shim_last_error(void) {
    return g_error.c_str();
}

const char * ggml_shim_build_info(void) {
    static std::string info;
    if (info.empty()) {
        char buf[512];
        std::snprintf(buf, sizeof buf, "{\"nvcc\": \"%d.%d.%d\", \"cuda_archs\": \"%s\", \"ggml_source\": \"%s\", \"cudart_runtime\": %d}",
                      __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__, GGML_SHIM_ARCHS,
                      GGML_SHIM_SOURCE_ID, [] { int v = 0; cudaRuntimeGetVersion(&v); return v; }());
        info = buf;
    }
    return info.c_str();
}

int ggml_shim_init(int device) {
    if (g_backend != nullptr) {
        return 0;
    }
    g_backend = ggml_backend_cuda_init(device);
    if (g_backend == nullptr) {
        return fail("ggml_backend_cuda_init failed");
    }
    g_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (g_galloc == nullptr) return fail("ggml_gallocr_new failed");
    ggml_backend_cuda_context & ctx = cuda_ctx();
    if (ctx.pools[ctx.device][ctx.curr_stream_no] != nullptr) return fail("pool already created");
    auto pool = std::make_unique<counting_pool>();
    g_pool = pool.get();
    ctx.pools[ctx.device][ctx.curr_stream_no] = std::move(pool);
    return 0;
}

// The last call's operation scratch: the pool's peak bytes and the graph's
// materialized intermediates (see alloc).
void ggml_shim_last_scratch(size_t * pool_peak, size_t * intermediates) {
    *pool_peak = g_pool != nullptr ? g_pool->peak : 0;
    *intermediates = g_intermediate_bytes;
}

// y[rows, cols] = rms_norm(x, eps) * w, x F32 [rows, cols], w F32 [cols]; y F32
// or F16. fused: one graph (the CUDA backend fuses rms_norm and mul);
// otherwise rms_norm and mul run as two graphs.
int ggml_shim_rms_norm(const void * x, const void * w, void * y, int y_type, int64_t rows, int64_t cols, float eps,
                       int fused) {
    if (sync_all("rms_norm entry")) return -1;
    call_ctx c;
    ggml_tensor * tx = input(c.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * tw = input(c.ctx, GGML_TYPE_F32, cols);
    ggml_tensor * norm = ggml_rms_norm(c.ctx, tx, eps);
    ggml_tensor * mul = ggml_mul(c.ctx, norm, tw);
    ggml_tensor * out = as_type(c.ctx, mul, y_type);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    if (alloc(graph) || copy_in(tx, x) || copy_in(tw, w)) return -1;
    if (fused) {
        if (compute(graph)) return -1;
    } else {
        // One node at a time: the graph compute loop can fuse only within a call.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_cgraph one = ggml_graph_view(graph, i, i + 1);
            if (compute(&one)) return -1;
        }
    }
    if (copy_out(y, out)) return -1;
    return sync_all("rms_norm exit");
}

// out = a + b, all F32 with n elements.
int ggml_shim_add(const void * a, const void * b, void * out, int64_t n) {
    if (sync_all("add entry")) return -1;
    call_ctx c;
    ggml_tensor * ta = input(c.ctx, GGML_TYPE_F32, n);
    ggml_tensor * tb = input(c.ctx, GGML_TYPE_F32, n);
    ggml_tensor * sum = ggml_add(c.ctx, ta, tb);
    ggml_set_output(sum);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, sum);
    if (alloc(graph) || copy_in(ta, a) || copy_in(tb, b) || compute(graph) || copy_out(out, sum)) return -1;
    return sync_all("add exit");
}

// NEOX RoPE over x [n_tok, n_head, dim] (row-major, dim fastest), positions
// pos0 + i, Qwen2's parameters (theta = freq_base, no scaling). compute_type
// F16 runs the rope kernel on F16 data; F32 converts the input to F32 first.
int ggml_shim_rope_neox(const void * x, int in_type, int compute_type, void * out, int out_type, int64_t n_tok,
                        int64_t n_head, int64_t dim, int64_t pos0, float freq_base, int n_ctx_orig) {
    if (sync_all("rope entry")) return -1;
    call_ctx c;
    ggml_tensor * tx = input(c.ctx, (ggml_type) in_type, dim, n_head, n_tok);
    ggml_tensor * pos = input(c.ctx, GGML_TYPE_I32, n_tok);
    ggml_tensor * src = as_type(c.ctx, tx, compute_type);
    ggml_tensor * r = ggml_rope_ext(c.ctx, src, pos, nullptr, (int) dim, GGML_ROPE_TYPE_NEOX, n_ctx_orig, freq_base,
                                    1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * o = as_type(c.ctx, r, out_type);
    ggml_set_output(o);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, o);
    if (alloc(graph) || copy_in(tx, x)) return -1;
    std::vector<int32_t> positions(n_tok);
    for (int64_t i = 0; i < n_tok; ++i) positions[i] = (int32_t) (pos0 + i);
    if (check_cuda(cudaMemcpy(pos->data, positions.data(), n_tok * sizeof(int32_t), cudaMemcpyHostToDevice), "pos")) return -1;
    if (compute(graph) || copy_out(out, o)) return -1;
    return sync_all("rope exit");
}

// out[rows, cols] = silu(g) * u (GGML's swiglu on split inputs).
int ggml_shim_swiglu(const void * g, const void * u, int in_type, void * out, int out_type, int64_t rows, int64_t cols) {
    if (sync_all("swiglu entry")) return -1;
    call_ctx c;
    ggml_tensor * tg = input(c.ctx, (ggml_type) in_type, cols, rows);
    ggml_tensor * tu = input(c.ctx, (ggml_type) in_type, cols, rows);
    ggml_tensor * s = ggml_swiglu_split(c.ctx, tg, tu);
    ggml_tensor * o = as_type(c.ctx, s, out_type);
    ggml_set_output(o);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, o);
    if (alloc(graph) || copy_in(tg, g) || copy_in(tu, u) || compute(graph) || copy_out(out, o)) return -1;
    return sync_all("swiglu exit");
}

// out[n_ids, cols] F32 = table[ids] with table [n_rows, cols] of table_type;
// ids are host int32.
int ggml_shim_get_rows(const void * table, int table_type, int64_t n_rows, int64_t cols, const int32_t * ids,
                       int64_t n_ids, void * out) {
    if (sync_all("get_rows entry")) return -1;
    call_ctx c;
    ggml_tensor * tt = input(c.ctx, (ggml_type) table_type, cols, n_rows);
    ggml_tensor * ti = input(c.ctx, GGML_TYPE_I32, n_ids);
    ggml_tensor * r = ggml_get_rows(c.ctx, tt, ti);
    ggml_tensor * o = as_type(c.ctx, r, GGML_TYPE_F32);
    ggml_set_output(o);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, o);
    if (alloc(graph) || copy_in(tt, table)) return -1;
    if (check_cuda(cudaMemcpy(ti->data, ids, n_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "ids")) return -1;
    if (compute(graph) || copy_out(out, o)) return -1;
    return sync_all("get_rows exit");
}

// out[n] = x[n] converted by a GGML cpy (the plan's casts, P3's operation-level
// check): F32 to F16 rounds to nearest even, F16 to F32 is exact.
int ggml_shim_cpy(const void * x, int in_type, void * out, int out_type, int64_t n) {
    if (!((in_type == GGML_TYPE_F32 && out_type == GGML_TYPE_F16) ||
          (in_type == GGML_TYPE_F16 && out_type == GGML_TYPE_F32))) {
        return fail("cpy: F32 to F16 or F16 to F32 only");
    }
    if (sync_all("cpy entry")) return -1;
    call_ctx c;
    ggml_tensor * tx = input(c.ctx, (ggml_type) in_type, n);
    ggml_tensor * o = as_type(c.ctx, tx, out_type);
    ggml_set_output(o);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, o);
    if (alloc(graph) || copy_in(tx, x) || compute(graph) || copy_out(out, o)) return -1;
    return sync_all("cpy exit");
}

// Causal attention of n_q queries at positions q_pos0 .. q_pos0 + n_q - 1 over
// the first n_kv cache positions. q [n_q, n_head, dim] F32, k and v
// [n_kv, n_head_kv, dim] F16 (the cache layout), out [n_q, n_head, dim] F32 or
// F16. The padded paths zero-fill K/V to a multiple of 256 positions and mask
// the padding, as llama.cpp's KV cache does. path: attn_path.
int ggml_shim_attention(const void * q, const void * k, const void * v, void * out, int out_type, int64_t n_q,
                        int64_t n_kv, int64_t n_head, int64_t n_head_kv, int64_t dim, int64_t q_pos0, float scale,
                        int path) {
    if (sync_all("attention entry")) return -1;
    if (q_pos0 + n_q > n_kv) return fail("queries beyond the cache");
    const bool fa = path <= FA_TILE;
    const int64_t n_kv_pad = path == FA_AUTO_UNPADDED ? n_kv : GGML_PAD(n_kv, 256);
    call_ctx c;
    ggml_tensor * tq = input(c.ctx, GGML_TYPE_F32, dim, n_head, n_q);
    ggml_tensor * tk = input(c.ctx, GGML_TYPE_F16, dim, n_head_kv, n_kv_pad);
    ggml_tensor * tv = input(c.ctx, GGML_TYPE_F16, dim, n_head_kv, n_kv_pad);
    ggml_tensor * mask = input(c.ctx, fa ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv_pad, n_q);
    ggml_tensor * qp = ggml_permute(c.ctx, tq, 0, 2, 1, 3);  // [dim, n_q, n_head]
    ggml_tensor * kp = ggml_permute(c.ctx, tk, 0, 2, 1, 3);  // [dim, n_kv, n_head_kv]
    ggml_tensor * vp = ggml_permute(c.ctx, tv, 0, 2, 1, 3);
    ggml_tensor * attn = nullptr;  // [dim, n_head, n_q] F32
    ggml_tensor * fa_node = nullptr;
    if (fa) {
        fa_node = ggml_flash_attn_ext(c.ctx, qp, kp, vp, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa_node, GGML_PREC_F32);  // as llama.cpp; the CUDA kernels do not read it
        attn = fa_node;
    } else {
        ggml_tensor * kq = ggml_mul_mat(c.ctx, kp, qp);  // [n_kv, n_q, n_head]
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        ggml_tensor * p = ggml_soft_max_ext(c.ctx, kq, mask, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(c.ctx, ggml_transpose(c.ctx, vp));  // [n_kv, dim, n_head_kv], llama.cpp's v_trans
        ggml_tensor * kqv = ggml_mul_mat(c.ctx, vt, p);  // [dim, n_q, n_head]
        if (path == NONFA_F32 || path == NONFA_F32_NO_TF32) {
            ggml_prec_set_acc(kqv, GGML_PREC_F32);
        }
        attn = ggml_cont(c.ctx, ggml_permute(c.ctx, kqv, 0, 2, 1, 3));
    }
    ggml_tensor * o = as_type(c.ctx, attn, out_type);
    ggml_set_output(o);
    ggml_cgraph * graph = ggml_new_graph_custom(c.ctx, 64, false);
    ggml_build_forward_expand(graph, o);
    if (alloc(graph) || copy_in(tq, q)) return -1;
    const size_t row = n_head_kv * dim * sizeof(half);
    if (check_cuda(cudaMemset(tk->data, 0, ggml_nbytes(tk)), "k pad") || check_cuda(cudaMemset(tv->data, 0, ggml_nbytes(tv)), "v pad")) return -1;
    if (check_cuda(cudaMemcpy(tk->data, k, n_kv * row, cudaMemcpyDeviceToDevice), "k") ||
        check_cuda(cudaMemcpy(tv->data, v, n_kv * row, cudaMemcpyDeviceToDevice), "v")) return -1;
    if (fa) {
        std::vector<half> m(n_kv_pad * n_q);
        for (int64_t i = 0; i < n_q; ++i)
            for (int64_t j = 0; j < n_kv_pad; ++j)
                m[i * n_kv_pad + j] = j <= q_pos0 + i ? __float2half(0.0f) : __float2half(-INFINITY);
        if (check_cuda(cudaMemcpy(mask->data, m.data(), m.size() * sizeof(half), cudaMemcpyHostToDevice), "mask")) return -1;
    } else {
        std::vector<float> m(n_kv_pad * n_q);
        for (int64_t i = 0; i < n_q; ++i)
            for (int64_t j = 0; j < n_kv_pad; ++j)
                m[i * n_kv_pad + j] = j <= q_pos0 + i ? 0.0f : -INFINITY;
        if (check_cuda(cudaMemcpy(mask->data, m.data(), m.size() * sizeof(float), cudaMemcpyHostToDevice), "mask")) return -1;
    }
    if (path == FA_VEC || path == FA_TILE) {
        // Node by node: the flash-attention node through the forced launcher,
        // the rest (views, the output conversion) through graph compute.
        ggml_backend_cuda_context & ctx = cuda_ctx();
        bool launched = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i) != fa_node) {
                ggml_cgraph one = ggml_graph_view(graph, i, i + 1);
                if (compute(&one)) return -1;
                continue;
            }
            if (dim != 64) return fail("the forced vector launcher is instantiated for head size 64 only");
            ggml_cuda_set_device(ctx.device);
            if (check_cuda(cudaDeviceSynchronize(), "inputs")) return -1;
            if (path == FA_VEC) {
                ggml_cuda_flash_attn_ext_vec_case<64, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, fa_node);
            } else {
                ggml_cuda_flash_attn_ext_tile(ctx, fa_node);
            }
            if (check_cuda(cudaStreamSynchronize(ctx.stream()), "forced launcher")) return -1;
            launched = true;
        }
        if (!launched) return fail("flash-attention node not in graph");
    } else {
        cublasHandle_t handle = nullptr;
        if (path == NONFA_F32_NO_TF32 || path == NONFA_LLAMA_NO_TF32) {
            handle = cuda_ctx().cublas_handle();
            if (cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) return fail("math mode");
        }
        const int rc = compute(graph);
        if (handle != nullptr && cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) {
            return fail("math mode restore");
        }
        if (rc) return -1;
    }
    if (copy_out(out, o)) return -1;
    return sync_all("attention exit");
}

// The matrix-multiply route ggml_cuda_mul_mat takes for the non-flash-attention
// KQ and KQV products at this shape (1 MMVF, 2 MMF, 3 cuBLAS), from GGML's own
// predicates; which = 0 for KQ, 1 for KQV.
int ggml_shim_nonfa_route(int64_t n_q, int64_t n_kv, int64_t n_head_kv, int64_t dim, int which) {
    ggml_backend_cuda_context & ctx = cuda_ctx();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int warp = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t n_kv_pad = GGML_PAD(n_kv, 256);
    int64_t ne[4];
    size_t nb[4];
    if (which == 0) {  // K view [dim, n_kv, n_head_kv] of a [dim, n_head_kv, n_kv] F16 tensor
        ne[0] = dim; ne[1] = n_kv_pad; ne[2] = n_head_kv; ne[3] = 1;
        nb[0] = 2; nb[1] = 2 * dim * n_head_kv; nb[2] = 2 * dim; nb[3] = nb[1] * n_kv_pad;
    } else {           // contiguous V^T [n_kv, dim, n_head_kv]
        ne[0] = n_kv_pad; ne[1] = dim; ne[2] = n_head_kv; ne[3] = 1;
        nb[0] = 2; nb[1] = 2 * n_kv_pad; nb[2] = nb[1] * dim; nb[3] = nb[2] * n_head_kv;
    }
    if (ggml_cuda_should_use_mmvf(GGML_TYPE_F16, cc, ne, nb, n_q)) return 1;
    if (ggml_cuda_should_use_mmf(GGML_TYPE_F16, cc, warp, ne, nb, (int) n_q, false)) return 2;
    return 3;
}

}  // extern "C"
