// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me>
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// Original Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976.
// Original attention numerical bodies and token-tile producer chains.
// Signatures borrow the native decode table/diagnostics; lexical aliases
// remove device globals without changing their arithmetic expressions.
// No original stream, handle, allocator, graph or cache registry.
// Source proof and original numerical compile flags are required.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_CORE_CUH_
#define JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_CORE_CUH_

#define DS4_CUDA_ATTENTION_SCORE_CAP 8192u
#define DS4_CUDA_ATTENTION_RAW_SCORE_CAP 256u
#define DS4_OPP_C_FP8_NOPE_DEV 448u
#define DS4_OPP_C_FP8_BLOCKS_DEV 7u
#define DS4_OPP_C_FP8_ROW_BYTES_DEV 704u
#define dsv4_e4m3fn_decode_table decode_table
#define g_fp8_kv_read_path_blocks (diagnostics->dense_packed_reads)
#define g_fp8_kv_indexed_read_path_blocks (diagnostics->indexed_packed_reads)

// clang-format off
struct ds4_decode_scalars {
    uint32_t pos0;        /* base sequence position; advances every token */
    uint32_t raw_row;     /* pos0 % raw_cap; KV ring-buffer slot */
    uint32_t raw_start;   /* window base in raw cache (post-mod) */
    uint32_t n_raw;       /* min(pos0 + 1, raw_window); raw count */
    uint32_t n_comp;      /* visible compressed tokens this step */
    uint32_t emit_phase;  /* pos0 % ratio; compressor cyclic slot */
    /* Row scalars for the row-view kernels (R1, Step-4 analyst review).
     * Per-layer state: layer_n_comp[il] and layer_n_index_comp[il].  Callers
     * write these via ds4_gpu_decode_scalars_set_emit_rows() + flush()
     * immediately before each per-layer emit, so subsequent kernels in that
     * layer's body see the right row.  Under future layer-graph capture
     * the flush becomes either a per-layer captured memcpy node or a per-
     * layer entry in a device-side array (Step 5/6 detail). */
    uint32_t comp_row;    /* layer_n_comp[il] at the current per-layer emit */
    uint32_t index_row;   /* layer_n_index_comp[il] at the current per-layer emit */
    uint32_t flags;       /* bit 0: emit FP8 KV this step
                           * bit 1: indexed-attention path active
                           * bit 2: ratio4 compressor schedule
                           * bits 3..31: reserved (must be 0) */
    uint32_t token;       /* decode token id this step.  Step 7 task #36:
                           * the hash-mode router-select kernels read the
                           * token from here (device buffer) instead of a
                           * by-value kernel arg, so a captured layer graph
                           * looks up the LIVE token on replay rather than
                           * the frozen capture-time value. */
};
static_assert(sizeof(struct ds4_decode_scalars) == 40u,
              "ds4_decode_scalars must be exactly 40 bytes");

struct ds4_layer_scalars {
    uint32_t n_comp;       /* attention compressed count, post-emit */
    uint32_t n_index_comp; /* indexer compressed count, post-emit (PC3) */
    uint32_t comp_row;     /* pre-emit row index for fp8 row-kernel */
    uint32_t index_row;    /* pre-emit row index for indexer_qat row-kernel */
};

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* Read one (row, dim) lane out of the packed FP8 mirror.  For d<448 the
 * code is 1 byte (sign in bit 7, magnitude index 0..126 in bits 0..6) and
 * the value reconstructs as sign * decode_table[idx] * scale[block].  For
 * d>=448 the lane is the FP32 rotary tail copied verbatim at emit.  The
 * mirror only carries the compressed-row payload, so the caller must have
 * already filtered out raw-row reads.
 *
 * P2 Inc2a: `c` is the ABSOLUTE mirror row.  The mirror slabs share the
 * F32 comp cache's bank layout (bank stride = comp_cap rows), so per-seq
 * callers must pass comp_seq_base + local_row, exactly mirroring their
 * F32 `comp_kv + (comp_seq_base + c) * head_dim` reads. */
__device__ static float fp8_kv_read(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t                          c,
        uint32_t                          d,
        const float* decode_table) {
    const uint64_t row_off = (uint64_t)c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
    if (d < DS4_OPP_C_FP8_NOPE_DEV) {
        const unsigned char code = codes_base[row_off + d];
        const float scale = scale_base[(uint64_t)c * DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6)];
        const float mag   = dsv4_e4m3fn_decode_table[code & 0x7fu];
        return ((code & 0x80u) ? -mag : mag) * scale;
    }
    const float *tail = (const float *)(codes_base + row_off + DS4_OPP_C_FP8_NOPE_DEV);
    return tail[d - DS4_OPP_C_FP8_NOPE_DEV];
}

/* P2 Inc2b: vector companion of fp8_kv_read() for the heads8/float4 staging
 * loops (kv_shared[off] = src[c4], c4 in [0, head_dim/4)).  Reads lanes
 * 4*c4 .. 4*c4+3 of row `c` (same ABSOLUTE row convention as fp8_kv_read).
 * For 4*c4 < 448 the four codes sit in one uchar4 -- the row start is
 * 16-byte aligned (704 = 44*16) and 4*c4 is 4-aligned -- and share one
 * 64-lane scale block (64 % 4 == 0: a quad never straddles blocks).  From
 * c4 >= 112 the lanes are the verbatim FP32 rotary tail, float4-aligned at
 * +448.  Each lane reconstructs exactly as fp8_kv_read would. */
__device__ static float4 fp8_kv_read4(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t                          c,
        uint32_t                          c4,
        const float* decode_table) {
    const uint64_t row_off = (uint64_t)c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
    const uint32_t d = c4 << 2u;
    if (d < DS4_OPP_C_FP8_NOPE_DEV) {
        const uchar4 q = *(const uchar4 *)(codes_base + row_off + d);
        const float scale = scale_base[(uint64_t)c * DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6u)];
        const float mx = dsv4_e4m3fn_decode_table[q.x & 0x7fu];
        const float my = dsv4_e4m3fn_decode_table[q.y & 0x7fu];
        const float mz = dsv4_e4m3fn_decode_table[q.z & 0x7fu];
        const float mw = dsv4_e4m3fn_decode_table[q.w & 0x7fu];
        return make_float4(((q.x & 0x80u) ? -mx : mx) * scale,
                           ((q.y & 0x80u) ? -my : my) * scale,
                           ((q.z & 0x80u) ? -mz : mz) * scale,
                           ((q.w & 0x80u) ? -mw : mw) * scale);
    }
    const float4 *tail = (const float4 *)(codes_base + row_off + DS4_OPP_C_FP8_NOPE_DEV);
    return tail[c4 - (DS4_OPP_C_FP8_NOPE_DEV >> 2u)];
}

// Forward the explicit immutable native table through original read calls.
#define fp8_kv_read(...) fp8_kv_read(__VA_ARGS__, decode_table)
#define fp8_kv_read4(...) fp8_kv_read4(__VA_ARGS__, decode_table)

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* P2 Inc2b: optional packed FP8 mirror of the compressed rows; when
         * both are non-NULL the comp lanes decode via fp8_kv_read() (bare
         * row index -- this kernel's comp_kv base is already the row-0 base
         * of the same cache the mirror views share). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && visible_comp != 0u && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    }
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            float dot = 0.0f;
            if (use_fp8) {
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * fp8_kv_read(comp_fp8, comp_scale, c, d);
            } else {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            }
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        if (use_fp8) {
            for (uint32_t c = 0; c < visible_comp; c++) acc += fp8_kv_read(comp_fp8, comp_scale, c, d) * scores[raw_count + c];
        } else {
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim,
        /* P2 Inc2b: optional packed FP8 mirror; the pack decodes comp lanes
         * once into the FP32 scratch the cublas score/value GEMMs consume
         * (bare row index, same base as comp_kv). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && n_comp != 0u && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    }
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    if (r < n_tokens) {
        dst[gid] = raw_kv[(uint64_t)r * head_dim + d];
    } else if (use_fp8) {
        dst[gid] = fp8_kv_read(comp_fp8, comp_scale, r - n_tokens, d);
    } else {
        dst[gid] = comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
    }
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        /* Phase 2 Step 4a: per-seq compressed-cache bank stride (rows).  When
         * seq_id is set, row t reads its compressed rows from bank seq_id[t] at
         * base seq_id[t]*comp_cap; seq_id==NULL keeps comp_seq_base=0 (single
         * bank, comp_cap ignored, bit-exact). */
        uint32_t comp_cap,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (n_tokens
         * entries).  positions -> row t's query position (replaces pos0+t,
         * required for multi-seq where rows are independent sequences);
         * seq_id -> per-seq raw-ring bank base seq_id[t]*raw_cap AND per-seq
         * compressed-cache bank base seq_id[t]*comp_cap (Step 4a).  Both NULL
         * keeps the single-sequence scalar path (bit-exact). */
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        /* Optional device-side scalars override (Step-4 Commit B / R5).
         * When non-NULL the kernel reads n_raw, raw_start from the struct
         * at execution time instead of using the inline args.  pos0,
         * n_tokens, window, ratio are passed = 0 by the decode shim and
         * remain inline (no per-token variation). */
        const struct ds4_decode_scalars * __restrict__ s_override,
        /* Optional per-layer scalars override (Step 4c A1).  When non-
         * NULL the kernel reads n_comp from ls_override->n_comp at
         * execution time, closing the R6 race for the attention path.
         * Pre-capture this produces the same value as the inline arg
         * (the prologue populates ls->n_comp = post-this-token's-emit
         * count); under capture (Step 5/6) ls_override becomes the
         * load-bearing source. */
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        /* Opp C Phase 1A.3: optional packed FP8 mirror of the compressed
         * rows.  When both pointers are non-NULL the kernel reads
         * compressed lanes via fp8_kv_read() instead of `comp_kv` -- the
         * decode table makes the reconstruction bit-identical to the FP32
         * cache (proven by the Phase-0 numerics test), so the captured
         * graph and the eager build still produce the same token ids.
         * NULL when DS4_CUDA_FP8_KV is off or no compressed rows exist;
         * raw_kv and the rotary tail handling are untouched. */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        /* M4b Inc3: optional per-row raw-span override (n_tokens entries).  The
         * batched MTP draft reuses this decode attention, but its KV ring is
         * sparsely filled (drafts only, not prefill/decode), so the positions-
         * derived span min(window, qpos+1) would read stale/never-written slots.
         * When non-NULL, row t uses draft_n_raw[t] = min(mtp_n_raw[bank]+1,
         * raw_window, raw_cap) raw rows ending at positions[t] -- matching the
         * serial drafter (metal_graph_eval_mtp_draft_from_hc) exactly.  NULL =
         * untouched (every non-draft caller stays bit-exact). */
        const int32_t       * __restrict__ draft_n_raw,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    if (s_override) {
        n_raw     = s_override->n_raw;
        raw_start = s_override->raw_start;
    }
    if (ls_override) {
        n_comp    = ls_override->n_comp;
    }
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    }
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    /* Phase 2 Step 3: per-row position.  positions[t] is the row's absolute
     * query position; first_raw_pos = positions[t]+1-n_raw is the oldest raw
     * row still in the (per-seq) window.  At single-seq positions[t]==pos0+t
     * and n_tokens-1 collapse to the original pos0+n_tokens-n_raw form. */
    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    /* Phase 2 Step 3c: per-row raw span.  In multi-seq each row is an
     * independent sequence whose visible raw count + ring base depend on ITS
     * own position, so derive row_n_raw/row_raw_start per row from positions[t]
     * (a pure function of qpos, window, raw_cap -- it reproduces the host scalar
     * n_raw/raw_start exactly when all rows share a length, so the Step-3b
     * equal-length gate is bit-identical).  Single-seq (positions==NULL) keeps
     * the scalar inline args. */
    uint32_t row_n_raw     = n_raw;
    uint32_t row_raw_start = raw_start;
    if (positions) {
        row_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (row_n_raw > raw_cap) row_n_raw = raw_cap;
        row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    }
    /* M4b Inc3: MTP draft per-row raw-span override (see signature).  The host
     * already clamps to <= min(raw_window, raw_cap); guard >=1 defensively. */
    if (draft_n_raw) {
        row_n_raw = (uint32_t)draft_n_raw[t];
        if (row_n_raw == 0u) row_n_raw = 1u;
        if (row_n_raw > raw_cap) row_n_raw = raw_cap;
        row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    }
    uint32_t first_raw_pos = positions ? (qpos + 1u - row_n_raw) : (pos0 + n_tokens - n_raw);
    uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Phase 2 Step 4a: per-seq compressed-cache bank base (rows).  Each row's
     * visible compressed rows live in its own bank at seq_id[t]*comp_cap. */
    uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    /* Phase 2 Step 4c: per-row compressed count for varlen multi-seq.  The
     * (qpos+1)/ratio formula over-counts by 1 when qpos ≡ ratio-1 (mod ratio);
     * single-seq relies on the scalar n_comp clamp below to correct it, but
     * varlen rows share one scalar (= max over seqs), so derive each row's true
     * count floor(qpos/ratio) here.  Equal-length is unchanged (floor(qpos/ratio)
     * == the scalar n_comp), so 3b/4a/4b gates stay bit-identical. */
    if (positions && ratio != 0u && !single_all && visible_comp > qpos / ratio)
        visible_comp = qpos / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    /* forum-#65 hardening: scores[] is DS4_CUDA_ATTENTION_SCORE_CAP entries
     * and raw_count can reach 256 -- n_score past the cap clobbers
     * raw_rows[] (the adjacent smem array) and the V pass then dereferences
     * score bit-patterns as ring row ids (sanitizer: wild global reads in
     * the raw V loop).  Healthy dispatch can't get here (the fits check
     * gates the launch at n_comp <= CAP-256), but a REPLAYED graph whose
     * live ls->n_comp crossed the boundary could -- seen via the serial
     * layer-graph cache in the ds4-bench 131k-alloc sweep (NVIDIA forum
     * #65).  The clamp only engages in states that would otherwise corrupt
     * smem; the band-keyed graph cache makes those unreachable again. */
    if (visible_comp > DS4_CUDA_ATTENTION_SCORE_CAP - 256u)
        visible_comp = DS4_CUDA_ATTENTION_SCORE_CAP - 256u;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (row_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + row_n_raw - 1u;
            if (single_all) {
                raw_count = row_n_raw > 256u ? 256u : row_n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = seq_base + ((row_raw_start + raw_first_idx + r) % raw_cap);
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                float dot = 0.0f;
                if (use_fp8) {
                    /* Hoisted fp8 row read: the generic fp8_kv_read re-derives
                     * the 64-bit row offset, codes-vs-tail branch, and scale
                     * index PER ELEMENT — issue-bound over a context-scaling
                     * row count.  Value math (sign*mag*scale) unchanged; the
                     * accumulation chain is PINNED below (source-level order
                     * alone is not enough under --use_fast_math). */
                    const unsigned char *cp = comp_fp8 +
                        (uint64_t)(comp_seq_base + c) * DS4_OPP_C_FP8_ROW_BYTES_DEV;
                    const float *scp = comp_scale +
                        (uint64_t)(comp_seq_base + c) * DS4_OPP_C_FP8_BLOCKS_DEV;
                    const float *tail = (const float *)(cp + DS4_OPP_C_FP8_NOPE_DEV);
                    uint32_t d = 0;
                    for (uint32_t b = 0; b < DS4_OPP_C_FP8_BLOCKS_DEV; b++) {
                        const float sc = scp[b];
                        for (uint32_t k = 0; k < 64u; k += 4u, d += 4u) {
                            /* uchar4 quad (row base is 16-aligned, d%4==0);
                             * lanes processed in ascending d.  Accumulation is
                             * PINNED to the scalar chain (mul then in-order
                             * FMA) with intrinsics: fast-math is otherwise
                             * free to tree-reduce the manual unroll, and that
                             * reassociation showed up as a +2 spec_hits drift
                             * on the teb think leg (draft-margin ULPs; text
                             * byte-identical).  Counter-identity with F32 is
                             * a load-bearing gate — keep the chain exact. */
                            const uchar4 q4 = *(const uchar4 *)(cp + d);
                            const float m0 = dsv4_e4m3fn_decode_table[q4.x & 0x7fu];
                            const float m1 = dsv4_e4m3fn_decode_table[q4.y & 0x7fu];
                            const float m2 = dsv4_e4m3fn_decode_table[q4.z & 0x7fu];
                            const float m3 = dsv4_e4m3fn_decode_table[q4.w & 0x7fu];
                            dot = __fmaf_rn(qh[d],
                                    __fmul_rn((q4.x & 0x80u) ? -m0 : m0, sc), dot);
                            dot = __fmaf_rn(qh[d + 1u],
                                    __fmul_rn((q4.y & 0x80u) ? -m1 : m1, sc), dot);
                            dot = __fmaf_rn(qh[d + 2u],
                                    __fmul_rn((q4.z & 0x80u) ? -m2 : m2, sc), dot);
                            dot = __fmaf_rn(qh[d + 3u],
                                    __fmul_rn((q4.w & 0x80u) ? -m3 : m3, sc), dot);
                        }
                    }
                    for (uint32_t k = 0; d < head_dim; k++, d++)
                        dot = __fmaf_rn(qh[d], tail[k], dot);
                } else {
                    /* F32 comp dot — the fp8 branch's identity partner.  Both
                     * sides are pinned to the same in-order FMA chain so
                     * fp8-vs-F32 counter identity is structural, not a
                     * compiled-form coincidence (an F32-only recompile of
                     * this kernel moved teb think spec_hits by -1 with the
                     * source untouched: fast-math re-scheduling). */
                    const float *kvrow = comp_kv + (uint64_t)(comp_seq_base + c) * head_dim;
                    for (uint32_t d = 0; d < head_dim; d++)
                        dot = __fmaf_rn(qh[d], kvrow[d], dot);
                }
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = NULL;
                bool fp8_row = false;
                uint32_t fp8_c = 0u;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) {
                        if (use_fp8) {
                            fp8_row = true;
                            fp8_c = comp_seq_base + c;
                        } else {
                            kvrow = comp_kv + (uint64_t)(comp_seq_base + c) * head_dim;
                        }
                    }
                }
                float s = -INFINITY;
                if (kvrow || fp8_row) {
                    float dot = 0.0f;
                    if (fp8_row) {
                        /* Same hoist as the single-token branch: row base
                         * pointers computed once, per-element work reduced to
                         * byte load + table + sign + FMA.  Accumulation
                         * pinned with intrinsics (see the quad-dot comment):
                         * this is the spec-verify dot — counter-identity
                         * with F32 depends on the exact chain. */
                        const unsigned char *cp = comp_fp8 +
                            (uint64_t)fp8_c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
                        const float *scp = comp_scale +
                            (uint64_t)fp8_c * DS4_OPP_C_FP8_BLOCKS_DEV;
                        const float *tail = (const float *)(cp + DS4_OPP_C_FP8_NOPE_DEV);
                        uint32_t d = qlane;
                        for (; d < DS4_OPP_C_FP8_NOPE_DEV; d += 8u) {
                            const unsigned char code = cp[d];
                            const float mag = dsv4_e4m3fn_decode_table[code & 0x7fu];
                            dot = __fmaf_rn(qh[d],
                                    __fmul_rn((code & 0x80u) ? -mag : mag, scp[d >> 6u]), dot);
                        }
                        for (; d < head_dim; d += 8u) {
                            dot = __fmaf_rn(qh[d], tail[d - DS4_OPP_C_FP8_NOPE_DEV], dot);
                        }
                    } else {
                        /* Pinned like its fp8 partner above (identity is
                         * structural only if BOTH branches fix the chain). */
                        for (uint32_t d = qlane; d < head_dim; d += 8u)
                            dot = __fmaf_rn(qh[d], kvrow[d], dot);
                    }
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        /* Consecutive-pair lane layout (d0,d1 = 2t,2t+1) instead of the old
         * split halves (t, t+256): the fp8 V loop reads its two codes as one
         * uchar2 and shares one scale block per pair; tail threads read
         * float2 — one stream instead of two.  Per-d values are unchanged
         * (each d column keeps its c-order), so outputs are bit-identical
         * under any lane mapping; F32/raw reads stay equally coalesced (a
         * warp covers a contiguous 256 B either way). */
        uint32_t d0 = threadIdx.x * 2u;
        uint32_t d1 = d0 + 1u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        if (use_fp8) {
            /* Hottest loop of the deep-decode ledger: one sequential pass
             * over EVERY visible comp row per thread.  With the pair lane
             * layout the two lanes are one uchar2 (or one float2 in the
             * rotary tail) sharing one scale block; all address math is
             * loop-invariant and pointers advance by the row stride.
             * Value math (sign*mag*scale) per lane unchanged.  Alignment:
             * row stride 704 and the 448 tail offset are 8-byte multiples
             * and d0 is even, so uchar2/float2 loads are aligned. */
            const uint64_t rb = (uint64_t)comp_seq_base;
            const unsigned char *row0 = comp_fp8 + rb * DS4_OPP_C_FP8_ROW_BYTES_DEV;
            if (d0 < DS4_OPP_C_FP8_NOPE_DEV) {          /* uniform per warp */
                const unsigned char *cp = row0 + d0;
                const float *scp = comp_scale + rb * DS4_OPP_C_FP8_BLOCKS_DEV + (d0 >> 6u);
                for (uint32_t c = 0; c < visible_comp; c++) {
                    const float s = scores[raw_count + c];
                    const uchar2 k = *(const uchar2 *)cp;
                    const float sc = *scp;
                    const float m0 = dsv4_e4m3fn_decode_table[k.x & 0x7fu];
                    const float m1 = dsv4_e4m3fn_decode_table[k.y & 0x7fu];
                    acc0 += (((k.x & 0x80u) ? -m0 : m0) * sc) * s;
                    acc1 += (((k.y & 0x80u) ? -m1 : m1) * sc) * s;
                    cp += DS4_OPP_C_FP8_ROW_BYTES_DEV;
                    scp += DS4_OPP_C_FP8_BLOCKS_DEV;
                }
            } else {
                const unsigned char *tp = row0 + DS4_OPP_C_FP8_NOPE_DEV +
                    (d0 - DS4_OPP_C_FP8_NOPE_DEV) * (uint32_t)sizeof(float);
                for (uint32_t c = 0; c < visible_comp; c++) {
                    const float s = scores[raw_count + c];
                    const float2 v = *(const float2 *)tp;
                    acc0 += v.x * s;
                    acc1 += v.y * s;
                    tp += DS4_OPP_C_FP8_ROW_BYTES_DEV;
                }
            }
        } else {
            for (uint32_t c = 0; c < visible_comp; c++) {
                float s = scores[raw_count + c];
                const float *kv = comp_kv + (uint64_t)(comp_seq_base + c) * head_dim;
                acc0 += kv[d0] * s;
                acc1 += kv[d1] * s;
            }
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            if (use_fp8) {
                /* Generic-width V path: d fixed per iteration of the outer
                 * loop — hoist the row-invariant address math (same value
                 * math and accumulation order as fp8_kv_read). */
                const uint64_t rb = (uint64_t)comp_seq_base;
                if (d < DS4_OPP_C_FP8_NOPE_DEV) {
                    const unsigned char *cp = comp_fp8 + rb * DS4_OPP_C_FP8_ROW_BYTES_DEV + d;
                    const float *scp = comp_scale + rb * DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6u);
                    for (uint32_t c = 0; c < visible_comp; c++) {
                        const unsigned char code = *cp;
                        const float mag = dsv4_e4m3fn_decode_table[code & 0x7fu];
                        acc += (((code & 0x80u) ? -mag : mag) * (*scp)) * scores[raw_count + c];
                        cp += DS4_OPP_C_FP8_ROW_BYTES_DEV;
                        scp += DS4_OPP_C_FP8_BLOCKS_DEV;
                    }
                } else {
                    const unsigned char *tp = comp_fp8 + rb * DS4_OPP_C_FP8_ROW_BYTES_DEV +
                        DS4_OPP_C_FP8_NOPE_DEV + (d - DS4_OPP_C_FP8_NOPE_DEV) * (uint32_t)sizeof(float);
                    for (uint32_t c = 0; c < visible_comp; c++) {
                        acc += *(const float *)tp * scores[raw_count + c];
                        tp += DS4_OPP_C_FP8_ROW_BYTES_DEV;
                    }
                }
            } else {
                for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)(comp_seq_base + c) * head_dim + d] * scores[raw_count + c];
            }
            oh[d] = acc / denom;
        }
    }
}

/* ---- Flash-decode split for the single-token attention decode path ----
 * attention_decode_mixed_kernel launches one block per head (grid = n_head),
 * which under-fills GPUs with many SMs (e.g. 64 heads on the PRO 6000's 188
 * SMs) and serializes each head's V-accumulation over the whole KV window.
 * These two kernels split each head's score rows across n_split blocks, each
 * emitting an online-softmax partial (m = chunk row-max, l = sum exp(s-m),
 * acc[d] = sum exp(s-m)*v), then a fixed-order combine merges the partials and
 * folds in the sink. Mathematically equal to the single-block path; fp rounding
 * differs (the split changes the softmax reference and summation order) -> a
 * one-time deterministic output shift. eager==captured still holds: n_split, the
 * row partition, and the sp-ascending combine are all deterministic and the
 * scratch pointer is allocated eager (capture-stable). Mirrors the kernel's
 * `single_all` decode branch exactly (t=0, pos0=0, window=0, ratio=0): raw_count
 * = min(n_raw,256), all n_comp visible, raw_rows[r] = (raw_start+r) % raw_cap. */
__global__ static void attention_decode_split_kernel(
        float *partials,            /* [(h*n_split+sp)*(head_dim+2)] = {m, l, acc[head_dim]} */
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_split,
        const struct ds4_decode_scalars * __restrict__ s_override,
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        /* P2 Inc2a note: this kernel is single-seq only (no positions/
         * seq_id), so its bare-row fp8_kv_read indices ARE the absolute
         * mirror rows (comp_seq_base == 0). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    if (s_override) { n_raw = s_override->n_raw; raw_start = s_override->raw_start; }
    if (ls_override) { n_comp = ls_override->n_comp; }
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && threadIdx.x == 0) atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    const uint32_t sp = blockIdx.x;
    const uint32_t h  = blockIdx.y;
    if (h >= n_head || sp >= n_split) return;
    const uint32_t raw_count = n_raw > 256u ? 256u : n_raw;
    const uint32_t visible_comp = n_comp;
    const uint32_t n_score = raw_count + visible_comp;
    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + (uint64_t)h * head_dim;
    __shared__ uint32_t raw_rows[256];
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ float partial[256];
    __shared__ float m_sh;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x)
        raw_rows[r] = (raw_start + r) % raw_cap;
    __syncthreads();
    /* this block's contiguous chunk of the row index range [0, n_score) */
    const uint32_t chunk = (n_score + n_split - 1u) / n_split;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_score) hi = n_score;
    float *out_m = partials + (uint64_t)(h * n_split + sp) * (head_dim + 2u);
    if (lo >= hi) {
        if (threadIdx.x == 0) { out_m[0] = -INFINITY; out_m[1] = 0.0f; }
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) out_m[2u + d] = 0.0f;
        return;
    }
    const uint32_t cn = hi - lo;
    float local_max = -INFINITY;
    for (uint32_t j = threadIdx.x; j < cn; j += blockDim.x) {
        const uint32_t row = lo + j;
        float s;
        if (row < raw_count) {
            const float *kv = raw_kv + (uint64_t)raw_rows[row] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            s = dot * scale;
        } else {
            const uint32_t c = row - raw_count;
            const float add = use_comp_mask ? comp_mask[c] : 0.0f;
            if (add > -1.0e20f) {
                float dot = 0.0f;
                if (use_fp8) {
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * fp8_kv_read(comp_fp8, comp_scale, c, d);
                } else {
                    const float *kv = comp_kv + (uint64_t)c * head_dim;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                }
                s = dot * scale + add;
            } else {
                s = -INFINITY;
            }
        }
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) m_sh = partial[0];
    __syncthreads();
    const float m = m_sh;
    if (m == -INFINITY) {
        /* whole chunk masked out -> contributes nothing to the combine */
        if (threadIdx.x == 0) { out_m[0] = -INFINITY; out_m[1] = 0.0f; }
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) out_m[2u + d] = 0.0f;
        return;
    }
    float den_local = 0.0f;
    for (uint32_t j = threadIdx.x; j < cn; j += blockDim.x) {
        const float e = expf(scores[j] - m);
        scores[j] = e;
        den_local += e;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) partial[threadIdx.x] += partial[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) { out_m[0] = m; out_m[1] = partial[0]; }
    float *out_acc = out_m + 2u;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < cn; j++) {
            const uint32_t row = lo + j;
            const float sc = scores[j];
            if (row < raw_count) {
                acc += raw_kv[(uint64_t)raw_rows[row] * head_dim + d] * sc;
            } else {
                const uint32_t c = row - raw_count;
                if (use_fp8) acc += fp8_kv_read(comp_fp8, comp_scale, c, d) * sc;
                else acc += comp_kv[(uint64_t)c * head_dim + d] * sc;
            }
        }
        out_acc[d] = acc;
    }
}

/* Combine the n_split online-softmax partials for each head into the final
 * attention output, folding in the sink. Fixed sp-ascending order keeps it
 * deterministic (eager==captured). One block per head. */
__global__ static void attention_decode_combine_kernel(
        float *heads,
        const float *sinks,
        const float *partials,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_split) {
    const uint32_t h = blockIdx.x;
    if (h >= n_head) return;
    const float *base = partials + (uint64_t)(h * n_split) * (head_dim + 2u);
    const uint32_t stride = head_dim + 2u;
    __shared__ float f_sh[64];
    __shared__ float denom_sh;
    if (threadIdx.x == 0) {
        const float sink = sinks[h];
        float gm = sink;
        for (uint32_t sp = 0; sp < n_split; sp++) gm = fmaxf(gm, base[(uint64_t)sp * stride]);
        float denom = expf(sink - gm);
        for (uint32_t sp = 0; sp < n_split; sp++) {
            const float m = base[(uint64_t)sp * stride];
            const float l = base[(uint64_t)sp * stride + 1u];
            const float f = (m == -INFINITY) ? 0.0f : expf(m - gm);
            f_sh[sp] = f;
            denom += f * l;
        }
        denom_sh = denom;
    }
    __syncthreads();
    const float denom = denom_sh;
    float *oh = heads + (uint64_t)h * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t sp = 0; sp < n_split; sp++) acc += f_sh[sp] * base[(uint64_t)sp * stride + 2u + d];
        oh[d] = acc / denom;
    }
}

/* ---- v0.4 head-group flash-decode (HG) ----
 * The per-head decode kernels above re-read and re-decode every visible
 * compressed row once PER HEAD (64x), one block per head, with a 32 KB
 * static score buffer capping residency at 2 blocks/SM -- ncu shows the
 * result is latency exposure, not bandwidth (240K decode shape: 654 us for
 * a 1.9 MB byte stream).  HG restructures the work: each block owns
 * DS4_ATTN_HG_HEADS heads and a contiguous row chunk, loops it in
 * DS4_ATTN_HG_ROWS-row tiles staged ONCE into f32 smem (fp8 decode shared
 * across the block's heads), computes warp-per-head dots from smem with q
 * held in registers, keeps an online-softmax partial {m, l, acc[]} per
 * head, and a fixed-order combine merges the n_split partials (same
 * deterministic numerics class as the flash-decode split above).  Proto
 * ladder + floor ledger: local/docs/briefs/brief-v04-arc.md (2026-07-20);
 * cuda/mmq/test/proto_attn_decode.cu is the isolated harness.
 *
 * fp8-vs-F32 config identity is STRUCTURAL here: both configs stage rows
 * to the same f32 smem tile (the fp8 decode -- table magnitude x pow2
 * block scale -- is exact), so the dot/V chains are shared instruction
 * streams, not twinned branches.
 *
 * Capture safety: n_split is a pure function of device SM count + n_head
 * (attention_decode_hg_n_split), the partials scratch is the eager
 * capture-stable g_attn_split_partials allocation, and live n_comp arrives
 * through ls_override exactly like the kernels above -- the in-kernel
 * chunk math adapts at replay while the grid stays fixed. */
#define DS4_ATTN_HG_HEADS 8u
#define DS4_ATTN_HG_ROWS  8u

__global__ static void __launch_bounds__(256, 4) attention_decode_hg_partial_kernel(
        float *partials,        /* [((t*n_head+h)*n_split+sp)*(head_dim+2)] = {m, l, acc[]} */
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t comp_cap,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,      /* dispatch-guarded == 512 */
        uint32_t n_split,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const struct ds4_decode_scalars * __restrict__ s_override,
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    __shared__ float kv_sm[DS4_ATTN_HG_ROWS][512 + 4];
    __shared__ float P_sm[DS4_ATTN_HG_HEADS][DS4_ATTN_HG_ROWS];
    __shared__ float m_sm[DS4_ATTN_HG_HEADS];
    __shared__ float l_sm[DS4_ATTN_HG_HEADS];
    __shared__ float f_sm[DS4_ATTN_HG_HEADS];
    if (s_override) {
        n_raw     = s_override->n_raw;
        raw_start = s_override->raw_start;
    }
    if (ls_override) {
        n_comp    = ls_override->n_comp;
    }
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    }
    const uint32_t sp = blockIdx.x;
    const uint32_t hg = blockIdx.y;
    const uint32_t t  = blockIdx.z;
    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t row_n_raw     = n_raw;
    uint32_t row_raw_start = raw_start;
    if (positions) {
        row_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (row_n_raw > raw_cap) row_n_raw = raw_cap;
        row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    }
    const uint32_t first_raw_pos = positions ? (qpos + 1u - row_n_raw)
                                             : (pos0 + n_tokens - n_raw);
    const uint32_t seq_base      = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    const uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    /* raw span + visibility: exact port of attention_decode_mixed_kernel's
     * derivation (computed per thread; scalar and cheap). */
    uint32_t raw_count = 0;
    uint32_t raw_first_idx = 0;
    if (row_n_raw != 0) {
        const uint32_t raw_last_pos = first_raw_pos + row_n_raw - 1u;
        if (single_all) {
            raw_count = row_n_raw > 256u ? 256u : row_n_raw;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo_p = first_raw_pos;
            if (window != 0 && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo_p) lo_p = wlo;
            }
            const uint32_t hi_p = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi_p >= lo_p) {
                raw_first_idx = lo_p - first_raw_pos;
                raw_count = hi_p - lo_p + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (positions && ratio != 0u && !single_all && visible_comp > qpos / ratio)
        visible_comp = qpos / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    const uint32_t n_score = raw_count + visible_comp;
    const uint32_t chunk = (n_score + n_split - 1u) / n_split;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_score) hi = n_score;
    const float scale = rsqrtf((float)head_dim);
    /* q for this warp's head, in registers (head_dim 512 / 32 lanes). */
    float q_reg[16];
    {
        const uint32_t h = hg * DS4_ATTN_HG_HEADS + warp;
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
#pragma unroll
        for (uint32_t k = 0; k < 16u; k++) q_reg[k] = qh[lane + 32u * k];
    }
    if (tid < DS4_ATTN_HG_HEADS) { m_sm[tid] = -INFINITY; l_sm[tid] = 0.0f; }
    __syncthreads();
    const uint32_t d0 = (tid * 2u) & 511u;
    const uint32_t d1 = d0 + 1u;
    float O0[DS4_ATTN_HG_HEADS], O1[DS4_ATTN_HG_HEADS];
#pragma unroll
    for (uint32_t h = 0; h < DS4_ATTN_HG_HEADS; h++) { O0[h] = 0.0f; O1[h] = 0.0f; }

    for (uint32_t tile = lo; tile < hi; tile += DS4_ATTN_HG_ROWS) {
        const uint32_t nrows = (hi - tile) < DS4_ATTN_HG_ROWS ? (hi - tile) : DS4_ATTN_HG_ROWS;
        /* Stage-decode the tile into f32 smem once for the whole head group.
         * Pad rows are ZEROED (a stale-NaN smem row survives P=0: 0*NaN). */
        for (uint32_t i4 = tid; i4 < DS4_ATTN_HG_ROWS * 128u; i4 += blockDim.x) {
            const uint32_t r = i4 >> 7u;
            const uint32_t d = (i4 & 127u) * 4u;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (r < nrows) {
                const uint32_t row = tile + r;
                if (row < raw_count) {
                    const uint32_t rr = seq_base +
                        ((row_raw_start + raw_first_idx + row) % raw_cap);
                    const float4 f4 = *(const float4 *)(raw_kv + (uint64_t)rr * head_dim + d);
                    v0 = f4.x; v1 = f4.y; v2 = f4.z; v3 = f4.w;
                } else if (use_fp8) {
                    const uint32_t c = row - raw_count;
                    const unsigned char *cp = comp_fp8 +
                        (uint64_t)(comp_seq_base + c) * DS4_OPP_C_FP8_ROW_BYTES_DEV;
                    if (d < DS4_OPP_C_FP8_NOPE_DEV) {
                        const uchar4 q4 = *(const uchar4 *)(cp + d);
                        const float sc = comp_scale[(uint64_t)(comp_seq_base + c) *
                                                    DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6u)];
                        const float m0 = dsv4_e4m3fn_decode_table[q4.x & 0x7fu];
                        const float m1 = dsv4_e4m3fn_decode_table[q4.y & 0x7fu];
                        const float m2 = dsv4_e4m3fn_decode_table[q4.z & 0x7fu];
                        const float m3 = dsv4_e4m3fn_decode_table[q4.w & 0x7fu];
                        v0 = ((q4.x & 0x80u) ? -m0 : m0) * sc;
                        v1 = ((q4.y & 0x80u) ? -m1 : m1) * sc;
                        v2 = ((q4.z & 0x80u) ? -m2 : m2) * sc;
                        v3 = ((q4.w & 0x80u) ? -m3 : m3) * sc;
                    } else {
                        const float4 f4 = *(const float4 *)(cp + DS4_OPP_C_FP8_NOPE_DEV +
                            (uint64_t)(d - DS4_OPP_C_FP8_NOPE_DEV) * sizeof(float));
                        v0 = f4.x; v1 = f4.y; v2 = f4.z; v3 = f4.w;
                    }
                } else {
                    const uint32_t c = row - raw_count;
                    const float4 f4 = *(const float4 *)(comp_kv +
                        (uint64_t)(comp_seq_base + c) * head_dim + d);
                    v0 = f4.x; v1 = f4.y; v2 = f4.z; v3 = f4.w;
                }
            }
            kv_sm[r][d]      = v0;
            kv_sm[r][d + 1u] = v1;
            kv_sm[r][d + 2u] = v2;
            kv_sm[r][d + 3u] = v3;
        }
        __syncthreads();
        /* warp-per-head dots from smem, q from registers */
        float s_r[DS4_ATTN_HG_ROWS];
#pragma unroll
        for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) s_r[r] = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 16u; k++) {
            const uint32_t d = lane + 32u * k;
            const float qv = q_reg[k];
#pragma unroll
            for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++)
                s_r[r] = __fmaf_rn(qv, kv_sm[r][d], s_r[r]);
        }
#pragma unroll
        for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
            for (uint32_t off = 16u; off > 0u; off >>= 1u)
                s_r[r] += __shfl_down_sync(0xffffffffu, s_r[r], off);
        }
        if (lane == 0) {
            float m_new = m_sm[warp];
            for (uint32_t r = 0; r < nrows; r++)
                m_new = fmaxf(m_new, s_r[r] * scale);
            const float f = expf(m_sm[warp] - m_new);
            float l_add = 0.0f;
            for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
                const float p = r < nrows ? expf(s_r[r] * scale - m_new) : 0.0f;
                P_sm[warp][r] = p;
                l_add += p;
            }
            l_sm[warp] = l_sm[warp] * f + l_add;
            m_sm[warp] = m_new;
            f_sm[warp] = f;
        }
        __syncthreads();
        /* V accumulate: preload the tile's kv at (d0, d1), run heads from regs */
        float kv0[DS4_ATTN_HG_ROWS], kv1[DS4_ATTN_HG_ROWS];
#pragma unroll
        for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
            kv0[r] = kv_sm[r][d0];
            kv1[r] = kv_sm[r][d1];
        }
#pragma unroll
        for (uint32_t h = 0; h < DS4_ATTN_HG_HEADS; h++) {
            const float f = f_sm[h];
            float o0 = O0[h] * f, o1 = O1[h] * f;
#pragma unroll
            for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
                const float p = P_sm[h][r];
                o0 = __fmaf_rn(p, kv0[r], o0);
                o1 = __fmaf_rn(p, kv1[r], o1);
            }
            O0[h] = o0; O1[h] = o1;
        }
        __syncthreads();
    }
    for (uint32_t h = 0; h < DS4_ATTN_HG_HEADS; h++) {
        const uint32_t gh = hg * DS4_ATTN_HG_HEADS + h;
        float *out = partials +
            (uint64_t)(((uint64_t)t * n_head + gh) * n_split + sp) * (head_dim + 2u);
        if (tid == 0) {
            out[0] = m_sm[h];
            out[1] = l_sm[h];
        }
        out[2u + d0] = O0[h];
        out[2u + d1] = O1[h];
    }
}

/* HG combine: one block per (token, head); merges the n_split partials in
 * fixed sp-ascending order and folds in the sink (deterministic,
 * eager==captured). */
__global__ static void attention_decode_hg_combine_kernel(
        float *heads,
        const float *sinks,
        const float *partials,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_split) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const float *base = partials +
        (uint64_t)(((uint64_t)t * n_head + h) * n_split) * (head_dim + 2u);
    const uint32_t stride = head_dim + 2u;
    __shared__ float f_sh[16];
    __shared__ float denom_sh;
    if (threadIdx.x == 0) {
        const float sink = sinks[h];
        float gm = sink;
        for (uint32_t sp = 0; sp < n_split; sp++) gm = fmaxf(gm, base[(uint64_t)sp * stride]);
        float denom = expf(sink - gm);
        for (uint32_t sp = 0; sp < n_split; sp++) {
            const float m = base[(uint64_t)sp * stride];
            const float l = base[(uint64_t)sp * stride + 1u];
            const float f = (m == -INFINITY) ? 0.0f : expf(m - gm);
            f_sh[sp] = f;
            denom += f * l;
        }
        denom_sh = denom;
    }
    __syncthreads();
    const float denom = denom_sh;
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t sp = 0; sp < n_split; sp++) acc += f_sh[sp] * base[(uint64_t)sp * stride + 2u + d];
        oh[d] = acc / denom;
    }
}

__global__ static void __launch_bounds__(256, 4) attention_indexed_hg_partial_kernel(
        float *partials,        /* [((t*n_head+h)*n_split+sp)*(head_dim+2)] = {m, l, acc[]} */
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t * __restrict__ topk,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t n_comp,
        uint32_t comp_cap,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,      /* dispatch-guarded == 512 */
        uint32_t n_split,
        const int32_t * __restrict__ positions,   /* dispatch-guarded != NULL */
        const int32_t * __restrict__ seq_id,
        /* C1 capture substrate.  The decode-scalars fields (n_raw/raw_start/
         * pos0) are all SUPERSEDED by the per-row positions[] derivation in
         * a positions-only kernel (same dead-under-positions property as the
         * generic kernels), so s_override is accepted for call-site
         * congruence but never read.  ls_override carries the LIVE n_comp --
         * without it a captured node replays the bake-time visible clamp
         * (the PC5 frozen-scalar class). */
        const struct ds4_decode_scalars * __restrict__ s_override,
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    __shared__ float kv_sm[DS4_ATTN_HG_ROWS][512 + 4];
    __shared__ float P_sm[DS4_ATTN_HG_HEADS][DS4_ATTN_HG_ROWS];
    __shared__ float m_sm[DS4_ATTN_HG_HEADS];
    __shared__ float l_sm[DS4_ATTN_HG_HEADS];
    __shared__ float f_sm[DS4_ATTN_HG_HEADS];
    __shared__ int32_t rowc_sm[DS4_ATTN_HG_ROWS];   /* comp pool id, -1 = invalid */
    (void)s_override;
    if (ls_override) {
        n_comp = ls_override->n_comp;
    }
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_indexed_read_path_blocks, 1ull);
    }
    const uint32_t sp = blockIdx.x;
    const uint32_t hg = blockIdx.y;
    const uint32_t t  = blockIdx.z;
    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || positions == NULL) return;
    const uint32_t qpos = (uint32_t)positions[t];
    uint32_t row_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
    if (row_n_raw > raw_cap) row_n_raw = raw_cap;
    const uint32_t row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    const uint32_t raw_count = row_n_raw > 256u ? 256u : row_n_raw;
    uint32_t visible = 0;
    if (ratio != 0u && n_comp != 0u) {
        visible = (qpos + 1u) / ratio;
        if (visible > qpos / ratio) visible = qpos / ratio;
        if (visible > n_comp) visible = n_comp;
    }
    const uint32_t seq_base      = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    const uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const uint32_t n_dom = raw_count + top_k;
    const uint32_t chunk = (n_dom + n_split - 1u) / n_split;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_dom) hi = n_dom;
    const float scale = rsqrtf((float)head_dim);
    /* q for this warp's head, in registers (head_dim 512 / 32 lanes). */
    float q_reg[16];
    {
        const uint32_t h = hg * DS4_ATTN_HG_HEADS + warp;
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
#pragma unroll
        for (uint32_t k = 0; k < 16u; k++) q_reg[k] = qh[lane + 32u * k];
    }
    if (tid < DS4_ATTN_HG_HEADS) { m_sm[tid] = -INFINITY; l_sm[tid] = 0.0f; }
    __syncthreads();
    const uint32_t d0 = (tid * 2u) & 511u;
    const uint32_t d1 = d0 + 1u;
    float O0[DS4_ATTN_HG_HEADS], O1[DS4_ATTN_HG_HEADS];
#pragma unroll
    for (uint32_t h = 0; h < DS4_ATTN_HG_HEADS; h++) { O0[h] = 0.0f; O1[h] = 0.0f; }

    for (uint32_t tile = lo; tile < hi; tile += DS4_ATTN_HG_ROWS) {
        const uint32_t nrows = (hi - tile) < DS4_ATTN_HG_ROWS ? (hi - tile) : DS4_ATTN_HG_ROWS;
        /* Resolve this tile's topk ids (one thread per row; broadcast reads). */
        if (tid < DS4_ATTN_HG_ROWS) {
            int32_t c = -1;
            const uint32_t row = tile + tid;
            if (tid < nrows && row >= raw_count) {
                const int32_t ci = topk[(uint64_t)t * top_k + (row - raw_count)];
                if (ci >= 0 && (uint32_t)ci < visible) c = ci;
            }
            rowc_sm[tid] = c;
        }
        __syncthreads();
        /* Stage-decode the tile into f32 smem once for the whole head group.
         * Pad AND invalid rows are ZEROED (a stale-NaN row survives P=0). */
        for (uint32_t i4 = tid; i4 < DS4_ATTN_HG_ROWS * 128u; i4 += blockDim.x) {
            const uint32_t r = i4 >> 7u;
            const uint32_t d = (i4 & 127u) * 4u;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (r < nrows) {
                const uint32_t row = tile + r;
                if (row < raw_count) {
                    const uint32_t rr = seq_base + ((row_raw_start + row) % raw_cap);
                    const float4 f4 = *(const float4 *)(raw_kv + (uint64_t)rr * head_dim + d);
                    v0 = f4.x; v1 = f4.y; v2 = f4.z; v3 = f4.w;
                } else if (rowc_sm[r] >= 0) {
                    const uint32_t c = comp_seq_base + (uint32_t)rowc_sm[r];
                    if (use_fp8) {
                        const unsigned char *cp = comp_fp8 +
                            (uint64_t)c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
                        if (d < DS4_OPP_C_FP8_NOPE_DEV) {
                            const uchar4 q4 = *(const uchar4 *)(cp + d);
                            const float sc = comp_scale[(uint64_t)c *
                                                        DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6u)];
                            const float m0 = dsv4_e4m3fn_decode_table[q4.x & 0x7fu];
                            const float m1 = dsv4_e4m3fn_decode_table[q4.y & 0x7fu];
                            const float m2 = dsv4_e4m3fn_decode_table[q4.z & 0x7fu];
                            const float m3 = dsv4_e4m3fn_decode_table[q4.w & 0x7fu];
                            v0 = ((q4.x & 0x80u) ? -m0 : m0) * sc;
                            v1 = ((q4.y & 0x80u) ? -m1 : m1) * sc;
                            v2 = ((q4.z & 0x80u) ? -m2 : m2) * sc;
                            v3 = ((q4.w & 0x80u) ? -m3 : m3) * sc;
                        } else {
                            const float4 f4 = *(const float4 *)(cp + DS4_OPP_C_FP8_NOPE_DEV +
                                (uint64_t)(d - DS4_OPP_C_FP8_NOPE_DEV) * sizeof(float));
                            v0 = f4.x; v1 = f4.y; v2 = f4.z; v3 = f4.w;
                        }
                    } else {
                        const float4 f4 = *(const float4 *)(comp_kv +
                            (uint64_t)c * head_dim + d);
                        v0 = f4.x; v1 = f4.y; v2 = f4.z; v3 = f4.w;
                    }
                }
            }
            kv_sm[r][d]      = v0;
            kv_sm[r][d + 1u] = v1;
            kv_sm[r][d + 2u] = v2;
            kv_sm[r][d + 3u] = v3;
        }
        __syncthreads();
        /* warp-per-head dots from smem, q from registers */
        float s_r[DS4_ATTN_HG_ROWS];
#pragma unroll
        for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) s_r[r] = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 16u; k++) {
            const uint32_t d = lane + 32u * k;
            const float qv = q_reg[k];
#pragma unroll
            for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++)
                s_r[r] = __fmaf_rn(qv, kv_sm[r][d], s_r[r]);
        }
#pragma unroll
        for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
            for (uint32_t off = 16u; off > 0u; off >>= 1u)
                s_r[r] += __shfl_down_sync(0xffffffffu, s_r[r], off);
        }
        if (lane == 0) {
            float m_new = m_sm[warp];
            for (uint32_t r = 0; r < nrows; r++) {
                const bool valid = (tile + r < raw_count) || rowc_sm[r] >= 0;
                if (valid) m_new = fmaxf(m_new, s_r[r] * scale);
            }
            /* All-invalid tile guard (see kernel comment): m_new can stay
             * -INF here; f=1 is exact (l_add 0, O rescale must be no-op). */
            const float f = (m_new == -INFINITY) ? 1.0f : expf(m_sm[warp] - m_new);
            float l_add = 0.0f;
            for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
                const bool valid = r < nrows &&
                    ((tile + r < raw_count) || rowc_sm[r] >= 0);
                const float p = valid ? expf(s_r[r] * scale - m_new) : 0.0f;
                P_sm[warp][r] = p;
                l_add += p;
            }
            l_sm[warp] = l_sm[warp] * f + l_add;
            m_sm[warp] = m_new;
            f_sm[warp] = f;
        }
        __syncthreads();
        /* V accumulate: preload the tile's kv at (d0, d1), run heads from regs */
        float kv0[DS4_ATTN_HG_ROWS], kv1[DS4_ATTN_HG_ROWS];
#pragma unroll
        for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
            kv0[r] = kv_sm[r][d0];
            kv1[r] = kv_sm[r][d1];
        }
#pragma unroll
        for (uint32_t h = 0; h < DS4_ATTN_HG_HEADS; h++) {
            const float f = f_sm[h];
            float o0 = O0[h] * f, o1 = O1[h] * f;
#pragma unroll
            for (uint32_t r = 0; r < DS4_ATTN_HG_ROWS; r++) {
                const float p = P_sm[h][r];
                o0 = __fmaf_rn(p, kv0[r], o0);
                o1 = __fmaf_rn(p, kv1[r], o1);
            }
            O0[h] = o0; O1[h] = o1;
        }
        __syncthreads();
    }
    for (uint32_t h = 0; h < DS4_ATTN_HG_HEADS; h++) {
        const uint32_t gh = hg * DS4_ATTN_HG_HEADS + h;
        float *out = partials +
            (uint64_t)(((uint64_t)t * n_head + gh) * n_split + sp) * (head_dim + 2u);
        if (tid == 0) {
            out[0] = m_sm[h];
            out[1] = l_sm[h];
        }
        out[2u + d0] = O0[h];
        out[2u + d1] = O1[h];
    }
}

__global__ static void attention_fp8_predecode_kernel(
        float * __restrict__ scratch,
        const int32_t * __restrict__ topk,
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t head_dim,
        uint32_t ratio,
        /* PC5 (live-pos substrate fix, 2026-05-28 postmortem): predecode
         * computes visible_comp from pos0 and applies the same filter as
         * attention_indexed_mixed_kernel (see line ~5779 for the full
         * postmortem).  Without s_override the filter uses captured pos0
         * on replay and writes a stale subset of comp rows into scratch,
         * which the FP32 attention kernel then reads -- producing a
         * predecode-specific capture attractor distinct from both the
         * (now-fixed) scalar-FP8 path and the FP32 path.  Read pos0 live
         * from the same substrate the attention kernel uses; the two
         * kernels must agree on which c values to decode and read. */
        const struct ds4_decode_scalars * __restrict__ s_override,
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    if (s_override) {
        pos0   = s_override->pos0;
    }
    if (ls_override) {
        n_comp = ls_override->n_comp;
    }
    uint32_t t = blockIdx.x;
    uint32_t i = blockIdx.y;
    if (t >= n_tokens || i >= top_k) return;

    uint32_t qpos = pos0 + t;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    int32_t c = topk[(uint64_t)t * top_k + i];
    if (c < 0 || (uint32_t)c >= visible_comp) return;

    float *out = scratch + (uint64_t)c * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = fp8_kv_read(comp_fp8, comp_scale, (uint32_t)c, d);
    }
}

/* P2 Inc2b: per-seq (t, slot)-packed predecode.  The serial predecode above
 * is c-indexed (scratch row = bank-LOCAL comp row), which collides across
 * banks when per-seq rows select the same local row -- so Inc2a kept per-seq
 * decode on in-kernel scalar fp8_kv_read() at a measured 15-20% step cost.
 * This variant packs by (t, compacted slot) instead: block (t, i) recomputes
 * the consumer's deterministic single-thread topk compaction (count of valid
 * ids before i; same order, same validity filter -- see the 2026-05-26
 * nondeterminism postmortem in attention_indexed_mixed_kernel) and decodes
 * absolute row comp_seq_base + c into scratch[(t*top_k + slot) * head_dim].
 * attention_indexed_mixed_kernel then reads slot s of row t directly: its
 * comp_rows[s] is the s-th valid id by construction.  Bit-identical to the
 * in-kernel path (same fp8_kv_read).  Counts on the indexed tripwire so the
 * MIRROR_READ gates stay armed when this path supersedes in-kernel reads. */
__global__ static void attention_fp8_predecode_tslot_kernel(
        float * __restrict__ scratch,
        const int32_t * __restrict__ topk,
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t head_dim,
        uint32_t ratio,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        uint32_t comp_cap,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = blockIdx.y;
    if (t >= n_tokens || i >= top_k) return;
    if (threadIdx.x == 0 && i == 0) {
        atomicAdd(&g_fp8_kv_indexed_read_path_blocks, 1ull);
    }
    /* Per-row visible_comp: EXACT copy of the consumer's perseq formula. */
    const uint32_t qpos = (uint32_t)positions[t];
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > qpos / ratio) visible_comp = qpos / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const int32_t c = topk[(uint64_t)t * top_k + i];
    if (c < 0 || (uint32_t)c >= visible_comp) return;
    /* Compacted slot = number of valid ids before i (order-preserving, every
     * thread computes the same value; the broadcast loads hit cache). */
    uint32_t slot = 0;
    for (uint32_t j = 0; j < i; j++) {
        const int32_t cj = topk[(uint64_t)t * top_k + j];
        if (cj >= 0 && (uint32_t)cj < visible_comp) slot++;
    }
    if (slot >= 512u) return;   /* consumer caps comp_rows[] at 512 */
    const uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    float *out = scratch + ((uint64_t)t * top_k + slot) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = fp8_kv_read(comp_fp8, comp_scale, comp_seq_base + (uint32_t)c, d);
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        /* Phase 2 Step 4b: per-seq compressed-cache bank stride (rows).  When
         * seq_id is set, row t reads its topk-selected compressed rows from
         * bank seq_id[t] at base seq_id[t]*comp_cap; seq_id==NULL keeps
         * comp_seq_base=0 (single bank, comp_cap ignored, bit-exact). */
        uint32_t comp_cap,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (n_tokens
         * entries).  positions -> row query position; seq_id -> per-seq raw
         * bank base AND (Step 4b) per-seq compressed bank base seq_id[t]*comp_cap.
         * Both NULL = single-sequence scalar path. */
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const struct ds4_decode_scalars * __restrict__ s_override,
        /* Step 4c A1: per-layer n_comp override.  See
         * attention_decode_mixed_kernel for rationale. */
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        /* Opp C Phase 1A.4: optional packed FP8 mirror of the compressed
         * rows.  When both pointers are non-NULL the kernel reads
         * compressed lanes (selected by topk -> comp_rows[]) via
         * fp8_kv_read() instead of `comp_kv`; bit-identical reconstruction
         * (see the dense decode kernel for the rationale and the Phase-0
         * numerics test).  NULL when DS4_CUDA_FP8_KV is off; raw KV and
         * the rotary tail handling are untouched. */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        /* P2 Inc2b: optional (t, slot)-packed predecoded comp rows (see
         * attention_fp8_predecode_tslot_kernel).  When non-NULL, comp slot s
         * of row t reads scratch[(t*top_k + s) * head_dim] -- the predecode
         * wrote the compacted slots in this kernel's exact order.  The
         * launch NULLs comp_fp8/comp_scale alongside, mirroring the serial
         * predecode's comp_kv swap. */
        const float * __restrict__ comp_predec,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    if (s_override) {
        n_raw     = s_override->n_raw;
        raw_start = s_override->raw_start;
        /* PC5 (live-pos substrate fix, 2026-05-28 long-context capture-vs-eager
         * postmortem): pos0 must be read live from the substrate too, not just
         * its derived n_raw/raw_start.  The kernel uses pos0 to compute
         * visible_comp = (qpos+1)/ratio, which is the upper bound on which
         * topk[]-selected compressed rows the attention loop considers
         * (filter at line ~5856: `if (c >= 0 && c < visible_comp)`).
         *
         * At capture time the by-value pos0 arg is baked into the kernel-node
         * arg list at queue time.  On replay at a much later token, frozen
         * pos0 gives a visible_comp far smaller than the live formula would
         * yield, and the `min(formula, live n_comp)` clamp picks the frozen
         * value -- silently dropping every topk entry whose compressed-row
         * index lies between visible_comp_capture and visible_comp_live.
         * The frozen value of n_raw/raw_start was already a symptom of the
         * same class (PC4); the override below was made live for those but
         * not for the underlying pos0, leaving the formula inconsistent
         * (frozen pos0 + live n_raw -> meaningless first_raw_pos).
         *
         * Manifestation: FP32 long-n=128 capture-vs-eager parity failed on
         * /tmp/long_prompt.txt with first divergent gen idx 40 (abs pos
         * ~10542).  Hash-dump bisect localized the first-divergent kernel
         * to attention_indexed_mixed_kernel at il=2 with all probed inputs
         * (q, raw_cache, comp_cache, comp_selected) bit-identical to BE,
         * but heads.after_rope_back differing.  Read pos0 live from the
         * decode-scalars substrate (sibling field to n_raw/raw_start) so
         * visible_comp tracks the live position. */
        pos0      = s_override->pos0;
    }
    if (ls_override) {
        n_comp    = ls_override->n_comp;
    }
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);
    if (use_fp8 && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_indexed_read_path_blocks, 1ull);
    }
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    /* Phase 2 Step 3: per-row position (see attention_decode_mixed_kernel). */
    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    /* Phase 2 Step 3c: per-row raw span (see attention_decode_mixed_kernel). */
    uint32_t row_n_raw     = n_raw;
    uint32_t row_raw_start = raw_start;
    if (positions) {
        row_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (row_n_raw > raw_cap) row_n_raw = raw_cap;
        row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    }
    uint32_t first_raw_pos = positions ? (qpos + 1u - row_n_raw) : (pos0 + n_tokens - n_raw);
    uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Phase 2 Step 4b: per-seq compressed bank base.  comp_rows[] indices are
     * into this row's OWN n_comp, so the absolute row is comp_seq_base+comp_rows[c]. */
    uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        /* Step 4c: per-row varlen compressed count (see attention_decode_mixed_kernel). */
        if (positions && visible_comp > qpos / ratio) visible_comp = qpos / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (row_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + row_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = seq_base + ((row_raw_start + raw_first_idx + r) % raw_cap);
    }
    /* Determinism fix (2026-05-26 long-context nondeterminism postmortem):
     * the prior parallel `atomicAdd(&comp_count, 1u)` compaction produced an
     * order-nondeterministic comp_rows[] across runs (CUDA does not specify
     * the order in which racing atomicAdds win), which in turn shuffled the
     * FP accumulation order in the score / output reductions below.  At long
     * input context (indexer-fires regime, n_comp > top_k) that 1 ULP per row
     * snowballs into different argmax tokens within ~32-64 decoded positions
     * -- see local/docs/ds4_long_context_nondeterminism_2026-05-26.md.
     *
     * Single-threaded sequential fill (matches the sibling
     * attention_indexed_mixed_heads8_rb4_kernel pattern at line 5935) keeps
     * comp_rows[] in the same order as the input topk every run, restoring
     * bit-identical decode output past the indexer-firing boundary.  The
     * extra single-thread loop costs ~top_k iterations of one-int reads per
     * block; at decode (top_k=512, ~64 blocks per layer call, ~21 ratio-4
     * layers, ~100 tok/s) this is a few percent of attention time -- a clear
     * win over keeping the bug. */
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) {
                comp_rows[comp_count++] = (uint32_t)c;
            }
        }
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const bool fp8_row = use_fp8 && (row >= raw_count);
                const uint32_t fp8_c = fp8_row ? comp_seq_base + comp_rows[row - raw_count] : 0u;
                const float *kvrow = NULL;
                if (!fp8_row) {
                    if (comp_predec != NULL && row >= raw_count) {
                        kvrow = comp_predec + ((uint64_t)t * top_k + (row - raw_count)) * head_dim;
                    } else {
                        kvrow = row < raw_count
                            ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                            : comp_kv + (uint64_t)(comp_seq_base + comp_rows[row - raw_count]) * head_dim;
                    }
                }
                float dot = 0.0f;
                if (fp8_row) {
                    /* Hoisted fp8 row read (see attention_decode_mixed): row
                     * base pointers once per row, per-element work reduced to
                     * byte load + table + sign + FMA.  d order (qlane-strided)
                     * preserved and accumulation pinned with intrinsics (see
                     * the quad-dot comment) — shuffle-reduce order, and thus
                     * spec-verify counter-identity, preserved exactly. */
                    const unsigned char *cp = comp_fp8 +
                        (uint64_t)fp8_c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
                    const float *scp = comp_scale +
                        (uint64_t)fp8_c * DS4_OPP_C_FP8_BLOCKS_DEV;
                    const float *tail = (const float *)(cp + DS4_OPP_C_FP8_NOPE_DEV);
                    uint32_t d = qlane;
                    for (; d < DS4_OPP_C_FP8_NOPE_DEV; d += 8u) {
                        const unsigned char code = cp[d];
                        const float mag = dsv4_e4m3fn_decode_table[code & 0x7fu];
                        dot = __fmaf_rn(qh[d],
                                __fmul_rn((code & 0x80u) ? -mag : mag, scp[d >> 6u]), dot);
                    }
                    for (; d < head_dim; d += 8u) {
                        dot = __fmaf_rn(qh[d], tail[d - DS4_OPP_C_FP8_NOPE_DEV], dot);
                    }
                } else {
                    /* Pinned like its fp8 partner above (identity is
                     * structural only if BOTH branches fix the chain).
                     * Also serves predecode-scratch reads: those hold
                     * bit-exact decoded F32 values, so the same pinned
                     * chain keeps them congruent too. */
                    for (uint32_t d = qlane; d < head_dim; d += 8u)
                        dot = __fmaf_rn(qh[d], kvrow[d], dot);
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        /* Pair lane layout (d0,d1 = 2t,2t+1) — see attention_decode_mixed:
         * enables the uchar2 fp8 V read; per-d values unchanged under any
         * lane mapping, raw/predec/F32 reads stay equally coalesced. */
        uint32_t d0 = threadIdx.x * 2u;
        uint32_t d1 = d0 + 1u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        if (comp_predec != NULL) {
            for (uint32_t c = 0; c < comp_count; c++) {
                float s = scores[raw_count + c];
                const float *kv = comp_predec + ((uint64_t)t * top_k + c) * head_dim;
                acc0 += kv[d0] * s;
                acc1 += kv[d1] * s;
            }
        } else if (use_fp8) {
            /* Pair-lane fp8 V read (see attention_decode_mixed): one uchar2
             * sharing one scale per pair; rows are top-k-scattered so the
             * row base is computed per ROW (hoisted from per-element).
             * Value math and per-d accumulation order unchanged. */
            if (d0 < DS4_OPP_C_FP8_NOPE_DEV) {              /* uniform per warp */
                const uint32_t blk = d0 >> 6u;
                for (uint32_t c = 0; c < comp_count; c++) {
                    const float s = scores[raw_count + c];
                    const uint64_t cr = (uint64_t)(comp_seq_base + comp_rows[c]);
                    const uchar2 k = *(const uchar2 *)(comp_fp8 +
                        cr * DS4_OPP_C_FP8_ROW_BYTES_DEV + d0);
                    const float sc = comp_scale[cr * DS4_OPP_C_FP8_BLOCKS_DEV + blk];
                    const float m0 = dsv4_e4m3fn_decode_table[k.x & 0x7fu];
                    const float m1 = dsv4_e4m3fn_decode_table[k.y & 0x7fu];
                    acc0 += (((k.x & 0x80u) ? -m0 : m0) * sc) * s;
                    acc1 += (((k.y & 0x80u) ? -m1 : m1) * sc) * s;
                }
            } else {
                const uint32_t toff = DS4_OPP_C_FP8_NOPE_DEV +
                    (d0 - DS4_OPP_C_FP8_NOPE_DEV) * (uint32_t)sizeof(float);
                for (uint32_t c = 0; c < comp_count; c++) {
                    const float s = scores[raw_count + c];
                    const uint64_t cr = (uint64_t)(comp_seq_base + comp_rows[c]);
                    const float2 v = *(const float2 *)(comp_fp8 +
                        cr * DS4_OPP_C_FP8_ROW_BYTES_DEV + toff);
                    acc0 += v.x * s;
                    acc1 += v.y * s;
                }
            }
        } else {
            for (uint32_t c = 0; c < comp_count; c++) {
                float s = scores[raw_count + c];
                const float *kv = comp_kv + (uint64_t)(comp_seq_base + comp_rows[c]) * head_dim;
                acc0 += kv[d0] * s;
                acc1 += kv[d1] * s;
            }
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            if (comp_predec != NULL) {
                for (uint32_t s = 0; s < comp_count; s++) acc += comp_predec[((uint64_t)t * top_k + s) * head_dim + d] * scores[raw_count + s];
            } else if (use_fp8) {
                for (uint32_t s = 0; s < comp_count; s++) {
                    acc += fp8_kv_read(comp_fp8, comp_scale, comp_seq_base + comp_rows[s], d) * scores[raw_count + s];
                }
            } else {
                for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)(comp_seq_base + comp_rows[s]) * head_dim + d] * scores[raw_count + s];
            }
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* P2 Inc2b: optional packed FP8 mirror (single-seq kernel: bare row
         * indices, same base as comp_kv -- absolute by construction). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();
    if (use_fp8 && comp_count != 0u && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_indexed_read_path_blocks, 1ull);
    }

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = ((const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim))[c4];
            } else if (use_fp8) {
                kv_shared[off] = fp8_kv_read4(comp_fp8, comp_scale, comp_rows[sr - raw_count], c4);
            } else {
                kv_shared[off] = ((const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim))[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = ((const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim))[c4];
            } else if (use_fp8) {
                kv_shared[off] = fp8_kv_read4(comp_fp8, comp_scale, comp_rows[sr - raw_count], c4);
            } else {
                kv_shared[off] = ((const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim))[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* min 2 blocks/SM caps ptxas at 64 regs/thread (512 thr x 64 x 2 = the full
 * 64K regfile).  Native sm_121 ptxas otherwise picks 72 regs, which fits only
 * ONE block/SM: occupancy halves and this kernel went 53 -> 85 ms/launch at
 * W4096 prefill (sm_75-JIT ran at 64 regs / 2 blocks -- the proven point). */
template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void __launch_bounds__(512, 2) attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* FB1 (per-seq admission chunks): optional per-row positions[]/seq_id[]
         * + per-seq compressed bank stride, mirroring the generic
         * attention_indexed_mixed_kernel semantics exactly: per-row raw window
         * derived from positions[t], raw ring base seq_id[t]*raw_cap, comp bank
         * base seq_id[t]*comp_cap, floor(qpos/ratio) per-row visible clamp, and
         * the deterministic single-thread topk compaction (the topk may carry
         * -1 / over-visible ids per row in multi-seq).  All NULL/0 keeps the
         * single-seq path bit-exact. */
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        uint32_t comp_cap,
        /* P2 Inc2b: optional packed FP8 mirror (absolute rows: this kernel
         * reads comp at comp_seq_base + id, matching the generic kernel). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count_s;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t row_n_raw     = n_raw;
    uint32_t row_raw_start = raw_start;
    if (positions) {
        row_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (row_n_raw > raw_cap) row_n_raw = raw_cap;
        row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    }
    uint32_t first_raw_pos = positions ? (qpos + 1u - row_n_raw) : (pos0 + n_tokens - n_raw);
    const uint32_t seq_base      = seq_id ? (uint32_t)seq_id[t] * raw_cap  : 0u;
    const uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (positions && visible_comp > qpos / ratio) visible_comp = qpos / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (row_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + row_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        /* FB1: deterministic per-row topk compaction (multi-seq only) -- the
         * same order-preserving single-thread fill as the generic kernel (see
         * the 2026-05-26 nondeterminism postmortem there). */
        comp_count_s = 0;
        if (positions) {
            uint32_t cc = 0;
            for (uint32_t i = 0; i < top_k && cc < 512u; i++) {
                const int32_t c = topk[(uint64_t)t * top_k + i];
                if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[cc++] = (uint32_t)c;
            }
            comp_count_s = cc;
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = seq_base + ((row_raw_start + raw_first_idx + r) % raw_cap);
    }
    __syncthreads();

    uint32_t comp_count;
    if (positions) {
        comp_count = comp_count_s;
    } else {
        comp_count = top_k < visible_comp ? top_k : visible_comp;
        if (comp_count > 512u) comp_count = 512u;
    }
    if (use_fp8 && comp_count != 0u && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_indexed_read_path_blocks, 1ull);
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = ((const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim))[c4];
            } else {
                const uint32_t comp_idx = positions
                    ? comp_rows[sr - raw_count]
                    : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
                kv_shared[off] = use_fp8
                    ? fp8_kv_read4(comp_fp8, comp_scale, comp_seq_base + comp_idx, c4)
                    : ((const float4 *)(comp_kv + (uint64_t)(comp_seq_base + comp_idx) * head_dim))[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* P2 Inc2b: optional packed FP8 mirror (single-seq prefill tier:
         * bare row indices, same base as comp_kv). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    if (use_fp8 && comp_count != 0u && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = ((const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim))[c4];
            } else if (use_fp8) {
                kv_shared[off] = fp8_kv_read4(comp_fp8, comp_scale, sr - raw_count, c4);
            } else {
                kv_shared[off] = ((const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim))[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

static constexpr uint32_t kTTTileTokens = 4u;
static constexpr uint32_t kTTG = 8u;
static constexpr uint32_t kTTM = 32u;
static constexpr uint32_t kTTStageRows = 32u;
static constexpr uint32_t kTTRawWindow = 128u;
static constexpr uint32_t kTTHeadDim = 512u;
static constexpr uint32_t kTTWarps = 16u;
static constexpr uint32_t kTTThreads = 512u;
static constexpr uint32_t kTTScoreKQuarters = 4u;
static constexpr uint32_t kTTScoreKSliceDim = kTTHeadDim / kTTScoreKQuarters;
static constexpr uint32_t kTTScoreKStepsPerQuarter = kTTScoreKSliceDim / 16u;
static constexpr uint32_t kTTRecordRingPlanes = 4u;
static constexpr uint32_t kTTProbStride = 40u;
static constexpr uint32_t kTTRingChunkBytes = 16u;
static constexpr uint32_t kTTRingChunksPerRow =
    (kTTHeadDim * sizeof(half)) / kTTRingChunkBytes;
static constexpr size_t kTTSmemHardCap = 90ull * 1024ull;

static_assert(kTTM == kTTTileTokens * kTTG, "token-tile M must be 4 tokens x G8");
static_assert(kTTProbStride == kTTStageRows + 8u, "token-tile prob stride changed");
static_assert(kTTScoreKSliceDim % 16u == 0, "token-tile score K split changed");
static_assert(kTTRingChunksPerRow == 64u, "token-tile KV ring expects 64 chunks");
static_assert(sizeof(int2) == 8u, "token-tile union record must remain 8 bytes");

template <uint32_t TT_STAGE_ROWS>
struct tt_TokentileLayout {
    static constexpr uint32_t prob_stride = TT_STAGE_ROWS + 8u;
    static constexpr uint32_t ring_plane_bytes =
        TT_STAGE_ROWS * kTTRingChunksPerRow * kTTRingChunkBytes;
    static constexpr uint32_t ring_plane_elems = ring_plane_bytes / sizeof(half);
};

__device__ static float tt_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float tt_warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

[[maybe_unused]] __device__ static float tt_dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ uint32_t tt_lane_id(void) {
    return threadIdx.x & 31u;
}

__device__ __forceinline__ uint32_t tt_warp_id(void) {
    return threadIdx.x >> 5u;
}

__device__ __forceinline__ int tt_mma_c_i(uint32_t lane, int l) {
    return ((l >> 1) << 3) + (int)(lane >> 2);
}

__device__ __forceinline__ int tt_mma_c_j(uint32_t lane, int l) {
    return (int)((lane & 3u) << 1) + (l & 1);
}

__device__ __forceinline__ unsigned tt_smem_addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t tt_ring_off_bytes(uint32_t row, uint32_t c) {
    return (row * kTTRingChunksPerRow + (c ^ (row & 7u))) * kTTRingChunkBytes;
}

__device__ __forceinline__ void tt_ldmatrix_x4_addr(uint32_t (&r)[4], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x4(uint32_t (&r)[4], const void *p) {
    tt_ldmatrix_x4_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_ldmatrix_x2(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_trans_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_mma_m16n8k16_f16_f32(
        float *d,
        const uint32_t (&a)[4],
        const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t d0 = __float_as_uint(d[0]);
    uint32_t d1 = __float_as_uint(d[1]);
    uint32_t d2 = __float_as_uint(d[2]);
    uint32_t d3 = __float_as_uint(d[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0);
    d[1] = __uint_as_float(d1);
    d[2] = __uint_as_float(d2);
    d[3] = __uint_as_float(d3);
#else
    (void)a;
    (void)b;
#endif
}

__device__ __forceinline__ unsigned char *tt_align16(unsigned char *p) {
    uintptr_t x = reinterpret_cast<uintptr_t>(p);
    x = (x + 15u) & ~uintptr_t(15u);
    return reinterpret_cast<unsigned char *>(x);
}

__device__ __forceinline__ void tt_zero_16B(void *dst) {
    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
}

__device__ __forceinline__ void tt_zero_8B(void *dst) {
    *reinterpret_cast<int2 *>(dst) = make_int2(0, 0);
}

__device__ __forceinline__ void tt_cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_16B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_16B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_8B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_8B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void tt_cp_async_wait_group(void) {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ uint32_t tt_score_partial_slot(
        uint32_t kq,
        uint32_t m,
        uint32_t r) {
    return (kq + r + m) & 3u;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_store_score_partial(
        float4 * __restrict__ partials,
        uint32_t kq,
        uint32_t m,
        uint32_t r,
        float v) {
    float *dst = &partials[m * TT_STAGE_ROWS + r].x;
    dst[tt_score_partial_slot(kq, m, r)] = v;
}

__device__ __forceinline__ float4 tt_load_score_partial_record(
        const float4 * __restrict__ p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float x, y, z, w;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "r"(tt_smem_addr(p)));
    return make_float4(x, y, z, w);
#else
    return *p;
#endif
}

__device__ __forceinline__ void tt_issue_cp_async_row(
        half * __restrict__ dst,
        uint32_t rr,
        uint32_t lane16,
        const half * __restrict__ src,
        bool live) {
    const char *src_b = reinterpret_cast<const char *>(src);
    char *dst_b = reinterpret_cast<char *>(dst);

#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t chunk = lane16 + i * 16u;
        char *db = dst_b + tt_ring_off_bytes(rr, chunk);
        tt_cp_async_16B(db, live ? static_cast<const void *>(src_b + chunk * 16u)
                                 : static_cast<const void *>(db),
                        live);
    }
}

__device__ __forceinline__ uint32_t tt_stage_raw_rows(
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count) {
    uint32_t raw_rows = 0;
    if (row0 < raw_union_count) {
        const uint32_t raw_left = raw_union_count - row0;
        raw_rows = raw_left < nr ? raw_left : nr;
    }
    return raw_rows;
}

__global__ static void __launch_bounds__(512, 1) attention_tokentile_union_build_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *topk,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    extern __shared__ uint32_t bitmap[];
    __shared__ uint32_t scan[513];
    __shared__ uint32_t running_s;

    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t bitmap_words = (n_comp + 1u) >> 1u;

    for (uint32_t w = tid; w < bitmap_words; w += blockDim.x) {
        bitmap[w] = 0u;
    }
    if (tid == 0u) running_s = 0u;
    __syncthreads();

    const uint32_t total_slots = tile_count * top_k;
    for (uint32_t idx = tid; idx < total_slots; idx += blockDim.x) {
        const uint32_t tok = idx / top_k;
        const uint32_t i = idx - tok * top_k;
        const uint32_t t = tile_base + tok;
        const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t visible = ratio ? (positions ? qpos / ratio : (qpos + 1u) / ratio) : n_comp;
        if (visible > n_comp) visible = n_comp;
        const int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible) {
            const uint32_t cu = (uint32_t)c;
            const uint32_t bits = ((uint32_t)(1u << tok)) << ((cu & 1u) * 16u);
            atomicOr(&bitmap[cu >> 1u], bits);
        }
    }
    __syncthreads();

    for (uint32_t base = 0u; base < n_comp; base += blockDim.x) {
        const uint32_t id = base + tid;
        uint32_t mask = 0u;
        if (id < n_comp) {
            const uint32_t word = bitmap[id >> 1u];
            mask = (id & 1u) ? (word >> 16u) : (word & 0xffffu);
        }
        const uint32_t pred = mask != 0u ? 1u : 0u;
        if (tid == 0u) scan[0] = 0u;
        scan[tid + 1u] = pred;
        __syncthreads();
        for (uint32_t off = 1u; off < blockDim.x; off <<= 1u) {
            uint32_t v = 0u;
            if (tid >= off) v = scan[tid + 1u - off];
            __syncthreads();
            scan[tid + 1u] += v;
            __syncthreads();
        }
        const uint32_t rank = scan[tid];
        const uint32_t total = scan[blockDim.x];
        const uint32_t running = running_s;
        if (pred) {
            records[(uint64_t)blockIdx.x * rec_stride + running + rank] =
                make_int2((int)id, (int)mask);
        }
        __syncthreads();
        if (tid == 0u) running_s = running + total;
        __syncthreads();
    }

    if (tid == 0u) counts[blockIdx.x] = running_s;
}

/* v0.5 inc-6: n_comp-independent union builder for the deep engage regime.
 * The bitmap builder above carries (n_comp+1)>>1 u32 words of dynamic smem
 * (16-bit token mask per comp) -- 64 KiB at n_comp 32768, past the device
 * ceiling beyond -- which was the whole n_comp <= 32768 engage gate
 * (attention stepped 569 -> 1014 ms/chunk past 131k tokens).  A tile only
 * ever holds kTTTileTokens * top_k = 2048 candidates, so instead: pack
 * (comp_id << 4) | tok into u32 (top_k == 512 by dispatch gate; comp ids
 * are far below 2^28), block-radix-sort ascending (33.8 KiB static temp,
 * n_comp-independent, no opt-in attribute), dedup adjacent ids, OR token
 * bits into the pre-zeroed record masks -- runs straddling thread
 * boundaries land via atomicOr at scan[tid]-1, and ascending order makes
 * the globally-first valid key always a start, so that rank exists.
 * Emits BIT-IDENTICAL records/counts to the bitmap builder (proven across
 * 72 shape-legs incl. both visibility branches, ragged tiles, -1 pads;
 * proto_attn_union_sort.cu).  Dispatch keeps the bitmap builder for
 * n_comp <= 32768 where it is faster (crossover ~40k; sort is flat
 * ~0.17 ms/launch at every depth). */
/* Items per thread = kTTTileTokens * top_k / 512 threads; top_k is pinned to
 * 512 by the dispatch gate, so this reduces to kTTTileTokens. */
static constexpr uint32_t kTTUnionItems = kTTTileTokens;
typedef cub::BlockRadixSort<uint32_t, 512, kTTUnionItems> tt_UnionSortT;

__global__ static void __launch_bounds__(512, 1) attention_tokentile_union_sort_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *topk,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    __shared__ typename tt_UnionSortT::TempStorage sort_tmp;
    __shared__ uint32_t scan[513];
    __shared__ uint32_t edge_uid[512];
    const uint32_t kInvalid = 0xFFFFFFFFu;

    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;

    int2 *rec = records + (uint64_t)blockIdx.x * rec_stride;
    for (uint32_t r = tid; r < rec_stride; r += blockDim.x)
        ((int *)&rec[r])[1] = 0;

    uint32_t keys[kTTUnionItems];
    const uint32_t total_slots = tile_count * top_k;
#pragma unroll
    for (uint32_t i = 0; i < kTTUnionItems; i++) {
        const uint32_t idx = tid * kTTUnionItems + i;
        uint32_t key = kInvalid;
        if (idx < total_slots) {
            const uint32_t tok = idx / top_k;
            const uint32_t s = idx - tok * top_k;
            const uint32_t t = tile_base + tok;
            const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
            uint32_t visible = ratio
                ? (positions ? qpos / ratio : (qpos + 1u) / ratio) : n_comp;
            if (visible > n_comp) visible = n_comp;
            const int32_t c = topk[(uint64_t)t * top_k + s];
            if (c >= 0 && (uint32_t)c < visible)
                key = ((uint32_t)c << 4u) | tok;
        }
        keys[i] = key;
    }
    __syncthreads();
    tt_UnionSortT(sort_tmp).Sort(keys);
    __syncthreads();

    edge_uid[tid] = keys[kTTUnionItems - 1u] >> 4u;
    __syncthreads();
    const uint32_t prev_edge = tid ? edge_uid[tid - 1u] : kInvalid;

    uint32_t local_starts = 0;
    uint32_t prev_uid = prev_edge;
#pragma unroll
    for (uint32_t i = 0; i < kTTUnionItems; i++) {
        const uint32_t uid = keys[i] >> 4u;
        if (keys[i] != kInvalid && uid != prev_uid) local_starts++;
        prev_uid = uid;
    }
    if (tid == 0u) scan[0] = 0u;
    scan[tid + 1u] = local_starts;
    __syncthreads();
    for (uint32_t off = 1u; off < blockDim.x; off <<= 1u) {
        uint32_t v = 0u;
        if (tid >= off) v = scan[tid + 1u - off];
        __syncthreads();
        scan[tid + 1u] += v;
        __syncthreads();
    }
    const uint32_t total = scan[blockDim.x];

    uint32_t rank = scan[tid];
    prev_uid = prev_edge;
#pragma unroll
    for (uint32_t i = 0; i < kTTUnionItems; i++) {
        const uint32_t key = keys[i];
        if (key != kInvalid) {
            const uint32_t uid = key >> 4u;
            if (uid != prev_uid) {
                ((int *)&rec[rank])[0] = (int)uid;
                rank++;
            }
            atomicOr((unsigned int *)&((int *)&rec[rank - 1u])[1],
                     1u << (key & 0xFu));
        }
        prev_uid = key >> 4u;
    }
    __syncthreads();
    if (tid == 0u) counts[blockIdx.x] = total;
}

__global__ static void __launch_bounds__(256, 1) attention_tokentile_raw_mirror_kernel(
        half *dst,
        const float *raw_kv,
        const int32_t *seq_id,
        uint32_t tt_run_pos0,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (d0 >= head_dim) return;
    half *dst_row = dst + (uint64_t)row * head_dim;
    if (row < raw_row_min) {
        dst_row[d0] = __float2half(0.0f);
        if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const int64_t p = (int64_t)tt_run_pos0 - (int64_t)(kTTRawWindow - 1u) + (int64_t)row;
    uint32_t slot = 0u;
    if (seq_id) {
        slot = (uint32_t)seq_id[0] * raw_cap + (uint32_t)((uint64_t)p % raw_cap);
    } else {
        const uint32_t rel = (uint32_t)(p - (int64_t)first_raw_pos);
        slot = (raw_start + rel) % raw_cap;
    }
    const float *src = raw_kv + (uint64_t)slot * head_dim;
    dst_row[d0] = __float2half(src[d0]);
    if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(src[d0 + 1u]);
}

__global__ static void __launch_bounds__(256, 1) attention_tokentile_comp_mirror_kernel(
        half *dst,
        const float *comp_kv,
        const unsigned char *comp_fp8,
        const float *comp_scale,
        const int32_t *seq_id,
        uint32_t comp_cap,
        uint32_t n_comp,
        uint32_t head_dim,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    const uint32_t row = blockIdx.x;
    const uint32_t c4 = threadIdx.x;
    if (row >= n_comp || c4 >= (head_dim >> 2u)) return;
    const uint32_t base = seq_id ? (uint32_t)seq_id[0] * comp_cap : 0u;
    float4 v;
    if (comp_fp8 && comp_scale) {
        v = fp8_kv_read4(comp_fp8, comp_scale, base + row, c4);
    } else {
        v = ((const float4 *)(comp_kv + (uint64_t)(base + row) * head_dim))[c4];
    }
    half *out = dst + (uint64_t)row * head_dim + (c4 << 2u);
    out[0] = __float2half(v.x);
    out[1] = __float2half(v.y);
    out[2] = __float2half(v.z);
    out[3] = __float2half(v.w);
}

/* Decode-mixed port: the non-indexed layers see comp rows as the causal
 * RANGE [0, visible(tok)) -- floor(qpos/ratio) per-seq, floor((qpos+1)/ratio)
 * single-seq, clamped to n_comp -- not a topk set, so the per-tile union is
 * [0, max visible) and the per-token masks are arithmetic.  Emits the same
 * ascending-id interleaved int2 {comp_id, mask} records the hmma kernel's
 * record ring consumes; no bitmap, no scan. */
__global__ static void __launch_bounds__(512, 1) attention_tokentile_dense_build_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    __shared__ uint32_t visible_s[kTTTileTokens];
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    if (tid < kTTTileTokens) {
        uint32_t visible = 0u;
        if (tid < tile_count && n_comp != 0u && ratio != 0u) {
            const uint32_t t = tile_base + tid;
            const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
            visible = positions ? qpos / ratio : (qpos + 1u) / ratio;
            if (visible > n_comp) visible = n_comp;
        }
        visible_s[tid] = visible;
    }
    __syncthreads();
    uint32_t vmax = 0u;
    for (uint32_t k = 0u; k < tile_count; k++) {
        if (visible_s[k] > vmax) vmax = visible_s[k];
    }
    for (uint32_t c = tid; c < vmax; c += blockDim.x) {
        uint32_t mask = 0u;
        for (uint32_t k = 0u; k < tile_count; k++) {
            if (c < visible_s[k]) mask |= 1u << k;
        }
        records[(uint64_t)blockIdx.x * rec_stride + c] = make_int2((int)c, (int)mask);
    }
    if (tid == 0u) counts[blockIdx.x] = vmax;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_record_stage_cp_async(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile) {
    static_assert(TT_STAGE_ROWS == 32u, "token-tile record issue is fixed at R32");
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;
    const bool comp_live = active && rr >= raw_rows && rr < nr;
    if (comp_live && lane16 == 0u) {
        int2 *dst = rec_plane + rr;
        const uint32_t ci = row0 + rr - raw_union_count;
        tt_cp_async_8B(dst, union_records_tile + ci, true);
    }
}

template <uint32_t TT_STAGE_ROWS, bool USE_SMEM_RECORDS>
__device__ __forceinline__ void tt_issue_kv_stage_cp_async(
        half * __restrict__ dst,
        const int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile,
        uint32_t tile_base,
        const half * __restrict__ raw_kv,
        const half * __restrict__ comp_kv,
        uint32_t tid) {
    constexpr uint32_t kCp16PerRow = (kTTHeadDim * sizeof(half)) / 16u;
    static_assert(kCp16PerRow == 64u, "expected 64 cp.async chunks per f16 KV row");
    static_assert(TT_STAGE_ROWS == 32u, "token-tile KV issue is fixed at R32");
    (void)tid;
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;

    if (active && rr < raw_rows) {
        const uint32_t sr = row0 + rr;
        const half *src = raw_kv + (uint64_t)(tile_base + sr) * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= raw_rows && rr < nr) {
        uint32_t comp_id = 0u;
        if (USE_SMEM_RECORDS) {
            comp_id = (uint32_t)rec_plane[rr].x;
        } else {
            const uint32_t ci = row0 + rr - raw_union_count;
            comp_id = (uint32_t)union_records_tile[ci].x;
        }
        const half *src = comp_kv + (uint64_t)comp_id * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= nr) {
        tt_issue_cp_async_row(dst, rr, lane16, NULL, false);
    }
}

__device__ __forceinline__ void tt_load_score_q_frag(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const float * __restrict__ q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t k_base = kq * kTTScoreKSliceDim;
    const uint32_t lane = tt_lane_id();
    const uint32_t a_group = lane >> 2u;
    const uint32_t a_col_pair = (lane & 3u) << 1u;
#pragma unroll
    for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
        const uint32_t k0 = k_base + kt * 16u;
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint32_t m = mtile * 16u + a_group + ((r & 1u) ? 8u : 0u);
            const uint32_t tok = m / kTTG;
            const uint32_t h = m - tok * kTTG;
            const uint32_t gt = tile_base + tok;
            const uint32_t gh = head_base + h;
            const uint32_t d = k0 + a_col_pair + ((r & 2u) ? 8u : 0u);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (gt < n_tokens && gh < n_head) {
                const float *q_row = q + ((uint64_t)gt * n_head + gh) * kTTHeadDim;
                x0 = q_row[d];
                x1 = q_row[d + 1u];
            }
            const half2 packed = __floats2half2_rn(x0, x1);
            q_frag[kt][r] =
                (uint32_t)__half_as_ushort(__low2half(packed)) |
                ((uint32_t)__half_as_ushort(__high2half(packed)) << 16);
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_hmma_score_stage(
        float4 * __restrict__ partial_scores,
        const uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ kv_cur,
        uint32_t nr,
        float score_scale) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    constexpr uint32_t kNtiles = TT_STAGE_ROWS / 8u;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t lane = tt_lane_id();
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t score_row_lane = lane & 7u;
    const uint32_t score_chunk_lane = (lane >> 3u) & 1u;
    const uint32_t score_chunk_base = kq * (kTTScoreKSliceDim / 8u) + score_chunk_lane;
#pragma unroll
    for (uint32_t ntile = 0; ntile < kNtiles; ++ntile) {
        const uint32_t row_base = ntile * 8u;
        if (row_base < nr) {
            float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint32_t score_row = row_base + score_row_lane;
#pragma unroll
            for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
                uint32_t b[2];
                tt_ldmatrix_x2_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(score_row, score_chunk_base + kt * 2u));
                tt_mma_m16n8k16_f16_f32(s_frag, q_frag[kt], b);
            }
            const uint32_t m0 = mtile * 16u + (lane >> 2u);
            const uint32_t r0 = row_base + ((lane & 3u) << 1u);
            const uint32_t r1 = r0 + 1u;
            const uint32_t m1 = m0 + 8u;
            if (r0 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r0, s_frag[0] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r0, s_frag[2] * score_scale);
            }
            if (r1 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r1, s_frag[1] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r1, s_frag[3] * score_scale);
            }
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_softmax_stage(
        half * __restrict__ probs,
        float * __restrict__ stage_rescale,
        float * __restrict__ max_s,
        float * __restrict__ sum_s,
        const float4 * __restrict__ scores,
        const int2 * __restrict__ records,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        uint32_t tile_count,
        uint32_t tile_base,
        uint32_t raw_row_min) {
    constexpr uint32_t kMPerWarp = kTTM / kTTWarps;
    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static_assert(kTTM % kTTWarps == 0u, "softmax maps integral m rows per warp");
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t sr = row0 + lane;
    const bool lane_live = lane < TT_STAGE_ROWS && lane < nr;
    const bool raw_slot = lane_live && sr < raw_union_count;
    uint16_t comp_mask = 0u;
    if (lane_live && !raw_slot) {
        comp_mask = (uint16_t)records[lane].y;
    }
    const uint32_t prob_lane_base = warp * kProbStride + lane;

#pragma unroll
    for (uint32_t mi = 0; mi < kMPerWarp; ++mi) {
        const uint32_t m = warp + mi * kTTWarps;
        const uint32_t tok = m / kTTG;
        const bool valid_token = tok < tile_count;
        const uint32_t score_idx = m * TT_STAGE_ROWS + lane;
        const uint32_t prob_idx = prob_lane_base + mi * kTTWarps * kProbStride;
        float score = -INFINITY;
        if (lane_live && valid_token) {
            const bool selected = raw_slot
                ? (((uint32_t)(sr - tok) < kTTRawWindow) && (tile_base + sr >= raw_row_min))
                : ((comp_mask & (uint16_t)(1u << tok)) != 0u);
            if (selected) {
                const float4 parts = tt_load_score_partial_record(scores + score_idx);
                const float s01 = parts.x + parts.y;
                const float s23 = parts.z + parts.w;
                score = s01 + s23;
            }
        }

        const float stage_m = tt_warp_max_f32(score);
        const float old_m = max_s[m];
        const float new_m = fmaxf(old_m, stage_m);
        float old_scale = 1.0f;
        if (new_m != -INFINITY) {
            old_scale = (old_m == -INFINITY) ? 0.0f : expf(old_m - new_m);
        }
        const float row_scale = (score == -INFINITY || new_m == -INFINITY)
            ? 0.0f
            : expf(score - new_m);
        const float stage_sum = tt_warp_sum_f32(row_scale);

        if (lane < TT_STAGE_ROWS) {
            probs[prob_idx] = __float2half(lane < nr ? row_scale : 0.0f);
        }
        if (lane == 0u) {
            max_s[m] = new_m;
            sum_s[m] = sum_s[m] * old_scale + stage_sum;
            stage_rescale[m] = old_scale;
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_pv_mma_stage(
        float (&o_acc)[2u * kTTTileTokens * kTTG],
        const half * __restrict__ probs,
        const float * __restrict__ stage_rescale,
        const half * __restrict__ kv_cur) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
        const float *scale0 = stage_rescale + mtile * 16u + (lane >> 2u);
        const float *scale1 = scale0 + 8u;
        const float rs0 = *scale0;
        const float rs1 = *scale1;
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
            const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
            o_acc[idx + 0u] *= rs0;
            o_acc[idx + 1u] *= rs0;
            o_acc[idx + 2u] *= rs1;
            o_acc[idx + 3u] *= rs1;
        }
    }

    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    constexpr unsigned kPvAStepBytes = 16u * sizeof(half);
    constexpr unsigned kPvMtileBytes = 16u * kProbStride * sizeof(half);
    const unsigned probs_lane_base =
        tt_smem_addr(probs) +
        (unsigned)(((lane & 15u) * (kProbStride / 2u) +
                    (lane >> 4u) * 4u) * sizeof(uint32_t));
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t pv_row_lane = lane & 15u;
    const uint32_t pv_chunk_base = pv_warp * kPvNTiles;
#pragma unroll
    for (uint32_t kt = 0; kt < TT_STAGE_ROWS / 16u; ++kt) {
        const unsigned probs_kt_base = probs_lane_base + (unsigned)(kt * kPvAStepBytes);
        const uint32_t pv_row = kt * 16u + pv_row_lane;
#pragma unroll
        for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
            uint32_t a[4];
            tt_ldmatrix_x4_addr(a, probs_kt_base + (unsigned)(mtile * kPvMtileBytes));
#pragma unroll
            for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
                uint32_t b[2];
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
                tt_ldmatrix_x2_trans_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(pv_row, pv_chunk_base + ntile));
                tt_mma_m16n8k16_f16_f32(o_acc + idx, a, b);
            }
        }
    }
}

__device__ __forceinline__ void tt_pv_mma_epilogue(
        const float (&o_acc)[2u * kTTTileTokens * kTTG],
        const float * __restrict__ final_scale,
        float * __restrict__ heads,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2) + (uint32_t)l;
                const uint32_t m = mtile * 16u + (uint32_t)tt_mma_c_i(lane, l);
                const uint32_t tok = m / kTTG;
                const uint32_t h = m - tok * kTTG;
                const uint32_t gt = tile_base + tok;
                const uint32_t gh = head_base + h;
                const uint32_t d =
                    pv_warp * 64u + ntile * 8u + (uint32_t)tt_mma_c_j(lane, l);
                if (gt < n_tokens && gh < n_head) {
                    heads[((uint64_t)gt * n_head + gh) * kTTHeadDim + d] =
                        o_acc[idx] * final_scale[m];
                }
            }
        }
    }
}

__global__ static void __launch_bounds__(512, 1) attention_tokentile_hmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min) {
    constexpr uint32_t kKvElems = tt_TokentileLayout<kTTStageRows>::ring_plane_elems;
    constexpr uint32_t kProbStride = tt_TokentileLayout<kTTStageRows>::prob_stride;
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t head_group = blockIdx.y;
    const uint32_t tile_base = tile_idx * kTTTileTokens;
    const uint32_t head_base = head_group * kTTG;
    if (tile_base >= n_tokens) {
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t raw_union_count = tile_count + kTTRawWindow - 1u;
    const uint32_t comp_union_count = union_counts[tile_idx];
    const uint32_t n_score = raw_union_count + comp_union_count;
    const uint64_t union_tile_off = (uint64_t)tile_idx * rec_stride;
    const int2 * __restrict__ union_records_tile = union_records + union_tile_off;
    const float score_scale = rsqrtf((float)kTTHeadDim);

    extern __shared__ unsigned char smem[];
    unsigned char *p = tt_align16(smem);
    half *kv_h = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * tt_TokentileLayout<kTTStageRows>::ring_plane_bytes);
    half *probs = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * kTTM * kProbStride * sizeof(half));
    float4 *score_scratch = reinterpret_cast<float4 *>(p);
    p = tt_align16(p + kTTM * kTTStageRows * sizeof(float4));
    float *max_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *sum_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *stage_rescale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *final_scale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    int2 *rec_ring = reinterpret_cast<int2 *>(p);
    p = tt_align16(p + kTTRecordRingPlanes * kTTStageRows * sizeof(int2));

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        max_s[m] = -INFINITY;
        sum_s[m] = 0.0f;
        stage_rescale[m] = 1.0f;
        final_scale[m] = 0.0f;
    }
    __syncthreads();

    union tt_TokentileRoleRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[2u * kTTTileTokens * kTTG];
    };
    tt_TokentileRoleRegs role_regs;
    tt_load_score_q_frag(
        role_regs.score_q_frag, q, n_tokens, n_head, tile_base, head_base);
    if (tt_warp_id() >= 8u) {
#pragma unroll
        for (uint32_t i = 0; i < 2u * kTTM; ++i) {
            role_regs.o_acc[i] = 0.0f;
        }
    }

    uint32_t cur = 0u;
    uint32_t free = 1u;
    if (n_score != 0u) {
        const uint32_t nr0 = n_score < kTTStageRows ? n_score : kTTStageRows;
        tt_issue_kv_stage_cp_async<kTTStageRows, false>(
            kv_h + cur * kKvElems,
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile,
            tile_base,
            raw_kv,
            comp_kv,
            tid);
        tt_issue_record_stage_cp_async<kTTStageRows>(
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile);
        if (kTTStageRows < n_score) {
            const uint32_t nr1 =
                n_score - kTTStageRows < kTTStageRows ? n_score - kTTStageRows : kTTStageRows;
            tt_issue_record_stage_cp_async<kTTStageRows>(
                rec_ring + kTTStageRows,
                kTTStageRows,
                nr1,
                raw_union_count,
                union_records_tile);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    uint32_t prob_cur = 0u;
    for (uint32_t row0 = 0u; row0 < n_score; row0 += kTTStageRows) {
        const uint32_t nr = n_score - row0 < kTTStageRows ? n_score - row0 : kTTStageRows;
        half *kv_cur = kv_h + cur * kKvElems;
        half *kv_free = kv_h + free * kKvElems;

        tt_hmma_score_stage<kTTStageRows>(
            score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
        if (row0 != 0u) {
            const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
            const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
            tt_pv_mma_stage<kTTStageRows>(
                role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
        }
        __syncthreads();

        const uint32_t next_row0 = row0 + kTTStageRows;
        const bool has_next = next_row0 < n_score;
        if (has_next) {
            const uint32_t next_nr =
                n_score - next_row0 < kTTStageRows ? n_score - next_row0 : kTTStageRows;
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_free,
                rec_ring + (((row0 / kTTStageRows) + 1u) & 3u) * kTTStageRows,
                next_row0,
                next_nr,
                raw_union_count,
                union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            const uint32_t prefetch_row0 = next_row0 + kTTStageRows;
            if (prefetch_row0 < n_score) {
                const uint32_t prefetch_nr =
                    n_score - prefetch_row0 < kTTStageRows
                        ? n_score - prefetch_row0
                        : kTTStageRows;
                tt_issue_record_stage_cp_async<kTTStageRows>(
                    rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                    prefetch_row0,
                    prefetch_nr,
                    raw_union_count,
                    union_records_tile);
            }
            tt_cp_async_commit();
        }

        half *probs_cur = probs + prob_cur * kTTM * kProbStride;
        tt_softmax_stage<kTTStageRows>(
            probs_cur,
            stage_rescale,
            max_s,
            sum_s,
            score_scratch,
            rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
            row0,
            nr,
            raw_union_count,
            tile_count,
            tile_base,
            raw_row_min);
        tt_cp_async_wait_group<0>();
        __syncthreads();
        cur ^= 1u;
        free ^= 1u;
        prob_cur ^= 1u;
    }

    if (n_score != 0u) {
        const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
        const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
        tt_pv_mma_stage<kTTStageRows>(
            role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
    }
    __syncthreads();

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        const uint32_t tok = m / kTTG;
        const uint32_t h = m - tok * kTTG;
        const uint32_t gt = tile_base + tok;
        const uint32_t gh = head_base + h;
        if (gt < n_tokens && gh < n_head) {
            const float sink = sinks[gh];
            const float old_m = max_s[m];
            const float new_m = fmaxf(old_m, sink);
            const float old_scale = old_m == -INFINITY ? 0.0f : expf(old_m - new_m);
            const float sink_scale = expf(sink - new_m);
            const float den = sum_s[m] * old_scale + sink_scale;
            final_scale[m] = den == 0.0f ? 0.0f : old_scale / den;
        } else {
            final_scale[m] = 0.0f;
        }
    }
    __syncthreads();

    tt_pv_mma_epilogue(
        role_regs.o_acc, final_scale, heads, n_tokens, n_head, tile_base, head_base);
}

constexpr size_t tt_align16_const(size_t x) {
    return (x + 15u) & ~size_t(15u);
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
struct tt_TokentileSmemBudget {
    static constexpr uint32_t M = kTTTileTokens * TT_G;
    static constexpr uint32_t prob_stride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static constexpr size_t q_bytes = 0;
    static constexpr size_t ring_bytes = 2ull * tt_TokentileLayout<TT_STAGE_ROWS>::ring_plane_bytes;
    static constexpr size_t p_bytes = 2ull * M * prob_stride * sizeof(half);
    static constexpr size_t partial_records = (size_t)M * TT_STAGE_ROWS;
    static constexpr size_t partial_bytes = partial_records * sizeof(float4);
    static constexpr size_t stats_bytes = 4ull * M * sizeof(float);
    static constexpr size_t record_bytes =
        (size_t)kTTRecordRingPlanes * TT_STAGE_ROWS * sizeof(int2);
    static constexpr size_t total =
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(q_bytes) + ring_bytes) + p_bytes) +
        partial_bytes) + stats_bytes) + record_bytes);
};

static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::p_bytes == 5120ull,
              "M32/R32 P double-buffer must use R+8 stride");
static_assert(sizeof(float4) == 16u, "score partial records must stay 16 bytes");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::partial_bytes == 16ull * 1024ull,
              "M32/R32 score partial records must be M*R float4");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::record_bytes == 1024ull,
              "M32/R32 record ring must be four R-row int2 planes");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::total == 88576ull,
              "M32/R32 total dynamic shared memory changed unexpectedly");
static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::total <= kTTSmemHardCap,
              "token-tile dynamic shared memory must stay under the 90 KiB pass gate");

/* min 4 blocks/SM caps ptxas at 64 regs/thread (256 thr): native sm_121 ptxas
 * picks 72 regs = 3 blocks/SM and long-ctx decode attention pays +18%
 * (14.7 -> 17.4 ms at 12k KV).  Same 64-reg pin as the prefill twin above. */
__global__ static void __launch_bounds__(256, 4) attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        /* FB1 (per-seq admission chunks): optional per-row positions[]/seq_id[]
         * + per-seq compressed bank stride, mirroring the generic
         * attention_decode_mixed_kernel semantics exactly (per-row raw window
         * from positions[t], raw ring base seq_id[t]*raw_cap, comp bank base
         * seq_id[t]*comp_cap, floor(qpos/ratio) per-row visible clamp).  All
         * NULL/0 keeps the single-seq path bit-exact. */
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        uint32_t comp_cap,
        const struct ds4_decode_scalars * __restrict__ s_override,
        /* Step 4c A1: per-layer n_comp override. */
        const struct ds4_layer_scalars  * __restrict__ ls_override,
        /* P2 Inc2b: optional packed FP8 mirror (absolute rows: this kernel
         * reads comp at comp_seq_base + row, matching the generic kernel). */
        const unsigned char * __restrict__ comp_fp8,
        const float         * __restrict__ comp_scale,
        const float* decode_table,
        Ds4AttentionDiagnostics* diagnostics) {
    if (s_override) {
        n_raw     = s_override->n_raw;
        raw_start = s_override->raw_start;
    }
    if (ls_override) {
        n_comp    = ls_override->n_comp;
    }
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;
    const bool use_fp8 = (comp_fp8 != NULL) && (comp_scale != NULL);

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t row_n_raw     = n_raw;
    uint32_t row_raw_start = raw_start;
    if (positions) {
        row_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (row_n_raw > raw_cap) row_n_raw = raw_cap;
        row_raw_start = (qpos + 1u - row_n_raw) % raw_cap;
    }
    const uint32_t first_raw_pos = positions ? (qpos + 1u - row_n_raw)
                                             : (pos0 + n_tokens - n_raw);
    const uint32_t seq_base      = seq_id ? (uint32_t)seq_id[t] * raw_cap  : 0u;
    const uint32_t comp_seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (positions && comp_count > qpos / ratio) comp_count = qpos / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (use_fp8 && comp_count != 0u && threadIdx.x == 0) {
        atomicAdd(&g_fp8_kv_read_path_blocks, 1ull);
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (row_n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + row_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = seq_base + ((row_raw_start + raw_first_idx + r) % raw_cap);
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = ((const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim))[c4];
            } else if (use_fp8) {
                kv_shared[off] = fp8_kv_read4(comp_fp8, comp_scale, comp_seq_base + (sr - raw_count), c4);
            } else {
                kv_shared[off] = ((const float4 *)(comp_kv + (uint64_t)(comp_seq_base + (sr - raw_count)) * head_dim))[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

// clang-format on
#undef fp8_kv_read
#undef fp8_kv_read4
#undef g_fp8_kv_read_path_blocks
#undef g_fp8_kv_indexed_read_path_blocks
#undef dsv4_e4m3fn_decode_table
#undef DS4_CUDA_ATTENTION_SCORE_CAP
#undef DS4_CUDA_ATTENTION_RAW_SCORE_CAP
#undef DS4_OPP_C_FP8_NOPE_DEV
#undef DS4_OPP_C_FP8_BLOCKS_DEV
#undef DS4_OPP_C_FP8_ROW_BYTES_DEV
#undef DS4_ATTN_HG_HEADS
#undef DS4_ATTN_HG_ROWS

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_CORE_CUH_
