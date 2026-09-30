// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me>
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// Original Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976.
// Numerical functions copied from ds4_cuda.cu; extraction provenance is
// in dsv4_ds4_cache_core.json. Only FP8 read/expand signatures receive an
// explicit caller-owned decode table instead of the original device global.
// No original runtime, allocation, stream or cache-counter code.
//
// Included only by the CUDA adapter, inside its anonymous namespace.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_CACHE_CORE_CUH_
#define JITLLM_KERNELS_GGML_DSV4_DS4_CACHE_CORE_CUH_

#define DS4_CUDA_UNUSED __attribute__((unused))
#define DS4_OPP_C_FP8_NOPE_DEV 448u
#define DS4_OPP_C_FP8_BLOCKS_DEV 7u
#define DS4_OPP_C_FP8_ROW_BYTES_DEV 704ull

// clang-format off
__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    /* signbit, not x<0: -0.0 must keep its sign through a re-encode round
     * trip (x<0 is false for -0.0).  Identical for all nonzero x. */
    float sign = signbit(x) ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static unsigned char dsv4_e4m3fn_encode_dev(float x) {
    const unsigned int sign_bit = signbit(x) ? 0x80u : 0x00u;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        const float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (unsigned char)(sign_bit | (unsigned int)best);
}

__device__ static float dsv4_pow2_ceil_scale(float r) {
    int e; float m = frexpf(r, &e);          /* r = m * 2^e, m in [0.5, 1) */
    return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

__device__ static float fp8_kv_read(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t                          c,
        uint32_t                          d,
        const float * __restrict__ dsv4_e4m3fn_decode_table) {
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

__global__ static void fp8_kv_dequant_rows_kernel(
        float * __restrict__ dst,
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t n_rows,
        uint32_t head_dim,
        const float * __restrict__ decode_table) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    float *out = dst + (uint64_t)r * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = fp8_kv_read(codes_base, scale_base, r, d, decode_table);
    }
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static int dsv4_e2m1fn_level_dev(float ax) {
    ax = fminf(ax, 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return best;
}

__device__ static DS4_CUDA_UNUSED float indexer_fp4_read(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t row, uint32_t d) {
    const unsigned char byte = codes_base[(uint64_t)row * 64u + (d >> 1u)];
    const uint32_t nib = ((uint32_t)byte >> (4u * (d & 1u))) & 0xFu;
    const float sign = (nib & 8u) ? -1.0f : 1.0f;
    const float scale = scale_base[(uint64_t)row * 4u + (d >> 5u)];
    return sign * dsv4_e2m1fn_value_dev((int)(nib & 7u)) * scale;
}

__global__ static void indexer_fp4_dequant_rows_kernel(
        float * __restrict__ dst,
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t n_rows, uint32_t head_dim) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    float *out = dst + (uint64_t)r * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = indexer_fp4_read(codes_base, scale_base, r, d);
    }
}

__global__ static void fp8_kv_quantize_kernel(
        float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        unsigned char * __restrict__ codes_base,
        float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    /* Opp C Phase 1A: see the matching comment in fp8_kv_quantize_row_kernel
     * for the packed mirror row layout (448 codes + FP32 rotary tail; 7
     * per-block scales).  When both bases are NULL the kernel collapses to
     * the original in-place FP32 quant. */
    const uint64_t row_stride = (uint64_t)n_nope + (uint64_t)n_rot * sizeof(float);
    const uint64_t scale_stride = (uint64_t)(n_nope / 64u);
    float *xr = x + (uint64_t)row * head_dim;
    unsigned char *codes_row = codes_base
        ? codes_base + (uint64_t)row * row_stride
        : (unsigned char *)0;
    float *scale_row = scale_base
        ? scale_base + (uint64_t)row * scale_stride
        : (float *)0;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = dsv4_pow2_ceil_scale(fmaxf(scratch[0], 1.0e-4f) / 448.0f);
        if (off + tid < n_nope) {
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_dev(clamp) * scale;
            if (codes_row) codes_row[off + tid] = dsv4_e4m3fn_encode_dev(clamp);
        }
        if (scale_row && tid == 0) scale_row[off / 64u] = scale;
        __syncthreads();
    }
    /* Copy FP32 rotary tail into the mirror row (one float per thread). */
    if (codes_row && tid < n_rot) {
        float *tail = (float *)(codes_row + (uint64_t)n_nope);
        tail[tid] = xr[(uint64_t)n_nope + tid];
    }
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim,
        /* P2 Inc3: optional packed FP4 mirror (NULL-passthrough, exactly like
         * fp8_kv_quantize_kernel).  When non-NULL, lane tid's sign+level nibble
         * packs into codes_base[row*64 + tid/2] (low nibble = even tid) and the
         * per-32-lane block scale into scale_base[row*4 + tid/32].  The packed
         * row reconstructs xr[tid] bit-for-bit via indexer_fp4_read. */
        unsigned char * __restrict__ codes_base,
        float         * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = dsv4_pow2_ceil_scale(amax / 6.0f);
    /* P2 Inc3: F32 write is bit-identical to the prior
     * dsv4_e2m1fn_dequant_dev(clamp)*scale (now factored through
     * dsv4_e2m1fn_level_dev), plus the optional packed-mirror emit.  The
     * even lane packs its own low nibble with the odd neighbour's nibble
     * (warp shuffle); lane 0 of each 32-lane block writes the block scale. */
    float clamp = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int level = dsv4_e2m1fn_level_dev(fabsf(clamp));
    xr[tid] = ((signbit(clamp) ? -1.0f : 1.0f) * dsv4_e2m1fn_value_dev(level)) * scale;
    if (codes_base) {
        uint32_t nib = (signbit(clamp) ? 8u : 0u) | (uint32_t)level;
        uint32_t hi  = __shfl_down_sync(0xffffffffu, nib, 1u);
        if ((tid & 1u) == 0u)  codes_base[(uint64_t)row * 64u + (tid >> 1u)] = (unsigned char)(nib | (hi << 4u));
    }
    /* v0.3 V5D: scale-only emit (scale_base set, codes_base NULL) hands the
     * indexer-Q QAT block scales to the WMMA scorer, which needs the exact
     * commit scale to stage unscaled levels (v / scale is exact only against
     * the scale this kernel divided by).  Comp-cache callers pass both mirror
     * pointers or neither, so their behavior is unchanged. */
    if (scale_base && (tid & 31u) == 0u)
        scale_base[(uint64_t)row * 4u + (tid >> 5u)] = scale;
}

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

__global__ static void store_raw_kv_batch_kernel(
        float *raw, const float *kv,
        uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (n_tokens
         * entries).  positions -> write slot from the row's absolute position
         * positions[t]; seq_id -> per-seq ring bank base seq_id[t]*raw_cap.
         * Both NULL keeps the single-sequence scalar path (bit-exact). */
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const struct ds4_decode_scalars * __restrict__ s_override) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    /* PC4 (K0): decode1 single-row path reads raw_row from the token-stable
     * substrate at execution time -- capture-safe.  Under layer-graph
     * capture, `pos0` was baked into the kernel-node arg list at queue
     * time and would replay the wrong slot.  Batch path (n_tokens > 1)
     * is not capture-targeted (decode2-exact per plan doc sec 8.3) and
     * keeps the inline arg. */
    uint32_t row;
    if (s_override != NULL && n_tokens == 1u) {
        row = s_override->raw_row;
    } else {
        uint32_t pos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
        row = seq_base + (pos % raw_cap);
    }
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}


#undef DS4_OPP_C_FP8_ROW_BYTES_DEV
#undef DS4_OPP_C_FP8_BLOCKS_DEV
#undef DS4_OPP_C_FP8_NOPE_DEV
#undef DS4_CUDA_UNUSED

// clang-format on

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_CACHE_CORE_CUH_
