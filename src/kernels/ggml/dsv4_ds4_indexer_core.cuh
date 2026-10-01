// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me>
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 Marco Palaferri
// SPDX-License-Identifier: MIT
//
// Original Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976.
// Numerical function bodies retained from ds4_cuda.cu. Only signatures
// expose caller-owned diagnostics; lexical counter aliases below remove
// the original device globals. rr_mma_mxf4 guards unsupported device code.
// No original allocator, graph, stream or registry is included.
// Frozen extraction proof and numerical flags are required before use.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_INDEXER_CORE_CUH_
#define JITLLM_KERNELS_GGML_DSV4_DS4_INDEXER_CORE_CUH_

#define DS4_CUDA_UNUSED __attribute__((unused))
#define DS4_CUDA_TOPK_MERGE_GROUP 8u
#define g_fp4_index_read_path_blocks (diagnostics->packed_reads)
#define g_topk_bound_viol_count (diagnostics->bound_count)
#define g_topk_bound_viol_n (diagnostics->bound_live)
#define g_topk_bound_viol_stride (diagnostics->bound_stride)
// Original harness-only instrumentation is not enabled by this derivative.
#if defined(DS4_S512_TRACE) || defined(DS4_S512_APPEND_GUARD)
#error Original ds4 harness-only globals are not native operands
#endif

// clang-format off
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

__device__ static float dsv4_pow2_ceil_scale(float r) {
    int e; float m = frexpf(r, &e);          /* r = m * 2^e, m in [0.5, 1) */
    return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
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

/* v0.3 V5D: signed e2m1 LEVEL of one packed FP4 lane -- indexer_fp4_read
 * WITHOUT the block-scale multiply.  The WMMA scorer stages these unscaled
 * levels as fp16 MMA operands (halves in [-6, 6]: always fp16-exact, unlike
 * the scaled values, whose block scales reach 2^-26 and underflow half --
 * refuted-raw-conversion proof in cuda/mmq/test/proto_indexer_score.cu) and
 * folds the F32 block scale back in after the 32-wide k-chunk accumulates. */
__device__ static float indexer_fp4_level_read(
        const unsigned char * __restrict__ codes_base,
        uint32_t row, uint32_t d) {
    const unsigned char byte = codes_base[(uint64_t)row * 64u + (d >> 1u)];
    const uint32_t nib = ((uint32_t)byte >> (4u * (d & 1u))) & 0xFu;
    const float sign = (nib & 8u) ? -1.0f : 1.0f;
    return sign * dsv4_e2m1fn_value_dev((int)(nib & 7u));
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        /* P2 Inc3b: optional packed FP4 indexer mirror (row stride 64 code
         * bytes + 4 block scales).  When non-NULL each lane reads via
         * indexer_fp4_read (bit-identical to index_comp).  NULL = F32. */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && threadIdx.x == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
            dot += qh[d] * (use_fp4 ? indexer_fp4_read(index_fp4, index_scale, c, d) : kh[d]);
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal,
        /* PC5 micro-pilot: optional per-layer substrate override.  When
         * non-NULL the kernel reads the runtime indexer count from
         * ls_override->n_index_comp instead of the inline n_comp arg.
         * Lets the shim launch with a session-stable max grid (e.g.
         * comp_cap) for capture-safety while preserving correctness
         * via this bounds check.  NULL = legacy path (n_comp == grid).*/
        const struct ds4_layer_scalars * __restrict__ ls_override,
        /* P2 Inc3: optional packed FP4 indexer mirror.  When non-NULL the krow
         * lane reads via indexer_fp4_read (bit-identical to index_comp) and the
         * block arms the FP4 read tripwire.  NULL = F32 (default / fp4 off). */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    if (c >= n_actual || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_actual;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);
    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) {
        krow[tid] = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, c, tid)
                            : index_comp[(uint64_t)c * 128u + tid];
    }
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

/* Phase 2 Step 4b: per-seq multi-sequence indexer scores.  The batched decode
 * scores path normally uses the WMMA tiled kernels, which share a tile's
 * index_comp rows across 16 query tokens -- incompatible with each token
 * reading its OWN sequence's index_comp bank.  So multi-seq (seq_id != NULL)
 * is forced onto this per-(comp-row, token) kernel, mirroring the raw/mixed
 * decode's `force_main_kernel`.  The per-token reduction is bit-identical to
 * indexer_score_one_direct_kernel (ls_override == NULL), so for equal-length
 * sequences each row reproduces its own n=1 standalone-decode oracle exactly.
 * Requires the indexer dims (head_dim==128, n_head==64). */
__global__ static void indexer_scores_multiseq_kernel(
        float *scores,            /* [n_tokens, n_comp]                       */
        const float *q,           /* [n_tokens, 64, 128]                      */
        const float *weights,     /* [n_tokens, 64]                           */
        const float *index_comp,  /* [N_banks * comp_cap, 128]                */
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t ratio,
        uint32_t comp_cap,
        float scale,
        int causal,
        /* Per-row query position (causal clamp) + per-seq bank base.  positions
         * NULL falls back to pos0 (equal-length); seq_id NULL falls back to
         * bank 0 (bit-exact single bank). */
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        /* P2 Inc3b: optional packed FP4 indexer mirror (whole slab; the kernel
         * applies the same seq_base+c row index as index_comp).  NULL = F32. */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
    const uint32_t c   = blockIdx.x;
    const uint32_t t   = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || t >= n_tokens || tid >= 128u) return;
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
    if (causal) {
        uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
        /* Step 4c: per-row varlen count -- floor(qpos/ratio) is the true number
         * of complete compressed rows; mark rows beyond it (unprimed bank slots
         * for shorter seqs) as -INF so topk never selects them. */
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
        if (c >= visible) {
            if (tid == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *qrow = q + (uint64_t)t * 64u * 128u;
    const float *wrow = weights + (uint64_t)t * 64u;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u)
        krow[tid] = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, seq_base + c, tid)
                            : index_comp[(uint64_t)(seq_base + c) * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(qrow + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * wrow[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[(uint64_t)t * n_comp + c] = total;
}

/* v0.3 V5D: WMMA rewrite of the per-(comp-row, token) multiseq scorer above
 * (proto: cuda/mmq/test/proto_indexer_score.cu, V5D leg -- 744 -> 188
 * us/launch at the 240K-deep 59082-row shape on GB10, 3.97x, bitwise
 * idempotent across the V5B..V5D family, top-k flips 0/2048 on every proto
 * leg).  The scalar kernel is latency-bound, not bandwidth-bound (33
 * CTA-wide barriers + a 32 KB q re-read per (row, token) block; fp4's 6.4x
 * byte reduction moved nothing), so this restates the scan as 32-row tiles:
 * the CTA stages one 32-row x 32-dim scale block of K as UNSCALED e2m1
 * levels (halves in [-6, 6], always fp16-exact -- the scaled values
 * underflow half at block scales ~2^-26, which is what refuted a raw fp16
 * conversion), each warp MMAs the tile against its own 16-head q tile, and
 * the F32 block scales fold back in per 32-wide k-chunk.  Chunk width ==
 * scale-block width, so the fold is exact.
 *
 * Scale sourcing (the same-binary cross-config identity argument):
 *  - q levels = q / q_block_scale, the EXACT commit scale the indexer-Q QAT
 *    divided by (scale-only emit from indexer_hadamard_fp4_kernel; both
 *    storage configs run that QAT, so staged q is config-invariant).
 *  - fp4-primary K: nibble levels via indexer_fp4_level_read + the resident
 *    index_scale mirror.
 *  - F32-primary K: block scale re-derived in-kernel with the emit QAT's
 *    identical derivation (warp amax -> subnormal floor ->
 *    dsv4_pow2_ceil_scale(amax/6); exact under fast-math).  On committed
 *    rows the re-derived scale can differ from the commit scale only by a
 *    power of two (re-encode idempotence, proto_kv_reencode_idem.cu), which
 *    rescales the staged levels by the inverse power: level products stay
 *    exact and the fold cancels the difference bitwise.
 *
 * Capture/VMM safety (engine deltas vs the proto; live-row outputs are
 * bitwise unchanged because MMA output rows are independent):
 *  - the launch grid is a pure function of the caller's n_comp (the capture
 *    band on capture steps); the causal clamp reads positions[] live.
 *  - staging reads are guarded at `comp < visible`, never `< n_comp`: the
 *    comp/index slabs are VMM demand-mapped and rows in [visible, band) may
 *    be unmapped.  Dead rows stage 0 and their outputs are overwritten with
 *    -INF in the epilogue (identical scores to the scalar kernel).
 *  - tiles entirely past `visible` write -INF and exit before staging (the
 *    scalar kernel's per-block early-out priced dead band rows at ~zero;
 *    without this, capture-band tiles would pay full MMA cost).
 *
 * Occupancy: static smem 10880 B; __launch_bounds__(128, 8) is the .15
 * bounds-sweep interior optimum ((128,6) 200.1 / (128,8) 187.8 / (128,12)
 * 200.3 us regs-40 spill cliff; this shape compiles to 64 regs).  Needs the
 * full smem carveout (PreferredSharedMemoryCarveout = 100, set once at the
 * launch site) to reach 8 blocks/SM on GB10's 100 KB SMs.
 * The 32-half smem stride is 4-way bank-conflicted but deliberately KEPT:
 * the smallest legal pad is 40 (wmma ldm must be a multiple of 8 halves --
 * 36 faults "misaligned address") and its +1536 B/block costs 8->7
 * blocks/SM, a measured wash (FP4 1.02x / F32 0.91x); see the V5E inc-2
 * note and proto_indexer_score_mt.cu. */
__global__ static void __launch_bounds__(128, 8) indexer_scores_multiseq_v5d_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale,
        Ds4IndexerDiagnostics* diagnostics) {
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 128u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u)
        atomicAdd(&g_fp4_index_read_path_blocks, 1ull);

    uint32_t visible = n_comp;
    if (causal) {
        const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
        visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
        if (visible > n_comp) visible = n_comp;
    }
    if (tile_c >= visible) {
        if (warp == 0u) {
            const uint32_t comp = tile_c + lane;
            if (comp < n_comp) scores[(uint64_t)t * n_comp + comp] = -INFINITY;
        }
        return;
    }

    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const uint32_t head0 = warp * 16u;
    const float *wrow = weights + (uint64_t)t * 64u;

    __shared__ __half a_sh[32 * 32];
    __shared__ __half b_sh[4 * 16 * 32];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float k_scale_block[32];
    __shared__ float warp_row_partial[4 * 32];

    __half *warp_b_sh = b_sh + warp * 16u * 32u;
    float *warp_c_sh = c_sh + warp * 16u * 16u;

    float head_total[16];
    #pragma unroll
    for (uint32_t slot = 0; slot < 16u; slot++) head_total[slot] = 0.0f;

    #pragma unroll
    for (uint32_t block = 0; block < 4u; block++) {
        for (uint32_t i = lane; i < 16u * 32u; i += 32u) {
            const uint32_t h = i >> 5u;
            const uint32_t block_d = i & 31u;
            const uint32_t d = block * 32u + block_d;
            const uint32_t head = head0 + h;
            float v = q[((uint64_t)t * 64u + head) * 128u + d];
            const float qs = q_block_scale[
                ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
            warp_b_sh[i] = __float2half(v / qs);
        }

        #pragma unroll
        for (uint32_t r = warp; r < 32u; r += 4u) {
            const uint32_t comp = tile_c + r;
            const uint32_t d = block * 32u + lane;
            float v = 0.0f;
            if (comp < visible) {
                v = use_fp4
                    ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                    : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
            }
            if (use_fp4) {
                if (lane == 0u) {
                    k_scale_block[r] = comp < visible
                        ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            } else {
                /* Emit-QAT scale derivation, verbatim (shfl_down max: only
                 * lane 0's value is the true amax, and only lane 0 writes --
                 * bitwise the proto's precomputed
                 * indexer_block_scale_rows_kernel). */
                float amax = fabsf(v);
                for (int offset = 16; offset > 0; offset >>= 1)
                    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
                if (lane == 0u) {
                    k_scale_block[r] = comp < visible
                        ? dsv4_pow2_ceil_scale(
                              fmaxf(amax, 7.052966104933725e-38f) / 6.0f)
                        : 1.0f;
                }
            }
            __syncwarp();
            const float level = use_fp4 ? v : v / k_scale_block[r];
            a_sh[r * 32u + lane] = __float2half(level);
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k += 16u) {
                wmma::load_matrix_sync(
                    a_frag, a_sh + row_tile * 16u * 32u + k, 32);
                wmma::load_matrix_sync(b_frag, warp_b_sh + k, 32);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(warp_c_sh, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (uint32_t slot = 0; slot < 8u; slot++) {
                const uint32_t i = lane + slot * 32u;
                const uint32_t r = row_tile * 16u + (i >> 4u);
                const uint32_t h = i & 15u;
                const uint32_t comp = tile_c + r;
                if (comp < n_comp) {
                    float chunk = warp_c_sh[i];
                    const float qs = q_block_scale[
                        ((uint64_t)t * 64u + head0 + h) * 4u + block];
                    chunk *= k_scale_block[r] * qs;
                    head_total[row_tile * 8u + slot] += chunk;
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++)
            warp_c_sh[lane + slot * 32u] =
                head_total[row_tile * 8u + slot];
        __syncthreads();

        if (lane < 16u) {
            float warp_total = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < 16u; h++) {
                const uint32_t head = head0 + h;
                warp_total += fmaxf(warp_c_sh[lane * 16u + h], 0.0f)
                            * wrow[head] * scale;
            }
            warp_row_partial[
                warp * 32u + row_tile * 16u + lane] = warp_total;
        }
        __syncthreads();
    }

    if (warp == 0u && lane < 32u) {
        const uint32_t comp = tile_c + lane;
        if (comp < n_comp) {
            float out = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < 4u; w++)
                out += warp_row_partial[w * 32u + lane];
            if (comp >= visible) out = -INFINITY;
            scores[(uint64_t)t * n_comp + comp] = out;
        }
    }
}

/* v0.4 V5E: verify-width (multi-token) variant of V5D (proto:
 * cuda/mmq/test/proto_indexer_score_mt.cu -- w5 897 -> 566 us FP4 at the
 * 59082-row shape, 1933 -> 1213 at 128745, bitwise-identical scores and
 * topk on every leg).  V5D launches one CTA per (32-row tile, TOKEN), so
 * at DSpark verify width 5 five CTAs re-stage the same K tile, re-derive
 * the same block scales, and pay the same barrier chain.  V5E drops the
 * token grid dimension and loops tokens inside the CTA: the full 32x128
 * K tile + scales stage ONCE per seq-run (restaged only when seq_id
 * changes between consecutive tokens -- mixed multiseq batches degrade
 * gracefully to per-token staging), and the per-token MMA chain is
 * CTA-barrier-free (a_sh_all stable, b_sh/c_sh warp-private).
 *
 * Correctness structure inherited from V5D verbatim: per-token staged
 * values, MMA fragment ops, fold order, and epilogue are IDENTICAL, so
 * live-row scores are bitwise V5D's; the staging guard is the seq-run's
 * MAX visible (VMM-safe: never past the seq's mapped band) and rows in
 * [visible_t, vis_max) carry real K but are -INF-overwritten per token
 * (MMA output rows are independent).  All intra-loop barriers are
 * CTA-uniform (visible/seq are per-token constants, staged_seq evolves
 * identically on every thread).
 *
 * Occupancy: static smem 20,480 B (40-half stride) + 96 regs cap at
 * 4 blocks/SM (was 17.4 KB / 5 at the conflicted 32-half stride; ncu
 * then: 41.7% achieved, SM 52% / Mem 63%, latency-bound, no saturated
 * pipe) -- the amortization wins 1.58-1.85x anyway; a smem/reg-diet V5F
 * or HMMA-class restructure is BANKED (trigger: scorer back in the
 * step-ledger top-3 post-V5E).  n_tokens 2..8 only; w1 keeps V5D's
 * 8-block occupancy, wide batches keep the V5D token grid.
 *
 * v0.5 scorer diet inc-2: the ldmatrix-staged tiles (a_sh_all/b_sh) use a
 * 40-half stride.  32 halves = 64 B put fragment rows on two bank quads
 * (4-way conflicts); 40 = the smallest LEGAL pad -- wmma load_matrix_sync
 * requires ldm % 8 == 0 for __half (every ldmatrix row 16-B aligned; the
 * first attempt used 36 and FAULTED "misaligned address" at every decode
 * step, proto_indexer_score_mt.cu --repro36-* reproduces) -- and 80 B
 * starts the 8 phase rows at banks 20*i mod 32, a perfect partition of
 * all 32 banks: zero conflicts.  Proto: bitwise + zero topk flips, FP4
 * 1.11x / F32 1.05x at 59082 rows net of the 5->4 occupancy loss.
 * V5D deliberately KEEPS the 32-half stride: its pad measured a wash
 * (FP4 1.02x / F32 0.91x) because +1536 B/block drops it 8->7 blocks/SM,
 * cancelling the conflict win. */
__global__ static void __launch_bounds__(128, 5) indexer_scores_multiseq_v5e_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale,
        Ds4IndexerDiagnostics* diagnostics) {
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (tid >= 128u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u)
        atomicAdd(&g_fp4_index_read_path_blocks, 1ull);
    const uint32_t head0 = warp * 16u;

    __shared__ __half a_sh_all[4 * 32 * 40];   /* [block][row][dim-in-block] */
    __shared__ float k_scale_all[32 * 4];      /* [row][block] */
    __shared__ __half b_sh[4 * 16 * 40];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float warp_row_partial[4 * 32];

    __half *warp_b_sh = b_sh + warp * 16u * 40u;
    float *warp_c_sh = c_sh + warp * 16u * 16u;

    uint32_t staged_seq = 0xFFFFFFFFu;

    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t visible = n_comp;
        if (causal) {
            const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
            visible = ratio ? (qpos + 1u) / ratio : n_comp;
            if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
            if (visible > n_comp) visible = n_comp;
        }
        if (tile_c >= visible) {
            if (warp == 0u) {
                const uint32_t comp = tile_c + lane;
                if (comp < n_comp) scores[(uint64_t)t * n_comp + comp] = -INFINITY;
            }
            continue;
        }

        const uint32_t sid = seq_id ? (uint32_t)seq_id[t] : 0u;
        const uint32_t seq_base = sid * comp_cap;

        if (sid != staged_seq) {
            uint32_t vis_max = visible;
            for (uint32_t u = t + 1u; u < n_tokens; u++) {
                if ((seq_id ? (uint32_t)seq_id[u] : 0u) != sid) break;
                uint32_t vu = n_comp;
                if (causal) {
                    const uint32_t qp = positions ? (uint32_t)positions[u] : pos0;
                    vu = ratio ? (qp + 1u) / ratio : n_comp;
                    if (positions && ratio && vu > qp / ratio) vu = qp / ratio;
                    if (vu > n_comp) vu = n_comp;
                }
                if (vu > vis_max) vis_max = vu;
            }

            __syncthreads();
            #pragma unroll
            for (uint32_t r0 = 0; r0 < 32u; r0 += 4u) {
                const uint32_t r = r0 + warp;
                const uint32_t comp = tile_c + r;
                #pragma unroll
                for (uint32_t block = 0; block < 4u; block++) {
                    const uint32_t d = block * 32u + lane;
                    float v = 0.0f;
                    if (comp < vis_max) {
                        v = use_fp4
                            ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                            : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
                    }
                    if (use_fp4) {
                        if (lane == 0u) {
                            k_scale_all[r * 4u + block] = comp < vis_max
                                ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                                : 1.0f;
                        }
                    } else {
                        float amax = fabsf(v);
                        for (int offset = 16; offset > 0; offset >>= 1)
                            amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
                        if (lane == 0u) {
                            k_scale_all[r * 4u + block] = comp < vis_max
                                ? dsv4_pow2_ceil_scale(
                                      fmaxf(amax, 7.052966104933725e-38f) / 6.0f)
                                : 1.0f;
                        }
                    }
                    __syncwarp();
                    const float level = use_fp4 ? v : v / k_scale_all[r * 4u + block];
                    a_sh_all[block * (32u * 40u) + r * 40u + lane] = __float2half(level);
                }
            }
            __syncthreads();
            staged_seq = sid;
        }

        const float *wrow = weights + (uint64_t)t * 64u;
        float head_total[16];
        #pragma unroll
        for (uint32_t slot = 0; slot < 16u; slot++) head_total[slot] = 0.0f;

        #pragma unroll
        for (uint32_t block = 0; block < 4u; block++) {
            for (uint32_t i = lane; i < 16u * 32u; i += 32u) {
                const uint32_t h = i >> 5u;
                const uint32_t block_d = i & 31u;
                const uint32_t d = block * 32u + block_d;
                const uint32_t head = head0 + h;
                float v = q[((uint64_t)t * 64u + head) * 128u + d];
                const float qs = q_block_scale[
                    ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
                warp_b_sh[h * 40u + block_d] = __float2half(v / qs);
            }
            __syncwarp();

            #pragma unroll
            for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (uint32_t k = 0; k < 32u; k += 16u) {
                    wmma::load_matrix_sync(
                        a_frag,
                        a_sh_all + block * (32u * 40u) + row_tile * 16u * 40u + k,
                        40);
                    wmma::load_matrix_sync(b_frag, warp_b_sh + k, 40);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(warp_c_sh, c_frag, 16, wmma::mem_row_major);
                __syncwarp();

                #pragma unroll
                for (uint32_t slot = 0; slot < 8u; slot++) {
                    const uint32_t i = lane + slot * 32u;
                    const uint32_t r = row_tile * 16u + (i >> 4u);
                    const uint32_t h = i & 15u;
                    const uint32_t comp = tile_c + r;
                    if (comp < n_comp) {
                        float chunk = warp_c_sh[i];
                        const float qs = q_block_scale[
                            ((uint64_t)t * 64u + head0 + h) * 4u + block];
                        chunk *= k_scale_all[r * 4u + block] * qs;
                        head_total[row_tile * 8u + slot] += chunk;
                    }
                }
                __syncwarp();
            }
        }

        __syncthreads();
        #pragma unroll
        for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
            #pragma unroll
            for (uint32_t slot = 0; slot < 8u; slot++)
                warp_c_sh[lane + slot * 32u] =
                    head_total[row_tile * 8u + slot];
            __syncthreads();

            if (lane < 16u) {
                float warp_total = 0.0f;
                #pragma unroll
                for (uint32_t h = 0; h < 16u; h++) {
                    const uint32_t head = head0 + h;
                    warp_total += fmaxf(warp_c_sh[lane * 16u + h], 0.0f)
                                * wrow[head] * scale;
                }
                warp_row_partial[
                    warp * 32u + row_tile * 16u + lane] = warp_total;
            }
            __syncthreads();
        }

        if (warp == 0u && lane < 32u) {
            const uint32_t comp = tile_c + lane;
            if (comp < n_comp) {
                float out = 0.0f;
                #pragma unroll
                for (uint32_t w = 0; w < 4u; w++)
                    out += warp_row_partial[w * 32u + lane];
                if (comp >= visible) out = -INFINITY;
                scores[(uint64_t)t * n_comp + comp] = out;
            }
        }
    }
}

__global__ static void indexer_scores_wmma_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        /* P2 Inc3b: optional packed FP4 indexer mirror (NULL = F32). */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    if (tid >= 32u || head_dim != 128u) return;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
                const uint32_t r = i >> 4u;
                const uint32_t c = i & 15u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[16 * 128];
    __shared__ float c_sh[16 * 16];
    __shared__ float acc_sh[16 * 16];

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, comp, d)
                                       : index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
            const uint32_t r = i >> 4u;
            const uint32_t token = tile_t + r;
            if (token < n_tokens) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
        const uint32_t r = i >> 4u;
        const uint32_t c = i & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        /* P2 Inc3b: optional packed FP4 indexer mirror (NULL = F32). */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 64u || head_dim != 128u) return;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 32u; i += 64u) {
                const uint32_t r = i >> 5u;
                const uint32_t c = i & 31u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[32 * 128];
    __shared__ float c_sh[2 * 16 * 16];
    __shared__ float acc_sh[2 * 16 * 16];

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 32u * 128u; i += 64u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, comp, d)
                                       : index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 64u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma64_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        /* P2 Inc3b: optional packed FP4 indexer mirror (NULL = F32). */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 64u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 128u || head_dim != 128u) return;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 64u; i += 128u) {
                const uint32_t r = i >> 6u;
                const uint32_t c = i & 63u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[64 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float acc_sh[4 * 16 * 16];

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 64u * 128u; i += 128u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, comp, d)
                                       : index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 128u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        /* P2 Inc3b: optional packed FP4 indexer mirror (NULL = F32). */
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale,
        Ds4IndexerDiagnostics* diagnostics) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 32u;  /* inc-3: 32-token tiles */
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    if (use_fp4 && tid == 0 && blockIdx.x == 0u && blockIdx.y == 0u) atomicAdd(&g_fp4_index_read_path_blocks, 1ull);

    if (causal) {
        const uint32_t last_token = min(tile_t + 32u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    /* v0.5 scorer diet inc-1 + inc-3.  Inc-1: 136-half smem stride (the
     * natural 128-half stride is 256 B = every ldmatrix row in one bank
     * group, ~8-way conflicts; ncu Mem Busy 86% at 15% of bandwidth; pad =
     * 2.2x bit-identical).  Inc-3: register blocking -- each warp keeps its
     * accumulators in registers (ReLU*w folded per element via the
     * runtime-verified m16n16k16 fragment mapping: quad = lane>>2,
     * row = quad + ((i&2)?8:0), col = (lane&3)*2 + (i&1) + ((i&4)?8:0)),
     * which deletes the per-head store_matrix_sync + 8-KB c_sh readback +
     * one of three per-head barriers; and the CTA covers 32 TOKENS so two
     * A fragments share every b_frag ldmatrix (3 loads per 2 MMAs) and the
     * 34.8-KB b_sh stage amortizes over twice the output.  Proto
     * (proto_indexer_score_prefill.cu k_rb2): 1.69-1.86x over the padded
     * baseline at every depth, bit-identical scores.  Grid y is
     * ceil(n_tokens/32) -- the launch site must match. */
    __shared__ __half a_sh[32 * 136];
    __shared__ __half b_sh[128 * 136];

    const uint32_t lane = tid & 31u;
    const uint32_t quad = lane >> 2u;
    const uint32_t tcol = (lane & 3u) * 2u;

    float acc0[8], acc1[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; }

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, comp, d)
                                       : index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 136u] = __float2half(v);
    }
    __syncthreads();

    const uint32_t t0_lo = tile_t + quad;
    const uint32_t t0_hi = tile_t + quad + 8u;
    const uint32_t t1_lo = tile_t + 16u + quad;
    const uint32_t t1_hi = tile_t + 24u + quad;

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[r * 136u + d] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
        wmma::fill_fragment(c0, 0.0f);
        wmma::fill_fragment(c1, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a0, a_sh + k0, 136);
            wmma::load_matrix_sync(a1, a_sh + 16u * 136u + k0, 136);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 136u + k0, 136);
            wmma::mma_sync(c0, a0, b_frag, c0);
            wmma::mma_sync(c1, a1, b_frag, c1);
        }

        /* OOB rows staged 0 -> dot 0 -> fmax(0)*0 adds 0; OOB comps staged
         * 0 likewise, and the write loop re-guards.  Bitwise-identical to
         * the smem-epilogue chain: same per-element op order. */
        const float w0_lo = t0_lo < n_tokens ? weights[(uint64_t)t0_lo * n_head + h] : 0.0f;
        const float w0_hi = t0_hi < n_tokens ? weights[(uint64_t)t0_hi * n_head + h] : 0.0f;
        const float w1_lo = t1_lo < n_tokens ? weights[(uint64_t)t1_lo * n_head + h] : 0.0f;
        const float w1_hi = t1_hi < n_tokens ? weights[(uint64_t)t1_hi * n_head + h] : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc0[i] += fmaxf(c0.x[i], 0.0f) * ((i & 2) ? w0_hi : w0_lo);
            acc1[i] += fmaxf(c1.x[i], 0.0f) * ((i & 2) ? w1_hi : w1_lo);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t row = quad + ((i & 2) ? 8u : 0u);
        const uint32_t col = tcol + (i & 1) + ((i & 4) ? 8u : 0u);
        const uint32_t comp = tile_c + warp * 16u + col;
        for (uint32_t rt = 0; rt < 2u; rt++) {
            const uint32_t token = tile_t + rt * 16u + row;
            if (token < n_tokens && comp < n_comp) {
                float out = (rt ? acc1[i] : acc0[i]) * scale;
                if (causal) {
                    const uint32_t visible = (pos0 + token + 1u) / ratio;
                    if (comp >= visible) out = -INFINITY;
                }
                scores[(uint64_t)token * n_comp + comp] = out;
            }
        }
    }
#endif
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k,
        /* PC5 substrate override (see indexer_topk_8192_cub_kernel docstring). */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_actual; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        /* PC5: optional per-layer substrate override.  When non-NULL the
         * kernel reads the live count from ls_override->n_index_comp at
         * execute time (capture-safe) instead of the inline n_comp arg
         * baked at capture.  Row stride remains n_comp because the
         * scores buffer's layout was written by the producer kernel with
         * that stride; only the data-vs-padding boundary uses n_actual.
         * See "live score producer, stale top-k consumer" notes. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_actual) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        /* PC5 substrate override; see indexer_topk_8192_cub_kernel docstring. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_actual) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        /* PC5 substrate override; see indexer_topk_8192_cub_kernel docstring. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_actual) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        /* PC5 substrate override; see indexer_topk_8192_cub_kernel docstring. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_actual) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride,
        /* PC5 substrate override; see indexer_topk_8192_cub_kernel docstring.
         * Chunk count (gridDim.y) must be sized off n_comp_max in the host
         * shim so a captured grid covers any future live n_index_comp; this
         * kernel then early-returns chunks fully past n_actual and partially
         * masks the boundary chunk. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_actual) {
        /* Whole chunk past the live count: emit -INFINITY sentinels so the
         * merge stage sees a deterministic empty set rather than stale data
         * left over from a previous-token's captured replay. */
        uint32_t *out_pad = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
        for (uint32_t i = tid; i < top_k; i += blockDim.x) {
            out_pad[i] = UINT32_MAX;
        }
        return;
    }
    const uint32_t chunk_n = n_actual - chunk_start < SORT_N ? n_actual - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride,
        /* PC5 substrate override; see indexer_topk_8192_cub_kernel docstring. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_actual) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride,
        /* PC5 substrate override; see indexer_topk_8192_cub_kernel docstring. */
        const struct ds4_layer_scalars * __restrict__ ls_override) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const uint32_t n_actual = ls_override ? ls_override->n_index_comp : n_comp;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_actual) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

/* inc-4 topk diet: single-pass streaming exact top-512 for the prefill
 * chunked tier (n_comp > 8192, n_tokens >= 32).  One block per token
 * streams the whole scores row once, keeping candidates above a rising
 * exact threshold (the 512th-best-so-far after each buffer compaction; a
 * prefix 512th-best key can only be <= the global one, so no true member
 * is ever rejected).  Total order = topk_pack_key descending == the
 * topk_score_better order used by the bitonic tiers (and already relied
 * on by the 8192-cub tier), so the selected array is byte-identical to
 * the chunked tree's, sentinel behavior included.  The rotated per-token
 * start offset breaks globally monotone score trends (recency-shaped
 * rows), the one measured adversarial accept-storm case; exactness is
 * stream-order independent so the rotation cannot change the result.
 * Replaces the chunk sorts, the merge tree, AND the per-launch candidate
 * scratch on this tier (proto_topk_diet.cu: 372-check bitwise sweep vs
 * the tree across shapes x adversarial patterns + CPU reference;
 * memcheck-clean; the 2026-07-28 racecheck "tool artifact" ruling was
 * re-adjudicated 2026-08-04: the tools DO over-report this pattern class
 * (condbar-control receipt), but a real compact-verdict race hid beneath
 * the noise — see the #9 ROOT CAUSE comment at the verdict snapshot). */
#ifdef DS4_S512_TRACE
#define S512_MARK(itv, ph) do { \
    if ((tid & 31u) == 0u && g_s512_trace != NULL) { \
        const uint32_t s_ = ((uint32_t)t * 16u + (tid >> 5u)) * 2u; \
        g_s512_trace[s_] = ((uint32_t)(itv) << 8) | (uint32_t)(ph); \
        g_s512_trace[s_ + 1u] = s_cnt; \
        __threadfence_system(); \
    } \
} while (0)
#else
#define S512_MARK(itv, ph) do { } while (0)
#endif
__global__ static void __launch_bounds__(512) indexer_topk_stream512_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        /* 13d-1: nullable live substrate.  Under capture the caller passes
         * &g_layer_dev[il]: n_comp stays the CAPTURED BAND (the scores row
         * stride the producer wrote), while the scan bound and rotation
         * modulo read n_index_comp at replay time.  Eager callers pass
         * NULL and scan exactly n_comp; for equal live counts the walk is
         * identical, so eager and captured selections are byte-identical. */
        const struct ds4_layer_scalars *ls,
        Ds4IndexerDiagnostics* diagnostics) {
    constexpr uint32_t STREAM_THREADS = 512u;
    constexpr uint32_t STREAM_ITEMS = 4u;
    constexpr uint32_t STREAM_CAP = STREAM_THREADS * STREAM_ITEMS;   /* 2048 */
    using StreamSort = cub::BlockRadixSort<uint64_t, STREAM_THREADS, STREAM_ITEMS>;
    /* buf and the cub scratch are deliberately SEPARATE regions: an earlier
     * draft aliased them and persisted candidates across sorts, which
     * violates the TempStorage reuse contract (racecheck-confirmed hazards,
     * in-vivo illegal access).  Keys cross between the regions in registers
     * only.  ~33 KB static smem total, 3 blocks/SM. */
    __shared__ uint64_t buf[STREAM_CAP];
    __shared__ typename StreamSort::TempStorage sort_tmp;
    __shared__ uint32_t s_cnt;
    __shared__ uint32_t s_vcnt;   /* frozen compact verdict, see loop */
    __shared__ uint64_t s_thr;

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const float *row = scores + (uint64_t)t * n_comp;   /* stride = caller extent (band under capture) */
    uint32_t n = ls ? ls->n_index_comp : n_comp;        /* live scan bound */
    /* #9 diagnosis (2026-08-04): the live bound must stay within (0,
     * n_comp] -- the scorer wrote rows at n_comp stride with [visible,
     * band) = -INF, and the rotation below takes % n.  A replay that
     * observes live > band (band-crossing race) or live == 0 (bank
     * reset mid-flight) is the suspected illegal-access class: record
     * it loudly and clamp instead of faulting. */
    if (ls != NULL && (n == 0u || n > n_comp)) {
        if (tid == 0u) {
            atomicAdd(&g_topk_bound_viol_count, 1u);
            g_topk_bound_viol_n = n;
            g_topk_bound_viol_stride = n_comp;
        }
        if (n == 0u) return;
        n = n_comp;
    }
    if (tid == 0) { s_cnt = 0; s_thr = 0; }
    __syncthreads();
    S512_MARK(0u, 1u);

    const uint32_t start = (uint32_t)(((uint64_t)(t + 1u) * 0x9E3779B9u) % n);
    /* tile = one read per thread: STREAM_CAP - tile = 1536 leaves 1024 keys
     * of slack above the kept top_k, so a compact runs only when the buffer
     * genuinely fills.  A wider tile with zero slack (CAP - tile == top_k)
     * degenerates to a full sort per tile at depth — slack, not capacity,
     * prices the compact cadence (proto receipt topk_diet_timing3 vs 4). */
    const uint32_t tile = blockDim.x;
    for (uint32_t base = 0; base < n; base += tile) {
        const uint64_t thr = s_thr;
        {
            const uint32_t i = base + tid;
            uint64_t key = 0;
            bool take = false;
            if (i < n) {
                uint32_t c = start + i;
                if (c >= n) c -= n;
                key = topk_pack_key(row[c], c);
                take = key > thr;   /* thr=0 accepts all real keys */
            }
            /* warp-aggregated append: one atomic per warp of accepts */
            const uint32_t ball = __ballot_sync(0xffffffffu, take);
            if (take) {
                const uint32_t lane = tid & 31u;
                const uint32_t rank = __popc(ball & ((1u << lane) - 1u));
                uint32_t pos = 0;
                if (rank == 0) pos = atomicAdd(&s_cnt, __popc(ball));
                pos = __shfl_sync(ball, pos, __ffs(ball) - 1);
#ifdef DS4_S512_APPEND_GUARD
                /* #9 harness-only probe: bound the append and count
                 * overflow attempts instead of faulting (adjudicated
                 * 2026-08-04: overflow here was NOT the fault site). */
                {
                    const uint32_t bidx = pos + rank;
                    if (bidx >= STREAM_CAP) {
                        atomicAdd(&g_s512_append_ovf_count, 1u);
                        atomicMax(&g_s512_append_ovf_max, bidx);
                    } else {
                        buf[bidx] = key;
                    }
                }
#else
                buf[pos + rank] = key;
#endif
            }
        }
        __syncthreads();
        /* #9 ROOT CAUSE (2026-08-04): the compact verdict must be read
         * from a value FROZEN under barriers.  Reading s_cnt directly
         * here races with the NEXT iteration's append atomicAdd on the
         * skip path (no barrier in between): a fast warp that reads
         * "skip" can loop around and bump s_cnt before a slow sibling
         * has read it, the sibling then crosses the threshold and enters
         * the compact alone.  Warps split across iterations, arrivals at
         * the single hardware barrier conflate, cub's rank state
         * corrupts, and a garbage rank scatters outside the smem window
         * (illegal access / field Xid 13 OOR).  Observed live via the
         * DS4_S512_TRACE breadcrumbs: sibling warps frozen in iter-k and
         * iter-k+1 compacts simultaneously (ledger 08-04).  The snapshot
         * barrier below closes the window by construction: no iter-k+1
         * append can start until every warp has passed it, so s_vcnt is
         * the exact completed-append count and the verdict is uniform. */
        if (tid == 0) s_vcnt = s_cnt;
        __syncthreads();
        S512_MARK(base / tile, 2u);
        if (s_vcnt > STREAM_CAP - tile) {
            /* compact: sort descending, keep top_k, raise the threshold */
            const uint32_t cnt = s_vcnt;
            uint64_t keys[STREAM_ITEMS];
            #pragma unroll
            for (uint32_t k = 0; k < STREAM_ITEMS; k++) {
                const uint32_t i = tid * STREAM_ITEMS + k;
                keys[k] = (i < cnt) ? buf[i] : 0;
            }
            __syncthreads();
            S512_MARK(base / tile, 3u);
            StreamSort(sort_tmp).SortDescending(keys);
            __syncthreads();
            S512_MARK(base / tile, 4u);
            if (tid < top_k / STREAM_ITEMS) {
                #pragma unroll
                for (uint32_t k = 0; k < STREAM_ITEMS; k++)
                    buf[tid * STREAM_ITEMS + k] = keys[k];
            }
            if (tid == top_k / STREAM_ITEMS - 1) s_thr = keys[STREAM_ITEMS - 1];
            if (tid == 0) s_cnt = top_k;
            __syncthreads();
            S512_MARK(base / tile, 5u);
        }
    }
    /* final: sort the survivors, emit the top_k ids in tree order */
    {
        const uint32_t cnt = s_cnt;
        uint64_t keys[STREAM_ITEMS];
        #pragma unroll
        for (uint32_t k = 0; k < STREAM_ITEMS; k++) {
            const uint32_t i = tid * STREAM_ITEMS + k;
            keys[k] = (i < cnt) ? buf[i] : 0;
        }
        __syncthreads();
        S512_MARK(0xFFFu, 6u);
        StreamSort(sort_tmp).SortDescending(keys);
        __syncthreads();
        S512_MARK(0xFFFu, 7u);
        if (tid < top_k / STREAM_ITEMS) {
            #pragma unroll
            for (uint32_t k = 0; k < STREAM_ITEMS; k++)
                selected[(uint64_t)t * top_k + tid * STREAM_ITEMS + k] =
                    0xffffffffu - (uint32_t)(keys[k] & 0xffffffffu);
        }
        S512_MARK(0xFFFu, 8u);
    }
}
#undef S512_MARK

#ifdef DS4_CUDA_HAVE_MXF4
/* ================= v0.5 inc-13a: exact mxf4 score+select ==================
 * Replaces the dense scores+topk pair on eager prefill chunks at ALL
 * depths with the exact minimal chain: (1) e2m1-quantize the indexer Q
 * rows, (2) mxf4 block-scale MMA coarse scores (F32 accumulate, F32 row
 * emit), (3) the shipped exact top-512 select over the f32 rows.  No
 * candidate pool, no rescore, no f16 anywhere in selection.
 *
 * Why exact: the model's DSA lightning indexer already quantizes q AND K
 * onto the e2m1 x pow2 grid (byte-verified across 21 layers on real
 * captures, rmse 0.00e+00 vs the F32 oracle), so the mxf4 coarse scores
 * are LOSSLESS in vivo and the f32-key select is deterministic (idx-asc
 * tie-break).  The retired inc-5/inc-7 pool+rescore chain was built
 * against a synthetic-data artifact and measurably DEGRADED real
 * selection (tie collapse in the f16 coarse row + f16-wmma reduction
 * noise); receipts in mxf4_diet/proto_real_sweep_l2-42.txt.
 *
 * Coarse geometry: 16-token x 256-comp tiles, Q staged in shared memory
 * with next-group cp.async prefetch.  The 16-token Q-staging tile shape
 * is adapted from xangel82/DS4-GB10-GX10-DSpark-CUDA
 * cuda/indexer/ds4_indexer_sm121.cu (MIT).
 * Portions Copyright (c) 2026 Marco Palaferri.
 * Our v3 evolution (receipts mxf4_diet/proto_rrv2_run5_v3.txt +
 * ncu_v3hg16w16.txt): paired-line swizzled Q smem (a row's four A-words
 * = one conflict-free LDS.128), 16-head cp.async staging groups (4
 * barriers/CTA vs 64), 16 warps x 2 col-groups, F32 emit.  1.24x the
 * imported geometry, 5.0x the shipped dense pair at n_comp 131072;
 * bitwise-twin gates at every proto shape incl. 3 real capture layers.
 *
 * Requires the sm_121a build (kind::mxf4 block-scale MMA): the Makefile
 * maps CUDA_ARCH=sm_121 to -gencode arch=compute_121a,code=sm_121a and
 * defines DS4_CUDA_HAVE_MXF4; -arch=sm_121a alone SILENTLY emits .target
 * sm_121 and ptxas rejects the MMA. */
#define RR_H 64u
#define RR_D 128u

/* nearest e2m1 magnitude level with ties toward the LOWER level (the host
 * quantizer's tie-break; proto-pinned so coarse scores match the receipts) */
__device__ static inline uint32_t rr_e2m1_nearest_lvl(float av) {
    if (av <= 0.25f) return 0u;
    if (av <= 0.75f) return 1u;
    if (av <= 1.25f) return 2u;
    if (av <= 1.75f) return 3u;
    if (av <= 2.5f)  return 4u;
    if (av <= 3.5f)  return 5u;
    if (av <= 5.0f)  return 6u;
    return 7u;
}

/* Q e2m1 quantization (mirror layout: 64 code bytes + 4 ue8m0 exponents
 * packed in one u32 per (token, head) row).  One warp per row; lane L owns
 * dims 4L..4L+3 (all inside 32-dim block L>>3). */
__global__ static void rr_q_quant_kernel(const float * __restrict__ q,
                                         unsigned char * __restrict__ qc,
                                         uint32_t * __restrict__ qsf,
                                         uint32_t rows) {
    const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= rows) return;
    const float *src = q + (uint64_t)row * RR_D;
    float v[4]; float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = src[lane * 4u + i];
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (uint32_t off = 4; off; off >>= 1u)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 8));
    /* pow2-ceil exponent of amax/6 (ue8m0; 127 = 2^0) */
    uint32_t e = 127u;
    if (amax > 0.0f) {
        const uint32_t b = __float_as_uint(amax / 6.0f);
        e = (b >> 23u) & 0xFFu;
        if (b & 0x7FFFFFu) e++;
        e = min(max(e, 1u), 254u);
    }
    const float rcp = __uint_as_float((254u - e) << 23u); /* 2^(127-e) */
    uint32_t nibs = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t lvl = rr_e2m1_nearest_lvl(fabsf(v[i]) * rcp);
        nibs |= (lvl | (v[i] < 0.0f ? 8u : 0u)) << (4 * i);
    }
    ((uint16_t *)qc)[(uint64_t)row * 32u + lane] = (uint16_t)nibs;
    const uint32_t e1 = __shfl_sync(0xffffffffu, e, 8);
    const uint32_t e2 = __shfl_sync(0xffffffffu, e, 16);
    const uint32_t e3 = __shfl_sync(0xffffffffu, e, 24);
    if (lane == 0)
        qsf[row] = e | (e1 << 8u) | (e2 << 16u) | (e3 << 24u);
}

/* 13a.2: mirror-path companion -- packs the producer QAT emit's four F32
 * pow2 block scales per row into the coarse kernel's ue8m0 qsf word.  A
 * 16-B read + 4-B write per row (the coarse kernel's qsf staging is
 * untouched, so its cp.async pipeline -- and SASS -- stay exactly the
 * 13a form; an ABBA showed a direct F32-scale read in the stager costs
 * ~1% at the 64k window). */
__global__ static void rr_q_scale_pack_kernel(const float * __restrict__ qs4,
                                              uint32_t * __restrict__ qsf,
                                              uint32_t rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint4 sb = *reinterpret_cast<const uint4 *>(qs4 + (uint64_t)row * 4u);
    qsf[row] = ((sb.x >> 23u) & 0xFFu)
             | (((sb.y >> 23u) & 0xFFu) << 8u)
             | (((sb.z >> 23u) & 0xFFu) << 16u)
             | (((sb.w >> 23u) & 0xFFu) << 24u);
}

__device__ static inline void rr_mma_mxf4(float d[4], const uint32_t a[4],
                                          uint32_t b0, uint32_t b1,
                                          uint32_t sfa, uint32_t sfb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1210
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(sfa), "r"(sfb));
#else
    // Native dispatch refuses MXF4 off sm_121a. Fail a stray unsupported launch.
    asm volatile("trap;");
#endif
}

/* cp.async helpers for the coarse Q staging (LDGSTS bypasses the RF; the
 * pred=false form copies zero bytes and zero-fills the smem word) */
__device__ __forceinline__ void idx_cp_async_u32(uint32_t *dst_smem,
                                                 const void *src_gmem,
                                                 bool pred) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(s), "l"(src_gmem), "r"(pred ? 4u : 0u));
}
__device__ __forceinline__ void idx_cp_commit(void) {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void idx_cp_wait0(void) {
    asm volatile("cp.async.wait_group 0;\n");
}

/* mxf4 coarse scorer v3: CTA = 16 tokens x 256 comps, 16 warps, each warp
 * owns 2 col-groups of 8 comps ((cg*16 + warp)*8).  K fragments (mirror
 * code words ARE the B operand) + ue8m0 exponents load ONCE per warp and
 * stay loop-invariant across all 64 heads.  Q rows stage into shared
 * memory in 16-head groups with cp.async ping-pong (4 barriers/CTA);
 * within a group buffer, two 17-word rows share one 128-B paired line
 * with code words in slot(w) = (w%4)*4 + w/4 order, so a thread's four
 * A-words for one row are ONE conflict-free LDS.128 (row scales live in a
 * separate side array).  w*ReLU folds on D fragments in registers; output
 * is F32 full rows (-INF past the causal clamp) at row stride out_stride,
 * feeding the exact top-512 select directly. */
__global__ static void __launch_bounds__(512) rr_coarse_mxf4_v3_kernel(
        float *scores, const unsigned char * __restrict__ qc,
        const uint32_t * __restrict__ qsf, const float * __restrict__ weights,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t ratio,
        float scale, const unsigned char * __restrict__ fp4,
        const float * __restrict__ fp4_sc, uint32_t out_stride) {
    const uint32_t HG = 16u;   /* heads per staging group */
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t tig = lane & 3u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tile_c = blockIdx.x * 256u;

    {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp) : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 256u; i += 512u) {
                const uint32_t r = i >> 8u, c = i & 255u;
                const uint32_t token = tile_t + r, comp = tile_c + c;
                if (token < n_tokens && comp < n_comp)
                    scores[(uint64_t)token * out_stride + comp] = -INFINITY;
            }
            return;
        }
    }

    __shared__ __align__(16) uint32_t q_sh[2][16][8][32];
    __shared__ uint32_t sfa_sh[2][16][16];
    __shared__ float w_sh[2][16][16];

    /* loop-invariant K fragments: cg col-group, regs {b0,b1} x k-halves */
    uint32_t bfrag[2][4], sfb[2][2];
#pragma unroll
    for (uint32_t cg = 0; cg < 2u; cg++) {
        const uint32_t col = tile_c + (cg * 16u + warp) * 8u + (lane >> 2u);
        if (col < n_comp) {
            const uint32_t *kw = (const uint32_t *)(fp4 + (uint64_t)col * 64u);
            bfrag[cg][0] = kw[tig];       bfrag[cg][1] = kw[4u + tig];
            bfrag[cg][2] = kw[8u + tig];  bfrag[cg][3] = kw[12u + tig];
            const uint32_t *sb = (const uint32_t *)(fp4_sc + (uint64_t)col * 4u);
            sfb[cg][0] = ((sb[0] >> 23u) & 0xFFu) | (((sb[1] >> 23u) & 0xFFu) << 8u);
            sfb[cg][1] = ((sb[2] >> 23u) & 0xFFu) | (((sb[3] >> 23u) & 0xFFu) << 8u);
        } else {
            bfrag[cg][0] = bfrag[cg][1] = bfrag[cg][2] = bfrag[cg][3] = 0u;
            sfb[cg][0] = sfb[cg][1] = 127u | (127u << 8u);
        }
    }

    /* group stager: HG heads x 16 tokens, 16 code words + 1 scale word
     * each; codes land permuted in the paired line, scales in the side
     * array, weights alongside */
    auto stage = [&](uint32_t h0, uint32_t buf) {
        for (uint32_t i = tid; i < HG * 16u * 17u; i += 512u) {
            const uint32_t hh = i / (16u * 17u);
            const uint32_t rem = i - hh * (16u * 17u);
            const uint32_t m = rem / 17u;
            const uint32_t word = rem - m * 17u;
            const uint32_t token = tile_t + m;
            const bool ok = token < n_tokens;
            const uint64_t row = (uint64_t)token * RR_H + h0 + hh;
            const void *src;
            uint32_t *dst;
            if (word < 16u) {
                src = ok ? (const void *)(qc + row * 64u + word * 4u)
                         : (const void *)qc;
                dst = &q_sh[buf][hh][m >> 1u]
                           [(m & 1u) * 16u + (word & 3u) * 4u + (word >> 2u)];
            } else {
                src = ok ? (const void *)(qsf + row) : (const void *)qsf;
                dst = &sfa_sh[buf][hh][m];
            }
            idx_cp_async_u32(dst, src, ok);
        }
        for (uint32_t i = tid; i < HG * 16u; i += 512u) {
            const uint32_t hh = i >> 4u, m = i & 15u;
            const uint32_t token = tile_t + m;
            const bool ok = token < n_tokens;
            const void *src = ok
                ? (const void *)(weights + (uint64_t)token * RR_H + h0 + hh)
                : (const void *)weights;
            idx_cp_async_u32((uint32_t *)&w_sh[buf][hh][m], src, ok);
        }
    };

    stage(0u, 0u);
    idx_cp_commit();
    idx_cp_wait0();
    __syncthreads();

    float acc[2][4];
#pragma unroll
    for (uint32_t cg = 0; cg < 2u; cg++)
        acc[cg][0] = acc[cg][1] = acc[cg][2] = acc[cg][3] = 0.0f;

    const uint32_t m0 = lane >> 2u;
    const uint32_t scale_m = (lane & 1u) * 8u + m0;
    for (uint32_t g = 0; g < RR_H / HG; g++) {
        const uint32_t cur = g & 1u;
        if (g + 1u < RR_H / HG) { stage((g + 1u) * HG, cur ^ 1u); idx_cp_commit(); }
#pragma unroll
        for (uint32_t hh = 0; hh < HG; hh++) {
            const uint4 va0 = *reinterpret_cast<const uint4 *>(
                &q_sh[cur][hh][m0 >> 1u][(m0 & 1u) * 16u + tig * 4u]);
            const uint4 va1 = *reinterpret_cast<const uint4 *>(
                &q_sh[cur][hh][(m0 + 8u) >> 1u][(m0 & 1u) * 16u + tig * 4u]);
            const uint32_t qs = sfa_sh[cur][hh][scale_m];
            const uint32_t sfa0 = qs & 0xFFFFu;
            const uint32_t sfa1 = qs >> 16u;
            const float w_lo = w_sh[cur][hh][m0];
            const float w_hi = w_sh[cur][hh][m0 + 8u];
            const uint32_t a_lo[4] = { va0.x, va1.x, va0.y, va1.y };
            const uint32_t a_hi[4] = { va0.z, va1.z, va0.w, va1.w };
#pragma unroll
            for (uint32_t cg = 0; cg < 2u; cg++) {
                float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                rr_mma_mxf4(d, a_lo, bfrag[cg][0], bfrag[cg][1], sfa0, sfb[cg][0]);
                rr_mma_mxf4(d, a_hi, bfrag[cg][2], bfrag[cg][3], sfa1, sfb[cg][1]);
                acc[cg][0] += fmaxf(d[0], 0.0f) * w_lo;
                acc[cg][1] += fmaxf(d[1], 0.0f) * w_lo;
                acc[cg][2] += fmaxf(d[2], 0.0f) * w_hi;
                acc[cg][3] += fmaxf(d[3], 0.0f) * w_hi;
            }
        }
        if (g + 1u < RR_H / HG) { idx_cp_wait0(); __syncthreads(); }
    }

    const uint32_t r0 = tile_t + m0;
    const uint32_t r1 = r0 + 8u;
    const uint32_t vis0 = (pos0 + r0 + 1u) / ratio;
    const uint32_t vis1 = (pos0 + r1 + 1u) / ratio;
#pragma unroll
    for (uint32_t cg = 0; cg < 2u; cg++) {
        const uint32_t colb = tile_c + (cg * 16u + warp) * 8u + tig * 2u;
#pragma unroll
        for (uint32_t j = 0; j < 2u; j++) {
            const uint32_t col = colb + j;
            if (col >= n_comp) continue;
            if (r0 < n_tokens)
                scores[(uint64_t)r0 * out_stride + col] =
                    col >= vis0 ? -INFINITY : acc[cg][j] * scale;
            if (r1 < n_tokens)
                scores[(uint64_t)r1 * out_stride + col] =
                    col >= vis1 ? -INFINITY : acc[cg][2u + j] * scale;
        }
    }
}

#endif  // DS4_CUDA_HAVE_MXF4

// clang-format on
#undef g_fp4_index_read_path_blocks
#undef g_topk_bound_viol_count
#undef g_topk_bound_viol_n
#undef g_topk_bound_viol_stride
#undef DS4_CUDA_UNUSED
#undef DS4_CUDA_TOPK_MERGE_GROUP
#ifdef RR_H
#undef RR_H
#undef RR_D
#endif

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_INDEXER_CORE_CUH_
