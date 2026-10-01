// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-License-Identifier: MIT

// Literal original numerical functions. Included only inside private linkage.
// See adjacent provenance. No foreign dispatcher or global substrate.

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

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (int i = 0; i < 256; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t]
                             : (scalars ? (int32_t)scalars->token : token_scalar);
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < 256; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < 256) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * 1.5f;
}

__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;
    __shared__ float sprob[256];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t]
                             : (scalars ? (int32_t)scalars->token : token_scalar);
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int j = 0; j < 6; j++) sel[j] = row[j];
    } else {
        for (int j = 0; j < 6; j++) sel[j] = -1;
        for (int e = 0; e < 256; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < 256) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < 6; j++) w[j] = w[j] / sum * 1.5f;
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t]
                                 : (scalars ? (int32_t)scalars->token : token_scalar);
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}

#define DS4_ROUTER_FUSE_IN     4096u
#define DS4_ROUTER_FUSE_OUT    256u
#define DS4_ROUTER_FUSE_KSPLIT 8u
#define DS4_ROUTER_FUSE_SEG    512u
#define DS4_ROUTER_FUSE_BLOCKS 128u


__global__ static void router_fused_coop_kernel(
        float *logits,            /* [256] */
        float *partials,          /* [256*8] g_router_fused_partials */
        int32_t *selected,        /* [6] */
        float *weights,           /* [6] */
        float *probs,             /* [256] */
        const __half *w,          /* [256][4096] f16 gate_inp */
        const float *x,           /* [4096] ffn_norm */
        const float *bias,        /* [256] or NULL */
        const int32_t *hash,      /* [hash_rows][6] or NULL */
        const int32_t *tokens,    /* decode: NULL */
        int32_t token_scalar,
        uint32_t hash_rows,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    __shared__ float sbias[DS4_ROUTER_FUSE_OUT];
    __shared__ float sprob[DS4_ROUTER_FUSE_OUT];
    __shared__ int32_t shash[6];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;

    /* Phase 0 (block 0): prefetch the tail's small model-map reads. */
    if (blockIdx.x == 0u) {
        if (has_bias) sbias[threadIdx.x] = bias[threadIdx.x];
        if (hash_mode && threadIdx.x == 0u) {
            int32_t tok = tokens ? tokens[0]
                                 : (scalars ? (int32_t)scalars->token : token_scalar);
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) shash[j] = row[j];
        }
    }

    /* Phase 1: split-K logits matmul, one warp per (row, kseg) tile. */
    const uint32_t nwarp = gridDim.x * (blockDim.x >> 5u);
    for (uint32_t t = blockIdx.x * (blockDim.x >> 5u) + warp;
         t < DS4_ROUTER_FUSE_OUT * DS4_ROUTER_FUSE_KSPLIT; t += nwarp) {
        const uint32_t row = t >> 3u;
        const uint32_t kseg = t & 7u;
        const uint64_t k0 = (uint64_t)kseg * DS4_ROUTER_FUSE_SEG;
        const __half *wr = w + (uint64_t)row * DS4_ROUTER_FUSE_IN;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        {
            const uint64_t i = k0 + (uint64_t)lane * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum0 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                  + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        {
            const uint64_t i = k0 + ((uint64_t)lane + 32u) * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum1 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                  + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        const float s0 = warp_sum_f32(sum0);
        const float s1 = warp_sum_f32(sum1);
        if (lane == 0u) partials[(uint64_t)row * DS4_ROUTER_FUSE_KSPLIT + kseg] = s0 + s1;
    }

    cooperative_groups::this_grid().sync();
    if (blockIdx.x != 0u) return;

    /* Phase 2: combine partials in fixed kseg order (verbatim combine kernel;
     * out_dim 256 == one 256-thread block). */
    {
        const uint64_t row = (uint64_t)threadIdx.x;
        const float *p = partials + row * DS4_ROUTER_FUSE_KSPLIT;
        float s = 0.0f;
        for (uint32_t k = 0u; k < DS4_ROUTER_FUSE_KSPLIT; k++) s += p[k];
        logits[row] = s;
    }
    __syncthreads();

    /* Phase 3: verbatim router_select_warp_topk_kernel warp body (t = 0,
     * row_in_block = 0). */
    if (warp != 0u) return;
    const float *log = logits;
    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? sbias[e] : 0.0f);
        sprob[e] = p;
        probs[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0u) {
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = shash[j];
                selected[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[(uint32_t)e] : 0.0f;
                weights[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) weights[j] = weights[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            selected[j] = (int32_t)out_idx[j];
            weights[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) weights[j] = weights[j] / sum * 1.5f;
    }
}

__global__ static void router_fused_rows_coop_kernel(
        float *logits,            /* [n,256] */
        float *partials,          /* [n*256*8] g_router_fused_partials */
        int32_t *selected,        /* [n,6] */
        float *weights,           /* [n,6] */
        float *probs,             /* [n,256] */
        const __half *w,          /* [256][4096] f16 gate_inp */
        const float *x,           /* [n,4096] batch_ffn_norm rows */
        const float *bias,        /* [256] or NULL */
        const int32_t *hash,      /* [hash_rows][6] or NULL */
        const int32_t *tokens,    /* [n] device token ids (hash mode) */
        uint32_t hash_rows,
        uint32_t n_tok,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;

    /* Phase 1: split-K logits matmul, one warp per (tok, row, kseg) tile.
     * Per-tok tile math identical to the serial kernel (and thus to the
     * unfused per-row split-K loop). */
    const uint32_t nwarp = gridDim.x * (blockDim.x >> 5u);
    const uint32_t tiles_per_tok = DS4_ROUTER_FUSE_OUT * DS4_ROUTER_FUSE_KSPLIT;
    for (uint32_t t = blockIdx.x * (blockDim.x >> 5u) + warp;
         t < n_tok * tiles_per_tok; t += nwarp) {
        const uint32_t tok = t / tiles_per_tok;
        const uint32_t tt = t - tok * tiles_per_tok;
        const uint32_t row = tt >> 3u;
        const uint32_t kseg = tt & 7u;
        const uint64_t k0 = (uint64_t)kseg * DS4_ROUTER_FUSE_SEG;
        const __half *wr = w + (uint64_t)row * DS4_ROUTER_FUSE_IN;
        const float *xr = x + (uint64_t)tok * DS4_ROUTER_FUSE_IN;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        {
            const uint64_t i = k0 + (uint64_t)lane * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(xr + i);
            const float4 xb = *reinterpret_cast<const float4 *>(xr + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum0 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                  + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        {
            const uint64_t i = k0 + ((uint64_t)lane + 32u) * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(xr + i);
            const float4 xb = *reinterpret_cast<const float4 *>(xr + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum1 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                  + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        const float s0 = warp_sum_f32(sum0);
        const float s1 = warp_sum_f32(sum1);
        if (lane == 0u)
            partials[((uint64_t)tok * DS4_ROUTER_FUSE_OUT + row) * DS4_ROUTER_FUSE_KSPLIT + kseg] = s0 + s1;
    }

    cooperative_groups::this_grid().sync();
    if (blockIdx.x != 0u) return;

    /* Phase 2: per-row combine in fixed kseg order (verbatim combine math). */
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        const uint64_t row = (uint64_t)threadIdx.x;
        const float *p = partials + ((uint64_t)tok * DS4_ROUTER_FUSE_OUT + row) * DS4_ROUTER_FUSE_KSPLIT;
        float s = 0.0f;
        for (uint32_t k = 0u; k < DS4_ROUTER_FUSE_KSPLIT; k++) s += p[k];
        logits[(uint64_t)tok * DS4_ROUTER_FUSE_OUT + row] = s;
    }
    __syncthreads();

    /* Phase 3: verbatim router_select_warp_topk_kernel warp body, one warp
     * per tok row (block 0 has 8 warps >= n_tok). */
    if (warp >= n_tok) return;
    {
        const uint32_t tok = warp;
        const float *log = logits + (uint64_t)tok * 256u;
        float *prob = probs + (uint64_t)tok * 256u;
        int32_t *sel = selected + (uint64_t)tok * 6u;
        float *wg = weights + (uint64_t)tok * 6u;
        __shared__ float sprob_rows[8][256];
        float local_prob[8];
        float local_score[8];
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float p = sqrtf(softplus_dev(log[e]));
            local_prob[j] = p;
            local_score[j] = p + (has_bias ? bias[e] : 0.0f);
            sprob_rows[tok][e] = p;
            prob[e] = p;
        }
        __syncwarp();

        if (hash_mode) {
            if (lane == 0u) {
                int32_t tk = tokens ? tokens[tok] : 0;
                if (tk < 0 || (uint32_t)tk >= hash_rows) tk = 0;
                const int32_t *hrow = hash + (uint64_t)tk * 6u;
                float sum = 0.0f;
                #pragma unroll
                for (uint32_t j = 0; j < 6u; j++) {
                    const int32_t e = hrow[j];
                    sel[j] = e;
                    const float v = (e >= 0 && e < 256) ? sprob_rows[tok][(uint32_t)e] : 0.0f;
                    wg[j] = v;
                    sum += v;
                }
                sum = fmaxf(sum, 6.103515625e-5f);
                #pragma unroll
                for (uint32_t j = 0; j < 6u; j++) wg[j] = wg[j] / sum * 1.5f;
            }
            return;
        }

        float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (uint32_t k = 0; k < 6u; k++) {
            float best_score = -INFINITY;
            float best_prob = 0.0f;
            uint32_t best_idx = UINT32_MAX;
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const uint32_t e = lane + j * 32u;
                const float s = local_score[j];
                if (router_score_better(s, e, best_score, best_idx)) {
                    best_score = s;
                    best_prob = local_prob[j];
                    best_idx = e;
                }
            }
            #pragma unroll
            for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
                const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
                const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
                const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
                if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                    best_score = other_score;
                    best_prob = other_prob;
                    best_idx = other_idx;
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const uint32_t e = lane + j * 32u;
                if (e == best_idx) local_score[j] = -INFINITY;
            }
            if (lane == 0) {
                out_idx[k] = best_idx;
                out_prob[k] = best_prob;
            }
        }

        if (lane == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                sel[j] = (int32_t)out_idx[j];
                wg[j] = out_prob[j];
                sum += out_prob[j];
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) wg[j] = wg[j] / sum * 1.5f;
        }
    }
}

__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

__global__ static void moe_mmq_swiglu_weighted_clamp_kernel(
        float *mid_out,
        float *gate_out_dbg, float *up_out_dbg,
        const float *gate_buf, const float *up_buf,
        const float *weights,
        uint32_t expert_mid_dim,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        float clamp) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_expert_used * expert_mid_dim;
    if (gid >= n) return;
    uint64_t slot_pair = gid / expert_mid_dim;
    uint32_t tok = (uint32_t)(slot_pair / n_expert_used);
    uint32_t slot = (uint32_t)(slot_pair - (uint64_t)tok * n_expert_used);
    float g = gate_buf[gid];
    float u = up_buf[gid];
    /* P3: sanitize at read (nonfinite -> 0, exactly what the standalone
     * ds4_mmq_sanitize_f32 pass wrote).  Lets the SoA mmq entries skip that
     * whole-buffer pass; sanitized producers are unaffected (values already
     * finite). */
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(u)) u = 0.0f;
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float w = weights[(uint64_t)tok * n_expert_used + slot];
    const float s = g / (1.0f + expf(-g));
    if (gate_out_dbg) gate_out_dbg[gid] = g;
    if (up_out_dbg)   up_out_dbg[gid]   = u;
    mid_out[gid] = s * u * w;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens, uint32_t guard_nonfinite) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        const float v = down[((uint64_t)tok * n_expert + e) * out_dim + row];
        if (!guard_nonfinite || isfinite(v)) acc += v;
    }
    out[gid] = acc;
}

// clang-format on
