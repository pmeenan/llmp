// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me>
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// Original Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976.
// Whole numerical functions copied verbatim. The unused decode-scalar
// layout only completes the original kernel signature; wrappers pass NULL,
// and no foreign global registry, scalar owner or runtime is copied.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_COMP_CORE_CUH_
#define JITLLM_KERNELS_GGML_DSV4_DS4_COMP_CORE_CUH_

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

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        /* Phase 2 Step 2: optional per-row absolute positions (n_tok entries).
         * NULL -> scalar fast-path (pos0 + t*pos_stride), unchanged behavior. */
        const int32_t *positions,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    int32_t eff_pos = positions ? positions[t] : (int32_t)(pos0 + t * pos_stride);
    float theta_extrap = (float)eff_pos * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        /* Step 4c C1: optional device-scalars override.  When non-NULL the
         * kernel reads pos0 from s->pos0 at execution time instead of using
         * the inline arg.  Same pattern as attention's s_override from
         * Step 4a Commit B.  Prefill callers pass NULL and keep the inline-
         * arg path; the decode-time caller passes g_decode_dev so the
         * kernel-node arg list bakes a session-stable pointer rather than
         * a per-token literal. */
        const struct ds4_decode_scalars * __restrict__ s_override,
        /* C2 Inc1 (bank-agnostic cont capture): per-row substrate.  When
         * pos_dev is non-NULL the row's absolute position is read from the
         * per-step device positions array at the BAKED row index (replaces
         * the C1b pos_delta-from-s->pos0 scheme; works for any bank and,
         * later, any multi-live row mix).  When seq_dev is non-NULL,
         * state_kv/state_score are FULL multi-bank slabs and the kernel
         * offsets to lane seq_dev[row_idx] (lane stride = coff*ratio*width,
         * derived from ratio/head_dim).  Both NULL = inline/serial paths,
         * bit-exact. */
        const int32_t * __restrict__ pos_dev,
        const int32_t * __restrict__ seq_dev,
        uint32_t row_idx) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    if (pos_dev) {
        pos0 = (uint32_t)pos_dev[row_idx];
    } else if (s_override) {
        pos0 = s_override->pos0;
    }
    if (seq_dev) {
        const uint64_t lane = (uint64_t)(uint32_t)seq_dev[row_idx] *
                              (uint64_t)(coff * ratio) * width;
        state_kv += lane;
        state_score += lane;
    }
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *base,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t comp_row_inline,
        const uint32_t * __restrict__ row_ptr_dev,
        /* C2 Inc1: state slab lane -- non-NULL seq_dev means state_kv/
         * state_score are FULL multi-bank slabs; offset to lane
         * seq_dev[row_idx] (stride coff*ratio*width).  NULL = pre-offset
         * views (serial/eager), bit-exact.  C2 Inc2: the C1b row_delta
         * arg is gone -- emit ordering is baked into the published row. */
        const int32_t * __restrict__ seq_dev,
        uint32_t row_idx) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    const uint32_t comp_row = row_ptr_dev ? *row_ptr_dev : comp_row_inline;
    float *row = base + (uint64_t)comp_row * head_dim;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    if (seq_dev) {
        const uint64_t lane = (uint64_t)(uint32_t)seq_dev[row_idx] *
                              (uint64_t)(coff * ratio) * width;
        state_kv += lane;
        state_score += lane;
    }
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width,
                                                      /* C2 Inc1: optional slab lane (see pool kernel). */
                                                      const int32_t * __restrict__ seq_dev,
                                                      uint32_t row_idx) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    if (seq_dev) {
        const uint64_t lane = (uint64_t)(uint32_t)seq_dev[row_idx] * 2ull * half;
        state_kv += lane;
        state_score += lane;
    }
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}
// clang-format on

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_COMP_CORE_CUH_
