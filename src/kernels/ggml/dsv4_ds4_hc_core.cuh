// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me>
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// Original Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976.
// Numerical functions copied verbatim from ds4_cuda.cu. Extraction and
// integration provenance is in dsv4_ds4_hc_core.json. Only included inside
// the CUDA adapter's anonymous namespace; no original runtime or globals.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_HC_CORE_CUH_
#define JITLLM_KERNELS_GGML_DSV4_DS4_HC_CORE_CUH_

// clang-format off
__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void rms_norm_plain_f16_rows_kernel(__half *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    __half *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = __float2half(xr[i] * scale);
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void rms_norm_weight_f16pair_kernel(float *out, __half *out16, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    __half *o16row = out16 + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i] * scale * w[i];
        orow[i] = v;
        o16row[i] = __float2half(v);
    }
}

__global__ static void rms_norm_weight_f16q8_kernel(float *out, __half *out16, char *y_q8, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    __half *o16row = out16 + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i] * scale * w[i];
        orow[i] = v;
        o16row[i] = __float2half(v);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t qb = threadIdx.x >> 5u; qb < (n >> 7u); qb += (blockDim.x >> 5u)) {
        const float4 xi = *reinterpret_cast<const float4 *>(orow + (uint64_t)qb * 128u + lane * 4u);
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 32));
        }
        float rcp_amax;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp_amax) : "f"(amax));
        const float d_inv = __fmul_rn(rcp_amax, 127.0f);
        char4 q;
        q.x = (int8_t)roundf(__fmul_rn(xi.x, d_inv));
        q.y = (int8_t)roundf(__fmul_rn(xi.y, d_inv));
        q.z = (int8_t)roundf(__fmul_rn(xi.z, d_inv));
        q.w = (int8_t)roundf(__fmul_rn(xi.w, d_inv));
        char *blk = y_q8 + ((uint64_t)qb * rows + row) * 144u;
        *reinterpret_cast<char4 *>(blk + 16u + lane * 4u) = q;
        if ((lane & 7u) == 0u) {
            float d;
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(d_inv));
            reinterpret_cast<float *>(blk)[lane >> 3u] = d;
        }
    }
}

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

template <uint32_t N_EMBD, uint32_t N_HC>
__global__ static void __launch_bounds__(256) hc_expand_split_rmsf16_kernel(
        float *out_hc,
        __half *xh_out,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add,
        float eps,
        /* v0.5 inc-8 F5: optional per-expert UNSUMMED MoE down output
         * ([token][n_expert_used][N_EMBD]).  Non-NULL replaces block_out
         * with the guarded expert sum (bit-identical to moe_sum_kernel:
         * ascending e, isfinite-guarded), staged once per token in shared
         * memory so the four dst_hc passes read it at smem cost. */
        const float *moe_unsummed,
        uint32_t n_expert_used) {
    constexpr uint32_t N = N_EMBD * N_HC;
    constexpr uint32_t NPT = N / 256u;
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    const float *bo = block_out + (uint64_t)t * N_EMBD;
    const float *ba = block_add + (uint64_t)t * N_EMBD;
    const float *res = residual_hc + (uint64_t)t * N;
    const float *po = post + (uint64_t)t * post_stride;
    const float *co = comb + (uint64_t)t * comb_stride;
    float *orow = out_hc + (uint64_t)t * N;
    __half *hrow = xh_out + (uint64_t)t * N;

    /* moe path: each thread's NPT unrolled elements revisit the same
     * N_EMBD/256 unique d values once per dst_hc, so the guarded expert
     * sum stages into registers (no smem, no occupancy change) and is
     * reused across the four dst_hc passes. */
    constexpr uint32_t NDU = N_EMBD / 256u;
    float bsum[NDU];
    if (moe_unsummed) {
        const float *md = moe_unsummed + (uint64_t)t * n_expert_used * N_EMBD;
#pragma unroll
        for (uint32_t k = 0; k < NDU; k++) {
            const uint32_t d = threadIdx.x + k * 256u;
            float acc = 0.0f;
            for (uint32_t e = 0; e < n_expert_used; e++) {
                const float mv = md[(uint64_t)e * N_EMBD + d];
                if (isfinite(mv)) acc += mv;
            }
            if (has_add) acc += ba[d];
            bsum[k] = acc;
        }
    }

    float v[NPT];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < NPT; j++) {
        const uint32_t i = threadIdx.x + j * 256u;
        const uint32_t dst_hc = i / N_EMBD;
        const uint32_t d = i % N_EMBD;
        float block_v;
        if (moe_unsummed) {
            block_v = bsum[j % NDU];
        } else {
            block_v = bo[d];
            if (has_add) block_v += ba[d];
        }
        float acc = block_v * po[dst_hc];
        for (uint32_t src_hc = 0; src_hc < N_HC; src_hc++) {
            acc += co[dst_hc + src_hc * N_HC] * res[(uint64_t)src_hc * N_EMBD + d];
        }
        orow[i] = acc;
        v[j] = acc;
        sum += acc * acc;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = 256u >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)N + eps);
#pragma unroll
    for (uint32_t j = 0; j < NPT; j++) {
        const uint32_t i = threadIdx.x + j * 256u;
        hrow[i] = __float2half(v[j] * scale);
    }
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}
// clang-format on

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_HC_CORE_CUH_
