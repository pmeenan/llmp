// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Verbatim numerical functions from Entrpi/ds4
// 76d51ef82a81b70b78e51a3a6ea11946286de976.
// ds4_cuda.cu SHA256 8d5de76a7aaaf9131ba8f9cf35412863fef88299b39676ea4bcfb07aef7386e3:
//   8262-8342, 8420-8477, 9797-10066.
// cuda/mmq/ds4_mmq_d2r.cu SHA256 9dd3be8c4af62a2ead319fe72e7ff6881472ce8a6749d7b6885a6956f2cf79b9:
//   205-208, 215-231, 251-263, 284-306, 3057-3308.
// cuda/mmq/ds4_mmq.cu SHA256 aaee3dcc55789efe54c17dad66cd39b24f2a8f28b7d119ded980a03f658cb7b8:
//   427-432.
// Original device compilation uses --use_fast_math. The adapter preserves
// all numerical bodies and supplies only explicit borrowed operands.
// The actual upstream permission notices remain in NOTICE/LICENSES.
//
// Private to dsv4_ds4_product_raw.cu: its original MMQ includes precede this.
//
// clang-format off
__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        /* v0.5 inc-8: optional per-row absolute positions (n_tok entries),
         * same source selection as rope_tail_kernel.  NULL -> scalar
         * fast-path (pos0 + t*pos_stride), byte-identical math. */
        const int32_t *positions,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
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
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const int32_t eff_pos = positions ? positions[t]
                                      : (int32_t)(pos0 + t * pos_stride);
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
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
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
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
/* ---- v0.5 flat-pool piece 1: fused own out_a --------------------------
 *
 * One kernel replaces the inverse-rope f16 pack + cublas strided-batched
 * f16 GEMM pair on the batch prefill path: it reads the f32 heads
 * DIRECTLY (staging converts in-register, the rope rotation rides the
 * staging loop via the (c,s) table below), dequants the aligned Q8_0
 * out_a rows in-register, and writes the interleaved f32 low rows.  That
 * retires the pack kernel's ~386 MB/launch f32->f16 round trip AND the
 * q8->f16 colmajor out_a cache (~2.9 GiB + the inc-12f boot prebuild) on
 * this path.  Proto receipts (cuda/mmq/test/proto_outa_own.cu, .33):
 * complex -22..25% vs pack+cublas iso at N=512/2048/4096; f64-oracle
 * error identical to the cublas leg at every shape.  VALUE-parity
 * increment (HMMA k-order differs from cutlass): goldens re-based at
 * ship.  Kill switch DS4_CUDA_NO_OUTA_OWN restores pack+cublas exactly.
 *
 * The (c,s) table companion follows the cp.async-pipeline law (13a.2):
 * transcendentals never ride a staging loop.  It computes cos/sin*mscale
 * per (token, rope pair) with EXPRESSION-IDENTICAL math to
 * attention_inverse_rope_pack_group_heads_f16_kernel (same compiled
 * transcendentals, same contraction-pinned rotation at the consumer), so
 * the staged f16 values match the pack path bit-for-bit; only the GEMM
 * accumulation order differs. */
static constexpr uint32_t kOAGroups = 8u;     /* n_groups   */
static constexpr uint32_t kOARank   = 1024u;  /* rank       */
static constexpr uint32_t kOAK      = 4096u;  /* group_dim  */
static constexpr uint32_t kOALow    = 8192u;  /* n_groups * rank */
static constexpr uint32_t kOAHD     = 512u;   /* head_dim   */
static constexpr uint32_t kOANRot   = 64u;
static constexpr uint32_t kOANNope  = kOAHD - kOANRot;
static constexpr uint32_t kOATileM  = 128u;   /* tokens per CTA    */
static constexpr uint32_t kOATileN  = 128u;   /* rank cols per CTA */
static constexpr uint32_t kOATileK  = 32u;    /* == QK8_0: one q8 block per row-stage */
static constexpr uint32_t kOAWarps  = 8u;
static constexpr uint32_t kOAPad    = 8u;     /* __half pad: wmma ldm %8 */

__global__ static void attention_outa_rope_cs_table_kernel(
        float2 *tab,
        uint32_t n_tokens,
        uint32_t pos0,
        const int32_t *positions,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t np = kOANRot / 2u;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_tokens * np) return;
    const uint32_t p = gid % np;
    const uint32_t t = gid / np;
    const uint32_t i = 2u * p;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)kOANRot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)kOANRot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(kOANRot - 1), corr1);
    }
    int32_t eff_pos = positions ? positions[t] : (int32_t)(pos0 + t);
    float theta_extrap = (float)eff_pos * powf(freq_base, -((float)i) / (float)kOANRot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    tab[gid] = make_float2(cosf(theta) * mscale, -sinf(theta) * mscale);
}

/* CTA tile 128 tokens x 128 rank cols, k32 double-buffered smem stages,
 * 8 warps x wmma m16n16k16 f16->f32 (32x64 per warp), 40 KiB smem => 2
 * CTAs/SM (the measured occupancy sweet spot; k64 at 1 CTA/SM lost ~5 TF
 * in the proto).  Grid (rank tiles, token tiles, groups): x walks the 8
 * rank tiles of one token stripe (stripe hot in L2), z keeps one group's
 * 4.25 MB q8 rows L2-resident across token tiles (d2r-schedule-order
 * law).  Q8 dequant is the byte-trick (biased byte ORed into a f16
 * mantissa, bias subtracted exactly, single-rounded __hmul2) --
 * bit-identical to round_f16(d*q), no F32 pipe in the staging loop. */
__global__ static void __launch_bounds__(kOAWarps * 32u, 2)
attention_outa_fused_own_kernel(
        float *low,                       /* [n_tokens][kOALow] interleaved */
        const float *heads,               /* [n_tokens][kOAGroups][kOAK]    */
        const float2 *tab,                /* [n_tokens][kOANRot/2]          */
        const __half *a_sc,               /* aligned q8: scales plane       */
        const int8_t *a_qs,               /* aligned q8: qs plane           */
        /* p5a: optional block_q8_1_mmq D4 dual-emit of `low` for the out_b
         * mmq (NULL = no emit).  One CTA tile = exactly one 128-element
         * k-segment block per token row (ib = kseg*n_tokens + row); layout
         * = 16 B d4[4] then qs[128] (144 B stride), math op-for-op equal
         * to quantize_mmq_q8_1<D4> (amax is order-insensitive; d_inv and
         * d replicate the reference's exact division expressions). */
        char *y_q8,
        uint32_t n_tokens) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ __half oa_smem[];
    constexpr uint32_t XS = kOATileM * (kOATileK + kOAPad);
    constexpr uint32_t WS = kOATileN * (kOATileK + kOAPad);
    __half *xs = oa_smem;            /* 2 * XS */
    __half *ws = oa_smem + 2 * XS;   /* 2 * WS */

    const uint32_t g    = blockIdx.z;
    const uint32_t row0 = blockIdx.y * kOATileM;   /* token base    */
    const uint32_t col0 = blockIdx.x * kOATileN;   /* rank-col base */
    const uint32_t tid  = threadIdx.x;
    const uint32_t warp = tid / 32u;

    /* X loaders: 16 B = 4 f32; A loaders: 16 B = 16 q8 (one half-block) */
    constexpr uint32_t XQPR   = kOATileK / 4u;
    constexpr uint32_t XRSTEP = (kOAWarps * 32u) / XQPR;
    constexpr uint32_t XSTEPS = kOATileM / XRSTEP;
    constexpr uint32_t WQPR   = kOATileK / 16u;
    constexpr uint32_t WRSTEP = (kOAWarps * 32u) / WQPR;
    constexpr uint32_t WSTEPS = kOATileN / WRSTEP;
    const uint32_t xlr = tid / XQPR;
    const uint32_t xlc = (tid % XQPR) * 4u;
    const uint32_t wlr = tid / WQPR;
    const uint32_t wlc = (tid % WQPR) * 16u;

    auto stage = [&](uint32_t buf, uint32_t k0) {
        int4 vx[XSTEPS];
#pragma unroll
        for (uint32_t s = 0; s < XSTEPS; s++) {
            const uint32_t t = row0 + xlr + s * XRSTEP;
            vx[s] = (t < n_tokens)
                ? *(const int4 *)(heads + ((uint64_t)t * kOAGroups + g) * kOAK + k0 + xlc)
                : make_int4(0, 0, 0, 0);
        }
#pragma unroll
        for (uint32_t s = 0; s < XSTEPS; s++) {
            __half *dst = xs + buf * XS + (xlr + s * XRSTEP) * (kOATileK + kOAPad) + xlc;
            float f0 = ((const float *)&vx[s])[0];
            float f1 = ((const float *)&vx[s])[1];
            float f2 = ((const float *)&vx[s])[2];
            float f3 = ((const float *)&vx[s])[3];
            const uint32_t hd = (k0 + xlc) % kOAHD;  /* 4-elem chunk never straddles */
            if (hd >= kOANNope) {
                const uint32_t t = row0 + xlr + s * XRSTEP;
                const uint32_t p0 = (hd - kOANNope) / 2u;
                const float2 cs0 = tab[(uint64_t)t * (kOANRot / 2u) + p0];
                const float2 cs1 = tab[(uint64_t)t * (kOANRot / 2u) + p0 + 1u];
                /* contraction pin, verbatim from the pack kernel */
                float r0 = __fmaf_rn(cs0.x, f0, -__fmul_rn(cs0.y, f1));
                f1 = __fmaf_rn(cs0.y, f0, __fmul_rn(cs0.x, f1));
                f0 = r0;
                float r2 = __fmaf_rn(cs1.x, f2, -__fmul_rn(cs1.y, f3));
                f3 = __fmaf_rn(cs1.y, f2, __fmul_rn(cs1.x, f3));
                f2 = r2;
            }
            ((__half2 *)dst)[0] = __floats2half2_rn(f0, f1);
            ((__half2 *)dst)[1] = __floats2half2_rn(f2, f3);
        }
#pragma unroll
        for (uint32_t s = 0; s < WSTEPS; s++) {
            const uint64_t r = (uint64_t)g * kOARank + col0 + wlr + s * WRSTEP;
            const int4 q16 = *(const int4 *)(a_qs + r * kOAK + k0 + wlc);
            const __half2 dh = __half2half2(a_sc[r * (kOAK / 32u) + (k0 + wlc) / 32u]);
            const __half2 bias = __half2half2(__ushort_as_half(0x6480u)); /* 1152.0 */
            const uint32_t *qw = (const uint32_t *)&q16;
            __half2 *h2 = (__half2 *)(ws + buf * WS + (wlr + s * WRSTEP) * (kOATileK + kOAPad) + wlc);
#pragma unroll
            for (uint32_t w4 = 0; w4 < 4u; w4++) {
                const uint32_t qq = qw[w4] ^ 0x80808080u;
                const uint32_t lo = __byte_perm(qq, 0x64u, 0x4140u);
                const uint32_t hi = __byte_perm(qq, 0x64u, 0x4342u);
                h2[2u * w4]      = __hmul2(dh, __hsub2(*(const __half2 *)&lo, bias));
                h2[2u * w4 + 1u] = __hmul2(dh, __hsub2(*(const __half2 *)&hi, bias));
            }
        }
    };

    const uint32_t wr0 = (warp / 2u) * 32u;   /* token offset in tile */
    const uint32_t wc0 = (warp % 2u) * 64u;   /* col offset in tile   */

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    stage(0, 0);
    __syncthreads();

    constexpr uint32_t k_stages = kOAK / kOATileK;
    for (uint32_t ks = 0; ks < k_stages; ks++) {
        const uint32_t cur = ks & 1u;
        if (ks + 1u < k_stages) stage(cur ^ 1u, (ks + 1u) * kOATileK);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[4];
        for (uint32_t kk = 0; kk < kOATileK; kk += 16u) {
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(fa[i],
                    xs + cur * XS + (wr0 + i * 16u) * (kOATileK + kOAPad) + kk,
                    kOATileK + kOAPad);
            for (int j = 0; j < 4; j++)
                wmma::load_matrix_sync(fb[j],
                    ws + cur * WS + (wc0 + j * 16u) * (kOATileK + kOAPad) + kk,
                    kOATileK + kOAPad);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            const uint32_t gt = row0 + wr0 + (uint32_t)i * 16u;
            const uint32_t gc = g * kOARank + col0 + wc0 + (uint32_t)j * 16u;
            if (gt < n_tokens)
                wmma::store_matrix_sync(low + (uint64_t)gt * kOALow + gc,
                                        acc[i][j], kOALow, wmma::mem_row_major);
        }
    }

    /* p5a q8_1 dual-emit: re-read this CTA's own just-stored C tile (L2-hot;
     * __syncthreads makes the block's global stores visible block-wide) and
     * write one 144-B block per token row.  512 (row, 32-col group) pairs
     * over 256 threads. */
    if (y_q8) {
        __syncthreads();
        const uint32_t kseg = (g * kOARank + col0) / 128u;
#pragma unroll
        for (uint32_t s = 0; s < 2u; s++) {
            const uint32_t p = tid + s * 256u;
            const uint32_t r = p / 4u;
            const uint32_t grp = p % 4u;
            const uint32_t gt = row0 + r;
            if (gt >= n_tokens) continue;
            const float *src = low + (uint64_t)gt * kOALow + g * kOARank + col0 + grp * 32u;
            float v[32];
            float amax = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 x4 = *(const float4 *)(src + j);
                v[j] = x4.x; v[j + 1] = x4.y; v[j + 2] = x4.z; v[j + 3] = x4.w;
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(x4.x), fabsf(x4.y)),
                                         fmaxf(fabsf(x4.z), fabsf(x4.w))));
            }
            /* SASS-pinned to quantize_mmq_q8_1's fast-math lowering (read
             * from the compiled object, 12c method): d_inv = FMUL(MUFU.RCP
             * (amax), 127) and the stored d = bare MUFU.RCP(d_inv).  The
             * straight-line source form let the optimizer rewrite
             * 1.0f/(127.0f/amax) algebraically -- an ulp-level scale
             * divergence the golden tripped on (top20_max_abs 8.55). */
            float rcp_amax;
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp_amax) : "f"(amax));
            const float d_inv = __fmul_rn(rcp_amax, 127.0f);
            char *blk = y_q8 + ((uint64_t)kseg * n_tokens + gt) * 144u;
            int4 qw[2];
            /* int8_t, NOT char: aarch64 char is UNSIGNED -- a bare-char cast
             * compiled to F2I.U32 and wrecked every negative quant (verify
             * receipt: all d4 exact, qs diverging).  The reference's char4
             * fields are signed char => signed F2I.TRUNC. */
            int8_t *qb = (int8_t *)qw;
#pragma unroll
            for (int j = 0; j < 32; j++) qb[j] = (int8_t)roundf(__fmul_rn(v[j], d_inv));
            int4 *dst_qs = (int4 *)(blk + 16u + grp * 32u);
            dst_qs[0] = qw[0];
            dst_qs[1] = qw[1];
            float d_store;
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d_store) : "f"(d_inv));
            ((float *)blk)[grp] = d_store;
        }
    }
}

__device__ __forceinline__ void zero_16B(void *dst) {
    int4 z = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(dst) = z;
}
__device__ __forceinline__ void cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_16B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    } else {
        zero_16B(dst);
    }
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}
__device__ __forceinline__ int d2r_lane() {
#if defined(__CUDA_ARCH__)
    uint32_t lane;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane));
    return (int)lane;
#else
    return (int)threadIdx.x;
#endif
}

__device__ __forceinline__ int d2r_warp() {
#if defined(__CUDA_ARCH__)
    uint32_t warp;
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(warp));
    return (int)warp;
#else
    return (int)threadIdx.y;
#endif
}

__device__ __forceinline__ int d2r_tid() {
    return (d2r_warp() << 5) | d2r_lane();
}
namespace dq8 {

constexpr int kDqMTile   = 128;
constexpr int kDqNTile   = 128;
constexpr int kDqRowWarps = 8;
constexpr int kDqColWarps = 2;
constexpr int kDqWarps   = kDqRowWarps * kDqColWarps;
constexpr int kDqThreads = 32 * kDqWarps;
constexpr int kDqStages  = 3;
constexpr int kDqRowPad  = 80;  // 64 B payload + 16 B: 20-int ldmatrix stride rotates banks
constexpr int kDqNFrag   = kDqNTile / 8;
constexpr int kDqNFragPerWarp = kDqNFrag / kDqColWarps;

constexpr size_t kDqSmemWQBytes  = (size_t)kDqStages * kDqMTile * kDqRowPad;
constexpr size_t kDqSmemQ8QBytes = (size_t)kDqStages * kDqNTile * kDqRowPad;
constexpr size_t kDqSmemQ8HBytes = (size_t)kDqStages * kDqNTile * 16;
constexpr size_t kDqSmemWDBytes  = (size_t)kDqStages * kDqMTile * sizeof(uint32_t);
constexpr size_t kDqSmemWQOff  = 0;
constexpr size_t kDqSmemQ8QOff = kDqSmemWQOff + kDqSmemWQBytes;
constexpr size_t kDqSmemQ8HOff = kDqSmemQ8QOff + kDqSmemQ8QBytes;
constexpr size_t kDqSmemWDOff  = kDqSmemQ8HOff + kDqSmemQ8HBytes;
constexpr size_t kDqSmemTotalBytes = kDqSmemWDOff + kDqSmemWDBytes;
static_assert(kDqSmemTotalBytes <= 99ull * 1024ull, "dense D2R dynamic smem exceeds sm limit");

struct DenseQ8Params {
    const char *wd_row0;       // dq plane at cta_row0 (row stride nb*2 = K/16)
    const char *wq_row0;       // qs plane at cta_row0 (row stride K)
    const char *q8_tile;       // q8 blocks at col_lo
    uint32_t q8_k128_stride_bytes;
    uint32_t wq_row_stride;    // K bytes
    int k64_iters;
    int k128_iters;
};

__device__ __forceinline__ void dq8_cp_async_4B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem), "l"(src));
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#else
    if (pred) {
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#endif
}

__device__ __forceinline__ void dq8_issue_q8_qs(
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const char *base = p.q8_tile +
                       (uint64_t)(k64_iter >> 1) * p.q8_k128_stride_bytes +
                       16u + (uint64_t)(k64_iter & 1) * 64u;
    const int buf = k64_iter % kDqStages;
    const int chunk = d2r_tid();
    const int col = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_q8q[buf][col][h * 16];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq) + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_q8_hdr(
        float (&s_q8h)[kDqStages][kDqNTile][4],
        const DenseQ8Params &p, int k128_iter) {
    if (d2r_tid() >= kDqNTile) {
        return;
    }
    const bool pred = k128_iter < p.k128_iters;
    const char *base = p.q8_tile + (uint64_t)k128_iter * p.q8_k128_stride_bytes;
    const int col = d2r_tid();
    void *dst = &s_q8h[k128_iter % kDqStages][col][0];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq);
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_w(
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const int wst = k64_iter % kDqStages;
    const char *wq = p.wq_row0 + (uint64_t)k64_iter * 64u;
    const char *wd = p.wd_row0 + (uint64_t)k64_iter * 4u;
    const uint32_t wq_row_stride = p.wq_row_stride;
    const uint32_t wd_row_stride = p.wq_row_stride >> 4;  // nb*2 = K/16
    const int chunk = d2r_tid();
    const int row = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_wq[wst][row][h * 16];
    const void *src = wq + (uint64_t)row * wq_row_stride + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
    if (d2r_tid() < kDqMTile) {
        void *ddst = &s_wd[wst][d2r_tid()];
        const void *dsrc = wd + (uint64_t)d2r_tid() * wd_row_stride;
        dq8_cp_async_4B(ddst, dsrc, pred);
    }
}

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void dq8_mainloop(
        float (&acc)[kDqNFragPerWarp][4],
        float (&s_q8h)[kDqStages][kDqNTile][4],
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int k64_iters = p.k64_iters;
    const int lane = d2r_lane();
    const int group = lane >> 2;
    const int wrow = (d2r_warp() >> 1) * 16;
    const int nf0 = (d2r_warp() & 1) * kDqNFragPerWarp;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    dq8_issue_w(s_wq, s_wd, p, 0);
    dq8_issue_q8_qs(s_q8q, p, 0);
    dq8_issue_q8_hdr(s_q8h, p, 0);
    cp_async_commit();
    dq8_issue_w(s_wq, s_wd, p, 1);
    dq8_issue_q8_qs(s_q8q, p, 1);
    dq8_issue_q8_hdr(s_q8h, p, 1);
    cp_async_commit();

    for (int i = 0; i < k64_iters; ++i) {
        // wait -> barrier -> issue: the barrier both publishes stage i CTA-wide
        // and proves all warps finished stage i-1, whose buffer ((i-1)%3 ==
        // (i+2)%3) the issues below overwrite.
        cp_async_wait_group<1>();
        __syncthreads();
        dq8_issue_w(s_wq, s_wd, p, i + 2);
        dq8_issue_q8_qs(s_q8q, p, i + 2);
        if ((i & 1) == 0) {
            dq8_issue_q8_hdr(s_q8h, p, (i >> 1) + 2);
        }
        cp_async_commit();

        const int wst = i % kDqStages;
        const int qbuf = i % kDqStages;
        const int hbuf = (i >> 1) % kDqStages;
        const uint32_t dw0_bits = s_wd[wst][wrow + group];
        const uint32_t dw1_bits = s_wd[wst][wrow + group + 8];
        const float2 dw0 = __half22float2(*reinterpret_cast<const half2 *>(&dw0_bits));
        const float2 dw1 = __half22float2(*reinterpret_cast<const half2 *>(&dw1_bits));

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            TileA A;
            ggml_cuda_mma::load_ldmatrix(
                A, reinterpret_cast<const int *>(&s_wq[wst][wrow][t * 32]),
                kDqRowPad / (int)sizeof(int));
            const int tq8 = (i & 1) * 2 + t;
            const float dwr0 = t ? dw0.y : dw0.x;
            const float dwr1 = t ? dw1.y : dw1.x;
#pragma unroll
            for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
                TileB B;
                TileC C;
                ggml_cuda_mma::load_ldmatrix(
                    B, reinterpret_cast<const int *>(&s_q8q[qbuf][(nf0 + nf) * 8][t * 32]),
                    kDqRowPad / (int)sizeof(int));
                ggml_cuda_mma::mma(C, A, B);
                const float da0 = s_q8h[hbuf][(nf0 + nf) * 8 + c0][tq8];
                const float da1 = s_q8h[hbuf][(nf0 + nf) * 8 + c1][tq8];
                acc[nf][0] = fmaf((float)C.x[0], dwr0 * da0, acc[nf][0]);
                acc[nf][1] = fmaf((float)C.x[1], dwr0 * da1, acc[nf][1]);
                acc[nf][2] = fmaf((float)C.x[2], dwr1 * da0, acc[nf][2]);
                acc[nf][3] = fmaf((float)C.x[3], dwr1 * da1, acc[nf][3]);
            }
        }
    }
}

__global__ __launch_bounds__(kDqThreads, 1)
void dense_q8_d2r_kernel(const char * __restrict__ wd_plane,
                         const char * __restrict__ wq_plane,
                         const block_q8_1_mmq * __restrict__ q8,
                         float * __restrict__ out,
                         int M, int N, int K, int group_m) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    extern __shared__ __align__(16) char dq8_smem[];
    auto &s_wq  = *reinterpret_cast<int8_t (*)[kDqStages][kDqMTile][kDqRowPad]>(dq8_smem + kDqSmemWQOff);
    auto &s_q8q = *reinterpret_cast<int8_t (*)[kDqStages][kDqNTile][kDqRowPad]>(dq8_smem + kDqSmemQ8QOff);
    auto &s_q8h = *reinterpret_cast<float (*)[kDqStages][kDqNTile][4]>(dq8_smem + kDqSmemQ8HOff);
    auto &s_wd  = *reinterpret_cast<uint32_t (*)[kDqStages][kDqMTile]>(dq8_smem + kDqSmemWDOff);

    // Grouped supertile order: group_m row-tiles stay L2-hot while col tiles
    // stream past them.
    const int num_m = M / kDqMTile;
    const int num_n = (N + kDqNTile - 1) / kDqNTile;
    const int width = group_m * num_n;
    const int g = (int)blockIdx.x / width;
    const int rem = (int)blockIdx.x - g * width;
    int gsize = num_m - g * group_m;
    if (gsize > group_m) {
        gsize = group_m;
    }
    const int pid_m = g * group_m + rem % gsize;
    const int pid_n = rem / gsize;

    const int cta_row0 = pid_m * kDqMTile;
    const int col_lo = pid_n * kDqNTile;

    DenseQ8Params p;
    p.wd_row0 = wd_plane + (uint64_t)cta_row0 * (uint64_t)(K >> 4);
    p.wq_row0 = wq_plane + (uint64_t)cta_row0 * (uint64_t)K;
    p.q8_tile = reinterpret_cast<const char *>(q8) +
                (uint64_t)col_lo * sizeof(block_q8_1_mmq);
    p.q8_k128_stride_bytes = (uint32_t)((uint64_t)N * sizeof(block_q8_1_mmq));
    p.wq_row_stride = (uint32_t)K;
    p.k64_iters = K >> 6;
    p.k128_iters = K >> 7;

    float acc[kDqNFragPerWarp][tile_C::ne] = {};

    dq8_mainloop<tile_A, tile_B, tile_C>(acc, s_q8h, s_q8q, s_wq, s_wd, p);

    // Column-major out [N][M]; rows always in range (M % 128 == 0 validated).
    // The isfinite guard preserves the sanitize contract of the mmq path.
    const int out_col_lo = col_lo + (d2r_warp() & 1) * (kDqNFragPerWarp * 8);
    const int out_row0 = cta_row0 + ((d2r_warp() >> 1) << 4);
#pragma unroll
    for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (col < N) {
                const float v = isfinite(acc[nf][l]) ? acc[nf][l] : 0.0f;
                out[(uint64_t)col * (uint64_t)M + (uint64_t)row] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wd_plane, wq_plane, q8, out, M, N, K, group_m);
    NO_DEVICE_CODE;
#endif
}

} // namespace dq8

__global__ static void ds4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}


typedef struct { __half2 ds; int8_t qs[32]; } ds4_hc_block_q8_1;

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps,
        /* M2-Inc2a: optional q8_1 emission of the q row (decode row 0) —
         * kills the q_b mmvq quantize prelude.  Bit-exact vs the vendored
         * quantize_q8_1 (proto_m2_qkv.cu 2a): the emission pass re-reads the
         * just-written orow values (identical bits) in warp granules. */
        ds4_hc_block_q8_1 *q81) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
    if (which == 0u && q81 != NULL && row == 0u && (n & 31u) == 0u) {
        __syncthreads();
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t warp = threadIdx.x >> 5u;
        const uint32_t nwarp = blockDim.x >> 5u;
        for (uint32_t qb = warp; qb < n / 32u; qb += nwarp) {
            const float v = orow[qb * 32u + lane];
            float amax = fabsf(v);
            float s = v;
            for (uint32_t off = 16u; off > 0u; off >>= 1u) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
                s += __shfl_xor_sync(0xffffffffu, s, off, 32);
            }
            const float d1 = amax / 127.0f;
            const int8_t qv = amax == 0.0f ? (int8_t)0 : (int8_t)roundf(v / d1);
            q81[qb].qs[lane] = qv;
            if (lane == 0u) q81[qb].ds = __floats2half2_rn(d1, s);
        }
    }
}

// clang-format on
