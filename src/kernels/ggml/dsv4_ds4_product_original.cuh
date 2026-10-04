// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Verbatim numerical functions from Entrpi/ds4
// 76d51ef82a81b70b78e51a3a6ea11946286de976: the fused own output-A core
// (its inverse-RoPE table and staged F16/HMMA product) and the YaRN ramp it
// calls.
// ds4_cuda.cu SHA256 8d5de76a7aaaf9131ba8f9cf35412863fef88299b39676ea4bcfb07aef7386e3:
//   8339-8342, 9797-10066.
// Original device compilation uses --use_fast_math. The adapter preserves
// all numerical bodies and supplies only explicit borrowed operands.
// The actual upstream permission notices remain in NOTICE/LICENSES.
//
// Private to dsv4_ds4_product_raw.cu.
//
// clang-format off
__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
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
