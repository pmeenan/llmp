// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-License-Identifier: MIT
// Verbatim Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976
// ds4_cuda.cu:6453-6479, 6629-6733, original SHA256
// 8d5de76a7aaaf9131ba8f9cf35412863fef88299b39676ea4bcfb07aef7386e3.
// clang-format off
__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Deterministic block reduction for a 256-thread (8-warp) block. Fixed
 * shuffle-tree order within each warp, then a fixed 8-element shuffle-tree
 * across warps -- scheduling-independent, so it is bit-identical run-to-run
 * and between eager and captured replay. Result is valid in thread 0. */
__device__ static float block_reduce_sum_256(float v) {
    v = warp_sum_f32(v);                       /* lane 0 of each warp holds its warp sum */
    __shared__ float s[8];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    float r = 0.0f;
    if (warp == 0u) {
        r = (lane < 8u) ? s[lane] : 0.0f;
        r += __shfl_down_sync(0xffffffffu, r, 4);
        r += __shfl_down_sync(0xffffffffu, r, 2);
        r += __shfl_down_sync(0xffffffffu, r, 1);
    }
    return r;                                   /* valid in thread 0 (warp 0, lane 0) */
}

/* Split-K F16 matmul, pass 1 (decode, n_tok==1). Each (row, kseg) block
 * reduces its contiguous in_dim segment [kseg*seg, (kseg+1)*seg) with
 * coalesced stride-blockDim loads, then writes a per-segment partial. Splitting
 * the row's dot product across kseg blocks raises the launched-block count so
 * the GPU's SMs actually fill (the single-block-per-row fast kernel left
 * waves_per_multiprocessor << 1 on the small-out_dim projections). */
__global__ static void matmul_f16_splitk_kernel(
        float *partial,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t ksplit,
        int vec_ok) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint32_t kseg = blockIdx.y;
    if (row >= out_dim) return;
    /* Round the segment length up to a multiple of 8 so every segment start k0
     * is 8-half aligned -> the uint4 (8-half) vector loads below stay 16B
     * aligned. Segments still tile [0,in_dim); a fully-trailing segment
     * (k0 >= in_dim) just contributes 0. */
    uint64_t seg = (in_dim + (uint64_t)ksplit - 1u) / (uint64_t)ksplit;
    seg = (seg + 7u) & ~(uint64_t)7u;
    const uint64_t k0 = (uint64_t)kseg * seg;
    if (k0 >= in_dim) {
        if (threadIdx.x == 0u) partial[row * (uint64_t)ksplit + (uint64_t)kseg] = 0.0f;
        return;
    }
    uint64_t k1 = k0 + seg;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const uint32_t tid = threadIdx.x;
    const uint32_t nt = blockDim.x;
    float sum = 0.0f;
    if (vec_ok) {
        /* Vectorized: each thread consumes 8 contiguous halves per step via a
         * 16B uint4 load (coalesced: consecutive threads -> consecutive 16B),
         * paired with two float4 activation loads. ~8x fewer memory
         * instructions than the scalar path -> better latency tolerance on a
         * reduction that ncu shows is latency-, not bandwidth-, bound. */
        const uint64_t vend = k0 + ((k1 - k0) & ~(uint64_t)7u);
        for (uint64_t i = k0 + (uint64_t)tid * 8u; i < vend; i += (uint64_t)nt * 8u) {
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                 + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        for (uint64_t i = vend + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    } else {
        for (uint64_t i = k0 + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    }
    sum = block_reduce_sum_256(sum);
    if (threadIdx.x == 0u) partial[row * (uint64_t)ksplit + (uint64_t)kseg] = sum;
}

/* Split-K F16 matmul, pass 2: combine the ksplit per-segment partials for each
 * row in a fixed (kseg ascending) order. Fixed order keeps the result
 * deterministic and eager==captured. */
__global__ static void matmul_f16_splitk_combine_kernel(
        float *out,
        const float *partial,
        uint64_t out_dim,
        uint32_t ksplit) {
    const uint64_t row = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (row >= out_dim) return;
    const float *p = partial + row * (uint64_t)ksplit;
    float s = 0.0f;
    for (uint32_t k = 0u; k < ksplit; k++) s += p[k];
    out[row] = s;
}

// clang-format on
