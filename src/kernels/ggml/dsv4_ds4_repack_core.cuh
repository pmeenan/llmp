// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me>
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// Three complete bit-preserving numerical kernels from pinned ds4_repack.cu.
// Only included in the native adapter's anonymous namespace; no upstream
// file mapping, allocation, runtime or weight server. See companion provenance.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_REPACK_CORE_CUH_
#define JITLLM_KERNELS_GGML_DSV4_DS4_REPACK_CORE_CUH_
// clang-format off
__global__ static void repack_iq2_xxs_aligned_kernel(
        __half *dq,
        uint2 *qs,
        const unsigned char *raw,
        uint64_t nblk) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 8ull) return;
    const uint64_t blk = i >> 3;
    const uint32_t p = (uint32_t)(i & 7u);
    const unsigned char *src = raw + blk * 66ull;
    if (p == 0u) {
        uint16_t h;
        memcpy(&h, src, 2u);
        dq[blk] = __ushort_as_half(h);
    }
    uint2 v;
    memcpy(&v, src + 2u + (uint64_t)p * 8u, 8u);
    qs[blk * 8ull + p] = v;
}

__global__ static void repack_q2_k_aligned_kernel(
        uint32_t *dm2,          // section base, uint32 view (2 words / pair blk)
        uint32_t *sc4,          // section base, uint32 view (8 words / pair blk)
        uint32_t *qs2,          // section base, uint32 view (32 words / pair blk)
        const unsigned char *raw,
        uint64_t g0,            // absolute raw-block index of raw[0]
        uint64_t cblk,          // raw blocks in this chunk
        uint32_t nb_row,        // blocks per row = K/256
        uint32_t nrows) {       // rows per expert = M
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cblk * 16ull) return;
    const uint64_t j = i >> 4;
    const uint32_t p = (uint32_t)(i & 15u);
    const uint64_t g = g0 + j;
    const uint32_t b = (uint32_t)(g % nb_row);
    const uint32_t r = (uint32_t)((g / nb_row) % nrows);
    const uint64_t e = g / ((uint64_t)nb_row * nrows);
    const uint64_t pblk = ((uint64_t)e * (nrows/2u) + r/2u) * nb_row + b;
    const uint32_t parity = r & 1u;
    const unsigned char *src = raw + j * 84ull;
    uint32_t w;
    memcpy(&w, src + 16u + (uint64_t)p * 4u, 4u);
    qs2[pblk * 32ull + (uint64_t)p * 2ull + parity] = w;
    if (p < 4u) {
        memcpy(&w, src + (uint64_t)p * 4u, 4u);
        /* scales word p covers bytes [4p, 4p+4) = window h = p>>1, half p&1 */
        sc4[pblk * 8ull + (uint64_t)(p >> 1) * 4ull + parity * 2ull + (p & 1u)] = w;
    }
    if (p == 0u) {
        memcpy(&w, src + 80u, 4u);
        dm2[pblk * 2ull + parity] = w;
    }
}

__global__ static void repack_q8_0_aligned_kernel(
        __half *dq,
        unsigned char *qs,
        const unsigned char *raw,
        uint64_t nblk) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 2ull) return;
    const uint64_t blk = i >> 1;
    const uint32_t p = (uint32_t)(i & 1u);
    const unsigned char *src = raw + blk * 34ull;
    if (p == 0u) {
        uint16_t h;
        memcpy(&h, src, 2u);
        dq[blk] = __ushort_as_half(h);
    }
    memcpy(qs + blk * 32ull + p * 16ull, src + 2u + p * 16ull, 16u);
}
// clang-format on
#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_REPACK_CORE_CUH_
