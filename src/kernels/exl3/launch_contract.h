// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// How ExLlamaV3's kernels must be launched (docs/backend-proof.md, P1 and
// P3). Every llmpalooza launcher of a kept ExLlamaV3 kernel includes this
// header (launch.h), as the kernel tables do (tests/unit/exl3_tables.h),
// and a tooling test (tools/tests/test_sources.py) requires it to name
// every __global__ kernel the locked subset keeps. Line numbers below are
// of the locked, patched tree.
//
// Cooperative kernels: exl3_gemm_kernel and exl3_mgemm_kernel, both in
// quant/exl3_gemm_kernel.cuh, in all 64 instances of the tables
// tfp_exl3_gemm_kernel_* and tfp_exl3_mgemm_kernel_*; and exl3_gemv_kernel
// (quant/exl3_gemv_kernel.cuh), in the eight K = 4 mcg instances of
// llmp/llmp_exl3_kernels.cu.
//
// 1. Launch cooperatively, with every block co-resident:
//    cudaLaunchCooperativeKernel (or cudaLaunchKernelEx with
//    cudaLaunchAttributeCooperative) and no more blocks than the occupancy
//    calculator allows across the device's SMs.
//    - exl3_gemm_kernel and exl3_mgemm_kernel: a block of
//      EXL3_GEMM_BLOCKDIM[shape] threads (quant/exl3_kernel_map.cuh) and
//      SMEM_MAX bytes of dynamic shared memory (quant/exl3_gemm_inner.cuh;
//      raised with cudaFuncSetAttribute first). Upstream's exl3_gemm.cu
//      launches exl3_gemm_kernel on (num_sms, 1, 1) and exl3_mgemm_kernel on
//      (num_sms, 1, concurrency) with num_sms * concurrency <= the SM count.
//    - exl3_gemv_kernel: 512 threads for configuration 0, 256 for 1, no
//      dynamic shared memory, on (min(n / columns, co-resident blocks), 1,
//      1) (exl3_gemv.cu, exl3_gemv_try_launch).
//    - All three call cooperative_groups' grid sync, which traps
//      (_CG_ABORT, __trap) when the launch was not cooperative, and a trap
//      loses the CUDA context. exl3_gemm_kernel always reaches one (lines
//      29 and 49), exl3_gemv_kernel always two (201 and 434).
//      exl3_mgemm_kernel reaches one only with min_index >= 0 (141),
//      had_src_list (167) or B_weights (292), and per matrix only below
//      sm_90 (210, 247).
//    - Their other waits spin on global memory and never trap: the split-K
//      locks of exl3_gemm_kernel_inner (barrier_acquire, ptx.cuh) and, from
//      sm_90 on, exl3_mgemm_kernel's group_barrier (ptx.cuh). A grid whose
//      blocks are not all resident at once hangs there. On sm_121 an
//      exl3_mgemm_kernel launch without min_index, had_src_list or
//      B_weights reaches no grid sync, so no trap reveals a launch that was
//      not cooperative: it hangs if its grid is too large, which only a
//      cooperative launch refuses up front
//      (cudaErrorCooperativeLaunchTooLarge).
//
// 2. Give each launch lock slots no concurrently running launch uses.
//    `locks` is the device context's int buffer (quant/exl3_devctx.cuh:
//    MAX_TILES_C split-K locks, then 2 * MAX_BARRIERS barrier counters and
//    sense words at BARRIER_LOCKS_OFFSET), zeroed before first use, as
//    upstream's DevCtx::get_locks does with cudaMemset. A launch that
//    completes returns its locks and counters to zero (sense words flip,
//    which is harmless); one that fails or traps leaves them undetermined,
//    so the buffer is zeroed again before reuse. exl3_gemv_kernel takes
//    the argument and never reads it. exl3_mgemm_kernel also
//    keeps its selection (v_indices, v_weights, bszm_sync, at most
//    MAX_INDICES = 128 entries) in __device__ variables of its compilation
//    unit, shared by every launch of that unit's kernels on the device, so
//    two of its launches with min_index >= 0 must not overlap either; it
//    touches them only then (lines 101-145).
//
// Ordinary kernels, launched on a normal grid and waiting on no other
// block: reconstruct_kernel and reconstruct_had_kernel (quant/
// reconstruct.cu), had_hf_r_128_kernel and had_ff_r_128_kernel
// (quant/hadamard.cu) and add_kernel_hhh (add.cu), the ones llmpalooza's
// linear launches (launch.h), with upstream's grids and block sizes. The
// kept files also define kernels llmpalooza never launches, and instantiates
// only where a kept file does so itself: reconstruct_batch_kernel,
// reconstruct_had_batch_kernel, had_hf_r_128_dual_kernel,
// had_ff_r_128_dual_kernel, had_hf_r_128_batch_kernel,
// had_ff_r_128_batch_kernel, copy2d_kernel, moe_bias_add_kernel,
// moe_bias_add_weighted_kernel and add.cu's other seven adds
// (add_kernel_hhf, add_kernel_hfh, add_kernel_hff, add_kernel_fhh,
// add_kernel_fhf, add_kernel_ffh, add_kernel_fff).

#ifndef LLMP_KERNELS_EXL3_LAUNCH_CONTRACT_H_
#define LLMP_KERNELS_EXL3_LAUNCH_CONTRACT_H_

#endif  // LLMP_KERNELS_EXL3_LAUNCH_CONTRACT_H_
