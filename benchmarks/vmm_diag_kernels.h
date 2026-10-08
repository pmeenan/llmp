// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The microkernels of the host-VMM diagnosis (vmm_diag_bench.cc;
// docs/experiments/host-vmm-diagnosis/). Each queues one launch on `stream`
// over device-accessible addresses and returns the launch's error. None is
// part of llmpalooza: they are probes of how the GPU reads a kind of memory.

#ifndef LLMP_BENCHMARKS_VMM_DIAG_KERNELS_H_
#define LLMP_BENCHMARKS_VMM_DIAG_KERNELS_H_

#include <cuda_runtime.h>

#include <cstdint>

namespace llmp::diag {

// Fills `words` 32-bit words at `data` with a pattern of `seed`.
cudaError_t Fill(cudaStream_t stream, std::uint64_t data, std::uint64_t words, std::uint32_t seed);

// Fills `count` F16 values in [-0.1, 0.1), or F32 values in [-1, 1).
cudaError_t FillHalf(cudaStream_t stream, std::uint64_t data, std::uint64_t count,
                     std::uint32_t seed);
cudaError_t FillFloat(cudaStream_t stream, std::uint64_t data, std::uint64_t count,
                      std::uint32_t seed);

// The M0 I/O experiment's resident scan (docs/experiments/io-path/kernel.cu),
// launched as it was: 256 blocks of 256 threads, one 32-bit load per thread
// and step, a grid-stride loop, a per-thread sum to `out` (256 x 256 words).
cudaError_t ScanWords(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes,
                      std::uint64_t out);

// A streaming read with 16-byte loads over `blocks` blocks of 256 threads.
cudaError_t ScanVector(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes,
                       std::uint64_t out, int blocks);

// Writes every 16 bytes of `data` once.
cudaError_t WriteVector(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes, int blocks);

// Writes `count` 32-bit words, one at the start of each `stride`-byte line
// of `data` (a product's scattered per-row outputs).
cudaError_t WriteSparse(cudaStream_t stream, std::uint64_t data, std::uint64_t count,
                        std::uint64_t stride, int blocks);

// `count` blocks of 256 threads, in which only thread 0 writes: one 32-bit
// word per block at `data` + 4 x block (a matrix-vector product's output
// pattern, without its reads).
cudaError_t WritePerBlock(cudaStream_t stream, std::uint64_t data, std::uint64_t count);

// Reads `bytes` at `data` `passes` times with L1-bypassing (L2-cached)
// 16-byte loads; each pass starts each block at a different offset, so a
// block does not read back what it read before.
cudaError_t Reread(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes, int passes,
                   std::uint64_t out, int blocks);

// Each warp reads `lines_per_warp` 128-byte lines chosen pseudo-randomly
// over `bytes`. With `window` non-zero, a warp stays in one randomly chosen
// `window`-byte aligned region for `run` consecutive lines before it moves
// to another (translation locality with the same line pattern).
cudaError_t RandomLines(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes,
                        std::uint64_t window, int run, int lines_per_warp, std::uint64_t out,
                        int blocks);

// Copies `bytes` (a multiple of 16) from `from` to `to` with 16-byte loads
// and stores.
cudaError_t CopyVector(cudaStream_t stream, std::uint64_t to, std::uint64_t from,
                       std::uint64_t bytes, int blocks);

}  // namespace llmp::diag

#endif  // LLMP_BENCHMARKS_VMM_DIAG_KERNELS_H_
