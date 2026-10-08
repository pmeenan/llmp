// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The wake benchmark's kernels (wake_bench.cc; docs/experiments/runtime-wake/):
// a step that keeps the GPU busy for a set time and stamps its start and end
// with the GPU's global timer, and a ticker that publishes that timer to
// host memory, to relate it to the host's clock. Probes, not part of llmpalooza.

#ifndef LLMP_BENCHMARKS_WAKE_BENCH_KERNELS_H_
#define LLMP_BENCHMARKS_WAKE_BENCH_KERNELS_H_

#include <cuda_runtime.h>

#include <cstdint>

namespace llmp::wake {

// One thread spins until `ns` of the global timer have passed, then writes
// its start and end times to stamps[0] and stamps[1] (device-accessible,
// such as mapped host memory).
cudaError_t Step(cudaStream_t stream, std::uint64_t ns, std::uint64_t* stamps);

// One thread writes the global timer to *out, again and again, for `ns`.
cudaError_t Tick(cudaStream_t stream, std::uint64_t ns, std::uint64_t* out);

}  // namespace llmp::wake

#endif  // LLMP_BENCHMARKS_WAKE_BENCH_KERNELS_H_
