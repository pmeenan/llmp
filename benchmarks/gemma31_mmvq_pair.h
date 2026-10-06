// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Private operator experiment; no registry, runner or production option.
#ifndef JITLLM_BENCHMARKS_GEMMA31_MMVQ_PAIR_H_
#define JITLLM_BENCHMARKS_GEMMA31_MMVQ_PAIR_H_

#include <cstdint>
#include <expected>

#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::benchmarks::gemma31_mmvq {
std::expected<std::uint64_t, kernels::ggml::KernelFailure> PlanOrdinaryC2Pair(
    const kernels::ggml::LaunchContext& launch, const ggml_tensor* gate, const ggml_tensor* up);
std::expected<void, kernels::ggml::KernelFailure> OrdinaryC2Pair(
    kernels::ggml::LaunchContext& launch, ggml_tensor* gate, ggml_tensor* up);
}  // namespace jitllm::benchmarks::gemma31_mmvq
#endif
