// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_KERNELS_GGML_SET_ROWS_GROUP_H_
#define JITLLM_KERNELS_GGML_SET_ROWS_GROUP_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string_view>

#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {
class LaunchContext;
inline constexpr std::size_t kSetRowsGroupMax = 16;
inline constexpr std::string_view kSetRowsGroupedName = "jitllm.set_rows.grouped";

// Two through sixteen independently addressed ordinary F32 -> F16/I64 stores.
// Every complete destination span must be disjoint from every other destination
// and every source/index span. Indices retain the primitive's device-payload
// contract; this check never reads them. Producers, keeps and all original nodes
// remain live. The planner groups only consecutive launch-bearing stores.
// No device scratch or host staging: bounded descriptors are copied into CUDA
// launch arguments (also by graph capture). Operands remain funded and leased
// through completion/replay exactly as for the individual primitive stores.
// Single-store eligibility, including checked U32 element-count arithmetic
// before the inherited primitive check evaluates its signed shape product.
std::expected<void, KernelFailure> CheckSetRowsForGrouping(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckSetRowsGroup(std::span<const ggml_tensor* const> nodes);
std::expected<void, KernelFailure> RunSetRowsGroup(LaunchContext& launch,
                                                   std::span<ggml_tensor* const> nodes);
}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_SET_ROWS_GROUP_H_
