// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_DSV4_SCORE_ORIGINAL_H_
#define LLMP_TESTS_SUPPORT_DSV4_SCORE_ORIGINAL_H_
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
namespace llmp::kernels::ggml {
class LaunchContext;
}
namespace llmp::test_support {
// The genuine legacy Lid node supplies operands and native validation. Output
// is an independently owned packed F32[C,T] observation, disjoint from operands.
std::expected<void, kernels::ggml::KernelFailure> OriginalLidScores(
    kernels::ggml::LaunchContext& launch, ggml_tensor* lid, ggml_tensor* output);
}  // namespace llmp::test_support
#endif  // LLMP_TESTS_SUPPORT_DSV4_SCORE_ORIGINAL_H_
