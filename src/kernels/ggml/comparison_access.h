// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_GGML_COMPARISON_ACCESS_H_
#define LLMP_KERNELS_GGML_COMPARISON_ACCESS_H_
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include "kernels/ggml/plan_step.h"
namespace llmp::kernels::ggml {
enum class ComparisonAccessScope { kWhole, kLocal };
struct ComparisonExplicitAccess {
  const ggml_tensor* tensor = nullptr;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  bool read = false;
  bool written = false;
  std::uint8_t phase = 0;
};
struct ComparisonExplicitAccesses {
  std::array<ComparisonExplicitAccess, 10> values{};
  std::size_t count = 0;
  std::array<const ggml_tensor*, 2> elided{};
  std::size_t elided_count = 0;
  std::optional<std::uint32_t> dispatch;
};
bool HasComparisonAccessRecipe(std::string_view name);
std::optional<ComparisonExplicitAccesses> DescribeComparisonAccess(const PlanStep& step,
                                                                   ComparisonAccessScope scope);
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_COMPARISON_ACCESS_H_
