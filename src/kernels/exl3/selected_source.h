// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_EXL3_SELECTED_SOURCE_H_
#define LLMP_KERNELS_EXL3_SELECTED_SOURCE_H_
#include <array>
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "kernels/exl3/packed_tuning.h"
#include "kernels/ggml/selected_operation.h"
namespace llmp::kernels::exl3 {
// Occupancy for output F16/F32, row case one/multiple, narrow/wide config.
// Queried without allocating payloads or submitting work. It is rechecked
// when the source launch is made; declarations include all eight facts.
struct GemvCapacity {
  std::array<std::array<std::array<int, 2>, 2>, 2> blocks{};
  std::uint32_t arch = 121;
  int sms = 48;
  bool operator==(const GemvCapacity&) const = default;
};
std::expected<GemvCapacity, std::string> QueryGemvCapacity(int device);

// The approved frozen G/O profile's selected-operation adapter. It builds
// no model graph and owns no runner: shared IR lowering supplies each exact
// selected product/bias, physical parts, and global activation placement.
// Packed tuning is the authenticated cache, never a new autotuning run.
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> SelectedSource(
    FrozenCache cache, std::optional<GemvCapacity> gemv = {},
    std::optional<PackedCapacity> defaults = {}, bool generic = false);
// All existing frozen pin sets plus one distinct occupancy-derived packed source.
// No model identity selects a profile; capabilities prove each physical problem.
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> DeviceSelectedSource(
    int device);
}  // namespace llmp::kernels::exl3
#endif  // LLMP_KERNELS_EXL3_SELECTED_SOURCE_H_
