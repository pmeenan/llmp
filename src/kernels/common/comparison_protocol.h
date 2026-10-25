// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_COMMON_COMPARISON_PROTOCOL_H_
#define LLMP_KERNELS_COMMON_COMPARISON_PROTOCOL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "base/sha256.h"

namespace llmp::kernels {

// A vendor-neutral view of immutable producer-owned device facts. The caller
// must source these fields from its retained backend declaration, never from
// a comparison request or a device label supplied by the planner.
struct ComparisonDeviceFacts {
  std::string_view device_class{};
  std::string_view device_model{};
  std::uint32_t arch = 0;
  std::uint32_t features = 0;
  std::uint32_t multiprocessors = 0;
};

inline base::Sha256Digest ComparisonDeviceIdentity(ComparisonDeviceFacts facts) {
  base::Sha256 hash;
  hash.Update("llmp-comparison-device-v1");
  const auto append_u64 = [&](std::uint64_t value) {
    std::array<std::byte, sizeof(value)> encoded{};
    for (std::size_t i = 0; i < encoded.size(); ++i)
      encoded[encoded.size() - i - 1] = static_cast<std::byte>((value >> (i * 8U)) & 0xffU);
    hash.Update(encoded);
  };
  const auto append_u32 = [&](std::uint32_t value) {
    std::array<std::byte, sizeof(value)> encoded{};
    for (std::size_t i = 0; i < encoded.size(); ++i)
      encoded[encoded.size() - i - 1] = static_cast<std::byte>((value >> (i * 8U)) & 0xffU);
    hash.Update(encoded);
  };
  const auto append_string = [&](std::string_view value) {
    append_u64(static_cast<std::uint64_t>(value.size()));
    hash.Update(value);
  };
  append_string(facts.device_class);
  append_string(facts.device_model);
  append_u32(facts.arch);
  append_u32(facts.features);
  append_u32(facts.multiprocessors);
  return hash.Finish();
}

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_COMPARISON_PROTOCOL_H_
