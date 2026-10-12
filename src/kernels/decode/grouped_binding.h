// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Geometry only, not source/residency authority. Callers derive each part
// from owned source metadata and keep the selected experts' leases current.
#ifndef LLMP_KERNELS_DECODE_GROUPED_BINDING_H_
#define LLMP_KERNELS_DECODE_GROUPED_BINDING_H_

#include <cstdint>
#include <expected>
#include <span>

#include "base/bytes.h"
#include "format/tensor_format.h"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {
struct PitchedExpertPart {
  format::PartRole role = format::PartRole::kData;
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
  std::uint64_t matrix_pitch = 0;
};
enum class GroupedBindingError : std::uint8_t { kFormat, kExtent, kParts, kAddress, kAlias };
struct GroupedBindingSize {
  std::uint64_t experts = 0;
  base::Bytes descriptors{};
};
std::expected<GroupedBindingSize, GroupedBindingError> GroupedBindingBound(
    const format::TensorFormat& format);
// Every expert's descriptor is initialized, including empty expert intervals.
// Entire geometry is checked before changing into. No payload is read and no
// allocation is made on success. Single-expert format retains stacked and all
// encoding parameters; per-part matrix_pitch, not packed share size, locates e.
// Parts and format must remain immutable through this synchronous call.
std::expected<void, GroupedBindingError> BindGroupedInto(const format::TensorFormat& format,
                                                         std::span<const PitchedExpertPart> parts,
                                                         std::span<Operand> into);
}  // namespace llmp::kernels::decode
#endif  // LLMP_KERNELS_DECODE_GROUPED_BINDING_H_
