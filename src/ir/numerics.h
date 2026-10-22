// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_IR_NUMERICS_H_
#define LLMP_IR_NUMERICS_H_

#include <cstdint>
#include <string_view>

namespace llmp::ir {

// Arithmetic changes whose same-quality evidence is distinct from structural
// fusion or sharing an incumbent's already mandatory conversion. The names
// describe rounding/reduction boundaries, not matchers or model families.
enum class NumericalChange : std::uint8_t {
  kBf16SwiGluIntermediates,
  kBf16ProductInputAndResult,
  kBf16AttentionIntermediates,
  kBf16ResidualCarry,
  kBf16HyperConnectionIntermediates,
  kMxfp8ProductInput,
  kQ8VectorReduction,
  kNvfp4ExpertIntermediates,
};
std::string_view NumericalChangeName(NumericalChange change);

}  // namespace llmp::ir

#endif  // LLMP_IR_NUMERICS_H_
