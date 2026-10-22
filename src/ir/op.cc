// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/op.h"

#include <string>
#include <string_view>

#include "ir/op_traits.h"

namespace llmp::ir {

std::string_view OpName(Op op) {
  const auto i = static_cast<std::size_t>(op);
  const OpTraits* traits = i < PrimitiveTraits().size() ? &PrimitiveTraits()[i] : nullptr;
  return traits ? traits->name : "unknown";
}

std::string_view FusedKindName(FusedKind kind) {
  const OpTraits* traits = Traits(Fused(kind));
  return traits ? traits->name : "unknown";
}

std::string OpKeyName(OpKey key) {
  if (key.op != Op::kFused) {
    return std::string(OpName(key.op));
  }
  return std::string(OpName(key.op)) + "." + std::string(FusedKindName(key.fused));
}

std::string_view ProductArithmeticName(ProductArithmetic arithmetic) {
  switch (arithmetic) {
    case ProductArithmetic::kImplementation:
      return "implementation";
    case ProductArithmetic::kBf16F32:
      return "bf16_rne_f32";
    case ProductArithmetic::kBlock32Q8SumF32:
      return "block32_q8_sum_f32";
  }
  return "unknown";
}

}  // namespace llmp::ir
