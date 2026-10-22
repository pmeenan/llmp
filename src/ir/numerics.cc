// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/numerics.h"

namespace llmp::ir {

std::string_view NumericalChangeName(NumericalChange change) {
  switch (change) {
    case NumericalChange::kBf16SwiGluIntermediates:
      return "bf16_swiglu_intermediates";
    case NumericalChange::kBf16ProductInputAndResult:
      return "bf16_product_input_and_result";
    case NumericalChange::kBf16AttentionIntermediates:
      return "bf16_attention_intermediates";
    case NumericalChange::kBf16ResidualCarry:
      return "bf16_residual_carry";
    case NumericalChange::kBf16HyperConnectionIntermediates:
      return "bf16_hyper_connection_intermediates";
    case NumericalChange::kMxfp8ProductInput:
      return "mxfp8_product_input";
    case NumericalChange::kQ8VectorReduction:
      return "q8_vector_reduction";
    case NumericalChange::kNvfp4ExpertIntermediates:
      return "nvfp4_expert_intermediates";
  }
  return {};
}

}  // namespace llmp::ir
