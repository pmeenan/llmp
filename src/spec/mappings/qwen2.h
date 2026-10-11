// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen2's approved FP16 GGUF and EXL3 fixtures as shared components.
// V0Values provides the compiled profile facts and the EXL3 arithmetic
// flag. Source normalization for native imports remains a separate step.
#ifndef LLMP_SPEC_MAPPINGS_QWEN2_H_
#define LLMP_SPEC_MAPPINGS_QWEN2_H_

#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"

namespace llmp::model {
struct Qwen2Profile;
}
namespace llmp::spec::mappings {
const Mapping& Qwen2Mapping();
std::expected<std::vector<SourceValue>, std::string> Qwen2V0Values(
    const model::Qwen2Profile& profile);
}  // namespace llmp::spec::mappings
#endif  // LLMP_SPEC_MAPPINGS_QWEN2_H_
