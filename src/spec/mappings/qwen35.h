// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.5's text mapping (qwen3_5, qwen3_5_text): three Gated DeltaNet
// layers (`attention.gdn`) to a gated full-attention one (`attention.dense`
// with QK norms and an output gate), a routed-expert block (`ffn.moe` with
// a gated shared expert) or a dense one (`ffn.glu`) by the checkpoint's
// expert count, zero-centered norms; from GGUF, safetensors and MLX
// checkpoints (the GGUF converter's folds asserted, the native importer's
// recorded).
#ifndef LLMP_SPEC_MAPPINGS_QWEN35_H_
#define LLMP_SPEC_MAPPINGS_QWEN35_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {
const Mapping& Qwen35Mapping();
}  // namespace llmp::spec::mappings
#endif  // LLMP_SPEC_MAPPINGS_QWEN35_H_
