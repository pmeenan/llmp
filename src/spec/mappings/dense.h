// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Generic pre-norm dense decoder mappings (Qwen2, Llama, Mistral) over shared
// components: `embedding.token`, `attention.dense` (GQA, RoPE with optional
// `rope_freqs` or YaRN), `ffn.glu` (`silu`), `norm.rms`, and `head.linear`
// (tied or untied). One canonical mapping an architecture, its GGUF,
// safetensors and EXL3 checkpoints its source dialects.
#ifndef LLMP_SPEC_MAPPINGS_DENSE_H_
#define LLMP_SPEC_MAPPINGS_DENSE_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {
const Mapping& Qwen2Mapping();
const Mapping& LlamaMapping();
const Mapping& MistralMapping();
}  // namespace llmp::spec::mappings
#endif  // LLMP_SPEC_MAPPINGS_DENSE_H_
