// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Generic pre-norm dense decoder mappings (Qwen2, Llama, Mistral) over shared
// components: `embedding.token`, `attention.dense` (GQA, RoPE with optional
// `rope_freqs` or YaRN), `ffn.glu` (`silu`), `norm.rms`, and `head.linear`
// (tied or untied).
#ifndef LLMP_SPEC_MAPPINGS_QWEN2_H_
#define LLMP_SPEC_MAPPINGS_QWEN2_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {
const Mapping& Qwen2Mapping();
// Container-named accessors remain aliases of the canonical mapping while
// source selection is expressed by its dialect registry.
const Mapping& Qwen2GgufMapping();
const Mapping& Qwen2Exl3Mapping();
const Mapping& Qwen2SafetensorsMapping();
const Mapping& Qwen35Mapping();

const Mapping& LlamaMapping();
// Legacy container-named accessors return the same canonical mapping.
const Mapping& LlamaGgufMapping();
const Mapping& LlamaExl3Mapping();
const Mapping& LlamaSafetensorsMapping();

const Mapping& MistralMapping();
const Mapping& MistralGgufMapping();
const Mapping& MistralExl3Mapping();
const Mapping& MistralSafetensorsMapping();
}  // namespace llmp::spec::mappings
#endif  // LLMP_SPEC_MAPPINGS_QWEN2_H_
