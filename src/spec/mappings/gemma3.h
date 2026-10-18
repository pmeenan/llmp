// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma native HF and GGUF mappings. HF uses its explicit query scalar
// and source zero-centered norms; GGUF consumes attention.scale when given
// and the converter-applied norm.plus_one convention.

#ifndef LLMP_SPEC_MAPPINGS_GEMMA3_H_
#define LLMP_SPEC_MAPPINGS_GEMMA3_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {

const Mapping& Gemma3Mapping();
const Mapping& Gemma3SafetensorsMapping();
const Mapping& Gemma3ConditionalSafetensorsMapping();

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_GEMMA3_H_
