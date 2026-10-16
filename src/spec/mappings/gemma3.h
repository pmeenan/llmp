// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 3's architecture mapping (spec/mapping.h), as llama.cpp's GGUF
// writes the checkpoint: Gemma 2's layer with RMS-normed queries and keys
// and no attention softcap (a final one only where the checkpoint has
// it), five sliding-window layers to a global one, the local layers
// rotating at the checkpoint's sliding-window base or llama.cpp's built-in
// 10,000 and the global ones at the checkpoint's base with its linear
// scaling. Queries are scaled by 1 / sqrt(key length), llama.cpp's rule
// below 27B: a 62-layer (27B) checkpoint's 1 / sqrt(width / heads) is not
// expressible yet, and the importer must not use this mapping for one.

#ifndef LLMP_SPEC_MAPPINGS_GEMMA3_H_
#define LLMP_SPEC_MAPPINGS_GEMMA3_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {

const Mapping& Gemma3Mapping();

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_GEMMA3_H_
