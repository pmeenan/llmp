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
// expressible yet, and the importer must not use this mapping for one. And
// the source values of a v0 artifact from today's compiled profile
// (model/gemma3.h).

#ifndef LLMP_SPEC_MAPPINGS_GEMMA3_H_
#define LLMP_SPEC_MAPPINGS_GEMMA3_H_

#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"

namespace llmp::model {
struct Gemma3Profile;
}

namespace llmp::spec::mappings {

const Mapping& Gemma3Mapping();

// The profile as the GGUF keys the mapping reads. Refused where the
// profile says what the mapping cannot (a scaling other than linear, other
// local layers).
std::expected<std::vector<SourceValue>, std::string> Gemma3V0Values(
    const model::Gemma3Profile& profile);

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_GEMMA3_H_
