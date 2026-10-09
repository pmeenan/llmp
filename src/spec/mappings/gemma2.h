// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 2's architecture mapping (spec/mapping.h), as llama.cpp's GGUF
// writes the checkpoint: sandwich-normed GQA attention with a logit
// softcap, alternating sliding-window (even) and global (odd) layers,
// NEOX rotation, a GELU-tanh GLU feed-forward, an embedding scaled by
// sqrt(width) and a tied output head with a final softcap; the GGUF
// converter's (1 + w) norm fold. Queries are scaled by 1 / sqrt(key
// length), llama.cpp's rule below 27B: a 46-layer (27B) checkpoint's 1 /
// sqrt(width / heads) is not expressible yet, and the importer must not use
// this mapping for one. And the source values of a v0 artifact from
// today's compiled profile (model/gemma2.h).

#ifndef LLMP_SPEC_MAPPINGS_GEMMA2_H_
#define LLMP_SPEC_MAPPINGS_GEMMA2_H_

#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"

namespace llmp::model {
struct Gemma2Profile;
}

namespace llmp::spec::mappings {

const Mapping& Gemma2Mapping();

// The profile as the GGUF keys the mapping reads. Refused where the
// profile says what the mapping cannot (an attention scale other than
// 1 / sqrt(key length), a rotation scaling, other local layers).
std::expected<std::vector<SourceValue>, std::string> Gemma2V0Values(
    const model::Gemma2Profile& profile);

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_GEMMA2_H_
