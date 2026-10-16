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
// this mapping for one.

#ifndef LLMP_SPEC_MAPPINGS_GEMMA2_H_
#define LLMP_SPEC_MAPPINGS_GEMMA2_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {

const Mapping& Gemma2Mapping();

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_GEMMA2_H_
