// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen2's approved FP16 GGUF and EXL3 fixtures as shared components: the
// mapping their stored specs were made by (the EXL3 arithmetic a flag,
// llmp.qwen2.exl3), and the importer's closed normalizations of the GGUF
// and the EXL3 checkpoints.
#ifndef LLMP_SPEC_MAPPINGS_QWEN2_H_
#define LLMP_SPEC_MAPPINGS_QWEN2_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {
const Mapping& Qwen2Mapping();
// Closed, byte-preserving GGUF target normalization.
const Mapping& Qwen2GgufMapping();
// Closed, multipart-copy EXL3 target normalization.
const Mapping& Qwen2Exl3Mapping();
}  // namespace llmp::spec::mappings
#endif  // LLMP_SPEC_MAPPINGS_QWEN2_H_
