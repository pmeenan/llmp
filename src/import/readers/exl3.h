// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The EXL3 container reader (D-109): an ExLlamaV3 checkpoint, a Hugging
// Face-style directory (readers/safetensors.h: config.json, shards, index;
// container "exl3") whose config.json's `quantization_config` has
// quant_method "exl3", as a checkpoint (checkpoint.h) with each quantized
// linear's tensors paired into one trellis format (format::kExl3Trellis in
// kExl3Tiles). ExLlamaV3's quantization_config.json beside it repeats what
// the tensors say and is not read.
//
// A linear M is quantized when M.trellis is present, as ExLlamaV3's
// LinearEXL3 reads it (exllamav3/modules/quant/exl3.py at the pin):
//
//   - M.trellis: I16 [K / 16, N / 16, 16 · k], k bits a weight from 1 to
//     8 (ExLlamaV3's K = shape[-1] / 16; a last dimension that is not 16 ·
//     k, as a half-integer rate's, is refused: the format has none);
//   - M.suh: F16, K values; M.svh: F16, N values (any shape of that many).
//     Without them, ExLlamaV3's packed sign bits M.su and M.sv need an
//     unpacking fold the importer does not have yet: refused. Beside suh
//     or svh, a packed su or sv is not read (upstream prefers the unpacked)
//     and stays dense;
//   - the codebook: mcg when M.mcg is present, mul1 when M.mul1 is, 3inst
//     when neither (both: refused, ambiguous); the marker is the format's
//     kCodebookMarker part, 4 bytes of any dtype, whose value upstream does
//     not read.
//
// Every part carries the trellis's format, {exl3_trellis, k_bits = k,
// codebook, kExl3Tiles, [K, N]}, and its role. Everything else (the
// embeddings, norms, a linear's bias, side vectors or markers without a
// trellis) is dense as its dtype. A config.json that also has MLX's
// `quantization` object is refused (which of the two describes the tensors
// is ambiguous).

#ifndef LLMP_IMPORT_READERS_EXL3_H_
#define LLMP_IMPORT_READERS_EXL3_H_

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>

#include "base/json.h"
#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

// Whether config.json's root has quantization_config.quant_method "exl3".
bool IsExl3Config(base::json::Value root);

std::expected<Checkpoint, Error> ReadExl3(const std::filesystem::path& directory);

// One shard's bytes with its config.json, as ReadExl3 reads a checkpoint
// of one shard: for tests and fuzzing (ParseSafetensors's file).
std::expected<Checkpoint, Error> ParseExl3(std::span<const std::byte> shard,
                                           std::string_view config);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_EXL3_H_
