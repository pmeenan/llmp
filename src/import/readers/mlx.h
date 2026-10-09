// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The MLX container reader (D-109): a checkpoint mlx-lm wrote, a Hugging
// Face-style directory (readers/safetensors.h: config.json, shards, index;
// container "mlx") whose config.json has a `quantization` object, as a
// checkpoint (checkpoint.h) with each quantized weight's parts paired into
// one tensor format. An unquantized one (no `quantization`) is known by its
// shards' `__metadata__` {"format": "mlx"}, which mx.save_safetensors
// writes: the safetensors reader hands it here (MlxCheckpoint), every
// tensor dense, and refuses shards that name MLX's format and another's.
//
// The container matters beyond quantization: MLX keeps its own axis
// orders, which the reader describes as stored (dims as written, reversed
// into GGML's order like every safetensors tensor) and leaves to the
// architecture mapping's folds. A convolution's weight is channels-last,
// mlx.nn.Conv1d's [out, kernel, in] where PyTorch's is [out, in, kernel]
// (mlx-lm's sanitize moves the axes on conversion; a depthwise conv1d is
// [channels, kernel, 1]), and a mixture's experts are stacked into one
// [experts, ...] array under `switch_mlp` (an expert array here).
//
// `quantization` holds the default scheme, `group_size`, `bits` and
// `mode` ("affine" when absent; "mxfp4", "nvfp4", "mxfp8"), and per module
// path an override, as mlx-lm reads it: an object of the same members
// (group_size and bits required, since MLX's own defaults for them differ
// between its versions; mode "affine" when absent), `false` (not
// quantized) or `true` (the default). config.json's `quantization_config`
// (mlx-lm writes a copy there) is not read. A module M is quantized
// exactly when M.weight and M.scales are both present (mlx-lm's own test);
// its scheme is its override's, else the default's. Its tensors:
//
//   - M.weight, the codes: an unsigned integer dtype (MLX writes U32)
//     [.., N, C], each row K · bits / 8 bytes, codes in a little-endian bit
//     stream (layout.h kSplit; mx.quantize packs "from the lower to upper
//     bits", its 3-, 5- and 6-bit packs alike);
//   - M.scales [.., N, K / group]: affine F16, BF16 or F32; mxfp4 and mxfp8
//     E8M0 bytes (U8 or F8_E8M0), nvfp4 E4M3 bytes (U8 or F8_E4M3);
//   - affine only, M.biases: the scales' dtype and shape.
//
// K is the scales' columns times the group, which the codes' bytes must
// match: affine is {affine, bits, group, the scales' type, kSplit}, the
// others {mxfp4 | nvfp4 | mxfp8, kSplit} (their bits and group fixed: 4
// and 32, 4 and 16, 8 and 32), with shape [K, N] or, three-dimensional,
// [K, N, E] as an expert array (mlx-lm's switch_mlp). Every other tensor is
// dense as its dtype (a dense three-dimensional one under a `switch_mlp` or
// `experts` component an expert array); a stray M.scales or M.biases with
// no weight is dense too, and an fp mode's M.biases is left dense. Refused:
// a quantized module whose override says false or whose scheme is missing
// or incomplete, a mode MLX does not have, a group the affine encoding
// does not take (a multiple of 8), parts of other dtypes, ranks or sizes,
// and too many overrides (kMaxQuantizedLayers) or override keys past
// kMaxKeyBytes.

#ifndef LLMP_IMPORT_READERS_MLX_H_
#define LLMP_IMPORT_READERS_MLX_H_

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>

#include "base/json.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/readers/safetensors.h"

namespace llmp::import {

// Whether config.json's root has MLX's `quantization` object.
bool IsMlxConfig(base::json::Value root);

// A directory the safetensors reader read (every tensor dense, as its
// dtype) as MLX's: container "mlx", `quantization` applied when config.json
// has one, expert arrays marked.
std::expected<Checkpoint, Error> MlxCheckpoint(HfDirectory dir);

std::expected<Checkpoint, Error> ReadMlx(const std::filesystem::path& directory);

// One shard's bytes with its config.json, as ReadMlx reads a checkpoint
// of one shard: for tests and fuzzing (ParseSafetensors's file).
std::expected<Checkpoint, Error> ParseMlx(std::span<const std::byte> shard,
                                          std::string_view config);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_MLX_H_
