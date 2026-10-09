// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A Hugging Face checkpoint's quantization (D-108, D-109): recognized from
// config.json's `quantization_config` and ModelOpt's hf_quant_config.json,
// and applied to the safetensors reader's tensors by pairing each quantized
// weight's parts into one tensor format. Recognized and described, never
// relaid out: each part keeps its own bytes.
//
// Described: NVIDIA ModelOpt's NVFP4 (quant_algo NVFP4 or W4A16_NVFP4,
// group 16) and MXFP8 (MXFP8, group 32), alone or per module
// (MIXED_PRECISION with `quantized_layers`, as Qwen3.8 Flash Next's native
// checkpoint has). A weight `M.weight` with block scales `M.weight_scale`
// is quantized:
//
//   - NVFP4: M.weight U8 [N, K / 2] (two E2M1 codes a byte, value 2i in the
//     low nibble), M.weight_scale F8_E4M3 [N, K / 16], and when present
//     M.weight_scale_2 F32 with one value: the format {nvfp4, kSplit,
//     [K, N]} with a tensor scale, its parts kCodes, kBlockScales and
//     kTensorScale. Without M.weight_scale_2, the nearest ancestor module
//     A with A.weight_scale_2 and no A.weight_scale gives the tensor scale
//     (a table stored as row shards under one scale, as Qwen3.8's n-gram
//     table): that tensor is the kTensorScale part of each such weight,
//     whose formats must then be equal;
//   - MXFP8: M.weight F8_E4M3 [N, K], M.weight_scale U8 or F8_E8M0
//     [N, K / 32]: {mxfp8, kSplit, [K, N]}, parts kCodes and kBlockScales.
//
// The scheme is the module's in `quantized_layers` (the longest module
// name that is M or an ancestor of it), or the single algorithm's unless an
// exclusion pattern (`ignore`, `exclude_modules`; '*' and '?' wildcards)
// names M (at most 4,096 patterns of at most 1,024 bytes, all matching of
// a checkpoint within a fixed budget of steps). A scaled weight no scheme
// covers (excluded, or in no configured module) takes the scheme its
// dtypes and shapes match under ModelOpt's conventions (Qwen3.8's n-gram
// table, which the config leaves out); one that matches neither is
// refused, since its scales could not be described. A configured scheme
// whose dtypes or part sizes the pair does not have is refused too. Every
// other tensor (activation scales `input_scale`, a `weight_scale` with no
// weight beside it, a tensor scale no weight takes, a weight without block
// scales even in a configured module) stays dense: the reader refuses
// only what it cannot describe correctly.
//
// Recognized and refused, naming why: compressed-tensors other than
// `dense` (its NVFP4 global scale is the reciprocal of NVFP4's tensor
// scale, so describing it needs a fold), AWQ and GPTQ (affine with zero
// points: no encoding yet), block FP8 (`fp8`), ModelOpt algorithms other
// than the three, MLX's `quantization` object (the MLX reader's), and any
// other method. With no quantization config every tensor is dense.

#ifndef LLMP_IMPORT_READERS_HF_QUANT_H_
#define LLMP_IMPORT_READERS_HF_QUANT_H_

#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "base/json.h"
#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

enum class QuantScheme : std::uint8_t { kNvfp4, kMxfp8 };

struct Quantization {
  std::string method{};                 // "modelopt"
  std::string algorithm{};              // "NVFP4", "W4A16_NVFP4", "MXFP8" or "MIXED_PRECISION"
  std::optional<QuantScheme> scheme{};  // a single algorithm's
  std::map<std::string, QuantScheme> layers{};  // MIXED_PRECISION's modules
  std::vector<std::string> exclude{};           // a single algorithm's exclusions
  bool operator==(const Quantization&) const = default;
};

// The quantization config.json's root (`config`) and hf_quant_config.json
// (`hf`, when present) describe: none, or the one both agree on (kConfig
// when they differ); refused as above.
std::expected<std::optional<Quantization>, Error> RecognizeQuantization(
    base::json::Value config, const base::json::Value* hf);

// Pairs the tensors' parts as `quantization` says (formats and parts set
// in place).
std::expected<void, Error> ApplyQuantization(const std::optional<Quantization>& quantization,
                                             std::vector<Tensor>& tensors);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_HF_QUANT_H_
