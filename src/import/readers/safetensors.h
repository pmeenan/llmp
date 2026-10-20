// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The safetensors container reader (D-109): a Hugging Face checkpoint
// directory, config.json with model.safetensors or the shards
// model.safetensors.index.json names, as a checkpoint (checkpoint.h). A
// diffusers component's directory (readers/diffusers.h) is read alike,
// its weights named diffusion_pytorch_model instead of model.
//
// config.json (at most kMaxConfigBytes, strict RFC 8259 JSON: well-formed
// UTF-8, duplicate keys refused, nesting at most 32) is an object with a
// `model_type` string of 1 to 200 bytes without control characters, the
// architecture, or without one a diffusers `_class_name` of the same
// rule. Its members become the metadata, objects flattened into
// dotted keys ("text_config.num_hidden_layers"): numbers as integers
// (written without fraction or exponent and within 64 bits) or finite
// reals, strings, flags, lists of integers, of numbers or of strings, and
// JSON null, and none for any other list. A member named "" is refused; a
// member whose name holds a '.' is left out with its contents (its dotted
// key could not be told from a nested member's, and transformers reads it
// only under its literal name).
// `quantization_config` (and MLX's `quantization` object) is not
// metadata: hf_quant.h reads it, and hf_quant_config.json beside it (the
// MLX and EXL3 readers read theirs).
//
// The index (at most kMaxIndexBytes) holds `weight_map`, tensor names to
// shard file names; its other members are not read. A shard name is a
// plain file name ending in ".safetensors" (no '/', not "." or ".."), at
// most kMaxShards of them, each opened beneath the checkpoint (and kept
// open: ReadCheckpoint raises the soft open-file limit for them). Every
// tensor a shard holds is the index's under that shard and every tensor
// the index names is in its shard. With no index, model.safetensors is
// the only shard; beside an index that does not name it, the checkpoint
// is refused (which of the two is meant is ambiguous), as is a directory
// holding weights under both base names. With no weights under the
// default names, a variant's (transformers' and diffusers' `variant`:
// <base>.<v>.safetensors or <base>.safetensors.index.<v>.json, v of
// letters, digits and '_') are read when they are the directory's only
// weights, as the default's would be; several variants, or variants of
// both bases, are refused as ambiguous (at most 4 · kMaxShards entries
// are listed). Beside the default names a variant is not read, as a
// library loading the directory without naming one does not.
//
// A shard is safetensors' format, read as permissively as still describes
// it correctly: an 8-byte little-endian header length of at most
// kMaxHeaderBytes, inside the file; the header a JSON object of tensors
// and an optional `__metadata__`, free text but for its `format`: shards
// that MLX wrote ("mlx") make the checkpoint MLX's (readers/mlx.h's
// MlxCheckpoint), and shards in MLX's format beside shards another
// library wrote are refused. A tensor is an object
// with `dtype`, `shape` and `data_offsets` (other members ignored): a
// dtype of the table below, at most kMaxRank non-negative integer
// dimensions whose product times the dtype's size is its bytes, and [begin, end] with
// end - begin those bytes, inside the data section and overlapping no
// other tensor's. (Safetensors' own reader also wants the tensors to tile
// the data section with no gap or byte left over, which this reader does
// not need.) Names are 1 to kMaxNameBytes bytes without control
// characters. Zero-element tensors have checked offsets and known dtypes,
// but are recorded in skipped_tensors rather than the executable inventory;
// they still count toward the aggregate tensor cap and shard index coverage.
//
// Dtypes and their encodings: BOOL, U8, I8, I16, U16, I32, U32, I64, U64,
// F16, BF16, F32, F64, F8_E4M3, F8_E5M2 and F8_E8M0 (E8M0); the 4- and
// 6-bit dtypes and complex ones are refused. A dense tensor's shape is its
// dimensions reversed into GGML's order, those past the fourth folded into
// ne[3] (as v0's plain descriptors). A three-dimensional tensor with a
// name component "experts" is an expert array (fused experts).

#ifndef LLMP_IMPORT_READERS_SAFETENSORS_H_
#define LLMP_IMPORT_READERS_SAFETENSORS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

inline constexpr std::uint64_t kMaxHeaderBytes = 100'000'000;  // safetensors' own limit
inline constexpr std::uint64_t kMaxConfigBytes = std::uint64_t{16} << 20U;
inline constexpr std::uint64_t kMaxIndexBytes = std::uint64_t{64} << 20U;
inline constexpr std::size_t kMaxShards = 4096;
inline constexpr std::size_t kMaxRank = 8;
inline constexpr std::size_t kMaxConfigDepth = 32;

std::expected<Checkpoint, Error> ReadSafetensors(const std::filesystem::path& directory);

// One shard's bytes with its config.json (and hf_quant_config.json, when
// not empty), as ReadSafetensors reads a checkpoint of one shard: for
// tests and fuzzing. The checkpoint's file is named "memory" and has no
// path (its reads fail).
std::expected<Checkpoint, Error> ParseSafetensors(std::span<const std::byte> shard,
                                                  std::string_view config,
                                                  std::string_view hf_quant_config = {});

// ------------------------------------------------ what the other readers share

// A directory read as above with every tensor dense, as its dtype, before
// any quantization pairs parts: the MLX and EXL3 readers read theirs from
// `config` and pair the tensors themselves.
struct HfDirectory {
  Checkpoint checkpoint{};  // container "safetensors"
  base::json::Document config{};
  std::optional<base::json::Document> hf_quant{};  // hf_quant_config.json, when present
  // A shard's `__metadata__.format` is "mlx" (and none names another):
  // MLX wrote the arrays (readers/mlx.h).
  bool mlx = false;
};
std::expected<HfDirectory, Error> ReadHfDirectory(const std::filesystem::path& directory);
// The same of one shard's bytes (ParseSafetensors's file).
std::expected<HfDirectory, Error> ParseHfShard(std::span<const std::byte> shard,
                                               std::string_view config,
                                               std::string_view hf_quant_config = {});

// config.json beneath `directory`, opened and checked as Read opens it,
// within its cap and parsed as an object: a reader's claim's look, none
// for anything else (the reader that reads it reports why).
std::optional<base::json::Document> PeekConfig(const std::filesystem::path& directory);

// A JSON object's members as metadata, as config.json's (sorted, keys
// unique, within the caps); `skip` names root members left out.
std::expected<std::vector<MetaEntry>, Error> FlattenMetadata(
    base::json::Value root, std::span<const std::string_view> skip = {});

// Whether a dot-separated component of `name` is `component`.
bool HasComponent(std::string_view name, std::string_view component);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_SAFETENSORS_H_
