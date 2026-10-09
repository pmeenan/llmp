// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The safetensors container reader (D-109): a Hugging Face checkpoint
// directory, config.json with model.safetensors or the shards
// model.safetensors.index.json names, as a checkpoint (checkpoint.h).
//
// config.json (at most kMaxConfigBytes, strict RFC 8259 JSON: well-formed
// UTF-8, duplicate keys refused, nesting at most 32) is an object with a
// `model_type` string of 1 to 200 bytes without control characters, the
// architecture. Its members become the metadata, objects flattened into
// dotted keys ("text_config.num_hidden_layers"): numbers as integers
// (written without fraction or exponent and within 64 bits) or finite
// reals, strings, flags, lists of integers, of numbers or of strings, and
// none for null and any other list. Two members flattening to one key are
// refused.
// `quantization_config` is not metadata: hf_quant.h reads it, and
// hf_quant_config.json beside it.
//
// The index (at most kMaxIndexBytes) holds `weight_map`, tensor names to
// shard file names; its other members are not read. A shard name is a
// plain file name ending in ".safetensors" (no '/', not "." or ".."), at
// most kMaxShards of them, each opened beneath the checkpoint. Every
// tensor a shard holds is the index's under that shard and every tensor
// the index names is in its shard. With no index, model.safetensors is
// the only shard; beside an index that does not name it, the checkpoint
// is refused (which of the two is meant is ambiguous).
//
// A shard is safetensors' format, read as permissively as still describes
// it correctly: an 8-byte little-endian header length of at most
// kMaxHeaderBytes, inside the file; the header a JSON object of tensors
// and an optional `__metadata__`, which is not read. A tensor is an object
// with `dtype`, `shape` and `data_offsets` (other members ignored): a
// dtype of the table below, at most kMaxRank non-negative integer
// dimensions whose product (no element count zero: the format layer has no
// empty shape) times the dtype's size is its bytes, and [begin, end] with
// end - begin those bytes, inside the data section and overlapping no
// other tensor's. (Safetensors' own reader also wants the tensors to tile
// the data section with no gap or byte left over, which this reader does
// not need.) Names are 1 to kMaxNameBytes bytes without control
// characters.
//
// Dtypes and their encodings: BOOL, U8, I8, I16, U16, I32, U32, I64, U64,
// F16, BF16, F32, F64, F8_E4M3, F8_E5M2 and F8_E8M0 (E8M0); the 4- and
// 6-bit dtypes and complex ones are refused. A dense tensor's shape is its
// dimensions reversed into GGML's order, those past the fourth folded into
// ne[3] (as v0's plain descriptors). A three-dimensional tensor with a
// name component "experts" is an expert array (fused experts).

#ifndef LLMP_IMPORT_READERS_SAFETENSORS_H_
#define LLMP_IMPORT_READERS_SAFETENSORS_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>

#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

inline constexpr std::uint64_t kMaxHeaderBytes = 100'000'000;  // safetensors' own limit
inline constexpr std::uint64_t kMaxConfigBytes = std::uint64_t{16} << 20U;
inline constexpr std::uint64_t kMaxIndexBytes = std::uint64_t{64} << 20U;
inline constexpr std::size_t kMaxShards = 4096;
inline constexpr std::size_t kMaxRank = 8;
inline constexpr std::size_t kMaxConfigDepth = 32;

std::expected<Checkpoint, Error> ReadSafetensors(const std::filesystem::path& directory,
                                                 uid_t trusted);

// One shard's bytes with its config.json (and hf_quant_config.json, when
// not empty), as ReadSafetensors reads a checkpoint of one shard: for
// tests and fuzzing. The checkpoint's file is named "memory" and has no
// descriptor.
std::expected<Checkpoint, Error> ParseSafetensors(std::span<const std::byte> shard,
                                                  std::string_view config,
                                                  std::string_view hf_quant_config = {});

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_SAFETENSORS_H_
