// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Why a checkpoint was refused (D-009, D-066): the rule it broke, a static
// reason, and where: which of its files (in the checkpoint's file order)
// and which item in it, a byte offset, a tensor, a metadata key or a shard,
// by index. As with the artifact reader, nothing from the checkpoint's own
// text (a name, a key, a value) is copied into an error, so a hostile name
// cannot reach a log or a terminal through one.

#ifndef LLMP_IMPORT_ERROR_H_
#define LLMP_IMPORT_ERROR_H_

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace llmp::import {

enum class Rule : std::uint8_t {
  kIo,            // a system call failed, or a file changed while it was read
  kUntrusted,     // a path others could change (platform/path_trust.h)
  kFileType,      // not a regular file or directory, or absent
  kContainer,     // the container's structure: truncation, types, offsets, alignment, overlap
  kBounds,        // beyond one of the reader's caps
  kUnsupported,   // a type, dtype or quantization the format layer cannot describe
  kShards,        // a split GGUF's or a sharded checkpoint's files disagree
  kConfig,        // config.json, the shard index or a quantization config is invalid
  kQuantization,  // a quantized tensor's parts do not pair as its scheme says
};

std::string_view RuleName(Rule rule);

inline constexpr std::uint64_t kNoItem = std::numeric_limits<std::uint64_t>::max();

enum class ItemKind : std::uint8_t { kNone, kByte, kTensor, kKey, kShard };

struct Error {
  Rule rule = Rule::kContainer;
  std::string_view reason{};     // static text
  std::uint64_t file = kNoItem;  // the file's index in the checkpoint's order
  ItemKind kind = ItemKind::kNone;
  std::uint64_t item = kNoItem;

  // "container: a tensor's data past the end of its file (file 1, tensor 17)".
  std::string ToString() const;
};

}  // namespace llmp::import

#endif  // LLMP_IMPORT_ERROR_H_
