// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGUF container reader (D-109): a llama.cpp checkpoint, one file or
// split into several, as a checkpoint (checkpoint.h).
//
// The header is base/gguf.h's, read from the file's start in growing
// prefixes (kGgufFirstRead, then four times as much). The headers of all
// of a checkpoint's files take at most kMaxGgufHeaderBytes, the metadata
// kept from them at most kMaxGgufMetadataBytes, and their tensors at most
// kMaxTensors, each charged as a file is read, so a split set cannot add up
// past what one file may hold. Beyond its structure the reader refuses
// only what it could not describe correctly, and so is more permissive
// than pinned gguf.cpp where a file is still unambiguous:
//
//   - general.architecture UTF-8 text of 1 to 200 bytes without control
//     characters; general.alignment, when present, a power of two up to
//     2^31 of any integer type (gguf.cpp writes a u32; 32 otherwise);
//   - tensor names of 1 to kMaxNameBytes bytes without control characters
//     (gguf.cpp stops at 63);
//   - every tensor's GGML type one the format layer names (format::Info's
//     GgmlBlock, which every type of ggml.h at the pin has; a removed or
//     unknown type is refused), its rows whole blocks, its bytes the size rule's
//     (format::PartsOf, checked arithmetic);
//   - zero-element tensors have checked offsets and known types, but are
//     recorded in skipped_tensors rather than the executable tensor inventory;
//   - every tensor's bytes, at its offset into the data section (which
//     starts at the header's end padded to the alignment), inside the file
//     and overlapping no other tensor's. gguf.cpp also packs them in info
//     order at the alignment; a file placed otherwise, or with bytes after
//     its last tensor (which gguf.cpp ignores), is read as it is;
//   - a split GGUF (split.count above 1, as gguf-split writes it: split.no
//     and split.count u16, split.tensors.count i32, though any integer type
//     is read) is read from its first
//     file, named <prefix>-00001-of-<count>.gguf, with the others beside it
//     by the same pattern: each file's split.no its position and its count
//     the first's, every tensor name once across them, their tensors the
//     first file's split.tensors.count in all when it gives one (llama.cpp
//     does not need it), and any key a later file repeats equal to the
//     first's (the split.* keys excepted; the first's are kept).
//
// A three-dimensional tensor whose name has a component ending in `_exps`
// (llama.cpp's routed-expert tensors, ffn_gate_exps and the like) is an
// expert array. The metadata holds every key; a GGUF bool array becomes a
// list of 0 and 1, and a u64 above INT64_MAX none.

#ifndef LLMP_IMPORT_READERS_GGUF_H_
#define LLMP_IMPORT_READERS_GGUF_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>

#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

// What one checkpoint's files may hold in all: header bytes (the first
// file's metadata holds a vocabulary), and the metadata kept, costed at
// its bytes plus 64 a key, 32 a string and 8 a number.
inline constexpr std::uint64_t kMaxGgufHeaderBytes = std::uint64_t{256} << 20U;
inline constexpr std::uint64_t kMaxGgufMetadataBytes = std::uint64_t{512} << 20U;
inline constexpr std::uint64_t kMaxGgufSplits = 4096;
// The first prefix read; each further one is four times as long.
inline constexpr std::uint64_t kGgufFirstRead = std::uint64_t{1} << 20U;
// A string value's or array's cap (a vocabulary's tokens fit).
inline constexpr std::uint64_t kMaxGgufString = std::uint64_t{16} << 20U;
inline constexpr std::uint64_t kMaxGgufArray = std::uint64_t{1} << 24U;
inline constexpr std::int64_t kMaxGgufAlignment = std::int64_t{1} << 31U;

// The GGUF at `path`, and its other splits beside it.
std::expected<Checkpoint, Error> ReadGguf(const std::filesystem::path& path);

// One whole GGUF file's bytes (no split of several files), as ReadGguf
// reads a file: for tests and fuzzing. The checkpoint has one file, named
// "memory", and no path (its reads fail).
std::expected<Checkpoint, Error> ParseGguf(std::span<const std::byte> file);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_GGUF_H_
