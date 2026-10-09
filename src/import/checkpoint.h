// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A checkpoint as a container reader normalizes it (D-108, D-109;
// docs/engine-components.md#the-importer): its files, opened and checked
// (files.h); every tensor as a tensor format (an encoding, its parameters,
// a layout and a logical shape in GGML's order), the part of that format
// it holds and the byte extent that holds it; and the source metadata as
// typed key/values. Where a quantization pairs several container tensors
// into one logical tensor (a ModelOpt NVFP4 weight's codes, block scales
// and tensor scale), each keeps its own extent and carries the pair's
// format with its own part, as spec::SourceTensor does. Nothing is
// relayed out or converted here: the bytes are the checkpoint's.
//
// SourceOf and ValuesOf give exactly what spec::Synthesize reads, so an
// architecture's mapping applies to a checkpoint as it applies to a v0
// artifact (spec/mapping.h).
//
// The output is deterministic: tensors in file order (a split GGUF's
// files in split order, a sharded checkpoint's shards by name) and, in a
// file, by offset; metadata by key.

#ifndef LLMP_IMPORT_CHECKPOINT_H_
#define LLMP_IMPORT_CHECKPOINT_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "import/error.h"
#include "import/files.h"
#include "spec/mapping.h"

namespace llmp::import {

// Caps of every reader.
inline constexpr std::size_t kMaxTensors = std::size_t{1} << 20U;
inline constexpr std::size_t kMaxMetadata = std::size_t{1} << 16U;
inline constexpr std::size_t kMaxNameBytes = 512;  // a tensor name
inline constexpr std::size_t kMaxKeyBytes = 1024;  // a metadata key
inline constexpr std::size_t kMaxArchitecture = 200;

// A metadata value: none (JSON null, or what the metadata cannot hold: a
// GGUF u64 above INT64_MAX, a JSON array of mixed kinds or of objects), an
// integer, a real, a flag, a string, or a list of integers, reals or
// strings. A GGUF bool array is a list of integers 0 and 1.
using MetaValue =
    std::variant<std::monostate, std::int64_t, double, bool, std::string, std::vector<std::int64_t>,
                 std::vector<double>, std::vector<std::string>>;

struct MetaEntry {
  std::string key{};
  MetaValue value{};
  bool operator==(const MetaEntry&) const = default;
};

// A file of the checkpoint: its name (relative to the checkpoint: a GGUF's
// file name, a shard's), and its size and identity when opened.
struct SourceFile {
  std::string name{};
  std::uint64_t bytes = 0;
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
};

struct Extent {
  std::uint32_t file = 0;
  std::uint64_t offset = 0;  // in the file
  std::uint64_t bytes = 0;
  bool operator==(const Extent&) const = default;
};

struct Tensor {
  std::string name{};
  format::TensorFormat format{};  // the logical tensor's
  format::PartRole part = format::PartRole::kData;
  bool experts = false;  // every expert's matrix as one tensor (ne[2] experts)
  Extent extent{};
  // As the container writes it: a GGML type's encoding name or a
  // safetensors dtype, and its dimensions in the container's order (GGUF's
  // innermost first, safetensors' outermost first).
  std::string_view dtype{};
  std::vector<std::uint64_t> dims{};
};

// A diffusers pipeline (container "diffusers"; readers/diffusers.h) is a
// checkpoint of components, each a checkpoint of its own with its role
// (D-089), and holds no tensors itself: its files are model_index.json
// and the files of its components that hold no weights (a scheduler's, a
// processor's), its metadata model_index.json's.
struct Checkpoint {
  // A pipeline component's role (model_index.json's name for it,
  // [a-z][a-z0-9_]{0,63}, also its directory); empty for any other.
  std::string role{};
  // "gguf", "safetensors", "mlx", "exl3" or "diffusers".
  std::string container{};
  // GGUF general.architecture; config.json model_type, or a diffusers
  // component's _class_name; a pipeline's model_index.json _class_name.
  std::string architecture{};
  // The recognized quantization ("modelopt NVFP4", "modelopt
  // MIXED_PRECISION", "mlx affine", "exl3"), or empty.
  std::string quantization{};
  std::vector<SourceFile> files{};
  std::vector<OpenFile> open{};  // files[i]'s descriptor
  std::vector<Tensor> tensors{};
  std::vector<MetaEntry> metadata{};     // sorted by key, keys unique
  std::vector<Checkpoint> components{};  // a pipeline's, sorted by role

  const Tensor* FindTensor(std::string_view name) const;
  const MetaValue* FindValue(std::string_view key) const;
  // Reads `out.size()` bytes of an extent's file at `offset` within the
  // extent (kBounds past it; kIo if the file shrank).
  std::expected<void, Error> Read(const Extent& extent, std::uint64_t offset,
                                  std::span<std::byte> out) const;
};

// Whether text holds no control character (below 0x20, 0x7F, or C1's
// U+0080 to U+009F in UTF-8): what a tensor name and an architecture must
// be, besides UTF-8.
bool NoControlCharacters(std::string_view text);

// The checkpoint as a mapping's source: every tensor's name, format, part
// and whether it is an expert array.
spec::Source SourceOf(const Checkpoint& checkpoint);

// The metadata a mapping can read: integers, reals, flags, strings and
// lists of integers or reals of at most spec::kMaxList values, under keys
// spec::IsSourceKey admits; everything else is left out.
std::vector<spec::SourceValue> ValuesOf(const Checkpoint& checkpoint);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_CHECKPOINT_H_
