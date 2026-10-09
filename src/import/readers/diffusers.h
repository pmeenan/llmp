// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The diffusers pipeline reader (D-089, D-109): a pipeline directory,
// model_index.json beside a directory per component, as a checkpoint of
// container "diffusers" whose components are checkpoints of their own
// (checkpoint.h), the source of a D-089 composition.
//
// model_index.json (at most kMaxModelIndexBytes, strict JSON as
// config.json) is an object with a `_class_name` string of 1 to 200 bytes
// without control characters, the pipeline's architecture; all its members
// are the pipeline's metadata, flattened as config.json's. A member whose
// name does not start with '_' and whose value is a list of two strings
// ([library, class]) names a component (a list holding null, such as
// [null, null], names none); its name is its role and its directory's
// name, and must be [a-z][a-z0-9_]{0,63} (a path component, and a
// composition's role). At most kMaxPipelineEntries components.
//
// A component's directory must exist, be trusted and, its links resolved,
// lie beneath the pipeline's (a link out of the pipeline is an escape and
// refused; files inside it may link anywhere trusted, as a Hugging Face
// cache's snapshot files link to its blobs). An error from inside a
// component carries its role's index (Error::component). With a config.json
// it holds weights and is read by the reader that claims it (reader.h;
// never as a pipeline itself): a transformers model's model.safetensors
// or a diffusers model's diffusion_pytorch_model.safetensors, sharded or
// not, or their one variant ("fp16": readers/safetensors.h), its
// architecture its model_type or _class_name. One to
// kMaxPipelineComponents such components. Without a config.json (a
// scheduler's, a tokenizer's or a processor's), its regular files, read
// from at most kMaxPipelineDirectoryEntries entries (subdirectories and
// what is not a regular file are left out), are the pipeline's own files,
// named "<role>/<file name>" (UTF-8 without control characters), at most
// kMaxPipelineFiles of them (each stays open, as every reader's files
// do), each at most kMaxPipelineFileBytes and all at most
// kMaxPipelineFilesBytes (a composition's caps for the metadata it keeps
// verbatim). The pipeline's files are model_index.json, then those, by
// role and name.

#ifndef LLMP_IMPORT_READERS_DIFFUSERS_H_
#define LLMP_IMPORT_READERS_DIFFUSERS_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

inline constexpr std::uint64_t kMaxModelIndexBytes = std::uint64_t{1} << 20U;
inline constexpr std::size_t kMaxPipelineEntries = 64;
inline constexpr std::size_t kMaxPipelineComponents = 16;
inline constexpr std::size_t kMaxPipelineDirectoryEntries = 4096;
inline constexpr std::size_t kMaxPipelineFiles = 256;  // kept files in all, each open
inline constexpr std::uint64_t kMaxPipelineFileBytes = std::uint64_t{64} << 20U;
inline constexpr std::uint64_t kMaxPipelineFilesBytes = std::uint64_t{128} << 20U;

// What model_index.json says: the pipeline's architecture, its metadata,
// and its components' roles, sorted.
struct ModelIndex {
  std::string architecture{};
  std::vector<MetaEntry> metadata{};
  std::vector<std::string> roles{};
};
std::expected<ModelIndex, Error> ParseModelIndex(std::string_view text);

std::expected<Checkpoint, Error> ReadDiffusers(const std::filesystem::path& directory,
                                               uid_t trusted);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READERS_DIFFUSERS_H_
