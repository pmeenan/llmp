// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The container readers (D-109; docs/engine-components.md#the-importer):
// one per container, each normalizing a checkpoint into a Checkpoint
// (checkpoint.h) of tensor formats, extents and source metadata, and
// nothing family-specific. A reader claims a path by its shape (a file
// starting with GGUF's magic; a directory holding config.json) and reads
// it as untrusted input (D-009): every path through the path-trust walk
// (files.h), every count, length and offset bounded and checked before it
// drives a read or an allocation, and every refusal an Error naming its
// rule and place.
//
// So far: GGUF (readers/gguf.h) and safetensors with config.json
// (readers/safetensors.h, with readers/hf_quant.h's quantizations).
// Diffusers pipelines, MLX and EXL3 add their readers here.

#ifndef LLMP_IMPORT_READER_H_
#define LLMP_IMPORT_READER_H_

#include <sys/types.h>

#include <expected>
#include <filesystem>
#include <span>
#include <string_view>

#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {

struct ContainerReader {
  std::string_view container{};  // the Checkpoint::container it produces
  // Whether the path is this container's (a cheap look at what it is,
  // trusting nothing: Read checks everything again).
  bool (*claims)(const std::filesystem::path& path) = nullptr;
  std::expected<Checkpoint, Error> (*read)(const std::filesystem::path& path,
                                           uid_t trusted) = nullptr;
};

std::span<const ContainerReader> Readers();

// The checkpoint at `path`, read by the first reader that claims it
// (kFileType when none does), trusting root and `trusted`.
std::expected<Checkpoint, Error> ReadCheckpoint(const std::filesystem::path& path, uid_t trusted);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READER_H_
