// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The container readers (D-109; docs/engine-components.md#the-importer):
// one per container, each normalizing a checkpoint into a Checkpoint
// (checkpoint.h) of tensor formats, extents and source metadata, and
// nothing family-specific. A reader claims a path by its shape and reads
// it as untrusted input (D-009): every path through the path-trust walk
// (files.h), every count, length and offset bounded and checked before it
// drives a read or an allocation, and every refusal an Error naming its
// rule and place.
//
// The readers, in the order they are asked: GGUF (readers/gguf.h; a file
// named *.gguf), a diffusers pipeline (readers/diffusers.h; a directory
// with model_index.json), EXL3 (readers/exl3.h) and MLX (readers/mlx.h)
// (a directory whose config.json, opened as the reader opens it, names
// their quantization), and any other directory as safetensors with
// config.json (readers/safetensors.h, with readers/hf_quant.h's
// quantizations).

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
  // Whether the path is this container's (a cheap look at what it is:
  // whatever it reads goes through the trust walk, and Read checks
  // everything again).
  bool (*claims)(const std::filesystem::path& path, uid_t trusted) = nullptr;
  std::expected<Checkpoint, Error> (*read)(const std::filesystem::path& path,
                                           uid_t trusted) = nullptr;
};

std::span<const ContainerReader> Readers();

// The checkpoint at `path`, read by the first reader that claims it
// (kFileType when none does), trusting root and `trusted`.
std::expected<Checkpoint, Error> ReadCheckpoint(const std::filesystem::path& path, uid_t trusted);

// The same, never as a pipeline: a pipeline's component.
std::expected<Checkpoint, Error> ReadComponentCheckpoint(const std::filesystem::path& path,
                                                         uid_t trusted);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_READER_H_
