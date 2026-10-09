// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/reader.h"

#include <sys/types.h>

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>
#include <system_error>

#include "import/checkpoint.h"
#include "import/error.h"
#include "import/readers/diffusers.h"
#include "import/readers/exl3.h"
#include "import/readers/gguf.h"
#include "import/readers/mlx.h"
#include "import/readers/safetensors.h"
#include "platform/interfaces.h"

namespace llmp::import {
namespace {

bool IsDirectory(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_directory(path, error);
}

// A file named *.gguf (a split GGUF by its first file).
bool ClaimsGguf(const std::filesystem::path& path, uid_t /*trusted*/) {
  return !IsDirectory(path) && path.extension() == ".gguf";
}

// A directory with model_index.json: a diffusers pipeline.
bool ClaimsDiffusers(const std::filesystem::path& path, uid_t /*trusted*/) {
  std::error_code error;
  return IsDirectory(path) && std::filesystem::exists(path / "model_index.json", error);
}

// A directory whose config.json names EXL3's quantization.
bool ClaimsExl3(const std::filesystem::path& path, uid_t trusted) {
  if (!IsDirectory(path)) {
    return false;
  }
  const auto config = PeekConfig(path, trusted);
  return config && IsExl3Config(config->root());
}

// A directory whose config.json has MLX's quantization object.
bool ClaimsMlx(const std::filesystem::path& path, uid_t trusted) {
  if (!IsDirectory(path)) {
    return false;
  }
  const auto config = PeekConfig(path, trusted);
  return config && IsMlxConfig(config->root());
}

// Any other directory: a Hugging Face checkpoint (config.json, safetensors).
bool ClaimsSafetensors(const std::filesystem::path& path, uid_t /*trusted*/) {
  return IsDirectory(path);
}

constexpr std::array<ContainerReader, 5> kReaders = {{
    {.container = "gguf", .claims = ClaimsGguf, .read = ReadGguf},
    {.container = "diffusers", .claims = ClaimsDiffusers, .read = ReadDiffusers},
    {.container = "exl3", .claims = ClaimsExl3, .read = ReadExl3},
    {.container = "mlx", .claims = ClaimsMlx, .read = ReadMlx},
    {.container = "safetensors", .claims = ClaimsSafetensors, .read = ReadSafetensors},
}};

std::expected<Checkpoint, Error> Read(const std::filesystem::path& path, uid_t trusted,
                                      bool pipelines) {
  for (const ContainerReader& reader : kReaders) {
    if (!pipelines && reader.container == "diffusers") {
      continue;
    }
    if (reader.claims(path, trusted)) {
      return reader.read(path, trusted);
    }
  }
  return std::unexpected(
      Error{.rule = Rule::kFileType, .reason = "a path no container reader claims"});
}

// The most files one checkpoint holds open (a pipeline's, the largest),
// with room for the process's own.
constexpr std::uint64_t kMostOpenFiles =
    (kMaxPipelineComponents * (kMaxShards + 3)) + kMaxPipelineFiles + 1 + 1024;
static_assert(kMostOpenFiles >= kMaxGgufSplits + 1024);

}  // namespace

std::span<const ContainerReader> Readers() { return kReaders; }

std::expected<Checkpoint, Error> ReadCheckpoint(const std::filesystem::path& path, uid_t trusted) {
  // A checkpoint keeps every file it read open (files.h): at most a
  // pipeline's kept files and its weight components' shards and documents,
  // more than the usual soft limit of 1,024. The soft limit is raised that
  // far (two system calls, or one when it is there already), as far as the
  // hard limit allows; past what it allows, an open fails as kIo.
  static_cast<void>(platform::RaiseOpenFileLimit(kMostOpenFiles));
  return Read(path, trusted, true);
}

std::expected<Checkpoint, Error> ReadComponentCheckpoint(const std::filesystem::path& path,
                                                         uid_t trusted) {
  return Read(path, trusted, false);
}

}  // namespace llmp::import
