// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/reader.h"

#include <array>
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

namespace llmp::import {
namespace {

bool IsDirectory(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_directory(path, error);
}

// A file named *.gguf (a split GGUF by its first file).
bool ClaimsGguf(const std::filesystem::path& path) {
  return !IsDirectory(path) && path.extension() == ".gguf";
}

// A directory with model_index.json: a diffusers pipeline.
bool ClaimsDiffusers(const std::filesystem::path& path) {
  std::error_code error;
  return IsDirectory(path) && std::filesystem::exists(path / "model_index.json", error);
}

// A directory whose config.json names EXL3's quantization.
bool ClaimsExl3(const std::filesystem::path& path) {
  if (!IsDirectory(path)) {
    return false;
  }
  const auto config = PeekConfig(path);
  return config && IsExl3Config(config->root());
}

// A directory whose config.json has MLX's quantization object.
bool ClaimsMlx(const std::filesystem::path& path) {
  if (!IsDirectory(path)) {
    return false;
  }
  const auto config = PeekConfig(path);
  return config && IsMlxConfig(config->root());
}

// Any other directory: a Hugging Face checkpoint (config.json, safetensors).
bool ClaimsSafetensors(const std::filesystem::path& path) { return IsDirectory(path); }

constexpr std::array<ContainerReader, 5> kReaders = {{
    {.container = "gguf", .claims = ClaimsGguf, .read = ReadGguf},
    {.container = "diffusers", .claims = ClaimsDiffusers, .read = ReadDiffusers},
    {.container = "exl3", .claims = ClaimsExl3, .read = ReadExl3},
    {.container = "mlx", .claims = ClaimsMlx, .read = ReadMlx},
    {.container = "safetensors", .claims = ClaimsSafetensors, .read = ReadSafetensors},
}};

std::expected<Checkpoint, Error> Read(const std::filesystem::path& path, bool pipelines) {
  for (const ContainerReader& reader : kReaders) {
    if (!pipelines && reader.container == "diffusers") {
      continue;
    }
    if (reader.claims(path)) {
      return reader.read(path);
    }
  }
  return std::unexpected(
      Error{.rule = Rule::kFileType, .reason = "a path no container reader claims"});
}

}  // namespace

std::span<const ContainerReader> Readers() { return kReaders; }

std::expected<Checkpoint, Error> ReadCheckpoint(const std::filesystem::path& path) {
  return Read(path, true);
}

std::expected<Checkpoint, Error> ReadComponentCheckpoint(const std::filesystem::path& path) {
  return Read(path, false);
}

}  // namespace llmp::import
