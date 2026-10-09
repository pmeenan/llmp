// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/reader.h"

#include <sys/types.h>

#include <array>
#include <expected>
#include <filesystem>
#include <span>
#include <system_error>

#include "import/checkpoint.h"
#include "import/error.h"
#include "import/readers/gguf.h"
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

// A directory: a Hugging Face checkpoint (config.json, safetensors).
bool ClaimsSafetensors(const std::filesystem::path& path) { return IsDirectory(path); }

constexpr std::array<ContainerReader, 2> kReaders = {{
    {.container = "gguf", .claims = ClaimsGguf, .read = ReadGguf},
    {.container = "safetensors", .claims = ClaimsSafetensors, .read = ReadSafetensors},
}};

}  // namespace

std::span<const ContainerReader> Readers() { return kReaders; }

std::expected<Checkpoint, Error> ReadCheckpoint(const std::filesystem::path& path, uid_t trusted) {
  for (const ContainerReader& reader : kReaders) {
    if (reader.claims(path)) {
      return reader.read(path, trusted);
    }
  }
  return std::unexpected(
      Error{.rule = Rule::kFileType, .reason = "a path no container reader claims"});
}

}  // namespace llmp::import
