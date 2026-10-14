// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_ARCHIVE_INPUT_H_
#define LLMP_IMPORT_ARCHIVE_INPUT_H_
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "import/checkpoint.h"

namespace llmp::import {
struct ArchivedFile {
  std::uint64_t bytes = 0;
  bool present = false;
  std::optional<base::Sha256Digest> sha256{};
};
struct ArchiveVariant {
  std::string id{};
  std::vector<std::string> files{};
};
struct ArchiveInput {
  std::filesystem::path revision{}, root{};
  std::string repo{}, commit{};
  std::map<std::string, ArchivedFile, std::less<>> files{};
  std::vector<ArchiveVariant> variants{};
};
// Nearest ancestor manifest, if any; an invalid manifest never falls back to
// ordinary import. The marker identifies an archive; a staged revision with
// no marker is authenticated identically and does not write catalog events.
std::expected<std::optional<ArchiveInput>, Error> FindArchiveInput(
    const std::filesystem::path& source);
// Resolves an archived directory to its single selected GGUF set, or the
// selected safetensors directory; ambiguity is refused, never guessed.
std::expected<std::filesystem::path, Error> SelectArchiveSource(
    const ArchiveInput& archive, const std::filesystem::path& requested);
std::expected<std::vector<std::optional<base::Sha256Digest>>, Error> ArchiveFileIdentities(
    const ArchiveInput& archive, std::span<const SourceFile> files);
std::expected<std::vector<std::optional<base::Sha256Digest>>, Error> ArchiveIdentities(
    const ArchiveInput& archive, const Checkpoint& source);
// Called only after publication. Returns no event for a staged copy. An
// event failure does not undo the published artifact and must be reported as
// such by the CLI. Independent event writers never take the checkpoint guard.
std::expected<void, Error> RecordArchiveImport(const ArchiveInput& archive,
                                               const Checkpoint& source,
                                               const artifact::Artifact& artifact);
std::expected<void, Error> RecordArchiveImport(const ArchiveInput& archive,
                                               const Checkpoint& source,
                                               const artifact::Composition& composition);
}  // namespace llmp::import
#endif  // LLMP_IMPORT_ARCHIVE_INPUT_H_
