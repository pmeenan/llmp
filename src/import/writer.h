// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_WRITER_H_
#define LLMP_IMPORT_WRITER_H_

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "import/normalize.h"

namespace llmp::import {
struct WriteOptions {
  std::filesystem::path store{};
  std::string build{};
  // One optional archive identity per source file, in the plan's order.
  std::span<const std::optional<base::Sha256Digest>> expected{};
  // Explicit ceiling for newly requested metadata, excluding the owned plan.
  std::uint64_t metadata_bytes = 0;
};
struct WriteResult {
  std::string id{};
  std::filesystem::path path{};
  bool reused = false;
};
// Writes a copy-only normalized plan. Sources remain open one at a time,
// payload working space is bounded, expert arrays become contiguous slices,
// and a failed stream leaves no published artifact. Publication is serialized
// within one store; an existing ID is fully verified before reuse.
std::expected<WriteResult, Error> WriteArtifact(const NormalizedPlan& plan,
                                                const WriteOptions& options);
}  // namespace llmp::import
#endif  // LLMP_IMPORT_WRITER_H_
