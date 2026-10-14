// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_WRITER_H_
#define LLMP_IMPORT_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "artifact/composition.h"
#include "import/normalize.h"

namespace llmp::import {
class ModelOptTargetPlan;
struct ComponentPlan {
  const Checkpoint& checkpoint;
  spec::Source canonical{};
  spec::ModelSpec spec{};
  std::vector<CopyRoute> routes{};
  std::uint64_t payload_bytes = 0;
};
struct CompositionWriteOptions {
  std::filesystem::path store{};
  std::string build{};
  std::string architecture{};
  std::vector<artifact::CompositionComponent> components{};
  std::vector<SourceFile> files{};
  std::span<const std::optional<base::Sha256Digest>> expected{};
};
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
// Writes a privately owned normalized plan. Sources remain open one at a time,
// payload working space is bounded, expert arrays become contiguous slices,
// and a failed stream leaves no published artifact. Publication is serialized
// within one store; an existing ID is fully verified before reuse.
std::expected<WriteResult, Error> WriteArtifact(const NormalizedPlan& plan,
                                                const WriteOptions& options);
std::expected<WriteResult, Error> WriteArtifact(const ModelOptTargetPlan& plan,
                                                const WriteOptions& options);
std::expected<WriteResult, Error> WriteArtifact(const ComponentPlan& plan,
                                                const WriteOptions& options);
std::expected<WriteResult, Error> WriteComposition(const CompositionWriteOptions& options);
namespace publication_detail {
// Internal metadata-only check of the exact placement/transfer planner used by
// WriteArtifact. No file opens, payload reads, or publication effects. The
// caller funds temporary layout metadata just as for publication.
struct PlacementSummary {
  std::uint64_t metadata_bytes = 0;
  std::size_t copies = 0, destinations = 0, tables = 0, globals = 0;
  std::size_t layers = 0, heads = 0, expert_groups = 0, row_tables = 0;
};
std::expected<PlacementSummary, Error> InspectPlacement(const NormalizedPlan& plan,
                                                        std::uint64_t metadata_bytes);
}  // namespace publication_detail
}  // namespace llmp::import
#endif  // LLMP_IMPORT_WRITER_H_
