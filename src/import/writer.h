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
#include "import/indexed_parts.h"
#include "import/normalize.h"

namespace llmp::import {
class ModelOptTargetPlan;
class ModelOptDrafterPlan;
// Where a component's tensor is stored when its spec's bindings do not say
// (an image component's spec binds every tensor in one spatial layer): its
// row table, its global group, one of its layers' (a transformer block's,
// a decoder up block's) or its head group.
struct ComponentGroup {
  enum class Kind : std::uint8_t { kTable, kGlobal, kLayer, kHead };
  Kind kind = Kind::kGlobal;
  std::uint32_t layer = 0;
};
struct ComponentPlan {
  const Checkpoint& checkpoint;
  spec::Source canonical{};
  spec::ModelSpec spec{};
  std::vector<CopyRoute> routes{};
  std::uint64_t payload_bytes = 0;
  // Each canonical tensor's group, in its order; empty: as the spec binds it.
  std::vector<ComponentGroup> groups{};
};
struct CompositionWriteOptions {
  std::filesystem::path store{};
  std::string build{};
  std::string architecture{};
  std::vector<artifact::CompositionComponent> components{};
  std::vector<SourceFile> files{};
  std::span<const std::optional<base::Sha256Digest>> expected{};
  // The spec of the model the composition makes (D-113), kept as its
  // spec.json, its stages' artifact indexes counting the components in
  // order; none: a composition without one.
  std::optional<spec::ModelSpec> spec{};
};
struct WriteOptions {
  std::filesystem::path store{};
  std::string build{};
  // One optional archive identity per source file, in the plan's order.
  std::span<const std::optional<base::Sha256Digest>> expected{};
  // Explicit ceiling for newly requested metadata, excluding the owned plan.
  std::uint64_t metadata_bytes = 0;
  // The import options that change what is published, as canonical JSON
  // (empty: none); the manifest's importer records their SHA-256 (D-113).
  std::string options{};
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
// A drafter carries no spec (D-113); its generated parts (a selected head's
// token map) are written once every source stream, the whole file each
// derives from included, has authenticated.
std::expected<WriteResult, Error> WriteArtifact(const ModelOptDrafterPlan& plan,
                                                const WriteOptions& options);
std::expected<void, Error> ValidateCompositionFiles(std::span<const SourceFile> files);
std::expected<WriteResult, Error> WriteComposition(const CompositionWriteOptions& options);
namespace publication_detail {
std::expected<WriteResult, Error> WriteIndexed(
    const IndexedPartPlan& parts, const spec::Source& source, const spec::ModelSpec& model,
    std::uint64_t payload, std::span<const NormalizedPartBinding> bindings,
    std::span<const SemanticReceipt> semantic, std::span<const PlannedPartFold> folds,
    std::span<const GeneratedPart> generated, const WriteOptions& options);

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
