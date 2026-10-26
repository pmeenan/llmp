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
#include <utility>
#include <vector>

#include "artifact/composition.h"
#include "import/indexed_parts.h"
#include "import/normalize.h"
#include "platform/lock_file.h"

namespace llmp::import {
struct ComponentPlan {
  const Checkpoint& checkpoint;
  spec::Source canonical{};
  spec::ModelSpec spec{};
  std::vector<CopyRoute> routes{};
  std::uint64_t payload_bytes = 0;
  // Each canonical tensor's group, in its order; empty: as the spec binds it.
  std::vector<ComponentGroup> groups{};
  // The mapping that planned it (an image pipeline's, or a configured
  // spatial component's) and its version, recorded in the manifest's
  // importer (and compared by image component reuse); empty: none.
  std::string mapping_architecture{};
  std::uint16_t mapping_version = 0;
};
// A store's native import lock (`.staging/native-import.lock`), which every
// publication into the store holds. An import takes it before it plans, so
// a second import into the same store fails at once rather than after its
// planning, and passes it to each publication (WriteOptions::lock). Taking
// it creates the store and its private staging directory, and removes the
// staging jobs an earlier holder abandoned.
class StoreLock {
 public:
  static std::expected<StoreLock, Error> Acquire(const std::filesystem::path& store);
  const std::filesystem::path& store() const { return store_; }

 private:
  StoreLock(std::filesystem::path store, platform::LockFile lock)
      : store_(std::move(store)), lock_(std::move(lock)) {}
  std::filesystem::path store_;
  platform::LockFile lock_;
};
struct CompositionWriteOptions {
  std::filesystem::path store{};
  // The lock the caller holds on `store`; null: publication takes it.
  const StoreLock* lock = nullptr;
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
  // The lock the caller holds on `store`; null: publication takes it.
  const StoreLock* lock = nullptr;
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
// and a failed stream leaves no published artifact. After the streams, one
// read of each written shard (shards in parallel) computes its digest and
// its chunks'. Publication is serialized within one store. An existing
// artifact of the same ID is reused after structural verification: its
// manifest, which the ID hashes, lists exactly the file and chunk digests
// this import computed from the bytes it wrote; its payload is not re-read
// (deep verification is install-time work, D-054).
std::expected<WriteResult, Error> WriteArtifact(const NormalizedPlan& plan,
                                                const WriteOptions& options);
std::expected<WriteResult, Error> WriteArtifact(const ComponentPlan& plan,
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
