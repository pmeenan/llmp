// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The prepared-artifact reader (docs/artifact-format.md; D-009, D-018,
// D-056, D-113). Artifact::Open treats the directory as untrusted input and
// either refuses it with the violated rule or returns its validated view:
// the manifest's identity and model, the dependency groups and their
// chunks, every resource's and expert slice's place in its group, and
// coalesced direct-read plans for missing chunks. It never reads tensor
// payload.
//
// It reads two format versions, told apart by the manifest's `format`: v0
// ("jitllm-artifact", format_version 0), whose resources carry v0's
// library-named representation descriptors, and v1 ("llmp-artifact",
// format_version 1), whose resources carry their tensor's format and the
// part of it they hold (format_json.h), whose manifest names the importer,
// the source container and the folds and relayouts the importer applied,
// and which may carry the model spec as a hashed document of its own,
// spec.json. Any other version of either is refused (kUnsupportedVersion,
// `item` the version found): a newer artifact needs a newer runtime, an
// older one a re-import; nothing is migrated (D-018). Groups, chunks,
// placements and read plans are the same for both, so nothing that pages
// an artifact in depends on which version it read.
//
// Open checks what the prototype verifier checks when a loader opens a v0
// artifact (layout.py's verify with deep=False), in its order, and the
// same for v1 with v1's descriptors:
//
//   - the directory (itself possibly reached through a link): holding
//     only data/ and meta/ and regular files, no links inside it, exactly
//     the files the manifest lists plus manifest.json;
//   - manifest.json (at most 1 MiB): strict JSON, a format and version
//     this reader implements and the spark-v0 profile exactly, canonical
//     bytes, the directory named by their SHA-256; every key and value
//     typed and patterned, lists in canonical order;
//   - every listed file's size, before anything reads it; kept metadata
//     within its caps and covered by the recorded sources;
//   - v1's spec.json, when listed (at most kMaxSpecBytes): its digest, and
//     strict canonical JSON whose root is an object. Its schema is the
//     spec module's (spec/stored.h), which comes after this one in the
//     layers; so is checking the applied folds' names against the fold
//     catalog;
//   - index.json (at most 64 MiB): its manifest digest, strict canonical
//     JSON, the profile's header, shards, groups tiling each shard in
//     order with chunks numbered in group order, resources and expert
//     arrays sized by their representations (v0) or formats (v1),
//     aligned, inside their groups and not overlapping, EXL3 closure (v0),
//     expert structure, and the manifest's model and transformations
//     agreeing with it;
//   - each shard's header: rebuilt from the index and compared byte for
//     byte, never parsed, and its digest checked.
//
// Deep verification (payload, chunk and kept-metadata digests, zero pads)
// is install-time work (D-054) and not done here. Nor are kept metadata
// files' contents read: in particular a kept `.kv.gguf` is not parsed
// (the prototype's loader-open does parse it); the reader does not use
// them, and whatever does must validate them.
//
// Caps bound every allocation by the documents' size limits: 2^18
// resources plus expert slices plus alias roles, 2^18 groups, 8 alias
// roles per resource, 2^18 tensors named by v1's folds and relayouts,
// 100,000,000-byte shard headers, 2^16 directory entries.

#ifndef LLMP_ARTIFACT_ARTIFACT_H_
#define LLMP_ARTIFACT_ARTIFACT_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "artifact/layout.h"
#include "artifact/representation.h"
#include "base/bytes.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::artifact {

inline constexpr std::uint64_t kMaxManifestBytes = std::uint64_t{1} << 20U;
inline constexpr std::uint64_t kMaxIndexBytes = std::uint64_t{64} << 20U;
inline constexpr std::uint64_t kMaxMetadataBytes = std::uint64_t{64} << 20U;
inline constexpr std::uint64_t kMaxMetadataTotal = std::uint64_t{128} << 20U;
inline constexpr std::uint64_t kMaxShardHeader = 100'000'000;
inline constexpr std::uint64_t kMaxEntries = std::uint64_t{1} << 18U;
inline constexpr std::uint64_t kMaxAliases = 8;
inline constexpr std::uint64_t kMaxExperts = std::uint64_t{1} << 20U;
inline constexpr std::uint64_t kMaxLayer = std::uint64_t{1} << 20U;
inline constexpr std::uint64_t kMaxDirectoryEntries = std::uint64_t{1} << 16U;
// v1's spec.json; spec::kMaxSpecBytes (spec/stored.cc checks they agree).
inline constexpr std::uint64_t kMaxSpecBytes = std::uint64_t{4} << 20U;
// The newest artifact format version this build reads; it reads 0 too.
inline constexpr std::uint32_t kFormatVersion = 1;

// An owned file descriptor.
class FileDescriptor {
 public:
  FileDescriptor() = default;
  explicit FileDescriptor(int fd) : fd_(fd) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  FileDescriptor& operator=(FileDescriptor&& other) noexcept;
  ~FileDescriptor();

  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  // The descriptor, no longer owned (the caller closes it).
  int release() { return std::exchange(fd_, -1); }

 private:
  int fd_ = -1;
};

struct Model {
  std::string architecture;
  std::uint32_t expert_count = 0;
  std::vector<Family> representation;  // v0: sorted by name; v1: empty
  // v1: the container the importer read ("gguf", "safetensors", "mlx",
  // "exl3"); v0: empty.
  std::string container;
};

struct Source {
  std::string name;
  Bytes bytes;
  Digest sha256{};
};

enum class FileRole : std::uint8_t { kIndex, kShard, kSourceMetadata, kSpec };

struct ListedFile {
  std::string path;
  FileRole role = FileRole::kShard;
  Bytes bytes;
  Digest sha256{};
};

// v0's `converter`, or v1's `importer`: its name and version, and for v1
// its build identity and the SHA-256 of the options it ran with (empty
// and zero for v0).
struct Converter {
  std::string name;
  std::string version;
  std::string build{};
  Digest options_sha256{};
};

struct Resource {
  std::string name;
  std::vector<std::string> roles;  // the name, then tied aliases, sorted
  std::uint32_t group = 0;
  Bytes offset;  // within the group, 256-byte aligned
  Bytes bytes;
  Bytes readable;         // bytes plus the backend's over-read (zero pad)
  Representation repr{};  // v0 only; a v1 resource's is empty
  // The format of the tensor it holds a part of, and that part: v1's as
  // recorded; v0's as format_map.h reads its representation (none where
  // that has no format: Q8_1).
  std::optional<format::TensorFormat> format;
  format::PartRole part = format::PartRole::kData;
  bool rows = false;  // a row table (`access: "rows"`)
};

// A sliced expert tensor: slice e lives in group first_group + e at
// group_offset.
struct ExpertArray {
  std::string name;
  std::uint32_t layer = 0;
  std::uint32_t count = 0;
  std::uint32_t first_group = 0;
  Bytes group_offset;
  Bytes slice_bytes;
  Bytes readable;
  Representation repr{};  // of one 2-D slice; v0 only
  // One slice's format (at most two dimensions) and the part each slice
  // holds, as for a resource.
  std::optional<format::TensorFormat> format;
  format::PartRole part = format::PartRole::kData;
};

// v1: a fold the importer applied (a name from the fold catalog,
// spec/spec.h's FoldInfo) and the tensors it applied it to, by resource
// name or expert array name, sorted.
struct AppliedFold {
  std::string name;
  std::vector<std::string> tensors;
};

// v1: tensors the importer relaid out losslessly from layout `from` into
// the one their format records.
struct Relayout {
  format::Layout from = format::Layout::kDense;
  std::vector<std::string> tensors;  // sorted
};

// Where a resource or slice lives: its byte range in its group and the
// chunks its readable range needs.
struct Placement {
  std::uint32_t group = 0;
  Bytes offset;
  Bytes bytes;
  Bytes readable;
  Closure closure{};
};

struct OpenOptions {
  // The ID the manifest must hash to; by default the directory's name.
  std::optional<std::string> expected_id;
};

class Artifact {
 public:
  static std::expected<Artifact, Error> Open(const std::filesystem::path& root,
                                             const OpenOptions& options = {});

  Artifact(const Artifact&) = delete;
  Artifact& operator=(const Artifact&) = delete;
  Artifact(Artifact&&) noexcept = default;
  Artifact& operator=(Artifact&&) noexcept = default;
  ~Artifact() = default;

  const std::string& id() const { return id_; }
  // 0 or 1.
  std::uint32_t format_version() const { return format_version_; }
  const Model& model() const { return model_; }
  const Converter& converter() const { return converter_; }
  std::span<const Source> sources() const { return sources_; }
  std::span<const ListedFile> files() const { return files_; }
  const Layout& layout() const { return layout_; }
  std::span<const Shard> shards() const { return layout_.shards; }
  std::span<const Group> groups() const { return layout_.groups; }
  std::span<const Resource> resources() const { return resources_; }
  std::span<const ExpertArray> expert_arrays() const { return expert_arrays_; }
  // One digest per chunk, by global chunk number (for verification and
  // replication; page-in never hashes).
  std::span<const Digest> chunk_sha256() const { return chunk_sha256_; }
  // v1: the spec.json bytes Open read, hashed and checked, when the
  // artifact carries a spec (spec/stored.h parses them). Their SHA-256 is
  // the spec's digest, the spec's JSON being canonical.
  std::optional<std::string_view> spec_json() const {
    return spec_json_ ? std::optional<std::string_view>(*spec_json_) : std::nullopt;
  }
  // v1: what the importer applied; empty for v0.
  std::span<const AppliedFold> folds() const { return folds_; }
  std::span<const Relayout> relayouts() const { return relayouts_; }

  // The resource bound to `role` (its name or a tied alias).
  std::optional<std::uint32_t> FindResource(std::string_view role) const;
  std::optional<std::uint32_t> FindExpertArray(std::string_view name) const;

  std::expected<Placement, Error> ResourcePlacement(std::uint32_t resource) const;
  std::expected<Placement, Error> SlicePlacement(std::uint32_t array, std::uint32_t expert) const;

  // A row table's geometry, from its representation.
  std::expected<RowGeometry, Error> Rows(std::uint32_t resource) const;
  // The chunks the given rows need, deduplicated and sorted. Row IDs are
  // untrusted: one outside the table is refused, never clamped.
  std::expected<std::vector<ChunkKey>, Error> RowChunks(std::uint32_t resource,
                                                        std::span<const std::uint64_t> rows) const;

  std::expected<std::vector<ReadRun>, Error> PlanReads(std::span<const ChunkKey> missing,
                                                       std::span<const ChunkKey> resident,
                                                       ReadLimits limits = {}) const {
    return artifact::PlanReads(layout_, missing, resident, limits);
  }

  // Shard `shard` opened read-only for direct I/O, beneath the artifact
  // and without following links; refused unless it is a regular file
  // (kFileType) of the size the installed manifest lists (kFileSize).
  // Called whenever the shard's model becomes active again
  // (engine/paged_weights.h), perhaps long after Open: the check is the
  // read's functional one, not the file's identity (D-056 as amended on
  // 2026-10-10), so a shard whose status changed (a mode, a new hard
  // link) still opens, and one whose contents changed at the same size is
  // the user's to answer for, as the store is (D-073 as amended).
  std::expected<FileDescriptor, Error> OpenShardForDirectRead(std::uint32_t shard) const;

  // A kept metadata file (`meta/<name>`, listed with the role
  // source-metadata), read beneath the artifact without following links:
  // refused (kFileSet) unless the manifest lists it, and (kFileSize,
  // kHash) unless it still has the listed size and SHA-256. Within
  // kMaxMetadataBytes by the manifest's caps.
  std::expected<std::string, Error> ReadMetadata(std::string_view name) const;

 private:
  friend class Opener;
  Artifact() = default;

  FileDescriptor root_;
  std::string id_;
  std::uint32_t format_version_ = 0;
  Model model_;
  Converter converter_;
  std::vector<Source> sources_;
  std::vector<ListedFile> files_;
  Layout layout_;
  std::vector<std::uint64_t> shard_bytes_;  // each shard file's listed size
  std::vector<Resource> resources_;
  std::vector<ExpertArray> expert_arrays_;
  std::vector<Digest> chunk_sha256_;
  std::optional<std::string> spec_json_;
  std::vector<AppliedFold> folds_;
  std::vector<Relayout> relayouts_;
  // (role, resource) and (name, array), sorted by name.
  std::vector<std::pair<std::string, std::uint32_t>> roles_;
  std::vector<std::pair<std::string, std::uint32_t>> array_names_;
};

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_ARTIFACT_H_
