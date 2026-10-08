// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The v0 prepared-artifact reader (docs/artifact-format.md; D-009, D-018,
// D-056). Artifact::Open treats the directory as untrusted input and
// either refuses it with the violated rule or returns its validated view:
// the manifest's identity and model, the dependency groups and their
// chunks, every resource's and expert slice's place in its group, and
// coalesced direct-read plans for missing chunks. It never reads tensor
// payload.
//
// Open checks what the prototype verifier checks when a loader opens an
// artifact (layout.py's verify with deep=False), in its order:
//
//   - the directory: a real directory, holding only data/ and meta/ and
//     regular, singly linked files, no links anywhere, exactly the files
//     the manifest lists plus manifest.json;
//   - manifest.json (at most 1 MiB): strict JSON, format "jitllm-artifact",
//     format_version 0 and the spark-v0 profile exactly, canonical bytes,
//     the directory named by their SHA-256; every key and value typed and
//     patterned, lists in canonical order;
//   - every listed file's size, before anything reads it; kept metadata
//     within its caps and covered by the recorded sources;
//   - index.json (at most 64 MiB): its manifest digest, strict canonical
//     JSON, the profile's header, shards, groups tiling each shard in
//     order with chunks numbered in group order, resources and expert
//     arrays sized by their representations, aligned, inside their groups
//     and not overlapping, EXL3 closure, expert structure, and the
//     manifest's model and transformations agreeing with it;
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
// roles per resource, 100,000,000-byte shard headers, 2^16 directory
// entries.

#ifndef LLMP_ARTIFACT_ARTIFACT_H_
#define LLMP_ARTIFACT_ARTIFACT_H_

#include <sys/types.h>

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

 private:
  int fd_ = -1;
};

struct Model {
  std::string architecture;
  std::uint32_t expert_count = 0;
  std::vector<Family> representation;  // sorted by name
};

struct Source {
  std::string name;
  Bytes bytes;
  Digest sha256{};
};

enum class FileRole : std::uint8_t { kIndex, kShard, kSourceMetadata };

struct ListedFile {
  std::string path;
  FileRole role = FileRole::kShard;
  Bytes bytes;
  Digest sha256{};
};

struct Converter {
  std::string name;
  std::string version;
};

struct Resource {
  std::string name;
  std::vector<std::string> roles;  // the name, then tied aliases, sorted
  std::uint32_t group = 0;
  Bytes offset;  // within the group, 256-byte aligned
  Bytes bytes;
  Bytes readable;  // bytes plus the backend's over-read (zero pad)
  Representation repr{};
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
  Representation repr{};  // of one 2-D slice
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
  // When set, the path to the artifact (platform/path_trust.h) and every
  // directory and file in it must be safe from users other than root and
  // this one (Rule::kUntrusted), since integrity then rests on the store's
  // permissions (D-056).
  std::optional<uid_t> trusted_owner;
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
  // and without following links; refused (kFileType) unless it is still
  // the file validated at open: the same device, inode, inode generation
  // (where the file system reports one), size and status-change time. A
  // file replaced since is refused even where the inode number is reused
  // (ext4 does): its generation differs, unless its owner set it back
  // (FS_IOC_SETVERSION). Where there is no generation (overlayfs, tmpfs,
  // NFS), such a replacement is refused only if its status-change time
  // differs. A rewrite in place is refused
  // where it moved the status-change time, which a file system with
  // coarse timestamps (ext4 with 128-byte inodes: whole seconds) may not
  // do within one tick; nor is a change after this call detected. Beyond
  // that identity, page-in rests on the store's permissions (D-056).
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

  struct FileIdentity {
    dev_t device = 0;
    ino_t inode = 0;
    // FS_IOC_GETVERSION (ext4, btrfs, xfs): a new file that reuses an inode
    // number gets another; nothing where the file system has none
    // (overlayfs, tmpfs, NFS), where identity rests on the rest.
    std::optional<std::uint64_t> generation;
    std::uint64_t size = 0;
    // st_ctim, from the stat taken before the header was read: a write,
    // truncation, link or mode change sets it to the current time, at the
    // file system's granularity (fine-grained once it has been queried, on
    // kernels and file systems with multigrain timestamps). Compared for
    // equality only, so a clock stepped back matters only if it reproduces
    // the recorded time exactly.
    std::int64_t changed_seconds = 0;
    std::int64_t changed_nanoseconds = 0;
  };

  FileDescriptor root_;
  std::string id_;
  Model model_;
  Converter converter_;
  std::vector<Source> sources_;
  std::vector<ListedFile> files_;
  Layout layout_;
  std::vector<FileIdentity> shard_files_;
  std::vector<Resource> resources_;
  std::vector<ExpertArray> expert_arrays_;
  std::vector<Digest> chunk_sha256_;
  // (role, resource) and (name, array), sorted by name.
  std::vector<std::pair<std::string, std::uint32_t>> roles_;
  std::vector<std::pair<std::string, std::uint32_t>> array_names_;
};

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_ARTIFACT_H_
