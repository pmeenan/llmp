// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A v0 artifact's weights as extents on a paged node (paged_node.h), for
// every M3 runner (dsv4_runner.h, qwen38_runner.h, qwen_image_runner.h;
// docs/engine.md, docs/experiments/fast-swap/):
//
// - The artifact opened, its shards open for direct reads, its identity
//   the catalog's content key.
// - Each dense group placed on the device gets a 2 MiB-aligned region of
//   one device reservation; its chunk k is an extent at the region's base +
//   k x 2 MiB, landed from its shard through the zone (D-081).
// - Each slab (a layer's routed experts at a uniform stride S, the resident
//   expert layout, docs/artifact-format.md#executable-views) is laid out by
//   LayOutSlab: an extent is a 2 MiB page of the slab's address range whose
//   contents are the stored bytes of the (at most two) groups it overlaps,
//   which are consecutive in the file, read as one 4 KiB-aligned range of
//   up to 2 MiB + 8 KiB into a landing slot and copied into the page in
//   pieces; the S - stored bytes between groups are never read by the
//   kernels and are not written. The slab starts δ bytes into its first
//   page (δ a multiple of its alignment) so that where the layer's groups
//   change shard, a page boundary falls in the gap between two groups: no
//   page needs two files.
// - A group placed on the host (DeepSeek's token table, whose rows the CPU
//   dequantizes as llama.cpp looks them up) gets a region of one host VMM
//   reservation, its chunks read there directly (D-034), not landed.
// - Extents are cataloged in file order (RE-026), all weights, restorable
//   from the artifact, with managed backing (D-033); their places are
//   pinned by the runner once registered (D-090).
// - Groups placed nowhere are not paged at all (Qwen3.8's n-gram table,
//   whose rows are read on demand, ple_rows.h; a component's parts its
//   phases never read).
//
// CUDA builds only.

#ifndef LLMP_ENGINE_PAGED_WEIGHTS_H_
#define LLMP_ENGINE_PAGED_WEIGHTS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "providers/device_memory.h"
#include "scheduler/scheduler.h"

namespace llmp::engine {

// ------------------------------------------------------------------ slabs

// A page of a slab: its read and the (at most kSlabPieces) pieces copied
// from the slot into the page. Exposed for tests.
inline constexpr std::size_t kSlabPieces = 4;  // at most kMaxDeviceCopies
struct SlabPage {
  std::uint32_t shard = 0;
  std::uint64_t file_offset = 0;  // 4 KiB-aligned
  std::uint64_t length = 0;       // 4 KiB-aligned
  std::array<std::uint64_t, kSlabPieces> slot_offset{};
  std::array<std::uint64_t, kSlabPieces> page_offset{};  // within the page
  std::array<std::uint64_t, kSlabPieces> bytes{};
  std::size_t pieces = 0;
};
// The most a page's read can take: a page's bytes, and 4 KiB before and
// after for alignment. The node's slot size for a model with slabs.
inline constexpr std::uint64_t kSlabSlotBytes = kPagedExtent + (std::uint64_t{8} * 1024);

// One slab: where its experts' groups are in the file (group e:
// `shard[e]`, `file[e]`, each `stored` bytes), its stride and the slab's
// offset δ in its first page, a multiple of `alignment` (a power of two,
// 16 to 4,096). Computes δ and the pages; refused if the groups are not
// consecutive in each shard, change shard more than once, or a stride is
// below the stored bytes.
struct SlabLayout {
  std::uint64_t stride = 0;
  std::uint64_t delta = 0;
  std::vector<SlabPage> pages;
  std::uint64_t bytes() const { return pages.size() * kPagedExtent; }
};
std::expected<SlabLayout, std::string> LayOutSlab(std::span<const std::uint32_t> shard,
                                                  std::span<const std::uint64_t> file,
                                                  std::uint64_t stored, std::uint64_t stride,
                                                  std::uint64_t alignment = 256);

// One layer's routed experts: `count` groups from `first_group`, each at
// `stride` in the slab, the slab's offset in its first page a multiple of
// `alignment` (LayOutSlab).
struct SlabSpec {
  std::uint32_t first_group = 0;
  std::uint32_t count = 0;
  std::uint64_t stride = 0;
  std::uint64_t alignment = 256;
};

// The slab of a layer whose expert arrays (each an index into the
// artifact's expert arrays and its GGML type's name) share their groups:
// its first group, and a stride of the stored bytes rounded up to 16 and to
// every array's element size. Refused if the arrays' groups differ.
std::expected<SlabSpec, std::string> ExpertSlab(
    const artifact::Artifact& artifact,
    std::span<const std::pair<std::uint32_t, std::string_view>> arrays, std::uint32_t count,
    std::uint64_t alignment, std::uint32_t layer);

// ------------------------------------------------------------------ weights

// Where a group's chunks go.
enum class GroupPlace : std::uint8_t {
  kNone,    // not paged
  kDevice,  // a region of the device reservation, landed through the zone
  kHost,    // a region of the host reservation, read there directly
};

class PagedWeights {
 public:
  using Status = engine::Status;

  // Opens the artifact (`options` its checks) and its shards for direct
  // reads. Once, first.
  Status Open(const std::filesystem::path& path, const artifact::OpenOptions& options = {});
  bool opened() const { return artifact_ != nullptr; }
  const artifact::Artifact& artifact() const { return *artifact_; }
  // A shard's direct-read descriptor, open while this is.
  int shard_fd(std::uint32_t shard) const { return shards_.at(shard).get(); }

  // Reserves and catalogs the places of the groups `place` names (one per
  // group) and of the slabs, in file order: one device reservation (dense
  // regions and slabs) and, if any group is kHost, one host reservation.
  // Refused if a placed group is an expert group, a slab's groups are not
  // uniform expert groups, or LayOutSlab refuses one.
  Status Reserve(PagedNode& node, std::span<const GroupPlace> place,
                 std::span<const SlabSpec> slabs);
  // After the node's Start: every extent's source, and its span for the
  // coverage check.
  Status Register(PagedNode& node, int owner);
  // Adds every extent that is not where it was registered, or not pinned
  // there (D-090), to `check`. On the scheduler's thread. A successful
  // weight-only scan is reused while that scheduler's placement stamp matches.
  void CheckPlaces(const scheduler::Scheduler& scheduler, PlaceCheck& check) const;
  // Frees the reservations (their extents must have been evicted).
  Status Release(providers::VmmProvider& memory);

  // The bytes of the device extents a load does not write: each dense
  // chunk's tail past its stored length, and a slab page's bytes outside
  // its pieces (the gaps between groups, before the first and after the
  // last). They hold whatever the backing held (a handoff's is the
  // outgoing model's), so a kernel that reads them reads stale bytes.
  struct Range {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    bool slab = false;
  };
  std::vector<Range> Unwritten() const;

  // A placed group's (or a slab expert's) address; 0 if not paged.
  std::uint64_t group_address(std::uint32_t group) const { return group_address_.at(group); }
  // A resource's and an expert array's (expert 0's slice) address.
  std::uint64_t resource_address(std::uint32_t resource) const;
  std::uint64_t array_address(std::uint32_t array) const;
  const std::vector<catalog::ExtentId>& extents() const { return extents_; }
  // The extents of the groups `which` picks (by group, and whether on the
  // host; a slab page's group is its layer's first).
  std::vector<catalog::ExtentId> ExtentsWhere(
      const std::function<bool(std::uint32_t group, bool host)>& which) const;
  std::uint64_t read_bytes() const { return read_bytes_; }  // one full load reads
  std::uint64_t slab_padding() const { return slab_padding_; }
  std::uint64_t bytes() const { return bytes_; }            // the device reservation
  std::uint64_t host_bytes() const { return host_bytes_; }  // the host reservation

 private:
  friend struct PagedWeightsTestAccess;
  bool CheckPlacesImpl(const scheduler::Scheduler& scheduler, PlaceCheck& check) const;
  mutable std::optional<scheduler::PlacementStamp> checked_places_;
  struct Source {
    std::uint64_t address = 0;
    std::uint32_t group = 0;
    bool host = false;
    scheduler::PageSource source;
  };
  std::unique_ptr<artifact::Artifact> artifact_;
  std::vector<artifact::FileDescriptor> shards_;
  std::array<std::uint8_t, 32> key_{};
  providers::ReservationId reservation_;
  providers::ReservationId host_reservation_;
  std::uint64_t bytes_ = 0;
  std::uint64_t host_bytes_ = 0;
  std::vector<std::uint64_t> group_address_;
  std::vector<catalog::ExtentId> extents_;
  std::vector<Source> sources_;  // by extent, in extents_' order
  std::uint64_t read_bytes_ = 0;
  std::uint64_t slab_padding_ = 0;
};

// The artifact's identity as the catalog's content key takes it.
std::array<std::uint8_t, 32> ArtifactKey(const artifact::Artifact& artifact);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PAGED_WEIGHTS_H_
