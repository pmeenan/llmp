// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a runner maps for its life beside its weights and state
// (docs/engine.md), whatever its kernels: device memory of its own (a
// verify's snapshot, working memory) mapped at setup with access and pinned
// in the catalog, and pinned host staging, cataloged. The engine core's
// part of a runner's resources; RunnerResources (runner_resources.h, CUDA
// builds) adds cuBLAS and the GGML launch context over it.
//
// Lifetimes are completion-aware (AGENTS.md rule 6): ReleaseMappings runs
// only after the node's teardown fenced the model's stream and stopped the
// scheduler, and after everything that names this memory (plans, graphs, a
// launch context) is gone. The staging is the node's to free
// (PagedNode::Pinned), at its close.

#ifndef LLMP_ENGINE_RUNNER_MEMORY_H_
#define LLMP_ENGINE_RUNNER_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "catalog/catalog.h"
#include "engine/paged_node.h"

namespace llmp::engine {

struct AlignedPinned {
  void* allocation = nullptr;  // PagedNode owns this address; retain/free this, not data.
  std::byte* data = nullptr;
  std::uint64_t bytes = 0;
  std::uint64_t allocation_bytes = 0;  // the whole cataloged charge, including slack
};

class RunnerMemory {
 public:
  using Status = engine::Status;

  RunnerMemory(PagedNode& node, int owner, std::uint32_t stream)
      : node_(node), owner_(owner), stream_(stream) {}
  RunnerMemory(const RunnerMemory&) = delete;
  RunnerMemory& operator=(const RunnerMemory&) = delete;
  RunnerMemory(RunnerMemory&&) = delete;
  RunnerMemory& operator=(RunnerMemory&&) = delete;
  ~RunnerMemory() = default;

  PagedNode& node() const { return node_; }
  int owner() const { return owner_; }
  std::uint32_t stream() const { return stream_; }

  // `bytes` of device VMM (rounded to extents) of `memory_class`, pinned,
  // mapped into `mapped` for the runner's life. Before the node runs.
  // `mapped` must remain alive at the same address through fenced release,
  // including when Map fails after making a partial allocation.
  Status Map(Mapped& mapped, std::string name, std::uint64_t bytes,
             catalog::MemoryClass memory_class, bool* over_budget = nullptr);
  // Pinned host memory, cataloged as the runner's staging. Before the
  // node runs.
  std::expected<void*, std::string> Pinned(std::uint64_t bytes, bool* over_budget = nullptr);
  // An aligned view of pinned staging. A nonzero size and power-of-two
  // alignment are required. The allocation includes alignment - 1 slack
  // bytes and is charged whole before use; node teardown owns its lifetime.
  // RetireRing/KeepPinned/FreePinned must receive allocation, never data.
  std::expected<AlignedPinned, std::string> PinnedAligned(std::uint64_t bytes,
                                                          std::uint64_t alignment);
  // Successful setup allocations only, before the scheduler starts. Device
  // mappings include extent alignment; pinned bytes include PagedNode's 256
  // byte minimum. This excludes the shared workspace and growing live state.
  std::uint64_t mapped_bytes() const;
  std::uint64_t pinned_bytes() const { return pinned_bytes_; }

  // Counted control capacities for an owner that will perform these exact
  // allocations. ReserveMetadata is called only after the owner prepays it.
  static std::expected<std::uint64_t, std::string> MetadataBytes(std::size_t mappings,
                                                                 std::size_t pinned);
  void ReserveMetadata(std::size_t mappings, std::size_t pinned);
  std::size_t ExtentCount() const;
  Status FillExtents(std::span<catalog::ExtentId> into) const;

  // Every extent mapped here and the staging's: a closure's share.
  std::vector<catalog::ExtentId> extents() const;

  // Every mapping released (the problems appended). Once, after the node's
  // teardown fenced the stream and after whatever named the memory is gone.
  void ReleaseMappings(std::vector<std::string>& problems);
  // Completed-boundary retirement of this owner's allocations only. Captures,
  // plans and every request lease must have retired first. Successful removal
  // updates both the node catalog and this owner's staging accounting.
  Status Retire();

 private:
  PagedNode& node_;
  int owner_;
  std::uint32_t stream_;
  std::vector<Mapped*> mapped_;  // the runner's members, in mapping order
  std::vector<catalog::ExtentId> staging_;
  struct PinnedAllocation {
    void* pointer = nullptr;
    catalog::ExtentId extent;
    std::uint64_t bytes = 0;
  };
  std::vector<PinnedAllocation> pinned_;
  std::uint64_t pinned_bytes_ = 0;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_RUNNER_MEMORY_H_
