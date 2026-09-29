// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A model's conversation state on a paged node (docs/engine.md), the part
// of a runner that holds what a conversation has computed:
//
// - Regions of live state (D-068: the target's caches and recurrent state,
//   then a drafter's), each laid out by the model (model/*.h) and reserved at
//   setup before the weights' places, so a closure (sorted by identity)
//   restores the state before paging the weights in. Every extent is
//   kPreserve with a write-back place, a 2 MiB range of one unnamed
//   direct-I/O spill file (numbered across the regions): evicting it writes
//   it back through the zone (D-081's reverse path) and loading it restores
//   it, at the place registered, pinned (D-090).
// - The quarantine: a job that failed after it may have written the state
//   leaves it unusable until it is cleared, so the state is never left
//   half-written and used.
// - A speculative verify's snapshot (D-068 working state): before a verify
//   writes the state, every range it will write is saved into device
//   memory (the model says which, by verify row, -1 for its scratch rows);
//   accepting its first rows owes the restore of the rest before the next
//   job's own work, and a model's commit of the kept rows after it
//   (Qwen3.8's recurrent state), so the state is what a verify of the kept
//   rows alone would have left. A failed verify is undone whole.
//
// A growing region keeps its layout's virtual bytes; only initialized
// extents have backing and spill contents. Turn-boundary
// checkpoints of recurrent or indexer state are saved ranges kept across
// jobs, the snapshot's mechanism with a longer life.

#ifndef JITLLM_ENGINE_LIVE_STATE_H_
#define JITLLM_ENGINE_LIVE_STATE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "scheduler/scheduler.h"

namespace jitllm::engine {

class RunnerResources;

class LiveState {
 public:
  using Status = engine::Status;
  // A model's commit of a verify's first `keep` rows, queued after the
  // rejected rows' restore (Qwen3.8's recurrent and convolution state).
  using Commit = std::function<std::expected<void, kernels::ggml::KernelFailure>(
      kernels::ggml::LaunchContext& launch, std::uint32_t keep)>;

  // `model` names it in refusals ("DeepSeek", "Qwen3.8").
  explicit LiveState(std::string model) : model_(std::move(model)) {}
  LiveState(const LiveState&) = delete;
  LiveState& operator=(const LiveState&) = delete;
  LiveState(LiveState&&) = delete;
  LiveState& operator=(LiveState&&) = delete;
  // Nothing is released here: Release does it, after the node's teardown
  // proved no read or write of the spill file or the regions is in flight.
  ~LiveState() = default;

  // ---------------------------------------------------------------- regions

  // A region of `bytes` (its layout's; mapped in whole extents), after the
  // ones before it. Before the weights' places are reserved.
  Status Add(PagedNode& node, std::string name, std::uint64_t bytes, int owner);
  // A stable virtual reservation with no backing until Use names a range.
  // Its sparse spill file initially supplies zero bytes; initialized
  // extents thereafter have the same write-back contract as Add's.
  Status AddGrowing(PagedNode& node, std::string name, std::uint64_t bytes, int owner);
  struct Range {
    std::size_t region = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };
  // Materializes the extents intersecting these ranges, without changing
  // any previous contents or addresses. True when the used set grew.
  std::expected<bool, std::string> Use(PagedNode& node, std::span<const Range> ranges,
                                       const catalog::Closure* keep = nullptr);
  // Drops initialized extents outside these already-used ranges, without
  // reloading them or writing them back. Called between jobs, with no
  // request lease on the state. Its saved file ranges return to sparse zeros.
  Status Retain(PagedNode& node, std::span<const Range> ranges);
  // Copies only these initialized ranges, packed in their given order,
  // between state and the caller's sufficiently large pinned buffer.
  Status Copy(PagedNode& node, const catalog::Closure& fence, std::uint32_t stream, void* host,
              std::span<const Range> ranges, bool to_host);
  std::size_t regions() const { return regions_.size(); }
  // A region's address and its layout's bytes; 0 for a region not added.
  std::uint64_t base(std::size_t region) const;
  std::uint64_t bytes(std::size_t region) const;
  std::uint64_t total_bytes() const;
  // Every extent, region by region.
  std::vector<catalog::ExtentId> extents() const;
  // Every extent of the virtual reservations, including unused ones:
  // sources and address pins are registered once, at setup.
  std::vector<catalog::ExtentId> reserved_extents() const;
  // Used extents in region order, bounded to each layout's bytes.
  std::vector<Range> used_ranges() const;
  std::uint64_t used_bytes() const;

  // After the node's Start: each extent's write-back place in an unnamed
  // direct-I/O spill file in `directory`, region by region.
  Status RegisterSpill(PagedNode& node, const std::filesystem::path& directory);
  // Adds every extent not where it was registered, or not pinned there
  // (D-090), to `check`. On the scheduler's thread.
  void CheckPlaces(const scheduler::Scheduler& scheduler, PlaceCheck& check) const;

  // Every region zeroed (a job over `fence` on `stream`); any snapshot,
  // restore or commit owed dropped. Unusable until it ran; a quarantine
  // lifts once it has.
  Status Clear(PagedNode& node, const catalog::Closure& fence, std::uint32_t stream,
               std::string_view what);
  // Every region read to the host (a job), one output per region (empty
  // for a region not added).
  Status Read(PagedNode& node, const catalog::Closure& fence, std::uint32_t stream,
              std::string_view what, std::span<std::vector<std::byte>* const> out);
  // Pinned host memory of total_bytes() (allocated on first use; null if
  // that failed): Read's copy, and a harness's scratch.
  void* HostCopy(PagedNode& node, std::uint64_t bytes = 0);
  // A failed copy may still own the DMA destination. Keep it until exit
  // and refuse subsequent reuse, even when the requested size is smaller.
  void HostCopyUnproven() {
    host_copy_unproven_ = true;
    if (host_node_ != nullptr) {
      host_node_->KeepPinned(host_copy_);
    }
  }

  // ------------------------------------------------------------- the guard

  bool quarantined() const { return quarantined_; }
  void Quarantine() { quarantined_ = true; }
  // Refused while quarantined.
  Status Usable() const;

  // ------------------------------------------------------------- speculation

  // A verify's snapshot: `capacity` save and restore descriptors (pinned
  // staging), the saved bytes in [base, base + bytes) of the model's own
  // device memory (SnapshotAt), and `commit` if the model commits kept rows.
  Status AllocateSnapshot(RunnerResources& resources, std::uint32_t capacity);
  void SnapshotAt(std::uint64_t base, std::uint64_t bytes) {
    snapshot_base_ = base;
    snapshot_bytes_ = bytes;
  }
  void SetCommit(Commit commit) { commit_ = std::move(commit); }

  // A verify's saves: Begin, then each range it writes (verify row `row`,
  // -1 its scratch rows), each at a 256-byte boundary of the snapshot.
  void BeginSaves();
  Status Save(std::uint64_t address, std::uint64_t bytes, std::int64_t row);
  // Queues the saves (in the verify's job, before its run).
  std::expected<void, kernels::ggml::KernelFailure> QueueSaves(
      kernels::ggml::LaunchContext& launch) const;
  // The verify ran: `rows` await Accept.
  void Verified(std::uint32_t rows) { verify_rows_ = rows; }
  std::uint32_t verify_rows() const { return verify_rows_; }
  Status AwaitingAccept() const;
  struct Saved {
    std::uint64_t address = 0;  // the state's bytes
    std::uint64_t saved = 0;    // their copy
    std::uint64_t bytes = 0;
    std::int64_t row = 0;  // -1: the chunk's scratch rows
  };
  const std::vector<Saved>& saved() const { return saved_; }

  // The last verify's first `keep` rows (1 to its rows) stay: the rest's
  // ranges (and the scratch rows') are owed back, then the commit.
  Status Accept(std::uint32_t keep);
  // Restores and commits owed, queued at the start of the next job.
  bool owed() const { return restore_count_ != 0 || commit_keep_ != 0; }
  // Queues what is owed; each part stays owed unless queued.
  std::expected<void, kernels::ggml::KernelFailure> QueueOwed(kernels::ggml::LaunchContext& launch);
  // What is owed, now, as a job of its own over `closure`.
  Status Rollback(PagedNode& node, const catalog::Closure& closure, std::uint32_t stream,
                  kernels::ggml::LaunchContext& launch, std::string_view what);

  // After a failed job, so the state is never left half-written: a verify
  // whose saves were queued (`saved`) is undone whole before the next job's
  // own work (nothing committed); anything else that may have written the
  // state (`wrote`), or a launch of unknown effect (`unknown`), quarantines
  // it until Clear. True if a verify was undone.
  bool Settle(bool saved, bool wrote, bool unknown);

  // Every region and the snapshot's descriptors released (the problems
  // appended), the spill file closed, the host copy freed. Once, after the
  // node's teardown.
  void Release(providers::VmmProvider& memory, std::vector<std::string>& problems);

 private:
  struct Region {
    Mapped mapped;
    bool growing = false;
    std::uint64_t bytes = 0;                     // the layout's
    std::vector<scheduler::PageSource> sources;  // registered, by extent
    std::vector<std::uint8_t> used;
  };

  std::string model_;
  std::vector<Region> regions_;
  int spill_fd_ = -1;
  void* host_copy_ = nullptr;
  PagedNode* host_node_ = nullptr;
  std::uint64_t host_capacity_ = 0;
  bool host_copy_unproven_ = false;
  bool quarantined_ = false;

  std::uint64_t snapshot_base_ = 0;
  std::uint64_t snapshot_bytes_ = 0;
  std::uint32_t capacity_ = 0;
  kernels::ggml::RangeCopy* save_ = nullptr;
  kernels::ggml::RangeCopy* restore_ = nullptr;
  std::uint32_t save_count_ = 0;     // the next verify's
  std::uint64_t save_at_ = 0;        // in the snapshot
  std::uint32_t restore_count_ = 0;  // owed
  std::uint32_t commit_keep_ = 0;    // owed: rows to commit (0: none)
  Commit commit_;
  std::vector<Saved> saved_;       // the last verify's
  std::uint32_t verify_rows_ = 0;  // its rows; 0 once accepted
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_LIVE_STATE_H_
