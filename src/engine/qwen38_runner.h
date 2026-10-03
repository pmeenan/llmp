// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 Flash Next as a model on a paged node (paged_node.h; M3's swap
// path, docs/experiments/fast-swap/swap.md): its v0 prepared artifact
// paged into device VMM through the node's landing zone, each chunk run as
// one device job on the model's own stream under a lease on its whole
// closure (D-086): the request's, held from its start to its end when the
// driver opens one on the model's stream (PagedNode::BeginRequest, M3's
// lease per request, D-093), else the job's own. With the graph, plan and
// kernels of the resident harness (benchmarks/qwen38_exec.cc, via
// qwen38_plan.h). Built on the engine's skeleton (docs/engine.md), as
// DeepSeek's runner is; what is Qwen3.8's own:
//
// - Weights (paged_weights.h): every dense group but the n-gram table's in
//   a 2 MiB-aligned region, a chunk an extent; each layer's routed experts
//   a slab at the resident layout's stride, an extent a 2 MiB page of it
//   landed in pieces: 2,764,800 bytes for an artifact in the CUTLASS
//   layout, whose slots the grouped GEMM and vector products read as they
//   land (no rewrite), or 2,768,976 for GGML's (mul_mat_id). The slab's
//   offset in its first page is a multiple of 16, the stride's own
//   alignment (the resident harness's expert e is 16-aligned for odd e
//   too): the 80-byte gap between GGML-layout groups cannot hold
//   DeepSeek's 256 where a layer's experts change shard.
// - The n-gram (PLE) table is not paged whole: before each chunk's job the
//   rows its tokens name are read on demand into a pinned landing and the
//   job gathers them into row slots (ple_rows.h), queued between the
//   inputs' copies and the plan (and so captured with it); the graph is
//   built over a copy of the binding whose table has the slots' rows, and
//   the chunk's row indices are the slots'. Only the table's group is left
//   unpaged: a resource sharing it is refused.
// - The state (model/qwen38.h: the QSA layers' K, V and indexer caches, the
//   linear-attention layers' recurrent and convolution state, the n-gram
//   layer's convolution history), and the MTP drafter's beside it, as live
//   state (live_state.h).
// - The n-gram hash's constants are read back and checked
//   (CheckQwen38PleHash) after every full load (ReadPleHash), as DeepSeek's
//   hash-routing tables are.
// - A chunk: its host-built inputs (Qwen38Chunk, over the whole history),
//   its rows read, the graph planned for its shape (kept per shape), and
//   one job that copies the inputs, gathers the rows, runs the bound plan
//   and copies the last row's logits out. The first chunk of each shape
//   checks every tensor the plan binds against the catalog (BP-A1).
// - Decode graphs (D-090, graph_runs.h): a one-row chunk (and a verify, and
//   a draft) whose shape has run once launch by launch is captured and
//   later runs of that shape replay it. The gather's row count is data the
//   device reads (ple_rows.h), never a launch parameter. Every plan and
//   graph, the drafter's and the waves' too, is charged to the node as it
//   is kept and given back through the node's one reclaim order
//   (planned.h); a graph with no room even after a reclaim is not made.
// - An idle slot's state spills to its spill file (Slot::Spill) and comes
//   back before its next work (Slot::Restore), exactly as it left.
// - Speculation (Qwen38Options::drafter; docs/experiments/qwen38-mtp/): the
//   MTP drafter's own v0 artifact paged beside the target's (its dense
//   groups as regions, its experts as a slab), binding the target's token
//   table and head (model/qwen38.h Qwen38MtpBinding). A prefill chunk with
//   `inject` also exports its rows' streams and runs the drafter's pass over
//   the positions whose next token it knows, in the same job. Draft runs the
//   drafter's catch-up over the rows the last verify kept (or the prefill
//   left) and its further passes, one job, returning the drafts. Verify runs
//   the target over the anchor and the drafts in the verify form (every
//   row's logits; the recurrent, convolution and n-gram state read but not
//   written, each row's inputs saved; the KV and indexer cells it writes
//   saved first), and Accept then owes the rejected rows' cells back and
//   the kept rows' commit (qwen38_commit.h, the live state's commit hook),
//   before the next job's own work (or at once, with Rollback): the state a
//   verify of the kept rows alone would have left. A failed verify is
//   undone whole; any other failure after a job may have written the state
//   quarantines it until Clear.

#ifndef JITLLM_ENGINE_QWEN38_RUNNER_H_
#define JITLLM_ENGINE_QWEN38_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "engine/graph_runs.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/planned.h"
#include "engine/ple_rows.h"
#include "engine/qwen38_plan.h"
#include "engine/qwen38_wave_plan.h"
#include "engine/request_cohort.h"
#include "engine/runner_resources.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/qwen38_commit.h"
#include "kernels/ggml/qwen38_graph.h"
#include "model/qwen38.h"
#include "providers/storage.h"

namespace jitllm::benchmarks::qwen_batch {
class Proof;
}

namespace jitllm::engine {

struct Qwen38Options {
  std::filesystem::path artifact;
  std::filesystem::path out;  // the spill file's directory
  std::uint32_t context = 8704;
  std::uint32_t max_rows = 512;
  bool graphs = true;  // decode graphs (D-090); set_graphs changes it between chunks
  // The MTP drafter's artifact (model/qwen38.h Qwen38MtpBinding); empty: no
  // speculation.
  std::filesystem::path drafter;
  // Drafts a step (the drafter's passes) and the draft head's rows (the
  // lowest token IDs or a selected head's ascending IDs; 0: every row).
  // The defaults were the
  // fastest of depths 2–3 and 32,768 rows to the whole vocabulary on
  // `prose` and `code` (docs/experiments/qwen38-mtp/).
  std::uint32_t draft_rows = 2;
  std::uint32_t draft_vocab = 65536;
  // Benchmark-only retained-head copies, bounded at setup; serving leaves
  // this off. Draft's optional capture selects the retained plan per call.
  bool draft_head_capture = false;
  // Benchmark-only: provision retained routed operands for these layers;
  // Verify's optional capture selects its distinct plan per call.
  std::uint64_t routed_capture = 0;
  // Internal opt-in provisioning, before Setup: 1 preserves scalar budgets;
  // 2..request_slots additionally bound decode waves. No runtime/API
  // capability is enabled merely by allocating this storage. Wide/injected
  // prefill stays scalar.
  std::uint32_t wave_slots = 1;
  // The request states provisioned (1 to kMaxRequestSlots, at least
  // wave_slots): each its own virtual state ceilings, verify snapshot and
  // plans. The harnesses keep four; serving sets the model's cap.
  std::uint32_t request_slots = 4;
  // The cells a wave of two or more reads, rounded up to this (a power of
  // two, at least 256: a lone request's alignment); serving sets the
  // model's wave_read_align (runtime/model_settings.h).
  std::uint32_t wave_read_align = 2048;
  // Where each slot's spill file lives (LiveState::SpillPlace), asked once
  // at Register; unset, an unnamed file in `out`. The runtime names them
  // to keep conversations across a restart (D-105).
  std::function<LiveState::SpillPlace(std::uint32_t slot)> spill_place;
};

// Setup-only diagnostic arithmetic. Scalar values use the same measured plan
// maxima with wave_slots=1; provisioned values are the actual allocations.
// Both preserve every provisioned branch's ceilings/snapshots. Virtual
// ceilings do not claim physical backing or simultaneous maximum-context
// admission.
struct Qwen38SetupBudget {
  std::uint64_t scalar_activations = 0;
  std::uint64_t scalar_scratch = 0;
  std::uint64_t scalar_staging_inputs = 0;
  std::uint64_t scalar_host_inputs = 0;
  std::uint64_t scalar_ple_mapped = 0;
  std::uint64_t scalar_pinned = 0;
  std::uint64_t scalar_snapshot_per_branch = 0;
  std::uint64_t activations = 0;
  std::uint64_t scratch = 0;
  std::uint64_t staging_inputs = 0;
  std::uint64_t host_inputs = 0;
  std::uint64_t ple_mapped = 0;
  std::uint64_t pinned = 0;
  std::uint64_t wave_output_pinned = 0;
  std::uint64_t snapshot_per_branch = 0;
  std::uint64_t runner_mapped = 0;
  std::uint64_t target_per_branch = 0;
  std::uint64_t drafter_per_branch = 0;
  std::uint64_t virtual_per_branch = 0;  // extent-rounded target plus drafter
  std::uint64_t initialized_logical = 0;
  std::uint64_t initialized_extent_bytes = 0;
};

struct Qwen38RoutedCapture {
  struct Layer {
    std::uint32_t layer = 0;
    std::vector<float> input, activation, down, shared, gate, weights, combined;
    std::vector<float> attention_input, attention_projection;
    std::vector<std::int32_t> ids;
  };
  std::uint32_t rows = 0;
  std::vector<Layer> layers;
};

struct Qwen38DraftHeadCapture {
  std::uint32_t catch_up_rows = 0;
  std::uint32_t head_rows = 0;
  // Pass-major unrounded F32 mixed inputs [passes, width] and F32 logits
  // [passes, head_rows], copied from runner-owned pinned staging only after
  // the device job completed. These vectors are never DMA destinations.
  std::vector<float> inputs;
  std::vector<float> logits;
  std::vector<std::int32_t> token_ids;  // head rows' validated original token IDs
};

// What the n-gram rows cost, summed over chunks.
struct PleStats {
  std::uint64_t chunks = 0;
  std::uint64_t lookups = 0;
  std::uint64_t rows = 0;          // distinct within each chunk, summed
  std::uint64_t reads = 0;         // direct reads
  std::uint64_t read_bytes = 0;    // their bytes
  std::uint64_t useful_bytes = 0;  // rows x 90
  std::uint64_t extent_bytes = 0;  // what whole 2 MiB chunks would have read
  double seconds = 0;              // planning and reading, on the caller's thread
};

class Qwen38Runner final : public PagedModel {
 public:
  using Status = engine::Status;

  Qwen38Runner(PagedNode& node, const Qwen38Options& options, int owner, std::uint32_t stream)
      : node_(node),
        o_(options),
        owner_(owner),
        stream_(stream),
        resources_(node, owner, stream),
        runs_(options.graphs) {}
  ~Qwen38Runner() override;
  Qwen38Runner(const Qwen38Runner&) = delete;
  Qwen38Runner& operator=(const Qwen38Runner&) = delete;
  Qwen38Runner(Qwen38Runner&&) = delete;
  Qwen38Runner& operator=(Qwen38Runner&&) = delete;

  // Before the scheduler exists: the artifact, binding and state layout,
  // the largest chunk shapes measured, the model's own memory mapped (state,
  // row slots, cuBLAS workspace, staging) and its weights' places reserved
  // and cataloged.
  Status Setup();
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  // The largest chunk's host-built inputs (model/qwen38.h's masks and
  // indices, the embedding rows), which a chunk allocates on the host
  // beside its staged copy: the bytes the staging is sized for (a bound).
  std::uint64_t host_input_bytes() const { return host_input_bytes_; }
  // After Start, before Run: every weight's and the state's source, their
  // places pinned (D-090).
  Status Register();
  // After the node's workspace, before Run: the closures, the launch
  // context and the registry.
  Status Bind();

  // After a full load: the n-gram hash's constants read back and checked.
  Status ReadPleHash();
  // Every weight and state extent is still pinned at the place registered
  // for it (D-090); refused, dropping every graph, if one has moved.
  Status CheckPlaces();
  // Fills the weight extents' unwritten bytes (PagedWeights::Unwritten):
  // the slab pages', the dense chunks' tails, or both.
  Status Scrub(std::uint8_t value, bool slabs, bool dense);
  // The state zeroed (a job leasing it), the drafter's too, and any pending
  // commit dropped.
  Status Clear();
  bool state_usable() const { return !cohort_faulted_ && !live_.quarantined(); }
  Status ReserveStateThrough(std::uint32_t positions) { return EnsureState(positions); }
  std::vector<LiveState::Range> used_state_ranges() const { return live_.used_ranges(); }
  std::uint64_t used_state_bytes() const { return live_.used_bytes(); }
  Status SaveUsedState(void* host, std::span<const LiveState::Range> ranges);
  Status RestoreUsedState(void* host, std::span<const LiveState::Range> ranges);
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const;
  Status PrepareRestoreState(std::span<const LiveState::Range> ranges);
  Status CopyCheckpointState(void* host, std::span<const LiveState::Range> ranges, bool to_host);
  // One chunk: history[n_past, end) after n_past (history holds every token
  // from position 0); the last row's logits in `logits`. With `inject`
  // (speculating), the drafter's streams and its pass over the chunk too.
  Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
               std::vector<float>& logits, bool inject = false);

  // Speculation (Qwen38Options::drafter).
  bool speculative() const { return dweights_.opened(); }
  std::uint32_t draft_rows() const { return o_.draft_rows; }
  // The drafter's catch-up and its passes: `history` every token through
  // the anchor (at position history.size() - 1, not yet in the target's
  // cache); the drafts of the next positions, draft_rows of them, and with
  // `probabilities` each draft's softmax probability over the draft head's
  // rows (the drafter's confidence; an adaptive window's input).
  Status Draft(std::span<const std::int32_t> history, std::vector<std::int32_t>& drafts,
               std::vector<float>* probabilities = nullptr, std::uint32_t passes = 0,
               Qwen38DraftHeadCapture* head_capture = nullptr);
  // A verify: history[n_past, end) the anchor and the drafts (history as
  // Chunk's), at most draft_rows + 1 rows; every row's argmax (the lowest
  // index among equals, on the device) and, with `logits`, every row's
  // logits (rows × vocab; a sampler's or a check's). The next job after one
  // must be preceded by its Accept.
  Status Verify(std::span<const std::int32_t> history, std::uint32_t n_past,
                std::vector<std::int32_t>& argmax, std::vector<float>* logits,
                Qwen38RoutedCapture* routed_capture = nullptr);
  // After a verify: its first `keep` rows (1 to its rows) stay; they are
  // committed and the rest's cells restored before the next job's own work
  // (Rollback runs that now).
  Status Accept(std::uint32_t keep);
  Status Rollback();
  // The target's state and the drafter's, read to the host (a job; any
  // pending commit runs first).
  Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter);
  const GraphStats& draft_stats() const { return draft_stats_; }
  const model::Qwen38MtpState& mtp_state() const { return mtp_layout_; }
  std::uint64_t drafter_read_bytes() const { return dweights_.read_bytes(); }

  // Forgets every planned shape and its graph.
  void DropPlans();
  std::size_t plans() const;
  std::size_t graphs() const;
  // What one step holds at once at most (its plans, PlannedHostBytes; a
  // wave's every slot's and its composition), measured at Setup: a chunk
  // beside a drafter pass, or a target wave beside a draft wave. The memory
  // guard sets it apart beside the catalog's budget (Served::
  // plan_floor_bytes); every plan and graph past it is charged inside the
  // budget (planned.h PlanAccount).
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }
  // How plan_floor_bytes() is made up, for the start's log.
  const std::string& plan_report() const { return plan_report_; }
  // What the kept plans hold now, as counted (PlannedHostBytes).
  std::uint64_t cached_plan_bytes() const;
  // What the kept graphs hold now, as counted (kGraphNodeHostBytes a node).
  std::uint64_t cached_graph_bytes() const;
  // What they took of the device's free memory at their captures.
  std::uint64_t graph_measured_bytes() const;
  // Its plans and graphs as candidates for the node's reclaim order
  // (memory/reclaim.h), and one's reclaim (Dsv4Runner's, the same).
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);
  std::uint64_t reclaimed_plans() const;
  std::uint64_t reclaimed_graphs() const;
  double plan_seconds() const { return plan_seconds_; }
  const PleStats& ple() const { return ple_; }
  // Decode graphs on or off for the next chunks; captured graphs are kept.
  void set_graphs(bool on) { runs_.set_graphs(on); }
  const GraphStats& graph_stats() const { return graph_stats_; }

  const catalog::Closure& everything() const { return everything_; }
  // The weight extents (the target's, then the drafter's) and the state's
  // (the target's, then the drafter's).
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  // A swap's incremental write-back (Dsv4Runner's).
  std::vector<catalog::ExtentId> unchanged_state() const;
  void StateWrittenBack(bool whole);
  std::uint64_t weight_read_bytes() const { return weights_.read_bytes() + dweights_.read_bytes(); }
  std::uint64_t state_bytes() const { return layout_.bytes; }
  std::uint64_t state_base() const { return live_.base(kTarget); }
  // The drafter's state (0 bytes without speculation), and the streams rows
  // its next draft catches up on (host-side): with the target's state, the
  // conversation state a check saves and puts back (fence_closure() leases
  // both regions). set_pending_rows only after Rollback, restoring a value
  // pending_rows() gave for the same state.
  std::uint64_t drafter_state_base() const { return live_.base(kDrafter); }
  std::uint64_t drafter_state_bytes() const { return live_.bytes(kDrafter); }
  std::uint32_t pending_rows() const { return pending_rows_; }
  void set_pending_rows(std::uint32_t rows) { pending_rows_ = rows; }
  std::uint64_t slab_padding() const { return weights_.slab_padding(); }
  std::uint64_t table_bytes() const { return table_.rows * table_.row_bytes; }
  std::uint64_t coverage_tensors() const { return coverage_.tensors; }
  std::uint64_t coverage_violations() const { return coverage_.violations; }
  const std::string& first_violation() const { return coverage_.first_violation; }
  std::uint32_t vocab() const { return profile_.vocab; }
  const artifact::Artifact& artifact() const { return weights_.artifact(); }
  const model::Qwen38StateLayout& state_layout() const { return layout_; }

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  // Private C2 benchmark borrows this one weight/PLE/launch owner; serving
  // never constructs it. Independent state stays in cataloged LiveState slots.
  friend class benchmarks::qwen_batch::Proof;
  static constexpr std::size_t kTarget = 0;  // live_'s regions
  static constexpr std::size_t kDrafter = 1;

  // A chunk plan's key; a verify's second variant copies its argmaxes out
  // without its logits.
  struct ChunkKey {
    kernels::ggml::Qwen38ChunkShape shape;
    Qwen38ChunkKind kind;
    bool operator==(const ChunkKey&) const = default;
  };
  using ChunkPlans = PlanCache<ChunkKey, Qwen38Planned, 2>;
  using MtpPlans = PlanCache<kernels::ggml::Qwen38MtpShape, Qwen38MtpPlanned>;
  static constexpr std::size_t kWithLogits = 0;  // ChunkPlans' variants
  static constexpr std::size_t kLean = 1;

  // One conversation's writable native state and address-bound plans.
  // The runner owns it at a stable address through fenced Release; shared
  // weights, PLE, launch/staging and layout geometry stay on the runner.
  // The aliases below preserve the legacy default request and private
  // benchmark entry points. Every other request has independent addresses.
  struct RequestState {
    explicit RequestState(std::uint32_t index = 0) : slot(index) {}
    RequestState(const RequestState&) = delete;
    RequestState& operator=(const RequestState&) = delete;
    RequestState(RequestState&&) = delete;
    RequestState& operator=(RequestState&&) = delete;
    ~RequestState() = default;

    LiveState live{"Qwen3.8"};  // target, then MTP drafter
    Qwen38Model model;
    ChunkPlans plans;
    MtpPlans mplans;
    // Working state, charged for the model's life and never spilled.
    Mapped commit;
    kernels::ggml::Qwen38CommitArgs commit_args;
    kernels::ggml::RangeCopy* carry = nullptr;
    std::uint32_t pending_rows = 0;
    bool verify_restores_streams = false;
    bool state_refused = false;  // the last EnsureState's (or Restore's) clean capacity refusal
    // Its state written to its spill file and its backing released
    // (Slot::Spill): outside every closure until Restore brings it back.
    bool spilled = false;
    SpillTrack track;  // what its spill file holds (an incremental spill)
    const std::uint32_t slot;
    catalog::Closure fence;
    // A conversation kept from the process before (D-105, Slot::Adopt):
    // the ranges its named spill file holds, spilled until Restore loads
    // them as its state (LiveState::Use reads fresh extents from the file).
    std::vector<LiveState::Range> adopted;
    std::uint64_t adopted_bytes = 0;
  };

 public:
  static constexpr std::size_t kRequestSlots = kMaxRequestSlots;
  static_assert(kQwen38WaveSlots == kRequestSlots);
  // Borrowed from its one runner through fenced Release. A slot cannot move,
  // be constructed by callers, or redirect work to another runner. Scalar
  // prefill writes directly into its own state; shared staging is reused
  // only after each native operation has completed.
  class Slot final {
   public:
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;
    ~Slot() = default;
    std::uint32_t index() const { return request_.slot; }
    bool state_usable() const { return !owner_.cohort_faulted_ && !request_.live.quarantined(); }
    std::vector<LiveState::Range> used_state_ranges() const { return request_.live.used_ranges(); }
    std::uint64_t used_state_bytes() const { return request_.live.used_bytes(); }
    std::uint64_t state_base() const { return request_.live.base(kTarget); }
    std::uint64_t state_bytes() const { return owner_.layout_.bytes; }
    std::uint64_t drafter_state_base() const { return request_.live.base(kDrafter); }
    std::uint64_t drafter_state_bytes() const { return request_.live.bytes(kDrafter); }
    std::uint32_t pending_rows() const { return request_.pending_rows; }
    // Only after completed rollback/restore of this slot's own snapshot.
    void set_pending_rows(std::uint32_t rows) { request_.pending_rows = rows; }
    Status Clear() { return owner_.Clear(request_); }
    // Discards the retained state of a slot outside the selected cohort (an
    // idle conversation's reuse cache), between completed units. Refused
    // for a selected slot, which clears through Clear.
    Status ClearIdle() { return owner_.ClearIdle(request_); }
    // Spills this slot's state to its spill file, between completed units
    // (an idle conversation, or a member set aside for its peers): its
    // backing released, the state exactly as it was once Restore brings it
    // back. Only what changed since the spill file last held the state is
    // written (SpillTrack). Refused while a verify awaits its Accept.
    Status Spill() { return owner_.Spill(request_); }
    // What a Spill now would write (0 while spilled).
    std::uint64_t spill_write_bytes() const { return owner_.SpillWriteBytes(request_); }
    // Brings a spilled slot's state back before its next work: a clean
    // capacity refusal sets state_refused(), the state still spilled.
    Status Restore() { return owner_.Restore(request_); }
    bool spilled() const { return request_.spilled; }
    // Whether the request open on the stream leases its state now.
    bool held() const {
      return (owner_.active_mask_ & (SlotMask{1} << request_.slot)) != 0 &&
             owner_.node_.InRequest(owner_.stream_);
    }
    // What its last refused growth asked for (LiveState::refused_bytes).
    std::uint64_t refused_bytes() const { return request_.live.refused_bytes(); }
    std::uint64_t spilled_bytes() const {
      if (!request_.adopted.empty()) {
        return request_.adopted_bytes;
      }
      return request_.spilled ? request_.live.extents().size() * kPagedExtent : 0;
    }
    // D-105: its state as a kept record describes it. The live state (its
    // regions and spill file), whether a restore or commit is still owed
    // by its last verify (Settle it first: the record must hold the state
    // whole), and whether its spill file holds all of its state, settled,
    // as written last (a spill or a swap's write-back completed).
    const LiveState& live() const { return request_.live; }
    bool owed() const { return request_.live.owed() || request_.live.verify_rows() != 0; }
    bool kept_whole() const {
      return request_.track.on_disk && request_.adopted.empty() && !request_.live.quarantined() &&
             !owed() && !request_.live.extents().empty();
    }
    // Adopts a conversation kept from the process before (D-105): its
    // spill file (registered kept) holds `used`; the slot is spilled until
    // Restore loads them. Before any work on it, after the node runs.
    Status Adopt(std::span<const LiveState::Range> used) { return owner_.Adopt(request_, used); }
    bool adopted() const { return !request_.adopted.empty(); }
    // A decode step's state, its caches backed through the alignment a
    // shared wave reads (DecodeReadAlign): a capacity refusal is then this
    // request's own, here, and never the shared wave's.
    Status ReserveStateThrough(std::uint32_t positions) {
      request_.state_refused = false;
      return owner_.EnsureState(request_, positions, owner_.DecodeReadAlign());
    }
    Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
                 std::vector<float>& logits, bool inject = false) {
      request_.state_refused = false;
      return owner_.Chunk(request_, history, n_past, logits, inject);
    }
    // After a failed ReserveStateThrough or Chunk: true when it failed only
    // because the state's growth did not fit the execution budget beside
    // what is leased (WorkError::kOverBudget). No graph ran; the state is
    // usable as it was, with any fresh zero pages that completed retained
    // and protected. A later call may succeed once leased state is freed.
    bool state_refused() const { return request_.state_refused; }
    // Before a refusal the caller makes without either call.
    void ForgetRefusal() { request_.state_refused = false; }
    Status Draft(std::span<const std::int32_t> history, std::vector<std::int32_t>& drafts,
                 std::vector<float>* probabilities = nullptr, std::uint32_t passes = 0,
                 Qwen38DraftHeadCapture* head_capture = nullptr) {
      return owner_.Draft(request_, history, drafts, probabilities, passes, head_capture);
    }
    Status Verify(std::span<const std::int32_t> history, std::uint32_t n_past,
                  std::vector<std::int32_t>& argmax, std::vector<float>* logits,
                  Qwen38RoutedCapture* routed_capture = nullptr) {
      return owner_.Verify(request_, history, n_past, argmax, logits, routed_capture);
    }
    Status Accept(std::uint32_t keep) { return owner_.Accept(request_, keep); }
    // Only after this slot's Verify completed successfully, before Accept.
    // Undo all saved target writes when host judgement could not select any
    // rows. Drains the restore job before success; a failed return retains
    // the runner's completion-aware ownership obligations.
    Status DiscardVerify() { return owner_.DiscardVerify(request_); }
    Status Rollback() { return owner_.Rollback(request_); }
    Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
      return owner_.ReadState(request_, target, drafter);
    }
    Status SaveUsedState(void* host, std::span<const LiveState::Range> ranges) {
      return owner_.SaveUsedState(request_, host, ranges);
    }
    Status RestoreUsedState(void* host, std::span<const LiveState::Range> ranges) {
      return owner_.RestoreUsedState(request_, host, ranges);
    }
    std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
        std::uint32_t positions) const {
      return owner_.CheckpointRanges(request_, positions);
    }
    Status PrepareRestoreState(std::span<const LiveState::Range> ranges) {
      return owner_.PrepareRestoreState(request_, ranges);
    }
    Status CopyCheckpointState(void* host, std::span<const LiveState::Range> ranges, bool to_host) {
      return owner_.CopyCheckpointState(request_, host, ranges, to_host);
    }

   private:
    friend class Qwen38Runner;
    Slot(Qwen38Runner& owner, RequestState& request) : owner_(owner), request_(request) {}
    Qwen38Runner& owner_;
    RequestState& request_;
  };
  // Host-only lookup; no admission, model activation or native dispatch.
  std::expected<Slot*, std::string> request_slot(std::size_t index);
  // What a kept conversation's record must match to be adopted (D-105):
  // the state format's version, the context, whether the MTP drafter's
  // state is kept beside the target's, and each region's name and bytes.
  // After Setup.
  std::string kept_layout() const;
  // D-102's hang recovery, rung 2: after its stream was fenced (nothing it
  // queued still runs) and with no request open on it, the runner usable
  // again. Every slot the failed work may have touched (resident while the
  // cohort faulted, or quarantined) has its state discarded; a spilled
  // slot's, on disk since before, is kept; the cohort's fault lifts; plans
  // and graphs are dropped. Refused with its launch context faulted (an
  // unknown launch is not proven retired). The slots discarded;
  // `before_discard` (if set) is told each slot before its state changes
  // (its kept record goes first, D-105).
  std::expected<SlotMask, std::string> RecoverInPlace(
      const std::function<void(std::uint32_t slot)>& before_discard = {});
  // Between completed native/copy units only. With several active slots the
  // driver must hold one request on this stream over execution_closure().
  // Refreshing an existing request retains their union; failure stops the
  // entire cohort. Empty selects shared resources alone for drained retirement.
  // Serial fallback selects slot zero before opening its ordinary request.
  Status SelectSlots(std::span<Slot* const> active);
  const catalog::Closure& execution_closure() const { return execution_; }
  bool cohort_usable() const { return !cohort_faulted_; }
  bool HasRetainedState() const;

  // Borrowed descriptors for one synchronous completed native unit. All slots
  // must belong to this runner, be selected, and occur once in ascending order.
  // History, descriptors and caller outputs stay alive through return; DMA
  // uses only the runner's pinned per-slot slices. Outputs change only after
  // the whole unit completed successfully. Cancellation is checked by the
  // driver between units, never by destroying an in-flight slot or plan.
  struct ChunkWork {
    Slot* slot = nullptr;
    std::span<const std::int32_t> history;
    std::uint32_t n_past = 0;
    std::vector<float>* logits = nullptr;
  };
  struct DraftWork {
    Slot* slot = nullptr;
    std::span<const std::int32_t> history;
    std::vector<std::int32_t>* drafts = nullptr;
    std::vector<float>* probabilities = nullptr;
    std::uint32_t passes = 0;  // zero: the configured maximum
  };
  struct VerifyWork {
    Slot* slot = nullptr;
    std::span<const std::int32_t> history;
    std::uint32_t n_past = 0;
    std::vector<std::int32_t>* argmax = nullptr;
    std::vector<float>* logits = nullptr;
  };
  bool waves_provisioned() const { return !released_ && wave_logits_ != nullptr; }
  std::uint32_t wave_capacity() const { return waves_provisioned() ? o_.wave_slots : 1; }
  // The request states Setup provisioned (Qwen38Options::request_slots).
  std::size_t request_slots() const { return slot_count_; }
  // What a slot's state holds once it has run through `positions` (Slot::
  // used_state_bytes then), from empty: an admission's estimate. Host-only.
  std::expected<std::uint64_t, std::string> StateBytesThrough(std::uint32_t positions) const;
  // Read after successful Setup and before node.Start. No native dispatch or
  // catalog access; initialized range accounting is separate from the unused
  // virtual ceilings.
  Qwen38SetupBudget setup_budget() const;
  // One to four rows each (Draft: <=4 pending rows, <=3 passes), up to
  // wave_capacity() slots. Groups of consecutive compatible slots share
  // products of at most sixteen rows (qwen38_wave_plan.h); incompatible
  // slots keep their original operations inside the owned joint plan.
  // paired=false is a paid composition control, not a different math tier.
  // ChunkWave never injects: prefill/injection and diagnostic captures use
  // the existing Slot scalar entry points. Mixed phases form separate units.
  // Status success covers the whole unit. A refusal gives no partial outputs;
  // partial growth may still be retained. A known fenced failure settles all
  // involved states; cohort_usable()/each state_usable() govern subsequent use.
  // A faulted cohort requires completion-aware retirement, retaining its plans
  // and pinned/native owners; an error return is not destruction permission.
  // Successful verify rows await separate per-slot Accept(keep), as scalar.
  Status ChunkWave(std::span<const ChunkWork> work, bool paired = true);
  Status DraftWave(std::span<const DraftWork> work, bool paired = true);
  Status VerifyWave(std::span<const VerifyWork> work, bool paired = true);
  const Qwen38WaveStats& last_wave() const { return last_wave_; }

 private:
  struct TargetWork {
    Slot* slot;
    std::span<const std::int32_t> history;
    std::uint32_t n_past;
    std::vector<std::int32_t>* argmax;
    std::vector<float>* logits;
  };
  struct TargetWaveKey {
    std::array<ChunkKey, kRequestSlots> slots{};
    SlotMask mask = 0;
    SlotMask logits = 0;  // fixed pinned output copy pattern
    bool paired = true;
    bool operator==(const TargetWaveKey&) const = default;
  };
  struct DraftWaveKey {
    std::array<kernels::ggml::Qwen38MtpShape, kRequestSlots> slots{};
    SlotMask mask = 0;
    bool paired = true;
    bool operator==(const DraftWaveKey&) const = default;
  };
  struct WaveCacheOwner {
    std::unique_ptr<Qwen38WavePlanned> plan;
    bool covered = false;  // BP-A1, once per bound plan, as scalar plans
  };
  using TargetWaves = PlanCache<TargetWaveKey, WaveCacheOwner>;
  using DraftWaves = PlanCache<DraftWaveKey, WaveCacheOwner>;
  Status TargetWave(std::span<const TargetWork> work, bool verify, bool paired);
  Status CheckWaveSlot(const Slot* slot, std::uint32_t previous) const;
  Status CheckWave(const Qwen38WavePlanned& planned);
  std::expected<TargetWaves::Entry*, std::string> PlannedWave(const TargetWaveKey& key);
  std::expected<DraftWaves::Entry*, std::string> PlannedWave(const DraftWaveKey& key);
  Status ReadRows(const PleRowPlan& planned, std::size_t lookups);
  Status CheckWaveSources(std::span<const std::pair<ggml_tensor*, const void*>> sources,
                          const Qwen38WavePlanned& planned) const;
  Status CheckWaveOutput(const ggml_tensor* tensor, ggml_type type, std::uint64_t bytes) const;
  // The provisioned request states, slot 0 first.
  std::span<RequestState* const> Requests() { return std::span(requests_).first(slot_count_); }
  std::span<const RequestState* const> Requests() const {
    return std::span(const_requests_).first(slot_count_);
  }
  Status CheckActive(const RequestState& request) const;
  void FaultCohort();
  // Job's Status alone does not distinguish a known fenced refusal from
  // a provider fault retaining an operation. Inspect scheduler/catalog
  // health on their owner thread before preserving a local failure.
  void CheckFailedJob();
  Status BindRequest(RequestState& request);
  Status SetupSnapshot(RequestState& request);
  // Every plan cache: each slot's chunk and drafter plans (every possible
  // slot's, empty unless provisioned), and the waves'.
  std::array<PlanCacheBase*, (2 * kRequestSlots) + 2> PlanCaches();
  std::array<const PlanCacheBase*, (2 * kRequestSlots) + 2> PlanCaches() const;
  // Active, and its state resident (not spilled).
  Status CheckResident(const RequestState& request) const;
  Status Spill(RequestState& request);
  Status Restore(RequestState& request);
  Status Adopt(RequestState& request, std::span<const LiveState::Range> used);
  // An incremental spill's split, and what it would write (Dsv4Runner's).
  void SplitForSpill(const RequestState& request, std::vector<catalog::ExtentId>& written,
                     std::vector<catalog::ExtentId>& unchanged) const;
  std::uint64_t SpillWriteBytes(const RequestState& request) const;
  // The state ranges writes from `positions` on may change (a turn
  // checkpoint's, and an incremental spill's record).
  std::expected<std::vector<LiveState::Range>, std::string> StateWrites(
      std::uint32_t positions) const;
  Status Clear(RequestState& request);
  Status ClearIdle(RequestState& request);
  // The state through `positions`, its caches read through `positions` rounded
  // up to `read_align` (a wave's WaveReadAlign; Qwen38UsedState).
  Status EnsureState(RequestState& request, std::uint32_t positions,
                     std::uint32_t read_align = 256);
  // The most coarsely a decode wave of this runner reads the caches.
  std::uint32_t DecodeReadAlign() const;
  // How coarsely a wave of `slots` requests reads the caches: a lone
  // request 256 cells, a wider wave Qwen38Options::wave_read_align.
  std::uint32_t WaveReadAlign(std::size_t slots) const;
  // The state ranges (the target's and the drafter's) a slot uses through
  // `positions`, its caches read through `read_align`.
  std::expected<std::vector<LiveState::Range>, std::string> StateRanges(
      std::uint32_t positions, std::uint32_t read_align) const;
  Status Chunk(RequestState& request, std::span<const std::int32_t> history, std::uint32_t n_past,
               std::vector<float>& logits, bool inject);
  Status Draft(RequestState& request, std::span<const std::int32_t> history,
               std::vector<std::int32_t>& drafts, std::vector<float>* probabilities,
               std::uint32_t passes, Qwen38DraftHeadCapture* head_capture);
  Status Verify(RequestState& request, std::span<const std::int32_t> history, std::uint32_t n_past,
                std::vector<std::int32_t>& argmax, std::vector<float>* logits,
                Qwen38RoutedCapture* routed_capture);
  Status Accept(RequestState& request, std::uint32_t keep);
  Status DiscardVerify(RequestState& request);
  Status Rollback(RequestState& request);
  Status ReadState(RequestState& request, std::vector<std::byte>& target,
                   std::vector<std::byte>& drafter);
  Status SaveUsedState(RequestState& request, void* host, std::span<const LiveState::Range> ranges);
  Status RestoreUsedState(RequestState& request, void* host,
                          std::span<const LiveState::Range> ranges);
  Status PrepareRestoreState(RequestState& request, std::span<const LiveState::Range> ranges);
  Status CopyCheckpointState(RequestState& request, void* host,
                             std::span<const LiveState::Range> ranges, bool to_host);
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      const RequestState& request, std::uint32_t positions) const;

  Status ReserveWeights(std::vector<std::uint64_t>& stride, std::uint64_t& mtp_stride);
  std::expected<ChunkPlans::Entry*, std::string> Planned(const ChunkKey& key);
  std::expected<ChunkPlans::Entry*, std::string> Planned(RequestState& request,
                                                         const ChunkKey& key);
  std::expected<MtpPlans::Entry*, std::string> PlannedMtp(
      const kernels::ggml::Qwen38MtpShape& shape);
  std::expected<MtpPlans::Entry*, std::string> PlannedMtp(
      RequestState& request, const kernels::ggml::Qwen38MtpShape& shape);
  void Check(const kernels::ggml::Qwen38Graph& graph);
  void CheckMtp(const kernels::ggml::Qwen38MtpGraph& graph);
  // The chunk's n-gram rows planned, read and their slots' sources set.
  std::expected<std::vector<std::int32_t>, std::string> ReadRows(
      const model::Qwen38ChunkInputs& in);
  // The gather of a chunk of `rows` rows' n-gram rows into their slots, a
  // run's work between its inputs and its plan.
  std::function<bool(void* stream)> Gather(std::uint32_t rows);
  // After a failed job (LiveState::Settle), with the launch context's
  // fault: an undone verify leaves no streams rows pending.
  void Settle(bool saved, bool wrote, bool unknown);
  void Settle(RequestState& request, bool saved, bool wrote, bool unknown);
  // The live state's, and the n-gram hash checked and no row reads stalled.
  Status Usable() const;
  Status Usable(const RequestState& request) const;
  Status RefreshClosures();
  Status RefreshClosures(SlotMask protected_mask);
  Status EnsureState(std::uint32_t positions);
  // The drafter's shape and pass inputs for `rows` rows from `first` and
  // `passes` - 1 single rows after them.
  std::expected<std::pair<kernels::ggml::Qwen38MtpShape, std::vector<model::Qwen38ChunkInputs>>,
                std::string>
  MtpInputs(std::uint32_t first, std::uint32_t rows, std::uint32_t passes, bool head,
            std::int64_t hidden_row, bool confidence = false, bool capture_head = false,
            std::uint32_t read_align = 256) const;

  PagedNode& node_;
  const Qwen38Options& o_;
  int owner_;
  std::uint32_t stream_;
  RunnerResources resources_;
  // Every possible slot's host object (each small; only the provisioned
  // ones, Requests(), hold state, snapshots or plans), at stable addresses.
  template <std::size_t... I>
  static std::array<RequestState, sizeof...(I)> MakeRequests(std::index_sequence<I...> /*slots*/) {
    return {{RequestState(static_cast<std::uint32_t>(I + 1))...}};
  }
  template <std::size_t... I>
  std::array<Slot, sizeof...(I) + 1> MakeSlots(std::index_sequence<I...> /*slots*/) {
    return {{Slot(*this, default_request_), Slot(*this, additional_requests_[I])...}};
  }
  template <typename Request, std::size_t... I>
  std::array<Request*, sizeof...(I) + 1> Pointers(std::index_sequence<I...> /*slots*/) {
    return {&default_request_, &additional_requests_[I]...};
  }
  RequestState default_request_;
  std::array<RequestState, kRequestSlots - 1> additional_requests_ =
      MakeRequests(std::make_index_sequence<kRequestSlots - 1>{});
  std::array<Slot, kRequestSlots> request_slots_ =
      MakeSlots(std::make_index_sequence<kRequestSlots - 1>{});
  std::array<RequestState*, kRequestSlots> requests_ =
      Pointers<RequestState>(std::make_index_sequence<kRequestSlots - 1>{});
  std::array<const RequestState*, kRequestSlots> const_requests_ =
      Pointers<const RequestState>(std::make_index_sequence<kRequestSlots - 1>{});
  std::size_t slot_count_ = 1;  // Requests(): Qwen38Options::request_slots from Setup
  LiveState& live_ = default_request_.live;
  GraphRuns runs_;

  const model::Qwen38Profile& profile_ = model::Qwen38Flash();
  PagedWeights weights_;
  model::Qwen38Binding binding_;        // the artifact's
  model::Qwen38Binding graph_binding_;  // the table's rows the slots'
  model::Qwen38StateLayout layout_;
  model::Qwen38PleHash hash_;
  bool hash_checked_ = false;
  Qwen38Model& model_ = default_request_.model;

  // The n-gram rows: the table in its shard, the slots (device), the
  // landing and the slots' sources (pinned), the runner's own ring.
  PleTable table_;
  std::uint64_t slots_ = 0;  // row slots: max_rows x ple_heads
  Mapped slot_memory_;
  std::byte* landing_ = nullptr;
  std::uint64_t landing_bytes_ = 0;
  std::uint32_t* sources_ = nullptr;
  std::uint32_t* ple_count_ = nullptr;  // pinned: the rows the next gather takes
  std::unique_ptr<providers::Storage> ring_;
  bool rows_stalled_ = false;  // reads left in flight: no chunk runs again
  PleStats ple_;

  void* logits_ = nullptr;
  void* hash_host_ = nullptr;       // pinned: the hash constants, read back
  void* draft_ids_host_ = nullptr;  // pinned: selected head's token map, checked after loading
  std::vector<PagedWeights::Range> unwritten_;
  std::uint64_t* scrub_ = nullptr;  // pinned: Scrub's ranges
  std::uint64_t activation_bytes_ = 0;
  std::uint64_t scratch_bytes_ = 0;
  std::uint64_t host_input_bytes_ = 0;
  std::uint64_t plan_floor_bytes_ = 0;  // plan_floor_bytes()
  std::string plan_report_;
  Qwen38SetupBudget setup_budget_;
  // Additional fixed pinned output slices, indexed by sealed slot rather
  // than compact wave order. A graph key fixes the full output copy pattern.
  float* wave_logits_ = nullptr;
  std::int32_t* wave_ids_ = nullptr;
  float* wave_probabilities_ = nullptr;
  std::uint64_t wave_logit_words_ = 0;  // each slot's stride

  catalog::Closure everything_;
  catalog::Closure fence_;       // the state: what a clear or a fence leases
  catalog::Closure execution_;   // shared resources and every active initialized slot
  SlotMask active_mask_ = 1;     // legacy default request until explicitly changed
  bool cohort_faulted_ = false;  // lost protection/unknown shared completion: retirement only

  ChunkPlans& plans_ = default_request_.plans;  // cleared before the launch context (Release)
  MtpPlans& mplans_ = default_request_.mplans;
  TargetWaves target_waves_;
  DraftWaves draft_waves_;
  // What the plans and graphs hold, charged to the node (Bind).
  PlanAccount account_;
  Qwen38WaveStats last_wave_;
  GraphStats graph_stats_;
  GraphStats draft_stats_;
  double plan_seconds_ = 0;
  Coverage coverage_;
  bool released_ = false;

  // The MTP drafter (Qwen38Options::drafter).
  PagedWeights dweights_;
  model::Qwen38MtpBinding dbinding_;
  model::Qwen38MtpState mtp_layout_;
  model::Qwen38CommitLayout commit_layout_;
  // A verify's saves and its cells' snapshot (D-068 working state, charged
  // with the model, never spilled: mapped for the model's life, it stays
  // across a swap, and a commit still pending then runs at the next job).
  Mapped& commit_ = default_request_.commit;
  kernels::ggml::Qwen38CommitArgs& commit_args_ = default_request_.commit_args;
  std::uint64_t mtp_base_ = 0;  // a drafter pass's inputs are staged from here
  kernels::ggml::RangeCopy*& carry_ = default_request_.carry;
  std::uint32_t& pending_rows_ = default_request_.pending_rows;
  void* drafts_ = nullptr;               // pinned: a draft's, then a verify's argmaxes
  float* draft_head_capture_ = nullptr;  // pinned, owned by the node through teardown
  std::uint32_t capture_head_rows_ = 0;
  std::byte* routed_capture_ = nullptr;  // runner-owned pinned DMA staging
  std::uint64_t routed_capture_bytes_ = 0;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_QWEN38_RUNNER_H_
