// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash as a model on a paged node (paged_node.h; M3's swap
// path, docs/experiments/fast-swap/swap.md): its v0 prepared artifact
// paged into device VMM through the node's landing zone, each chunk run as
// one device job on the model's own stream under a lease on its whole
// closure (D-086): the request's, held from its start to its end when the
// driver opens one on the model's stream (PagedNode::BeginRequest, M3's
// lease per request, D-093), else the job's own. With the graph, plan and
// kernels of the resident harness (benchmarks/dsv4_exec.cc, via
// dsv4_plan.h). Built on the engine's skeleton (docs/engine.md); what is
// DeepSeek's own:
//
// - Weights (paged_weights.h): each dense group a 2 MiB-aligned region;
//   each layer's routed experts a slab at the resident layout's stride
//   (the stored bytes rounded to 16 and the arrays' block sizes), a page
//   an extent, the slab 256-aligned in its first page; the token table on
//   the host, its chunks read into host VMM directly: the embedding rows
//   are dequantized on the CPU, as llama.cpp looks them up.
// - The state (model/dsv4.h: the window cache, the compressed and indexer
//   caches and the compressor rings, three D-068 representations; the
//   window cache a ring of the window and a chunk in the fast plan, a cell
//   per position in the reference mode and with Dsv4Options::full_window),
//   and the DSpark ring beside it, as live state (live_state.h).
// - The cuBLAS workspace (the router's BF16 products run there at prefill
//   widths); the activations and the GGML pool: the node's shared
//   workspace; the input staging, the logits rows and the hash tables'
//   copy: pinned host memory, cataloged.
// - A chunk: its host-built inputs, the graph planned for its shape, kind
//   and injection (kept per shape, at most kMaxChunkPlans of them across
//   every request slot, the least recently used dropped; DropPlans forgets
//   them, for first-use measurements), and one job that copies the inputs,
//   runs the bound plan under the K-C launch context and copies the last
//   row's logits out. The first chunk of each shape checks every tensor
//   the plan binds against the catalog (BP-A1).
// - Decode graphs (D-090, graph_runs.h), with graphs on: a one-row chunk
//   (or a verify) whose shape has run once launch by launch is captured,
//   the input copies, the plan's 4,972 steps and the logits copy together,
//   and every later chunk of that shape replays it. What varies between
//   steps of a shape (the token, its position, the cache cells, the masks
//   and the compressors' indices) is the inputs' data. Every graph counts
//   toward one cap, the draft blocks' and the waves' too (kMaxGraphs and
//   the draft's with one slot, kMaxWaveGraphs with several), and toward
//   kMaxGraphBytes as counted: a capture past either destroys the least
//   recently used plan's graph. Prefill chunks run launch by launch.
// - The plans' and graphs' host and driver memory, outside the catalog, is
//   bounded by those caps: plan_host_bytes(), measured from the largest
//   plans at Setup, is what the memory guard counts for it.
// - The hash-routed layers' token-to-expert tables, which the kernels index
//   unchecked, are checked after every full load (CheckHashRouting).
// - Speculation (Dsv4Options::drafter; docs/experiments/dspark/): the
//   DSpark drafter's own v0 artifact paged beside the target's, the same
//   way, binding the target's head and token table (model/dspark.h). A
//   chunk of kind kInject (prefill) or kVerify also computes the target's
//   features and injects its last rows into the ring, in the same graph. A
//   verify runs the row-invariant plan in the exact mode (D-092), returns
//   every row's logits, and first saves every state byte it will write (the
//   target's cells, ring rows and compressed rows, the drafter's ring
//   cells: model/dsv4.h Dsv4ChunkWrites, model/dspark.h DsparkWrites) into
//   the snapshot; Accept then owes the rejected rows' bytes and the
//   chunk's scratch rows back, restored before the next job's own work (or
//   at once, with Rollback), which leaves exactly what a verify of the
//   accepted rows alone would have left (D-068 truncation). Draft runs the
//   drafter's block (a job of its own, captured as a graph from its second
//   run) and returns its drafts. DraftVerify chains both in one job, saving
//   a round trip a step: the draft's graph, its drafts copied into the
//   verify's staged tokens and their embedding rows looked up on the
//   device into its staged rows (equal to the host's lookup,
//   CheckDeviceEmbedding), then the verify's graph.
// - Independent requests (Dsv4Options::wave_slots; docs/engine.md,
//   "Independent request state"): up to four request slots, each its own
//   live state (the target's and the DSpark ring), verify snapshot, output
//   staging and plans bound to its state; the weights, workspace, staging
//   and launch context shared (request_cohort.h). A wave (dsv4_graph.h
//   Dsv4WaveGraph) runs one decode step, or each slot's draft and one joined
//   verify, for several slots in one job: every row-local product reads its
//   weights once for all of them.

#ifndef JITLLM_ENGINE_DSV4_RUNNER_H_
#define JITLLM_ENGINE_DSV4_RUNNER_H_

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
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "engine/dsv4_plan.h"
#include "engine/graph_runs.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/planned.h"
#include "engine/request_cohort.h"
#include "engine/runner_resources.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/dspark.h"
#include "model/dsv4.h"

namespace jitllm::engine {

struct Dsv4Options {
  std::filesystem::path artifact;
  std::filesystem::path out;  // the spill file's directory
  std::uint32_t context = 8704;
  std::uint32_t max_rows = 512;
  bool graphs = true;  // decode graphs (D-090); set_graphs changes it between chunks
  // The DSpark drafter's artifact (model/dspark.h); empty: no speculation.
  std::filesystem::path drafter;
  // The most rows a verify takes: the anchor and its drafts.
  std::uint32_t max_verify = 4;
  // The rows of a draft block (the drafts it proposes).
  std::uint32_t draft_rows = 3;
  // The reference mode (dsv4_common.h Dsv4Model::exact): llama.cpp's
  // unfused graph and D-092's row-invariant verify; off, the fast plan.
  bool exact = false;
  // The fast plan keeps its window cache as a ring (model/dsv4.h
  // Dsv4Window::kRing); with this (or exact) the full-size cache, a cell per
  // position, which the reference mode needs: a probe that runs both plans
  // over one state (set_exact) sets it.
  bool full_window = false;
  // Production fast prefill needs only the last head row. Generic and
  // diagnostic callers opt in explicitly; exact/verify stay all-row.
  bool frontier_head = false;
  // Internal experiment, off by default; no public runtime/config option.
  bool prefill_outa_hca = false;
  // Independent request slots (1 to 4): 1 keeps the default request alone,
  // as the harnesses run; more provision that many request states (each
  // its own virtual state ceiling, snapshot and output staging) and waves.
  std::uint32_t wave_slots = 1;
};

// What a chunk computes beside its target rows' own work.
enum class Dsv4ChunkKind : std::uint8_t {
  kPlain,   // the target alone
  kInject,  // and the DSpark drafter's features and injection (a prefill beside it)
  kVerify,  // a speculative verify: every row's logits, the injection, and a
            // snapshot of what it writes
};

class Dsv4Runner final : public PagedModel {
 public:
  using Status = engine::Status;
  // The most decode graphs kept with one request slot (D-090), the draft
  // block's beside them: driver memory outside the catalog, tens of MiB
  // each; capturing another destroys the least recently used.
  static constexpr std::size_t kMaxGraphs = 8;
  // With several request slots: every graph, their decode, verify and
  // draft graphs and the waves', together.
  static constexpr std::size_t kMaxWaveGraphs = 16;
  // And what every graph may hold, as counted (kGraphNodeHostBytes a node
  // its plan launches): a capture past it drops the least recently used,
  // and one past it alone is not made. The measured working set: up to 143
  // MiB at 12 KiB a node in the C4 HTTP cells (3 graphs).
  static constexpr std::uint64_t kMaxGraphBytes = std::uint64_t{256} << 20U;
  // The most chunk plans kept, every slot's together, and the most wave
  // plans: host memory outside the catalog (plan_host_bytes()). The measured
  // working set: up to 26 plans of every kind, 330 MiB, in the C4 HTTP cells.
  static constexpr std::size_t kMaxChunkPlans = 24;
  static constexpr std::size_t kMaxWavePlans = 6;
  static constexpr std::size_t kRequestSlots = RequestCohort::kSlots;
  // A wave's most rows, every slot's together (dsv4_graph.h).
  static constexpr std::int64_t kWaveRows = kernels::ggml::kDsv4WaveRows;

 private:
  static constexpr std::size_t kTarget = 0;  // live state's regions
  static constexpr std::size_t kDrafter = 1;

  // A chunk plan's key: its shape, kind and injected rows.
  struct ChunkKey {
    kernels::ggml::Dsv4ChunkShape shape;
    Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain;
    std::int64_t inject_rows = 0;
    // HCA's host scalar; present only for eligible 4K/256-compressed plans.
    std::optional<std::uint32_t> first_position = std::nullopt;
    bool operator==(const ChunkKey&) const = default;
  };
  using ChunkPlans = PlanCache<ChunkKey, Dsv4Planned>;
  using DraftPlans = PlanCache<std::uint32_t, DsparkPlanned>;  // by the block's rows

  // One conversation's writable native state and the plans bound to it
  // (request_cohort.h). The runner owns it at a stable address through
  // Release; the weights, staging, workspace and launch context stay the
  // runner's. Slot 0 is the default request every scalar method runs on.
  struct RequestState {
    explicit RequestState(std::uint32_t index = 0) : slot(index) {}
    RequestState(const RequestState&) = delete;
    RequestState& operator=(const RequestState&) = delete;
    RequestState(RequestState&&) = delete;
    RequestState& operator=(RequestState&&) = delete;
    ~RequestState() = default;

    LiveState live{"DeepSeek"};        // the target's state, then the DSpark ring
    Dsv4Model model;                   // the shared weights' places, this state's
    DsparkModel dmodel;                // the drafter's, this ring's
    ChunkPlans plans{kMaxChunkPlans};  // destroyed before the launch context (Release)
    DraftPlans dplans{1};
    Mapped snapshot;  // a verify's saves (runtime)
    // Pinned: this slot's scalar logits rows and drafts. A captured graph
    // copies its outputs where it was captured to, so each slot's plans
    // always copy to its own.
    void* logits = nullptr;
    void* drafts = nullptr;
    const std::uint32_t slot;
    bool provisioned = false;  // its live state exists (slot 0 always)
    // The last EnsureState's clean capacity refusal (Slot::state_refused).
    bool state_refused = false;
    catalog::Closure fence;  // its state alone
  };

 public:
  Dsv4Runner(PagedNode& node, const Dsv4Options& options, int owner, std::uint32_t stream)
      : node_(node),
        o_(options),
        owner_(owner),
        stream_(stream),
        resources_(node, owner, stream),
        runs_(options.graphs) {}
  ~Dsv4Runner() override = default;
  Dsv4Runner(const Dsv4Runner&) = delete;
  Dsv4Runner& operator=(const Dsv4Runner&) = delete;
  Dsv4Runner(Dsv4Runner&&) = delete;
  Dsv4Runner& operator=(Dsv4Runner&&) = delete;

  // Before the scheduler exists: the artifact, binding and state layout,
  // the largest chunk shapes measured, the model's own memory mapped
  // (state, cuBLAS workspace, staging) and its weights' places reserved
  // and cataloged.
  Status Setup();
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  // The largest chunk's host-built inputs (model/dsv4.h Dsv4ChunkInputs, the
  // embedding rows), which a chunk allocates on the host beside its staged
  // copy: the bytes the staging is sized for (a bound, not a measurement).
  std::uint64_t host_input_bytes() const { return host_input_bytes_; }
  // After Start, before Run: every weight's and the state's source, their
  // places pinned (D-090).
  Status Register();
  // After the node's workspace, before Run: the closures, the launch
  // context and the registry.
  Status Bind();

  // The default request's (slot 0's) state and steps; Slot has the same
  // for every slot.
  // After Run. The state discarded (zeros on its next use), the drafter's
  // ring too, and any pending restore dropped.
  Status Clear() { return Clear(default_request_); }
  bool state_usable() const { return !cohort_.faulted() && !live_.quarantined(); }
  Status ReserveStateThrough(std::uint32_t positions) {
    return EnsureState(default_request_, positions);
  }
  // After a failed ReserveStateThrough, Chunk, Draft or DraftVerify: true
  // when only the state's growth did not fit the execution budget beside
  // what is leased (LiveState::Use's over_budget). Nothing was dispatched;
  // the state is usable as it was, with any fresh zero pages that completed
  // retained. A later call may succeed once capacity is freed.
  bool state_refused() const { return default_request_.state_refused; }
  std::vector<LiveState::Range> used_state_ranges() const { return live_.used_ranges(); }
  std::uint64_t used_state_bytes() const { return live_.used_bytes(); }
  Status SaveUsedState(void* host, std::span<const LiveState::Range> ranges) {
    return SaveUsedState(default_request_, host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const LiveState::Range> ranges) {
    return RestoreUsedState(default_request_, host, ranges);
  }
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const {
    return CheckpointRanges(default_request_, positions);
  }
  Status PrepareRestoreState(std::span<const LiveState::Range> ranges) {
    return PrepareRestoreState(default_request_, ranges);
  }
  Status CopyCheckpointState(void* host, std::span<const LiveState::Range> ranges, bool to_host) {
    return CopyCheckpointState(default_request_, host, ranges, to_host);
  }
  // One chunk of `tokens` after n_past: the last row's logits in `logits`
  // (every row's, rows × vocab, for kVerify).
  // With frontier_head, fast prefill computes only that row's head;
  // DSpark features/injection still cover the full chunk's streams.
  // With `meanwhile`, the chunk's job is submitted without waiting, and
  // `meanwhile` runs on this thread while it is in flight (the RE-029
  // probe: a page-in beside a long job); its failure fails the chunk.
  // kInject and kVerify need the drafter; a verify's rows must run at the
  // mask widths their one-row steps would (model/dsv4.h Dsv4SameWidths)
  // and number at most max_verify, and the next chunk after one must be
  // preceded by its Accept. A failed verify is undone before the next job;
  // any other failure after the chunk may have written the state
  // quarantines it until Clear.
  Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, const std::function<Status()>& meanwhile = {},
               Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain) {
    return Chunk(default_request_, n_past, tokens, logits, meanwhile, kind);
  }

  // Speculation (Dsv4Options::drafter).
  bool speculative() const { return dweights_.opened(); }
  // After a verify: its first `keep` rows (1 to its rows) stay; the rest,
  // and its scratch rows, are restored from its snapshot at the start of
  // the next job (Rollback runs that now).
  Status Accept(std::uint32_t keep) { return Accept(default_request_, keep); }
  Status Rollback() { return Rollback(default_request_); }
  // The drafter's block of draft_rows rows from pos0 (the anchor's position,
  // the target's state holding every position before it), anchored on
  // `anchor`: each slot's draft.
  Status Draft(std::uint32_t pos0, std::int32_t anchor, std::vector<std::int32_t>& drafts) {
    return Draft(default_request_, pos0, anchor, drafts);
  }
  // Draft and a verify of its first `rows` - 1 drafts in one job: the
  // drafts go from the draft's output into the verify's staged tokens, and
  // their embedding rows are looked up on the device from the host table
  // into its staged rows (which CheckDeviceEmbedding proves equal to the
  // host's), so the host sees the drafts only with the verify's logits.
  Status DraftVerify(std::uint32_t pos, std::int32_t anchor, std::uint32_t rows,
                     std::vector<std::int32_t>& drafts, std::vector<float>& logits) {
    return DraftVerify(default_request_, pos, anchor, rows, drafts, logits);
  }
  // Every token's embedding row looked up on the device equals the host's
  // lookup, bit for bit (a job per 2,048 tokens); the tokens checked.
  std::expected<std::uint64_t, std::string> CheckDeviceEmbedding();
  // The target's state and the drafter's ring, read to the host (a job).
  Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
    return ReadState(default_request_, target, drafter);
  }
  // The last verify's writes (model/dsv4.h Dsv4ChunkWrites, model/dspark.h
  // DsparkWrites): each range's offset in the target's state, or with
  // `ring` in the drafter's ring (as ReadState reads them), its bytes and
  // its row (-1: the chunk's scratch rows). Nothing else of either is
  // written by a verify.
  struct VerifyWrite {
    bool ring = false;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::int64_t row = 0;
  };
  std::vector<VerifyWrite> last_verify_writes() const {
    return last_verify_writes(default_request_);
  }
  // The reference mode on or off for the next chunks (Dsv4Options::exact):
  // every plan and graph dropped. The fast plan runs over either window;
  // the reference mode needs the full one (Dsv4Options::full_window), so
  // over a ring it is refused, and waves refuse it.
  Status set_exact(bool on) {
    if (on && layout_.window != model::Dsv4Window::kFull) {
      return std::unexpected(
          std::string("the reference mode needs the full window cache (Dsv4Options::full_window)"));
    }
    for (RequestState* request : Requests()) {
      request->model.exact = on;
      request->dmodel.exact = on;
    }
    DropPlans();
    return {};
  }
  // Runs of the drafter's block: how they ran, and its last job's host time.
  const GraphStats& draft_stats() const { return draft_stats_; }
  const model::DsparkProfile& dspark_profile() const { return dprofile_; }
  const model::Dsv4StateLayout& state_layout() const { return layout_; }
  const model::Dsv4Profile& profile() const { return profile_; }
  std::uint64_t drafter_read_bytes() const { return dweights_.read_bytes(); }
  std::uint32_t draft_rows() const { return o_.draft_rows; }
  std::uint32_t max_verify() const { return o_.max_verify; }
  // The device's time for `count` replays of the decode graph of the step
  // at n_past, queued back to back in one job with one input (the same
  // token and position each time): the decode step's GPU time without the
  // host's part. Changes the state (clear it after); refused if that shape
  // has no graph yet.
  std::expected<double, std::string> TimeReplays(std::uint32_t n_past, std::int32_t token,
                                                 std::uint32_t count);
  // The hash-routed layers' tables name only experts.
  Status CheckHashRouting();
  // Every weight and state extent is still pinned at the place registered
  // for it, which every captured graph names (D-090); refused, dropping
  // every graph, if one has moved. Between chunks, with the scheduler
  // running.
  Status CheckPlaces();
  // Forgets every planned shape and its graph: the next chunk of each
  // plans it again.
  void DropPlans();
  // A diagnostic: the next chunks' plans keep these llama.cpp callback
  // names alive ("*": every named tensor; empty: none), every plan dropped;
  // DumpLast then reads the last chunk's kept tensors. Turn decode graphs
  // off (set_graphs) while dumping: a replayed graph keeps nothing.
  void set_dump(std::vector<std::string> names) {
    dump_ = std::move(names);
    DropPlans();
  }
  struct Dumped {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    std::array<std::int64_t, 4> ne{};
    std::vector<std::byte> bytes;
  };
  // The last chunk's kept, contiguous named tensors, read to the host (a job).
  Status DumpLast(std::vector<Dumped>& out);
  std::size_t plans() const;
  std::size_t graphs() const;
  // The most host and driver memory the plans and graphs may hold: every
  // cap times its kind's largest plan (PlannedHostBytes) or graph
  // (kGraphNodeHostBytes a node), measured at Setup. The memory guard
  // counts it beside the catalog's budget (Served::plan_host_bytes).
  std::uint64_t plan_host_bytes() const { return plan_host_bytes_; }
  // How plan_host_bytes() is made up, for the start's log.
  const std::string& plan_report() const { return plan_report_; }
  // What the kept plans hold now, as counted (PlannedHostBytes).
  std::uint64_t cached_plan_bytes() const;
  // What the kept graphs hold now, as counted (kGraphNodeHostBytes a node).
  std::uint64_t cached_graph_bytes() const;
  double plan_seconds() const { return plan_seconds_; }  // spent planning, in all
  // Decode graphs on or off for the next chunks; captured graphs are kept.
  void set_graphs(bool on) { runs_.set_graphs(on); }
  const GraphStats& graph_stats() const { return graph_stats_; }
  // Runs of the waves (their graphs are counted in graph_stats too).
  const GraphStats& wave_stats() const { return wave_stats_; }
  // The last chunk: how it ran, and the job's host time (the inputs built
  // and staged, and everything queued, waits for room in the stream
  // included, RE-029).
  RunPath last_path() const { return last_path_; }
  double last_submit_seconds() const { return last_submit_seconds_; }

  // Every extent the model holds: weights, every slot's state, workspace
  // and staging (a swap's load).
  const catalog::Closure& everything() const { return everything_; }
  // What a job leases: the shared extents and the active slots' state
  // (with one request slot, everything).
  const catalog::Closure& execution_closure() const { return execution_; }
  // The weight extents (the target's in file order, its host table among
  // them; then the drafter's), and the state's (every slot's: the target's,
  // then the drafter's ring).
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  bool HasRetainedState() const;
  std::uint64_t weight_read_bytes() const { return weights_.read_bytes() + dweights_.read_bytes(); }
  std::uint64_t state_bytes() const { return layout_.bytes; }
  std::uint64_t state_base() const { return live_.base(kTarget); }
  // The drafter's ring (0 bytes without speculation): with the target's
  // state, the conversation state a check saves and puts back
  // (fence_closure() leases both).
  std::uint64_t drafter_state_base() const { return live_.base(kDrafter); }
  std::uint64_t drafter_state_bytes() const { return live_.bytes(kDrafter); }
  std::uint64_t slab_padding() const { return weights_.slab_padding() + dweights_.slab_padding(); }
  std::uint64_t coverage_tensors() const { return coverage_.tensors; }
  std::uint64_t coverage_violations() const { return coverage_.violations; }
  const std::string& first_violation() const { return coverage_.first_violation; }
  std::uint32_t vocab() const { return profile_.vocab; }
  const artifact::Artifact& artifact() const { return weights_.artifact(); }

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

  // ------------------------------------------------------- request slots

  // One request slot, borrowed from its runner through Release: it cannot
  // move, be built by callers or redirect work to another runner. Its
  // methods are the runner's scalar ones over this slot's state; a slot
  // must be selected (SelectSlots) before its work.
  class Slot final {
   public:
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;
    ~Slot() = default;
    std::uint32_t index() const { return request_.slot; }
    bool state_usable() const { return !owner_.cohort_.faulted() && !request_.live.quarantined(); }
    std::vector<LiveState::Range> used_state_ranges() const { return request_.live.used_ranges(); }
    std::uint64_t used_state_bytes() const { return request_.live.used_bytes(); }
    std::uint64_t state_base() const { return request_.live.base(kTarget); }
    std::uint64_t state_bytes() const { return owner_.layout_.bytes; }
    std::uint64_t drafter_state_base() const { return request_.live.base(kDrafter); }
    std::uint64_t drafter_state_bytes() const { return request_.live.bytes(kDrafter); }
    Status Clear() { return owner_.Clear(request_); }
    // Discards the retained state of a slot outside the selected cohort (an
    // idle conversation's reuse cache), between completed units. Refused
    // for a selected slot, which clears through Clear.
    Status ClearIdle() { return owner_.ClearIdle(request_); }
    Status ReserveStateThrough(std::uint32_t positions) {
      return owner_.EnsureState(request_, positions);
    }
    Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
                 std::vector<float>& logits, Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain) {
      return owner_.Chunk(request_, n_past, tokens, logits, {}, kind);
    }
    Status Draft(std::uint32_t pos0, std::int32_t anchor, std::vector<std::int32_t>& drafts) {
      return owner_.Draft(request_, pos0, anchor, drafts);
    }
    Status DraftVerify(std::uint32_t pos, std::int32_t anchor, std::uint32_t rows,
                       std::vector<std::int32_t>& drafts, std::vector<float>& logits) {
      return owner_.DraftVerify(request_, pos, anchor, rows, drafts, logits);
    }
    // After a failed ReserveStateThrough, Chunk, Draft or DraftVerify: true
    // when it failed only because the state's growth did not fit the
    // execution budget beside what is leased (WorkError::kOverBudget). No
    // graph ran; the state is usable as it was, with any fresh zero pages
    // that completed retained and protected. A later call may succeed once
    // leased state is freed. (ForgetRefusal: before a refusal the caller
    // makes without these.)
    bool state_refused() const { return request_.state_refused; }
    void ForgetRefusal() { request_.state_refused = false; }
    Status Accept(std::uint32_t keep) { return owner_.Accept(request_, keep); }
    // Only after this slot's verify completed and before its Accept: every
    // saved range restored (no row kept), drained before success.
    Status DiscardVerify() { return owner_.DiscardVerify(request_); }
    Status Rollback() { return owner_.Rollback(request_); }
    Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
      return owner_.ReadState(request_, target, drafter);
    }
    std::vector<VerifyWrite> last_verify_writes() const {
      return Dsv4Runner::last_verify_writes(request_);
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
    friend class Dsv4Runner;
    Slot(Dsv4Runner& owner, RequestState& request) : owner_(owner), request_(request) {}
    Dsv4Runner& owner_;
    RequestState& request_;
  };
  // Host-only lookup of a provisioned slot; no admission or native work.
  std::expected<Slot*, std::string> request_slot(std::size_t index);
  // Between completed units only. With several active slots the driver
  // holds one request on this stream over execution_closure(); refreshing
  // an open request keeps their union. A failure faults the cohort. An
  // empty set selects the shared extents alone (a drained retirement).
  Status SelectSlots(std::span<Slot* const> active);
  bool cohort_usable() const { return !cohort_.faulted(); }
  bool waves_provisioned() const { return !released_ && o_.wave_slots > 1; }
  std::uint32_t wave_capacity() const { return waves_provisioned() ? o_.wave_slots : 1; }

  // One wave step of a slot: its anchor (the last sampled token, at `pos`,
  // not yet in its state) and, for a speculative wave, the verify's rows
  // (the anchor and rows - 1 of the draft block's drafts; a plain wave's
  // rows are 1). Outputs are the slot's own: every row's logits (rows ×
  // vocab) and the draft block's drafts. Each must stay alive through the
  // call; they change only once the whole wave completed.
  struct WaveWork {
    Slot* slot = nullptr;
    std::uint32_t pos = 0;
    std::int32_t anchor = 0;
    std::uint32_t rows = 1;
    std::vector<std::int32_t>* drafts = nullptr;
    std::vector<float>* logits = nullptr;
  };
  // Slots active, each once, in ascending order, at most wave_capacity()
  // and kDsv4WaveRows rows in all. A decode wave is one step of every slot
  // (beside a drafter, with its injection, as a prefill's: its ring keeps
  // every position); a speculative wave each slot's draft block and then
  // one joined verify of every slot's rows (each at its steps' mask
  // widths), whose rows then await each slot's own Accept. One job: a
  // refusal before it gives no outputs (state growth may stay); a known
  // failure settles every slot (a verify undone, other writes
  // quarantined); an unknown one faults the cohort.
  Status DecodeWave(std::span<const WaveWork> work);
  Status DraftVerifyWave(std::span<const WaveWork> work);

 private:
  // The wave plans' key: each slot's chunk shape in order, and each its
  // slot index and injected rows, and whether it verifies.
  struct WaveKey {
    kernels::ggml::Dsv4WaveShape shape;
    std::array<std::uint32_t, kRequestSlots> slots{};
    bool verify = false;
    bool operator==(const WaveKey&) const = default;
  };
  using WavePlans = PlanCache<WaveKey, Dsv4WavePlanned>;

  // `part`'s places (paged_weights.h): its dense groups, a slab per layer
  // (`experts` each; the strides in `stride`), and with `table` the token
  // table on the host.
  Status ReservePart(PagedWeights& part, const model::Dsv4Binding& binding, std::uint32_t layers,
                     std::uint32_t experts, bool table, std::vector<std::uint64_t>& stride);
  std::array<RequestState*, kRequestSlots> Requests();
  std::array<const RequestState*, kRequestSlots> Requests() const;
  std::array<LiveState*, kRequestSlots> States();
  Status SetupSlot(RequestState& request, std::uint64_t snapshot_bytes);
  void BindSlot(RequestState& request);
  std::expected<ChunkPlans::Entry*, std::string> Planned(RequestState& request,
                                                         const ChunkKey& key);
  std::expected<DraftPlans::Entry*, std::string> PlannedDraft(RequestState& request);
  std::expected<WavePlans::Entry*, std::string> PlannedWave(const WaveKey& key);
  void Check(const kernels::ggml::Dsv4Graph& graph);
  void CheckWave(const kernels::ggml::Dsv4WaveGraph& graph);
  // After a failed job (LiveState::Settle), with the launch context's
  // fault; an unknown effect faults the cohort.
  void Settle(RequestState& request, bool saved, bool wrote, bool unknown);
  // The host token table, which a job's lease holds resident.
  std::span<const std::byte> table() const;
  // The verify's embedding rows of `n` drafts, looked up on the device
  // from the host table into the staging at `staging_at` (DraftVerify's,
  // and a wave slot's), from `drafts` (a draft plan's output).
  std::expected<ggml_tensor*, std::string> DraftRowsNode(std::uint32_t slot, std::uint32_t n,
                                                         const void* drafts,
                                                         std::uint64_t staging_at);
  // A verify's snapshot: the ranges it writes, saved.
  Status PlanSnapshot(RequestState& request, const model::Dsv4ChunkInputs& in);
  Status RefreshClosures() { return RefreshClosures(cohort_.active()); }
  Status RefreshClosures(std::uint8_t protected_mask);
  // Before `adding` captures of `adding_bytes` (planned.h RoomForGraphs):
  // room under the graph caps (every graph); false if they alone exceed
  // them (not captured then).
  bool RoomForGraphs(std::size_t adding, std::uint64_t adding_bytes);
  // Before planning a chunk shape: room under kMaxChunkPlans, every slot's.
  void RoomForChunkPlan();
  // Where a slot's draft block's inputs are staged (fixed per slot: its
  // captured graph copies from there).
  std::uint64_t DraftStagingAt(const RequestState& request) const {
    return request.slot * draft_staging_;
  }
  Status CheckActive(const RequestState& request) const {
    if (released_) {
      return std::unexpected(std::string("the DeepSeek runner is released"));
    }
    return cohort_.Check(node_, stream_, request.slot);
  }
  Status Usable(const RequestState& request) const {
    if (auto active = CheckActive(request); !active) {
      return active;
    }
    return request.live.Usable();
  }
  Status Wave(std::span<const WaveWork> work, bool speculative);

  Status Clear(RequestState& request);
  Status ClearIdle(RequestState& request);
  Status EnsureState(RequestState& request, std::uint32_t positions);
  Status Chunk(RequestState& request, std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, const std::function<Status()>& meanwhile,
               Dsv4ChunkKind kind);
  Status Accept(RequestState& request, std::uint32_t keep);
  Status DiscardVerify(RequestState& request);
  Status Rollback(RequestState& request);
  Status Draft(RequestState& request, std::uint32_t pos0, std::int32_t anchor,
               std::vector<std::int32_t>& drafts);
  Status DraftVerify(RequestState& request, std::uint32_t pos, std::int32_t anchor,
                     std::uint32_t rows, std::vector<std::int32_t>& drafts,
                     std::vector<float>& logits);
  Status ReadState(RequestState& request, std::vector<std::byte>& target,
                   std::vector<std::byte>& drafter);
  static std::vector<VerifyWrite> last_verify_writes(const RequestState& request);
  Status SaveUsedState(RequestState& request, void* host, std::span<const LiveState::Range> ranges);
  Status RestoreUsedState(RequestState& request, void* host,
                          std::span<const LiveState::Range> ranges);
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      const RequestState& request, std::uint32_t positions) const;
  Status PrepareRestoreState(RequestState& request, std::span<const LiveState::Range> ranges);
  Status CopyCheckpointState(RequestState& request, void* host,
                             std::span<const LiveState::Range> ranges, bool to_host);

  PagedNode& node_;
  const Dsv4Options& o_;
  int owner_;
  std::uint32_t stream_;
  RunnerResources resources_;
  RequestState default_request_;
  std::array<RequestState, kRequestSlots - 1> additional_requests_{
      {RequestState(1), RequestState(2), RequestState(3)}};
  std::array<Slot, kRequestSlots> request_slots_{
      {Slot(*this, default_request_), Slot(*this, additional_requests_[0]),
       Slot(*this, additional_requests_[1]), Slot(*this, additional_requests_[2])}};
  LiveState& live_ = default_request_.live;  // the default request's
  RequestCohort cohort_{"DeepSeek"};
  GraphRuns runs_;

  const model::Dsv4Profile& profile_ = model::Dsv4Flash();
  PagedWeights weights_;
  model::Dsv4Binding binding_;
  model::Dsv4StateLayout layout_;
  Dsv4Model& model_ = default_request_.model;  // the shared weights' places
  std::uint32_t table_group_ = 0;

  void* hash_tables_ = nullptr;  // pinned: the hash-routed layers' tables, read back
  // Pinned: a wave's every row's logits (kDsv4WaveRows × vocab) and its
  // slots' drafts.
  float* wave_logits_ = nullptr;
  std::uint64_t activation_bytes_ = 0;
  std::uint64_t scratch_bytes_ = 0;
  std::uint64_t host_input_bytes_ = 0;
  std::uint64_t snapshot_bytes_ = 0;   // a slot's verify snapshot
  std::uint64_t draft_staging_ = 0;    // a slot's draft inputs' staging stride
  std::uint64_t plan_host_bytes_ = 0;  // plan_host_bytes()
  std::string plan_report_;

  catalog::Closure everything_;
  catalog::Closure execution_;  // the shared extents and the active slots' state
  catalog::Closure fence_;      // every slot's state: what a fence leases
  // A draft's closure: the drafter's weights, the target's head and token
  // table, the active states and their snapshots, the workspace and the
  // staging (a lease of a tenth of `everything_`'s extents).
  catalog::Closure draft_closure_;

  WavePlans waves_{kMaxWavePlans};
  std::vector<std::string> dump_;              // set_dump
  const Dsv4Planned* last_planned_ = nullptr;  // the last chunk's plan (DumpLast)
  GraphStats graph_stats_;
  GraphStats draft_stats_;
  GraphStats wave_stats_;
  RunPath last_path_ = RunPath::kEager;
  double last_submit_seconds_ = 0;
  double plan_seconds_ = 0;
  Coverage coverage_;
  bool released_ = false;

  // The DSpark drafter (Dsv4Options::drafter).
  const model::DsparkProfile& dprofile_ = model::DsparkDeepSeekV4Flash();
  PagedWeights dweights_;
  model::DsparkBinding dbinding_;
  model::DsparkStateLayout dlayout_;
  DsparkModel& dmodel_ = default_request_.dmodel;
  // A verify's inputs are staged from here, a draft's below it (each
  // slot's at its own place), so one job can stage both.
  std::uint64_t verify_base_ = 0;
  // DraftRowsNode's nodes (and CheckDeviceEmbedding's), in their own arena:
  // each slot's, by draft count - 1.
  std::optional<kernels::ggml::TensorArena> rows_arena_;
  ggml_tensor* rows_table_ = nullptr;
  std::array<std::vector<ggml_tensor*>, kRequestSlots> draft_rows_;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_DSV4_RUNNER_H_
