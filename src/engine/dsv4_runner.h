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
//   and injection (kept per shape; DropPlans forgets them, for first-use
//   measurements), and one job that copies the inputs, runs the bound plan
//   under the K-C launch context and copies the last row's logits out. The
//   first chunk of each shape checks every tensor the plan binds against
//   the catalog (BP-A1).
// - Decode graphs (D-090, graph_runs.h), with graphs on: a one-row chunk
//   (or a verify) whose shape has run once launch by launch is captured,
//   the input copies, the plan's 4,972 steps and the logits copy together,
//   and every later chunk of that shape replays it. What varies between
//   steps of a shape (the token, its position, the cache cells, the masks
//   and the compressors' indices) is the inputs' data. At most kMaxGraphs
//   target graphs are kept (a capture past it destroys the oldest); the
//   draft block's one graph is its own. Prefill chunks run launch by
//   launch.
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

#ifndef JITLLM_ENGINE_DSV4_RUNNER_H_
#define JITLLM_ENGINE_DSV4_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
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
  // The most decode graphs kept (D-090): driver memory outside the
  // catalog, tens of MiB each; capturing another destroys the oldest.
  static constexpr std::size_t kMaxGraphs = 8;

  Dsv4Runner(PagedNode& node, const Dsv4Options& options, int owner, std::uint32_t stream)
      : node_(node),
        o_(options),
        owner_(owner),
        stream_(stream),
        resources_(node, owner, stream),
        runs_(options.graphs) {}

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

  // After Run. The state zeroed (a job leasing it), the drafter's ring
  // too, and any pending restore dropped.
  Status Clear();
  bool state_usable() const { return !live_.quarantined(); }
  Status ReserveStateThrough(std::uint32_t positions) { return EnsureState(positions); }
  std::vector<LiveState::Range> used_state_ranges() const { return live_.used_ranges(); }
  std::uint64_t used_state_bytes() const { return live_.used_bytes(); }
  Status SaveUsedState(void* host, std::span<const LiveState::Range> ranges);
  Status RestoreUsedState(void* host, std::span<const LiveState::Range> ranges);
  // One chunk of `tokens` after n_past: the last row's logits in `logits`
  // (every row's, rows × vocab, for kVerify).
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
               Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain);

  // Speculation (Dsv4Options::drafter).
  bool speculative() const { return dweights_.opened(); }
  // After a verify: its first `keep` rows (1 to its rows) stay; the rest,
  // and its scratch rows, are restored from its snapshot at the start of
  // the next job (Rollback runs that now).
  Status Accept(std::uint32_t keep) { return live_.Accept(keep); }
  Status Rollback();
  // The drafter's block of draft_rows rows from pos0 (the anchor's position,
  // the target's state holding every position before it), anchored on
  // `anchor`: each slot's draft.
  Status Draft(std::uint32_t pos0, std::int32_t anchor, std::vector<std::int32_t>& drafts);
  // Draft and a verify of its first `rows` - 1 drafts in one job: the
  // drafts go from the draft's output into the verify's staged tokens, and
  // their embedding rows are looked up on the device from the host table
  // into its staged rows (which CheckDeviceEmbedding proves equal to the
  // host's), so the host sees the drafts only with the verify's logits.
  Status DraftVerify(std::uint32_t pos, std::int32_t anchor, std::uint32_t rows,
                     std::vector<std::int32_t>& drafts, std::vector<float>& logits);
  // Every token's embedding row looked up on the device equals the host's
  // lookup, bit for bit (a job per 2,048 tokens); the tokens checked.
  std::expected<std::uint64_t, std::string> CheckDeviceEmbedding();
  // The target's state and the drafter's ring, read to the host (a job).
  Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter);
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
  std::vector<VerifyWrite> last_verify_writes() const;
  // The reference mode on or off for the next chunks (Dsv4Options::exact):
  // every plan and graph dropped. The fast plan runs over either window;
  // the reference mode needs the full one (Dsv4Options::full_window), so
  // over a ring it is refused.
  Status set_exact(bool on) {
    if (on && layout_.window != model::Dsv4Window::kFull) {
      return std::unexpected(
          std::string("the reference mode needs the full window cache (Dsv4Options::full_window)"));
    }
    model_.exact = on;
    dmodel_.exact = on;
    DropPlans();
    return {};
  }
  // Runs of the drafter's block: how they ran, and its last job's host time.
  const GraphStats& draft_stats() const { return draft_stats_; }
  const model::DsparkProfile& dspark_profile() const { return dprofile_; }
  const model::Dsv4StateLayout& state_layout() const { return layout_; }
  const model::Dsv4Profile& profile() const { return profile_; }
  std::uint64_t drafter_read_bytes() const { return dweights_.read_bytes(); }
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
  void DropPlans() {
    plans_.Clear();
    dplans_.Clear();
    last_planned_ = nullptr;
  }
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
  std::size_t plans() const { return plans_.size(); }
  std::size_t graphs() const { return plans_.graphs(); }
  double plan_seconds() const { return plan_seconds_; }  // spent planning, in all
  // Decode graphs on or off for the next chunks; captured graphs are kept.
  void set_graphs(bool on) { runs_.set_graphs(on); }
  const GraphStats& graph_stats() const { return graph_stats_; }
  // The last chunk: how it ran, and the job's host time (the inputs built
  // and staged, and everything queued, waits for room in the stream
  // included, RE-029).
  RunPath last_path() const { return last_path_; }
  double last_submit_seconds() const { return last_submit_seconds_; }

  // Every extent a chunk leases: weights, state, workspace and staging.
  const catalog::Closure& everything() const { return everything_; }
  // The weight extents (the target's in file order, its host table among
  // them; then the drafter's), and the state's (the target's, then the
  // drafter's ring).
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const { return live_.extents(); }
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

 private:
  static constexpr std::size_t kTarget = 0;  // live_'s regions
  static constexpr std::size_t kDrafter = 1;

  // A chunk plan's key: its shape, kind and injected rows.
  struct ChunkKey {
    kernels::ggml::Dsv4ChunkShape shape;
    Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain;
    std::int64_t inject_rows = 0;
    bool operator==(const ChunkKey&) const = default;
  };
  using ChunkPlans = PlanCache<ChunkKey, Dsv4Planned>;
  using DraftPlans = PlanCache<std::uint32_t, DsparkPlanned>;  // by the block's rows

  // `part`'s places (paged_weights.h): its dense groups, a slab per layer
  // (`experts` each; the strides in `stride`), and with `table` the token
  // table on the host.
  Status ReservePart(PagedWeights& part, const model::Dsv4Binding& binding, std::uint32_t layers,
                     std::uint32_t experts, bool table, std::vector<std::uint64_t>& stride);
  std::expected<ChunkPlans::Entry*, std::string> Planned(const ChunkKey& key);
  std::expected<DraftPlans::Entry*, std::string> PlannedDraft();
  void Check(const kernels::ggml::Dsv4Graph& graph);
  // After a failed job (LiveState::Settle), with the launch context's fault.
  void Settle(bool saved, bool wrote, bool unknown);
  // The host token table, which a job's lease holds resident.
  std::span<const std::byte> table() const;
  // The verify's embedding rows of `n` drafts, looked up on the device
  // from the host table into the verify's staging (DraftVerify).
  std::expected<ggml_tensor*, std::string> DraftRowsNode(std::uint32_t n, const void* drafts);
  // A verify's snapshot: the ranges it writes, saved.
  Status PlanSnapshot(const model::Dsv4ChunkInputs& in);
  Status RefreshClosures();
  Status EnsureState(std::uint32_t positions);

  PagedNode& node_;
  const Dsv4Options& o_;
  int owner_;
  std::uint32_t stream_;
  RunnerResources resources_;
  LiveState live_{"DeepSeek"};  // the target's state, then the DSpark ring
  GraphRuns runs_;

  const model::Dsv4Profile& profile_ = model::Dsv4Flash();
  PagedWeights weights_;
  model::Dsv4Binding binding_;
  model::Dsv4StateLayout layout_;
  Dsv4Model model_;
  std::uint32_t table_group_ = 0;

  Mapped snapshot_;  // a verify's saves (runtime, pinned)
  void* logits_ = nullptr;
  void* hash_tables_ = nullptr;  // pinned: the hash-routed layers' tables, read back
  std::uint64_t activation_bytes_ = 0;
  std::uint64_t scratch_bytes_ = 0;
  std::uint64_t host_input_bytes_ = 0;

  catalog::Closure everything_;
  catalog::Closure fence_;  // the state: what a clear or a fence leases
  // A draft's closure: the drafter's weights, the target's head and token
  // table, both states and the snapshot, the workspace and the staging (a
  // lease of a tenth of `everything_`'s extents).
  catalog::Closure draft_closure_;

  ChunkPlans plans_{32};  // destroyed before the launch context (Release)
  DraftPlans dplans_{1};
  std::vector<std::string> dump_;              // set_dump
  const Dsv4Planned* last_planned_ = nullptr;  // the last chunk's plan (DumpLast)
  GraphStats graph_stats_;
  GraphStats draft_stats_;
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
  DsparkModel dmodel_;
  void* drafts_ = nullptr;  // pinned: a draft's drafts
  // A verify's inputs are staged from here, a draft's from 0, so one job
  // can stage both.
  std::uint64_t verify_base_ = 0;
  // DraftRowsNode's nodes (and CheckDeviceEmbedding's), in their own arena.
  std::optional<kernels::ggml::TensorArena> rows_arena_;
  std::vector<ggml_tensor*> draft_rows_;  // by draft count - 1
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_DSV4_RUNNER_H_
