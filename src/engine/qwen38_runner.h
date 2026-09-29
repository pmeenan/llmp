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
//   device reads (ple_rows.h), never a launch parameter. At most kMaxGraphs
//   are kept, the target's and the drafter's together.
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
#include "engine/runner_resources.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/qwen38_commit.h"
#include "kernels/ggml/qwen38_graph.h"
#include "model/qwen38.h"
#include "providers/storage.h"

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
  // The most graphs kept (D-090): with speculation a context window holds
  // a decode step's, a verify's two (with and without its logits' copy) and
  // a draft's four (its catch-up of 1 to 4 rows); two windows' worth, so a
  // step past a 256-cell boundary does not recapture the steps before it.
  static constexpr std::size_t kMaxGraphs = 16;

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
               std::vector<float>* probabilities = nullptr, std::uint32_t passes = 0);
  // A verify: history[n_past, end) the anchor and the drafts (history as
  // Chunk's), at most draft_rows + 1 rows; every row's argmax (the lowest
  // index among equals, on the device) and, with `logits`, every row's
  // logits (rows × vocab; a sampler's or a check's). The next job after one
  // must be preceded by its Accept.
  Status Verify(std::span<const std::int32_t> history, std::uint32_t n_past,
                std::vector<std::int32_t>& argmax, std::vector<float>* logits);
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
  void DropPlans() {
    plans_.Clear();
    mplans_.Clear();
  }
  std::size_t plans() const { return plans_.size(); }
  std::size_t graphs() const { return plans_.graphs() + mplans_.graphs(); }
  double plan_seconds() const { return plan_seconds_; }
  const PleStats& ple() const { return ple_; }
  // Decode graphs on or off for the next chunks; captured graphs are kept.
  void set_graphs(bool on) { runs_.set_graphs(on); }
  const GraphStats& graph_stats() const { return graph_stats_; }

  const catalog::Closure& everything() const { return everything_; }
  // The weight extents (the target's, then the drafter's) and the state's
  // (the target's, then the drafter's).
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const { return live_.extents(); }
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

  Status ReserveWeights(std::vector<std::uint64_t>& stride, std::uint64_t& mtp_stride);
  std::expected<ChunkPlans::Entry*, std::string> Planned(const ChunkKey& key);
  std::expected<MtpPlans::Entry*, std::string> PlannedMtp(
      const kernels::ggml::Qwen38MtpShape& shape);
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
  // The live state's, and the n-gram hash checked and no row reads stalled.
  Status Usable() const;
  // The drafter's shape and pass inputs for `rows` rows from `first` and
  // `passes` - 1 single rows after them.
  std::expected<std::pair<kernels::ggml::Qwen38MtpShape, std::vector<model::Qwen38ChunkInputs>>,
                std::string>
  MtpInputs(std::uint32_t first, std::uint32_t rows, std::uint32_t passes, bool head,
            std::int64_t hidden_row, bool confidence = false) const;

  PagedNode& node_;
  const Qwen38Options& o_;
  int owner_;
  std::uint32_t stream_;
  RunnerResources resources_;
  LiveState live_{"Qwen3.8"};  // the target's state, then the MTP drafter's
  GraphRuns runs_;

  const model::Qwen38Profile& profile_ = model::Qwen38Flash();
  PagedWeights weights_;
  model::Qwen38Binding binding_;        // the artifact's
  model::Qwen38Binding graph_binding_;  // the table's rows the slots'
  model::Qwen38StateLayout layout_;
  model::Qwen38PleHash hash_;
  bool hash_checked_ = false;
  Qwen38Model model_;

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

  catalog::Closure everything_;
  catalog::Closure fence_;  // the state: what a clear or a fence leases

  ChunkPlans plans_{32};  // destroyed before the launch context (Release)
  MtpPlans mplans_{32};
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
  Mapped commit_;
  kernels::ggml::Qwen38CommitArgs commit_args_;  // every place but `keep`, from Bind
  std::uint64_t mtp_base_ = 0;                   // a drafter pass's inputs are staged from here
  kernels::ggml::RangeCopy* carry_ = nullptr;    // a prefill's pending streams row
  std::uint32_t pending_rows_ = 0;               // streams rows the next draft catches up on
  void* drafts_ = nullptr;                       // pinned: a draft's, then a verify's argmaxes
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_QWEN38_RUNNER_H_
