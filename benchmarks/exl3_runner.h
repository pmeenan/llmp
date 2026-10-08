// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// An EXL3 fixture as a model on a paged node (tests/support/paged_node.h):
// its v0 prepared artifact paged into device VMM through the node's landing
// zone, run through the native operation plan (kernels/exl3/qwen2.h) as
// device jobs on the model's own stream, each holding a lease on everything
// it touches. What llmp_exl3_paged (exl3_paged.cc) runs alone, and
// llmp_alternate_paged (alternate_paged.cc) runs beside the FP16 fixture
// for BP-S3. fp16_runner.h's model, for the EXL3 plan. CUDA builds only.
//
// - Memory, registered in the node's one catalog domain:
//   - every chunk of every group is an extent of device VMM with managed
//     backing (D-033), landed from its shard (D-081). Group g has a 2 MiB-
//     aligned region; its chunk k maps at the region's base + k x 2 MiB;
//   - derived at load, device VMM mapped at setup: the norms widened
//     exactly to F32 (weights, pinned), and each layer's multi-GEMM tables
//     (runtime objects), rewritten whenever the weights move (BP-P5);
//   - the cache (live state) and the EXL3 lock area (runtime): device VMM
//     mapped at setup; the activation region and the GGML pool (scratch):
//     the node's shared workspace; the host inputs, the logits' host copy
//     and the derivation's staging: pinned host memory, cataloged.
// - Every phase is planned first, as rung 3's harness plans it
//   (exl3_exec.cc); each is bound (Qwen2Program::Bind) just before it runs,
//   against the weights' current addresses, and every address the bound
//   program reads or writes is checked to lie in cataloged, resident
//   extents of device memory of its class, the model's own or the node's
//   shared workspace (BP-A1's in-process check).
// - The EXL3 launch context is made at setup, before the scheduler runs:
//   it queues its lock area's zeroing on the model's stream, ahead of every
//   job there. The GGML pool's size comes from every phase bound on a
//   planning context (which queues nothing), before the node maps the
//   shared workspace.
// - Rung 4: every weight is loaded before the first evaluation, which runs
//   every prefix's trajectory; a second repeats it.
// - Rung 5 (restores N): N more evaluations, each of which, after every
//   prefix's prefill, evicts every weight extent, releasing its backing,
//   and pages them all back in before the 16 steps. With relocate, every
//   second restore registers the weights at a second reservation, so they
//   come back at other addresses: the tables are rewritten and every phase
//   is bound anew. The cache stays resident.
// - Record (with inline lanes): the first evaluation, as exl3_exec.cc
//   records it (op_plan_compare.py).
// - BP-P2 (partial): one more evaluation, which after the first prefix's
//   prefill runs each of paging_cases.h's partial evictions (one layer,
//   side vectors and biases, the trellis only, a shared small-tensor
//   chunk, padded tails, a tensor crossing a chunk boundary): it evicts
//   exactly those extents, checks that a phase's job submitted without
//   materializing is refused before anything runs (invariants 1-2), and
//   pages only them back in.
// - BP-P4 (spill premapped|managed): one more evaluation, which after
//   every prefix's prefill and again after its eighth step writes the
//   cache back through the zone to an unnamed direct-I/O spill file in the
//   output directory and evicts it, then restores it; `premapped` keeps
//   its backing mapped and poisons it with 0xff first, `managed` releases
//   the backing (D-033). The cache's bytes before and after must match.
// - BP-P3: the head is a representation of its own, never the embedding:
//   checked at setup (distinct resources whose chunks do not overlap).
// - BP-L1 and BP-L3 (cancel in flight, with threads): after the
//   evaluations, the largest reconstruction phase (the 1,023-row prefill,
//   GGML and EXL3 work, each reconstruction slice followed by its GEMM)
//   is submitted behind a gate its job queues first (a stream wait on a
//   host flag), and its request is cancelled once the job has queued
//   everything. While the gate holds, the phase's lease must still hold
//   every extent it touches, the activations and reconstruction scratch
//   among them; the task retires, cancelled, only after the gate opens and
//   the fence completes, and only then are the leases gone.
// - Explainable plans: for each phase kind, the guaranteed bound (the
//   plan's activation region and the GGML pool's bound) against the
//   observed peak (the highest region byte an operation reaches, and the
//   pool's peak in that phase).

#ifndef LLMP_BENCHMARKS_EXL3_RUNNER_H_
#define LLMP_BENCHMARKS_EXL3_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "execution/registry.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/qwen2.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/ggml/launch.h"
#include "launch_recorder.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "paged_node.h"

namespace llmp::benchmarks {

struct Exl3Options {
  std::filesystem::path artifact;
  std::string fixture;
  model::Exl3Arm arm = model::Exl3Arm::kG;
  std::filesystem::path plan;
  std::filesystem::path ids;
  std::filesystem::path out;
  std::vector<int> prefixes = {32, 144, 145, 1023, 1024};
  int restores = 0;
  bool relocate = false;
  bool record = false;
  bool partial = false;
  std::string spill;  // empty, premapped or managed
  bool cancel = false;
};

class Exl3Runner final : public test_support::PagedModel {
 public:
  using Status = test_support::Status;

  Exl3Runner(test_support::PagedNode& node, const Exl3Options& options, int owner,
             std::uint32_t stream)
      : node_(node), o_(options), owner_(owner), stream_(stream) {}

  // Before the scheduler exists: the artifact, every phase planned, the
  // model's own memory mapped, its weights' places reserved, the EXL3
  // launch context made and the GGML pool measured.
  Status Setup();
  std::uint64_t activations_needed() const { return region_bytes_; }
  std::uint64_t pool_needed() const { return pool_bytes_; }
  // After Start, before Run: the weights' (and the cache's) sources.
  Status Register();
  // After the node's workspace, before Run: the closures and the GGML
  // launch context over the shared pool.
  Status Bind();
  // After Run, once every weight is resident: the norms and tables
  // derived from the weights.
  Status Derive();
  // After Run: llmp_exl3_paged's whole run, its outputs written.
  Status RunAlone();

  // BP-S3's hooks. One evaluation of every prefix (numbered as RunAlone
  // numbers them), its logits by prefix in `result`.
  Status Evaluate(int evaluation, std::map<int, std::vector<float>>& result);
  const catalog::Closure& everything() const { return everything_; }
  const std::vector<catalog::ExtentId>& weights() const { return device_weights_; }
  Status Write(const std::vector<std::map<int, std::vector<float>>>& results);

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return cache_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  struct Coverage {
    std::uint64_t ranges = 0;
    std::array<std::uint64_t, catalog::kMemoryClassCount> by_class{};
    std::uint64_t violations = 0;
    std::string first;
  };
  // A phase kind's guaranteed bound against its observed peak (bytes).
  struct Peak {
    std::uint64_t phases = 0;
    std::uint64_t region_bound = 0;
    std::uint64_t region_seen = 0;
    std::uint64_t pool_bound = 0;
    std::uint64_t pool_seen = 0;
  };
  struct PagingEvent {
    std::string what;
    std::uint64_t extents = 0;
    double seconds = 0;
    std::string detail;
  };
  struct PhaseRun {
    int prefix = 0;
    int index = 0;
    model::Exl3PhasePlan plan;
  };

  Status ReserveWeights();
  Status Place(std::size_t which);
  std::uint64_t WeightAddress(std::uint32_t resource) const;
  kernels::exl3::Qwen2Linear Linear(const model::Exl3LinearBinding& l) const;
  void MapWeights();
  void Expect(std::string_view what, std::uint64_t address, std::uint64_t bytes,
              catalog::MemoryClass memory_class);
  void Check(const kernels::exl3::Qwen2Program& program, const PhaseRun& phase);
  Status Job(const catalog::Closure& closure, scheduler::DeviceJob job, std::string_view what);
  Status RunPhase(const PhaseRun& phase, int evaluation, std::vector<float>& out);
  Status Restore(int evaluation);
  Status Partial();
  Status Spill(std::string_view where);
  Status Snapshot(std::vector<std::byte>& out);
  Status RegisterCache();
  Status HeadIsItsOwn() const;
  Status CancelInFlight();

  test_support::PagedNode& node_;
  const Exl3Options& o_;
  int owner_;
  std::uint32_t stream_;
  const model::Qwen2Profile& profile_ = model::Qwen25Instruct05BExl3();
  std::unique_ptr<artifact::Artifact> artifact_;
  model::Exl3Binding binding_;
  std::optional<model::Exl3LaunchTable> table_;
  std::vector<std::int32_t> ids_;
  std::vector<artifact::FileDescriptor> shards_;
  std::vector<PhaseRun> phases_;

  std::array<providers::ReservationId, 2> weights_{};
  std::array<std::uint64_t, 2> weights_base_{};
  std::uint64_t weights_bytes_ = 0;
  std::size_t place_ = 0;
  std::vector<std::uint64_t> group_region_;
  std::vector<std::vector<catalog::ExtentId>> chunk_extents_;
  std::vector<catalog::ExtentId> device_weights_;
  std::uint64_t stored_bytes_ = 0;

  test_support::Mapped kv_;
  test_support::Mapped locks_;
  test_support::Mapped norms_;
  test_support::Mapped tables_;
  std::vector<catalog::ExtentId> staging_;
  void* inputs_ = nullptr;
  void* logits_ = nullptr;
  void* derive_ = nullptr;     // the derivation's staging
  std::vector<void*> pinned_;  // uncataloged: the cache's host copy, the gate
  std::uint64_t inputs_bytes_ = 0;
  std::uint64_t logits_bytes_ = 0;
  std::uint64_t region_bytes_ = 0;
  std::uint64_t pool_bytes_ = 0;
  std::uint64_t kv_bytes_ = 0;

  std::unique_ptr<kernels::ggml::LaunchContext> ggml_;
  std::unique_ptr<kernels::exl3::LaunchContext> launch_;
  std::unique_ptr<kernels::exl3::ReconGemm> gemm_;
  std::unique_ptr<execution::Registry> registry_;
  kernels::exl3::Qwen2Memory memory_map_;
  catalog::Closure everything_;
  catalog::Closure cache_;

  std::unique_ptr<test_support::Recording> recording_;
  std::string record_;
  Coverage coverage_;
  std::vector<test_support::LoadStats> loads_;
  std::map<std::string, std::string> identities_;  // "evaluation/prefix/phase" -> plan identity
  bool released_ = false;

  // --spill: the unnamed spill file, and a pinned copy of the cache.
  int spill_fd_ = -1;
  void* kv_copy_ = nullptr;
  std::vector<PagingEvent> events_;
  std::uint64_t kv_mismatches_ = 0;
  std::uint64_t refusals_ = 0;                 // incomplete closures refused before launch
  std::map<std::pair<int, int>, Peak> peaks_;  // by (rows, padded K)
  std::string cancel_result_;                  // --cancel-in-flight's, as JSON
};

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_EXL3_RUNNER_H_
