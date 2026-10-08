// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen2.5-0.5B FP16 fixture as a model on a paged node
// (tests/support/paged_node.h): its v0 prepared artifact paged into device
// VMM through the node's landing zone, each chunk run as a device job on
// the model's own stream under a lease on everything it touches. What
// llmp_fp16_paged (fp16_paged.cc) runs alone, and llmp_alternate_paged
// (alternate_paged.cc) runs beside the EXL3 fixture for BP-S3. CUDA builds
// only.
//
// - Memory, registered in the node's one catalog domain:
//   - every chunk of every group is an extent of device VMM with managed
//     backing (D-033), landed from its shard (D-081): read with O_DIRECT
//     into a landing slot, copied into its place by the copy engine, and
//     published once the copy's fence completes. Group g has a 2 MiB-
//     aligned region; its chunk k maps at the region's base + k x 2 MiB;
//   - the token table's chunks are also extents of host VMM, read there
//     directly: the embedding lookup runs on the CPU, as the bridge's does
//     from its CUDA_Host copy (the tied output head reads the device copy);
//   - the cache and the cuBLAS workspace: device VMM mapped at setup in
//     2 MiB extents; the activations and the GGML pool: the node's shared
//     workspace; the input staging and logits: pinned host memory,
//     cataloged.
// - Rung 4: every weight is loaded before the first evaluation, which runs
//   the trajectory; a second repeats it from a cleared cache.
// - Rung 5 (restores N): N more evaluations, each of which, at the
//   profile's restore point (after 32 tokens for control, 33 for heldout),
//   evicts every weight extent (device and host), releasing its backing,
//   and pages them all back in before continuing. With relocate, every
//   second restore registers the weights at a second reservation, so they
//   come back at other addresses and every chunk's tensors are bound anew
//   (BP-P5). The cache stays resident.
// - BP-P2 (partial): one more evaluation, which at the restore point runs
//   each of paging_cases.h's partial evictions in turn (one layer, side
//   vectors and biases, a shared small-tensor chunk, padded tails, a
//   tensor crossing a chunk boundary): it evicts exactly those extents,
//   checks that a chunk's job submitted without materializing is refused
//   before anything runs (invariants 1-2), and pages only them back in.
// - BP-P4 (spill): one more evaluation, which after the first chunk (a
//   prefill) and again mid-decode (8 tokens after the restore point)
//   writes the cache back through the zone to an unnamed direct-I/O spill
//   file in the output directory and evicts it, then restores it.
//   `premapped` keeps the cache's backing mapped (its eviction is the
//   catalog's alone) and poisons it with 0xff before the restore; `managed`
//   releases the backing (D-033) and the restore maps fresh backing.
//   Either way the cache is copied back to the host before and after, and
//   must match.
// - BP-P3 (shared embeddings): the token table is held once, in device
//   VMM: each chunk's embedding rows are copied from it to pinned staging
//   (a job leasing both) and widened on the host, instead of read from the
//   host copy. The logits must equal the duplicated run's.
// - Explainable plans: for each chunk shape, the guaranteed bound (the
//   placement's activation extent and the plan's scratch bound) against
//   the observed peak (the highest activation byte a bound tensor reaches,
//   and the pool's peak in that chunk).
// - Each chunk is one device job (scheduler::LaunchWork) on the model's
//   stream, holding a lease on the whole closure until the fence after it
//   completes: it builds the embedding rows from the host table, copies the
//   inputs in the bridge's order, runs the plan bound through the registry
//   under the K-C launch context, and copies the logits back. Before each
//   chunk every tensor the plan binds is checked to lie in cataloged
//   extents of device memory, the model's own or the node's shared
//   workspace, of the class it belongs to (BP-A1's in-process check).
// - The cuBLAS handle and its 32 MiB workspace are made current where the
//   bridge creates them: before the first chunk whose plan calls cuBLAS.

#ifndef LLMP_BENCHMARKS_FP16_RUNNER_H_
#define LLMP_BENCHMARKS_FP16_RUNNER_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "execution/registry.h"
#include "fp16_common.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/qwen2_graph.h"
#include "launch_recorder.h"
#include "model/qwen2.h"
#include "paged_node.h"

namespace llmp::benchmarks {

struct Fp16Options {
  std::filesystem::path artifact;
  std::string trajectory;
  std::filesystem::path tokens;
  bool fusion = true;
  std::filesystem::path out;
  int restores = 0;
  bool relocate = false;
  int load_only = 0;
  bool weights_host = false;
  bool premapped = false;
  bool partial = false;
  std::string spill;  // empty, premapped or managed
  bool shared_embeddings = false;
};

class Fp16Runner final : public test_support::PagedModel {
 public:
  using Status = test_support::Status;

  // Model `owner` on the node's compute stream `stream`. With `recording`,
  // each chunk's launches are noted in `record` (plan_compare.py).
  Fp16Runner(test_support::PagedNode& node, const Fp16Options& options, int owner,
             std::uint32_t stream, test_support::Recording* recording, std::string& record)
      : node_(node),
        o_(options),
        owner_(owner),
        stream_(stream),
        recording_(recording),
        record_(record) {}

  // Before the scheduler exists: the artifact, the trajectory, every
  // chunk measured, the model's own memory mapped and its weights' places
  // reserved.
  Status Setup();
  std::uint64_t activations_needed() const { return most_activations_; }
  std::uint64_t pool_needed() const { return most_scratch_; }
  // After Start, before Run: the weights' (and the cache's) sources.
  Status Register();
  // After the node's workspace, before Run: the closures, the launch
  // context and the registry.
  Status Bind();
  // After Run: llmp_fp16_paged's whole run, its outputs written to the
  // output directory.
  Status RunAlone();

  // BP-S3's hooks. One evaluation of the trajectory from a cleared cache
  // (numbered as RunAlone numbers them), its logits in `result`.
  Status Evaluate(int evaluation, std::vector<float>& result);
  // When the last evaluation's first chunk (the prompt's prefill) had its
  // logits back: the first generated token's time (M3's swap runner).
  std::chrono::steady_clock::time_point first_chunk_done() const { return first_chunk_done_; }
  // Everything a chunk leases: every weight, the cache, the workspace and
  // the staging.
  const catalog::Closure& everything() const { return everything_; }
  // The weight extents: the device chunks, then the host table's.
  std::vector<catalog::ExtentId> weights() const;
  // The bytes a load of every weight extent reads.
  std::uint64_t weight_read_bytes() const;
  // Summary and paging.json for `results` (the first is written); an error
  // for any failed check.
  Status Write(const std::vector<std::vector<float>>& results);
  const llmp::benchmarks::Trajectory& trajectory() const { return t_; }

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return cache_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  struct Coverage {
    std::uint64_t tensors = 0;
    std::array<std::uint64_t, catalog::kMemoryClassCount> by_class{};
    std::uint64_t violations = 0;
    std::string first;
  };
  // A chunk shape's guaranteed bound against its observed peak (bytes).
  struct Peak {
    std::uint64_t chunks = 0;
    std::uint64_t activations_bound = 0;
    std::uint64_t activations_seen = 0;
    std::uint64_t scratch_bound = 0;
    std::uint64_t scratch_seen = 0;
  };
  struct PagingEvent {
    std::string what;
    std::uint64_t extents = 0;
    double seconds = 0;
    std::string detail;
  };

  Status ReserveWeights();
  Status Place(std::size_t which);
  std::uint64_t WeightAddress(std::uint32_t resource) const;
  void Check(const kernels::ggml::Qwen2Graph& graph);
  Status Job(const catalog::Closure& closure, scheduler::DeviceJob job, std::string_view what);
  Status Restore(int evaluation);
  Status Partial();
  Status Spill(std::string_view where);
  Status Snapshot(std::vector<std::byte>& out);
  Status GatherRows(std::span<const std::int32_t> tokens, std::vector<float>& embd);
  Status RegisterCache();
  Status WriteLoads();

  test_support::PagedNode& node_;
  const Fp16Options& o_;
  int owner_;
  std::uint32_t stream_;
  test_support::Recording* recording_;
  std::string& record_;

  llmp::benchmarks::Trajectory t_;
  std::unique_ptr<artifact::Artifact> artifact_;
  const model::Qwen2Profile& profile_ = model::Qwen25Instruct05B();
  model::Qwen2Binding binding_;
  std::vector<artifact::FileDescriptor> shards_;

  // Weights: two places (the second for relocation), group regions, one
  // extent per chunk; the token table's host copy.
  std::array<providers::ReservationId, 2> weights_{};
  std::array<std::uint64_t, 2> weights_base_{};
  std::uint64_t weights_bytes_ = 0;
  std::size_t place_ = 0;
  std::vector<std::uint64_t> group_region_;
  std::vector<std::vector<catalog::ExtentId>> chunk_extents_;  // by group, chunk
  std::vector<catalog::ExtentId> device_weights_;
  providers::ReservationId table_;
  std::uint64_t table_base_ = 0;
  std::vector<catalog::ExtentId> table_extents_;
  std::uint64_t stored_bytes_ = 0;
  std::vector<providers::BackingId> premapped_;  // --backing premapped, place 0
  std::array<std::uint8_t, 32> id_{};

  test_support::Mapped kv_;
  test_support::Mapped workspace_;
  std::vector<catalog::ExtentId> staging_;
  void* inputs_ = nullptr;
  void* logits_ = nullptr;
  std::uint64_t input_bytes_ = 0;
  std::uint64_t logits_bytes_ = 0;
  std::uint64_t most_activations_ = 0;
  std::uint64_t most_scratch_ = 0;
  std::uint64_t cublas_bytes_ = 0;
  std::uint32_t most_rows_ = 0;

  std::unique_ptr<kernels::ggml::LaunchContext> launch_;
  std::unique_ptr<kernels::ggml::CublasHandle> cublas_;
  std::unique_ptr<execution::Registry> registry_;
  catalog::Closure everything_;  // what a chunk leases
  catalog::Closure cache_;       // what a cache clear leases

  Coverage coverage_;
  std::vector<test_support::LoadStats> loads_;
  bool released_ = false;

  // --spill: the unnamed spill file, and a pinned copy of the cache.
  int spill_fd_ = -1;
  void* kv_copy_ = nullptr;
  std::uint64_t kv_used_ = 0;  // the cache's bytes (kv_.bytes rounds up)
  // --embeddings shared: pinned staging for a chunk's rows.
  void* rows_ = nullptr;
  std::uint64_t rows_bytes_ = 0;
  catalog::Closure gather_;  // the device table and the staging
  std::vector<PagingEvent> events_;
  std::uint64_t kv_mismatches_ = 0;
  std::uint64_t refusals_ = 0;           // incomplete closures refused before launch
  std::map<std::uint32_t, Peak> peaks_;  // by chunk rows
  std::chrono::steady_clock::time_point first_chunk_done_;
};

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_FP16_RUNNER_H_
