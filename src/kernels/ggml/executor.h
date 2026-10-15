// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Running a planned GGML graph (graph_plan.h) under the K-C launch context
// (D-053): the plan is resolved against the implementation registry once
// (execution/registry.h), every step bound to the module's implementation
// it names, identity and all, and then each run walks the steps in order on
// the context's stream. Nothing is substituted: a plan naming an
// implementation this build lacks, or one whose identity changed, binds
// nothing. CUDA builds only.

#ifndef LLMP_KERNELS_GGML_EXECUTOR_H_
#define LLMP_KERNELS_GGML_EXECUTOR_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/selected_operation.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

// The device's part of upstream's choices on the context's device
// (ops.h SelectMulMat and MulMatVecFusible), for PlanGraph. The context
// must outlive the result.
DeviceChoices DeviceChoicesOf(const LaunchContext& launch);

// The most GGML pool scratch any step of `plan` draws at once on the
// context's device: its cuBLAS products' plans (ops.h PlanMulMatCublas),
// and the quantized products', top-k's and tensor-core attention's
// (ops_ext.h). The other implementations draw none.
std::expected<std::uint64_t, KernelFailure> PlanScratch(const LaunchContext& launch,
                                                        const GraphPlan& plan);
// The same over the steps of the concurrent lanes (graph_plan.h AssignLanes),
// each of which draws from a lane's pool; PlanScratch counts the others.
std::expected<std::uint64_t, KernelFailure> PlanLaneScratch(const LaunchContext& launch,
                                                            const GraphPlan& plan);
std::expected<std::uint64_t, KernelFailure> PlanScratchOn(const LaunchContext& launch,
                                                          const GraphPlan& plan, bool lanes);

// SCRATCH DIAGNOSTIC ONLY: an installed observer sees the existing bound
// launches without adding uses or changing placement. Its owning arm and
// paid readback storage survive until submission completion, or forever on
// an unknown result. Install/remove only between completed jobs.
class FirstAttentionObserver {
 public:
  virtual ~FirstAttentionObserver() = default;
  virtual bool active() const = 0;
  virtual std::expected<void, KernelFailure> Observe(LaunchContext& launch,
                                                     std::span<ggml_tensor* const> nodes,
                                                     std::string_view implementation,
                                                     bool after) = 0;
};
bool SetFirstAttentionObserver(FirstAttentionObserver* expected,
                               FirstAttentionObserver* replacement);

// The run-time fusions BoundGraph::Run applies to a spatial or region graph
// (docs/experiments/m36-runtime-fusion/README.md), to be replaced by planner
// patterns: the matchers it tries in order, and the variants they record.
enum class RuntimeFusion : std::uint8_t {
  kSplitBroadcast,         // 0B: two repeated vectors' concat kept as two rows
  kSpatialNorm,            // 0: channel RMS norm (+ SiLU) of a spatial tensor
  kModulate,               // 1A: layer norm · (1 + scale), spatial input
  kModulateRows,           // 1B: the same, rows input
  kNormRope,               // 2: RMS norm · w and up to three rotations
  kGatedResidualModulate,  // 3A with 1A: x + tanh(g) · y, then its modulation
  kGatedResidualRows,      // 3A: x + tanh(g) · y with the branch in rows
  kGatedResidual,          // 3B: x + tanh(g) · y
  kRowsProjections,        // 4A: 1 to 3 products of one rows input
  kSpatialProjections,     // 4B: a product of a spatial input
  kQkvAttention,           // in 4A/4B: QKV products, norm-rope, attention
  kMlp,                    // in 4A/4B: gate/up products, SwiGLU, down
  kConvBias,               // 5: tiled 3x3 convolution and its bias
  kDupUpAdd,               // 6: x plus the duplicate-up of a shortcut
  kConv1x1,                // 7: 1x1 convolution and its bias
  // Variants (counted, not separately controlled).
  kQkvBf16,               // the QKV group with BF16 operands
  kQueryNormInAttention,  // the query's norm and rotation inside attention
  kAttentionBf16Out,      // attention writing the next product's BF16 input
  kMaskPrepacked,         // an attention reusing the previous packed mask
  kReuseConvertedInput,   // a product reusing a BF16 conversion in workspace
  kSkipF32Modulation,     // a modulation writing only its BF16 copy
  kBf16Branch,            // a product writing BF16 for the gated residual
};
inline constexpr std::size_t kRuntimeFusions = 22;
static_assert(kRuntimeFusions == static_cast<std::size_t>(RuntimeFusion::kBf16Branch) + 1,
              "kRuntimeFusions counts every RuntimeFusion");
std::string_view RuntimeFusionName(RuntimeFusion fusion);
// TESTS ONLY: which matchers may apply (a disabled one's steps run as
// planned) and how often each applied. One control at a time, process-wide;
// install/remove only between completed jobs.
struct RuntimeFusionControl {
  std::array<bool, kRuntimeFusions> enabled = [] {
    std::array<bool, kRuntimeFusions> all{};
    all.fill(true);
    return all;
  }();
  std::array<std::atomic<std::uint64_t>, kRuntimeFusions> hits{};
};
// TESTS ONLY: installs `replacement` where `expected` is installed (false
// otherwise); nullptr removes it. Production code never installs one, and
// with none installed every matcher may apply.
bool SetRuntimeFusionControl(RuntimeFusionControl* expected, RuntimeFusionControl* replacement);

// A plan whose every step is bound to an implementation of this build.
class BoundGraph {
 public:
  // Resolves `plan` against `registry` and binds each step; refused with
  // the registry's rejection, or if a step's nodes fail its host checks.
  static std::expected<BoundGraph, KernelFailure> Bind(const execution::Registry& registry,
                                                       const GraphPlan& plan,
                                                       std::span<const SelectedNode> selected = {},
                                                       SelectedLaunch* native = nullptr,
                                                       SelectedWorkspace scratch = {});

  // Runs every step in order on the context's stream, a region's lane steps
  // on the context's lanes (graph_plan.h AssignLanes; on the stream if the
  // context has none); stops at the first step refused or faulted
  // (launch.h), which is returned, every lane joined back first.
  std::expected<void, KernelFailure> Run(LaunchContext& launch) const;

  const execution::BoundPlan& bound() const { return bound_; }
  bool has_lanes() const { return !regions_.empty(); }

 private:
  friend class GgmlStepProgram;
  static std::expected<BoundGraph, KernelFailure> BindImpl(const execution::Registry&,
                                                           const GraphPlan&,
                                                           std::span<const SelectedNode>,
                                                           SelectedLaunch*, SelectedWorkspace,
                                                           bool);
  std::expected<void, KernelFailure> RunRange(LaunchContext&, std::size_t first,
                                              std::size_t end) const;
  bool routed_ = false;
  struct Step {
    std::variant<Kernel, RmsNormMulKernel, std::unique_ptr<BoundSelectedCall>> kernel;
    std::vector<ggml_tensor*> nodes;
    std::uint8_t lane = 0;
  };
  BoundGraph(execution::BoundPlan bound, std::vector<Step> steps,
             std::vector<GraphPlan::Region> regions)
      : bound_(std::move(bound)), steps_(std::move(steps)), regions_(std::move(regions)) {}
  std::expected<void, KernelFailure> RunStep(LaunchContext& launch, std::size_t i) const;
  std::expected<void, KernelFailure> RunLanes(LaunchContext& launch) const;
  std::expected<void, KernelFailure> RunLanesRange(LaunchContext& launch, std::size_t first,
                                                   std::size_t end) const;

  execution::BoundPlan bound_;
  std::vector<Step> steps_;
  std::vector<GraphPlan::Region> regions_;
  bool has_spatial_ops_ = false;
  // The roots of the step's outputs and kept tensors (GraphPlan::kept, from
  // PlanGraph's keep), which Run's run-time matchers must never leave
  // unwritten (until they are planner patterns, whose liveness the
  // placement gives).
  std::vector<const ggml_tensor*> kept_;
};

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_EXECUTOR_H_
