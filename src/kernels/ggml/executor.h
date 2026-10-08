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

#include <cstdint>
#include <expected>
#include <utility>
#include <variant>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
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

// A plan whose every step is bound to an implementation of this build.
class BoundGraph {
 public:
  // Resolves `plan` against `registry` and binds each step; refused with
  // the registry's rejection, or if a step's nodes fail its host checks.
  static std::expected<BoundGraph, KernelFailure> Bind(const execution::Registry& registry,
                                                       const GraphPlan& plan);

  // Runs every step in order on the context's stream, a region's lane steps
  // on the context's lanes (graph_plan.h AssignLanes; on the stream if the
  // context has none); stops at the first step refused or faulted
  // (launch.h), which is returned, every lane joined back first.
  std::expected<void, KernelFailure> Run(LaunchContext& launch) const;

  const execution::BoundPlan& bound() const { return bound_; }
  bool has_lanes() const { return !regions_.empty(); }

 private:
  struct Step {
    std::variant<Kernel, RmsNormMulKernel> kernel;
    std::vector<ggml_tensor*> nodes;
    std::uint8_t lane = 0;
  };
  BoundGraph(execution::BoundPlan bound, std::vector<Step> steps,
             std::vector<GraphPlan::Region> regions)
      : bound_(std::move(bound)), steps_(std::move(steps)), regions_(std::move(regions)) {}
  std::expected<void, KernelFailure> RunStep(LaunchContext& launch, std::size_t i) const;
  std::expected<void, KernelFailure> RunLanes(LaunchContext& launch) const;

  execution::BoundPlan bound_;
  std::vector<Step> steps_;
  std::vector<GraphPlan::Region> regions_;
};

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_EXECUTOR_H_
