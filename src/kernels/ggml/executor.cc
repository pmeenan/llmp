// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/executor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {
namespace {

// A refused step's last node and its first operand, by name and shape, so
// a refusal at a shape no test planned names what it refused.
std::string Describe(std::span<ggml_tensor* const> nodes) {
  if (nodes.empty() || nodes.back() == nullptr) {
    return {};
  }
  const auto shape = [](const ggml_tensor* t) {
    return std::format("'{}' {} [{}, {}, {}, {}] nb [{}, {}, {}, {}]", ggml_get_name(t),
                       ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0],
                       t->nb[1], t->nb[2], t->nb[3]);
  };
  const ggml_tensor* node = nodes.back();
  std::string out = " (node " + shape(node);
  if (node->src[0] != nullptr) {
    out += ", from " + shape(node->src[0]);
  }
  return out + ")";
}

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

}  // namespace

DeviceChoices DeviceChoicesOf(const LaunchContext& launch) {
  return {
      .mul_mat = [&launch](const ggml_tensor* node) { return SelectMulMat(launch, node); },
      .vector_fusible =
          [&launch](const ggml_tensor* node) { return MulMatVecFusible(launch, node); },
      .quant = [&launch](const ggml_tensor* node) { return SelectMulMatQ(launch, node); },
      .q2_d2r_fits = [&launch](const ggml_tensor* node) { return MulMatIdQ2D2rFits(launch, node); },
      .ds4_hca_fits =
          [&launch](const ggml_tensor* node) { return Dsv4HcaTokentileFits(launch, node); },
      .pair_glu_fits =
          [&launch](const ggml_tensor* up, const ggml_tensor* gate) {
            return MulMatIdQPairGluSupported(launch, up, gate);
          }};
}

std::expected<std::uint64_t, KernelFailure> PlanScratch(const LaunchContext& launch,
                                                        const GraphPlan& plan) {
  std::uint64_t most = 0;
  for (const PlanStep& step : plan.steps) {
    std::expected<std::uint64_t, KernelFailure> planned = 0;
    if (step.implementation == kMulMatCublas) {
      auto cublas = PlanMulMatCublas(launch, step.nodes.front());
      if (!cublas) {
        return std::unexpected(cublas.error());
      }
      planned = cublas->scratch;
    } else if (step.implementation == kMulMatVecQ || step.implementation == kMulMatIdVecQ) {
      planned = PlanMulMatVecQ(launch, step.nodes.front());
    } else if (step.implementation == kMulMatVecQRows || step.implementation == kMulMatIdVecQRows) {
      planned = PlanMulMatVecQRows(launch, step.nodes.front());
    } else if (step.implementation == kMulMatQ || step.implementation == kMulMatIdQ) {
      planned = PlanMulMatQ(launch, step.nodes.front());
    } else if (step.implementation == kMulMatIdQPair ||
               step.implementation == kMulMatIdQPairCompact ||
               step.implementation == kMulMatIdQPairGlu ||
               step.implementation == kMulMatIdQPairGluQ8) {
      planned = PlanMulMatIdQPair(launch, step.nodes[0], step.nodes[1],
                                  step.implementation != kMulMatIdQPair);
    } else if (step.implementation == kMulMatIdQCompact ||
               step.implementation == kMulMatIdQCompactPrequant) {
      planned = PlanMulMatIdQCompact(launch, step.nodes.front());
    } else if (step.implementation == kMulMatIdQ2D2r) {
      planned = PlanMulMatIdQ2D2r(launch, step.nodes.front());
    } else if (step.implementation == kTopKName) {
      planned = PlanTopK(launch, step.nodes.front());
    } else if (step.implementation == kDsv4LidTopKName) {
      planned = PlanDsv4LidTopK(launch, step.nodes.front());
    } else if (step.implementation == kMoeGemmName) {
      planned = PlanMoeGemm(launch, step.nodes.front());
    } else if (step.implementation == kMxfp8GemmName) {
      planned = PlanMxfp8Gemm(launch, step.nodes.front());
    } else if (step.implementation == kQsaTopKName) {
      planned = PlanQsaTopK(step.nodes.front());
    } else if (step.implementation == kQsaAttnName) {
      planned = PlanQsaAttn(step.nodes.front());
    } else if (step.implementation == kDsv4HcaTokentileName) {
      planned = PlanDsv4HcaTokentile(launch, step.nodes.front());
    } else if (step.implementation == kMulMatQPairDense) {
      planned = PlanMulMatQPairDense(launch, step.nodes[0], step.nodes[1]);
    } else if (step.implementation == kDsv4OutAName ||
               step.implementation == kDsv4OutAFastPackName) {
      planned = PlanDsv4OutA(launch, step.nodes.front());
    } else if (step.implementation == kFlashAttnMmaName ||
               step.implementation == kFlashAttnMmaWideName) {
      auto attention = PlanFlashAttnMma(launch, step.nodes.front(),
                                        step.implementation == kFlashAttnMmaWideName);
      if (!attention) {
        return std::unexpected(attention.error());
      }
      planned = attention->scratch;
    }
    if (!planned) {
      return std::unexpected(planned.error());
    }
    most = std::max(most, *planned);
  }
  return most;
}

std::expected<BoundGraph, KernelFailure> BoundGraph::Bind(const execution::Registry& registry,
                                                          const GraphPlan& plan) {
  const std::vector<execution::Choice> choices = plan.Choices();
  const auto built = execution::Plan::Build(registry, choices);
  if (!built) {
    return Rejected(std::format("the plan does not build: {} at step {}", built.error().detail,
                                built.error().operation));
  }
  auto bound = execution::Resolve(registry, *built);
  if (!bound) {
    return Rejected(std::format("the plan does not bind: {} at step {}", bound.error().detail,
                                bound.error().operation));
  }
  std::vector<Step> steps;
  steps.reserve(plan.steps.size());
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const PlanStep& planned = plan.steps[i];
    const execution::Implementation& implementation = bound->at(i);
    std::vector<const ggml_tensor*> view(planned.nodes.begin(), planned.nodes.end());
    if (planned.operation == execution::Operation::kRmsNormMul) {
      auto kernel = RmsNormMulKernel::Bind(implementation);
      if (!kernel) {
        return std::unexpected(kernel.error());
      }
      if (planned.nodes.size() != 2) {
        return Rejected(std::format("step {}: RMSNorm-mul takes two nodes", i));
      }
      if (auto checked = kernel->Check(planned.nodes[0], planned.nodes[1]); !checked) {
        return Rejected(std::format("step {} ({}): {}{}", i, kernel->name(), checked.error().detail,
                                    Describe(planned.nodes)));
      }
      steps.push_back({.kernel = *kernel, .nodes = planned.nodes});
      continue;
    }
    auto kernel = Kernel::Bind(implementation);
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    if (auto checked = kernel->Check(view); !checked) {
      return Rejected(std::format("step {} ({}): {}{}", i, kernel->name(), checked.error().detail,
                                  Describe(planned.nodes)));
    }
    steps.push_back({.kernel = *kernel, .nodes = planned.nodes});
  }
  return BoundGraph(std::move(*bound), std::move(steps));
}

std::expected<void, KernelFailure> BoundGraph::Run(LaunchContext& launch) const {
  for (std::size_t i = 0; i < steps_.size(); ++i) {
    const Step& step = steps_[i];
    std::expected<void, KernelFailure> ran;
    if (const auto* rms = std::get_if<RmsNormMulKernel>(&step.kernel)) {
      ran = rms->Run(launch, step.nodes[0], step.nodes[1]);
    } else {
      ran = std::get<Kernel>(step.kernel).Run(launch, step.nodes);
    }
    if (!ran) {
      return std::unexpected(KernelFailure{
          .error = ran.error().error, .detail = std::format("step {}: {}", i, ran.error().detail)});
    }
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
