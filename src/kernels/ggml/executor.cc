// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/executor.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {
namespace {
std::atomic<FirstAttentionObserver*> first_attention_observer{nullptr};
}
bool SetFirstAttentionObserver(FirstAttentionObserver* expected,
                               FirstAttentionObserver* replacement) {
  return first_attention_observer.compare_exchange_strong(expected, replacement);
}

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
      .flash_attn_vec256 =
          [&launch](const ggml_tensor* node) { return FlashAttnVec256Selected(launch, node); },
      .vector_fusible =
          [&launch](const ggml_tensor* node) { return MulMatVecFusible(launch, node); },
      .quant = [&launch](const ggml_tensor* node) { return SelectMulMatQ(launch, node); },
      .dense_mmvq_shape =
          [&launch](ggml_type type, std::int64_t columns) {
            return DenseMmvqShapeSelected(launch, type, columns);
          },
      .q2_d2r_fits = [&launch](const ggml_tensor* node) { return MulMatIdQ2D2rFits(launch, node); },
      .ds4_hca_fits =
          [&launch](const ggml_tensor* node) { return Dsv4HcaTokentileFits(launch, node); },
      .pair_glu_fits =
          [&launch](const ggml_tensor* up, const ggml_tensor* gate) {
            return MulMatIdQPairGluSupported(launch, up, gate);
          },
      .quant_glu_fusible =
          [&launch](const ggml_tensor* gate, const ggml_tensor* up, const ggml_tensor* glu) {
            return MulMatVecQGluFusible(launch, gate, up, glu);
          }};
}

std::expected<std::uint64_t, KernelFailure> PlanScratch(const LaunchContext& launch,
                                                        const GraphPlan& plan) {
  return PlanScratchOn(launch, plan, false);
}

std::expected<std::uint64_t, KernelFailure> PlanLaneScratch(const LaunchContext& launch,
                                                            const GraphPlan& plan) {
  return PlanScratchOn(launch, plan, true);
}

std::expected<std::uint64_t, KernelFailure> PlanScratchOn(const LaunchContext& launch,
                                                          const GraphPlan& plan, bool lanes) {
  std::uint64_t most = 0;
  for (const PlanStep& step : plan.steps) {
    if ((step.lane != 0) != lanes) {
      continue;
    }
    std::expected<std::uint64_t, KernelFailure> planned = 0;
    if (step.implementation == kConv2dTiledF32) {
      planned = PlanConv2dTiledF32(launch, step.nodes.front());
    } else if (step.implementation == kMulMatCublas) {
      auto cublas = PlanMulMatCublas(launch, step.nodes.front());
      if (!cublas) {
        return std::unexpected(cublas.error());
      }
      planned = cublas->scratch;
    } else if (step.implementation == kMmvqPreparedName) {
      planned = PlanMmvqPrepared(launch, step.nodes.front());
    } else if (step.implementation == kMulMatVecQ || step.implementation == kMulMatIdVecQ) {
      planned = PlanMulMatVecQ(launch, step.nodes.front());
    } else if (step.implementation == kMulMatGluQFused ||
               step.implementation == kMulMatGeGluQFused) {
      planned = PlanMulMatVecQGlu(launch, step.nodes[0], step.nodes[1], step.nodes[2]);
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
    } else if (step.implementation == kVisibleTopKName) {
      planned = PlanVisibleTopK(launch, step.nodes.front());
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
    } else if (step.implementation == kFlashAttnOwnersName) {
      auto in = FlashAttnOwnersFromNode(step.nodes.front());
      if (!in) return std::unexpected(in.error());
      auto attention = PlanFlashAttnOwners(launch, *in);
      if (!attention) return std::unexpected(attention.error());
      planned = attention->original.scratch;
    } else if (step.implementation == "ggml.flash_attn_ext.vec") {
      auto attention = PlanFlashAttnVec(launch, step.nodes.front());
      if (!attention) return std::unexpected(attention.error());
      planned = attention->scratch;
    } else if (step.implementation == kFlashAttnVec256Name) {
      auto attention = PlanFlashAttnVec256(launch, step.nodes.front());
      if (!attention) return std::unexpected(attention.error());
      planned = attention->scratch;
    } else if (step.implementation == kFlashAttnVecQ8Name ||
               step.implementation == kFlashAttnVecQ4Name) {
      auto attention = PlanFlashAttnVecQuant(launch, step.nodes.front());
      if (!attention) return std::unexpected(attention.error());
      planned = attention->scratch;
    } else if (step.implementation == kAttentionF32) {
      planned = PlanAttentionF32(launch, step.nodes.front());
    } else if (step.implementation == kFlashAttnMmaGqa2Name) {
      auto attention = PlanFlashAttnMmaGqa2(launch, step.nodes.front());
      if (!attention) return std::unexpected(attention.error());
      planned = attention->scratch;
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
                                                          const GraphPlan& plan,
                                                          std::span<const SelectedNode> selected,
                                                          SelectedLaunch* native,
                                                          SelectedWorkspace scratch) {
  return BindImpl(registry, plan, selected, native, scratch, false);
}

std::expected<BoundGraph, KernelFailure> BoundGraph::BindImpl(
    const execution::Registry& registry, const GraphPlan& plan,
    std::span<const SelectedNode> selected, SelectedLaunch* native, SelectedWorkspace scratch,
    bool routed) {
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
  // A resolved registry is immutable for this call. Validate each distinct
  // declaration once; every occurrence still checks its fresh descriptors.
  // The two-word entries borrow wrappers already owned by steps, and fit the
  // runner's per-node binding allowance. Small plans reserve only their bound.
  using CachedWrapper = std::pair<const execution::Implementation*, std::size_t>;
  static_assert(sizeof(CachedWrapper) <= 16);
  std::vector<CachedWrapper> wrappers;
  wrappers.reserve(std::min(plan.steps.size(), execution::Registry::kMaxImplementations));
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const PlanStep& planned = plan.steps[i];
    const execution::Implementation& implementation = bound->at(i);
    const auto external = planned.nodes.size() == 1
                              ? std::ranges::find(selected, planned.nodes[0], &SelectedNode::tensor)
                              : selected.end();
    if (external != selected.end()) {
      if (external->call->route() && !routed)
        return Rejected("routed selected calls require a segmented owned program");
      if (native == nullptr || native->owner() != external->call->owner() || planned.lane != 0 ||
          external->implementation != &implementation || !plan.regions.empty())
        return Rejected("a selected-source call has foreign binding or concurrent lanes");
      auto call = external->call->Bind(implementation, *native, scratch);
      if (!call) return std::unexpected(call.error());
      steps.push_back({.kernel = std::move(*call), .nodes = planned.nodes, .lane = 0});
      continue;
    }
    const auto cached = std::ranges::find(wrappers, &implementation, &CachedWrapper::first);
    std::vector<const ggml_tensor*> view(planned.nodes.begin(), planned.nodes.end());
    if (planned.operation == execution::Operation::kRmsNormMul) {
      auto kernel = cached == wrappers.end()
                        ? RmsNormMulKernel::Bind(implementation)
                        : std::expected<RmsNormMulKernel, KernelFailure>(
                              std::get<RmsNormMulKernel>(steps[cached->second].kernel));
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
      steps.push_back({.kernel = *kernel, .nodes = planned.nodes, .lane = planned.lane});
      if (cached == wrappers.end()) wrappers.emplace_back(&implementation, steps.size() - 1);
      continue;
    }
    auto kernel =
        cached == wrappers.end()
            ? Kernel::Bind(implementation)
            : std::expected<Kernel, KernelFailure>(std::get<Kernel>(steps[cached->second].kernel));
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    if (auto checked = kernel->Check(view); !checked) {
      return Rejected(std::format("step {} ({}): {}{}", i, kernel->name(), checked.error().detail,
                                  Describe(planned.nodes)));
    }
    if (planned.lane != 0 && UsesCublas(kernel->name())) {
      // Its handle and workspace are the context stream's (graph_plan.h
      // AssignLanes): on a lane it would queue on the stream, out of order.
      return Rejected(std::format("step {} ({}) borrows cuBLAS but is on lane {}", i,
                                  kernel->name(), planned.lane));
    }
    steps.push_back({.kernel = *kernel, .nodes = planned.nodes, .lane = planned.lane});
    if (cached == wrappers.end()) wrappers.emplace_back(&implementation, steps.size() - 1);
  }
  BoundGraph result(std::move(*bound), std::move(steps), plan.regions);
  result.routed_ = routed;
  return result;
}

std::expected<void, KernelFailure> BoundGraph::RunStep(LaunchContext& launch, std::size_t i) const {
  const Step& step = steps_[i];
  auto* observer = first_attention_observer.load();
  if (observer != nullptr && !observer->active()) observer = nullptr;
  const auto name = std::visit(
      [](const auto& k) -> std::string_view {
        using T = std::decay_t<decltype(k)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<BoundSelectedCall>>) {
          return "selected";
        } else {
          return k.name();
        }
      },
      step.kernel);
  if (observer != nullptr)
    if (auto observed = observer->Observe(launch, step.nodes, name, false); !observed)
      return observed;
  std::expected<void, KernelFailure> ran;
  if (const auto* rms = std::get_if<RmsNormMulKernel>(&step.kernel)) {
    ran = rms->Run(launch, step.nodes[0], step.nodes[1]);
  } else if (const auto* native = std::get_if<std::unique_ptr<BoundSelectedCall>>(&step.kernel)) {
    ran = (*native)->Run(launch.native_stream());
  } else {
    ran = std::get<Kernel>(step.kernel).Run(launch, step.nodes);
  }
  if (!ran) {
    return std::unexpected(KernelFailure{
        .error = ran.error().error, .detail = std::format("step {}: {}", i, ran.error().detail)});
  }
  if (observer != nullptr)
    if (auto observed = observer->Observe(launch, step.nodes, name, true); !observed)
      return observed;
  return {};
}

namespace {

const ggml_tensor* RootTensor(const ggml_tensor* t) {
  while (t != nullptr && t->view_src != nullptr) {
    t = t->view_src;
  }
  return t;
}

bool DisjointBytes(const ggml_tensor* a, const ggml_tensor* b) {
  if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr) {
    return false;
  }
  const auto a0 = reinterpret_cast<std::uintptr_t>(a->data);
  const auto a1 = a0 + ggml_nbytes(a);
  const auto b0 = reinterpret_cast<std::uintptr_t>(b->data);
  const auto b1 = b0 + ggml_nbytes(b);
  return a1 <= b0 || b1 <= a0;
}

bool IsPermute1203(const ggml_tensor* t) {
  return t != nullptr && t->op == GGML_OP_PERMUTE && t->op_params[0] == 1 && t->op_params[1] == 2 &&
         t->op_params[2] == 0 && t->op_params[3] == 3 && t->src[0] != nullptr;
}

bool IsPermute1023(const ggml_tensor* t) {
  return t != nullptr && t->op == GGML_OP_PERMUTE && t->op_params[0] == 1 && t->op_params[1] == 0 &&
         t->op_params[2] == 2 && t->op_params[3] == 3 && t->src[0] != nullptr;
}

bool IsSpatialFromRowsCont(const ggml_tensor* cont) {
  if (cont == nullptr || cont->op != GGML_OP_CONT || cont->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(cont) || cont->ne[0] != 1 || cont->ne[3] != 1) {
    return false;
  }
  const ggml_tensor* src = cont->src[0];
  if (src == nullptr || src->type != GGML_TYPE_F32) {
    return false;
  }
  const std::int64_t rows = cont->ne[1];
  const std::int64_t cols = cont->ne[2];
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const ggml_tensor* root = RootTensor(src);
  if (root == nullptr || root->type != GGML_TYPE_F32 || !ggml_is_contiguous(root) ||
      ggml_nelements(root) != rows * cols) {
    return false;
  }
  if (src->op == GGML_OP_PERMUTE && src->op_params[0] == 2 && src->op_params[1] == 0 &&
      src->op_params[2] == 1 && src->op_params[3] == 3) {
    return src->nb[1] == static_cast<std::size_t>(cols) * sizeof(float) &&
           src->nb[2] == sizeof(float);
  }
  if ((src->op == GGML_OP_TRANSPOSE ||
       (src->op == GGML_OP_PERMUTE && src->op_params[0] == 1 && src->op_params[1] == 0 &&
        src->op_params[2] == 2 && src->op_params[3] == 3)) &&
      cont->ne[1] == src->ne[0] && cont->ne[2] == src->ne[1]) {
    return src->nb[0] == static_cast<std::size_t>(cols) * sizeof(float) &&
           src->nb[1] == sizeof(float);
  }
  return false;
}

bool DisjointSpan(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
  if (a == nullptr || b == nullptr) return false;
  const auto a0 = reinterpret_cast<std::uintptr_t>(a);
  const auto a1 = a0 + a_bytes;
  const auto b0 = reinterpret_cast<std::uintptr_t>(b);
  const auto b1 = b0 + b_bytes;
  return a1 <= b0 || b1 <= a0;
}

}  // namespace

std::expected<void, KernelFailure> BoundGraph::Run(LaunchContext& launch) const {
  if (routed_) return Rejected("routed graph cannot run through the whole-step entry");
  if (auto* observer = first_attention_observer.load(); observer && observer->active()) {
    if (launch.capturing() || !regions_.empty())
      return std::unexpected(KernelFailure{
          .error = KernelError::kUnknown,
          .detail = "first attention diagnostic requires an eager single-stream graph"});
  }
  if (!regions_.empty() && launch.lanes() >= kMaxLanes) {
    return RunLanes(launch);
  }

  const auto is_kernel = [&](std::size_t idx, std::string_view name) -> ggml_tensor* {
    if (idx >= steps_.size()) return nullptr;
    const auto* k = std::get_if<Kernel>(&steps_[idx].kernel);
    if (k == nullptr || k->name() != name || steps_[idx].nodes.size() != 1) return nullptr;
    return steps_[idx].nodes[0];
  };

  const auto used_after = [&](std::size_t after_idx, const ggml_tensor* target) -> bool {
    if (target == nullptr) return false;
    const ggml_tensor* root_target = RootTensor(target);
    for (std::size_t j = after_idx + 1; j < steps_.size(); ++j) {
      for (const ggml_tensor* node : steps_[j].nodes) {
        if (node == nullptr) continue;
        for (const ggml_tensor* src : node->src) {
          if (src == nullptr) continue;
          if (src == target || src == root_target || RootTensor(src) == root_target) {
            return true;
          }
        }
      }
      if (!steps_[j].nodes.empty()) {
        const ggml_tensor* out = steps_[j].nodes.back();
        if (out != nullptr && out->view_src == nullptr && out->data != nullptr &&
            out->data == root_target->data && ggml_nbytes(out) >= ggml_nbytes(root_target)) {
          return false;
        }
      }
    }
    return false;
  };

  const auto wrap_err =
      [](std::size_t idx,
         std::expected<void, KernelFailure> r) -> std::expected<void, KernelFailure> {
    if (!r) {
      return std::unexpected(KernelFailure{
          .error = r.error().error, .detail = std::format("step {}: {}", idx, r.error().detail)});
    }
    return {};
  };

  struct Rope3dMatch {
    const ggml_tensor* x_in = nullptr;
    const ggml_tensor* weight = nullptr;
    ggml_tensor* last_rope = nullptr;
    float eps = 0.0f;
    float freq_base = 0.0f;
    bool neox = false;
    std::array<const ggml_tensor*, 3> pos_views{};
    std::array<int, 3> axis_dims{};
    std::size_t num_axes = 0;
    std::size_t end_idx = 0;
  };

  const auto match_rope_3d = [&](std::size_t start_idx, Rope3dMatch* out) -> bool {
    ggml_tensor* weight_conv = is_kernel(start_idx, "ggml.convert");
    const std::size_t rms_idx = (weight_conv != nullptr) ? (start_idx + 1) : start_idx;
    if (rms_idx >= steps_.size() ||
        !std::holds_alternative<RmsNormMulKernel>(steps_[rms_idx].kernel) ||
        steps_[rms_idx].nodes.size() != 2) {
      return false;
    }
    ggml_tensor* norm = steps_[rms_idx].nodes[0];
    ggml_tensor* mul = steps_[rms_idx].nodes[1];
    const ggml_tensor* x_in = norm != nullptr ? norm->src[0] : nullptr;
    const ggml_tensor* weight =
        (mul != nullptr && norm != nullptr)
            ? (mul->src[0] == norm ? mul->src[1] : (mul->src[1] == norm ? mul->src[0] : nullptr))
            : nullptr;
    if (weight_conv != nullptr) {
      if (weight != weight_conv || weight_conv->src[0] == nullptr ||
          weight_conv->src[0]->type != GGML_TYPE_BF16 || !ggml_is_contiguous(weight_conv->src[0]) ||
          !ggml_are_same_shape(weight_conv, weight_conv->src[0])) {
        return false;
      }
      weight = weight_conv->src[0];
    }
    if (x_in == nullptr || weight == nullptr || x_in->type != GGML_TYPE_F32 ||
        (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_BF16) ||
        !ggml_is_contiguous(x_in) || !ggml_is_contiguous(weight)) {
      return false;
    }
    const std::int64_t dim = x_in->ne[0];
    const std::int64_t heads = x_in->ne[1];
    const std::int64_t rows = x_in->ne[2];
    if (dim <= 0 || (dim % 2) != 0 || dim > 512 || heads <= 0 || heads > 65535 || rows <= 0 ||
        rows > 65535 || x_in->ne[3] != 1 || weight->ne[0] != dim || weight->ne[1] != 1 ||
        weight->ne[2] != 1 || weight->ne[3] != 1) {
      return false;
    }
    std::array<const ggml_tensor*, 3> pos_views{};
    std::array<int, 3> axis_dims{};
    std::array<const ggml_tensor*, 6> temps{};
    std::size_t num_axes = 0;
    std::int64_t expected_offset = 0;
    float freq_base = 0.0f;
    bool neox = false;
    ggml_tensor* prev_x = mul;
    for (std::size_t a = 0; a < 3; ++a) {
      ggml_tensor* pos_cont = is_kernel(rms_idx + 1 + 2 * a, kContName);
      ggml_tensor* rope = is_kernel(rms_idx + 2 + 2 * a, kRopeExtName);
      if (pos_cont == nullptr || rope == nullptr || pos_cont->src[0] == nullptr) break;
      const ggml_tensor* pos_src = pos_cont->src[0];
      if (pos_src->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos_src) ||
          pos_src->ne[0] != rows || ggml_nelements(pos_src) != rows) {
        break;
      }
      if (rope->op != GGML_OP_ROPE || rope->src[0] != prev_x || rope->src[1] != pos_cont ||
          rope->src[2] != nullptr || rope->type != GGML_TYPE_F32 || !ggml_is_contiguous(rope) ||
          !ggml_are_same_shape(x_in, rope)) {
        break;
      }
      const int n_dims = rope->op_params[1];
      const int mode = rope->op_params[2];
      const int offset = rope->op_params[15];
      float f_base = 0.0f;
      float f_scale = 0.0f;
      float ext_factor = 0.0f;
      float attn_factor = 0.0f;
      std::memcpy(&f_base, &rope->op_params[5], sizeof(float));
      std::memcpy(&f_scale, &rope->op_params[6], sizeof(float));
      std::memcpy(&ext_factor, &rope->op_params[7], sizeof(float));
      std::memcpy(&attn_factor, &rope->op_params[8], sizeof(float));
      if (n_dims <= 0 || (n_dims % 2) != 0 ||
          static_cast<std::int64_t>(offset) != expected_offset ||
          (mode != 0 && mode != GGML_ROPE_TYPE_NEOX) || f_scale != 1.0f || ext_factor != 0.0f ||
          attn_factor != 1.0f) {
        break;
      }
      const bool step_neox = (mode == GGML_ROPE_TYPE_NEOX);
      if (a == 0) {
        freq_base = f_base;
        neox = step_neox;
      } else if (f_base != freq_base || step_neox != neox) {
        break;
      }
      pos_views[a] = pos_src;
      axis_dims[a] = n_dims;
      temps[2 * a] = prev_x;
      temps[2 * a + 1] = pos_cont;
      expected_offset += n_dims;
      prev_x = rope;
      ++num_axes;
      if (expected_offset == dim) break;
    }
    if (num_axes == 0 || expected_offset != dim) return false;
    const std::size_t end_idx = rms_idx + 2 * num_axes;
    if (weight_conv != nullptr && used_after(end_idx, weight_conv)) return false;
    for (std::size_t t_idx = 0; t_idx < 2 * num_axes; ++t_idx) {
      if (used_after(end_idx, temps[t_idx])) return false;
    }
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(float));
    *out = Rope3dMatch{
        .x_in = x_in,
        .weight = weight,
        .last_rope = prev_x,
        .eps = eps,
        .freq_base = freq_base,
        .neox = neox,
        .pos_views = pos_views,
        .axis_dims = axis_dims,
        .num_axes = num_axes,
        .end_idx = end_idx,
    };
    return true;
  };

  const ggml_tensor* ws_bf16_spatial = nullptr;
  const ggml_tensor* ws_bf16_rows = nullptr;
  std::int64_t ws_bf16_ne0 = 0;
  std::int64_t ws_bf16_ne1 = 0;
  const ggml_tensor* direct_bf16_rows = nullptr;
  const ggml_tensor* bf16_branch_tensor = nullptr;
  const ggml_tensor* rows_layout_tensor = nullptr;
  bool rows_layout_is_bf16 = false;
  const ggml_tensor* cached_attn_mask = nullptr;
  std::int64_t cached_attn_rows = 0;
  std::int64_t cached_attn_heads_k = 0;
  std::int64_t cached_attn_dim = 0;
  std::size_t cached_rope_num_axes = 0;
  float cached_rope_freq_base = 0.0f;
  std::array<const void*, 3> cached_rope_pos{};

  struct ElidedSplitBcast {
    const ggml_tensor* concat_root = nullptr;
    SplitBroadcastWhc bcast{};
  };
  std::vector<ElidedSplitBcast> elided_bcasts;
  const auto get_split_bcast = [&](const ggml_tensor* whc) -> SplitBroadcastWhc {
    if (whc == nullptr) return {};
    const ggml_tensor* root = RootTensor(whc);
    for (const auto& e : elided_bcasts) {
      if (e.concat_root == root) return e.bcast;
    }
    return {};
  };

  const auto match_mlp_swiglu = [&](std::size_t gate_cont_idx, const ggml_tensor* mm_gate,
                                    const ggml_tensor* spatial_src,
                                    ggml_tensor** out_gate_tensor) -> bool {
    ggml_tensor* gate = is_kernel(gate_cont_idx, kContName);
    ggml_tensor* in_up = is_kernel(gate_cont_idx + 1, kContName);
    ggml_tensor* mm_up = is_kernel(gate_cont_idx + 2, kMulMatCublas);
    ggml_tensor* up = is_kernel(gate_cont_idx + 3, kContName);
    ggml_tensor* glu = is_kernel(gate_cont_idx + 4, kSwiGluName);
    ggml_tensor* in_down = is_kernel(gate_cont_idx + 5, kContName);
    ggml_tensor* mm_down = is_kernel(gate_cont_idx + 6, kMulMatCublas);
    if (gate == nullptr || in_up == nullptr || mm_up == nullptr || up == nullptr ||
        glu == nullptr || in_down == nullptr || mm_down == nullptr) {
      return false;
    }
    if (!IsSpatialFromRowsCont(gate) || RootTensor(gate->src[0]) != mm_gate ||
        !IsPermute1203(in_up->src[0]) || in_up->src[0]->src[0] != spatial_src ||
        RootTensor(mm_up->src[1]) != in_up || !IsSpatialFromRowsCont(up) ||
        RootTensor(up->src[0]) != mm_up || glu->src[0] != gate || glu->src[1] != up ||
        !IsPermute1203(in_down->src[0]) || in_down->src[0]->src[0] != glu ||
        RootTensor(mm_down->src[1]) != in_down) {
      return false;
    }
    if (!ggml_are_same_shape(mm_gate, mm_up) || mm_gate->ne[2] != 1 || mm_gate->ne[3] != 1 ||
        mm_down->src[1]->ne[0] != mm_gate->ne[0] || mm_down->src[1]->ne[1] != mm_gate->ne[1]) {
      return false;
    }
    if (!DisjointBytes(gate, mm_up)) {
      return false;
    }
    if (used_after(gate_cont_idx + 6, mm_gate) || used_after(gate_cont_idx + 6, gate) ||
        used_after(gate_cont_idx + 6, in_up) || used_after(gate_cont_idx + 6, mm_up) ||
        used_after(gate_cont_idx + 6, up) || used_after(gate_cont_idx + 6, glu) ||
        used_after(gate_cont_idx + 6, in_down)) {
      return false;
    }
    *out_gate_tensor = gate;
    return true;
  };

  const auto run_mlp_tail = [&](std::size_t gate_cont_idx, const ggml_tensor* mm_gate,
                                ggml_tensor* gate, bool gate_redirected, bool gate_bf16,
                                bool down_bf16_dst = false) -> std::expected<void, KernelFailure> {
    ggml_tensor* mm_up = is_kernel(gate_cont_idx + 2, kMulMatCublas);
    ggml_tensor* glu = is_kernel(gate_cont_idx + 4, kSwiGluName);
    ggml_tensor* mm_down = is_kernel(gate_cont_idx + 6, kMulMatCublas);
    void* gate_rows_data = gate->data;
    if (!gate_redirected) {
      if (mm_gate->data == mm_up->data) {
        ggml_tensor copy_node = *mm_gate;
        copy_node.op = GGML_OP_CONT;
        copy_node.data = gate->data;
        copy_node.src[0] = const_cast<ggml_tensor*>(mm_gate);
        if (auto copied = Cont(launch, &copy_node); !copied) {
          return wrap_err(gate_cont_idx, copied);
        }
      } else {
        gate_rows_data = mm_gate->data;
      }
    }
    if (auto r = MulMatCublasWithReuse(launch, mm_up, nullptr, /*reuse_converted_src1=*/true,
                                       /*bf16_dst=*/gate_bf16, /*src1_is_bf16=*/false,
                                       /*allow_lt_pin=*/true);
        !r) {
      return wrap_err(gate_cont_idx + 2, r);
    }
    ggml_tensor s0 = *mm_gate;
    s0.data = gate_rows_data;
    ggml_tensor s1 = *mm_up;
    ggml_tensor down_src1 = *mm_down->src[1];
    down_src1.view_src = nullptr;
    down_src1.view_offs = 0;
    down_src1.data = gate_rows_data;
    if (gate_bf16) {
      if (auto r = SwiGluToBf16(launch, &s0, &s1, glu->op_params[1] != 0, /*bf16_inputs=*/true,
                                /*dst_override=*/gate_rows_data);
          !r) {
        return wrap_err(gate_cont_idx + 4, r);
      }
      if (auto r = MulMatCublasWithReuse(launch, mm_down, &down_src1,
                                         /*reuse_converted_src1=*/false,
                                         /*bf16_dst=*/down_bf16_dst,
                                         /*src1_is_bf16=*/true, /*allow_lt_pin=*/true);
          !r) {
        return wrap_err(gate_cont_idx + 6, r);
      }
      return {};
    }
    const bool direct_bf16 =
        mm_down->src[0] != nullptr && mm_down->src[0]->type == GGML_TYPE_BF16 &&
        static_cast<std::uint64_t>(ggml_nelements(&s0)) * 2U <= launch.workspace().size.value();
    if (direct_bf16) {
      cached_attn_mask = nullptr;
      if (auto r = SwiGluToBf16(launch, &s0, &s1, glu->op_params[1] != 0); !r) {
        return wrap_err(gate_cont_idx + 4, r);
      }
      if (auto r = MulMatCublasWithReuse(launch, mm_down, &down_src1, /*reuse_converted_src1=*/true,
                                         /*bf16_dst=*/false, /*src1_is_bf16=*/false,
                                         /*allow_lt_pin=*/true);
          !r) {
        return wrap_err(gate_cont_idx + 6, r);
      }
    } else {
      cached_attn_mask = nullptr;
      ggml_tensor dst = *mm_gate;
      dst.op = glu->op;
      std::memcpy(dst.op_params, glu->op_params, sizeof(dst.op_params));
      dst.data = gate_rows_data;
      dst.src[0] = &s0;
      dst.src[1] = &s1;
      if (auto r = SwiGlu(launch, &dst); !r) {
        return wrap_err(gate_cont_idx + 4, r);
      }
      if (auto r = MulMatCublasWithReuse(launch, mm_down, &down_src1,
                                         /*reuse_converted_src1=*/false, /*bf16_dst=*/false,
                                         /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
          !r) {
        return wrap_err(gate_cont_idx + 6, r);
      }
    }
    return {};
  };

  const auto try_qkv_rope_attn =
      [&](std::size_t q_cont_idx, ggml_tensor* mm_q, const ggml_tensor* spatial_src,
          const ggml_tensor* q_src1_override, bool reuse_q_bf16, std::size_t* out_attn_idx,
          bool* out_attn_bf16_direct, std::expected<void, KernelFailure>* out_status) -> bool {
    *out_attn_bf16_direct = false;
    ggml_tensor* q = is_kernel(q_cont_idx, kContName);
    ggml_tensor* in_k = is_kernel(q_cont_idx + 1, kContName);
    ggml_tensor* mm_k = is_kernel(q_cont_idx + 2, kMulMatCublas);
    ggml_tensor* k = is_kernel(q_cont_idx + 3, kContName);
    ggml_tensor* in_v = is_kernel(q_cont_idx + 4, kContName);
    ggml_tensor* mm_v = is_kernel(q_cont_idx + 5, kMulMatCublas);
    ggml_tensor* v = is_kernel(q_cont_idx + 6, kContName);
    ggml_tensor* rq_in = is_kernel(q_cont_idx + 7, kContName);
    if (q == nullptr || in_k == nullptr || mm_k == nullptr || k == nullptr || in_v == nullptr ||
        mm_v == nullptr || v == nullptr || rq_in == nullptr) {
      return false;
    }
    if (!IsSpatialFromRowsCont(q) || RootTensor(q->src[0]) != mm_q ||
        !IsPermute1203(in_k->src[0]) || in_k->src[0]->src[0] != spatial_src ||
        RootTensor(mm_k->src[1]) != in_k || !IsSpatialFromRowsCont(k) ||
        RootTensor(k->src[0]) != mm_k || !IsPermute1203(in_v->src[0]) ||
        in_v->src[0]->src[0] != spatial_src || RootTensor(mm_v->src[1]) != in_v ||
        !IsSpatialFromRowsCont(v) || RootTensor(v->src[0]) != mm_v ||
        !IsPermute1203(rq_in->src[0]) || rq_in->src[0]->src[0] != q) {
      return false;
    }
    Rope3dMatch rope_q;
    if (!match_rope_3d(q_cont_idx + 8, &rope_q) || RootTensor(rope_q.x_in) != rq_in) {
      return false;
    }
    const std::size_t rq_out_idx = rope_q.end_idx + 1;
    ggml_tensor* rq = is_kernel(rq_out_idx, kContName);
    ggml_tensor* rk_in = is_kernel(rq_out_idx + 1, kContName);
    if (rq == nullptr || rk_in == nullptr || !IsSpatialFromRowsCont(rq) ||
        RootTensor(rq->src[0]) != rope_q.last_rope || !IsPermute1203(rk_in->src[0]) ||
        rk_in->src[0]->src[0] != k) {
      return false;
    }
    const std::size_t rk_rms_idx = rq_out_idx + 2;
    Rope3dMatch rope_k;
    if (!match_rope_3d(rk_rms_idx, &rope_k) || RootTensor(rope_k.x_in) != rk_in) {
      return false;
    }
    const std::size_t rk_out_idx = rope_k.end_idx + 1;
    ggml_tensor* rk = is_kernel(rk_out_idx, kContName);
    ggml_tensor* qk = is_kernel(rk_out_idx + 1, kConcatName);
    ggml_tensor* qkv = is_kernel(rk_out_idx + 2, kConcatName);
    ggml_tensor* q_attn = is_kernel(rk_out_idx + 3, kContName);
    ggml_tensor* k_attn = is_kernel(rk_out_idx + 4, kContName);
    const std::size_t attn_idx = rk_out_idx + 5;
    ggml_tensor* attn = is_kernel(attn_idx, kAttentionF32);
    if (rk == nullptr || qk == nullptr || qkv == nullptr || q_attn == nullptr ||
        k_attn == nullptr || attn == nullptr) {
      return false;
    }
    if (!IsSpatialFromRowsCont(rk) || RootTensor(rk->src[0]) != rope_k.last_rope ||
        qk->op_params[0] != 2 || qk->src[0] != rq || qk->src[1] != rk || qkv->op_params[0] != 2 ||
        qkv->src[0] != qk || qkv->src[1] != v || RootTensor(q_attn->src[0]) != qkv ||
        RootTensor(k_attn->src[0]) != qkv || attn->src[0] != q_attn || attn->src[1] != k_attn ||
        RootTensor(attn->src[2]) != qkv) {
      return false;
    }
    const std::int64_t dim = rope_q.x_in->ne[0];
    const std::int64_t heads_q = rope_q.x_in->ne[1];
    const std::int64_t rows = rope_q.x_in->ne[2];
    const std::int64_t heads_k = rope_k.x_in->ne[1];
    if (rope_k.x_in->ne[0] != dim || rope_k.x_in->ne[2] != rows || mm_q->ne[0] != dim * heads_q ||
        mm_q->ne[1] != rows || mm_q->ne[2] != 1 || mm_q->ne[3] != 1 ||
        mm_k->ne[0] != dim * heads_k || mm_k->ne[1] != rows || mm_k->ne[2] != 1 ||
        mm_k->ne[3] != 1 || mm_v->ne[0] != dim * heads_k || mm_v->ne[1] != rows ||
        mm_v->ne[2] != 1 || mm_v->ne[3] != 1 || !ggml_are_same_shape(mm_q->src[1], mm_k->src[1]) ||
        !ggml_are_same_shape(mm_q->src[1], mm_v->src[1]) || q_attn->ne[0] != dim ||
        q_attn->ne[1] != rows || q_attn->ne[2] != heads_q || q_attn->ne[3] != 1 ||
        !ggml_is_contiguous(q_attn) || k_attn->ne[0] != dim || k_attn->ne[1] != rows ||
        k_attn->ne[2] != heads_k || k_attn->ne[3] != 1 || !ggml_is_contiguous(k_attn)) {
      return false;
    }
    const std::array<const ggml_tensor*, 18> intermediates{mm_q,
                                                           q,
                                                           in_k,
                                                           mm_k,
                                                           k,
                                                           in_v,
                                                           mm_v,
                                                           v,
                                                           rq_in,
                                                           rope_q.last_rope,
                                                           rq,
                                                           rk_in,
                                                           rope_k.last_rope,
                                                           rk,
                                                           qk,
                                                           qkv,
                                                           q_attn,
                                                           k_attn};
    for (const ggml_tensor* t : intermediates) {
      if (used_after(attn_idx, t)) return false;
    }
    const ggml_tensor* q_src1 = q_src1_override != nullptr ? q_src1_override : mm_q->src[1];
    if (!DisjointBytes(qkv, q_attn) || !DisjointBytes(qkv, k_attn) || !DisjointBytes(qkv, attn) ||
        ggml_nbytes(qkv) < ggml_nbytes(mm_k) || ggml_nbytes(qkv) < ggml_nbytes(mm_v)) {
      return false;
    }
    const ggml_tensor* tmp_q = nullptr;
    for (const ggml_tensor* cand : {qkv, q, k, v, rq_in, rk_in, mm_q}) {
      if (cand != nullptr && DisjointBytes(cand, q_src1) && DisjointBytes(cand, q_attn) &&
          ggml_nbytes(cand) >= ggml_nbytes(mm_q)) {
        tmp_q = cand;
        break;
      }
    }
    if (tmp_q == nullptr) return false;

    const std::int64_t cells = ((rows + 255) / 256) * 256;
    const std::uint64_t kv_bytes =
        static_cast<std::uint64_t>(dim * cells * heads_k) * sizeof(std::uint16_t);
    const std::size_t k_bf16_bytes =
        static_cast<std::size_t>(ggml_nelements(mm_k)) * sizeof(std::uint16_t);
    const std::size_t v_bf16_bytes =
        static_cast<std::size_t>(ggml_nelements(mm_v)) * sizeof(std::uint16_t);
    const bool fast_qkv_f16 = mm_q->src[0] != nullptr && mm_q->src[0]->type == GGML_TYPE_BF16 &&
                              mm_k->src[0] != nullptr && mm_k->src[0]->type == GGML_TYPE_BF16 &&
                              mm_v->src[0] != nullptr && mm_v->src[0]->type == GGML_TYPE_BF16 &&
                              ggml_nbytes(qkv) >= k_bf16_bytes + v_bf16_bytes &&
                              2 * kv_bytes <= launch.workspace().size.value();

    if (fast_qkv_f16) {
      const std::size_t q_bf16_bytes =
          static_cast<std::size_t>(ggml_nelements(mm_q)) * sizeof(std::uint16_t);
      const ggml_tensor* q_buf = nullptr;
      for (const ggml_tensor* cand : {q_attn, k_attn, q, k, v, rq_in, rk_in, mm_q}) {
        if (cand != nullptr && (reuse_q_bf16 || DisjointBytes(cand, q_src1)) &&
            DisjointBytes(cand, qkv) && DisjointBytes(cand, attn) &&
            ggml_nbytes(cand) >= q_bf16_bytes) {
          q_buf = cand;
          break;
        }
      }
      bool same_rope_grid = (q_buf != nullptr && dim == 128 && heads_k > 0 &&
                             (heads_q % heads_k) == 0 && rope_q.num_axes == rope_k.num_axes &&
                             rope_q.freq_base == rope_k.freq_base && rope_q.neox == rope_k.neox);
      for (std::size_t a = 0; same_rope_grid && a < rope_q.num_axes; ++a) {
        if (rope_q.axis_dims[a] != rope_k.axis_dims[a] ||
            rope_q.pos_views[a]->data != rope_k.pos_views[a]->data) {
          same_rope_grid = false;
        }
      }

      if (same_rope_grid) {
        ggml_tensor mm_q_node = *mm_q;
        mm_q_node.data = q_buf->data;
        if (auto r = MulMatCublasWithReuse(launch, &mm_q_node, q_src1_override, reuse_q_bf16,
                                           /*bf16_dst=*/true, /*src1_is_bf16=*/false,
                                           /*allow_lt_pin=*/true);
            !r) {
          *out_status = wrap_err(q_cont_idx - 1, r);
          return true;
        }
        ggml_tensor mm_k_node = *mm_k;
        mm_k_node.data = qkv->data;
        if (auto r = MulMatCublasWithReuse(launch, &mm_k_node, nullptr,
                                           /*reuse_converted_src1=*/true, /*bf16_dst=*/true,
                                           /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
            !r) {
          *out_status = wrap_err(q_cont_idx + 2, r);
          return true;
        }
        ggml_tensor mm_v_node = *mm_v;
        mm_v_node.data = static_cast<char*>(qkv->data) + k_bf16_bytes;
        if (auto r = MulMatCublasWithReuse(launch, &mm_v_node, nullptr,
                                           /*reuse_converted_src1=*/true, /*bf16_dst=*/true,
                                           /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
            !r) {
          *out_status = wrap_err(q_cont_idx + 5, r);
          return true;
        }
        const std::uint64_t freqs_bytes =
            ((static_cast<std::uint64_t>(rows * dim) * sizeof(float)) + 255) & ~std::uint64_t{255};
        const std::uint64_t q_src1_bf16_bytes =
            static_cast<std::uint64_t>(ggml_nelements(q_src1)) * sizeof(std::uint16_t);
        const bool norm_q_in_attn = !rope_q.neox && rope_q.weight->type == GGML_TYPE_BF16 &&
                                    q_src1_bf16_bytes + freqs_bytes <= 2 * kv_bytes;
        const std::uint64_t freqs_ws_offset =
            norm_q_in_attn ? (2 * kv_bytes - freqs_bytes) : UINT64_MAX;
        const bool mask_prepacked = attn->src[3] != nullptr && attn->src[3] == cached_attn_mask &&
                                    cached_attn_rows == rows && cached_attn_heads_k == heads_k &&
                                    cached_attn_dim == dim;
        const bool freqs_precomputed =
            norm_q_in_attn && mask_prepacked && cached_rope_num_axes == rope_q.num_axes &&
            cached_rope_freq_base == rope_q.freq_base &&
            cached_rope_pos[0] == rope_q.pos_views[0]->data &&
            (rope_q.num_axes < 2 || cached_rope_pos[1] == rope_q.pos_views[1]->data) &&
            (rope_q.num_axes < 3 || cached_rope_pos[2] == rope_q.pos_views[2]->data);
        ggml_tensor rq_in_node = *rope_q.x_in;
        rq_in_node.view_src = nullptr;
        rq_in_node.view_offs = 0;
        rq_in_node.data = q_buf->data;
        ggml_tensor rk_in_node = *rope_k.x_in;
        rk_in_node.view_src = nullptr;
        rk_in_node.view_offs = 0;
        rk_in_node.data = qkv->data;
        if (auto r = RmsNormMulRope3dQKInplaceBf16(
                launch, norm_q_in_attn ? nullptr : &rq_in_node,
                norm_q_in_attn ? nullptr : rope_q.weight, norm_q_in_attn ? 0.0f : rope_q.eps,
                &rk_in_node, rope_k.weight, rope_k.eps,
                std::span(rope_q.pos_views).first(rope_q.num_axes),
                std::span(rope_q.axis_dims).first(rope_q.num_axes), rope_q.freq_base, rope_q.neox,
                freqs_ws_offset, freqs_precomputed);
            !r) {
          *out_status = wrap_err(q_cont_idx + 8, r);
          return true;
        }
        ggml_tensor q_node = *q_attn;
        q_node.view_src = nullptr;
        q_node.view_offs = 0;
        q_node.data = q_buf->data;
        ggml_tensor k_node = *k_attn;
        k_node.view_src = nullptr;
        k_node.view_offs = 0;
        k_node.data = qkv->data;
        ggml_tensor v_node = *attn->src[2];
        v_node.view_src = nullptr;
        v_node.view_offs = 0;
        v_node.data = static_cast<char*>(qkv->data) + k_bf16_bytes;
        v_node.ne[0] = dim;
        v_node.ne[1] = rows;
        v_node.ne[2] = heads_k;
        v_node.ne[3] = 1;
        v_node.nb[0] = sizeof(float);
        v_node.nb[1] = static_cast<std::size_t>(dim * heads_k) * sizeof(float);
        v_node.nb[2] = static_cast<std::size_t>(dim) * sizeof(float);
        v_node.nb[3] = static_cast<std::size_t>(dim * heads_k * rows) * sizeof(float);
        ggml_tensor attn_node = *attn;
        attn_node.src[0] = &q_node;
        attn_node.src[1] = &k_node;
        attn_node.src[2] = &v_node;

        ggml_tensor* spatial_next = is_kernel(attn_idx + 1, kContName);
        ggml_tensor* in_next = is_kernel(attn_idx + 2, kContName);
        ggml_tensor* mm_next = is_kernel(attn_idx + 3, kMulMatCublas);
        const bool attn_bf16_direct =
            spatial_next != nullptr && in_next != nullptr && mm_next != nullptr &&
            IsSpatialFromRowsCont(spatial_next) && RootTensor(spatial_next->src[0]) == attn &&
            IsPermute1203(in_next->src[0]) && in_next->src[0]->src[0] == spatial_next &&
            RootTensor(mm_next->src[1]) == in_next && mm_next->src[0] != nullptr &&
            mm_next->src[0]->type == GGML_TYPE_BF16 && !used_after(attn_idx + 1, attn) &&
            !used_after(attn_idx + 3, spatial_next) && !used_after(attn_idx + 3, in_next);

        AttentionF32Options attn_opts{
            .mask_prepacked = mask_prepacked,
            .bf16_dst = attn_bf16_direct,
            .qkv_bf16_row_major = true,
            .norm_q_in_attn = norm_q_in_attn,
            .q_weight = norm_q_in_attn ? rope_q.weight->data : nullptr,
            .q_eps = norm_q_in_attn ? rope_q.eps : 0.0f,
        };
        if (auto r = AttentionF32WithOptions(launch, &attn_node, attn_opts); !r) {
          *out_status = wrap_err(attn_idx, r);
          return true;
        }
        if (attn->src[3] != nullptr) {
          cached_attn_mask = attn->src[3];
          cached_attn_rows = rows;
          cached_attn_heads_k = heads_k;
          cached_attn_dim = dim;
          cached_rope_num_axes = norm_q_in_attn ? rope_q.num_axes : 0;
          cached_rope_freq_base = rope_q.freq_base;
          for (std::size_t a = 0; a < rope_q.num_axes && a < 3; ++a) {
            cached_rope_pos[a] = rope_q.pos_views[a]->data;
          }
        }
        *out_attn_idx = attn_idx;
        *out_attn_bf16_direct = attn_bf16_direct;
        *out_status = {};
        return true;
      }

      ggml_tensor mm_q_node = *mm_q;
      mm_q_node.data = tmp_q->data;
      if (auto r = MulMatCublasWithReuse(launch, &mm_q_node, q_src1_override, reuse_q_bf16,
                                         /*bf16_dst=*/true, /*src1_is_bf16=*/false,
                                         /*allow_lt_pin=*/true);
          !r) {
        *out_status = wrap_err(q_cont_idx - 1, r);
        return true;
      }
      ggml_tensor rq_in_node = *rope_q.x_in;
      rq_in_node.view_src = nullptr;
      rq_in_node.view_offs = 0;
      rq_in_node.data = tmp_q->data;
      if (auto r = RmsNormMulRope3d(launch, &rq_in_node, rope_q.weight, rope_q.eps,
                                    std::span(rope_q.pos_views).first(rope_q.num_axes),
                                    std::span(rope_q.axis_dims).first(rope_q.num_axes),
                                    rope_q.freq_base, rope_q.neox, q_attn, /*transpose_out=*/true,
                                    /*x_is_bf16=*/true, /*dst_is_f16=*/true,
                                    /*padded_rows=*/0, /*dst_in_workspace=*/false);
          !r) {
        *out_status = wrap_err(q_cont_idx + 8, r);
        return true;
      }
      ggml_tensor mm_k_node = *mm_k;
      mm_k_node.data = qkv->data;
      if (auto r = MulMatCublasWithReuse(launch, &mm_k_node, nullptr,
                                         /*reuse_converted_src1=*/true, /*bf16_dst=*/true,
                                         /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
          !r) {
        *out_status = wrap_err(q_cont_idx + 2, r);
        return true;
      }
      ggml_tensor mm_v_node = *mm_v;
      mm_v_node.data = static_cast<char*>(qkv->data) + k_bf16_bytes;
      if (auto r = MulMatCublasWithReuse(launch, &mm_v_node, nullptr,
                                         /*reuse_converted_src1=*/true, /*bf16_dst=*/true,
                                         /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
          !r) {
        *out_status = wrap_err(q_cont_idx + 5, r);
        return true;
      }
      ggml_tensor rk_in_node = *rope_k.x_in;
      rk_in_node.view_src = nullptr;
      rk_in_node.view_offs = 0;
      rk_in_node.data = qkv->data;
      if (auto r = RmsNormMulRope3d(
              launch, &rk_in_node, rope_k.weight, rope_k.eps,
              std::span(rope_k.pos_views).first(rope_k.num_axes),
              std::span(rope_k.axis_dims).first(rope_k.num_axes), rope_k.freq_base, rope_k.neox,
              k_attn, /*transpose_out=*/true, /*x_is_bf16=*/true, /*dst_is_f16=*/true,
              /*padded_rows=*/static_cast<int>(cells), /*dst_in_workspace=*/true);
          !r) {
        *out_status = wrap_err(rk_rms_idx, r);
        return true;
      }
      ggml_tensor v_node = *attn->src[2];
      v_node.view_src = nullptr;
      v_node.view_offs = 0;
      v_node.data = static_cast<char*>(qkv->data) + k_bf16_bytes;
      v_node.ne[0] = dim;
      v_node.ne[1] = rows;
      v_node.ne[2] = heads_k;
      v_node.ne[3] = 1;
      v_node.nb[0] = sizeof(float);
      v_node.nb[1] = static_cast<std::size_t>(dim * heads_k) * sizeof(float);
      v_node.nb[2] = static_cast<std::size_t>(dim) * sizeof(float);
      v_node.nb[3] = static_cast<std::size_t>(dim * heads_k * rows) * sizeof(float);
      ggml_tensor attn_node = *attn;
      attn_node.src[2] = &v_node;

      ggml_tensor* spatial_next = is_kernel(attn_idx + 1, kContName);
      ggml_tensor* in_next = is_kernel(attn_idx + 2, kContName);
      ggml_tensor* mm_next = is_kernel(attn_idx + 3, kMulMatCublas);
      const bool attn_bf16_direct =
          spatial_next != nullptr && in_next != nullptr && mm_next != nullptr &&
          IsSpatialFromRowsCont(spatial_next) && RootTensor(spatial_next->src[0]) == attn &&
          IsPermute1203(in_next->src[0]) && in_next->src[0]->src[0] == spatial_next &&
          RootTensor(mm_next->src[1]) == in_next && mm_next->src[0] != nullptr &&
          mm_next->src[0]->type == GGML_TYPE_BF16 && !used_after(attn_idx + 1, attn) &&
          !used_after(attn_idx + 3, spatial_next) && !used_after(attn_idx + 3, in_next);

      const bool mask_prepacked = attn->src[3] != nullptr && attn->src[3] == cached_attn_mask &&
                                  cached_attn_rows == rows && cached_attn_heads_k == heads_k &&
                                  cached_attn_dim == dim;
      AttentionF32Options attn_opts{
          .q_is_f16 = true,
          .k_prepacked = true,
          .v_is_bf16 = true,
          .mask_prepacked = mask_prepacked,
          .bf16_dst = attn_bf16_direct,
      };
      if (auto r = AttentionF32WithOptions(launch, &attn_node, attn_opts); !r) {
        *out_status = wrap_err(attn_idx, r);
        return true;
      }
      if (attn->src[3] != nullptr) {
        cached_attn_mask = attn->src[3];
        cached_attn_rows = rows;
        cached_attn_heads_k = heads_k;
        cached_attn_dim = dim;
      }
      *out_attn_idx = attn_idx;
      *out_attn_bf16_direct = attn_bf16_direct;
      *out_status = {};
      return true;
    }

    cached_attn_mask = nullptr;
    ggml_tensor mm_q_node = *mm_q;
    mm_q_node.data = tmp_q->data;
    if (auto r = MulMatCublasWithReuse(launch, &mm_q_node, q_src1_override, reuse_q_bf16); !r) {
      *out_status = wrap_err(q_cont_idx - 1, r);
      return true;
    }
    ggml_tensor rq_in_node = *rope_q.x_in;
    rq_in_node.view_src = nullptr;
    rq_in_node.view_offs = 0;
    rq_in_node.data = tmp_q->data;
    if (auto r = RmsNormMulRope3d(launch, &rq_in_node, rope_q.weight, rope_q.eps,
                                  std::span(rope_q.pos_views).first(rope_q.num_axes),
                                  std::span(rope_q.axis_dims).first(rope_q.num_axes),
                                  rope_q.freq_base, rope_q.neox, q_attn, /*transpose_out=*/true);
        !r) {
      *out_status = wrap_err(q_cont_idx + 8, r);
      return true;
    }
    ggml_tensor mm_k_node = *mm_k;
    mm_k_node.data = qkv->data;
    if (auto r = MulMatCublasWithReuse(launch, &mm_k_node, nullptr,
                                       /*reuse_converted_src1=*/true);
        !r) {
      *out_status = wrap_err(q_cont_idx + 2, r);
      return true;
    }
    ggml_tensor rk_in_node = *rope_k.x_in;
    rk_in_node.view_src = nullptr;
    rk_in_node.view_offs = 0;
    rk_in_node.data = qkv->data;
    if (auto r = RmsNormMulRope3d(launch, &rk_in_node, rope_k.weight, rope_k.eps,
                                  std::span(rope_k.pos_views).first(rope_k.num_axes),
                                  std::span(rope_k.axis_dims).first(rope_k.num_axes),
                                  rope_k.freq_base, rope_k.neox, k_attn, /*transpose_out=*/true);
        !r) {
      *out_status = wrap_err(rk_rms_idx, r);
      return true;
    }
    ggml_tensor mm_v_node = *mm_v;
    mm_v_node.data = qkv->data;
    if (auto r = MulMatCublasWithReuse(launch, &mm_v_node, nullptr,
                                       /*reuse_converted_src1=*/true);
        !r) {
      *out_status = wrap_err(q_cont_idx + 5, r);
      return true;
    }
    ggml_tensor v_node = *attn->src[2];
    v_node.view_src = nullptr;
    v_node.view_offs = 0;
    v_node.data = qkv->data;
    v_node.ne[0] = dim;
    v_node.ne[1] = rows;
    v_node.ne[2] = heads_k;
    v_node.ne[3] = 1;
    v_node.nb[0] = sizeof(float);
    v_node.nb[1] = static_cast<std::size_t>(dim * heads_k) * sizeof(float);
    v_node.nb[2] = static_cast<std::size_t>(dim) * sizeof(float);
    v_node.nb[3] = static_cast<std::size_t>(dim * heads_k * rows) * sizeof(float);
    ggml_tensor attn_node = *attn;
    attn_node.src[2] = &v_node;
    if (auto r = AttentionF32(launch, &attn_node); !r) {
      *out_status = wrap_err(attn_idx, r);
      return true;
    }
    *out_attn_idx = attn_idx;
    *out_status = {};
    return true;
  };

  const auto can_elide_concat_bcast = [&](std::size_t concat_idx, const ggml_tensor* c) -> bool {
    if (c == nullptr || c->data == nullptr) return false;
    bool has_consumer = false;
    for (std::size_t j = concat_idx + 1; j < steps_.size(); ++j) {
      bool reads_c = false;
      for (const ggml_tensor* node : steps_[j].nodes) {
        if (node == nullptr) continue;
        for (const ggml_tensor* src : node->src) {
          if (src != nullptr && (src == c || RootTensor(src) == c)) {
            reads_c = true;
            break;
          }
        }
        if (reads_c) break;
      }
      if (reads_c) {
        ggml_tensor* side_cont = is_kernel(j, kContName);
        if (side_cont != nullptr && IsPermute1203(side_cont->src[0]) &&
            RootTensor(side_cont->src[0]->src[0]) == c &&
            is_kernel(j + 1, "ggml.norm") != nullptr && is_kernel(j + 2, kScaleName) != nullptr &&
            is_kernel(j + 3, kMulName) != nullptr && !used_after(j + 3, side_cont)) {
          has_consumer = true;
        } else {
          ggml_tensor* t = is_kernel(j, kUnaryName);
          if (t != nullptr && j >= 1 && t->op == GGML_OP_UNARY &&
              ggml_get_unary_op(t) == GGML_UNARY_OP_TANH && RootTensor(t->src[0]) == c &&
              IsSpatialFromRowsCont(is_kernel(j - 1, kContName)) &&
              is_kernel(j + 1, kMulName) != nullptr && is_kernel(j + 2, kAddName) != nullptr &&
              !used_after(j + 2, t)) {
            has_consumer = true;
          } else {
            return false;
          }
        }
      }
      if (!steps_[j].nodes.empty()) {
        const ggml_tensor* out = steps_[j].nodes.back();
        if (out != nullptr && out->view_src == nullptr && out->data != nullptr &&
            out->data == c->data && ggml_nbytes(out) >= ggml_nbytes(c)) {
          break;
        }
      }
    }
    return has_consumer;
  };

  const auto can_skip_mod_f32_dst = [&](std::size_t mod_idx, const ggml_tensor* mod,
                                        bool write_bf16) -> bool {
    if (!write_bf16 || mod == nullptr) return false;
    ggml_tensor* spatial_in = is_kernel(mod_idx + 1, kContName);
    ggml_tensor* in_0 = is_kernel(mod_idx + 2, kContName);
    ggml_tensor* mm_0 = is_kernel(mod_idx + 3, kMulMatCublas);
    if (spatial_in == nullptr || in_0 == nullptr || mm_0 == nullptr ||
        !IsSpatialFromRowsCont(spatial_in) || RootTensor(spatial_in->src[0]) != mod ||
        !IsPermute1203(in_0->src[0]) || in_0->src[0]->src[0] != spatial_in ||
        RootTensor(mm_0->src[1]) != in_0 || mm_0->src[0] == nullptr ||
        mm_0->src[0]->type != GGML_TYPE_BF16 || used_after(mod_idx + 1, mod) ||
        used_after(mod_idx + 3, in_0)) {
      return false;
    }
    std::size_t num_proj = 1;
    for (std::size_t p = 1; p < 3; ++p) {
      ggml_tensor* prev_out = is_kernel(mod_idx + 1 + 3 * p, kContName);
      ggml_tensor* in_p = is_kernel(mod_idx + 2 + 3 * p, kContName);
      ggml_tensor* mm_p = is_kernel(mod_idx + 3 + 3 * p, kMulMatCublas);
      if (prev_out == nullptr || in_p == nullptr || mm_p == nullptr ||
          !IsPermute1203(in_p->src[0]) || in_p->src[0]->src[0] != spatial_in ||
          RootTensor(mm_p->src[1]) != in_p || mm_p->src[1]->ne[0] != mm_0->src[1]->ne[0] ||
          mm_p->src[1]->ne[1] != mm_0->src[1]->ne[1] || mm_p->src[1]->ne[2] != 1 ||
          mm_p->src[1]->ne[3] != 1 || used_after(mod_idx + 3 + 3 * p, in_p)) {
        break;
      }
      ++num_proj;
    }
    if (used_after(mod_idx + 3 * num_proj, spatial_in)) return false;
    ggml_tensor* dummy_gate = nullptr;
    if (num_proj == 2 && match_mlp_swiglu(mod_idx + 4, mm_0, spatial_in, &dummy_gate)) {
      return true;
    }
    if (num_proj == 3) {
      ggml_tensor* mm_k = is_kernel(mod_idx + 6, kMulMatCublas);
      ggml_tensor* mm_v = is_kernel(mod_idx + 9, kMulMatCublas);
      if (mm_k != nullptr && mm_v != nullptr && mm_k->src[0] != nullptr &&
          mm_k->src[0]->type == GGML_TYPE_BF16 && mm_v->src[0] != nullptr &&
          mm_v->src[0]->type == GGML_TYPE_BF16) {
        return true;
      }
    }
    return false;
  };

  const auto only_used_by_norm_and_next_residual = [&](std::size_t out_idx,
                                                       const ggml_tensor* out) -> bool {
    if (out == nullptr || out->data == nullptr) return false;
    for (std::size_t j = out_idx + 1; j < steps_.size(); ++j) {
      bool reads_out = false;
      for (const ggml_tensor* node : steps_[j].nodes) {
        if (node == nullptr) continue;
        for (const ggml_tensor* src : node->src) {
          if (src != nullptr && (src == out || RootTensor(src) == out)) {
            reads_out = true;
            break;
          }
        }
        if (reads_out) break;
      }
      if (reads_out) {
        if (j == out_idx + 1) continue;
        ggml_tensor* branch_cont_next = j >= 3 ? is_kernel(j - 3, kContName) : nullptr;
        ggml_tensor* t_next = j >= 2 ? is_kernel(j - 2, kUnaryName) : nullptr;
        ggml_tensor* m_next = j >= 1 ? is_kernel(j - 1, kMulName) : nullptr;
        ggml_tensor* out_next = is_kernel(j, kAddName);
        if (branch_cont_next != nullptr && t_next != nullptr && m_next != nullptr &&
            out_next != nullptr && IsSpatialFromRowsCont(branch_cont_next) &&
            t_next->op == GGML_OP_UNARY && ggml_get_unary_op(t_next) == GGML_UNARY_OP_TANH &&
            ((out_next->src[0] == m_next && out_next->src[1] == out) ||
             (out_next->src[1] == m_next && out_next->src[0] == out)) &&
            !used_after(j, out)) {
          continue;
        }
        return false;
      }
      if (!steps_[j].nodes.empty()) {
        const ggml_tensor* step_out = steps_[j].nodes.back();
        if (step_out != nullptr && step_out->view_src == nullptr && step_out->data != nullptr &&
            step_out->data == out->data && ggml_nbytes(step_out) >= ggml_nbytes(out)) {
          break;
        }
      }
    }
    return true;
  };

  const auto only_used_by_gated_residual = [&](std::size_t mm_idx,
                                               const ggml_tensor* mm_out) -> bool {
    if (mm_out == nullptr) return false;
    ggml_tensor* branch_cont = is_kernel(mm_idx + 1, kContName);
    ggml_tensor* t = is_kernel(mm_idx + 2, kUnaryName);
    ggml_tensor* m = is_kernel(mm_idx + 3, kMulName);
    ggml_tensor* out = is_kernel(mm_idx + 4, kAddName);
    if (branch_cont == nullptr || t == nullptr || m == nullptr || out == nullptr ||
        !IsSpatialFromRowsCont(branch_cont) || t->op != GGML_OP_UNARY ||
        ggml_get_unary_op(t) != GGML_UNARY_OP_TANH) {
      return false;
    }
    const ggml_tensor* gate = t->src[0];
    const ggml_tensor* branch = m->src[0] == t ? m->src[1] : (m->src[1] == t ? m->src[0] : nullptr);
    const ggml_tensor* carry =
        out->src[0] == m ? out->src[1] : (out->src[1] == m ? out->src[0] : nullptr);
    const ggml_tensor* branch_rows = RootTensor(branch_cont->src[0]);
    if (branch != branch_cont || branch_rows != mm_out || gate == nullptr || carry == nullptr ||
        gate->type != GGML_TYPE_F32 || carry->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(carry) || !ggml_is_contiguous(out) ||
        !ggml_is_contiguous(branch_rows) || ggml_nbytes(branch_rows) != ggml_nbytes(out) ||
        !ggml_are_same_shape(carry, branch_cont) || !ggml_are_same_shape(carry, gate) ||
        !ggml_are_same_shape(carry, out) || !DisjointBytes(branch_rows, out) ||
        !DisjointBytes(gate, out) || used_after(mm_idx + 1, mm_out) ||
        used_after(mm_idx + 4, branch_cont) || used_after(mm_idx + 4, t) ||
        used_after(mm_idx + 4, m)) {
      return false;
    }
    const bool has_carry_scratch =
        DisjointBytes(carry, out) ||
        (DisjointBytes(branch_cont, out) && DisjointBytes(branch_cont, carry) &&
         DisjointBytes(branch_cont, branch_rows) && DisjointBytes(branch_cont, gate)) ||
        (DisjointBytes(m, out) && DisjointBytes(m, carry) && DisjointBytes(m, branch_rows) &&
         DisjointBytes(m, gate)) ||
        (DisjointBytes(t, out) && DisjointBytes(t, carry) && DisjointBytes(t, branch_rows) &&
         DisjointBytes(t, gate));
    return has_carry_scratch;
  };

  for (std::size_t i = 0; i < steps_.size(); ++i) {
    auto iter_res = [&]() -> std::expected<void, KernelFailure> {
      // Pattern 0B: Elide Cont(v0) -> Repeat(r0) -> Cont(v1) -> Repeat(r1) -> Concat(c)
      if (ggml_tensor* v0 = is_kernel(i, kContName)) {
        ggml_tensor* r0 = is_kernel(i + 1, "ggml.repeat");
        ggml_tensor* v1 = is_kernel(i + 2, kContName);
        ggml_tensor* r1 = is_kernel(i + 3, "ggml.repeat");
        ggml_tensor* c = is_kernel(i + 4, kConcatName);
        if (r0 != nullptr && v1 != nullptr && r1 != nullptr && c != nullptr &&
            v0->type == GGML_TYPE_F32 && r0->type == GGML_TYPE_F32 && v1->type == GGML_TYPE_F32 &&
            r1->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32 && ggml_is_contiguous(v0) &&
            ggml_is_contiguous(v1) && ggml_is_contiguous(c) && v0->ne[0] == 1 && v0->ne[1] > 0 &&
            v0->ne[2] == 1 && v0->ne[3] == 1 && v1->ne[0] == 1 && v1->ne[1] == v0->ne[1] &&
            v1->ne[2] == 1 && v1->ne[3] == 1 && r0->src[0] == v0 && r1->src[0] == v1 &&
            r0->ne[0] > 0 && r0->ne[1] == v0->ne[1] && r0->ne[2] == 1 && r0->ne[3] == 1 &&
            r1->ne[0] > 0 && r1->ne[1] == v0->ne[1] && r1->ne[2] == 1 && r1->ne[3] == 1 &&
            c->op_params[0] == 0 && c->src[0] == r0 && c->src[1] == r1 &&
            c->ne[0] == r0->ne[0] + r1->ne[0] && c->ne[1] == v0->ne[1] && c->ne[2] == 1 &&
            c->ne[3] == 1 && c->ne[0] >= 2 && v0->src[0] != nullptr && v1->src[0] != nullptr &&
            !used_after(i + 4, v0) && !used_after(i + 4, r0) && !used_after(i + 4, v1) &&
            !used_after(i + 4, r1) && can_elide_concat_bcast(i + 4, c)) {
          const std::size_t cols = static_cast<std::size_t>(v0->ne[1]);
          const std::size_t rows = static_cast<std::size_t>(c->ne[0]);
          const std::size_t vec_bytes = cols * sizeof(float);
          float* vec0_dst = static_cast<float*>(c->data) + (rows - 2) * cols;
          float* vec1_dst = static_cast<float*>(c->data) + (rows - 1) * cols;
          if (DisjointSpan(v0->src[0]->data, vec_bytes, vec0_dst, 2 * vec_bytes) &&
              DisjointSpan(v1->src[0]->data, vec_bytes, vec0_dst, 2 * vec_bytes)) {
            ggml_tensor v0_node = *v0;
            v0_node.data = vec0_dst;
            if (auto r = Cont(launch, &v0_node); !r) return wrap_err(i, r);
            ggml_tensor v1_node = *v1;
            v1_node.data = vec1_dst;
            if (auto r = Cont(launch, &v1_node); !r) return wrap_err(i + 2, r);
            std::erase_if(elided_bcasts, [&](const ElidedSplitBcast& e) {
              return e.concat_root == c || e.concat_root->data == c->data;
            });
            elided_bcasts.push_back(ElidedSplitBcast{
                .concat_root = c,
                .bcast =
                    SplitBroadcastWhc{
                        .vec0 = vec0_dst,
                        .vec1 = vec1_dst,
                        .split_row = static_cast<int>(r0->ne[0]),
                    },
            });
            i += 4;
            return {};
          }
        }
      }

      // Pattern 0: SpatialNorm (Cont(Permute) -> L2Norm -> Scale -> Mul -> [SiLU] -> Cont(Permute))
      if (ggml_tensor* rows_cont = is_kernel(i, kContName)) {
        ggml_tensor* l2 = is_kernel(i + 1, "ggml.l2_norm");
        ggml_tensor* sc = is_kernel(i + 2, kScaleName);
        ggml_tensor* mul = is_kernel(i + 3, kMulName);
        if (l2 != nullptr && sc != nullptr && mul != nullptr && IsPermute1203(rows_cont->src[0]) &&
            RootTensor(l2->src[0]) == rows_cont && sc->src[0] == l2) {
          ggml_tensor* silu = is_kernel(i + 4, kUnaryName);
          const bool has_silu = silu != nullptr && silu->op == GGML_OP_UNARY &&
                                ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU &&
                                silu->src[0] == mul;
          const std::size_t out_idx = has_silu ? (i + 5) : (i + 4);
          ggml_tensor* out_whc = is_kernel(out_idx, kContName);
          const ggml_tensor* x_whc = rows_cont->src[0]->src[0];
          const ggml_tensor* gamma =
              mul->src[0] == sc ? mul->src[1] : (mul->src[1] == sc ? mul->src[0] : nullptr);
          const ggml_tensor* pre_out = has_silu ? silu : mul;
          float scale_v = 0.0f;
          float bias_v = 0.0f;
          std::memcpy(&scale_v, &sc->op_params[0], sizeof(float));
          std::memcpy(&bias_v, &sc->op_params[1], sizeof(float));
          if (out_whc != nullptr && x_whc != nullptr && gamma != nullptr &&
              out_whc->src[0] != nullptr && out_whc->src[0]->op == GGML_OP_PERMUTE &&
              out_whc->src[0]->op_params[0] == 2 && out_whc->src[0]->op_params[1] == 0 &&
              out_whc->src[0]->op_params[2] == 1 && out_whc->src[0]->op_params[3] == 3 &&
              RootTensor(out_whc->src[0]->src[0]) == pre_out && x_whc->type == GGML_TYPE_F32 &&
              gamma->type == GGML_TYPE_F32 && out_whc->type == GGML_TYPE_F32 &&
              ggml_is_contiguous(x_whc) && ggml_is_contiguous(gamma) &&
              ggml_is_contiguous(out_whc) && ggml_are_same_shape(x_whc, out_whc) &&
              x_whc->ne[0] > 0 && x_whc->ne[1] > 0 && x_whc->ne[2] > 0 && x_whc->ne[3] == 1 &&
              gamma->ne[0] == x_whc->ne[2] && ggml_nelements(gamma) == x_whc->ne[2] &&
              bias_v == 0.0f && !used_after(out_idx, rows_cont) && !used_after(out_idx, l2) &&
              !used_after(out_idx, sc) && !used_after(out_idx, mul) &&
              (!has_silu || !used_after(out_idx, silu))) {
            float eps = 0.0f;
            std::memcpy(&eps, l2->op_params, sizeof(float));
            if (auto r =
                    SpatialChannelNormF32(launch, x_whc, gamma, out_whc, eps, scale_v, has_silu);
                !r) {
              return wrap_err(i + 1, r);
            }
            i = out_idx;
            return {};
          }
        }
      }

      // Pattern 1A: AdaptiveSpatial with x_cont and side_cont ->
      // LayerNormModulateF32(x_is_whc=true)
      if (ggml_tensor* x_cont = is_kernel(i, kContName)) {
        ggml_tensor* side_cont = is_kernel(i + 1, kContName);
        ggml_tensor* ln = is_kernel(i + 2, "ggml.norm");
        ggml_tensor* sc = is_kernel(i + 3, kScaleName);
        ggml_tensor* mod = is_kernel(i + 4, kMulName);
        if (side_cont != nullptr && ln != nullptr && sc != nullptr && mod != nullptr &&
            IsPermute1203(x_cont->src[0]) && IsPermute1203(side_cont->src[0])) {
          const ggml_tensor* x_whc = x_cont->src[0]->src[0];
          const ggml_tensor* side_whc = side_cont->src[0]->src[0];
          const ggml_tensor* x_rows = ln->src[0];
          float scale_v = 0.0f;
          float bias_v = 0.0f;
          std::memcpy(&scale_v, &sc->op_params[0], sizeof(float));
          std::memcpy(&bias_v, &sc->op_params[1], sizeof(float));
          if (x_whc != nullptr && side_whc != nullptr && x_rows != nullptr &&
              x_whc->type == GGML_TYPE_F32 && side_whc->type == GGML_TYPE_F32 &&
              mod->type == GGML_TYPE_F32 && ggml_is_contiguous(x_whc) &&
              ggml_is_contiguous(side_whc) && ggml_is_contiguous(mod) &&
              ggml_are_same_shape(x_whc, side_whc) && x_rows->ne[0] > 0 && x_rows->ne[1] > 0 &&
              x_rows->ne[2] == 1 && x_rows->ne[3] == 1 && ggml_are_same_shape(x_rows, mod) &&
              side_whc->ne[0] * side_whc->ne[1] == x_rows->ne[1] &&
              side_whc->ne[2] == x_rows->ne[0] && side_whc->ne[3] == 1 &&
              RootTensor(x_rows) == x_cont && RootTensor(sc->src[0]) == side_cont &&
              scale_v == 1.0f && bias_v == 1.0f &&
              ((mod->src[0] == ln && mod->src[1] == sc) ||
               (mod->src[0] == sc && mod->src[1] == ln)) &&
              DisjointBytes(x_whc, mod) && DisjointBytes(side_whc, mod) &&
              !used_after(i + 4, x_cont) && !used_after(i + 4, side_cont) &&
              !used_after(i + 4, ln) && !used_after(i + 4, sc)) {
            float eps = 0.0f;
            std::memcpy(&eps, ln->op_params, sizeof(float));
            const bool write_bf16 = static_cast<std::uint64_t>(ggml_nelements(mod)) * 2U <=
                                    launch.workspace().size.value();
            const bool skip_f32 = can_skip_mod_f32_dst(i + 4, mod, write_bf16);
            const SplitBroadcastWhc side_bcast = get_split_bcast(side_whc);
            const bool x_is_whc = (x_whc != rows_layout_tensor);
            if (auto r = LayerNormModulateF32(launch, x_whc, side_whc, mod, eps, x_is_whc,
                                              write_bf16, skip_f32, side_bcast);
                !r) {
              return wrap_err(i + 2, r);
            }
            ws_bf16_spatial = nullptr;
            ws_bf16_rows = write_bf16 ? mod : nullptr;
            ws_bf16_ne0 = mod->ne[0];
            ws_bf16_ne1 = mod->ne[1];
            direct_bf16_rows = nullptr;
            i += 4;
            return {};
          }
        }
      }

      // Pattern 1B: AdaptiveSpatial with side_cont -> LayerNormModulateF32(x_is_whc=false)
      if (ggml_tensor* side_cont = is_kernel(i, kContName)) {
        ggml_tensor* ln = is_kernel(i + 1, "ggml.norm");
        ggml_tensor* sc = is_kernel(i + 2, kScaleName);
        ggml_tensor* mod = is_kernel(i + 3, kMulName);
        if (ln != nullptr && sc != nullptr && mod != nullptr && IsPermute1203(side_cont->src[0])) {
          const ggml_tensor* side_whc = side_cont->src[0]->src[0];
          const ggml_tensor* x_rows = ln->src[0];
          float scale_v = 0.0f;
          float bias_v = 0.0f;
          std::memcpy(&scale_v, &sc->op_params[0], sizeof(float));
          std::memcpy(&bias_v, &sc->op_params[1], sizeof(float));
          if (x_rows != nullptr && side_whc != nullptr && x_rows->type == GGML_TYPE_F32 &&
              side_whc->type == GGML_TYPE_F32 && mod->type == GGML_TYPE_F32 &&
              ggml_is_contiguous(x_rows) && ggml_is_contiguous(side_whc) &&
              ggml_is_contiguous(mod) && x_rows->ne[0] > 0 && x_rows->ne[1] > 0 &&
              x_rows->ne[2] == 1 && x_rows->ne[3] == 1 && ggml_are_same_shape(x_rows, mod) &&
              side_whc->ne[0] * side_whc->ne[1] == x_rows->ne[1] &&
              side_whc->ne[2] == x_rows->ne[0] && side_whc->ne[3] == 1 &&
              RootTensor(sc->src[0]) == side_cont && scale_v == 1.0f && bias_v == 1.0f &&
              ((mod->src[0] == ln && mod->src[1] == sc) ||
               (mod->src[0] == sc && mod->src[1] == ln)) &&
              DisjointBytes(side_whc, mod) && !used_after(i + 3, side_cont) &&
              !used_after(i + 3, ln) && !used_after(i + 3, sc)) {
            float eps = 0.0f;
            std::memcpy(&eps, ln->op_params, sizeof(float));
            const bool write_bf16 = static_cast<std::uint64_t>(ggml_nelements(mod)) * 2U <=
                                    launch.workspace().size.value();
            const SplitBroadcastWhc side_bcast = get_split_bcast(side_whc);
            if (auto r = LayerNormModulateF32(launch, x_rows, side_whc, mod, eps,
                                              /*x_is_whc=*/false, write_bf16,
                                              /*skip_f32_dst=*/false, side_bcast);
                !r) {
              return wrap_err(i + 1, r);
            }
            ws_bf16_spatial = nullptr;
            ws_bf16_rows = write_bf16 ? mod : nullptr;
            ws_bf16_ne0 = mod->ne[0];
            ws_bf16_ne1 = mod->ne[1];
            direct_bf16_rows = nullptr;
            i += 3;
            return {};
          }
        }
      }

      // Pattern 2: RmsNormMul + 1..3 (Cont(pos), RopeExt) -> RmsNormMulRope3d
      {
        Rope3dMatch rope;
        if (match_rope_3d(i, &rope)) {
          if (auto r =
                  RmsNormMulRope3d(launch, rope.x_in, rope.weight, rope.eps,
                                   std::span(rope.pos_views).first(rope.num_axes),
                                   std::span(rope.axis_dims).first(rope.num_axes), rope.freq_base,
                                   rope.neox, rope.last_rope, /*transpose_out=*/false);
              !r) {
            return wrap_err(i, r);
          }
          i = rope.end_idx;
          return {};
        }
      }

      // Pattern 3A: Cont(FromChannelRows(branch_rows)) + GatedResidual [+ Pattern 1A]
      if (ggml_tensor* branch_cont = is_kernel(i, kContName)) {
        ggml_tensor* t = is_kernel(i + 1, kUnaryName);
        ggml_tensor* m = is_kernel(i + 2, kMulName);
        ggml_tensor* out = is_kernel(i + 3, kAddName);
        if (t != nullptr && m != nullptr && out != nullptr && IsSpatialFromRowsCont(branch_cont) &&
            t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_TANH) {
          const ggml_tensor* gate = t->src[0];
          const ggml_tensor* branch =
              m->src[0] == t ? m->src[1] : (m->src[1] == t ? m->src[0] : nullptr);
          const ggml_tensor* carry =
              out->src[0] == m ? out->src[1] : (out->src[1] == m ? out->src[0] : nullptr);
          const ggml_tensor* branch_rows = RootTensor(branch_cont->src[0]);
          const auto find_scratch =
              [&](std::initializer_list<const ggml_tensor*> pool,
                  std::initializer_list<const ggml_tensor*> avoid) -> const ggml_tensor* {
            for (const ggml_tensor* cand : pool) {
              if (cand == nullptr || !ggml_is_contiguous(cand) ||
                  ggml_nbytes(cand) != ggml_nbytes(out)) {
                continue;
              }
              bool ok = true;
              for (const ggml_tensor* a : avoid) {
                if (a != nullptr && !DisjointBytes(cand, a)) {
                  ok = false;
                  break;
                }
              }
              if (ok) return cand;
            }
            return nullptr;
          };
          const auto stage_into =
              [&](const ggml_tensor* src, const ggml_tensor* scratch,
                  ggml_tensor* staged) -> std::expected<const ggml_tensor*, KernelFailure> {
            if (scratch == nullptr) return src;
            *staged = *src;
            staged->op = GGML_OP_CONT;
            staged->view_src = nullptr;
            staged->view_offs = 0;
            staged->data = scratch->data;
            staged->src[0] = const_cast<ggml_tensor*>(src);
            if (auto r = Cont(launch, staged); !r) return std::unexpected(r.error());
            return staged;
          };
          if (branch == branch_cont && gate != nullptr && carry != nullptr &&
              branch_rows != nullptr && gate->type == GGML_TYPE_F32 &&
              carry->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32 &&
              ggml_is_contiguous(gate) && ggml_is_contiguous(carry) && ggml_is_contiguous(out) &&
              ggml_is_contiguous(branch_rows) && ggml_nbytes(branch_rows) == ggml_nbytes(out) &&
              ggml_are_same_shape(carry, branch_cont) && ggml_are_same_shape(carry, gate) &&
              ggml_are_same_shape(carry, out) && !used_after(i + 3, branch_cont) &&
              !used_after(i + 3, t) && !used_after(i + 3, m)) {
            const bool carry_is_rows = (carry == rows_layout_tensor);
            const bool branch_is_bf16 = (branch_rows == bf16_branch_tensor);
            const SplitBroadcastWhc gate_bcast = get_split_bcast(gate);

            // Check if immediately followed by Pattern 1A consuming `out`, and `out` is only
            // ever used by that Pattern 1A and the next Pattern 3A's `carry`.
            ggml_tensor* x_cont = is_kernel(i + 4, kContName);
            ggml_tensor* side_cont = is_kernel(i + 5, kContName);
            ggml_tensor* ln = is_kernel(i + 6, "ggml.norm");
            ggml_tensor* sc = is_kernel(i + 7, kScaleName);
            ggml_tensor* mod = is_kernel(i + 8, kMulName);
            if (x_cont != nullptr && side_cont != nullptr && ln != nullptr && sc != nullptr &&
                mod != nullptr && IsPermute1203(x_cont->src[0]) &&
                IsPermute1203(side_cont->src[0]) && x_cont->src[0]->src[0] == out) {
              const ggml_tensor* side_whc = side_cont->src[0]->src[0];
              const ggml_tensor* x_rows = ln->src[0];
              float scale_v = 0.0f;
              float bias_v = 0.0f;
              std::memcpy(&scale_v, &sc->op_params[0], sizeof(float));
              std::memcpy(&bias_v, &sc->op_params[1], sizeof(float));
              const bool need_stage_carry_1a = !carry_is_rows && !DisjointBytes(carry, out);
              const ggml_tensor* carry_scratch_1a =
                  need_stage_carry_1a ? find_scratch({branch_cont, m, t, x_cont, side_cont, ln, sc},
                                                     {out, carry, branch_rows, gate, side_whc, mod})
                                      : nullptr;
              if (side_whc != nullptr && x_rows != nullptr && side_whc->type == GGML_TYPE_F32 &&
                  mod->type == GGML_TYPE_F32 && ggml_is_contiguous(side_whc) &&
                  ggml_is_contiguous(mod) && ggml_are_same_shape(out, side_whc) &&
                  x_rows->ne[0] > 0 && x_rows->ne[1] > 0 && x_rows->ne[2] == 1 &&
                  x_rows->ne[3] == 1 && ggml_are_same_shape(x_rows, mod) &&
                  side_whc->ne[0] * side_whc->ne[1] == x_rows->ne[1] &&
                  side_whc->ne[2] == x_rows->ne[0] && side_whc->ne[3] == 1 &&
                  RootTensor(x_rows) == x_cont && RootTensor(sc->src[0]) == side_cont &&
                  scale_v == 1.0f && bias_v == 1.0f &&
                  ((mod->src[0] == ln && mod->src[1] == sc) ||
                   (mod->src[0] == sc && mod->src[1] == ln)) &&
                  DisjointBytes(out, mod) && DisjointBytes(side_whc, mod) &&
                  DisjointBytes(side_whc, out) && DisjointBytes(branch_rows, mod) &&
                  DisjointBytes(gate, out) && DisjointBytes(gate, mod) &&
                  (!need_stage_carry_1a || carry_scratch_1a != nullptr) &&
                  !used_after(i + 8, x_cont) && !used_after(i + 8, side_cont) &&
                  !used_after(i + 8, ln) && !used_after(i + 8, sc) &&
                  only_used_by_norm_and_next_residual(i + 3, out)) {
                ggml_tensor staged_carry{};
                auto actual_carry = stage_into(carry, carry_scratch_1a, &staged_carry);
                if (!actual_carry) return wrap_err(i, std::unexpected(actual_carry.error()));
                float eps = 0.0f;
                std::memcpy(&eps, ln->op_params, sizeof(float));
                const bool write_bf16 = static_cast<std::uint64_t>(ggml_nelements(mod)) * 2U <=
                                        launch.workspace().size.value();
                const bool skip_f32 = can_skip_mod_f32_dst(i + 8, mod, write_bf16);
                const SplitBroadcastWhc side_bcast = get_split_bcast(side_whc);
                const bool carry_is_bf16 = carry_is_rows && rows_layout_is_bf16;
                const bool out_rows_bf16 = branch_is_bf16 && mod->ne[0] == 4096;
                if (auto r = GatedResidualLayerNormModulateF32(
                        launch, *actual_carry, branch_rows, gate, out, side_whc, mod, eps,
                        carry_is_rows, write_bf16, skip_f32, gate_bcast, side_bcast, branch_is_bf16,
                        carry_is_bf16, out_rows_bf16);
                    !r) {
                  return wrap_err(i + 1, r);
                }
                rows_layout_tensor = out;
                rows_layout_is_bf16 = out_rows_bf16;
                bf16_branch_tensor = nullptr;
                ws_bf16_spatial = nullptr;
                ws_bf16_rows = write_bf16 ? mod : nullptr;
                ws_bf16_ne0 = mod->ne[0];
                ws_bf16_ne1 = mod->ne[1];
                direct_bf16_rows = nullptr;
                i += 8;
                return {};
              }
            }

            const bool need_stage_branch = !DisjointBytes(branch_rows, out);
            const ggml_tensor* branch_scratch =
                need_stage_branch
                    ? find_scratch({branch_cont, m, t}, {out, carry, branch_rows, gate})
                    : nullptr;
            const bool need_stage_carry = carry_is_rows && !DisjointBytes(carry, out);
            const ggml_tensor* carry_scratch =
                need_stage_carry ? find_scratch({branch_cont, m, t},
                                                {out, carry, branch_rows, gate, branch_scratch})
                                 : nullptr;
            if ((!need_stage_branch || branch_scratch != nullptr) &&
                (!need_stage_carry || carry_scratch != nullptr) &&
                (gate_bcast.vec0 == nullptr || DisjointBytes(gate, out))) {
              ggml_tensor staged_branch{};
              auto actual_branch = stage_into(branch_rows, branch_scratch, &staged_branch);
              if (!actual_branch) return wrap_err(i, std::unexpected(actual_branch.error()));
              ggml_tensor staged_carry{};
              auto actual_carry = stage_into(carry, carry_scratch, &staged_carry);
              if (!actual_carry) return wrap_err(i, std::unexpected(actual_carry.error()));
              const bool carry_is_bf16 = carry_is_rows && rows_layout_is_bf16;
              if (auto r = GatedResidualF32(launch, *actual_carry, *actual_branch, gate, out,
                                            /*branch_is_rows=*/true, carry_is_rows, gate_bcast,
                                            branch_is_bf16, carry_is_bf16);
                  !r) {
                return wrap_err(i + 1, r);
              }
              rows_layout_tensor = nullptr;
              rows_layout_is_bf16 = false;
              bf16_branch_tensor = nullptr;
              i += 3;
              return {};
            }
          }
        }
      }

      // Pattern 3B: GatedResidual (Unary(Tanh) -> Mul -> Add) -> GatedResidualF32
      if (ggml_tensor* t = is_kernel(i, kUnaryName)) {
        ggml_tensor* m = is_kernel(i + 1, kMulName);
        ggml_tensor* out = is_kernel(i + 2, kAddName);
        if (m != nullptr && out != nullptr && t->op == GGML_OP_UNARY &&
            ggml_get_unary_op(t) == GGML_UNARY_OP_TANH) {
          const ggml_tensor* gate = t->src[0];
          const ggml_tensor* branch =
              m->src[0] == t ? m->src[1] : (m->src[1] == t ? m->src[0] : nullptr);
          const ggml_tensor* carry =
              out->src[0] == m ? out->src[1] : (out->src[1] == m ? out->src[0] : nullptr);
          if (gate != nullptr && branch != nullptr && carry != nullptr &&
              gate->type == GGML_TYPE_F32 && branch->type == GGML_TYPE_F32 &&
              carry->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32 &&
              ggml_is_contiguous(gate) && ggml_is_contiguous(branch) && ggml_is_contiguous(carry) &&
              ggml_is_contiguous(out) && ggml_are_same_shape(carry, branch) &&
              ggml_are_same_shape(carry, gate) && ggml_are_same_shape(carry, out) &&
              !used_after(i + 2, t) && !used_after(i + 2, m)) {
            if (auto r = GatedResidualF32(launch, carry, branch, gate, out,
                                          /*branch_is_rows=*/false, /*carry_is_rows=*/false,
                                          get_split_bcast(gate));
                !r) {
              return wrap_err(i, r);
            }
            i += 2;
            return {};
          }
        }
      }

      // Pattern 4A: Elide roundtrip [C, rows] -> [1, rows, C, 1] -> [C, rows] before 1..3
      // projections
      if (ggml_tensor* spatial_in = is_kernel(i, kContName)) {
        ggml_tensor* in_0 = is_kernel(i + 1, kContName);
        ggml_tensor* mm_0 = is_kernel(i + 2, kMulMatCublas);
        if (in_0 != nullptr && mm_0 != nullptr && IsSpatialFromRowsCont(spatial_in) &&
            IsPermute1203(in_0->src[0]) && in_0->src[0]->src[0] == spatial_in &&
            RootTensor(mm_0->src[1]) == in_0 && mm_0->src[1]->ne[0] == spatial_in->ne[2] &&
            mm_0->src[1]->ne[1] == spatial_in->ne[1] && mm_0->src[1]->ne[2] == 1 &&
            mm_0->src[1]->ne[3] == 1 && !used_after(i + 2, in_0)) {
          const ggml_tensor* prev_rows = RootTensor(spatial_in->src[0]);
          std::size_t num_proj = 1;
          for (std::size_t p = 1; p < 3; ++p) {
            ggml_tensor* prev_out = is_kernel(i + 3 * p, kContName);
            ggml_tensor* in_p = is_kernel(i + 1 + 3 * p, kContName);
            ggml_tensor* mm_p = is_kernel(i + 2 + 3 * p, kMulMatCublas);
            if (prev_out == nullptr || in_p == nullptr || mm_p == nullptr ||
                !IsPermute1203(in_p->src[0]) || in_p->src[0]->src[0] != spatial_in ||
                RootTensor(mm_p->src[1]) != in_p || mm_p->src[1]->ne[0] != mm_0->src[1]->ne[0] ||
                mm_p->src[1]->ne[1] != mm_0->src[1]->ne[1] || mm_p->src[1]->ne[2] != 1 ||
                mm_p->src[1]->ne[3] != 1 || used_after(i + 2 + 3 * p, in_p)) {
              break;
            }
            ++num_proj;
          }
          if (!used_after(i + 3 * num_proj - 1, spatial_in)) {
            ggml_tensor src1_view = *mm_0->src[1];
            src1_view.view_src = nullptr;
            src1_view.view_offs = 0;
            src1_view.data = prev_rows->data;
            const bool reuse_first_bf16 =
                (prev_rows == ws_bf16_rows && src1_view.ne[0] == ws_bf16_ne0 &&
                 src1_view.ne[1] == ws_bf16_ne1);
            const bool first_src1_is_bf16 = (prev_rows == direct_bf16_rows);

            std::size_t attn_idx = 0;
            bool attn_bf16_direct = false;
            std::expected<void, KernelFailure> qkv_status;
            if (num_proj == 3 &&
                try_qkv_rope_attn(i + 3, mm_0, spatial_in, &src1_view, reuse_first_bf16, &attn_idx,
                                  &attn_bf16_direct, &qkv_status)) {
              if (!qkv_status) return qkv_status;
              ws_bf16_spatial = nullptr;
              ws_bf16_rows = nullptr;
              direct_bf16_rows = attn_bf16_direct ? is_kernel(attn_idx, kAttentionF32) : nullptr;
              bf16_branch_tensor = nullptr;
              i = attn_idx;
              return {};
            }

            ggml_tensor* mlp_gate = nullptr;
            if (num_proj == 2 && match_mlp_swiglu(i + 3, mm_0, spatial_in, &mlp_gate)) {
              const bool redirect_gate = reuse_first_bf16 || DisjointBytes(mlp_gate, &src1_view);
              ggml_tensor* mm_up = is_kernel(i + 5, kMulMatCublas);
              ggml_tensor* mm_down = is_kernel(i + 9, kMulMatCublas);
              const bool gate_bf16 = redirect_gate && mm_0->src[0] != nullptr &&
                                     mm_0->src[0]->type == GGML_TYPE_BF16 && mm_up != nullptr &&
                                     mm_up->src[0] != nullptr &&
                                     mm_up->src[0]->type == GGML_TYPE_BF16 && mm_down != nullptr &&
                                     mm_down->src[0] != nullptr &&
                                     mm_down->src[0]->type == GGML_TYPE_BF16;
              const bool down_bf16_dst = gate_bf16 && only_used_by_gated_residual(i + 9, mm_down);
              ggml_tensor mm_gate_node = *mm_0;
              if (redirect_gate) mm_gate_node.data = mlp_gate->data;
              if (auto r = MulMatCublasWithReuse(launch, &mm_gate_node, &src1_view,
                                                 reuse_first_bf16, /*bf16_dst=*/gate_bf16,
                                                 /*src1_is_bf16=*/false,
                                                 /*allow_lt_pin=*/true);
                  !r) {
                return wrap_err(i + 2, r);
              }
              if (auto r =
                      run_mlp_tail(i + 3, mm_0, mlp_gate, redirect_gate, gate_bf16, down_bf16_dst);
                  !r) {
                return r;
              }
              ws_bf16_spatial = nullptr;
              ws_bf16_rows = nullptr;
              direct_bf16_rows = nullptr;
              bf16_branch_tensor = down_bf16_dst ? mm_down : nullptr;
              i += 9;
              return {};
            }

            const bool single_bf16_dst =
                num_proj == 1 && first_src1_is_bf16 && mm_0->src[0] != nullptr &&
                mm_0->src[0]->type == GGML_TYPE_BF16 && only_used_by_gated_residual(i + 2, mm_0);
            if (auto r = MulMatCublasWithReuse(launch, mm_0, &src1_view, reuse_first_bf16,
                                               /*bf16_dst=*/single_bf16_dst, first_src1_is_bf16,
                                               /*allow_lt_pin=*/true);
                !r) {
              return wrap_err(i + 2, r);
            }
            if (!first_src1_is_bf16) {
              ws_bf16_spatial = spatial_in;
              ws_bf16_rows = prev_rows;
              ws_bf16_ne0 = src1_view.ne[0];
              ws_bf16_ne1 = src1_view.ne[1];
            } else {
              ws_bf16_spatial = nullptr;
              ws_bf16_rows = nullptr;
            }
            direct_bf16_rows = nullptr;
            bf16_branch_tensor = single_bf16_dst ? mm_0 : nullptr;
            i += 2;
            return {};
          }
        }
      }

      // Pattern 4B: Projection Cont(Permute(spatial_in, {1,2,0,3})) -> MulMatCublas
      if (ggml_tensor* in_p = is_kernel(i, kContName)) {
        ggml_tensor* mm_p = is_kernel(i + 1, kMulMatCublas);
        if (mm_p != nullptr && IsPermute1203(in_p->src[0]) && RootTensor(mm_p->src[1]) == in_p &&
            mm_p->src[1]->ne[2] == 1 && mm_p->src[1]->ne[3] == 1 && !used_after(i + 1, in_p)) {
          const ggml_tensor* spatial_in = in_p->src[0]->src[0];
          if (spatial_in != nullptr && spatial_in == ws_bf16_spatial &&
              mm_p->src[1]->ne[0] == ws_bf16_ne0 && mm_p->src[1]->ne[1] == ws_bf16_ne1) {
            if (auto r = MulMatCublasWithReuse(launch, mm_p, nullptr,
                                               /*reuse_converted_src1=*/true, /*bf16_dst=*/false,
                                               /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
                !r) {
              return wrap_err(i + 1, r);
            }
            i += 1;
            return {};
          }
          // First projection in a group where spatial_in is still needed elsewhere:
          // run in_p, then check if this starts a QKV+RoPE+Attention or MLP sequence.
          if (auto r = RunStep(launch, i); !r) return r;
          std::size_t attn_idx = 0;
          bool attn_bf16_direct = false;
          std::expected<void, KernelFailure> qkv_status;
          if (spatial_in != nullptr &&
              try_qkv_rope_attn(i + 2, mm_p, spatial_in, nullptr, /*reuse_q_bf16=*/false, &attn_idx,
                                &attn_bf16_direct, &qkv_status)) {
            if (!qkv_status) return qkv_status;
            ws_bf16_spatial = nullptr;
            ws_bf16_rows = nullptr;
            direct_bf16_rows = attn_bf16_direct ? is_kernel(attn_idx, kAttentionF32) : nullptr;
            i = attn_idx;
            return {};
          }
          ggml_tensor* mlp_gate = nullptr;
          if (spatial_in != nullptr && match_mlp_swiglu(i + 2, mm_p, spatial_in, &mlp_gate)) {
            const bool redirect_gate = DisjointBytes(mlp_gate, in_p);
            ggml_tensor mm_gate_node = *mm_p;
            if (redirect_gate) mm_gate_node.data = mlp_gate->data;
            if (auto r = MulMatCublasWithReuse(launch, &mm_gate_node, nullptr,
                                               /*reuse_converted_src1=*/false, /*bf16_dst=*/false,
                                               /*src1_is_bf16=*/false, /*allow_lt_pin=*/true);
                !r) {
              return wrap_err(i + 1, r);
            }
            if (auto r = run_mlp_tail(i + 2, mm_p, mlp_gate, redirect_gate, /*gate_bf16=*/false);
                !r) {
              return r;
            }
            ws_bf16_spatial = nullptr;
            ws_bf16_rows = nullptr;
            direct_bf16_rows = nullptr;
            i += 8;
            return {};
          }
          if (auto r = RunStep(launch, i + 1); !r) return r;
          ws_bf16_spatial = spatial_in;
          ws_bf16_rows = nullptr;
          ws_bf16_ne0 = mm_p->src[1]->ne[0];
          ws_bf16_ne1 = mm_p->src[1]->ne[1];
          direct_bf16_rows = nullptr;
          i += 1;
          return {};
        }
      }

      // Pattern 5: Conv2dTiledF32 + in-place bias Add
      if (ggml_tensor* conv = is_kernel(i, kConv2dTiledF32)) {
        if (const ggml_tensor* add = is_kernel(i + 1, kAddName)) {
          const ggml_tensor* bias =
              add->src[0] == conv ? add->src[1] : (add->src[1] == conv ? add->src[0] : nullptr);
          if (bias != nullptr && add->data == conv->data && bias->type == GGML_TYPE_F32 &&
              ggml_is_contiguous(bias) && bias->ne[0] == 1 && bias->ne[1] == 1 &&
              bias->ne[2] == conv->ne[2] && bias->ne[3] == 1) {
            ws_bf16_spatial = nullptr;
            ws_bf16_rows = nullptr;
            direct_bf16_rows = nullptr;
            cached_attn_mask = nullptr;
            auto r = Conv2dTiledF32WithBias(launch, conv, bias);
            i += 1;
            return r;
          }
        }
      }

      // Pattern 6: DupUpAdd (Cont -> Repeat -> Cont(View) -> Cont(Permute) -> Cont(Permute) -> Add)
      if (ggml_tensor* c0 = is_kernel(i, kContName)) {
        ggml_tensor* rep = is_kernel(i + 1, "ggml.repeat");
        ggml_tensor* frame = is_kernel(i + 2, kContName);
        ggml_tensor* dx = is_kernel(i + 3, kContName);
        ggml_tensor* out_up = is_kernel(i + 4, kContName);
        ggml_tensor* add = is_kernel(i + 5, kAddName);
        if (rep != nullptr && frame != nullptr && dx != nullptr && out_up != nullptr &&
            add != nullptr && c0->src[0] != nullptr && rep->src[0] != nullptr &&
            frame->src[0] != nullptr && dx->src[0] != nullptr && out_up->src[0] != nullptr &&
            RootTensor(rep->src[0]) == c0 && frame->src[0]->op == GGML_OP_VIEW &&
            RootTensor(frame->src[0]) == rep && IsPermute1203(dx->src[0]) &&
            RootTensor(dx->src[0]) == frame && out_up->src[0]->op == GGML_OP_PERMUTE &&
            out_up->src[0]->op_params[0] == 0 && out_up->src[0]->op_params[1] == 2 &&
            out_up->src[0]->op_params[2] == 1 && out_up->src[0]->op_params[3] == 3 &&
            RootTensor(out_up->src[0]) == dx) {
          const ggml_tensor* sc = c0->src[0];
          const ggml_tensor* x = RootTensor(add->src[1]) == out_up
                                     ? add->src[0]
                                     : (RootTensor(add->src[0]) == out_up ? add->src[1] : nullptr);
          if (sc != nullptr && x != nullptr && sc->type == GGML_TYPE_F32 &&
              x->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && ggml_is_contiguous(sc) &&
              ggml_is_contiguous(x) && ggml_is_contiguous(add) && ggml_are_same_shape(x, add) &&
              sc->ne[0] > 0 && sc->ne[1] > 0 && sc->ne[2] > 0 && sc->ne[3] == 1 &&
              x->ne[0] == 2 * sc->ne[0] && x->ne[1] == 2 * sc->ne[1] && x->ne[2] > 0 &&
              x->ne[3] == 1 && rep->ne[0] == sc->ne[0] * sc->ne[1] && rep->ne[1] > 0 &&
              rep->ne[2] == sc->ne[2] && rep->ne[3] == 1 && frame->ne[0] == sc->ne[0] * sc->ne[1] &&
              frame->ne[1] == 4 && frame->ne[2] == x->ne[2] && frame->ne[3] == 1 &&
              (rep->ne[1] * sc->ne[2]) % (x->ne[2] * 4) == 0 &&
              (add->data == x->data || DisjointBytes(x, add)) && !used_after(i + 5, c0) &&
              !used_after(i + 5, rep) && !used_after(i + 5, frame) && !used_after(i + 5, dx) &&
              !used_after(i + 5, out_up)) {
            const int repeats = static_cast<int>(rep->ne[1]);
            const int ft = static_cast<int>((rep->ne[1] * sc->ne[2]) / (x->ne[2] * 4));
            const std::size_t expected_offs = static_cast<std::size_t>(ft - 1) * 4 *
                                              static_cast<std::size_t>(sc->ne[0] * sc->ne[1]) *
                                              sizeof(float);
            if (ft >= 1 && frame->src[0]->view_offs == expected_offs &&
                (DisjointBytes(sc, add) || (DisjointBytes(c0, add) && DisjointBytes(c0, sc)))) {
              const ggml_tensor* actual_sc = sc;
              if (!DisjointBytes(sc, add)) {
                if (auto r = Cont(launch, c0); !r) return wrap_err(i, r);
                actual_sc = c0;
              }
              if (auto r = DupUpAddF32(launch, x, actual_sc, add, ft, repeats); !r) {
                return wrap_err(i + 5, r);
              }
              i += 5;
              return {};
            }
          }
        }
      }

      // Pattern 7: SpatialConv1x1 (Cont(Transpose(Reshape(x))) -> MulMatCublas -> Cont(Transpose)
      // -> Add)
      if (ggml_tensor* in_cont = is_kernel(i, kContName)) {
        ggml_tensor* mm = is_kernel(i + 1, kMulMatCublas);
        ggml_tensor* out_cont = is_kernel(i + 2, kContName);
        ggml_tensor* add = is_kernel(i + 3, kAddName);
        if (mm != nullptr && out_cont != nullptr && add != nullptr &&
            IsPermute1023(in_cont->src[0]) && in_cont->src[0]->src[0] != nullptr &&
            in_cont->src[0]->src[0]->op == GGML_OP_RESHAPE && mm->src[1] == in_cont &&
            mm->src[0] != nullptr && IsPermute1023(out_cont->src[0]) &&
            out_cont->src[0]->src[0] == mm) {
          const ggml_tensor* x_whc = in_cont->src[0]->src[0]->src[0];
          const ggml_tensor* weight = mm->src[0];
          const ggml_tensor* bias = add->src[0] == out_cont
                                        ? add->src[1]
                                        : (add->src[1] == out_cont ? add->src[0] : nullptr);
          if (x_whc != nullptr && bias != nullptr && x_whc->type == GGML_TYPE_F32 &&
              weight->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 &&
              add->type == GGML_TYPE_F32 && ggml_is_contiguous(x_whc) &&
              ggml_is_contiguous(weight) && ggml_is_contiguous(bias) && ggml_is_contiguous(add) &&
              ggml_are_same_shape(out_cont, add) && x_whc->ne[0] > 0 && x_whc->ne[1] > 0 &&
              x_whc->ne[2] > 0 && x_whc->ne[3] == 1 && in_cont->ne[0] == x_whc->ne[2] &&
              in_cont->ne[1] == x_whc->ne[0] * x_whc->ne[1] && in_cont->ne[2] == 1 &&
              in_cont->ne[3] == 1 && weight->ne[0] == x_whc->ne[2] && weight->ne[1] == add->ne[2] &&
              weight->ne[2] == 1 && weight->ne[3] == 1 && add->ne[0] == x_whc->ne[0] &&
              add->ne[1] == x_whc->ne[1] && add->ne[3] == 1 && bias->ne[0] == 1 &&
              bias->ne[1] == 1 && bias->ne[2] == add->ne[2] && bias->ne[3] == 1 &&
              DisjointBytes(weight, add) && DisjointBytes(bias, add) &&
              !used_after(i + 3, in_cont) && !used_after(i + 3, mm) &&
              !used_after(i + 3, out_cont)) {
            const ggml_tensor* actual_x = x_whc;
            ggml_tensor staged_x{};
            bool can_run = DisjointBytes(x_whc, add);
            if (!can_run) {
              for (const ggml_tensor* cand : {in_cont, mm, out_cont}) {
                if (cand != nullptr && ggml_is_contiguous(cand) &&
                    ggml_nbytes(cand) >= ggml_nbytes(x_whc) && DisjointBytes(cand, x_whc) &&
                    DisjointBytes(cand, add) && DisjointBytes(cand, weight) &&
                    DisjointBytes(cand, bias)) {
                  staged_x = *x_whc;
                  staged_x.op = GGML_OP_CONT;
                  staged_x.view_src = nullptr;
                  staged_x.view_offs = 0;
                  staged_x.data = cand->data;
                  staged_x.src[0] = const_cast<ggml_tensor*>(x_whc);
                  if (auto r = Cont(launch, &staged_x); !r) return wrap_err(i, r);
                  actual_x = &staged_x;
                  can_run = true;
                  break;
                }
              }
            }
            if (can_run) {
              ws_bf16_spatial = nullptr;
              ws_bf16_rows = nullptr;
              direct_bf16_rows = nullptr;
              auto r = SpatialConv1x1F32(launch, actual_x, weight, bias, add);
              i += 3;
              return r;
            }
          }
        }
      }

      // Track workspace invalidation across ordinary steps.
      const auto* k_step = std::get_if<Kernel>(&steps_[i].kernel);
      const bool preserves_ws =
          k_step != nullptr && (k_step->name() == kContName || k_step->name() == kAddName ||
                                k_step->name() == kMulName || k_step->name() == kUnaryName ||
                                k_step->name() == kScaleName);
      if (!preserves_ws) {
        cached_attn_mask = nullptr;
      }
      if (!steps_[i].nodes.empty()) {
        const ggml_tensor* out = steps_[i].nodes.back();
        if (out != nullptr && out->data != nullptr) {
          std::erase_if(elided_bcasts, [&](const ElidedSplitBcast& e) {
            return e.concat_root->data == out->data;
          });
        }
      }
      if (ws_bf16_spatial != nullptr || ws_bf16_rows != nullptr) {
        const ggml_tensor* out = steps_[i].nodes.empty() ? nullptr : steps_[i].nodes.back();
        const bool overwrites_spatial = ws_bf16_spatial != nullptr && out != nullptr &&
                                        RootTensor(out)->data == RootTensor(ws_bf16_spatial)->data;
        const bool overwrites_rows = ws_bf16_rows != nullptr && out != nullptr &&
                                     RootTensor(out)->data == RootTensor(ws_bf16_rows)->data;
        if (!preserves_ws) {
          ws_bf16_spatial = nullptr;
          ws_bf16_rows = nullptr;
        } else {
          if (overwrites_spatial) ws_bf16_spatial = nullptr;
          if (overwrites_rows) ws_bf16_rows = nullptr;
        }
      }

      return RunStep(launch, i);
    }();
    if (!iter_res) return iter_res;
  }
  return {};
}

std::expected<void, KernelFailure> BoundGraph::RunRange(LaunchContext& launch, std::size_t first,
                                                        std::size_t end) const {
  if (!routed_ || !regions_.empty() || first > end || end > steps_.size())
    return Rejected("invalid owned routed segment");
  for (auto i = first; i != end; ++i)
    if (auto r = RunStep(launch, i); !r) return r;
  return {};
}

// The lanes' protocol (graph_plan.h AssignLanes, LaneOrder). Lane 0 is the
// context's stream. At a region's first step every lane waits for the
// stream (so for everything queued before), a step waits for each other
// lane that last wrote what it reads or writes, or read what it writes
// since, and at the region's last step the stream waits for every lane.
std::expected<void, KernelFailure> BoundGraph::RunLanes(LaunchContext& launch) const {
  constexpr std::size_t kLanes = LaneOrder::kLanes;
  LaneOrder order;
  std::size_t region = 0;
  bool open = false;
  const auto storage = [](const ggml_tensor* t) {
    while (t->view_src != nullptr) {
      t = t->view_src;
    }
    return t;
  };
  const auto wait = [&](std::uint8_t waiter,
                        std::uint8_t on) -> std::expected<void, KernelFailure> {
    if (auto r = launch.LaneWait(waiter, on); !r) {
      return r;
    }
    order.Waited(waiter, on);
    return {};
  };
  // The stream waits for every lane (each forked lane is joined back, as a
  // capture requires, whether or not it ran anything).
  const auto close = [&]() -> std::expected<void, KernelFailure> {
    std::expected<void, KernelFailure> joined;
    for (std::uint8_t lane = 1; lane < kLanes; ++lane) {
      auto r = wait(0, lane);
      if (!r && (joined || r.error().error == KernelError::kUnknown)) {
        joined = r;
      }
    }
    launch.SelectLane(0);
    open = false;
    order.Reset();
    return joined;
  };
  // Every lane waits for the stream: what a region reads, and the bytes its
  // tensors take, are the stream's until it opens.
  const auto fork = [&]() -> std::expected<void, KernelFailure> {
    open = true;
    for (std::uint8_t lane = 1; lane < kLanes; ++lane) {
      if (auto r = wait(lane, 0); !r) {
        return r;
      }
    }
    return {};
  };
  // What a step reads and writes, as storage.
  std::vector<const ggml_tensor*> reads;
  std::vector<const ggml_tensor*> writes;
  const auto accesses = [&](const Step& step) {
    reads.clear();
    writes.clear();
    for (const ggml_tensor* node : step.nodes) {
      writes.push_back(storage(node));
      for (const ggml_tensor* src : node->src) {
        if (src != nullptr) {
          reads.push_back(storage(src));
        }
      }
    }
  };
  // Waits for the other lanes `step` must follow (LaneOrder).
  const auto inputs = [&](std::uint8_t lane) -> std::expected<void, KernelFailure> {
    for (const std::uint8_t other : order.Before(lane, reads, writes)) {
      if (auto r = wait(lane, other); !r) {
        return r;
      }
    }
    return {};
  };
  std::expected<void, KernelFailure> result;
  for (std::size_t i = 0; i < steps_.size() && result; ++i) {
    while (region < regions_.size() && regions_[region].last < i) {
      ++region;
    }
    const bool inside = region < regions_.size() && i >= regions_[region].first;
    if (inside && !open) {
      result = fork();
      if (!result) {
        break;
      }
    }
    const Step& step = steps_[i];
    const std::uint8_t lane = inside ? step.lane : 0;
    if (open) {
      accesses(step);
      result = inputs(lane);
      if (!result) {
        break;
      }
    }
    launch.SelectLane(lane);
    result = RunStep(launch, i);
    if (open) {
      order.Queued(lane, reads, writes);
    }
    if (result && open && i == regions_[region].last) {
      result = close();
    }
  }
  if (open) {
    // Joined even after a failure, so the stream's fence covers every lane.
    auto closed = close();
    if (!closed) {
      // A rejected kernel is not evidence that already queued lane work
      // retired. Preserve an unproven join as kUnknown for the owner even
      // when the earlier step failed cleanly.
      if (!result)
        closed.error().detail = result.error().detail + "; lane closure: " + closed.error().detail;
      if (result || closed.error().error == KernelError::kUnknown) result = std::move(closed);
    }
  }
  launch.SelectLane(0);
  return result;
}

}  // namespace llmp::kernels::ggml
