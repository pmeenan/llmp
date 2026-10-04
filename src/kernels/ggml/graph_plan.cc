// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/graph_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::kernels::ggml {
namespace {

using execution::Operation;

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

std::string Where(GraphNodes graph, std::size_t i) {
  return std::format("node {} ({} {})", i, ggml_op_desc(graph[i]), graph[i]->name);
}

// A computed tensor with memory of its own.
bool Computed(const ggml_tensor* t) { return t->op != GGML_OP_NONE && t->view_src == nullptr; }

// The tensor whose memory `t` is: itself, or the end of its view chain
// (GGML points a view of a view at the first source, but that is
// upstream's choice, not a contract).
const ggml_tensor* Storage(const ggml_tensor* t) {
  while (t->view_src != nullptr) {
    t = t->view_src;
  }
  return t;
}

// Whether `reader`, reading `tensor` itself, is the only reader of
// `tensor`'s memory: no other node reads it, directly or through a view, no
// view of it or write into it is in the graph, and neither the graph's last
// node nor a tensor `keep` names (read after the run) is it or views it.
// For an implementation that leaves `tensor` unwritten or overwrites it
// with another form. dsv4_hc_norm.cc ReadElsewhere's scan, by storage.
bool OnlyReader(GraphNodes graph, std::span<ggml_tensor* const> keep, const ggml_tensor* tensor,
                const ggml_tensor* reader) {
  const ggml_tensor* storage = Storage(tensor);
  if (graph.empty() || Storage(graph.back()) == storage) {
    return false;
  }
  for (const ggml_tensor* kept : keep) {
    if (kept != nullptr && Storage(kept) == storage) {
      return false;
    }
  }
  for (const ggml_tensor* node : graph) {
    if (node == tensor) {
      continue;
    }
    if (Storage(node) == storage) {
      return false;
    }
    for (const ggml_tensor* src : node->src) {
      if (src != nullptr && Storage(src) == storage && (node != reader || src != tensor)) {
        return false;
      }
    }
  }
  return true;
}

std::string_view MulMatName(MulMatPath path) {
  switch (path) {
    case MulMatPath::kVector:
      return kMulMatVector;
    case MulMatPath::kTensorCore:
      return kMulMatTensorCore;
    case MulMatPath::kCublas:
      return kMulMatCublas;
  }
  return kMulMatCublas;
}

std::uint64_t Rounded(std::uint64_t bytes, std::uint64_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

// ggml_mul_mat_set_hint's hint (op_params[1]).
std::int32_t MulMatHint(const ggml_tensor* node) {
  std::int32_t hint = 0;
  std::memcpy(&hint, &node->op_params[1], sizeof(hint));
  return hint;
}

}  // namespace

std::vector<execution::Choice> GraphPlan::Choices() const {
  std::vector<execution::Choice> choices;
  choices.reserve(steps.size());
  for (const PlanStep& step : steps) {
    choices.push_back(
        {.operation = step.operation, .implementation = std::string(step.implementation)});
  }
  return choices;
}

bool LaunchesNothing(const ggml_tensor* node) {
  return ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
         node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE;
}

std::expected<GraphPlan, KernelFailure> PlanGraph(GraphNodes graph, bool fusion,
                                                  const DeviceChoices& device,
                                                  std::span<ggml_tensor* const> keep) {
  if (fusion && device.row_invariant) {
    // Upstream's fused vector products pick their launch by the column
    // count: a row-invariant plan (D-092) cannot take them.
    return Rejected("a row-invariant plan is planned without fusion");
  }
  GraphPlan plan;
  std::vector<bool> taken(graph.size(), false);
  const auto add = [&](Operation operation, std::string_view name, std::size_t first,
                       std::vector<ggml_tensor*> nodes, std::size_t span) {
    for (std::size_t k = first; k < first + span; ++k) {
      taken[k] = true;
    }
    plan.steps.push_back(
        {.operation = operation, .implementation = name, .nodes = std::move(nodes), .lane = 0});
  };
  std::unordered_map<std::size_t, Dsv4HcPostExpertsNodes> deferred;
  for (std::size_t i = 0; i < graph.size(); ++i) {
    ggml_tensor* node = graph[i];
    if (taken[i] || LaunchesNothing(node)) {
      continue;
    }
    if (const auto d = deferred.find(i); d != deferred.end()) {
      const Dsv4HcPostExpertsNodes& f = d->second;
      add(Operation::kHcPost, kDsv4HcPostExpertsNormF16Name, i, {f.reduce, f.add, f.post, f.norm},
          4);
      continue;
    }
    if (fusion) {
      // ggml_cuda_try_fuse's order: the patterns jitLLM lacks must not
      // apply, then RoPE and its store, the gate/up products, the product
      // and its add, and the RMSNorm and its mul.
      if (const auto pattern = UnimplementedFusionAt(graph, i)) {
        return Rejected(
            std::format("{}: upstream's {} fusion might apply there, and no "
                        "implementation here reproduces it",
                        Where(graph, i), *pattern));
      }
      if (node->op == GGML_OP_ROPE) {
        if (const auto f = RopeSetRowsFusionAt(graph, i)) {
          add(Operation::kRopeSetRows, kRopeSetRowsFused, i, {f->rope, f->set_rows}, 3);
          continue;
        }
      }
      if (node->op == GGML_OP_MUL_MAT) {
        if (const auto f = MulMatGluFusionAt(graph, i); f && device.vector_fusible(f->up)) {
          add(Operation::kMulMatGlu, kMulMatGluFused, i, {f->gate, f->up, f->glu}, 3);
          continue;
        }
        if (const auto f = MulMatAddFusionAt(graph, i); f && device.vector_fusible(f->mul_mat)) {
          add(Operation::kMulMatAdd, kMulMatAddFused, i, {f->mul_mat, f->add}, 2);
          continue;
        }
      }
      if (node->op == GGML_OP_RMS_NORM) {
        if (const auto f = RmsNormMulFusionAt(graph, i)) {
          add(Operation::kRmsNormMul, kRmsNormMulFused, i, {f->norm, f->mul}, 2);
          continue;
        }
      }
    }
    switch (node->op) {
      case GGML_OP_RMS_NORM: {
        ggml_tensor* next = i + 1 < graph.size() ? graph[i + 1] : nullptr;
        if (const auto f =
                !fusion && device.fuse_norms ? RmsNormMulFusionAt(graph, i) : std::nullopt) {
          add(Operation::kRmsNormMul, kRmsNormMulFused, i, {f->norm, f->mul}, 2);
        } else if (!fusion && next != nullptr && next->op == GGML_OP_MUL && next->src[0] == node) {
          add(Operation::kRmsNormMul, kRmsNormMulUnfused, i, {node, next}, 2);
        } else {
          add(Operation::kRmsNorm, kRmsNormName, i, {node}, 1);
        }
        break;
      }
      case GGML_OP_MUL:
        add(Operation::kMul, kMulName, i, {node}, 1);
        break;
      case GGML_OP_ADD:
        add(Operation::kAdd, kAddName, i, {node}, 1);
        break;
      case GGML_OP_MUL_MAT: {
        if (MulMatHint(node) == GGML_HINT_SRC0_IS_HADAMARD) {
          add(Operation::kMatMul, kMulMatHadamard, i, {node}, 1);
          break;
        }
        if (device.row_invariant) {
          if (node->src[1] == nullptr || node->src[1]->ne[1] > kRowInvariantColumns) {
            return Rejected(
                std::format("{}: a row-invariant plan takes products of at most {} "
                            "columns",
                            Where(graph, i), kRowInvariantColumns));
          }
          add(Operation::kMatMul,
              IsQuantizedWeightType(node->src[0]->type) ? kMulMatVecQRows : kMulMatVecFRows, i,
              {node}, 1);
          break;
        }
        if (IsQuantizedWeightType(node->src[0]->type)) {
          if (!device.quant) {
            return Rejected(std::format("{}: no quantized product here", Where(graph, i)));
          }
          const auto path = device.quant(node);
          if (!path) {
            return Rejected(std::format("{}: {}", Where(graph, i), path.error().detail));
          }
          if (device.dense_pair && *path == QuantMulMatPath::kTile) {
            // A later dense product of the same block-quantized type and
            // activation (MulMatQPairDenseFits) joins this one.
            bool paired = false;
            for (std::size_t k = i + 1; k < graph.size() && k < i + 512; ++k) {
              ggml_tensor* other = graph[k];
              if (other->view_src != nullptr && Storage(other) == Storage(node->src[1]) &&
                  !LaunchesNothing(other)) {
                break;  // a write into the activation: the later product reads it changed
              }
              if (taken[k] || !MulMatQPairDenseFits(node, other)) {
                continue;
              }
              if (Storage(other->src[0])->op != GGML_OP_NONE) {
                // Its weights are computed (perhaps after this node): it
                // cannot run here.
                continue;
              }
              const auto other_path = device.quant(other);
              if (other_path && *other_path == QuantMulMatPath::kTile) {
                add(Operation::kMatMul, kMulMatQPairDense, i, {node, other}, 1);
                taken[k] = true;
                paired = true;
              }
              break;
            }
            if (paired) {
              break;
            }
          }
          add(Operation::kMatMul, *path == QuantMulMatPath::kVector ? kMulMatVecQ : kMulMatQ, i,
              {node}, 1);
          break;
        }
        if ((device.vector_floats ||
             (device.vector_float_node && device.vector_float_node(node))) &&
            node->src[1] != nullptr && node->src[1]->ne[1] <= kRowInvariantColumns &&
            (node->src[0]->type == GGML_TYPE_F32 || node->src[0]->type == GGML_TYPE_F16 ||
             node->src[0]->type == GGML_TYPE_BF16)) {
          add(Operation::kMatMul, kMulMatVecFRows, i, {node}, 1);
          break;
        }
        const auto path = device.mul_mat(node);
        if (!path) {
          return Rejected(std::format("{}: {}", Where(graph, i), path.error().detail));
        }
        add(Operation::kMatMul, MulMatName(*path), i, {node}, 1);
        break;
      }
      case GGML_OP_MUL_MAT_ID: {
        if (device.row_invariant) {
          if (node->ne[2] > kRowInvariantColumns) {
            return Rejected(
                std::format("{}: a row-invariant plan takes expert products of at "
                            "most {} tokens",
                            Where(graph, i), kRowInvariantColumns));
          }
          add(Operation::kMulMatId, kMulMatIdVecQRows, i, {node}, 1);
          break;
        }
        if (!device.quant) {
          return Rejected(std::format("{}: no quantized product here", Where(graph, i)));
        }
        const auto path = device.quant(node);
        if (!path) {
          return Rejected(std::format("{}: {}", Where(graph, i), path.error().detail));
        }
        if (device.d2r_experts && *path == QuantMulMatPath::kTile && device.q2_d2r_fits &&
            device.q2_d2r_fits(node)) {
          add(Operation::kMulMatId, kMulMatIdQ2D2r, i, {node}, 1);
          break;
        }
        if (device.pair_experts && *path == QuantMulMatPath::kTile && i + 1 < graph.size()) {
          ggml_tensor* second = graph[i + 1];
          if (CheckMulMatIdQPair(node, second)) {
            const auto second_path = device.quant(second);
            if (second_path && *second_path == QuantMulMatPath::kTile) {
              // The graph orders the pair as its GLU's sources: gate, then up.
              ggml_tensor* glu = i + 2 < graph.size() ? graph[i + 2] : nullptr;
              // The write-back leaves the up output unwritten: only the
              // activation may read it.
              if (device.pair_glu && device.compact_experts && device.pair_glu_fits &&
                  glu != nullptr && glu->src[0] == node && glu->src[1] == second &&
                  MulMatIdQPairGluFits(second, node, glu) && device.pair_glu_fits(second, node) &&
                  OnlyReader(graph, keep, second, glu)) {
                ggml_tensor* down = i + 3 < graph.size() ? graph[i + 3] : nullptr;
                // D2R reads the F32 activation and quantizes it itself
                // (measured faster than the quantizing write-back).
                const bool d2r_down = device.d2r_experts && device.q2_d2r_fits && down != nullptr &&
                                      down->op == GGML_OP_MUL_MAT_ID && device.q2_d2r_fits(down);
                // The quantizing write-back leaves D2S6 blocks where the F32
                // activation was: only the down product may read it.
                if (device.pair_glu_q8 && !d2r_down && down != nullptr &&
                    down->op == GGML_OP_MUL_MAT_ID && MulMatIdQCompactPrequantFits(down, glu) &&
                    CheckMulMatIdQCompact(down) && OnlyReader(graph, keep, glu, down)) {
                  const auto down_path = device.quant(down);
                  if (down_path && *down_path == QuantMulMatPath::kTile) {
                    add(Operation::kMulMatId, kMulMatIdQPairGluQ8, i, {second, node, glu}, 3);
                    add(Operation::kMulMatId, kMulMatIdQCompactPrequant, i + 3, {down}, 1);
                    break;
                  }
                }
                add(Operation::kMulMatId, kMulMatIdQPairGlu, i, {second, node, glu}, 3);
                break;
              }
              add(Operation::kMulMatId,
                  device.compact_experts ? kMulMatIdQPairCompact : kMulMatIdQPair, i,
                  {node, second}, 2);
              break;
            }
          }
        }
        const auto tile_name =
            device.compact_experts && CheckMulMatIdQCompact(node) ? kMulMatIdQCompact : kMulMatIdQ;
        add(Operation::kMulMatId, *path == QuantMulMatPath::kVector ? kMulMatIdVecQ : tile_name, i,
            {node}, 1);
        break;
      }
      case GGML_OP_ROPE:
        // The backend proof's RoPE for NEOX without frequency factors or an
        // offset (op_params[15]); any other mode or offset through the
        // extended implementation. The choice is structural, never a
        // check's verdict, so that a plan
        // names the same implementation whatever the addresses.
        if (node->op_params[2] == GGML_ROPE_TYPE_NEOX && node->src[2] == nullptr &&
            node->op_params[15] == 0) {
          add(Operation::kRope, kRopeName, i, {node}, 1);
        } else {
          add(Operation::kRope, kRopeExtName, i, {node}, 1);
        }
        break;
      case GGML_OP_ROPE_BACK:
        add(Operation::kRope, kRopeExtName, i, {node}, 1);
        break;
      case GGML_OP_SET_ROWS:
        // The backend proof's KV write takes F32 rows into F16 at I64 row
        // indices; everything else the extended implementation. (A
        // set_rows node's sources are the rows, the indices and then the
        // destination, ggml.c:4021-4023.)
        add(Operation::kSetRows,
            node->type == GGML_TYPE_F16 && node->src[0] != nullptr &&
                    node->src[0]->type == GGML_TYPE_F32 && node->src[1] != nullptr &&
                    node->src[1]->type == GGML_TYPE_I64
                ? kSetRowsName
                : kSetRowsExtName,
            i, {node}, 1);
        break;
      case GGML_OP_GET_ROWS:
        add(Operation::kGetRows,
            node->src[0]->type == GGML_TYPE_F32 || node->src[0]->type == GGML_TYPE_F16 ||
                    node->src[0]->type == GGML_TYPE_BF16
                ? kGetRowsName
                : kGetRowsExtName,
            i, {node}, 1);
        break;
      case GGML_OP_SOFT_MAX:
        add(Operation::kSoftMax, kSoftMaxName, i, {node}, 1);
        break;
      case GGML_OP_CONT:
        add(Operation::kCont, kContName, i, {node}, 1);
        break;
      case GGML_OP_GLU:
        if (ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU_CLAMP) {
          add(Operation::kSwiGluClamp, kSwiGluClampName, i, {node}, 1);
          break;
        }
        if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU || node->src[1] == nullptr) {
          return Rejected(std::format("{}: only a split SwiGLU is implemented", Where(graph, i)));
        }
        add(Operation::kSwiGlu, kSwiGluName, i, {node}, 1);
        break;
      case GGML_OP_SUB:
        add(Operation::kSub, kSubName, i, {node}, 1);
        break;
      case GGML_OP_DIV:
        add(Operation::kDiv, kDivName, i, {node}, 1);
        break;
      case GGML_OP_SCALE:
        add(Operation::kScale, kScaleName, i, {node}, 1);
        break;
      case GGML_OP_UNARY:
      case GGML_OP_SQRT:
        add(Operation::kUnary, kUnaryName, i, {node}, 1);
        break;
      case GGML_OP_CLAMP:
        add(Operation::kClamp, kClampName, i, {node}, 1);
        break;
      case GGML_OP_FILL:
        add(Operation::kFill, kFillName, i, {node}, 1);
        break;
      case GGML_OP_REPEAT:
        add(Operation::kRepeat, kRepeatName, i, {node}, 1);
        break;
      case GGML_OP_CONCAT:
        add(Operation::kConcat, kConcatName, i, {node}, 1);
        break;
      case GGML_OP_SUM_ROWS:
        add(Operation::kSumRows, kSumRowsName, i, {node}, 1);
        break;
      case GGML_OP_ARGSORT:
        add(Operation::kArgsort, kArgsortName, i, {node}, 1);
        break;
      case GGML_OP_TOP_K:
        add(Operation::kTopK, kTopKName, i, {node}, 1);
        break;
      case GGML_OP_LIGHTNING_INDEXER:
        add(Operation::kLightningIndexer, kLightningIndexerName, i, {node}, 1);
        break;
      case GGML_OP_DSV4_HC_COMB:
        add(Operation::kHcComb, kHcCombName, i, {node}, 1);
        break;
      case GGML_OP_DSV4_HC_PRE:
        add(Operation::kHcPre, kHcPreName, i, {node}, 1);
        break;
      case GGML_OP_DSV4_HC_POST:
        if (const auto f = Dsv4HcPostNormF16At(graph, i)) {
          // The post, its flat reshape and the F16 mix-input norm: one kernel.
          add(Operation::kHcPost, kDsv4HcPostNormF16Name, i, {f->post, f->norm}, 3);
          break;
        }
        add(Operation::kHcPost, kHcPostName, i, {node}, 1);
        break;
      case GGML_OP_SSM_CONV:
        add(Operation::kSsmConv, kSsmConvName, i, {node}, 1);
        break;
      case GGML_OP_GATED_DELTA_NET:
        // jitLLM's column-blocked recurrence where it takes the shape (the
        // same arithmetic, jitllm_ops.h), else upstream's.
        if (GatedDeltaNetLanesFits(node)) {
          add(Operation::kGatedDeltaNet, kGatedDeltaNetLanesName, i, {node}, 1);
        } else if (GatedDeltaNetColumnsFits(node)) {
          add(Operation::kGatedDeltaNet, kGatedDeltaNetColumnsName, i, {node}, 1);
        } else {
          add(Operation::kGatedDeltaNet, kGatedDeltaNetName, i, {node}, 1);
        }
        break;
      case GGML_OP_CUSTOM:
        switch (JitllmOpOf(node)) {
          case JitllmOp::kMxfp8MulMatVec:
            add(Operation::kMatMul, kMxfp8MulMatVecName, i, {node}, 1);
            break;
          case JitllmOp::kMxfp8Dequant:
            add(Operation::kConvert, kMxfp8DequantName, i, {node}, 1);
            break;
          case JitllmOp::kNvfp4Rows:
            add(Operation::kGetRows, kNvfp4RowsName, i, {node}, 1);
            break;
          case JitllmOp::kQRows:
            add(Operation::kGetRows, kQRowsName, i, {node}, 1);
            break;
          case JitllmOp::kHcCombine:
            add(Operation::kHcCombine, kHcCombineName, i, {node}, 1);
            break;
          case JitllmOp::kHcNorm:
            add(Operation::kHcNorm, kHcNormName, i, {node}, 1);
            break;
          case JitllmOp::kHcMix:
            add(Operation::kHcMix, kHcMixName, i, {node}, 1);
            break;
          case JitllmOp::kMoeGlu:
            add(Operation::kMoeGlu, kMoeGluName, i, {node}, 1);
            break;
          case JitllmOp::kMoeCombine:
            add(Operation::kMoeCombine, kMoeCombineName, i, {node}, 1);
            break;
          case JitllmOp::kBf16:
            add(Operation::kConvert, kBf16Name, i, {node}, 1);
            break;
          case JitllmOp::kGemmBf16:
            add(Operation::kMatMul, kGemmBf16Name, i, {node}, 1);
            break;
          case JitllmOp::kMoeRoute:
            add(Operation::kMoeRoute, kMoeRouteName, i, {node}, 1);
            break;
          case JitllmOp::kMoeQuantize:
            add(Operation::kQuantize, kMoeQuantizeName, i, {node}, 1);
            break;
          case JitllmOp::kMoeGemm:
            add(Operation::kMulMatId, kMoeGemmName, i, {node}, 1);
            break;
          case JitllmOp::kMoeGluQuantize:
            add(Operation::kMoeGlu, kMoeGluQuantizeName, i, {node}, 1);
            break;
          case JitllmOp::kMoeCombineSorted:
            add(Operation::kMoeCombine, kMoeCombineSortedName, i, {node}, 1);
            break;
          case JitllmOp::kMoeGemv:
            add(Operation::kMulMatId, kMoeGemvName, i, {node}, 1);
            break;
          case JitllmOp::kGdnConv:
            add(Operation::kSsmConv, kGdnConvName, i, {node}, 1);
            break;
          case JitllmOp::kGdnNormGate:
            add(Operation::kNormGate, kGdnNormGateName, i, {node}, 1);
            break;
          case JitllmOp::kArgmax:
            add(Operation::kTopK, kArgmaxName, i, {node}, 1);
            break;
          case JitllmOp::kMxfp8Quantize:
            add(Operation::kQuantize, kMxfp8QuantizeName, i, {node}, 1);
            break;
          case JitllmOp::kMxfp8Swizzle:
            add(Operation::kConvert, kMxfp8SwizzleName, i, {node}, 1);
            break;
          case JitllmOp::kMxfp8Gemm:
            add(Operation::kMatMul, kMxfp8GemmName, i, {node}, 1);
            break;
          case JitllmOp::kHcPrep:
            add(Operation::kHcNorm, kHcPrepName, i, {node}, 1);
            break;
          case JitllmOp::kHcLo:
            add(Operation::kUnary, kHcLoName, i, {node}, 1);
            break;
          case JitllmOp::kHcMixBf16:
            add(Operation::kHcMix, kHcMixBf16Name, i, {node}, 1);
            break;
          case JitllmOp::kMoeRouter:
            add(Operation::kArgsort, kMoeRouterName, i, {node}, 1);
            break;
          case JitllmOp::kGdnHistory:
            add(Operation::kCont, kGdnHistoryName, i, {node}, 1);
            break;
          case JitllmOp::kGdnGates:
            add(Operation::kUnary, kGdnGatesName, i, {node}, 1);
            break;
          case JitllmOp::kGdnStep:
            add(Operation::kGatedDeltaNet, kGdnStepName, i, {node}, 1);
            break;
          case JitllmOp::kQsaPrep:
            add(Operation::kRope, kQsaPrepName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4QHead:
            add(Operation::kRope, kDsv4QHeadName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4OutA:
            add(Operation::kMatMul, device.outa_fast_pack ? kDsv4OutAFastPackName : kDsv4OutAName,
                i, {node}, 1);
            break;
          case JitllmOp::kDsv4HcNormF16:
            add(Operation::kRmsNorm, kDsv4HcNormF16Name, i, {node}, 1);
            break;
          case JitllmOp::kDsv4F16Copy:
            add(Operation::kConvert, kDsv4F16CopyName, i, {node}, 1);
            break;
          case JitllmOp::kQsaGateQuantize:
            add(Operation::kQuantize, kQsaGateQuantizeName, i, {node}, 1);
            break;
          case JitllmOp::kQsaPool:
            add(Operation::kRope, kQsaPoolName, i, {node}, 1);
            break;
          case JitllmOp::kQsaTopK:
            add(Operation::kTopK, kQsaTopKName, i, {node}, 1);
            break;
          case JitllmOp::kQsaAttn:
            add(Operation::kFlashAttn, kQsaAttnName, i, {node}, 1);
            break;
          case JitllmOp::kQuantizeQ8:
            add(Operation::kQuantize, kQuantizeQ8Name, i, {node}, 1);
            break;
          case JitllmOp::kVecQ:
            add(Operation::kMatMul, kVecQName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4Route:
            add(Operation::kMoeRoute, kDsv4RouteName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4Combine:
            add(Operation::kMoeCombine, kDsv4CombineName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4WeightedReduce:
            if (device.hc_post_experts) {
              std::size_t at = 0;
              if (const auto f = Dsv4HcPostExpertsAt(graph, i, &at)) {
                // Deferred to the post: it runs at the add's place.
                taken[i] = true;
                deferred.emplace(at, *f);
                break;
              }
            }
            add(Operation::kMoeCombine, kDsv4WeightedReduceName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4HcMix:
            add(Operation::kHcMix, kDsv4HcMixName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4HcPre:
            add(Operation::kHcPre, kDsv4HcPreName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4Compress:
            add(Operation::kSoftMax, kDsv4CompressName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4LidTopK:
            add(Operation::kLightningIndexer, kDsv4LidTopKName, i, {node}, 1);
            break;
          case JitllmOp::kDsv4SparseMask:
            add(Operation::kFill, kDsv4SparseMaskName, i, {node}, 1);
            break;
          case JitllmOp::kNone:
            return Rejected(
                std::format("{}: a custom operation jitLLM does not name", Where(graph, i)));
        }
        break;
      case GGML_OP_FLASH_ATTN_EXT:
        if (node->src[0] == nullptr || (node->src[0]->ne[0] != 256 && node->src[0]->ne[0] != 512)) {
          return Rejected(
              std::format("{}: flash attention is planned at head dimensions 256 and "
                          "512 only",
                          Where(graph, i)));
        }
        if (device.ds4_hca && device.ds4_hca_fits && device.ds4_hca_fits(node)) {
          add(Operation::kFlashAttn, kDsv4HcaTokentileName, i, {node}, 1);
          break;
        }
        add(Operation::kFlashAttn,
            device.wide_sparse_attention &&
                    (JitllmOpOf(node->src[3]) != JitllmOp::kDsv4SparseMask ||
                     JitllmOpInt(node->src[3], 1) != 1)
                ? kFlashAttnMmaWideName
                : kFlashAttnMmaName,
            i, {node}, 1);
        break;
      default:
        return Rejected(std::format("{}: no implementation of this operation", Where(graph, i)));
    }
  }
  return plan;
}

bool SamePlan(const GraphPlan& a, const GraphPlan& b) {
  return std::ranges::equal(a.steps, b.steps, [](const PlanStep& x, const PlanStep& y) {
    return x.operation == y.operation && x.implementation == y.implementation &&
           x.nodes == y.nodes && x.lane == y.lane;
  });
}

void AssignLanes(GraphPlan& plan, const LaneTags& tags,
                 const std::function<bool(std::string_view)>& on_stream) {
  plan.regions.clear();
  if (tags.empty()) {
    return;
  }
  const auto stays = [&](std::string_view implementation) {
    return on_stream ? on_stream(implementation)
                     : implementation == kMulMatCublas || implementation == kGemmBf16Name;
  };
  std::unordered_map<const ggml_tensor*, LaneTag> tag;
  tag.reserve(tags.size());
  for (const auto& [tensor, t] : tags) {
    tag.emplace(tensor, t);
  }
  std::map<std::uint32_t, GraphPlan::Region> spans;
  for (std::size_t s = 0; s < plan.steps.size(); ++s) {
    PlanStep& step = plan.steps[s];
    step.lane = 0;
    for (const ggml_tensor* node : step.nodes) {
      const auto found = tag.find(node);
      if (found == tag.end() || found->second.region == 0) {
        continue;
      }
      const LaneTag& t = found->second;
      const auto at = static_cast<std::uint32_t>(s);
      const auto [it, added] = spans.try_emplace(t.region, GraphPlan::Region{at, at});
      if (!added) {
        it->second.first = std::min(it->second.first, at);
        it->second.last = std::max(it->second.last, at);
      }
      if (t.lane <= kMaxLanes && !stays(step.implementation)) {
        step.lane = t.lane;
      }
      break;
    }
  }
  std::vector<GraphPlan::Region> ordered;
  ordered.reserve(spans.size());
  for (const auto& [region, span] : spans) {
    ordered.push_back(span);
  }
  std::ranges::sort(ordered, {}, &GraphPlan::Region::first);
  for (const GraphPlan::Region& span : ordered) {
    if (!plan.regions.empty() && span.first <= plan.regions.back().last) {
      plan.regions.back().last = std::max(plan.regions.back().last, span.last);
    } else {
      plan.regions.push_back(span);
    }
  }
  // A lane step outside every region would have no fork or join.
  std::size_t r = 0;
  for (std::size_t s = 0; s < plan.steps.size(); ++s) {
    while (r < plan.regions.size() && plan.regions[r].last < s) {
      ++r;
    }
    if (r == plan.regions.size() || s < plan.regions[r].first) {
      plan.steps[s].lane = 0;
    }
  }
}

std::vector<std::uint8_t> LaneOrder::Before(std::uint8_t lane,
                                            std::span<const ggml_tensor* const> reads,
                                            std::span<const ggml_tensor* const> writes) const {
  std::array<bool, kLanes> need{};
  const auto after = [&](std::uint8_t other, std::uint64_t at) {
    if (other != lane && at != 0 && seen_[lane][other] < at) {
      need[other] = true;
    }
  };
  for (const ggml_tensor* t : reads) {
    if (const auto found = access_.find(t); found != access_.end()) {
      after(found->second.writer, found->second.written);
    }
  }
  for (const ggml_tensor* t : writes) {
    if (const auto found = access_.find(t); found != access_.end()) {
      after(found->second.writer, found->second.written);
      for (std::size_t other = 0; other < kLanes; ++other) {
        after(static_cast<std::uint8_t>(other), found->second.read[other]);
      }
    }
  }
  std::vector<std::uint8_t> out;
  for (std::size_t other = 0; other < kLanes; ++other) {
    if (need[other]) {
      out.push_back(static_cast<std::uint8_t>(other));
    }
  }
  return out;
}

void LaneOrder::Waited(std::uint8_t waiter, std::uint8_t on) { seen_[waiter][on] = queued_[on]; }

void LaneOrder::Queued(std::uint8_t lane, std::span<const ggml_tensor* const> reads,
                       std::span<const ggml_tensor* const> writes) {
  const std::uint64_t at = ++queued_[lane];
  for (const ggml_tensor* t : reads) {
    access_[t].read[lane] = at;
  }
  // A write waited for every earlier reader and writer on other lanes, and
  // follows its own lane's in order: only it is outstanding now.
  for (const ggml_tensor* t : writes) {
    Access& a = access_[t];
    a = Access{};
    a.writer = lane;
    a.written = at;
  }
}

void LaneOrder::Reset() { access_.clear(); }

std::expected<Placement, KernelFailure> PlaceActivations(GraphNodes graph, const GraphPlan& plan,
                                                         std::span<ggml_tensor* const> inputs,
                                                         std::uint64_t alignment,
                                                         std::span<ggml_tensor* const> keep) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Rejected("the alignment is not a power of two");
  }
  struct Life {
    ggml_tensor* tensor = nullptr;
    std::int64_t first = 0;  // the step computing it; -1 for an input
    std::int64_t last = 0;   // the last step reading it
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;
    bool placed = false;
  };
  std::vector<Life> lives;
  std::unordered_map<const ggml_tensor*, std::size_t> index;
  const auto track = [&](ggml_tensor* t, std::int64_t first) {
    if (index.contains(t)) {
      return;
    }
    index.emplace(t, lives.size());
    lives.push_back(
        {.tensor = t, .first = first, .last = first, .bytes = Rounded(ggml_nbytes(t), alignment)});
  };
  for (ggml_tensor* input : inputs) {
    if (input->op != GGML_OP_NONE || input->view_src != nullptr) {
      return Rejected("an input to place is not a leaf");
    }
    track(input, -1);
  }
  for (std::size_t s = 0; s < plan.steps.size(); ++s) {
    for (ggml_tensor* node : plan.steps[s].nodes) {
      if (Computed(node)) {
        track(node, static_cast<std::int64_t>(s));
      }
    }
  }
  for (const ggml_tensor* node : graph) {
    if (Computed(node) && !index.contains(node)) {
      return Rejected(std::format("{} is computed by no step", node->name));
    }
  }
  for (std::size_t s = 0; s < plan.steps.size(); ++s) {
    for (const ggml_tensor* node : plan.steps[s].nodes) {
      for (const ggml_tensor* src : node->src) {
        if (src == nullptr) {
          continue;
        }
        if (const auto found = index.find(Storage(src)); found != index.end()) {
          Life& life = lives[found->second];
          life.last = std::max(life.last, static_cast<std::int64_t>(s));
        }
      }
    }
  }
  // Concurrent lanes (AssignLanes): what a region's steps compute or read
  // lives for the region's whole span, as its steps run in any order.
  for (const GraphPlan::Region& region : plan.regions) {
    const auto first = static_cast<std::int64_t>(region.first);
    const auto last = static_cast<std::int64_t>(region.last);
    const auto widen = [&](const ggml_tensor* t) {
      if (const auto found = index.find(Storage(t)); found != index.end()) {
        Life& life = lives[found->second];
        life.first = life.first < 0 ? life.first : std::min(life.first, first);
        life.last = std::max(life.last, last);
      }
    };
    for (std::size_t s = region.first; s <= region.last && s < plan.steps.size(); ++s) {
      for (const ggml_tensor* node : plan.steps[s].nodes) {
        widen(node);
        for (const ggml_tensor* src : node->src) {
          if (src != nullptr) {
            widen(src);
          }
        }
      }
    }
  }
  // The graph's output lives to the end, in whatever tensor it views.
  if (!graph.empty()) {
    if (const auto found = index.find(Storage(graph.back())); found != index.end()) {
      lives[found->second].last = static_cast<std::int64_t>(plan.steps.size());
    }
  }
  for (const ggml_tensor* kept : keep) {
    if (kept == nullptr) {
      continue;
    }
    if (const auto found = index.find(Storage(kept)); found != index.end()) {
      lives[found->second].last = static_cast<std::int64_t>(plan.steps.size());
    }
  }
  std::vector<std::size_t> order(lives.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::ranges::stable_sort(
      order, [&](std::size_t a, std::size_t b) { return lives[a].bytes > lives[b].bytes; });
  Placement placement;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> busy;  // offset, end
  for (const std::size_t i : order) {
    Life& life = lives[i];
    busy.clear();
    for (const Life& other : lives) {
      if (other.placed && other.first <= life.last && life.first <= other.last) {
        busy.emplace_back(other.offset, other.offset + other.bytes);
      }
    }
    std::ranges::sort(busy);
    std::uint64_t offset = 0;
    for (const auto& [start, end] : busy) {
      if (start >= offset + life.bytes) {
        break;
      }
      offset = std::max(offset, end);
    }
    life.offset = offset;
    life.placed = true;
    placement.extent = std::max(placement.extent, offset + life.bytes);
  }
  placement.offsets.reserve(lives.size());
  for (const Life& life : lives) {
    placement.offsets.emplace_back(life.tensor, life.offset);
  }
  return placement;
}

void BindViews(std::span<ggml_tensor* const> nodes) {
  for (ggml_tensor* node : nodes) {
    if (node->view_src != nullptr) {
      TensorArena::Bind(node,
                        reinterpret_cast<std::uintptr_t>(node->view_src->data) + node->view_offs);
    }
  }
}

void BindDistinct(GraphNodes graph, std::uint64_t base) {
  std::uint64_t next = base;
  for (ggml_tensor* node : graph) {
    if (Computed(node)) {
      TensorArena::Bind(node, next);
      next += Rounded(ggml_nbytes(node), 256) + 256;
    }
  }
  BindViews(graph);
}

}  // namespace jitllm::kernels::ggml
