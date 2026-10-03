// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen2 chunk graph (kernels/ggml/qwen2_graph.h) and its planning and
// placement (graph_plan.h), in every profile (backend-proof P2):
// - the graph has llama.cpp's nodes in llama.cpp's order: 941 on the GPU,
//   which with the embedding lookup it runs on the CPU are the 942 "graph
//   nodes" the bridge logs, 39 per layer in the order its recorded plan
//   launches them;
// - with fusion on, upstream's patterns fuse where the recorded plan fused
//   (the RMSNorm and its mul, RoPE and the K write, and in a decode step
//   the products with their bias or residual and the gate and up products
//   with SwiGLU), and no pattern jitLLM lacks applies anywhere; with fusion
//   off, only the unfused RMSNorm-mul pairs its two launchers;
// - the planner refuses a graph where an unimplemented pattern might apply;
// - placed activations reproduce the pre-registered limit A exactly (rows ×
//   611,328 bytes: the head's F32 input and logits), and no step's output
//   shares a byte with anything it reads, and a graph ending in a view
//   keeps the viewed tensor's bytes to the end.
// A model of GB10's kernel-family choice stands in for the device, which
// the GPU runs check (docs/experiments/backend-proof-p2/).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/qwen2_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen2.h"

namespace {

namespace kg = jitllm::kernels::ggml;
using kg::MulMatPath;

// GB10's choices for this model, as the recorded plan shows them: MMVF for
// one column, MMF up to 16, cuBLAS beyond.
kg::DeviceChoices ModelDevice() {
  return {.mul_mat = [](const ggml_tensor* node) -> std::expected<MulMatPath, kg::KernelFailure> {
            const std::int64_t columns = node->src[1]->ne[1];
            if (columns == 1) {
              return MulMatPath::kVector;
            }
            return columns <= 16 ? MulMatPath::kTensorCore : MulMatPath::kCublas;
          },
          .vector_fusible = [](const ggml_tensor* node) { return node->src[1]->ne[1] == 1; },
          .quant = nullptr};
}

struct Chunk {
  std::optional<kg::TensorArena> arena;
  kg::Qwen2Graph graph;
};

Chunk Build(std::int64_t rows, std::int64_t n_kv, std::int64_t cells) {
  Chunk c;
  auto arena = kg::TensorArena::Create(kg::Qwen2GraphTensors(jitllm::model::Qwen25Instruct05B()));
  EXPECT_TRUE(arena.has_value());
  c.arena.emplace(std::move(*arena));
  auto graph = kg::BuildQwen2Graph(*c.arena, jitllm::model::Qwen25Instruct05B(),
                                   {.rows = rows, .n_kv = n_kv, .cells = cells});
  EXPECT_TRUE(graph.has_value()) << (graph ? "" : graph.error().detail);
  c.graph = std::move(*graph);
  // Leaves anywhere 256-aligned, computed tensors at distinct addresses.
  std::uint64_t leaf = std::uint64_t{1} << 36U;
  for (ggml_tensor* t : c.graph.inputs()) {
    kg::TensorArena::Bind(t, leaf);
    leaf += (ggml_nbytes(t) + 511) / 256 * 256;
  }
  kg::BindDistinct(c.graph.nodes, std::uint64_t{1} << 40U);
  return c;
}

std::vector<ggml_op> Ops(const std::vector<ggml_tensor*>& nodes, std::size_t from,
                         std::size_t count) {
  std::vector<ggml_op> ops;
  for (std::size_t i = from; i < from + count; ++i) {
    ops.push_back(nodes[i]->op);
  }
  return ops;
}

constexpr std::size_t kLayerNodes = 39;

const std::vector<ggml_op>& LayerOps() {
  static const std::vector<ggml_op> ops = {
      GGML_OP_RMS_NORM, GGML_OP_MUL,      GGML_OP_MUL_MAT,  GGML_OP_ADD,      GGML_OP_RESHAPE,
      GGML_OP_ROPE,     GGML_OP_MUL_MAT,  GGML_OP_ADD,      GGML_OP_RESHAPE,  GGML_OP_MUL_MAT,
      GGML_OP_ADD,      GGML_OP_RESHAPE,  GGML_OP_ROPE,     GGML_OP_VIEW,     GGML_OP_SET_ROWS,
      GGML_OP_RESHAPE,  GGML_OP_RESHAPE,  GGML_OP_RESHAPE,  GGML_OP_SET_ROWS, GGML_OP_VIEW,
      GGML_OP_PERMUTE,  GGML_OP_VIEW,     GGML_OP_PERMUTE,  GGML_OP_VIEW,     GGML_OP_PERMUTE,
      GGML_OP_MUL_MAT,  GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT,  GGML_OP_PERMUTE,  GGML_OP_CONT,
      GGML_OP_MUL_MAT,  GGML_OP_ADD,      GGML_OP_RMS_NORM, GGML_OP_MUL,      GGML_OP_MUL_MAT,
      GGML_OP_MUL_MAT,  GGML_OP_GLU,      GGML_OP_MUL_MAT,  GGML_OP_ADD};
  return ops;
}

TEST(Qwen2GraphTest, HasLlamaCppsNodesInItsOrder) {
  const Chunk c = Build(32, 256, 512);
  const auto& nodes = c.graph.nodes;
  ASSERT_EQ(nodes.size(), 941U);  // + the CPU's embedding lookup = the bridge's 942
  for (std::size_t layer = 0; layer < 23; ++layer) {
    EXPECT_EQ(Ops(nodes, layer * kLayerNodes, kLayerNodes), LayerOps()) << "layer " << layer;
  }
  // The last layer keeps only the output rows before its residual add.
  const std::size_t last = 23 * kLayerNodes;
  std::vector<ggml_op> want(LayerOps().begin(), LayerOps().begin() + 31);
  want.push_back(GGML_OP_GET_ROWS);
  want.push_back(GGML_OP_GET_ROWS);
  want.insert(want.end(), LayerOps().begin() + 31, LayerOps().end());
  EXPECT_EQ(Ops(nodes, last, want.size()), want);
  EXPECT_EQ(Ops(nodes, nodes.size() - 3, 3),
            (std::vector<ggml_op>{GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT}));
  EXPECT_EQ(nodes.back(), c.graph.logits);
  EXPECT_EQ(c.graph.logits->ne[0], 151936);
  EXPECT_EQ(c.graph.logits->ne[1], 32);
  // Attention reads n_kv cells of the 512-cell cache, and KQ accumulates in F32.
  const ggml_tensor* kq = nodes[25];
  EXPECT_EQ(kq->ne[0], 256);
  EXPECT_EQ(kq->ne[1], 32);
  EXPECT_EQ(kq->ne[2], 14);
  EXPECT_EQ(kq->op_params[0], GGML_PREC_F32);
  EXPECT_EQ(c.graph.mask->ne[0], 256);
  EXPECT_EQ(c.graph.v_idxs->ne[0], 32 * 128);
}

TEST(Qwen2GraphTest, RefusesAShapeTheCacheCannotHold) {
  auto arena = kg::TensorArena::Create(kg::Qwen2GraphTensors(jitllm::model::Qwen25Instruct05B()));
  ASSERT_TRUE(arena.has_value());
  const auto& p = jitllm::model::Qwen25Instruct05B();
  EXPECT_FALSE(kg::BuildQwen2Graph(*arena, p, {.rows = 32, .n_kv = 16, .cells = 512}).has_value());
  EXPECT_FALSE(kg::BuildQwen2Graph(*arena, p, {.rows = 1, .n_kv = 1024, .cells = 512}).has_value());
  EXPECT_FALSE(kg::BuildQwen2Graph(*arena, p, {.rows = 0, .n_kv = 256, .cells = 512}).has_value());
}

std::vector<std::string_view> Names(const kg::GraphPlan& plan, std::size_t from,
                                    std::size_t count) {
  std::vector<std::string_view> names;
  for (std::size_t i = from; i < from + count && i < plan.steps.size(); ++i) {
    names.push_back(plan.steps[i].implementation);
  }
  return names;
}

TEST(Qwen2GraphTest, FusesADecodeStepWhereTheRecordFused) {
  const Chunk c = Build(1, 256, 512);
  for (std::size_t i = 0; i < c.graph.nodes.size(); ++i) {
    EXPECT_FALSE(kg::UnimplementedFusionAt(c.graph.nodes, i).has_value()) << i;
  }
  const auto plan = kg::PlanGraph(c.graph.nodes, true, ModelDevice());
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  // The recorded plan's fused decode layer: kernels 23, 17, 25, 17, 17, 24,
  // 8, 21, 27, 14, a copy, 17, 23, 20, 19.
  const std::vector<std::string_view> layer = {
      kg::kRmsNormMulFused, kg::kMulMatAddFused,   kg::kRopeName,      kg::kMulMatAddFused,
      kg::kMulMatAddFused,  kg::kRopeSetRowsFused, kg::kSetRowsName,   kg::kMulMatVector,
      kg::kSoftMaxName,     kg::kMulMatVector,     kg::kContName,      kg::kMulMatAddFused,
      kg::kRmsNormMulFused, kg::kMulMatGluFused,   kg::kMulMatAddFused};
  ASSERT_EQ(plan->steps.size(), (23 * layer.size()) + 18 + 2);
  for (std::size_t l = 0; l < 23; ++l) {
    EXPECT_EQ(Names(*plan, l * layer.size(), layer.size()), layer) << "layer " << l;
  }
  EXPECT_EQ(
      Names(*plan, 23 * layer.size(), 17),
      (std::vector<std::string_view>{
          kg::kRmsNormMulFused, kg::kMulMatAddFused, kg::kRopeName, kg::kMulMatAddFused,
          kg::kMulMatAddFused, kg::kRopeSetRowsFused, kg::kSetRowsName, kg::kMulMatVector,
          kg::kSoftMaxName, kg::kMulMatVector, kg::kContName, kg::kMulMatVector, kg::kGetRowsName,
          kg::kGetRowsName, kg::kAddName, kg::kRmsNormMulFused, kg::kMulMatGluFused}));
  EXPECT_EQ(Names(*plan, plan->steps.size() - 3, 3),
            (std::vector<std::string_view>{kg::kMulMatAddFused, kg::kRmsNormMulFused,
                                           kg::kMulMatVector}));
  // The fused products name their nodes in the implementations' order.
  const kg::PlanStep& glu = plan->steps[13];
  ASSERT_EQ(glu.nodes.size(), 3U);
  EXPECT_EQ(glu.nodes[2]->src[0], glu.nodes[0]);  // gate
  EXPECT_EQ(glu.nodes[2]->src[1], glu.nodes[1]);  // up
}

TEST(Qwen2GraphTest, RunsNodeByNodeWithFusionOff) {
  const Chunk c = Build(1, 256, 512);
  const auto plan = kg::PlanGraph(c.graph.nodes, false, ModelDevice());
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  const std::vector<std::string_view> layer = {
      kg::kRmsNormMulUnfused, kg::kMulMatVector,      kg::kAddName,      kg::kRopeName,
      kg::kMulMatVector,      kg::kAddName,           kg::kMulMatVector, kg::kAddName,
      kg::kRopeName,          kg::kSetRowsName,       kg::kSetRowsName,  kg::kMulMatVector,
      kg::kSoftMaxName,       kg::kMulMatVector,      kg::kContName,     kg::kMulMatVector,
      kg::kAddName,           kg::kRmsNormMulUnfused, kg::kMulMatVector, kg::kMulMatVector,
      kg::kSwiGluName,        kg::kMulMatVector,      kg::kAddName};
  ASSERT_EQ(plan->steps.size(), (23 * layer.size()) + 25 + 2);
  for (std::size_t l = 0; l < 23; ++l) {
    EXPECT_EQ(Names(*plan, l * layer.size(), layer.size()), layer) << "layer " << l;
  }
  EXPECT_EQ(Names(*plan, plan->steps.size() - 2, 2),
            (std::vector<std::string_view>{kg::kRmsNormMulUnfused, kg::kMulMatVector}));
}

TEST(Qwen2GraphTest, APrefillFusesOnlyWhatNeedsNoVectorKernel) {
  const Chunk c = Build(32, 256, 512);
  const auto plan = kg::PlanGraph(c.graph.nodes, true, ModelDevice());
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(Names(*plan, 0, 12),
            (std::vector<std::string_view>{kg::kRmsNormMulFused, kg::kMulMatCublas, kg::kAddName,
                                           kg::kRopeName, kg::kMulMatCublas, kg::kAddName,
                                           kg::kMulMatCublas, kg::kAddName, kg::kRopeSetRowsFused,
                                           kg::kSetRowsName, kg::kMulMatCublas, kg::kSoftMaxName}));
  EXPECT_EQ(std::ranges::count(Names(*plan, 0, plan->steps.size()), kg::kMulMatGluFused), 0);
}

TEST(Qwen2GraphTest, RefusesAPatternWithNoImplementation) {
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  ggml_tensor* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 4);
  ggml_tensor* b = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 4);
  ggml_tensor* w = ggml_new_tensor_2d(c, GGML_TYPE_F16, 64, 64);
  const auto refused = [&](ggml_tensor* out, std::string_view pattern) {
    std::vector<ggml_tensor*> outputs = {out};
    const std::vector<ggml_tensor*> nodes = kg::GraphOrder(outputs);
    kg::BindDistinct(nodes, std::uint64_t{1} << 40U);
    EXPECT_EQ(kg::UnimplementedFusionAt(nodes, 0), pattern);
    EXPECT_FALSE(kg::PlanGraph(nodes, true, ModelDevice()).has_value());
    EXPECT_TRUE(kg::PlanGraph(nodes, false, ModelDevice()).has_value() || pattern == "SCALE");
  };
  refused(ggml_add(c, ggml_add(c, a, b), b), "add or mul chain");
  refused(ggml_mul(c, ggml_mul_mat(c, w, a), b), "mul_mat and scale");
  refused(ggml_add(c, ggml_mul(c, ggml_rms_norm(c, a, 1e-6f), b), b),
          "rms_norm, mul and rope or add");
  refused(ggml_scale(c, a, 2.0f), "SCALE");
}

// Where each placed tensor lives.
std::map<const ggml_tensor*, std::pair<std::uint64_t, std::uint64_t>> Ranges(
    const kg::Placement& placement) {
  std::map<const ggml_tensor*, std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (const auto& [tensor, offset] : placement.offsets) {
    ranges[tensor] = {offset, offset + ggml_nbytes(tensor)};
  }
  return ranges;
}

TEST(Qwen2GraphTest, PlacesActivationsInTheLimitA) {
  struct Shape {
    std::int64_t rows, n_kv, cells;
  };
  for (const Shape s : {Shape{1, 256, 512}, Shape{32, 256, 512}, Shape{16, 256, 1024},
                        Shape{17, 256, 1024}, Shape{512, 768, 1024}, Shape{1, 768, 1024}}) {
    for (const bool fusion : {true, false}) {
      const Chunk c = Build(s.rows, s.n_kv, s.cells);
      const auto plan = kg::PlanGraph(c.graph.nodes, fusion, ModelDevice());
      ASSERT_TRUE(plan.has_value()) << plan.error().detail;
      const auto inputs = c.graph.inputs();
      const auto placement = kg::PlaceActivations(c.graph.nodes, *plan, inputs, 128);
      ASSERT_TRUE(placement.has_value()) << placement.error().detail;
      EXPECT_EQ(placement->extent, static_cast<std::uint64_t>(s.rows) * 611328)
          << s.rows << " rows, fusion " << fusion;
      // No step writes a byte that it, or a node fused into it, reads.
      const auto ranges = Ranges(*placement);
      for (const kg::PlanStep& step : plan->steps) {
        for (const ggml_tensor* node : step.nodes) {
          if (!ranges.contains(node)) {
            continue;
          }
          const auto [lo, hi] = ranges.at(node);
          for (const ggml_tensor* src : node->src) {
            if (src == nullptr) {
              continue;
            }
            const ggml_tensor* root = src->view_src != nullptr ? src->view_src : src;
            if (!ranges.contains(root) || root == node) {
              continue;
            }
            const auto [slo, shi] = ranges.at(root);
            EXPECT_TRUE(hi <= slo || shi <= lo) << ggml_op_desc(node) << " over its input";
          }
        }
      }
      for (const auto& [tensor, offset] : placement->offsets) {
        EXPECT_EQ(offset % 128, 0U);
      }
    }
  }
}

// A graph whose last node is a view keeps the viewed tensor alive to the
// end: in `a, b, view(a)`, b must not take a's bytes, or it overwrites the
// output. The same through a view of the view.
TEST(Qwen2GraphTest, AFinalViewKeepsItsSourceAlive) {
  for (const bool chained : {false, true}) {
    auto arena = kg::TensorArena::Create(8);
    ASSERT_TRUE(arena.has_value());
    ggml_context* c = arena->context();
    ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
    ggml_tensor* a = ggml_add(c, x, x);
    ggml_tensor* b = ggml_mul(c, x, x);
    ggml_tensor* last = ggml_view_1d(c, a, 512, 0);
    std::vector<ggml_tensor*> nodes = {a, b, last};
    if (chained) {
      last = ggml_view_1d(c, last, 256, 0);
      nodes.push_back(last);
    }
    const auto plan = kg::PlanGraph(nodes, false, ModelDevice());
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    ASSERT_EQ(plan->steps.size(), 2U);
    const std::array<ggml_tensor*, 1> inputs = {x};
    const auto placement = kg::PlaceActivations(nodes, *plan, inputs, 128);
    ASSERT_TRUE(placement.has_value()) << placement.error().detail;
    const auto ranges = Ranges(*placement);
    ASSERT_TRUE(ranges.contains(a) && ranges.contains(b));
    const auto [alo, ahi] = ranges.at(a);
    const auto [blo, bhi] = ranges.at(b);
    EXPECT_TRUE(ahi <= blo || bhi <= alo)
        << "chained " << chained << ": a at " << alo << ", b at " << blo;
  }
}

// Concurrent lanes (graph_plan.h AssignLanes): steps take their nodes'
// lanes, a region spans its tagged steps (overlapping ones joined), and
// what a region's steps compute or read lives for the whole span, so two
// lanes' tensors never share bytes though each dies early in step order.
TEST(Qwen2GraphTest, LanesSpanRegionsAndKeepTheirTensorsApart) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
  ggml_tensor* before = ggml_add(c, x, x);           // the stream's, before
  ggml_tensor* a1 = ggml_mul(c, before, before);     // lane 1
  ggml_tensor* a2 = ggml_add(c, a1, a1);             // lane 1, a1 dies
  ggml_tensor* b1 = ggml_mul(c, before, x);          // lane 2
  ggml_tensor* b2 = ggml_add(c, b1, b1);             // lane 2, b1 dies
  ggml_tensor* joined = ggml_add(c, a2, b2);         // the stream's, in the region
  ggml_tensor* after = ggml_mul(c, joined, joined);  // after the region
  const std::vector<ggml_tensor*> nodes = {before, a1, a2, b1, b2, joined, after};
  kg::LaneTags tags = {{a1, {1, 7}}, {a2, {1, 7}}, {b1, {2, 7}}, {b2, {2, 7}}, {joined, {0, 7}}};
  auto plan = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  ASSERT_EQ(plan->steps.size(), nodes.size());
  kg::AssignLanes(*plan, tags);
  std::vector<std::uint8_t> lanes;
  for (const kg::PlanStep& step : plan->steps) {
    lanes.push_back(step.lane);
  }
  EXPECT_EQ(lanes, (std::vector<std::uint8_t>{0, 1, 1, 2, 2, 0, 0}));
  ASSERT_EQ(plan->regions.size(), 1U);
  EXPECT_EQ(plan->regions[0].first, 1U);
  EXPECT_EQ(plan->regions[0].last, 5U);
  const std::array<ggml_tensor*, 1> inputs = {x};
  const auto placement = kg::PlaceActivations(nodes, *plan, inputs, 128);
  ASSERT_TRUE(placement.has_value()) << placement.error().detail;
  const auto ranges = Ranges(*placement);
  // On one stream a1 could share bytes with b1 or b2; on lanes they run at
  // once, so the region's tensors all stay apart.
  const std::vector<ggml_tensor*> region = {before, a1, a2, b1, b2, joined};
  for (std::size_t i = 0; i < region.size(); ++i) {
    for (std::size_t j = i + 1; j < region.size(); ++j) {
      const auto [ilo, ihi] = ranges.at(region[i]);
      const auto [jlo, jhi] = ranges.at(region[j]);
      EXPECT_TRUE(ihi <= jlo || jhi <= ilo) << i << " and " << j;
    }
  }
  // The same plan without lanes: no regions, every step on the stream.
  auto plain = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(plain.has_value());
  EXPECT_TRUE(plain->regions.empty());
  EXPECT_FALSE(kg::SamePlan(*plan, *plain));
  // Overlapping regions are joined; a lane step outside any region (none
  // tagged with a region) stays on the stream; a cuBLAS product keeps
  // lane 0 within its region.
  kg::LaneTags overlapping = {{a1, {1, 1}}, {a2, {1, 2}}, {b1, {2, 2}}, {b2, {2, 1}}};
  kg::AssignLanes(*plain, overlapping);
  ASSERT_EQ(plain->regions.size(), 1U);
  EXPECT_EQ(plain->regions[0].first, 1U);
  EXPECT_EQ(plain->regions[0].last, 4U);
  kg::AssignLanes(*plain, {{a1, {1, 0}}});
  EXPECT_TRUE(plain->regions.empty());
  EXPECT_EQ(plain->steps[1].lane, 0U);
  // A step whose implementation must stay on the stream (one borrowing
  // cuBLAS) keeps lane 0 inside its region; the region is unchanged.
  kg::AssignLanes(*plan, tags,
                  [](std::string_view implementation) { return implementation == kg::kMulName; });
  lanes.clear();
  for (const kg::PlanStep& step : plan->steps) {
    lanes.push_back(step.lane);
  }
  EXPECT_EQ(lanes, (std::vector<std::uint8_t>{0, 0, 1, 0, 2, 0, 0}));
  ASSERT_EQ(plan->regions.size(), 1U);
  EXPECT_EQ(plan->regions[0].first, 1U);
  EXPECT_EQ(plan->regions[0].last, 5U);
}

// The order concurrent lanes keep inside a region (graph_plan.h LaneOrder):
// a read waits for the tensor's last writer on another lane, a write for its
// last writer and for every other lane that read it since, a lane never
// waits for itself, a wait covers everything the awaited lane queued, and a
// region's end clears it all.
TEST(Qwen2GraphTest, LanesOrderEveryReadAndWriteAcrossLanes) {
  auto arena = kg::TensorArena::Create(8);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  const ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
  const ggml_tensor* y = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
  const ggml_tensor* z = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
  using Lanes = std::vector<std::uint8_t>;
  using Ts = std::vector<const ggml_tensor*>;
  kg::LaneOrder order;
  // Lane 1 writes x; lane 2 reads it (read after write).
  EXPECT_EQ(order.Before(1, Ts{}, Ts{x}), Lanes{});
  order.Queued(1, Ts{}, Ts{x});
  EXPECT_EQ(order.Before(1, Ts{x}, Ts{y}), Lanes{});  // its own lane, in order
  EXPECT_EQ(order.Before(2, Ts{x}, Ts{y}), Lanes{1});
  order.Waited(2, 1);
  EXPECT_EQ(order.Before(2, Ts{x}, Ts{y}), Lanes{});  // the wait covers it
  order.Queued(2, Ts{x}, Ts{y});
  // Lane 3 overwrites x: after its writer (lane 1) and its reader (lane 2).
  EXPECT_EQ(order.Before(3, Ts{}, Ts{x}), (Lanes{1, 2}));
  // The stream writes y, which lane 2 wrote (write after write).
  EXPECT_EQ(order.Before(0, Ts{}, Ts{y}), Lanes{2});
  order.Waited(3, 1);
  order.Waited(3, 2);
  order.Queued(3, Ts{}, Ts{x});
  // Readers before lane 3's write are settled; a later read of x follows
  // lane 3 alone; z, untouched, needs nothing.
  EXPECT_EQ(order.Before(1, Ts{x}, Ts{z}), Lanes{3});
  EXPECT_EQ(order.Before(4, Ts{z}, Ts{z}), Lanes{});
  // A wait covers only what was queued by then.
  order.Waited(4, 3);
  order.Queued(3, Ts{}, Ts{z});
  EXPECT_EQ(order.Before(4, Ts{z}, Ts{}), Lanes{3});
  order.Reset();
  EXPECT_EQ(order.Before(4, Ts{x, y, z}, Ts{x, y, z}), Lanes{});
}

TEST(Qwen2GraphTest, AnUnchangedPlanIsTheSamePlan) {
  const Chunk c = Build(1, 256, 512);
  const auto a = kg::PlanGraph(c.graph.nodes, true, ModelDevice());
  const auto b = kg::PlanGraph(c.graph.nodes, true, ModelDevice());
  const auto u = kg::PlanGraph(c.graph.nodes, false, ModelDevice());
  ASSERT_TRUE(a && b && u);
  EXPECT_TRUE(kg::SamePlan(*a, *b));
  EXPECT_FALSE(kg::SamePlan(*a, *u));
  const auto choices = a->Choices();
  ASSERT_EQ(choices.size(), a->steps.size());
  EXPECT_EQ(choices.front().implementation, "ggml.rms_norm_mul.fused");
  EXPECT_EQ(choices.front().operation, jitllm::execution::Operation::kRmsNormMul);
}

}  // namespace
