// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_BENCHMARK_GEMMA26_LATE_MOE_GRAPHS_H_
#define LLMP_BENCHMARK_GEMMA26_LATE_MOE_GRAPHS_H_
#include <array>
#include <limits>
#include <vector>

#include "ggml.h"
namespace late_moe {
static_assert(GGML_TYPE_F32 == 0 && GGML_TYPE_I32 == 26);
inline constexpr int kRows = 64;
struct Graph {
  std::vector<ggml_tensor*> inputs, nodes, outputs;
};
inline Graph Routing(ggml_context* context) {
  auto* logits = ggml_new_tensor_2d(context, GGML_TYPE_F32, 128, kRows);
  auto* probabilities = ggml_soft_max(context, logits);
  auto* reshaped = ggml_reshape_3d(context, probabilities, 1, 128, kRows);
  auto* ids = ggml_argsort_top_k(context, probabilities, 8);
  auto* gathered = ggml_get_rows(context, reshaped, ids);
  auto* selected = ggml_reshape_2d(context, gathered, 8, kRows);
  auto* denominator = ggml_sum_rows(context, selected);
  auto* clamp = ggml_clamp(context, denominator, 0x1p-14f, std::numeric_limits<float>::infinity());
  auto* divided = ggml_div(context, selected, clamp);
  auto* weights = ggml_reshape_3d(context, divided, 1, 8, kRows);
  return {{logits},
          {probabilities, reshaped, ids->view_src, ids, gathered, selected, denominator, clamp,
           divided, weights},
          {ids, weights}};
}
inline Graph Reduction(ggml_context* context) {
  auto* experts = ggml_new_tensor_3d(context, GGML_TYPE_F32, 2816, 8, kRows);
  auto* scales = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 8, kRows);
  auto* weights = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 8, kRows);
  auto* scaled = ggml_mul(context, experts, scales);
  auto* weighted = ggml_mul(context, scaled, weights);
  Graph graph{{experts, scales, weights}, {scaled, weighted}, {}};
  std::array<ggml_tensor*, 8> parts{};
  for (std::size_t i = 0; i < parts.size(); ++i) {
    parts[i] = ggml_view_2d(context, weighted, 2816, kRows, weighted->nb[2], i * weighted->nb[1]);
    graph.nodes.push_back(parts[i]);
  }
  auto* value = parts[0];
  for (std::size_t i = 1; i < parts.size(); ++i) {
    value = ggml_add(context, value, parts[i]);
    graph.nodes.push_back(value);
  }
  graph.outputs = {value};
  return graph;
}
}  // namespace late_moe
#endif
