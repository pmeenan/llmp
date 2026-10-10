// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Internal to src/components (and its tests): the pieces feed-forward
// components are built of, each an IR fragment and a scalar reference
// written independently of it, so each is tested alone and in the
// components that use it (ffn.glu, ffn.moe).
//
//   - A gated feed-forward product: the gate and up products of one input
//     (the gate's first), the GLU, the down product.
//   - A router: its input (Gemma 4's: an unweighted RMS norm of the
//     stream, scaled by 1 / sqrt(width) and a learned per-channel scale),
//     its product with the router matrix, and the selection (ir::kRoute:
//     softmax over the experts, the `used` highest, their weights
//     renormalized when `normalize`).
//   - Routed experts: each selected expert's gate and up products of the
//     routed input (a stacked gate-up array's halves, or two arrays), the
//     GLU, the down product, each slot's learned expert scale gathered
//     (when there is one), and the ordered reduce (ir::kExpertReduce: each
//     slot's output times its scale, times its weight, summed in slot
//     order).
//
// The fragments emit in the retired Gemma 4 builder's order (its
// GraphOrder), with the router's selection and the
// reduce as the op set's composite ops, whose decompositions are GGML's
// node sequences (llama.cpp's build_moe_ffn): a plan runs them as those
// primitives or, where its policy keeps the planner's route_reduce
// pattern, as an implementation of each composite (GGML's fused Gemma
// routing and scaled reduce).

#ifndef LLMP_COMPONENTS_MOE_H_
#define LLMP_COMPONENTS_MOE_H_

#include <cstdint>
#include <span>
#include <vector>

#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::components::moe {

// ------------------------------------------------------ gated feed-forward

// down · glu(gate · h, up · h): h [width, rows] to [width, rows]; a clamped
// SwiGLU's `limit` (ir::GluKind::kSwiGluClamp: silu(min(gate, limit)) ·
// clamp(up, -limit, limit)).
ir::TensorId EmitGlu(ir::Builder& b, ir::TensorId h, ir::TensorId gate, ir::TensorId up,
                     ir::TensorId down, ir::GluKind kind, float limit = 0.0F);
// The same over decoded weights ([width, hidden] gate and up, [hidden,
// width] down, GGML order) and `rows` rows of h.
std::vector<double> GluReference(std::span<const double> h, std::span<const float> gate,
                                 std::span<const float> up, std::span<const float> down,
                                 std::uint64_t width, std::uint64_t hidden, ir::GluKind kind,
                                 float limit = 0.0F);
// glu(gate, up) of one value pair.
double GluOf(ir::GluKind kind, double gate, double up, float limit);

// ------------------------------------------------------------------ router

struct RouterParams {
  std::uint32_t experts = 0;
  std::uint32_t used = 0;
  bool normalize = false;
  float eps = 0.0F;
  // Its input Gemma 4's scaled norm of the stream (with `scale`); false:
  // the stream itself (Qwen3.8's).
  bool rms_input = true;
  // The probabilities: softmax (kSoftmaxTopK), or sqrt(softplus) chosen by
  // the highest plus a bias per expert (kSqrtSoftplusBias) or by each
  // row's ids looked up already (kHash: a token-hash side input's), as
  // DeepSeek V4 routes; the selected weights times `route_scale`; the
  // router's product summed in F32 (`f32`).
  ir::RouteKind kind = ir::RouteKind::kSoftmaxTopK;
  float route_scale = 1.0F;
  bool f32 = false;
};
struct Routed {
  ir::TensorId ids{};      // I32 [used, rows]
  ir::TensorId weights{};  // F32 [used, rows]
};
// x [width, rows], router [width, experts], scale [width] (none without
// rms_input), bias [experts] (kSqrtSoftplusBias's), ids I32 [used, rows]
// (kHash's).
Routed EmitRouter(ir::Builder& b, ir::TensorId x, ir::TensorId router, ir::TensorId scale,
                  const RouterParams& p, ir::TensorId bias = {}, ir::TensorId ids = {});
struct RoutedRef {
  std::vector<std::uint32_t> ids;  // [rows][used]
  std::vector<double> weights;     // [rows][used]
};
// `ids` [rows][used]: kHash's.
RoutedRef RouterReference(std::span<const double> x, std::uint64_t width,
                          std::span<const float> router, std::span<const float> scale,
                          const RouterParams& p, std::span<const float> bias = {},
                          std::span<const std::int64_t> ids = {});

// ----------------------------------------------------------------- experts

struct ExpertParams {
  std::uint64_t width = 0;
  std::uint64_t hidden = 0;  // an expert's
  bool stacked = false;      // one gate-up array: gate's rows, then up's
  ir::GluKind kind = ir::GluKind::kGeGluTanh;
  float limit = 0.0F;  // a clamped SwiGLU's
};
// h [width, rows]; gate_up [width, 2 · hidden, experts] when stacked, else
// gate and up [width, hidden, experts]; down [hidden, width, experts];
// scale [experts] or none: each slot's down output's; gate_scale and
// up_scale [experts] or none: each slot's gate and up products' (NVFP4's
// per-expert tensor scales kept beside the experts, applied after each
// product as llama.cpp's build_lora_mm_id applies a `_s` tensor). The
// reduced output [width, rows].
ir::TensorId EmitExperts(ir::Builder& b, ir::TensorId h, const Routed& routed, ir::TensorId gate,
                         ir::TensorId up, ir::TensorId down, ir::TensorId scale,
                         const ExpertParams& p, ir::TensorId gate_scale = {},
                         ir::TensorId up_scale = {});
// The same over decoded expert arrays (each expert's matrix after the
// last, GGML order; `up` unused when stacked) and the routing.
std::vector<double> ExpertsReference(std::span<const double> h, const RoutedRef& routed,
                                     std::span<const float> gate, std::span<const float> up,
                                     std::span<const float> down, std::span<const float> scale,
                                     const ExpertParams& p, std::span<const float> gate_scale = {},
                                     std::span<const float> up_scale = {});

}  // namespace llmp::components::moe

#endif  // LLMP_COMPONENTS_MOE_H_
