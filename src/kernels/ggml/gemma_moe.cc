// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_moe.h"

#include <array>
#include <limits>
#include <span>

#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
bool Shape(const ggml_tensor* t, const std::array<std::int64_t, 4>& shape) {
  return t != nullptr && std::ranges::equal(std::span(t->ne), shape);
}
bool Current(const ggml_tensor* t) {
  // Bound the borrowed chain before using the ordinary stale-view guard.
  const auto* p = t;
  for (unsigned i = 0; p != nullptr && i < 64; ++i) {
    if (p->view_src == nullptr) return detail::Current(t);
    if (p->view_offs > std::numeric_limits<std::uint64_t>::max() -
                           reinterpret_cast<std::uintptr_t>(p->view_src->data))
      return false;
    p = p->view_src;
  }
  return false;
}
bool Storage(const GemmaMoeOperand& o, std::uint64_t needed) {
  if (!detail::Bound(o.tensor) || !detail::Aligned(o.tensor, 4) || !Current(o.tensor)) return false;
  const auto extent = detail::Extent(o.tensor);
  const auto address = reinterpret_cast<std::uintptr_t>(o.tensor->data);
  return extent && *extent <= needed && o.bytes.value() >= needed &&
         o.bytes.value() <= std::numeric_limits<std::uint64_t>::max() - address;
}
bool Overlap(const ggml_tensor* a, std::uint64_t a_bytes, const ggml_tensor* b,
             std::uint64_t b_bytes) {
  const auto ab = reinterpret_cast<std::uintptr_t>(a->data);
  const auto bb = reinterpret_cast<std::uintptr_t>(b->data);
  return ab < bb + b_bytes && bb < ab + a_bytes;
}
bool Canonical(const ggml_tensor* t, ggml_type type, const std::array<std::int64_t, 4>& shape) {
  // Check fixed bounded shapes before any GGML stride/byte arithmetic.
  return Shape(t, shape) && t->type == type && detail::Packed(t);
}
}  // namespace

std::expected<void, KernelFailure> CheckGemmaRouting(const GemmaRouting& d) {
  const auto* x = d.logits.tensor;
  const auto rows = x == nullptr ? 0 : x->ne[1];
  const auto experts = x == nullptr ? 0 : x->ne[0];
  const auto used = d.weights.tensor == nullptr ? 0 : d.weights.tensor->ne[1];
  if (!GemmaRoutingShapeFits(experts, used, rows) ||
      !Canonical(x, GGML_TYPE_F32, {experts, rows, 1, 1}) ||
      !Canonical(d.weights.tensor, GGML_TYPE_F32, {1, used, rows, 1}) ||
      d.ids_use != GemmaRouteIdsUse::kSelectedTop8 || d.denominator_min != kGemmaRouteClamp)
    return detail::Rejected(
        "routing requires a compiled expert count, valid top-k and exact clamp");
  const auto* ids = d.ids.tensor;
  const auto* root = ids == nullptr ? nullptr : ids->view_src;
  if (!Shape(ids, {used, rows, 1, 1}) || ids->type != GGML_TYPE_I32 || ids->op != GGML_OP_VIEW ||
      ids->view_offs != 0 || !Canonical(root, GGML_TYPE_I32, {experts, rows, 1, 1}) ||
      ids->nb[0] != 4 || ids->nb[1] != static_cast<std::uint64_t>(experts) * 4 ||
      ids->nb[2] != static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(experts) * 4 ||
      ids->nb[3] != ids->nb[2] || ids->data != root->data)
    return detail::Rejected(
        "Gemma routing requires a current selected view of full expert-pitch backing");
  const auto xb = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(experts) * 4;
  const auto wb = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(used) * 4;
  if (!Storage(d.logits, xb) || !Storage(d.weights, wb) || !Storage(d.ids, xb) ||
      !Storage({root, d.ids.bytes}, xb))
    return detail::Rejected("Gemma routing operand backing is short, stale or invalid");
  if (Overlap(x, xb, ids, xb) || Overlap(x, xb, d.weights.tensor, wb) ||
      Overlap(ids, xb, d.weights.tensor, wb))
    return detail::Rejected("Gemma routing outputs overlap operands or full sort backing");
  return {};
}

std::expected<void, KernelFailure> CheckGemmaScaledReduction(const GemmaScaledReduction& d) {
  const auto rows = d.experts.tensor == nullptr ? 0 : d.experts.tensor->ne[2];
  const auto width = d.experts.tensor == nullptr ? 0 : d.experts.tensor->ne[0];
  const auto used = d.experts.tensor == nullptr ? 0 : d.experts.tensor->ne[1];
  if (!GemmaReductionShapeFits(width, used, rows) ||
      !Canonical(d.experts.tensor, GGML_TYPE_F32, {width, used, rows, 1}) ||
      !Canonical(d.scales.tensor, GGML_TYPE_F32, {1, used, rows, 1}) ||
      !Canonical(d.weights.tensor, GGML_TYPE_F32, {1, used, rows, 1}) ||
      !Canonical(d.values.tensor, GGML_TYPE_F32, {width, rows, 1, 1}))
    return detail::Rejected(
        "scaled reduction requires canonical F32 rows within grid/index bounds");
  const auto r = static_cast<std::uint64_t>(rows);
  const std::array operands = {d.experts, d.scales, d.weights, d.values};
  const std::array<std::uint64_t, 4> bytes = {
      r * static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(used) * 4,
      r * static_cast<std::uint64_t>(used) * 4, r * static_cast<std::uint64_t>(used) * 4,
      r * static_cast<std::uint64_t>(width) * 4};
  for (std::size_t i = 0; i < operands.size(); ++i)
    if (!Storage(operands[i], bytes[i]))
      return detail::Rejected("Gemma scaled reduction operand backing is short, stale or invalid");
  for (std::size_t i = 0; i < 3; ++i)
    if (Overlap(d.values.tensor, bytes[3], operands[i].tensor, bytes[i]))
      return detail::Rejected("Gemma scaled reduction output overlaps an input");
  return {};
}
}  // namespace llmp::kernels::ggml
