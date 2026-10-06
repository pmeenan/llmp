// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/fattn_owner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {
namespace {
using detail::Rejected;
constexpr std::uint64_t kMaxSpan = 64ULL << 20U;

// Bound types/dimensions before any GGML helper; bounded view traversal also
// rejects cycles and stale/overflowed addresses without following source DAGs.
std::optional<std::uint64_t> Span(const ggml_tensor* t) {
  if (!t || !t->data || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32)) return std::nullopt;
  std::uint64_t bytes = t->type == GGML_TYPE_F16 ? 2 : 4;
  for (int i = 0; i < 4; ++i) {
    if (t->ne[i] <= 0 || t->ne[i] > 16777216 || t->nb[i] > kMaxSpan) return std::nullopt;
    std::uint64_t part = 0;
    if (__builtin_mul_overflow(std::uint64_t(t->ne[i] - 1), t->nb[i], &part) ||
        __builtin_add_overflow(bytes, part, &bytes) || bytes > kMaxSpan)
      return std::nullopt;
  }
  std::uint64_t end = 0;
  if (__builtin_add_overflow(std::uint64_t(reinterpret_cast<std::uintptr_t>(t->data)), bytes, &end))
    return std::nullopt;
  return bytes;
}
bool Current(const ggml_tensor* t) {
  std::array<const ggml_tensor*, 64> seen{};
  std::size_t count = 0;
  while (t) {
    if (count == seen.size() ||
        std::find(seen.begin(), seen.begin() + count, t) != seen.begin() + count)
      return false;
    seen[count++] = t;
    const auto bytes = Span(t);
    if (!bytes || reinterpret_cast<std::uintptr_t>(t->data) % 16 != 0) return false;
    const auto* parent = t->view_src;
    if (!parent) return true;
    const auto parent_bytes = Span(parent);
    std::uint64_t current = 0;
    if (!parent_bytes || t->view_offs > *parent_bytes || *bytes > *parent_bytes - t->view_offs ||
        __builtin_add_overflow(std::uint64_t(reinterpret_cast<std::uintptr_t>(parent->data)),
                               std::uint64_t(t->view_offs), &current) ||
        current != reinterpret_cast<std::uintptr_t>(t->data))
      return false;
    t = parent;
  }
  return false;
}
bool Shape(const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne,
           std::array<std::size_t, 4> nb) {
  return t && t->type == type && std::equal(ne.begin(), ne.end(), t->ne) &&
         std::equal(nb.begin(), nb.end(), t->nb) && Current(t);
}
bool Overlap(const ggml_tensor* a, const ggml_tensor* b) {
  const auto first = reinterpret_cast<std::uintptr_t>(a->data);
  const auto second = reinterpret_cast<std::uintptr_t>(b->data);
  return first < second + *Span(b) && second < first + *Span(a);
}
}  // namespace

std::expected<void, KernelFailure> CheckFlashAttnOwners(const FlashAttnOwners& in) {
  if (!in.q || !in.output || (in.q->ne[0] != 256 && in.q->ne[0] != 512) ||
      (in.q->ne[2] != 16 && in.q->ne[2] != 32))
    return Rejected("owner MMA requires the closed Gemma head dimensions");
  const auto d = std::size_t(in.q->ne[0]), heads = std::size_t(in.q->ne[2]);
  const auto kvh = heads / (d == 256 ? 2 : 8);
  const auto qrow = d * heads * sizeof(float), kvrow = d * kvh * 2;
  if (!Shape(in.q, GGML_TYPE_F32, {std::int64_t(d), 1, std::int64_t(heads), 4},
             {4, qrow, d * 4, qrow}) ||
      !Shape(in.mask, GGML_TYPE_F16, {256, 32, 1, 4}, {2, 512, 16384, 16384}) ||
      !Shape(in.output, GGML_TYPE_F32, {std::int64_t(d), std::int64_t(heads), 1, 4},
             {4, d * 4, qrow, qrow}) ||
      in.output->view_src)
    return Rejected("owner MMA requires current packed Q/mask/output stream layouts");
  std::array<const ggml_tensor*, 10> reads{in.q, in.mask};
  for (std::size_t owner = 0; owner < 4; ++owner) {
    for (const auto* tensor : {in.k[owner], in.v[owner]})
      if (!Shape(tensor, GGML_TYPE_F16, {std::int64_t(d), 256, std::int64_t(kvh), 1},
                 {2, kvrow, d * 2, kvrow * 256}))
        return Rejected("owner MMA requires actual current cell-major F16 cache views");
    reads[2 + owner] = in.k[owner];
    reads[6 + owner] = in.v[owner];
  }
  for (const auto* input : reads)
    if (Overlap(in.output, input)) return Rejected("owner MMA output overlaps a real operand");
  // Independent owner storage: aliases within an owner are read-only, but one
  // owner's K/V must not be another owner's payload.
  for (std::size_t a = 0; a < 4; ++a)
    for (std::size_t b = a + 1; b < 4; ++b)
      for (const auto* first : {in.k[a], in.v[a]})
        for (const auto* second : {in.k[b], in.v[b]})
          if (Overlap(first, second)) return Rejected("owner MMA cache owners overlap");
  return {};
}
std::expected<FlashAttnOwners, KernelFailure> FlashAttnOwnersFromNode(ggml_tensor* node) {
  static_assert(GGML_MAX_SRC == 10);
  if (!node || JitllmOpOf(node) != JitllmOp::kFlashAttnOwners || node->view_src)
    return Rejected("not a ten-source owner attention custom node");
  for (int i = 0; i < 8; ++i)
    if (JitllmOpInt(node, i) != 0) return Rejected("owner attention has unsupported parameters");
  FlashAttnOwners in{.q = node->src[0], .mask = node->src[1], .output = node};
  for (std::size_t owner = 0; owner < 4; ++owner) {
    in.k[owner] = node->src[2 + owner];
    in.v[owner] = node->src[6 + owner];
  }
  if (auto checked = CheckFlashAttnOwners(in); !checked) return std::unexpected(checked.error());
  return in;
}
std::expected<void, KernelFailure> CheckFlashAttnOwnersNode(const ggml_tensor* node) {
  auto in = FlashAttnOwnersFromNode(const_cast<ggml_tensor*>(node));
  if (!in) return std::unexpected(in.error());
  return {};
}
}  // namespace jitllm::kernels::ggml
