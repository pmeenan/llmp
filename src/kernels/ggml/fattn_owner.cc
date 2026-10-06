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
// A cache view executes only its bounded prefix. Its full approved 262K
// cell-major backing is an address-validation parent, never a launch operand.
constexpr std::uint64_t kMaxParentSpan = 1ULL << 30U;

// Bound types/dimensions before any GGML helper; bounded view traversal also
// rejects cycles and stale/overflowed addresses without following source DAGs.
std::optional<std::uint64_t> Span(const ggml_tensor* t, std::uint64_t limit = kMaxSpan) {
  if (!t || !t->data || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32)) return std::nullopt;
  std::uint64_t bytes = t->type == GGML_TYPE_F16 ? 2 : 4;
  for (int i = 0; i < 4; ++i) {
    if (t->ne[i] <= 0 || t->ne[i] > 16777216 || t->nb[i] > limit) return std::nullopt;
    std::uint64_t part = 0;
    if (__builtin_mul_overflow(std::uint64_t(t->ne[i] - 1), t->nb[i], &part) ||
        __builtin_add_overflow(bytes, part, &bytes) || bytes > limit)
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
    const auto bytes = Span(t, count == 1 ? kMaxSpan : kMaxParentSpan);
    if (!bytes || reinterpret_cast<std::uintptr_t>(t->data) % 16 != 0) return false;
    const auto* parent = t->view_src;
    if (!parent) return true;
    const auto parent_bytes = Span(parent, kMaxParentSpan);
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

std::expected<detail::OwnerPartition, KernelFailure> detail::PlanOwnerPartition(
    int max_blocks, int kv_tiles, int kv_heads, std::uint32_t logical_cohort) {
  if (max_blocks <= 0 || kv_tiles <= 0 || kv_tiles > 512 || kv_heads <= 0 || kv_heads > 16 ||
      (logical_cohort != 2 && logical_cohort != 3 && logical_cohort != 4 && logical_cohort != 8 &&
       logical_cohort != 12))
    return Rejected("owner MMA partition inputs are outside the closed grid bounds");
  const auto grid = [&](std::uint32_t cohort) {
    const auto tiles = static_cast<std::int64_t>(kv_heads) * cohort;
    const auto raw = std::min(static_cast<std::int64_t>(max_blocks), kv_tiles * tiles);
    const auto rounded = raw / tiles * tiles;
    const auto loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    return static_cast<int>(loss <= 5 ? rounded : raw);
  };
  if (logical_cohort == 2 || logical_cohort == 3) {
    const auto blocks = grid(logical_cohort);
    return OwnerPartition{
        .cohort_blocks = blocks, .quad_blocks = blocks, .effective_cohort = logical_cohort};
  }
  auto cohort = logical_cohort;
  auto blocks = grid(cohort);
  if (blocks % static_cast<int>(cohort / 4) != 0) {
    cohort = 4;
    blocks = grid(cohort);
  }
  return OwnerPartition{.cohort_blocks = blocks,
                        .quad_blocks = blocks / static_cast<int>(cohort / 4),
                        .effective_cohort = cohort};
}

std::expected<void, KernelFailure> CheckFlashAttnOwners(const FlashAttnOwners& in) {
  if (in.owner_count < 2 || in.owner_count > 4 ||
      (in.owner_count < 4
           ? in.logical_cohort != in.owner_count
           : (in.logical_cohort != 4 && in.logical_cohort != 8 && in.logical_cohort != 12)))
    return Rejected("owner MMA requires actual roots2/3 or four-root cohort4/8/12");
  if (!in.q || !in.mask || !in.output || (in.q->ne[0] != 256 && in.q->ne[0] != 512) ||
      (in.q->ne[2] != 16 && in.q->ne[2] != 32))
    return Rejected("owner MMA requires the closed Gemma head dimensions");
  if (in.mask->ne[0] < 256 || in.mask->ne[0] > 16384 || in.mask->ne[0] % 256 != 0)
    return Rejected("owner MMA requires bounded actual padded cache widths");
  const auto cells = std::size_t(in.mask->ne[0]);
  const auto d = std::size_t(in.q->ne[0]), heads = std::size_t(in.q->ne[2]);
  const auto kvh = heads / (d == 256 ? 2 : 8);
  const auto qrow = d * heads * sizeof(float), kvrow = d * kvh * 2;
  if (!Shape(in.q, GGML_TYPE_F32, {std::int64_t(d), 1, std::int64_t(heads), in.owner_count},
             {4, qrow, d * 4, qrow}) ||
      !Shape(in.mask, GGML_TYPE_F16, {std::int64_t(cells), 32, 1, in.owner_count},
             {2, cells * 2, cells * 64, cells * 64}) ||
      !Shape(in.output, GGML_TYPE_F32, {std::int64_t(d), std::int64_t(heads), 1, in.owner_count},
             {4, d * 4, qrow, qrow}) ||
      in.output->view_src)
    return Rejected("owner MMA requires current packed Q/mask/output stream layouts");
  std::array<const ggml_tensor*, 10> reads{in.q, in.mask};
  for (std::size_t owner = in.owner_count; owner < 4; ++owner)
    if (in.k[owner] || in.v[owner]) return Rejected("owner MMA has extra inactive cache roots");
  for (std::size_t owner = 0; owner < in.owner_count; ++owner) {
    for (const auto* tensor : {in.k[owner], in.v[owner]})
      if (!Shape(tensor, GGML_TYPE_F16,
                 {std::int64_t(d), std::int64_t(cells), std::int64_t(kvh), 1},
                 {2, kvrow, d * 2, kvrow * cells}))
        return Rejected("owner MMA requires actual current cell-major F16 cache views");
    reads[2 + owner] = in.k[owner];
    reads[6 + owner] = in.v[owner];
  }
  for (const auto* input : reads)
    if (input && Overlap(in.output, input))
      return Rejected("owner MMA output overlaps a real operand");
  // Independent owner storage: aliases within an owner are read-only, but one
  // owner's K/V must not be another owner's payload.
  for (std::size_t a = 0; a < in.owner_count; ++a)
    for (std::size_t b = a + 1; b < in.owner_count; ++b)
      for (const auto* first : {in.k[a], in.v[a]})
        for (const auto* second : {in.k[b], in.v[b]})
          if (Overlap(first, second)) return Rejected("owner MMA cache owners overlap");
  return {};
}
std::expected<FlashAttnOwners, KernelFailure> FlashAttnOwnersFromNode(ggml_tensor* node) {
  static_assert(GGML_MAX_SRC == 10);
  if (!node || JitllmOpOf(node) != JitllmOp::kFlashAttnOwners || node->view_src)
    return Rejected("not a ten-source owner attention custom node");
  const auto cohort = JitllmOpInt(node, 0);
  if (cohort != 2 && cohort != 3 && cohort != 4 && cohort != 8 && cohort != 12)
    return Rejected("owner attention has an unsupported logical cohort");
  const auto encoded_count = JitllmOpInt(node, 1);
  if (encoded_count != 0 && encoded_count != 2 && encoded_count != 3)
    return Rejected("owner attention has an unsupported active root count");
  for (int i = 2; i < 8; ++i)
    if (JitllmOpInt(node, i) != 0) return Rejected("owner attention has unsupported parameters");
  FlashAttnOwners in{
      .q = node->src[0],
      .mask = node->src[1],
      .output = node,
      .logical_cohort = static_cast<std::uint32_t>(cohort),
      .owner_count = encoded_count == 0 ? 4U : static_cast<std::uint32_t>(encoded_count)};
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
