// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/fattn_owner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
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
bool Current(const ggml_tensor* t, std::uint64_t limit) {
  std::array<const ggml_tensor*, 64> seen{};
  std::size_t count = 0;
  while (t) {
    if (count == seen.size() ||
        std::find(seen.begin(), seen.begin() + count, t) != seen.begin() + count)
      return false;
    seen[count++] = t;
    const auto bytes = Span(t, count == 1 ? limit : kMaxParentSpan);
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
           std::array<std::size_t, 4> nb, std::uint64_t limit = kMaxSpan) {
  return t && t->type == type && std::equal(ne.begin(), ne.end(), t->ne) &&
         std::equal(nb.begin(), nb.end(), t->nb) && Current(t, limit);
}
bool Overlap(const ggml_tensor* a, const ggml_tensor* b, std::uint64_t a_limit,
             std::uint64_t b_limit) {
  const auto first = reinterpret_cast<std::uintptr_t>(a->data);
  const auto second = reinterpret_cast<std::uintptr_t>(b->data);
  const auto a_bytes = Span(a, a_limit), b_bytes = Span(b, b_limit);
  if (!a_bytes || !b_bytes) return true;
  return first < second + *b_bytes && second < first + *a_bytes;
}
}  // namespace

std::expected<detail::OwnerPartition, KernelFailure> detail::PlanOwnerPartition(
    int max_blocks, int kv_tiles, int tiles_per_owner, std::uint32_t logical_cohort,
    bool prefer_whole_tiles, bool wide_kv, int query_tiles) {
  if (max_blocks <= 0 || kv_tiles <= 0 || kv_tiles > (wide_kv ? 4096 : 512) || query_tiles < 0 ||
      query_tiles > 16 || tiles_per_owner <= 0 || tiles_per_owner > (query_tiles ? 64 : 16) ||
      (logical_cohort != 2 && logical_cohort != 3 && logical_cohort != 4 && logical_cohort != 8 &&
       logical_cohort != 12 && !PartialOwnerCohort(logical_cohort)))
    return Rejected("owner MMA partition inputs are outside the closed grid bounds");
  if ((wide_kv && (logical_cohort != 2 || tiles_per_owner % 4 != 0)) ||
      (query_tiles && (logical_cohort != 2 || tiles_per_owner != 4 * query_tiles)))
    return Rejected("multirow owner partition requires C2 and one-to-sixteen GQA2 query tiles");
  const auto grid = [&](std::uint32_t cohort) {
    const auto tiles = static_cast<std::int64_t>(tiles_per_owner) * cohort;
    const auto waves = (tiles + max_blocks - 1) / max_blocks;
    if (prefer_whole_tiles && 100 * tiles / (max_blocks * waves) >= 75)
      return static_cast<int>(tiles);
    const auto raw = std::min(static_cast<std::int64_t>(max_blocks), kv_tiles * tiles);
    const auto rounded = raw / tiles * tiles;
    const auto loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    return static_cast<int>(loss <= 5 ? rounded : raw);
  };
  if (logical_cohort == 2 || logical_cohort == 3 || PartialOwnerCohort(logical_cohort)) {
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
  const bool partial = detail::PartialOwnerCohort(in.logical_cohort);
  if (partial) {
    if (in.owner_offset >= in.logical_cohort || in.owner_offset % 4 != 0 ||
        in.owner_count != std::min(4U, in.logical_cohort - in.owner_offset))
      return Rejected("partial owner MMA requires a canonical real root group");
  } else if (in.owner_offset != 0 || in.owner_count < 2 || in.owner_count > 4 ||
             (in.owner_count < 4 ? in.logical_cohort != in.owner_count
                                 : (in.logical_cohort != 4 && in.logical_cohort != 8 &&
                                    in.logical_cohort != 12))) {
    return Rejected("owner MMA requires actual roots2/3 or four-root cohort4/8/12");
  }
  if (!in.q || !in.mask || !in.output || (in.q->ne[0] != 256 && in.q->ne[0] != 512) ||
      (in.q->ne[2] != 16 && in.q->ne[2] != 32 &&
       !(in.q->ne[0] == 256 && in.q->ne[2] == 8 &&
         ((in.logical_cohort >= 2 && in.logical_cohort <= 4) || in.logical_cohort == 8 ||
          in.logical_cohort == 12 || partial))))
    return Rejected("owner MMA requires the closed Gemma head dimensions");
  if (in.logit_softcap != 0 &&
      (in.logit_softcap != 50 || in.q->ne[0] != 256 || in.q->ne[2] != 8 || in.owner_count != 2 ||
       in.logical_cohort != 2 || in.owner_offset != 0))
    return Rejected("owner MMA softcap requires the closed Gemma2 D256/H8/C2 cap50 path");
  const bool bounded_shape = (in.logit_softcap == 50 && in.q->ne[0] == 256 && in.q->ne[2] == 8) ||
                             (in.logit_softcap == 0 && ((in.q->ne[0] == 256 && in.q->ne[2] == 8) ||
                                                        in.q->ne[2] == 16 || in.q->ne[2] == 32));
  const bool bounded_whole12 = in.logit_softcap == 0 && in.q->ne[0] == 256 && in.q->ne[2] == 8 &&
                               in.owner_count == 4 && in.logical_cohort == 12 &&
                               in.owner_offset == 0;
  const bool bounded_c2 =
      bounded_shape && in.owner_count == 2 && in.logical_cohort == 2 && in.owner_offset == 0;
  if (in.bounded_roots && !bounded_c2 && !bounded_whole12)
    return Rejected("bounded owner roots require closed C2 or Gemma3 whole-C12 carriers");
  const auto rows = in.q->ne[1];
  if (rows != 1 && (rows < 2 || rows > 512 || in.q->ne[0] != 256 || in.q->ne[2] != 8 ||
                    in.owner_count != 2 || in.logical_cohort != 2 || in.owner_offset != 0 ||
                    (in.logit_softcap != 0 && in.logit_softcap != 50) || in.bounded_roots))
    return Rejected("multirow owner MMA requires Gemma2/Gemma3 cap0/50 C2/2-to-512-row inputs");
  const bool wide_prefill = rows > 1 && in.logit_softcap == 0;
  const auto maximum_cells = wide_prefill ? 131072 : 16384;
  // Only the closed Gemma3 multirow contract needs 256 MiB per actual root.
  // Q/output remain at most 8 MiB; the largest joined mask is exactly 256 MiB.
  const auto operand_limit = wide_prefill ? (256ULL << 20U) : kMaxSpan;
  const auto mask_limit = wide_prefill ? (256ULL << 20U) : kMaxSpan;
  if (in.mask->ne[0] < 256 || in.mask->ne[0] > maximum_cells || in.mask->ne[0] % 256 != 0)
    return Rejected("owner MMA requires bounded actual padded cache widths");
  const auto cells = std::size_t(in.mask->ne[0]);
  const auto d = std::size_t(in.q->ne[0]), heads = std::size_t(in.q->ne[2]);
  const auto kvh = heads / (d == 256 ? 2 : 8);
  const auto qrow = d * heads * sizeof(float), kvrow = d * kvh * 2;
  const auto mask_rows = std::size_t((rows + 31) / 32 * 32);
  if (!Shape(in.q, GGML_TYPE_F32, {std::int64_t(d), rows, std::int64_t(heads), in.owner_count},
             {4, qrow, d * 4, qrow * std::size_t(rows)}) ||
      !Shape(in.mask, GGML_TYPE_F16,
             {std::int64_t(cells), std::int64_t(mask_rows), 1, in.owner_count},
             {2, cells * 2, cells * mask_rows * 2, cells * mask_rows * 2}, mask_limit) ||
      !Shape(in.output, GGML_TYPE_F32, {std::int64_t(d), std::int64_t(heads), rows, in.owner_count},
             {4, d * 4, qrow, qrow * std::size_t(rows)}) ||
      in.output->view_src)
    return Rejected("owner MMA requires current packed Q/mask/output stream layouts");
  std::array<const ggml_tensor*, 10> reads{in.q, in.mask};
  for (std::size_t owner = in.owner_count; owner < 4; ++owner)
    if (in.k[owner] || in.v[owner]) return Rejected("owner MMA has extra inactive cache roots");
  std::size_t largest_actual = 0;
  for (std::size_t owner = 0; owner < in.owner_count; ++owner) {
    const auto* key = in.k[owner];
    const auto* value = in.v[owner];
    if (!key || !value || key->ne[1] != value->ne[1])
      return Rejected("owner MMA requires matching actual K/V widths");
    if (in.bounded_roots && (key->ne[1] < 256 || key->ne[1] > static_cast<std::int64_t>(cells) ||
                             key->ne[1] % 256 != 0))
      return Rejected("bounded owner roots require aligned actual widths within the logical mask");
    const auto actual = in.bounded_roots ? std::size_t(key->ne[1]) : cells;
    largest_actual = std::max(largest_actual, actual);
    for (const auto* tensor : {key, value})
      if (!Shape(tensor, GGML_TYPE_F16,
                 {std::int64_t(d), std::int64_t(actual), std::int64_t(kvh), 1},
                 {2, kvrow, d * 2, kvrow * actual}, operand_limit))
        return Rejected("owner MMA requires actual current cell-major F16 cache views");
    reads[2 + owner] = in.k[owner];
    reads[6 + owner] = in.v[owner];
  }
  if (in.bounded_roots && !bounded_whole12 && largest_actual != cells)
    return Rejected("bounded owner mask must match the largest actual width");
  for (std::size_t i = 0; i < reads.size(); ++i)
    if (reads[i] && Overlap(in.output, reads[i], kMaxSpan,
                            i == 0   ? kMaxSpan
                            : i == 1 ? mask_limit
                                     : operand_limit))
      return Rejected("owner MMA output overlaps a real operand");
  // Independent owner storage: aliases within an owner are read-only, but one
  // owner's K/V must not be another owner's payload.
  for (std::size_t a = 0; a < in.owner_count; ++a)
    for (std::size_t b = a + 1; b < in.owner_count; ++b)
      for (const auto* first : {in.k[a], in.v[a]})
        for (const auto* second : {in.k[b], in.v[b]})
          if (Overlap(first, second, operand_limit, operand_limit))
            return Rejected("owner MMA cache owners overlap");
  return {};
}
std::expected<FlashAttnOwners, KernelFailure> FlashAttnOwnersFromNode(ggml_tensor* node) {
  static_assert(GGML_MAX_SRC == 10);
  if (!node || LlmpOpOf(node) != LlmpOp::kFlashAttnOwners || node->view_src)
    return Rejected("not a ten-source owner attention custom node");
  const auto cohort = LlmpOpInt(node, 0);
  if (cohort != 2 && cohort != 3 && cohort != 4 && cohort != 8 && cohort != 12 &&
      !detail::PartialOwnerCohort(static_cast<std::uint32_t>(cohort)))
    return Rejected("owner attention has an unsupported logical cohort");
  const auto encoded_count = LlmpOpInt(node, 1);
  if (encoded_count != 0 && encoded_count != 1 && encoded_count != 2 && encoded_count != 3)
    return Rejected("owner attention has an unsupported active root count");
  const auto offset = LlmpOpInt(node, 2);
  if (offset < 0) return Rejected("owner attention has a negative group offset");
  for (int i = 5; i < 8; ++i)
    if (LlmpOpInt(node, i) != 0) return Rejected("owner attention has unsupported parameters");
  const auto bounded = LlmpOpInt(node, 4);
  if (bounded != 0 && bounded != 1)
    return Rejected("owner attention has an unsupported root bound");
  const auto cap = LlmpOpInt(node, 3);
  if (cap != 0 && cap != 50) return Rejected("owner attention has an unsupported softcap");
  FlashAttnOwners in{
      .q = node->src[0],
      .mask = node->src[1],
      .output = node,
      .logical_cohort = static_cast<std::uint32_t>(cohort),
      .owner_count = encoded_count == 0 ? 4U : static_cast<std::uint32_t>(encoded_count),
      .owner_offset = static_cast<std::uint32_t>(offset),
      .logit_softcap = static_cast<std::uint32_t>(cap),
      .bounded_roots = bounded != 0};
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
}  // namespace llmp::kernels::ggml
