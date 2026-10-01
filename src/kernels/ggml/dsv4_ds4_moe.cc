// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_moe.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>

#include "kernels/ggml/dsv4_ds4_repack.h"

namespace jitllm::kernels::ggml {
namespace {
std::unexpected<KernelFailure> Reject(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
bool Absent(Ds4CacheBuffer b) { return b.address == 0 && b.bytes == 0; }
bool Rows(std::uint32_t n) { return n > 0 && n <= kDs4MoeMaxTokens; }
struct Access {
  Ds4CacheBuffer buffer;
  std::uint64_t bytes = 0, alignment = 4;
  bool write = false;
};
std::expected<void, KernelFailure> Ranges(std::span<const Access> ranges) {
  for (const auto& a : ranges) {
    if (a.bytes == 0) {
      if (!Absent(a.buffer)) return Reject("ds4 MoE unused range must be absent");
    } else if (a.buffer.address == 0 || a.buffer.address % a.alignment != 0 ||
               a.buffer.bytes < a.bytes ||
               a.buffer.bytes > std::numeric_limits<std::uint64_t>::max() - a.buffer.address) {
      return Reject("ds4 MoE range is short, misaligned or wraps");
    }
  }
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    for (std::size_t j = i + 1; j < ranges.size(); ++j) {
      const auto& a = ranges[i];
      const auto& b = ranges[j];
      if (a.bytes == 0 || b.bytes == 0 || (!a.write && !b.write)) continue;
      // Complete declared ranges protect producer slack and adjacent views.
      if (a.buffer.address < b.buffer.address + b.buffer.bytes &&
          b.buffer.address < a.buffer.address + a.buffer.bytes)
        return Reject("ds4 MoE writable range aliases an operand");
    }
  }
  return {};
}
std::uint64_t Strided(std::uint32_t rows, std::uint32_t stride) {
  return (((static_cast<std::uint64_t>(rows) - 1) * stride) + 6) * 4;
}
}  // namespace

std::expected<void, KernelFailure> CheckDs4MoeIds(std::span<const std::int32_t> ids,
                                                  std::uint32_t rows, std::uint32_t stride) {
  if (!Rows(rows) || stride < 6 || stride > 256 ||
      ids.size() < ((static_cast<std::uint64_t>(rows) - 1) * stride) + 6)
    return Reject("ds4 authoritative ID rows/stride are invalid");
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t s = 0; s < 6; ++s) {
      const auto id = ids[(static_cast<std::size_t>(r) * stride) + s];
      if (id < 0 || id >= 256) return Reject("ds4 selected expert is outside [0,256)");
      for (std::uint32_t before = 0; before < s; ++before)
        if (id == ids[(static_cast<std::size_t>(r) * stride) + before])
          return Reject("ds4 selected experts repeat within a token");
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4Router(const Ds4Router& d) {
  if (!Rows(d.rows) ||
      (d.select != Ds4RouterSelect::kWarp && d.select != Ds4RouterSelect::kParallel &&
       d.select != Ds4RouterSelect::kScalar) ||
      (Absent(d.hash) != (d.hash_rows == 0)) || (d.hash_rows != 0 && !Absent(d.bias)) ||
      (d.hash_rows == 0 && !Absent(d.tokens)))
    return Reject("ds4 router mode/rows/hash is invalid");
  const auto rows = static_cast<std::uint64_t>(d.rows);
  const std::array ranges = {Access{d.logits, rows * 256 * 4},
                             Access{d.bias, Absent(d.bias) ? 0U : 256U * 4U},
                             Access{d.hash, static_cast<std::uint64_t>(d.hash_rows) * 6 * 4},
                             Access{d.tokens, d.hash_rows != 0 ? rows * 4 : 0},
                             Access{d.selected, rows * 6 * 4, 4, true},
                             Access{d.weights, rows * 6 * 4, 4, true},
                             Access{d.probabilities, rows * 256 * 4, 4, true}};
  return Ranges(ranges);
}
std::expected<void, KernelFailure> CheckDs4RouterCooperative(const Ds4RouterCooperative& d) {
  if (auto c = CheckDs4Router(d.router); !c) return c;
  if (d.router.rows > 8 || d.router.select != Ds4RouterSelect::kWarp)
    return Reject("ds4 cooperative router is an original 1..8 row warp tier");
  const auto rows = static_cast<std::uint64_t>(d.router.rows);
  const auto& r = d.router;
  const std::array ranges = {Access{d.input, rows * 4096 * 4, 16},
                             Access{d.projection, 256ULL * 4096 * 2, 16},
                             Access{d.partials, rows * 256 * 8 * 4, 4, true},
                             Access{r.logits, rows * 256 * 4, 4, true},
                             Access{r.bias, Absent(r.bias) ? 0U : 256U * 4U},
                             Access{r.hash, static_cast<std::uint64_t>(r.hash_rows) * 6 * 4},
                             Access{r.tokens, r.hash_rows != 0 ? rows * 4 : 0},
                             Access{r.selected, rows * 6 * 4, 4, true},
                             Access{r.weights, rows * 6 * 4, 4, true},
                             Access{r.probabilities, rows * 256 * 4, 4, true}};
  return Ranges(ranges);
}

std::expected<Ds4MoeLayout, KernelFailure> Ds4MoeLayoutOf(const Ds4MoeShape& s, Ds4MoeTier tier) {
  if (!Rows(s.rows) || s.input == 0 || s.input > 4096 || s.input % 1024 != 0 || s.middle == 0 ||
      s.middle > 2048 || s.middle % 256 != 0 || s.output == 0 || s.output > 4096 ||
      s.output % 2 != 0 ||
      (tier != Ds4MoeTier::kVector && tier != Ds4MoeTier::kDirect &&
       tier != Ds4MoeTier::kMaterialized && tier != Ds4MoeTier::kClassic) ||
      (tier == Ds4MoeTier::kVector && s.rows > 16) ||
      ((tier == Ds4MoeTier::kDirect || tier == Ds4MoeTier::kMaterialized) &&
       static_cast<std::uint64_t>(s.rows) * 6 < 1024))
    return Reject("ds4 MoE shape/tier is outside the original aligned contract");
  // The literal down D2R kernel has eight 16-row warps per 128-row CTA.
  // Its full/guarded mainloops contain CTA barriers, so every warp must
  // choose the same branch even when an expert has a full 64-column tile.
  if ((tier == Ds4MoeTier::kDirect || tier == Ds4MoeTier::kMaterialized) && s.output % 128 >= 16)
    return Reject("ds4 MoE down D2R output tail would diverge at a CTA barrier");
  const auto g = Ds4AlignedLayoutOf(
      {.kind = Ds4AlignedKind::kIq2Xxs, .input = s.input, .output = s.middle, .groups = 256});
  const auto down = Ds4AlignedLayoutOf(
      {.kind = Ds4AlignedKind::kQ2K, .input = s.middle, .output = s.output, .groups = 256});
  if (!g) return std::unexpected(g.error());
  if (!down) return std::unexpected(down.error());
  const auto rows = static_cast<std::uint64_t>(s.rows);
  const auto pairs = rows * 6;
  const bool vector = tier == Ds4MoeTier::kVector;
  const auto input_payload = (tier == Ds4MoeTier::kDirect ? rows : pairs) * (s.input / 128) * 144;
  const auto down_payload = pairs * (s.middle / 128) * 144;
  return Ds4MoeLayout{
      .gate_weight_bytes = g->packed_bytes.value(),
      .down_weight_bytes = down->packed_bytes.value(),
      .input_quant_bytes =
          vector ? rows * (s.input / 32) * 36 : (pairs * (s.input / 128) * 144) + (128ULL * 144ULL),
      .down_quant_bytes = vector ? pairs * (s.middle / 32) * 36 : down_payload + (128ULL * 144ULL),
      .work_bytes =
          (vector || tier == Ds4MoeTier::kClassic) ? 0 : (((pairs + 31) / 32) + 256 + 1) * 4,
      .middle_bytes = pairs * s.middle * 4,
      .down_bytes = pairs * s.output * 4,
      .input_payload_bytes = vector ? rows * (s.input / 32) * 36 : input_payload,
      .down_payload_bytes = vector ? pairs * (s.middle / 32) * 36 : down_payload};
}
std::expected<void, KernelFailure> CheckDs4Moe(const Ds4Moe& d) {
  const auto layout = Ds4MoeLayoutOf(d.shape, d.tier);
  if (!layout) return std::unexpected(layout.error());
  if (d.selected_stride < 6 || d.selected_stride > 256 || d.weight_stride < 6 ||
      d.weight_stride > 256)
    return Reject("ds4 MoE selected/weight stride is invalid");
  const auto& l = *layout;
  const auto rows = static_cast<std::uint64_t>(d.shape.rows);
  const auto pairs = rows * 6;
  const bool vector = d.tier == Ds4MoeTier::kVector;
  const bool materialized = d.tier == Ds4MoeTier::kMaterialized || d.tier == Ds4MoeTier::kClassic;
  const bool produced = !Absent(d.producer.storage);
  if (produced) {
    const auto& p = d.producer;
    if ((!vector && d.tier != Ds4MoeTier::kDirect) || d.generation == 0 ||
        p.source_address != d.input.address || p.generation != d.generation ||
        p.rows != d.shape.rows || p.width != d.shape.input ||
        p.kind != (vector ? Ds4MoeQuant::kQ81 : Ds4MoeQuant::kD4))
      return Reject("ds4 MoE producer source/generation/shape/layout differs");
  } else if (d.producer.source_address != 0 || d.producer.generation != 0 || d.producer.rows != 0 ||
             d.producer.width != 0) {
    return Reject("ds4 MoE absent producer has nonzero identity");
  }
  const std::array ranges = {
      Access{d.input, rows * d.shape.input * 4, 16},
      Access{d.gate_weights, l.gate_weight_bytes, 64},
      Access{d.up_weights, l.gate_weight_bytes, 64},
      Access{d.down_weights, l.down_weight_bytes, 64},
      Access{d.selected, Strided(d.shape.rows, d.selected_stride)},
      Access{d.weights, Strided(d.shape.rows, d.weight_stride)},
      Access{d.compact_ids, d.selected_stride != 6 ? pairs * 4 : 0, 4, true},
      Access{d.compact_weights, d.weight_stride != 6 ? pairs * 4 : 0, 4, true},
      Access{d.ids_source, vector ? 0 : pairs * 4, 4, true},
      Access{d.ids_destination, vector ? 0 : pairs * 4, 4, true},
      Access{d.expert_bounds, vector ? 0 : 257U * 4U, 4, true},
      Access{d.work, l.work_bytes, 4, true},
      Access{d.input_quant, l.input_quant_bytes, 16, !produced},
      Access{d.producer.storage, produced ? l.input_payload_bytes + (vector ? 0 : 128 * 144) : 0,
             16},
      Access{d.down_quant, l.down_quant_bytes, 16, true},
      Access{d.gate, materialized ? l.middle_bytes : 0, 4, true},
      Access{d.up, materialized ? l.middle_bytes : 0, 4, true},
      Access{d.middle, (materialized || vector) ? l.middle_bytes : 0, 4, true},
      Access{d.down, l.down_bytes, 4, true},
      Access{d.sum, Absent(d.sum) ? 0 : rows * d.shape.output * 4, 4, true}};
  if (auto checked = Ranges(ranges); !checked) return checked;
  if (produced && d.input.address < d.producer.storage.address + d.producer.storage.bytes &&
      d.producer.storage.address < d.input.address + d.input.bytes)
    return Reject("ds4 MoE input producer aliases its differently formatted F32 source");
  return {};
}
std::expected<void, KernelFailure> CheckDs4SharedSwiglu(const Ds4SharedSwiglu& d) {
  if (!Rows(d.rows) || d.width == 0 || d.width > 2048)
    return Reject("ds4 shared SwiGLU shape is invalid");
  const auto bytes = static_cast<std::uint64_t>(d.rows) * d.width * 4;
  const std::array ranges = {Access{d.gate, bytes}, Access{d.up, bytes},
                             Access{d.output, bytes, 4, true}};
  return Ranges(ranges);
}
std::expected<void, KernelFailure> CheckDs4MoeSum(const Ds4MoeSum& d) {
  if (!Rows(d.rows) || d.width == 0 || d.width > 4096)
    return Reject("ds4 MoE sum shape is invalid");
  const auto bytes = static_cast<std::uint64_t>(d.rows) * d.width * 4;
  const std::array ranges = {Access{d.slots, bytes * 6}, Access{d.output, bytes, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4MoePostPair(const Ds4MoePostPair& d) {
  constexpr Ds4MoeShape shape{.rows = 4096, .input = 4096, .middle = 2048, .output = 4096};
  const auto layout = Ds4MoeLayoutOf(shape, Ds4MoeTier::kMaterialized);
  if (!layout) return std::unexpected(layout.error());
  const auto& l = *layout;
  const std::array ranges = {Access{d.gate, l.middle_bytes},
                             Access{d.up, l.middle_bytes},
                             Access{d.weights, 4096ULL * 6 * 4},
                             Access{d.ids_destination, 4096ULL * 6 * 4},
                             Access{d.expert_bounds, 257ULL * 4},
                             Access{d.down_weights, l.down_weight_bytes, 64},
                             Access{d.middle, l.middle_bytes, 4, true},
                             Access{d.down_quant, l.down_quant_bytes, 16, true},
                             Access{d.work, l.work_bytes, 4, true},
                             Access{d.down, l.down_bytes, 4, true},
                             Access{d.sum, Absent(d.sum) ? 0 : 4096ULL * 4096 * 4, 4, true}};
  if (auto checked = Ranges(ranges); !checked) return checked;
  for (const auto protected_buffer : d.retained) {
    const std::array protected_range = {Access{protected_buffer, protected_buffer.bytes, 1}};
    if (auto checked = Ranges(protected_range); !checked) return checked;
    for (const auto& writable : ranges) {
      if (!writable.write || writable.bytes == 0 || protected_buffer.bytes == 0) continue;
      if (writable.buffer.address < protected_buffer.address + protected_buffer.bytes &&
          protected_buffer.address < writable.buffer.address + writable.buffer.bytes)
        return Reject("ds4 post-pair writable range aliases a retained original operand");
    }
  }
  return {};
}
std::expected<void, KernelFailure> CheckDs4MoeMaps(const Ds4MoeMaps& d) {
  const std::array ranges = {Access{d.selected, 4096ULL * 6 * 4},
                             Access{d.ids_source, 4096ULL * 6 * 4, 4, true},
                             Access{d.ids_destination, 4096ULL * 6 * 4, 4, true},
                             Access{d.expert_bounds, 257ULL * 4, 4, true}};
  if (auto checked = Ranges(ranges); !checked) return checked;
  for (const auto protected_buffer : d.retained) {
    const std::array protected_range = {Access{protected_buffer, protected_buffer.bytes, 1}};
    if (auto checked = Ranges(protected_range); !checked) return checked;
    for (const auto& writable : ranges) {
      if (!writable.write || protected_buffer.bytes == 0) continue;
      if (writable.buffer.address < protected_buffer.address + protected_buffer.bytes &&
          protected_buffer.address < writable.buffer.address + writable.buffer.bytes)
        return Reject("ds4 map adapter writable range aliases a retained original operand");
    }
  }
  return {};
}
}  // namespace jitllm::kernels::ggml
