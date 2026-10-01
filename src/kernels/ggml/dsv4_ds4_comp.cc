// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_comp.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Reject(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }
std::uint32_t Head(const Ds4CompState& state) {
  if (state.kind == Ds4CacheKind::kKv512 && (state.ratio == 4 || state.ratio == 128)) return 512;
  if (state.kind == Ds4CacheKind::kIndexer128 && state.ratio == 4) return 128;
  return 0;
}
std::uint32_t Width(const Ds4CompState& state) {
  return Head(state) * (state.ratio == 4 ? 2U : 1U);
}
std::uint64_t StateBytes(const Ds4CompState& state) {
  return static_cast<std::uint64_t>(state.ratio == 4 ? 8U : state.ratio) * Width(state) * 4;
}
bool Span(std::uint32_t first, std::uint32_t tokens) {
  return tokens != 0 && tokens <= kDs4CompMaxTokens &&
         first <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - tokens;
}
std::uint64_t ApeBytes(const Ds4CompState& state, Ds4CompApe format) {
  return static_cast<std::uint64_t>(state.ratio) * Width(state) *
         (format == Ds4CompApe::kF16 ? 2U : 4U);
}
bool Ape(Ds4CompApe format) { return format == Ds4CompApe::kF32 || format == Ds4CompApe::kF16; }
bool Positive(float value) { return std::isfinite(value) && value > 0; }
bool Rope(const Ds4CompRope& rope) {
  if (!Positive(rope.frequency_base) || !Positive(rope.frequency_scale) ||
      !Positive(rope.attention_factor) || !std::isfinite(rope.extension) || rope.extension < 0)
    return false;
  return rope.extension == 0 || (rope.frequency_base > 1 && rope.original_context != 0 &&
                                 Positive(rope.beta_fast) && Positive(rope.beta_slow));
}

struct Access {
  Ds4CacheBuffer buffer;
  std::uint64_t bytes{};
  std::uint64_t alignment = 4;
  bool write = false;
};

std::expected<void, KernelFailure> Ranges(std::span<const Access> ranges) {
  for (const auto& range : ranges) {
    if (range.bytes == 0) {
      if (!Absent(range.buffer)) return Reject("ds4 compressor unused range must be absent");
    } else if (range.buffer.address == 0 || range.buffer.address % range.alignment != 0 ||
               range.buffer.bytes < range.bytes ||
               range.buffer.bytes >
                   std::numeric_limits<std::uint64_t>::max() - range.buffer.address) {
      return Reject("ds4 compressor range is short, misaligned or has an unrepresentable end");
    }
  }
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    for (std::size_t j = i + 1; j < ranges.size(); ++j) {
      const auto& a = ranges[i];
      const auto& b = ranges[j];
      if (a.bytes == 0 || b.bytes == 0 || (!a.write && !b.write)) continue;
      if (a.buffer.address < b.buffer.address + b.bytes &&
          b.buffer.address < a.buffer.address + a.bytes)
        return Reject("ds4 compressor output/state aliases an operand");
    }
  }
  return {};
}

}  // namespace

std::expected<std::uint32_t, KernelFailure> Ds4CompCausalCount(std::uint32_t position,
                                                               std::uint32_t ratio) {
  if ((ratio != 4 && ratio != 128) ||
      position >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return Reject("ds4 compressor causal position or ratio is invalid");
  return (position + 1) / ratio;
}

std::expected<Ds4CompPlan, KernelFailure> PlanDs4Comp(const Ds4CompChunk& desc) {
  if (Head(desc.state) == 0 || !Span(desc.first, desc.tokens) ||
      desc.before != desc.first / desc.state.ratio)
    return Reject("ds4 compressor shape/position/count is invalid");
  const auto after = (desc.first + desc.tokens) / desc.state.ratio;
  if (after > desc.capacity) return Reject("ds4 compressor cache capacity is short");
  auto path = Ds4CompPath::kRows;
  if (desc.first == 0) {
    path = Ds4CompPath::kZeroPrefix;
  } else if (desc.first % desc.state.ratio == 0 && desc.tokens % desc.state.ratio == 0) {
    path = Ds4CompPath::kAligned;
  }
  return Ds4CompPlan{
      .path = path,
      .emitted = after - desc.before,
      .after = after,
      .refresh_required = desc.state.ratio == 4 && desc.tokens >= 4 && path != Ds4CompPath::kRows};
}

std::expected<void, KernelFailure> CheckDs4CompState(const Ds4CompState& desc) {
  if (Head(desc) == 0) return Reject("ds4 compressor state kind/ratio is invalid");
  const auto bytes = StateBytes(desc);
  const std::array ranges = {Access{desc.kv, bytes, 4, true}, Access{desc.score, bytes, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4CompPool(const Ds4CompPool& desc) {
  if (Head(desc.state) == 0 || !Ape(desc.ape_format) || !Span(desc.first, desc.tokens) ||
      desc.first % desc.state.ratio != 0 || desc.tokens % desc.state.ratio != 0)
    return Reject("ds4 compressor pool is not an original aligned shape");
  const auto source = static_cast<std::uint64_t>(desc.tokens) * Width(desc.state) * 4;
  const auto output =
      static_cast<std::uint64_t>(desc.tokens / desc.state.ratio) * Head(desc.state) * 4;
  const std::array ranges = {Access{desc.state.kv, StateBytes(desc.state)},
                             Access{desc.state.score, StateBytes(desc.state)},
                             Access{desc.kv, source},
                             Access{desc.score, source},
                             Access{desc.ape, ApeBytes(desc.state, desc.ape_format),
                                    desc.ape_format == Ds4CompApe::kF16 ? 2U : 4U},
                             Access{desc.values, output, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4CompChunk(const Ds4CompChunk& desc) {
  const auto plan = PlanDs4Comp(desc);
  if (!plan) return std::unexpected(plan.error());
  if (!Ape(desc.ape_format) || !Positive(desc.rms_epsilon) || !Rope(desc.rope) ||
      Absent(desc.codes) != Absent(desc.scales))
    return Reject("ds4 compressor transform/mirror contract is invalid");
  const bool packed = !Absent(desc.codes);
  const auto source = static_cast<std::uint64_t>(desc.tokens) * Width(desc.state) * 4;
  const auto emitted = static_cast<std::uint64_t>(plan->emitted);
  const auto code_row =
      desc.state.kind == Ds4CacheKind::kKv512 ? kDs4KvPackedRowBytes : kDs4IndexerPackedRowBytes;
  const auto scale_row =
      desc.state.kind == Ds4CacheKind::kKv512 ? kDs4KvScaleRowBytes : kDs4IndexerScaleRowBytes;
  const std::array ranges = {Access{desc.state.kv, StateBytes(desc.state), 4, true},
                             Access{desc.state.score, StateBytes(desc.state), 4, true},
                             Access{desc.kv, source},
                             Access{desc.score, source},
                             Access{desc.ape, ApeBytes(desc.state, desc.ape_format),
                                    desc.ape_format == Ds4CompApe::kF16 ? 2U : 4U},
                             Access{desc.norm, static_cast<std::uint64_t>(Head(desc.state)) * 4},
                             Access{desc.values, emitted * Head(desc.state) * 4, 4, true},
                             Access{desc.codes, packed ? emitted * code_row : 0,
                                    desc.state.kind == Ds4CacheKind::kKv512 ? 4U : 1U, true},
                             Access{desc.scales, packed ? emitted * scale_row : 0, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4CompRefresh(const Ds4CompRefresh& desc) {
  if (Head(desc.state) == 0 || desc.state.ratio != 4 || !Ape(desc.ape_format) ||
      !Span(desc.first, 4))
    return Reject("ds4 compressor refresh shape/position is invalid");
  const auto source = static_cast<std::uint64_t>(4) * Width(desc.state) * 4;
  const std::array ranges = {Access{desc.state.kv, StateBytes(desc.state), 4, true},
                             Access{desc.state.score, StateBytes(desc.state), 4, true},
                             Access{desc.kv, source}, Access{desc.score, source},
                             Access{desc.ape, ApeBytes(desc.state, desc.ape_format),
                                    desc.ape_format == Ds4CompApe::kF16 ? 2U : 4U}};
  return Ranges(ranges);
}

}  // namespace jitllm::kernels::ggml
