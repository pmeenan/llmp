// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_product.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

namespace jitllm::kernels::ggml {
namespace {
constexpr std::uint64_t kMax = std::numeric_limits<std::int32_t>::max();
constexpr std::uint64_t kTailBlocks = 256;

auto Refused(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
bool Range(Ds4ProductRead s, std::uint64_t need, std::uint64_t alignment) {
  std::uintptr_t end = 0;
  return s.data != nullptr && need != 0 && s.bytes >= need &&
         reinterpret_cast<std::uintptr_t>(s.data) % alignment == 0 &&
         !__builtin_add_overflow(reinterpret_cast<std::uintptr_t>(s.data), s.bytes, &end);
}
Ds4ProductRead Read(Ds4ProductWrite s) { return {.data = s.data, .bytes = s.bytes}; }
bool Separate(Ds4ProductRead a, Ds4ProductRead b) {
  if (a.data == nullptr || b.data == nullptr) return true;
  const auto x = reinterpret_cast<std::uintptr_t>(a.data);
  const auto y = reinterpret_cast<std::uintptr_t>(b.data);
  std::uintptr_t xe = 0;
  std::uintptr_t ye = 0;
  return !__builtin_add_overflow(x, a.bytes, &xe) && !__builtin_add_overflow(y, b.bytes, &ye) &&
         (xe <= y || ye <= x);
}
bool SeparateFrom(Ds4ProductWrite out, std::initializer_list<Ds4ProductRead> inputs) {
  return std::ranges::all_of(inputs,
                             [out](const auto input) { return Separate(Read(out), input); });
}
bool Matrix(Ds4ProductRead s, std::uint32_t rows, std::uint32_t columns, std::uint64_t stride,
            std::uint64_t element, std::uint64_t alignment, bool packed,
            std::uint64_t max_rows = 65535) {
  std::uint64_t width = 0;
  std::uint64_t last = 0;
  std::uint64_t need = 0;
  return rows != 0 && rows <= max_rows && columns != 0 && columns <= kMax &&
         !__builtin_mul_overflow(static_cast<std::uint64_t>(columns), element, &width) &&
         stride >= width && stride <= kMax && stride % alignment == 0 &&
         (!packed || stride == width) &&
         !__builtin_mul_overflow(static_cast<std::uint64_t>(rows - 1), stride, &last) &&
         !__builtin_add_overflow(last, width, &need) && Range(s, need, alignment);
}
bool Input(const Ds4ProductMatrix& d, std::uint64_t element, std::uint64_t alignment, bool packed) {
  return Matrix(d.storage, d.rows, d.columns, d.row_stride, element, alignment, packed);
}
bool Output(const Ds4ProductOutput& d, std::uint64_t element, std::uint64_t alignment,
            bool packed) {
  return Matrix(Read(d.storage), d.rows, d.columns, d.row_stride, element, alignment, packed);
}
bool Empty(Ds4ProductRead s) { return s.data == nullptr && s.bytes == 0; }
bool Empty(const Ds4D4Sidecar& d) {
  return d.storage.data == nullptr && d.storage.bytes == 0 && d.source == nullptr &&
         d.generation == 0 && d.rows == 0 && d.columns == 0;
}
bool AlignedQ8(const Ds4Q8Weights& w) {
  return w.columns != 0 && w.columns % 32 == 0 && w.rows != 0 && w.rows <= kMax &&
         w.scale_row_stride == static_cast<std::uint64_t>(w.columns / 32) * 2 &&
         w.code_row_stride == w.columns &&
         Range(w.scales, static_cast<std::uint64_t>(w.rows) * w.scale_row_stride, 16) &&
         Range(w.codes, static_cast<std::uint64_t>(w.rows) * w.code_row_stride, 16) &&
         Separate(w.scales, w.codes);
}
bool Rope(const Ds4ProductRope& r, std::uint32_t rows, std::uint32_t head_width) {
  if (r.rotary == 0 || r.rotary % 2 != 0 || r.rotary > head_width || !std::isfinite(r.base) ||
      r.base <= 0 || !std::isfinite(r.scale) || r.scale <= 0 || !std::isfinite(r.extension) ||
      r.extension < 0 || !std::isfinite(r.attention) || r.attention <= 0 ||
      !std::isfinite(r.beta_fast) || r.beta_fast <= 0 || !std::isfinite(r.beta_slow) ||
      r.beta_slow <= 0)
    return false;
  if (r.extension != 0 && (r.original_context == 0 || r.base == 1.0F)) return false;
  if (r.positions.data != nullptr)
    return Range(r.positions, static_cast<std::uint64_t>(rows) * 4, 4);
  return Empty(r.positions) &&
         static_cast<std::uint64_t>(r.first) + (static_cast<std::uint64_t>(rows - 1) * r.step) <=
             kMax;
}
}  // namespace

std::expected<void, KernelFailure> CheckDs4Embedding(const Ds4Embedding& d) {
  const auto width = static_cast<std::uint64_t>(d.weights.columns) * d.hyper_connections;
  if (!Matrix(d.weights.storage, d.weights.rows, d.weights.columns, d.weights.row_stride, 2, 2,
              true, kMax) ||
      !Output(d.output, 4, 4, true) || d.hyper_connections != 4 || width > kMax ||
      d.output.columns != width ||
      ((static_cast<std::uint64_t>(d.output.rows) * width) + 255) / 256 > kMax ||
      !Range(d.tokens, static_cast<std::uint64_t>(d.output.rows) * 4, 4) ||
      !SeparateFrom(d.output.storage, {d.weights.storage, d.tokens}))
    return Refused("ds4 embedding requires bounded packed F16 rows and separate HC4 output");
  return {};
}

std::expected<void, KernelFailure> CheckDs4F16Conversion(const Ds4F16Conversion& d) {
  if (!Input(d.input, 4, 4, true) || !Output(d.output, 2, 2, true) ||
      d.output.rows != d.input.rows || d.output.columns != d.input.columns ||
      ((static_cast<std::uint64_t>(d.input.rows) * d.input.columns) + 255) / 256 > kMax ||
      !Separate(Read(d.output.storage), d.input.storage))
    return Refused("ds4 F16 conversion requires bounded disjoint packed matrices");
  return {};
}

std::expected<void, KernelFailure> CheckDs4QkvNorm(const Ds4QkvNorm& d) {
  const auto valid = [](const Ds4ProductMatrix& input, const Ds4ProductOutput& output,
                        Ds4ProductRead weight) {
    const auto elements = static_cast<std::uint64_t>(input.rows) * input.columns;
    const bool exact =
        output.storage.data == input.storage.data && output.storage.bytes == input.storage.bytes;
    return Input(input, 4, 4, true) && Output(output, 4, 4, true) && input.rows == output.rows &&
           input.columns == output.columns &&
           Range(weight, static_cast<std::uint64_t>(input.columns) * 4, 4) &&
           (exact || Separate(Read(output.storage), input.storage)) &&
           Separate(Read(output.storage), weight) && elements <= kMax;
  };
  if (!valid(d.query, d.query_output, d.query_weight) || !valid(d.kv, d.kv_output, d.kv_weight) ||
      d.query.rows <= 8 || d.query.rows != d.kv.rows || !std::isfinite(d.epsilon) ||
      d.epsilon <= 0 ||
      !SeparateFrom(d.query_output.storage,
                    {d.kv.storage, Read(d.kv_output.storage), d.kv_weight}) ||
      !SeparateFrom(d.kv_output.storage, {d.query.storage, d.query_weight}))
    return Refused("ds4 fused QKV RMS requires independent bounded wide rows and exact aliases");
  return {};
}

std::expected<void, KernelFailure> CheckDs4F16Product(const Ds4F16Product& d) {
  if (!Matrix(d.weights.storage, d.weights.rows, d.weights.columns, d.weights.row_stride, 2, 2,
              false, kMax) ||
      !Input(d.input, 2, 2, false) || !Output(d.output, 4, 4, false) || d.input.rows <= 8 ||
      d.weights.columns != d.input.columns || d.output.rows != d.input.rows ||
      d.output.columns != d.weights.rows ||
      !SeparateFrom(d.output.storage, {d.weights.storage, d.input.storage}))
    return Refused("ds4 wide F16 requires bounded disjoint F16 matrices and T>8");
  return {};
}

std::expected<void, KernelFailure> CheckDs4F16Vector(const Ds4F16Vector& d) {
  if (!Matrix(d.weights.storage, d.weights.rows, d.weights.columns, d.weights.row_stride, 2, 2,
              true, kMax) ||
      !Input(d.input, 4, 4, true) || !Output(d.output, 4, 4, true) || d.input.rows > 8 ||
      d.weights.columns != d.input.columns || d.output.rows != d.input.rows ||
      d.output.columns != d.weights.rows ||
      !SeparateFrom(d.output.storage, {d.weights.storage, d.input.storage}))
    return Refused("ds4 small F16 requires bounded disjoint packed F16/F32 matrices and T1..8");
  return {};
}

std::expected<std::uint64_t, KernelFailure> PlanDs4F16Vector(const Ds4F16Vector& d) {
  if (auto c = CheckDs4F16Vector(d); !c) return std::unexpected(c.error());
  const auto m = static_cast<std::uint64_t>(d.weights.rows);
  const auto k = static_cast<std::uint64_t>(d.input.columns);
  if (m >= 2048 || k < 1024) return 0;
  const auto split = std::min((2048 + m - 1) / m, k / 512);
  return split > 1 ? m * split * 4 : 0;
}

std::expected<std::uint64_t, KernelFailure> Ds4D4Bytes(std::uint32_t rows, std::uint32_t columns) {
  if (rows == 0 || rows > 65535 || columns == 0 || columns % 512 != 0 || columns / 512 > 65535 ||
      columns > kMax)
    return Refused("ds4 D4 requires positive bounded rows and whole 512-element input");
  const auto blocks = static_cast<std::uint64_t>(rows) * (columns / 128);
  // Original quantizer and MMQ stride fields index the payload in signed ints.
  if ((blocks + kTailBlocks) * 144 > kMax)
    return Refused("ds4 D4 payload exceeds the original signed indexing bound");
  return (blocks + kTailBlocks) * 144;
}

std::expected<void, KernelFailure> CheckDs4D4(const Ds4ProductMatrix& input,
                                              std::uint64_t generation, const Ds4D4Sidecar& d) {
  const auto bytes = Ds4D4Bytes(input.rows, input.columns);
  if (!bytes) return std::unexpected(bytes.error());
  if (!Input(input, 4, 16, true) || generation == 0 || d.generation != generation ||
      d.source != input.storage.data || d.rows != input.rows || d.columns != input.columns ||
      !Range(Read(d.storage), *bytes, 16) || !Separate(Read(d.storage), input.storage))
    return Refused("ds4 D4 requires current source/shape/generation and disjoint padded storage");
  return {};
}

std::expected<void, KernelFailure> CheckDs4Q8Product(const Ds4Q8Product& d) {
  if (auto q = CheckDs4D4(d.input, d.generation, d.quantized); !q) return q;
  const auto& w = d.weights;
  if (!Output(d.output, 4, 16, true) || d.input.rows <= 8 || w.columns != d.input.columns ||
      w.rows != d.output.columns || d.output.rows != d.input.rows ||
      !SeparateFrom(d.output.storage,
                    {d.input.storage, Read(d.quantized.storage), w.raw, w.scales, w.codes}) ||
      !SeparateFrom(d.quantized.storage, {w.raw, w.scales, w.codes}))
    return Refused("ds4 Q8 requires bounded disjoint packed wide products");
  if (d.path == Ds4Q8Path::kMmq) {
    const auto stride = static_cast<std::uint64_t>(w.columns / 32) * 34;
    if (static_cast<std::uint64_t>(w.rows) * d.input.rows > kMax ||
        static_cast<std::uint64_t>(w.rows) * (w.columns / 32) > kMax ||
        w.raw_row_stride != stride ||
        !Range(w.raw, static_cast<std::uint64_t>(w.rows) * stride, 16))
      return Refused("ds4 MMQ requires original packed Q8_0 rows and signed offsets");
  } else if (d.path == Ds4Q8Path::kDenseD2r) {
    if (d.input.rows < 512 || w.rows < 2048 || w.rows % 128 != 0 || w.columns > 4096 ||
        w.columns % 1024 != 0 || !AlignedQ8(w))
      return Refused("ds4 dense D2R requires its original wide kind-5 shape and planes");
  } else {
    return Refused("unknown ds4 Q8 product path");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4HeadRope(const Ds4HeadRope& d) {
  const auto heads = static_cast<std::uint64_t>(d.heads);
  const auto columns = heads * d.head_width;
  if (!Output(d.input, 4, 4, true) || d.heads == 0 || d.head_width == 0 || columns > kMax ||
      d.input.columns != columns || heads * d.input.rows > kMax ||
      !Rope(d.rope, d.input.rows, d.head_width) ||
      heads * d.input.rows * (d.rope.rotary / 2) > kMax ||
      !Separate(Read(d.input.storage), d.rope.positions) ||
      (d.normalize && (!std::isfinite(d.epsilon) || d.epsilon <= 0)))
    return Refused("ds4 head RoPE requires bounded packed heads and finite original parameters");
  return {};
}

std::expected<std::uint64_t, KernelFailure> Ds4Q81Bytes(std::uint32_t rows, std::uint32_t columns) {
  if (rows == 0 || rows > 8 || columns == 0 || columns > kMax || columns % 256 != 0)
    return Refused("ds4 Q8_1 requires T1..8 and whole 256-element input");
  const auto padded = ((static_cast<std::uint64_t>(columns) + 511) / 512) * 512;
  const auto bytes = static_cast<std::uint64_t>(rows) * (padded / 32) * 36;
  if (padded > kMax || bytes > kMax)
    return Refused("ds4 Q8_1 exceeds the original signed indexing bound");
  return bytes;
}

std::expected<void, KernelFailure> CheckDs4Q81(const Ds4ProductMatrix& input,
                                               std::uint64_t generation, const Ds4Q81Sidecar& d) {
  const auto bytes = Ds4Q81Bytes(input.rows, input.columns);
  if (!bytes) return std::unexpected(bytes.error());
  if (!Input(input, 4, 4, true) || generation == 0 || d.generation != generation ||
      d.source != input.storage.data || d.rows != input.rows || d.columns != input.columns ||
      !Range(Read(d.storage), *bytes, 16) || !Separate(Read(d.storage), input.storage))
    return Refused("ds4 Q8_1 needs current source/shape/generation and disjoint padded bytes");
  return {};
}

std::expected<void, KernelFailure> CheckDs4Q8Vector(const Ds4Q8Vector& d) {
  if (auto q = CheckDs4Q81(d.input, d.generation, d.quantized); !q) return q;
  const auto& w = d.weights;
  if (!Output(d.output, 4, 4, true) || w.rows == 0 || w.rows > kMax ||
      w.columns != d.input.columns || w.rows != d.output.columns || d.output.rows != d.input.rows ||
      static_cast<std::uint64_t>(w.rows) * d.input.rows > kMax ||
      !SeparateFrom(d.output.storage,
                    {d.input.storage, Read(d.quantized.storage), w.raw, w.scales, w.codes}) ||
      !SeparateFrom(d.quantized.storage, {w.raw, w.scales, w.codes}))
    return Refused("ds4 small Q8 needs bounded disjoint packed full-output matrices");
  if (d.path == Ds4Q8VectorPath::kAligned) {
    if (w.columns % 1024 != 0 || !AlignedQ8(w))
      return Refused("ds4 aligned Q8 vector needs original kind-5 planes and K%1024==0");
  } else if (d.path == Ds4Q8VectorPath::kRaw) {
    const auto single_rpb = w.columns < 1024 ? 4ULL : 1ULL;
    const auto rpb = d.input.rows == 1 ? single_rpb : 2ULL;
    const auto physical = ((static_cast<std::uint64_t>(w.rows) + rpb - 1) / rpb) * rpb;
    const auto blocks = static_cast<std::uint64_t>(w.columns / 32);
    const auto stride = blocks * 34;
    if (physical * blocks > kMax || w.raw_row_stride != stride ||
        !Range(w.raw, physical * stride, 16))
      return Refused("ds4 raw Q8 vector needs bounded whole readable original row tiles");
  } else {
    return Refused("unknown ds4 small Q8 vector path");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4OutA(const Ds4OutA& d) {
  const auto rows = static_cast<std::uint64_t>(d.heads.rows);
  const auto table_rows = ((rows + 127) / 128) * 128;
  const auto store_rows = ((rows + 15) / 16) * 16;
  if (!Input(d.heads, 4, 16, true) || d.heads.columns != 32768 || d.heads.rows <= 8 ||
      !Output(d.low, 4, 32, true) || d.low.rows != rows || d.low.columns != 8192 ||
      !Range(Read(d.low.storage), store_rows * 8192 * 4, 32) || d.weights.rows != 8192 ||
      d.weights.columns != 4096 || !AlignedQ8(d.weights) || !Rope(d.rope, d.heads.rows, 512) ||
      !d.rope.inverse || d.rope.rotary != 64 || d.rope.step != 1 ||
      !Range(Read(d.rope_table), table_rows * 32 * 8, 16) ||
      !SeparateFrom(d.low.storage, {d.heads.storage, d.weights.scales, d.weights.codes,
                                    Read(d.rope_table), d.rope.positions}) ||
      !SeparateFrom(d.rope_table,
                    {d.heads.storage, d.weights.scales, d.weights.codes, d.rope.positions}))
    return Refused("ds4 own out-a requires exact G8 geometry and charged physical tile tails");
  if (!Empty(d.quantized)) {
    const Ds4ProductMatrix low{.storage = Read(d.low.storage),
                               .rows = d.low.rows,
                               .columns = d.low.columns,
                               .row_stride = d.low.row_stride};
    if (auto q = CheckDs4D4(low, d.generation, d.quantized); !q) return q;
    if (!SeparateFrom(d.quantized.storage, {d.heads.storage, d.weights.scales, d.weights.codes,
                                            Read(d.rope_table), d.rope.positions}))
      return Refused("ds4 own out-a D4 output overlaps an input or its rope table");
  }
  return {};
}
}  // namespace jitllm::kernels::ggml
