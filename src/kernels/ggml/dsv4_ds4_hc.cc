// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_hc.h"

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
bool Rows(std::uint32_t rows) { return rows != 0 && rows <= kDs4HcMaxRows; }
bool Width(std::uint32_t width) { return width != 0 && width <= kDs4HcWidth; }
bool Epsilon(float epsilon) { return std::isfinite(epsilon) && epsilon > 0; }

struct Access {
  Ds4CacheBuffer buffer;
  std::uint64_t bytes{};
  std::uint64_t alignment = alignof(float);
  bool write = false;
};

std::uint64_t Optional(const Ds4CacheBuffer& buffer, std::uint64_t bytes) {
  return Absent(buffer) ? 0 : bytes;
}

// Only kernel-accessed prefixes participate in alias checks. The full supplied
// range must still have a representable end. Read-only views may overlap;
// writes cannot, except an explicitly allowed exact RMS source/F32-output pair.
std::expected<void, KernelFailure> Ranges(std::span<const Access> ranges,
                                          bool rms_in_place = false) {
  for (const auto& range : ranges) {
    if (range.bytes == 0) {
      if (!Absent(range.buffer)) return Reject("ds4 HC unused range must be absent");
      continue;
    }
    if (range.buffer.address == 0 || range.buffer.address % range.alignment != 0 ||
        range.buffer.bytes < range.bytes ||
        range.buffer.bytes > std::numeric_limits<std::uint64_t>::max() - range.buffer.address) {
      return Reject("ds4 HC operand is short, misaligned or has an unrepresentable end");
    }
  }
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    for (std::size_t j = i + 1; j < ranges.size(); ++j) {
      const auto& a = ranges[i];
      const auto& b = ranges[j];
      if (a.bytes == 0 || b.bytes == 0 || (!a.write && !b.write)) continue;
      if (rms_in_place && i == 0 && j == 2 && a.buffer.address == b.buffer.address &&
          a.bytes == b.bytes)
        continue;
      if (a.buffer.address < b.buffer.address + b.bytes &&
          b.buffer.address < a.buffer.address + a.bytes) {
        return Reject("ds4 HC output overlaps an operand");
      }
    }
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckDs4Rms(const Ds4Rms& desc) {
  if (!Rows(desc.rows) || desc.width == 0 || desc.width > 4 * kDs4HcWidth || !Epsilon(desc.epsilon))
    return Reject("ds4 RMS shape or epsilon is invalid");
  const bool weighted = !Absent(desc.weights);
  const bool f32 = !Absent(desc.values);
  const bool f16 = !Absent(desc.values_f16);
  const bool q8 = !Absent(desc.q8_d4);
  if ((!weighted && (f32 == f16 || q8)) || (weighted && !f32) ||
      (q8 && (!f16 || desc.rows < 64 || desc.width % 128 != 0))) {
    return Reject("ds4 RMS output combination or original Q8 eligibility is invalid");
  }
  const auto elements = static_cast<std::uint64_t>(desc.rows) * desc.width;
  const std::array ranges = {
      Access{desc.source, elements * 4},
      Access{desc.weights, weighted ? static_cast<std::uint64_t>(desc.width) * 4 : 0},
      Access{desc.values, Optional(desc.values, elements * 4), q8 ? 16U : 4U, true},
      Access{desc.values_f16, Optional(desc.values_f16, elements * 2), 2, true},
      Access{desc.q8_d4, q8 ? elements / 128 * 144 : 0, 16, true}};
  return Ranges(ranges, true);
}

std::expected<void, KernelFailure> CheckDs4HcSplit(const Ds4HcSplit& desc) {
  if (!Rows(desc.rows) || desc.iterations == 0 || desc.iterations > 20 || !Epsilon(desc.epsilon))
    return Reject("ds4 HC split shape/iterations/epsilon is invalid");
  const auto bytes = static_cast<std::uint64_t>(desc.rows) * 24 * 4;
  const std::array ranges = {Access{desc.mix, bytes}, Access{desc.scale, 12}, Access{desc.base, 96},
                             Access{desc.split, bytes, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4HcWeighted(const Ds4HcWeighted& desc) {
  if (!Rows(desc.rows) || !Width(desc.width) ||
      (desc.weight_stride != 4 && desc.weight_stride != 24)) {
    return Reject("ds4 HC weighted shape/stride is invalid");
  }
  const auto bytes = static_cast<std::uint64_t>(desc.rows) * desc.width * 4;
  const auto weights = (((static_cast<std::uint64_t>(desc.rows) - 1) * desc.weight_stride) + 4) * 4;
  const std::array ranges = {Access{desc.residual, bytes * 4}, Access{desc.weights, weights},
                             Access{desc.values, bytes, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4HcPre(const Ds4HcPre& desc) {
  if (auto checked = CheckDs4HcSplit(desc.coefficients); !checked) return checked;
  if (!Width(desc.width)) return Reject("ds4 HC pre width is invalid");
  const auto rows = desc.coefficients.rows;
  const auto bytes = static_cast<std::uint64_t>(rows) * desc.width * 4;
  const auto split = static_cast<std::uint64_t>(rows) * 24 * 4;
  const std::array ranges = {
      Access{desc.coefficients.mix, split}, Access{desc.coefficients.scale, 12},
      Access{desc.coefficients.base, 96},   Access{desc.coefficients.split, split, 4, true},
      Access{desc.residual, bytes * 4},     Access{desc.values, bytes, 4, true}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4HcExpand(const Ds4HcExpand& desc) {
  if (!Rows(desc.rows) || !Width(desc.width)) return Reject("ds4 HC expand shape is invalid");
  const bool f16 = !Absent(desc.values_f16);
  const bool moe = !Absent(desc.moe_unsummed);
  const bool add = !Absent(desc.add);
  if ((f16 && (desc.width != 4096 || desc.rows <= 8 || !Epsilon(desc.epsilon))) ||
      (moe && (!f16 || !add || !Absent(desc.block))) || (!moe && Absent(desc.block))) {
    return Reject("ds4 HC expand operands or original folded eligibility is invalid");
  }
  const auto block = static_cast<std::uint64_t>(desc.rows) * desc.width * 4;
  const std::array ranges = {
      Access{desc.block, moe ? 0 : block},
      Access{desc.add, Optional(desc.add, block)},
      Access{desc.residual, block * 4},
      Access{desc.split, static_cast<std::uint64_t>(desc.rows) * 24 * 4},
      Access{desc.values, block * 4, 4, true},
      Access{desc.values_f16, Optional(desc.values_f16, block * 4 / 2), 2, true},
      Access{desc.moe_unsummed, moe ? block * 6 : 0}};
  return Ranges(ranges);
}

std::expected<void, KernelFailure> CheckDs4HcHeadWeights(const Ds4HcHeadWeights& desc) {
  if (!Rows(desc.rows) || !Epsilon(desc.epsilon))
    return Reject("ds4 HC head shape/epsilon is invalid");
  const auto bytes = static_cast<std::uint64_t>(desc.rows) * 4 * 4;
  const std::array ranges = {Access{desc.pre, bytes}, Access{desc.scale, 4}, Access{desc.base, 16},
                             Access{desc.values, bytes, 4, true}};
  return Ranges(ranges);
}

}  // namespace jitllm::kernels::ggml
