// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_cache.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

struct Geometry {
  std::uint32_t width;
  std::uint32_t code_bytes;
  std::uint32_t scale_bytes;
};

std::optional<Geometry> Shape(Ds4CacheKind kind) {
  switch (kind) {
    case Ds4CacheKind::kKv512:
      return Geometry{kDs4KvWidth, kDs4KvPackedRowBytes, kDs4KvScaleRowBytes};
    case Ds4CacheKind::kIndexer128:
      return Geometry{kDs4IndexerWidth, kDs4IndexerPackedRowBytes, kDs4IndexerScaleRowBytes};
  }
  return std::nullopt;
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }

// Kernels access only the logical prefix; the provided view can be larger,
// but its entire range must have a representable end. Device membership and
// backing are the checked bound plan's responsibility, like other launchers.
bool Holds(const Ds4CacheBuffer& buffer, std::uint64_t bytes, std::uint64_t alignment) {
  return buffer.address != 0 && buffer.address % alignment == 0 && buffer.bytes >= bytes &&
         buffer.bytes <= std::numeric_limits<std::uint64_t>::max() - buffer.address;
}

bool Overlap(const Ds4CacheBuffer& a, std::uint64_t a_bytes, const Ds4CacheBuffer& b,
             std::uint64_t b_bytes) {
  return a.address < b.address + b_bytes && b.address < a.address + a_bytes;
}

bool Rows(std::uint32_t rows) {
  // One block per row; this is CUDA's x-grid bound for the target device.
  return rows != 0 && rows <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
}

}  // namespace

std::array<float, 128> Ds4CacheDecodeTable() {
  std::array<float, 128> table{};
  // Same exact power-of-two construction as original ds4_cuda.cu:8710.
  // Caller uploads once before measured work; there is no device global.
  for (std::uint32_t i = 0; i < table.size(); ++i) {
    const auto exponent = static_cast<int>((i >> 3) & 15);
    const auto mantissa = static_cast<float>(i & 7);
    table[i] = exponent == 0 ? mantissa * 0.001953125f
                             : (1.0f + (mantissa * 0.125f)) * std::ldexp(1.0f, exponent - 7);
  }
  return table;
}

std::expected<void, KernelFailure> CheckDs4CacheQat(const Ds4CacheQat& desc) {
  const auto shape = Shape(desc.kind);
  if (!shape || !Rows(desc.rows)) return Rejected("ds4 QAT requires a supported kind and rows");
  const auto values = static_cast<std::uint64_t>(desc.rows) * shape->width * sizeof(float);
  const auto codes = static_cast<std::uint64_t>(desc.rows) * shape->code_bytes;
  const auto scales = static_cast<std::uint64_t>(desc.rows) * shape->scale_bytes;
  if (!Holds(desc.values, values, alignof(float))) return Rejected("ds4 QAT F32 range is invalid");
  const bool have_codes = !Absent(desc.codes);
  const bool have_scales = !Absent(desc.scales);
  if ((have_codes && !have_scales) ||
      (desc.kind == Ds4CacheKind::kKv512 && have_codes != have_scales)) {
    return Rejected("ds4 QAT packed outputs are incomplete");
  }
  // KV carries a F32 rotary tail starting at byte448 of each704-byte row.
  if (have_codes && (!Holds(desc.codes, codes, desc.kind == Ds4CacheKind::kKv512 ? 4 : 1) ||
                     Overlap(desc.values, values, desc.codes, codes))) {
    return Rejected("ds4 QAT codes are invalid or overlap values");
  }
  if (have_scales && (!Holds(desc.scales, scales, alignof(float)) ||
                      Overlap(desc.values, values, desc.scales, scales) ||
                      (have_codes && Overlap(desc.codes, codes, desc.scales, scales)))) {
    return Rejected("ds4 QAT scales are invalid or overlap another operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4CacheExpand(const Ds4CacheExpand& desc) {
  const auto shape = Shape(desc.kind);
  if (!shape || !Rows(desc.rows))
    return Rejected("ds4 expansion requires a supported kind and rows");
  const auto values = static_cast<std::uint64_t>(desc.rows) * shape->width * sizeof(float);
  const auto codes = static_cast<std::uint64_t>(desc.rows) * shape->code_bytes;
  const auto scales = static_cast<std::uint64_t>(desc.rows) * shape->scale_bytes;
  if (!Holds(desc.values, values, alignof(float)) ||
      !Holds(desc.codes, codes, desc.kind == Ds4CacheKind::kKv512 ? 4 : 1) ||
      !Holds(desc.scales, scales, alignof(float))) {
    return Rejected("ds4 expansion operand is invalid");
  }
  if (Overlap(desc.values, values, desc.codes, codes) ||
      Overlap(desc.values, values, desc.scales, scales) ||
      Overlap(desc.codes, codes, desc.scales, scales)) {
    return Rejected("ds4 expansion operands overlap");
  }
  if (desc.kind == Ds4CacheKind::kKv512) {
    constexpr std::uint64_t kTableBytes = 128 * sizeof(float);
    if (!Holds(desc.decode_table, kTableBytes, alignof(float)) ||
        Overlap(desc.decode_table, kTableBytes, desc.values, values) ||
        Overlap(desc.decode_table, kTableBytes, desc.codes, codes) ||
        Overlap(desc.decode_table, kTableBytes, desc.scales, scales)) {
      return Rejected("ds4 KV expansion decode table is invalid or overlaps an operand");
    }
  } else if (!Absent(desc.decode_table)) {
    return Rejected("ds4 indexer expansion has no decode table operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4CacheRawStore(const Ds4CacheRawStore& desc) {
  if (!Rows(desc.rows) || desc.cells == 0 || desc.cells > 8192 || desc.rows > desc.cells ||
      desc.rows - 1 > std::numeric_limits<std::uint32_t>::max() - desc.first) {
    return Rejected("ds4 raw ring dimensions/position are invalid or write slots repeat");
  }
  const auto source = static_cast<std::uint64_t>(desc.rows) * kDs4KvWidth * sizeof(float);
  const auto ring = static_cast<std::uint64_t>(desc.cells) * kDs4KvWidth * sizeof(float);
  if (!Holds(desc.source, source, alignof(float)) || !Holds(desc.ring, ring, alignof(float)) ||
      Overlap(desc.source, source, desc.ring, ring)) {
    return Rejected("ds4 raw store ranges are invalid or overlap");
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
