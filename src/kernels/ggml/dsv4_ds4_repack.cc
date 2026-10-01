// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_repack.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Reject(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

std::optional<std::uint64_t> Multiply(std::uint64_t a, std::uint64_t b) {
  std::uint64_t value = 0;
  if (__builtin_mul_overflow(a, b, &value)) return std::nullopt;
  return value;
}

std::optional<std::uint64_t> Add(std::uint64_t a, std::uint64_t b) {
  std::uint64_t value = 0;
  if (__builtin_add_overflow(a, b, &value)) return std::nullopt;
  return value;
}

std::optional<std::uint64_t> Align64(std::uint64_t value) {
  const auto rounded = Add(value, 63);
  if (!rounded) return std::nullopt;
  return *rounded & ~std::uint64_t{63};
}

bool Range(Ds4CacheBuffer buffer, std::uint64_t bytes, std::uint64_t alignment) {
  return buffer.address != 0 && buffer.address % alignment == 0 && buffer.bytes >= bytes &&
         buffer.bytes <= std::numeric_limits<std::uint64_t>::max() - buffer.address;
}

}  // namespace

std::expected<Ds4AlignedLayout, KernelFailure> Ds4AlignedLayoutOf(const Ds4AlignedShape& shape) {
  std::uint32_t quantum = 0;
  std::uint32_t raw_block = 0;
  switch (shape.kind) {
    case Ds4AlignedKind::kIq2Xxs:
      quantum = 256;
      raw_block = 66;
      break;
    case Ds4AlignedKind::kQ2K:
      quantum = 256;
      raw_block = 84;
      break;
    case Ds4AlignedKind::kQ8Dense:
      quantum = 32;
      raw_block = 34;
      break;
    default:
      return Reject("unknown ds4 aligned weight kind");
  }
  if (shape.input == 0 || shape.input % quantum != 0 || shape.output == 0 || shape.groups == 0 ||
      (shape.kind == Ds4AlignedKind::kQ2K && shape.output % 2 != 0) ||
      (shape.kind == Ds4AlignedKind::kQ8Dense && shape.groups != 1)) {
    return Reject("invalid ds4 aligned weight geometry");
  }
  const auto row_blocks = shape.input / quantum;
  const auto rows = Multiply(shape.output, shape.groups);
  const auto blocks = rows ? Multiply(*rows, row_blocks) : std::nullopt;
  const auto raw_bytes = blocks ? Multiply(*blocks, raw_block) : std::nullopt;
  if (!raw_bytes) return Reject("ds4 aligned weight size overflows");

  std::optional<std::uint64_t> scales;
  std::optional<std::uint64_t> codes;
  std::optional<std::uint64_t> payload;
  if (shape.kind == Ds4AlignedKind::kQ2K) {
    const auto pairs = *blocks / 2;
    const auto dm = Multiply(pairs, 8);
    const auto sc = Multiply(pairs, 32);
    const auto qs = Multiply(pairs, 128);
    scales = dm ? Align64(*dm) : std::nullopt;
    const auto aligned_sc = sc ? Align64(*sc) : std::nullopt;
    codes = scales && aligned_sc ? Add(*scales, *aligned_sc) : std::nullopt;
    payload = codes && qs ? Add(*codes, *qs) : std::nullopt;
  } else {
    scales = 0;
    const auto dq = Multiply(*blocks, 2);
    const auto qs = Multiply(*blocks, shape.kind == Ds4AlignedKind::kIq2Xxs ? 64U : 32U);
    codes = dq ? Align64(*dq) : std::nullopt;
    payload = codes && qs ? Add(*codes, *qs) : std::nullopt;
  }
  if (!scales || !codes || !payload) return Reject("ds4 aligned sections overflow");
  return Ds4AlignedLayout{.raw_bytes = base::Bytes(*raw_bytes),
                          .packed_bytes = base::Bytes(*payload),
                          .blocks = *blocks,
                          .scales_offset = *scales,
                          .codes_offset = *codes,
                          .blocks_per_row = row_blocks,
                          .raw_block_bytes = raw_block};
}

std::expected<void, KernelFailure> CheckDs4RepackInitialize(const Ds4AlignedShape& shape,
                                                            Ds4CacheBuffer packed) {
  const auto layout = Ds4AlignedLayoutOf(shape);
  if (!layout) return std::unexpected(layout.error());
  if (!Range(packed, layout->packed_bytes.value(), 64)) {
    return Reject("ds4 aligned output is short, misaligned or wraps");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4RepackChunk(const Ds4RepackChunk& desc) {
  const auto layout = Ds4AlignedLayoutOf(desc.shape);
  if (!layout) return std::unexpected(layout.error());
  if (desc.blocks == 0 || desc.first_block > layout->blocks ||
      desc.blocks > layout->blocks - desc.first_block) {
    return Reject("ds4 repack chunk is empty or exceeds the tensor");
  }
  if (desc.shape.kind == Ds4AlignedKind::kQ2K) {
    const auto pair_blocks = std::uint64_t{layout->blocks_per_row} * 2;
    if (desc.first_block % pair_blocks != 0 || desc.blocks % pair_blocks != 0) {
      return Reject("ds4 Q2 chunk must contain complete row pairs");
    }
  }
  std::uint32_t threads = 2;
  if (desc.shape.kind == Ds4AlignedKind::kIq2Xxs) {
    threads = 8;
  } else if (desc.shape.kind == Ds4AlignedKind::kQ2K) {
    threads = 16;
  }
  const auto work = Multiply(desc.blocks, threads);
  const auto rounded = work ? Add(*work, 255) : std::nullopt;
  if (!rounded || *rounded / 256 > std::uint64_t{2147483647}) {
    return Reject("ds4 repack chunk exceeds the CUDA X grid");
  }
  // The complete tensor raw size was checked, so its bounded chunk product
  // cannot overflow. Conservative whole-output aliasing prevents the
  // independent CUDA threads from overwriting unread source blocks.
  const auto raw_bytes = desc.blocks * layout->raw_block_bytes;
  const auto packed_bytes = layout->packed_bytes.value();
  if (!Range(desc.raw, raw_bytes, 2) || !Range(desc.packed, packed_bytes, 64)) {
    return Reject("ds4 repack operands are short, misaligned or wrap");
  }
  if (desc.raw.address < desc.packed.address + packed_bytes &&
      desc.packed.address < desc.raw.address + raw_bytes) {
    return Reject("ds4 repack output overlaps the source");
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
