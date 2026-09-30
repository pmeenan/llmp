// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original ds4 cache/QAT stages for the complete benchmark plan. Every
// range is a caller-owned logical view, backed and leased until completion.
// No allocation, cache counters, global sidecar registry or stream ownership.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_CACHE_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_CACHE_H_

#include <array>
#include <cstdint>
#include <expected>

#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

class LaunchContext;

struct Ds4CacheBuffer {
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
};

enum class Ds4CacheKind : std::uint8_t { kKv512, kIndexer128 };

inline constexpr std::uint32_t kDs4KvWidth = 512;
inline constexpr std::uint32_t kDs4KvPackedRowBytes = 704;
inline constexpr std::uint32_t kDs4KvScaleRowBytes = 28;
inline constexpr std::uint32_t kDs4IndexerWidth = 128;
inline constexpr std::uint32_t kDs4IndexerPackedRowBytes = 64;
inline constexpr std::uint32_t kDs4IndexerScaleRowBytes = 16;

// In-place F32 QAT. KV rounds only the 448 non-RoPE channels. Indexer
// applies the original normalized Hadamard before E2M1 quantization.
// With neither codes nor scales, keep only the original F32 result.
// Indexer permits scales alone (the Q producer's scoring sidecar); KV
// requires both packed arrays or neither. Packed arrays are row-0 views.
struct Ds4CacheQat {
  Ds4CacheKind kind = Ds4CacheKind::kKv512;
  Ds4CacheBuffer values;
  Ds4CacheBuffer codes;
  Ds4CacheBuffer scales;
  std::uint32_t rows = 0;
};

// Exact packed-row reconstruction to F32. The caller initializes/uploads
// the immutable 128-entry KV decode table, accounts its 512 bytes and
// retains it through completion. Indexer does not read a decode table.
struct Ds4CacheExpand {
  Ds4CacheKind kind = Ds4CacheKind::kKv512;
  Ds4CacheBuffer values;
  Ds4CacheBuffer codes;
  Ds4CacheBuffer scales;
  Ds4CacheBuffer decode_table;
  std::uint32_t rows = 0;
};

// Single sequence only. Writes contiguous QAT rows at (first+t)%cells,
// rounding every value through F16 into physically F32 ring storage,
// exactly as the original batch store. This has fixed host position;
// this launcher refuses graph capture; submit each new position eagerly.
struct Ds4CacheRawStore {
  Ds4CacheBuffer source;
  Ds4CacheBuffer ring;
  std::uint32_t first = 0;
  std::uint32_t rows = 0;
  std::uint32_t cells = 0;
};

std::array<float, 128> Ds4CacheDecodeTable();
std::expected<void, KernelFailure> CheckDs4CacheQat(const Ds4CacheQat& desc);
std::expected<void, KernelFailure> CheckDs4CacheExpand(const Ds4CacheExpand& desc);
std::expected<void, KernelFailure> CheckDs4CacheRawStore(const Ds4CacheRawStore& desc);

// CUDA builds only. Refusal queues nothing. An unknown CUDA failure
// faults the LaunchContext; caller retains every operand until recovery.
// These queue on jitLLM's stream, use no temporary workspace and do not
// prove completion. Caller fences the complete stage/plan normally.
std::expected<void, KernelFailure> RunDs4CacheQat(LaunchContext& launch, const Ds4CacheQat& desc);
std::expected<void, KernelFailure> RunDs4CacheExpand(LaunchContext& launch,
                                                     const Ds4CacheExpand& desc);
std::expected<void, KernelFailure> RunDs4CacheRawStore(LaunchContext& launch,
                                                       const Ds4CacheRawStore& desc);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_CACHE_H_
