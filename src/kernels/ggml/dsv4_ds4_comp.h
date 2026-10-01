// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original ds4 compressor primitives under jitLLM ownership. Projections,
// state, output rows and packed mirrors are explicit borrowed views. No
// allocator, global row registry, device scalar substrate or foreign stream.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_COMP_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_COMP_H_

#include <cstdint>
#include <expected>

#include "kernels/ggml/dsv4_ds4_cache.h"

namespace jitllm::kernels::ggml {

inline constexpr std::uint32_t kDs4CompMaxTokens = 4096;
inline constexpr float kDs4CompBootScore = -1.0e30f;

enum class Ds4CompApe : std::uint8_t { kF32, kF16 };
enum class Ds4CompPath : std::uint8_t { kZeroPrefix, kAligned, kRows };

// KV512 permits ratio4 or128; indexer128 permits ratio4. Ratio4 state is
// [8][2*head_width] F32; ratio128 state is[128][512] F32. The plan owns
// position/count metadata and advances it only after proved completion.
struct Ds4CompState {
  Ds4CacheBuffer kv{};
  Ds4CacheBuffer score{};
  Ds4CacheKind kind = Ds4CacheKind::kKv512;
  std::uint32_t ratio = 4;
};

// Model-derived parameters must be supplied; there are no guessed model
// defaults. Original rotary tail is64 for both Flash compressor kinds.
struct Ds4CompRope {
  std::uint32_t original_context = 0;
  float frequency_base = 0;
  float frequency_scale = 0;
  float extension = 0;
  float attention_factor = 0;
  float beta_fast = 0;
  float beta_slow = 0;
};

// Stateless aligned pool. Input projections are[tokens][coff*head_width]
// F32, APE[ratio][coff*head_width], output[tokens/ratio][head_width] F32.
// first is ratio-aligned; a nonzero first at ratio4 reads the previous
// four rows of state for its first overlap window. No state write, norm,
// RoPE or QAT here; this is also the fixed-shape current-operand graph seam.
struct Ds4CompPool {
  Ds4CompState state{};
  Ds4CacheBuffer kv{};
  Ds4CacheBuffer score{};
  Ds4CacheBuffer ape{};
  Ds4CacheBuffer values{};
  Ds4CompApe ape_format = Ds4CompApe::kF32;
  std::uint32_t first = 0;
  std::uint32_t tokens = 0;
};

// A complete compressor chunk after the root-owned projection products:
// original zero-prefix bulk, aligned replay, or per-row ragged path. Every
// emitted row receives weighted RMS, original RoPE and mandatory KV FP8 or
// indexer Hadamard/FP4 QAT, even without packed mirrors. Packed codes/scales
// are both present or both absent, and address only newly emitted rows.
// values likewise addresses only new rows; before/capacity refer to the
// complete cache. No hidden F32 cache or host count mutation.
struct Ds4CompChunk {
  Ds4CompState state{};
  Ds4CacheBuffer kv{};
  Ds4CacheBuffer score{};
  Ds4CacheBuffer ape{};
  Ds4CacheBuffer norm{};
  Ds4CacheBuffer values{};
  Ds4CacheBuffer codes{};
  Ds4CacheBuffer scales{};
  Ds4CompApe ape_format = Ds4CompApe::kF32;
  Ds4CompRope rope{};
  std::uint32_t first = 0;
  std::uint32_t tokens = 0;
  std::uint32_t before = 0;
  std::uint32_t capacity = 0;
  float rms_epsilon = 0;
};

struct Ds4CompPlan {
  Ds4CompPath path;
  std::uint32_t emitted;
  std::uint32_t after;
  bool refresh_required;
};

// Original ratio4 post-bulk refresh is a separate marked stage: after
// RunChunk, the caller must compute BOTH last-four-token small projections
// with the original product path, then RunRefresh before continuation.
// Reusing a slice of the wide projections changes original state rounding.
// Their buffers may reuse retired bulk storage; no extra allocation here.
struct Ds4CompRefresh {
  Ds4CompState state{};
  Ds4CacheBuffer kv{};
  Ds4CacheBuffer score{};
  Ds4CacheBuffer ape{};
  Ds4CompApe ape_format = Ds4CompApe::kF32;
  std::uint32_t first = 0;  // absolute position of the four refreshed tokens
};

std::expected<Ds4CompPlan, KernelFailure> PlanDs4Comp(const Ds4CompChunk& desc);
// Number of complete groups after processing absolute token position.
std::expected<std::uint32_t, KernelFailure> Ds4CompCausalCount(std::uint32_t position,
                                                               std::uint32_t ratio);
std::expected<void, KernelFailure> CheckDs4CompState(const Ds4CompState& desc);
std::expected<void, KernelFailure> CheckDs4CompPool(const Ds4CompPool& desc);
std::expected<void, KernelFailure> CheckDs4CompChunk(const Ds4CompChunk& desc);
std::expected<void, KernelFailure> CheckDs4CompRefresh(const Ds4CompRefresh& desc);

// CUDA only. Init implements C boot/Clear's finite -1e30 score sentinel.
// Original bulk/replay/refresh subsequently reset scores to IEEE -Inf.
// Refusal submits nothing; success queues work and does not prove completion.
// Full chunk/refresh capture is refused because position/emit/reset choices
// are host-bound. Pool captures fixed shape and reads current operands.
std::expected<void, KernelFailure> RunDs4CompInitialize(LaunchContext& launch,
                                                        const Ds4CompState& desc);
std::expected<void, KernelFailure> RunDs4CompPool(LaunchContext& launch, const Ds4CompPool& desc);
std::expected<void, KernelFailure> RunDs4CompChunk(LaunchContext& launch, const Ds4CompChunk& desc);
std::expected<void, KernelFailure> RunDs4CompRefresh(LaunchContext& launch,
                                                     const Ds4CompRefresh& desc);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_COMP_H_
