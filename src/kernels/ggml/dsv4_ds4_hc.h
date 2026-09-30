// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Borrowed views for the original ds4 wide-prefill HC/norm stages. No
// allocation, original runtime, dispatch policy or sidecar registry. The
// bound plan accounts and leases every input/output until completion.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_HC_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_HC_H_

#include <cstdint>
#include <expected>

#include "kernels/ggml/dsv4_ds4_cache.h"

namespace jitllm::kernels::ggml {

inline constexpr std::uint32_t kDs4HcLanes = 4;
inline constexpr std::uint32_t kDs4HcSplitWidth = 24;
inline constexpr std::uint32_t kDs4HcWidth = 4096;
inline constexpr std::uint32_t kDs4HcMaxRows = 4096;

// Contiguous rows. Absent weights select plain RMS, with exactly one of
// values(F32) or values_f16. Weighted RMS requires values and optionally
// dual-emits values_f16. q8_d4 additionally emits the original producer's
// [width/128][rows] blocks of144 bytes (16 F32-scale bytes +128 signed codes),
// eligible only with F16 output, rows>=64 and width divisible by128.
// Exact source==values is supported; every other write overlap is refused.
// Input width is1..16384, rows1..4096, epsilon finite and positive.
struct Ds4Rms {
  Ds4CacheBuffer source{};
  Ds4CacheBuffer weights{};
  Ds4CacheBuffer values{};
  Ds4CacheBuffer values_f16{};
  Ds4CacheBuffer q8_d4{};
  std::uint32_t width = kDs4HcWidth;
  std::uint32_t rows = 0;
  float epsilon = 1e-6f;
};

// Per-row mix/split is24 F32: pre[4], post[4], column-major comb[16].
// Scale is3 F32, base24 F32. Iterations1..20; the complete Flash plan uses20.
struct Ds4HcSplit {
  Ds4CacheBuffer mix{};
  Ds4CacheBuffer scale{};
  Ds4CacheBuffer base{};
  Ds4CacheBuffer split{};
  std::uint32_t rows = 0;
  std::uint32_t iterations = 20;
  float epsilon = 1e-6f;
};

// Residual layout [rows][4][width], values[rows][width]. Each coefficient
// row has4 standalone weights or24 split values; only its first4 are read.
struct Ds4HcWeighted {
  Ds4CacheBuffer residual{};
  Ds4CacheBuffer weights{};
  Ds4CacheBuffer values{};
  std::uint32_t width = kDs4HcWidth;
  std::uint32_t rows = 0;
  std::uint32_t weight_stride = kDs4HcSplitWidth;
};

// Original fused split+weighted-sum primitive for wide prefill. The next
// weighted norm is an explicit Ds4Rms producer, preserving the original
// wide-row eligibility (no small cooperative whole-HC substitution).
struct Ds4HcPre {
  Ds4HcSplit coefficients{};
  Ds4CacheBuffer residual{};
  Ds4CacheBuffer values{};
  std::uint32_t width = kDs4HcWidth;
};

// Original expand, with optional shared/block add. split[rows][24],
// residual/output[rows][4][width], block/add[rows][width]. With values_f16
// present, select the original4096x4 fused expand+next-RMS/F16 kernel and
// require rows>8 (original default native-F16 ceiling). This fused kernel
// has the original contraction behavior; it is not claimed byte-equal to
// plain expand followed by RMS. Without F16 output, the final layer uses
// the plain runtime-loop expand. No hidden temporary or row-count state.
// moe_unsummed[rows][6][4096] selects the ascending, isfinite-guarded6-slot
// sum inside fused expand. Then block must be absent and add must be present.
struct Ds4HcExpand {
  Ds4CacheBuffer block{};
  Ds4CacheBuffer add{};
  Ds4CacheBuffer residual{};
  Ds4CacheBuffer split{};
  Ds4CacheBuffer values{};
  Ds4CacheBuffer values_f16{};
  Ds4CacheBuffer moe_unsummed{};
  std::uint32_t width = kDs4HcWidth;
  std::uint32_t rows = 0;
  float epsilon = 1e-6f;
};

// Final head's sigmoid coefficients: pre/output[rows][4], scale1, base4.
// Compose with Ds4HcWeighted(weight_stride4) and the output-weighted RMS.
struct Ds4HcHeadWeights {
  Ds4CacheBuffer pre{};
  Ds4CacheBuffer scale{};
  Ds4CacheBuffer base{};
  Ds4CacheBuffer values{};
  std::uint32_t rows = 0;
  float epsilon = 1e-6f;
};

std::expected<void, KernelFailure> CheckDs4Rms(const Ds4Rms& desc);
std::expected<void, KernelFailure> CheckDs4HcSplit(const Ds4HcSplit& desc);
std::expected<void, KernelFailure> CheckDs4HcWeighted(const Ds4HcWeighted& desc);
std::expected<void, KernelFailure> CheckDs4HcPre(const Ds4HcPre& desc);
std::expected<void, KernelFailure> CheckDs4HcExpand(const Ds4HcExpand& desc);
std::expected<void, KernelFailure> CheckDs4HcHeadWeights(const Ds4HcHeadWeights& desc);

// CUDA only. Refusal submits nothing; normal Run queues on jitLLM's stream
// without proving completion. Graph replay reads current contents of every
// caller-owned view; no host position, cache cursor or dynamic shape is baked.
std::expected<void, KernelFailure> RunDs4Rms(LaunchContext& launch, const Ds4Rms& desc);
std::expected<void, KernelFailure> RunDs4HcSplit(LaunchContext& launch, const Ds4HcSplit& desc);
std::expected<void, KernelFailure> RunDs4HcWeighted(LaunchContext& launch,
                                                    const Ds4HcWeighted& desc);
std::expected<void, KernelFailure> RunDs4HcPre(LaunchContext& launch, const Ds4HcPre& desc);
std::expected<void, KernelFailure> RunDs4HcExpand(LaunchContext& launch, const Ds4HcExpand& desc);
std::expected<void, KernelFailure> RunDs4HcHeadWeights(LaunchContext& launch,
                                                       const Ds4HcHeadWeights& desc);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_HC_H_
