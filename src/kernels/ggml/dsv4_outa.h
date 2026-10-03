// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_KERNELS_GGML_DSV4_OUTA_H_
#define JITLLM_KERNELS_GGML_DSV4_OUTA_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

class LaunchContext;
inline constexpr const char* kDsv4OutAName = "jitllm.dsv4.outa_prefill";
// Experimental: the same, with a coalesced weight repack (identical bytes).
inline constexpr const char* kDsv4OutAFastPackName = "jitllm.dsv4.outa_prefill_fastpack";

struct Dsv4OutAParams {
  std::int32_t original_context = 0;
  float base = 0;
  float scale = 0;
  float extension = 0;
  float attention = 0;
  float beta_fast = 0;
  float beta_slow = 0;
};

// A prefill chunk of T rows, kDsv4OutAMinRows <= T <= kDsv4OutAMaxRows:
// unrotated packed F32[512,64,T,1], raw Q8_0[4096,1024,8,1], and I32
// positions [T]. Emits F32[8192,Dsv4OutARows(T),1,1]: the core stores whole
// 16-row WMMA tiles, so the output holds T rounded up to 16 rows, of which
// the graph reads the first T. Decode and verify chunks, and other graph
// shapes, retain inverse RoPE, native MMQ and its layout copy.
inline constexpr std::int64_t kDsv4OutAMinRows = 64;
inline constexpr std::int64_t kDsv4OutAMaxRows = 4096;
constexpr std::int64_t Dsv4OutARows(std::int64_t rows) { return (rows + 15) / 16 * 16; }
bool Dsv4OutAFits(const ggml_tensor* w, const ggml_tensor* x, const ggml_tensor* pos,
                  const Dsv4OutAParams& params);
ggml_tensor* Dsv4OutA(ggml_context* context, ggml_tensor* weights, ggml_tensor* heads,
                      ggml_tensor* positions, const Dsv4OutAParams& params);
Dsv4OutAParams Dsv4OutAParamsOf(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4OutA(const ggml_tensor* node);

// Provider-owned scratch contains one reused weight-plane pair and a
// rotation table. Packing, table preparation and the staged F16 product
// are paid on each invocation; no persistent expanded weight copies.
bool Dsv4OutASupported(const LaunchContext& launch);
std::expected<std::uint64_t, KernelFailure> PlanDsv4OutA(const LaunchContext& launch,
                                                         const ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4OutA(LaunchContext& launch, ggml_tensor* node,
                                               bool fast_pack = false);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_OUTA_H_
