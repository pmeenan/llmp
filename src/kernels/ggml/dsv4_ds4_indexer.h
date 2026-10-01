// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary original ds4 indexer reference for the complete-plan benchmark.
// No production plan
// selects this interface. Buffers are native owned/leased logical views.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_INDEXER_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_INDEXER_H_

#include <cstdint>
#include <expected>
#include <limits>

#include "kernels/ggml/dsv4_ds4_cache.h"

namespace jitllm::kernels::ggml {

class LaunchContext;

// Original per-layer live-count ABI; caller publishes on the native stream
// and retains the stable address through every graph replay/completion.
struct Ds4IndexerLayerScalars {
  std::uint32_t n_comp = 0;
  std::uint32_t n_index_comp = 0;
  std::uint32_t comp_row = 0;
  std::uint32_t index_row = 0;
};
static_assert(sizeof(Ds4IndexerLayerScalars) == 16);

// Original numerical tripwires, now an explicit caller-owned device operand.
// Initialize before launch; read after completion. Bound violations remain
// diagnostic failures even where the original selector clamps defensively.
struct Ds4IndexerDiagnostics {
  unsigned long long packed_reads = 0;
  std::uint32_t bound_count = 0;
  std::uint32_t bound_live = 0;
  std::uint32_t bound_stride = 0;
  std::uint32_t reserved = 0;
};
static_assert(sizeof(Ds4IndexerDiagnostics) == 24);

enum class Ds4IndexerScoreKind : std::uint8_t {
  kOriginalDefault,
  kScalar,
  kDirectOne,
  kMultisequenceScalar,
  kMultisequenceV5d,
  kMultisequenceV5e,
  kWmma16,
  kWmma32,
  kWmma64,
  kWmma128,
  kMxf4,
};

enum class Ds4IndexerSelectKind : std::uint8_t {
  kOriginalDefault,
  kInsertion,
  kBitonic1024,
  kBitonic2048,
  kBitonic4096,
  kCub8192,
  kBitonic8192,
  kStream512,
  kChunkTree,
};

// D128/H64 original model geometry. Keys have cells_per_bank rows/bank,
// rounded F32 plus optional packed E2M1 mirror(64B codes+16B scales/row).
// A packed primary still supplies the original valid F32 logical view;
// readers taking packed operands do not require its payload to be backed.
// Dispatch must never select an F32 read when that payload is absent.
// Q is the original Hadamard/E2M1 rounded F32[tokens][64][128]; weights
// are F32[tokens][64]. Scales alone are legal for V5D/V5E; Q codes require
// the scales, in the original64B+16B per(token,head) producer layout.
struct Ds4IndexerScores {
  Ds4CacheBuffer query{};
  Ds4CacheBuffer weights{};
  Ds4CacheBuffer keys{};
  Ds4CacheBuffer key_codes{};
  Ds4CacheBuffer key_scales{};
  Ds4CacheBuffer query_codes{};
  Ds4CacheBuffer query_scales{};
  Ds4CacheBuffer positions{};
  Ds4CacheBuffer bank_ids{};
  Ds4CacheBuffer layer_scalars{};
  Ds4CacheBuffer scores{};
  Ds4CacheBuffer scratch{};
  Ds4CacheBuffer diagnostics{};
  std::uint32_t tokens = 0;
  std::uint32_t cells = 0;
  std::uint32_t cells_per_bank = 0;
  std::uint32_t banks = 1;
  std::uint32_t first = 0;
  std::uint32_t ratio = 4;
  // Eager equals cells. Captured direct-one equals the stable capacity;
  // live n_index_comp must be1..score_band and is published by the caller.
  std::uint32_t score_band = 0;
  // Trusted consecutive same-bank claim, proved from host row mirrors.
  // UINT32_MAX means no claim. Clearing per-row arrays is valid only for
  // this original FE1 path, with at least two consecutive rows.
  std::uint32_t consecutive_bank = std::numeric_limits<std::uint32_t>::max();
  float scale = 1.0f;
  bool causal = true;
  bool quality_mode = false;
  Ds4IndexerScoreKind kind = Ds4IndexerScoreKind::kOriginalDefault;
};

// Original score/index descending total order, lower index on equal scores.
// Sentinels/causal -infinity retain the original selector behavior. Native
// score producers provide finite values or -infinity; NaN is not an input.
struct Ds4IndexerSelect {
  Ds4CacheBuffer scores{};
  Ds4CacheBuffer selected{};
  Ds4CacheBuffer layer_scalars{};
  Ds4CacheBuffer scratch{};
  Ds4CacheBuffer diagnostics{};
  std::uint32_t tokens = 0;
  std::uint32_t cells = 0;
  std::uint32_t score_band = 0;
  std::uint32_t top_k = 512;
  Ds4IndexerSelectKind kind = Ds4IndexerSelectKind::kOriginalDefault;
};

struct Ds4IndexerDispatch {
  Ds4IndexerScoreKind score_kind = Ds4IndexerScoreKind::kScalar;
  Ds4IndexerSelectKind select_kind = Ds4IndexerSelectKind::kInsertion;
  // Diagnostic facts from the actual compiled CUB type/native device. No
  // caller-provided Boolean selects a numerical tier or changes arithmetic.
  std::uint64_t cub_temp_storage_bytes = 0, select_dynamic_shared_bytes = 0,
                device_shared_optin = 0;
  bool cub_available = false;
};

// Device capability and capture status are checked again by the launcher.
// MXF4 is only eligible on sm_121a, eager, ratio4/top512, tokens>=32 and
// cells>=1024, with packed keys. Original query producer mirrors skip the
// requantizer; missing mirrors use the original requantizer and paid scratch.
std::expected<void, KernelFailure> CheckDs4IndexerScores(const Ds4IndexerScores& desc);
std::expected<void, KernelFailure> CheckDs4IndexerSelect(const Ds4IndexerSelect& desc);
std::expected<std::uint64_t, KernelFailure> Ds4IndexerScoreScratchBytes(
    const Ds4IndexerScores& desc);
std::expected<std::uint64_t, KernelFailure> Ds4IndexerSelectScratchBytes(
    const Ds4IndexerSelect& desc);
// Set original CUB dynamic-smem and V5D/V5E carveout before capture.
// Caller invokes once per device setup; no queued numerical work.
std::expected<void, KernelFailure> PrepareDs4Indexer(LaunchContext& launch);
// No numerical work is queued. Freeze the original chosen score/select
// tiers with operand identities before replacing one benchmark stage.
std::expected<Ds4IndexerDispatch, KernelFailure> DescribeDs4IndexerDispatch(
    const LaunchContext& launch, const Ds4IndexerScores& scores, const Ds4IndexerSelect& select);

// Refusal submits nothing. Unknown device failure faults LaunchContext;
// caller retains owners until recovery. No allocation, stream ownership or
// completion proof here. Original CUDA numerical flags are required.
std::expected<void, KernelFailure> RunDs4IndexerScores(LaunchContext& launch,
                                                       const Ds4IndexerScores& desc);
std::expected<void, KernelFailure> RunDs4IndexerSelect(LaunchContext& launch,
                                                       const Ds4IndexerSelect& desc);
std::expected<void, KernelFailure> RunDs4IndexerScoreSelect(LaunchContext& launch,
                                                            const Ds4IndexerScores& scores,
                                                            const Ds4IndexerSelect& select);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_INDEXER_H_
