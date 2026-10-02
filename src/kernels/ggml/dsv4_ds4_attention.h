// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary original ds4 attention reference for the complete-plan benchmark.
// Native production attention selection is unchanged.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_H_

#include <cstdint>
#include <expected>
#include <limits>

#include "kernels/ggml/dsv4_ds4_indexer.h"

namespace jitllm::kernels::ggml {

// Original token-stable ABI. Caller publishes the live values on jitLLM's
// stream before captured kernels and keeps the buffer address stable.
struct Ds4AttentionDecodeScalars {
  std::uint32_t pos0 = 0;
  std::uint32_t raw_row = 0;
  std::uint32_t raw_start = 0;
  std::uint32_t n_raw = 0;
  std::uint32_t n_comp = 0;
  std::uint32_t emit_phase = 0;
  std::uint32_t comp_row = 0;
  std::uint32_t index_row = 0;
  std::uint32_t flags = 0;
  std::uint32_t token = 0;
};
static_assert(sizeof(Ds4AttentionDecodeScalars) == 40);

struct Ds4AttentionDiagnostics {
  unsigned long long dense_packed_reads = 0;
  unsigned long long indexed_packed_reads = 0;
};
static_assert(sizeof(Ds4AttentionDiagnostics) == 16);

enum class Ds4AttentionDomain : std::uint8_t {
  kRawPrefill,
  kMixedPrefill,
  kMaskedPrefill,
  kDecodeHeads,
  kMixedRing,
  kIndexedRing,
};

enum class Ds4AttentionKind : std::uint8_t {
  kOriginalDefault,
  kScalar,
  kHeads8Online,
  kIndexedTwoPass,
  kTokenTile,
  kHeadGroup,
  kPerHeadSplit,
  kCublas,
};

// AttentionD512/H64; raw cells physically F32 with original F16-roundtrip
// values. Compressed F32 logical views remain valid even when packed FP8
// is the backed primary. Packed readers take704B codes/tail+28B scales/row
// and a caller-owned128-entry F32 decode table. Sink logits are finite.
// Original F32 Q, F16 mirrors/HMMA probability and F32 accumulation remain
// distinct from the ordinary native attention's value arithmetic.
struct Ds4Attention {
  Ds4CacheBuffer query{};
  Ds4CacheBuffer output{};
  Ds4CacheBuffer sinks{};
  Ds4CacheBuffer raw{};
  Ds4CacheBuffer compressed{};
  Ds4CacheBuffer compressed_codes{};
  Ds4CacheBuffer compressed_scales{};
  Ds4CacheBuffer decode_table{};
  Ds4CacheBuffer selected{};
  // A live layer requires tokens*compressed_cells F32 capacity; caller
  // publishes each live row at the original n_comp stride before replay.
  Ds4CacheBuffer mask{};
  Ds4CacheBuffer positions{};
  Ds4CacheBuffer bank_ids{};
  Ds4CacheBuffer draft_raw_count{};
  Ds4CacheBuffer decode_scalars{};
  Ds4CacheBuffer layer_scalars{};
  Ds4CacheBuffer scratch{};
  Ds4CacheBuffer diagnostics{};
  std::uint32_t tokens = 0;
  std::uint32_t first = 0;
  std::uint32_t raw_cells = 0;
  std::uint32_t raw_count = 0;
  std::uint32_t raw_start = 0;
  std::uint32_t compressed_cells = 0;
  std::uint32_t compressed_count = 0;
  std::uint32_t banks = 1;
  std::uint32_t window = 128;
  std::uint32_t ratio = 4;
  std::uint32_t top_k = 512;
  // Consecutive, same-bank host proof. UINT32_MAX means no such proof.
  // With per-row arrays, their host mirrors must establish this position
  // and uniform bank before token-tile mirrors may read bank_ids[0].
  std::uint32_t consecutive_first = std::numeric_limits<std::uint32_t>::max();
  bool allow_multisequence_heads8 = false;
  bool quality_mode = false;
  Ds4AttentionDomain domain = Ds4AttentionDomain::kMixedRing;
  Ds4AttentionKind kind = Ds4AttentionKind::kOriginalDefault;
};

// Layout of the selected complete producer/consumer chain in one charged
// caller-owned scratch view. Regions are256B aligned, disjoint and retained
// through completion. Counts/records are original int2{id,mask} unions,
// never an uncharged cache of previous selections.
struct Ds4AttentionScratch {
  std::uint64_t records = 0;
  std::uint64_t counts = 0;
  std::uint64_t raw_mirror = 0;
  std::uint64_t compressed_mirror = 0;
  std::uint64_t sorted_ids = 0;
  std::uint64_t predecoded = 0;
  std::uint64_t partials = 0;
  std::uint64_t gemm_keys = 0;
  std::uint64_t gemm_scores = 0;
  std::uint64_t gemm_output = 0;
  std::uint64_t bytes = 0;
  std::uint32_t record_stride = 0;
  std::uint32_t splits = 0;
  bool sort_ids = false;
  bool predecode = false;
};

struct Ds4AttentionDispatch {
  Ds4AttentionKind kind = Ds4AttentionKind::kScalar;
  Ds4AttentionScratch scratch{};
};

struct Ds4AttentionMask {
  Ds4CacheBuffer selected{};
  Ds4CacheBuffer mask{};
  std::uint32_t tokens = 0;
  std::uint32_t cells = 0;
  std::uint32_t top_k = 512;
};

// Shape validation is independent of CUDA. Original trusted device array
// contents must satisfy published bank/position/live-count capacities; an
// API client can never supply these addresses or payloads.
std::expected<void, KernelFailure> CheckDs4Attention(const Ds4Attention& desc);
std::expected<void, KernelFailure> CheckDs4AttentionMask(const Ds4AttentionMask& desc);
std::expected<Ds4AttentionScratch, KernelFailure> PlanDs4AttentionScratch(const Ds4Attention& desc,
                                                                          Ds4AttentionKind selected,
                                                                          std::uint32_t device_sms,
                                                                          bool predecode = true);

// Original shared-memory attributes are prepared once before graph capture.
// No numerical work is queued, no scratch allocated, no stream owned.
std::expected<void, KernelFailure> PrepareDs4Attention(LaunchContext& launch);
// No numerical work is queued. Record this resolved tier and paid layout
// with operand identities before a matched one-stage benchmark substitution.
std::expected<Ds4AttentionDispatch, KernelFailure> DescribeDs4AttentionDispatch(
    const LaunchContext& launch, const Ds4Attention& desc);
std::expected<void, KernelFailure> RunDs4AttentionMask(LaunchContext& launch,
                                                       const Ds4AttentionMask& desc);
// Auto preserves original eligibility/fallback order. Missing paid token-tile
// or HG scratch falls through before submission. Known CUDA failures fault
// the native context; no partially submitted alternate retry occurs.
std::expected<void, KernelFailure> RunDs4Attention(LaunchContext& launch, const Ds4Attention& desc);

// Experimental (docs/experiments/ds4-prefill-stages): the token-tile HCA core
// over F16 Q rows; arguments as the patched ds4 jitllm_ds4_hca_core_launch's.
// Prepare sets the kernel's shared-memory opt-in outside graph capture.
int Ds4HcaCoreQ16Prepare();
int Ds4HcaCoreQ16Launch(float* out, const float* sinks, const void* q, const void* raw,
                        const void* compressed, const void* records, const void* counts,
                        std::uint32_t stride, std::uint32_t tokens, std::uint32_t heads,
                        std::uint32_t raw_row_min, void* stream);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_H_
