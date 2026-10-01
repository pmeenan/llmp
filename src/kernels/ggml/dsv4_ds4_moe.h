// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original ds4 router/routed activation and aligned expert tiers. Explicit
// borrowed operands; jitLLM owns plans, streams, storage and completion.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_MOE_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_MOE_H_

#include <cstdint>
#include <expected>
#include <span>

#include "kernels/ggml/dsv4_ds4_cache.h"

namespace jitllm::kernels::ggml {

inline constexpr std::uint32_t kDs4MoeExperts = 256;
inline constexpr std::uint32_t kDs4MoeSelected = 6;
inline constexpr std::uint32_t kDs4MoeMaxTokens = 4096;

// Original router output is COMPACT [T,6]. Existing jitLLM captures keep
// six selected values in stride256; routed input expresses that stride and
// packs into distinct compact scratch. Never reinterpret 6-of256 as T*6.
enum class Ds4RouterSelect : std::uint8_t { kWarp, kParallel, kScalar };
struct Ds4Router {
  Ds4CacheBuffer logits{}, bias{}, hash{}, tokens{};
  Ds4CacheBuffer selected{}, weights{}, probabilities{};
  std::uint32_t rows = 0;
  std::uint32_t hash_rows = 0;
  Ds4RouterSelect select = Ds4RouterSelect::kWarp;
};
struct Ds4RouterCooperative {
  // Original batch cooperative tier, including a one-row batch. This
  // does not implement serial decode's router_fused_coop live-scalar tier.
  Ds4Router router{};
  Ds4CacheBuffer input{}, projection{}, partials{};  // F32 [T,4096], F16 [256,4096]
};
// Hash values must be import-validated distinct expert IDs. Live router
// projections/bias are finite model operands. These are internal binder
// contracts, not caller booleans that authorize unchecked device values.
std::expected<void, KernelFailure> CheckDs4Router(const Ds4Router& desc);
std::expected<void, KernelFailure> CheckDs4RouterCooperative(const Ds4RouterCooperative& desc);
std::expected<void, KernelFailure> CheckDs4MoeIds(std::span<const std::int32_t> ids,
                                                  std::uint32_t rows, std::uint32_t stride);

struct Ds4SharedSwiglu {
  Ds4CacheBuffer gate{}, up{}, output{};
  std::uint32_t rows = 0, width = 0;
};
struct Ds4MoeSum {
  Ds4CacheBuffer slots{}, output{};
  std::uint32_t rows = 0, width = 0;
};

enum class Ds4MoeTier : std::uint8_t { kVector, kDirect, kMaterialized, kClassic };
// Materialized implements the original >=1024-pair D2R tier only. Its
// default small shared-MMQ + ds4_swiglu_weighted_f32 tier is a follow-on;
// Classic is a distinct diagnostic tier and cannot stand in for it.
struct Ds4MoeShape {
  std::uint32_t rows = 0, input = 0, middle = 0, output = 0;
};
enum class Ds4MoeQuant : std::uint8_t { kD4, kQ81 };
struct Ds4MoeProducer {
  // Emitted by a checked producer; D4 slack must already be initialized.
  // Producer and source storage are disjoint, with both leased until the
  // consumer completes. These identity fields do not replace byte controls.
  Ds4CacheBuffer storage{};
  std::uint64_t source_address = 0, generation = 0;
  std::uint32_t rows = 0, width = 0;
  Ds4MoeQuant kind = Ds4MoeQuant::kD4;
};
// Caller owns the replacement physical IQ2 gate/up and paired-Q2 down
// sets from the checked repack binder. No second raw expert weight set.
// Input is contiguous F32 [T,input]; selected/weights have explicit
// element row strides >=6. Content is original router or validated fixed
// IDs; all six must be valid/distinct per row. A graph reads current data.
// Direct/materialized down D2R requires output%128==0 or output%128<16
// so its eight warps follow the same CTA-barrier path. All outputs are even.
struct Ds4Moe {
  Ds4MoeShape shape{};
  Ds4MoeTier tier = Ds4MoeTier::kDirect;
  std::uint64_t generation = 0;
  Ds4MoeProducer producer{};  // absent means original fresh quantization
  Ds4CacheBuffer input{}, gate_weights{}, up_weights{}, down_weights{};
  Ds4CacheBuffer selected{}, weights{};
  std::uint32_t selected_stride = 6, weight_stride = 6;
  Ds4CacheBuffer compact_ids{}, compact_weights{};  // only for noncompact input
  Ds4CacheBuffer ids_source{}, ids_destination{}, expert_bounds{}, work{};
  Ds4CacheBuffer input_quant{}, down_quant{}, gate{}, up{}, middle{}, down{}, sum{};
};
// Physical original host input arena remains conservatively gathered-sized
// even for the direct token-compact consumer. Guard128 blocks are explicitly
// zeroed safety work (original default YBUF memset is OFF); it is charged.
struct Ds4MoeLayout {
  std::uint64_t gate_weight_bytes = 0, down_weight_bytes = 0;
  std::uint64_t input_quant_bytes = 0, down_quant_bytes = 0;
  std::uint64_t work_bytes = 0, middle_bytes = 0, down_bytes = 0;
  std::uint64_t input_payload_bytes = 0, down_payload_bytes = 0;
};
std::expected<Ds4MoeLayout, KernelFailure> Ds4MoeLayoutOf(const Ds4MoeShape& shape,
                                                          Ds4MoeTier tier);
std::expected<void, KernelFailure> CheckDs4Moe(const Ds4Moe& desc);
std::expected<void, KernelFailure> CheckDs4SharedSwiglu(const Ds4SharedSwiglu& desc);
std::expected<void, KernelFailure> CheckDs4MoeSum(const Ds4MoeSum& desc);

// CUDA only. No allocator/global original row registry/stream. Refusal
// queues nothing. Success submits work, not proof of completion; operands,
// scratch and any graph stay leased through native completion. All tiers
// are selected before launch; failures never retry another numerical tier.
std::expected<void, KernelFailure> RunDs4Router(LaunchContext& launch, const Ds4Router& desc);
std::expected<void, KernelFailure> RunDs4RouterCooperative(LaunchContext& launch,
                                                           const Ds4RouterCooperative& desc);
std::expected<void, KernelFailure> RunDs4SharedSwiglu(LaunchContext& launch,
                                                      const Ds4SharedSwiglu& desc);
std::expected<void, KernelFailure> RunDs4MoeSum(LaunchContext& launch, const Ds4MoeSum& desc);
// Exact native borrowed-workspace need for classic MMQ's reused stream-K
// fixup; other tiers use only explicitly described scratch and return zero.
std::expected<std::uint64_t, KernelFailure> PlanDs4MoeScratch(const LaunchContext& launch,
                                                              const Ds4Moe& desc);
std::expected<void, KernelFailure> RunDs4Moe(LaunchContext& launch, const Ds4Moe& desc);

}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_MOE_H_
