// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_ENGINE_QWEN38_WAVE_PLAN_H_
#define JITLLM_ENGINE_QWEN38_WAVE_PLAN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine/qwen38_plan.h"

namespace jitllm::engine {

inline constexpr std::size_t kQwen38WaveSlots = 4;

// Native runner bindings, in strictly ascending slot order. Model metadata,
// weights and mapped places remain the caller's. No admission or lease is
// obtained here: these must be owned slots of one live Qwen38Runner, with
// its current authenticated weight bindings and independent mutable places.
struct Qwen38TargetWaveInput {
  std::uint32_t slot = 0;
  const Qwen38Model* model = nullptr;
  kernels::ggml::Qwen38ChunkShape shape;
  Qwen38ChunkKind kind;
};

struct Qwen38DraftWaveInput {
  std::uint32_t slot = 0;
  const Qwen38Model* model = nullptr;
  kernels::ggml::Qwen38MtpShape shape;
};

struct Qwen38WavePlacement {
  std::uint64_t activations = 0;  // zero: measure only, as PlanQwen38Chunk
  std::uint64_t bytes = 0;
  bool paired = true;
  // The target runner enables sharing only through the guarded MMF path.
  bool share_target_head = false;
};

struct Qwen38WaveStats {
  std::uint64_t mxfp8_pairs = 0;
  std::uint64_t routed_pairs = 0;
  std::uint64_t packed_bytes = 0;
  std::uint64_t full_head_pairs = 0;
  std::uint64_t head_packed_bytes = 0;
  std::uint8_t paired_slots = 0;
};

// Owns fresh per-slot descriptors and a separate composition arena. It never
// borrows or rewrites a runner's scalar cached plan. The per-slot graph
// accessors supply source/output/state descriptors, not runnable scalar plans:
// only this object's final joint plan is placed and subsequently bound.
//
// No work is launched or captured here. BindPlanned performs the ordinary
// workspace/registry checks. A cache must destroy captured PlanRuns before
// this owner, after proven completion; neither destructor proves retirement.
// Mapped operands, activations and model metadata must outlive all uses.
class Qwen38WavePlanned : public PlannedBase {
 public:
  Qwen38WavePlanned() = default;
  Qwen38WavePlanned(const Qwen38WavePlanned&) = delete;
  Qwen38WavePlanned& operator=(const Qwen38WavePlanned&) = delete;
  Qwen38WavePlanned(Qwen38WavePlanned&&) = delete;
  Qwen38WavePlanned& operator=(Qwen38WavePlanned&&) = delete;
  ~Qwen38WavePlanned();

  const kernels::ggml::Qwen38Graph* target(std::size_t slot) const;
  const kernels::ggml::Qwen38MtpGraph* draft(std::size_t slot) const;
  std::span<ggml_tensor* const> nodes() const { return nodes_; }
  std::span<ggml_tensor* const> inputs() const { return inputs_; }
  std::uint8_t active_slots() const { return active_slots_; }
  const Qwen38WaveStats& stats() const { return stats_; }
  // What it holds on the host as its runner counts it (planned.h
  // PlannedHostBytes): every slot's plan, its own arenas and joined plan.
  std::uint64_t host_bytes() const;
  // One shared product: the slots' original descriptors, in slot order, and
  // their replacement.
  struct Product {
    std::vector<ggml_tensor*> originals;
    ggml_tensor* together = nullptr;
  };
  // Original head descriptors and their replacement, for selector/shape proof.
  std::span<const Product> head_products() const { return head_products_; }

 private:
  friend struct Qwen38WaveBuilder;
  std::array<std::unique_ptr<Qwen38Planned>, kQwen38WaveSlots> target_;
  std::array<std::unique_ptr<Qwen38MtpPlanned>, kQwen38WaveSlots> draft_;
  std::vector<ggml_tensor*> nodes_;
  std::vector<ggml_tensor*> inputs_;
  std::vector<ggml_tensor*> keep_;
  std::vector<Product> products_;
  std::optional<kernels::ggml::TensorArena> head_arena_;
  std::vector<Product> head_products_;
  Qwen38WaveStats stats_;
  std::uint8_t active_slots_ = 0;
};

// The most a wave of `slots` slots holds on the host (host_bytes()) when
// each slot's plan holds at most `slot_bytes` (PlannedHostBytes), launches
// at most `slot_nodes` nodes and has at most `products` products eligible to
// be shared: every slot's plan, the composition's arenas and the joined plan.
std::uint64_t Qwen38WaveHostBound(std::uint64_t slot_bytes, std::uint64_t slot_nodes,
                                  std::uint64_t products, std::uint64_t slots);

// Geometry-only eligibility, shared with conservative provisioning. The
// composer separately authenticates the immutable leaf and MMF selectors.
bool Qwen38FullHeadPairCandidate(const ggml_tensor* tensor);

// Only the original two-to-four-row cuBLAS BF16 product, with contiguous
// operands/output. The composer also checks shared immutable weights,
// parameters and compatible layouts; one-row vector products stay scalar.
bool Qwen38HcPairCandidate(const ggml_tensor* tensor);

// One to four actual slots, each one to four rows. Groups of consecutive
// compatible slots (in ascending order, up to all four) coalesce
// corresponding MXFP8-vector and routed-GEMV products, each at most
// sixteen rows, with paid GGML concats and split views. A lone slot,
// incompatible phase/product sequence or diagnostic capture retains the
// original operations. Ragged row counts may group; MTP pass/head/confidence
// differences do not. Compatible target verification groups also coalesce
// guarded two-to-four-row HC BF16 products on their original cuBLAS path.
// Target-head sharing may group compatible contiguous three- or four-row BF16
// full target heads into an ordinary MMF product of up to sixteen columns
// with paid F32 concats and a view a slot. Actual scalar/replacement
// selectors must all remain MMF; otherwise the heads stay original.
// Draft heads and all stateful operations stay original.
// Every unmodified registry identity and each replacement's precision tier
// are checked against fresh scalar plans. No dispatch occurs on any error.
std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38TargetWave(
    std::span<const Qwen38TargetWaveInput> requests, const kernels::ggml::DeviceChoices& choices,
    Qwen38WavePlacement placement = {});

std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38DraftWave(
    std::span<const Qwen38DraftWaveInput> requests, const kernels::ggml::DeviceChoices& choices,
    Qwen38WavePlacement placement = {});

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_QWEN38_WAVE_PLAN_H_
