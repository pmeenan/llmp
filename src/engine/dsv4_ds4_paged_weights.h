// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary matched-ds4 benchmark: original aligned weight files paged by
// the native catalog, storage and device providers. Canonical raw expert
// groups are not registered alongside these replacement tensors. This is
// neither an on-disk artifact format nor another serving implementation.
#ifndef JITLLM_ENGINE_DSV4_DS4_PAGED_WEIGHTS_H_
#define JITLLM_ENGINE_DSV4_DS4_PAGED_WEIGHTS_H_

#include <cstdint>
#include <span>
#include <vector>

#include "engine/dsv4_ds4_weights.h"
#include "engine/paged_node.h"
#include "scheduler/scheduler.h"

namespace jitllm::engine {

struct Ds4AlignedWeightView {
  bool expert_array = false;
  std::uint32_t index = 0;
  kernels::ggml::Ds4AlignedShape shape{};
  kernels::ggml::Ds4AlignedLayout layout{};
  std::uint64_t address = 0;
  base::Sha256Digest stored_sha256{};
  base::Sha256Digest content_key{};
};

class Ds4PagedAlignedWeights {
 public:
  Ds4PagedAlignedWeights() = default;
  Ds4PagedAlignedWeights(const Ds4PagedAlignedWeights&) = delete;
  Ds4PagedAlignedWeights& operator=(const Ds4PagedAlignedWeights&) = delete;
  Ds4PagedAlignedWeights(Ds4PagedAlignedWeights&&) = delete;
  Ds4PagedAlignedWeights& operator=(Ds4PagedAlignedWeights&&) = delete;
  // Caller performs node teardown and Release before destroying file owners.
  ~Ds4PagedAlignedWeights() = default;

  // Before node.Start. Files are complete outputs of PrepareDs4WeightFile,
  // in the exact planned tensor order. Ownership moves here only after the
  // complete file/geometry inventory passes validation. A reservation and
  // catalog extents consume virtual address space, not resident backing.
  Status Reserve(PagedNode& node, const Ds4PreparedWeightSet& plan,
                 std::vector<Ds4PreparedWeightFile>& files);
  // After Start, before Run: register native immutable page sources/spans.
  Status Register(PagedNode& node, int owner);
  void CheckPlaces(const scheduler::Scheduler& scheduler, PlaceCheck& check) const;
  const Ds4AlignedWeightView* Find(std::uint32_t index, bool expert_array) const;
  std::span<const Ds4AlignedWeightView> views() const { return views_; }
  const std::vector<catalog::ExtentId>& extents() const { return extents_; }
  std::uint64_t bytes() const { return bytes_; }
  std::uint64_t read_bytes() const { return read_bytes_; }

  // Once after native teardown fenced every consumer and evicted extents.
  // On refusal, retains the reservation and open files for safe teardown.
  Status Release(providers::VmmProvider& memory);

 private:
  struct Source {
    std::uint64_t address = 0;
    catalog::MemoryClass memory_class = catalog::MemoryClass::kWeights;
    scheduler::PageSource source;
  };
  providers::ReservationId reservation_;
  std::uint64_t bytes_ = 0;
  std::uint64_t read_bytes_ = 0;
  bool registered_ = false;
  std::vector<Ds4PreparedWeightFile> files_;
  std::vector<Ds4AlignedWeightView> views_;
  std::vector<catalog::ExtentId> extents_;
  std::vector<Source> sources_;
};

}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_DSV4_DS4_PAGED_WEIGHTS_H_
