// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_RUNTIME_COMPONENT_IMAGE_BUILD_H_
#define LLMP_RUNTIME_COMPONENT_IMAGE_BUILD_H_

#include <array>
#include <memory>
#include <vector>

#include "artifact/composition.h"
#include "engine/execution_contract.h"
#include "runtime/engine_image.h"

namespace llmp::runtime {
// Immutable request input owner. Its caller pays artifact parsing, native
// tokenizer/rendering and noise acquisition before constructing these values.
// The factory checks the exact composition and role inventory again, then
// prepays every generated mapping/graph/compiler owner through construction.
// The build consumes this owner once; request replacement is a separate lifecycle.
// This is not serving admission; the caller retains the entire node/provider
// environment after any unknown completion or unsuccessful first teardown.
struct ImageGenerationSource {
  std::shared_ptr<const artifact::Composition> composition;
  std::array<std::unique_ptr<engine::PagedWeights>, 3> weights;
  std::vector<std::int32_t> prompt;
  std::uint32_t drop = 0;
  // Both zero: use settings.image_size for a square request. Otherwise both
  // dimensions are explicit output pixels; source model divisibility applies.
  std::uint32_t width = 0, height = 0;
  // The prepared initial noise's BF16 boundary, in source [row,channel] order.
  std::vector<std::uint16_t> noise;
  std::shared_ptr<void> keep;
};
// One request-shaped generation using settings.image_steps and source geometry.
// Takes already-open, paid artifact owners; no checkpoint payload copies or
// residency admission happen here. EngineImage measures all stage envelopes
// before Start and holds only each executing stage's current weight closure.
// Optional internal tuning owns its names and is paid by the caller before
// construction. Only its tuning tuples are forwarded; provider device facts
// remain authoritative and an absent record preserves default selection.
EngineImageBuild ComponentImageGenerationBuild(
    engine::PagedNode& node, int owner, std::shared_ptr<ImageGenerationSource> source,
    std::shared_ptr<engine::ExecutionMetadataAllowance> construction,
    std::shared_ptr<const planner::OwnedDeviceFacts> tuning = {});
}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_IMAGE_BUILD_H_
