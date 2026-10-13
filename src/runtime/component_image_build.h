// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_RUNTIME_COMPONENT_IMAGE_BUILD_H_
#define LLMP_RUNTIME_COMPONENT_IMAGE_BUILD_H_

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

#include "artifact/composition.h"
#include "engine/execution_contract.h"
#include "runtime/engine_image.h"

namespace llmp::runtime {
// Caller-paid content for one request. No artifacts, kernels or mappings are
// reopened when preparing another request of the same layout.
struct ImageGenerationRequest {
  std::vector<std::int32_t> prompt;
  std::uint32_t drop = 0;
  // Both zero: settings.image_size; otherwise both output dimensions.
  std::uint32_t width = 0, height = 0;
  std::vector<std::uint16_t> noise;  // source BF16 [row,channel] boundary
  std::shared_ptr<void> keep;        // caller's paid prompt/noise owner
  std::filesystem::path output{};    // explicit completed RGBA8 export, empty normally
};
// Initial model source retains the paid artifact owner while its input content
// moves to the independently owned first request. Setup consumes weights once.
struct ImageGenerationModel;
struct ImageGenerationSource : ImageGenerationRequest {
  std::shared_ptr<const artifact::Composition> composition;
  std::array<std::unique_ptr<engine::PagedWeights>, 3> weights;
  std::weak_ptr<ImageGenerationModel> prepared;
  // Ordinary serving may grow its one retained host payment under this node's
  // catalog owner. Null retains the fixed caller-paid diagnostic contract.
  engine::PagedNode* construction_node = nullptr;
  std::uint64_t construction_cap = 0;
};
// Internal serving handoff from paid source preparation. It names the exact
// opened, stopped node that holds the source/construction payment. Server
// snapshots this owner, validates the configured alias/composition, and uses
// the same EngineImage factory as every other caller. No raw parser or runner.
struct PreparedImageGeneration {
  std::string alias;
  engine::PagedNode* node = nullptr;
  int owner = -1;
  std::shared_ptr<ImageGenerationSource> source;
  std::shared_ptr<engine::ExecutionMetadataAllowance> construction;
};
// Pays generated request data before allocation. EngineImage verifies its layout
// against the retained graphs/ports before accepting a completed-boundary change.
std::expected<ImageRequest, std::string> ComponentImageGenerationRequest(
    ImageGenerationRequest&& request, const ModelSettings& settings,
    std::shared_ptr<engine::ExecutionMetadataAllowance> construction,
    engine::PagedNode* construction_node = nullptr, std::uint64_t construction_cap = 0);
// Builds only paid geometry/request owners against the existing bound source.
// EngineImage publishes this at a completed boundary after measured admission.
std::expected<ImageGeometry, std::string> ComponentImageGenerationGeometry(
    const std::shared_ptr<ImageGenerationSource>& source, ImageGenerationRequest&& request,
    const ModelSettings& settings,
    std::shared_ptr<engine::ExecutionMetadataAllowance> construction);
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
// Existing installed readers and native tokenizer, under the Server's parser
// loan. Keeps one model/source allowance and source-owned request preparation.
EngineImageBuild ComponentImageServingBuild(engine::PagedNode& node, std::filesystem::path store,
                                            std::string composition_id, int owner,
                                            std::string prompt, std::filesystem::path noise,
                                            std::uint64_t parser_loan, std::uint64_t cap);
}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_IMAGE_BUILD_H_
