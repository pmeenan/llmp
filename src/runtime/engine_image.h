// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One pipeline-kind adapter over shared typed graph/program mechanics.
// Phase composition is source-owned data; no token state, kernel dispatcher,
// or per-family execution loop is introduced here.

#ifndef LLMP_RUNTIME_ENGINE_IMAGE_H_
#define LLMP_RUNTIME_ENGINE_IMAGE_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "components/spatial.h"
#include "config/node_config.h"
#include "engine/paged_node.h"
#include "engine/staged_execution.h"
#include "kernels/common/lower.h"
#include "runtime/serving.h"

namespace llmp::runtime {

// Source construction and request parsing remain runtime concerns. Execution
// descriptors belong to the engine and contain no filesystem or serving types.
// Validate the exact paid source representation before parsing or copying it.
std::expected<std::span<const std::byte>, std::string> CheckedImageNoise(const ImageNoise& noise,
                                                                         std::uint64_t elements);

using ImagePort = engine::StagedPort;
using ImageStage = engine::StagedProgram;
using ImageInvocation = engine::StagedInvocation;
using ImageRequest = engine::StagedRequest;
using ImageGeometry = engine::StagedGeometry;
struct EngineImageParts : engine::StagedExecutionParts {
  std::function<std::expected<std::variant<ImageRequest, ImageGeometry>, std::string>(
      const ImageInput&, const base::Sha256Digest&)>
      prepare_request;
};

using EngineImageBuild =
    std::function<std::expected<EngineImageParts, std::string>(const ModelSettings& settings)>;

class EngineImage final : public Image {
 public:
  // Exact hidden adapter/control objects for startup/current-candidate admission.
  static std::uint64_t ControlMetadataBytes();
  EngineImage(engine::PagedNode& node, const config::ModelEntry& entry,
              const ModelSettings& settings, int index, EngineImageBuild build);
  EngineImage(const EngineImage&) = delete;
  EngineImage& operator=(const EngineImage&) = delete;
  EngineImage(EngineImage&&) = delete;
  EngineImage& operator=(EngineImage&&) = delete;
  ~EngineImage() override;

  engine::PagedModel& paged() override;
  bool has_paged() const override { return execution_ && execution_->has_state(); }
  Status Setup() override;
  void set_log(std::function<void(std::string_view)> log) override;
  Status Register() override;
  Status Bind() override;
  std::uint64_t activations_needed() const override;
  std::uint64_t pool_needed() const override;
  std::uint64_t host_input_bytes() const override;
  std::uint64_t startup_host_bytes() const override;
  std::uint64_t plan_floor_bytes() const override;
  std::string plan_report() const override;
  void set_plan_report_sink(kernels::PlanRecordSink sink) override;
  void ReportPlans(const kernels::PlanRecordSink& sink) const override;
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override;
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override;
  std::vector<catalog::ExtentId> weights() const override;
  std::vector<catalog::ExtentId> state() const override;
  const catalog::Closure& everything() const override;
  std::uint64_t weight_read_bytes() const override;
  Status AfterLoad() override;
  Status CheckPlaces() override;
  void DropPlans() override;
  std::string extra() const override;

  std::uint32_t stream() const { return static_cast<std::uint32_t>(index_); }
  const catalog::Closure& fence_closure() const;
  std::vector<catalog::ExtentId> managed_extents() const;
  Status Release();
  const catalog::Closure& request_closure() const override { return fence_closure(); }

  // Before the first launch or after completed Finish, without a request lease.
  // Content replacement preserves maps; geometry uses measured admission.
  Status PrepareRequest(const ImageInput& input) override;
  Status ReplaceRequest(ImageRequest request);
  Status ReplaceGeometry(ImageGeometry geometry);
  Status FirstOutput(std::string& sha) override;
  Status Finish(std::string& sha) override;
  Status Abandon() override;

 private:
  EngineImageBuild build_;
  kernels::PlanRecordSink plan_report_sink_;
  engine::PagedNode& node_;
  int index_;
  engine::StagedExecutionOwner execution_;
  std::function<std::expected<std::variant<ImageRequest, ImageGeometry>, std::string>(
      const ImageInput&, const base::Sha256Digest&)>
      prepare_request_;
  base::Sha256Digest layout_{};
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_ENGINE_IMAGE_H_
