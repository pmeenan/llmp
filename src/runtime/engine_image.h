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
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "kernels/common/lower.h"
#include "runtime/serving.h"

namespace llmp::runtime {

// Source-owned typed stage composition. The adapter executes these graphs
// through the same lower/program lifecycle as token models; no token state or
// family executor is encoded here. A port is stable device storage, copied at
// stage boundaries only after its producer completes.
struct ImagePort {
  format::TensorFormat format;
  std::uint8_t owner = 0;
  std::vector<std::byte> initial;  // exact encoded bytes, or produced by an earlier stage
};
struct ImageStage {
  std::shared_ptr<const components::SpatialGraph> graph;
  // Immutable source/geometry-bound arithmetic facts, independently paid.
  std::shared_ptr<const planner::NumericalQualifications> numerical{};
  std::uint16_t weights = 0;
  std::uint32_t observation = 0;
  // Source role indices are local to this graph, even when component numbers
  // coincide with another stage. The retained owner authenticates this map.
  kernels::Addresses addresses;
  // One entry per graph port. Absent input: host-staged by fill. An output
  // may be observed on the host and/or carried to this device port.
  std::vector<std::optional<std::uint32_t>> inputs, outputs;
  // The graph inputs (by index) the source keeps the same between the
  // stage's invocations of a generation (planner/plan.h
  // PlanOptions::invariant): a run after a run of the stage skips them.
  std::vector<std::uint32_t> invariant{};
};
struct ImageInvocation {
  std::uint32_t stage = 0;
  std::uint32_t iteration = 0;
};
// Separately paid request content. Replacements carry the same source-defined
// layout contract and exact typed ports; they do not replace graphs or kernels.
struct ImageRequest {
  ImageRequest() = default;
  ImageRequest(const ImageRequest&) = delete;
  ImageRequest& operator=(const ImageRequest&) = delete;
  ImageRequest(ImageRequest&&) = default;
  ImageRequest& operator=(ImageRequest&&) = default;
  std::shared_ptr<void> keep;
  base::Sha256Digest identity{}, layout{};
  std::vector<ImagePort> ports;
  std::vector<ImageInvocation> sequence;
  std::uint32_t first_output = 0;  // inclusive invocation index
  std::function<Status(const ImageInvocation&, std::size_t, std::span<std::byte>)> fill;
  // Final conversion is host-side pipeline output formatting, after the last
  // typed F32 output has completed. It cannot submit work or change the plan.
  std::function<std::expected<std::string, std::string>(std::span<const std::byte>)> finish;
};

// A separately paid geometry candidate over the retained model source/backend.
// It owns only graphs, typed ports and request content, never a second model.
struct ImageGeometry {
  std::shared_ptr<void> keep;
  base::Sha256Digest identity{};
  std::vector<ImageStage> stages;
  ImageRequest request;
};

struct EngineImageParts {
  // These source/kernel borrowers share one complete owner. It outlives every
  // graph, lowered program, recording and mapped workspace on all paths.
  std::shared_ptr<void> keep;
  std::string architecture;
  std::shared_ptr<const engine::ExecutionMetadataAllowance> construction;
  std::shared_ptr<void> geometry_keep;
  base::Sha256Digest identity{};
  std::vector<engine::StageWeights> weights;
  std::vector<engine::RunnerConfig::KernelMemory> kernel_memory;
  std::function<Status(std::span<const std::uint64_t>)> prepare;
  std::function<void()> retire;
  std::function<std::expected<std::unique_ptr<kernels::StepLaunch>, std::string>()> launch;
  std::shared_ptr<const kernels::StepBackend> backend;
  planner::Policy policy;
  std::vector<ImageStage> stages;
  ImageRequest request;
  // Paid source-owned request construction, not a second execution path.
  std::function<std::expected<std::variant<ImageRequest, ImageGeometry>, std::string>(
      const ImageInput&, const base::Sha256Digest&)>
      prepare_request;
  bool graphs = true;
};

using EngineImageBuild =
    std::function<std::expected<EngineImageParts, std::string>(const ModelSettings& settings)>;

class EngineImage final : public Image, public engine::PagedModel {
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
  bool has_paged() const override { return state_ != nullptr; }
  Status Setup() override;
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

  std::uint32_t stream() const override { return static_cast<std::uint32_t>(index_); }
  const catalog::Closure& fence_closure() const override;
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;
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
  engine::PagedNode& node_;
  int index_;
  EngineImageBuild build_;
  kernels::PlanRecordSink plan_report_sink_{};
  struct State;
  std::shared_ptr<State> state_;
  std::vector<std::shared_ptr<State>> transitions_;
  Status PrepareState(State& state);
  Status BindState(State& state);
  Status RunThrough(std::size_t end, std::string& sha);
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_ENGINE_IMAGE_H_
