// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_ENGINE_STAGED_EXECUTION_H_
#define LLMP_ENGINE_STAGED_EXECUTION_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/spatial.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "kernels/common/lower.h"

namespace llmp::engine {
// Stateless typed stages use the same plan cache, ProgramLifecycle, StepRuns
// and RunnerMemory as token execution, without autoregressive state slots.
struct StagedPort {
  format::TensorFormat format;
  std::uint8_t owner = 0;
  std::vector<std::byte> initial;  // exact encoded bytes, or produced by an earlier stage
};
struct StagedProgram {
  std::shared_ptr<const components::SpatialGraph> graph;
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
struct StagedInvocation {
  std::uint32_t stage = 0;
  std::uint32_t iteration = 0;
};
// Separately paid request content. Replacements carry the same source-defined
// layout contract and exact typed ports; they do not replace graphs or kernels.
struct StagedRequest {
  StagedRequest() = default;
  StagedRequest(const StagedRequest&) = delete;
  StagedRequest& operator=(const StagedRequest&) = delete;
  StagedRequest(StagedRequest&&) = default;
  StagedRequest& operator=(StagedRequest&&) = default;
  std::shared_ptr<void> keep;
  base::Sha256Digest identity{}, layout{};
  std::vector<StagedPort> ports;
  std::vector<StagedInvocation> sequence;
  std::uint32_t first_output = 0;  // inclusive invocation index
  std::function<Status(const StagedInvocation&, std::size_t, std::span<std::byte>)> fill;
  // Final conversion is host-side pipeline output formatting, after the last
  // typed F32 output has completed. It cannot submit work or change the plan.
  std::function<std::expected<std::string, std::string>(std::span<const std::byte>)> finish;
};

// A separately paid geometry candidate over the retained model source/backend.
// It owns only graphs, typed ports and request content, never a second model.
struct StagedGeometry {
  std::shared_ptr<void> keep;
  base::Sha256Digest identity{};
  std::vector<StagedProgram> stages;
  StagedRequest request;
};

struct StagedExecutionParts {
  // These source/kernel borrowers share one complete owner. It outlives every
  // graph, lowered program, recording and mapped workspace on all paths.
  std::shared_ptr<void> keep;
  std::string architecture;
  std::function<void(std::string_view)> log;
  std::shared_ptr<const ExecutionMetadataAllowance> construction;
  std::shared_ptr<void> geometry_keep;
  base::Sha256Digest identity{};
  std::vector<StageWeights> weights;
  std::vector<RunnerConfig::KernelMemory> kernel_memory;
  std::function<Status(std::span<const std::uint64_t>)> prepare;
  std::function<void()> retire;
  std::function<std::expected<std::unique_ptr<kernels::StepLaunch>, std::string>()> launch;
  std::shared_ptr<const kernels::StepBackend> backend;
  planner::Policy policy;
  std::vector<StagedProgram> stages;
  StagedRequest request;
  bool graphs = true;
};

class StagedExecution;
// A frontend's view of a complete paid node-owned executor. The node retains
// its own reference through positive fences and physical release, including
// when an adapter or request callback drops its last reference after failure.
class StagedExecutionOwner {
 public:
  StagedExecutionOwner() = default;
  explicit operator bool() const { return static_cast<bool>(keep_); }
  StagedExecution* operator->() const { return execution_; }
  StagedExecution& operator*() const { return *execution_; }

 private:
  friend class StagedExecution;
  StagedExecutionOwner(kernels::ResourceKeep keep, StagedExecution* execution)
      : keep_(std::move(keep)), execution_(execution) {}
  kernels::ResourceKeep keep_;
  StagedExecution* execution_ = nullptr;
};

class StagedExecution final : public PagedModel {
 public:
  static std::expected<StagedExecutionOwner, std::string> Create(PagedNode& node, int owner);
  StagedExecution(const StagedExecution&) = delete;
  StagedExecution& operator=(const StagedExecution&) = delete;
  ~StagedExecution() override;
  bool has_state() const { return state_ != nullptr; }
  Status Setup(StagedExecutionParts parts);
  void SetLog(std::function<void(std::string_view)> log);
  // Request callbacks must only run at an idle initial/completed boundary.
  Status CheckRequestBoundary() const;
  Status Register();
  Status Bind();
  std::uint64_t activations_needed() const;
  std::uint64_t pool_needed() const;
  std::uint64_t host_input_bytes() const;
  std::uint64_t startup_host_bytes() const;
  std::uint64_t plan_floor_bytes() const;
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  const catalog::Closure& everything() const;
  std::uint64_t weight_read_bytes() const;
  Status AfterLoad();
  Status CheckPlaces();
  void DropPlans();
  std::string extra() const;
  std::string plan_report() const;
  void SetPlanReportSink(kernels::PlanRecordSink sink);
  void ReportPlans(const kernels::PlanRecordSink& sink) const;
  static std::uint64_t ControlMetadataBytes();

  std::uint32_t stream() const override { return static_cast<std::uint32_t>(index_); }
  const catalog::Closure& fence_closure() const override;
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;
  Status AcknowledgeFence() override;
  bool unproven() const noexcept;

  // Before the first launch or after completed Finish, without a request lease.
  // Content replacement preserves maps; geometry uses measured admission.
  Status ReplaceRequest(StagedRequest request);
  Status ReplaceGeometry(StagedGeometry geometry);
  Status FirstOutput(std::string& sha);
  Status Finish(std::string& sha);
  Status Abandon();

 private:
  StagedExecution(PagedNode& node, int owner);
  struct Lifetime;
  Lifetime* lifetime_ = nullptr;
  bool drained_ = false, release_failed_ = false;
  kernels::PlanRecordSink plan_report_sink_;
  Status RetireAfterFence() override;
  void RetireRoot();
  PagedNode& node_;
  int index_;
  struct State;
  std::shared_ptr<State> state_;
  std::vector<std::shared_ptr<State>> transitions_;
  Status PrepareState(State& state);
  Status BindState(State& state);
  Status RunThrough(std::size_t end, std::string& sha);
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_STAGED_EXECUTION_H_
