// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The step source of a model made of components (D-107;
// docs/engine-components.md; engine/model_runner.h StepSource): for each
// step shape, the stage's primitive step from its spec (components/step.h
// BuildStep), planned over a kernel source's registry for its device under
// the alias's policy (planner/plan.h), and lowered by that source into a
// step program (kernels/common/lower.h). The kernel source is a table of
// functions: the engine names no kernel library.
//
// Its policy is the architecture mapping's default for the source
// (spec/mapping.h DefaultPolicy: the family's recipe as data), unless the
// caller gives one (an alias's settings, later). Its plans' key is the
// spec's bound digest, the policy's patterns and the policy (planner/
// plan.h KeyOf); the runner adds the shape.
//
// Part of the engine core: every build profile.

#ifndef LLMP_ENGINE_COMPONENT_SOURCE_H_
#define LLMP_ENGINE_COMPONENT_SOURCE_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>

#include "base/sha256.h"
#include "components/step.h"
#include "engine/execution_contract.h"
#include "engine/model_runner.h"
#include "kernels/common/lower.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::engine {

// One catalog-backed transient compiler ceiling shared by sources on the
// serving driver. Retained source metadata and installed plans pay separately.
// Foreign-thread/reentrant builds cannot borrow uncharged construction room.
class SharedCompilerAllowance {
 public:
  SharedCompilerAllowance() = default;
  SharedCompilerAllowance(const SharedCompilerAllowance&) = delete;
  SharedCompilerAllowance& operator=(const SharedCompilerAllowance&) = delete;
  class Use {
   public:
    Use(Use&& other) noexcept;
    Use& operator=(Use&&) = delete;
    Use(const Use&) = delete;
    Use& operator=(const Use&) = delete;
    ~Use();

   private:
    friend class SharedCompilerAllowance;
    Use(std::shared_ptr<ExecutionMetadataAllowance> owner, std::uint64_t bytes)
        : owner_(std::move(owner)), bytes_(bytes), thread_(std::this_thread::get_id()) {}
    std::shared_ptr<ExecutionMetadataAllowance> owner_;
    std::uint64_t bytes_;
    std::thread::id thread_;
  };
  // Startup only, under the stopped node's catalog owner. Total includes the
  // metadata owner overhead and caller's page rounding. Grow the same extent.
  Status Reserve(catalog::Catalog& catalog, catalog::DomainId domain, std::uint64_t total,
                 std::uint64_t cap, std::uint64_t available, std::uint64_t margin);
  std::expected<Use, std::string> Acquire(std::uint64_t bytes);
  std::uint64_t paid_bytes() const {
    return owner_ ? owner_->capacity() + ExecutionMetadataAllowance::OverheadBytes() : 0;
  }

 private:
  const std::thread::id thread_ = std::this_thread::get_id();
  std::shared_ptr<ExecutionMetadataAllowance> owner_;
};

struct ComponentSourceOptions {
  // The model's spec and the stage this source plans; both outlive it, as
  // do the bindings' tensors and layout.
  const spec::ModelSpec* spec = nullptr;
  std::uint16_t stage = 0;
  components::Bindings bindings{};
  // The spec bound to its artifacts (spec.h BoundDigest).
  base::Sha256Digest bound_digest{};
  // The model's tokenizer vocabulary where its head is wider (RE-050): the
  // stage's greedy choices read no row past it (components::StepOptions::
  // vocabulary), and its plans' key and execution identity name it. 0: the
  // head's own rows.
  std::uint32_t vocabulary = 0;
  // The architecture's mapping, whose default policy for the source's
  // `values` (spec/mapping.h DefaultPolicy; read only by Create) plans the
  // steps, and a policy that replaces it (an alias's); one of the two.
  const spec::Mapping* mapping = nullptr;
  spec::PolicyContext policy_context{};
  std::span<const spec::SourceValue> values{};
  // The owners the runner serves: with the layout's context, the serving
  // envelope the mapping's rule is chosen for (spec::Envelope; 0: unknown,
  // so no envelope-bounded rule).
  std::uint32_t slots = 0;
  std::optional<planner::Policy> policy{};
  kernels::StepBackend backend{};
  // Required whole composition; every staged source names the same immutable
  // domain. Requires a real backend owner. Joined pending replay additionally
  // requires its explicit use and complete compatible composition layouts.
  std::shared_ptr<const NumericalDomain> execution_domain{};
  // Require graph-capture-safe implementations. Eager-capable callers may
  // opt out; each actual program still reports whether it can be captured.
  bool require_capture = true;
  // Diagnostic validation only, after the real backend lowers both measuring
  // and bound plans. No callback means the ordinary path is unchanged.
  kernels::ProgramValidator validate_program{};
  // Nonzero and owned together. Charged before any BuildStep/graph allocation
  // and retained with the produced program until measurement destruction or
  // transfer into an independently paid live plan/future.
  std::shared_ptr<SharedCompilerAllowance> compiler{};
  std::uint64_t compiler_bytes = 0;
};

class ComponentStepSource final : public StepSource {
 public:
  // Refused for a missing spec, layout or backend, no policy (given or the
  // mapping's), or one its own rules refuse.
  static std::expected<std::unique_ptr<ComponentStepSource>, std::string> Create(
      ComponentSourceOptions options);

  planner::PlanKey KeyBase() const override;
  Status PrepareExecution() override;
  Status CheckExecutionUse(NumericalUse use) const override;
  Status CheckExecutionShape(const planner::StepShape& shape) const override;
  std::expected<std::optional<base::Sha256Digest>, std::string> ExecutionIdentity() const override;
  void EmitShapeParameters(const planner::StepShape& shape,
                           model::ShapeParameterSink& out) const override;
  std::expected<SpecializationChange, SpecializationError> NextSpecializationChange(
      const SpecializationRequest& request) const override;
  StageExecutionFacts StageFacts(std::uint16_t stage) const override;
  bool UsesSideRows(std::uint16_t stage) const override;
  Status SideRows(std::uint16_t stage, std::uint16_t component,
                  std::span<const std::int32_t> history, std::uint32_t position,
                  std::span<std::int32_t> out) const override;
  std::expected<std::vector<CommitBuffer>, std::string> CommitBuffers(
      std::uint16_t stage, std::uint32_t rows) const override;
  const planner::Policy& policy() const { return policy_; }
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Build(
      const planner::StepShape& shape, const kernels::Addresses& addresses) override;
  std::shared_ptr<const kernels::BackendOwner> ResourceBackend(std::uint16_t stage) const override {
    return stage == o_.stage ? o_.backend.owner : nullptr;
  }
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildPrivate(
      const planner::StepShape& shape, const kernels::Addresses& addresses,
      const kernels::ExecutionResources& resources) override;

  std::uint16_t stage() const { return o_.stage; }
  const base::Sha256Digest& bound_digest() const { return o_.bound_digest; }

 private:
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildWith(
      const planner::StepShape& shape, const kernels::Addresses& addresses,
      const kernels::ExecutionResources* resources);
  ComponentStepSource(ComponentSourceOptions options, planner::Policy policy)
      : o_(std::move(options)), policy_(std::move(policy)) {}
  friend class StagedStepSource;
  struct ExecutionState;
  static std::expected<std::shared_ptr<ExecutionState>, std::string> SealExecution(
      std::span<ComponentStepSource* const> stages);
  std::shared_ptr<ExecutionState> execution_{};
  ComponentSourceOptions o_;
  const kernels::StepBackend& Backend() const {
    return o_.backend.owner ? o_.backend.owner->backend() : o_.backend;
  }
  planner::Policy policy_;
  planner::PlanKey key_base_;
  std::vector<components::Instance> shaped_, side_;
  base::Sha256Digest layout_identity_{}, semantic_identity_{};
  planner::StateSpans StateSpans(const planner::StepShape& shape,
                                 const kernels::Addresses& addresses) const;
};

// The step source of a model of several stages (a target and its
// drafters, D-068): each stage's component source, the one a shape's
// stage names. Its plans share one key base: the spec bound to every
// stage's artifacts (each source's, which must agree), and digests over
// every stage's patterns and policy in stage order, so a plan of any stage
// is keyed by all of them and its shape (which names its stage).
class StagedStepSource final : public StepSource {
 public:
  // Refused for no source, a source of another stage than its index, or
  // sources of different bound specs.
  static std::expected<std::unique_ptr<StagedStepSource>, std::string> Create(
      std::vector<std::unique_ptr<ComponentStepSource>> stages);

  planner::PlanKey KeyBase() const override { return key_; }
  Status PrepareExecution() override;
  Status CheckExecutionUse(NumericalUse use) const override;
  Status CheckExecutionShape(const planner::StepShape& shape) const override;
  std::expected<std::optional<base::Sha256Digest>, std::string> ExecutionIdentity() const override;
  void EmitShapeParameters(const planner::StepShape& shape,
                           model::ShapeParameterSink& out) const override;
  std::expected<SpecializationChange, SpecializationError> NextSpecializationChange(
      const SpecializationRequest& request) const override {
    if (!ValidSpecializationRequest(request) || request.shape.stage >= stages_.size())
      return std::unexpected(SpecializationError::kRequest);
    return stages_[request.shape.stage]->NextSpecializationChange(request);
  }
  StageExecutionFacts StageFacts(std::uint16_t stage) const override;
  bool UsesSideRows(std::uint16_t stage) const override;
  Status SideRows(std::uint16_t stage, std::uint16_t component,
                  std::span<const std::int32_t> history, std::uint32_t position,
                  std::span<std::int32_t> out) const override;
  std::expected<std::vector<CommitBuffer>, std::string> CommitBuffers(
      std::uint16_t stage, std::uint32_t rows) const override;
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Build(
      const planner::StepShape& shape, const kernels::Addresses& addresses) override;
  std::shared_ptr<const kernels::BackendOwner> ResourceBackend(std::uint16_t stage) const override {
    return stage < stages_.size() ? stages_[stage]->ResourceBackend(stage) : nullptr;
  }
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildPrivate(
      const planner::StepShape& shape, const kernels::Addresses& addresses,
      const kernels::ExecutionResources& resources) override {
    if (shape.stage >= stages_.size()) return std::unexpected("private step stage is absent");
    return stages_[shape.stage]->BuildPrivate(shape, addresses, resources);
  }
  const ComponentStepSource& stage(std::size_t index) const { return *stages_.at(index); }

 private:
  StagedStepSource(std::vector<std::unique_ptr<ComponentStepSource>> stages, planner::PlanKey key)
      : stages_(std::move(stages)), key_(std::move(key)) {}
  std::vector<std::unique_ptr<ComponentStepSource>> stages_;
  planner::PlanKey key_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_COMPONENT_SOURCE_H_
