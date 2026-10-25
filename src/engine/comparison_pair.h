// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_COMPARISON_PAIR_H_
#define LLMP_ENGINE_COMPARISON_PAIR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "components/comparison_boundary.h"
#include "components/step.h"
#include "execution/registry.h"
#include "ir/storage_admission.h"
#include "kernels/common/comparison.h"
#include "kernels/common/comparison_region.h"
#include "kernels/common/execution_resources.h"
#include "kernels/common/lower.h"
#include "planner/alternative.h"
#include "planner/step_shape.h"

namespace llmp::engine {

// Only this typed, stateless recording protocol is admitted for local recipe
// comparisons. The digest is shared with measurement-row construction; callers
// cannot supply a free-form protocol label or preparation hash.
enum class ComparisonRecipePreparation : std::uint8_t {
  kFullStatelessRestagedInputsPurePrefixCapturedFullBody = 1,
  // A distinct typed protocol for stateless ordinary-input replays whose
  // backend-proved pure prefix produces ready boundary operands before the
  // full local body is captured. Eligibility comes from both backend proofs,
  // not from a caller-provided source-graph grammar or label.
  kStatelessReadyBoundaryOperands = 2,
  // Complete ordinary program, one invocation per independently restored
  // full state/save image and canonically restaged original inputs.
  kWholeProgramRestoredStateSingleInvocation = 3,
  // Same complete reset/staging, one ordinary Run. Derived from an immutable
  // ordinary-only selected declaration in at least one arm, never a label.
  kWholeProgramRestoredStateOrdinaryInvocation = 4,
};

std::optional<base::Sha256Digest> ComparisonPreparationDomain(
    ComparisonRecipePreparation preparation, execution::Phase phase) noexcept;

// A separate complete-program domain. The caller retains the authenticated
// checkpoint/own alternative and verifies each program's live plan identity.
// The accepted site is never inferred from an arbitrary boundary group.
std::optional<execution::RecipeComparisonKey> RecipeComparisonKeyForWholePrograms(
    const components::OwnedComparisonBoundary& full,
    const components::OwnedComparisonRegionBoundary& site_boundary,
    const planner::ComparisonSite& site, execution::Phase phase,
    const base::Sha256Digest& selection_environment, const kernels::ComparisonRecipeSeal& incumbent,
    const kernels::ComparisonRecipeSeal& alternative) noexcept;

// Persistent equality through DescriptorVerified -> BoundVerified promotion.
bool SameWholeProgramRecipe(const kernels::ComparisonRecipeSeal& descriptor,
                            const kernels::ComparisonRecipeSeal& bound) noexcept;

// Shared exact-key constructor for a pair of actual ready-boundary
// applicability seals. Callers must obtain both seals from the same retained
// programs and authenticate their source identities before using this value.
std::optional<execution::RecipeComparisonKey> RecipeComparisonKeyForReadyCertificates(
    const components::OwnedComparisonRegionBoundary& boundary, ir::OpKey discovery_operation,
    execution::Phase phase, const base::Sha256Digest& selection_environment,
    const kernels::ComparisonRegionRecipeSeal& incumbent,
    const kernels::ComparisonRegionRecipeSeal& alternative) noexcept;

enum class ComparisonPairArm : std::uint8_t { kIncumbent, kAlternative };

struct CapturedRegionInventory {
  std::size_t recorded_nodes = 0;
  std::uint32_t repetitions = 0;
};

// Chooses how the retained StepGraph is handed to the existing shared lowerer.
// Token-step inputs preserve the normal runner contract; standalone component
// graphs use their typed GraphBuildIO ports without a token phase.
enum class ComparisonInputDomain : std::uint8_t { kTokenStep, kComponentPorts };

struct ComparisonPairFailure {
  enum class Code : std::uint8_t {
    kAdmission,
    kInvalid,
    kNotReady,
    kUnproved,
    kStale,
    kNoProgress,
  } code = Code::kInvalid;
  std::string_view detail{};
};

enum class ComparisonExpansionMerge : std::uint8_t {
  kProgress,
  kNoProgress,
  kOutOfRange,
  kCapacity,
};

// A paid range vector for one completed plan. Its values are synchronous
// physical facts for the same retained source graph; it contains no plan or
// graph pointers and can be handed to the existing pair binding API.
class ComparisonResolvedBindings {
 public:
  ComparisonResolvedBindings() = default;
  ComparisonResolvedBindings(ComparisonResolvedBindings&&) noexcept = default;
  ComparisonResolvedBindings& operator=(ComparisonResolvedBindings&&) noexcept = delete;
  ComparisonResolvedBindings(const ComparisonResolvedBindings&) = delete;
  ComparisonResolvedBindings& operator=(const ComparisonResolvedBindings&) = delete;

  std::span<const kernels::ComparisonBindingRange> ranges() const { return ranges_; }

 private:
  friend class ComparisonSourceBindings;
  ir::StorageLoan loan_{};
  std::vector<kernels::ComparisonBindingRange> ranges_;
};

// Paid, source-scoped external declarations shared by cumulative plans. The
// caller retains the original graph, IO spans, and optional StepGraph metadata
// for this transaction. Create records their complete typed IO/state inventory; each Resolve
// authenticates one exact finished plan and produces a separately paid physical binding vector. The
// first successful Resolve fixes the canonical external declarations and
// later plans must preserve them.
class ComparisonSourceBindings {
 public:
  static std::expected<ComparisonSourceBindings, ComparisonPairFailure> Create(
      std::shared_ptr<const ir::StorageFunder> funder, const components::StepGraph& source,
      std::uint16_t active_stage);
  // Ordinary component ports carry no token, generated-input or state facts.
  static std::expected<ComparisonSourceBindings, ComparisonPairFailure> Create(
      std::shared_ptr<const ir::StorageFunder> funder, const components::GraphBuildIO& source,
      std::uint16_t active_stage);

  ComparisonSourceBindings(ComparisonSourceBindings&&) noexcept = default;
  ComparisonSourceBindings& operator=(ComparisonSourceBindings&&) noexcept = delete;
  ComparisonSourceBindings(const ComparisonSourceBindings&) = delete;
  ComparisonSourceBindings& operator=(const ComparisonSourceBindings&) = delete;

  std::expected<ComparisonResolvedBindings, ComparisonPairFailure> Resolve(
      const planner::StepPlan& plan, const kernels::Addresses& addresses,
      const planner::RootBytesOf& written = {});
  std::optional<components::ComparisonBoundaryView> boundary_view() const;

 private:
  ComparisonSourceBindings(ir::StorageLoan loan, std::shared_ptr<const ir::StorageFunder> funder,
                           const components::GraphBuildIO& io, const components::StepGraph* source,
                           std::uint16_t active_stage)
      : loan_(std::move(loan)),
        funder_(std::move(funder)),
        io_(io),
        source_(source),
        active_stage_(active_stage) {}
  static std::expected<ComparisonSourceBindings, ComparisonPairFailure> CreateCore(
      std::shared_ptr<const ir::StorageFunder> funder, const components::GraphBuildIO& io,
      const components::StepGraph* source, std::uint16_t active_stage);

  ir::StorageLoan loan_{};
  std::shared_ptr<const ir::StorageFunder> funder_{};
  components::GraphBuildIO io_;  // graph and spans borrowed for the transaction lifetime
  const components::StepGraph* source_ = nullptr;  // optional exact token metadata
  std::uint16_t active_stage_ = 0;
  std::uint64_t source_identity_ = 0;
  base::Sha256Digest source_digest_{};
  base::Sha256Digest metadata_digest_{};
  std::vector<components::ComparisonExternalBinding> external_;
  std::vector<components::ComparisonUnstoredInput> unstored_;
  std::size_t external_count_ = 0;
  bool resolved_ = false;
};

// Allocation-free union used by closure. `membership` is indexed only by
// authenticated original source NodeIds; `additions` already has source-node
// capacity paid by the enclosing package.
ComparisonExpansionMerge MergeComparisonExpansions(std::span<const ir::NodeId> incumbent,
                                                   std::span<const ir::NodeId> alternative,
                                                   std::span<std::uint8_t> membership,
                                                   std::vector<ir::NodeId>& additions);

// The explicit caller creates this paid owner before planning or lowering,
// then installs the two actual plans and bound programs. It owns the original
// semantic source and never stores a LowerInput or a caller's borrowed spans.
class ComparisonPair final {
 public:
  static std::expected<std::unique_ptr<ComparisonPair>, ComparisonPairFailure> Create(
      std::shared_ptr<const ir::StorageFunder> funder, components::StepGraph&& source,
      planner::StepShape&& shape, std::vector<components::ComparisonExternalBinding>&& external,
      std::uint32_t context, std::uint16_t active_stage = 0,
      ComparisonInputDomain input_domain = ComparisonInputDomain::kTokenStep);

  ComparisonPair(const ComparisonPair&) = delete;
  ComparisonPair& operator=(const ComparisonPair&) = delete;
  ComparisonPair(ComparisonPair&&) = delete;
  ComparisonPair& operator=(ComparisonPair&&) = delete;
  ~ComparisonPair() = default;

  std::expected<void, ComparisonPairFailure> Plan(const planner::ComparisonSiteSelector& selector,
                                                  const execution::Registry& registry,
                                                  const planner::DeviceFacts& device,
                                                  const planner::Policy& policy,
                                                  const planner::PlanOptions& options = {});

  // The returned input and its pointers are valid only for the caller's
  // synchronous backend Lower call. The pair owns every referenced object.
  std::expected<kernels::LowerInput, ComparisonPairFailure> LoweringInput(
      ComparisonPairArm arm, const kernels::Addresses& addresses);

  // Install actual arm-specific backing after placement and allocation. The
  // input spans are borrowed synchronously; their range facts are copied only
  // after the package owner has grown its admission loan.
  std::expected<void, ComparisonPairFailure> InstallBindings(
      std::array<std::span<const kernels::ComparisonBindingRange>, 2> bindings);

  // Programs are the actual outputs of lowering and must already be bound to
  // their admitted launch environment. Launch/resource ownership remains in
  // the concrete enclosing fixture or the actual ExecutionResources owner.
  std::expected<void, ComparisonPairFailure> InstallPrograms(
      std::unique_ptr<kernels::StepProgram> incumbent,
      std::unique_ptr<kernels::StepProgram> alternative);

  // Installs the same actual lowered owners before Bind. Descriptor projection
  // is a selection input only; it grants no execution authority.
  std::expected<void, ComparisonPairFailure> InstallDescriptorPrograms(
      std::unique_ptr<kernels::StepProgram> incumbent,
      std::unique_ptr<kernels::StepProgram> alternative);

  // Binds the exact already-installed descriptor programs for measurement.
  // Both launches must be the admitted launches retained by their programs.
  // Any attempt is terminal; a partial bind is never undone or retried.
  std::expected<void, ComparisonPairFailure> BindPrograms(kernels::StepLaunch& incumbent_launch,
                                                          kernels::StepLaunch& alternative_launch);

  const components::StepGraph& source() const { return source_; }
  const planner::StepShape& shape() const { return shape_; }
  const planner::ComparisonInventory* inventory() const {
    return inventory_ ? &*inventory_ : nullptr;
  }
  const planner::OwnedComparisonPlan* plan(ComparisonPairArm arm) const;

  // Consumes the prepared package. Failure occurs before submission and
  // destroys it; success keeps both arms and their projections together in
  // one heap-stable owner.
  friend std::expected<std::unique_ptr<ComparisonPair>, ComparisonPairFailure> CloseComparisonPair(
      std::unique_ptr<ComparisonPair> pair);
  friend std::expected<std::unique_ptr<kernels::StepProgram>, ComparisonPairFailure>
  SelectRecipeComparison(std::unique_ptr<ComparisonPair> pair, const execution::Registry& registry,
                         ComparisonRecipePreparation preparation);
  friend std::expected<std::optional<execution::RecipeComparisonKey>, ComparisonPairFailure>
  DescribeRecipeComparison(ComparisonPair& pair, const execution::Registry& registry);
  friend std::expected<std::optional<execution::RecipeComparisonKey>, ComparisonPairFailure>
  DescribeRecipeComparison(ComparisonPair& pair, const execution::Registry& registry,
                           ComparisonRecipePreparation preparation);

  std::expected<void, kernels::StepFailure> CaptureRegion(ComparisonPairArm arm,
                                                          kernels::StepLaunch& launch,
                                                          providers::NativeStream stream,
                                                          std::uint32_t repetitions = 1);
  // Runs only the authenticated pure prefix for a new frame. The paired
  // captured body remains separately replayable for body-only timing.
  std::expected<void, kernels::StepFailure> PrepareFrame(ComparisonPairArm arm,
                                                         kernels::StepLaunch& launch,
                                                         providers::NativeStream stream) const;
  std::expected<void, kernels::StepFailure> ReplayCapturedRegion(
      ComparisonPairArm arm, kernels::StepLaunch& launch, providers::NativeStream stream) const;
  std::expected<void, kernels::StepFailure> RunFrame(ComparisonPairArm arm,
                                                     kernels::StepLaunch& launch,
                                                     providers::NativeStream stream) const;
  std::optional<CapturedRegionInventory> captured_region_inventory(
      ComparisonPairArm arm, const kernels::StepLaunch& launch) const;
  std::expected<double, kernels::StepFailure> captured_region_duration_ns(
      ComparisonPairArm arm, const kernels::StepLaunch& launch) const;
  std::span<const kernels::InputSlot> input_slots(ComparisonPairArm arm) const;
  std::span<const kernels::ComparisonRegionValuePort> value_ports(ComparisonPairArm arm) const;
  bool closed() const { return closed_; }

 private:
  class SelectedRecipeProgram;

  struct Arm {
    kernels::ExecutionResources resources{};
    std::unique_ptr<kernels::StepProgram> program;
    std::optional<kernels::ComparisonRegionProjection> projection;
    std::span<const kernels::ComparisonRegionValuePort> ports{};
    std::unique_ptr<kernels::Recorded> recording;
    std::uint32_t recording_repetitions = 0;
  };

  ComparisonPair(ir::StorageLoan package_loan, std::shared_ptr<const ir::StorageFunder> funder,
                 components::StepGraph&& source, planner::StepShape&& shape,
                 std::vector<components::ComparisonExternalBinding>&& external,
                 std::uint32_t context, std::uint16_t active_stage,
                 ComparisonInputDomain input_domain);

  std::expected<void, ComparisonPairFailure> BuildSeedBoundary();
  std::expected<void, ComparisonPairFailure> ValidatePlans();
  std::expected<void, ComparisonPairFailure> BuildCandidateBoundary();
  components::ComparisonBoundaryView SourceBoundaryView() const;
  kernels::ComparisonRequest& Request(ComparisonPairArm arm);
  kernels::ComparisonRegionCapture& Capture(ComparisonPairArm arm);
  Arm& GetArm(ComparisonPairArm arm);
  const Arm& GetArm(ComparisonPairArm arm) const;

  // Declaration order is ownership order. Pair storage is refunded last;
  // source and plans outlive programs, and projections/recordings outlive
  // nothing they name because Arm members are destroyed in reverse order.
  ir::StorageLoan package_loan_;
  std::shared_ptr<const ir::StorageFunder> funder_;
  components::StepGraph source_;
  planner::StepShape shape_;
  std::vector<components::ComparisonExternalBinding> external_;
  std::array<std::vector<kernels::ComparisonBindingRange>, 2> bindings_;
  std::uint32_t context_ = 0;
  std::uint16_t active_stage_ = 0;
  ComparisonInputDomain input_domain_ = ComparisonInputDomain::kTokenStep;
  bool plan_attempted_ = false;
  bool lowering_started_ = false;
  bool bindings_installed_ = false;
  bool descriptor_programs_installed_ = false;
  bool measurement_bind_attempted_ = false;
  bool measurement_bind_succeeded_ = false;
  base::Sha256Digest registry_digest_{};
  std::optional<execution::RecipeComparisonKey> descriptor_key_;
  ComparisonRecipePreparation descriptor_preparation_ =
      ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody;
  std::optional<planner::ComparisonInventory> inventory_;
  std::optional<planner::OwnedComparisonPlan> alternative_plan_;
  std::optional<components::OwnedComparisonRegionBoundary> seed_boundary_;
  std::vector<std::uint32_t> candidate_offsets_;
  std::vector<ir::NodeId> candidate_members_;
  std::vector<ir::NodeId> additions_;
  std::vector<std::uint8_t> membership_;
  std::optional<components::OwnedComparisonRegionBoundary> boundary_;
  std::array<std::optional<kernels::ComparisonRegionRecipeSeal>, 2> descriptor_seals_{};
  std::array<std::optional<kernels::ComparisonRegionRecipeSeal>, 2> descriptor_certificate_seals_{};
  std::array<std::optional<kernels::ComparisonRegionRecipeSeal>, 2> bound_certificate_seals_{};
  std::array<kernels::ComparisonRequest, 2> requests_{};
  std::array<kernels::ComparisonRegionCapture, 2> captures_{};
  Arm incumbent_;
  Arm alternative_;
  bool closed_ = false;
};

std::expected<std::unique_ptr<ComparisonPair>, ComparisonPairFailure> CloseComparisonPair(
    std::unique_ptr<ComparisonPair> pair);

// Consumes a descriptor-ready pair. A miss or unsupported descriptor returns
// the exact ordinary incumbent; an exact measured hit returns a retained,
// bind-revalidated owner for the selected actual lowered program.
std::expected<std::unique_ptr<kernels::StepProgram>, ComparisonPairFailure> SelectRecipeComparison(
    std::unique_ptr<ComparisonPair> pair, const execution::Registry& registry,
    ComparisonRecipePreparation preparation =
        ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody);

// Nonconsuming authoritative key description for the current actual pair.
// Unsupported local proofs yield an empty optional; stale ownership and
// admission failures remain typed errors. Selection calls this same producer.
std::expected<std::optional<execution::RecipeComparisonKey>, ComparisonPairFailure>
DescribeRecipeComparison(ComparisonPair& pair, const execution::Registry& registry);
std::expected<std::optional<execution::RecipeComparisonKey>, ComparisonPairFailure>
DescribeRecipeComparison(ComparisonPair& pair, const execution::Registry& registry,
                         ComparisonRecipePreparation preparation);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_COMPARISON_PAIR_H_
