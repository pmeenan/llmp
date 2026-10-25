// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "engine/comparison_pair.h"
#include "engine/graph_plan.h"
#include "execution/fusion_ranks.h"
#include "ir/storage_admission.h"
#include "planner/alternative.h"

namespace llmp::engine {
namespace {

constexpr std::uint64_t kAllocatorEnvelope = 64;

bool AddBytes(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
  out = a + b;
  return true;
}

class BackendStorageFunder final : public ir::StorageFunder {
 public:
  BackendStorageFunder(std::shared_ptr<const kernels::BackendOwner> owner,
                       std::shared_ptr<const kernels::ConstructionFunder> funder,
                       std::uint64_t control)
      : owner_(std::move(owner)), funder_(std::move(funder)), control_(control) {}
  ~BackendStorageFunder() override {
    if (owner_)
      owner_->Release(control_);
    else if (funder_)
      funder_->release(control_);
  }
  bool Charge(std::uint64_t bytes) const override {
    if (owner_) return owner_->Charge(bytes);
    return funder_ && funder_->charge && funder_->charge(bytes);
  }
  void Release(std::uint64_t bytes) const override {
    if (owner_)
      owner_->Release(bytes);
    else if (funder_ && funder_->release)
      funder_->release(bytes);
  }
  static std::shared_ptr<const ir::StorageFunder> Create(const kernels::StepBackend& backend) {
    constexpr std::uint64_t control = sizeof(BackendStorageFunder) + 2 * kAllocatorEnvelope;
    const auto backing = backend.ConstructionFunding();
    if (backend.owner) {
      if (!backend.owner->Charge(control)) return {};
      return std::make_shared<BackendStorageFunder>(backend.owner, backing, control);
    }
    if (!backing || !backing->charge || !backing->release || !backing->charge(control)) return {};
    return std::make_shared<BackendStorageFunder>(nullptr, backing, control);
  }

 private:
  std::shared_ptr<const kernels::BackendOwner> owner_;
  std::shared_ptr<const kernels::ConstructionFunder> funder_;
  std::uint64_t control_;
};

bool SamePersistent(const kernels::ComparisonRegionRecipeSeal& a,
                    const kernels::ComparisonRegionRecipeSeal& b) {
  return a.version == b.version && a.missing == b.missing && a.subject == b.subject &&
         a.boundary == b.boundary && a.planned_region == b.planned_region &&
         a.physical_recipe == b.physical_recipe && a.policy == b.policy &&
         a.registry == b.registry && a.backend_recipe == b.backend_recipe &&
         a.backend_build == b.backend_build && a.backend_selection == b.backend_selection &&
         a.backend_device == b.backend_device;
}

struct LocalPairProof {
  // Declared first so the boundary is destroyed before this payment refunds.
  ir::StorageLoan retained_payment;
  std::uint64_t retained_charge = 0;
  components::OwnedComparisonRegionBoundary boundary;
  kernels::ComparisonRegionRecipeSeal incumbent;
  kernels::ComparisonRegionRecipeSeal alternative;
  kernels::ComparisonRegionRecipeSeal chosen;
  execution::RecipeComparisonKey key;
};

// All copied expansions are held in source-sized capacity before the first
// backend query. Each backend span is copied before another request to that
// same program can invalidate its scratch.
std::optional<LocalPairProof> ProveLocalPair(const components::ComparisonBoundaryView& source,
                                             const planner::ComparisonSite& site,
                                             std::shared_ptr<const ir::StorageFunder> funder,
                                             const kernels::StepProgram& incumbent,
                                             const kernels::StepProgram& alternative,
                                             ir::OpKey discovery, planner::Phase phase,
                                             const base::Sha256Digest& selection) {
  if (site.whole_checkpoint.whole_only()) return {};
  // Local certificate closure proves the changed recipe body. Whole-program
  // remainder equality authenticates public slots and outside-region resource
  // envelopes, so global pool or activation sizes are not a safe proxy here.
  const std::size_t node_count = source.io.graph.nodes().size();
  std::uint64_t bytes = 0;
  if (node_count > std::numeric_limits<std::uint32_t>::max() ||
      !AddBytes((node_count + 1) * sizeof(std::uint32_t),
                6 * node_count * sizeof(ir::NodeId) + node_count, bytes) ||
      !AddBytes(bytes, 8 * kAllocatorEnvelope, bytes))
    return {};
  auto workspace_loan = ir::AcquireStorageLoan(funder, bytes);
  if (!workspace_loan) return {};
  std::vector<std::uint32_t> offsets;
  std::vector<ir::NodeId> members, left_expand, right_expand, left_remainder, right_remainder;
  std::vector<ir::NodeId> additions;
  offsets.reserve(node_count + 1);
  members.reserve(node_count);
  left_expand.reserve(node_count);
  right_expand.reserve(node_count);
  left_remainder.reserve(node_count);
  right_remainder.reserve(node_count);
  additions.reserve(node_count);
  if (site.accepted_member_origins.empty()) return {};
  if (site.accepted_member_origins.size() > node_count) return {};
  offsets.push_back(0);
  for (const auto& group : site.accepted_member_origins) {
    if (group.empty()) return {};
    for (const auto node : group) {
      if (!node.valid() || node.v >= node_count || members.size() == node_count) return {};
      members.push_back(node);
    }
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }

  std::vector<std::uint8_t> membership(node_count, 0);
  auto mark_members = [&]() -> bool {
    std::fill(membership.begin(), membership.end(), 0);
    for (const auto node : members) {
      if (!node.valid() || node.v >= membership.size() || membership[node.v]) return false;
      membership[node.v] = 1;
    }
    return true;
  };
  if (!mark_members()) return {};

  for (;;) {
    auto boundary_result = components::MakeComparisonRegionBoundary(
        {.source = source, .member_offsets = offsets, .members = members}, funder);
    if (!boundary_result) return {};
    components::OwnedComparisonRegionBoundary boundary = std::move(*boundary_result);
    const auto left = incumbent.ProjectComparisonRegionCertificate(
        boundary, kernels::ComparisonCertificatePhase::kDescriptor);
    if (left.required_members.size() > node_count) return {};
    left_expand.assign(left.required_members.begin(), left.required_members.end());
    const auto right = alternative.ProjectComparisonRegionCertificate(
        boundary, kernels::ComparisonCertificatePhase::kDescriptor);
    if (right.required_members.size() > node_count) return {};
    right_expand.assign(right.required_members.begin(), right.required_members.end());
    const auto left_outside = incumbent.CompareComparisonRemainder(
        alternative, boundary, kernels::ComparisonCertificatePhase::kDescriptor);
    if (left_outside.required_members.size() > node_count) return {};
    left_remainder.assign(left_outside.required_members.begin(),
                          left_outside.required_members.end());
    const auto right_outside = alternative.CompareComparisonRemainder(
        incumbent, boundary, kernels::ComparisonCertificatePhase::kDescriptor);
    if (right_outside.required_members.size() > node_count) return {};
    right_remainder.assign(right_outside.required_members.begin(),
                           right_outside.required_members.end());

    const bool statuses_supported =
        (left.status == kernels::ComparisonRegionStatus::kClosed ||
         left.status == kernels::ComparisonRegionStatus::kExpand) &&
        (right.status == kernels::ComparisonRegionStatus::kClosed ||
         right.status == kernels::ComparisonRegionStatus::kExpand) &&
        (left_outside.status == kernels::ComparisonRemainderStatus::kSame ||
         left_outside.status == kernels::ComparisonRemainderStatus::kExpand) &&
        (right_outside.status == kernels::ComparisonRemainderStatus::kSame ||
         right_outside.status == kernels::ComparisonRemainderStatus::kExpand);
    if (!statuses_supported ||
        (left.status == kernels::ComparisonRegionStatus::kClosed && !left_expand.empty()) ||
        (right.status == kernels::ComparisonRegionStatus::kClosed && !right_expand.empty()) ||
        (left_outside.status == kernels::ComparisonRemainderStatus::kSame &&
         !left_remainder.empty()) ||
        (right_outside.status == kernels::ComparisonRemainderStatus::kSame &&
         !right_remainder.empty()))
      return {};
    const bool certs_closed =
        left.status == kernels::ComparisonRegionStatus::kClosed &&
        right.status == kernels::ComparisonRegionStatus::kClosed &&
        left.seal.state == kernels::ComparisonProofState::kDescriptorVerified &&
        right.seal.state == kernels::ComparisonProofState::kDescriptorVerified;
    const bool remainders_same = left_outside.status == kernels::ComparisonRemainderStatus::kSame &&
                                 right_outside.status == kernels::ComparisonRemainderStatus::kSame;
    if (certs_closed && remainders_same) {
      auto key = RecipeComparisonKeyForReadyCertificates(boundary, discovery, phase, selection,
                                                         left.seal, right.seal);
      if (!key) return {};
      return LocalPairProof{{}, 0, std::move(boundary), left.seal, right.seal, {}, *key};
    }

    additions.clear();
    const auto append = [&](std::span<const ir::NodeId> requested) {
      for (const auto node : requested) {
        if (!node.valid() || node.v >= membership.size()) return false;
        if (membership[node.v]) continue;
        if (additions.size() == additions.capacity()) return false;
        membership[node.v] = 1;
        additions.push_back(node);
      }
      return true;
    };
    if (!append(left_expand) || !append(right_expand) || !append(left_remainder) ||
        !append(right_remainder))
      return {};
    std::ranges::sort(additions, {}, &ir::NodeId::v);
    if (additions.empty()) return {};
    for (const auto node : additions) {
      if (members.size() == node_count) return {};
      members.push_back(node);
      offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    if (!mark_members()) return {};
  }
}

bool RevalidateCertificates(const kernels::StepProgram& candidate,
                            std::span<const LocalPairProof> prior) {
  for (const auto& saved : prior) {
    const auto cert = candidate.ProjectComparisonRegionCertificate(
        saved.boundary, kernels::ComparisonCertificatePhase::kDescriptor);
    if (cert.status != kernels::ComparisonRegionStatus::kClosed || !cert.required_members.empty() ||
        cert.seal.state != kernels::ComparisonProofState::kDescriptorVerified ||
        !SamePersistent(saved.chosen, cert.seal))
      return false;
    const auto same = candidate.CompareComparisonRemainder(
        candidate, saved.boundary, kernels::ComparisonCertificatePhase::kDescriptor);
    if (same.status != kernels::ComparisonRemainderStatus::kSame || !same.required_members.empty())
      return false;
  }
  return true;
}

std::optional<components::OwnedComparisonRegionBoundary> SeedBoundary(
    const components::ComparisonBoundaryView& source, std::span<const std::uint32_t> offsets,
    std::span<const ir::NodeId> members, std::shared_ptr<const ir::StorageFunder> funder) {
  if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != members.size()) return {};
  for (std::size_t i = 1; i < offsets.size(); ++i)
    if (offsets[i - 1] > offsets[i] || offsets[i] > members.size()) return {};
  auto made = components::MakeComparisonRegionBoundary(
      {.source = source, .member_offsets = offsets, .members = members}, std::move(funder));
  if (!made) return {};
  return std::move(*made);
}

std::optional<execution::RecipeComparisonKey> ProveWholePair(
    const components::ComparisonBoundaryView& source,
    const components::OwnedComparisonBoundary& full, const planner::ComparisonSite& site,
    std::shared_ptr<const ir::StorageFunder> funder, execution::Phase phase,
    const base::Sha256Digest& selection, const kernels::ComparisonRecipeSeal& incumbent,
    const kernels::ComparisonRecipeSeal& alternative) {
  const auto count = source.io.graph.nodes().size();
  if (site.accepted_member_origins.empty() || site.accepted_member_origins.size() > count ||
      count > UINT32_MAX)
    return {};
  std::uint64_t bytes = 0;
  if (!AddBytes((count + 1) * sizeof(std::uint32_t), count * sizeof(ir::NodeId), bytes) ||
      !AddBytes(bytes, 2 * kAllocatorEnvelope, bytes))
    return {};
  auto payment = ir::AcquireStorageLoan(funder, bytes);
  if (!payment) return {};
  std::vector<std::uint32_t> offsets;
  std::vector<ir::NodeId> members;
  offsets.reserve(count + 1);
  members.reserve(count);
  offsets.push_back(0);
  for (const auto& group : site.accepted_member_origins) {
    if (group.empty() || group.size() > count - members.size()) return {};
    members.insert(members.end(), group.begin(), group.end());
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  auto boundary = SeedBoundary(source, offsets, members, funder);
  if (!boundary) return {};
  return RecipeComparisonKeyForWholePrograms(full, *boundary, site, phase, selection, incumbent,
                                             alternative);
}

class AdoptedRecipeProgram final : public kernels::StepProgram {
 public:
  static std::unique_ptr<AdoptedRecipeProgram> Prepare(
      std::shared_ptr<const ir::StorageFunder> funder, std::size_t certificate_capacity,
      const kernels::StepProgram& incumbent) {
    std::uint64_t capacity_bytes = 0;
    if (certificate_capacity > std::numeric_limits<std::uint64_t>::max() / sizeof(LocalPairProof) ||
        !AddBytes(sizeof(AdoptedRecipeProgram), certificate_capacity * sizeof(LocalPairProof),
                  capacity_bytes) ||
        !AddBytes(capacity_bytes, 2 * kAllocatorEnvelope, capacity_bytes))
      return {};
    auto loan = ir::AcquireStorageLoan(funder, capacity_bytes);
    if (!loan) return {};
    auto pending = std::unique_ptr<AdoptedRecipeProgram>(new (std::nothrow) AdoptedRecipeProgram(
        std::move(*loan), std::move(funder), capacity_bytes, certificate_capacity));
    if (!pending) return {};
    const auto resources = incumbent.RetainedResources();
    if (resources && !pending->HoldResources(resources)) return {};
    return pending;
  }

  bool AddProof(LocalPairProof&& proof) {
    if (proofs_.size() == proofs_.capacity()) return false;
    const auto bytes = proof.boundary.storage_bytes();
    std::uint64_t next_retained = 0;
    if (!AddBytes(retained_bytes_, bytes, next_retained)) return false;
    auto payment = ir::AcquireStorageLoan(funder_, bytes);
    if (!payment) return false;
    proof.retained_payment = std::move(*payment);
    proof.retained_charge = bytes;
    proof.boundary.ConstructionPaid();
    retained_bytes_ = next_retained;
    proofs_.push_back(std::move(proof));
    return true;
  }
  void PopProof() {
    if (proofs_.empty()) return;
    retained_bytes_ -= proofs_.back().retained_charge;
    proofs_.pop_back();
  }
  std::span<const LocalPairProof> proofs() const { return proofs_; }
  void SetWholeProof(const kernels::ComparisonRecipeSeal& proof, bool ordinary_only) {
    whole_proof_ = proof;
    ordinary_only_ = ordinary_only;
  }
  bool has_whole_proof() const { return whole_proof_.has_value(); }
  bool SameResources(const kernels::StepProgram& candidate) const {
    const auto retained = RetainedResources();
    const auto candidate_resources = candidate.RetainedResources();
    return static_cast<bool>(retained) == static_cast<bool>(candidate_resources) &&
           (!retained || candidate.UsesResources(retained));
  }
  void InstallFinal(std::unique_ptr<kernels::StepProgram> program,
                    kernels::ComparisonPlanIdentity identity) {
    program_ = std::move(program);
    identity_ = identity;
    ready_ = false;
  }
  void ConstructionPaid() override {
    if (paid_) return;
    paid_ = true;
    if (program_) program_->ConstructionPaid();
    for (auto& proof : proofs_) {
      proof.boundary.ConstructionPaid();
      proof.retained_payment.Reset();
    }
    loan_.Reset();
    funder_.reset();
    kernels::StepProgram::ConstructionPaid();
  }
  std::uint64_t scratch_bytes() const override { return program_->scratch_bytes(); }
  std::uint64_t pool_bytes() const override { return program_->pool_bytes(); }
  std::uint64_t private_pool_bytes() const override { return program_->private_pool_bytes(); }
  std::uint64_t host_bytes() const override {
    std::uint64_t bytes = program_ ? program_->host_bytes() : 0;
    if (!AddBytes(bytes, retained_bytes_, bytes)) return std::numeric_limits<std::uint64_t>::max();
    return bytes;
  }
  std::uint64_t launches() const override { return program_->launches(); }
  std::span<const kernels::InputSlot> inputs() const override {
    return program_ ? program_->inputs() : std::span<const kernels::InputSlot>{};
  }
  std::span<const kernels::OutputSlot> outputs() const override {
    return program_ ? program_->outputs() : std::span<const kernels::OutputSlot>{};
  }
  std::span<const kernels::TensorUse> tensors() const override {
    return program_ ? program_->tensors() : std::span<const kernels::TensorUse>{};
  }
  const execution::BoundPlan* bound() const override {
    return ready_ && program_ ? program_->bound() : nullptr;
  }
  std::optional<kernels::ComparisonRecipeSeal> comparison_recipe() const override {
    return program_ ? program_->comparison_recipe() : std::nullopt;
  }
  std::optional<kernels::ComparisonPlanIdentity> comparison_plan_identity() const override {
    if (!program_) return std::nullopt;
    const auto identity = program_->comparison_plan_identity();
    return identity && identity->source_graph_identity == identity_.source_graph_identity &&
                   identity->planned_graph_identity == identity_.planned_graph_identity
               ? identity
               : std::nullopt;
  }
  bool capturable() const override { return !ordinary_only_ && program_ && program_->capturable(); }
  std::optional<std::span<const kernels::RoutedWeight>> weight_reads() const override {
    return program_ ? program_->weight_reads() : std::nullopt;
  }
  kernels::ComparisonRemainderResult CompareComparisonRemainder(
      const kernels::StepProgram& candidate,
      const components::OwnedComparisonRegionBoundary& boundary,
      kernels::ComparisonCertificatePhase phase) const override {
    (void)candidate;
    (void)boundary;
    (void)phase;
    return {};
  }
  const kernels::StepProgram& comparison_program() const override { return *this; }
  kernels::ComparisonRegionResult ProjectComparisonRegion(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    (void)boundary;
    return {};
  }
  kernels::ComparisonRegionDescriptorResult ProjectComparisonRegionDescriptor(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    (void)boundary;
    return {};
  }
  kernels::ComparisonRegionDescriptorResult ProjectComparisonRegionCertificate(
      const components::OwnedComparisonRegionBoundary& boundary,
      kernels::ComparisonCertificatePhase phase) const override {
    (void)boundary;
    (void)phase;
    return {};
  }
  bool ValidateComparisonProjection(
      const kernels::ComparisonRegionProjection& projection) const override {
    (void)projection;
    return false;
  }
  std::expected<void, kernels::StepFailure> RunComparisonPrefix(
      const kernels::ComparisonRegionProjection& projection, kernels::StepLaunch& launch,
      providers::NativeStream stream) const override {
    (void)projection;
    (void)launch;
    (void)stream;
    return std::unexpected(
        kernels::StepFailure{.detail = "runtime recipe wrapper has no comparison executor"});
  }
  std::expected<void, kernels::StepFailure> RunComparisonRegion(
      const kernels::ComparisonRegionProjection& projection, kernels::StepLaunch& launch,
      providers::NativeStream stream) const override {
    (void)projection;
    (void)launch;
    (void)stream;
    return std::unexpected(
        kernels::StepFailure{.detail = "runtime recipe wrapper has no comparison executor"});
  }
  kernels::RoutedProgram* routed() override {
    return ready_ && program_ ? program_->routed() : nullptr;
  }
  const kernels::RoutedProgram* routed() const override {
    return ready_ && program_ ? program_->routed() : nullptr;
  }
  std::expected<void, kernels::StepFailure> Bind(kernels::StepLaunch& launch) override {
    ready_ = false;
    if (!program_)
      return std::unexpected(kernels::StepFailure{.detail = "recipe program is not installed"});
    auto bound = program_->Bind(launch);
    if (!bound) return bound;
    const auto identity = program_->comparison_plan_identity();
    if (!identity || identity->source_graph_identity != identity_.source_graph_identity ||
        identity->planned_graph_identity != identity_.planned_graph_identity)
      return std::unexpected(
          kernels::StepFailure{.detail = "recipe adoption plan identity changed at Bind"});
    if (whole_proof_) {
      const auto actual = program_->comparison_recipe();
      if (!actual || !SameWholeProgramRecipe(*whole_proof_, *actual))
        return std::unexpected(
            kernels::StepFailure{.detail = "whole program recipe changed at Bind"});
    }
    for (const auto& certificate : proofs_) {
      const auto proof = program_->ProjectComparisonRegionCertificate(
          certificate.boundary, kernels::ComparisonCertificatePhase::kBound);
      if (proof.status != kernels::ComparisonRegionStatus::kClosed ||
          !proof.required_members.empty() ||
          proof.seal.state != kernels::ComparisonProofState::kBoundVerified ||
          !SamePersistent(certificate.chosen, proof.seal))
        return std::unexpected(
            kernels::StepFailure{.detail = "recipe adoption certificate changed at Bind"});
      const auto remainder = program_->CompareComparisonRemainder(
          *program_, certificate.boundary, kernels::ComparisonCertificatePhase::kBound);
      if (remainder.status != kernels::ComparisonRemainderStatus::kSame ||
          !remainder.required_members.empty())
        return std::unexpected(
            kernels::StepFailure{.detail = "recipe adoption remainder changed at Bind"});
    }
    ready_ = true;
    return {};
  }
  std::expected<void, kernels::StepFailure> Position(std::span<const std::uint32_t> past) override {
    if (!ready_)
      return std::unexpected(kernels::StepFailure{.detail = "recipe program is not bound"});
    return program_->Position(past);
  }
  std::expected<void, kernels::StepFailure> TryUpdateGeneratedInputs(
      const planner::StepShape& shape) override {
    if (!ready_)
      return std::unexpected(kernels::StepFailure{.detail = "recipe program is not bound"});
    return program_->TryUpdateGeneratedInputs(shape);
  }
  std::expected<void, kernels::StepFailure> Run(kernels::StepLaunch& launch,
                                                providers::NativeStream stream) const override {
    if (!ready_)
      return std::unexpected(kernels::StepFailure{.detail = "recipe program is not bound"});
    return program_->Run(launch, stream);
  }
  bool warm() const override { return program_->warm(); }
  std::expected<void, kernels::StepFailure> RunWarm(kernels::StepLaunch& launch,
                                                    providers::NativeStream stream) const override {
    if (!ready_)
      return std::unexpected(kernels::StepFailure{.detail = "recipe program is not bound"});
    return program_->RunWarm(launch, stream);
  }
  std::expected<std::unique_ptr<kernels::Recorded>, kernels::StepFailure> Record(
      kernels::StepLaunch& launch, providers::NativeStream stream,
      const kernels::StepLaunch::Body& body) const override {
    if (ordinary_only_)
      return std::unexpected(kernels::StepFailure{
          .detail = "ordinary whole-program ranking does not authorize captured execution"});
    if (!ready_)
      return std::unexpected(kernels::StepFailure{.detail = "recipe program is not bound"});
    return program_->Record(launch, stream, body);
  }
  void ReportPlan(const kernels::PlanReportSink& sink) const override {
    program_->ReportPlan(sink);
  }
  void StreamPlanReport(const kernels::PlanReportSink& sink) const override {
    program_->StreamPlanReport(sink);
  }
  kernels::PlanCoverage plan_coverage() const override {
    return program_ ? program_->plan_coverage() : kernels::PlanCoverage{};
  }

 private:
  AdoptedRecipeProgram(ir::StorageLoan loan, std::shared_ptr<const ir::StorageFunder> funder,
                       std::uint64_t base_bytes, std::size_t capacity)
      : loan_(std::move(loan)), funder_(std::move(funder)), retained_bytes_(base_bytes) {
    proofs_.reserve(capacity);
  }
  ir::StorageLoan loan_;
  std::shared_ptr<const ir::StorageFunder> funder_;
  std::uint64_t retained_bytes_;
  std::unique_ptr<kernels::StepProgram> program_;
  std::vector<LocalPairProof> proofs_;
  std::optional<kernels::ComparisonRecipeSeal> whole_proof_;
  bool ordinary_only_ = false;
  kernels::ComparisonPlanIdentity identity_;
  bool ready_ = false;
  bool paid_ = false;
};

}  // namespace

namespace {
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildCumulativeCore(
    const GraphBuildIO& io, std::optional<kernels::TokenLowerInput> token, execution::Phase phase,
    std::uint16_t active_stage, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources) {
  if (token && (&token->step.graph != &io.graph || options.shape != &token->shape))
    return std::unexpected("token lowering metadata differs from the source graph or shape");
  if (!token && (options.shape || options.state_spans))
    return std::unexpected("ordinary component comparison cannot borrow token shape or state");
  const auto state = token ? std::span<const components::StepState>(token->step.state)
                           : std::span<const components::StepState>{};
  const auto token_for =
      [&](planner::StateSpans state_spans) -> std::optional<kernels::TokenLowerInput> {
    if (!token) return {};
    return kernels::TokenLowerInput{token->step, token->shape, token->context,
                                    std::move(state_spans)};
  };
  std::shared_ptr<const ir::StorageFunder> funder;
  const auto ordinary =
      [&](planner::PlanOptions&& fallback_options,
          bool options_are_prepared =
              false) -> std::expected<std::unique_ptr<kernels::StepProgram>, std::string> {
    if (options_are_prepared) {
      auto original_graph = io.graph.Clone(funder);
      if (!original_graph) return std::unexpected("ordinary graph copy failed");
      auto plan = planner::Plan(std::move(*original_graph), *backend.registry, backend.device,
                                policy, fallback_options);
      if (!plan) return std::unexpected("ordinary component planning failed");
      const kernels::LowerInput input{*plan, io, addresses,
                                      token_for(fallback_options.state_spans)};
      return LowerComponentPlan(input, backend, policy, validate, report_owner, resources);
    }
    auto state_spans = fallback_options.state_spans;
    return detail::BuildOrdinaryComponentProgram(
        io, backend, policy, addresses, std::move(fallback_options),
        token_for(std::move(state_spans)), validate, report_owner, resources);
  };
  if (backend.registry == nullptr || backend.registry->ranks().recipe_comparisons.empty() ||
      addresses.measuring() || options.owner_products != nullptr ||
      (token && !token->step.owner_products.empty()) || backend.lanes.count != 0)
    return ordinary(std::move(options));
  funder = BackendStorageFunder::Create(backend);
  if (!funder) return ordinary(std::move(options));
  auto prepared_result = PrepareComponentPlanOptionsInPlace(io, backend, addresses, options);
  if (!prepared_result) return ordinary(std::move(options));
  auto prepared = std::move(options);
  bool incumbent_lower_started = false;
  const auto attempt = [&]() -> std::expected<std::unique_ptr<kernels::StepProgram>, std::string> {
    auto source_bindings = token
                               ? ComparisonSourceBindings::Create(funder, token->step, active_stage)
                               : ComparisonSourceBindings::Create(funder, io, active_stage);
    if (!source_bindings) return std::unexpected("comparison source inventory unavailable");
    {
      auto original_graph = io.graph.Clone(funder);
      if (!original_graph) return std::unexpected("comparison source clone unavailable");
      auto baseline_plan = planner::Plan(std::move(*original_graph), *backend.registry,
                                         backend.device, policy, prepared);
      if (!baseline_plan) return std::unexpected("comparison baseline plan unavailable");
      auto baseline_ranges = source_bindings->Resolve(*baseline_plan, addresses, backend.written);
      if (!baseline_ranges) return std::unexpected("comparison source ranges unavailable");
    }
    auto source_view = source_bindings->boundary_view();
    if (!source_view) return std::unexpected("comparison source view unavailable");
    auto source_boundary_result = components::MakeComparisonBoundary(*source_view, funder);
    if (!source_boundary_result) return std::unexpected("comparison source boundary unavailable");
    auto source_boundary = std::move(*source_boundary_result);

    const auto& rows = backend.registry->ranks().recipe_comparisons;
    std::uint64_t filter_bytes = 0;
    if (rows.size() > std::numeric_limits<std::uint64_t>::max() / sizeof(ir::OpKey) ||
        !AddBytes(rows.size() * sizeof(ir::OpKey), kAllocatorEnvelope, filter_bytes))
      return std::unexpected("comparison operation inventory unavailable");
    auto filter_loan = ir::AcquireStorageLoan(funder, filter_bytes);
    if (!filter_loan) return std::unexpected("comparison operation admission unavailable");
    std::vector<ir::OpKey> filter;
    filter.reserve(rows.size());
    for (const auto& row : rows) filter.push_back(row.key.discovery_operation);
    std::ranges::sort(filter);
    filter.erase(std::unique(filter.begin(), filter.end()), filter.end());
    planner::ComparisonPlanningEnvironment environment(*source_view, *backend.registry,
                                                       backend.device, policy, prepared,
                                                       token ? &token->shape : nullptr);
    auto ledger_result = planner::ComparisonDecisionLedger::Create(environment, funder);
    if (!ledger_result) return std::unexpected("comparison ledger unavailable");
    auto ledger = std::move(*ledger_result);
    auto discovered = planner::DiscoverComparisonPlan(
        *source_view, filter, false, funder, *backend.registry, backend.device, policy, prepared);
    if (!discovered) return std::unexpected("comparison discovery unavailable");
    auto discovery = std::move(discovered->discovery);
    const auto tickets = discovery.tickets();
    const planner::ComparisonDiscoveryTicket* seed_ticket = nullptr;
    for (const auto& ticket : tickets) {
      if (ticket.member_offsets.size() >= 2) {
        seed_ticket = &ticket;
        break;
      }
    }
    if (seed_ticket == nullptr)
      return std::unexpected("comparison has no representable seed ticket");
    auto seed_boundary = SeedBoundary(*source_view, seed_ticket->member_offsets,
                                      seed_ticket->member_origins, funder);
    if (!seed_boundary) return std::unexpected("comparison seed boundary unavailable");
    std::expected<std::unique_ptr<kernels::StepProgram>, std::string> current_program =
        std::unexpected("cumulative incumbent was not lowered");
    base::Sha256Digest current_planned_graph{};
    {
      auto current_plan = std::move(discovered->incumbent);
      current_planned_graph = current_plan.plan.graph.Digest();
      auto current_ranges = source_bindings->Resolve(current_plan.plan, addresses, backend.written);
      if (!current_ranges) return std::unexpected("comparison incumbent ranges unavailable");
      const auto discovered_tensor_map = discovery.source_to_planned();
      const auto discovered_node_origins = discovery.planned_node_origins();
      if (discovery.source_identity() != io.graph.identity() ||
          !std::ranges::equal(discovered_tensor_map, current_plan.plan.tensors) ||
          discovered_node_origins.size() != current_plan.plan.graph.nodes().size() ||
          current_plan.plan.graph.identity() == 0 ||
          discovered_tensor_map.size() != io.graph.tensor_count())
        return std::unexpected("comparison incumbent origins unavailable");
      if (discovered_node_origins.size() >
          std::numeric_limits<std::uint64_t>::max() / sizeof(planner::ComparisonOriginSetId))
        return std::unexpected("comparison incumbent origin inventory overflow");
      std::uint64_t origin_bytes =
          discovered_node_origins.size() * sizeof(planner::ComparisonOriginSetId);
      if (!AddBytes(origin_bytes, kAllocatorEnvelope, origin_bytes))
        return std::unexpected("comparison incumbent origin inventory overflow");
      auto origin_loan = ir::AcquireStorageLoan(funder, origin_bytes);
      if (!origin_loan) return std::unexpected("comparison incumbent origin admission unavailable");
      std::vector<std::uint32_t> copied_node_origins;
      copied_node_origins.reserve(discovered_node_origins.size());
      copied_node_origins.insert(copied_node_origins.end(), discovered_node_origins.begin(),
                                 discovered_node_origins.end());
      planner::ComparisonFinalNodeOrigins incumbent_final_nodes(discovery.source_identity());
      incumbent_final_nodes.planned_graph_identity = current_plan.plan.graph.identity();
      incumbent_final_nodes.node_origins =
          planner::OwnedRewriteOrigins(std::move(*origin_loan), std::move(copied_node_origins));
      const kernels::ComparisonRequest comparison{
          .mode = kernels::ComparisonRunMode::kFull,
          .scope = kernels::ComparisonProofScope::kWholeProgramNative,
          .bindings = current_ranges->ranges(),
          .state = state,
          .active_stage = active_stage,
          .expected_boundary = &source_boundary};
      const kernels::ComparisonRegionCapture capture{.seed = &*seed_boundary,
                                                     .origins = &discovery.origins(),
                                                     .final_nodes = &incumbent_final_nodes};
      const kernels::LowerInput incumbent_input{
          current_plan.plan, io, addresses, token_for(prepared.state_spans), &comparison, &capture};
      // From here onward the helper must never rebuild an ordinary plan: even
      // a failed backend Lower may have consumed one-shot construction state.
      incumbent_lower_started = true;
      current_program =
          LowerComponentPlan(incumbent_input, backend, policy, validate, report_owner, resources);
      if (!current_program) return std::unexpected(std::move(current_program.error()));
      const auto identity = (*current_program)->comparison_plan_identity();
      if (identity && (identity->source_graph_identity != io.graph.identity() ||
                       identity->planned_graph_identity != current_plan.plan.graph.identity()))
        return std::unexpected("comparison incumbent lower changed its tracked graph identity");
      // An unsupported backend may successfully lower the unchanged incumbent
      // without carrying comparison provenance. Keep that exact owner as the
      // ordinary unproved program; identity absence cannot authorize trials.
      if (!identity) return std::move(*current_program);
    }
    std::optional<kernels::ComparisonPlanIdentity> current_identity =
        (*current_program)->comparison_plan_identity();

    // Until a certificate and an exact key both close, this exact lowered
    // incumbent remains the only program returned to the normal runner.
    auto pending_owner = AdoptedRecipeProgram::Prepare(funder, tickets.size(), **current_program);
    if (!pending_owner) return std::move(*current_program);
    for (const auto& ticket : tickets) {
      if (ticket.member_offsets.size() < 2 ||
          (!ticket.boundary_representable && ticket.kind != planner::ComparisonSiteKind::kRewrite))
        continue;
      const auto group_count = ticket.member_offsets.size() - 1;
      std::uint64_t selector_bytes = 0;
      if (ticket.member_offsets.size() >
              std::numeric_limits<std::uint64_t>::max() / sizeof(std::vector<ir::NodeId>) ||
          ticket.member_origins.size() >
              std::numeric_limits<std::uint64_t>::max() / sizeof(ir::NodeId) ||
          ticket.boundary_ports.size() >
              std::numeric_limits<std::uint64_t>::max() / sizeof(planner::ComparisonSitePortRef) ||
          !AddBytes(ticket.member_offsets.size() * sizeof(std::vector<ir::NodeId>),
                    ticket.member_origins.size() * sizeof(ir::NodeId), selector_bytes) ||
          !AddBytes(selector_bytes, ticket.owner_segments.size(), selector_bytes) ||
          !AddBytes(selector_bytes,
                    ticket.boundary_ports.size() * sizeof(planner::ComparisonSitePortRef),
                    selector_bytes) ||
          !AddBytes(selector_bytes, ticket.pattern.size(), selector_bytes) ||
          group_count > std::numeric_limits<std::uint64_t>::max() - 4 ||
          static_cast<std::uint64_t>(group_count) + 4 >
              std::numeric_limits<std::uint64_t>::max() / kAllocatorEnvelope ||
          !AddBytes(selector_bytes, (group_count + 4) * kAllocatorEnvelope, selector_bytes))
        break;
      auto selector_loan = ir::AcquireStorageLoan(funder, selector_bytes);
      if (!selector_loan) break;
      planner::ComparisonSiteSelector selector;
      selector.source_graph_identity = discovery.source_identity();
      selector.kind = ticket.kind;
      selector.pattern.assign(ticket.pattern);
      selector.version = ticket.version;
      selector.member_owner_segments = ticket.owner_segments;
      selector.boundary_ports = ticket.boundary_ports;
      if (!ticket.boundary_representable) selector.checkpoint_match = ticket.checkpoint_match;
      selector.member_origins.reserve(ticket.member_offsets.size() - 1);
      bool valid_ticket = true;
      for (std::size_t group = 0; group + 1 < ticket.member_offsets.size(); ++group) {
        const auto first = ticket.member_offsets[group];
        const auto last = ticket.member_offsets[group + 1];
        if (first > last || last > ticket.member_origins.size()) {
          valid_ticket = false;
          break;
        }
        selector.member_origins.emplace_back(ticket.member_origins.begin() + first,
                                             ticket.member_origins.begin() + last);
      }
      if (!valid_ticket) continue;
      auto trial =
          ticket.boundary_representable
              ? planner::CaptureComparisonTrial(environment, ledger, selector, funder)
              : planner::CaptureWholeComparisonTrial(environment, ledger, selector, funder);
      if (!trial) continue;
      auto alternative = planner::MakeAlternative(*trial);
      if (!alternative) continue;
      auto planned = planner::PlanAlternative(environment, *trial, std::move(*alternative), ledger);
      if (!planned) continue;
      auto ranges = source_bindings->Resolve(planned->plan, addresses, backend.written);
      if (!ranges) continue;
      if (!planned->final_origins ||
          planned->final_node_origins.source_graph_identity != io.graph.identity() ||
          planned->final_node_origins.planned_graph_identity != planned->plan.graph.identity())
        continue;
      const kernels::ComparisonRequest comparison{
          .mode = kernels::ComparisonRunMode::kFull,
          .scope = kernels::ComparisonProofScope::kWholeProgramNative,
          .bindings = ranges->ranges(),
          .state = state,
          .active_stage = active_stage,
          .expected_boundary = &source_boundary};
      const kernels::ComparisonRegionCapture capture{.seed = &*seed_boundary,
                                                     .origins = &*planned->final_origins,
                                                     .final_nodes = &planned->final_node_origins};
      const kernels::LowerInput candidate_input{
          planned->plan, io, addresses, token_for(prepared.state_spans), &comparison, &capture};
      auto candidate_program =
          LowerComponentPlan(candidate_input, backend, policy, validate, {}, resources);
      if (!candidate_program) continue;
      if (!pending_owner->SameResources(**candidate_program)) continue;
      auto proof =
          ProveLocalPair(*source_view, trial->site, funder, **current_program, **candidate_program,
                         ticket.registered_operation, phase, backend.registry->SelectionDigest());
      const auto* measurement =
          proof ? backend.registry->FindRecipeComparison(proof->key) : nullptr;
      if (measurement && measurement->kind != execution::RecipeComparisonKind::kLatency)
        measurement = nullptr;
      const auto left = (*current_program)->comparison_recipe();
      const auto right = (*candidate_program)->comparison_recipe();
      const auto candidate_identity = (*candidate_program)->comparison_plan_identity();
      const execution::RecipeComparison* whole_row = nullptr;
      if (left && right && current_identity && candidate_identity &&
          current_identity->source_graph_identity == io.graph.identity() &&
          candidate_identity->source_graph_identity == io.graph.identity() &&
          candidate_identity->planned_graph_identity == planned->plan.graph.identity() &&
          left->planned_graph == current_planned_graph &&
          right->planned_graph == planned->plan.graph.Digest()) {
        const auto key = ProveWholePair(*source_view, source_boundary, trial->site, funder, phase,
                                        backend.registry->SelectionDigest(), *left, *right);
        whole_row = key ? backend.registry->FindRecipeComparison(*key) : nullptr;
      }
      // A complete-program quality rejection dominates a local latency row
      // for this exact pair. Ordinary latency precedence remains unchanged.
      if (whole_row &&
          (!measurement ||
           whole_row->kind == execution::RecipeComparisonKind::kRejectIncumbentQuality)) {
        const bool checkpoint_only = trial->site.whole_checkpoint.whole_only();
        if (checkpoint_only &&
            !ledger.ValidateTerminalWholeAlternative(environment, *trial, *planned))
          continue;
        const bool choose = execution::RecipeComparisonPreference(*whole_row) ==
                            execution::RecipePreference::kAlternative;
        if (choose) {
          if (!RevalidateCertificates(**candidate_program, pending_owner->proofs())) continue;
          if (!checkpoint_only && !ledger.CommitAcceptedAlternative(environment, *trial, *planned))
            continue;
          current_program = std::move(candidate_program);
          current_identity = *candidate_identity;
          current_planned_graph = right->planned_graph;
        }
        pending_owner->SetWholeProof(
            choose ? *right : *left,
            left->execution_method == kernels::ComparisonExecutionMethod::kOrdinaryOnly ||
                right->execution_method == kernels::ComparisonExecutionMethod::kOrdinaryOnly);
        // An exact whole-program decision ends cumulative selection, including
        // an incumbent or neutral decision: later changes invalidate its seal.
        break;
      }
      if (!measurement) continue;
      const auto preference =
          execution::ClassifyRecipeLatency(measurement->incumbent_ns, measurement->alternative_ns);
      if (preference == execution::RecipePreference::kNeutral) continue;
      const bool choose_alternative = preference == execution::RecipePreference::kAlternative;
      kernels::StepProgram& proposed = choose_alternative ? **candidate_program : **current_program;
      if (choose_alternative && !RevalidateCertificates(proposed, pending_owner->proofs()))
        continue;
      auto identity = proposed.comparison_plan_identity();
      if (!identity || identity->source_graph_identity != io.graph.identity() ||
          (choose_alternative &&
           identity->planned_graph_identity != planned->plan.graph.identity()) ||
          (!choose_alternative &&
           (!current_identity ||
            identity->source_graph_identity != current_identity->source_graph_identity ||
            identity->planned_graph_identity != current_identity->planned_graph_identity)))
        continue;
      proof->chosen = choose_alternative ? proof->alternative : proof->incumbent;
      if (!pending_owner->AddProof(std::move(*proof))) continue;
      if (choose_alternative) {
        auto committed = ledger.CommitAcceptedAlternative(environment, *trial, *planned);
        if (!committed) {
          pending_owner->PopProof();
          continue;
        }
        current_program = std::move(candidate_program);
        current_planned_graph = planned->plan.graph.Digest();
      }
      current_identity = *identity;
    }
    if (pending_owner->proofs().empty() && !pending_owner->has_whole_proof())
      return std::move(*current_program);
    if (!current_identity || current_identity->source_graph_identity != io.graph.identity())
      return std::unexpected("an accepted cumulative recipe lost its live plan identity");
    pending_owner->InstallFinal(std::move(*current_program), *current_identity);
    return std::unique_ptr<kernels::StepProgram>(std::move(pending_owner));
  };
  auto attempted = attempt();
  if (attempted) return attempted;
  if (incumbent_lower_started) return std::unexpected(std::move(attempted.error()));
  return ordinary(std::move(prepared), true);
}

}  // namespace

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildCumulativeRecipeProgram(
    const GraphBuildIO& io, const components::StepGraph& token_step,
    const planner::StepShape& shape, std::uint16_t active_stage, std::uint32_t context,
    const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources) {
  const kernels::TokenLowerInput token{token_step, shape, context, options.state_spans};
  return BuildCumulativeCore(io, token, shape.phase, active_stage, backend, policy, addresses,
                             std::move(options), validate, std::move(report_owner), resources);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildCumulativeRecipeProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources) {
  return BuildCumulativeCore(io, std::nullopt, execution::Phase::kPrefill, policy.active_stage,
                             backend, policy, addresses, std::move(options), validate,
                             std::move(report_owner), resources);
}

}  // namespace llmp::engine
