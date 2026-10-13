// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/execution_contract.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <format>
#include <limits>
#include <string_view>
#include <utility>

#include "format/encoding.h"

namespace llmp::engine {

std::uint64_t ExecutionMetadataAllowance::OverheadBytes() {
  return sizeof(ExecutionMetadataAllowance) + 512;
}

std::expected<std::shared_ptr<ExecutionMetadataAllowance>, std::string>
ExecutionMetadataAllowance::Admit(catalog::Catalog& catalog, catalog::DomainId domain,
                                  std::uint64_t bytes, std::uint64_t available,
                                  std::uint64_t margin) {
  // Includes this owner/control block and the pinned catalog extent's metadata.
  const auto overhead = OverheadBytes();
  if (bytes <= overhead || margin == 0 || bytes > available || margin > available - bytes)
    return std::unexpected("execution metadata allowance lacks paid physical headroom");
  auto extent = catalog.AddExtent({.domain = domain,
                                   .memory_class = catalog::MemoryClass::kRuntime,
                                   .recovery = catalog::Recovery::kPinned,
                                   .size = base::Bytes(bytes),
                                   .content = {}},
                                  true);
  if (!extent) return std::unexpected("execution metadata allowance could not be cataloged");
  return std::shared_ptr<ExecutionMetadataAllowance>(
      new ExecutionMetadataAllowance(catalog, domain, *extent, bytes - overhead));
}

bool ExecutionMetadataAllowance::HeldBy(const catalog::Catalog& catalog,
                                        catalog::DomainId domain) const {
  if (catalog_ != &catalog || domain_ != domain) return false;
  const auto extent = catalog.Describe(extent_);
  return extent && extent->state == catalog::ExtentState::kResident &&
         extent->descriptor.domain == domain &&
         extent->descriptor.recovery == catalog::Recovery::kPinned &&
         extent->descriptor.memory_class == catalog::MemoryClass::kRuntime &&
         extent->descriptor.size.value() >= capacity_ + OverheadBytes();
}

std::expected<void, std::string> ExecutionMetadataAllowance::Grow(
    catalog::Catalog& catalog, catalog::DomainId domain, std::uint64_t capacity,
    std::uint64_t budget, std::uint64_t available, std::uint64_t margin) {
  if (!HeldBy(catalog, domain)) return std::unexpected("metadata growth lost its held payment");
  if (capacity <= capacity_) return {};
  const auto delta = capacity - capacity_;
  const auto occupied = catalog.OccupancyOf(domain).Total().value();
  if (capacity > UINT64_MAX - OverheadBytes() || occupied > budget || delta > budget - occupied ||
      margin == 0 || delta > available || margin > available - delta)
    return std::unexpected("metadata growth exceeds admitted physical headroom");
  if (!catalog.GrowRuntimeCharge(extent_, base::Bytes(capacity + OverheadBytes())))
    return std::unexpected("metadata growth could not retain its catalog charge");
  capacity_ = capacity;
  return {};
}

bool ExecutionMetadataAllowance::Charge(std::uint64_t bytes) {
  if (bytes > capacity_ - used_) return false;
  used_ += bytes;
  return true;
}

void ExecutionMetadataAllowance::Release(std::uint64_t bytes) {
  // Releases are paired by the immutable metadata owner; an over-release must
  // not invent capacity on a malformed caller path.
  if (bytes <= used_) used_ -= bytes;
}

namespace {

// Explicit little-endian scalars and length-prefixed bytes; no structure
// padding, enum representation or delimiter-dependent field boundaries.
class Identity {
 public:
  explicit Identity(std::string_view domain) { Text(domain); }
  void Number(std::uint64_t number) {
    std::array<std::byte, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::byte>(number >> (i * 8));
    hash_.Update(bytes);
  }
  void Text(std::string_view text) {
    Number(text.size());
    hash_.Update(text);
  }
  void Digest(const base::Sha256Digest& digest) {
    Number(digest.size());
    hash_.Update(std::as_bytes(std::span(digest)));
  }
  base::Sha256Digest Finish() { return hash_.Finish(); }

 private:
  base::Sha256 hash_;
};

bool Known(const base::Sha256Digest& digest) {
  return std::ranges::any_of(digest, [](auto byte) { return byte != 0; });
}

bool Known(planner::Phase phase) {
  switch (phase) {
    case planner::Phase::kPrefill:
    case planner::Phase::kDecode:
    case planner::Phase::kVerify:
    case planner::Phase::kDraft:
    case planner::Phase::kScore:
    case planner::Phase::kCommit:
      return true;
  }
  return false;
}

bool Known(NumericalUse use) {
  switch (use) {
    case NumericalUse::kPrompt:
    case NumericalUse::kDecode:
    case NumericalUse::kScore:
    case NumericalUse::kPromptCompanions:
    case NumericalUse::kSpeculative:
    case NumericalUse::kBlock:
    case NumericalUse::kJoinedRepeatPending:
      return true;
  }
  return false;
}

}  // namespace

std::expected<NumericalDomain, std::string> NumericalDomain::Create(
    std::vector<NumericalStage> stages) {
  if (stages.empty() || stages.size() > std::uint64_t{UINT16_MAX} + 1) {
    return std::unexpected("a numerical domain needs an addressable ordered stage set");
  }
  Identity hash("llmp-numerical-domain-v1");
  hash.Number(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    if (stage.stage != i || stage.name.empty() || stage.uses.empty() || stage.phases.empty()) {
      return std::unexpected("a numerical stage is missing or out of order");
    }
    hash.Number(stage.stage);
    hash.Text(stage.name);
    hash.Number(stage.uses.size());
    for (std::size_t u = 0; u < stage.uses.size(); ++u) {
      if (!Known(stage.uses[u]) || (u != 0 && stage.uses[u - 1] >= stage.uses[u])) {
        return std::unexpected("numerical uses must be known, ordered and unique");
      }
      hash.Number(static_cast<std::uint8_t>(stage.uses[u]));
    }
    hash.Number(stage.phases.size());
    for (std::size_t p = 0; p < stage.phases.size(); ++p) {
      const auto& phase = stage.phases[p];
      if (!Known(phase.phase) || (p != 0 && stage.phases[p - 1].phase >= phase.phase) ||
          phase.max_owners == 0 || phase.max_owners > std::uint64_t{UINT8_MAX} + 1 ||
          phase.max_owner_rows == 0 || phase.max_total_rows == 0) {
        return std::unexpected("numerical phases need known unique phases and positive bounds");
      }
      hash.Number(static_cast<std::uint8_t>(phase.phase));
      hash.Number(phase.max_owners);
      hash.Number(phase.max_owner_rows);
      hash.Number(phase.max_total_rows);
      hash.Number(phase.row_invariant ? 1 : 0);
    }
  }
  return NumericalDomain(std::move(stages), hash.Finish());
}

std::expected<void, std::string> NumericalDomain::CheckShape(
    const planner::StepShape& shape) const {
  if (shape.stage >= stages_.size() || !Known(shape.phase)) {
    return std::unexpected("a step names no qualified numerical stage or phase");
  }
  const auto& stage = stages_[shape.stage];
  const auto phase = std::ranges::find(stage.phases, shape.phase, &NumericalPhase::phase);
  if (phase == stage.phases.end() || (shape.row_invariant && !phase->row_invariant)) {
    return std::unexpected("the source does not qualify this step's numerical mode");
  }
  if (shape.segments.empty() || shape.segments.size() > phase->max_owners) {
    return std::unexpected("the source does not qualify this step's owner count");
  }
  std::uint64_t rows = 0;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const auto& segment = shape.segments[i];
    if (segment.rows == 0 || segment.rows > phase->max_owner_rows ||
        (i != 0 && shape.segments[i - 1].slot >= segment.slot)) {
      return std::unexpected("the source does not qualify these ordered owner rows");
    }
    // At most 256 owners of uint32 rows, so this addition cannot overflow.
    rows += segment.rows;
  }
  if (rows > phase->max_total_rows) {
    return std::unexpected("the source does not qualify this step's total rows");
  }
  return {};
}

std::expected<void, std::string> NumericalDomain::CheckMode(NumericalUse use) const {
  for (const auto& stage : stages_) {
    if (std::ranges::find(stage.uses, use) == stage.uses.end()) {
      return std::unexpected("the execution composition does not qualify this numerical mode");
    }
  }
  return {};
}

std::expected<void, std::string> NumericalDomain::CheckUse(
    NumericalUse use, std::span<const planner::StepShape> shapes) const {
  if (!Known(use) || shapes.empty()) {
    return std::unexpected("a numerical operation needs its known use and complete shapes");
  }
  for (const auto& shape : shapes) {
    if (auto checked = CheckShape(shape); !checked) return checked;
    const auto& allowed = stages_[shape.stage].uses;
    if (std::ranges::find(allowed, use) == allowed.end()) {
      return std::unexpected("a participating stage does not qualify this numerical operation");
    }
  }
  return {};
}

std::expected<ExecutionContract, std::string> ExecutionContract::Create(
    NumericalDomain domain, std::span<const StageExecutionInput> stages) {
  if (stages.size() != domain.stages().size()) {
    return std::unexpected("execution compatibility must seal every numerical stage");
  }
  Identity hash("llmp-execution-contract-v2");
  hash.Digest(domain.identity());
  hash.Number(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    if (stage.stage != i || stage.name != domain.stages()[i].name || stage.registry == nullptr ||
        stage.registry->size() == 0 || !Known(stage.patterns) || !Known(stage.backend.recipe) ||
        !Known(stage.backend.build) || !Known(stage.backend.selection) ||
        stage.device.device_class.empty() ||
        (stage.reservation_alignment != 0 && !std::has_single_bit(stage.reservation_alignment)) ||
        !std::has_single_bit(stage.placement_alignment)) {
      return std::unexpected("execution compatibility lacks an immutable stage declaration");
    }
    if (auto checked = planner::Validate(stage.policy); !checked) {
      return std::unexpected(std::format("execution policy: {}", checked.error()));
    }
    hash.Number(stage.stage);
    hash.Text(stage.name);
    hash.Text(planner::ToText(stage.policy));
    hash.Digest(stage.patterns);
    hash.Number(stage.registry->size());
    for (std::size_t r = 0; r < stage.registry->size(); ++r) {
      hash.Text(stage.registry->at(r).name);
      hash.Digest(stage.registry->identity(r));
    }
    hash.Text(stage.device.device_class);
    hash.Number(stage.device.arch);
    hash.Number(stage.device.features);
    hash.Number(stage.device.tuned.size());
    for (const auto& tune : stage.device.tuned) {
      if (tune.implementation.empty() || tune.rows == 0 || tune.k == 0 || tune.n == 0 ||
          !stage.registry->Find(tune.implementation) ||
          (tune.output_encoding &&
           std::ranges::find(format::Encodings(), *tune.output_encoding,
                             &format::EncodingInfo::id) == format::Encodings().end())) {
        return std::unexpected("execution compatibility has an invalid immutable tuning key");
      }
      hash.Text(tune.implementation);
      hash.Number(tune.rows);
      hash.Number(tune.k);
      hash.Number(tune.n);
      hash.Number(tune.output_encoding.has_value());
      if (tune.output_encoding) hash.Number(static_cast<std::uint16_t>(*tune.output_encoding));
    }
    hash.Digest(stage.backend.recipe);
    hash.Digest(stage.backend.build);
    hash.Digest(stage.backend.selection);
    hash.Number(stage.reservation_alignment);
    hash.Number(stage.placement_alignment);
  }
  return ExecutionContract(std::move(domain), hash.Finish());
}

}  // namespace llmp::engine
