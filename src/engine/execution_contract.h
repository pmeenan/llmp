// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Immutable execution compatibility and numerical qualification. Neither is a
// model spec identity, a request shape/cache key, nor permission to read state.
// These are owned value types. ComponentStepSource composes them from owned
// backend declarations; complete operation admission and numerical evidence
// remain separate requirements. This header contains no vendor types.

#ifndef LLMP_ENGINE_EXECUTION_CONTRACT_H_
#define LLMP_ENGINE_EXECUTION_CONTRACT_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "catalog/catalog.h"
#include "execution/registry.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::engine {

// Fixed pre-start host payment for the backend and its source declarations.
// Admission pins the entire allowance in the node catalog before allocating
// this owner. Local charges never exceed it, including temporary construction.
// The pinned extent stays charged until the node is destroyed: borrower
// destruction must not mutate a running scheduler's catalog. A build retains
// and reuses one owner across failed/retried setup, rather than admitting again.
class ExecutionMetadataAllowance {
 public:
  static std::expected<std::shared_ptr<ExecutionMetadataAllowance>, std::string> Admit(
      catalog::Catalog& catalog, catalog::DomainId domain, std::uint64_t bytes,
      std::uint64_t available, std::uint64_t margin);
  ExecutionMetadataAllowance(const ExecutionMetadataAllowance&) = delete;
  ExecutionMetadataAllowance& operator=(const ExecutionMetadataAllowance&) = delete;
  bool Charge(std::uint64_t bytes);
  void Release(std::uint64_t bytes);
  std::uint64_t used() const { return used_; }
  std::uint64_t capacity() const { return capacity_; }

 private:
  explicit ExecutionMetadataAllowance(std::uint64_t capacity) : capacity_(capacity) {}
  std::uint64_t capacity_ = 0, used_ = 0;
};

enum class NumericalUse : std::uint8_t {
  kPrompt,
  kDecode,
  kScore,
  kPromptCompanions,
  kSpeculative,
  kBlock,
};

struct NumericalPhase {
  planner::Phase phase = planner::Phase::kPrefill;
  std::uint32_t max_owners = 0;
  std::uint32_t max_owner_rows = 0;
  std::uint64_t max_total_rows = 0;
  bool row_invariant = false;
  bool operator==(const NumericalPhase&) const = default;
};

struct NumericalStage {
  std::uint16_t stage = 0;
  std::string name;
  // Exact declared set; factory validates ordering and duplicates. These are
  // operation-wide modes, independent of the shape's immediate phase.
  std::vector<NumericalUse> uses;
  std::vector<NumericalPhase> phases;
  bool operator==(const NumericalStage&) const = default;
};

class NumericalDomain {
 public:
  // Every stage of the source composition, in stage order starting at zero.
  // Empty or unknown declarations fail closed. Legacy sources do not construct
  // this required-domain object and keep their existing explicit legacy path.
  static std::expected<NumericalDomain, std::string> Create(std::vector<NumericalStage> stages);

  std::span<const NumericalStage> stages() const { return stages_; }
  const base::Sha256Digest& identity() const { return identity_; }
  std::expected<void, std::string> CheckMode(NumericalUse use) const;
  std::expected<void, std::string> CheckShape(const planner::StepShape& shape) const;
  // The caller supplies all possible shapes of the operation, including both
  // futures, companions, owed settle/commit and adaptive alternatives. This
  // pure value check does not discover omitted work or evaluate callbacks.
  std::expected<void, std::string> CheckUse(NumericalUse use,
                                            std::span<const planner::StepShape> shapes) const;

 private:
  NumericalDomain(std::vector<NumericalStage> stages, base::Sha256Digest identity)
      : stages_(std::move(stages)), identity_(identity) {}
  std::vector<NumericalStage> stages_;
  base::Sha256Digest identity_{};
};

// Producer authority belongs to the actual backend owner. The qualified
// build/install receipt and actual immutable runtime configuration define
// compatibility; this value does not perform runtime code attestation.
using BackendExecutionDeclaration = execution::BackendDeclaration;

// Borrowed only during Create; nothing in the finished object retains a span,
// registry pointer, tuning string_view, callback or caller-owned configuration.
struct StageExecutionInput {
  std::uint16_t stage = 0;
  std::string name;
  planner::Policy policy;
  base::Sha256Digest patterns{};
  const execution::Registry* registry = nullptr;
  planner::DeviceFacts device;
  BackendExecutionDeclaration backend;
  // Actual provider guarantee, zero when unknown. No physical address belongs
  // in this identity. Placement alignment affects selection independently.
  std::uint64_t reservation_alignment = 0;
  std::uint64_t placement_alignment = 256;
};

class ExecutionContract {
 public:
  // A required contract seals the whole ordered composition, including any
  // legacy companion. Missing backend/registry/stage facts are refused.
  static std::expected<ExecutionContract, std::string> Create(
      NumericalDomain domain, std::span<const StageExecutionInput> stages);

  const base::Sha256Digest& identity() const { return identity_; }
  const NumericalDomain& domain() const { return domain_; }

 private:
  ExecutionContract(NumericalDomain domain, base::Sha256Digest identity)
      : domain_(std::move(domain)), identity_(identity) {}
  NumericalDomain domain_;
  base::Sha256Digest identity_{};
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_EXECUTION_CONTRACT_H_
