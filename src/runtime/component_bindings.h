// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Stored-v1 source binding, independent of serving admission and planning.
#ifndef LLMP_RUNTIME_COMPONENT_BINDINGS_H_
#define LLMP_RUNTIME_COMPONENT_BINDINGS_H_
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "engine/execution_contract.h"
#include "engine/model_runner.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::runtime {
struct ComponentWeightPlace {
  std::uint16_t artifact = 0;
  bool array = false;
  std::uint32_t index = 0;
  bool operator==(const ComponentWeightPlace&) const = default;
};
using ComponentWeightPlaces =
    std::map<std::pair<ir::WeightRef, format::PartRole>, ComponentWeightPlace>;

// Bound from already-opened header bytes/counts, without parsing the spec or
// constructing a compiler. Payment must precede the first StoredSpec/SourceOf.
std::expected<std::uint64_t, std::string> StoredBindingBytes(const artifact::Artifact& artifact);
class StoredBindingPayment {
 public:
  static std::expected<std::unique_ptr<StoredBindingPayment>, std::string> Create(
      const artifact::Artifact&, std::shared_ptr<engine::ExecutionMetadataAllowance>);
  ~StoredBindingPayment();
  StoredBindingPayment(const StoredBindingPayment&) = delete;
  StoredBindingPayment& operator=(const StoredBindingPayment&) = delete;
  bool Names(const artifact::Artifact& artifact) const;

 private:
  StoredBindingPayment(std::shared_ptr<engine::ExecutionMetadataAllowance> allowance,
                       std::uint64_t bytes, std::string artifact)
      : allowance_(std::move(allowance)), bytes_(bytes), artifact_(std::move(artifact)) {}
  std::shared_ptr<engine::ExecutionMetadataAllowance> allowance_;
  std::uint64_t bytes_;
  std::string artifact_;
};
struct StoredComponentBindings {
  explicit StoredComponentBindings(std::unique_ptr<StoredBindingPayment> admitted)
      : payment(std::move(admitted)) {}
  StoredComponentBindings(StoredComponentBindings&&) = default;
  // Memberwise assignment would refund the old payment before replacing its
  // still-live payload. Transfer this complete owner by move construction.
  StoredComponentBindings& operator=(StoredComponentBindings&&) = delete;
  // First: all owned source/metadata dies before this local charge is refunded.
  std::unique_ptr<StoredBindingPayment> payment;
  spec::ModelSpec model;
  spec::Source source;
  ComponentWeightPlaces places;
  std::vector<engine::RunnerConfig::RowTable> rows;
  std::uint32_t logit_width = 0;
};
// Generic source validity only; does not approve any serving policy/envelope.
std::expected<StoredComponentBindings, std::string> ReadStoredComponents(
    const artifact::Artifact&, std::unique_ptr<StoredBindingPayment> payment,
    std::optional<std::pair<spec::ModelSpec, spec::Source>> pre_read = std::nullopt);
}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_BINDINGS_H_
