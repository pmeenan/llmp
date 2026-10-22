// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_bindings.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string_view>

#include "components/step.h"
#include "format/tensor_format.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {
std::unexpected<std::string> Refuse(std::string_view what) {
  return std::unexpected(std::string(what));
}
}  // namespace
std::expected<std::unique_ptr<StoredBindingPayment>, std::string> StoredBindingPayment::Create(
    const artifact::Artifact& artifact,
    std::shared_ptr<engine::ExecutionMetadataAllowance> allowance) {
  const auto bytes = StoredBindingBytes(artifact);
  if (!bytes) return std::unexpected(bytes.error());
  if (!allowance || !allowance->Charge(*bytes))
    return Refuse("stored binding construction was not prepaid");
  return std::unique_ptr<StoredBindingPayment>(
      new StoredBindingPayment(std::move(allowance), *bytes, std::string(artifact.id())));
}
StoredBindingPayment::~StoredBindingPayment() { allowance_->Release(bytes_); }
bool StoredBindingPayment::Names(const artifact::Artifact& artifact) const {
  return artifact_ == artifact.id();
}
std::expected<StoredComponentBindings, std::string> ReadStoredComponents(
    const artifact::Artifact& artifact, std::unique_ptr<StoredBindingPayment> payment,
    std::optional<std::pair<spec::ModelSpec, spec::Source>> pre_read) {
  if (!payment || !payment->Names(artifact))
    return Refuse("stored binding has no matching payment");
  StoredComponentBindings out(std::move(payment));
  if (pre_read.has_value()) {
    out.model = std::move(pre_read->first);
    out.source = std::move(pre_read->second);
  } else {
    auto model = spec::StoredSpec(artifact);
    if (!model) return std::unexpected(model.error());
    out.model = std::move(*model);
    auto source = spec::SourceOf(artifact);
    if (!source) return std::unexpected(source.error());
    out.source = std::move(*source);
  }
  // First consumable domain is one target in one artifact, like the existing
  // stored Gemma consumer. The component visitor itself supports compositions.
  if (out.model.stages.size() != 1 || out.model.stages[0].artifact != 0)
    return Refuse("stored runtime binding requires one target artifact stage");
  const std::array<std::span<const components::BoundTensor>, 1> tables{out.source.tensors};
  auto visited = components::VisitWeightBindings(
      out.model, tables,
      [&](const components::WeightBindingView& weight) -> std::expected<void, std::string> {
        if (weight.artifact != 0 || weight.ref.stage != 0)
          return Refuse("stored runtime weight names an unavailable artifact stage");
        if (weight.head_rows) {
          if (weight.head_rows > static_cast<std::uint64_t>(INT32_MAX) ||
              (out.logit_width && out.logit_width != weight.head_rows))
            return Refuse("stored output head vocabulary differs");
          out.logit_width = static_cast<std::uint32_t>(weight.head_rows);
        }
        for (const auto& [name, role] : weight.parts) {
          const auto index = weight.ref.expert_array ? artifact.FindExpertArray(name)
                                                     : artifact.FindResource(name);
          if (!index) return Refuse("stored weight part is missing from the opened artifact");
          if (weight.ref.expert_array) {
            const auto& array = artifact.expert_arrays()[*index];
            if (!array.format || array.format->shape.rank > 2)
              return Refuse("stored expert has no supported source slice");
            auto full = *array.format;
            full.shape.rank = 3;
            full.shape.ne[2] = array.count;
            if (full != weight.format || array.part != role)
              return Refuse("stored expert role format or count differs from its source");
            const auto& matrix = *array.format;
            // Parts are format facts. Resident grouped execution advertises
            // its own capabilities after these exact placements are bound.
            const auto parts = format::PartsOf(matrix);
            const auto part = parts ? parts->Find(role) : std::optional<format::PartExtent>{};
            if (!part || part->bytes != array.slice_bytes.value())
              return Refuse("stored expert part has no exact source slice extent");
          } else {
            const auto& resource = artifact.resources()[*index];
            const auto parts = format::PartsOf(weight.format);
            const auto part = parts ? parts->Find(role) : std::optional<format::PartExtent>{};
            if (!resource.format || *resource.format != weight.format || resource.part != role ||
                !part || part->bytes != resource.bytes.value())
              return Refuse("stored role format or extent differs from its source");
          }
          const ComponentWeightPlace place{.artifact = static_cast<std::uint16_t>(weight.artifact),
                                           .array = weight.ref.expert_array,
                                           .index = *index};
          const auto [old, inserted] = out.places.emplace(std::pair{weight.ref, role}, place);
          if (!inserted && old->second != place)
            return Refuse("stored weight role has conflicting aliases");
          if (!weight.rows_per_token || !format::RowBytes(weight.format, role)) continue;
          if (place.array) return Refuse("stored row table cannot be an expert array");
          const auto& resource = artifact.resources()[*index];
          if (!resource.rows)
            return Refuse("stored row role lacks row-addressable source placement");
          for (std::size_t i = 0; i < artifact.resources().size(); ++i)
            if (i != *index && artifact.resources()[i].group == resource.group)
              return Refuse("stored row table shares its group with another resource");
          if (std::ranges::any_of(out.rows, [&](const auto& row) {
                return row.weight == weight.ref && row.part == role;
              }))
            continue;
          out.rows.push_back({.weight = weight.ref,
                              .resource = *index,
                              .format = weight.format,
                              .part = role,
                              .rows_per_token = weight.rows_per_token});
        }
        return {};
      });
  if (!visited) return std::unexpected(visited.error());
  if (!out.logit_width) return Refuse("stored target has no bound output head");
  return out;
}
}  // namespace llmp::runtime
