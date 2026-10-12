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
std::expected<std::uint64_t, std::string> StoredBindingBytes(const artifact::Artifact& artifact) {
  const auto text = artifact.spec_json();
  if (artifact.format_version() != 1 || !text || text->size() > spec::kMaxSpecBytes)
    return Refuse("stored binding requires bounded native spec bytes");
  // The strict JSON parser has 24-byte nodes + four-byte child indices.
  // At most one of each per source byte, three overlapping vector capacities.
  // Strings, numeric lists, canonicalization and owned spec growth share the
  // byte term; object records and catalog defaults use the separate term.
  std::uint64_t objects = 0;
  bool quoted = false;
  for (const char c : *text) {
    if (c == '\"')
      quoted = !quoted;  // canonical subset has no escapes
    else if (!quoted && c == '{')
      ++objects;
  }
  std::uint64_t defaults = 0;
  for (const auto* def : components::Catalog()) {
    std::uint64_t size = 0;
    for (const auto& param : def->params) {
      const auto choice = param.choices.empty() ? 0 : param.choices.front().size();
      size += 3 * (sizeof(spec::Param) + sizeof(std::string) + param.name.size() + choice + 2);
    }
    defaults = std::max(defaults, size);
  }
  // All possible spec record types fit Stage, ComponentRef, or SideInput;
  // retain the explicit maximum so additions cannot silently exceed payment.
  const auto record =
      std::max({sizeof(spec::Stage), sizeof(spec::ComponentRef), sizeof(spec::SideInput),
                sizeof(spec::WeightBinding), sizeof(components::Instance)});
  std::uint64_t bytes = sizeof(StoredComponentBindings) + sizeof(StoredBindingPayment) + 4096;
  const auto add = [&](std::uint64_t count, std::uint64_t unit) {
    std::uint64_t size = 0;
    return !__builtin_mul_overflow(count, unit, &size) &&
           !__builtin_add_overflow(bytes, size, &bytes) && bytes <= SIZE_MAX;
  };
  if (!add(text->size(), 3 * (32 + sizeof(std::int64_t)) + 16) ||
      !add(objects, 8 * record + 4 * defaults))
    return Refuse("stored binding spec inventory overflows");
  std::uint64_t tensors = 0;
  const auto tensor = [&](std::string_view name) {
    if (++tensors > spec::kMaxSourceTensors) return false;
    return add(1, 4 * sizeof(spec::SourceTensor) + 4 * sizeof(components::WeightBindingView) +
                      4 * sizeof(ComponentWeightPlaces::value_type) +
                      4 * sizeof(engine::RunnerConfig::RowTable) + 256) &&
           add(name.size() + 1, 8);
  };
  for (const auto& r : artifact.resources())
    for (const auto& role : r.roles)
      if (!tensor(role)) return Refuse("stored binding source inventory overflows");
  for (const auto& a : artifact.expert_arrays())
    if (!tensor(a.name)) return Refuse("stored binding source inventory overflows");
  for (const auto& fold : artifact.folds())
    if (!add(fold.name.size() + 1, 8)) return Refuse("stored binding fold inventory overflows");
  if (!add(artifact.model().architecture.size() + artifact.model().container.size() + 2, 8) ||
      !add(artifact.id().size() + 1, 2))
    return Refuse("stored binding source identity inventory overflows");
  if (!add(artifact.groups().size(), 64) || !add(artifact.expert_arrays().size(), 256))
    return Refuse("stored binding placement inventory overflows");
  return bytes;
}
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
    const artifact::Artifact& artifact, std::unique_ptr<StoredBindingPayment> payment) {
  if (!payment || !payment->Names(artifact))
    return Refuse("stored binding has no matching payment");
  StoredComponentBindings out(std::move(payment));
  auto model = spec::StoredSpec(artifact);
  if (!model) return std::unexpected(model.error());
  out.model = std::move(*model);
  auto source = spec::SourceOf(artifact);
  if (!source) return std::unexpected(source.error());
  out.source = std::move(*source);
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
            if (role != format::PartRole::kData || (matrix.layout != format::Layout::kDense &&
                                                    matrix.layout != format::Layout::kGgmlBlocks))
              return Refuse("stored expert placement does not yet support multipart slabs");
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
