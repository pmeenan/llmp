// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

#include "components/step.h"
#include "runtime/component_bindings.h"
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

}  // namespace llmp::runtime
