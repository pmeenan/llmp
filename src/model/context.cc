// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"

namespace llmp::model {
namespace {

Bytes InDomain(const catalog::Closure& closure, catalog::DomainId domain) {
  const auto found = closure.bytes_by_domain.find(domain);
  return found != closure.bytes_by_domain.end() ? found->second : Bytes();
}

}  // namespace

std::string ToString(ComponentRole role) {
  switch (role) {
    case ComponentRole::kMain:
      return "main";
    case ComponentRole::kDrafter:
      return "drafter";
    case ComponentRole::kEncoder:
      return "encoder";
  }
  return "unknown";
}

std::string ToString(ContextError error) {
  switch (error) {
    case ContextError::kNoMain:
      return "a model context needs exactly one main component";
    case ContextError::kEmpty:
      return "a component names no resources";
    case ContextError::kDuplicate:
      return "the same role of the same artifact appears twice";
    case ContextError::kBadResource:
      return "a component names an invalid resource";
  }
  return "unknown context error";
}

std::expected<ModelContext, ContextError> ModelContext::Create(std::vector<Component> components) {
  if (std::ranges::count(components, ComponentRole::kMain, &Component::role) != 1) {
    return std::unexpected(ContextError::kNoMain);
  }
  for (const Component& component : components) {
    if (component.resources.empty()) {
      return std::unexpected(ContextError::kEmpty);
    }
    if (!std::ranges::all_of(component.resources,
                             [](catalog::ResourceId id) { return id.valid(); })) {
      return std::unexpected(ContextError::kBadResource);
    }
  }
  // The identity over (role, artifact) in canonical order, so the order
  // components were listed in does not change it.
  std::vector<std::pair<ComponentRole, base::Sha256Digest>> keys;
  keys.reserve(components.size());
  for (const Component& component : components) {
    keys.emplace_back(component.role, component.artifact);
  }
  std::ranges::sort(keys);
  if (std::ranges::adjacent_find(keys) != keys.end()) {
    return std::unexpected(ContextError::kDuplicate);
  }
  base::Sha256 hash;
  hash.Update("llmp.model_context.v0");
  for (const auto& [role, artifact] : keys) {
    const std::array<std::uint8_t, 1> tag = {static_cast<std::uint8_t>(role)};
    hash.Update(std::as_bytes(std::span(tag)));
    hash.Update(std::as_bytes(std::span(artifact)));
  }
  return ModelContext(std::move(components), hash.Finish());
}

std::vector<catalog::ResourceId> ModelContext::ResourcesOf(
    std::span<const ComponentRole> roles) const {
  std::vector<catalog::ResourceId> resources;
  for (const Component& component : components_) {
    if (std::ranges::find(roles, component.role) != roles.end()) {
      resources.insert(resources.end(), component.resources.begin(), component.resources.end());
    }
  }
  std::ranges::sort(resources);
  const auto [first, last] = std::ranges::unique(resources);
  resources.erase(first, last);
  return resources;
}

std::expected<catalog::Closure, catalog::CatalogError> ModelContext::ClosureOf(
    const catalog::Catalog& catalog, std::span<const ComponentRole> roles) const {
  return catalog.ClosureOf(ResourcesOf(roles));
}

std::expected<Bytes, catalog::CatalogError> ModelContext::SharedBytes(
    const catalog::Catalog& catalog, catalog::DomainId domain) const {
  std::vector<catalog::ResourceId> all;
  Bytes summed;
  for (const Component& component : components_) {
    auto closure = catalog.ClosureOf(component.resources);
    if (!closure) {
      return std::unexpected(closure.error());
    }
    const std::optional<Bytes> sum = summed.Plus(InDomain(*closure, domain));
    if (!sum) {
      return std::unexpected(catalog::CatalogError::kBadRange);
    }
    summed = *sum;
    all.insert(all.end(), component.resources.begin(), component.resources.end());
  }
  auto united = catalog.ClosureOf(all);
  if (!united) {
    return std::unexpected(united.error());
  }
  return summed.Minus(InDomain(*united, domain)).value_or(Bytes());
}

}  // namespace llmp::model
