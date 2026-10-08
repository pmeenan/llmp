// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model context's components (D-068; docs/architecture.md#model-lifecycle):
// the runtime's unit of execution composes one or more components, each an
// installed artifact or a declared part of one, in roles. A stored MTP
// layer is a drafter component of the main artifact; a companion drafter
// is a drafter component of its own artifact. Internal: how a manifest
// references another artifact stays open (artifact-format.md), and the
// operation-contract decision (backend-proof P6) settles these types.
//
// Each component names the catalog resources it reads. A resource that
// components share, such as a drafter's use of its target's embedding
// table, is one resource named by both: a closure over several components
// holds its extents once, so it is leased and charged once.
//
// The context's identity covers every component's role and artifact, in a
// canonical order: retained state that depends on the context depends on
// every component (D-055, D-068).

#ifndef LLMP_MODEL_CONTEXT_H_
#define LLMP_MODEL_CONTEXT_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"

namespace llmp::model {

using base::Bytes;

enum class ComponentRole : std::uint8_t {
  kMain,     // the model whose output the request returns
  kDrafter,  // a speculative drafter: stored MTP layers or a companion artifact
  kEncoder,  // a modality encoder or projector
};

std::string ToString(ComponentRole role);

struct Component {
  ComponentRole role = ComponentRole::kMain;
  base::Sha256Digest artifact{};  // the artifact's identity (D-056)
  std::vector<catalog::ResourceId> resources;
};

enum class ContextError : std::uint8_t {
  kNoMain,       // not exactly one main component
  kEmpty,        // a component with no resources
  kDuplicate,    // the same role of the same artifact twice
  kBadResource,  // an invalid resource identity
};

std::string ToString(ContextError error);

class ModelContext {
 public:
  static std::expected<ModelContext, ContextError> Create(std::vector<Component> components);

  std::span<const Component> components() const { return components_; }
  const base::Sha256Digest& identity() const { return identity_; }

  // Every resource of the components in `roles`, each once.
  std::vector<catalog::ResourceId> ResourcesOf(std::span<const ComponentRole> roles) const;
  // Their closure: each extent once, however many components share it.
  std::expected<catalog::Closure, catalog::CatalogError> ClosureOf(
      const catalog::Catalog& catalog, std::span<const ComponentRole> roles) const;
  // The bytes in `domain` that charging each component's closure
  // separately would count more than once: the components' closures
  // summed, less their union's.
  std::expected<Bytes, catalog::CatalogError> SharedBytes(const catalog::Catalog& catalog,
                                                          catalog::DomainId domain) const;

 private:
  ModelContext(std::vector<Component> components, base::Sha256Digest identity)
      : components_(std::move(components)), identity_(identity) {}

  std::vector<Component> components_;
  base::Sha256Digest identity_;
};

}  // namespace llmp::model

#endif  // LLMP_MODEL_CONTEXT_H_
