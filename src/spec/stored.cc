// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/stored.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "format/tensor_format.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::spec {
namespace {

static_assert(artifact::kMaxSpecBytes == kMaxSpecBytes,
              "the artifact reader reads every spec FromJson takes");

std::unexpected<std::string> Refuse(std::string_view where, std::string_view what) {
  return std::unexpected(std::format("{}: {}", where, what));
}

// The importer's folds are catalog names.
std::expected<void, std::string> CheckCatalog(const artifact::Artifact& a) {
  for (const artifact::AppliedFold& f : a.folds()) {
    if (FindFold(f.name) == nullptr) {
      return Refuse(std::format("artifact {} fold {}", a.id(), f.name),
                    "not in the fold catalog, so what its tensors mean is unknown");
    }
  }
  return {};
}

template <typename F>
void EachComponent(const Stage& s, const F& f) {
  const auto maybe = [&](const std::optional<ComponentRef>& c) {
    if (c) {
      f(*c);
    }
  };
  maybe(s.embedding);
  maybe(s.stream.mixing);
  for (const SideInput& in : s.side) {
    f(in.source);
  }
  for (const Layer& l : s.layers) {
    for (const ComponentRef& c : l.components) {
      f(c);
    }
  }
  maybe(s.heads.final_norm);
  maybe(s.heads.output);
  for (const ComponentRef& c : s.heads.mtp) {
    f(c);
  }
}

// The importer's folds of artifact `index` (`a`) against the spec: catalog
// names, each among the folds of every stage that runs from the artifact
// (a stage's folds are its own artifact's, D-113), and each naming a
// tensor (by resource or expert array name) that some binding of the spec
// reads from the artifact, so that a fold record describes tensors the
// stages use.
std::expected<void, std::string> CheckApplied(const artifact::Artifact& a, std::uint32_t index,
                                              const ModelSpec& spec) {
  if (auto ok = CheckCatalog(a); !ok) {
    return ok;
  }
  if (a.folds().empty()) {
    return {};
  }
  std::set<std::string_view> bound;
  for (const Stage& s : spec.stages) {
    EachComponent(s, [&](const ComponentRef& c) {
      for (const WeightBinding& w : c.weights) {
        // Validated: every binding's holding stage is the spec's.
        const Stage* holder = HoldingStage(spec, s, w);
        if (holder == nullptr || holder->artifact != index) {
          continue;
        }
        if (w.experts) {
          bound.insert(w.tensor);
        } else if (const auto r = a.FindResource(w.tensor); r) {
          bound.insert(a.resources()[*r].name);
        }
      }
    });
  }
  for (const artifact::AppliedFold& f : a.folds()) {
    const std::string where = std::format("artifact {} fold {}", a.id(), f.name);
    for (const Stage& s : spec.stages) {
      if (s.artifact == index && !std::ranges::contains(s.folds, f.name)) {
        return Refuse(
            where, std::format("applied by the importer but not among stage {}'s folds", s.name));
      }
    }
    if (std::ranges::none_of(f.tensors, [&](const std::string& t) { return bound.contains(t); })) {
      return Refuse(where, "names no tensor a stage of the spec binds from the artifact");
    }
  }
  return {};
}

// The stored bytes are the spec's canonical JSON, so that their SHA-256,
// which the manifest records, is the spec's digest.
std::expected<ModelSpec, std::string> ReadStored(std::string_view where, std::string_view text) {
  auto spec = FromJson(text);
  if (!spec) {
    return std::unexpected(std::format("{} spec.json: {}", where, spec.error()));
  }
  if (ToJson(*spec) != text) {
    return Refuse(std::format("{} spec.json", where), "not the spec's own canonical JSON");
  }
  return spec;
}

}  // namespace

std::expected<ModelSpec, std::string> StoredSpec(const artifact::Artifact& artifact) {
  const std::string where = std::format("artifact {}", artifact.id());
  if (artifact.format_version() == 0) {
    return Refuse(where, "a v0 artifact stores no spec (re-import it with llmp import, D-118)");
  }
  const auto text = artifact.spec_json();
  if (!text) {
    return Refuse(where, "carries no spec of its own (a composition's describes it)");
  }
  auto spec = ReadStored(where, *text);
  if (!spec) {
    return spec;
  }
  for (const Stage& s : spec->stages) {
    if (s.artifact != 0) {
      return Refuse(
          where, std::format("stage {} runs from artifact {}, not this one", s.name, s.artifact));
    }
  }
  if (auto ok = CheckApplied(artifact, 0, *spec); !ok) {
    return std::unexpected(ok.error());
  }
  return spec;
}

std::expected<ModelSpec, std::string> StoredSpec(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components) {
  const std::string where = std::format("composition {}", composition.id());
  if (composition.format_version() == 0) {
    return Refuse(where, "a v0 composition stores no spec");
  }
  const auto text = composition.spec_json();
  if (!text) {
    return Refuse(where, "carries no spec");
  }
  const auto& named = composition.components();
  if (components.size() != named.size()) {
    return Refuse(where,
                  std::format("{} components opened for its {}", components.size(), named.size()));
  }
  for (std::size_t i = 0; i < named.size(); ++i) {
    const artifact::Artifact* a = components[i];
    if (a == nullptr || a->id() != named[i].artifact ||
        a->model().architecture != named[i].architecture) {
      return Refuse(
          where, std::format("component {} ({}) is not the artifact it names", i, named[i].role));
    }
  }
  auto spec = ReadStored(where, *text);
  if (!spec) {
    return spec;
  }
  for (const Stage& s : spec->stages) {
    if (s.artifact >= named.size()) {
      return Refuse(where, std::format("stage {} runs from artifact {} of {}", s.name, s.artifact,
                                       named.size()));
    }
  }
  for (std::size_t i = 0; i < components.size(); ++i) {
    if (auto ok = CheckApplied(*components[i], static_cast<std::uint32_t>(i), *spec); !ok) {
      return std::unexpected(ok.error());
    }
  }
  return spec;
}

std::expected<Source, std::string> SourceOf(const artifact::Artifact& artifact) {
  if (artifact.format_version() != 1) {
    return Refuse(std::format("artifact {}", artifact.id()),
                  std::format("format version {}: a source is a v1 artifact's (re-import it with "
                              "llmp import, D-118)",
                              artifact.format_version()));
  }
  if (auto ok = CheckCatalog(artifact); !ok) {
    return std::unexpected(ok.error());
  }
  Source s{.architecture = artifact.model().architecture,
           .container = artifact.model().container,
           .tensors = {},
           .folds = {}};
  for (const artifact::AppliedFold& f : artifact.folds()) {
    s.folds.push_back(f.name);
  }
  for (const artifact::Resource& r : artifact.resources()) {
    if (!r.format) {
      return Refuse(r.name, "no tensor format");  // every v1 resource has one
    }
    for (const std::string& role : r.roles) {
      s.tensors.push_back({.name = role, .format = *r.format, .part = r.part, .experts = false});
    }
  }
  for (const artifact::ExpertArray& a : artifact.expert_arrays()) {
    if (!a.format || a.format->shape.rank > 2) {
      return Refuse(a.name, "no slice format of at most two dimensions");
    }
    format::TensorFormat f = *a.format;
    f.shape.ne[2] = a.count;
    f.shape.rank = 3;
    s.tensors.push_back({.name = a.name, .format = f, .part = a.part, .experts = true});
  }
  if (s.tensors.size() > kMaxSourceTensors) {
    return Refuse("source", std::format("more than {} tensors", kMaxSourceTensors));
  }
  return s;
}

}  // namespace llmp::spec
