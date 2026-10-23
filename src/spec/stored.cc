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

// Both preprocessing origins use catalog fold names.
std::expected<void, std::string> CheckCatalog(const artifact::Artifact& a) {
  for (auto records : {a.folds(), a.source_folds()})
    for (const artifact::AppliedFold& f : records) {
      if (FindFold(f.name) == nullptr) {
        return Refuse(std::format("artifact {} fold {}", a.id(), f.name),
                      "not in the fold catalog, so what its tensors mean is unknown");
      }
    }
  return {};
}

// Both origins' folds of artifact `index` (`a`) against the spec: catalog
// names, together exactly the folds of every stage that runs from the artifact
// (a stage's folds are its own artifact's, D-113), and each naming a
// tensor (by resource or expert array name) that some binding of the spec
// reads from the artifact, so that a fold record describes tensors the
// stages use.
std::expected<void, std::string> CheckApplied(const artifact::Artifact& a, std::uint32_t index,
                                              const ModelSpec& spec) {
  if (auto ok = CheckCatalog(a); !ok) {
    return ok;
  }
  std::set<std::string_view> recorded;
  for (auto records : {a.folds(), a.source_folds()})
    for (const auto& fold : records) recorded.insert(fold.name);
  for (const auto& stage : spec.stages)
    if (stage.artifact == index && !std::ranges::equal(stage.folds, recorded))
      return Refuse(a.id(), "stored stage folds disagree with preprocessing receipts");
  std::set<std::string_view> bound;
  for (const Stage& s : spec.stages) {
    EachComponent(s, [&](const ComponentRef& c, const std::string&) {
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
  for (auto records : {a.folds(), a.source_folds()})
    for (const artifact::AppliedFold& f : records) {
      const std::string where = std::format("artifact {} fold {}", a.id(), f.name);
      for (const Stage& s : spec.stages) {
        if (s.artifact == index && !std::ranges::contains(s.folds, f.name)) {
          return Refuse(
              where, std::format("applied by the importer but not among stage {}'s folds", s.name));
        }
      }
      if (std::ranges::any_of(f.tensors,
                              [&](const std::string& t) { return !bound.contains(t); })) {
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
  const bool is_drafter = spec->stages.size() == 2 && spec->stages[0].role == StageRole::kTarget &&
                          spec->stages[0].artifact == 0 &&
                          spec->stages[1].role == StageRole::kDrafter &&
                          spec->stages[1].artifact == 1;
  if (!is_drafter) {
    for (const Stage& s : spec->stages) {
      if (s.artifact != 0) {
        return Refuse(
            where, std::format("stage {} runs from artifact {}, not this one", s.name, s.artifact));
      }
    }
    if (auto ok = CheckApplied(artifact, 0, *spec); !ok) {
      return std::unexpected(ok.error());
    }
  } else {
    if (auto ok = CheckApplied(artifact, 1, *spec); !ok) {
      return std::unexpected(ok.error());
    }
  }
  return spec;
}

std::expected<void, std::string> ConfigureComposedDraftVocabulary(ModelSpec& model,
                                                                  std::uint32_t rows) {
  if (rows == 0 || model.stages.size() != 2 || model.stages.back().role != StageRole::kDrafter)
    return Refuse("draft vocabulary", "requires a positive extent and one composed drafter");
  auto& drafter = model.stages.back();
  const auto set_rows = [&](ComponentRef& c, bool add_if_missing) {
    const auto it = std::ranges::find(c.params, "rows", &Param::name);
    if (it != c.params.end()) {
      it->value = static_cast<std::int64_t>(rows);
    } else if (add_if_missing) {
      c.params.push_back({.name = "rows", .value = ParamValue(static_cast<std::int64_t>(rows))});
    }
  };
  if (drafter.heads.output.has_value()) {
    set_rows(*drafter.heads.output, true);
  }
  for (Layer& layer : drafter.layers) {
    for (ComponentRef& c : layer.components) {
      const Param* sel = c.Find("selected");
      const auto* is_sel = sel != nullptr ? std::get_if<bool>(&sel->value) : nullptr;
      if (is_sel != nullptr && *is_sel) {
        set_rows(c, false);
      }
    }
  }
  Canonicalize(model);
  return Validate(model);
}

std::expected<ModelSpec, std::string> ComposeStoredDrafter(
    const ModelSpec& target_spec, const ModelSpec& drafter_spec,
    std::optional<std::uint32_t> draft_vocab) {
  constexpr std::string_view kWhere = "drafter composition";
  if (auto ok = Validate(target_spec); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = Validate(drafter_spec); !ok) {
    return std::unexpected(ok.error());
  }
  if (target_spec.stages.size() != 1 || target_spec.stages[0].role != StageRole::kTarget ||
      target_spec.stages[0].artifact != 0) {
    return Refuse(kWhere, "target spec must have one target stage on artifact 0");
  }
  if (drafter_spec.stages.size() != 2 || drafter_spec.stages[0].role != StageRole::kTarget ||
      drafter_spec.stages[0].artifact != 0 || drafter_spec.stages[1].role != StageRole::kDrafter ||
      drafter_spec.stages[1].artifact != 1 || target_spec.pipeline != drafter_spec.pipeline) {
    return Refuse(kWhere,
                  "drafter spec must have a stub target on artifact 0 and a drafter stage on "
                  "artifact 1 of the target's pipeline");
  }
  if (std::ranges::contains(target_spec.stages, drafter_spec.stages[1].name, &Stage::name)) {
    return Refuse(kWhere,
                  std::format("target spec already has a stage {}", drafter_spec.stages[1].name));
  }
  ModelSpec out = target_spec;
  Stage& target_stage = out.stages[0];
  for (const FeatureTap& tap : drafter_spec.stages[0].taps) {
    if (tap.point == TapPoint::kPostFinalNorm
            ? (tap.layer != 0 || !target_stage.heads.final_norm.has_value())
            : tap.layer >= target_stage.layers.size()) {
      return Refuse(kWhere, std::format("tap {} at layer {} exceeds target's {} layers", tap.name,
                                        tap.layer, target_stage.layers.size()));
    }
    const auto existing = std::ranges::find(target_stage.taps, tap.name, &FeatureTap::name);
    if (existing == target_stage.taps.end()) {
      target_stage.taps.push_back(tap);
    } else if (existing->layer != tap.layer || existing->point != tap.point) {
      return Refuse(kWhere, std::format("target tap {} conflicts with the drafter's", tap.name));
    }
  }
  Stage drafter = drafter_spec.stages[1];
  const std::string_view stub_target_name = drafter_spec.stages[0].name;
  for (Read& r : drafter.reads) {
    if (r.stage == stub_target_name) {
      r.stage = target_stage.name;
    }
  }
  EachComponent(drafter, [&](ComponentRef& c, const std::string&) {
    for (WeightBinding& w : c.weights) {
      if (w.stage == stub_target_name) {
        w.stage = target_stage.name;
      }
    }
  });
  out.stages.push_back(std::move(drafter));
  if (draft_vocab) {
    auto configured = ConfigureComposedDraftVocabulary(out, *draft_vocab);
    if (!configured) return std::unexpected(configured.error());
  }
  Canonicalize(out);
  if (auto ok = Validate(out); !ok) {
    return std::unexpected(ok.error());
  }
  return out;
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
    s.preprocessing.push_back({.name = f.name, .tensors = f.tensors});
  }
  for (const artifact::AppliedFold& f : artifact.source_folds())
    s.preprocessing.push_back({.name = f.name, .tensors = f.tensors, .source = true});
  for (const artifact::Resource& r : artifact.resources()) {
    if (!r.format) {
      return Refuse(r.name, "no tensor format");  // every v1 resource has one
    }
    for (const std::string& role : r.roles) {
      s.tensors.push_back({.name = role,
                           .format = *r.format,
                           .part = r.part,
                           .experts = false,
                           .storage = role == r.name ? std::string{} : r.name});
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
