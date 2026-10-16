// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/parameter_folds.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <utility>
#include <variant>

#include "components/component.h"
#include "format/encoding.h"
#include "format/layout.h"

namespace llmp::import {
namespace {

std::unexpected<Error> Bad(std::string_view why) {
  return std::unexpected(Error{.rule = Rule::kUnsupported, .reason = why});
}

// The scalar fold that computes a catalog fold, if this importer has one.
std::optional<ScalarFold> ScalarOf(std::string_view fold) {
  if (fold == "norm.plus_one") return ScalarFold::kNormPlusOne;
  if (fold == "ssm.a_neg_exp") return ScalarFold::kANegExp;
  return std::nullopt;
}

bool FlagSet(const spec::ComponentRef& ref, std::string_view flag) {
  const spec::Param* p = ref.Find(flag);
  return p != nullptr && std::holds_alternative<bool>(p->value) && std::get<bool>(p->value);
}

// A folded tensor's F32 bytes at most: a per-channel vector, never a bulk
// weight (the header's "bulk weights never widen").
constexpr std::uint64_t kMaxFoldedBytes = std::uint64_t{1} << 20U;

// Whether the output is a whole dense BF16 vector of one copied source,
// small enough to widen.
bool Foldable(const TensorAssembly& out) {
  return !out.fold && !out.relayout_from && out.format.encoding == format::Encoding::kBF16 &&
         out.format.layout == format::Layout::kDense && out.format.stacked == 1 &&
         out.format.shape.rank == 1 && out.format.shape.ne[0] != 0 &&
         out.format.shape.ne[0] <= kMaxFoldedBytes / 4 && out.parts.size() == 1 &&
         out.parts[0].part == format::PartRole::kData &&
         out.parts[0].kind == PartAssemblyKind::kCopy && out.parts[0].sources.size() == 1;
}

}  // namespace

std::expected<std::vector<AppliedParameterFold>, Error> FoldDerivedParameters(
    const spec::Stage& stage, std::vector<TensorAssembly>& outputs) {
  std::map<std::string_view, std::uint32_t> by_name;
  for (std::uint32_t i = 0; i < outputs.size(); ++i) by_name.emplace(outputs[i].name, i);
  // Per fold: the outputs it governs, and whether it can cover them all.
  struct Governed {
    std::set<std::uint32_t> outputs;
    bool whole = true;
  };
  std::map<std::string_view, Governed> folds;
  // Every binding's tensor and the fold its role derives under its
  // instance's flags (empty: none).
  std::vector<std::pair<std::string_view, std::string_view>> bound;
  const auto visit = [&](const spec::ComponentRef& ref) -> std::expected<void, Error> {
    const components::ComponentDef* def = components::Find(ref.kind, ref.version);
    if (def == nullptr) return Bad("a parameter fold's spec names a component the catalog lacks");
    for (const spec::WeightBinding& w : ref.weights) {
      const auto role = components::RoleIndex(*def, w.role);
      if (!role) return Bad("a parameter fold's spec binds a role its component lacks");
      const components::DerivedValue& derived = def->roles[*role].derived;
      const bool derives = !derived.fold.empty() && FlagSet(ref, derived.flag);
      bound.emplace_back(w.tensor, derives ? derived.fold : std::string_view{});
      if (!derives) continue;
      if (!ScalarOf(derived.fold)) return Bad("a derived role's fold has no importer transform");
      Governed& g = folds[derived.fold];
      const auto at = by_name.find(w.tensor);
      if (!w.stage.empty() || w.part != format::PartRole::kData || w.experts ||
          at == by_name.end() || !Foldable(outputs[at->second])) {
        g.whole = false;
        continue;
      }
      g.outputs.insert(at->second);
    }
    return {};
  };
  const auto visit_optional =
      [&](const std::optional<spec::ComponentRef>& ref) -> std::expected<void, Error> {
    return ref ? visit(*ref) : std::expected<void, Error>{};
  };
  if (auto ok = visit_optional(stage.embedding); !ok) return std::unexpected(ok.error());
  if (auto ok = visit_optional(stage.stream.mixing); !ok) return std::unexpected(ok.error());
  for (const auto& side : stage.side)
    if (auto ok = visit(side.source); !ok) return std::unexpected(ok.error());
  for (const auto& layer : stage.layers)
    for (const auto& c : layer.components)
      if (auto ok = visit(c); !ok) return std::unexpected(ok.error());
  if (auto ok = visit_optional(stage.heads.final_norm); !ok) return std::unexpected(ok.error());
  if (auto ok = visit_optional(stage.heads.output); !ok) return std::unexpected(ok.error());
  for (const auto& c : stage.heads.mtp)
    if (auto ok = visit(c); !ok) return std::unexpected(ok.error());
  // A tensor bound to a role that derives another fold, or none, keeps its
  // folds unapplied: its bytes cannot serve both readings.
  for (const auto& [tensor, fold] : bound)
    for (auto& [name, g] : folds) {
      const auto at = by_name.find(tensor);
      if (name != fold && at != by_name.end() && g.outputs.contains(at->second)) g.whole = false;
    }
  // A fold the stage already carries cannot apply twice.
  std::vector<AppliedParameterFold> applied;
  for (auto& [name, g] : folds) {
    if (!g.whole || g.outputs.empty() || std::ranges::contains(stage.folds, name)) continue;
    AppliedParameterFold a{.fold = std::string(name), .outputs = {}};
    for (const std::uint32_t o : g.outputs) {
      outputs[o].fold = ScalarOf(name);
      outputs[o].format.encoding = format::Encoding::kF32;
      a.outputs.push_back(o);
    }
    applied.push_back(std::move(a));
  }
  return applied;
}

}  // namespace llmp::import
