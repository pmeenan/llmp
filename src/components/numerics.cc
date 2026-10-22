// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/numerics.h"

#include <algorithm>
#include <format>

#include "components/component.h"

namespace llmp::components {

std::expected<void, std::string> ValidateNumericalSites(const ComponentDef& def) {
  const auto error = [&](std::string_view what) {
    return std::unexpected(std::format("{}: invalid numerical site ({})", def.kind, what));
  };
  for (const auto& site : def.numerical) {
    if (site.semantic.empty() || ir::NumericalChangeName(site.change).empty() ||
        site.anchors.empty() || site.encodings.empty())
      return error("semantic, change, anchors and encodings must be explicit");
    const auto known_anchor = [](ir::OpKey anchor) {
      return ir::OpName(anchor.op) != "unknown" &&
             (anchor.op == ir::Op::kFused ? anchor.fused != ir::FusedKind::kNone &&
                                                anchor.fused != ir::FusedKind::kGroupedStores &&
                                                ir::FusedKindName(anchor.fused) != "unknown"
                                          : anchor.fused == ir::FusedKind::kNone);
    };
    for (const auto origin : site.origins)
      if (!known_anchor(origin)) return error("unknown origin anchor");
    for (const auto anchor : site.anchors)
      if (!known_anchor(anchor)) return error("unknown anchor");
    for (const auto encoding : site.encodings) {
      if (std::ranges::find(format::Encodings(), encoding, &format::EncodingInfo::id) ==
          format::Encodings().end())
        return error("unknown encoding");
    }
    for (const auto role : site.roles) {
      const auto found = std::ranges::find(def.roles, role, &WeightRole::name);
      if (found == def.roles.end()) return error("unknown weight role");
      for (const auto encoding : site.encodings)
        if (!found->encodings.empty() && !std::ranges::contains(found->encodings, encoding))
          return error("encoding outside weight role domain");
    }
    for (const auto& condition : site.when) {
      const auto found = std::ranges::find(def.params, condition.parameter, &ParamSpec::name);
      if (found == def.params.end() ||
          (condition.flag ? found->type != ParamType::kBool || !condition.choice.empty()
                          : found->type != ParamType::kName ||
                                !std::ranges::contains(found->choices, condition.choice)))
        return error("unknown semantic parameter or choice");
    }
  }
  return {};
}

bool NumericalSiteApplies(const NumericalSite& site, const Instance& instance, ir::OpKey anchor,
                          std::string_view role, format::Encoding encoding) {
  if (!std::ranges::contains(instance.def->numerical, &site,
                             [](const NumericalSite& declared) { return &declared; }) ||
      !std::ranges::contains(site.anchors, anchor) ||
      !std::ranges::contains(site.encodings, encoding) ||
      (site.roles.empty() ? !role.empty() : !std::ranges::contains(site.roles, role)))
    return false;
  return std::ranges::all_of(site.when, [&](const NumericalCondition& condition) {
    return condition.flag ? instance.params.Flag(condition.parameter) == *condition.flag
                          : instance.params.Name(condition.parameter) == condition.choice;
  });
}

}  // namespace llmp::components
