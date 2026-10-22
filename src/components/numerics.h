// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_NUMERICS_H_
#define LLMP_COMPONENTS_NUMERICS_H_

#include <array>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "format/encoding.h"
#include "ir/numerics.h"
#include "ir/op.h"

namespace llmp::components {

struct ComponentDef;
struct Instance;

// A semantic choice or flag, checked against the component's parameter domain.
// Conditions are ANDed; different alternatives use separate site declarations.
struct NumericalCondition {
  std::string_view parameter;
  std::string_view choice{};
  std::optional<bool> flag{};
};

// A component's arithmetic site, independent of checkpoint, container, device
// or evidence. This declares where a rewrite's requirement makes sense; it
// never grants permission to change arithmetic. Evidence is joined separately.
struct NumericalSite {
  std::string_view semantic;
  ir::NumericalChange change;
  std::span<const ir::OpKey> anchors;
  // Exact component weight-role names. Empty means an activation-only site.
  std::span<const std::string_view> roles{};
  // Stored weight encoding for a role-backed site; source activation encoding
  // for a site without weights. Empty never means every encoding.
  std::span<const format::Encoding> encodings;
  std::span<const NumericalCondition> when{};
  // Primitive origin of an anchor introduced by structural planning. Empty:
  // the declared anchor itself. Keeps origin ownership in the component.
  std::span<const ir::OpKey> origins{};
};

// Validates compiled declarations against their component's parameter and
// role domains. Called during ordinary instance construction, before shapes.
std::expected<void, std::string> ValidateNumericalSites(const ComponentDef& def);
// Exact matching of a declared site against a validated instance. `site` must
// be an entry in instance.def->numerical, not a borrowed foreign declaration.
// This has no
// evidence/device fallback and does not authorize a planner transformation.
bool NumericalSiteApplies(const NumericalSite& site, const Instance& instance, ir::OpKey anchor,
                          std::string_view role, format::Encoding encoding);

namespace numerics {
using format::Encoding;
using ir::NumericalChange;
inline constexpr std::array kProductAnchors{ir::Primitive(ir::Op::kMatMul)};
inline constexpr std::array kExpertAnchors{ir::Primitive(ir::Op::kMatMulId)};
inline constexpr std::array kNormAnchors{ir::Primitive(ir::Op::kRmsNorm)};
inline constexpr std::array kReduceAnchors{ir::Primitive(ir::Op::kExpertReduce)};
inline constexpr std::array kGluAnchors{ir::Primitive(ir::Op::kGlu)};
inline constexpr std::array kAttentionAnchors{ir::Primitive(ir::Op::kAttention)};
inline constexpr std::array kMultiplyAnchors{ir::Primitive(ir::Op::kMul)};
inline constexpr std::array kCarryAnchors{ir::Fused(ir::FusedKind::kLayerNormModulateSplit)};
inline constexpr std::array kFloatEncodings{Encoding::kF32, Encoding::kF16, Encoding::kBF16};
inline constexpr std::array kPreparedWeightEncodings{Encoding::kMxfp8, Encoding::kNvfp4};
inline constexpr std::array kNvfp4Encoding{Encoding::kNvfp4};
// Block-quantized weight encodings whose product consumes a block32 Q8 input.
// Capability checks retain the individual implementation's narrower limits.
inline constexpr std::array kBlockWeightEncodings{
    Encoding::kQ1_0,  Encoding::kQ2_0,    Encoding::kQ4_0,   Encoding::kQ4_1,  Encoding::kQ5_0,
    Encoding::kQ5_1,  Encoding::kQ8_0,    Encoding::kQ8_1,   Encoding::kQ2_K,  Encoding::kQ3_K,
    Encoding::kQ4_K,  Encoding::kQ5_K,    Encoding::kQ6_K,   Encoding::kQ8_K,  Encoding::kIQ1_S,
    Encoding::kIQ1_M, Encoding::kIQ2_XXS, Encoding::kIQ2_XS, Encoding::kIQ2_S, Encoding::kIQ3_XXS,
    Encoding::kIQ3_S, Encoding::kIQ4_NL,  Encoding::kIQ4_XS, Encoding::kTQ1_0, Encoding::kTQ2_0,
    Encoding::kMxfp4};
}  // namespace numerics
}  // namespace llmp::components

#endif  // LLMP_COMPONENTS_NUMERICS_H_
