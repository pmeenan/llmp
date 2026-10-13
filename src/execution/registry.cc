// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/registry.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"

namespace llmp::execution {
namespace {

std::unexpected<PlanRejection> Rejected(PlanError error, std::size_t operation,
                                        std::string detail) {
  return std::unexpected(
      PlanRejection{.error = error, .operation = operation, .detail = std::move(detail)});
}

// Fields are length-prefixed, so no two sequences of fields hash alike by
// moving bytes from one field to the next.
void Field(base::Sha256& hash, std::string_view text) {
  std::array<std::uint8_t, 8> length{};
  for (std::size_t i = 0; i < length.size(); ++i) {
    length[i] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(text.size()) >> (8 * i));
  }
  hash.Update(std::as_bytes(std::span(length)));
  hash.Update(text);
}

bool ValidName(std::string_view name) {
  return !name.empty() && name.size() <= Registry::kMaxName &&
         std::ranges::all_of(name, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_';
         });
}

// The registry index of the implementation a plan names for its operation
// at `position`: never another implementation, whatever the registry has.
std::expected<std::size_t, PlanRejection> Lookup(const Registry& registry, Operation operation,
                                                 std::string_view name, std::size_t position) {
  if (!ValidName(name)) {
    return Rejected(PlanError::kInvalid, position, "not an implementation name");
  }
  const auto found = registry.Find(name);
  if (!found) {
    return Rejected(PlanError::kUnsupported, position,
                    std::format("no implementation {} in this build", name));
  }
  const Implementation& implementation = registry.at(*found);
  if (implementation.operation != operation) {
    return Rejected(PlanError::kWrongOperation, position,
                    std::format("{} implements {}, not {}", name,
                                OperationName(implementation.operation), OperationName(operation)));
  }
  return *found;
}

// A capability's identity text: every field but the predicate.
std::string CapabilityText(const Capability& c) {
  std::string text = std::format("op {}; operands", ir::OpKeyName(c.op));
  for (const OperandRule& rule : c.operands) {
    text += std::format(" [{}:", rule.operand);
    for (const format::Encoding e : rule.encodings) {
      text += std::format(" {}", format::Info(e).name);
    }
    text += " /";
    for (const format::Layout l : rule.layouts) {
      text += std::format(" {}", format::Info(l).name);
    }
    text += "]";
  }
  text += std::format(
      "; device {}-{} needs {}; rows {}-{} k%{} n%{}; rung {}; expansion {}; lossy {}; "
      "row_invariant {}; capturable {}; group {}; priority {}",
      c.device.min_arch, c.device.max_arch, c.device.needs, c.shape.min_rows, c.shape.max_rows,
      c.shape.k_multiple, c.shape.n_multiple, static_cast<int>(c.rung),
      static_cast<int>(c.expansion), c.lossy, c.row_invariant, c.capturable, c.group, c.priority);
  // Added only when set, so that the identities of capabilities without
  // them are unchanged.
  if (c.needs_state_span) text += "; needs contiguous state span";
  if (c.needs_tuning) {
    text += "; needs tuning";
  }
  if (c.batch != 0) {
    text += std::format("; batch {}", c.batch);
  }
  if (!c.launches.empty()) {
    text += "; launches";
    for (const std::string_view l : c.launches) {
      text += std::format(" {}", l);
    }
  }
  return text;
}

// A primitive op, or kFused with the kind it names; both in the op set.
bool ValidOpKey(ir::OpKey key) {
  return ir::OpName(key.op) != "unknown" && ir::FusedKindName(key.fused) != "unknown" &&
         (key.op == ir::Op::kFused) == (key.fused != ir::FusedKind::kNone);
}

bool ValidCapability(const Capability& c) {
  if (!ValidOpKey(c.op)) {
    return false;
  }
  const bool known_rung = c.rung == Rung::kNative || c.rung == Rung::kOnChipDecode ||
                          c.rung == Rung::kBf16Decode || c.rung == Rung::kPrimitive;
  if (!known_rung || static_cast<std::uint8_t>(c.expansion) > 1 ||
      (!c.group.empty() && !ValidName(c.group)) || c.shape.min_rows > c.shape.max_rows ||
      c.shape.k_multiple == 0 || c.shape.n_multiple == 0 || c.device.min_arch > c.device.max_arch) {
    return false;
  }
  for (std::size_t i = 0; i < c.operands.size(); ++i) {
    for (std::size_t j = i + 1; j < c.operands.size(); ++j) {
      if (c.operands[i].operand == c.operands[j].operand) {
        return false;
      }
    }
    for (const format::Encoding e : c.operands[i].encodings) {
      if (static_cast<std::size_t>(e) >= format::Encodings().size()) {
        return false;
      }
    }
    for (const format::Layout l : c.operands[i].layouts) {
      if (static_cast<std::size_t>(l) >= format::Layouts().size()) {
        return false;
      }
    }
  }
  return true;
}

// Every rule that constrains an operand finds it in the query, in a format
// the rule takes: a query that leaves a constrained operand out does not
// match (an operand no rule names is unconstrained).
bool OperandsMatch(const Capability& c, std::span<const OperandFormat> operands) {
  for (const OperandRule& rule : c.operands) {
    if (rule.encodings.empty() && rule.layouts.empty()) {
      continue;
    }
    const auto operand = std::ranges::find_if(
        operands, [&](const OperandFormat& o) { return o.operand == rule.operand; });
    if (operand == operands.end() ||
        (!rule.encodings.empty() &&
         std::ranges::find(rule.encodings, operand->encoding) == rule.encodings.end()) ||
        (!rule.layouts.empty() &&
         std::ranges::find(rule.layouts, operand->layout) == rule.layouts.end())) {
      return false;
    }
  }
  return true;
}

bool Computes(const Capability& c, const CapabilityQuery& q) {
  if (c.op != q.op || !OperandsMatch(c, q.operands)) {
    return false;
  }
  if (q.arch < c.device.min_arch || q.arch > c.device.max_arch ||
      (q.features & c.device.needs) != c.device.needs) {
    return false;
  }
  if (q.rows < c.shape.min_rows || q.rows > c.shape.max_rows || q.k % c.shape.k_multiple != 0 ||
      q.n % c.shape.n_multiple != 0) {
    return false;
  }
  // D-108: a decode-shaped product reads each weight once, compressed.
  if (c.expansion == Expansion::kTransientWorkspace && q.rows <= q.policy.max_decode_rows) {
    return false;
  }
  if (c.needs_state_span && !q.state_span) return false;
  if ((c.lossy && !q.policy.allow_lossy) || (q.row_invariant && !c.row_invariant) ||
      (q.capturable && !c.capturable)) {
    return false;
  }
  return c.accepts == nullptr || c.accepts(q);
}

// Tuning data for exactly the queried shape, where the capability needs it.
bool Tuned(const Implementation& implementation, const CapabilityQuery& q) {
  return !implementation.capability->needs_tuning ||
         std::ranges::any_of(q.tuned, [&](const TunedShape& t) {
           return t.implementation == implementation.name && t.rows == q.rows && t.k == q.k &&
                  t.n == q.n &&
                  (!t.output_encoding || std::ranges::any_of(q.operands, [&](const auto& o) {
                    return o.operand == kOutputOperand && o.encoding == *t.output_encoding;
                  }));
         });
}

}  // namespace

std::string_view OperationName(Operation operation) {
  switch (operation) {
    case Operation::kLayerNorm:
      return "layer_norm";
    case Operation::kL2Norm:
      return "l2_norm";
    case Operation::kRmsNorm:
      return "rms_norm";
    case Operation::kRmsNormMul:
      return "rms_norm_mul";
    case Operation::kRmsNormMulRope:
      return "rms_norm_mul_rope";
    case Operation::kRmsNormMulAdd:
      return "rms_norm_mul_add";
    case Operation::kGemmaRoute:
      return "gemma_route";
    case Operation::kGemmaScaledReduce:
      return "gemma_scaled_reduce";
    case Operation::kAdd:
      return "add";
    case Operation::kMul:
      return "mul";
    case Operation::kMatMul:
      return "mul_mat";
    case Operation::kGetRows:
      return "get_rows";
    case Operation::kSetRows:
      return "set_rows";
    case Operation::kRope:
      return "rope";
    case Operation::kRopeSetRows:
      return "rope_set_rows";
    case Operation::kSoftMax:
      return "soft_max";
    case Operation::kCont:
      return "cont";
    case Operation::kSwiGlu:
      return "swiglu";
    case Operation::kGeGlu:
      return "geglu";
    case Operation::kMulMatGeGlu:
      return "mul_mat_geglu";
    case Operation::kMulMatAdd:
      return "mul_mat_add";
    case Operation::kMulMatGlu:
      return "mul_mat_glu";
    case Operation::kQuantLinear:
      return "quant_linear";
    case Operation::kQuantMultiLinear:
      return "quant_multi_linear";
    case Operation::kConvert:
      return "convert";
    case Operation::kFlashAttn:
      return "flash_attn";
    case Operation::kBiasAdd:
      return "bias_add";
    case Operation::kMulMatId:
      return "mul_mat_id";
    case Operation::kSub:
      return "sub";
    case Operation::kDiv:
      return "div";
    case Operation::kScale:
      return "scale";
    case Operation::kUnary:
      return "unary";
    case Operation::kClamp:
      return "clamp";
    case Operation::kFill:
      return "fill";
    case Operation::kRepeat:
      return "repeat";
    case Operation::kConcat:
      return "concat";
    case Operation::kSumRows:
      return "sum_rows";
    case Operation::kArgsort:
      return "argsort";
    case Operation::kTopK:
      return "top_k";
    case Operation::kSwiGluClamp:
      return "swiglu_clamp";
    case Operation::kSsmConv:
      return "ssm_conv";
    case Operation::kGatedDeltaNet:
      return "gated_delta_net";
    case Operation::kLightningIndexer:
      return "lightning_indexer";
    case Operation::kHcComb:
      return "dsv4_hc_comb";
    case Operation::kHcPre:
      return "dsv4_hc_pre";
    case Operation::kHcPost:
      return "dsv4_hc_post";
    case Operation::kHcCombine:
      return "hc_combine";
    case Operation::kHcNorm:
      return "hc_norm";
    case Operation::kHcMix:
      return "hc_mix";
    case Operation::kMoeGlu:
      return "moe_glu";
    case Operation::kMoeCombine:
      return "moe_combine";
    case Operation::kMoeRoute:
      return "moe_route";
    case Operation::kQuantize:
      return "quantize";
    case Operation::kNormGate:
      return "norm_gate";
    case Operation::kHeadNormRope:
      return "head_norm_rope";
    case Operation::kLayerNormModulate:
      return "layer_norm_modulate";
    case Operation::kGatedResidual:
      return "gated_residual";
    case Operation::kGatedResidualNorm:
      return "gated_residual_norm";
    case Operation::kEulerStep:
      return "euler_step";
    case Operation::kConv2d:
      return "conv2d";
    case Operation::kChannelNorm:
      return "channel_norm";
    case Operation::kUpsample:
      return "upsample";
    case Operation::kDupUpAdd:
      return "dup_up_add";
  }
  return "unknown";
}

ir::OpKey OpKeyOf(Operation operation) {
  using ir::Fused;
  using ir::FusedKind;
  using ir::Op;
  using ir::Primitive;
  switch (operation) {
    case Operation::kLayerNorm:
      return Primitive(Op::kLayerNorm);
    case Operation::kL2Norm:
      return Primitive(Op::kL2Norm);
    case Operation::kRmsNorm:
      return Primitive(Op::kRmsNorm);
    case Operation::kRmsNormMul:
      return Fused(FusedKind::kRmsNormMul);
    case Operation::kAdd:
    case Operation::kBiasAdd:
      return Primitive(Op::kAdd);
    case Operation::kMul:
      return Primitive(Op::kMul);
    case Operation::kMatMul:
    case Operation::kQuantLinear:
      return Primitive(Op::kMatMul);
    case Operation::kGetRows:
      return Primitive(Op::kGetRows);
    case Operation::kSetRows:
      return Primitive(Op::kCacheStore);
    case Operation::kRope:
      return Primitive(Op::kRope);
    case Operation::kRopeSetRows:
      return Fused(FusedKind::kRopeStore);
    case Operation::kSoftMax:
      return Primitive(Op::kSoftMax);
    case Operation::kCont:
      return Primitive(Op::kCont);
    case Operation::kSwiGlu:
    case Operation::kSwiGluClamp:
    case Operation::kGeGlu:
      return Primitive(Op::kGlu);
    case Operation::kMulMatAdd:
      return Fused(FusedKind::kMatMulAdd);
    case Operation::kMulMatGlu:
    case Operation::kMulMatGeGlu:
      return Fused(FusedKind::kMatMulGlu);
    case Operation::kQuantMultiLinear:
      return Fused(FusedKind::kPreparedProducts);
    case Operation::kConvert:
      return Primitive(Op::kConvert);
    case Operation::kFlashAttn:
      return Primitive(Op::kAttention);
    case Operation::kMulMatId:
      return Primitive(Op::kMatMulId);
    case Operation::kSub:
      return Primitive(Op::kSub);
    case Operation::kDiv:
      return Primitive(Op::kDiv);
    case Operation::kScale:
      return Primitive(Op::kScale);
    case Operation::kUnary:
      return Primitive(Op::kUnary);
    case Operation::kClamp:
      return Primitive(Op::kClamp);
    case Operation::kFill:
      return Primitive(Op::kFill);
    case Operation::kRepeat:
      return Primitive(Op::kRepeat);
    case Operation::kConcat:
      return Primitive(Op::kConcat);
    case Operation::kSumRows:
      return Primitive(Op::kSumRows);
    case Operation::kArgsort:
      return Primitive(Op::kArgsort);
    case Operation::kTopK:
      return Primitive(Op::kTopK);
    case Operation::kSsmConv:
      return Primitive(Op::kSsmConv);
    case Operation::kGatedDeltaNet:
      return Primitive(Op::kGatedDeltaNet);
    case Operation::kLightningIndexer:
      return Primitive(Op::kIndexScore);
    case Operation::kHcComb:
      return Primitive(Op::kHcMix);
    case Operation::kHcPre:
    case Operation::kHcMix:  // Qwen3.8's: gated streams folded into one
      return Primitive(Op::kHcPre);
    case Operation::kHcPost:
    case Operation::kHcCombine:  // Qwen3.8's: an output added to each stream, gated
      return Primitive(Op::kHcPost);
    case Operation::kHcNorm:
      return Fused(FusedKind::kHcNorm);
    case Operation::kMoeGlu:
      return Fused(FusedKind::kMoeGlu);
    case Operation::kMoeCombine:
      return Fused(FusedKind::kMoeCombine);
    case Operation::kMoeRoute:
      return Primitive(Op::kExpertSort);
    case Operation::kQuantize:
      return Primitive(Op::kPrepare);
    case Operation::kNormGate:
      return Fused(FusedKind::kNormGate);
    case Operation::kHeadNormRope:
      return Fused(FusedKind::kQkPrep);
    case Operation::kLayerNormModulate:
      return Fused(FusedKind::kLayerNormModulate);
    case Operation::kGatedResidual:
      return Fused(FusedKind::kGatedResidual);
    case Operation::kGatedResidualNorm:
      return Fused(FusedKind::kGatedResidualNorm);
    case Operation::kEulerStep:
      return Fused(FusedKind::kEulerStep);
    case Operation::kConv2d:
      return Primitive(Op::kConv2d);
    case Operation::kChannelNorm:
      return Primitive(Op::kChannelNorm);
    case Operation::kUpsample:
      return Primitive(Op::kUpsample);
    case Operation::kDupUpAdd:
      return Fused(FusedKind::kDupUpAdd);
    case Operation::kRmsNormMulRope:
      return Fused(FusedKind::kRmsNormMulRope);
    case Operation::kRmsNormMulAdd:
      return Fused(FusedKind::kRmsNormMulAdd);
    case Operation::kGemmaRoute:
      return Primitive(Op::kRoute);
    case Operation::kGemmaScaledReduce:
      return Primitive(Op::kExpertReduce);
  }
  base::Fatal("execution::OpKeyOf of an operation outside the enum");
}

base::Sha256Digest IdentityOf(const Implementation& implementation) {
  base::Sha256 hash;
  Field(hash, "llmp.implementation.v0");
  Field(hash, implementation.name);
  Field(hash, OperationName(implementation.operation));
  Field(hash, implementation.source);
  Field(hash, implementation.revision);
  Field(hash, implementation.build);
  Field(hash, implementation.variant);
  // Only a declared capability adds a field, so the identities of
  // implementations declared without one are unchanged.
  if (implementation.capability) {
    Field(hash, CapabilityText(*implementation.capability));
  }
  return hash.Finish();
}

Registry::Registry(std::vector<Implementation> implementations,
                   std::vector<base::Sha256Digest> identities, std::vector<std::size_t> by_name)
    : implementations_(std::move(implementations)),
      identities_(std::move(identities)),
      by_name_(std::move(by_name)) {}

std::expected<Registry, PlanRejection> Registry::Create(
    std::vector<Implementation> implementations) {
  if (implementations.size() > kMaxImplementations) {
    return Rejected(PlanError::kInvalid, 0,
                    std::format("{} implementations, more than {}", implementations.size(),
                                kMaxImplementations));
  }
  std::vector<base::Sha256Digest> identities;
  identities.reserve(implementations.size());
  std::vector<std::size_t> by_name(implementations.size());
  for (std::size_t i = 0; i < implementations.size(); ++i) {
    if (!ValidName(implementations[i].name)) {
      return Rejected(PlanError::kInvalid, 0,
                      "an implementation name that is empty, too long "
                      "or not lower-case letters, digits, '.' and '_'");
    }
    if (implementations[i].capability && !ValidCapability(*implementations[i].capability)) {
      return Rejected(PlanError::kInvalid, 0,
                      std::format("{} declares a malformed capability", implementations[i].name));
    }
    identities.push_back(IdentityOf(implementations[i]));
    by_name[i] = i;
  }
  std::ranges::sort(by_name, [&](std::size_t a, std::size_t b) {
    return implementations[a].name < implementations[b].name;
  });
  const auto duplicate = std::ranges::adjacent_find(by_name, [&](std::size_t a, std::size_t b) {
    return implementations[a].name == implementations[b].name;
  });
  if (duplicate != by_name.end()) {
    return Rejected(PlanError::kInvalid, 0,
                    std::format("two implementations named {}", implementations[*duplicate].name));
  }
  return Registry(std::move(implementations), std::move(identities), std::move(by_name));
}

std::optional<std::size_t> Registry::Find(std::string_view name) const {
  const auto found = std::ranges::lower_bound(
      by_name_, name, {},
      [this](std::size_t i) -> std::string_view { return implementations_[i].name; });
  if (found == by_name_.end() || implementations_[*found].name != name) {
    return std::nullopt;
  }
  return *found;
}

const Implementation& Registry::at(std::size_t index) const {
  base::Check(index < implementations_.size(), "registry index in range");
  return implementations_[index];
}

const base::Sha256Digest& Registry::identity(std::size_t index) const {
  base::Check(index < identities_.size(), "registry index in range");
  return identities_[index];
}

std::expected<Plan, PlanRejection> Plan::Build(const Registry& registry,
                                               std::span<const Choice> choices) {
  if (choices.empty() || choices.size() > kMaxOperations) {
    return Rejected(
        PlanError::kInvalid, 0,
        std::format("{} operations; a plan has 1 to {}", choices.size(), kMaxOperations));
  }
  std::vector<Step> steps;
  steps.reserve(choices.size());
  base::Sha256 hash;
  Field(hash, "llmp.plan.v0");
  Field(hash, std::format("{}", choices.size()));
  for (std::size_t i = 0; i < choices.size(); ++i) {
    const Choice& choice = choices[i];
    const auto found = Lookup(registry, choice.operation, choice.implementation, i);
    if (!found) {
      return std::unexpected(found.error());
    }
    const Implementation& implementation = registry.at(*found);
    const base::Sha256Digest& identity = registry.identity(*found);
    Field(hash, OperationName(choice.operation));
    Field(hash, implementation.name);
    hash.Update(std::as_bytes(std::span(identity)));
    steps.push_back({.operation = choice.operation,
                     .implementation = implementation.name,
                     .identity = identity});
  }
  return Plan(std::move(steps), hash.Finish());
}

const Implementation& BoundPlan::at(std::size_t operation) const {
  base::Check(operation < bound_.size(), "plan operation in range");
  return *bound_[operation];
}

std::expected<BoundPlan, PlanRejection> Resolve(const Registry& registry, const Plan& plan) {
  std::vector<const Implementation*> bound;
  bound.reserve(plan.steps().size());
  for (std::size_t i = 0; i < plan.steps().size(); ++i) {
    const Plan::Step& step = plan.steps()[i];
    const auto found = Lookup(registry, step.operation, step.implementation, i);
    if (!found) {
      return std::unexpected(found.error());
    }
    const Implementation& implementation = registry.at(*found);
    if (registry.identity(*found) != step.identity) {
      return Rejected(
          PlanError::kStale, i,
          std::format("{} is {} in this build; the plan was made for {}", implementation.name,
                      base::ToHex(registry.identity(*found)), base::ToHex(step.identity)));
    }
    bound.push_back(&implementation);
  }
  return BoundPlan(std::move(bound), plan.identity());
}

std::vector<std::size_t> Candidates(const Registry& registry, const CapabilityQuery& query) {
  struct Ranked {
    std::size_t index = 0;
    std::uint8_t rung = 0;
    std::uint64_t measured = 0;  // unranked last
    std::uint16_t priority = 0;
    std::string_view name;
  };
  std::vector<Ranked> ranked;
  for (std::size_t i = 0; i < registry.size(); ++i) {
    const Implementation& implementation = registry.at(i);
    if (!implementation.capability || !Computes(*implementation.capability, query) ||
        !Tuned(implementation, query)) {
      continue;
    }
    const auto rank = std::ranges::find_if(query.measured, [&](const MeasuredRank& m) {
      return m.implementation == implementation.name;
    });
    ranked.push_back(
        {.index = i,
         .rung = static_cast<std::uint8_t>(implementation.capability->rung),
         .measured = rank == query.measured.end() ? ~std::uint64_t{0} : std::uint64_t{rank->rank},
         .priority = implementation.capability->priority,
         .name = implementation.name});
  }
  // D-053: correctness (the filter above), then measured speed; the
  // ladder's rung and the static priority order what is not measured.
  std::ranges::sort(ranked, [](const Ranked& a, const Ranked& b) {
    if (a.measured != b.measured) {
      return a.measured < b.measured;
    }
    if (a.rung != b.rung) {
      return a.rung < b.rung;
    }
    if (a.priority != b.priority) {
      return a.priority > b.priority;
    }
    return a.name < b.name;
  });
  std::vector<std::size_t> out;
  out.reserve(ranked.size());
  for (const Ranked& r : ranked) {
    out.push_back(r.index);
  }
  return out;
}

}  // namespace llmp::execution
