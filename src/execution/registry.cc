// SPDX-FileCopyrightText: 2026 jitLLM contributors
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

namespace jitllm::execution {
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

}  // namespace

std::string_view OperationName(Operation operation) {
  switch (operation) {
    case Operation::kRmsNorm:
      return "rms_norm";
    case Operation::kRmsNormMul:
      return "rms_norm_mul";
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

base::Sha256Digest IdentityOf(const Implementation& implementation) {
  base::Sha256 hash;
  Field(hash, "jitllm.implementation.v0");
  Field(hash, implementation.name);
  Field(hash, OperationName(implementation.operation));
  Field(hash, implementation.source);
  Field(hash, implementation.revision);
  Field(hash, implementation.build);
  Field(hash, implementation.variant);
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
  Field(hash, "jitllm.plan.v0");
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

}  // namespace jitllm::execution
