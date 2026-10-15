// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/policy.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "planner/pattern.h"

namespace llmp::planner {
namespace {

std::unexpected<std::string> Error(std::string detail) {
  return std::unexpected(std::move(detail));
}

bool NameChars(std::string_view s, bool dots) {
  return !s.empty() && s.size() <= execution::Registry::kMaxName &&
         std::ranges::all_of(s, [&](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                  (dots && c == '.');
         });
}

auto RankOrder(const ClassRank& r) { return std::tie(r.device_class, r.rank, r.implementation); }
auto RuleOrder(const StepRule& r) {
  return std::tie(r.scope, r.test, r.rows, r.action, r.pattern, r.value);
}

constexpr std::array<std::string_view, 3> kScopes{"any", "solo", "joined"};
constexpr std::array<std::string_view, 2> kTests{"past", "under"};
constexpr std::array<std::string_view, 3> kActions{"disable", "max_decode_rows", "vector_rows"};

template <typename E, std::size_t N>
std::optional<E> Named(const std::array<std::string_view, N>& names, std::string_view word) {
  const auto at = std::ranges::find(names, word);
  if (at == names.end()) return std::nullopt;
  return static_cast<E>(at - names.begin());
}

bool Bound(std::uint32_t n) { return n != 0 && n <= kMaxDecodeRowsBound; }

std::string RuleText(const StepRule& r) {
  return std::format("when {} {} {} {} {}\n", kScopes.at(static_cast<std::size_t>(r.scope)),
                     kTests.at(static_cast<std::size_t>(r.test)), r.rows,
                     kActions.at(static_cast<std::size_t>(r.action)),
                     r.action == StepAction::kDisable ? r.pattern : std::to_string(r.value));
}

bool Holds(const StepRule& r, std::uint64_t rows, std::size_t segments) {
  if ((r.scope == StepScope::kSolo && segments != 1) ||
      (r.scope == StepScope::kJoined && segments <= 1))
    return false;
  return r.test == RowTest::kPast ? rows > r.rows : rows < r.rows;
}

// A decimal of at most ten digits, no sign or leading zero.
std::expected<std::uint32_t, std::string> Number(std::string_view s) {
  std::uint32_t v = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc{} || end != s.data() + s.size() || s.empty() ||
      (s.size() > 1 && s.front() == '0')) {
    return Error(std::format("not a number: {}", s));
  }
  return v;
}

std::vector<std::string_view> Words(std::string_view line) {
  std::vector<std::string_view> out;
  std::size_t at = 0;
  while (at <= line.size()) {
    const std::size_t space = line.find(' ', at);
    const std::size_t end = space == std::string_view::npos ? line.size() : space;
    out.push_back(line.substr(at, end - at));
    if (space == std::string_view::npos) {
      break;
    }
    at = space + 1;
  }
  return out;
}

}  // namespace

bool Disabled(const Policy& policy, std::string_view pattern) {
  if (std::ranges::contains(policy.disabled, pattern)) {
    return true;
  }
  const Pattern* p = FindPattern(pattern);
  return p != nullptr && p->opt_in && !std::ranges::contains(policy.enabled, pattern);
}

void Canonicalize(Policy& policy) {
  std::ranges::sort(policy.enabled);
  std::ranges::sort(policy.disabled);
  std::ranges::sort(policy.ranks, [](const ClassRank& a, const ClassRank& b) {
    return RankOrder(a) < RankOrder(b);
  });
  std::ranges::sort(policy.rules, [](const StepRule& a, const StepRule& b) {
    return RuleOrder(a) < RuleOrder(b);
  });
}

std::expected<void, std::string> Validate(const Policy& policy) {
  if (policy.max_decode_rows == 0 || policy.max_decode_rows > kMaxDecodeRowsBound) {
    return Error(std::format("max_decode_rows {} outside 1 to {}", policy.max_decode_rows,
                             kMaxDecodeRowsBound));
  }
  if (policy.vector_rows != 0 && !Bound(policy.vector_rows)) {
    return Error(
        std::format("vector_rows {} outside 1 to {}", policy.vector_rows, kMaxDecodeRowsBound));
  }
  if (policy.disabled.size() > kMaxDisabled || policy.enabled.size() > kMaxDisabled ||
      policy.ranks.size() > kMaxRanks || policy.rules.size() > kMaxStepRules) {
    return Error("more enabled or disabled patterns, ranks or step rules than a policy holds");
  }
  for (std::size_t i = 0; i < policy.enabled.size(); ++i) {
    const std::string& name = policy.enabled[i];
    const Pattern* p = FindPattern(name);
    if (p == nullptr || !p->opt_in) {
      return Error(std::format("no opt-in pattern {} to enable", name));
    }
    if (i > 0 && policy.enabled[i - 1] >= name) {
      return Error("enabled patterns are not ascending and unique");
    }
    if (std::ranges::contains(policy.disabled, name)) {
      return Error(std::format("{} both enabled and disabled", name));
    }
  }
  for (std::size_t i = 0; i < policy.disabled.size(); ++i) {
    const std::string& name = policy.disabled[i];
    if (FindPattern(name) == nullptr) {
      return Error(std::format("no pattern {} to disable", name));
    }
    if (i > 0 && policy.disabled[i - 1] >= name) {
      return Error("disabled patterns are not ascending and unique");
    }
  }
  for (std::size_t i = 0; i < policy.rules.size(); ++i) {
    const StepRule& r = policy.rules[i];
    if (static_cast<std::size_t>(r.scope) >= kScopes.size() ||
        static_cast<std::size_t>(r.test) >= kTests.size() ||
        static_cast<std::size_t>(r.action) >= kActions.size()) {
      return Error("a step rule of no scope, test or action");
    }
    if (r.test == RowTest::kUnder && r.rows == 0) {
      return Error("a step rule under no rows holds for no step");
    }
    if (r.action == StepAction::kDisable ? (r.value != 0 || FindPattern(r.pattern) == nullptr)
                                         : (!r.pattern.empty() || !Bound(r.value))) {
      return Error(std::format("a step rule's action is not one: {}", RuleText(r)));
    }
    if (i > 0 && !(RuleOrder(policy.rules[i - 1]) < RuleOrder(r))) {
      return Error("step rules are not ascending and unique");
    }
  }
  for (std::size_t i = 0; i < policy.ranks.size(); ++i) {
    const ClassRank& r = policy.ranks[i];
    if (!NameChars(r.device_class, false) || !NameChars(r.implementation, true)) {
      return Error(std::format("a rank's class or implementation is not a name: {} {}",
                               r.device_class, r.implementation));
    }
    if (i > 0 && !(RankOrder(policy.ranks[i - 1]) < RankOrder(r))) {
      return Error("ranks are not in order");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (policy.ranks[j].device_class == r.device_class &&
          policy.ranks[j].implementation == r.implementation) {
        return Error(std::format("{} ranked twice on {}", r.implementation, r.device_class));
      }
    }
  }
  return {};
}

std::string ToText(const Policy& policy) {
  std::string text = std::format("max_decode_rows {}\n", policy.max_decode_rows);
  if (policy.vector_rows != 0) {
    text += std::format("vector_rows {}\n", policy.vector_rows);
  }
  if (policy.row_invariant) {
    text += "row_invariant\n";
  }
  if (policy.allow_lossy) {
    text += "allow_lossy\n";
  }
  for (const std::string& name : policy.enabled) {
    text += std::format("enable {}\n", name);
  }
  for (const std::string& name : policy.disabled) {
    text += std::format("disable {}\n", name);
  }
  for (const StepRule& r : policy.rules) {
    text += RuleText(r);
  }
  for (const ClassRank& r : policy.ranks) {
    text += std::format("rank {} {} {}\n", r.device_class, r.implementation, r.rank);
  }
  return text;
}

std::expected<Policy, std::string> ParsePolicy(std::string_view text) {
  if (text.size() > kMaxPolicyBytes) {
    return Error("a policy past its bound");
  }
  if (text.empty() || text.back() != '\n') {
    return Error("a policy is lines, each ending in a newline");
  }
  Policy policy;
  bool rows = false;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t end = text.find('\n', at);
    const std::string_view line = text.substr(at, end - at);
    at = end + 1;
    const std::vector<std::string_view> w = Words(line);
    if (w.size() == 2 && w[0] == "max_decode_rows" && !rows) {
      const auto n = Number(w[1]);
      if (!n) {
        return std::unexpected(n.error());
      }
      policy.max_decode_rows = *n;
      rows = true;
    } else if (w.size() == 2 && w[0] == "vector_rows" && policy.vector_rows == 0) {
      const auto n = Number(w[1]);
      if (!n) {
        return std::unexpected(n.error());
      }
      if (*n == 0) {
        return Error("vector_rows 0 is its absence");
      }
      policy.vector_rows = *n;
    } else if (w.size() == 6 && w[0] == "when") {
      const auto scope = Named<StepScope>(kScopes, w[1]);
      const auto test = Named<RowTest>(kTests, w[2]);
      const auto action = Named<StepAction>(kActions, w[4]);
      const auto n = Number(w[3]);
      if (!scope || !test || !action || !n) {
        return Error(std::format("not a step rule: {}", line.substr(0, 64)));
      }
      if (policy.rules.size() == kMaxStepRules) {
        return Error("more step rules than a policy holds");
      }
      StepRule rule{.scope = *scope, .test = *test, .rows = *n, .action = *action};
      if (*action == StepAction::kDisable) {
        rule.pattern = std::string(w[5]);
      } else {
        const auto value = Number(w[5]);
        if (!value) {
          return std::unexpected(value.error());
        }
        rule.value = *value;
      }
      policy.rules.push_back(std::move(rule));
    } else if (w.size() == 1 && w[0] == "row_invariant") {
      policy.row_invariant = true;
    } else if (w.size() == 1 && w[0] == "allow_lossy") {
      policy.allow_lossy = true;
    } else if (w.size() == 2 && w[0] == "enable") {
      if (policy.enabled.size() == kMaxDisabled) {
        return Error("more enabled patterns than a policy holds");
      }
      policy.enabled.emplace_back(w[1]);
    } else if (w.size() == 2 && w[0] == "disable") {
      if (policy.disabled.size() == kMaxDisabled) {
        return Error("more disabled patterns than a policy holds");
      }
      policy.disabled.emplace_back(w[1]);
    } else if (w.size() == 4 && w[0] == "rank") {
      const auto n = Number(w[3]);
      if (!n) {
        return std::unexpected(n.error());
      }
      if (policy.ranks.size() == kMaxRanks) {
        return Error("more ranks than a policy holds");
      }
      policy.ranks.push_back(
          {.device_class = std::string(w[1]), .implementation = std::string(w[2]), .rank = *n});
    } else {
      return Error(std::format("not a policy directive: {}", line.substr(0, 64)));
    }
  }
  if (auto valid = Validate(policy); !valid) {
    return std::unexpected(valid.error());
  }
  if (ToText(policy) != text) {
    return Error("a policy's text is not canonical");
  }
  return policy;
}

std::expected<Policy, std::string> ForStep(const Policy& policy, std::uint64_t rows,
                                           std::size_t segments) {
  Policy out = policy;
  out.rules.clear();
  for (const StepRule& r : policy.rules) {
    if (!Holds(r, rows, segments)) {
      continue;
    }
    switch (r.action) {
      case StepAction::kDisable:
        std::erase(out.enabled, r.pattern);
        if (!std::ranges::contains(out.disabled, r.pattern)) {
          out.disabled.push_back(r.pattern);
        }
        break;
      case StepAction::kMaxDecodeRows:
        out.max_decode_rows = std::min(out.max_decode_rows, r.value);
        break;
      case StepAction::kVectorRows:
        out.vector_rows = out.vector_rows == 0 ? r.value : std::min(out.vector_rows, r.value);
        break;
    }
  }
  Canonicalize(out);
  if (auto valid = Validate(out); !valid) {
    return Error(std::format("a step's policy: {}", valid.error()));
  }
  return out;
}

base::Sha256Digest Digest(const Policy& policy) {
  base::Sha256 hash;
  hash.Update(std::string_view("llmp.planner.policy.v1\n"));
  hash.Update(ToText(policy));
  return hash.Finish();
}

std::vector<execution::MeasuredRank> RanksFor(const Policy& policy, std::string_view device_class) {
  std::vector<execution::MeasuredRank> out;
  for (const ClassRank& r : policy.ranks) {
    if (r.device_class == device_class) {
      out.push_back({.implementation = r.implementation, .rank = r.rank});
    }
  }
  return out;
}

}  // namespace llmp::planner
