// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/policy.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
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
  return std::ranges::contains(policy.disabled, pattern);
}

void Canonicalize(Policy& policy) {
  std::ranges::sort(policy.disabled);
  std::ranges::sort(policy.ranks, [](const ClassRank& a, const ClassRank& b) {
    return RankOrder(a) < RankOrder(b);
  });
}

std::expected<void, std::string> Validate(const Policy& policy) {
  if (policy.max_decode_rows == 0 || policy.max_decode_rows > kMaxDecodeRowsBound) {
    return Error(std::format("max_decode_rows {} outside 1 to {}", policy.max_decode_rows,
                             kMaxDecodeRowsBound));
  }
  if (policy.disabled.size() > kMaxDisabled || policy.ranks.size() > kMaxRanks) {
    return Error("more disabled patterns or ranks than a policy holds");
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
  if (policy.row_invariant) {
    text += "row_invariant\n";
  }
  if (policy.allow_lossy) {
    text += "allow_lossy\n";
  }
  for (const std::string& name : policy.disabled) {
    text += std::format("disable {}\n", name);
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
    } else if (w.size() == 1 && w[0] == "row_invariant") {
      policy.row_invariant = true;
    } else if (w.size() == 1 && w[0] == "allow_lossy") {
      policy.allow_lossy = true;
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
