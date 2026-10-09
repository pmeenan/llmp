// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "base/check.h"
#include "components/component.h"
#include "spec/spec.h"

namespace llmp::components {
namespace {

bool Within(const ParamSpec& s, double v) {
  return std::isfinite(v) && v >= s.least && v <= s.most;
}

// The value of `s` given `v`, or why not.
std::expected<spec::ParamValue, std::string> Typed(const ParamSpec& s, const spec::ParamValue& v) {
  const auto out_of_range = [&] {
    return std::unexpected(
        std::format("parameter {} is outside [{}, {}]", s.name, s.least, s.most));
  };
  switch (s.type) {
    case ParamType::kInt: {
      const auto* i = std::get_if<std::int64_t>(&v);
      if (i == nullptr) break;
      if (!Within(s, static_cast<double>(*i))) return out_of_range();
      return v;
    }
    case ParamType::kReal: {
      // An integer is a real too (a mapping's constant may be written so).
      double r = 0.0;
      if (const auto* d = std::get_if<double>(&v)) {
        r = *d;
      } else if (const auto* i = std::get_if<std::int64_t>(&v)) {
        r = static_cast<double>(*i);
      } else {
        break;
      }
      if (!Within(s, r)) return out_of_range();
      return spec::ParamValue(r);
    }
    case ParamType::kBool:
      if (std::holds_alternative<bool>(v)) return v;
      break;
    case ParamType::kName: {
      const auto* n = std::get_if<std::string>(&v);
      if (n == nullptr) break;
      if (std::ranges::find(s.choices, std::string_view(*n)) == s.choices.end()) {
        return std::unexpected(std::format("parameter {} has no choice {}", s.name, *n));
      }
      return v;
    }
    case ParamType::kInts: {
      const auto* l = std::get_if<std::vector<std::int64_t>>(&v);
      if (l == nullptr) break;
      for (const std::int64_t i : *l) {
        if (!Within(s, static_cast<double>(i))) return out_of_range();
      }
      return v;
    }
    case ParamType::kReals: {
      const auto* l = std::get_if<std::vector<double>>(&v);
      if (l == nullptr) break;
      for (const double d : *l) {
        if (!Within(s, d)) return out_of_range();
      }
      return v;
    }
  }
  return std::unexpected(std::format("parameter {} is not {}", s.name, ParamTypeName(s.type)));
}

spec::ParamValue Fallback(const ParamSpec& s) {
  switch (s.type) {
    case ParamType::kInt:
      return static_cast<std::int64_t>(s.fallback);
    case ParamType::kReal:
      return s.fallback;
    case ParamType::kBool:
      return s.fallback != 0.0;
    case ParamType::kName:
      return std::string(s.choices.empty() ? std::string_view{} : s.choices.front());
    case ParamType::kInts:
      return std::vector<std::int64_t>{};
    case ParamType::kReals:
      return std::vector<double>{};
  }
  return std::int64_t{0};
}

}  // namespace

std::string_view ParamTypeName(ParamType type) {
  switch (type) {
    case ParamType::kInt:
      return "an integer";
    case ParamType::kReal:
      return "a real";
    case ParamType::kBool:
      return "a flag";
    case ParamType::kName:
      return "a name";
    case ParamType::kInts:
      return "a list of integers";
    case ParamType::kReals:
      return "a list of reals";
  }
  return "unknown";
}

std::expected<Params, std::string> Params::Of(std::span<const ParamSpec> specs,
                                              const spec::ComponentRef& ref) {
  Params out;
  for (const spec::Param& p : ref.params) {
    if (std::ranges::find(specs, std::string_view(p.name), &ParamSpec::name) == specs.end()) {
      return std::unexpected(std::format("{} has no parameter {}", ref.kind, p.name));
    }
  }
  for (const ParamSpec& s : specs) {
    const spec::Param* given = ref.Find(s.name);
    if (given == nullptr) {
      if (s.required) {
        return std::unexpected(std::format("{} needs parameter {}", ref.kind, s.name));
      }
      out.values_.push_back({.name = std::string(s.name), .value = Fallback(s)});
      continue;
    }
    auto typed = Typed(s, given->value);
    if (!typed) return std::unexpected(std::format("{}: {}", ref.kind, typed.error()));
    out.values_.push_back({.name = std::string(s.name), .value = std::move(*typed)});
    out.given_.emplace_back(s.name);
  }
  std::ranges::sort(out.values_, {}, &spec::Param::name);
  return out;
}

const spec::ParamValue& Params::At(std::string_view name) const {
  const auto at = std::ranges::lower_bound(
      values_, name, {}, [](const spec::Param& p) -> std::string_view { return p.name; });
  base::Check(at != values_.end() && at->name == name, "a component reads a parameter it lists");
  return at->value;
}

std::int64_t Params::Int(std::string_view name) const {
  const auto* v = std::get_if<std::int64_t>(&At(name));
  base::Check(v != nullptr, "an integer parameter");
  return *v;
}
double Params::Real(std::string_view name) const {
  const auto* v = std::get_if<double>(&At(name));
  base::Check(v != nullptr, "a real parameter");
  return *v;
}
bool Params::Flag(std::string_view name) const {
  const auto* v = std::get_if<bool>(&At(name));
  base::Check(v != nullptr, "a flag parameter");
  return *v;
}
std::string_view Params::Name(std::string_view name) const {
  const auto* v = std::get_if<std::string>(&At(name));
  base::Check(v != nullptr, "a name parameter");
  return *v;
}
std::span<const std::int64_t> Params::Ints(std::string_view name) const {
  const auto* v = std::get_if<std::vector<std::int64_t>>(&At(name));
  base::Check(v != nullptr, "an integer list parameter");
  return *v;
}
std::span<const double> Params::Reals(std::string_view name) const {
  const auto* v = std::get_if<std::vector<double>>(&At(name));
  base::Check(v != nullptr, "a real list parameter");
  return *v;
}
bool Params::Given(std::string_view name) const {
  return std::ranges::find(given_, name) != given_.end();
}

std::string_view SplitKindName(SplitKind kind) {
  switch (kind) {
    case SplitKind::kNone:
      return "none";
    case SplitKind::kHeads:
      return "heads";
    case SplitKind::kColumns:
      return "columns";
    case SplitKind::kRows:
      return "rows";
    case SplitKind::kExperts:
      return "experts";
    case SplitKind::kWholeLayer:
      return "whole_layer";
  }
  return "unknown";
}

}  // namespace llmp::components
