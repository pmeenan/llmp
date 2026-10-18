// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_SPEC_NAMES_H_
#define LLMP_SPEC_NAMES_H_
#include <algorithm>
#include <string_view>

#include "spec/spec.h"
namespace llmp::spec::names {
inline bool Lower(char c) { return c >= 'a' && c <= 'z'; }
inline bool Digit(char c) { return c >= '0' && c <= '9'; }
inline bool IsName(std::string_view s) {
  return !s.empty() && s.size() <= kMaxText && Lower(s.front()) &&
         std::ranges::all_of(s, [](char c) { return Lower(c) || Digit(c) || c == '_'; });
}
inline bool IsKind(std::string_view s) {
  if (s.empty() || s.size() > kMaxText) return false;
  std::size_t first = 0;
  while (true) {
    const auto dot = s.find('.', first);
    if (!IsName(s.substr(first, dot == std::string_view::npos ? dot : dot - first))) return false;
    if (dot == std::string_view::npos) return true;
    first = dot + 1;
  }
}
}  // namespace llmp::spec::names
#endif  // LLMP_SPEC_NAMES_H_
