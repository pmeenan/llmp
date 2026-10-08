// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "dsv4_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <sstream>
#include <string_view>
#include <utility>

namespace llmp::benchmarks {

std::expected<std::vector<TokenLine>, std::string> ReadTokenLines(const std::filesystem::path& p) {
  std::ifstream file(p);
  if (!file) {
    return std::unexpected(std::format("cannot read {}", p.string()));
  }
  std::vector<TokenLine> out;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    TokenLine t;
    const auto tab = line.find('\t');
    std::string_view rest = line;
    if (tab != std::string::npos) {
      t.name = line.substr(0, tab);
      rest = std::string_view(line).substr(tab + 1);
    }
    std::istringstream ids{std::string(rest)};
    std::int64_t id = 0;
    while (ids >> id) {
      t.ids.push_back(static_cast<std::int32_t>(id));
    }
    out.push_back(std::move(t));
  }
  return out;
}

double Nll(std::span<const float> row, std::int32_t target) {
  const double most = *std::ranges::max_element(row);
  double sum = 0;
  for (const float v : row) {
    sum += std::exp(static_cast<double>(v) - most);
  }
  return (most + std::log(sum)) - static_cast<double>(row[static_cast<std::size_t>(target)]);
}

std::expected<void, std::string> WriteFloats(const std::filesystem::path& p,
                                             std::span<const float> v) {
  std::ofstream out(p, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size_bytes()));
  if (!out) {
    return std::unexpected(std::format("cannot write {}", p.string()));
  }
  return {};
}

}  // namespace llmp::benchmarks
