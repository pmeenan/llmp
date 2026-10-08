// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "model/host_mask.h"

namespace {
namespace md = jitllm::model;
constexpr std::uint32_t kRows = 512;
constexpr std::uint32_t kWidth = 4608;
constexpr std::uint32_t kPast = 8191;
constexpr std::uint32_t kWindow = 4096;

// These are the prior per-cell formulas, including the blocked initialization.
[[gnu::noinline]] void OldFill(std::span<std::uint16_t> output, bool ring) {
  std::ranges::fill(output, 0xFC00);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto pos = (ring ? kPast : 4095U) + r;
    for (std::uint32_t c = 0; c < kWidth; ++c) {
      bool visible = c <= pos;
      if (ring) {
        const auto held = c + ((kPast + kRows - 1 - c) / kWidth) * kWidth;
        visible = held <= pos && pos - held < kWindow;
      }
      if (visible) output[std::size_t{r} * kWidth + c] = 0;
    }
  }
}
[[gnu::noinline]] void NewFill(std::span<std::uint16_t> output, bool ring) {
  std::ranges::fill(output, 0xFC00);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto runs = ring ? md::HostMaskRing(kWidth, kPast + r, kPast + kRows, kWidth, kWindow)
                           : md::HostMaskPrefix(kWidth, 4095U + r + 1);
    md::FillHostMaskVisible(output.subspan(std::size_t{r} * kWidth, kWidth), *runs,
                            std::uint16_t{0});
  }
}
[[gnu::noinline]] bool OldCheck(std::span<const std::uint16_t> output, bool ring) {
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::uint32_t c = 0; c < kWidth; ++c) {
      const auto pos = (ring ? kPast : 4095U) + r;
      bool visible = c <= pos;
      if (ring) {
        const auto held = c + ((kPast + kRows - 1 - c) / kWidth) * kWidth;
        visible = held <= pos && pos - held < kWindow;
      }
      if (output[std::size_t{r} * kWidth + c] != (visible ? 0 : 0xFC00)) return false;
    }
  return true;
}
[[gnu::noinline]] bool NewCheck(std::span<const std::uint16_t> output, bool ring) {
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto runs = ring ? md::HostMaskRing(kWidth, kPast + r, kPast + kRows, kWidth, kWindow)
                           : md::HostMaskPrefix(kWidth, 4095U + r + 1);
    if (!md::MatchHostMask(output.subspan(std::size_t{r} * kWidth, kWidth), *runs)) return false;
  }
  return true;
}
[[gnu::noinline]] void OldCompressed(std::span<std::uint16_t> output) {
  std::ranges::fill(output, 0xFC00);
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::int64_t c = 0; c < std::int64_t{4095U + r + 1}; ++c)
      output[std::size_t{r} * kWidth + static_cast<std::size_t>(c)] = 0;
}
[[gnu::noinline]] void OldQsa(std::span<float> output) {
  constexpr std::uint32_t full = (kPast + kRows) / 4;
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto tail = (kPast + r + 1) / 4 * 4;
    for (std::uint32_t b = 0; b < kWidth; ++b)
      output[std::size_t{r} * kWidth + b] = b >= full ? -std::numeric_limits<float>::infinity()
                                            : b * 4 >= tail ? 1e9f
                                                            : 0.0f;
    output[std::size_t{r} * kWidth + full] = 1e9f;
  }
}
[[gnu::noinline]] void NewQsa(std::span<float> output) {
  for (std::uint32_t r = 0; r < kRows; ++r)
    md::FillHostQsaBias(output.subspan(std::size_t{r} * kWidth, kWidth), (kPast + kRows) / 4,
                        (kPast + r + 1) / 4);
}
}  // namespace

int main() {
  std::vector<std::uint16_t> half(std::size_t{kRows} * kWidth), oracle(half.size());
  std::vector<float> bias(half.size()), qsa_oracle(half.size());
  std::uint64_t sink = 0;
  for (const std::string_view formula :
       {"prefix-fill", "compressed-fill", "ring-fill", "prefix-check", "ring-check", "qsa-fill"}) {
    const bool ring = formula.starts_with("ring");
    OldFill(oracle, ring);
    NewFill(half, ring);
    OldQsa(qsa_oracle);
    NewQsa(bias);
    if (half != oracle || !OldCheck(half, ring) || !NewCheck(half, ring) ||
        !std::ranges::equal(bias, qsa_oracle, [](float a, float b) {
          return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        }))
      return 1;
    for (const std::string_view arm : {"O1", "A1", "A2", "O2"}) {
      const bool old = arm.starts_with("O");
      const auto start = std::chrono::steady_clock::now();
      for (std::uint32_t call = 0; call < 16; ++call) {
        if (formula == "qsa-fill") {
          old ? OldQsa(bias) : NewQsa(bias);
          sink += std::bit_cast<std::uint32_t>(bias[call * 7919 % bias.size()]);
        } else if (formula.ends_with("check")) {
          const bool matched = old ? OldCheck(half, ring) : NewCheck(half, ring);
          if (!matched) return 1;
          sink += matched;
        } else {
          if (old && formula == "compressed-fill")
            OldCompressed(half);
          else
            old ? OldFill(half, ring) : NewFill(half, ring);
          sink += half[call * 7919 % half.size()];
        }
      }
      const auto seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      if (half != oracle || !std::ranges::equal(bias, qsa_oracle, [](float a, float b) {
            return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
          }))
        return 1;
      std::cout << std::setprecision(12) << "formula=" << formula << " arm=" << arm
                << " calls=16 rows=" << kRows << " width=" << kWidth << " seconds=" << seconds
                << '\n';
    }
  }
  std::cout << "exact=1 sink=" << sink << '\n';
}
