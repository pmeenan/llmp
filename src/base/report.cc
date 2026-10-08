// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/report.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace llmp::base {

namespace {

// Formatting characters that change how text around them reads: the
// bidirectional marks, embeddings, overrides and isolates, and the
// zero-width ones.
bool Invisible(char32_t c) {
  return c == 0x00AD || c == 0x034F || c == 0x061C || c == 0x115F || c == 0x1160 || c == 0x17B4 ||
         c == 0x17B5 || c == 0x180E || (c >= 0x200B && c <= 0x200F) ||
         (c >= 0x2028 && c <= 0x202E) || (c >= 0x2060 && c <= 0x206F) || c == 0x3164 ||
         (c >= 0xFE00 && c <= 0xFE0F) || c == 0xFEFF || c == 0xFFA0 ||
         (c >= 0xFFF0 && c <= 0xFFFB) || (c >= 0xE0000 && c <= 0xE0FFF);
}

// The code point starting text[at] and its length, or length 0 if the
// bytes there are not UTF-8.
std::pair<char32_t, std::size_t> Decode(std::string_view text, std::size_t at) {
  const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(text[at + i]); };
  const unsigned char lead = byte(0);
  std::size_t length = 0;
  char32_t c = 0;
  if (lead < 0x80) {
    return {lead, 1};
  }
  if (lead >= 0xC2 && lead <= 0xDF) {
    length = 2;
    c = lead & 0x1FU;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    length = 3;
    c = lead & 0x0FU;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    length = 4;
    c = lead & 0x07U;
  } else {
    return {0, 0};
  }
  if (at + length > text.size()) {
    return {0, 0};
  }
  for (std::size_t i = 1; i < length; ++i) {
    if ((byte(i) & 0xC0U) != 0x80) {
      return {0, 0};
    }
    c = (c << 6U) | (byte(i) & 0x3FU);
  }
  const bool overlong = (length == 3 && c < 0x800) || (length == 4 && c < 0x10000);
  if (overlong || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) {
    return {0, 0};
  }
  return {c, length};
}

}  // namespace

std::string Printable(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t at = 0; at < text.size();) {
    const auto [c, length] = Decode(text, at);
    if (length == 0) {
      out += std::format("\\x{:02x}", static_cast<unsigned char>(text[at]));
      ++at;
      continue;
    }
    if (c < 0x20 || c == 0x7F) {
      out += std::format("\\x{:02x}", static_cast<unsigned>(c));
    } else if ((c >= 0x80 && c <= 0x9F) || Invisible(c)) {
      out += std::format("\\u{:04x}", static_cast<unsigned>(c));
    } else {
      out.append(text.substr(at, length));
    }
    at += length;
  }
  return out;
}

std::string FormatBytes(std::uint64_t bytes) {
  static constexpr std::array<std::string_view, 6> kUnits = {"KiB", "MiB", "GiB",
                                                             "TiB", "PiB", "EiB"};
  if (bytes < 1024) {
    return std::format("{} B", bytes);
  }
  std::size_t unit = 0;
  std::uint64_t scale = 1024;
  while (unit + 1 < kUnits.size() && bytes / scale >= 1024) {
    scale *= 1024;
    ++unit;
  }
  if (bytes % scale == 0) {
    return std::format("{} {}", bytes / scale, kUnits.at(unit));
  }
  double value = static_cast<double>(bytes) / static_cast<double>(scale);
  // Just under the next unit rounds up to it: "1.0 GiB", never "1024.0 MiB".
  if (std::round(value * 10) >= 10240 && unit + 1 < kUnits.size()) {
    value /= 1024;
    ++unit;
  }
  return std::format("{:.1f} {}", value, kUnits.at(unit));
}

}  // namespace llmp::base
