// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Allocation-free JSON fragments. The borrowed sink consumes every fragment
// synchronously; count and fill use the identical writer, including escaping.
#ifndef LLMP_BASE_JSON_STREAM_H_
#define LLMP_BASE_JSON_STREAM_H_

#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>

#include "base/sha256.h"

namespace llmp::base::json {
using FragmentSink = std::function<void(std::string_view)>;
class Stream {
 public:
  explicit Stream(const FragmentSink& sink) : sink_(sink) {}
  void Raw(std::string_view text) const {
    if (sink_) sink_(text);
  }
  void Number(std::uint64_t value) const {
    std::array<char, 20> bytes{};
    const auto result = std::to_chars(bytes.data(), bytes.data() + bytes.size(), value);
    Raw({bytes.data(), static_cast<std::size_t>(result.ptr - bytes.data())});
  }
  void Boolean(bool value) const { Raw(value ? "true" : "false"); }
  void Quoted(std::string_view text) const {
    Raw("\"");
    constexpr char hex[] = "0123456789abcdef";
    std::size_t begin = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
      const auto byte = static_cast<unsigned char>(text[i]);
      if (byte != '"' && byte != '\\' && byte >= 0x20) continue;
      Raw(text.substr(begin, i - begin));
      if (byte == '"' || byte == '\\') {
        const char escaped[] = {'\\', static_cast<char>(byte)};
        Raw({escaped, sizeof(escaped)});
      } else {
        const char escaped[] = {'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15]};
        Raw({escaped, sizeof(escaped)});
      }
      begin = i + 1;
    }
    Raw(text.substr(begin));
    Raw("\"");
  }
  void Hex(const Sha256Digest& digest) const {
    constexpr char hex[] = "0123456789abcdef";
    std::array<char, 64> out{};
    for (std::size_t i = 0; i < digest.size(); ++i) {
      out[2 * i] = hex[digest[i] >> 4];
      out[2 * i + 1] = hex[digest[i] & 15];
    }
    Raw({out.data(), out.size()});
  }

 private:
  const FragmentSink& sink_;
};
}  // namespace llmp::base::json

#endif  // LLMP_BASE_JSON_STREAM_H_
