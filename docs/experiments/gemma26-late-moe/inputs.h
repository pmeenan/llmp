// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_BENCHMARK_GEMMA26_LATE_MOE_INPUTS_H_
#define JITLLM_BENCHMARK_GEMMA26_LATE_MOE_INPUTS_H_
#include <unistd.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
namespace late_moe {
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
inline constexpr std::uint64_t kHostAllowance = 64U << 20U;
inline std::string Hash(std::span<const std::byte> data) {
  return jitllm::base::ToHex(jitllm::base::Sha256().Update(data).Finish());
}
inline bool Hex(std::string_view text) {
  return text.size() == 64 && text.find_first_not_of("0123456789abcdef") == text.npos;
}
inline bool Finite(std::span<const std::byte> data) {
  if (data.size() % 4) return false;
  for (std::size_t i = 0; i < data.size(); i += 4) {
    float value = 0;
    std::memcpy(&value, data.data() + i, 4);
    if (!std::isfinite(value)) return false;
  }
  return true;
}
inline bool Read(const std::filesystem::path& path, std::size_t bytes,
                 std::vector<std::byte>& data) {
  std::error_code error;
  if (std::filesystem::symlink_status(path, error).type() != std::filesystem::file_type::regular ||
      error || std::filesystem::file_size(path, error) != bytes || error)
    return false;
  data.resize(bytes);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
  return stream && stream.peek() == std::char_traits<char>::eof();
}
struct Inputs {
  std::array<std::array<std::vector<std::byte>, 3>, 2> layers;
  bool Load(const std::filesystem::path& directory, std::string_view manifest_sha, bool route) {
    if (!Hex(manifest_sha)) return false;
    std::error_code error;
    const auto bytes = std::filesystem::file_size(directory / "inputs.tsv", error);
    if (error || bytes > 2048) return false;
    std::vector<std::byte> raw;
    if (!Read(directory / "inputs.tsv", static_cast<std::size_t>(bytes), raw) ||
        Hash(raw) != manifest_sha)
      return false;
    std::istringstream manifest(std::string(reinterpret_cast<const char*>(raw.data()), raw.size()));
    const std::array<const char*, 3> roles{"experts", "scales", "weights"};
    const std::array<std::size_t, 3> sizes{5767168, 2048, 2048};
    for (std::size_t layer = 0; layer < 2; ++layer) {
      for (std::size_t i = 0; i < (route ? 1U : 3U); ++i) {
        const std::string name =
            "layer-" + std::to_string(layer + 28) + "-" + (route ? "logits" : roles[i]) + ".bin";
        const auto size = route ? 32768U : sizes[i];
        std::string line;
        if (!std::getline(manifest, line) || line.size() > 256) return false;
        const auto first = line.find('\t'), second = line.find('\t', first + 1);
        if (first == line.npos || second == line.npos || line.substr(0, first) != name ||
            line.substr(first + 1, second - first - 1) != std::to_string(size) ||
            !Hex(line.substr(second + 1)))
          return false;
        if (!Read(directory / name, size, layers[layer][i]) ||
            Hash(layers[layer][i]) != line.substr(second + 1) || !Finite(layers[layer][i]))
          return false;
      }
    }
    return manifest.peek() == std::char_traits<char>::eof();
  }
};
inline bool Write(const std::filesystem::path& path, std::span<const std::byte> data) {
  FILE* stream = std::fopen(path.c_str(), "wx");
  if (!stream) return false;
  const bool complete = std::fwrite(data.data(), 1, data.size(), stream) == data.size() &&
                        std::fflush(stream) == 0 && fsync(fileno(stream)) == 0;
  const bool closed = std::fclose(stream) == 0;
  return complete && closed;
}
}  // namespace late_moe
#endif
