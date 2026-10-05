// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_BENCHMARK_DENSE_FFN_INPUTS_H_
#define JITLLM_BENCHMARK_DENSE_FFN_INPUTS_H_
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "ggml.h"
namespace ffn_replay {
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
static_assert(GGML_TYPE_Q4_K == 12 && GGML_TYPE_Q6_K == 14);
inline constexpr std::size_t kWidth = 5376, kHidden = 21504;
inline constexpr std::array<const char*, 4> kNames{"input", "gate", "up", "down"};
inline constexpr std::array<std::size_t, 4> kLogical{kWidth * 4, 65028096, 65028096, 94832640};
inline constexpr std::array<std::size_t, 4> kBytes{kWidth * 4, 65028240, 65028240, 94832640};
inline constexpr std::size_t kOutputElements = kHidden + kWidth, kOutputBytes = kOutputElements * 4;
// Full input payload plus largest device-root witness, fresh input and retained
// full outputs. Metadata/allocator overhead has a separate fixed allowance.
inline constexpr std::uint64_t kHostAllowance = 384ULL << 20;
inline bool Finite(std::span<const float> values) {
  for (auto v : values)
    if (!std::isfinite(v)) return false;
  return true;
}
inline bool Write(const std::filesystem::path& path, std::span<const float> values) {
  std::ofstream f(path, std::ios::binary | std::ios::noreplace);
  f.write(reinterpret_cast<const char*>(values.data()),
          static_cast<std::streamsize>(values.size_bytes()));
  f.flush();
  return bool(f);
}
struct Inputs {
  std::array<std::vector<std::byte>, 4> data;
  std::array<std::array<std::byte, GGML_MAX_OP_PARAMS>, 4> params{};
  std::vector<float> captured;
  bool Matches(const std::array<ggml_tensor*, 3>& products, ggml_tensor* glu) const {
    for (std::size_t i = 0; i < 3; ++i)
      if (std::memcmp(params[i].data(), products[i]->op_params, GGML_MAX_OP_PARAMS) != 0)
        return false;
    return std::memcmp(params[3].data(), glu->op_params, GGML_MAX_OP_PARAMS) == 0;
  }
  bool Read(const std::filesystem::path& path) {
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i].resize(kBytes[i]);
      std::ifstream f(path / (std::string(kNames[i]) + ".bin"), std::ios::binary);
      f.read(reinterpret_cast<char*>(data[i].data()), static_cast<std::streamsize>(data[i].size()));
      if (!f || f.peek() != std::char_traits<char>::eof()) return false;
      if (i == 1 || i == 2)
        for (std::size_t j = kLogical[i]; j < kBytes[i]; ++j)
          if (data[i][j] != std::byte{0}) return false;
    }
    std::ifstream parameters(path / "params.bin", std::ios::binary);
    parameters.read(reinterpret_cast<char*>(params.data()), sizeof(params));
    if (!parameters || parameters.peek() != std::char_traits<char>::eof()) return false;
    std::array<float, kWidth> x{};
    std::memcpy(x.data(), data[0].data(), kLogical[0]);
    captured.resize(kOutputElements);
    for (std::size_t i = 0; i < 2; ++i) {
      std::ifstream f(path / (i == 0 ? "activation.bin" : "down-output.bin"), std::ios::binary);
      const auto count = i == 0 ? kHidden : kWidth;
      f.read(reinterpret_cast<char*>(captured.data() + (i == 0 ? 0 : kHidden)),
             static_cast<std::streamsize>(count * 4));
      if (!f || f.peek() != std::char_traits<char>::eof()) return false;
    }
    return Finite(x) && Finite(captured);
  }
};
}  // namespace ffn_replay
#endif
