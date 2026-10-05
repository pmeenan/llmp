// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_BENCHMARK_GEMMA_ATTENTION_REPLAY_INPUTS_H_
#define JITLLM_BENCHMARK_GEMMA_ATTENTION_REPLAY_INPUTS_H_
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "ggml.h"
namespace attention_replay {
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
static_assert(GGML_MAX_OP_PARAMS == 64 && GGML_TYPE_F32 == 0 && GGML_TYPE_F16 == 1 &&
              GGML_PREC_F32 == 10);
inline constexpr std::array<const char*, 5> kNames{"q", "k", "v", "mask", "params"};
inline constexpr std::array<std::size_t, 5> kBytes{131072, 8388608, 8388608, 65536, 64};
inline constexpr std::size_t kOutputElements = 4 * 32 * 256, kOutputBytes = kOutputElements * 4;
inline constexpr std::uint64_t kHostAllowance = 64U << 20U;
struct Inputs {
  std::array<std::vector<std::byte>, 5> data;
  bool Read(const std::filesystem::path& source) {
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i].resize(kBytes[i]);
      std::ifstream file(source / (std::string(kNames[i]) + ".bin"), std::ios::binary);
      file.read(reinterpret_cast<char*>(data[i].data()),
                static_cast<std::streamsize>(data[i].size()));
      if (!file || file.peek() != std::char_traits<char>::eof()) return false;
    }
    std::array<std::int32_t, 16> parameters{};
    std::memcpy(parameters.data(), data[4].data(), 64);
    const std::array<std::int32_t, 16> expected{1065353216, 0, 0, 10};
    return parameters == expected;
  }
};
inline ggml_tensor* Attention(ggml_context* ctx, ggml_tensor* q_root, ggml_tensor* k_root,
                              ggml_tensor* v_root, ggml_tensor* mask, const Inputs& input) {
  auto* q = ggml_permute(ctx, q_root, 0, 2, 1, 3);
  auto* k = ggml_permute(ctx, k_root, 0, 2, 1, 3);
  auto* v = ggml_permute(ctx, v_root, 0, 2, 1, 3);
  auto* out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, 0);
  std::memcpy(out->op_params, input.data[4].data(), 64);
  return out;
}
inline bool Write(const std::filesystem::path& path, std::span<const float> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  return bool(file);
}
inline bool Finite(std::span<const float> values) {
  for (float v : values)
    if (!std::isfinite(v)) return false;
  return true;
}
}  // namespace attention_replay
#endif
