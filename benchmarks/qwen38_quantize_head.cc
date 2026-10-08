// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Offline selected-head preparation only. No runtime conversion or CUDA.
// Reads BF16 rows at a validated prepared-artifact offset and writes Q4_1
// using the locked GGML quantizer; the Python preparer authenticates and
// publishes the containing artifact. Buffers hold at most 128 rows.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <print>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/sha256.h"
#include "ggml.h"

namespace {
bool Number(const char* text, std::uint64_t& value) {
  const std::string_view s(text);
  const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == s.data() + s.size();
}
}  // namespace

int main(int argc, char** argv) {
  std::uint64_t offset = 0, width = 0, rows = 0;
  if (argc != 6 || !Number(argv[2], offset) || !Number(argv[3], width) || !Number(argv[4], rows) ||
      width != 2560 || rows == 0 || rows > 248320 ||
      offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) -
                   width * rows * sizeof(ggml_bf16_t)) {
    std::println(stderr, "usage: quantizer INPUT OFFSET WIDTH ROWS NEW_OUTPUT");
    return 1;
  }
  if (ggml_blck_size(GGML_TYPE_Q4_1) != 32 || ggml_type_size(GGML_TYPE_Q4_1) != 20) return 1;
  std::ifstream input(argv[1], std::ios::binary);
  std::ofstream output(argv[5], std::ios::binary | std::ios::noreplace);
  if (!input || !output) return 1;
  input.seekg(static_cast<std::streamoff>(offset));
  constexpr std::uint64_t batch_rows = 128;
  const auto batch = std::min(batch_rows, rows);
  std::vector<ggml_bf16_t> bf16(static_cast<std::size_t>(batch * width));
  std::vector<float> expanded(bf16.size());
  const auto row_bytes = ggml_row_size(GGML_TYPE_Q4_1, static_cast<std::int64_t>(width));
  std::vector<std::byte> packed(static_cast<std::size_t>(batch) * row_bytes);
  llmp::base::Sha256 input_digest;
  for (std::uint64_t first = 0; first < rows; first += batch_rows) {
    const auto count = std::min(batch_rows, rows - first);
    const auto values = static_cast<std::size_t>(count * width);
    const auto bytes = static_cast<std::streamsize>(values * sizeof(ggml_bf16_t));
    if (!input.read(reinterpret_cast<char*>(bf16.data()), bytes)) return 1;
    input_digest.Update(std::as_bytes(std::span(bf16).first(values)));
    for (std::size_t i = 0; i < values; ++i) {
      expanded[i] = ggml_bf16_to_fp32(bf16[i]);
      if (!std::isfinite(expanded[i])) return 1;
    }
    // Authenticate the reference quantizer's actual coefficient/reciprocal
    // range before its float-to-integer casts, without restricting raw
    // values that a finite Q4_1 scale/minimum can represent.
    for (std::size_t first_value = 0; first_value < values; first_value += 32) {
      const auto begin = expanded.begin() + static_cast<std::ptrdiff_t>(first_value);
      const auto [minimum, maximum] = std::minmax_element(begin, begin + 32);
      const float span = *maximum - *minimum;
      const float scale = span / 15.0F;
      const float inverse = scale != 0.0F ? 1.0F / scale : 0.0F;
      if (!std::isfinite(span) || !std::isfinite(scale) || !std::isfinite(inverse) ||
          !std::isfinite(ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale))) ||
          !std::isfinite(ggml_fp16_to_fp32(ggml_fp32_to_fp16(*minimum))))
        return 1;
      for (auto value = begin; value != begin + 32; ++value) {
        const float normalized = (*value - *minimum) * inverse;
        if (!std::isfinite(normalized) || normalized < 0.0F || normalized > 16.0F) return 1;
      }
    }
    const auto written = ggml_quantize_chunk(GGML_TYPE_Q4_1, expanded.data(), packed.data(), 0,
                                             static_cast<std::int64_t>(count),
                                             static_cast<std::int64_t>(width), nullptr);
    if (written != static_cast<std::size_t>(count) * row_bytes) return 1;
    // A finite BF16 row can exceed F16 coefficient range. Refuse rather
    // than publish a quantized head with nonfinite scale/minimum.
    constexpr std::size_t block_bytes = 20;
    for (std::size_t i = 0; i < written; i += block_bytes) {
      ggml_fp16_t scale{}, minimum{};
      std::memcpy(&scale, packed.data() + i, sizeof(scale));
      std::memcpy(&minimum, packed.data() + i + sizeof(scale), sizeof(minimum));
      if (!std::isfinite(ggml_fp16_to_fp32(scale)) || !std::isfinite(ggml_fp16_to_fp32(minimum)))
        return 1;
    }
    output.write(reinterpret_cast<const char*>(packed.data()),
                 static_cast<std::streamsize>(written));
    if (!output) return 1;
  }
  output.flush();
  if (!output) return 1;
  ggml_quantize_free();
  std::println("QWEN_SELECTED_HEAD_QUANTIZED rows={} width={} bytes={} input_sha256={}", rows,
               width, rows * row_bytes, llmp::base::ToHex(input_digest.Finish()));
  return 0;
}
