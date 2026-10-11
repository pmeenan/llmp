// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/reconstruction_tuning.h"

#include <algorithm>
#include <array>
#include <format>

namespace llmp::kernels::exl3 {
namespace {
// Exact copy of the 21 approved zero-workspace algorithms from
// docs/experiments/backend-proof-p0/exl3-recon-pin.json;
// SHA-256 10f16b9ba855dd1c612e3db57411be39b57ac6ac39774d3517e7b38607c571eb.
constexpr std::array kPins{
    LtAlgorithm{.config = {67, 315, 1, 0, 0, 29, 35, 0, 0},
                .m = 1023,
                .k = 896,
                .n = 128,
                .ldc = 128,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 315, 1, 0, 0, 29, 35, 0, 0},
                .m = 1024,
                .k = 896,
                .n = 128,
                .ldc = 128,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 312, 1, 0, 0, 105, 36, 0, 0},
                .m = 145,
                .k = 896,
                .n = 128,
                .ldc = 128,
                .output = Output::kF16},
    LtAlgorithm{.config = {21, 20, 1, 0, 0, 0, 11, 0, 0},
                .m = 1023,
                .k = 896,
                .n = 20864,
                .ldc = 151936,
                .output = Output::kF16},
    LtAlgorithm{.config = {21, 20, 1, 0, 0, 0, 11, 0, 0},
                .m = 1024,
                .k = 896,
                .n = 20864,
                .ldc = 151936,
                .output = Output::kF16},
    LtAlgorithm{.config = {21, 20, 1, 0, 0, 0, 11, 0, 0},
                .m = 145,
                .k = 896,
                .n = 20864,
                .ldc = 151936,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 412, 1, 0, 0, 31, 35, 0, 0},
                .m = 1023,
                .k = 896,
                .n = 32768,
                .ldc = 151936,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 412, 1, 0, 0, 31, 35, 0, 0},
                .m = 1024,
                .k = 896,
                .n = 32768,
                .ldc = 151936,
                .output = Output::kF16},
    LtAlgorithm{.config = {21, 15, 1, 0, 0, 0, 12, 0, 0},
                .m = 145,
                .k = 896,
                .n = 32768,
                .ldc = 151936,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 409, 1, 0, 0, 30, 35, 0, 0},
                .m = 1023,
                .k = 896,
                .n = 896,
                .ldc = 896,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 409, 1, 0, 0, 30, 35, 0, 0},
                .m = 1024,
                .k = 896,
                .n = 896,
                .ldc = 896,
                .output = Output::kF16},
    LtAlgorithm{.config = {67, 316, 1, 0, 0, 66, 35, 0, 0},
                .m = 145,
                .k = 896,
                .n = 896,
                .ldc = 896,
                .output = Output::kF16},
    LtAlgorithm{.config = {21, 20, 1, 0, 0, 0, 11, 0, 0},
                .m = 1023,
                .k = 896,
                .n = 4864,
                .ldc = 4864,
                .output = Output::kF32},
    LtAlgorithm{.config = {21, 20, 1, 0, 0, 0, 11, 0, 0},
                .m = 1024,
                .k = 896,
                .n = 4864,
                .ldc = 4864,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 193, 1, 0, 0, 18, 35, 0, 0},
                .m = 145,
                .k = 896,
                .n = 4864,
                .ldc = 4864,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 30, 1, 0, 0, 23, 35, 0, 0},
                .m = 1023,
                .k = 4864,
                .n = 896,
                .ldc = 896,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 30, 1, 0, 0, 23, 35, 0, 0},
                .m = 1024,
                .k = 4864,
                .n = 896,
                .ldc = 896,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 316, 1, 0, 0, 29, 36, 0, 0},
                .m = 145,
                .k = 4864,
                .n = 896,
                .ldc = 896,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 409, 1, 0, 0, 30, 35, 0, 0},
                .m = 1023,
                .k = 896,
                .n = 896,
                .ldc = 896,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 409, 1, 0, 0, 30, 35, 0, 0},
                .m = 1024,
                .k = 896,
                .n = 896,
                .ldc = 896,
                .output = Output::kF32},
    LtAlgorithm{.config = {67, 316, 1, 0, 0, 66, 35, 0, 0},
                .m = 145,
                .k = 896,
                .n = 896,
                .ldc = 896,
                .output = Output::kF32},
};
}  // namespace

std::expected<std::vector<LtAlgorithm>, std::string> ReconstructionAlgorithms(std::uint32_t arch,
                                                                              int rows, int k,
                                                                              int n,
                                                                              Output output) {
  if (arch != 121 || (rows != 145 && rows != 1023 && rows != 1024) || k < 128 ||
      k > kMaxDimension || n < 128 || n > kMaxDimension || k % 128 != 0 || n % 128 != 0) {
    return std::unexpected("no approved EXL3 reconstruction for this device and shape");
  }
  std::vector<LtAlgorithm> out;
  for (const int columns : ReconstructSlices(n)) {
    const auto pin = std::ranges::find_if(kPins, [&](const auto& p) {
      return p.m == rows && p.k == k && p.n == columns && p.ldc == n && p.output == output;
    });
    if (pin == kPins.end()) {
      return std::unexpected(std::format("no EXL3 reconstruction pin for {}x{}x{}, ldc {}, {}",
                                         rows, k, columns, n,
                                         output == Output::kF32 ? "F32" : "F16"));
    }
    out.push_back(*pin);
  }
  return out;
}

std::vector<execution::TunedShape> ReconstructionTuning(std::uint32_t arch) {
  std::vector<execution::TunedShape> out;
  if (arch != 121) return out;
  for (const auto& p : kPins) {
    // A slice is not a tunable full matrix. All slices, their full output
    // pitch and their output encoding must have pins before publishing it.
    if (!ReconstructionAlgorithms(arch, p.m, p.k, p.ldc, p.output)) continue;
    const execution::TunedShape shape{
        .implementation = p.m >= 1024 ? "exl3.linear.reconstruct_fused" : "exl3.linear.reconstruct",
        .rows = static_cast<std::uint64_t>(p.m),
        .k = static_cast<std::uint64_t>(p.k),
        .n = static_cast<std::uint64_t>(p.ldc),
        .output_encoding =
            p.output == Output::kF32 ? format::Encoding::kF32 : format::Encoding::kF16};
    if (std::ranges::find(out, shape) == out.end()) out.push_back(shape);
  }
  return out;
}
}  // namespace llmp::kernels::exl3
