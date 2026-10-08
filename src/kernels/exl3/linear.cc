// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/linear.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "kernels/exl3/launch.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// The bias's operands over the linear's output, in place; checked before
// the product is queued, so that a refused bias queues nothing.
std::expected<BiasOperands, KernelFailure> BiasOver(std::uint64_t y, Output output, int m, int n,
                                                    std::uint64_t bias) {
  if (output != Output::kF16) {
    return Rejected("a bias needs an F16 output (add_kernel_hhh)");
  }
  const BiasOperands operands{.x = y, .bias = bias, .y = y, .rows = m, .columns = n};
  if (auto checked = CheckBias(operands); !checked) {
    return std::unexpected(checked.error());
  }
  return operands;
}

}  // namespace

std::expected<void, KernelFailure> PackedGemmLinear(LaunchContext& launch,
                                                    const LinearOperands& operands,
                                                    const GemmPlan& plan, std::uint64_t bias) {
  std::expected<BiasOperands, KernelFailure> add = BiasOperands{};
  if (bias != 0) {
    add = BiasOver(operands.y, operands.output, operands.m, operands.weights.n, bias);
    if (!add) {
      return std::unexpected(add.error());
    }
    if (auto apart = CheckBiasApart(operands, bias, launch.locks()); !apart) {
      return apart;
    }
  }
  if (auto product = launch.Gemm(operands, plan); !product) {
    return product;
  }
  return bias == 0 ? std::expected<void, KernelFailure>() : launch.Bias(*add);
}

std::expected<void, KernelFailure> PackedGemvLinear(LaunchContext& launch,
                                                    const LinearOperands& operands,
                                                    const GemvPlan& plan, std::uint64_t bias) {
  std::expected<BiasOperands, KernelFailure> add = BiasOperands{};
  if (bias != 0) {
    add = BiasOver(operands.y, operands.output, operands.m, operands.weights.n, bias);
    if (!add) {
      return std::unexpected(add.error());
    }
    if (auto apart = CheckBiasApart(operands, bias, launch.locks()); !apart) {
      return apart;
    }
  }
  if (auto product = launch.Gemv(operands, plan); !product) {
    return product;
  }
  return bias == 0 ? std::expected<void, KernelFailure>() : launch.Bias(*add);
}

std::expected<void, KernelFailure> MultiLinear(LaunchContext& launch,
                                               const MultiLinearOperands& operands,
                                               const MultiGemmPlan& plan) {
  return launch.MultiGemm(operands, plan);
}

std::expected<void, KernelFailure> ReconstructedLinear(LaunchContext& launch, ReconGemm& gemm,
                                                       const ReconstructedOperands& o, bool fused,
                                                       std::span<const LtAlgorithm> algorithms) {
  // Everything is checked before the first launch, so that a refusal queues
  // nothing: the transforms, every slice and its GEMM, and the bias.
  const Weights& w = o.weights;
  if (auto weights = CheckWeights(w); !weights) {
    return weights;
  }
  if (auto scratch = CheckReconstructedScratch(o, fused); !scratch) {
    return scratch;
  }
  const std::vector<int> slices = ReconstructSlices(w.n);
  if (algorithms.size() != slices.size()) {
    return Rejected(
        std::format("{} pinned algorithms for {} slices", algorithms.size(), slices.size()));
  }
  const HadamardOperands input{.x = o.x,
                               .y = o.xh,
                               .scale = w.suh,
                               .rows = o.m,
                               .columns = w.k,
                               .type = Output::kF16,
                               .input_scale = true};
  const HadamardOperands output{.x = o.y,
                                .y = o.y,
                                .scale = w.svh,
                                .rows = o.m,
                                .columns = w.n,
                                .type = o.output,
                                .input_scale = false};
  if (!fused) {
    for (const HadamardOperands* transform : {&input, &output}) {
      if (auto checked = CheckHadamard(*transform); !checked) {
        return checked;
      }
    }
  }
  const std::uint64_t product_input = fused ? o.x : o.xh;
  const auto out_bytes = static_cast<std::uint64_t>(OutputBytes(o.output));
  std::vector<ReconstructOperands> reconstructs;
  std::vector<ReconGemmOperands> products;
  int column = 0;
  for (const int columns : slices) {
    reconstructs.push_back(
        {.weights = w, .w = o.w, .column = column, .columns = columns, .fused = fused});
    products.push_back({.w = o.w,
                        .x = product_input,
                        .y = o.y + (static_cast<std::uint64_t>(column) * out_bytes),
                        .m = o.m,
                        .k = w.k,
                        .n = columns,
                        .ldc = w.n,
                        .output = o.output});
    if (auto checked = CheckReconstruct(reconstructs.back()); !checked) {
      return checked;
    }
    if (auto checked =
            gemm.Check(products.back(), algorithms[reconstructs.size() - 1], launch.sm_count());
        !checked) {
      return checked;
    }
    column += columns;
  }
  std::expected<BiasOperands, KernelFailure> add = BiasOperands{};
  if (o.bias != 0) {
    add = BiasOver(o.y, o.output, o.m, w.n, o.bias);
    if (!add) {
      return std::unexpected(add.error());
    }
  }

  if (!fused) {
    if (auto transformed = launch.Hadamard(input); !transformed) {
      return transformed;
    }
  }
  for (std::size_t i = 0; i < slices.size(); ++i) {
    if (auto reconstructed = launch.Reconstruct(reconstructs[i]); !reconstructed) {
      return reconstructed;
    }
    if (auto product = gemm.Run(launch, products[i], algorithms[i], launch.sm_count()); !product) {
      return product;
    }
  }
  if (!fused) {
    if (auto transformed = launch.Hadamard(output); !transformed) {
      return transformed;
    }
  }
  return o.bias == 0 ? std::expected<void, KernelFailure>() : launch.Bias(*add);
}

}  // namespace llmp::kernels::exl3
