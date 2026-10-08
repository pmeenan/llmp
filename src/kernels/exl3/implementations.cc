// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/implementations.h"

#include <cublas_api.h>

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/linear.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(LLMP_EXL3_SOURCE_TREE) || !defined(LLMP_EXL3_SDK) || !defined(LLMP_EXL3_TARGET) || \
    !defined(LLMP_EXL3_CUDA_ARCHITECTURES) || !defined(LLMP_EXL3_BUILD_TYPE) ||                 \
    !defined(LLMP_EXL3_SANITIZE)
#error \
    "implementations.cc needs the ExLlamaV3 source tree, SDK, target, architectures and build type"
#endif

namespace llmp::kernels::exl3 {
namespace {

enum class Path : std::uint8_t { kGemm, kGemv, kReconstruct, kReconstructFused, kMultiGemm, kBias };

}  // namespace

struct Kernel::Entry {
  std::string_view name;
  execution::Operation operation;
  Path path;
  std::string_view variant;
};

namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif
// And whether it checks libstdc++'s preconditions (D-083), as the GGML
// module's identities record.
#ifdef _GLIBCXX_ASSERTIONS
constexpr std::string_view kLibraryAsserts = "libstdc++ assertions";
#else
constexpr std::string_view kLibraryAsserts = "no libstdc++ assertions";
#endif

constexpr std::array<Kernel::Entry, 6> kEntries = {{
    {.name = "exl3.linear.gemm",
     .operation = execution::Operation::kQuantLinear,
     .path = Path::kGemm,
     .variant = "exl3_gemm_kernel<K, false, fp32, 1, shape> at the plan's tile shape and grid, "
                "cooperative, the context's zeroed lock slots; then add_kernel_hhh for a bias"},
    {.name = "exl3.linear.gemv",
     .operation = execution::Operation::kQuantLinear,
     .path = Path::kGemv,
     .variant = "exl3_gemv_kernel<4, fp32, 1, m == 1 ? 0 : 1, cfg, false> at the plan's "
                "configuration and grid, cooperative; then add_kernel_hhh for a bias"},
    {.name = "exl3.linear.reconstruct",
     .operation = execution::Operation::kQuantLinear,
     .path = Path::kReconstruct,
     .variant = "had_hf_r_128_kernel<true, false>; per slice of at most 32768 columns "
                "reconstruct_kernel<K, 1, false> and cublasLtMatmul (COMPUTE_32F, the plan's "
                "pinned algorithm, no workspace); had_{hf,ff}_r_128_kernel<false, true>; then "
                "add_kernel_hhh for a bias"},
    {.name = "exl3.linear.reconstruct_fused",
     .operation = execution::Operation::kQuantLinear,
     .path = Path::kReconstructFused,
     .variant = "per slice of at most 32768 columns reconstruct_had_kernel<K, 1, false> and "
                "cublasLtMatmul (COMPUTE_32F, the plan's pinned algorithm, no workspace); then "
                "add_kernel_hhh for a bias"},
    {.name = "exl3.multi_linear.mgemm",
     .operation = execution::Operation::kQuantMultiLinear,
     .path = Path::kMultiGemm,
     .variant = "exl3_mgemm_kernel<K, false, fp32, 1, shape> for two matrices of one input at "
                "the plan's tile shape, grid and concurrency, cooperative, no indices, weights, "
                "range or slices"},
    {.name = "exl3.bias_add",
     .operation = execution::Operation::kBiasAdd,
     .path = Path::kBias,
     .variant = "add_kernel_hhh as add_gr launches it (grid ceil(rows x columns / 1024), 1024 "
                "threads): F16 + F16, one rounding; the linear's bias on every path, as its own "
                "operation of the plan"},
}};

bool Reconstructs(Path path) {
  return path == Path::kReconstruct || path == Path::kReconstructFused;
}

execution::Implementation Declaration(const Kernel::Entry& entry) {
  std::string build = std::format(
      "sdk {}; target {}; cuda architectures {} (upstream's units, llmp_exl3_cuda); build type "
      "{} ({}, {}); sanitize {}; module sources {}",
      LLMP_EXL3_SDK, LLMP_EXL3_TARGET, LLMP_EXL3_CUDA_ARCHITECTURES, LLMP_EXL3_BUILD_TYPE, kAsserts,
      kLibraryAsserts, LLMP_EXL3_SANITIZE, ModuleSourcesDigest());
  if (Reconstructs(entry.path)) {
    build += std::format("; cuBLASLt {}", CUBLAS_VERSION);
  }
  return execution::Implementation{.name = std::string(entry.name),
                                   .operation = entry.operation,
                                   .source = "exllamav3",
                                   .revision = LLMP_EXL3_SOURCE_TREE,
                                   .build = std::move(build),
                                   .variant = std::string(entry.variant)};
}

std::expected<void, KernelFailure> Takes(const Kernel::Entry& entry, Path path,
                                         std::string_view call) {
  if (entry.path != path) {
    return Rejected(std::format("{} does not take {}", entry.name, call));
  }
  return {};
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kEntries.size());
  for (const Kernel::Entry& entry : kEntries) {
    declared.push_back(Declaration(entry));
  }
  return declared;
}

std::expected<Kernel, KernelFailure> Kernel::Bind(const execution::Implementation& implementation) {
  for (const Entry& entry : kEntries) {
    if (entry.name != implementation.name) {
      continue;
    }
    const execution::Implementation declared = Declaration(entry);
    if (execution::IdentityOf(declared) != execution::IdentityOf(implementation)) {
      return Rejected(std::format("{} is not this build's {}", implementation.name, entry.name));
    }
    return Kernel(entry);
  }
  return Rejected(std::format("the EXL3 module has no implementation {}", implementation.name));
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               const LinearOperands& operands, const GemmPlan& plan,
                                               std::uint64_t bias) const {
  if (auto takes = Takes(*entry_, Path::kGemm, "a GEMM plan"); !takes) {
    return takes;
  }
  return PackedGemmLinear(launch, operands, plan, bias);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               const LinearOperands& operands, const GemvPlan& plan,
                                               std::uint64_t bias) const {
  if (auto takes = Takes(*entry_, Path::kGemv, "a GEMV plan"); !takes) {
    return takes;
  }
  return PackedGemvLinear(launch, operands, plan, bias);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch, ReconGemm& gemm,
                                               const ReconstructedOperands& operands,
                                               std::span<const LtAlgorithm> algorithms) const {
  if (!Reconstructs(entry_->path)) {
    return Rejected(std::format("{} does not reconstruct", entry_->name));
  }
  return ReconstructedLinear(launch, gemm, operands, entry_->path == Path::kReconstructFused,
                             algorithms);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               const MultiLinearOperands& operands,
                                               const MultiGemmPlan& plan) const {
  if (auto takes = Takes(*entry_, Path::kMultiGemm, "a multi-GEMM plan"); !takes) {
    return takes;
  }
  return MultiLinear(launch, operands, plan);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               const BiasOperands& operands) const {
  if (auto takes = Takes(*entry_, Path::kBias, "a bias add"); !takes) {
    return takes;
  }
  return launch.Bias(operands);
}

std::string_view Kernel::name() const { return entry_->name; }

execution::Operation Kernel::operation() const { return entry_->operation; }

}  // namespace llmp::kernels::exl3
