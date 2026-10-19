// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/implementations.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(LLMP_GGML_SOURCE_TREE) || !defined(LLMP_DS4_SOURCE_TREE) ||        \
    !defined(LLMP_GGML_SDK) || !defined(LLMP_GGML_TARGET) ||                    \
    !defined(LLMP_GGML_CUDA_ARCHITECTURES) || !defined(LLMP_GGML_BUILD_TYPE) || \
    !defined(LLMP_GGML_SANITIZE)
#error "implementations.cc needs the GGML source tree, SDK, target, architectures and build type"
#endif

namespace llmp::kernels::ggml {

struct RmsNormMulKernel::Entry {
  std::string_view name;
  std::string_view variant;
  std::expected<void, KernelFailure> (*check)(const ggml_tensor* norm, const ggml_tensor* mul);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch, ggml_tensor* norm,
                                            ggml_tensor* mul);
};

struct Kernel::Entry {
  std::string_view name;
  llmp::ir::OpKey operation;
  std::string_view variant;
  std::size_t arity;
  // Called with exactly `arity` nodes, or validator-owned bounded arity if zero.
  std::expected<void, KernelFailure> (*check)(std::span<const ggml_tensor* const> nodes);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch,
                                            std::span<ggml_tensor* const> nodes);
};

namespace {

// Whether this build keeps asserts, GGML's device asserts among them.
#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif
// And whether it checks libstdc++'s preconditions (D-083).
#ifdef _GLIBCXX_ASSERTIONS
constexpr std::string_view kLibraryAsserts = "libstdc++ assertions";
#else
constexpr std::string_view kLibraryAsserts = "no libstdc++ assertions";
#endif

#define LLMP_GGML_NORM_RECIPE(NAME, VARIANT, CHECK, RUN) \
  {.name = NAME, .variant = VARIANT, .check = CHECK, .run = RUN},
#define LLMP_GGML_RECIPE(NAME, KEY, VARIANT, ARITY, CHECK, RUN)
constexpr std::array<RmsNormMulKernel::Entry, 2> kRmsNormMul = {{
#include "kernels/ggml/recipe_entries.inc"
}};
#undef LLMP_GGML_RECIPE
#undef LLMP_GGML_NORM_RECIPE

// The other implementations (implementations.h), each checking and running
// its nodes through ops.h.
using Nodes = std::span<ggml_tensor* const>;
using ConstNodes = std::span<const ggml_tensor* const>;

// Matching only reads descriptors. Borrow a fixed stack view for the shared
// graph matcher; no tensor metadata or ownership is changed here.
template <std::size_t N>
std::array<ggml_tensor*, N> Borrow(ConstNodes nodes) {
  std::array<ggml_tensor*, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = const_cast<ggml_tensor*>(nodes[i]);
  return out;
}
std::unexpected<KernelFailure> InvalidGemmaChain() {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = "invalid checked Gemma MoE chain"});
}

#define LLMP_GGML_NORM_RECIPE(NAME, VARIANT, CHECK, RUN)
#define LLMP_GGML_RECIPE(NAME, KEY, VARIANT, ARITY, CHECK, RUN) \
  {.name = NAME, .operation = KEY, .variant = VARIANT, .arity = ARITY, .check = CHECK, .run = RUN},
constexpr std::array<Kernel::Entry, 155> kKernels = {{
#include "kernels/ggml/recipe_entries.inc"
}};
#undef LLMP_GGML_RECIPE
#undef LLMP_GGML_NORM_RECIPE

// Its capability is capabilities.h's, whose predicate its variant names.
execution::Implementation Declare(std::string_view name, llmp::ir::OpKey operation,
                                  std::string_view variant) {
  const execution::Capability* capability = CapabilityOf(name);
  // The grouped GEMM and the MXFP8 product are CUTLASS's kernels: their
  // identities name that tree too.
  const bool cutlass = name == kMoeGemmName || name == kMxfp8GemmName;
  const bool ds4 = name == kMulMatIdQ2D2r || name == kDsv4HcaTokentileName ||
                   name == kDsv4OutAName || name == kDsv4OutAFastPackName;
  std::string source = "ggml";
  std::string revision =
      std::format("ggml tree {}; llmp module {}", LLMP_GGML_SOURCE_TREE, ModuleSourcesDigest());
  if (ds4) {
    source = "ds4";
    revision = std::format("ds4 tree {}; ggml tree {}; llmp module {}", LLMP_DS4_SOURCE_TREE,
                           LLMP_GGML_SOURCE_TREE, ModuleSourcesDigest());
  } else if (cutlass) {
    source = "cutlass";
    revision = std::format("cutlass tree {}; ggml tree {}; llmp module {}",
                           LLMP_CUTLASS_SOURCE_TREE, LLMP_GGML_SOURCE_TREE, ModuleSourcesDigest());
  }
  return {.name = std::string(name),
          .operation = capability ? capability->op : operation,
          .source = std::move(source),
          .revision = std::move(revision),
          .build = std::format("sdk {}; target {}; cuda {}; build type {}; {}; {}; sanitizers {}",
                               LLMP_GGML_SDK, LLMP_GGML_TARGET, LLMP_GGML_CUDA_ARCHITECTURES,
                               LLMP_GGML_BUILD_TYPE, kAsserts, kLibraryAsserts, LLMP_GGML_SANITIZE),
          .variant = capability == nullptr ? std::string(variant)
                                           : std::format("{}{}", variant, AcceptsTextOf(name)),
          .capability = capability == nullptr ? std::nullopt : std::optional(*capability)};
}

execution::Implementation Declare(const RmsNormMulKernel::Entry& entry) {
  return Declare(entry.name, llmp::ir::Fused(llmp::ir::FusedKind::kRmsNormMul), entry.variant);
}

execution::Implementation Declare(const Kernel::Entry& entry) {
  return Declare(entry.name, entry.operation, entry.variant);
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kRmsNormMul.size() + kKernels.size());
  for (const RmsNormMulKernel::Entry& entry : kRmsNormMul) {
    declared.push_back(Declare(entry));
  }
  for (const Kernel::Entry& entry : kKernels) {
    declared.push_back(Declare(entry));
  }
  // The planner's implementations that are no kernel of their own: a node
  // whose lowering launches others' (capabilities.h, Capability::launches).
  for (const CapabilityDeclaration& d : CapabilityDeclarations()) {
    if (std::ranges::none_of(
            declared, [&](const execution::Implementation& i) { return i.name == d.name; })) {
      declared.push_back(
          Declare(d.name, d.capability->op, "launches the module's steps its capability lists"));
    }
  }
  return declared;
}

std::expected<RmsNormMulKernel, KernelFailure> RmsNormMulKernel::Bind(
    const execution::Implementation& implementation) {
  for (const Entry& entry : kRmsNormMul) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return RmsNormMulKernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML rms_norm_mul implementation of this build",
                            implementation.name)});
}

std::expected<void, KernelFailure> RmsNormMulKernel::Check(const ggml_tensor* norm,
                                                           const ggml_tensor* mul) const {
  return entry_->check(norm, mul);
}

std::expected<void, KernelFailure> RmsNormMulKernel::Run(LaunchContext& launch, ggml_tensor* norm,
                                                         ggml_tensor* mul) const {
  return entry_->run(launch, norm, mul);
}

std::string_view RmsNormMulKernel::name() const { return entry_->name; }

bool UsesCublas(std::string_view implementation) {
  // The declarations above whose launchers take the lent cuBLAS handle
  // (MulMatCublas, RunGemmBf16); a new one belongs here.
  return implementation == "ggml.mul_mat.cublas" || implementation == "llmp.gemm.bf16" ||
         implementation == kGemmBf16LtName || implementation == kConv2dTiledF32 ||
         implementation == kConv1x1Name;
}

std::expected<Kernel, KernelFailure> Kernel::Bind(const execution::Implementation& implementation) {
  for (const Entry& entry : kKernels) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return Kernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML implementation of this build", implementation.name)});
}

std::expected<void, KernelFailure> Kernel::Check(std::span<const ggml_tensor* const> nodes) const {
  if (entry_->arity != 0 && nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->check(nodes);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               std::span<ggml_tensor* const> nodes) const {
  if (entry_->arity != 0 && nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->run(launch, nodes);
}

std::string_view Kernel::name() const { return entry_->name; }

llmp::ir::OpKey Kernel::operation() const {
  const auto* capability = CapabilityOf(entry_->name);
  return capability ? capability->op : entry_->operation;
}

std::size_t Kernel::arity() const { return entry_->arity; }

}  // namespace llmp::kernels::ggml
