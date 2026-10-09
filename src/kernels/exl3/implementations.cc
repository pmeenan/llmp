// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/implementations.h"

#include <cublas_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"
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

// ---------------------------------------------------------------- capabilities

// What each path computes, for the planner (D-108; execution/registry.h):
// a product of an EXL3 trellis in its tiles (operand 0; the multi-GEMM's
// second matrix is operand 1) with F16 activations (operand 1, or 2), at
// rung 1, the encoding's own kernels. The reconstruction paths expand a
// slice of the weights to F16 in workspace (D-052's bounded
// reconstruction), so Candidates() never offers them for a decode-shaped
// product; their predicate cannot see whether a plan holds each slice's
// pinned algorithm, which ReconGemm requires (implementations.h). Nothing
// is declared row-invariant or capturable: neither is established for
// these cooperative kernels (the M3 fixture path runs them uncaptured, so
// a plan that needs neither selects them as today). The bias add declares
// none: it is the linear's own operation, not a product.

constexpr std::array<format::Encoding, 1> kTrellis = {format::Encoding::kExl3Trellis};
constexpr std::array<format::Layout, 1> kTiles = {format::Layout::kExl3Tiles};
constexpr std::array<format::Encoding, 1> kF16 = {format::Encoding::kF16};
constexpr std::array<format::Layout, 1> kDense = {format::Layout::kDense};
constexpr std::array<execution::OperandRule, 2> kLinearRules = {{
    {.operand = 0, .encodings = kTrellis, .layouts = kTiles},
    {.operand = 1, .encodings = kF16, .layouts = kDense},
}};
constexpr std::array<execution::OperandRule, 3> kMultiRules = {{
    {.operand = 0, .encodings = kTrellis, .layouts = kTiles},
    {.operand = 1, .encodings = kTrellis, .layouts = kTiles},
    {.operand = 2, .encodings = kF16, .layouts = kDense},
}};

// The build's architectures ("121-real,86-real"): a device runs SASS built
// for its major version and an equal or lower minor, or for exactly its
// own with an "a" suffix.
struct Arch {
  std::uint32_t arch = 0;
  bool specific = false;
};

std::vector<Arch> CompiledArchitectures() {
  std::vector<Arch> out;
  const std::string_view list = LLMP_EXL3_CUDA_ARCHITECTURES;
  std::size_t at = 0;
  while (at < list.size()) {
    const std::size_t comma = std::min(list.find(',', at), list.size());
    const std::string_view entry = list.substr(at, comma - at);
    Arch a;
    const auto [end, error] = std::from_chars(entry.data(), entry.data() + entry.size(), a.arch);
    if (error == std::errc{} && a.arch != 0) {
      a.specific = end != entry.data() + entry.size() && *end == 'a';
      out.push_back(a);
    }
    at = comma + 1;
  }
  return out;
}

bool CompiledFor(std::uint32_t arch) {
  static const std::vector<Arch> compiled = CompiledArchitectures();
  return std::ranges::any_of(compiled, [&](const Arch& a) {
    return a.specific ? arch == a.arch : (arch / 10 == a.arch / 10 && arch >= a.arch);
  });
}

const execution::OperandFormat* OperandOf(const execution::CapabilityQuery& q,
                                          std::uint8_t operand) {
  const auto found = std::ranges::find_if(
      q.operands, [&](const execution::OperandFormat& o) { return o.operand == operand; });
  return found == q.operands.end() ? nullptr : &*found;
}

// What validate.h's checks of a packed or reconstructed linear refuse
// that the rules above cannot say: a rate and codebook without kernels in
// this build (CheckWeights: mcg at K = 4, 5, 6 or 8; the GEMV K = 4 only),
// a side beyond 2^19 or a trellis beyond the kernels' int indices
// (CheckWeights), rows beyond 2^16 (CheckRows), and rows times either
// side beyond int (CheckExtent, twice for the multi-GEMM's two matrices).
bool Accepts(const execution::CapabilityQuery& q, std::span<const std::uint8_t> weights,
             bool gemv) {
  if (!CompiledFor(q.arch)) {
    return false;
  }
  const execution::OperandFormat* first = nullptr;
  for (const std::uint8_t operand : weights) {
    const execution::OperandFormat* w = OperandOf(q, operand);
    if (w == nullptr || w->params.codebook != format::Exl3Codebook::kMcg ||
        !(gemv ? w->params.k_bits == 4 : RateCompiled(w->params.k_bits)) ||
        (first != nullptr && w->params != first->params)) {
      return false;
    }
    first = w;
  }
  if (first == nullptr) {
    return false;
  }
  const auto matrices = static_cast<std::uint64_t>(weights.size());
  constexpr auto kIntMax = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  constexpr auto kSideMin = static_cast<std::uint64_t>(kHadamardBlock);
  constexpr auto kSideMax = static_cast<std::uint64_t>(kMaxDimension);
  // Within these bounds no product below overflows 64 bits.
  return q.k >= kSideMin && q.n >= kSideMin && q.k <= kSideMax && q.n <= kSideMax && q.rows >= 1 &&
         q.rows <= static_cast<std::uint64_t>(kMaxRows) &&
         (q.k / 16) * (q.n / 16) * 16 * first->params.k_bits <= kIntMax &&
         q.rows * q.k * matrices <= kIntMax && q.rows * q.n * matrices <= kIntMax;
}

constexpr std::array<std::uint8_t, 1> kOneWeight = {0};
constexpr std::array<std::uint8_t, 2> kTwoWeights = {0, 1};

bool AcceptsLinear(const execution::CapabilityQuery& q) { return Accepts(q, kOneWeight, false); }
bool AcceptsGemv(const execution::CapabilityQuery& q) { return Accepts(q, kOneWeight, true); }
bool AcceptsMulti(const execution::CapabilityQuery& q) { return Accepts(q, kTwoWeights, false); }

// The predicate's checks, which the variant names (the identity covers the
// capability but not its predicate).
constexpr std::string_view kAcceptsText =
    "; selectable for the build's SASS architectures, the mcg codebook at K = 4, 5, 6 or 8, "
    "k and n from 128 to 2^19, 1 to 65,536 rows, the trellis and rows times k and n within int";
constexpr std::string_view kAcceptsGemvText =
    "; selectable for the build's SASS architectures, the mcg codebook at K = 4, k and n from 128 "
    "to 2^19, 1 to 8 rows, the trellis and rows times k and n within int";
constexpr std::string_view kAcceptsMultiText =
    "; selectable for the build's SASS architectures, two trellises of one rate and codebook "
    "(mcg at K = 4, 5, 6 or 8), k and n from 128 to 2^19, 1 to 65,536 rows, the trellis and "
    "twice rows times k and n within int";
// The reconstruction GEMM runs only the algorithms exl3-recon-pin.json
// pins, recorded on the GB10, for exactly its GEMM: the plan supplies them.
constexpr std::string_view kAcceptsReconstructText =
    "; selectable on the GB10 (sm_121, where its cuBLASLt algorithms were pinned) with the "
    "build's SASS, the mcg codebook at K = 4, 5, 6 or 8, k and n from 128 to 2^19, 1 to 65,536 "
    "rows, the trellis and rows times k and n within int; a plan supplies the pinned algorithm "
    "of each slice's GEMM";

std::optional<execution::Capability> CapabilityOf(Path path) {
  execution::Capability c;
  c.op = ir::Primitive(ir::Op::kMatMul);
  c.operands = kLinearRules;
  c.device.min_arch = 80;  // mma.sync m16n8k16 F16, cp.async, ldmatrix
  c.shape = {.min_rows = 1,
             .max_rows = static_cast<std::uint64_t>(kMaxRows),
             .k_multiple = static_cast<std::uint64_t>(kHadamardBlock),
             .n_multiple = static_cast<std::uint64_t>(kHadamardBlock)};
  c.rung = execution::Rung::kNative;
  c.expansion = execution::Expansion::kNone;
  c.row_invariant = false;
  c.capturable = false;
  c.accepts = &AcceptsLinear;
  switch (path) {
    case Path::kGemm:
      c.priority = 2;
      return c;
    case Path::kGemv:
      c.shape.max_rows = static_cast<std::uint64_t>(kGemvMaxRows);
      c.priority = 3;
      c.accepts = &AcceptsGemv;
      return c;
    case Path::kReconstruct:
    case Path::kReconstructFused:
      c.device.min_arch = 121;
      c.device.max_arch = 121;
      c.expansion = execution::Expansion::kTransientWorkspace;
      // Only for the shapes exl3-recon-pin.json pins (the GEMM refuses
      // any other): the planner drops it elsewhere.
      c.needs_tuning = true;
      c.priority = path == Path::kReconstruct ? 1 : 0;
      return c;
    case Path::kMultiGemm:
      c.op = ir::Fused(ir::FusedKind::kPreparedProducts);
      c.operands = kMultiRules;
      c.accepts = &AcceptsMulti;
      return c;
    case Path::kBias:
      break;
  }
  return std::nullopt;
}

std::string_view AcceptsText(Path path) {
  switch (path) {
    case Path::kGemm:
      return kAcceptsText;
    case Path::kGemv:
      return kAcceptsGemvText;
    case Path::kReconstruct:
    case Path::kReconstructFused:
      return kAcceptsReconstructText;
    case Path::kMultiGemm:
      return kAcceptsMultiText;
    case Path::kBias:
      break;
  }
  return "";
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
  return execution::Implementation{
      .name = std::string(entry.name),
      .operation = entry.operation,
      .source = "exllamav3",
      .revision = LLMP_EXL3_SOURCE_TREE,
      .build = std::move(build),
      .variant = std::format("{}{}", entry.variant, AcceptsText(entry.path)),
      .capability = CapabilityOf(entry.path)};
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
