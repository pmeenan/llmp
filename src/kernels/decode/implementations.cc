// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/implementations.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"
#include "kernels/decode/launch.h"
#include "kernels/decode/operand.h"
#include "kernels/decode/products.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(LLMP_DECODE_SDK) || !defined(LLMP_DECODE_TARGET) ||                    \
    !defined(LLMP_DECODE_CUDA_ARCHITECTURES) || !defined(LLMP_DECODE_BUILD_TYPE) || \
    !defined(LLMP_DECODE_SANITIZE)
#error "implementations.cc needs the SDK, target, architectures and build type"
#endif

namespace llmp::kernels::decode {
namespace {

using execution::Capability;
using execution::Implementation;
using execution::OperandRule;
using execution::Operation;
using format::Encoding;
using format::Layout;

#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif
#ifdef _GLIBCXX_ASSERTIONS
constexpr std::string_view kLibraryAsserts = "libstdc++ assertions";
#else
constexpr std::string_view kLibraryAsserts = "no libstdc++ assertions";
#endif

constexpr std::size_t kDecoderCount = static_cast<std::size_t>(DecoderId::kCount);

// Each decoder's encoding and layout, for the operand rules' spans.
constexpr std::array<Encoding, kDecoderCount> kEncodings = {
    Encoding::kBF16,  Encoding::kF16,   Encoding::kQ8_0,  Encoding::kNvfp4, Encoding::kNvfp4,
    Encoding::kNvfp4, Encoding::kNvfp4, Encoding::kMxfp8, Encoding::kMxfp8, Encoding::kAffine};
constexpr std::array<Layout, kDecoderCount> kLayouts = {
    Layout::kDense,      Layout::kDense,          Layout::kGgmlBlocks,      Layout::kSplit,
    Layout::kScaleAtoms, Layout::kRowInterleaved, Layout::kGgmlNvfp4Blocks, Layout::kSplit,
    Layout::kScaleAtoms, Layout::kSplit};
constexpr std::array<Encoding, 1> kF32 = {Encoding::kF32};
constexpr std::array<Encoding, 1> kI32 = {Encoding::kI32};
constexpr std::array<Layout, 1> kDense = {Layout::kDense};

// The GLUs over two decoders that decode_mixed.cu instantiates.
constexpr std::array<std::array<DecoderId, 2>, 2> kMixedGlu = {{
    {DecoderId::kMxfp8Split, DecoderId::kBf16Dense},
    {DecoderId::kNvfp4Split, DecoderId::kQ8_0Blocks},
}};

constexpr std::array<std::string_view, static_cast<std::size_t>(Template::kCount)> kTemplateNames =
    {"vector", "vector_glu", "matrix", "matrix_fma", "grouped", "rows"};

bool Dense(DecoderId id) { return id == DecoderId::kBf16Dense || id == DecoderId::kF16Dense; }

OperandRule WeightRule(std::uint8_t operand, DecoderId id) {
  const auto i = static_cast<std::size_t>(id);
  return {.operand = operand,
          .encodings = std::span(kEncodings).subspan(i, 1),
          .layouts = std::span(kLayouts).subspan(i, 1)};
}

OperandRule DenseRule(std::uint8_t operand, std::span<const Encoding> encodings) {
  return {.operand = operand, .encodings = encodings, .layouts = kDense};
}

bool Accepts(const execution::CapabilityQuery& q) {
  if (!CompiledFor(q.arch) || q.k >= (std::uint64_t{1} << 30U) ||
      q.n >= (std::uint64_t{1} << 31U)) {
    return false;
  }
  for (const auto& o : q.operands) {
    if (o.encoding == Encoding::kAffine && o.params.bits != 0) {
      if (!((o.params.bits == 4 || o.params.bits == 8) &&
            (o.params.group == 32 || o.params.group == 64 || o.params.group == 128) &&
            (o.params.scale == format::AffineScale::kBF16 ||
             o.params.scale == format::AffineScale::kF16) &&
            !o.params.zero_point && (q.k == 0 || q.k % o.params.group == 0))) {
        return false;
      }
    }
  }
  return true;
}

std::string TemplateText(Template t, std::string_view mma) {
  switch (t) {
    case Template::kVector:
      return "VectorKernel<D, cols 1|2|4|8|16>: a warp per output row, 8 warps a block; each lane "
             "decodes 16-value segments at a 32-segment stride into registers and sums each input "
             "row with fmaf in k order, then a warp sum, times the tensor scale";
    case Template::kVectorGlu:
      return "VectorGluKernel<gate, up, cols 1|2|4|8|16>: VectorKernel's sums of the gate's and "
             "the up's rows, each times its tensor scale, then ir::GluKind's GLU in F32";
    case Template::kMatrix:
      return std::format(
          "TileKernel<D, 16|64 rows, mma>: 64 outputs by 16 or 64 rows a block of 128 threads, "
          "k 32 a step; inputs rounded to {0} and the weight tile decoded to {0} in shared memory "
          "(rows padded by 8), mma.sync m16n8k16 {0} with F32 accumulators, times the tensor "
          "scale",
          mma);
    case Template::kMatrixFma:
      return "TileKernel<D, 16|64 rows, fma>: the matrix tiles in F32 shared memory (rows padded "
             "by 1), fmaf in k order, times the tensor scale";
    case Template::kGrouped:
      return std::format(
          "TileKernel<D, 16|64 rows, mma, grouped>: the matrix tiles ({} mma.sync) over each "
          "expert's pairs of a worklist sorted by expert, a block per 64 outputs and expert, rows "
          "gathered from the input and scattered to the output, each expert's operand from a "
          "device array",
          mma);
    case Template::kRows:
      return "RowsKernel<D>: a block of 128 threads per id, segments decoded to F32 times the "
             "tensor scale; an id outside the matrix writes zeros and sets the flag";
    case Template::kCount:
      break;
  }
  return "";
}

// One declaration, with its operand rules in storage that outlives the
// registry (Table() below never reallocates).
struct Declared {
  Bound bound;
  std::string name;
  std::string variant;
  std::array<OperandRule, 3> rules{};
  std::size_t rule_count = 0;
};

Declared Declare(const Bound& b) {
  Declared d;
  d.bound = b;
  d.name = ImplementationName(b);
  const std::uint32_t k_multiple = std::max(DecoderKMultiple(b.gate), DecoderKMultiple(b.up));
  d.variant = std::format("{}; decoder {}", TemplateText(b.tmpl, DecoderMmaElement(b.gate)),
                          DecoderName(b.gate));
  if (b.tmpl == Template::kVectorGlu) {
    d.variant += std::format(" (gate), {} (up)", DecoderName(b.up));
  }
  d.variant += std::format(
      "; segments of {} values, K a multiple of {}; accepts the build's SASS architectures, K "
      "below 2^30, N below 2^31",
      kSegment, k_multiple);
  switch (b.tmpl) {
    case Template::kVector:
    case Template::kMatrix:
    case Template::kMatrixFma:
      d.rules = {WeightRule(0, b.gate), DenseRule(1, kF32), {}};
      d.rule_count = 2;
      break;
    case Template::kVectorGlu:
      d.rules = {WeightRule(0, b.gate), WeightRule(1, b.up), DenseRule(2, kF32)};
      d.rule_count = 3;
      break;
    case Template::kGrouped:
      d.rules = {WeightRule(0, b.gate), DenseRule(1, kF32), DenseRule(2, kI32)};
      d.rule_count = 3;
      break;
    case Template::kRows:
      d.rules = {WeightRule(0, b.gate), DenseRule(1, kI32), {}};
      d.rule_count = 2;
      break;
    case Template::kCount:
      break;
  }
  return d;
}

const std::vector<Declared>& Table() {
  static const std::vector<Declared> table = [] {
    std::vector<Declared> t;
    for (std::size_t i = 0; i < kDecoderCount; ++i) {
      const auto id = static_cast<DecoderId>(i);
      for (std::size_t j = 0; j < static_cast<std::size_t>(Template::kCount); ++j) {
        t.push_back(Declare({.tmpl = static_cast<Template>(j), .gate = id, .up = id}));
      }
    }
    for (const auto& [gate, up] : kMixedGlu) {
      t.push_back(Declare({.tmpl = Template::kVectorGlu, .gate = gate, .up = up}));
    }
    return t;
  }();
  return table;
}

Implementation Declaration(const Declared& d) {
  Capability c;
  c.operands = std::span(d.rules).first(d.rule_count);
  c.rung = Dense(d.bound.gate) && Dense(d.bound.up) ? execution::Rung::kNative
                                                    : execution::Rung::kOnChipDecode;
  c.expansion = execution::Expansion::kNone;
  c.row_invariant = true;
  c.capturable = true;
  c.shape.k_multiple = std::max(DecoderKMultiple(d.bound.gate), DecoderKMultiple(d.bound.up));
  c.accepts = &Accepts;
  Operation operation = Operation::kMatMul;
  switch (d.bound.tmpl) {
    case Template::kVector:
      c.op = ir::Primitive(ir::Op::kMatMul);
      c.shape.max_rows = kMaxVectorRows;
      c.priority = 3;
      break;
    case Template::kMatrix:
      c.op = ir::Primitive(ir::Op::kMatMul);
      c.shape.max_rows = kMaxMatrixRows;
      c.device.min_arch = 80;
      c.device.needs = execution::kBf16Mma;
      c.priority = 2;
      break;
    case Template::kMatrixFma:
      c.op = ir::Primitive(ir::Op::kMatMul);
      c.shape.max_rows = kMaxMatrixRows;
      c.priority = 1;
      break;
    case Template::kVectorGlu:
      c.op = ir::Fused(ir::FusedKind::kMatMulGlu);
      c.shape.max_rows = kMaxVectorRows;
      operation = Operation::kMulMatGlu;
      break;
    case Template::kGrouped:
      c.op = ir::Primitive(ir::Op::kMatMulId);
      c.device.min_arch = 80;
      c.device.needs = execution::kBf16Mma;
      operation = Operation::kMulMatId;
      break;
    case Template::kRows:
      c.op = ir::Primitive(ir::Op::kGetRows);
      operation = Operation::kGetRows;
      break;
    case Template::kCount:
      break;
  }
  const std::string build = std::format(
      "sdk {}; target {}; cuda architectures {}; build type {} ({}, {}); sanitize {}; module "
      "sources {}",
      LLMP_DECODE_SDK, LLMP_DECODE_TARGET, LLMP_DECODE_CUDA_ARCHITECTURES, LLMP_DECODE_BUILD_TYPE,
      kAsserts, kLibraryAsserts, LLMP_DECODE_SANITIZE, ModuleSourcesDigest());
  return Implementation{.name = d.name,
                        .operation = operation,
                        .source = "llmp",
                        .revision = std::string(ModuleSourcesDigest()),
                        .build = build,
                        .variant = d.variant,
                        .capability = c};
}

// The build's architectures: "121a-real,86-real" and the like.
struct Arch {
  std::uint32_t arch = 0;
  bool specific = false;  // sm_XYa: X.Y only
};

std::vector<Arch> CompiledArchitectures() {
  std::vector<Arch> out;
  const std::string_view list = LLMP_DECODE_CUDA_ARCHITECTURES;
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

}  // namespace

std::string_view TemplateName(Template t) {
  const auto i = static_cast<std::size_t>(t);
  return i < kTemplateNames.size() ? kTemplateNames.at(i) : std::string_view("unknown");
}

std::string ImplementationName(const Bound& b) {
  std::string name = std::format("llmp.decode.{}.{}", TemplateName(b.tmpl), DecoderName(b.gate));
  if (b.up != b.gate) {
    name += std::format(".{}", DecoderName(b.up));
  }
  return name;
}

std::vector<Implementation> Implementations() {
  std::vector<Implementation> declared;
  declared.reserve(Table().size());
  for (const Declared& d : Table()) {
    declared.push_back(Declaration(d));
  }
  return declared;
}

std::expected<Bound, std::string> Bind(const Implementation& implementation) {
  for (const Declared& d : Table()) {
    if (d.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declaration(d)) != execution::IdentityOf(implementation)) {
      return std::unexpected(std::format("{} is not this build's {}", implementation.name, d.name));
    }
    return d.bound;
  }
  return std::unexpected(
      std::format("the decode module has no implementation {}", implementation.name));
}

bool CompiledFor(std::uint32_t arch) {
  static const std::vector<Arch> compiled = CompiledArchitectures();
  for (const Arch& a : compiled) {
    if (a.specific ? arch == a.arch : (arch / 10 == a.arch / 10 && arch >= a.arch)) {
      return true;
    }
  }
  return false;
}

}  // namespace llmp::kernels::decode
