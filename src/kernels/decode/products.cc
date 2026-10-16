// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/products.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/op.h"
#include "kernels/decode/launch.h"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {
namespace {

using format::Encoding;
using format::Layout;
using format::PartRole;

struct DecoderRow {
  DecoderId id;
  Encoding encoding;
  Layout layout;
  std::string_view name;
  std::uint32_t k_multiple;
  std::string_view mma;
  std::uint32_t align;  // of the data or codes part's address and rows
};

constexpr std::array<DecoderRow, static_cast<std::size_t>(DecoderId::kCount)> kDecoders = {{
    {DecoderId::kBf16Dense, Encoding::kBF16, Layout::kDense, "bf16.dense", kSegment, "bf16", 16},
    {DecoderId::kF16Dense, Encoding::kF16, Layout::kDense, "f16.dense", kSegment, "f16", 16},
    {DecoderId::kQ8_0Blocks, Encoding::kQ8_0, Layout::kGgmlBlocks, "q8_0.ggml_blocks", kQ8_0Block,
     "f16", 2},
    {DecoderId::kNvfp4Split, Encoding::kNvfp4, Layout::kSplit, "nvfp4.split", kNvfp4Block, "bf16",
     8},
    {DecoderId::kNvfp4Atoms, Encoding::kNvfp4, Layout::kScaleAtoms, "nvfp4.scale_atoms",
     kNvfp4Block * kAtomBlocks, "bf16", 8},
    {DecoderId::kNvfp4Interleaved, Encoding::kNvfp4, Layout::kRowInterleaved,
     "nvfp4.row_interleaved", kNvfp4Block, "bf16", 1},
    {DecoderId::kNvfp4GgmlBlocks, Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks,
     "nvfp4.ggml_nvfp4_blocks", kGgmlNvfp4Block, "bf16", 4},
    {DecoderId::kMxfp8Split, Encoding::kMxfp8, Layout::kSplit, "mxfp8.split", kMxfp8Block, "bf16",
     16},
    {DecoderId::kMxfp8Atoms, Encoding::kMxfp8, Layout::kScaleAtoms, "mxfp8.scale_atoms",
     kMxfp8Block * kAtomBlocks, "bf16", 16},
    {DecoderId::kAffineSplit, Encoding::kAffine, Layout::kSplit, "affine.split", kSegment, "bf16",
     8},
}};

constexpr bool InIdOrder() {
  for (std::size_t i = 0; i < kDecoders.size(); ++i) {
    if (static_cast<std::size_t>(kDecoders[i].id) != i) {
      return false;
    }
  }
  return true;
}
static_assert(InIdOrder(), "the decoder table lists every decoder in DecoderId order");

const DecoderRow& Row(DecoderId id) { return kDecoders.at(static_cast<std::size_t>(id)); }

// What the kernels index in 32 bits (K and N below these, as the
// registry declarations accept), and CUDA's grid y.
constexpr std::uint64_t kMaxK = std::uint64_t{1} << 30U;
constexpr std::uint64_t kMaxRows = std::uint64_t{1} << 31U;
constexpr std::uint64_t kMaxGridY = 65535;

std::unexpected<std::string> Refused(std::string message) {
  return std::unexpected(std::move(message));
}

bool Aligned(const void* p, std::uint64_t alignment) {
  return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

const DecoderKernels* KernelsFor(DecoderId id) {
  for (const DecoderKernels* k :
       {DenseKernels(id), Q8_0Kernels(id), Nvfp4Kernels(id), Mxfp8Kernels(id), AffineKernels(id)}) {
    if (k != nullptr) {
      return k;
    }
  }
  return nullptr;
}

Launcher LauncherFor(DecoderId id, Template t) {
  const DecoderKernels* k = KernelsFor(id);
  return k != nullptr ? k->by_template[static_cast<int>(t)] : nullptr;
}

// The input and output rows' checks shared by every product.
Status CheckRows(const float* x, std::uint64_t rows, std::uint64_t x_stride, std::uint64_t k,
                 const float* y, std::uint64_t y_stride, std::uint64_t outputs) {
  if (x == nullptr || y == nullptr) {
    return Refused("a product needs its input and output");
  }
  if (!Aligned(x, 16) || x_stride % 4 != 0) {
    return Refused("the input must be 16-byte aligned with a row stride of whole float4s");
  }
  if (rows > 1 && (x_stride < k || y_stride < outputs)) {
    return Refused("row strides shorter than a row");
  }
  return {};
}

}  // namespace

std::optional<DecoderId> DecoderOf(const format::TensorFormat& format) {
  for (const DecoderRow& row : kDecoders) {
    if (row.encoding == format.encoding && row.layout == format.layout) {
      return row.id;
    }
  }
  return std::nullopt;
}

std::string_view DecoderName(DecoderId id) { return Row(id).name; }
format::Encoding DecoderEncoding(DecoderId id) { return Row(id).encoding; }
format::Layout DecoderLayout(DecoderId id) { return Row(id).layout; }
std::uint32_t DecoderKMultiple(DecoderId id) { return Row(id).k_multiple; }
std::string_view DecoderMmaElement(DecoderId id) { return Row(id).mma; }

std::expected<Operand, std::string> OperandOf(const Weight& weight) {
  const format::TensorFormat& f = weight.format;
  const auto id = DecoderOf(f);
  if (!id) {
    return Refused(std::format("no decoder for {}.{}", format::Info(f.encoding).name,
                               format::Info(f.layout).name));
  }
  const DecoderRow& d = Row(*id);
  const auto extents = format::PartsOf(f);
  if (!extents) {
    return Refused(std::format("{}: {}", d.name, format::FormatErrorName(extents.error())));
  }
  const std::uint64_t k = f.shape.ne[0];
  const std::uint64_t n = f.shape.ne[1];
  if (k % d.k_multiple != 0 || k >= kMaxK || n >= kMaxRows) {
    return Refused(std::format("{}: K {} must be a multiple of {} below 2^30, N {} below 2^31",
                               d.name, k, d.k_multiple, n));
  }
  const std::uint64_t matrices = format::Matrices(f);
  if (weight.matrix >= matrices) {
    return Refused(std::format("{}: matrix {} of {}", d.name, weight.matrix, matrices));
  }
  if (weight.parts.size() != extents->size()) {
    return Refused(std::format("{}: {} parts given for a format of {}", d.name, weight.parts.size(),
                               extents->size()));
  }
  if (f.encoding == Encoding::kAffine) {
    if (f.params.bits != 4 && f.params.bits != 8) {
      return Refused(
          std::format("{}: affine decoder supports 4 or 8 bits, got {}", d.name, f.params.bits));
    }
    if (f.params.group != 32 && f.params.group != 64 && f.params.group != 128) {
      return Refused(std::format("{}: affine decoder supports group 32, 64, or 128, got {}", d.name,
                                 f.params.group));
    }
    if (f.params.scale != format::AffineScale::kBF16 &&
        f.params.scale != format::AffineScale::kF16) {
      return Refused(std::format("{}: affine decoder supports BF16 or F16 scales", d.name));
    }
  }
  Operand op;
  op.k = static_cast<std::uint32_t>(k);
  op.rows = static_cast<std::uint32_t>(n);
  op.rows_per_scale = static_cast<std::uint32_t>(n / f.stacked);
  for (const format::PartExtent& extent : extents->parts()) {
    const DevicePart* part = nullptr;
    for (const DevicePart& p : weight.parts) {
      if (p.role == extent.role) {
        if (part != nullptr) {
          return Refused(
              std::format("{}: part {} given twice", d.name, format::PartRoleName(extent.role)));
        }
        part = &p;
      }
    }
    if (part == nullptr || part->address == nullptr || part->bytes < extent.bytes) {
      return Refused(std::format("{}: part {} missing or shorter than its {} bytes", d.name,
                                 format::PartRoleName(extent.role), extent.bytes));
    }
    const std::uint64_t share = extent.bytes / matrices;
    const auto* base = static_cast<const std::uint8_t*>(part->address) + (weight.matrix * share);
    switch (extent.role) {
      case PartRole::kData:
      case PartRole::kCodes:
        op.data = base;
        op.row_bytes = share / n;
        break;
      case PartRole::kBlockScales:
        op.scales = base;
        break;
      case PartRole::kTensorScale:
        op.tensor_scale = reinterpret_cast<const float*>(base);
        break;
      case PartRole::kBiases:
      case PartRole::kZeros:
        op.biases = base;
        break;
      default:
        return Refused(
            std::format("{}: unexpected part {}", d.name, format::PartRoleName(extent.role)));
    }
  }
  if (f.encoding == Encoding::kAffine) {
    op.group = static_cast<std::uint16_t>(f.params.group);
    op.bits = f.params.bits;
    op.scale_type = static_cast<std::uint8_t>(f.params.scale);
    if (f.params.zero_point) {
      op.scale_type |= 0x80U;
    }
    const std::uint64_t groups = k / f.params.group;
    const std::uint64_t scale_elem_bytes = f.params.scale == format::AffineScale::kF32 ? 4 : 2;
    op.scale_row_bytes = static_cast<std::uint32_t>(groups * scale_elem_bytes);
  }
  if (!Aligned(op.data, d.align) || op.row_bytes % d.align != 0 ||
      (op.scales != nullptr && !Aligned(op.scales, 2)) ||
      (op.biases != nullptr && !Aligned(op.biases, 2)) ||
      (op.tensor_scale != nullptr && !Aligned(op.tensor_scale, 4))) {
    return Refused(
        std::format("{}: parts must be {}-byte aligned (tensor scales 4)", d.name, d.align));
  }
  return op;
}

Status RunVector(const Product& p, Stream stream) {
  const auto op = OperandOf(p.weight);
  if (!op) {
    return std::unexpected(op.error());
  }
  if (p.rows == 0 || p.rows > kMaxVectorRows || p.outputs == 0 || p.first > op->rows ||
      p.outputs > op->rows - p.first) {
    return Refused(std::format("vector: rows {} (1 to 16), outputs [{}, {}) of {}", p.rows, p.first,
                               p.first + p.outputs, op->rows));
  }
  if (auto ok = CheckRows(p.x, p.rows, p.x_stride, op->k, p.y, p.y_stride, p.outputs); !ok) {
    return ok;
  }
  Args a;
  a.w = *op;
  a.first = static_cast<std::uint32_t>(p.first);
  a.outputs = static_cast<std::uint32_t>(p.outputs);
  a.rows = static_cast<std::uint32_t>(p.rows);
  a.x = p.x;
  a.x_stride = p.x_stride;
  a.y = p.y;
  a.y_stride = p.y_stride;
  return LauncherFor(*DecoderOf(p.weight.format), Template::kVector)(a, stream);
}

Status RunMatrix(const Product& p, MatrixPath path, Stream stream) {
  const auto op = OperandOf(p.weight);
  if (!op) {
    return std::unexpected(op.error());
  }
  const std::uint64_t tile = p.rows <= 16 ? 16 : 64;
  if (p.rows == 0 || p.rows > kMaxMatrixRows || p.outputs == 0 || p.first > op->rows ||
      p.outputs > op->rows - p.first) {
    return Refused(std::format("matrix: rows {} (1 to {}), outputs [{}, {}) of {}", p.rows,
                               kMaxMatrixRows, p.first, p.first + p.outputs, op->rows));
  }
  if (auto ok = CheckRows(p.x, p.rows, p.x_stride, op->k, p.y, p.y_stride, p.outputs); !ok) {
    return ok;
  }
  Args a;
  a.w = *op;
  a.first = static_cast<std::uint32_t>(p.first);
  a.outputs = static_cast<std::uint32_t>(p.outputs);
  a.rows = static_cast<std::uint32_t>(p.rows);
  a.x = p.x;
  a.x_stride = p.x_stride;
  a.y = p.y;
  a.y_stride = p.y_stride;
  a.tile_rows = static_cast<std::uint32_t>(tile);
  const Template t = path == MatrixPath::kMma ? Template::kMatrix : Template::kMatrixFma;
  return LauncherFor(*DecoderOf(p.weight.format), t)(a, stream);
}

Status RunVectorGlu(const GluProduct& p, Stream stream) {
  const auto gate = OperandOf(p.gate);
  if (!gate) {
    return std::unexpected("gate: " + gate.error());
  }
  const auto up = OperandOf(p.up);
  if (!up) {
    return std::unexpected("up: " + up.error());
  }
  if (gate->k != up->k) {
    return Refused(std::format("vector_glu: gate's K {} and up's {} differ", gate->k, up->k));
  }
  if (p.rows == 0 || p.rows > kMaxVectorRows || p.outputs == 0 || p.gate_first > gate->rows ||
      p.outputs > gate->rows - p.gate_first || p.up_first > up->rows ||
      p.outputs > up->rows - p.up_first) {
    return Refused(
        std::format("vector_glu: rows {} (1 to 16), outputs {} from gate row {} of {} "
                    "and up row {} of {}",
                    p.rows, p.outputs, p.gate_first, gate->rows, p.up_first, up->rows));
  }
  if (static_cast<std::uint8_t>(p.glu.kind) >
      static_cast<std::uint8_t>(ir::GluKind::kSwiGluClamp)) {
    return Refused("vector_glu: unknown GLU");
  }
  if (p.glu.swapped) {
    return Refused("vector_glu: a swapped GLU is the caller's (it names the gate and the up)");
  }
  if (auto ok = CheckRows(p.x, p.rows, p.x_stride, gate->k, p.y, p.y_stride, p.outputs); !ok) {
    return ok;
  }
  const DecoderId g = *DecoderOf(p.gate.format);
  const DecoderId u = *DecoderOf(p.up.format);
  const Launcher launch = g == u ? LauncherFor(g, Template::kVectorGlu) : MixedGluLauncher(g, u);
  if (launch == nullptr) {
    return Refused(std::format("vector_glu: no instantiation for gate {} with up {}",
                               DecoderName(g), DecoderName(u)));
  }
  Args a;
  a.w = *gate;
  a.w2 = *up;
  a.first = static_cast<std::uint32_t>(p.gate_first);
  a.first2 = static_cast<std::uint32_t>(p.up_first);
  a.outputs = static_cast<std::uint32_t>(p.outputs);
  a.rows = static_cast<std::uint32_t>(p.rows);
  a.x = p.x;
  a.x_stride = p.x_stride;
  a.y = p.y;
  a.y_stride = p.y_stride;
  a.glu = static_cast<std::uint8_t>(p.glu.kind);
  a.limit = p.glu.limit;
  return launch(a, stream);
}

Status RunGrouped(const GroupedProduct& p, Stream stream) {
  const auto id = DecoderOf(p.format);
  if (!id) {
    return Refused(std::format("grouped: no decoder for {}.{}",
                               format::Info(p.format.encoding).name,
                               format::Info(p.format.layout).name));
  }
  if (auto valid = format::Validate(p.format); !valid) {
    return Refused(std::format("grouped: {}", format::FormatErrorName(valid.error())));
  }
  const std::uint64_t k = p.format.shape.ne[0];
  const std::uint64_t n = p.format.shape.ne[1];
  if (k % Row(*id).k_multiple != 0 || k >= kMaxK || n >= kMaxRows) {
    return Refused(
        std::format("grouped: K {} must be a multiple of {} below 2^30", k, Row(*id).k_multiple));
  }
  if (p.experts == nullptr || !Aligned(p.experts, alignof(Operand)) || p.offsets == nullptr ||
      p.pair_rows == nullptr || p.experts_count == 0 || p.experts_count > kMaxGridY ||
      p.outputs == 0 || p.first > n || p.outputs > n - p.first || p.pairs > kMaxRows ||
      p.x_rows > kMaxRows || p.y_rows > kMaxRows) {
    return Refused(std::format("grouped: {} experts (1 to 65535), outputs [{}, {}) of {}, {} pairs",
                               p.experts_count, p.first, p.first + p.outputs, n, p.pairs));
  }
  if (auto ok = CheckRows(p.x, 2, p.x_stride, k, p.y, p.y_stride, p.outputs); !ok) {
    return ok;
  }
  Args a;
  a.w.k = static_cast<std::uint32_t>(k);  // each expert's operand must match
  a.w.rows = static_cast<std::uint32_t>(n);
  a.experts = p.experts;
  a.first = static_cast<std::uint32_t>(p.first);
  a.outputs = static_cast<std::uint32_t>(p.outputs);
  a.x = p.x;
  a.x_stride = p.x_stride;
  a.y = p.y;
  a.y_stride = p.y_stride;
  a.offsets = p.offsets;
  a.pair_rows = p.pair_rows;
  a.pair_out = p.pair_out;
  a.experts_count = static_cast<std::uint32_t>(p.experts_count);
  a.pairs = static_cast<std::uint32_t>(p.pairs);
  a.x_rows = static_cast<std::uint32_t>(p.x_rows);
  a.y_rows = static_cast<std::uint32_t>(p.y_rows);
  a.tile_rows = p.rows_hint > 16 ? 64 : 16;
  return LauncherFor(*id, Template::kGrouped)(a, stream);
}

Status RunRows(const RowsGather& g, Stream stream) {
  const auto op = OperandOf(g.weight);
  if (!op) {
    return std::unexpected(op.error());
  }
  if (g.ids == nullptr || g.out == nullptr || g.count == 0 || g.count >= kMaxRows ||
      (g.count > 1 && g.out_stride < op->k)) {
    return Refused(std::format("rows: {} ids into rows of {} at a stride of {}", g.count, op->k,
                               g.out_stride));
  }
  Args a;
  a.w = *op;
  a.ids = g.ids;
  a.rows = static_cast<std::uint32_t>(g.count);
  a.y = g.out;
  a.y_stride = g.out_stride;
  a.bad = g.bad;
  return LauncherFor(*DecoderOf(g.weight.format), Template::kRows)(a, stream);
}

}  // namespace llmp::kernels::decode
