// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Products over weights in their stored encodings (D-108;
// docs/engine-components.md#weight-and-state-formats), vendor-free: a block
// decoder per encoding and layout (decoders.cuh) under shared templates
// (templates.cuh), so every product shape runs from the decoder alone and
// a weight is never expanded into device memory. CUDA builds only.
//
// Decoders: BF16 and F16 dense; Q8_0 in GGML's blocks; NVFP4 split, in
// CUTLASS's scale atoms, row-interleaved and in GGML's blocks (with or
// without a tensor scale); MXFP8 and MXFP4 split and in scale atoms;
// affine split with its explicit per-group scales/biases.
// Templates:
//   RunVector     decode-shaped products, 1 to 16 input rows;
//   RunVectorGlu  the same over a gate and an up weight (each its own
//                 format) into a GLU of the two;
//   RunMatrix     1 to kMaxMatrixRows input rows: tiles decoded to shared memory
//                 and multiplied with mma.sync (sm_80 on), or with F32 FMA;
//   RunGrouped    an expert worklist's rows, each expert its own weight;
//   RunRows       rows gathered by index, decoded to F32.
//
// Arithmetic. Every decoder decodes a value exactly as the CPU reference
// (kernels/reference/decode.h) does before the tensor scale (code times
// block scale is exact in F32 for these encodings), and the tensor scale
// multiplies each output once. The vector templates and the FMA matrix
// path keep the inputs in F32 and sum in F32. The mma path rounds inputs
// to the decoder's matrix element: BF16 for BF16, NVFP4 and MXFP8 weights
// (whose decoded values are exactly BF16, MXFP8's for block scales from
// 2^-124 to 2^119), F16 for F16 and Q8_0 weights (Q8_0's d · q rounded to F16, as
// GGML's dequantization to F16 rounds it); it sums in F32. On the F16 path
// an input of magnitude 65520 or more rounds to infinity, as on GGML's F16
// path. Each product is
// row-invariant: an input row's outputs do not depend on the other rows or
// how many there are (D-092).
//
// Inputs are F32 rows (16-byte aligned, a row stride that is a multiple
// of 4 floats); outputs are F32. A weight's part addresses must be aligned
// for its decoder's loads (16 bytes for dense, MXFP8 and 8-bit affine
// codes, 8 for NVFP4 and 4-bit affine codes, 4 for GGML's NVFP4 blocks,
// tensor scales and F32 affine scales, 2 for Q8_0's blocks and other
// scales). Every Run checks its operands completely and refuses, with
// nothing queued, a format without a decoder, parts that do not hold the
// format's bytes, misaligned addresses, a K that is not a multiple of the
// decoder's segment, or sizes the kernels cannot index; then it only
// queues the kernel on `stream` (a cudaStream_t): completion is the
// caller's. Device data a product reads (a worklist, ids, the grouped
// product's operands' K and N) is checked on the device: out-of-range
// entries are skipped or reported, never read past. No Run allocates
// device memory or takes a workspace, and each queues one kernel, so it
// can be captured into a graph.

#ifndef LLMP_KERNELS_DECODE_PRODUCTS_H_
#define LLMP_KERNELS_DECODE_PRODUCTS_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/op.h"
#include "kernels/decode/launch.h"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {

// One part of a weight on the device: all of its matrices' bytes.
struct DevicePart {
  format::PartRole role = format::PartRole::kData;
  const void* address = nullptr;
  std::uint64_t bytes = 0;
};

// Matrix `matrix` of a weight of format `format` whose parts are `parts`.
struct Weight {
  format::TensorFormat format{};
  std::span<const DevicePart> parts{};
  std::uint64_t matrix = 0;
};

// The decoder a format takes, if the module has one.
std::optional<DecoderId> DecoderOf(const format::TensorFormat& format);
std::string_view DecoderName(DecoderId id);  // "nvfp4.scale_atoms"
format::Encoding DecoderEncoding(DecoderId id);
format::Layout DecoderLayout(DecoderId id);
// The multiple of K a decoder's rows must be: its blocks, scale atoms and
// 16-value segments.
std::uint32_t DecoderKMultiple(DecoderId id);
// What a matrix kernel decodes a decoder's values to: "bf16" or "f16".
std::string_view DecoderMmaElement(DecoderId id);

// One matrix of a weight as the device decoders read it, checked: a
// decoder exists, the parts hold the format's bytes (format::PartsOf) with
// each role once, the matrix exists, and the addresses are aligned. The
// grouped product takes one per expert in device memory.
std::expected<Operand, std::string> OperandOf(const Weight& weight);

// y[t, j] = Σ_k W[first + j, k] · x[t, k] (times the weight's tensor
// scale of row first + j), for t < rows and j < outputs; row t of x at
// x + t · x_stride, of y at y + t · y_stride.
struct Product {
  Weight weight{};
  std::uint64_t first = 0;
  std::uint64_t outputs = 0;
  const float* x = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t x_stride = 0;
  float* y = nullptr;
  std::uint64_t y_stride = 0;
};

// Decode-shaped: 1 to 16 rows, each weight byte read once.
Status RunVector(const Product& product, Stream stream);

enum class MatrixPath : std::uint8_t { kMma, kFma };
Status RunMatrix(const Product& product, MatrixPath path, Stream stream);

// out[t, j] = glu(gate[t, j], up[t, j]) of the two products' rows
// gate_first + j and up_first + j (one weight may be both, as a stacked
// gate and up), 1 to 16 rows; the weights may differ in format, not in K.
// The caller resolves ir::GluParams::swapped into the two (refused here).
struct GluProduct {
  Weight gate{};
  std::uint64_t gate_first = 0;
  Weight up{};
  std::uint64_t up_first = 0;
  std::uint64_t outputs = 0;
  ir::GluParams glu{};
  const float* x = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t x_stride = 0;
  float* y = nullptr;
  std::uint64_t y_stride = 0;
};
Status RunVectorGlu(const GluProduct& product, Stream stream);

// An expert worklist's products: for each expert e, its pairs p in
// [offsets[e], offsets[e + 1]) compute y[pair_out[p] (or p), j] = Σ_k
// W_e[first + j, k] · x[pair_rows[p], k]. `format` is the experts' format
// (its decoder, K and N) and `experts` a device array of `experts_count`
// operands of it (OperandOf: one matrix each, any base address each),
// whose addresses the device trusts as OperandOf made them. offsets,
// pair_rows and pair_out are device I32 arrays; the device skips an expert
// whose operand's K or N is not the format's, a pair past `pairs`, and one
// naming a row past x_rows or y_rows. `rows_hint`, the pairs
// an expert is expected to have, picks the tile (16 or 64 rows).
struct GroupedProduct {
  format::TensorFormat format{};
  const Operand* experts = nullptr;
  std::uint64_t experts_count = 0;
  std::uint64_t first = 0;
  std::uint64_t outputs = 0;
  const std::int32_t* offsets = nullptr;
  const std::int32_t* pair_rows = nullptr;
  const std::int32_t* pair_out = nullptr;
  std::uint64_t pairs = 0;
  const float* x = nullptr;
  std::uint64_t x_rows = 0;
  std::uint64_t x_stride = 0;
  float* y = nullptr;
  std::uint64_t y_rows = 0;
  std::uint64_t y_stride = 0;
  std::uint64_t rows_hint = 0;
};
Status RunGrouped(const GroupedProduct& product, Stream stream);

// out[i, :] = W[ids[i], :] as F32 (times its tensor scale), for i <
// count; ids are a device I32 array. An id outside the matrix writes zeros
// and sets *bad (a device I32, if given) to 1.
struct RowsGather {
  Weight weight{};
  const std::int32_t* ids = nullptr;
  std::uint64_t count = 0;
  float* out = nullptr;
  std::uint64_t out_stride = 0;
  std::int32_t* bad = nullptr;
};
Status RunRows(const RowsGather& gather, Stream stream);

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_PRODUCTS_H_
