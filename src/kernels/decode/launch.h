// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The module's internal launch interface, vendor-free: the arguments every
// template kernel takes, and each decoder's launchers, one per template,
// which the per-encoding units (decode_*.cu) define. products.cc checks a
// product completely before it calls a launcher; a launcher only picks the
// kernel's instantiation and grid and queues it.

#ifndef LLMP_KERNELS_DECODE_LAUNCH_H_
#define LLMP_KERNELS_DECODE_LAUNCH_H_

#include <cstdint>
#include <expected>
#include <string>

#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {

using Status = std::expected<void, std::string>;
using Stream = void*;  // a cudaStream_t

// The product templates (templates.cuh).
enum class Template : std::uint8_t {
  kVector,     // decode-shaped: rows <= 16, each weight read once
  kVectorGlu,  // the same over a gate and an up weight, then a GLU
  kMatrix,     // tiles decoded to shared memory, mma.sync
  kMatrixFma,  // the same tiles, F32 FMA
  kGrouped,    // tiles of an expert's rows from a sorted worklist, mma.sync
  kRows,       // rows gathered by index, decoded to F32
  kCount
};

inline constexpr std::uint32_t kMaxVectorRows = 16;
// The matrix templates' input rows: 65,535 tiles of 64 (CUDA's grid y).
inline constexpr std::uint32_t kMaxMatrixRows = 65535U * 64U;

// Everything a template kernel reads; each uses its own fields.
struct Args {
  // The weight (a GLU's gate); kGrouped: only its k and rows, the format's,
  // which each expert's operand must have.
  Operand w;
  Operand w2;                        // a GLU's up weight
  const Operand* experts = nullptr;  // kGrouped: a device array, one per expert
  std::uint32_t first = 0;           // the weight's first output row
  std::uint32_t first2 = 0;          // a GLU's up weight's
  std::uint32_t outputs = 0;         // output rows (N) computed
  std::uint32_t rows = 0;            // input rows: tokens, or kRows' ids
  const float* x = nullptr;
  std::uint64_t x_stride = 0;  // floats
  float* y = nullptr;
  std::uint64_t y_stride = 0;  // floats
  // kGrouped: expert e's pairs [offsets[e], offsets[e + 1]), each reading
  // row pair_rows[p] of x into row pair_out[p] (or p) of y; pairs at or
  // past `pairs`, and rows of x or y at or past x_rows or y_rows, are
  // skipped.
  const std::int32_t* offsets = nullptr;
  const std::int32_t* pair_rows = nullptr;
  const std::int32_t* pair_out = nullptr;
  std::uint32_t experts_count = 0;
  std::uint32_t pairs = 0;
  std::uint32_t x_rows = 0;
  std::uint32_t y_rows = 0;
  std::uint32_t tile_rows = 0;  // kMatrix and kGrouped: 16 or 64
  // kRows: the ids, and a flag set when one is outside the weight.
  const std::int32_t* ids = nullptr;
  std::int32_t* bad = nullptr;
  // kVectorGlu: ir::GluKind's value and kSwiGluClamp's limit.
  std::uint8_t glu = 0;
  float limit = 0.0f;
};

using Launcher = Status (*)(const Args& args, Stream stream);

// A decoder's launchers, one per template (indexed by Template); its
// kVectorGlu launcher takes it for both weights.
struct DecoderKernels {
  Launcher by_template[static_cast<int>(Template::kCount)] = {};
};

// Each per-encoding unit's decoders (null for another unit's).
const DecoderKernels* DenseKernels(DecoderId id);  // decode_dense.cu: BF16, F16
const DecoderKernels* Q8_0Kernels(DecoderId id);   // decode_q8_0.cu
const DecoderKernels* Nvfp4Kernels(DecoderId id);  // decode_nvfp4.cu
const DecoderKernels* Mxfp8Kernels(DecoderId id);  // decode_mxfp8.cu
// The GLU of two different decoders, where decode_mixed.cu instantiates
// the pair (null otherwise); a pair of one decoder is in its unit's table.
Launcher MixedGluLauncher(DecoderId gate, DecoderId up);

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_LAUNCH_H_
