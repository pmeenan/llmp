// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The routed experts' CUTLASS layout (the grouped GEMM's executable view,
// docs/artifact-format.md#executable-views), and the conversions between it
// and GGML's block_nvfp4 layout the artifact stores. No CUDA types: the
// conversions (llmp_moe.cu) take device addresses and a cudaStream_t as
// void*.
//
// An expert's slot in its layer's slab holds, from its start:
//   gate and up codes   2f rows (gate's f, then up's f) of w / 2 bytes,
//                       element 2i in the low nibble (E2M1);
//   gate and up scales  their E4M3 scales (one per 16 elements) in the
//                       swizzled layout of 2f rows (moe_cutlass.h SfOffset);
//   down codes          w rows of f / 2 bytes;
//   down scales         their scales in the swizzled layout of w rows;
// f the expert's hidden width (640), w the model's width (2560). Each part
// is a multiple of 16 bytes; together they are exactly the bytes GGML's
// blocks of the three projections hold (36 bytes a 64-element block), so
// the layout fits the slot GGML's layout did. The conversion is a
// permutation of those bytes, lossless both ways.

#ifndef LLMP_KERNELS_GGML_MOE_LAYOUT_H_
#define LLMP_KERNELS_GGML_MOE_LAYOUT_H_

#include <cstdint>

namespace llmp::kernels::ggml::moe {

struct ExpertLayout {
  std::uint64_t ffn = 0;    // f: a multiple of 64
  std::uint64_t width = 0;  // w: a multiple of 128
  static constexpr std::uint64_t gate_up_codes() { return 0; }
  constexpr std::uint64_t gate_up_scales() const { return 2 * ffn * (width / 2); }
  constexpr std::uint64_t down_codes() const { return gate_up_scales() + (2 * ffn * (width / 16)); }
  constexpr std::uint64_t down_scales() const { return down_codes() + (width * (ffn / 2)); }
  constexpr std::uint64_t bytes() const { return down_scales() + (width * (ffn / 16)); }
};

// One layer's slab: `experts` slots `stride` bytes apart, each holding (in
// GGML's layout) the gate, up and down projections' block_nvfp4 rows at
// those byte offsets in the slot.
struct ExpertSlab {
  void* base = nullptr;
  std::uint64_t stride = 0;
  std::int64_t experts = 0;
  std::uint64_t gate = 0;
  std::uint64_t up = 0;
  std::uint64_t down = 0;
  ExpertLayout layout;
};

// The tokens a block of llmp.moe.route counts.
inline constexpr std::int64_t kRouteChunk = 64;
// The rows a group's scales are padded to (the scale layout's atom).
inline constexpr std::int64_t kScaleRows = 128;

// llmp.moe.route's output, I32: for `experts` experts, `slots` routed
// rows (tokens · experts used, slot t · used + j) and `chunks` counting
// blocks, in this order:
//   offsets     experts + 1: expert e's rows are rows offsets[e] ..
//               offsets[e + 1] of the sorted order (slots in token order
//               within an expert);
//   scale rows  experts + 1: the prefix of each expert's rows rounded up to
//               kScaleRows, where its scales start;
//   position    slots: each slot's row in the sorted order (-1 for an id
//               outside the experts);
//   token       slots: each sorted row's token;
//   expert      slots: each sorted row's expert;
//   counts      chunks · experts: the counting blocks' work space.
struct RouteLayout {
  std::int64_t experts = 0;
  std::int64_t slots = 0;
  std::int64_t chunks = 0;
  static constexpr std::int64_t offsets() { return 0; }
  constexpr std::int64_t scale_rows() const { return experts + 1; }
  constexpr std::int64_t position() const { return 2 * (experts + 1); }
  constexpr std::int64_t token() const { return position() + slots; }
  constexpr std::int64_t expert() const { return token() + slots; }
  constexpr std::int64_t counts() const { return expert() + slots; }
  constexpr std::int64_t ints() const { return counts() + (chunks * experts); }
};

// Quantized activations (llmp.moe.quantize and llmp.moe.glu_quantize),
// bytes: `slots` sorted rows of k values as E2M1 codes (k / 2 bytes a row),
// their E4M3 scales in each expert's swizzled block (at its scale rows),
// then each row's F32 global scale; the parts 256-byte aligned. The scale
// part holds the most rows the padding can reach.
struct QuantLayout {
  std::uint64_t k = 0;
  std::uint64_t slots = 0;
  std::uint64_t experts = 0;
  static constexpr std::uint64_t Align(std::uint64_t b) { return (b + 255) / 256 * 256; }
  static constexpr std::uint64_t codes() { return 0; }
  constexpr std::uint64_t scale_rows_max() const {
    constexpr auto kRows = static_cast<std::uint64_t>(kScaleRows);
    const std::uint64_t rows = slots + ((kRows - 1) * experts);
    return (rows + kRows - 1) / kRows * kRows;
  }
  constexpr std::uint64_t scales() const { return Align(slots * (k / 2)); }
  constexpr std::uint64_t row_scales() const {
    return Align(scales() + (scale_rows_max() * (k / 16)));
  }
  constexpr std::uint64_t bytes() const { return row_scales() + (slots * 4); }
};

// Rewrites the slab's slots from GGML's layout into the CUTLASS layout,
// `batch` experts at a time through `temp` (batch · stride bytes of device
// memory), on `stream`. The slot bytes past the layout are zeroed. Returns
// false (nothing queued) for a slab whose parts do not fit its stride.
bool ToCutlassLayout(const ExpertSlab& slab, void* temp, std::int64_t batch, void* stream);
// The inverse, into `out` (experts · stride bytes, GGML's layout at the
// same offsets, other bytes untouched): the layout proof's check.
bool ToGgmlLayout(const ExpertSlab& slab, void* out, void* stream);

}  // namespace llmp::kernels::ggml::moe

#endif  // LLMP_KERNELS_GGML_MOE_LAYOUT_H_
