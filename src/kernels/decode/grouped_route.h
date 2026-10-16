// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A grouped product's worklist built on the device from the router's IDs, for
// resident experts (every expert's parts at a fixed pitch from expert 0's):
// no host round trip, so the preparation and the product capture into a
// graph. CUDA builds only.
//
// One block counts each expert's pairs, compacts the experts that have any
// into `groups` slots in ascending expert order (slot c's descriptor is
// expert e's operand: expert 0's with each part pointer moved e times its
// pitch; `offsets[c]` its first sorted pair), and scatters every pair to its
// expert's run: `pair_rows` the pair's input row (its token, or its slot
// with one input row a slot), `pair_out` its output row (its slot). Unused
// slots get an empty run and a zero descriptor, which the grouped template
// skips. Pairs within an expert's run are in no fixed order; the grouped
// product is row-invariant, so its outputs do not depend on it. An ID outside
// [0, experts) drops its pair: its output row, when `y` is given, is zeroed.
// The compaction is a block-wide scan (each thread a run of experts).
#ifndef LLMP_KERNELS_DECODE_GROUPED_ROUTE_H_
#define LLMP_KERNELS_DECODE_GROUPED_ROUTE_H_

#include <cstdint>

#include "kernels/decode/launch.h"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {

inline constexpr std::uint32_t kMaxResidentExperts = 4096;

struct ResidentRoute {
  Operand first{};               // expert 0's operand
  std::uint64_t data_pitch = 0;  // bytes an expert, for each non-null pointer
  std::uint64_t scales_pitch = 0;
  std::uint64_t tensor_scale_pitch = 0;
  std::uint64_t biases_pitch = 0;
  std::uint32_t experts = 0;          // at most kMaxResidentExperts
  const std::int32_t* ids = nullptr;  // [used, tokens] I32
  std::uint64_t ids_stride = 0;       // elements between tokens' IDs
  std::uint32_t used = 0;
  std::uint32_t tokens = 0;
  bool per_slot_input = false;        // input rows are slots, not tokens
  std::uint32_t groups = 0;           // min(experts, used * tokens)
  Operand* descriptors = nullptr;     // [groups]
  std::int32_t* offsets = nullptr;    // [groups + 1]
  std::int32_t* pair_rows = nullptr;  // [used * tokens]
  std::int32_t* pair_out = nullptr;   // [used * tokens]
  // The product's output: pair p's row at y + p · y_stride, its first
  // `y_outputs` values; null: dropped pairs' rows are left as they are.
  float* y = nullptr;
  std::uint64_t y_stride = 0;
  std::uint32_t y_outputs = 0;
};

// Queues the preparation on `stream`; refuses, with nothing queued, sizes the
// kernel cannot index or null buffers.
Status PrepareResidentRoute(const ResidentRoute& route, Stream stream);

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_GROUPED_ROUTE_H_
