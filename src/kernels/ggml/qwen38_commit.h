// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A Qwen3.8 speculative verify's commit (docs/experiments/qwen38-mtp/).
// A verify in the fast graph's verify form (qwen38_graph.h
// Qwen38GraphOptions::verify) reads the recurrent, convolution and n-gram
// state but writes none of it; it saves each row's inputs instead
// (model/qwen38.h Qwen38CommitLayout). Once the host knows how many rows it
// keeps, the commit leaves the state those rows alone would have left,
// with nothing re-run but the recurrences themselves (D-068's truncation):
//
// - each linear-attention layer's gated delta rule over the kept rows, from
//   the saved convolution output (q, k, v), gate and beta, in place on its
//   recurrent state, with llmp.gated_delta_net.columns' arithmetic
//   operation for operation, so the state is the one the verify's own
//   recurrence reached at its last kept row, bit for bit
//   (tests/unit/qwen38_commit_test.cc);
// - each convolution history (the last taps of the old history followed by
//   the kept rows' saved inputs), in place, and the n-gram layer's the same.
//
// Queued outside any graph, before the next job's own work, like DSpark's
// restore (engine/live_state.h). CUDA builds only.

#ifndef LLMP_KERNELS_GGML_QWEN38_COMMIT_H_
#define LLMP_KERNELS_GGML_QWEN38_COMMIT_H_

#include <cstdint>
#include <expected>

#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

class LaunchContext;

// One linear-attention layer's state and the verify's saves for it (device
// addresses): its recurrent state F32 [128, 128, v_heads] (column-major per
// head, as gated_delta_net's), its convolution history F32 [taps ·
// channels] (tap j of channel c at c · taps + j), and the saved rows
// ([width, rows], row r at r · width).
struct Qwen38CommitLayer {
  float* state = nullptr;
  float* history = nullptr;
  const float* conv = nullptr;  // [channels]: q heads, k heads, v heads
  const float* qkv = nullptr;   // [channels]
  const float* gate = nullptr;  // [v_heads]
  const float* beta = nullptr;  // [v_heads]
};

inline constexpr int kQwen38CommitLayers = 48;

struct Qwen38CommitArgs {
  int layers = 0;  // at most kQwen38CommitLayers
  int keep = 0;    // rows kept, 1 to the verify's
  int channels = 0;
  int qk_heads = 0;  // the key heads (q and k), 128 wide
  int v_heads = 0;   // the value heads, 128 wide, a multiple of qk_heads
  int taps = 0;      // the convolution's history: kernel - 1
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a kernel argument, copied to the device by value.
  Qwen38CommitLayer layer[kQwen38CommitLayers] = {};
  // The n-gram layer's convolution history F32 [ple_taps · ple_width] and
  // its saved input rows [ple_width].
  float* ple_history = nullptr;
  const float* ple_rows = nullptr;
  int ple_width = 0;
  int ple_taps = 0;
};

// Refused (kRejected, nothing queued) if the extents are not these kernels'
// (128-wide heads, keep 1 to 8, taps 1 to 16, pointers 16-byte aligned);
// kUnknown if a launch fails.
std::expected<void, KernelFailure> Qwen38Commit(LaunchContext& launch,
                                                const Qwen38CommitArgs& args);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_QWEN38_COMMIT_H_
