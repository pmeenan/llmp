// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// How a runner's runs went and the copies a run queues, whatever its
// kernels (D-090): the engine core's part of the graph runs, shared by the
// GGML runs (graph_runs.h) and the step runs (step_runs.h).

#ifndef LLMP_ENGINE_RUN_STATS_H_
#define LLMP_ENGINE_RUN_STATS_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace llmp::engine {

// How a run went (D-090).
enum class RunPath : std::uint8_t {
  kEager,     // launch by launch
  kCaptured,  // captured: beside a launch-by-launch run, or then replayed once
  kReplayed,  // one launch of a graph captured earlier
};

// A model's runs of one kind (its target's, or its drafter's), summed.
struct GraphStats {
  std::uint64_t eager = 0;
  std::uint64_t captured = 0;
  std::uint64_t replayed = 0;
  std::uint64_t refused = 0;       // captures refused: those plans stay launch by launch
  std::uint64_t dropped = 0;       // graphs destroyed to stay within the model's cap
  double capture_seconds = 0;      // capturing
  double instantiate_seconds = 0;  // instantiating and uploading
  std::uint64_t nodes = 0;         // in every graph captured
  std::int64_t memory_bytes = 0;   // the drop in the device's free memory across captures
  std::string first_refusal;
};

// Counts a run that went `path`.
void Count(GraphStats& stats, RunPath path);

// Copies a run queues, three addresses or sizes each: an input's device
// address, its bytes and its offset in the staging (Stage's); an output's
// host address, device address and bytes.
using RunCopy = std::array<std::uint64_t, 3>;
using Copies = std::vector<RunCopy>;

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_RUN_STATS_H_
