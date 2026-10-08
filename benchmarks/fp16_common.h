// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the backend proof's FP16 harnesses share (docs/backend-proof.md, P2):
// the two declared trajectories and each chunk's graph, built, bound,
// planned and placed as rung 3's harness (fp16_exec.cc) does it, with the
// weights' addresses given by the caller, so the paged harness
// (fp16_paged.cc) plans over device VMM exactly as rung 3 plans over
// cudaMalloc. CUDA builds only.

#ifndef LLMP_BENCHMARKS_FP16_COMMON_H_
#define LLMP_BENCHMARKS_FP16_COMMON_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/qwen2_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen2.h"

namespace llmp::benchmarks {

struct Trajectory {
  std::string name;
  std::vector<std::int32_t> tokens;
  std::vector<std::uint32_t> chunks;
  std::uint32_t cells = 0;
  // The profile's restore point (backend-proof.md, numerical profiles):
  // the tokens evaluated before the restored evaluation's restore.
  std::uint32_t restore_after = 0;
};

// `control`: the bridge's tokens.txt (76 IDs, one per line), chunks 32
// then 44 × 1, 512 cells, restore after 32. `heldout`: the declared
// held-out IDs (1,040 little-endian int64, SHA-256 6dd8da89..., the first
// 577 used), chunks 16, 17, 16 × 1, 512, 16 × 1, 1,024 cells, restore
// after 33.
std::expected<Trajectory, std::string> LoadTrajectory(std::string_view name,
                                                      const std::filesystem::path& tokens);

struct ChunkGraph {
  std::optional<kernels::ggml::TensorArena> arena;
  kernels::ggml::Qwen2Graph graph;
  kernels::ggml::GraphPlan plan;
  kernels::ggml::Placement placement;
};

// Where the chunk's tensors live: each weight resource's address, the
// cache's, and the activations' region (0 to measure the placement only).
struct ChunkMemory {
  std::function<std::uint64_t(std::uint32_t resource)> weight;
  std::uint64_t kv = 0;
  std::uint64_t activations = 0;
  std::uint64_t activation_bytes = 0;
};

// Builds, binds, plans and places one chunk's graph as fp16_exec.cc does:
// every computed tensor first at its own address, then placed largest
// first in the activations' region (128-byte aligned), planned again over
// the real addresses, which must give the same plan; every bound tensor is
// checked for 128-byte alignment.
std::expected<ChunkGraph, std::string> PlanChunk(const model::Qwen2Profile& profile,
                                                 const model::Qwen2Binding& binding,
                                                 const ChunkMemory& memory, std::uint32_t cells,
                                                 std::uint32_t rows, std::uint32_t n_kv,
                                                 bool fusion,
                                                 const kernels::ggml::DeviceChoices& choices);

inline std::uint64_t RoundUp(std::uint64_t bytes, std::uint64_t to) {
  return (bytes + to - 1) / to * to;
}

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_FP16_COMMON_H_
