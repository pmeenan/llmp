// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a GGML runner sets up for its life beside its weights and state
// (docs/engine.md): the engine core's runner memory (runner_memory.h: its
// own device memory, a cuBLAS workspace among it, and pinned staging);
// cuBLAS with upstream's workspace for the device on the model's stream;
// and the K-C launch context over the node's pool and the registry its
// plans bind against (D-053). CUDA builds only (llmp_engine).
//
// Lifetimes are completion-aware (AGENTS.md rule 6): Release runs only
// after the node's teardown fenced the model's stream and stopped the
// scheduler, and the runner destroys its plans and graphs first (they
// name the launch context and this memory, D-090); then the launch
// context, cuBLAS and the memory go, in that order. The staging is the
// node's to free (PagedNode::Pinned), at its close.

#ifndef LLMP_ENGINE_RUNNER_RESOURCES_H_
#define LLMP_ENGINE_RUNNER_RESOURCES_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "engine/paged_node.h"
#include "engine/runner_memory.h"
#include "execution/registry.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"

namespace llmp::engine {

class RunnerResources : public RunnerMemory {
 public:
  using Status = engine::Status;

  RunnerResources(PagedNode& node, int owner, std::uint32_t stream)
      : RunnerMemory(node, owner, stream) {}
  RunnerResources(const RunnerResources&) = delete;
  RunnerResources& operator=(const RunnerResources&) = delete;
  RunnerResources(RunnerResources&&) = delete;
  RunnerResources& operator=(RunnerResources&&) = delete;
  ~RunnerResources() = default;

  // cuBLAS on the model's stream, with upstream's workspace for the
  // device mapped as `name` (runtime memory).
  Status OpenCublas(std::string name);
  kernels::ggml::CublasHandle& cublas() const { return *cublas_; }
  const Mapped& cublas_workspace() const { return cublas_workspace_; }
  std::uint64_t cublas_bytes() const { return cublas_bytes_; }
  // Kernel source state is private pinned runtime memory, never shared scratch.
  Status OpenSelectedWorkspace(std::string name, std::uint64_t bytes);
  const Mapped& selected_workspace() const { return selected_workspace_; }

  // A launch context without a pool, to measure plans at setup.
  std::expected<std::unique_ptr<kernels::ggml::LaunchContext>, std::string> MeasuringContext()
      const;
  // The launch context over the node's pool (its first `pool_bytes`) and
  // the registry of GGML's implementations. After the node's workspace.
  Status BindLaunch(std::uint64_t pool_bytes);
  kernels::ggml::LaunchContext& launch() const { return *launch_; }
  const execution::Registry& registry() const { return *registry_; }

  // The launch context and cuBLAS destroyed, then every mapping released
  // (the problems appended). Once, after the node's teardown fenced the
  // stream, and after the runner destroyed its plans and graphs.
  void Release(std::vector<std::string>& problems);

 private:
  Mapped cublas_workspace_;
  Mapped selected_workspace_;
  std::uint64_t cublas_bytes_ = 0;
  std::unique_ptr<kernels::ggml::CublasHandle> cublas_;
  std::unique_ptr<kernels::ggml::LaunchContext> launch_;
  std::unique_ptr<execution::Registry> registry_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_RUNNER_RESOURCES_H_
