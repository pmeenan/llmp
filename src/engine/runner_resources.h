// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a runner sets up for its life beside its weights and state
// (docs/engine.md): device memory of its own (a cuBLAS workspace, a
// verify's snapshot, working memory) mapped at setup with access and
// pinned in the catalog; pinned host staging, cataloged; cuBLAS with
// upstream's workspace for the device on the model's stream; and, for a
// runner of GGML plans, the K-C launch context over the node's pool and
// the registry its plans bind against (D-053).
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

#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "execution/registry.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"

namespace llmp::engine {

class RunnerResources {
 public:
  using Status = engine::Status;

  RunnerResources(PagedNode& node, int owner, std::uint32_t stream)
      : node_(node), owner_(owner), stream_(stream) {}
  RunnerResources(const RunnerResources&) = delete;
  RunnerResources& operator=(const RunnerResources&) = delete;
  RunnerResources(RunnerResources&&) = delete;
  RunnerResources& operator=(RunnerResources&&) = delete;
  ~RunnerResources() = default;

  PagedNode& node() const { return node_; }
  int owner() const { return owner_; }
  std::uint32_t stream() const { return stream_; }

  // `bytes` of device VMM (rounded to extents) of `memory_class`, pinned,
  // mapped into `mapped` for the runner's life. Before the node runs.
  // `mapped` must remain alive at the same address through fenced Release,
  // including when Map fails after making a partial allocation.
  Status Map(Mapped& mapped, std::string name, std::uint64_t bytes,
             catalog::MemoryClass memory_class);
  // Pinned host memory, cataloged as the runner's staging. Before the
  // node runs.
  std::expected<void*, std::string> Pinned(std::uint64_t bytes);
  // Successful setup allocations only, before the scheduler starts. Device
  // mappings include extent alignment; pinned bytes include PagedNode's 256
  // byte minimum. This excludes the shared workspace and growing live state.
  std::uint64_t mapped_bytes() const;
  std::uint64_t pinned_bytes() const { return pinned_bytes_; }

  // cuBLAS on the model's stream, with upstream's workspace for the
  // device mapped as `name` (runtime memory).
  Status OpenCublas(std::string name);
  kernels::ggml::CublasHandle& cublas() const { return *cublas_; }
  const Mapped& cublas_workspace() const { return cublas_workspace_; }
  std::uint64_t cublas_bytes() const { return cublas_bytes_; }

  // A launch context without a pool, to measure plans at setup.
  std::expected<std::unique_ptr<kernels::ggml::LaunchContext>, std::string> MeasuringContext()
      const;
  // The launch context over the node's pool (its first `pool_bytes`) and
  // the registry of GGML's implementations. After the node's workspace.
  Status BindLaunch(std::uint64_t pool_bytes);
  kernels::ggml::LaunchContext& launch() const { return *launch_; }
  const execution::Registry& registry() const { return *registry_; }

  // Every extent mapped here and the staging's: a closure's share.
  std::vector<catalog::ExtentId> extents() const;

  // The launch context and cuBLAS destroyed, then every mapping released
  // (the problems appended). Once, after the node's teardown fenced the
  // stream, and after the runner destroyed its plans and graphs.
  void Release(std::vector<std::string>& problems);

 private:
  PagedNode& node_;
  int owner_;
  std::uint32_t stream_;
  std::vector<Mapped*> mapped_;  // the runner's members, in mapping order
  std::vector<catalog::ExtentId> staging_;
  std::uint64_t pinned_bytes_ = 0;
  Mapped cublas_workspace_;
  std::uint64_t cublas_bytes_ = 0;
  std::unique_ptr<kernels::ggml::CublasHandle> cublas_;
  std::unique_ptr<kernels::ggml::LaunchContext> launch_;
  std::unique_ptr<execution::Registry> registry_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_RUNNER_RESOURCES_H_
