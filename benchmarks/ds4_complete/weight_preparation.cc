// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "ds4_complete/weight_preparation.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <limits>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "engine/runner_resources.h"

namespace jitllm::benchmarks::ds4_complete {
namespace {
namespace en = engine;
namespace ca = catalog;
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point first) {
  return std::chrono::duration<double>(Clock::now() - first).count();
}

class PreparationModel final : public en::PagedModel {
 public:
  explicit PreparationModel(en::PagedNode& node) : resources(node, 0, 0) {}
  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return closure; }
  // These pinned setup mappings are released by resources after retirement.
  std::vector<ca::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    std::vector<std::string> problems;
    resources.Release(problems);
    if (!problems.empty()) return std::unexpected(problems.front());
    return {};
  }

  en::Mapped raw;
  en::Mapped packed;
  en::RunnerResources resources;
  ca::Closure closure;
};
}  // namespace

std::expected<PreparedModelWeights, std::string> PrepareModelWeights(
    const artifact::Artifact& artifact, const std::filesystem::path& scratch,
    std::uint64_t payload_limit) {
  if (payload_limit == 0 || payload_limit > (std::uint64_t{256} << 20U) ||
      payload_limit % artifact::kFileAlignment != 0) {
    return std::unexpected("complete preparation requires a bounded aligned read size");
  }
  auto plan = en::PlanDs4PreparedWeights(artifact);
  if (!plan) return std::unexpected(plan.error());
  PreparedModelWeights result;
  result.plan = std::move(*plan);
  // Preflight every tensor's complete direct-read tiling before opening a
  // device or creating output. These counts describe actual aligned reads,
  // including overlap at raw block boundaries, rather than payload only.
  for (const auto& tensor : result.plan.tensors) {
    auto chunks = en::PlanDs4WeightChunks(tensor, payload_limit);
    if (!chunks) return std::unexpected(chunks.error());
    for (const auto& chunk : *chunks) {
      if (__builtin_add_overflow(result.direct_read_bytes, chunk.read_bytes,
                                 &result.direct_read_bytes)) {
        return std::unexpected("complete preparation read count overflows");
      }
    }
  }
  std::vector<artifact::FileDescriptor> shards;
  shards.reserve(artifact.shards().size());
  for (std::uint32_t i = 0; i < artifact.shards().size(); ++i) {
    auto shard = artifact.OpenShardForDirectRead(i);
    if (!shard) return std::unexpected("could not open validated raw preparation shard");
    shards.push_back(std::move(*shard));
  }
  const auto setup_started = Clock::now();
  en::PagedNode node({.compute_streams = 1});
  PreparationModel model(node);
  // Even a partial Open can own streams or landing-zone backing. Every
  // attempted setup passes through TearDown before captured owners leave.
  auto prepared = [&]() -> en::Status {
    if (auto opened = node.Open(); !opened) return opened;
    if (auto mapped = model.resources.Map(model.raw, "ds4 preparation raw", payload_limit,
                                          ca::MemoryClass::kScratch);
        !mapped)
      return mapped;
    if (auto mapped =
            model.resources.Map(model.packed, "ds4 preparation aligned",
                                result.plan.largest_tensor_bytes, ca::MemoryClass::kScratch);
        !mapped)
      return mapped;
    const auto host_bytes = payload_limit + (2 * artifact::kFileAlignment);
    auto pinned = model.resources.Pinned(host_bytes);
    if (!pinned) return std::unexpected(pinned.error());
    if (auto workspace = node.MapWorkspace(en::kPagedExtent, en::kPagedExtent); !workspace)
      return workspace;
    if (auto launch = model.resources.BindLaunch(0); !launch) return launch;
    auto extents = model.resources.extents();
    extents.insert(extents.end(), node.pool().extents.begin(), node.pool().extents.end());
    extents.insert(extents.end(), node.activations().extents.begin(),
                   node.activations().extents.end());
    auto closure = node.catalog().ClosureOfExtents(extents);
    if (!closure) return std::unexpected("could not bind complete preparation memory closure");
    model.closure = std::move(*closure);
    const auto physical = node.catalog().OccupancyOf(node.domain()).Total().value();
    if (physical > std::numeric_limits<std::uint64_t>::max() - (2 * en::kPagedExtent)) {
      return std::unexpected("preparation execution budget overflows");
    }
    result.preparation_physical_bytes = physical;
    if (auto started = node.Start(base::Bytes(physical + (2 * en::kPagedExtent))); !started)
      return started;
    node.Run();
    result.setup_seconds = Seconds(setup_started);
    const en::Ds4WeightPreparationBuffers buffers{
        .host = *pinned,
        .host_bytes = host_bytes,
        .raw = {.address = model.raw.base, .bytes = model.raw.bytes},
        .packed = {.address = model.packed.base, .bytes = model.packed.bytes}};
    const auto preparation_started = Clock::now();
    result.files.reserve(result.plan.tensors.size());
    for (const auto& tensor : result.plan.tensors) {
      auto file =
          en::PrepareDs4WeightFile(node, model.resources.launch(), model.closure, 0, artifact,
                                   shards, scratch, tensor, payload_limit, buffers);
      if (!file) return std::unexpected(tensor.name + ": " + file.error());
      if (__builtin_add_overflow(result.stored_bytes, file->stored_bytes.value(),
                                 &result.stored_bytes)) {
        return std::unexpected("complete prepared file count overflows");
      }
      result.files.push_back(std::move(*file));
    }
    result.preparation_seconds = Seconds(preparation_started);
    result.steps = node.TakeTimes(0);
    result.preparation_jobs = result.steps.steps;
    return {};
  }();
  const auto teardown_started = Clock::now();
  const std::array<en::PagedModel*, 1> models = {&model};
  auto retired = node.TearDown(models);
  result.teardown_seconds = Seconds(teardown_started);
  if (!retired) {
    // A scope exit cannot prove GPU or storage retirement. This diagnostic
    // process must stop before releasing any owner captured by unknown work.
    std::println(stderr, "ds4 preparation retirement failed: {}", retired.error());
    std::abort();
  }
  if (!prepared) return std::unexpected(prepared.error());
  if (result.files.size() != result.plan.tensors.size()) {
    return std::unexpected("complete prepared weight inventory is missing a tensor");
  }
  return result;
}

}  // namespace jitllm::benchmarks::ds4_complete
