// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_KERNELS_GGML_GRAPH_READ_INDEX_H_
#define JITLLM_KERNELS_GGML_GRAPH_READ_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "kernels/ggml/fusion.h"

namespace jitllm::kernels::ggml::detail {
// Borrowed, immutable descriptors for one PlanGraph invocation only. Rebuild
// after placement: neither logical identity nor validation survives mutation.
class GraphReadIndex {
 public:
  GraphReadIndex(GraphNodes graph, std::span<ggml_tensor* const> keep);
  // Conservative capacity allowance, not a measured physical peak. Each
  // occurrence covers all three hash tables, reader vectors (including
  // geometric growth/rehash), and allocator metadata. Sequential passes
  // share one envelope; no index survives a pass.
  static std::optional<std::uint64_t> ScratchBytes(GraphNodes graph,
                                                   std::span<ggml_tensor* const> keep);
  bool Matches(GraphNodes graph, std::span<ggml_tensor* const> keep) const;
  // Exact source-pointer occurrences, including duplicate and self edges.
  // A null query or different borrowed graph must use the original scan.
  std::optional<std::size_t> SourceUses(GraphNodes graph, const ggml_tensor* tensor) const;
  bool OnlyReader(const ggml_tensor* tensor, const ggml_tensor* reader) const;
  bool Private(std::span<ggml_tensor* const> nodes, const ggml_tensor* output,
               const ggml_tensor* selected_ids = nullptr) const;
  std::optional<const ggml_tensor*> StrictRoot(const ggml_tensor* tensor) const;

 private:
  struct Descriptor {
    const ggml_tensor* root = nullptr;
    std::size_t source_uses = 0;
  };
  struct Edge {
    const ggml_tensor* node;
    const ggml_tensor* source;
  };
  struct Readers {
    std::vector<const ggml_tensor*> producers;
    std::vector<Edge> edges;
    std::vector<const ggml_tensor*> kept;
  };
  const ggml_tensor* Root(const ggml_tensor* tensor) const;
  bool StrictlyValid() const;
  GraphNodes graph_;
  std::span<ggml_tensor* const> keep_;
  std::unordered_map<const ggml_tensor*, Descriptor> roots_;
  std::unordered_map<const ggml_tensor*, Readers> readers_;
  mutable std::unordered_map<const ggml_tensor*, std::optional<const ggml_tensor*>> strict_;
  mutable std::optional<bool> strictly_valid_;
  bool bounded_ = true;
};
}  // namespace jitllm::kernels::ggml::detail
#endif  // JITLLM_KERNELS_GGML_GRAPH_READ_INDEX_H_
