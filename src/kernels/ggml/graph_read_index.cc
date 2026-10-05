// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/ggml/graph_read_index.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml::detail {
namespace {
// Keep the original matcher validation, including the depth bound for EACH
// starting descriptor; memoize only the complete result, not partial walks.
std::optional<const ggml_tensor*> CheckedRoot(const ggml_tensor* t) {
  for (unsigned depth = 0; t != nullptr && depth < 64; ++depth) {
    if (static_cast<unsigned>(t->type) >= GGML_TYPE_COUNT || !Extent(t)) return std::nullopt;
    if (t->view_src == nullptr) return t;
    const auto* p = t->view_src;
    if (static_cast<unsigned>(p->type) >= GGML_TYPE_COUNT) return std::nullopt;
    const auto child = Extent(t), parent = Extent(p);
    if (!child || !parent || t->view_offs > *parent || *child > *parent - t->view_offs)
      return std::nullopt;
    const auto address = reinterpret_cast<std::uintptr_t>(p->data);
    if (p->data == nullptr || t->view_offs > std::numeric_limits<std::uint64_t>::max() - address ||
        reinterpret_cast<std::uintptr_t>(t->data) != address + t->view_offs)
      return std::nullopt;
    t = p;
  }
  return std::nullopt;
}
// OnlyReader historically uses logical identity, not the checked matcher's
// depth/offset/span contract. Accept arbitrarily long finite view chains.
const ggml_tensor* LogicalRoot(const ggml_tensor* t) {
  const auto* slow = t;
  const auto* fast = t;
  while (fast != nullptr && fast->view_src != nullptr) {
    slow = slow->view_src;
    fast = fast->view_src->view_src;
    if (slow == fast) return nullptr;
  }
  while (t != nullptr && t->view_src != nullptr) t = t->view_src;
  return t;
}

}  // namespace

std::optional<std::uint64_t> GraphReadIndex::ScratchBytes(GraphNodes graph,
                                                          std::span<ggml_tensor* const> keep) {
  constexpr std::uint64_t kOccurrenceBytes = 1024;
  constexpr std::uint64_t kFixedBytes = 4096;
  std::uint64_t occurrences = graph.size();
  const auto add = [&occurrences](std::uint64_t n) {
    return !__builtin_add_overflow(occurrences, n, &occurrences);
  };
  if (!add(keep.size())) return std::nullopt;
  for (const auto* node : graph) {
    if (node == nullptr) return std::nullopt;
    for (const auto* source : node->src)
      if (source != nullptr && !add(1)) return std::nullopt;
  }
  std::uint64_t bytes = 0;
  if (__builtin_mul_overflow(occurrences, kOccurrenceBytes, &bytes) ||
      __builtin_add_overflow(bytes, kFixedBytes, &bytes))
    return std::nullopt;
  return bytes;
}

GraphReadIndex::GraphReadIndex(GraphNodes graph, std::span<ggml_tensor* const> keep)
    : graph_(graph), keep_(keep) {
  roots_.reserve(graph.size());
  readers_.reserve(graph.size());
  const auto root = [this](const ggml_tensor* t) {
    const auto [it, inserted] = roots_.try_emplace(t, nullptr);
    if (inserted) it->second = LogicalRoot(t);
    bounded_ = bounded_ && it->second != nullptr;
    return it->second;
  };
  for (const auto* node : graph) {
    if (node == nullptr) {
      bounded_ = false;
      continue;
    }
    readers_[root(node)].producers.push_back(node);
    for (const auto* source : node->src)
      if (source != nullptr) readers_[root(source)].edges.push_back({node, source});
  }
  for (const auto* kept : keep)
    if (kept != nullptr) readers_[root(kept)].kept.push_back(kept);
}
bool GraphReadIndex::Matches(GraphNodes graph, std::span<ggml_tensor* const> keep) const {
  return graph.data() == graph_.data() && graph.size() == graph_.size() &&
         keep.data() == keep_.data() && keep.size() == keep_.size();
}
const ggml_tensor* GraphReadIndex::Root(const ggml_tensor* t) const {
  const auto found = roots_.find(t);
  return found == roots_.end() ? LogicalRoot(t) : found->second;
}
std::optional<const ggml_tensor*> GraphReadIndex::StrictRoot(const ggml_tensor* t) const {
  const auto found = strict_.find(t);
  if (found != strict_.end()) return found->second;
  const auto result = CheckedRoot(t);
  strict_.emplace(t, result);
  return result;
}
bool GraphReadIndex::StrictlyValid() const {
  if (strictly_valid_) return *strictly_valid_;
  bool valid = bounded_;
  for (const auto& [tensor, root] : roots_) {
    (void)root;
    if (!StrictRoot(tensor)) valid = false;
  }
  strictly_valid_ = valid;
  return valid;
}
bool GraphReadIndex::OnlyReader(const ggml_tensor* tensor, const ggml_tensor* reader) const {
  const auto* storage = Root(tensor);
  if (!bounded_ || storage == nullptr || graph_.empty() || Root(graph_.back()) == storage)
    return false;
  const auto found = readers_.find(storage);
  if (found == readers_.end()) return true;
  const auto& uses = found->second;
  if (!uses.kept.empty()) return false;
  for (const auto* producer : uses.producers)
    if (producer != tensor) return false;
  for (const auto& edge : uses.edges)
    if (edge.node != tensor && (edge.node != reader || edge.source != tensor)) return false;
  return true;
}
bool GraphReadIndex::Private(std::span<ggml_tensor* const> nodes, const ggml_tensor* output,
                             const ggml_tensor* selected_ids) const {
  if (!StrictlyValid()) return false;
  const auto out = StrictRoot(output);
  if (!out) return false;
  const auto internal = [nodes](const ggml_tensor* t) {
    return std::ranges::find(nodes, t) != nodes.end();
  };
  const auto readable = [&](const ggml_tensor* t) {
    return StrictRoot(t) == out || t == selected_ids;
  };
  for (const auto* node : nodes) {
    const auto root = StrictRoot(node);
    if (!root || ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 && !readable(node))) return false;
    const auto found = readers_.find(*root);
    if (found == readers_.end()) continue;
    const auto& uses = found->second;
    for (const auto* kept : uses.kept)
      if (!readable(kept)) return false;
    for (const auto* producer : uses.producers)
      if (!internal(producer) && !readable(producer)) return false;
    for (const auto& edge : uses.edges)
      if (!internal(edge.node) && !readable(edge.source)) return false;
    if (!graph_.empty() && Root(graph_.back()) == *root && !readable(graph_.back())) return false;
  }
  return true;
}
}  // namespace jitllm::kernels::ggml::detail
