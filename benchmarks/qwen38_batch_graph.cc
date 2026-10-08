// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_batch_graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/llmp_ops.h"

namespace llmp::benchmarks::qwen_batch {
namespace {
namespace kg = kernels::ggml;

std::unexpected<std::string> Error(std::string s) { return std::unexpected(std::move(s)); }

bool Eligible(const ggml_tensor* t) {
  const auto op = kg::LlmpOpOf(t);
  return op == kg::LlmpOp::kMxfp8MulMatVec || op == kg::LlmpOp::kMoeGemv;
}

bool SameLeaf(const ggml_tensor* a, const ggml_tensor* b) {
  return a != nullptr && b != nullptr && a->op == GGML_OP_NONE && b->op == GGML_OP_NONE &&
         a->data != nullptr && a->data == b->data && a->type == b->type &&
         std::ranges::equal(a->ne, b->ne) && std::ranges::equal(a->nb, b->nb);
}

std::expected<void, std::string> Match(const ggml_tensor* a, const ggml_tensor* b) {
  if (!Eligible(a) || kg::LlmpOpOf(a) != kg::LlmpOpOf(b) ||
      std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0 ||
      !SameLeaf(a->src[0], b->src[0]) || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
    return Error("C2 product phase/order, immutable weights or parameters differ");
  }
  if (kg::LlmpOpOf(a) == kg::LlmpOp::kMxfp8MulMatVec) {
    if (!SameLeaf(a->src[1], b->src[1]) || a->ne[0] != b->ne[0] || a->ne[1] <= 0 || b->ne[1] <= 0 ||
        a->ne[1] > 4 || b->ne[1] > 4 || a->src[2]->type != GGML_TYPE_F32 ||
        b->src[2]->type != GGML_TYPE_F32 || a->src[2]->ne[0] != b->src[2]->ne[0]) {
      return Error("C2 MXFP8 products require two F32 windows of one to four rows");
    }
  } else {
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1] || a->ne[2] <= 0 || b->ne[2] <= 0 ||
        a->ne[2] > 4 || b->ne[2] > 4 || a->src[1]->type != GGML_TYPE_F32 ||
        b->src[1]->type != GGML_TYPE_F32 || a->src[1]->ne[0] != b->src[1]->ne[0] ||
        a->src[1]->ne[1] != b->src[1]->ne[1]) {
      return Error("C2 routed products require matching F32 expert-slot windows");
    }
    if (kg::IsMoeGemvSwiglu(a) &&
        (!SameLeaf(a->src[3], b->src[3]) || !SameLeaf(a->src[4], b->src[4]))) {
      return Error("C2 routed products have different gate/up scales");
    }
  }
  return {};
}

void Redirect(std::span<ggml_tensor* const> nodes, ggml_tensor* from, ggml_tensor* to) {
  for (ggml_tensor* n : nodes) {
    for (ggml_tensor*& s : n->src) {
      if (s == from) {
        s = to;
      }
    }
    if (n->view_src == from) {
      n->view_src = to;
    }
  }
}

// Original lists are already topological *and* preserve mutable state
// ordering. Advance all active lists to the corresponding product barrier,
// then emit paid pack/product/splits for ascending-slot pairs. Each pair has
// at most eight rows; an odd active slot retains its original product.
std::expected<void, std::string> Coalesce(Plan& out,
                                          const Requests<std::vector<ggml_tensor*>>& lists) {
  auto arena = kg::TensorArena::Create(16384);
  if (!arena) {
    return Error(arena.error().detail);
  }
  out.arena.emplace(std::move(*arena));
  ggml_context* c = out.arena->context();
  Requests<std::size_t> at{};
  while (true) {
    std::vector<std::size_t> ready;
    for (std::size_t s = 0; s < kMaxRequests; ++s) {
      while (at[s] < lists[s].size() && !Eligible(lists[s][at[s]])) {
        out.nodes.push_back(lists[s][at[s]++]);
      }
      if (at[s] != lists[s].size()) {
        ready.push_back(s);
      }
    }
    if (ready.empty()) {
      break;
    }
    for (std::size_t s = 0; s < kMaxRequests; ++s) {
      if (!lists[s].empty() && at[s] == lists[s].size()) {
        return Error("Shared-row graphs have different row-local product closures");
      }
    }
    // A slot cannot drift to a different numerical stage merely because its
    // current row count differs from the other active slots.
    for (std::size_t i = 1; i < ready.size(); ++i) {
      if (auto matched = Match(lists[ready[0]][at[ready[0]]], lists[ready[i]][at[ready[i]]]);
          !matched) {
        return matched;
      }
    }
    for (std::size_t i = 0; i < ready.size(); i += 2) {
      const auto sa = ready[i];
      ggml_tensor* a = lists[sa][at[sa]++];
      if (i + 1 == ready.size()) {
        out.nodes.push_back(a);
        continue;
      }
      const auto sb = ready[i + 1];
      ggml_tensor* b = lists[sb][at[sb]++];
      if (auto matched = Match(a, b); !matched) {
        return matched;
      }
      if (auto room = out.arena->Reserve(5); !room) {
        return Error(room.error().detail);
      }
      ggml_tensor* both = nullptr;
      std::array<ggml_tensor*, 2> split{};
      if (kg::LlmpOpOf(a) == kg::LlmpOp::kMxfp8MulMatVec) {
        ggml_tensor* x = ggml_concat(c, a->src[2], b->src[2], 1);
        out.nodes.push_back(x);
        both = kg::Mxfp8MulMatVec(c, a->src[0], a->src[1], x);
        split[0] = ggml_view_2d(c, both, a->ne[0], a->ne[1], both->nb[1], 0);
        split[1] = ggml_view_2d(c, both, b->ne[0], b->ne[1], both->nb[1],
                                static_cast<std::size_t>(a->ne[1]) * both->nb[1]);
        out.packed_bytes += ggml_nbytes(x);
        ++out.mxfp8_pairs;
      } else {
        ggml_tensor* x = ggml_concat(c, a->src[1], b->src[1], 2);
        ggml_tensor* ids = ggml_concat(c, a->src[2], b->src[2], 1);
        out.nodes.push_back(x);
        out.nodes.push_back(ids);
        if (kg::IsMoeGemvSwiglu(a)) {
          both = kg::MoeGemvSwiglu(c, a->src[0], x, ids, a->ne[0], a->src[3], a->src[4],
                                   static_cast<std::uint64_t>(kg::LlmpOpInt(a, 2)),
                                   static_cast<std::uint64_t>(kg::LlmpOpInt(a, 3)));
        } else {
          both = kg::MoeGemv(c, a->src[0], x, ids, a->ne[0], kg::LlmpOpInt(a, 0),
                             kg::LlmpOpInt(a, 1), static_cast<std::uint64_t>(kg::LlmpOpInt(a, 2)),
                             static_cast<std::uint64_t>(kg::LlmpOpInt(a, 3)));
        }
        split[0] = ggml_view_3d(c, both, a->ne[0], a->ne[1], a->ne[2], both->nb[1], both->nb[2], 0);
        split[1] = ggml_view_3d(c, both, b->ne[0], b->ne[1], b->ne[2], both->nb[1], both->nb[2],
                                static_cast<std::size_t>(a->ne[2]) * both->nb[2]);
        out.packed_bytes += ggml_nbytes(x) + ggml_nbytes(ids);
        ++out.routed_pairs;
      }
      out.nodes.push_back(both);
      out.products.push_back({a, b, both});
      out.nodes.insert(out.nodes.end(), split.begin(), split.end());
      for (ggml_tensor*& held : out.keep) {
        if (held == a) {
          held = split[0];
        }
        if (held == b) {
          held = split[1];
        }
      }
      for (const auto& list : lists) {
        Redirect(list, a, split[0]);
        Redirect(list, b, split[1]);
      }
    }
  }
  if (out.routed_pairs == 0) {
    return Error("C2 graph coalesced no routed products");
  }
  return {};
}

std::expected<void, std::string> UnmodifiedSteps(const Requests<kg::GraphPlan>& originals,
                                                 const Plan& out) {
  const auto& joint = out.plan;
  for (const auto& plan : originals) {
    for (const auto& step : plan.steps) {
      if (step.nodes.size() == 1 && std::ranges::any_of(out.products, [&](const auto& pair) {
            return pair[0] == step.nodes.front() || pair[1] == step.nodes.front();
          })) {
        continue;
      }
      const auto count = std::ranges::count_if(joint.steps, [&](const kg::PlanStep& s) {
        return s.operation == step.operation && s.implementation == step.implementation &&
               s.nodes == step.nodes;
      });
      if (count != 1) {
        return Error("C2 plan changed an unbatched implementation or fusion");
      }
    }
  }
  for (const auto& pair : out.products) {
    const auto find = [](const kg::GraphPlan& plan, ggml_tensor* node) {
      return std::ranges::find_if(plan.steps, [node](const kg::PlanStep& s) {
        return s.nodes.size() == 1 && s.nodes.front() == node;
      });
    };
    const auto original = [&](ggml_tensor* node) -> const kg::PlanStep* {
      const kg::PlanStep* found = nullptr;
      for (const auto& plan : originals) {
        const auto at = find(plan, node);
        if (at != plan.steps.end()) {
          if (found != nullptr) {
            return nullptr;
          }
          found = &*at;
        }
      }
      return found;
    };
    const auto* a = original(pair[0]);
    const auto* b = original(pair[1]);
    const auto together = find(joint, pair[2]);
    if (a == nullptr || b == nullptr || together == joint.steps.end() ||
        a->operation != b->operation || a->implementation != b->implementation ||
        a->operation != together->operation || a->implementation != together->implementation) {
      return Error("C2 packed product changed its implementation/precision tier");
    }
  }
  return {};
}

std::expected<void, std::string> Finish(Plan& out, const Requests<std::vector<ggml_tensor*>>& lists,
                                        const Requests<kg::GraphPlan>& originals,
                                        const kg::DeviceChoices& choices, bool batch,
                                        std::uint64_t activations, std::uint64_t bytes) {
  if (batch) {
    if (auto made = Coalesce(out, lists); !made) {
      return made;
    }
  } else {
    for (const auto& list : lists) {
      out.nodes.insert(out.nodes.end(), list.begin(), list.end());
    }
  }
  std::unordered_set<const ggml_tensor*> seen;
  for (const auto* node : out.nodes) {
    if (node == nullptr || seen.contains(node)) {
      return Error("C2 joint graph repeats an operation descriptor");
    }
    for (const auto* parent : node->src) {
      if (parent != nullptr && parent->op != GGML_OP_NONE && !seen.contains(parent)) {
        return Error("C2 joint graph has an uncomputed parent/causal cycle");
      }
    }
    if (node->view_src != nullptr && node->view_src->op != GGML_OP_NONE &&
        !seen.contains(node->view_src)) {
      return Error("C2 joint graph has an unordered view producer");
    }
    seen.insert(node);
  }
  if (auto made =
          engine::PlaceAndPlan(out, out.nodes, out.inputs, out.keep, choices, activations, bytes);
      !made) {
    return made;
  }
  return UnmodifiedSteps(originals, out);
}
}  // namespace

std::expected<std::unique_ptr<Plan>, std::string> TargetPlan(
    const Requests<engine::Qwen38Model>& models, const Requests<kg::Qwen38ChunkShape>& shapes,
    const kg::DeviceChoices& choices, bool batch, std::uint64_t activations, std::uint64_t bytes,
    std::uint32_t active) {
  if (active == 0 || active >= (1U << kMaxRequests)) {
    return Error("Target active-slot mask invalid");
  }
  auto out = std::make_unique<Plan>();
  Requests<std::vector<ggml_tensor*>> lists;
  Requests<kg::GraphPlan> originals;
  for (std::size_t s = 0; s < kMaxRequests; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    if (shapes[s].rows <= 0 || shapes[s].rows > 4) {
      return Error("C2 target window must contain one to four rows per request");
    }
    auto p = engine::PlanQwen38Chunk(models[s], shapes[s], choices, 0, 0, {},
                                     {.verify = true, .export_streams = true});
    if (!p) {
      return std::unexpected(p.error());
    }
    out->target[s] = std::move(*p);
    auto& g = out->target[s]->graph;
    lists[s] = g.nodes;
    originals[s] = out->target[s]->plan;
    const auto inputs = g.inputs();
    out->inputs.insert(out->inputs.end(), inputs.begin(), inputs.end());
    out->keep.push_back(g.logits);
    out->keep.push_back(g.argmax);
  }
  if (auto made = Finish(*out, lists, originals, choices, batch && std::popcount(active) > 1,
                         activations, bytes);
      !made) {
    return std::unexpected(made.error());
  }
  return out;
}

std::expected<std::unique_ptr<Plan>, std::string> DraftPlan(
    const Requests<engine::Qwen38Model>& models, const Requests<kg::Qwen38MtpShape>& shapes,
    const kg::DeviceChoices& choices, bool batch, std::uint64_t activations, std::uint64_t bytes,
    std::uint32_t active) {
  if (active == 0 || active >= (1U << kMaxRequests)) {
    return Error("Draft active-slot mask invalid");
  }
  auto out = std::make_unique<Plan>();
  Requests<std::vector<ggml_tensor*>> lists;
  Requests<kg::GraphPlan> originals;
  for (std::size_t s = 0; s < kMaxRequests; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    if (shapes[s].rows <= 0 || shapes[s].rows > 4 || shapes[s].passes != 3 || !shapes[s].head) {
      return Error("C2 MTP requires fixed three passes with one to four catch-up rows");
    }
    auto p = engine::PlanQwen38Mtp(models[s], shapes[s], choices, 0, 0);
    if (!p) {
      return std::unexpected(p.error());
    }
    out->mtp[s] = std::move(*p);
    auto& g = out->mtp[s]->graph;
    lists[s] = g.nodes;
    originals[s] = out->mtp[s]->plan;
    const auto inputs = g.inputs();
    out->inputs.insert(out->inputs.end(), inputs.begin(), inputs.end());
    out->keep.insert(out->keep.end(), g.drafts.begin(), g.drafts.end());
    out->keep.insert(out->keep.end(), g.probabilities.begin(), g.probabilities.end());
    out->keep.insert(out->keep.end(), g.head_inputs.begin(), g.head_inputs.end());
    out->keep.insert(out->keep.end(), g.head_logits.begin(), g.head_logits.end());
  }
  if (auto made = Finish(*out, lists, originals, choices, batch && std::popcount(active) > 1,
                         activations, bytes);
      !made) {
    return std::unexpected(made.error());
  }
  return out;
}
}  // namespace llmp::benchmarks::qwen_batch
