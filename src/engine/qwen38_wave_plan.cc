// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen38_wave_plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engine/support.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"

namespace jitllm::engine {

bool Qwen38FullHeadPairCandidate(const ggml_tensor* t) {
  if (t == nullptr || t->op != GGML_OP_MUL_MAT || t->type != GGML_TYPE_F32 ||
      t->src[0] == nullptr || t->src[1] == nullptr) {
    return false;
  }
  const auto* w = t->src[0];
  const auto* x = t->src[1];
  return w->type == GGML_TYPE_BF16 && w->ne[0] == 2560 && w->ne[1] == 248320 && w->ne[2] == 1 &&
         w->ne[3] == 1 && ggml_is_contiguous(w) && x->type == GGML_TYPE_F32 && x->ne[0] == 2560 &&
         (x->ne[1] == 3 || x->ne[1] == 4) && x->ne[2] == 1 && x->ne[3] == 1 &&
         ggml_is_contiguous(x) && t->ne[0] == 248320 && t->ne[1] == x->ne[1] && t->ne[2] == 1 &&
         t->ne[3] == 1 && ggml_is_contiguous(t);
}

// Target-wave geometry shared with conservative workspace provisioning. Original one-row GemvBf16
// uses a different vector kernel; only the existing multirow cuBLAS path may be combined.
bool Qwen38HcPairCandidate(const ggml_tensor* t) {
  namespace kg = kernels::ggml;
  return t != nullptr && kg::IsGemvBf16(t) && t->src[0] != nullptr && t->src[1] != nullptr &&
         t->src[0]->type == GGML_TYPE_BF16 && t->src[1]->type == GGML_TYPE_BF16 &&
         (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) && t->ne[1] >= 2 &&
         t->ne[1] <= 4 && t->ne[1] > kg::kGemvBf16FastColumns && t->ne[2] == 1 && t->ne[3] == 1 &&
         t->src[1]->ne[1] == t->ne[1] && t->src[1]->ne[2] == 1 && t->src[1]->ne[3] == 1 &&
         ggml_is_contiguous(t->src[0]) && ggml_is_contiguous(t->src[1]) && ggml_is_contiguous(t);
}

namespace {
namespace kg = kernels::ggml;
using support::Address;
using support::Error;

template <typename T>
using Slots = std::array<T, kQwen38WaveSlots>;
using Lists = Slots<std::vector<ggml_tensor*>>;
using Originals = Slots<kg::GraphPlan>;

struct HeadPair {
  std::array<ggml_tensor*, 2> originals{};
  // Paid concat, ordinary product, and the two independent output views.
  std::array<ggml_tensor*, 4> nodes{};
};
using HeadPairs = Slots<std::optional<HeadPair>>;

// Only composition metadata is capped here; the original graph builders
// retain their own bounded arenas. Exceeding this cap keeps scalar products.
constexpr std::size_t kMaxPairedProducts = 4096;

struct Range {
  std::uint64_t first = 0;
  std::uint64_t end = 0;
};

bool Overlap(Range a, Range b) { return a.first < b.end && b.first < a.end; }

bool AddMutable(std::vector<Range>& ranges, std::uint64_t address, std::uint64_t bytes) {
  if (address == 0 || bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) {
    return false;
  }
  const Range added{address, address + bytes};
  if (std::ranges::any_of(ranges, [&](Range r) { return Overlap(r, added); })) {
    return false;
  }
  ranges.push_back(added);
  return true;
}

bool SameModel(const Qwen38Model& a, const Qwen38Model& b) {
  return a.artifact == b.artifact && a.profile == b.profile && a.binding == b.binding &&
         a.state == b.state && a.drafter == b.drafter && a.mtp_state == b.mtp_state &&
         a.commit == b.commit && a.mtp_stride == b.mtp_stride && a.fused == b.fused &&
         a.exact == b.exact && a.cutlass == b.cutlass && a.places.stride == b.places.stride &&
         a.places.ple_table == b.places.ple_table;
}

template <typename Request>
std::expected<std::vector<Range>, std::string> MutablePlaces(std::span<const Request> requests) {
  if (requests.empty() || requests.size() > kQwen38WaveSlots) {
    return Error("a Qwen3.8 wave requires one to four slots");
  }
  std::vector<Range> ranges;
  const Qwen38Model* first = nullptr;
  std::uint32_t previous = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (r.slot >= kQwen38WaveSlots || (i != 0 && r.slot <= previous) || r.model == nullptr) {
      return Error("Qwen3.8 wave slots must be unique, live and in ascending order");
    }
    previous = r.slot;
    const Qwen38Model& m = *r.model;
    if (m.artifact == nullptr || m.profile == nullptr || m.binding == nullptr ||
        m.state == nullptr || !m.places.resource || !m.places.array || m.places.ple_table == 0 ||
        (first != nullptr && !SameModel(*first, m))) {
      return Error("Qwen3.8 wave slots require one common native model binding");
    }
    first = &m;
    if (!AddMutable(ranges, m.places.state, m.state->bytes)) {
      return Error("Qwen3.8 wave target places are empty, overflowing or overlap");
    }
    if (m.mtp_state != nullptr && !AddMutable(ranges, m.places.mtp_state, m.mtp_state->bytes)) {
      return Error("Qwen3.8 wave MTP places are empty, overflowing or overlap");
    }
    if (m.commit != nullptr && !AddMutable(ranges, m.places.commit, m.commit->bytes)) {
      return Error("Qwen3.8 wave commit places are empty, overflowing or overlap");
    }
    if (m.drafter != nullptr &&
        (m.mtp_state == nullptr || !m.places.mtp_resource || !m.places.mtp_array)) {
      return Error("Qwen3.8 wave drafter binding is incomplete");
    }
  }
  return ranges;
}

bool Eligible(const ggml_tensor* t) {
  const auto op = kg::JitllmOpOf(t);
  return op == kg::JitllmOp::kMxfp8MulMatVec || op == kg::JitllmOp::kMoeGemv;
}

bool SameImmutableLeaf(const ggml_tensor* a, const ggml_tensor* b,
                       std::span<const Range> mutable_places) {
  if (a == nullptr || b == nullptr || a->op != GGML_OP_NONE || b->op != GGML_OP_NONE ||
      a->data == nullptr || a->data != b->data || a->type != b->type ||
      !std::ranges::equal(a->ne, b->ne) || !std::ranges::equal(a->nb, b->nb)) {
    return false;
  }
  const std::uint64_t address = Address(a->data);
  const std::uint64_t bytes = ggml_nbytes(a);
  if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) {
    return false;
  }
  return std::ranges::none_of(mutable_places,
                              [&](Range r) { return Overlap(r, {address, address + bytes}); });
}

bool Concatenable(const ggml_tensor* a, const ggml_tensor* b, std::size_t dim, ggml_type type) {
  if (a == nullptr || b == nullptr || a->type != type || b->type != type) {
    return false;
  }
  for (std::size_t i = 0; i < GGML_MAX_DIMS; ++i) {
    if (a->ne[i] <= 0 || b->ne[i] <= 0 || (i != dim && a->ne[i] != b->ne[i])) {
      return false;
    }
  }
  return true;
}

bool MatchHc(const ggml_tensor* a, const ggml_tensor* b, std::span<const Range> mutable_places) {
  return Qwen38HcPairCandidate(a) && Qwen38HcPairCandidate(b) && a->type == b->type &&
         a->ne[0] == b->ne[0] && a->ne[1] + b->ne[1] <= 8 &&
         std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) == 0 &&
         SameImmutableLeaf(a->src[0], b->src[0], mutable_places) &&
         Concatenable(a->src[1], b->src[1], 1, GGML_TYPE_BF16);
}

bool OrdinaryHead(const kg::GraphPlan& plan, const ggml_tensor* node) {
  const kg::PlanStep* found = nullptr;
  for (const auto& step : plan.steps) {
    if (step.nodes.size() == 1 && step.nodes.front() == node) {
      if (found != nullptr) {
        return false;
      }
      found = &step;
    }
  }
  return found != nullptr && found->operation == execution::Operation::kMatMul &&
         found->implementation == kg::kMulMatTensorCore;
}

bool Match(const ggml_tensor* a, const ggml_tensor* b, std::span<const Range> mutable_places) {
  if (!Eligible(a) || !Eligible(b) || kg::JitllmOpOf(a) != kg::JitllmOpOf(b) ||
      std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0 ||
      !SameImmutableLeaf(a->src[0], b->src[0], mutable_places) || a->type != GGML_TYPE_F32 ||
      b->type != GGML_TYPE_F32 || a->ne[0] != b->ne[0]) {
    return false;
  }
  if (kg::JitllmOpOf(a) == kg::JitllmOp::kMxfp8MulMatVec) {
    return SameImmutableLeaf(a->src[1], b->src[1], mutable_places) &&
           Concatenable(a->src[2], b->src[2], 1, GGML_TYPE_F32) && a->ne[1] >= 1 && a->ne[1] <= 4 &&
           b->ne[1] >= 1 && b->ne[1] <= 4 && a->ne[1] + b->ne[1] <= 8 && a->ne[2] == 1 &&
           b->ne[2] == 1 && a->ne[3] == 1 && b->ne[3] == 1 && a->src[2]->ne[1] == a->ne[1] &&
           b->src[2]->ne[1] == b->ne[1] && a->src[2]->ne[2] == 1 && a->src[2]->ne[3] == 1;
  }
  if (a->ne[1] != b->ne[1] || a->ne[2] < 1 || a->ne[2] > 4 || b->ne[2] < 1 || b->ne[2] > 4 ||
      a->ne[2] + b->ne[2] > 8 || a->ne[3] != 1 || b->ne[3] != 1 ||
      !Concatenable(a->src[1], b->src[1], 2, GGML_TYPE_F32) ||
      !Concatenable(a->src[2], b->src[2], 1, GGML_TYPE_I32) || a->src[1]->ne[2] != a->ne[2] ||
      b->src[1]->ne[2] != b->ne[2] || a->src[1]->ne[3] != 1 || a->src[2]->ne[0] != a->ne[1] ||
      a->src[2]->ne[1] != a->ne[2] || b->src[2]->ne[1] != b->ne[2] || a->src[2]->ne[2] != 1 ||
      a->src[2]->ne[3] != 1) {
    return false;
  }
  return !kg::IsMoeGemvSwiglu(a) || (SameImmutableLeaf(a->src[3], b->src[3], mutable_places) &&
                                     SameImmutableLeaf(a->src[4], b->src[4], mutable_places));
}

void Redirect(const Lists& lists, ggml_tensor* from, ggml_tensor* to) {
  for (const auto& list : lists) {
    for (ggml_tensor* n : list) {
      for (ggml_tensor*& source : n->src) {
        if (source == from) {
          source = to;
        }
      }
      if (n->view_src == from) {
        n->view_src = to;
      }
    }
  }
}

bool SameDraftPhase(const kg::Qwen38MtpShape& a, const kg::Qwen38MtpShape& b) {
  return a.passes == b.passes && a.head == b.head && a.head_rows == b.head_rows &&
         a.confidence == b.confidence && !a.capture_head && !b.capture_head;
}

std::expected<void, std::string> Topological(std::span<ggml_tensor* const> nodes) {
  std::unordered_set<const ggml_tensor*> seen;
  for (const ggml_tensor* node : nodes) {
    if (node == nullptr || seen.contains(node)) {
      return Error("Qwen3.8 wave repeats an operation descriptor");
    }
    for (const ggml_tensor* parent : node->src) {
      if (parent != nullptr && parent->op != GGML_OP_NONE && !seen.contains(parent)) {
        return Error("Qwen3.8 wave has an uncomputed parent or causal cycle");
      }
    }
    if (node->view_src != nullptr && node->view_src->op != GGML_OP_NONE &&
        !seen.contains(node->view_src)) {
      return Error("Qwen3.8 wave has an unordered view producer");
    }
    seen.insert(node);
  }
  return {};
}

}  // namespace

struct Qwen38WaveBuilder {
  static std::expected<HeadPairs, std::string> PrepareHeads(
      Qwen38WavePlanned& out, const Originals& originals, std::span<const std::uint32_t> order,
      const Slots<bool>& compatible, std::span<const Range> mutable_places,
      const kg::DeviceChoices& choices, bool share) {
    HeadPairs pairs;
    if (!share) {
      return pairs;
    }
    for (std::size_t i = 0; i + 1 < order.size(); i += 2) {
      const auto a_slot = order[i];
      const auto b_slot = order[i + 1];
      if (!compatible[a_slot] || out.target_[a_slot] == nullptr || out.target_[b_slot] == nullptr) {
        continue;
      }
      ggml_tensor* a = out.target_[a_slot]->graph.logits;
      ggml_tensor* b = out.target_[b_slot]->graph.logits;
      if (!Qwen38FullHeadPairCandidate(a) || !Qwen38FullHeadPairCandidate(b) ||
          !SameImmutableLeaf(a->src[0], b->src[0], mutable_places) ||
          std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0 ||
          !OrdinaryHead(originals[a_slot], a) || !OrdinaryHead(originals[b_slot], b)) {
        continue;
      }
      if (!out.head_arena_.has_value()) {
        auto arena = kg::TensorArena::Create((kQwen38WaveSlots / 2) * 4);
        if (!arena) {
          return Error(arena.error().detail);
        }
        out.head_arena_.emplace(std::move(*arena));
      }
      if (auto room = out.head_arena_->Reserve(4); !room) {
        return Error(room.error().detail);
      }
      ggml_context* c = out.head_arena_->context();
      ggml_tensor* x = ggml_concat(c, a->src[1], b->src[1], 1);
      ggml_tensor* both = ggml_mul_mat(c, a->src[0], x);
      std::memcpy(both->op_params, a->op_params, sizeof(both->op_params));
      HeadPair candidate{{a, b},
                         {x, both, ggml_view_2d(c, both, a->ne[0], a->ne[1], both->nb[1], 0),
                          ggml_view_2d(c, both, b->ne[0], b->ne[1], both->nb[1],
                                       static_cast<std::size_t>(a->ne[1]) * both->nb[1])}};
      // This metadata-only probe authenticates the actual device selector
      // before rewriting any original descriptor. Unsupported pairs retain
      // the scalar heads. Final placement rebinds these nodes and checks again.
      kg::BindDistinct(candidate.nodes, std::uint64_t{1} << 46U);
      auto probe = kg::PlanGraph(candidate.nodes, false, choices);
      if (!probe || !OrdinaryHead(*probe, both)) {
        continue;
      }
      pairs[a_slot] = candidate;
    }
    return pairs;
  }

  // Preflight every barrier before mutating any descriptor. Pair membership
  // is fixed by the ascending request list, never shifted as a graph ends.
  static std::expected<void, std::string> Compose(Qwen38WavePlanned& out, const Lists& lists,
                                                  std::span<const std::uint32_t> order,
                                                  const Slots<bool>& compatible,
                                                  std::span<const Range> mutable_places,
                                                  const HeadPairs& heads) {
    Lists barriers;
    for (std::size_t s = 0; s < kQwen38WaveSlots; ++s) {
      std::ranges::copy_if(lists[s], std::back_inserter(barriers[s]), Eligible);
    }
    Slots<bool> pair_first{};
    std::size_t products = 0;
    for (std::size_t i = 0; i + 1 < order.size(); i += 2) {
      const auto a = order[i];
      const auto b = order[i + 1];
      if (!compatible[a] || barriers[a].empty() || barriers[a].size() != barriers[b].size() ||
          barriers[a].size() > kMaxPairedProducts - products) {
        continue;
      }
      bool match = true;
      for (std::size_t n = 0; n < barriers[a].size(); ++n) {
        if (!Match(barriers[a][n], barriers[b][n], mutable_places)) {
          match = false;
          break;
        }
      }
      if (match) {
        pair_first[a] = true;
        products += barriers[a].size();
      }
    }
    // HC eligibility never enters the original all-product preflight above.
    // Authenticate the extra barriers independently, including their order
    // among the existing barriers. A refusal leaves all old pairing intact.
    Slots<bool> hc_slots{};
    for (std::size_t i = 0; i + 1 < order.size(); i += 2) {
      const auto a = order[i];
      const auto b = order[i + 1];
      if (!pair_first[a] || out.target_[a] == nullptr || out.target_[b] == nullptr ||
          out.target_[a]->graph.row_ids == nullptr || out.target_[b]->graph.row_ids == nullptr) {
        continue;
      }
      std::vector<ggml_tensor*> a_barriers;
      std::vector<ggml_tensor*> b_barriers;
      const auto extra_barrier = [](const ggml_tensor* t) {
        return Eligible(t) || Qwen38HcPairCandidate(t);
      };
      std::ranges::copy_if(lists[a], std::back_inserter(a_barriers), extra_barrier);
      std::ranges::copy_if(lists[b], std::back_inserter(b_barriers), extra_barrier);
      bool match = a_barriers.size() == b_barriers.size();
      std::size_t hc_products = 0;
      for (std::size_t n = 0; match && n < a_barriers.size(); ++n) {
        if (Qwen38HcPairCandidate(a_barriers[n])) {
          match = MatchHc(a_barriers[n], b_barriers[n], mutable_places);
          ++hc_products;
        } else {
          match = Match(a_barriers[n], b_barriers[n], mutable_places);
        }
      }
      if (match && hc_products != 0 && hc_products <= kMaxPairedProducts - products) {
        hc_slots[a] = true;
        hc_slots[b] = true;
        products += hc_products;
      }
    }
    if (products != 0) {
      auto arena = kg::TensorArena::Create(products * 5);
      if (!arena) {
        return Error(arena.error().detail);
      }
      out.arena.emplace(std::move(*arena));
    }
    Slots<ggml_tensor*> head_at{};
    for (std::size_t i = 0; i + 1 < order.size(); i += 2) {
      const auto a = order[i];
      const auto b = order[i + 1];
      if (heads[a].has_value()) {
        head_at[a] = heads[a]->originals[0];
        head_at[b] = heads[a]->originals[1];
      }
    }
    const auto barrier = [&](std::uint32_t slot, const ggml_tensor* t) {
      return Eligible(t) || t == head_at[slot] || (hc_slots[slot] && Qwen38HcPairCandidate(t));
    };
    Slots<std::size_t> at{};
    while (true) {
      bool ready = false;
      for (const auto s : order) {
        while (at[s] < lists[s].size() && !barrier(s, lists[s][at[s]])) {
          out.nodes_.push_back(lists[s][at[s]++]);
        }
        ready = ready || at[s] != lists[s].size();
      }
      if (!ready) {
        break;
      }
      for (std::size_t i = 0; i < order.size(); ++i) {
        const auto a_slot = order[i];
        if (at[a_slot] == lists[a_slot].size()) {
          continue;
        }
        if (lists[a_slot][at[a_slot]] == head_at[a_slot]) {
          // A shorter original product sequence may reach its head first.
          // Hold only that head until its fixed peer has computed its input.
          if (!heads[a_slot].has_value() || i + 1 >= order.size()) {
            continue;
          }
          const auto b_slot = order[i + 1];
          if (at[b_slot] == lists[b_slot].size() || lists[b_slot][at[b_slot]] != head_at[b_slot]) {
            continue;
          }
          const auto& head = *heads[a_slot];
          ++at[a_slot];
          ++at[b_slot];
          ++i;
          out.nodes_.insert(out.nodes_.end(), head.nodes.begin(), head.nodes.end());
          out.products_.push_back({head.originals[0], head.originals[1], head.nodes[1]});
          out.head_products_.push_back(out.products_.back());
          const auto packed = ggml_nbytes(head.nodes[0]);
          out.stats_.packed_bytes += packed;
          out.stats_.head_packed_bytes += packed;
          ++out.stats_.full_head_pairs;
          out.stats_.paired_slots |= static_cast<std::uint8_t>((1U << a_slot) | (1U << b_slot));
          for (std::size_t j = 0; j < 2; ++j) {
            ggml_tensor* original = head.originals[j];
            ggml_tensor* split = head.nodes[j + 2];
            for (ggml_tensor*& held : out.keep_) {
              if (held == original) {
                held = split;
              }
            }
            Redirect(lists, original, split);
            auto& graph = out.target_[j == 0 ? a_slot : b_slot]->graph;
            graph.logits = split;
            for (auto& named : graph.named) {
              if (named.second == original) {
                named.second = split;
              }
            }
          }
          continue;
        }
        ggml_tensor* a = lists[a_slot][at[a_slot]++];
        if (!pair_first[a_slot]) {
          out.nodes_.push_back(a);
          continue;
        }
        const auto b_slot = order[++i];
        if (at[b_slot] == lists[b_slot].size() || !out.arena.has_value()) {
          return Error("Qwen3.8 wave lost a preflighted product barrier");
        }
        ggml_tensor* b = lists[b_slot][at[b_slot]++];
        if (auto room = out.arena->Reserve(5); !room) {
          return Error(room.error().detail);
        }
        ggml_context* c = out.arena->context();
        ggml_tensor* both = nullptr;
        std::array<ggml_tensor*, 2> split{};
        if (Qwen38HcPairCandidate(a)) {
          if (!hc_slots[a_slot] || !hc_slots[b_slot] || !MatchHc(a, b, mutable_places)) {
            return Error("Qwen3.8 wave lost a preflighted HC barrier");
          }
          ggml_tensor* x = ggml_concat(c, a->src[1], b->src[1], 1);
          out.nodes_.push_back(x);
          both = kg::GemvBf16(c, a->src[0], x, a->type);
          std::memcpy(both->op_params, a->op_params, sizeof(both->op_params));
          split[0] = ggml_view_2d(c, both, a->ne[0], a->ne[1], both->nb[1], 0);
          split[1] = ggml_view_2d(c, both, b->ne[0], b->ne[1], both->nb[1],
                                  static_cast<std::size_t>(a->ne[1]) * both->nb[1]);
          out.stats_.packed_bytes += ggml_nbytes(x);
        } else if (kg::JitllmOpOf(a) == kg::JitllmOp::kMxfp8MulMatVec) {
          ggml_tensor* x = ggml_concat(c, a->src[2], b->src[2], 1);
          out.nodes_.push_back(x);
          both = kg::Mxfp8MulMatVec(c, a->src[0], a->src[1], x);
          split[0] = ggml_view_2d(c, both, a->ne[0], a->ne[1], both->nb[1], 0);
          split[1] = ggml_view_2d(c, both, b->ne[0], b->ne[1], both->nb[1],
                                  static_cast<std::size_t>(a->ne[1]) * both->nb[1]);
          out.stats_.packed_bytes += ggml_nbytes(x);
          ++out.stats_.mxfp8_pairs;
        } else {
          ggml_tensor* x = ggml_concat(c, a->src[1], b->src[1], 2);
          ggml_tensor* ids = ggml_concat(c, a->src[2], b->src[2], 1);
          out.nodes_.push_back(x);
          out.nodes_.push_back(ids);
          if (kg::IsMoeGemvSwiglu(a)) {
            both = kg::MoeGemvSwiglu(c, a->src[0], x, ids, a->ne[0], a->src[3], a->src[4],
                                     static_cast<std::uint64_t>(kg::JitllmOpInt(a, 2)),
                                     static_cast<std::uint64_t>(kg::JitllmOpInt(a, 3)));
          } else {
            both = kg::MoeGemv(c, a->src[0], x, ids, a->ne[0], kg::JitllmOpInt(a, 0),
                               kg::JitllmOpInt(a, 1),
                               static_cast<std::uint64_t>(kg::JitllmOpInt(a, 2)),
                               static_cast<std::uint64_t>(kg::JitllmOpInt(a, 3)));
          }
          split[0] =
              ggml_view_3d(c, both, a->ne[0], a->ne[1], a->ne[2], both->nb[1], both->nb[2], 0);
          split[1] = ggml_view_3d(c, both, b->ne[0], b->ne[1], b->ne[2], both->nb[1], both->nb[2],
                                  static_cast<std::size_t>(a->ne[2]) * both->nb[2]);
          out.stats_.packed_bytes += ggml_nbytes(x) + ggml_nbytes(ids);
          ++out.stats_.routed_pairs;
        }
        out.nodes_.push_back(both);
        out.nodes_.insert(out.nodes_.end(), split.begin(), split.end());
        out.products_.push_back({a, b, both});
        out.stats_.paired_slots |= static_cast<std::uint8_t>((1U << a_slot) | (1U << b_slot));
        for (ggml_tensor*& held : out.keep_) {
          if (held == a) {
            held = split[0];
          } else if (held == b) {
            held = split[1];
          }
        }
        Redirect(lists, a, split[0]);
        Redirect(lists, b, split[1]);
      }
    }
    return Topological(out.nodes_);
  }

  static std::expected<void, std::string> Authenticate(const Originals& originals,
                                                       const Qwen38WavePlanned& out) {
    for (const auto& plan : originals) {
      for (const auto& step : plan.steps) {
        if (step.nodes.size() == 1 && std::ranges::any_of(out.products_, [&](const auto& pair) {
              return pair[0] == step.nodes.front() || pair[1] == step.nodes.front();
            })) {
          continue;
        }
        const auto count = std::ranges::count_if(out.plan.steps, [&](const kg::PlanStep& s) {
          return s.operation == step.operation && s.implementation == step.implementation &&
                 s.nodes == step.nodes;
        });
        if (count != 1) {
          return Error("Qwen3.8 wave changed an unmodified implementation or fusion");
        }
      }
    }
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
    for (const auto& pair : out.products_) {
      const auto* a = original(pair[0]);
      const auto* b = original(pair[1]);
      const auto together = find(out.plan, pair[2]);
      if (a == nullptr || b == nullptr || together == out.plan.steps.end() ||
          a->operation != b->operation || a->implementation != b->implementation ||
          a->operation != together->operation || a->implementation != together->implementation) {
        return Error("Qwen3.8 wave changed a paired implementation or precision tier");
      }
    }
    for (const auto& pair : out.head_products_) {
      const auto* a = original(pair[0]);
      const auto* b = original(pair[1]);
      if (a == nullptr || b == nullptr || a->operation != execution::Operation::kMatMul ||
          a->implementation != kg::kMulMatTensorCore ||
          b->operation != execution::Operation::kMatMul ||
          b->implementation != kg::kMulMatTensorCore || !OrdinaryHead(out.plan, pair[2])) {
        return Error("Qwen3.8 full target heads did not retain the ordinary MMF selector");
      }
    }
    return {};
  }

  static std::expected<void, std::string> Finish(Qwen38WavePlanned& out, const Lists& lists,
                                                 const Originals& originals,
                                                 std::span<const std::uint32_t> order,
                                                 const Slots<bool>& compatible,
                                                 std::span<const Range> mutable_places,
                                                 const kg::DeviceChoices& choices,
                                                 Qwen38WavePlacement placement) {
    if (placement.activations != 0 &&
        placement.bytes > std::numeric_limits<std::uint64_t>::max() - placement.activations) {
      return Error("Qwen3.8 wave activation range overflows");
    }
    Slots<bool> allowed = compatible;
    if (!placement.paired) {
      allowed.fill(false);
    }
    auto heads = PrepareHeads(out, originals, order, compatible, mutable_places, choices,
                              placement.paired && placement.share_target_head);
    if (!heads) {
      return std::unexpected(heads.error());
    }
    if (auto made = Compose(out, lists, order, allowed, mutable_places, *heads); !made) {
      return made;
    }
    if (auto made = PlaceAndPlan(out, out.nodes_, out.inputs_, out.keep_, choices,
                                 placement.activations, placement.bytes);
        !made) {
      return made;
    }
    return Authenticate(originals, out);
  }

  static std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> Target(
      std::span<const Qwen38TargetWaveInput> requests, const kg::DeviceChoices& choices,
      Qwen38WavePlacement placement) {
    auto mutable_places = MutablePlaces(requests);
    if (!mutable_places) {
      return std::unexpected(mutable_places.error());
    }
    auto out = std::make_unique<Qwen38WavePlanned>();
    Lists lists;
    Originals originals;
    std::vector<std::uint32_t> order;
    Slots<bool> compatible{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& r = requests[i];
      if (r.shape.rows < 1 || r.shape.rows > 4) {
        return Error("Qwen3.8 target wave requires one to four rows per slot");
      }
      auto p = PlanQwen38Chunk(*r.model, r.shape, choices, 0, 0, {}, r.kind);
      if (!p) {
        return std::unexpected(p.error());
      }
      const auto s = r.slot;
      out->target_[s] = std::move(*p);
      const auto& g = out->target_[s]->graph;
      lists[s] = g.nodes;
      originals[s] = out->target_[s]->plan;
      const auto inputs = g.inputs();
      out->inputs_.insert(out->inputs_.end(), inputs.begin(), inputs.end());
      out->keep_.push_back(g.logits);
      if (g.argmax != nullptr) {
        out->keep_.push_back(g.argmax);
      }
      for (const auto& layer : g.routed) {
        for (ggml_tensor* t :
             {layer.input, layer.activation, layer.down, layer.shared, layer.gate, layer.weights,
              layer.ids, layer.combined, layer.attention_input, layer.attention_projection}) {
          if (t != nullptr) {
            out->keep_.push_back(t);
          }
        }
      }
      order.push_back(s);
      out->active_slots_ |= static_cast<std::uint8_t>(1U << s);
      if (i % 2 == 0 && i + 1 < requests.size()) {
        compatible[s] = r.kind == requests[i + 1].kind && r.kind.capture_routed == 0;
      }
    }
    if (auto made =
            Finish(*out, lists, originals, order, compatible, *mutable_places, choices, placement);
        !made) {
      return std::unexpected(made.error());
    }
    return out;
  }

  static std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> Draft(
      std::span<const Qwen38DraftWaveInput> requests, const kg::DeviceChoices& choices,
      Qwen38WavePlacement placement) {
    auto mutable_places = MutablePlaces(requests);
    if (!mutable_places) {
      return std::unexpected(mutable_places.error());
    }
    auto out = std::make_unique<Qwen38WavePlanned>();
    Lists lists;
    Originals originals;
    std::vector<std::uint32_t> order;
    Slots<bool> compatible{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& r = requests[i];
      if (r.shape.rows < 1 || r.shape.rows > 4 || r.shape.passes < 1 || r.shape.passes > 8) {
        return Error("Qwen3.8 draft wave requires one to four rows and one to eight passes");
      }
      auto p = PlanQwen38Mtp(*r.model, r.shape, choices, 0, 0);
      if (!p) {
        return std::unexpected(p.error());
      }
      const auto s = r.slot;
      out->draft_[s] = std::move(*p);
      const auto& g = out->draft_[s]->graph;
      lists[s] = g.nodes;
      originals[s] = out->draft_[s]->plan;
      const auto inputs = g.inputs();
      out->inputs_.insert(out->inputs_.end(), inputs.begin(), inputs.end());
      for (const auto* held : {&g.drafts, &g.probabilities, &g.head_inputs, &g.head_logits}) {
        out->keep_.insert(out->keep_.end(), held->begin(), held->end());
      }
      order.push_back(s);
      out->active_slots_ |= static_cast<std::uint8_t>(1U << s);
      if (i % 2 == 0 && i + 1 < requests.size()) {
        compatible[s] = SameDraftPhase(r.shape, requests[i + 1].shape);
      }
    }
    if (auto made =
            Finish(*out, lists, originals, order, compatible, *mutable_places, choices, placement);
        !made) {
      return std::unexpected(made.error());
    }
    return out;
  }
};

Qwen38WavePlanned::~Qwen38WavePlanned() {
  // Drop bound descriptor references while every original arena is alive.
  // Captured graphs and completion-aware retirement remain the caller's.
  bound.reset();
}

const kg::Qwen38Graph* Qwen38WavePlanned::target(std::size_t slot) const {
  return slot < target_.size() && target_[slot] != nullptr ? &target_[slot]->graph : nullptr;
}

const kg::Qwen38MtpGraph* Qwen38WavePlanned::draft(std::size_t slot) const {
  return slot < draft_.size() && draft_[slot] != nullptr ? &draft_[slot]->graph : nullptr;
}

std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38TargetWave(
    std::span<const Qwen38TargetWaveInput> requests, const kg::DeviceChoices& choices,
    Qwen38WavePlacement placement) {
  return Qwen38WaveBuilder::Target(requests, choices, placement);
}

std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38DraftWave(
    std::span<const Qwen38DraftWaveInput> requests, const kg::DeviceChoices& choices,
    Qwen38WavePlacement placement) {
  return Qwen38WaveBuilder::Draft(requests, choices, placement);
}

}  // namespace jitllm::engine
