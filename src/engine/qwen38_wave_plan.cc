// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen38_wave_plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
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
#include "kernels/ggml/set_rows_group.h"

namespace jitllm::engine {

bool Qwen38SelectedQ4HeadPrefix(const ggml_tensor* weight, const ggml_tensor* parent) {
  constexpr std::uint64_t kRowBytes = 1600;
  const auto shape = [](const ggml_tensor* t) {
    return t != nullptr && t->type == GGML_TYPE_Q4_1 && t->ne[0] == 2560 && t->ne[1] >= 1 &&
           t->ne[1] <= 248320 && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == 20 &&
           t->nb[1] == kRowBytes && t->nb[2] == kRowBytes * static_cast<std::uint64_t>(t->ne[1]) &&
           t->nb[3] == t->nb[2] && t->data != nullptr;
  };
  if (!shape(parent) || parent->op != GGML_OP_NONE || parent->view_src != nullptr ||
      !std::ranges::all_of(parent->src, [](const auto* src) { return src == nullptr; }))
    return false;
  const auto address = reinterpret_cast<std::uintptr_t>(parent->data);
  const auto bytes = kRowBytes * static_cast<std::uint64_t>(parent->ne[1]);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
  if (weight == parent) return true;
  if (!shape(weight) || weight->op != GGML_OP_VIEW || weight->view_src != parent ||
      weight->src[0] != parent || weight->view_offs != 0 || weight->data != parent->data ||
      weight->ne[1] > parent->ne[1] ||
      !std::ranges::all_of(std::span(weight->src).subspan(1),
                           [](const auto* src) { return src == nullptr; }))
    return false;
  std::size_t encoded_offset = 0;
  std::memcpy(&encoded_offset, weight->op_params, sizeof(encoded_offset));
  return encoded_offset == 0 && ggml_nbytes(weight) <= bytes;
}

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

// Target-wave geometry. Original one-row GemvBf16
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
using Product = Qwen38WavePlanned::Product;

// Consecutive slots of the ascending request order; the first leads.
using Group = std::vector<std::uint32_t>;

struct HeadGroup {
  std::vector<ggml_tensor*> originals;
  // Paid concats, the ordinary product, then a view a slot.
  std::vector<ggml_tensor*> nodes;
  ggml_tensor* product = nullptr;
  std::vector<ggml_tensor*> views;
};
using HeadGroups = Slots<std::optional<HeadGroup>>;  // at each group's leader

// Only composition metadata is capped here; the original graph builders
// retain their own bounded arenas. Groups past this cap refuse the wave;
// HC products past it stay scalar.
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
    return Error(std::format("a Qwen3.8 wave requires one to {} slots", kQwen38WaveSlots));
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

bool VecQRouted(const ggml_tensor* t) {
  return t->src[2] != nullptr && t->src[2]->type == GGML_TYPE_I32;
}

// Selected MTP heads share GGML's one-column vector arithmetic. The full
// target head and other BF16 products do not qualify.
bool DraftHead(const ggml_tensor* t) {
  return t != nullptr && t->op == GGML_OP_MUL_MAT && t->type == GGML_TYPE_F32 &&
         t->src[0] != nullptr && t->src[1] != nullptr && t->src[0]->type == GGML_TYPE_BF16 &&
         t->src[0]->ne[0] == 2560 && t->src[0]->ne[1] >= 16384 && t->src[0]->ne[1] <= 65536 &&
         t->src[1]->type == GGML_TYPE_F32 && t->src[1]->ne[0] == 2560 && t->src[1]->ne[1] >= 1 &&
         t->src[1]->ne[1] <= 8 && t->ne[0] == t->src[0]->ne[1] && t->ne[1] == t->src[1]->ne[1] &&
         t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t->src[0]) &&
         ggml_is_contiguous(t->src[1]) && ggml_is_contiguous(t);
}

bool Eligible(const ggml_tensor* t) {
  const auto op = kg::JitllmOpOf(t);
  // Only one-row GGUF steps: the shared launch retains their reduction.
  // Multirow products and grouped products keep their original operations.
  return (DraftHead(t) && t->ne[1] == 1) || op == kg::JitllmOp::kMxfp8MulMatVec ||
         op == kg::JitllmOp::kMoeGemv ||
         (op == kg::JitllmOp::kVecQ && kg::JitllmOpInt(t, 0) == 1 &&
          (VecQRouted(t) || t->src[0]->ne[2] == 1));
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

bool SameHeadWeight(const ggml_tensor* a, const ggml_tensor* b,
                    std::span<const Range> mutable_places) {
  if (SameImmutableLeaf(a, b, mutable_places)) return true;
  if (a == nullptr || b == nullptr || a->op != GGML_OP_VIEW || b->op != GGML_OP_VIEW ||
      a->type != b->type || a->view_offs != b->view_offs || !std::ranges::equal(a->ne, b->ne) ||
      !std::ranges::equal(a->nb, b->nb) ||
      std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0)
    return false;
  return SameImmutableLeaf(a->view_src, b->view_src, mutable_places);
}

bool SameSelectedQ4Prefix(const ggml_tensor* a, const ggml_tensor* b,
                          std::span<const Range> mutable_places) {
  if (a == nullptr || b == nullptr || a->op != GGML_OP_VIEW || b->op != GGML_OP_VIEW ||
      !Qwen38SelectedQ4HeadPrefix(a, a->view_src) || !Qwen38SelectedQ4HeadPrefix(b, b->view_src) ||
      !std::ranges::equal(a->ne, b->ne) || !std::ranges::equal(a->nb, b->nb) ||
      std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0)
    return false;
  // Authenticate the complete immutable parent, including its unused suffix.
  return SameImmutableLeaf(a->view_src, b->view_src, mutable_places);
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
         a->ne[0] == b->ne[0] &&
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

// Whether b may join a's product (each at most four rows; the group's sum
// is checked separately against the kernels' limits).
bool Match(const ggml_tensor* a, const ggml_tensor* b, std::span<const Range> mutable_places) {
  if (a == nullptr || b == nullptr) return false;
  if (DraftHead(a) || DraftHead(b)) {
    return DraftHead(a) && DraftHead(b) && a->ne[1] == 1 && b->ne[1] == 1 && a->ne[0] == b->ne[0] &&
           std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) == 0 &&
           SameHeadWeight(a->src[0], b->src[0], mutable_places);
  }
  if (!Eligible(a) || !Eligible(b) || kg::JitllmOpOf(a) != kg::JitllmOpOf(b) ||
      std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0 ||
      !(SameImmutableLeaf(a->src[0], b->src[0], mutable_places) ||
        (kg::JitllmOpOf(a) == kg::JitllmOp::kVecQ && !VecQRouted(a) && !VecQRouted(b) &&
         SameSelectedQ4Prefix(a->src[0], b->src[0], mutable_places))) ||
      a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || a->ne[0] != b->ne[0]) {
    return false;
  }
  if (kg::JitllmOpOf(a) == kg::JitllmOp::kVecQ) {
    const bool routed = VecQRouted(a);
    if (routed != VecQRouted(b) || a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3] ||
        a->ne[1] != b->ne[1] || a->src[1] == nullptr || b->src[1] == nullptr ||
        kg::JitllmOpOf(a->src[1]) != kg::JitllmOp::kQuantizeQ8 ||
        kg::JitllmOpOf(b->src[1]) != kg::JitllmOp::kQuantizeQ8) {
      return false;
    }
    const auto* gate_a = a->src[routed ? 3 : 2];
    const auto* gate_b = b->src[routed ? 3 : 2];
    const bool per_slot = kg::JitllmOpInt(a, 1) != 0;
    return ((gate_a == nullptr && gate_b == nullptr) ||
            SameImmutableLeaf(gate_a, gate_b, mutable_places)) &&
           Concatenable(a->src[1]->src[0], b->src[1]->src[0], per_slot ? 2 : 1, GGML_TYPE_F32) &&
           (!routed || Concatenable(a->src[2], b->src[2], 1, GGML_TYPE_I32));
  }
  if (kg::JitllmOpOf(a) == kg::JitllmOp::kMxfp8MulMatVec) {
    return SameImmutableLeaf(a->src[1], b->src[1], mutable_places) &&
           Concatenable(a->src[2], b->src[2], 1, GGML_TYPE_F32) && a->ne[1] >= 1 && a->ne[1] <= 4 &&
           b->ne[1] >= 1 && b->ne[1] <= 4 && a->ne[2] == 1 && b->ne[2] == 1 && a->ne[3] == 1 &&
           b->ne[3] == 1 && a->src[2]->ne[1] == a->ne[1] && b->src[2]->ne[1] == b->ne[1] &&
           a->src[2]->ne[2] == 1 && a->src[2]->ne[3] == 1;
  }
  if (a->ne[1] != b->ne[1] || a->ne[2] < 1 || a->ne[2] > 4 || b->ne[2] < 1 || b->ne[2] > 4 ||
      a->ne[3] != 1 || b->ne[3] != 1 || !Concatenable(a->src[1], b->src[1], 2, GGML_TYPE_F32) ||
      !Concatenable(a->src[2], b->src[2], 1, GGML_TYPE_I32) || a->src[1]->ne[2] != a->ne[2] ||
      b->src[1]->ne[2] != b->ne[2] || a->src[1]->ne[3] != 1 || a->src[2]->ne[0] != a->ne[1] ||
      a->src[2]->ne[1] != a->ne[2] || b->src[2]->ne[1] != b->ne[2] || a->src[2]->ne[2] != 1 ||
      a->src[2]->ne[3] != 1) {
    return false;
  }
  return !kg::IsMoeGemvSwiglu(a) || (SameImmutableLeaf(a->src[3], b->src[3], mutable_places) &&
                                     SameImmutableLeaf(a->src[4], b->src[4], mutable_places));
}

// The rows a product joins (columns; a routed product's tokens) and the
// most its shared kernel takes.
std::int64_t JoinedRows(const ggml_tensor* t) {
  if (kg::JitllmOpOf(t) == kg::JitllmOp::kVecQ) {
    return kg::JitllmOpInt(t, 0);
  }
  return kg::JitllmOpOf(t) == kg::JitllmOp::kMoeGemv ? t->ne[2] : t->ne[1];
}

std::int64_t JoinedLimit(const ggml_tensor* t) {
  if (DraftHead(t)) {
    return 8;
  }
  switch (kg::JitllmOpOf(t)) {
    case kg::JitllmOp::kVecQ:
      return VecQRouted(t) ? std::min(kg::kVecQMaxTokens, 128 / t->ne[1]) : kg::kVecQMaxTokens;
    case kg::JitllmOp::kMoeGemv:
      return kg::kMoeGemvWaveTokens;
    case kg::JitllmOp::kMxfp8MulMatVec:
      return kg::kMxfp8VecWaveColumns;
    default:
      return 16;  // HC's cuBLAS product
  }
}

// `xs` joined along `dim` by paid GGML concats, each appended to `nodes`.
ggml_tensor* ConcatAll(ggml_context* c, std::span<ggml_tensor* const> xs, int dim,
                       std::vector<ggml_tensor*>& nodes) {
  ggml_tensor* x = xs[0];
  for (std::size_t i = 1; i < xs.size(); ++i) {
    x = ggml_concat(c, x, xs[i], dim);
    nodes.push_back(x);
  }
  return x;
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
  // Consecutive slots of `order`, each `compatible` with the one before it,
  // whose product barriers match the group leader's in sequence and whose
  // rows fit each shared kernel. A slot that does not match its leader
  // starts the next group. Without `allowed`, every slot is its own group.
  static std::vector<Group> Groups(const Lists& lists, std::span<const std::uint32_t> order,
                                   const Slots<bool>& compatible,
                                   std::span<const Range> mutable_places, bool allowed) {
    Lists barriers;
    for (const auto s : order) {
      std::ranges::copy_if(lists[s], std::back_inserter(barriers[s]), Eligible);
    }
    const auto joins = [&](const Group& g, std::uint32_t s) {
      const auto a = g.front();
      if (barriers[a].empty() || barriers[a].size() != barriers[s].size()) {
        return false;
      }
      for (std::size_t n = 0; n < barriers[a].size(); ++n) {
        if (!Match(barriers[a][n], barriers[s][n], mutable_places)) {
          return false;
        }
        std::int64_t rows = JoinedRows(barriers[s][n]);
        for (const auto m : g) {
          rows += JoinedRows(barriers[m][n]);
        }
        if (rows > JoinedLimit(barriers[a][n])) {
          return false;
        }
      }
      return true;
    };
    std::vector<Group> groups;
    for (std::size_t i = 0; i < order.size(); ++i) {
      const auto s = order[i];
      if (i == 0 || !allowed || !compatible[s] || !joins(groups.back(), s)) {
        groups.push_back({s});
      } else {
        groups.back().push_back(s);
      }
    }
    return groups;
  }

  static std::expected<HeadGroups, std::string> PrepareHeads(
      Qwen38WavePlanned& out, const Originals& originals, std::span<const Group> groups,
      std::span<const Range> mutable_places, const kg::DeviceChoices& choices, bool share) {
    HeadGroups heads;
    if (!share) {
      return heads;
    }
    for (const Group& g : groups) {
      if (g.size() < 2) {
        continue;
      }
      std::vector<ggml_tensor*> logits;
      bool eligible = true;
      std::int64_t columns = 0;
      for (const auto s : g) {
        if (out.target_[s] == nullptr) {
          eligible = false;
          break;
        }
        ggml_tensor* t = out.target_[s]->graph.logits;
        const ggml_tensor* lead = logits.empty() ? t : logits.front();
        if (!Qwen38FullHeadPairCandidate(t) || !OrdinaryHead(originals[s], t) ||
            !SameImmutableLeaf(lead->src[0], t->src[0], mutable_places) ||
            std::memcmp(lead->op_params, t->op_params, sizeof(t->op_params)) != 0) {
          eligible = false;
          break;
        }
        columns += t->ne[1];
        logits.push_back(t);
      }
      if (!eligible || columns > 16) {
        continue;
      }
      if (!out.head_arena_.has_value()) {
        auto arena = kg::TensorArena::Create(4 * kQwen38WaveSlots);
        if (!arena) {
          return Error(arena.error().detail);
        }
        out.head_arena_.emplace(std::move(*arena));
      }
      if (auto room = out.head_arena_->Reserve(2 * g.size()); !room) {
        return Error(room.error().detail);
      }
      ggml_context* c = out.head_arena_->context();
      HeadGroup candidate;
      candidate.originals = logits;
      std::vector<ggml_tensor*> xs;
      xs.reserve(logits.size());
      for (ggml_tensor* t : logits) {
        xs.push_back(t->src[1]);
      }
      ggml_tensor* x = ConcatAll(c, xs, 1, candidate.nodes);
      candidate.product = ggml_mul_mat(c, logits.front()->src[0], x);
      std::memcpy(candidate.product->op_params, logits.front()->op_params,
                  sizeof(candidate.product->op_params));
      candidate.nodes.push_back(candidate.product);
      std::size_t offset = 0;
      for (ggml_tensor* t : logits) {
        ggml_tensor* view = ggml_view_2d(c, candidate.product, t->ne[0], t->ne[1],
                                         candidate.product->nb[1], offset);
        offset += static_cast<std::size_t>(t->ne[1]) * candidate.product->nb[1];
        candidate.views.push_back(view);
        candidate.nodes.push_back(view);
      }
      // This metadata-only probe authenticates the actual device selector
      // before rewriting any original descriptor. Unsupported groups retain
      // the scalar heads. Final placement rebinds these nodes and checks again.
      kg::BindDistinct(candidate.nodes, std::uint64_t{1} << 46U);
      auto probe = kg::PlanGraph(candidate.nodes, false, choices);
      if (!probe || !OrdinaryHead(*probe, candidate.product)) {
        continue;
      }
      heads[g.front()] = std::move(candidate);
    }
    return heads;
  }

  // Every group's barriers were preflighted (Groups) before any descriptor
  // changes; membership never shifts as a graph ends.
  static std::expected<void, std::string> Compose(Qwen38WavePlanned& out, const Lists& lists,
                                                  std::span<const Group> groups,
                                                  std::span<const Range> mutable_places,
                                                  const HeadGroups& heads, bool lanes) {
    std::size_t products = 0;
    // A shared product's descriptors: its members' concats, the product and
    // a view a member, within four a member of the widest group.
    std::size_t widest = 1;
    for (const Group& g : groups) {
      if (g.size() > 1) {
        products += static_cast<std::size_t>(std::ranges::count_if(lists[g.front()], Eligible));
      }
      widest = std::max(widest, g.size());
    }
    if (products > kMaxPairedProducts) {
      return Error("Qwen3.8 wave exceeds its shared product bound");
    }
    // HC eligibility never enters the product preflight. Authenticate the
    // extra barriers independently, including their order among the
    // existing barriers; HC products join only target verifications. A
    // refusal leaves the other grouping intact.
    Slots<bool> hc_slots{};
    for (const Group& g : groups) {
      if (g.size() < 2 || !std::ranges::all_of(g, [&](auto s) {
            return out.target_[s] != nullptr && out.target_[s]->graph.row_ids != nullptr;
          })) {
        continue;
      }
      const auto extra_barrier = [](const ggml_tensor* t) {
        return Eligible(t) || Qwen38HcPairCandidate(t);
      };
      std::vector<ggml_tensor*> a_barriers;
      std::ranges::copy_if(lists[g.front()], std::back_inserter(a_barriers), extra_barrier);
      bool match = true;
      std::size_t hc_products = 0;
      std::vector<std::int64_t> rows(a_barriers.size(), 0);
      for (std::size_t k = 0; match && k < g.size(); ++k) {
        std::vector<ggml_tensor*> s_barriers;
        std::ranges::copy_if(lists[g[k]], std::back_inserter(s_barriers), extra_barrier);
        match = s_barriers.size() == a_barriers.size();
        for (std::size_t n = 0; match && n < a_barriers.size(); ++n) {
          if (Qwen38HcPairCandidate(a_barriers[n])) {
            rows[n] += s_barriers[n]->ne[1];
            match = MatchHc(a_barriers[n], s_barriers[n], mutable_places) && rows[n] <= 16;
            hc_products += k == 0 ? 1 : 0;
          } else {
            match = Match(a_barriers[n], s_barriers[n], mutable_places);
          }
        }
      }
      if (match && hc_products != 0 && hc_products <= kMaxPairedProducts - products) {
        for (const auto s : g) {
          hc_slots[s] = true;
        }
        products += hc_products;
      }
    }
    if (products != 0) {
      auto arena = kg::TensorArena::Create(products * 4 * widest);
      if (!arena) {
        return Error(arena.error().detail);
      }
      out.arena.emplace(std::move(*arena));
    }
    Slots<ggml_tensor*> head_at{};
    for (const Group& g : groups) {
      if (heads[g.front()].has_value()) {
        for (std::size_t k = 0; k < g.size(); ++k) {
          head_at[g[k]] = heads[g.front()]->originals[k];
        }
      }
    }
    const auto barrier = [&](std::uint32_t slot, const ggml_tensor* t) {
      return Eligible(t) || t == head_at[slot] || (hc_slots[slot] && Qwen38HcPairCandidate(t));
    };
    Slots<std::size_t> at{};
    // With lanes (graph_plan.h AssignLanes): each slot's own operations
    // between two product barriers on a lane of its own (the slots in wave
    // order, past kMaxLanes sharing lanes in turn), each pass a region; the
    // shared products, their concatenations and views stay on the stream.
    std::size_t slots = 0;
    for (const Group& g : groups) {
      slots += g.size();
    }
    lanes = lanes && slots >= kQwen38LaneSlots;
    std::uint32_t region = 0;
    while (true) {
      // Every pass emits at least one descriptor or ends; a group whose
      // members wait on each other at different barriers would otherwise
      // spin here.
      const std::size_t emitted = out.nodes_.size();
      bool ready = false;
      ++region;
      std::uint32_t position = 0;
      for (const Group& g : groups) {
        for (const auto s : g) {
          const auto lane = static_cast<std::uint8_t>((position++ % kg::kMaxLanes) + 1);
          while (at[s] < lists[s].size() && !barrier(s, lists[s][at[s]])) {
            ggml_tensor* node = lists[s][at[s]++];
            out.nodes_.push_back(node);
            if (lanes) {
              out.lanes_.emplace_back(node, kg::LaneTag{.lane = lane, .region = region});
            }
          }
          ready = ready || at[s] != lists[s].size();
        }
      }
      if (!ready) {
        break;
      }
      for (const Group& g : groups) {
        const auto a_slot = g.front();
        if (g.size() == 1) {
          if (at[a_slot] != lists[a_slot].size()) {
            out.nodes_.push_back(lists[a_slot][at[a_slot]++]);
          }
          continue;
        }
        if (at[a_slot] == lists[a_slot].size()) {
          if (std::ranges::any_of(g, [&](auto s) { return at[s] != lists[s].size(); })) {
            return Error("Qwen3.8 wave lost a preflighted product barrier");
          }
          continue;
        }
        if (lists[a_slot][at[a_slot]] == head_at[a_slot]) {
          // A shorter original product sequence may reach its head first.
          // Hold the heads until every member has computed its input.
          if (!std::ranges::all_of(g, [&](auto s) {
                return at[s] != lists[s].size() && lists[s][at[s]] == head_at[s];
              })) {
            continue;
          }
          const auto& head = *heads[a_slot];
          out.nodes_.insert(out.nodes_.end(), head.nodes.begin(), head.nodes.end());
          out.products_.push_back({head.originals, head.product});
          out.head_products_.push_back(out.products_.back());
          for (const ggml_tensor* n : head.nodes) {
            if (n->op == GGML_OP_CONCAT) {
              out.stats_.packed_bytes += ggml_nbytes(n);
              out.stats_.head_packed_bytes += ggml_nbytes(n);
            }
          }
          ++out.stats_.full_head_pairs;
          for (std::size_t k = 0; k < g.size(); ++k) {
            const auto s = g[k];
            ++at[s];
            out.stats_.paired_slots |= std::uint32_t{1} << s;
            ggml_tensor* original = head.originals[k];
            ggml_tensor* split = head.views[k];
            for (ggml_tensor*& held : out.keep_) {
              if (held == original) {
                held = split;
              }
            }
            Redirect(lists, original, split);
            auto& graph = out.target_[s]->graph;
            graph.logits = split;
            for (auto& named : graph.named) {
              if (named.second == original) {
                named.second = split;
              }
            }
          }
          continue;
        }
        std::vector<ggml_tensor*> members;
        for (const auto s : g) {
          if (at[s] == lists[s].size() || !out.arena.has_value()) {
            return Error("Qwen3.8 wave lost a preflighted product barrier");
          }
          members.push_back(lists[s][at[s]++]);
        }
        ggml_tensor* a = members.front();
        const bool hc = !Eligible(a) && hc_slots[a_slot] && Qwen38HcPairCandidate(a);
        for (std::size_t k = 1; k < members.size(); ++k) {
          if (hc ? (!hc_slots[g[k]] || !MatchHc(a, members[k], mutable_places))
                 : !Match(a, members[k], mutable_places)) {
            return Error("Qwen3.8 wave lost a preflighted product barrier");
          }
        }
        if (auto room = out.arena->Reserve(4 * widest); !room) {
          return Error(room.error().detail);
        }
        ggml_context* c = out.arena->context();
        std::vector<ggml_tensor*> made;
        std::vector<ggml_tensor*> xs;
        ggml_tensor* both = nullptr;
        std::vector<ggml_tensor*> split;
        if (hc) {
          for (ggml_tensor* m : members) {
            xs.push_back(m->src[1]);
          }
          ggml_tensor* x = ConcatAll(c, xs, 1, made);
          both = kg::GemvBf16(c, a->src[0], x, a->type);
          std::memcpy(both->op_params, a->op_params, sizeof(both->op_params));
          std::size_t offset = 0;
          for (ggml_tensor* m : members) {
            split.push_back(ggml_view_2d(c, both, m->ne[0], m->ne[1], both->nb[1], offset));
            offset += static_cast<std::size_t>(m->ne[1]) * both->nb[1];
          }
          out.stats_.packed_bytes += ggml_nbytes(x);
        } else if (DraftHead(a)) {
          for (ggml_tensor* m : members) {
            xs.push_back(m->src[1]);
          }
          ggml_tensor* x = ConcatAll(c, xs, 1, made);
          both = ggml_mul_mat(c, a->src[0], x);
          std::memcpy(both->op_params, a->op_params, sizeof(a->op_params));
          for (std::size_t k = 0; k < members.size(); ++k) {
            split.push_back(ggml_view_2d(c, both, a->ne[0], 1, both->nb[1], k * both->nb[1]));
          }
          out.stats_.packed_bytes += ggml_nbytes(x);
          ++out.stats_.draft_head_pairs;
        } else if (kg::JitllmOpOf(a) == kg::JitllmOp::kVecQ) {
          const bool routed = VecQRouted(a);
          const bool per_slot = kg::JitllmOpInt(a, 1) != 0;
          std::vector<ggml_tensor*> ids;
          for (ggml_tensor* m : members) {
            xs.push_back(m->src[1]->src[0]);
            if (routed) {
              ids.push_back(m->src[2]);
            }
          }
          // Keep the scalar Q8 nodes for any other consumers. Packing F32
          // inputs and quantizing the joined rows preserves the ordinary
          // Q8 producer contract; all packing and repeated quantization is paid.
          ggml_tensor* x = ConcatAll(c, xs, per_slot ? 2 : 1, made);
          ggml_tensor* q8 = kg::QuantizeQ8(c, x);
          made.push_back(q8);
          ggml_tensor* joined_ids = routed ? ConcatAll(c, ids, 1, made) : nullptr;
          both = kg::VecQ(c, a->src[0], q8, joined_ids, static_cast<std::int64_t>(members.size()),
                          per_slot, a->src[routed ? 3 : 2],
                          static_cast<kg::VecQGlu>(kg::JitllmOpInt(a, 2)), kg::JitllmOpFloat(a, 3));
          kg::SetVecQOneToken(both);
          for (std::size_t k = 0; k < members.size(); ++k) {
            const ggml_tensor* m = members[k];
            if (routed) {
              split.push_back(ggml_view_3d(c, both, m->ne[0], m->ne[1], 1, both->nb[1], both->nb[2],
                                           k * both->nb[2]));
            } else {
              split.push_back(ggml_view_2d(c, both, m->ne[0], 1, both->nb[1], k * both->nb[1]));
            }
          }
          out.stats_.packed_bytes += ggml_nbytes(x) + (routed ? ggml_nbytes(joined_ids) : 0);
          ++out.stats_.vecq_pairs;
          bool selected_head = a->src[0]->type == GGML_TYPE_Q4_1;
          for (std::size_t k = 0; k < g.size(); ++k) {
            const auto* draft = out.draft_[g[k]].get();
            selected_head = selected_head && draft != nullptr &&
                            draft->graph.draft_ids != nullptr &&
                            Qwen38SelectedQ4HeadPrefix(members[k]->src[0], draft->graph.output) &&
                            members[k]->ne[0] == members[k]->src[0]->ne[1];
          }
          if (selected_head) ++out.stats_.quantized_draft_head_pairs;
        } else if (kg::JitllmOpOf(a) == kg::JitllmOp::kMxfp8MulMatVec) {
          for (ggml_tensor* m : members) {
            xs.push_back(m->src[2]);
          }
          ggml_tensor* x = ConcatAll(c, xs, 1, made);
          both = kg::Mxfp8MulMatVec(c, a->src[0], a->src[1], x);
          std::size_t offset = 0;
          for (ggml_tensor* m : members) {
            split.push_back(ggml_view_2d(c, both, m->ne[0], m->ne[1], both->nb[1], offset));
            offset += static_cast<std::size_t>(m->ne[1]) * both->nb[1];
          }
          out.stats_.packed_bytes += ggml_nbytes(x);
          ++out.stats_.mxfp8_pairs;
        } else {
          std::vector<ggml_tensor*> ids;
          for (ggml_tensor* m : members) {
            xs.push_back(m->src[1]);
            ids.push_back(m->src[2]);
          }
          ggml_tensor* x = ConcatAll(c, xs, 2, made);
          ggml_tensor* joined = ConcatAll(c, ids, 1, made);
          if (kg::IsMoeGemvSwiglu(a)) {
            both = kg::MoeGemvSwiglu(c, a->src[0], x, joined, a->ne[0], a->src[3], a->src[4],
                                     static_cast<std::uint64_t>(kg::JitllmOpInt(a, 2)),
                                     static_cast<std::uint64_t>(kg::JitllmOpInt(a, 3)));
          } else {
            both = kg::MoeGemv(c, a->src[0], x, joined, a->ne[0], kg::JitllmOpInt(a, 0),
                               kg::JitllmOpInt(a, 1),
                               static_cast<std::uint64_t>(kg::JitllmOpInt(a, 2)),
                               static_cast<std::uint64_t>(kg::JitllmOpInt(a, 3)));
          }
          std::size_t offset = 0;
          for (ggml_tensor* m : members) {
            split.push_back(ggml_view_3d(c, both, m->ne[0], m->ne[1], m->ne[2], both->nb[1],
                                         both->nb[2], offset));
            offset += static_cast<std::size_t>(m->ne[2]) * both->nb[2];
          }
          out.stats_.packed_bytes += ggml_nbytes(x) + ggml_nbytes(joined);
          ++out.stats_.routed_pairs;
        }
        out.nodes_.insert(out.nodes_.end(), made.begin(), made.end());
        out.nodes_.push_back(both);
        out.nodes_.insert(out.nodes_.end(), split.begin(), split.end());
        out.products_.push_back({members, both});
        for (std::size_t k = 0; k < members.size(); ++k) {
          out.stats_.paired_slots |= std::uint32_t{1} << g[k];
          for (ggml_tensor*& held : out.keep_) {
            if (held == members[k]) {
              held = split[k];
            }
          }
          Redirect(lists, members[k], split[k]);
          if (out.target_[g[k]] != nullptr) {
            auto& graph = out.target_[g[k]]->graph;
            if (graph.logits == members[k]) {
              graph.logits = split[k];
            }
            for (auto& named : graph.named) {
              if (named.second == members[k]) {
                named.second = split[k];
              }
            }
          }
        }
      }
      if (out.nodes_.size() == emitted) {
        return Error("Qwen3.8 wave members wait at different product barriers");
      }
    }
    return Topological(out.nodes_);
  }

  static std::expected<void, std::string> Authenticate(const Originals& originals,
                                                       const Qwen38WavePlanned& out,
                                                       bool group_requested) {
    const auto grouped = [](const auto& plan) {
      return std::ranges::any_of(plan.steps, [](const auto& step) {
        return step.implementation == kg::kSetRowsGroupedName;
      });
    };
    const bool expand_stores =
        group_requested && (grouped(out.plan) || std::ranges::any_of(originals, grouped));
    const auto replaced = [&](const ggml_tensor* node) {
      return std::ranges::any_of(out.products_, [&](const Product& p) {
        return std::ranges::find(p.originals, node) != p.originals.end();
      });
    };
    for (const auto& plan : originals) {
      if (expand_stores) {
        std::vector<const ggml_tensor*> expected_stores;
        for (const auto& step : plan.steps)
          if (step.implementation == kg::kSetRowsName ||
              step.implementation == kg::kSetRowsGroupedName)
            expected_stores.insert(expected_stores.end(), step.nodes.begin(), step.nodes.end());
        std::vector<const ggml_tensor*> actual_stores;
        for (const auto& step : out.plan.steps)
          if (step.implementation == kg::kSetRowsName ||
              step.implementation == kg::kSetRowsGroupedName)
            for (const auto* node : step.nodes)
              if (std::ranges::find(expected_stores, node) != expected_stores.end())
                actual_stores.push_back(node);
        if (actual_stores != expected_stores)
          return Error("Qwen3.8 wave changed cache store order or count");
      }
      for (const auto& step : plan.steps) {
        if (step.nodes.size() == 1 && replaced(step.nodes.front())) {
          continue;
        }
        // Grouping may change only the packaging of these exact primitive
        // stores, including lane-boundary splits. All other fusions stay exact.
        if (expand_stores && (step.implementation == kg::kSetRowsName ||
                              step.implementation == kg::kSetRowsGroupedName)) {
          for (const auto* node : step.nodes) {
            const auto count = std::ranges::count_if(out.plan.steps, [&](const kg::PlanStep& s) {
              return (s.implementation == kg::kSetRowsName ||
                      s.implementation == kg::kSetRowsGroupedName) &&
                     std::ranges::count(s.nodes, node) == 1;
            });
            if (count != 1) return Error("Qwen3.8 wave changed a physical cache store");
          }
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
    for (const auto& product : out.products_) {
      const auto together = find(out.plan, product.together);
      if (product.originals.size() < 2 || together == out.plan.steps.end()) {
        return Error("Qwen3.8 wave changed a shared implementation or precision tier");
      }
      for (ggml_tensor* node : product.originals) {
        const auto* step = original(node);
        if (step == nullptr || step->operation != together->operation ||
            step->implementation != together->implementation) {
          return Error(std::format(
              "Qwen3.8 wave changed a shared implementation or precision tier ({} for {})",
              together->implementation, step == nullptr ? "?" : step->implementation));
        }
      }
    }
    for (const auto& product : out.head_products_) {
      for (ggml_tensor* node : product.originals) {
        const auto* step = original(node);
        if (step == nullptr || step->operation != execution::Operation::kMatMul ||
            step->implementation != kg::kMulMatTensorCore) {
          return Error("Qwen3.8 full target heads did not retain the ordinary MMF selector");
        }
      }
      if (!OrdinaryHead(out.plan, product.together)) {
        return Error("Qwen3.8 full target heads did not retain the ordinary MMF selector");
      }
    }
    return {};
  }

  static std::expected<void, std::string> Finish(
      Qwen38WavePlanned& out, const Lists& lists, const Originals& originals,
      std::span<const std::uint32_t> order, const Slots<bool>& compatible,
      std::span<const Range> mutable_places, const kg::DeviceChoices& choices,
      Qwen38WavePlacement placement, std::optional<ActivationMeasurement> measurement) {
    if (placement.activations != 0 &&
        placement.bytes > std::numeric_limits<std::uint64_t>::max() - placement.activations) {
      return Error("Qwen3.8 wave activation range overflows");
    }
    const auto groups = Groups(lists, order, compatible, mutable_places, placement.paired);
    auto heads = PrepareHeads(out, originals, groups, mutable_places, choices,
                              placement.paired && placement.share_target_head);
    if (!heads) {
      return std::unexpected(heads.error());
    }
    if (auto made = Compose(out, lists, groups, mutable_places, *heads, placement.lanes); !made) {
      return made;
    }
    if (auto made =
            PlaceAndPlan(out, out.nodes_, out.inputs_, out.keep_, choices, placement.activations,
                         placement.bytes, measurement, out.lanes_.empty() ? nullptr : &out.lanes_);
        !made) {
      return made;
    }
    for (const auto& step : out.plan.steps) {
      if (step.implementation == kg::kSetRowsGroupedName) {
        ++out.stats_.grouped_store_steps;
        out.stats_.grouped_stores += step.nodes.size();
      } else if (step.implementation == kg::kSetRowsName) {
        ++out.stats_.primitive_store_steps;
      }
    }
    return Authenticate(originals, out, choices.group_set_rows);
  }

  static std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> Target(
      std::span<const Qwen38TargetWaveInput> requests, const kg::DeviceChoices& choices,
      Qwen38WavePlacement placement, std::optional<ActivationMeasurement> measurement) {
    if (std::ranges::any_of(requests, [](const auto& r) { return r.kind.state_only; }))
      return Error("state-only prefill is scalar; target waves require headed requests");
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
      if (r.shape.token != requests.front().shape.token)
        return Error("Qwen3.8 target publication mode must be homogeneous across the wave");
      auto p = PlanQwen38Chunk(*r.model, r.shape, choices, 0, 0, {}, r.kind, measurement);
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
      out->active_slots_ |= std::uint32_t{1} << s;
      // Whether this slot may join the group of the slot before it.
      if (i != 0) {
        compatible[s] = r.kind == requests[i - 1].kind && r.kind.capture_routed == 0 &&
                        requests[i - 1].kind.capture_routed == 0;
      }
    }
    if (auto made = Finish(*out, lists, originals, order, compatible, *mutable_places, choices,
                           placement, measurement);
        !made) {
      return std::unexpected(made.error());
    }
    return out;
  }

  static std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> Draft(
      std::span<const Qwen38DraftWaveInput> requests, const kg::DeviceChoices& choices,
      Qwen38WavePlacement placement, std::optional<ActivationMeasurement> measurement) {
    kg::DeviceChoices draft_choices = choices;
    draft_choices.vector_float_node = [prior = choices.vector_float_node](const ggml_tensor* t) {
      return DraftHead(t) || (prior && prior(t));
    };
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
      auto p = PlanQwen38Mtp(*r.model, r.shape, draft_choices, 0, 0, measurement);
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
      out->active_slots_ |= std::uint32_t{1} << s;
      // Whether this slot may join the group of the slot before it.
      if (i != 0) {
        compatible[s] = SameDraftPhase(requests[i - 1].shape, r.shape);
      }
    }
    if (auto made = Finish(*out, lists, originals, order, compatible, *mutable_places,
                           draft_choices, placement, measurement);
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

std::uint64_t Qwen38WavePlanned::host_bytes() const {
  std::uint64_t bytes = PlannedHostBytes(*this) + (head_arena_ ? head_arena_->bytes() : 0) +
                        ((nodes_.size() + inputs_.size() + keep_.size()) * sizeof(ggml_tensor*));
  for (const auto& planned : target_) {
    bytes += planned != nullptr ? PlannedHostBytes(*planned) : 0;
  }
  for (const auto& planned : draft_) {
    bytes += planned != nullptr ? PlannedHostBytes(*planned) : 0;
  }
  return bytes;
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
  return PlanQwen38TargetWave(requests, choices, placement, std::nullopt);
}

std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38TargetWave(
    std::span<const Qwen38TargetWaveInput> requests, const kg::DeviceChoices& choices,
    Qwen38WavePlacement placement, std::optional<ActivationMeasurement> measurement) {
  if (measurement && placement.activations != 0)
    return Error("measurement-only wave planning cannot use activation storage");
  return Qwen38WaveBuilder::Target(requests, choices, placement, measurement);
}

std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38DraftWave(
    std::span<const Qwen38DraftWaveInput> requests, const kg::DeviceChoices& choices,
    Qwen38WavePlacement placement) {
  return PlanQwen38DraftWave(requests, choices, placement, std::nullopt);
}

std::expected<std::unique_ptr<Qwen38WavePlanned>, std::string> PlanQwen38DraftWave(
    std::span<const Qwen38DraftWaveInput> requests, const kg::DeviceChoices& choices,
    Qwen38WavePlacement placement, std::optional<ActivationMeasurement> measurement) {
  if (measurement && placement.activations != 0)
    return Error("measurement-only wave planning cannot use activation storage");
  return Qwen38WaveBuilder::Draft(requests, choices, placement, measurement);
}

}  // namespace jitllm::engine
