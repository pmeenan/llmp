// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma4.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/layout.h"
#include "artifact/representation.h"
#include "base/check.h"
#include "model/host_mask.h"

namespace jitllm::model {
namespace {
constexpr std::uint64_t kExtent = 2U << 20U;
constexpr std::uint16_t kZero = 0, kNegInf = 0xFC00;
std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}
std::uint64_t Pad(std::uint64_t bytes, std::uint64_t align) {
  return ((bytes + align - 1) / align) * align;
}
bool ProfileValid(const Gemma4Profile& p) { return p == Gemma4_26BA4B() || p == Gemma4_31B(); }
bool AlignValid(std::uint32_t align) { return align >= 256 && align <= 8192 && align % 256 == 0; }
std::uint32_t ReadCells(std::uint32_t positions, std::uint32_t cells, std::uint32_t align) {
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(cells, Pad(positions, align)));
}
// Public layouts are checked too: an edited offset/width must never produce
// out-of-region input indices or understate materialization ranges.
bool LayoutValid(const Gemma4Profile& p, const Gemma4StateLayout& s) {
  if (!ProfileValid(p) || s.context == 0 || s.context > p.context || s.max_rows == 0 ||
      s.max_rows > std::min(s.context, kGemma4MaxRows) || s.global_cells != Pad(s.context, 256) ||
      s.local_cells != Pad(std::min<std::uint64_t>(s.context, p.window + s.max_rows), 256) ||
      s.tensors.size() != std::size_t{p.layers} * 2) {
    return false;
  }
  std::uint64_t at = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto& t = s.tensors[std::size_t{il} * 2 + (value ? 1U : 0U)];
      const auto cells = p.local(il) ? s.local_cells : s.global_cells;
      const auto width = p.head_dim(il) * p.kv_heads(il);
      const auto bytes = std::uint64_t{width} * cells * 2;
      if (t.layer != il || t.value != value || t.local != p.local(il) || t.width != width ||
          t.cells != cells || t.offset != at || t.bytes != bytes) {
        return false;
      }
      at += Pad(bytes, kExtent);
    }
  }
  return s.bytes == at;
}
std::expected<std::uint64_t, std::string> InputBytes(const Gemma4Profile& p,
                                                     const Gemma4StateLayout& s,
                                                     std::span<const Gemma4Segment> segments,
                                                     bool masks, std::uint32_t align) {
  if (!LayoutValid(p, s) || !AlignValid(align) || segments.empty() ||
      segments.size() > kGemma4MaxSlots) {
    return Refused("invalid Gemma4 layout, alignment or segment count");
  }
  std::array<bool, kGemma4MaxSlots> seen{};
  std::uint64_t bytes = segments.size() * sizeof(Gemma4SegmentInputs), rows = 0;
  for (const auto& seg : segments) {
    if (seg.slot >= kGemma4MaxSlots || seen[seg.slot] || seg.tokens.empty() ||
        seg.n_past > s.context || seg.tokens.size() > s.context - seg.n_past ||
        seg.tokens.size() > s.max_rows - rows) {
      return Refused("empty, repeated or out-of-bounds Gemma4 segment");
    }
    seen[seg.slot] = true;
    for (const auto token : seg.tokens) {
      if (token < 0 || std::cmp_greater_equal(token, p.vocab)) {
        return Refused("Gemma4 token is outside the vocabulary");
      }
    }
    rows += seg.tokens.size();
    // Token, position, output ID and the global/local I64 cell indices.
    bytes += seg.tokens.size() * (3 * sizeof(std::int32_t) + 2 * sizeof(std::int64_t));
    if (masks) {
      const auto end = seg.n_past + static_cast<std::uint32_t>(seg.tokens.size());
      const auto global =
          std::uint64_t{ReadCells(end, s.global_cells, align)} * seg.tokens.size() * 2;
      const auto local =
          std::uint64_t{ReadCells(end, s.local_cells, align)} * seg.tokens.size() * 2;
      if (global > std::numeric_limits<std::int32_t>::max() ||
          local > std::numeric_limits<std::int32_t>::max()) {
        return Refused("Gemma4 reference mask exceeds the GGML I32 byte stride");
      }
      bytes += global + local;
    }
  }
  return bytes;
}
}  // namespace

const Gemma4Profile& Gemma4_26BA4B() {
  static constexpr Gemma4Profile p{.name = "gemma-4-26B-A4B-it",
                                   .layers = 30,
                                   .width = 2816,
                                   .heads = 16,
                                   .local_kv_heads = 8,
                                   .global_kv_heads = 2,
                                   .local_head_dim = 256,
                                   .global_head_dim = 512,
                                   .local_rope_dims = 256,
                                   .global_rope_dims = 512,
                                   .ffn = 2112,
                                   .experts = 128,
                                   .experts_used = 8,
                                   .expert_ffn = 704,
                                   .vocab = 262144,
                                   .context = kGemma4Context,
                                   .window = 1024,
                                   .local_rope_base = 10000,
                                   .global_rope_base = 1000000,
                                   .rms_eps = 1e-6f,
                                   .final_softcap = 30};
  return p;
}
const Gemma4Profile& Gemma4_31B() {
  static constexpr Gemma4Profile p{.name = "gemma-4-31B-it",
                                   .layers = 60,
                                   .width = 5376,
                                   .heads = 32,
                                   .local_kv_heads = 16,
                                   .global_kv_heads = 4,
                                   .local_head_dim = 256,
                                   .global_head_dim = 512,
                                   .local_rope_dims = 256,
                                   .global_rope_dims = 512,
                                   .ffn = 21504,
                                   .experts = 0,
                                   .experts_used = 0,
                                   .expert_ffn = 0,
                                   .vocab = 262144,
                                   .context = kGemma4Context,
                                   .window = 1024,
                                   .local_rope_base = 10000,
                                   .global_rope_base = 1000000,
                                   .rms_eps = 1e-6f,
                                   .final_softcap = 30};
  return p;
}

std::expected<Gemma4Binding, std::string> BindGemma4(const Gemma4Profile& p,
                                                     std::string_view architecture,
                                                     std::span<const Gemma4Resource> resources) {
  if (!ProfileValid(p) || architecture != "gemma4") {
    return Refused("not an approved Gemma4 text profile/architecture");
  }
  std::unordered_map<std::string_view, std::size_t> roles;
  std::vector<std::uint32_t> indices;
  std::uint32_t normal = 0, expert = 0;
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto& r = resources[i];
    indices.push_back(r.expert_array ? expert++ : normal++);
    if (r.roles.empty()) {
      return Refused("Gemma4 resource has no role");
    }
    for (const auto& role : r.roles) {
      if (!roles.emplace(role, i).second) {
        return Refused(std::format("duplicate Gemma4 role {}", role));
      }
    }
  }
  std::vector<std::string_view> used;
  const auto want = [&](std::string_view name, std::vector<std::uint64_t> ne, bool array = false,
                        bool matrix = false) -> std::expected<Gemma4Tensor, std::string> {
    const auto it = roles.find(name);
    if (it == roles.end()) {
      return Refused(std::format("missing Gemma4 tensor {}", name));
    }
    const auto& r = resources[it->second];
    const auto* type = artifact::FindGgmlType(r.type);
    const bool weight_type =
        type != nullptr && (r.type == "F32" || r.type == "F16" || r.type == "BF16" ||
                            (type->block_elements > 1 && r.type != "Q8_1" && r.type != "Q8_K"));
    if (r.ne != ne || r.expert_array != array || (array ? r.count != p.experts : r.count != 0) ||
        !type || (matrix ? !weight_type : r.type != "F32") || ne[0] % type->block_elements != 0) {
      return Refused(std::format("Gemma4 tensor {} has wrong shape, type or expert count", name));
    }
    std::uint64_t bytes = (ne[0] / type->block_elements) * type->block_bytes;
    for (std::size_t i = 1; i < ne.size(); ++i) {
      bytes *= ne[i];  // every wanted shape is bounded by the fixed profile
    }
    // Match the prepared-artifact GGML contract: quantized tensors whose
    // row width is not a multiple of 512 need the backend's tail over-read.
    if (type->block_elements > 1 && ne[0] % artifact::kGgmlRowPadding != 0) {
      bytes += (artifact::kGgmlRowPadding - ne[0] % artifact::kGgmlRowPadding) /
               type->block_elements * type->block_bytes;
    }
    if (r.readable < bytes || r.group_offset > UINT64_MAX - r.readable ||
        r.group_offset % artifact::kMemberAlignment != 0 || (!array && r.group_offset != 0)) {
      return Refused(std::format("Gemma4 tensor {} has insufficient or malformed storage", name));
    }
    used.push_back(it->first);
    return Gemma4Tensor{.index = indices[it->second],
                        .type = r.type,
                        .ne = r.ne,
                        .expert_array = array,
                        .group_offset = r.group_offset,
                        .readable = r.readable};
  };
  Gemma4Binding b;
  const auto put = [&](Gemma4Tensor& out, std::string name, std::vector<std::uint64_t> ne,
                       bool array = false,
                       bool matrix = false) -> std::expected<void, std::string> {
    auto t = want(name, std::move(ne), array, matrix);
    if (!t) {
      return std::unexpected(t.error());
    }
    out = std::move(*t);
    return {};
  };
  const auto optional = [&](std::optional<Gemma4Tensor>& out, std::string name,
                            std::vector<std::uint64_t> ne, bool array = false,
                            bool matrix = false) -> std::expected<void, std::string> {
    auto t = want(name, std::move(ne), array, matrix);
    if (!t) {
      return std::unexpected(t.error());
    }
    out = std::move(*t);
    return {};
  };
  if (auto r = put(b.token_embd, "token_embd.weight", {p.width, p.vocab}, false, true); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = put(b.output_norm, "output_norm.weight", {p.width}); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = put(b.rope_freqs, "rope_freqs.weight", {p.global_head_dim / 2}); !r) {
    return std::unexpected(r.error());
  }
  b.output = b.token_embd;
  if (roles.contains("output.weight")) {
    auto t = want("output.weight", {p.width, p.vocab}, false, true);
    if (!t) {
      return std::unexpected(t.error());
    }
    if (t->index != b.token_embd.index || t->expert_array) {
      return Refused("Gemma4 output head must alias its tied token embedding");
    }
  }
  b.layers.resize(p.layers);
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    auto& l = b.layers[il];
    const auto prefix = std::format("blk.{}.", il);
    const auto d = p.head_dim(il), kv = p.kv_heads(il) * d, q = p.heads * d;
    const auto add = [&](Gemma4Tensor& out, std::string_view suffix, std::vector<std::uint64_t> ne,
                         bool matrix = false) {
      return put(out, prefix + std::string(suffix), std::move(ne), false, matrix);
    };
    for (auto [out, suffix, ne, matrix] :
         std::vector<std::tuple<Gemma4Tensor*, std::string_view, std::vector<std::uint64_t>, bool>>{
             {&l.attn_norm, "attn_norm.weight", {p.width}, false},
             {&l.q, "attn_q.weight", {p.width, q}, true},
             {&l.k, "attn_k.weight", {p.width, kv}, true},
             {&l.out, "attn_output.weight", {q, p.width}, true},
             {&l.q_norm, "attn_q_norm.weight", {d}, false},
             {&l.k_norm, "attn_k_norm.weight", {d}, false},
             {&l.attn_post_norm, "post_attention_norm.weight", {p.width}, false},
             {&l.ffn_norm, "ffn_norm.weight", {p.width}, false},
             {&l.gate, "ffn_gate.weight", {p.width, p.ffn}, true},
             {&l.up, "ffn_up.weight", {p.width, p.ffn}, true},
             {&l.down, "ffn_down.weight", {p.ffn, p.width}, true},
             {&l.ffn_post_norm, "post_ffw_norm.weight", {p.width}, false}}) {
      if (auto r = add(*out, suffix, std::move(ne), matrix); !r) {
        return std::unexpected(r.error());
      }
    }
    l.tied_kv = !p.local(il);
    if (l.tied_kv) {
      l.v = l.k;
    } else if (auto r = add(l.v, "attn_v.weight", {p.width, kv}, true); !r) {
      return std::unexpected(r.error());
    }
    if (auto r = optional(l.output_scale, prefix + "layer_output_scale.weight", {1}); !r) {
      return std::unexpected(r.error());
    }
    if (p.experts != 0) {
      for (auto [out, suffix, ne, matrix] :
           std::vector<std::tuple<std::optional<Gemma4Tensor>*, std::string_view,
                                  std::vector<std::uint64_t>, bool>>{
               {&l.router, "ffn_gate_inp.weight", {p.width, p.experts}, true},
               {&l.router_scale, "ffn_gate_inp.scale", {p.width}, false},
               {&l.ffn_pre_norm_2, "pre_ffw_norm_2.weight", {p.width}, false},
               {&l.ffn_post_norm_1, "post_ffw_norm_1.weight", {p.width}, false},
               {&l.ffn_post_norm_2, "post_ffw_norm_2.weight", {p.width}, false},
               {&l.expert_scale, "ffn_down_exps.scale", {p.experts}, false}}) {
        if (auto r = optional(*out, prefix + std::string(suffix), std::move(ne), false, matrix);
            !r) {
          return std::unexpected(r.error());
        }
      }
      if (roles.contains(prefix + "ffn_gate_up_exps.weight")) {
        if (auto r = optional(l.gate_up_exps, prefix + "ffn_gate_up_exps.weight",
                              {p.width, 2U * p.expert_ffn}, true, true);
            !r) {
          return std::unexpected(r.error());
        }
      } else {
        if (auto r = optional(l.gate_exps, prefix + "ffn_gate_exps.weight", {p.width, p.expert_ffn},
                              true, true);
            !r) {
          return std::unexpected(r.error());
        }
        if (auto r = optional(l.up_exps, prefix + "ffn_up_exps.weight", {p.width, p.expert_ffn},
                              true, true);
            !r) {
          return std::unexpected(r.error());
        }
      }
      if (auto r = optional(l.down_exps, prefix + "ffn_down_exps.weight", {p.expert_ffn, p.width},
                            true, true);
          !r) {
        return std::unexpected(r.error());
      }
    }
  }
  for (const auto& [role, index] : roles) {
    if (!std::ranges::contains(used, role)) {
      return Refused(std::format("Gemma4 does not read role {}", role));
    }
    std::ignore = index;
  }
  return b;
}

std::expected<Gemma4Binding, std::string> BindGemma4(const Gemma4Profile& p,
                                                     const artifact::Artifact& artifact) {
  if (artifact.model().expert_count != p.experts) {
    return Refused("Gemma4 artifact expert count differs from its profile");
  }
  std::vector<Gemma4Resource> resources;
  for (const auto& r : artifact.resources()) {
    if (r.repr.family != artifact::Family::kGgml) {
      return Refused("Gemma4 foundation requires GGML representations");
    }
    resources.push_back({.roles = r.roles,
                         .type = std::string(r.repr.type),
                         .ne = r.repr.dims,
                         .readable = r.readable.value()});
  }
  for (const auto& a : artifact.expert_arrays()) {
    if (a.repr.family != artifact::Family::kGgml) {
      return Refused("Gemma4 experts require GGML representations");
    }
    resources.push_back({.roles = {a.name},
                         .type = std::string(a.repr.type),
                         .ne = a.repr.dims,
                         .expert_array = true,
                         .count = a.count,
                         .group_offset = a.group_offset.value(),
                         .readable = a.readable.value()});
  }
  return BindGemma4(p, artifact.model().architecture, resources);
}

std::expected<void, std::string> CheckGemma4Binding(const Gemma4Profile& p,
                                                    const Gemma4Binding& b) {
  if (!ProfileValid(p) || b.layers.size() != p.layers) {
    return Refused("not a complete approved Gemma4 binding");
  }
  const auto same = [](const Gemma4Tensor& a, const Gemma4Tensor& z) {
    return a.index == z.index && a.type == z.type && a.ne == z.ne &&
           a.expert_array == z.expert_array && a.group_offset == z.group_offset &&
           a.readable == z.readable;
  };
  if (!same(b.token_embd, b.output)) {
    return Refused("Gemma4 head no longer aliases its embedding");
  }
  // Reconstruct roles from public descriptors and reuse artifact admission's
  // checked shape/type/readability contract. Shared identities must agree;
  // resource numbering is retained for the execution caller, not redefined.
  std::vector<Gemma4Resource> resources;
  std::vector<const Gemma4Tensor*> identities;
  bool consistent = true;
  const auto add = [&](const Gemma4Tensor& t, std::string role) {
    if (!consistent || t.ne.empty() || t.ne.size() > 2 ||
        artifact::FindGgmlType(t.type) == nullptr) {
      consistent = false;
      return;  // Do not duplicate unbounded public rank/type storage before refusal.
    }
    const auto at = std::ranges::find_if(identities, [&](const Gemma4Tensor* old) {
      return old->expert_array == t.expert_array && old->index == t.index;
    });
    if (at != identities.end()) {
      consistent = consistent && same(**at, t);
      resources[static_cast<std::size_t>(at - identities.begin())].roles.push_back(std::move(role));
    } else {
      identities.push_back(&t);
      resources.push_back({.roles = {std::move(role)},
                           .type = t.type,
                           .ne = t.ne,
                           .expert_array = t.expert_array,
                           .count = t.expert_array ? p.experts : 0,
                           .group_offset = t.group_offset,
                           .readable = t.readable});
    }
  };
  add(b.token_embd, "token_embd.weight");
  add(b.output_norm, "output_norm.weight");
  add(b.rope_freqs, "rope_freqs.weight");
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const auto& l = b.layers[il];
    if (l.tied_kv != !p.local(il) || (l.tied_kv && !same(l.k, l.v))) {
      return Refused("Gemma4 global K-as-V identity differs from its profile");
    }
    const auto prefix = std::format("blk.{}.", il);
    for (const auto& [t, name] : std::vector<std::pair<const Gemma4Tensor*, std::string_view>>{
             {&l.attn_norm, "attn_norm.weight"},
             {&l.q, "attn_q.weight"},
             {&l.k, "attn_k.weight"},
             {&l.out, "attn_output.weight"},
             {&l.q_norm, "attn_q_norm.weight"},
             {&l.k_norm, "attn_k_norm.weight"},
             {&l.attn_post_norm, "post_attention_norm.weight"},
             {&l.ffn_norm, "ffn_norm.weight"},
             {&l.gate, "ffn_gate.weight"},
             {&l.up, "ffn_up.weight"},
             {&l.down, "ffn_down.weight"},
             {&l.ffn_post_norm, "post_ffw_norm.weight"}}) {
      add(*t, prefix + std::string(name));
    }
    if (!l.tied_kv) add(l.v, prefix + "attn_v.weight");
    for (const auto& [t, name] :
         std::vector<std::pair<const std::optional<Gemma4Tensor>*, std::string_view>>{
             {&l.output_scale, "layer_output_scale.weight"},
             {&l.router, "ffn_gate_inp.weight"},
             {&l.router_scale, "ffn_gate_inp.scale"},
             {&l.ffn_pre_norm_2, "pre_ffw_norm_2.weight"},
             {&l.ffn_post_norm_1, "post_ffw_norm_1.weight"},
             {&l.ffn_post_norm_2, "post_ffw_norm_2.weight"},
             {&l.gate_up_exps, "ffn_gate_up_exps.weight"},
             {&l.gate_exps, "ffn_gate_exps.weight"},
             {&l.up_exps, "ffn_up_exps.weight"},
             {&l.down_exps, "ffn_down_exps.weight"},
             {&l.expert_scale, "ffn_down_exps.scale"}}) {
      if (t->has_value()) add(**t, prefix + std::string(name));
    }
  }
  if (!consistent) return Refused("Gemma4 shared resource descriptors disagree");
  const auto arrays = static_cast<std::uint32_t>(
      std::ranges::count_if(identities, [](const Gemma4Tensor* t) { return t->expert_array; }));
  const auto ordinary = static_cast<std::uint32_t>(identities.size()) - arrays;
  for (const auto* t : identities) {
    if (t->index >= (t->expert_array ? arrays : ordinary)) {
      return Refused("Gemma4 binding resource identities have gaps or exceed their domain");
    }
  }
  auto checked = BindGemma4(p, "gemma4", resources);
  if (!checked) return std::unexpected(checked.error());
  return {};
}

std::expected<void, std::string> CheckGemma4RopeFactors(const Gemma4Profile& p,
                                                        std::span<const float> factors) {
  if (!ProfileValid(p) || factors.size() != p.global_head_dim / 2 ||
      std::ranges::any_of(factors, [](float f) { return !std::isfinite(f) || f <= 0; })) {
    return Refused("Gemma4 global RoPE needs finite positive F32 factors for its complete head");
  }
  return {};
}

std::expected<Gemma4StateLayout, std::string> Gemma4State(const Gemma4Profile& p,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows) {
  if (!ProfileValid(p) || context == 0 || context > p.context || max_rows == 0 ||
      max_rows > std::min(context, kGemma4MaxRows)) {
    return Refused("Gemma4 context or chunk bound is invalid");
  }
  Gemma4StateLayout s;
  s.context = context;
  s.max_rows = max_rows;
  s.global_cells = static_cast<std::uint32_t>(Pad(context, 256));
  s.local_cells =
      static_cast<std::uint32_t>(Pad(std::min<std::uint64_t>(context, p.window + max_rows), 256));
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto width = p.head_dim(il) * p.kv_heads(il);
      const auto cells = p.local(il) ? s.local_cells : s.global_cells;
      const auto bytes = std::uint64_t{width} * cells * 2;
      s.tensors.push_back({.layer = il,
                           .value = value,
                           .local = p.local(il),
                           .width = width,
                           .cells = cells,
                           .offset = s.bytes,
                           .bytes = bytes});
      s.bytes += Pad(bytes, kExtent);
    }
  }
  return s;
}
std::expected<std::vector<StateRepresentation>, std::string> Gemma4StateLayout::Representations(
    const Gemma4Profile& p) const {
  if (!LayoutValid(p, *this)) {
    return Refused("Gemma4 representation layout is invalid");
  }
  std::vector<StateRepresentation> out;
  for (const auto& t : tensors) {
    out.push_back({.name = std::format("gemma4.{}.{}", t.layer, t.value ? "v" : "k"),
                   .block_positions = 0,
                   .block_bytes = base::Bytes{Pad(t.bytes, kExtent)},
                   .capabilities = t.local ? static_cast<std::uint8_t>(StateCapability::kAppend)
                                           : StateCapability::kAppend | StateCapability::kTruncate,
                   .max_snapshots = 0,
                   .snapshot_bytes = base::Bytes{0}});
  }
  return out;
}
std::expected<std::vector<StateRange>, std::string> Gemma4UsedState(const Gemma4Profile& p,
                                                                    const Gemma4StateLayout& s,
                                                                    std::uint32_t positions,
                                                                    std::uint32_t align) {
  if (!LayoutValid(p, s) || !AlignValid(align) || positions > s.context) {
    return Refused("Gemma4 initialized footprint is out of bounds");
  }
  std::vector<StateRange> out;
  out.reserve(s.tensors.size());
  if (positions != 0) {
    for (const auto& t : s.tensors) {
      out.push_back({.offset = t.offset,
                     .bytes = std::uint64_t{ReadCells(positions, t.cells, align)} * t.width * 2});
    }
  }
  return out;
}
std::expected<std::uint64_t, std::string> Gemma4HostInputBytes(
    const Gemma4Profile& p, const Gemma4StateLayout& s, std::span<const Gemma4Segment> segments,
    bool masks, std::uint32_t align) {
  return InputBytes(p, s, segments, masks, align);
}
std::expected<Gemma4ChunkInputs, std::string> Gemma4Chunk(const Gemma4Profile& p,
                                                          const Gemma4StateLayout& s,
                                                          std::span<const Gemma4Segment> segments,
                                                          bool masks, std::uint32_t align) {
  const auto bytes = InputBytes(p, s, segments, masks, align);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  Gemma4ChunkInputs in;
  std::size_t total_rows = 0;
  for (const auto& seg : segments) {
    total_rows += seg.tokens.size();
  }
  in.tokens.reserve(total_rows);
  in.positions.reserve(total_rows);
  in.out_ids.reserve(total_rows);
  in.segments.reserve(segments.size());
  for (const auto& seg : segments) {
    Gemma4SegmentInputs x;
    x.slot = seg.slot;
    x.n_past = seg.n_past;
    x.first_row = static_cast<std::uint32_t>(in.tokens.size());
    x.rows = static_cast<std::uint32_t>(seg.tokens.size());
    const auto end = x.n_past + x.rows;
    x.global_n_kv = ReadCells(end, s.global_cells, align);
    x.local_n_kv = ReadCells(end, s.local_cells, align);
    x.global_cells.resize(x.rows);
    x.local_cells.resize(x.rows);
    if (masks) {
      x.global_mask.assign(std::size_t{x.rows} * x.global_n_kv, kNegInf);
      x.local_mask.assign(std::size_t{x.rows} * x.local_n_kv, kNegInf);
    }
    for (std::uint32_t i = 0; i < x.rows; ++i) {
      const auto pos = x.n_past + i;
      in.tokens.push_back(seg.tokens[i]);
      in.positions.push_back(static_cast<std::int32_t>(pos));
      in.out_ids.push_back(static_cast<std::int32_t>(in.out_ids.size()));
      x.global_cells[i] = pos;
      x.local_cells[i] = pos % s.local_cells;
      if (masks) {
        const auto global = HostMaskPrefix(x.global_n_kv, std::uint64_t{pos} + 1);
        const auto local = HostMaskRing(x.local_n_kv, pos, end, s.local_cells, p.window);
        if (!global || !local) return Refused("invalid host causal-mask interval");
        FillHostMaskVisible(
            std::span{x.global_mask}.subspan(std::size_t{i} * x.global_n_kv, x.global_n_kv),
            *global, kZero);
        FillHostMaskVisible(
            std::span{x.local_mask}.subspan(std::size_t{i} * x.local_n_kv, x.local_n_kv), *local,
            kZero);
      }
    }
    in.segments.push_back(std::move(x));
  }
  std::uint64_t allocated =
      (in.tokens.capacity() + in.positions.capacity() + in.out_ids.capacity()) *
          sizeof(std::int32_t) +
      in.segments.capacity() * sizeof(Gemma4SegmentInputs);
  for (const auto& seg : in.segments) {
    allocated += (seg.global_cells.capacity() + seg.local_cells.capacity()) * sizeof(std::int64_t);
    allocated += (seg.global_mask.capacity() + seg.local_mask.capacity()) * sizeof(std::uint16_t);
  }
  base::Check(allocated == *bytes, "Gemma4 inputs exceeded their preflight storage envelope");
  return in;
}
std::expected<std::vector<StateRange>, std::string> Gemma4ChunkWrites(const Gemma4Profile& p,
                                                                      const Gemma4StateLayout& s,
                                                                      std::uint32_t past,
                                                                      std::uint32_t rows) {
  if (!LayoutValid(p, s) || rows == 0 || rows > s.max_rows || past > s.context ||
      rows > s.context - past) {
    return Refused("Gemma4 write footprint is out of bounds");
  }
  std::vector<StateRange> out;
  for (const auto& t : s.tensors) {
    const auto first = t.local ? past % t.cells : past;
    const auto count = std::min(rows, t.cells - first);
    const auto row_bytes = std::uint64_t{t.width} * 2;
    out.push_back({.offset = t.offset + first * row_bytes, .bytes = count * row_bytes});
    if (count < rows) {
      out.push_back({.offset = t.offset, .bytes = (rows - count) * row_bytes});
    }
  }
  return out;
}
}  // namespace jitllm::model
