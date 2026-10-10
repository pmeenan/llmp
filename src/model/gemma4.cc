// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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
#include "model/v0_artifact.h"

namespace llmp::model {
namespace {
std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}
bool ProfileValid(const Gemma4Profile& p) { return p == Gemma4_26BA4B() || p == Gemma4_31B(); }
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
  if (auto v0 = RequireV0Artifact(artifact, "Gemma4"); !v0) {
    return std::unexpected(v0.error());
  }
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

}  // namespace llmp::model
