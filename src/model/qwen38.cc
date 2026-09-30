// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Qwen38Chunk's QSA block tables and n-gram hash port llama.cpp b29c606e2's
// llama_memory_hybrid_idx::set_input_qsa and llm_graph_input_ple::set_input
// (MIT) for one sequence, so the file carries GGML's notice
// (docs/licensing.md).

#include "model/qwen38.h"

#include <algorithm>
#include <bit>
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
#include "artifact/representation.h"
#include "model/state.h"

namespace jitllm::model {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

std::uint64_t Pad(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

// The bytes of an element of a chunk's widest [n_kv, rows] tensor (F32).
constexpr std::uint64_t kMaskBytes = 4;

std::string Shape(std::span<const std::uint64_t> ne) {
  std::string out = "[";
  for (std::size_t i = 0; i < ne.size(); ++i) {
    out += std::format("{}{}", i == 0 ? "" : ", ", ne[i]);
  }
  return out + "]";
}

bool ProfileIsSane(const Qwen38Profile& p) {
  return p.layers > 0 && p.layers % 4 == 0 && p.width > 0 && p.width % 64 == 0 && p.heads > 0 &&
         p.kv_heads > 0 && p.heads % p.kv_heads == 0 && p.head_dim > 0 && p.rope_dims > 0 &&
         p.rope_dims <= p.head_dim && p.lin_k_heads > 0 && p.lin_v_heads % p.lin_k_heads == 0 &&
         p.lin_head_dim > 0 && p.conv > 1 && p.experts > 0 && p.experts_used > 0 &&
         p.experts_used <= p.experts && p.expert_ffn % 64 == 0 && p.shared_ffn % 32 == 0 &&
         p.hc > 1 && p.hc_rank > 0 && p.indexer_heads > 0 && p.indexer_head_dim > 0 &&
         p.indexer_ratio > 0 && p.indexer_ratio <= 64 && p.indexer_budget > 0 &&
         p.ple_layer < p.layers && p.linear(p.ple_layer) && p.ngram >= 2 && p.ngram <= 8 &&
         p.heads_per_ngram > 0 && p.ple_row % 16 == 0 && p.ple_row <= 1024 && p.ple_conv > 1 &&
         p.vocab > 0 && p.ple_eos >= 0 && std::cmp_less(p.ple_eos, p.vocab) && p.ple_row > 0 &&
         p.expert_ffn > 0 && p.shared_ffn > 0 && p.lin_v_heads > 0 && p.rms_eps > 0.0f &&
         p.rope_base > 0.0f && p.indexer_budget % p.indexer_ratio == 0 &&
         // The uint32 widths (hc_width, conv_channels, ple_width) cannot wrap.
         p.width <= (1U << 16) && p.hc <= 16 && p.lin_k_heads <= 256 && p.lin_v_heads <= 256 &&
         p.lin_head_dim <= 1024 && p.heads_per_ngram <= 64;
}

// What the binding expects of one role.
struct Want {
  std::string role;
  bool plain = false;
  std::string_view type;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  Qwen38Tensor* into = nullptr;
};

// The wants' builders.
class Wants {
 public:
  explicit Wants(std::vector<Want>& want) : want_(want) {}
  void Ggml(std::string role, std::string_view type, std::vector<std::uint64_t> ne,
            Qwen38Tensor* into) {
    want_.push_back(
        {.role = std::move(role), .plain = false, .type = type, .ne = std::move(ne), .into = into});
  }
  void Plain(std::string role, std::string_view type, std::vector<std::uint64_t> ne,
             Qwen38Tensor* into) {
    want_.push_back(
        {.role = std::move(role), .plain = true, .type = type, .ne = std::move(ne), .into = into});
  }
  // A linear y[n] = W x[k]: MXFP8 codes and scales, or with `bf16` the
  // drafter's GGML BF16 matrix.
  void Linear(const std::string& name, std::uint64_t k, std::uint64_t n, Qwen38Mxfp8* into,
              bool bf16) {
    if (bf16) {
      Ggml(name + ".weight", "BF16", {k, n}, &into->bf16);
      return;
    }
    Plain(name + ".weight", "F8_E4M3", {k, n}, &into->codes);
    Plain(name + ".weight_scale", "U8", {k / 32, n}, &into->scales);
  }
  void Array(std::string role, std::string_view type, std::vector<std::uint64_t> ne,
             Qwen38Tensor* into) {
    want_.push_back({.role = std::move(role),
                     .plain = false,
                     .type = type,
                     .ne = std::move(ne),
                     .expert_array = true,
                     .into = into});
  }

 private:
  std::vector<Want>& want_;
};

std::uint64_t Atoms(std::uint64_t rows, std::uint64_t k) { return Pad(rows, 128) / 128 * (k / 64); }

// One layer's wants under `n` ("blk.<i>."): the target's layer `il` (MXFP8
// linears, its kind by the profile, the n-gram layer's extras), or with
// `mtp` the drafter's full-attention layer (BF16 linears). The routed
// experts in GGML's layout, or the CUTLASS layout's four arrays.
void AddLayer(Wants& w, const Qwen38Profile& p, const std::string& n, std::uint32_t il,
              Qwen38Layer& l, bool cutlass, bool mtp) {
  const std::uint64_t width = p.width;
  const std::uint64_t hd = p.head_dim;
  l.linear = !mtp && p.linear(il);
  for (const auto& [kind, norm, down, up, inject] :
       {std::tuple{"attn", &l.hc_attn_norm, &l.hc_attn_down, &l.hc_attn_up, &l.hc_attn_inject},
        std::tuple{"ffn", &l.hc_ffn_norm, &l.hc_ffn_down, &l.hc_ffn_up, &l.hc_ffn_inject}}) {
    w.Ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {p.hc_width()}, norm);
    w.Ggml(std::format("{}hc_{}_down.weight", n, kind), "BF16", {p.hc_width(), p.hc_rank}, down);
    w.Ggml(std::format("{}hc_{}_up.weight", n, kind), "BF16", {p.hc_rank, p.hc_width()}, up);
    w.Ggml(std::format("{}hc_{}_inject.weight", n, kind), "BF16", {p.hc_width(), p.hc}, inject);
  }
  if (l.linear) {
    w.Linear(n + "attn_qkv", width, p.conv_channels(), &l.qkv, false);
    w.Linear(n + "attn_gate", width, p.lin_v_width(), &l.z, false);
    w.Linear(n + "ssm_beta", width, p.lin_v_heads, &l.beta, false);
    w.Linear(n + "ssm_alpha", width, p.lin_v_heads, &l.alpha, false);
    w.Ggml(n + "ssm_dt.bias", "F32", {p.lin_v_heads}, &l.dt_bias);
    w.Ggml(n + "ssm_a", "F32", {p.lin_v_heads}, &l.ssm_a);
    w.Ggml(n + "ssm_conv1d.weight", "F32", {p.conv, p.conv_channels()}, &l.conv1d);
    w.Ggml(n + "ssm_norm.weight", "F32", {p.lin_head_dim}, &l.ssm_norm);
    w.Linear(n + "ssm_out", p.lin_v_width(), width, &l.ssm_out, false);
  } else {
    w.Linear(n + "attn_q", width, 2 * hd * p.heads, &l.q, mtp);
    w.Linear(n + "attn_k", width, hd * p.kv_heads, &l.k, mtp);
    w.Linear(n + "attn_v", width, hd * p.kv_heads, &l.v, mtp);
    w.Linear(n + "attn_output", hd * p.heads, width, &l.o, mtp);
    w.Ggml(n + "attn_q_norm.weight", "F32", {hd}, &l.q_norm);
    w.Ggml(n + "attn_k_norm.weight", "F32", {hd}, &l.k_norm);
    w.Linear(n + "indexer.qk_proj", width, std::uint64_t{p.indexer_heads + 1} * p.indexer_head_dim,
             &l.idx_qk, mtp);
    w.Ggml(n + "indexer.q_norm.weight", "F32", {p.indexer_head_dim}, &l.idx_q_norm);
    w.Ggml(n + "indexer.k_norm.weight", "F32", {p.indexer_head_dim}, &l.idx_k_norm);
  }
  if (!mtp && il == p.ple_layer) {
    w.Ggml(n + "ple_key.weight", "BF16", {p.ple_width(), p.hc_width()}, &l.ple_key);
    w.Ggml(n + "ple_value.weight", "BF16", {p.ple_width(), width}, &l.ple_value);
    w.Ggml(n + "ple_norm_key.weight", "F32", {p.hc_width()}, &l.ple_norm_key);
    w.Ggml(n + "ple_norm_query.weight", "F32", {p.hc_width()}, &l.ple_norm_query);
    w.Ggml(n + "ple_norm_conv.weight", "F32", {p.hc_width()}, &l.ple_norm_conv);
    w.Ggml(n + "ple_conv1d.weight", "F32", {p.ple_conv, p.hc_width()}, &l.ple_conv1d);
    w.Plain(n + "ple_multipliers", "I64", {p.ngram}, &l.ple_multipliers);
    w.Plain(n + "ple_head_offsets", "I64", {p.ple_heads()}, &l.ple_head_offsets);
    w.Plain(n + "ple_head_vocab", "I64", {p.ple_heads()}, &l.ple_head_vocab);
  }
  w.Ggml(n + "ffn_gate_inp.weight", "BF16", {width, p.experts}, &l.router);
  w.Ggml(n + "ffn_gate_inp_shexp.weight", "BF16", {width}, &l.shared_gate);
  w.Linear(n + "ffn_gate_shexp", width, p.shared_ffn, &l.gate_shexp, mtp);
  w.Linear(n + "ffn_up_shexp", width, p.shared_ffn, &l.up_shexp, mtp);
  w.Linear(n + "ffn_down_shexp", p.shared_ffn, width, &l.down_shexp, mtp);
  const std::uint64_t f = p.expert_ffn;
  for (const auto& [proj, into, scale, k, out] :
       {std::tuple{"gate", &l.gate_exps, &l.gate_exps_scale, width, f},
        std::tuple{"up", &l.up_exps, &l.up_exps_scale, width, f},
        std::tuple{"down", &l.down_exps, &l.down_exps_scale, f, width}}) {
    if (!cutlass) {
      w.Array(std::format("{}ffn_{}_exps.weight", n, proj), "NVFP4", {k, out}, into);
    }
    w.Ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {p.experts}, scale);
  }
  if (cutlass) {
    // A matrix's swizzled scales: 512-byte atoms of 128 rows by 4 scales
    // (64 of k), the rows padded to 128.
    w.Array(n + "ffn_gate_up_exps.codes", "I8", {width / 2, 2 * f}, &l.gate_up_codes);
    w.Array(n + "ffn_gate_up_exps.scales", "I8", {512, Atoms(2 * f, width)}, &l.gate_up_scales);
    w.Array(n + "ffn_down_exps.codes", "I8", {f / 2, width}, &l.down_codes);
    w.Array(n + "ffn_down_exps.scales", "I8", {512, Atoms(width, f)}, &l.down_scales);
  }
}

// Binds every want to the resource of its role, which must have its
// representation, type and shape exactly; refused, naming the tensor, if
// one is missing or differs, or if the artifact binds a role no want reads.
std::expected<void, std::string> Match(const Qwen38Profile& p, std::span<const Want> want,
                                       std::span<const Qwen38Resource> resources,
                                       std::string_view model) {
  std::unordered_map<std::string, std::uint32_t> by_role;
  for (std::uint32_t i = 0; i < resources.size(); ++i) {
    for (const std::string& role : resources[i].roles) {
      if (!by_role.emplace(role, i).second) {
        return Refused(std::format("the artifact binds {} twice", role));
      }
    }
  }
  std::unordered_map<std::string_view, const Want*> wanted;
  for (const Want& x : want) {
    wanted.emplace(x.role, &x);
  }
  for (const auto& [role, index] : by_role) {
    if (!wanted.contains(role)) {
      return Refused(std::format("the artifact binds {}, which {} does not read", role, model));
    }
  }
  for (const Want& x : want) {
    const auto found = by_role.find(x.role);
    if (found == by_role.end()) {
      return Refused(std::format("the artifact has no {}", x.role));
    }
    const Qwen38Resource& r = resources[found->second];
    if (r.expert_array != x.expert_array) {
      return Refused(std::format("{} is {}an expert array", x.role, r.expert_array ? "" : "not "));
    }
    if (x.expert_array && r.count != p.experts) {
      return Refused(std::format("{} has {} experts, not {}", x.role, r.count, p.experts));
    }
    // A zero in the wanted shape leaves that extent to a later check.
    const bool shape_fits =
        r.ne.size() == x.ne.size() &&
        std::ranges::equal(r.ne, x.ne, [](std::uint64_t got, std::uint64_t want_ne) {
          return want_ne == 0 ? got > 0 : got == want_ne;
        });
    if (r.plain != x.plain || r.type != x.type || !shape_fits) {
      return Refused(std::format("{} is {} {} {}, not {} {} {}", x.role, r.plain ? "plain" : "GGML",
                                 r.type, Shape(r.ne), x.plain ? "plain" : "GGML", x.type,
                                 Shape(x.ne)));
    }
    *x.into = {.index = found->second,
               .plain = r.plain,
               .type = r.type,
               .ne = r.ne,
               .group_offset = r.group_offset,
               .readable = r.readable};
  }
  return {};
}

// The CUTLASS layout's four arrays back to back from each group's start, as
// the kernels read a slot (the artifact reader already put every layer's
// arrays in the same expert groups).
bool PackedFromTheStart(const Qwen38Layer& l) {
  std::uint64_t at = 0;
  for (const Qwen38Tensor* t : l.expert_arrays(true)) {
    if (t->group_offset != at) {
      return false;
    }
    at += t->ne[0] * t->ne[1];
  }
  return true;
}

// The resources of an artifact as the adapter sees them; `first_array` the
// index of its first expert array.
std::expected<std::vector<Qwen38Resource>, std::string> ResourcesOf(
    const artifact::Artifact& artifact, std::size_t& first_array) {
  std::vector<Qwen38Resource> all;
  all.reserve(artifact.resources().size() + artifact.expert_arrays().size());
  for (const artifact::Resource& resource : artifact.resources()) {
    if (resource.repr.family == artifact::Family::kExl3) {
      return Refused(std::format("{} is an EXL3 representation", resource.name));
    }
    const bool plain = resource.repr.family == artifact::Family::kPlain;
    std::vector<std::uint64_t> ne = resource.repr.dims;
    if (plain) {
      std::ranges::reverse(ne);  // GGML order
    }
    all.push_back({.roles = resource.roles,
                   .plain = plain,
                   .type = std::string(resource.repr.type),
                   .ne = std::move(ne)});
  }
  first_array = all.size();
  for (const artifact::ExpertArray& array : artifact.expert_arrays()) {
    if (array.repr.family != artifact::Family::kGgml) {
      return Refused(std::format("{} is not a GGML representation", array.name));
    }
    all.push_back({.roles = {array.name},
                   .plain = false,
                   .type = std::string(array.repr.type),
                   .ne = array.repr.dims,
                   .expert_array = true,
                   .count = array.count,
                   .group_offset = array.group_offset.value(),
                   .readable = array.readable.value()});
  }
  return all;
}

// Expert arrays are indexed among the artifact's expert arrays.
void Rebase(Qwen38Layer& l, bool cutlass, std::size_t first_array) {
  const auto rebase = [&](Qwen38Tensor& t) { t.index -= static_cast<std::uint32_t>(first_array); };
  if (cutlass) {
    rebase(l.gate_up_codes);
    rebase(l.gate_up_scales);
    rebase(l.down_codes);
    rebase(l.down_scales);
  } else {
    rebase(l.gate_exps);
    rebase(l.up_exps);
    rebase(l.down_exps);
  }
}

}  // namespace

const Qwen38Profile& Qwen38Flash() {
  // config.json's text_config (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6,
  // the base model's), which the importer validated
  // (docs/experiments/artifact-layout/modelopt_qwen38.py Config).
  static const Qwen38Profile kProfile = [] {
    Qwen38Profile p;
    p.name = "qwen3.8-flash-next";
    p.layers = 48;
    p.width = 2560;
    p.heads = 24;
    p.kv_heads = 2;
    p.head_dim = 256;
    p.rope_dims = 64;  // partial_rotary_factor 0.25
    p.rope_sections = {11, 11, 10, 0};
    p.rope_base = 10000000.0f;
    p.lin_k_heads = 16;
    p.lin_v_heads = 48;
    p.lin_head_dim = 128;
    p.conv = 4;
    p.experts = 512;
    p.experts_used = 10;
    p.expert_ffn = 640;
    p.shared_ffn = 640;
    p.hc = 4;
    p.hc_rank = 320;
    p.indexer_heads = 4;
    p.indexer_head_dim = 128;
    p.indexer_ratio = 4;
    p.indexer_budget = 2048;
    p.ple_layer = 1;  // ple_layer_ids [2], 1-based
    p.ngram = 3;
    p.heads_per_ngram = 8;
    p.ple_row = 160;
    p.ple_conv = 4;
    p.ple_eos = 248044;  // text_config.eos_token_id
    p.vocab = 248320;
    p.rms_eps = 1e-6f;
    return p;
  }();
  return kProfile;
}

std::expected<Qwen38Binding, std::string> BindQwen38(const Qwen38Profile& p,
                                                     std::string_view architecture,
                                                     std::span<const Qwen38Resource> resources) {
  if (architecture != "qwen4exp") {
    return Refused(std::format("the artifact's architecture is {}, not qwen4exp", architecture));
  }
  if (!ProfileIsSane(p)) {
    return Refused("the profile is not a Qwen3.8 model's");
  }
  Qwen38Binding b;
  b.layers.resize(p.layers);
  // The experts' layout: the CUTLASS layout's arrays where the artifact has
  // them (every layer's, or the binding is refused below), else GGML's.
  const bool cutlass = std::ranges::any_of(resources, [](const Qwen38Resource& r) {
    return r.expert_array && std::ranges::contains(r.roles, "blk.0.ffn_gate_up_exps.codes");
  });
  b.experts = cutlass ? Qwen38Experts::kCutlass : Qwen38Experts::kGgml;
  const std::uint64_t w = p.width;
  std::vector<Want> want;
  Wants wants(want);
  wants.Ggml("token_embd.weight", "BF16", {w, p.vocab}, &b.token_embd);
  wants.Ggml("output.weight", "BF16", {w, p.vocab}, &b.output);
  wants.Ggml("output_hc_norm.weight", "F32", {p.hc_width()}, &b.output_hc_norm);
  wants.Ggml("output_hc_down.weight", "BF16", {p.hc_width(), p.hc_rank}, &b.output_hc_down);
  wants.Ggml("output_hc_up.weight", "BF16", {p.hc_rank, p.hc_width()}, &b.output_hc_up);
  // The table's rows are checked against the hash's ranges, not the profile.
  wants.Plain("per_layer_token_embd.weight", "U8", {(p.ple_row / 2) + (p.ple_row / 16), 0},
              &b.ple_table);
  wants.Plain("per_layer_token_embd.weight_scale_2", "F32", {1}, &b.ple_table_scale);
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    AddLayer(wants, p, std::format("blk.{}.", il), il, b.layers[il], cutlass, false);
  }
  if (auto matched = Match(p, want, resources, "Qwen3.8"); !matched) {
    return std::unexpected(matched.error());
  }
  if (cutlass) {
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      if (!PackedFromTheStart(b.layers[il])) {
        return Refused(std::format(
            "layer {}'s CUTLASS expert arrays are not packed from each group's start", il));
      }
    }
  }
  return b;
}

std::expected<Qwen38Binding, std::string> BindQwen38(const Qwen38Profile& profile,
                                                     const artifact::Artifact& artifact) {
  std::size_t first_array = 0;
  auto all = ResourcesOf(artifact, first_array);
  if (!all) {
    return std::unexpected(all.error());
  }
  auto bound = BindQwen38(profile, artifact.model().architecture, *all);
  if (!bound) {
    return bound;
  }
  for (Qwen38Layer& l : bound->layers) {
    Rebase(l, bound->cutlass(), first_array);
  }
  return bound;
}

std::expected<void, std::string> CheckQwen38DraftIds(std::span<const std::int32_t> ids,
                                                     std::uint32_t vocab) {
  if (ids.empty() || ids.size() > vocab) {
    return Refused("an empty or oversized draft vocabulary");
  }
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || std::cmp_greater_equal(ids[i], vocab) || (i > 0 && ids[i - 1] >= ids[i])) {
      return Refused("draft vocabulary IDs must be in range and strictly ascending");
    }
  }
  return {};
}

std::expected<Qwen38MtpBinding, std::string> BindQwen38Mtp(
    const Qwen38Profile& p, std::string_view architecture,
    std::span<const Qwen38Resource> resources) {
  if (architecture != "qwen4exp-mtp") {
    return Refused(std::format("the drafter's architecture is {}, not qwen4exp-mtp", architecture));
  }
  if (!ProfileIsSane(p)) {
    return Refused("the profile is not a Qwen3.8 model's");
  }
  Qwen38MtpBinding b;
  const std::uint64_t w = p.width;
  std::vector<Want> want;
  Wants wants(want);
  wants.Ggml("fc_embd.weight", "BF16", {w, w}, &b.fc_embd);
  wants.Ggml("fc_hidden.weight", "BF16", {w, w}, &b.fc_hidden);
  wants.Ggml("norm_embd.weight", "F32", {w}, &b.norm_embd);
  wants.Ggml("norm_hidden.weight", "F32", {p.hc_width()}, &b.norm_hidden);
  wants.Ggml("output_hc_norm.weight", "F32", {p.hc_width()}, &b.output_hc_norm);
  wants.Ggml("output_hc_down.weight", "BF16", {p.hc_width(), p.hc_rank}, &b.output_hc_down);
  wants.Ggml("output_hc_up.weight", "BF16", {p.hc_rank, p.hc_width()}, &b.output_hc_up);
  for (const Qwen38Resource& r : resources) {
    if (std::ranges::contains(r.roles, "draft_output.weight")) {
      if (r.ne.size() != 2 || r.ne[1] == 0 || r.ne[1] > p.vocab) {
        return Refused("draft_output.weight has no bounded draft vocabulary");
      }
      wants.Ggml("draft_output.weight", "BF16", {w, r.ne[1]}, &b.draft_output);
      wants.Ggml("draft_output.ids", "I32", {1, r.ne[1]}, &b.draft_ids);
      break;
    }
  }
  // One full-attention layer, its routed experts in the CUTLASS layout.
  AddLayer(wants, p, "blk.0.", 0, b.layer, true, true);
  if (auto matched = Match(p, want, resources, "Qwen3.8's MTP drafter"); !matched) {
    return std::unexpected(matched.error());
  }
  if (!PackedFromTheStart(b.layer)) {
    return Refused("the drafter's CUTLASS expert arrays are not packed from each group's start");
  }
  return b;
}

std::expected<Qwen38MtpBinding, std::string> BindQwen38Mtp(const Qwen38Profile& profile,
                                                           const artifact::Artifact& artifact) {
  std::size_t first_array = 0;
  auto all = ResourcesOf(artifact, first_array);
  if (!all) {
    return std::unexpected(all.error());
  }
  auto bound = BindQwen38Mtp(profile, artifact.model().architecture, *all);
  if (!bound) {
    return bound;
  }
  Rebase(bound->layer, true, first_array);
  return bound;
}

std::expected<Qwen38PleHash, std::string> CheckQwen38PleHash(
    const Qwen38Profile& p, std::span<const std::int64_t> multipliers,
    std::span<const std::int64_t> offsets, std::span<const std::int64_t> vocab,
    std::uint64_t table_rows) {
  if (!ProfileIsSane(p) || multipliers.size() != p.ngram || offsets.size() != p.ple_heads() ||
      vocab.size() != p.ple_heads()) {
    return Refused("n-gram hash constants that are not the profile's");
  }
  // The lookup indexes rows in I32 (llama.cpp's qwen4exp_require checks).
  const auto limit = std::min<std::uint64_t>(table_rows, std::numeric_limits<std::int32_t>::max());
  Qwen38PleHash h;
  h.table_rows = table_rows;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    if (vocab[i] <= 0 || offsets[i] < 0 ||
        static_cast<std::uint64_t>(offsets[i]) + static_cast<std::uint64_t>(vocab[i]) > limit) {
      return Refused(std::format("n-gram head {}'s rows [{}, {} + {}) are outside the table's {}",
                                 i, offsets[i], offsets[i], vocab[i], table_rows));
    }
    h.offsets.push_back(static_cast<std::uint64_t>(offsets[i]));
    h.vocab.push_back(static_cast<std::uint64_t>(vocab[i]));
  }
  for (const std::int64_t m : multipliers) {
    h.multipliers.push_back(static_cast<std::uint64_t>(m));
  }
  return h;
}

// ---------------------------------------------------------------- state

std::int64_t Qwen38StateLayout::Find(std::uint32_t layer, Qwen38StateTensor::Kind kind) const {
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].layer == layer && tensors[i].kind == kind) {
      return static_cast<std::int64_t>(i);
    }
  }
  return -1;
}

std::vector<StateRepresentation> Qwen38StateLayout::Representations() const {
  using K = Qwen38StateTensor::Kind;
  std::uint64_t kv = 0;
  std::uint64_t indexer = 0;
  std::uint64_t recurrent = 0;
  for (const Qwen38StateTensor& t : tensors) {
    if (t.kind == K::kK || t.kind == K::kV) {
      kv += t.bytes;
    } else if (t.kind == K::kIndexerK || t.kind == K::kIndexerBlocks) {
      indexer += t.bytes;
    } else {
      recurrent += t.bytes;
    }
  }
  const auto fixed = [](std::string name, std::uint64_t bytes) {
    return StateRepresentation{.name = std::move(name),
                               .block_positions = 0,
                               .block_bytes = Bytes(bytes),
                               .capabilities = static_cast<std::uint8_t>(StateCapability::kAppend),
                               .max_snapshots = 0,
                               .snapshot_bytes = Bytes(0)};
  };
  return {fixed("qwen38.kv", kv), fixed("qwen38.indexer", indexer),
          fixed("qwen38.recurrent", recurrent)};
}

std::expected<Qwen38StateLayout, std::string> Qwen38State(const Qwen38Profile& p,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows, bool host_masks) {
  if (!ProfileIsSane(p)) {
    return Refused("the profile is not a Qwen3.8 model's");
  }
  // The reference and unfused graphs' [n_kv, rows] tensors (the host-built
  // masks, F16 and F32, and the indexer's expanded F32 scores) have planes
  // of up to n_kv x rows x 4 bytes, a stride GGML's flash attention takes as
  // a 32-bit int and ggml_permute truncates to one (RE-037): for them the
  // chunk bound and the padded context are bounded together by those bytes.
  if (context == 0 || max_rows == 0 || max_rows > context || max_rows > kQwen38MaxRows ||
      context > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - 255 ||
      (host_masks && Pad(context, 256) * max_rows * kMaskBytes >
                         std::uint64_t{std::numeric_limits<std::int32_t>::max()})) {
    return Refused(std::format("no state for {} positions in chunks of {}", context, max_rows));
  }
  using K = Qwen38StateTensor::Kind;
  Qwen38StateLayout s;
  s.context = context;
  s.max_rows = max_rows;
  s.cells = static_cast<std::uint32_t>(Pad(context, 256));
  const auto add = [&](K kind, std::uint32_t layer, bool f16, std::uint64_t ne0,
                       std::uint64_t ne1) {
    const std::uint64_t bytes = ne0 * ne1 * (f16 ? 2 : 4);
    s.tensors.push_back({.kind = kind,
                         .layer = layer,
                         .f16 = f16,
                         .ne0 = ne0,
                         .ne1 = ne1,
                         .offset = s.bytes,
                         .bytes = bytes});
    s.bytes = Pad(s.bytes + bytes, 256);
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.linear(il)) {
      add(K::kConv, il, false, std::uint64_t{p.conv - 1} * p.conv_channels(), 1);
      add(K::kRecurrent, il, false, std::uint64_t{p.lin_head_dim} * p.lin_head_dim * p.lin_v_heads,
          1);
      if (il == p.ple_layer) {
        add(K::kPleConv, il, false, std::uint64_t{p.ple_history()} * p.hc_width(), 1);
      }
    } else {
      add(K::kK, il, true, std::uint64_t{p.head_dim} * p.kv_heads, s.cells);
      add(K::kV, il, true, std::uint64_t{p.head_dim} * p.kv_heads, s.cells);
      add(K::kIndexerK, il, false, p.indexer_head_dim, s.cells);
      add(K::kIndexerBlocks, il, true, p.indexer_head_dim,
          (std::uint64_t{s.cells} + p.indexer_ratio - 1) / p.indexer_ratio);
    }
  }
  return s;
}

std::expected<std::vector<StateRange>, std::string> Qwen38UsedState(const Qwen38Profile& p,
                                                                    const Qwen38StateLayout& state,
                                                                    std::uint32_t positions) {
  if (positions > state.context || p.indexer_ratio == 0) {
    return Refused("used state passes the Qwen3.8 context");
  }
  using K = Qwen38StateTensor::Kind;
  const std::uint64_t read = std::min<std::uint64_t>(state.cells, Pad(positions, 256));
  std::vector<StateRange> ranges;
  for (const Qwen38StateTensor& t : state.tensors) {
    std::uint64_t rows = t.ne1;
    if (t.kind == K::kK || t.kind == K::kV || t.kind == K::kIndexerK) {
      rows = std::min(t.ne1, read);
    } else if (t.kind == K::kIndexerBlocks) {
      rows = std::min(t.ne1, (read + p.indexer_ratio - 1) / p.indexer_ratio);
    }
    if (rows != 0) {
      ranges.push_back({.offset = t.offset, .bytes = t.ne0 * rows * (t.f16 ? 2 : 4)});
    }
  }
  return ranges;
}

std::expected<std::vector<StateRange>, std::string> Qwen38CheckpointWrites(
    const Qwen38Profile& p, const Qwen38StateLayout& state, std::uint32_t positions) {
  if (positions > state.context || p.indexer_ratio == 0) {
    return Refused("checkpoint passes the Qwen3.8 context");
  }
  using K = Qwen38StateTensor::Kind;
  std::vector<StateRange> writes;
  for (const Qwen38StateTensor& t : state.tensors) {
    std::uint64_t first = 0;
    if (t.kind == K::kK || t.kind == K::kV || t.kind == K::kIndexerK) {
      first = positions;
    } else if (t.kind == K::kIndexerBlocks) {
      first = positions / p.indexer_ratio;  // a partly filled pool block is mutable
    }
    first = std::min(first, t.ne1);
    const std::uint64_t row = t.ne0 * (t.f16 ? 2 : 4);
    if (first < t.ne1) {
      writes.push_back({.offset = t.offset + (first * row), .bytes = (t.ne1 - first) * row});
    }
  }
  return writes;
}

std::uint32_t Qwen38MostRows(std::uint32_t context, bool host_masks) {
  if (context == 0 ||
      context > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - 255) {
    return 0;
  }
  const std::uint64_t masks = host_masks ? std::uint64_t{std::numeric_limits<std::int32_t>::max()} /
                                               (Pad(context, 256) * kMaskBytes)
                                         : std::uint64_t{kQwen38MaxRows};
  return static_cast<std::uint32_t>(std::min<std::uint64_t>({context, kQwen38MaxRows, masks}));
}

// ---------------------------------------------------------------- chunk inputs

std::vector<std::int32_t> Qwen38PleRows(const Qwen38Profile& p, const Qwen38PleHash& h,
                                        std::span<const std::int32_t> history,
                                        std::uint32_t position) {
  const std::uint32_t n_gram = p.ngram;
  const auto eos = static_cast<std::uint64_t>(p.ple_eos);
  std::vector<std::uint64_t> ctx(n_gram);
  ctx[0] = static_cast<std::uint64_t>(history[position]);
  bool cut = false;
  for (std::uint32_t s = 1; s < n_gram; ++s) {
    // The predecessor s positions back; missing (before the sequence) or
    // an EOS cuts everything at or before it, which then reads as EOS.
    const bool missing = s > position;
    const std::int64_t t = missing ? -1 : history[position - s];
    cut = cut || t < 0 || std::cmp_equal(t, p.ple_eos);
    ctx[s] = cut ? eos : static_cast<std::uint64_t>(t);
  }
  std::vector<std::int32_t> rows(p.ple_heads());
  for (std::uint32_t n = 2; n <= n_gram; ++n) {
    std::uint64_t mixed = ctx[0] * h.multipliers[0];
    for (std::uint32_t j = 1; j < n; ++j) {
      mixed ^= ctx[j] * h.multipliers[j];
    }
    const std::uint32_t base = (n - 2) * p.heads_per_ngram;
    for (std::uint32_t g = 0; g < p.heads_per_ngram; ++g) {
      const std::uint32_t head = base + g;
      rows[head] = static_cast<std::int32_t>((mixed % h.vocab[head]) + h.offsets[head]);
    }
  }
  return rows;
}

std::expected<Qwen38ChunkInputs, std::string> Qwen38Chunk(const Qwen38Profile& p,
                                                          const Qwen38StateLayout& state,
                                                          const Qwen38PleHash& hash,
                                                          std::span<const std::int32_t> history,
                                                          std::uint32_t n_past, std::uint32_t rows,
                                                          bool selection_masks) {
  const std::uint64_t end = std::uint64_t{n_past} + rows;
  if (rows == 0 || rows > state.max_rows || end > state.context) {
    return Refused(
        std::format("a chunk of {} rows at {} does not fit a context of {} in chunks "
                    "of {}",
                    rows, n_past, state.context, state.max_rows));
  }
  if (history.size() != end) {
    return Refused(std::format("the history holds {} tokens, not {}", history.size(), end));
  }
  if (hash.multipliers.size() != p.ngram || hash.offsets.size() != p.ple_heads() ||
      hash.vocab.size() != p.ple_heads()) {
    return Refused("n-gram hash constants that are not the profile's");
  }
  // Every head's rows inside the table and the I32 index (as
  // CheckQwen38PleHash), so no row the hash yields is out of range.
  const std::uint64_t limit =
      std::min<std::uint64_t>(hash.table_rows, std::numeric_limits<std::int32_t>::max());
  for (std::size_t i = 0; i < hash.vocab.size(); ++i) {
    if (hash.vocab[i] == 0 || hash.offsets[i] > limit || hash.vocab[i] > limit - hash.offsets[i]) {
      return Refused(std::format("n-gram head {}'s rows are outside the table", i));
    }
  }
  for (const std::int32_t t : history) {
    if (t < 0 || std::cmp_greater_equal(t, p.vocab)) {
      return Refused(std::format("token {} is outside the vocabulary", t));
    }
  }
  auto placed =
      Qwen38Rows(p, state.cells, n_past, rows,
                 static_cast<std::uint32_t>(std::min<std::uint64_t>(Pad(end, 256), state.cells)),
                 selection_masks);
  if (!placed) {
    return placed;
  }
  Qwen38ChunkInputs in = std::move(*placed);
  in.tokens.assign(history.begin() + n_past, history.end());
  in.ple_rows.resize(std::size_t{p.ple_heads()} * rows);
  for (std::uint32_t i = 0; i < rows; ++i) {
    const auto r = Qwen38PleRows(p, hash, history, n_past + i);
    std::ranges::copy(r, in.ple_rows.begin() + (std::ptrdiff_t{i} * p.ple_heads()));
  }
  return in;
}

std::expected<Qwen38ChunkInputs, std::string> Qwen38Rows(const Qwen38Profile& p,
                                                         std::uint32_t cells, std::uint32_t n_past,
                                                         std::uint32_t rows, std::uint32_t read,
                                                         bool selection_masks) {
  const std::uint64_t end = std::uint64_t{n_past} + rows;
  if (rows == 0 || end > cells || read < Pad(end, 256) || read > cells || read % 256 != 0 ||
      p.indexer_ratio == 0) {
    return Refused(std::format("{} rows at {} reading {} cells do not fit a cache of {}", rows,
                               n_past, read, cells));
  }
  Qwen38ChunkInputs in;
  in.n_past = n_past;
  in.rows = rows;
  in.n_kv = read;
  in.positions.resize(4 * std::size_t{rows});
  in.cells.resize(rows);
  for (std::uint32_t i = 0; i < rows; ++i) {
    const auto pos = static_cast<std::int32_t>(n_past + i);
    for (std::size_t sec = 0; sec < 4; ++sec) {
      in.positions[(sec * rows) + i] = pos;
    }
    in.cells[i] = pos;
  }
  const std::size_t n_kv = in.n_kv;
  // QSA: the budget keeps indexer_budget + ratio - 1 cells (whole blocks,
  // plus the tail); a chunk whose attention reads no more keeps every cell,
  // so the selection changes nothing and is not built.
  const std::uint32_t ratio = p.indexer_ratio;
  const std::uint64_t width = std::uint64_t{p.indexer_budget} + ratio - 1;
  in.qsa_select = n_kv > width;
  if (!in.qsa_select || selection_masks) {
    in.mask.assign(n_kv * rows, kQwen38HalfNegInf);
    for (std::uint32_t i = 0; i < rows; ++i) {
      std::fill_n(in.mask.begin() + static_cast<std::ptrdiff_t>(i * n_kv),
                  static_cast<std::ptrdiff_t>(std::uint64_t{n_past} + i + 1), kQwen38HalfZero);
    }
  }
  if (selection_masks) {
    in.mask_f32.assign(n_kv * rows, -std::numeric_limits<float>::infinity());
    for (std::uint32_t i = 0; i < rows; ++i) {
      std::fill_n(in.mask_f32.begin() + static_cast<std::ptrdiff_t>(i * n_kv),
                  static_cast<std::ptrdiff_t>(std::uint64_t{n_past} + i + 1), 0.0f);
    }
  }
  if (in.qsa_select) {
    in.qsa.blocks = static_cast<std::uint32_t>((n_kv + ratio - 1) / ratio);
  }
  // The fast graph selects from the cached block keys on the device: no
  // block tables.
  if (in.qsa_select && selection_masks) {
    Qwen38QsaInputs& q = in.qsa;
    // Cell j holds position j; the full blocks are those every one of whose
    // ratio positions is written.
    const auto full = static_cast<std::uint32_t>(end / ratio);
    const bool have_dead = full < q.blocks;
    const std::uint32_t dead = have_dead ? full : q.blocks - 1;
    q.cell_block.resize(n_kv);
    for (std::size_t j = 0; j < n_kv; ++j) {
      const auto b = static_cast<std::uint32_t>(j / ratio);
      q.cell_block[j] = static_cast<std::int32_t>(j < end && b < full ? b : dead);
    }
    q.block_cells.assign(std::size_t{ratio} * q.blocks, 0);
    q.block_pos.assign(4 * std::size_t{q.blocks}, 0);
    for (std::uint32_t b = 0; b < full; ++b) {
      for (std::uint32_t k = 0; k < ratio; ++k) {
        q.block_cells[(std::size_t{b} * ratio) + k] = static_cast<std::int32_t>((b * ratio) + k);
      }
      for (std::size_t sec = 0; sec < 4; ++sec) {
        q.block_pos[(sec * q.blocks) + b] = static_cast<std::int32_t>(b * ratio);
      }
    }
    q.bias.assign(std::size_t{q.blocks} * rows, 0.0f);
    for (std::uint32_t i = 0; i < rows; ++i) {
      const std::uint64_t pos = std::uint64_t{n_past} + i;
      // The incomplete tail is always visible.
      const std::uint64_t tail_start = (pos + 1) / ratio * ratio;
      float* bias = q.bias.data() + (std::size_t{i} * q.blocks);
      for (std::uint32_t b = 0; b < q.blocks; ++b) {
        if (b >= full) {
          bias[b] = -std::numeric_limits<float>::infinity();
          continue;
        }
        bias[b] = std::uint64_t{b} * ratio >= tail_start ? 1e9f : 0.0f;
      }
      if (have_dead) {
        bias[dead] = 1e9f;
      }
    }
  }
  return in;
}

// ---------------------------------------------------------------- the MTP drafter

std::vector<StateRepresentation> Qwen38MtpState::Representations() const {
  return {StateRepresentation{.name = "qwen38.mtp.kv",
                              .block_positions = 0,
                              .block_bytes = Bytes(hidden - k),
                              .capabilities = static_cast<std::uint8_t>(StateCapability::kAppend),
                              .max_snapshots = 0,
                              .snapshot_bytes = Bytes(0)},
          StateRepresentation{.name = "qwen38.mtp.streams",
                              .block_positions = 0,
                              .block_bytes = Bytes(bytes - hidden),
                              .capabilities = static_cast<std::uint8_t>(StateCapability::kAppend),
                              .max_snapshots = 0,
                              .snapshot_bytes = Bytes(0)}};
}

std::expected<Qwen38MtpState, std::string> Qwen38MtpStateOf(const Qwen38Profile& p,
                                                            const Qwen38StateLayout& state) {
  if (!ProfileIsSane(p) || state.cells == 0 || state.max_rows == 0 ||
      state.max_rows > kQwen38MaxRows) {
    return Refused("no MTP state for that target state");
  }
  Qwen38MtpState s;
  s.context = state.context;
  s.cells = state.cells;
  s.hidden_rows = state.max_rows + 1;
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * s.cells * 2;
  s.k = 0;
  s.v = Pad(kv, 256);
  s.indexer = s.v + Pad(kv, 256);
  s.blocks = s.indexer + Pad(std::uint64_t{p.indexer_head_dim} * s.cells * 4, 256);
  s.hidden =
      s.blocks + Pad(std::uint64_t{p.indexer_head_dim} *
                         ((std::uint64_t{s.cells} + p.indexer_ratio - 1) / p.indexer_ratio) * 2,
                     256);
  s.bytes = s.hidden + Pad(std::uint64_t{p.hc_width()} * s.hidden_rows * 4, 256);
  return s;
}

std::expected<Qwen38CommitLayout, std::string> Qwen38Commit(const Qwen38Profile& p,
                                                            std::uint32_t rows) {
  if (!ProfileIsSane(p) || rows == 0 || rows > 8) {
    return Refused(std::format("no commit layout for verifies of {} rows", rows));
  }
  Qwen38CommitLayout c;
  c.rows = rows;
  c.channels = p.conv_channels();
  c.v_heads = p.lin_v_heads;
  c.hc_width = p.hc_width();
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.linear(il)) {
      c.layers.push_back(il);
    }
  }
  c.layer_bytes = (2 * c.Part(c.channels)) + (2 * c.Part(c.v_heads));
  c.bytes = c.ple() + c.Part(c.hc_width);
  return c;
}

}  // namespace jitllm::model
