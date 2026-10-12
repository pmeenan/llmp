// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen2_exl3.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/representation.h"
#include "model/v0_artifact.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

// ------------------------------------------------------------------ binding

class Binder {
 public:
  explicit Binder(std::span<const Exl3Resource> resources) : resources_(resources) {
    for (std::uint32_t i = 0; i < resources_.size(); ++i) {
      by_name_.emplace(resources_[i].name, i);
    }
  }

  std::expected<std::uint32_t, std::string> Plain(const std::string& name, std::string_view dtype,
                                                  const std::vector<std::uint64_t>& shape) {
    auto index = Find(name);
    if (!index) {
      return std::unexpected(index.error());
    }
    const Exl3Resource& r = resources_[*index];
    if (r.family != "plain" || r.dtype != dtype || r.shape != shape) {
      return Refused(std::format("{} is not plain {} of the profile's shape", name, dtype));
    }
    return *index;
  }

  std::expected<Exl3LinearBinding, std::string> Linear(std::string name, int k, int n, bool bias) {
    Exl3LinearBinding linear;
    linear.name = std::move(name);
    linear.k = k;
    linear.n = n;
    auto trellis = Find(linear.name + ".trellis");
    if (!trellis) {
      return std::unexpected(trellis.error());
    }
    const Exl3Resource& t = resources_[*trellis];
    const auto bits = static_cast<int>(t.k_bits);
    if (t.family != "exl3" || t.role != "trellis" || t.codebook != "mcg" || t.dtype != "I16" ||
        std::cmp_not_equal(t.in_features, k) || std::cmp_not_equal(t.out_features, n) ||
        (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
        t.shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(k / 16),
                                              static_cast<std::uint64_t>(n / 16),
                                              static_cast<std::uint64_t>(16 * bits)}) {
      return Refused(std::format("{}.trellis is not an mcg trellis of {} × {} at K = 4, 5, 6 or 8",
                                 linear.name, k, n));
    }
    linear.trellis = *trellis;
    linear.bits = bits;
    for (const auto& [role, length, out] :
         {std::tuple{"suh", k, &linear.suh}, std::tuple{"svh", n, &linear.svh}}) {
      auto index = Find(std::format("{}.{}", linear.name, role));
      if (!index) {
        return std::unexpected(index.error());
      }
      const Exl3Resource& r = resources_[*index];
      if (r.family != "exl3" || r.role != role || r.dtype != "F16" ||
          r.shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(length)}) {
        return Refused(
            std::format("{}.{} is not an F16 side vector of {}", linear.name, role, length));
      }
      *out = *index;
    }
    auto mcg = Find(linear.name + ".mcg");
    if (!mcg) {
      return std::unexpected(mcg.error());
    }
    const Exl3Resource& m = resources_[*mcg];
    if (m.family != "exl3" || m.role != "mcg" || m.dtype != "I32" || !m.shape.empty()) {
      return Refused(linear.name + ".mcg is not the mcg flag");
    }
    if (bias) {
      auto b = Plain(linear.name + ".bias", "F16", {static_cast<std::uint64_t>(n)});
      if (!b) {
        return std::unexpected(b.error());
      }
      linear.bias = *b;
    }
    return linear;
  }

  // Every resource was bound.
  std::expected<void, std::string> AllBound() const {
    for (std::uint32_t i = 0; i < resources_.size(); ++i) {
      if (!bound_.contains(i)) {
        return Refused(
            std::format("the artifact holds {}, which the plan does not read", resources_[i].name));
      }
    }
    return {};
  }

 private:
  std::expected<std::uint32_t, std::string> Find(const std::string& name) {
    const auto found = by_name_.find(name);
    if (found == by_name_.end()) {
      return Refused(std::format("the artifact has no {}", name));
    }
    bound_.insert({found->second, true});
    return found->second;
  }

  std::span<const Exl3Resource> resources_;
  std::map<std::string, std::uint32_t, std::less<>> by_name_;
  std::map<std::uint32_t, bool> bound_;
};

}  // namespace

const Qwen2Profile& Qwen25Instruct05BExl3() {
  // config.json of both EXL3 fixtures (docs/exl3-bringup.md): the same
  // model as the FP16 fixture; max_position_embeddings 32,768 is RoPE's
  // original context (exl3-op-plan.json, rope.n_ctx_orig).
  static constexpr Qwen2Profile kProfile = {.name = "qwen2.5-0.5b-instruct-exl3",
                                            .layers = 24,
                                            .width = 896,
                                            .heads = 14,
                                            .kv_heads = 2,
                                            .head_dim = 64,
                                            .ffn = 4864,
                                            .vocab = 151936,
                                            .train_context = 32768,
                                            .rms_eps = 1e-6f,
                                            .rope_base = 1000000.0f,
                                            .weight_type = "exl3"};
  return kProfile;
}

std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& profile,
                                                      const artifact::Artifact& artifact) {
  if (auto v0 = RequireV0Artifact(artifact, "Qwen2 EXL3"); !v0) {
    return std::unexpected(v0.error());
  }
  if (!artifact.expert_arrays().empty()) {
    return Refused("a dense Qwen2 artifact has no expert arrays");
  }
  std::vector<Exl3Resource> resources;
  resources.reserve(artifact.resources().size());
  for (const artifact::Resource& r : artifact.resources()) {
    if (r.roles.size() != 1 || r.roles.front() != r.name) {
      return Refused(std::format("{} has aliases, which the EXL3 plan does not bind", r.name));
    }
    resources.push_back({.name = r.name,
                         .family = std::string(artifact::FamilyName(r.repr.family)),
                         .dtype = std::string(r.repr.type),
                         .shape = r.repr.dims,
                         .role = std::string(r.repr.role),
                         .k_bits = r.repr.k_bits,
                         .in_features = r.repr.in_features,
                         .out_features = r.repr.out_features,
                         .codebook = std::string(r.repr.codebook)});
  }
  return BindQwen2Exl3(profile, artifact.model().architecture, resources);
}

std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& p,
                                                      std::string_view architecture,
                                                      std::span<const Exl3Resource> resources) {
  if (architecture != "qwen2") {
    return Refused(std::format("the artifact's architecture is {}, not qwen2", architecture));
  }
  // Every count nonzero before any is divided by or taken as a shape.
  if (p.layers == 0 || p.width == 0 || p.heads == 0 || p.kv_heads == 0 || p.head_dim == 0 ||
      p.ffn == 0 || p.vocab == 0 || p.width % 128 != 0 || p.ffn % 128 != 0 ||
      p.heads % p.kv_heads != 0 || (std::uint64_t{p.heads} * p.head_dim) != p.width ||
      p.kv_width() % 128 != 0) {
    return Refused("a profile the EXL3 plan cannot run");
  }
  Binder bind(resources);
  Exl3Binding out;
  const auto width = static_cast<int>(p.width);
  const auto ffn = static_cast<int>(p.ffn);
  const auto kv = static_cast<int>(p.kv_width());
  auto embed = bind.Plain("model.embed_tokens.weight", "BF16", {p.vocab, p.width});
  auto norm = bind.Plain("model.norm.weight", "BF16", {p.width});
  auto head = bind.Linear("lm_head", width, static_cast<int>(p.vocab), false);
  if (!embed) {
    return std::unexpected(embed.error());
  }
  if (!norm) {
    return std::unexpected(norm.error());
  }
  if (!head) {
    return std::unexpected(head.error());
  }
  out.embed = *embed;
  out.final_norm = *norm;
  out.lm_head = std::move(*head);
  for (std::uint32_t l = 0; l < p.layers; ++l) {
    const std::string b = std::format("model.layers.{}.", l);
    Exl3LayerBinding layer;
    auto attn = bind.Plain(b + "input_layernorm.weight", "BF16", {p.width});
    auto mlp = bind.Plain(b + "post_attention_layernorm.weight", "BF16", {p.width});
    if (!attn || !mlp) {
      return std::unexpected(!attn ? attn.error() : mlp.error());
    }
    layer.attn_norm = *attn;
    layer.mlp_norm = *mlp;
    struct Want {
      Exl3LinearBinding* into;
      std::string name;
      int k, n;
      bool bias;
    };
    for (const Want& w : {Want{&layer.q, b + "self_attn.q_proj", width, width, true},
                          Want{&layer.k, b + "self_attn.k_proj", width, kv, true},
                          Want{&layer.v, b + "self_attn.v_proj", width, kv, true},
                          Want{&layer.o, b + "self_attn.o_proj", width, width, false},
                          Want{&layer.gate, b + "mlp.gate_proj", width, ffn, false},
                          Want{&layer.up, b + "mlp.up_proj", width, ffn, false},
                          Want{&layer.down, b + "mlp.down_proj", ffn, width, false}}) {
      auto linear = bind.Linear(w.name, w.k, w.n, w.bias);
      if (!linear) {
        return std::unexpected(linear.error());
      }
      *w.into = std::move(*linear);
    }
    out.layers.push_back(std::move(layer));
  }
  if (auto all = bind.AllBound(); !all) {
    return std::unexpected(all.error());
  }
  return out;
}

}  // namespace llmp::model
