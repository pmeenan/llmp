// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma3.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/representation.h"

namespace jitllm::model {
const Gemma3Profile& Gemma3_4BQat() {
  // Facts from bbcac0d0's approved Q4_0 file, not a Gemma4-derived profile.
  static constexpr Gemma3Profile profile{
      34, 2560, 10240, 8, 4, 256, 256, 262208, 131072, 1024, 1000000.0F, 8.0F, 1.0e-6F, "linear"};
  return profile;
}

std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& p,
                                                     std::string_view architecture,
                                                     std::span<const Gemma3Resource> resources) {
  if (p != Gemma3_4BQat() || architecture != "gemma3" || resources.size() != 444) {
    return std::unexpected("unsupported Gemma3 profile, architecture or resource count");
  }
  std::unordered_map<std::string, std::uint32_t> roles;
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto& resource = resources[i];
    if (resource.roles.empty() || resource.roles.size() > 2) {
      return std::unexpected("invalid Gemma3 resource roles");
    }
    for (const auto& role : resource.roles) {
      if (role.empty() || !roles.emplace(role, static_cast<std::uint32_t>(i)).second) {
        return std::unexpected("empty or duplicate Gemma3 role");
      }
    }
  }
  const auto take =
      [&](const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<Gemma3Tensor, std::string> {
    const auto found = roles.find(role);
    if (found == roles.end()) {
      return std::unexpected("missing Gemma3 role: " + role);
    }
    const auto index = found->second;
    const auto& resource = resources[index];
    const std::vector<std::uint64_t> ne(shape);
    const auto* traits = artifact::FindGgmlType(type);
    if (traits == nullptr || resource.type != type || resource.ne != ne ||
        ne[0] % traits->block_elements != 0) {
      return std::unexpected("wrong Gemma3 type or shape: " + role);
    }
    // Multiplication follows exact bounded profile shapes, after equality
    // above. Every approved quantized row is a multiple of the GGML 512-row
    // padding, so its canonical readable bytes equal its stored bytes.
    auto bytes = ne[0] / traits->block_elements * traits->block_bytes;
    for (std::size_t dim = 1; dim < ne.size(); ++dim) {
      bytes *= ne[dim];
    }
    if (resource.readable < bytes) {
      return std::unexpected("short Gemma3 readable storage: " + role);
    }
    roles.erase(found);
    return Gemma3Tensor{index, resource.type, resource.ne, resource.readable};
  };
  Gemma3Binding binding;
  const auto assign =
      [&](Gemma3Tensor& target, const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<void, std::string> {
    auto tensor = take(role, type, shape);
    if (!tensor) {
      return std::unexpected(tensor.error());
    }
    target = std::move(*tensor);
    return {};
  };
  if (auto result = assign(binding.token_embd, "token_embd.weight", "Q8_0", {p.width, p.vocab});
      !result) {
    return std::unexpected(result.error());
  }
  binding.output = binding.token_embd;
  if (const auto head = roles.find("output.weight"); head != roles.end()) {
    if (head->second != binding.token_embd.index) {
      return std::unexpected("Gemma3 output must alias token embeddings");
    }
    roles.erase(head);
  }
  if (auto result = assign(binding.output_norm, "output_norm.weight", "F32", {p.width}); !result) {
    return std::unexpected(result.error());
  }
  binding.layers.resize(p.layers);
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    auto& layer = binding.layers[il];
    const auto prefix = "blk." + std::to_string(il) + '.';
    const auto norm = [&](Gemma3Tensor& tensor, std::string_view suffix, std::uint64_t width) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "F32", {width});
    };
    const auto matrix = [&](Gemma3Tensor& tensor, std::string_view suffix, std::uint64_t input,
                            std::uint64_t output) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "Q4_0", {input, output});
    };
    // V is a separate actual resource in every layer; do not inherit
    // Gemma4's global-layer K/V identity rule.
    for (auto result :
         {norm(layer.attn_norm, "attn_norm", p.width),
          matrix(layer.q, "attn_q", p.width, p.heads * p.key_dim),
          matrix(layer.k, "attn_k", p.width, p.kv_heads * p.key_dim),
          matrix(layer.v, "attn_v", p.width, p.kv_heads * p.value_dim),
          matrix(layer.out, "attn_output", p.heads * p.value_dim, p.width),
          norm(layer.q_norm, "attn_q_norm", p.key_dim),
          norm(layer.k_norm, "attn_k_norm", p.key_dim),
          norm(layer.attn_post_norm, "post_attention_norm", p.width),
          norm(layer.ffn_norm, "ffn_norm", p.width), matrix(layer.gate, "ffn_gate", p.width, p.ffn),
          matrix(layer.up, "ffn_up", p.width, p.ffn),
          matrix(layer.down, "ffn_down", p.ffn, p.width),
          norm(layer.ffn_post_norm, "post_ffw_norm", p.width)}) {
      if (!result) {
        return std::unexpected(result.error());
      }
    }
  }
  if (!roles.empty()) {
    return std::unexpected("unconsumed Gemma3 roles");
  }
  return binding;
}

std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& p,
                                                     const artifact::Artifact& artifact) {
  if (artifact.model().expert_count != 0 || !artifact.expert_arrays().empty() ||
      artifact.resources().size() != 444) {
    return std::unexpected("Gemma3 requires 444 ordinary resources and no expert storage");
  }
  std::vector<Gemma3Resource> resources;
  resources.reserve(artifact.resources().size());
  for (const auto& resource : artifact.resources()) {
    if (resource.repr.family != artifact::Family::kGgml) {
      return std::unexpected("Gemma3 requires GGML resources");
    }
    resources.push_back({resource.roles, std::string(resource.repr.type), resource.repr.dims,
                         resource.readable.value()});
  }
  return BindGemma3(p, artifact.model().architecture, resources);
}
}  // namespace jitllm::model
