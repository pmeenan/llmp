// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma2.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <unordered_map>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/representation.h"

namespace jitllm::model {
namespace {
// Every multiplication follows equality with the closed, bounded profile.
// Keep the existing artifact's 512-element quantized over-read contract:
// width2304 requires one extra 256-element Q8_0 row tail (272 bytes).
bool StorageValid(std::string_view type, std::span<const std::uint64_t> ne, std::uint64_t readable,
                  std::string_view expected_type, std::span<const std::uint64_t> expected_ne) {
  const auto* traits = artifact::FindGgmlType(type);
  if (type != expected_type || traits == nullptr || ne.empty() ||
      !std::ranges::equal(ne, expected_ne) || ne[0] % traits->block_elements != 0)
    return false;
  auto bytes = ne[0] / traits->block_elements * traits->block_bytes;
  for (std::size_t i = 1; i < ne.size(); ++i) bytes *= ne[i];
  if (traits->block_elements > 1 && ne[0] % artifact::kGgmlRowPadding != 0) {
    bytes += (artifact::kGgmlRowPadding - ne[0] % artifact::kGgmlRowPadding) /
             traits->block_elements * traits->block_bytes;
  }
  return readable >= bytes;
}
}  // namespace

const Gemma2Profile& Gemma2_2B() {
  // Actual metadata plus d812 defaults: absent RoPE base/scale => 10000/1.
  // The 2B attention scale is 1/sqrt(256), not 1/sqrt(width/heads).
  static constexpr Gemma2Profile profile{26,      2304,    9216,  8,    4,        256,
                                         256,     256000,  8192,  4096, 10000.0F, 1.0F,
                                         1.0e-6F, 0.0625F, 50.0F, 30.0F};
  return profile;
}

std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& p,
                                                     std::string_view architecture,
                                                     std::span<const Gemma2Resource> resources) {
  if (p != Gemma2_2B() || architecture != "gemma2" || resources.size() != 288) {
    return std::unexpected("unsupported Gemma2 profile, architecture or resource count");
  }
  std::unordered_map<std::string, std::uint32_t> roles;
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto& resource = resources[i];
    if (resource.roles.empty() || resource.roles.size() > 2) {
      return std::unexpected("invalid Gemma2 resource roles");
    }
    for (const auto& role : resource.roles) {
      if (role.empty() || !roles.emplace(role, static_cast<std::uint32_t>(i)).second) {
        return std::unexpected("empty or duplicate Gemma2 role");
      }
    }
  }
  const auto take =
      [&](const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<Gemma2Tensor, std::string> {
    const auto found = roles.find(role);
    if (found == roles.end()) {
      return std::unexpected("missing Gemma2 role: " + role);
    }
    const auto index = found->second;
    const auto& resource = resources[index];
    const std::vector<std::uint64_t> ne(shape);
    if (!StorageValid(resource.type, resource.ne, resource.readable, type, ne)) {
      return std::unexpected("wrong Gemma2 type, shape or readable storage: " + role);
    }
    roles.erase(found);
    return Gemma2Tensor{index, resource.type, resource.ne, resource.readable};
  };
  Gemma2Binding binding;
  const auto assign =
      [&](Gemma2Tensor& target, const std::string& role, std::string_view type,
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
      return std::unexpected("Gemma2 output must alias token embeddings");
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
    const auto norm = [&](Gemma2Tensor& tensor, std::string_view suffix, std::uint64_t width) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "F32", {width});
    };
    const auto matrix = [&](Gemma2Tensor& tensor, std::string_view suffix, std::uint64_t input,
                            std::uint64_t output) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "Q8_0", {input, output});
    };
    // Q/K/V are distinct resources; this profile has no Q/K norms.
    for (auto result :
         {norm(layer.attn_norm, "attn_norm", p.width),
          matrix(layer.q, "attn_q", p.width, p.heads * p.key_dim),
          matrix(layer.k, "attn_k", p.width, p.kv_heads * p.key_dim),
          matrix(layer.v, "attn_v", p.width, p.kv_heads * p.value_dim),
          matrix(layer.out, "attn_output", p.heads * p.value_dim, p.width),
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
    return std::unexpected("unconsumed Gemma2 roles");
  }
  return binding;
}

std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& p,
                                                     const artifact::Artifact& artifact) {
  if (artifact.model().expert_count != 0 || !artifact.expert_arrays().empty() ||
      artifact.resources().size() != 288) {
    return std::unexpected("Gemma2 requires 288 ordinary resources and no expert storage");
  }
  std::vector<Gemma2Resource> resources;
  resources.reserve(artifact.resources().size());
  for (const auto& resource : artifact.resources()) {
    if (resource.repr.family != artifact::Family::kGgml) {
      return std::unexpected("Gemma2 requires GGML resources");
    }
    resources.push_back({resource.roles, std::string(resource.repr.type), resource.repr.dims,
                         resource.readable.value()});
  }
  return BindGemma2(p, artifact.model().architecture, resources);
}
std::expected<void, std::string> CheckGemma2Binding(const Gemma2Profile& p,
                                                    const Gemma2Binding& binding) {
  if (p != Gemma2_2B() || binding.layers.size() != p.layers ||
      binding.output != binding.token_embd) {
    return std::unexpected("invalid Gemma2 profile, layer count or tied head");
  }
  // Fixed stack membership replaces reconstruction/rebinding on future waves.
  std::array<bool, 288> seen{};
  const auto check = [&](const Gemma2Tensor& t, std::string_view type,
                         std::initializer_list<std::uint64_t> shape) {
    if (t.index >= seen.size() || seen[t.index] ||
        !StorageValid(t.type, t.ne, t.readable, type, shape))
      return false;
    seen[t.index] = true;
    return true;
  };
  if (!check(binding.token_embd, "Q8_0", {p.width, p.vocab}) ||
      !check(binding.output_norm, "F32", {p.width})) {
    return std::unexpected("invalid Gemma2 embedding/head descriptors");
  }
  for (const auto& l : binding.layers) {
    if (!check(l.attn_norm, "F32", {p.width}) ||
        !check(l.q, "Q8_0", {p.width, p.heads * p.key_dim}) ||
        !check(l.k, "Q8_0", {p.width, p.kv_heads * p.key_dim}) ||
        !check(l.v, "Q8_0", {p.width, p.kv_heads * p.value_dim}) ||
        !check(l.out, "Q8_0", {p.heads * p.value_dim, p.width}) ||
        !check(l.attn_post_norm, "F32", {p.width}) || !check(l.ffn_norm, "F32", {p.width}) ||
        !check(l.gate, "Q8_0", {p.width, p.ffn}) || !check(l.up, "Q8_0", {p.width, p.ffn}) ||
        !check(l.down, "Q8_0", {p.ffn, p.width}) || !check(l.ffn_post_norm, "F32", {p.width})) {
      return std::unexpected("invalid Gemma2 layer descriptor or resource identity");
    }
  }
  return {};
}
}  // namespace jitllm::model
