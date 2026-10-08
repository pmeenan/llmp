// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma4_assistant.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "artifact/representation.h"

namespace llmp::model {
namespace {
auto Refused(std::string_view why) { return std::unexpected(std::string(why)); }
bool Approved(const Gemma4AssistantProfile& p) {
  return p == Gemma4Assistant26() || p == Gemma4Assistant31();
}
struct Role {
  std::string name;
  std::vector<std::uint64_t> ne;
  std::string type;
};
std::array<Role, kGemma4AssistantResources> Roles(const Gemma4AssistantProfile& p) {
  std::array<Role, kGemma4AssistantResources> roles;
  std::size_t at = 0;
  const auto add = [&](std::string name, std::vector<std::uint64_t> ne, bool matrix = false) {
    roles[at++] = {std::move(name), std::move(ne), matrix ? "Q8_0" : "F32"};
  };
  add("token_embd.weight", {1024, 262144}, true);
  add("output_norm.weight", {1024});
  add("rope_freqs.weight", {256});
  add("nextn.pre_projection.weight", {2ULL * p.target_width, 1024}, true);
  add("nextn.post_projection.weight", {1024, p.target_width}, true);
  for (unsigned i = 0; i < 4; ++i) {
    const auto prefix = std::format("blk.{}.", i);
    const std::uint32_t dim = i == 3 ? 512 : 256;
    add(prefix + "attn_norm.weight", {1024});
    add(prefix + "attn_q.weight", {1024, std::uint64_t{dim} * p.heads}, true);
    add(prefix + "attn_q_norm.weight", {dim});
    add(prefix + "attn_output.weight", {std::uint64_t{dim} * p.heads, 1024}, true);
    add(prefix + "post_attention_norm.weight", {1024});
    add(prefix + "ffn_norm.weight", {1024});
    add(prefix + "ffn_gate.weight", {1024, 8192}, true);
    add(prefix + "ffn_up.weight", {1024, 8192}, true);
    add(prefix + "ffn_down.weight", {8192, 1024}, true);
    add(prefix + "post_ffw_norm.weight", {1024});
    add(prefix + "layer_output_scale.weight", {1});
  }
  return roles;
}
std::array<Gemma4Tensor*, kGemma4AssistantResources> Tensors(Gemma4AssistantBinding& b) {
  std::array<Gemma4Tensor*, kGemma4AssistantResources> result;
  std::size_t at = 0;
  for (auto* tensor :
       {&b.embedding, &b.output_norm, &b.rope_freqs, &b.pre_projection, &b.post_projection})
    result[at++] = tensor;
  for (auto& l : b.layers)
    for (auto* tensor : {&l.attn_norm, &l.q, &l.q_norm, &l.out, &l.attn_post_norm, &l.ffn_norm,
                         &l.gate, &l.up, &l.down, &l.ffn_post_norm, &l.output_scale})
      result[at++] = tensor;
  return result;
}
std::array<const Gemma4Tensor*, kGemma4AssistantResources> Tensors(
    const Gemma4AssistantBinding& b) {
  // The const traversal mirrors the bounded role list without copying any
  // public descriptor before its rank/type/identity checks.
  std::array<const Gemma4Tensor*, kGemma4AssistantResources> result;
  std::size_t at = 0;
  for (const auto* tensor :
       {&b.embedding, &b.output_norm, &b.rope_freqs, &b.pre_projection, &b.post_projection})
    result[at++] = tensor;
  for (const auto& l : b.layers)
    for (const auto* tensor :
         {&l.attn_norm, &l.q, &l.q_norm, &l.out, &l.attn_post_norm, &l.ffn_norm, &l.gate, &l.up,
          &l.down, &l.ffn_post_norm, &l.output_scale})
      result[at++] = tensor;
  return result;
}
}  // namespace
const Gemma4AssistantProfile& Gemma4Assistant26() {
  static constexpr Gemma4AssistantProfile p{
      "gemma-4-26B-A4B-it-assistant-q8_0", 2816, 16, 8, 2, 28, 29};
  return p;
}
const Gemma4AssistantProfile& Gemma4Assistant31() {
  static constexpr Gemma4AssistantProfile p{
      "gemma-4-31B-it-assistant-q8_0", 5376, 32, 16, 4, 58, 59};
  return p;
}
std::expected<Gemma4AssistantBinding, std::string> BindGemma4Assistant(
    const Gemma4AssistantProfile& p, std::string_view architecture,
    std::span<const Gemma4Resource> resources) {
  if (!Approved(p) || architecture != "gemma4-assistant" ||
      resources.size() != kGemma4AssistantResources)
    return Refused("not a complete approved Q8_0 Gemma4 assistant");
  // Inspect bounded descriptors before copying names, types or ranks.
  for (const auto& r : resources)
    if (r.roles.size() != 1 || r.roles[0].size() > 64 || r.ne.empty() || r.ne.size() > 2 ||
        (r.type != "Q8_0" && r.type != "F32") || r.expert_array || r.count != 0 ||
        r.group_offset != 0 ||
        r.readable > std::numeric_limits<std::uint64_t>::max() - r.group_offset)
      return Refused("malformed assistant resource descriptor");
  const auto roles = Roles(p);
  Gemma4AssistantBinding result;
  auto tensors = Tensors(result);
  std::array<bool, kGemma4AssistantResources> used{};
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const auto& want = roles[i];
    const auto found =
        std::ranges::find_if(resources, [&](const auto& r) { return r.roles[0] == want.name; });
    if (found == resources.end()) return Refused("missing assistant tensor role");
    const auto index = static_cast<std::uint32_t>(found - resources.begin());
    if (used[index] || found->type != want.type || found->ne != want.ne)
      return Refused("assistant tensor role, shape or type differs");
    used[index] = true;
    const auto* type = artifact::FindGgmlType(want.type);
    std::uint64_t bytes = want.ne[0] / type->block_elements * type->block_bytes;
    for (std::size_t dim = 1; dim < want.ne.size(); ++dim) bytes *= want.ne[dim];
    // All Q8_0 input dimensions are multiples of the 512-element GGML
    // tail-padding quantum, so these approved matrices need no extra tail.
    if (found->readable != bytes) return Refused("assistant tensor storage is truncated");
    *tensors[i] = {
        .index = index, .type = found->type, .ne = found->ne, .readable = found->readable};
  }
  result.head = result.embedding;
  return result;
}
std::expected<Gemma4AssistantBinding, std::string> BindGemma4Assistant(
    const Gemma4AssistantProfile& p, const artifact::Artifact& a) {
  if (!Approved(p) || a.model().architecture != "gemma4-assistant" || a.model().expert_count != 0 ||
      !a.expert_arrays().empty() || a.resources().size() != kGemma4AssistantResources)
    return Refused("not an approved ordinary assistant artifact");
  const artifact::ListedFile* metadata = nullptr;
  for (const auto& file : a.files()) {
    if (file.role != artifact::FileRole::kSourceMetadata) continue;
    if (metadata != nullptr || !file.path.starts_with("meta/") ||
        !file.path.ends_with(".kv.gguf") || file.bytes.value() > (16ULL << 20))
      return Refused("assistant requires one bounded kept GGUF metadata file");
    metadata = &file;
  }
  if (!metadata) return Refused("assistant kept GGUF metadata is missing");
  auto bytes = a.ReadMetadata(std::string_view(metadata->path).substr(5));
  if (!bytes) return Refused("assistant kept GGUF metadata could not be authenticated");
  if (auto checked =
          CheckGemma4AssistantMetadata(p, std::as_bytes(std::span(bytes->data(), bytes->size())));
      !checked)
    return Refused(checked.error());
  std::vector<Gemma4Resource> resources;
  resources.reserve(kGemma4AssistantResources);
  for (const auto& r : a.resources()) {
    if (r.repr.family != artifact::Family::kGgml || r.roles.size() != 1 || r.roles[0].size() > 64 ||
        r.repr.dims.empty() || r.repr.dims.size() > 2 ||
        (r.repr.type != "Q8_0" && r.repr.type != "F32"))
      return Refused("unsupported assistant artifact representation");
    resources.push_back({.roles = r.roles,
                         .type = std::string(r.repr.type),
                         .ne = r.repr.dims,
                         .readable = r.readable.value()});
  }
  return BindGemma4Assistant(p, a.model().architecture, resources);
}
std::expected<void, std::string> CheckGemma4AssistantMetadata(const Gemma4AssistantProfile& p,
                                                              std::span<const std::byte> bytes) {
  if (!Approved(p) || bytes.size() > (16ULL << 20))
    return Refused("assistant metadata profile or byte bound differs");
  constexpr std::string_view prefix = "gemma4-assistant.";
  constexpr std::array integer_keys{"block_count",
                                    "nextn_predict_layers",
                                    "embedding_length",
                                    "embedding_length_out",
                                    "feed_forward_length",
                                    "context_length",
                                    "attention.head_count",
                                    "attention.shared_kv_layers",
                                    "embedding_length_per_layer_input",
                                    "attention.key_length",
                                    "attention.value_length",
                                    "attention.key_length_swa",
                                    "attention.value_length_swa",
                                    "attention.sliding_window",
                                    "rope.dimension_count",
                                    "rope.dimension_count_swa"};
  const std::array<std::int64_t, integer_keys.size()> integers{
      4, 4, 1024, p.target_width, 8192, 262144, p.heads, 4, 0, 512, 512, 256, 256, 1024, 512, 256};
  constexpr std::array extra_keys{"attention.head_count_kv",
                                  "attention.sliding_window_pattern",
                                  "rope.freq_base",
                                  "rope.freq_base_swa",
                                  "attention.layer_norm_rms_epsilon",
                                  "final_logit_softcapping",
                                  "use_ordered_embeddings",
                                  "rope.scaling.factor",
                                  "rope.scaling.type",
                                  "rope.scale_linear",
                                  "rope.scaling.attn_factor",
                                  "rope.scaling.alpha",
                                  "rope.scaling.original_context_length",
                                  "rope.scaling.finetuned",
                                  "attention.scale",
                                  "attention.causal"};
  std::vector<std::string> names;
  names.reserve(integer_keys.size() + extra_keys.size() + 1);
  names.emplace_back("general.architecture");
  for (const auto* suffix : integer_keys) names.push_back(std::string(prefix) + suffix);
  for (const auto* suffix : extra_keys) names.push_back(std::string(prefix) + suffix);
  std::vector<std::string_view> wanted;
  wanted.reserve(names.size());
  for (const auto& name : names) wanted.push_back(name);
  auto metadata = artifact::ReadGgufMetadata(bytes, wanted);
  if (!metadata) return Refused("assistant kept GGUF metadata is malformed");
  using Kind = artifact::GgufValue::Kind;
  const auto found = [&](std::string_view suffix) -> const artifact::GgufValue* {
    const auto it = metadata->find(std::string(prefix) + std::string(suffix));
    return it == metadata->end() ? nullptr : &it->second;
  };
  const auto architecture = metadata->find("general.architecture");
  if (architecture == metadata->end() || architecture->second.kind != Kind::kString ||
      architecture->second.text != "gemma4-assistant")
    return Refused("assistant metadata architecture differs");
  for (std::size_t i = 0; i < integer_keys.size(); ++i) {
    const auto* value = found(integer_keys[i]);
    if (!value || value->kind != Kind::kInteger || value->integer != integers[i])
      return Refused("assistant metadata dimension or sharing contract differs");
  }
  const auto* kv = found("attention.head_count_kv");
  const auto* pattern = found("attention.sliding_window_pattern");
  const std::vector<std::int64_t> expected_kv{p.local_kv_heads, p.local_kv_heads, p.local_kv_heads,
                                              p.global_kv_heads};
  if (!kv || kv->kind != Kind::kIntegers || kv->integers != expected_kv || !pattern ||
      pattern->kind != Kind::kIntegers ||
      pattern->integers != std::vector<std::int64_t>{1, 1, 1, 0})
    return Refused("assistant metadata attention layer mapping differs");
  for (const auto [key, expected] :
       {std::pair{"rope.freq_base", 1000000.f}, std::pair{"rope.freq_base_swa", 10000.f},
        std::pair{"attention.layer_norm_rms_epsilon", 1e-6f}}) {
    const auto* value = found(key);
    if (!value || value->kind != Kind::kFloat || !std::isfinite(value->real) ||
        static_cast<float>(value->real) != expected)
      return Refused("assistant metadata rotary or norm contract differs");
  }
  const auto optional = [&](std::string_view key, std::int64_t expected) {
    const auto* value = found(key);
    return !value || (value->kind == Kind::kInteger && value->integer == expected) ||
           (value->kind == Kind::kFloat && value->real == static_cast<double>(expected));
  };
  if (!optional("final_logit_softcapping", 0) || !optional("use_ordered_embeddings", 0) ||
      !optional("rope.scaling.factor", 1) || found("rope.scaling.type") ||
      found("rope.scale_linear") || found("rope.scaling.attn_factor") ||
      found("rope.scaling.alpha") || found("rope.scaling.original_context_length") ||
      found("rope.scaling.finetuned") || !optional("attention.scale", 1) ||
      !optional("attention.causal", 1))
    return Refused("unsupported assistant softcap, centroid, scaling or attention semantics");
  return {};
}
std::expected<void, std::string> CheckGemma4AssistantBinding(const Gemma4AssistantProfile& p,
                                                             const Gemma4AssistantBinding& b) {
  if (!Approved(p)) return Refused("assistant profile differs");
  if (b.head.index >= kGemma4AssistantResources || b.head.ne.empty() || b.head.ne.size() > 2 ||
      (b.head.type != "Q8_0" && b.head.type != "F32") || b.head.expert_array ||
      b.head.group_offset != 0)
    return Refused("assistant tied head descriptor is outside the bounded domain");
  const auto roles = Roles(p);
  const auto tensors = Tensors(b);
  std::array<bool, kGemma4AssistantResources> used{};
  std::array<Gemma4Resource, kGemma4AssistantResources> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = *tensors[i];
    if (t.index >= resources.size() || used[t.index] || t.ne.empty() || t.ne.size() > 2 ||
        (t.type != "Q8_0" && t.type != "F32") || t.expert_array || t.group_offset != 0)
      return Refused("assistant binding identity, rank or storage differs");
    used[t.index] = true;
    resources[t.index] = {
        .roles = {roles[i].name}, .type = t.type, .ne = t.ne, .readable = t.readable};
  }
  if (b.head != b.embedding) return Refused("assistant tied canonical head differs");
  if (auto bound = BindGemma4Assistant(p, "gemma4-assistant", resources); !bound)
    return Refused(bound.error());
  return {};
}
std::expected<Gemma4AssistantTarget, std::string> CheckGemma4AssistantTarget(
    const Gemma4AssistantProfile& p, const Gemma4AssistantBinding& b, const Gemma4Profile& t,
    const Gemma4Binding& target_binding) {
  if (auto checked = CheckGemma4AssistantBinding(p, b); !checked) return Refused(checked.error());
  const auto& expected = p == Gemma4Assistant26() ? Gemma4_26BA4B() : Gemma4_31B();
  if (t != expected) return Refused("assistant and target profiles differ");
  if (auto checked = CheckGemma4Binding(t, target_binding); !checked)
    return Refused(checked.error());
  if (!t.local(p.local_target_layer) || t.local(p.global_target_layer))
    return Refused("assistant shared layer attention kinds differ");
  Gemma4AssistantTarget result;
  for (unsigned i = 0; i < 4; ++i) {
    const auto layer = i == 3 ? p.global_target_layer : p.local_target_layer;
    result.layers[i] = {.target_layer = layer,
                        .head_dim = t.head_dim(layer),
                        .kv_heads = t.kv_heads(layer),
                        .rope_dims = i == 3 ? t.global_rope_dims : t.local_rope_dims,
                        .local = i != 3};
  }
  return result;
}
std::expected<void, std::string> CheckGemma4AssistantVocabulary(
    const Gemma4AssistantVocabulary& t, const Gemma4AssistantVocabulary& a) {
  constexpr std::size_t vocab = 262144, merges = 514906;
  for (const auto* v : {&t, &a})
    if (v->tokens.size() != vocab || v->scores.size() != vocab || v->types.size() != vocab ||
        v->merges.size() != merges)
      return Refused("assistant vocabulary values are incomplete");
  std::uint64_t string_bytes = 0;
  const auto string_ok = [&](std::string_view s) {
    if (s.size() > 65536 || string_bytes > (32ULL << 20) - s.size()) return false;
    string_bytes += s.size();
    return true;
  };
  for (std::size_t i = 0; i < vocab; ++i) {
    if (!string_ok(t.tokens[i]) || !string_ok(a.tokens[i]) || t.tokens[i] != a.tokens[i] ||
        !std::isfinite(t.scores[i]) || !std::isfinite(a.scores[i]) ||
        std::bit_cast<std::uint64_t>(t.scores[i]) != std::bit_cast<std::uint64_t>(a.scores[i]))
      return Refused("assistant canonical token or score differs");
    const auto valid_type = [](std::int64_t kind) {
      return kind == 1 || kind == 3 || kind == 4 || kind == 5 || kind == 6;
    };
    if (!valid_type(t.types[i]) || !valid_type(a.types[i]))
      return Refused("unsupported assistant token kind");
    if (t.types[i] == a.types[i]) continue;
    if (i == 1 && t.tokens[i] == "<eos>" && t.types[i] == 1 && a.types[i] == 3) continue;
    if (i == 258884 && t.tokens[i] == "<|video|>" && t.types[i] == 3 && a.types[i] == 1) continue;
    return Refused("assistant token-kind difference is not an approved target-authority exception");
  }
  for (std::size_t i = 0; i < merges; ++i) {
    if (!string_ok(t.merges[i].first) || !string_ok(t.merges[i].second) ||
        !string_ok(a.merges[i].first) || !string_ok(a.merges[i].second) ||
        t.merges[i] != a.merges[i])
      return Refused("assistant canonical merge rank differs");
  }
  return {};
}
}  // namespace llmp::model
