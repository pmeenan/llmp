// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The source names the mappings (spec/mapping.h) share, in the spirit of
// gguf-py's TENSOR_MAPPINGS and its key tables: a tensor role's llama.cpp
// GGUF spelling, which the mappings' tensor rules match, beside its
// transformers spelling, which a safetensors, MLX or EXL3 dialect's
// TensorAlias renames to it (or which the rules of a container read under its
// transformers names match directly); and a transformers config key beside
// the GGUF key the mappings' parameter rules read. A mapping names these
// roles, its slots and its own family's names; a role here is a spelling, not
// a component's weight role (one tensor binds different slots in different
// families: Gemma's sandwich norms). {layer} and {expert} are a tensor rule's
// (spec/mapping.h kLayerField, kExpertField). Internal to src/spec/mappings.

#ifndef LLMP_SPEC_MAPPINGS_NAMES_H_
#define LLMP_SPEC_MAPPINGS_NAMES_H_

#include <array>
#include <cstddef>
#include <string_view>

#include "format/encoding.h"
#include "spec/mapping.h"

namespace llmp::spec::mappings {

// A tensor's GGUF and transformers spellings (empty: the container has none).
struct TensorName {
  std::string_view gguf{};
  std::string_view hf{};
};

namespace names {
// The token table, the final norm and the output head.
inline constexpr TensorName kTokenEmbd{"token_embd.weight", "model.embed_tokens.weight"};
inline constexpr TensorName kOutputNorm{"output_norm.weight", "model.norm.weight"};
inline constexpr TensorName kOutput{"output.weight", "lm_head.weight"};
// A shared rotation-factor table (llama.cpp's; transformers computes its own).
inline constexpr TensorName kRopeFreqs{"rope_freqs.weight", {}};

// Attention.
inline constexpr TensorName kAttnNorm{"blk.{layer}.attn_norm.weight",
                                      "model.layers.{layer}.input_layernorm.weight"};
inline constexpr TensorName kAttnQ{"blk.{layer}.attn_q.weight",
                                   "model.layers.{layer}.self_attn.q_proj.weight"};
inline constexpr TensorName kAttnK{"blk.{layer}.attn_k.weight",
                                   "model.layers.{layer}.self_attn.k_proj.weight"};
inline constexpr TensorName kAttnV{"blk.{layer}.attn_v.weight",
                                   "model.layers.{layer}.self_attn.v_proj.weight"};
inline constexpr TensorName kAttnQBias{"blk.{layer}.attn_q.bias",
                                       "model.layers.{layer}.self_attn.q_proj.bias"};
inline constexpr TensorName kAttnKBias{"blk.{layer}.attn_k.bias",
                                       "model.layers.{layer}.self_attn.k_proj.bias"};
inline constexpr TensorName kAttnVBias{"blk.{layer}.attn_v.bias",
                                       "model.layers.{layer}.self_attn.v_proj.bias"};
inline constexpr TensorName kAttnOutput{"blk.{layer}.attn_output.weight",
                                        "model.layers.{layer}.self_attn.o_proj.weight"};
inline constexpr TensorName kAttnQNorm{"blk.{layer}.attn_q_norm.weight",
                                       "model.layers.{layer}.self_attn.q_norm.weight"};
inline constexpr TensorName kAttnKNorm{"blk.{layer}.attn_k_norm.weight",
                                       "model.layers.{layer}.self_attn.k_norm.weight"};

// The FFN of a pre-norm decoder: its norm is transformers' post-attention one.
inline constexpr TensorName kFfnNorm{"blk.{layer}.ffn_norm.weight",
                                     "model.layers.{layer}.post_attention_layernorm.weight"};
inline constexpr TensorName kFfnGate{"blk.{layer}.ffn_gate.weight",
                                     "model.layers.{layer}.mlp.gate_proj.weight"};
inline constexpr TensorName kFfnUp{"blk.{layer}.ffn_up.weight",
                                   "model.layers.{layer}.mlp.up_proj.weight"};
inline constexpr TensorName kFfnDown{"blk.{layer}.ffn_down.weight",
                                     "model.layers.{layer}.mlp.down_proj.weight"};

// A sandwich-norm block's (Gemma's): the attention's output norm, and the
// FFN's input and output norms.
inline constexpr TensorName kAttnPostNorm{"blk.{layer}.post_attention_norm.weight",
                                          "model.layers.{layer}.post_attention_layernorm.weight"};
inline constexpr TensorName kFfnPreNorm{"blk.{layer}.ffn_norm.weight",
                                        "model.layers.{layer}.pre_feedforward_layernorm.weight"};
inline constexpr TensorName kFfnPostNorm{"blk.{layer}.post_ffw_norm.weight",
                                         "model.layers.{layer}.post_feedforward_layernorm.weight"};

// A routed-expert block's router, shared expert and expert arrays
// (transformers' per-expert matrices; llama.cpp's stacked arrays).
inline constexpr TensorName kFfnGateInp{"blk.{layer}.ffn_gate_inp.weight",
                                        "model.layers.{layer}.mlp.gate.weight"};
inline constexpr TensorName kFfnGateInpShexp{"blk.{layer}.ffn_gate_inp_shexp.weight",
                                             "model.layers.{layer}.mlp.shared_expert_gate.weight"};
inline constexpr TensorName kFfnGateShexp{
    "blk.{layer}.ffn_gate_shexp.weight", "model.layers.{layer}.mlp.shared_expert.gate_proj.weight"};
inline constexpr TensorName kFfnUpShexp{"blk.{layer}.ffn_up_shexp.weight",
                                        "model.layers.{layer}.mlp.shared_expert.up_proj.weight"};
inline constexpr TensorName kFfnDownShexp{
    "blk.{layer}.ffn_down_shexp.weight", "model.layers.{layer}.mlp.shared_expert.down_proj.weight"};
inline constexpr TensorName kFfnGateExps{
    "blk.{layer}.ffn_gate_exps.weight",
    "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"};
inline constexpr TensorName kFfnUpExps{"blk.{layer}.ffn_up_exps.weight",
                                       "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"};
inline constexpr TensorName kFfnDownExps{
    "blk.{layer}.ffn_down_exps.weight",
    "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"};
// transformers' fused expert tables (one tensor a layer, with and without
// the `.weight` suffix by exporter).
inline constexpr TensorName kFfnGateUpExpsFused{"blk.{layer}.ffn_gate_up_exps.weight",
                                                "model.layers.{layer}.mlp.experts.gate_up_proj"};
inline constexpr TensorName kFfnGateUpExpsFusedWeight{
    "blk.{layer}.ffn_gate_up_exps.weight", "model.layers.{layer}.mlp.experts.gate_up_proj.weight"};
inline constexpr TensorName kFfnDownExpsFused{"blk.{layer}.ffn_down_exps.weight",
                                              "model.layers.{layer}.mlp.experts.down_proj"};
inline constexpr TensorName kFfnDownExpsFusedWeight{
    "blk.{layer}.ffn_down_exps.weight", "model.layers.{layer}.mlp.experts.down_proj.weight"};
// MLX's stacked expert arrays (mlx-lm's SwitchGLU).
inline constexpr TensorName kFfnGateExpsSwitch{
    "blk.{layer}.ffn_gate_exps.weight", "model.layers.{layer}.mlp.switch_mlp.gate_proj.weight"};
inline constexpr TensorName kFfnUpExpsSwitch{"blk.{layer}.ffn_up_exps.weight",
                                             "model.layers.{layer}.mlp.switch_mlp.up_proj.weight"};
inline constexpr TensorName kFfnDownExpsSwitch{
    "blk.{layer}.ffn_down_exps.weight", "model.layers.{layer}.mlp.switch_mlp.down_proj.weight"};

// Gated DeltaNet (linear attention: Qwen3.5's and Qwen3.8's).
inline constexpr TensorName kAttnQkv{"blk.{layer}.attn_qkv.weight",
                                     "model.layers.{layer}.linear_attn.in_proj_qkv.weight"};
inline constexpr TensorName kAttnGate{"blk.{layer}.attn_gate.weight",
                                      "model.layers.{layer}.linear_attn.in_proj_z.weight"};
inline constexpr TensorName kSsmBeta{"blk.{layer}.ssm_beta.weight",
                                     "model.layers.{layer}.linear_attn.in_proj_b.weight"};
inline constexpr TensorName kSsmAlpha{"blk.{layer}.ssm_alpha.weight",
                                      "model.layers.{layer}.linear_attn.in_proj_a.weight"};
inline constexpr TensorName kSsmDt{"blk.{layer}.ssm_dt.bias",
                                   "model.layers.{layer}.linear_attn.dt_bias"};
inline constexpr TensorName kSsmA{"blk.{layer}.ssm_a", "model.layers.{layer}.linear_attn.A_log"};
inline constexpr TensorName kSsmConv1d{"blk.{layer}.ssm_conv1d.weight",
                                       "model.layers.{layer}.linear_attn.conv1d.weight"};
inline constexpr TensorName kSsmNorm{"blk.{layer}.ssm_norm.weight",
                                     "model.layers.{layer}.linear_attn.norm.weight"};
inline constexpr TensorName kSsmOut{"blk.{layer}.ssm_out.weight",
                                    "model.layers.{layer}.linear_attn.out_proj.weight"};
}  // namespace names

// The transformers config keys the mappings read under their GGUF keys (a
// dialect's KeyAlias: source, canonical), and the ones they read as they are.
namespace keys {
inline constexpr KeyAlias kLayerCount{"num_hidden_layers", "block_count"};
inline constexpr KeyAlias kContext{"max_position_embeddings", "context_length"};
inline constexpr KeyAlias kWidth{"hidden_size", "embedding_length"};
inline constexpr KeyAlias kFfnWidth{"intermediate_size", "feed_forward_length"};
inline constexpr KeyAlias kRmsEps{"rms_norm_eps", "attention.layer_norm_rms_epsilon"};
inline constexpr KeyAlias kHeads{"num_attention_heads", "attention.head_count"};
inline constexpr KeyAlias kKvHeads{"num_key_value_heads", "attention.head_count_kv"};
inline constexpr KeyAlias kHeadDim{"head_dim", "attention.key_length"};
inline constexpr KeyAlias kRopeTheta{"rope_theta", "rope.freq_base"};
inline constexpr KeyAlias kRopeParametersTheta{"rope_parameters.rope_theta", "rope.freq_base"};
inline constexpr KeyAlias kRopeParametersType{"rope_parameters.rope_type", "rope.scaling.type"};
inline constexpr KeyAlias kRopeParametersPartial{"rope_parameters.partial_rotary_factor",
                                                 "rope.partial_rotary_factor"};
inline constexpr KeyAlias kPartialRotary{"partial_rotary_factor", "rope.partial_rotary_factor"};
inline constexpr KeyAlias kRopeParametersMropeSection{"rope_parameters.mrope_section",
                                                      "rope.dimension_sections"};
inline constexpr KeyAlias kRopeParametersMropeInterleaved{"rope_parameters.mrope_interleaved",
                                                          "rope.mrope_interleaved"};
inline constexpr KeyAlias kSlidingWindow{"sliding_window", "attention.sliding_window"};
// A routed-expert block's.
inline constexpr KeyAlias kExperts{"num_experts", "expert_count"};
inline constexpr KeyAlias kExpertsUsed{"num_experts_per_tok", "expert_used_count"};
inline constexpr KeyAlias kExpertWidth{"moe_intermediate_size", "expert_feed_forward_length"};
inline constexpr KeyAlias kSharedExpertWidth{"shared_expert_intermediate_size",
                                             "expert_shared_feed_forward_length"};
// Gated DeltaNet's.
inline constexpr KeyAlias kLinearConvKernel{"linear_conv_kernel_dim", "ssm.conv_kernel"};
inline constexpr KeyAlias kLinearKeyHeads{"linear_num_key_heads", "ssm.group_count"};
inline constexpr KeyAlias kLinearValueHeads{"linear_num_value_heads", "ssm.time_step_rank"};
inline constexpr KeyAlias kLinearKeyHeadDim{"linear_key_head_dim", "ssm.state_size"};
inline constexpr KeyAlias kLinearValueHeadDim{"linear_value_head_dim", "ssm.value_head_dim"};
// Keys read under their own spelling (requirements on the source).
inline constexpr KeyAlias kHiddenAct{"hidden_act", "hidden_act"};
inline constexpr KeyAlias kAttentionBias{"attention_bias", "attention_bias"};
inline constexpr KeyAlias kEncoderDecoder{"is_encoder_decoder", "is_encoder_decoder"};
inline constexpr KeyAlias kMambaDtype{"mamba_ssm_dtype", "mamba_ssm_dtype"};
inline constexpr KeyAlias kOutputGateType{"output_gate_type", "output_gate_type"};
inline constexpr KeyAlias kLayerTypes{"layer_types", "layer_types"};
inline constexpr KeyAlias kBidirectional{"use_bidirectional_attention",
                                         "use_bidirectional_attention"};
}  // namespace keys

// A dialect's alias of a dictionary tensor's transformers spelling to its
// GGUF one (the rule pattern it binds by), or a family's own alias.
constexpr TensorAlias AliasOf(const TensorName& name) { return {name.hf, name.gguf}; }
constexpr TensorAlias AliasOf(const TensorAlias& alias) { return alias; }
template <class... Entries>
constexpr std::array<TensorAlias, sizeof...(Entries)> Aliases(const Entries&... entries) {
  return {AliasOf(entries)...};
}

// A tensor name made at compile time from a stem and a suffix (a matrix's
// EXL3 parts: its trellis codes, signs and codebook marker). The constexpr
// table that holds it keeps it in static storage, which the rules naming
// it point into.
struct ComposedName {
  std::array<char, 96> text{};
  std::size_t size = 0;
  constexpr std::string_view view() const { return {text.data(), size}; }
};
// `name` without `suffix` (which it must end with), then `replacement`;
// too long a name does not compile (an index past the array).
constexpr ComposedName Respell(std::string_view name, std::string_view suffix,
                               std::string_view replacement) {
  ComposedName out;
  const std::string_view stem =
      name.ends_with(suffix) ? name.substr(0, name.size() - suffix.size()) : std::string_view{};
  for (const char c : stem) out.text[out.size++] = c;
  for (const char c : replacement) out.text[out.size++] = c;
  return out;
}

// A matrix's parts in an EXL3 (trellis-quantized) checkpoint, under its
// transformers name without `.weight`: the trellis codes, the input and
// output sign vectors, and the optional codebook marker (exllamav3's `mcg`,
// which older exports name `mul1`).
struct TrellisPart {
  std::string_view suffix{};
  format::PartRole part = format::PartRole::kData;
  bool optional = false;
};
inline constexpr std::array kTrellisParts{
    TrellisPart{".trellis", format::PartRole::kCodes},
    TrellisPart{".suh", format::PartRole::kInputSigns},
    TrellisPart{".svh", format::PartRole::kOutputSigns},
    TrellisPart{".mcg", format::PartRole::kCodebookMarker, true},
};
inline constexpr std::string_view kTrellisMarkerAlias = ".mul1";

// A trellis-quantized matrix: its dictionary name and the slot and role it binds.
struct TrellisMatrix {
  TensorName name{};
  std::string_view slot{}, role{};
};
// The names a table of trellis-quantized matrices binds: each matrix's
// parts, in kTrellisParts' order, then its older marker spelling.
template <std::size_t N>
constexpr auto TrellisNames(const std::array<TrellisMatrix, N>& matrices) {
  std::array<ComposedName, N*(kTrellisParts.size() + 1)> out{};
  std::size_t next = 0;
  for (const auto& m : matrices) {
    for (const auto& part : kTrellisParts) out[next++] = Respell(m.name.hf, ".weight", part.suffix);
    out[next++] = Respell(m.name.hf, ".weight", kTrellisMarkerAlias);
  }
  return out;
}
// Their tensor rules, matrix by matrix, part by part.
template <std::size_t N, std::size_t M>
constexpr auto TrellisRules(const std::array<TrellisMatrix, N>& matrices,
                            const std::array<ComposedName, M>& names) {
  static_assert(M == N * (kTrellisParts.size() + 1));
  std::array<TensorRule, N * kTrellisParts.size()> out{};
  std::size_t next = 0;
  for (std::size_t m = 0; m < N; ++m)
    for (std::size_t p = 0; p < kTrellisParts.size(); ++p)
      out[next++] = TensorRule{.pattern = names[(m * (kTrellisParts.size() + 1)) + p].view(),
                               .slot = matrices[m].slot,
                               .role = matrices[m].role,
                               .part = kTrellisParts[p].part,
                               .optional = kTrellisParts[p].optional};
  return out;
}
// Part `part` (kTrellisParts' index) of matrix `matrix`'s names.
template <std::size_t M>
constexpr std::string_view TrellisPartName(const std::array<ComposedName, M>& names,
                                           std::size_t matrix, std::size_t part) {
  return names[(matrix * (kTrellisParts.size() + 1)) + part].view();
}
// The older codebook-marker spelling of each matrix, an alias of its marker.
template <std::size_t M>
constexpr auto TrellisMarkerAliases(const std::array<ComposedName, M>& names) {
  constexpr std::size_t kStride = kTrellisParts.size() + 1;
  static_assert(M % kStride == 0);
  std::array<TensorAlias, M / kStride> out{};
  for (std::size_t m = 0; m < out.size(); ++m)
    out[m] = TensorAlias{.source = names[(m * kStride) + kTrellisParts.size()].view(),
                         .canonical = names[(m * kStride) + kTrellisParts.size() - 1].view()};
  return out;
}

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_NAMES_H_
