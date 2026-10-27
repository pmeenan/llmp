// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shorthand for writing mapping tables (spec/mapping.h) as constant data,
// and for the source values the mapping data carries (a drafter import's
// target, the native MTP drafter's block). Internal to src/spec/mappings
// (and the tests' family fixtures).

#ifndef LLMP_SPEC_MAPPINGS_RULES_H_
#define LLMP_SPEC_MAPPINGS_RULES_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "format/encoding.h"
#include "spec/mapping.h"
#include "spec/mappings/names.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
inline constexpr std::uint32_t kLayerLimit = 1024;
inline constexpr std::array<std::string_view, 1> kGgufContainers{"gguf"};
// The generic decoder policy (the dense decoders' and Qwen3.5's): batched
// decode up to 16 rows with the shared runner's default lookahead and graph
// capture, paired MMQ dense products, and solo prompt chunks.
inline constexpr std::array kDensePolicies = {
    PolicyRule{.text = "max_decode_rows 16\nvector_rows 16\nproduct dense_pair\n",
               .prompts_alone = true},
};

// The arrays `parts`, one after another.
template <class T, std::size_t... N>
constexpr std::array<T, (N + ...)> Concat(const std::array<T, N>&... parts) {
  std::array<T, (N + ...)> out{};
  std::size_t next = 0;
  ((std::ranges::copy(parts, out.begin() + next), next += N), ...);
  return out;
}

// ---------------------------------------------------------------- overlays
//
// A container's form of a canonical rule table: one canonical table plus the
// container's overlay, as data (the dense dialects' tensor parts and
// parameter overrides are resolved the same way, spec/mapping.h
// SourceDialect): canonical rules matched by their identity (a tensor rule's
// slot, role and part; a parameter rule's slot, parameter and key), dropped
// or edited field by field, then the container's own rules appended. Every
// edit applies to at least one rule, or the table does not compile.

// A tensor rule's edit: of the rules of `slot` and `role` (and `part` when
// engaged), dropped, or with the fields given replaced.
struct TensorEdit {
  std::string_view slot{}, role{};
  std::optional<format::PartRole> part{};
  bool drop = false;
  std::string_view pattern{};  // nonempty: the container's spelling
  std::optional<format::PartRole> set_part{};
  std::optional<bool> optional{}, any_part{};
};
constexpr bool EditApplies(const TensorEdit& edit, const TensorRule& rule) {
  return edit.slot == rule.slot && edit.role == rule.role &&
         (!edit.part || *edit.part == rule.part);
}
constexpr TensorRule Edited(TensorRule rule, const TensorEdit& edit) {
  if (!edit.pattern.empty()) rule.pattern = edit.pattern;
  if (edit.set_part) rule.part = *edit.set_part;
  if (edit.optional) rule.optional = *edit.optional;
  if (edit.any_part) rule.any_part = *edit.any_part;
  return rule;
}

// A parameter rule's edit: of the rules of `slot` (empty: every slot's) and
// `param` (and of `key` when engaged: empty, a constant's), dropped, or with
// the fields given replaced, and `then` inserted after it.
struct ParamEdit {
  std::string_view slot{}, param{};
  std::optional<std::string_view> key{};
  bool drop = false;
  std::optional<std::string_view> set_key{}, numerator_key{}, divisor_key{}, tensor{}, text{};
  std::optional<Transform> transform{};
  std::optional<bool> optional{};
  std::optional<ParamRule> then{};
};
constexpr bool EditApplies(const ParamEdit& edit, const ParamRule& rule) {
  return !edit.param.empty() && (edit.slot.empty() || edit.slot == rule.slot) &&
         edit.param == rule.param && (!edit.key || *edit.key == rule.key);
}
constexpr ParamRule Edited(ParamRule rule, const ParamEdit& edit) {
  if (edit.set_key) rule.key = *edit.set_key;
  if (edit.numerator_key) rule.numerator_key = *edit.numerator_key;
  if (edit.divisor_key) rule.divisor_key = *edit.divisor_key;
  if (edit.tensor) rule.tensor = *edit.tensor;
  if (edit.text) rule.value.text = *edit.text;
  if (edit.transform) rule.transform = *edit.transform;
  if (edit.optional) rule.optional = *edit.optional;
  return rule;
}

inline constexpr std::array<TensorRule, 0> kNoTensorRules{};
inline constexpr std::array<ParamRule, 0> kNoParamRules{};

namespace overlay_detail {
template <class Rule, std::size_t N, std::size_t E, class Edit>
constexpr bool EveryEditApplies(const std::array<Rule, N>& base, const std::array<Edit, E>& edits) {
  return std::ranges::all_of(edits, [&](const Edit& edit) {
    return std::ranges::any_of(base, [&](const Rule& rule) { return EditApplies(edit, rule); });
  });
}
template <class Rule, std::size_t N, std::size_t E, class Edit>
constexpr std::size_t OverlaidSize(const std::array<Rule, N>& base,
                                   const std::array<Edit, E>& edits) {
  std::size_t size = 0;
  for (const Rule& rule : base) {
    bool dropped = false;
    std::size_t inserted = 0;
    for (const Edit& edit : edits) {
      if (!EditApplies(edit, rule)) continue;
      dropped = dropped || edit.drop;
      if constexpr (requires { edit.then; }) inserted += edit.then ? 1 : 0;
    }
    if (!dropped) size += 1 + inserted;
  }
  return size;
}
}  // namespace overlay_detail

// `kBase` (a std::array of TensorRule or ParamRule) with `kEdits` (of
// TensorEdit or ParamEdit) applied in order, then `kExtra`'s rules.
template <const auto& kBase, const auto& kEdits, const auto& kExtra>
constexpr auto Overlay() {
  static_assert(overlay_detail::EveryEditApplies(kBase, kEdits),
                "an overlay edit matches no canonical rule");
  using Rule = std::ranges::range_value_t<decltype(kBase)>;
  std::array<Rule, overlay_detail::OverlaidSize(kBase, kEdits) + kExtra.size()> out{};
  std::size_t next = 0;
  for (const Rule& base : kBase) {
    bool dropped = false;
    Rule rule = base;
    for (const auto& edit : kEdits) {
      if (!EditApplies(edit, base)) continue;
      dropped = dropped || edit.drop;
      rule = Edited(rule, edit);
    }
    if (dropped) continue;
    out[next++] = rule;
    if constexpr (std::is_same_v<Rule, ParamRule>)
      for (const auto& edit : kEdits)
        if (EditApplies(edit, base) && edit.then) out[next++] = *edit.then;
  }
  for (const Rule& rule : kExtra) out[next++] = rule;
  return out;
}

// The canonical table of a mapping whose dialects bind roles (set
// parameters) its base table lacks: `kBase`, then each rule of `kAdded`
// whose slot and role (slot and parameter) `kBase` has none of.
namespace overlay_detail {
constexpr bool SameIdentity(const TensorRule& a, const TensorRule& b) {
  return a.slot == b.slot && a.role == b.role;
}
constexpr bool SameIdentity(const ParamRule& a, const ParamRule& b) {
  return a.slot == b.slot && a.param == b.param;
}
template <class Rule, std::size_t N, std::size_t M>
constexpr std::size_t NewIdentities(const std::array<Rule, N>& base,
                                    const std::array<Rule, M>& added) {
  return static_cast<std::size_t>(std::ranges::count_if(added, [&](const Rule& rule) {
    return std::ranges::none_of(base, [&](const Rule& b) { return SameIdentity(b, rule); });
  }));
}
}  // namespace overlay_detail
template <const auto& kBase, const auto& kAdded>
constexpr auto WithNewIdentities() {
  using Rule = std::ranges::range_value_t<decltype(kBase)>;
  std::array<Rule, kBase.size() + overlay_detail::NewIdentities(kBase, kAdded)> out{};
  std::ranges::copy(kBase, out.begin());
  std::size_t next = kBase.size();
  for (const Rule& rule : kAdded)
    if (std::ranges::none_of(kBase,
                             [&](const Rule& b) { return overlay_detail::SameIdentity(b, rule); }))
      out[next++] = rule;
  return out;
}

inline constexpr std::array kGgufAuxiliary{
    AuxiliaryNamespace{"general.", "container provenance"},
    AuxiliaryNamespace{"tokenizer.", "tokenizer metadata retained for the writer"},
    AuxiliaryNamespace{"quantize.imatrix.", "source quantization calibration provenance"}};

inline constexpr std::array kSafetensorsAuxiliary{
    AuxiliaryNamespace{"architectures", "transformers class provenance"},
    AuxiliaryNamespace{"transformers_version", "transformers writer version provenance"},
    AuxiliaryNamespace{"attention_dropout", "training dropout configuration"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer bos token id"},
    AuxiliaryNamespace{"eos_token_id", "tokenizer eos token id"},
    AuxiliaryNamespace{"initializer_range", "weight initializer scale"},
    AuxiliaryNamespace{"max_window_layers", "sliding window layer count"},
    AuxiliaryNamespace{"sliding_window", "sliding window length"},
    AuxiliaryNamespace{"torch_dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"use_cache", "transformers generation cache default"},
    AuxiliaryNamespace{"vocab_size", "vocabulary size stated by embedding table",
                       names::kTokenEmbd.hf, 1},
    AuxiliaryNamespace{"layer_types", "transformer block layer types"},
    AuxiliaryNamespace{"text_config.layer_types", "text transformer block layer types"},
    AuxiliaryNamespace{"mlp_bias", "MLP projection biases: a bias tensor no rule reads is refused"},
};

// Gemma owns its mathematical keys rather than inheriting Qwen's auxiliary windows.
inline constexpr auto kGemmaHfAuxiliary = [] {
  std::array<AuxiliaryNamespace, kSafetensorsAuxiliary.size() - 2> out{};
  std::size_t next = 0;
  for (const auto& rule : kSafetensorsAuxiliary)
    if (rule.prefix != "sliding_window" && rule.prefix != "max_window_layers") out[next++] = rule;
  return out;
}();

inline constexpr std::array kNativeAuxiliary{
    AuxiliaryNamespace{"architectures", "transformers class provenance"},
    AuxiliaryNamespace{"transformers_version", "transformers writer version provenance"},
    AuxiliaryNamespace{"attention_dropout", "training dropout configuration"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer bos token id"},
    AuxiliaryNamespace{"eos_token_id", "tokenizer eos token id"},
    AuxiliaryNamespace{"text_config.eos_token_id", "tokenizer eos token id"},
    AuxiliaryNamespace{"initializer_range", "weight initializer scale"},
    AuxiliaryNamespace{"torch_dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"use_cache", "transformers generation cache default"},
    AuxiliaryNamespace{"tie_word_embeddings", "weight tying configuration"},
    AuxiliaryNamespace{"text_config.tie_word_embeddings", "weight tying configuration"},
    AuxiliaryNamespace{"vocab_size", "vocabulary size stated by embedding table"},
    AuxiliaryNamespace{"text_config.vocab_size", "vocabulary size stated by embedding table"},
    AuxiliaryNamespace{"vision_config.", "vision component outside this text-only import"},
    AuxiliaryNamespace{"audio_config.", "audio component outside this text-only import"},
    AuxiliaryNamespace{"quantization.", "checkpoint quantization metadata"},
    AuxiliaryNamespace{"full_attention_interval",
                       "the full-attention interval; the required layer_types states each "
                       "layer's type, as transformers reads it"},
    AuxiliaryNamespace{"mlp_only_layers",
                       "a MoE model's dense-FFN layers: such a layer's dense tensors have no "
                       "rule and are refused"},
    AuxiliaryNamespace{"mtp_num_hidden_layers",
                       "the MTP drafter's layers, outside the target (its tensors are refused "
                       "unless excluded)"},
    AuxiliaryNamespace{"mtp_use_dedicated_embeddings",
                       "the MTP drafter's embedding sharing, outside the target"},
    AuxiliaryNamespace{"vocab_size_per_layer_input",
                       "the per-layer input table's rows, stated by its tensor's extent"},
    AuxiliaryNamespace{"quantization_config.", "checkpoint quantization metadata"},
};

inline constexpr std::array kGemmaHfRequirements{
    ScalarRequirement{.key = "query_pre_attn_scalar", .positive = true}};
inline constexpr std::array kGemma2HfRequirements{
    kGemmaHfRequirements[0],
    ScalarRequirement{
        .key = "sliding_window_size", .equals_key = "sliding_window", .optional = true}};

// Gemma's transformers tensor names (a sandwich-norm block's), and Gemma 3's
// with its QK norms.
inline constexpr auto kGemmaHfTensors =
    Aliases(names::kTokenEmbd, names::kAttnNorm, names::kAttnQ, names::kAttnK, names::kAttnV,
            names::kAttnOutput, names::kAttnPostNorm, names::kFfnPreNorm, names::kFfnPostNorm,
            names::kFfnGate, names::kFfnUp, names::kFfnDown, names::kOutputNorm, names::kOutput);
inline constexpr auto kGemma3HfTensors =
    Concat(kGemmaHfTensors, Aliases(names::kAttnQNorm, names::kAttnKNorm));

inline constexpr std::array kGemmaSafetensorsKeys{
    KeyAlias{"rope_parameters.rope_theta", "rope_theta"},
    KeyAlias{"text_config.rope_parameters.rope_theta", "rope_theta"},
};

inline constexpr std::array kHfHeadPresence{
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = names::kOutput.gguf}};
inline constexpr std::array<std::string_view, 2> kHfIgnored{
    "model.rotary_emb.inv_freq", "model.layers.{layer}.self_attn.rotary_emb.inv_freq"};
inline constexpr std::array<std::string_view, 5> kGemmaHfNulls{
    "rope_scaling", "rope_parameters", "final_logit_softcapping", "attn_logit_softcapping",
    "use_bidirectional_attention"};
inline constexpr std::array<std::string_view, 19> kHfSemantic{"qwen2.",
                                                              "rope_*",
                                                              "rotary_*",
                                                              "quantization*",
                                                              "sliding_window*",
                                                              "attention_*",
                                                              "attn_*",
                                                              "partial_rotary_factor",
                                                              "softcap*",
                                                              "hidden_*",
                                                              "rms_norm_*",
                                                              "norm_*",
                                                              "num_*",
                                                              "activation*",
                                                              "query_*",
                                                              "head_dim",
                                                              "use_bidirectional_attention",
                                                              "attn_logit_softcapping",
                                                              "final_logit_softcapping"};

// A parameter from a source key; `optional`: left out when the key is
// absent and no later rule gives it.
constexpr ParamRule FromKey(std::string_view slot, std::string_view param, std::string_view key,
                            ValueType type, Transform transform = Transform::kNone,
                            std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from a source key, or if absent, a ratio of two keys;
// `optional`: left out when both the key and the ratio are absent and no later rule gives it.
constexpr ParamRule FromOrRatio(std::string_view slot, std::string_view param, std::string_view key,
                                std::string_view numerator_key, std::string_view divisor_key,
                                ValueType type, Transform transform = Transform::kNone,
                                std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from the ratio of two keys; `optional`: left out when absent and no later rule gives
// it.
constexpr ParamRule FromRatio(std::string_view slot, std::string_view param,
                              std::string_view numerator_key, std::string_view divisor_key,
                              ValueType type, Transform transform = Transform::kNone,
                              std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from the product of two keys.
constexpr ParamRule FromProduct(std::string_view slot, std::string_view param,
                                std::string_view numerator_key, std::string_view divisor_key,
                                ValueType type, std::string_view layer_types = {},
                                bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = Transform::kProduct,
          .optional = optional};
}

constexpr ParamRule ZeroCenteredNorm(std::string_view slot) {
  return {.slot = slot,
          .param = "norm_zero_centered",
          .type = ValueType::kBool,
          .value = {.flag = true},
          .unless_fold = "norm.plus_one"};
}

// A parameter from a key listing a value a layer (or one for all).
constexpr ParamRule PerLayer(std::string_view slot, std::string_view param, std::string_view key,
                             ValueType type, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = type,
          .per_layer = true};
}

// Constant parameters.
constexpr ParamRule Int(std::string_view slot, std::string_view param, std::int64_t value,
                        std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kInt,
          .value = {.integer = value}};
}
constexpr ParamRule Real(std::string_view slot, std::string_view param, double value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kReal,
          .value = {.real = value}};
}
constexpr ParamRule Flag(std::string_view slot, std::string_view param, bool value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kBool,
          .value = {.flag = value}};
}
constexpr ParamRule Name(std::string_view slot, std::string_view param, std::string_view value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kString,
          .value = {.text = value}};
}

// A requirement: the key's value is `value`, or (optional) the key is
// absent.
constexpr ParamRule Require(std::string_view key, std::string_view value, bool optional) {
  return {.key = key, .type = ValueType::kString, .value = {.text = value}, .optional = optional};
}
constexpr ParamRule RequireInt(std::string_view key, std::int64_t value, bool optional) {
  return {.key = key, .type = ValueType::kInt, .value = {.integer = value}, .optional = optional};
}
constexpr ParamRule RequireReal(std::string_view key, double value, bool optional) {
  return {.key = key, .type = ValueType::kReal, .value = {.real = value}, .optional = optional};
}

// `rule`, applying only when the source's `key` is the string `value`, or
// (an empty `value`) only when `key` is absent (ParamRule::when_key).
constexpr ParamRule When(ParamRule rule, std::string_view key, std::string_view value) {
  rule.when_key = key;
  rule.when_value = value;
  return rule;
}

// A boolean parameter set true when `tensor` is present in the source;
// absent when the tensor is absent, so a subsequent rule or optional omission
// applies.
constexpr ParamRule FromTensor(std::string_view slot, std::string_view param,
                               std::string_view tensor, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .tensor = tensor,
          .type = ValueType::kBool,
          .optional = true};
}

// An optional boolean parameter set true when `key` equals `value` (and left
// out when `key` is absent or `"none"`, or refused for any other string).
constexpr ParamRule KeyEquals(std::string_view slot, std::string_view param, std::string_view key,
                              std::string_view value, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = ValueType::kBool,
          .value = {.flag = true, .text = value},
          .optional = true};
}

// Shared native dialect data. Canonical architecture parameters occur once;
// source keys and exporter defaults do not redefine the component structure.
inline constexpr std::array kGemmaHfKeys{
    SourceKey{"block_count", "num_hidden_layers"},
    SourceKey{"context_length", "max_position_embeddings"},
    SourceKey{"embedding_length", "hidden_size"},
    SourceKey{"attention.layer_norm_rms_epsilon", "rms_norm_eps"},
    SourceKey{"attention.head_count", "num_attention_heads"},
    SourceKey{"attention.head_count_kv", "num_key_value_heads"},
    SourceKey{"attention.key_length", "head_dim"},
    SourceKey{"attention.value_length", "head_dim"},
    SourceKey{"rope.dimension_count", "head_dim"},
    SourceKey{"attention.sliding_window", "sliding_window"},
    SourceKey{"feed_forward_length", "intermediate_size"},
    SourceKey{"rope.freq_base", "rope_theta"},
    SourceKey{"rope.scaling.type", "rope_scaling.rope_type"},
    SourceKey{"attention.sliding_window_pattern", "sliding_window_pattern"},
};
inline constexpr ParamOverride kGemmaHfQueryScale{
    .slot = "attention",
    .param = "scale",
    .rule = FromKey("attention", "scale", "query_pre_attn_scalar", ValueType::kReal,
                    Transform::kRsqrt)};
inline constexpr std::array kGemmaHfParams{
    ZeroCenteredNorm("attention"),
    ZeroCenteredNorm("ffn"),
    ZeroCenteredNorm("final_norm"),
    Require("hidden_activation", "gelu_pytorch_tanh", true),
    ParamRule{.key = "attention_bias",
              .type = ValueType::kBool,
              .value = {.flag = false},
              .optional = true},
    Require("hidden_act", "gelu_pytorch_tanh", true),
    ParamRule{.key = "use_bidirectional_attention",
              .type = ValueType::kBool,
              .value = {.flag = false},
              .optional = true},
};
// A dense decoder's matrices, which a transformers checkpoint may keep
// whole or as codes beside their scales: bound with whatever part they hold.
inline constexpr std::array kHfMatrixParts{
    TensorPartOverride{"embedding", "table"},  TensorPartOverride{"attention", "q"},
    TensorPartOverride{"attention", "k"},      TensorPartOverride{"attention", "v"},
    TensorPartOverride{"attention", "output"}, TensorPartOverride{"ffn", "gate"},
    TensorPartOverride{"ffn", "up"},           TensorPartOverride{"ffn", "down"},
    TensorPartOverride{"output", "weight"},
};
inline constexpr std::array kGemmaGgufFolds{FoldRule{.fold = "norm.plus_one", .container = "gguf"}};
inline constexpr std::array kGemmaHfFolds{
    FoldRule{.fold = "norm.plus_one", .container = "safetensors", .imported = true}};
inline constexpr std::array<std::string_view, 2> kGemmaContainers{"gguf", "safetensors"};

// ------------------------------------------------------- drafter values
//
// A drafter's extraction rules (spec/mapping.h ValueRule, read where its own
// source lacks a key) inverted from the forward rules it shares with its
// target: each source key a rule of `kParams` reads as the parameter itself
// (one key, no transform, no gate) for a slot of `kSlots` (kComponent: its
// target's last component of the slot's kind, that parameter) or for the
// stage or stream (kContext, kWidth, kStreams), from the key's first such
// rule; and after it, each key a requirement of `kRequirements` requires
// equal to that key (the same origin). Binding facts (llmp. keys) are the
// drafter's own rules, never a target's.
namespace overlay_detail {
template <std::size_t S>
constexpr std::optional<ValueRule> Inverted(const ParamRule& rule,
                                            const std::array<SlotRule, S>& slots) {
  if (rule.key.empty() || rule.key.starts_with("llmp.") || !rule.numerator_key.empty() ||
      !rule.divisor_key.empty() || !rule.tensor.empty() || rule.transform != Transform::kNone ||
      !rule.when_key.empty() || !rule.unless_fold.empty())
    return std::nullopt;
  if (rule.slot == kStageSlot && rule.param == "context")
    return ValueRule{.key = rule.key, .origin = ValueOrigin::kContext};
  if (rule.slot == kStreamSlot && rule.param == "width")
    return ValueRule{.key = rule.key, .origin = ValueOrigin::kWidth};
  if (rule.slot == kStreamSlot && rule.param == "streams")
    return ValueRule{.key = rule.key, .origin = ValueOrigin::kStreams};
  for (const SlotRule& slot : slots)
    if (slot.slot == rule.slot)
      return ValueRule{.key = rule.key,
                       .origin = ValueOrigin::kComponent,
                       .kind = slot.kind,
                       .param = rule.param,
                       .list = rule.per_layer};
  return std::nullopt;
}
template <std::size_t P, std::size_t S, std::size_t R>
constexpr std::pair<std::array<ValueRule, P + R>, std::size_t> InvertAll(
    const std::array<ParamRule, P>& params, const std::array<SlotRule, S>& slots,
    const std::array<ScalarRequirement, R>& requirements) {
  std::array<ValueRule, P + R> out{};
  std::size_t n = 0;
  const auto seen = [&](std::string_view key) {
    return std::ranges::any_of(out.begin(), out.begin() + n,
                               [&](const ValueRule& v) { return v.key == key; });
  };
  for (const ParamRule& rule : params) {
    const auto value = Inverted(rule, slots);
    if (!value || seen(value->key)) continue;
    out[n++] = *value;
    for (const ScalarRequirement& r : requirements)
      if (r.equals_key == value->key && !seen(r.key)) {
        out[n] = *value;
        out[n++].key = r.key;
      }
  }
  return {out, n};
}
}  // namespace overlay_detail
template <const auto& kParams, const auto& kSlots, const auto& kRequirements>
constexpr auto InvertedValues() {
  constexpr auto kAll = overlay_detail::InvertAll(kParams, kSlots, kRequirements);
  std::array<ValueRule, kAll.second> out{};
  std::ranges::copy(kAll.first.begin(), kAll.first.begin() + kAll.second, out.begin());
  return out;
}

// A profile's value under a source key: integers as integers, floats as the
// reals they are.
inline SourceValue Value(std::string_view key, std::uint32_t v) {
  return {.key = std::string(key), .value = ParamValue(static_cast<std::int64_t>(v))};
}
inline SourceValue Value(std::string_view key, float v) {
  return {.key = std::string(key), .value = ParamValue(static_cast<double>(v))};
}
inline SourceValue Value(std::string_view key, std::string_view v) {
  return {.key = std::string(key), .value = ParamValue(std::string(v))};
}
// A list of integers (GGUF's arrays of integers or flags).
inline SourceValue Value(std::string_view key, std::vector<std::int64_t> v) {
  return {.key = std::string(key), .value = ParamValue(std::move(v))};
}
// A flag (a source's fact, an approval's gate).
inline SourceValue FlagValue(std::string_view key, bool v) {
  return {.key = std::string(key), .value = ParamValue(v)};
}

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_RULES_H_
