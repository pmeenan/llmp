// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/source_mapping.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <string_view>

#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace nd = normalization_detail;
namespace {
using ErrorResult = std::unexpected<Error>;
ErrorResult Invalid() {
  return ErrorResult(Error{.rule = Rule::kUnsupported,
                           .reason = "invalid or unsupported source normalization table"});
}
constexpr std::array kAuxiliary{
    AuxiliaryNamespace{"general.", "container provenance"},
    AuxiliaryNamespace{"tokenizer.", "tokenizer metadata retained for the writer"},
    AuxiliaryNamespace{"quantize.imatrix.", "source quantization calibration provenance"}};
constexpr std::array kGemma2Requirements{ScalarRequirement{
    .key = "gemma2.block_count", .reject_equal = 46, .unless_present = "gemma2.attention.scale"}};
constexpr std::array kGemma3Requirements{ScalarRequirement{
    .key = "gemma3.block_count", .reject_equal = 62, .unless_present = "gemma3.attention.scale"}};
constexpr std::array kGemmaHfTensors{
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.layers.{layer}.input_layernorm.weight", "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.post_attention_norm.weight"},
    TensorAlias{"model.layers.{layer}.pre_feedforward_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.post_feedforward_layernorm.weight",
                "blk.{layer}.post_ffw_norm.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.weight", "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.weight", "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.weight", "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.norm.weight", "output_norm.weight"},
    TensorAlias{"lm_head.weight", "output.weight"},
};
constexpr auto kGemma3HfTensors = [] {
  std::array<TensorAlias, kGemmaHfTensors.size() + 2> aliases{};
  std::ranges::copy(kGemmaHfTensors, aliases.begin());
  aliases[kGemmaHfTensors.size()] = {"model.layers.{layer}.self_attn.q_norm.weight",
                                     "blk.{layer}.attn_q_norm.weight"};
  aliases[kGemmaHfTensors.size() + 1] = {"model.layers.{layer}.self_attn.k_norm.weight",
                                         "blk.{layer}.attn_k_norm.weight"};
  return aliases;
}();
constexpr std::array kGemmaHfRequirements{
    ScalarRequirement{.key = "query_pre_attn_scalar", .positive = true}};
constexpr std::array kQwen2HfTensors{
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.layers.{layer}.input_layernorm.weight", "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.bias", "blk.{layer}.attn_q.bias"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.bias", "blk.{layer}.attn_k.bias"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.bias", "blk.{layer}.attn_v.bias"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.weight", "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.weight", "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.weight", "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.norm.weight", "output_norm.weight"},
    TensorAlias{"lm_head.weight", "output.weight"},
};
// Marker names select the reader's codebook; canonical binding names do not.
// 3inst has no marker, so the corresponding mapping bindings are optional.
constexpr std::array kQwen2Exl3Tensors{
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.mul1",
                "model.layers.{layer}.self_attn.q_proj.mcg"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.mul1",
                "model.layers.{layer}.self_attn.k_proj.mcg"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.mul1",
                "model.layers.{layer}.self_attn.v_proj.mcg"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.mul1",
                "model.layers.{layer}.self_attn.o_proj.mcg"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.mul1",
                "model.layers.{layer}.mlp.gate_proj.mcg"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.mul1", "model.layers.{layer}.mlp.up_proj.mcg"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.mul1",
                "model.layers.{layer}.mlp.down_proj.mcg"},
    TensorAlias{"lm_head.mul1", "lm_head.mcg"},
};
constexpr std::array kQwen2SafetensorsKeys{
    KeyAlias{"head_dim", "qwen2.attention.key_length"},
    KeyAlias{"hidden_act", "qwen2.hidden_act"},
    KeyAlias{"hidden_size", "qwen2.embedding_length"},
    KeyAlias{"intermediate_size", "qwen2.feed_forward_length"},
    KeyAlias{"max_position_embeddings", "qwen2.context_length"},
    KeyAlias{"num_attention_heads", "qwen2.attention.head_count"},
    KeyAlias{"num_hidden_layers", "qwen2.block_count"},
    KeyAlias{"num_key_value_heads", "qwen2.attention.head_count_kv"},
    KeyAlias{"rms_norm_eps", "qwen2.attention.layer_norm_rms_epsilon"},
    KeyAlias{"rope_theta", "qwen2.rope.freq_base"},
    KeyAlias{"rope_parameters.rope_theta", "qwen2.rope.freq_base"},
    KeyAlias{"use_sliding_window", "qwen2.use_sliding_window"},
};
constexpr std::array kSafetensorsAuxiliary{
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
                       "model.embed_tokens.weight", 1},
    AuxiliaryNamespace{"layer_types", "transformer block layer types"},
    AuxiliaryNamespace{"text_config.layer_types", "text transformer block layer types"},
};

// Gemma owns its mathematical keys rather than inheriting Qwen's auxiliary windows.
constexpr auto kGemmaHfAuxiliary = [] {
  std::array<AuxiliaryNamespace, kSafetensorsAuxiliary.size() - 2> out{};
  std::size_t next = 0;
  for (const auto& rule : kSafetensorsAuxiliary)
    if (rule.prefix != "sliding_window" && rule.prefix != "max_window_layers") out[next++] = rule;
  return out;
}();
// transformers' tying flag: untied, the checkpoint must hold its own head
// (the output rule would otherwise fall back to the token table). EXL3
// quantizes the head even when tied, so its tables require it anyway; the
// flag still selects, and an untied one without the head is refused.
constexpr std::array kHfHeadPresence{
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = "output.weight"}};
constexpr std::array kExl3HeadPresence{
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = "lm_head.trellis"}};
constexpr std::array kGemmaConditionalHeadPresence{
    TensorPresenceRequirement{.flag = "text_config.tie_word_embeddings", .tensor = "output.weight"},
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = "output.weight"}};
constexpr std::array kGemma2HfRequirements{
    kGemmaHfRequirements[0],
    ScalarRequirement{
        .key = "sliding_window_size", .equals_key = "sliding_window", .optional = true}};
constexpr std::array<std::string_view, 5> kGemmaHfNulls{
    "rope_scaling", "rope_parameters", "final_logit_softcapping", "attn_logit_softcapping",
    "use_bidirectional_attention"};
constexpr std::array<std::string_view, 5> kGemma3ConditionalNulls{
    "text_config.rope_scaling", "text_config.rope_parameters",
    "text_config.final_logit_softcapping", "text_config.attn_logit_softcapping",
    "text_config.use_bidirectional_attention"};
constexpr std::array kGemma3ConditionalRequirements{
    ScalarRequirement{.key = "text_config.query_pre_attn_scalar", .positive = true}};
constexpr auto kGemma3ConditionalAuxiliary = [] {
  std::array<AuxiliaryNamespace, kGemmaHfAuxiliary.size() + 7> out{};
  std::ranges::copy(kGemmaHfAuxiliary, out.begin());
  // Root tokenizer/provenance fields are kept by the common declarations above.
  auto i = kGemmaHfAuxiliary.size();
  out[i++] = {"vision_config.", "vision component outside this text-only import"};
  out[i++] = {"text_config.initializer_range", "weight initializer scale"};
  out[i++] = {"text_config.attention_dropout", "training dropout configuration"};
  out[i++] = {"text_config.cache_implementation", "transformers cache implementation"};
  out[i++] = {"text_config.torch_dtype", "checkpoint nominal dtype"};
  out[i++] = {"text_config.use_cache", "transformers generation cache default"};
  out[i] = {"text_config.vocab_size", "vocabulary stated by embedding table",
            "language_model.model.embed_tokens.weight", 1};
  return out;
}();
constexpr std::array<std::string_view, 2> kGemma3Excluded{"vision_tower.",
                                                          "multi_modal_projector."};
constexpr std::array<std::string_view, 1> kGemma3ConditionalSemantic{"text_config."};
constexpr auto kGemma3ConditionalTensors = [] {
  auto tensors = kGemma3HfTensors;
  for (auto& t : tensors)
    if (t.source == "model.embed_tokens.weight")
      t.source = "language_model.model.embed_tokens.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.input_layernorm.weight")
      t.source = "language_model.model.layers.{layer}.input_layernorm.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.self_attn.q_proj.weight")
      t.source = "language_model.model.layers.{layer}.self_attn.q_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.self_attn.k_proj.weight")
      t.source = "language_model.model.layers.{layer}.self_attn.k_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.self_attn.v_proj.weight")
      t.source = "language_model.model.layers.{layer}.self_attn.v_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.self_attn.o_proj.weight")
      t.source = "language_model.model.layers.{layer}.self_attn.o_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.post_attention_layernorm.weight")
      t.source = "language_model.model.layers.{layer}.post_attention_layernorm.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.pre_feedforward_layernorm.weight")
      t.source = "language_model.model.layers.{layer}.pre_feedforward_layernorm.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.post_feedforward_layernorm.weight")
      t.source = "language_model.model.layers.{layer}.post_feedforward_layernorm.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.mlp.gate_proj.weight")
      t.source = "language_model.model.layers.{layer}.mlp.gate_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.mlp.up_proj.weight")
      t.source = "language_model.model.layers.{layer}.mlp.up_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.mlp.down_proj.weight")
      t.source = "language_model.model.layers.{layer}.mlp.down_proj.weight";
  for (auto& t : tensors)
    if (t.source == "model.norm.weight") t.source = "language_model.model.norm.weight";
  for (auto& t : tensors)
    if (t.source == "lm_head.weight") t.source = "language_model.lm_head.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.self_attn.q_norm.weight")
      t.source = "language_model.model.layers.{layer}.self_attn.q_norm.weight";
  for (auto& t : tensors)
    if (t.source == "model.layers.{layer}.self_attn.k_norm.weight")
      t.source = "language_model.model.layers.{layer}.self_attn.k_norm.weight";
  return tensors;
}();
constexpr std::array kQwenRequirements{ScalarRequirement{
    .key = "qwen4exp.attention.value_length", .equals_key = "qwen4exp.attention.key_length"}};
constexpr std::array kQwenAuxiliary{
    kAuxiliary[0], kAuxiliary[1], kAuxiliary[2],
    AuxiliaryNamespace{"qwen4exp.ssm.inner_size",
                       "linear value width stated by its output projection", "blk.0.ssm_out.weight",
                       0},
    // The text-only PLE hashes use eos_token_id. This marks an image input
    // placeholder in the checkpoint; no image input path is admitted here.
    AuxiliaryNamespace{"qwen4exp.ple.image_token_id",
                       "image input placeholder outside the text pipeline"}};
constexpr std::array kQwenLists{
    ListRequirement{.key = "qwen4exp.ple.layers", .size = nd::kLayerLimit, .min_size = 1},
    ListRequirement{.key = "qwen4exp.rope.dimension_sections",
                    .size = 4,
                    .sections_of = "qwen4exp.rope.dimension_count"},
    ListRequirement{.key = "qwen4exp.attention.compress_ratios",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .per_layer = true},
    ListRequirement{
        .key = "qwen4exp.ple.layer_multipliers", .size = nd::kLayerLimit, .min_size = 1},
    ListRequirement{.key = "qwen4exp.ple.head_offsets", .size = nd::kLayerLimit, .min_size = 1},
    ListRequirement{
        .key = "qwen4exp.ple.head_vocab_sizes", .size = nd::kLayerLimit, .min_size = 1}};
constexpr std::array kQwenRows{RowIndexRequirement{
    "qwen4exp.ple.head_offsets", "qwen4exp.ple.head_vocab_sizes", "per_layer_token_embd.weight"}};
constexpr std::array kDsv4Requirements{ScalarRequirement{
    .key = "deepseek4.attention.value_length", .equals_key = "deepseek4.attention.key_length"}};
// antirez's GGUFs state the checkpoint's MTP depth, whose layers' tensors
// are not imported (a source holding any is refused, their names unmapped),
// and its vocabulary, which the token table's shape states.
constexpr std::array kDsv4Auxiliary{
    kAuxiliary[0], kAuxiliary[1], kAuxiliary[2],
    AuxiliaryNamespace{"deepseek4.nextn_predict_layers", "MTP layers the import does not hold"},
    AuxiliaryNamespace{"deepseek4.vocab_size", "the vocabulary the token table states",
                       "token_embd.weight", 1}};
// The layer types may list the MTP layers' after the 43 blocks'; the shared
// expert's clamps may be absent (antirez's GGUFs), when the mapping clamps
// it as the routed experts.
constexpr std::array kDsv4Lists{
    ListRequirement{.key = "deepseek4.attention.compress_ratios",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .per_layer = true,
                    .extra_layers_key = "deepseek4.nextn_predict_layers",
                    .allow_unstated_extra = true},
    ListRequirement{.key = "deepseek4.swiglu_clamp_exp",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .per_layer = true},
    ListRequirement{.key = "deepseek4.swiglu_clamp_shexp",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .optional = true,
                    .per_layer = true}};
constexpr std::array kDsparkRequirements{
    ScalarRequirement{.key = "dflash.attention.value_length",
                      .equals_key = "dflash.attention.key_length"},
    ScalarRequirement{.key = "dflash.block_size", .positive = true}};
constexpr std::array kDsparkLists{
    ListRequirement{.key = "dflash.attention.compress_ratios",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .per_layer = true},
    ListRequirement{.key = "dflash.swiglu_clamp_exp",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .per_layer = true},
    ListRequirement{.key = "dflash.swiglu_clamp_shexp",
                    .size = nd::kLayerLimit,
                    .min_size = 1,
                    .per_layer = true},
    ListRequirement{.key = "dflash.target_layers", .size = nd::kLayerLimit, .min_size = 1}};
constexpr std::array<std::string_view, 2> kHfIgnored{
    "model.rotary_emb.inv_freq", "model.layers.{layer}.self_attn.rotary_emb.inv_freq"};
constexpr std::array<std::string_view, 2> kHfNullSemantics{"rope_scaling", "rope_parameters"};
constexpr std::array<std::string_view, 19> kHfSemantic{"qwen2.",
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
constexpr std::array kLlamaAuxiliary{
    kAuxiliary[0], kAuxiliary[1], kAuxiliary[2],
    AuxiliaryNamespace{"llama.vocab_size", "the vocabulary the token table states",
                       "token_embd.weight", 1}};
constexpr std::array kMistralAuxiliary{
    kAuxiliary[0], kAuxiliary[1], kAuxiliary[2],
    AuxiliaryNamespace{"mistral.vocab_size", "the vocabulary the token table states",
                       "token_embd.weight", 1}};
#define LLMP_DENSE_SAFETENSORS_KEYS(P)                                                    \
  KeyAlias{"head_dim", P "attention.key_length"}, KeyAlias{"hidden_act", P "hidden_act"}, \
      KeyAlias{"hidden_size", P "embedding_length"},                                      \
      KeyAlias{"intermediate_size", P "feed_forward_length"},                             \
      KeyAlias{"max_position_embeddings", P "context_length"},                            \
      KeyAlias{"num_attention_heads", P "attention.head_count"},                          \
      KeyAlias{"num_hidden_layers", P "block_count"},                                     \
      KeyAlias{"num_key_value_heads", P "attention.head_count_kv"},                       \
      KeyAlias{"rms_norm_eps", P "attention.layer_norm_rms_epsilon"},                     \
      KeyAlias{"rope_theta", P "rope.freq_base"},                                         \
      KeyAlias{"sliding_window", P "attention.sliding_window"},                           \
      KeyAlias{"attention_bias", P "attention_bias"},                                     \
      KeyAlias{"rope_scaling.rope_type", P "rope.scaling.type"},                          \
      KeyAlias{"rope_scaling.type", P "rope.scaling.type"},                               \
      KeyAlias{"rope_scaling.factor", P "rope.scaling.factor"},                           \
      KeyAlias{"rope_scaling.original_max_position_embeddings",                           \
               P "rope.scaling.original_context_length"},                                 \
      KeyAlias{"rope_scaling.beta_fast", P "rope.scaling.yarn_beta_fast"},                \
      KeyAlias{"rope_scaling.beta_slow", P "rope.scaling.yarn_beta_slow"},                \
      KeyAlias{"rope_parameters.rope_theta", P "rope.freq_base"},                         \
      KeyAlias{"rope_parameters.rope_type", P "rope.scaling.type"},                       \
      KeyAlias{"rope_parameters.factor", P "rope.scaling.factor"}
constexpr std::array kLlamaSafetensorsKeys{LLMP_DENSE_SAFETENSORS_KEYS("llama.")};
constexpr std::array kMistralSafetensorsKeys{LLMP_DENSE_SAFETENSORS_KEYS("mistral.")};
#undef LLMP_DENSE_SAFETENSORS_KEYS
constexpr std::array kGemma2SafetensorsKeys{
    KeyAlias{"rope_parameters.rope_theta", "rope_theta"},
};
constexpr std::array kGemma3SafetensorsKeys{
    KeyAlias{"rope_parameters.rope_theta", "rope_theta"},
};
constexpr std::array kGemma3ConditionalKeys{
    KeyAlias{"text_config.rope_parameters.rope_theta", "text_config.rope_theta"},
    KeyAlias{"rope_parameters.rope_theta", "text_config.rope_theta"},
};
constexpr std::array<std::string_view, 3> kDenseHfNulls{"rope_scaling", "rope_parameters",
                                                        "sliding_window"};

bool SameShape(std::string_view a, std::string_view b) {
  return (a.find(spec::kLayerField) == std::string_view::npos) ==
         (b.find(spec::kLayerField) == std::string_view::npos);
}
// Bounded finite pattern language: at most 1024 decimal layer values. Prefix
// and suffix rejection makes unrelated compiled rules constant-time.
bool Overlap(std::string_view a, std::string_view b) {
  const auto at = a.find(spec::kLayerField), bt = b.find(spec::kLayerField);
  if (at == std::string_view::npos) return nd::Match(b, a, nd::kLayerLimit).has_value();
  if (bt == std::string_view::npos) return nd::Match(a, b, nd::kLayerLimit).has_value();
  const auto ap = a.substr(0, at), bp = b.substr(0, bt);
  const auto as = a.substr(at + spec::kLayerField.size());
  const auto bs = b.substr(bt + spec::kLayerField.size());
  if ((!ap.starts_with(bp) && !bp.starts_with(ap)) || (!as.ends_with(bs) && !bs.ends_with(as)))
    return false;
  std::array<char, spec::kMaxText> text{};
  for (std::uint32_t layer = 0; layer < nd::kLayerLimit; ++layer) {
    auto* end = std::copy(ap.begin(), ap.end(), text.begin());
    end = std::to_chars(end, text.data() + text.size(), layer).ptr;
    end = std::copy(as.begin(), as.end(), end);
    if (nd::Match(b, {text.data(), static_cast<std::size_t>(end - text.data())}, nd::kLayerLimit))
      return true;
  }
  return false;
}
bool Structural(std::string_view key) {
  return key == "general.architecture" || key == "general.alignment" || key == "split.no" ||
         key == "split.count" || key == "split.tensors.count" || key == "model_type";
}
bool IsName(std::string_view s) {
  return !s.empty() && s.size() <= spec::kMaxText && (s.front() >= 'a' && s.front() <= 'z') &&
         std::ranges::all_of(s, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
         });
}
}  // namespace

const SourceMapping* FindSourceMapping(std::string_view container, std::string_view architecture) {
  static const std::array tables{
      SourceMapping{.container = "safetensors",
                    .architecture = &spec::mappings::Gemma3ConditionalSafetensorsMapping(),
                    .tensors = kGemma3ConditionalTensors,
                    .keys = kGemma3ConditionalKeys,
                    .auxiliary = kGemma3ConditionalAuxiliary,
                    .requirements = kGemma3ConditionalRequirements,
                    .semantic = kGemma3ConditionalSemantic,
                    .null_semantics = kGemma3ConditionalNulls,
                    .ignored_tensor_prefixes = kGemma3Excluded,
                    .tensor_presence = kGemmaConditionalHeadPresence},
      SourceMapping{.container = "safetensors",
                    .architecture = &spec::mappings::Gemma2SafetensorsMapping(),
                    .tensors = kGemmaHfTensors,
                    .keys = kGemma2SafetensorsKeys,
                    .auxiliary = kGemmaHfAuxiliary,
                    .requirements = kGemma2HfRequirements,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kGemmaHfNulls,
                    .tensor_presence = kHfHeadPresence},
      SourceMapping{.container = "safetensors",
                    .architecture = &spec::mappings::Gemma3SafetensorsMapping(),
                    .tensors = kGemma3HfTensors,
                    .keys = kGemma3SafetensorsKeys,
                    .auxiliary = kGemmaHfAuxiliary,
                    .requirements = kGemmaHfRequirements,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kGemmaHfNulls,
                    .tensor_presence = kHfHeadPresence},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Gemma2Mapping(),
                    .auxiliary = kAuxiliary,
                    .requirements = kGemma2Requirements},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Gemma3Mapping(),
                    .auxiliary = kAuxiliary,
                    .requirements = kGemma3Requirements},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Gemma4Mapping(),
                    .auxiliary = kAuxiliary},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Gemma4AssistantMapping(),
                    .auxiliary = kAuxiliary},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Qwen2GgufMapping(),
                    .auxiliary = kAuxiliary},
      SourceMapping{.container = "exl3",
                    .architecture = &spec::mappings::Qwen2Exl3Mapping(),
                    .tensors = kQwen2Exl3Tensors,
                    .keys = kQwen2SafetensorsKeys,
                    .auxiliary = kSafetensorsAuxiliary,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kHfNullSemantics,
                    .tensor_presence = kExl3HeadPresence},
      SourceMapping{
          .container = "safetensors",
          .architecture = &spec::mappings::Qwen2SafetensorsMapping(),
          .tensors = kQwen2HfTensors,
          .keys = kQwen2SafetensorsKeys,
          .auxiliary = kSafetensorsAuxiliary,
          .semantic = kHfSemantic,
          .ignored_tensors = kHfIgnored,
          .null_semantics = kHfNullSemantics,
          .tensor_presence = kHfHeadPresence,
      },
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::LlamaGgufMapping(),
                    .auxiliary = kLlamaAuxiliary},
      SourceMapping{.container = "exl3",
                    .architecture = &spec::mappings::LlamaExl3Mapping(),
                    .tensors = kQwen2Exl3Tensors,
                    .keys = kLlamaSafetensorsKeys,
                    .auxiliary = kGemmaHfAuxiliary,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kDenseHfNulls,
                    .tensor_presence = kExl3HeadPresence},
      SourceMapping{.container = "safetensors",
                    .architecture = &spec::mappings::LlamaSafetensorsMapping(),
                    .tensors = kQwen2HfTensors,
                    .keys = kLlamaSafetensorsKeys,
                    .auxiliary = kGemmaHfAuxiliary,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kDenseHfNulls,
                    .tensor_presence = kHfHeadPresence},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::MistralGgufMapping(),
                    .auxiliary = kMistralAuxiliary},
      SourceMapping{.container = "exl3",
                    .architecture = &spec::mappings::MistralExl3Mapping(),
                    .tensors = kQwen2Exl3Tensors,
                    .keys = kMistralSafetensorsKeys,
                    .auxiliary = kGemmaHfAuxiliary,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kDenseHfNulls,
                    .tensor_presence = kExl3HeadPresence},
      SourceMapping{.container = "safetensors",
                    .architecture = &spec::mappings::MistralSafetensorsMapping(),
                    .tensors = kQwen2HfTensors,
                    .keys = kMistralSafetensorsKeys,
                    .auxiliary = kGemmaHfAuxiliary,
                    .semantic = kHfSemantic,
                    .ignored_tensors = kHfIgnored,
                    .null_semantics = kDenseHfNulls,
                    .tensor_presence = kHfHeadPresence},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Qwen38GgufMapping(),
                    .auxiliary = kQwenAuxiliary,
                    .requirements = kQwenRequirements,
                    .lists = kQwenLists,
                    .extended_components = true,
                    .row_indices = kQwenRows},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::Dsv4Mapping(),
                    .auxiliary = kDsv4Auxiliary,
                    .requirements = kDsv4Requirements,
                    .lists = kDsv4Lists,
                    .extended_components = true},
      SourceMapping{.container = "gguf",
                    .architecture = &spec::mappings::DsparkMapping(),
                    .auxiliary = kAuxiliary,
                    .requirements = kDsparkRequirements,
                    .lists = kDsparkLists,
                    .extended_components = true}};
  for (const auto& table : tables)
    if (container == table.container && architecture == table.architecture->architecture)
      return &table;
  return nullptr;
}

namespace normalization_detail {
std::uint8_t KeyTypes(const spec::Mapping& m, std::string_view key) {
  std::uint8_t types = 0;
  bool seen = false;
  const auto add = [&](std::uint8_t allowed) {
    types = seen ? static_cast<std::uint8_t>(types & allowed) : allowed;
    seen = true;
  };
  if (key == m.layer_count_key) add(kInteger);
  if (!m.layer_type_key.empty() && key == m.layer_type_key) add(kIntegers);
  if (!m.layer_flags_key.empty() && key == m.layer_flags_key) add(kIntegers);
  for (const auto& rule : m.params) {
    if (!key.empty() && (key == rule.numerator_key || key == rule.divisor_key)) {
      add(kInteger | kReal);
    }
    if (key != rule.key || rule.key.empty()) continue;
    switch (rule.type) {
      case spec::ValueType::kInt:
        add(rule.per_layer ? kInteger | kIntegers : kInteger);
        break;
      case spec::ValueType::kReal:
        add(rule.per_layer ? kInteger | kReal | kReals : kInteger | kReal);
        break;
      case spec::ValueType::kBool:
        add(!rule.value.text.empty() ? kText : kFlag);
        break;
      case spec::ValueType::kString:
        add(kText);
        break;
      case spec::ValueType::kInts:
        add(kIntegers);
        break;
    }
  }
  for (const auto& slot : m.slots)
    if (!slot.layers_key.empty() && key == slot.layers_key) add(kIntegers);
  for (const auto& read : m.reads)
    if (!read.layers_key.empty() && key == read.layers_key) add(kIntegers);
  for (const auto& rule : m.slots)
    if ((!rule.gate.empty() && key == rule.gate) ||
        (!rule.count_key.empty() && key == rule.count_key))
      add(kInteger);
  for (const auto& rule : m.policies)
    if (!rule.gate.empty() && key == rule.gate) add(kInteger);
  return types;
}

bool Pattern(std::string_view pattern) {
  if (pattern.empty() || pattern.size() > spec::kMaxText - 4) return false;
  const auto field = pattern.find(spec::kLayerField);
  if (field == std::string_view::npos) return spec::IsSourceKey(pattern);
  if (pattern.find(spec::kLayerField, field + spec::kLayerField.size()) != std::string_view::npos)
    return false;
  // IsSourceKey excludes braces; inspect the two literal pieces directly.
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    if (i == field) {
      i += spec::kLayerField.size() - 1;
      continue;
    }
    const char c = pattern[i];
    if (c <= 32 || c >= 127 || c == '"' || c == '\\' || c == '{' || c == '}') return false;
  }
  return true;
}

std::optional<NameMatch> Match(std::string_view pattern, std::string_view name,
                               std::uint32_t layers) {
  const auto at = pattern.find(spec::kLayerField);
  if (at == std::string_view::npos)
    return name == pattern ? std::optional<NameMatch>(NameMatch{}) : std::nullopt;
  const auto prefix = pattern.substr(0, at), suffix = pattern.substr(at + spec::kLayerField.size());
  if (name.size() <= prefix.size() + suffix.size() || !name.starts_with(prefix) ||
      !name.ends_with(suffix))
    return std::nullopt;
  const auto digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  if (digits.size() > 4 || (digits.size() > 1 && digits.front() == '0')) return std::nullopt;
  std::uint32_t layer = 0;
  const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
  if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || layer >= layers)
    return std::nullopt;
  return NameMatch{digits};
}
}  // namespace normalization_detail

std::expected<void, Error> ValidateSourceMapping(const SourceMapping& table) {
  if ((table.container != "gguf" && table.container != "exl3" &&
       table.container != "safetensors") ||
      table.version == 0 || !table.architecture || table.route != RouteOperation::kCopyPart ||
      table.tensors.size() > nd::kRuleLimit || table.keys.size() > 256 ||
      table.auxiliary.size() > nd::kRuleLimit || table.requirements.size() > 16 ||
      table.ignored_tensor_prefixes.size() > 8 || table.tensor_presence.size() > 8 ||
      ((table.container == "exl3" || table.container == "safetensors") && table.auxiliary.empty()))
    return Invalid();
  const auto& m = *table.architecture;
  const auto* registered = FindSourceMapping(table.container, m.architecture);
  if (!registered || registered->architecture != &m ||
      table.extended_components != registered->extended_components ||
      table.lists.size() != registered->lists.size() ||
      table.shapes.size() != registered->shapes.size() ||
      !std::ranges::equal(table.row_indices, registered->row_indices) ||
      !std::ranges::equal(table.encodings, registered->encodings) ||
      !std::ranges::equal(table.semantic, registered->semantic) ||
      !std::ranges::equal(table.ignored_tensors, registered->ignored_tensors) ||
      !std::ranges::equal(table.null_semantics, registered->null_semantics) ||
      !std::ranges::equal(table.ignored_tensor_prefixes, registered->ignored_tensor_prefixes) ||
      table.tensor_presence.size() != registered->tensor_presence.size())
    return Invalid();
  for (std::size_t i = 0; i < table.tensor_presence.size(); ++i) {
    const auto& rule = table.tensor_presence[i];
    const auto& expected = registered->tensor_presence[i];
    if (rule.flag != expected.flag || rule.tensor != expected.tensor || rule.when != expected.when)
      return Invalid();
  }
  for (std::size_t i = 0; i < table.shapes.size(); ++i) {
    const auto& a = table.shapes[i];
    const auto& b = registered->shapes[i];
    if (a.name != b.name || !std::ranges::equal(a.shape, b.shape)) return Invalid();
  }
  for (std::size_t i = 0; i < table.lists.size(); ++i) {
    const auto& a = table.lists[i];
    const auto& b = registered->lists[i];
    if (a.key != b.key || a.size != b.size || a.size > nd::kLayerLimit ||
        a.min_size != b.min_size || a.min_size > a.size || a.optional != b.optional ||
        a.per_layer != b.per_layer || a.extra_layers_key != b.extra_layers_key ||
        a.allow_unstated_extra != b.allow_unstated_extra || a.sections_of != b.sections_of ||
        (!a.sections_of.empty() && (nd::KeyTypes(m, a.sections_of) & nd::kInteger) == 0) ||
        (!a.exact.empty() && a.exact.size() != a.size) || !std::ranges::equal(a.exact, b.exact) ||
        (nd::KeyTypes(m, a.key) & (nd::kIntegers | nd::kReals)) == 0)
      return Invalid();
  }
  for (const auto& required : registered->requirements) {
    bool found = false;
    for (const auto& rule : table.requirements)
      found |= rule.key == required.key && rule.reject_equal == required.reject_equal &&
               rule.require_equal == required.require_equal &&
               rule.equals_key == required.equals_key && rule.positive == required.positive &&
               rule.unless_present == required.unless_present && rule.optional == required.optional;
    if (!found) return Invalid();
  }
  const bool drafter = m.role == spec::StageRole::kDrafter;
  if ((m.role != spec::StageRole::kTarget && !drafter) ||
      (drafter ? m.stage.empty() || !IsName(m.stage) || m.stage == "target" || m.reads.size() > 8
               : !m.stage.empty() || !m.reads.empty()))
    return Invalid();
  if (m.slots.empty() || m.slots.size() > (table.extended_components ? 16U : 8U) ||
      m.tensors.size() > nd::kRuleLimit || m.params.size() > 512 || m.folds.size() > 8 ||
      m.policies.size() > 8 || m.pipeline != spec::PipelineKind::kLlm ||
      (!table.extended_components && !m.layer_type_key.empty()) ||
      !spec::IsSourceKey(m.layer_count_key))
    return Invalid();
  for (const auto& slot : m.slots)
    if (!table.extended_components &&
        (!slot.count_key.empty() || slot.per_layer || !slot.layers_key.empty() ||
         slot.site == spec::Site::kSide || slot.site == spec::Site::kMixing))
      return Invalid();
  for (const auto& p : m.params)
    if ((p.per_layer && p.type != spec::ValueType::kInt &&
         !(table.extended_components && p.type == spec::ValueType::kReal)) ||
        (!table.extended_components && p.type == spec::ValueType::kInts))
      return Invalid();
  for (const auto& r : m.tensors)
    if (!nd::Pattern(r.pattern) || (!drafter && r.target) || r.any_part ||
        (!drafter && r.unbound) || (r.target && r.unbound) ||
        ((table.container == "gguf" || table.container == "safetensors") &&
         r.part != format::PartRole::kData) ||
        (table.container == "exl3" && r.part != format::PartRole::kData &&
         r.part != format::PartRole::kCodes && r.part != format::PartRole::kInputSigns &&
         r.part != format::PartRole::kOutputSigns && r.part != format::PartRole::kCodebookMarker))
      return Invalid();
  const auto count = m.tensors.size() + table.tensors.size();
  const auto spelling = [&](std::size_t i) {
    return i < m.tensors.size() ? m.tensors[i].pattern : table.tensors[i - m.tensors.size()].source;
  };
  for (const auto& a : table.tensors) {
    if (!nd::Pattern(a.source) || !SameShape(a.source, a.canonical)) return Invalid();
    std::size_t matches = 0;
    for (const auto& r : m.tensors) matches += (!r.target && r.pattern == a.canonical);
    if (matches == 0) return Invalid();
  }
  // Equal canonical names can describe opposite sides of one declared gate.
  // They still identify one physical tensor. No alias may overlap any spelling.
  const auto exclusive = [&](const spec::TensorRule& a, const spec::TensorRule& b) {
    if (a.pattern != b.pattern || a.experts != b.experts) return false;
    const spec::SlotRule *left = nullptr, *right = nullptr;
    for (const auto& slot : m.slots) {
      if (slot.slot == a.slot) left = &slot;
      if (slot.slot == b.slot) right = &slot;
    }
    return left && right &&
           ((!left->gate.empty() && left->gate == right->gate &&
             left->gate_absent != right->gate_absent) ||
            (!left->count_key.empty() && left->count_key == right->count_key &&
             left->count_above != right->count_above));
  };
  const auto is_target = [&](std::size_t k) { return k < m.tensors.size() && m.tensors[k].target; };
  for (std::size_t i = 0; i < count; ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (is_target(i) == is_target(j) && Overlap(spelling(i), spelling(j)) &&
          !(i < m.tensors.size() && exclusive(m.tensors[i], m.tensors[j])))
        return Invalid();
  for (std::size_t i = 0; i < table.keys.size(); ++i) {
    const auto& a = table.keys[i];
    if (!spec::IsSourceKey(a.source) || !nd::KeyTypes(m, a.canonical) ||
        nd::KeyTypes(m, a.source) || Structural(a.source))
      return Invalid();
    for (std::size_t j = 0; j < i; ++j)
      if (a.source == table.keys[j].source) return Invalid();
  }
  for (std::size_t i = 0; i < table.auxiliary.size(); ++i) {
    const auto& a = table.auxiliary[i];
    if (a.prefix.empty() || !spec::IsSourceKey(a.prefix) || a.reason.empty() ||
        a.reason.size() > spec::kMaxText ||
        (!a.equals_tensor.empty() &&
         (a.prefix.ends_with('.') || !spec::IsSourceKey(a.equals_tensor) || a.equals_dim >= 4)))
      return Invalid();
    const auto matches_aux = [&](std::string_view key) {
      return a.prefix.ends_with('.') ? key.starts_with(a.prefix) : key == a.prefix;
    };
    if (matches_aux(m.layer_count_key) ||
        (!m.layer_flags_key.empty() && matches_aux(m.layer_flags_key)))
      return Invalid();
    for (const auto& slot : m.slots)
      if (!slot.gate.empty() && matches_aux(slot.gate)) return Invalid();
    for (const auto& read : m.reads)
      if (!read.layers_key.empty() && matches_aux(read.layers_key)) return Invalid();
    for (const auto& policy : m.policies)
      if (!policy.gate.empty() && matches_aux(policy.gate)) return Invalid();
    for (const auto& p : m.params)
      if ((!p.key.empty() && matches_aux(p.key)) ||
          (!p.numerator_key.empty() && matches_aux(p.numerator_key)) ||
          (!p.divisor_key.empty() && matches_aux(p.divisor_key)))
        return Invalid();
    for (const auto& k : table.keys)
      if (matches_aux(k.source)) return Invalid();
    for (std::size_t j = 0; j < i; ++j)
      if (a.prefix.starts_with(table.auxiliary[j].prefix) ||
          table.auxiliary[j].prefix.starts_with(a.prefix))
        return Invalid();
  }
  for (const auto& r : table.requirements) {
    if (!r.unless_present.empty() && (!nd::KeyTypes(m, r.unless_present) || r.require_equal ||
                                      r.positive || !r.equals_key.empty()))
      return Invalid();
    if (!r.require_equal && r.equals_key.empty() && !r.positive &&
        nd::KeyTypes(m, r.key) != nd::kInteger)
      return Invalid();
    if ((!r.equals_key.empty() || r.positive) &&
        std::ranges::none_of(registered->requirements, [&](const auto& known) {
          return known.key == r.key && known.equals_key == r.equals_key &&
                 known.positive == r.positive;
        }))
      return Invalid();
    if (r.require_equal && (!spec::IsSourceKey(r.key) || Structural(r.key) ||
                            std::ranges::none_of(registered->requirements, [&](const auto& known) {
                              return known.require_equal && known.key == r.key &&
                                     known.reject_equal == r.reject_equal;
                            })))
      return Invalid();
  }
  return {};
}
}  // namespace llmp::import
