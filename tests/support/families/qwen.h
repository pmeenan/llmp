// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Qwen2 and Qwen3.8 as test fixtures: the approved checkpoints' facts (the
// Qwen2.5 0.5B FP16 GGUF and EXL3 fixtures', Qwen3.8 Flash Next's
// config.json), the source values their architecture mappings read, so
// that a test can synthesize a real family's spec from a fixture's tensor
// table without an artifact, the fixture tables' resource description, and
// the oracles tests still compare generic code against: llama.cpp
// b29c606e2's n-gram hash (llm_graph_input_ple::set_input, MIT, so this
// file carries GGML's notice; docs/licensing.md) and the M3 runner's state
// sizes.
//
// Serving reads the spec a v1 artifact stores (D-118); nothing in src/
// holds these facts. A change here changes only what the tests synthesize.

#ifndef LLMP_TESTS_SUPPORT_FAMILIES_QWEN_H_
#define LLMP_TESTS_SUPPORT_FAMILIES_QWEN_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
#include "spec/mappings/rules.h"

namespace llmp::test_support::families {

// ---------------------------------------------------------------- Qwen2

// A Qwen2 dense checkpoint's hyperparameters.
struct Qwen2Profile {
  std::string_view name;
  std::uint32_t layers = 0;
  std::uint32_t width = 0;     // n_embd
  std::uint32_t heads = 0;     // query heads
  std::uint32_t kv_heads = 0;  // key and value heads
  std::uint32_t head_dim = 0;  // also the RoPE dimensions (NEOX over the whole head)
  std::uint32_t ffn = 0;       // feed-forward width
  std::uint32_t vocab = 0;
  std::uint32_t train_context = 0;  // RoPE's original context (n_ctx_orig)
  float rms_eps = 0.0f;
  float rope_base = 0.0f;
  // Weights of matrices ("F16" or "exl3"); norms and biases are F32.
  std::string_view weight_type;

  std::uint32_t kv_width() const { return kv_heads * head_dim; }
};

// Qwen2.5-0.5B-Instruct (the FP16 fixture of docs/first-slice.md): qwen2.*
// key/values of qwen2.5-0.5b-instruct-fp16.gguf (SHA-256 8e0ae260...).
inline const Qwen2Profile& Qwen25Instruct05B() {
  static constexpr Qwen2Profile kProfile = {.name = "qwen2.5-0.5b-instruct",
                                            .layers = 24,
                                            .width = 896,
                                            .heads = 14,
                                            .kv_heads = 2,
                                            .head_dim = 64,
                                            .ffn = 4864,
                                            .vocab = 151936,
                                            .train_context = 8192,
                                            .rms_eps = 1e-6f,
                                            .rope_base = 1000000.0f,
                                            .weight_type = "F16"};
  return kProfile;
}

// The same model's EXL3 fixtures (docs/exl3-bringup.md): config.json's
// max_position_embeddings 32,768 is RoPE's original context.
inline const Qwen2Profile& Qwen25Instruct05BExl3() {
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

// The profile as the keys the qwen2 mapping reads. Refused for another
// weight type or inconsistent attention dimensions.
inline std::expected<std::vector<spec::SourceValue>, std::string> Qwen2Values(
    const Qwen2Profile& p) {
  using spec::mappings::FlagValue;
  using spec::mappings::Value;
  if (p.weight_type != "F16" && p.weight_type != "exl3") {
    return std::unexpected("Qwen2 values need the approved FP16 or EXL3 profile");
  }
  if (p.layers == 0 || p.head_dim == 0 || p.kv_heads == 0 || p.heads % p.kv_heads != 0 ||
      std::uint64_t{p.heads} * p.head_dim != p.width) {
    return std::unexpected("Qwen2 profile has inconsistent attention dimensions");
  }
  return std::vector<spec::SourceValue>{
      Value("qwen2.context_length", p.train_context),
      Value("qwen2.embedding_length", p.width),
      Value("qwen2.block_count", p.layers),
      Value("qwen2.feed_forward_length", p.ffn),
      Value("qwen2.attention.head_count", p.heads),
      Value("qwen2.attention.head_count_kv", p.kv_heads),
      Value("qwen2.attention.key_length", p.head_dim),
      Value("qwen2.attention.layer_norm_rms_epsilon", p.rms_eps),
      Value("qwen2.rope.freq_base", p.rope_base),
      Value("qwen2.hidden_act", "silu"),
      FlagValue("qwen2.use_sliding_window", false),
  };
}

// ---------------------------------------------------------------- Qwen3.8

struct Qwen38Profile {
  std::string_view name;
  std::uint32_t layers = 0;
  std::uint32_t width = 0;     // n_embd
  std::uint32_t heads = 0;     // QSA query heads
  std::uint32_t kv_heads = 0;  // QSA KV heads
  std::uint32_t head_dim = 0;
  std::uint32_t rope_dims = 0;
  std::array<std::int32_t, 4> rope_sections{};  // interleaved mrope
  float rope_base = 0.0f;
  std::uint32_t lin_k_heads = 0;  // Gated DeltaNet
  std::uint32_t lin_v_heads = 0;
  std::uint32_t lin_head_dim = 0;  // key and value
  std::uint32_t conv = 0;          // its convolution kernel
  std::uint32_t experts = 0;
  std::uint32_t experts_used = 0;
  std::uint32_t expert_ffn = 0;
  std::uint32_t shared_ffn = 0;
  std::uint32_t hc = 0;  // hyper-connection streams
  std::uint32_t hc_rank = 0;
  std::uint32_t indexer_heads = 0;
  std::uint32_t indexer_head_dim = 0;
  std::uint32_t indexer_ratio = 0;   // tokens a block pools
  std::uint32_t indexer_budget = 0;  // tokens attention keeps (whole blocks, plus the tail)
  std::uint32_t ple_layer = 0;       // the n-gram embedding's layer
  std::uint32_t ngram = 0;
  std::uint32_t heads_per_ngram = 0;
  std::uint32_t ple_row = 0;  // values a table row holds
  std::uint32_t ple_conv = 0;
  std::int32_t ple_eos = 0;  // resets the n-gram window
  std::uint32_t vocab = 0;
  float rms_eps = 0.0f;

  // Every full_attention_interval-th layer is QSA, the rest linear.
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static): the profile's rule
  bool linear(std::uint32_t layer) const { return (layer + 1) % 4 != 0; }
  std::uint32_t hc_width() const { return hc * width; }
  std::uint32_t lin_k_width() const { return lin_k_heads * lin_head_dim; }
  std::uint32_t lin_v_width() const { return lin_v_heads * lin_head_dim; }
  std::uint32_t conv_channels() const { return (2 * lin_k_width()) + lin_v_width(); }
  std::uint32_t ple_heads() const { return (ngram - 1) * heads_per_ngram; }
  std::uint32_t ple_width() const { return ple_heads() * ple_row; }
  // The n-gram convolution's history: (kernel - 1) taps dilated by the
  // n-gram size.
  std::uint32_t ple_history() const { return (ple_conv - 1) * ngram; }
};

// Qwen3.8 Flash Next (Qwen/Qwen3.8-Flash-Next@de4b8e4d's text model, as
// Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6 carries it).
inline constexpr std::uint32_t kQwen38FlashContext = 262144;
inline constexpr std::uint32_t kQwen38FlashVocab = 248320;
inline const Qwen38Profile& Qwen38Flash() {
  // config.json's text_config (the base model's).
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
    p.vocab = kQwen38FlashVocab;
    p.rms_eps = 1e-6f;
    return p;
  }();
  return kProfile;
}

// A fixture table's tensor (a v0 descriptor's facts): a plain
// (checkpoint-layout) or GGML representation, its type name (GGML's, or
// the container dtype) and its shape in GGML order (`ne`; a plain [rows,
// cols] is ne [cols, rows]); an expert array's `ne` is one slice's.
struct Qwen38Resource {
  std::vector<std::string> roles;  // for an expert array, its name
  bool plain = false;
  std::string type;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  std::uint32_t count = 0;
  std::uint64_t group_offset = 0;  // an expert array's offset in its expert group
  std::uint64_t readable = 0;
};

// The n-gram hash's constants: each head's rows are offset + (hash mod
// vocab).
struct Qwen38PleHash {
  std::vector<std::uint64_t> multipliers;  // ngram
  std::vector<std::uint64_t> offsets;      // ple_heads
  std::vector<std::uint64_t> vocab;        // ple_heads
  std::uint64_t table_rows = 0;            // the table's rows it was checked against
};

namespace qwen_detail {
inline bool ProfileIsSane(const Qwen38Profile& p) {
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
         p.rope_base > 0.0f && p.indexer_budget % p.indexer_ratio == 0 && p.width <= (1U << 16) &&
         p.hc <= 16 && p.lin_k_heads <= 256 && p.lin_v_heads <= 256 && p.lin_head_dim <= 1024 &&
         p.heads_per_ngram <= 64;
}
inline std::uint64_t Pad(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

// The keys both mappings' blocks read, under `prefix`.
inline void BlockValues(std::vector<spec::SourceValue>& out, const std::string& prefix,
                        const Qwen38Profile& q, std::vector<std::int64_t> ratios) {
  using spec::mappings::Value;
  const auto key = [&](std::string_view k) { return prefix + std::string(k); };
  out.push_back(Value(key("context_length"), kQwen38FlashContext));
  out.push_back(Value(key("embedding_length"), q.width));
  out.push_back(Value(key("hyper_connection.count"), q.hc));
  out.push_back(Value(key("hyper_connection.low_rank"), q.hc_rank));
  out.push_back(Value(key("attention.layer_norm_rms_epsilon"), q.rms_eps));
  out.push_back(Value(key("attention.head_count"), q.heads));
  out.push_back(Value(key("attention.head_count_kv"), q.kv_heads));
  out.push_back(Value(key("attention.key_length"), q.head_dim));
  out.push_back(Value(key("attention.value_length"), q.head_dim));
  out.push_back(Value(key("rope.dimension_count"), q.rope_dims));
  out.push_back(Value(key("rope.freq_base"), q.rope_base));
  out.push_back(Value(key("rope.dimension_sections"),
                      std::vector<std::int64_t>{q.rope_sections[0], q.rope_sections[1],
                                                q.rope_sections[2], q.rope_sections[3]}));
  out.push_back(Value(key("attention.indexer.head_count"), q.indexer_heads));
  out.push_back(Value(key("attention.indexer.key_length"), q.indexer_head_dim));
  out.push_back(Value(key("attention.indexer.top_k"), q.indexer_budget));
  out.push_back(Value(key("attention.compress_ratios"), std::move(ratios)));
  out.push_back(Value(key("expert_count"), q.experts));
  out.push_back(Value(key("expert_used_count"), q.experts_used));
  out.push_back(Value(key("expert_feed_forward_length"), q.expert_ffn));
  out.push_back(Value(key("expert_shared_feed_forward_length"), q.shared_ffn));
}
}  // namespace qwen_detail

// The hash's constants checked against the profile and a table of
// `table_rows` rows (each head's range within it and the I32 row index the
// lookup takes); refused otherwise, naming the head.
inline std::expected<Qwen38PleHash, std::string> CheckQwen38PleHash(
    const Qwen38Profile& p, std::span<const std::int64_t> multipliers,
    std::span<const std::int64_t> offsets, std::span<const std::int64_t> vocab,
    std::uint64_t table_rows) {
  if (!qwen_detail::ProfileIsSane(p) || multipliers.size() != p.ngram ||
      offsets.size() != p.ple_heads() || vocab.size() != p.ple_heads()) {
    return std::unexpected("n-gram hash constants that are not the profile's");
  }
  const auto limit = std::min<std::uint64_t>(table_rows, std::numeric_limits<std::int32_t>::max());
  Qwen38PleHash h;
  h.table_rows = table_rows;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    if (vocab[i] <= 0 || offsets[i] < 0 ||
        static_cast<std::uint64_t>(offsets[i]) + static_cast<std::uint64_t>(vocab[i]) > limit) {
      return std::unexpected(
          std::format("n-gram head {}'s rows [{}, {} + {}) are outside the table's {}", i,
                      offsets[i], offsets[i], vocab[i], table_rows));
    }
    h.offsets.push_back(static_cast<std::uint64_t>(offsets[i]));
    h.vocab.push_back(static_cast<std::uint64_t>(vocab[i]));
  }
  for (const std::int64_t m : multipliers) h.multipliers.push_back(static_cast<std::uint64_t>(m));
  return h;
}

// The n-gram rows of the token at `position`: ple_heads rows, llama.cpp's
// llm_graph_input_ple::set_input hash (the oracle the side.ngram
// component is compared with).
inline std::vector<std::int32_t> Qwen38PleRows(const Qwen38Profile& p, const Qwen38PleHash& h,
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
    for (std::uint32_t j = 1; j < n; ++j) mixed ^= ctx[j] * h.multipliers[j];
    const std::uint32_t base = (n - 2) * p.heads_per_ngram;
    for (std::uint32_t g = 0; g < p.heads_per_ngram; ++g) {
      const std::uint32_t head = base + g;
      rows[head] = static_cast<std::int32_t>((mixed % h.vocab[head]) + h.offsets[head]);
    }
  }
  return rows;
}

// The profile and hash as the GGUF keys the qwen4exp mapping reads.
// Refused if the profile's linear layers are not the mapping's.
inline std::expected<std::vector<spec::SourceValue>, std::string> Qwen38GgufValues(
    const Qwen38Profile& p, const Qwen38PleHash& hash) {
  using spec::mappings::Value;
  const std::string_view types = spec::mappings::Qwen38Mapping().layer_types;
  std::vector<std::int64_t> ratios;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const bool full = !types.empty() && types.at(il % types.size()) == 'f';
    if (p.linear(il) == full) {
      return std::unexpected("the Qwen3.8 profile's linear layers are not the mapping's");
    }
    ratios.push_back(full ? std::int64_t{p.indexer_ratio} : 0);
  }
  const auto list = [](const std::vector<std::uint64_t>& v) {
    return std::vector<std::int64_t>(v.begin(), v.end());
  };
  std::vector<spec::SourceValue> out;
  qwen_detail::BlockValues(out, "qwen4exp.", p, std::move(ratios));
  for (spec::SourceValue v : std::vector<spec::SourceValue>{
           Value("qwen4exp.block_count", p.layers),
           Value("qwen4exp.full_attention_interval", 4U),
           Value("qwen4exp.ssm.conv_kernel", p.conv),
           Value("qwen4exp.ssm.group_count", p.lin_k_heads),
           Value("qwen4exp.ssm.time_step_rank", p.lin_v_heads),
           Value("qwen4exp.ssm.state_size", p.lin_head_dim),
           Value("qwen4exp.ssm.inner_size", p.lin_v_width()),
           Value("qwen4exp.ple.ngram_size", p.ngram),
           Value("qwen4exp.ple.heads_per_ngram", p.heads_per_ngram),
           Value("qwen4exp.ple.conv_kernel", p.ple_conv),
           Value("qwen4exp.ple.eos_token_id", static_cast<std::uint32_t>(p.ple_eos)),
           Value("qwen4exp.embedding_length_per_layer_input", p.ple_row),
           Value("qwen4exp.ple.layers", std::vector<std::int64_t>{p.ple_layer}),
           Value("qwen4exp.ple.layer_multipliers", list(hash.multipliers)),
           Value("qwen4exp.ple.head_offsets", list(hash.offsets)),
           Value("qwen4exp.ple.head_vocab_sizes", list(hash.vocab)),
       }) {
    out.push_back(std::move(v));
  }
  return out;
}

// The routed experts' layout in a ModelOpt-form fixture: GGML's NVFP4
// blocks per projection (the first Qwen3.8 import), or the CUTLASS
// scale-atom layout with gate and up stacked.
enum class Qwen38Experts : std::uint8_t { kGgml, kCutlass };

// The ModelOpt form's values: the GGUF keys plus the form (the indexer's
// fused projection, the experts' stacking and their per-expert scales).
inline std::expected<std::vector<spec::SourceValue>, std::string> Qwen38ModelOptValues(
    const Qwen38Profile& p, const Qwen38PleHash& hash, Qwen38Experts experts) {
  using spec::mappings::FlagValue;
  using spec::mappings::Value;
  auto out = Qwen38GgufValues(p, hash);
  if (!out) return out;
  out->push_back(Value("llmp.qwen4exp.indexer", "fused"));
  out->push_back(FlagValue("llmp.qwen4exp.experts_stacked", experts == Qwen38Experts::kCutlass));
  out->push_back(FlagValue("llmp.qwen4exp.expert_scales", true));
  return out;
}

// The MTP drafter's values (stacked scale-atom expert tables with their
// tensor scales' vectors), with a selected head of `selected_rows` rows (0:
// none, its target's head). spec::mappings::Qwen38MtpNativeValues gives
// the same for Qwen3.8 Flash.
inline std::expected<std::vector<spec::SourceValue>, std::string> Qwen38MtpValues(
    const Qwen38Profile& p, std::uint64_t selected_rows) {
  using spec::mappings::FlagValue;
  using spec::mappings::Value;
  if (selected_rows > p.vocab)
    return std::unexpected("a selected head of more rows than the vocabulary");
  std::vector<spec::SourceValue> out;
  qwen_detail::BlockValues(out, "qwen4exp-mtp.", p, {std::int64_t{p.indexer_ratio}});
  out.push_back(Value("qwen4exp-mtp.block_count", 1U));
  out.push_back(Value("llmp.qwen4exp.indexer", "fused"));
  out.push_back(FlagValue("llmp.qwen4exp.experts_stacked", true));
  out.push_back(FlagValue("llmp.qwen4exp.expert_scales", true));
  std::uint64_t head_rows = p.vocab;
  if (selected_rows != 0) {
    out.push_back(Value("qwen4exp-mtp.draft_head_rows", static_cast<std::uint32_t>(selected_rows)));
    head_rows = selected_rows;
  }
  out.push_back(
      Value("qwen4exp-mtp.draft_vocab",
            static_cast<std::uint32_t>(std::min<std::uint64_t>(
                spec::mappings::Qwen38Mapping().serving->default_draft_vocab, head_rows))));
  return out;
}

// The M3 runner's state region at a context (its layout: each tensor
// 256-byte aligned; the target's, then the MTP drafter's), which the
// component stage layouts must hold (the target's does not depend on the
// chunk's rows).
inline std::uint64_t Qwen38StateBytes(const Qwen38Profile& p, std::uint32_t context,
                                      std::uint32_t /*max_rows*/) {
  using qwen_detail::Pad;
  const std::uint64_t cells = Pad(context, 256);
  std::uint64_t bytes = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.linear(il)) {
      bytes = Pad(bytes + (std::uint64_t{p.conv - 1} * p.conv_channels() * 4), 256);
      bytes =
          Pad(bytes + (std::uint64_t{p.lin_head_dim} * p.lin_head_dim * p.lin_v_heads * 4), 256);
      if (il == p.ple_layer) {
        bytes = Pad(bytes + (std::uint64_t{p.ple_history()} * p.hc_width() * 4), 256);
      }
    } else {
      bytes = Pad(bytes + (std::uint64_t{p.head_dim} * p.kv_heads * cells * 2), 256);
      bytes = Pad(bytes + (std::uint64_t{p.head_dim} * p.kv_heads * cells * 2), 256);
      bytes = Pad(bytes + (p.indexer_head_dim * cells * 4), 256);
      bytes =
          Pad(bytes + (p.indexer_head_dim * ((cells + p.indexer_ratio - 1) / p.indexer_ratio) * 2),
              256);
    }
  }
  return bytes;
}
inline std::uint64_t Qwen38MtpStateBytes(const Qwen38Profile& p, std::uint32_t context,
                                         std::uint32_t max_rows) {
  using qwen_detail::Pad;
  const std::uint64_t cells = Pad(context, 256);
  const std::uint64_t hidden_rows = std::uint64_t{max_rows} + 1;
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * cells * 2;
  std::uint64_t bytes = 0;
  bytes = Pad(bytes + kv, 256);                                // k
  bytes = Pad(bytes + kv, 256);                                // v
  bytes = Pad(bytes + (cells * p.indexer_head_dim * 4), 256);  // indexer
  bytes = Pad(bytes + (((cells + p.indexer_ratio - 1) / p.indexer_ratio) * p.indexer_head_dim * 2),
              256);                                                           // blocks
  bytes = Pad(bytes + (std::uint64_t{p.hc_width()} * hidden_rows * 4), 256);  // hidden
  return bytes;
}

}  // namespace llmp::test_support::families

#endif  // LLMP_TESTS_SUPPORT_FAMILIES_QWEN_H_
