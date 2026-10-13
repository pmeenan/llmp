// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_UNIT_QWEN2_EXL3_NORMALIZATION_FIXTURE_H_
#define LLMP_TESTS_UNIT_QWEN2_EXL3_NORMALIZATION_FIXTURE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/check.h"
#include "format/layout.h"
#include "import/checkpoint.h"

namespace llmp::test_support {
// Independent bounded header inventory of Qwen2.5-0.5B-Instruct-exl3-4.0bpw
// (588,866,468 payload bytes) and Qwen2.5-0.5B-Instruct-exl3-4.5bpw
// (611,173,284 payload bytes): 798 physical tensors (50 BF16 weights, 72 F16
// biases, and 169 four-part EXL3 mcg trellises); synthetic extents, no payload.
inline import::Checkpoint Qwen2Exl3Checkpoint(bool rate_4_5 = false) {
  import::Checkpoint out{.container = "exl3",
                         .architecture = "qwen2",
                         .quantization = "exl3",
                         .files = {{.name = "model.safetensors", .bytes = 0}}};
  out.metadata.push_back({"architectures", std::vector<std::string>{"Qwen2ForCausalLM"}});
  out.metadata.push_back({"attention_dropout", 0.0});
  out.metadata.push_back({"bos_token_id", std::int64_t{151643}});
  out.metadata.push_back({"eos_token_id", std::int64_t{151645}});
  out.metadata.push_back({"hidden_act", std::string("silu")});
  out.metadata.push_back({"hidden_size", std::int64_t{896}});
  out.metadata.push_back({"initializer_range", 0.02});
  out.metadata.push_back({"intermediate_size", std::int64_t{4864}});
  out.metadata.push_back({"max_position_embeddings", std::int64_t{32768}});
  out.metadata.push_back({"max_window_layers", std::int64_t{21}});
  out.metadata.push_back({"model_type", std::string("qwen2")});
  out.metadata.push_back({"num_attention_heads", std::int64_t{14}});
  out.metadata.push_back({"num_hidden_layers", std::int64_t{24}});
  out.metadata.push_back({"num_key_value_heads", std::int64_t{2}});
  out.metadata.push_back({"rms_norm_eps", 1e-6});
  out.metadata.push_back({"rope_theta", 1000000.0});
  out.metadata.push_back({"sliding_window", std::int64_t{32768}});
  out.metadata.push_back({"tie_word_embeddings", true});
  out.metadata.push_back({"torch_dtype", std::string("bfloat16")});
  out.metadata.push_back(
      {"transformers_version", std::string(rate_4_5 ? "4.52.0.dev0" : "4.43.1")});
  out.metadata.push_back({"use_cache", true});
  out.metadata.push_back({"use_sliding_window", false});
  out.metadata.push_back({"vocab_size", std::int64_t{151936}});
  std::ranges::sort(out.metadata, {}, &import::MetaEntry::key);

  const auto add_dense = [&](std::string name, std::vector<std::uint64_t> dims,
                             format::Encoding encoding, std::string_view dtype) {
    std::vector<std::uint64_t> ne(dims.rbegin(), dims.rend());
    auto shape = format::ShapeOf(ne);
    base::Check(shape.has_value(), "captured Qwen2 EXL3 dense shape");
    format::TensorFormat f{.encoding = encoding, .layout = format::Layout::kDense, .shape = *shape};
    auto parts = format::PartsOf(f);
    base::Check(parts && parts->size() == 1, "captured Qwen2 EXL3 dense part");
    const auto at = out.files[0].bytes;
    out.tensors.push_back({.name = std::move(name),
                           .format = f,
                           .part = format::PartRole::kData,
                           .extent = {.offset = at, .bytes = (*parts)[0].bytes},
                           .dtype = dtype,
                           .dims = std::move(dims)});
    out.files[0].bytes = at + (*parts)[0].bytes;
  };
  const auto add_trellis = [&](std::string prefix, std::uint64_t k, std::uint64_t n,
                               std::uint8_t bits) {
    format::TensorFormat f{
        .encoding = format::Encoding::kExl3Trellis,
        .params = {.k_bits = bits, .codebook = format::Exl3Codebook::kMcg, .tensor_scale = false},
        .layout = format::Layout::kExl3Tiles,
        .shape = {.ne = {k, n, 1, 1}, .rank = 2}};
    auto parts = format::PartsOf(f);
    base::Check(parts && parts->size() == 4, "captured Qwen2 EXL3 trellis parts");
    const auto push = [&](std::string_view suffix, format::PartRole role, std::string_view dtype,
                          std::vector<std::uint64_t> dims) {
      const auto part = parts->Find(role);
      base::Check(part.has_value(), "captured Qwen2 EXL3 trellis part");
      const auto at = out.files[0].bytes;
      out.tensors.push_back({.name = prefix + std::string(suffix),
                             .format = f,
                             .part = role,
                             .extent = {.offset = at, .bytes = part->bytes},
                             .dtype = dtype,
                             .dims = std::move(dims)});
      out.files[0].bytes = at + part->bytes;
    };
    push(".trellis", format::PartRole::kCodes, "I16", {k / 16, n / 16, 16ULL * bits});
    push(".suh", format::PartRole::kInputSigns, "F16", {k});
    push(".svh", format::PartRole::kOutputSigns, "F16", {n});
    push(".mcg", format::PartRole::kCodebookMarker, "I32", {});
  };

  add_dense("model.embed_tokens.weight", {151936, 896}, format::Encoding::kBF16, "BF16");
  add_dense("model.norm.weight", {896}, format::Encoding::kBF16, "BF16");
  add_trellis("lm_head", 896, 151936, 8);
  struct Linear {
    std::string_view suffix;
    std::uint64_t k, n;
  };
  constexpr std::array<Linear, 7> kLinears{{
      {"self_attn.q_proj", 896, 896},
      {"self_attn.k_proj", 896, 128},
      {"self_attn.v_proj", 896, 128},
      {"self_attn.o_proj", 896, 896},
      {"mlp.gate_proj", 896, 4864},
      {"mlp.up_proj", 896, 4864},
      {"mlp.down_proj", 4864, 896},
  }};
  for (unsigned layer = 0; layer < 24; ++layer) {
    const auto base = std::format("model.layers.{}.", layer);
    add_dense(base + "input_layernorm.weight", {896}, format::Encoding::kBF16, "BF16");
    add_dense(base + "post_attention_layernorm.weight", {896}, format::Encoding::kBF16, "BF16");
    add_dense(base + "self_attn.q_proj.bias", {896}, format::Encoding::kF16, "F16");
    add_dense(base + "self_attn.k_proj.bias", {128}, format::Encoding::kF16, "F16");
    add_dense(base + "self_attn.v_proj.bias", {128}, format::Encoding::kF16, "F16");
    for (std::size_t i = 0; i < kLinears.size(); ++i) {
      const std::uint8_t bits =
          !rate_4_5 ? 4 : (i < 4 ? (layer < 2 ? 6 : 5) : (layer < 10 ? 5 : 4));
      add_trellis(base + std::string(kLinears[i].suffix), kLinears[i].k, kLinears[i].n, bits);
    }
  }
  return out;
}
}  // namespace llmp::test_support
#endif  // LLMP_TESTS_UNIT_QWEN2_EXL3_NORMALIZATION_FIXTURE_H_
