// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_UNIT_QWEN38_GGUF_NORMALIZATION_FIXTURE_H_
#define LLMP_TESTS_UNIT_QWEN38_GGUF_NORMALIZATION_FIXTURE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "format/layout.h"
#include "import/checkpoint.h"

namespace llmp::test_support {
// Independent metadata inventory of the archived 38bb39ee UD-IQ3_XXS
// headers (manifest506dfcaf..., header inventorybc35f7f6...). Synthetic
// extents, no tensor payload allocation and no claim of payload integrity.
inline import::Checkpoint Qwen38GgufCheckpoint() {
  import::Checkpoint out{
      .container = "gguf", .architecture = "qwen4exp", .files = {{.name = "memory", .bytes = 0}}};
  out.metadata.push_back({"general.architecture", std::string("qwen4exp")});
  out.metadata.push_back({"qwen4exp.block_count", std::int64_t{48}});
  out.metadata.push_back({"qwen4exp.context_length", std::int64_t{262144}});
  out.metadata.push_back({"qwen4exp.embedding_length", std::int64_t{2560}});
  out.metadata.push_back({"qwen4exp.attention.head_count", std::int64_t{24}});
  out.metadata.push_back({"qwen4exp.attention.head_count_kv", std::int64_t{2}});
  out.metadata.push_back(
      {"qwen4exp.rope.dimension_sections", std::vector<std::int64_t>{11, 11, 10, 0}});
  out.metadata.push_back({"qwen4exp.rope.freq_base", 10000000.0});
  out.metadata.push_back({"qwen4exp.attention.layer_norm_rms_epsilon", 9.999999974752427e-07});
  out.metadata.push_back({"qwen4exp.expert_count", std::int64_t{512}});
  out.metadata.push_back({"qwen4exp.expert_used_count", std::int64_t{10}});
  out.metadata.push_back({"qwen4exp.attention.key_length", std::int64_t{256}});
  out.metadata.push_back({"qwen4exp.attention.value_length", std::int64_t{256}});
  out.metadata.push_back({"qwen4exp.expert_feed_forward_length", std::int64_t{640}});
  out.metadata.push_back({"qwen4exp.expert_shared_feed_forward_length", std::int64_t{640}});
  out.metadata.push_back({"qwen4exp.ssm.conv_kernel", std::int64_t{4}});
  out.metadata.push_back({"qwen4exp.ssm.state_size", std::int64_t{128}});
  out.metadata.push_back({"qwen4exp.ssm.group_count", std::int64_t{16}});
  out.metadata.push_back({"qwen4exp.ssm.time_step_rank", std::int64_t{48}});
  out.metadata.push_back({"qwen4exp.ssm.inner_size", std::int64_t{6144}});
  out.metadata.push_back({"qwen4exp.full_attention_interval", std::int64_t{4}});
  out.metadata.push_back({"qwen4exp.rope.dimension_count", std::int64_t{64}});
  out.metadata.push_back({"qwen4exp.hyper_connection.count", std::int64_t{4}});
  out.metadata.push_back({"qwen4exp.hyper_connection.low_rank", std::int64_t{320}});
  out.metadata.push_back({"qwen4exp.attention.indexer.head_count", std::int64_t{4}});
  out.metadata.push_back({"qwen4exp.attention.indexer.key_length", std::int64_t{128}});
  out.metadata.push_back({"qwen4exp.attention.indexer.top_k", std::int64_t{2048}});
  out.metadata.push_back(
      {"qwen4exp.attention.compress_ratios",
       std::vector<std::int64_t>{0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4,
                                 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4,
                                 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4}});
  out.metadata.push_back({"qwen4exp.ple.layers", std::vector<std::int64_t>{1}});
  out.metadata.push_back({"qwen4exp.ple.ngram_size", std::int64_t{3}});
  out.metadata.push_back({"qwen4exp.ple.heads_per_ngram", std::int64_t{8}});
  out.metadata.push_back({"qwen4exp.ple.conv_kernel", std::int64_t{4}});
  out.metadata.push_back({"qwen4exp.ple.eos_token_id", std::int64_t{248044}});
  out.metadata.push_back({"qwen4exp.embedding_length_per_layer_input", std::int64_t{160}});
  out.metadata.push_back(
      {"qwen4exp.ple.layer_multipliers",
       std::vector<std::int64_t>{23703573157769, 20109073645365, 8052911324071}});
  out.metadata.push_back(
      {"qwen4exp.ple.head_offsets",
       std::vector<std::int64_t>{0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
                                 140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
                                 260000955, 280001114, 300001275}});
  out.metadata.push_back(
      {"qwen4exp.ple.head_vocab_sizes",
       std::vector<std::int64_t>{20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                                 20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                                 20000153, 20000159, 20000161, 20000171}});
  out.metadata.push_back({"qwen4exp.ple.image_token_id", std::int64_t{248056}});
  std::ranges::sort(out.metadata, {}, &import::MetaEntry::key);
  struct Descriptor {
    std::string_view name;
    std::array<std::uint64_t, 3> dims;
    std::uint8_t rank;
    format::Encoding encoding;
    std::uint64_t layers;
  };
  constexpr std::array descriptors{
      Descriptor{"blk.{}.attn_gate.weight", {2560, 6144, 1}, 2, format::Encoding::kQ8_0, 0x4ULL},
      Descriptor{"blk.{}.attn_gate.weight",
                 {2560, 6144, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0x777777777773ULL},
      Descriptor{
          "blk.{}.attn_k.weight", {2560, 512, 1}, 2, format::Encoding::kQ6_K, 0x888888888888ULL},
      Descriptor{
          "blk.{}.attn_k_norm.weight", {256, 1, 1}, 1, format::Encoding::kF32, 0x888888888888ULL},
      Descriptor{"blk.{}.attn_output.weight",
                 {6144, 2560, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0x888888888888ULL},
      Descriptor{
          "blk.{}.attn_q.weight", {2560, 12288, 1}, 2, format::Encoding::kQ6_K, 0x888888888888ULL},
      Descriptor{
          "blk.{}.attn_q_norm.weight", {256, 1, 1}, 1, format::Encoding::kF32, 0x888888888888ULL},
      Descriptor{"blk.{}.attn_qkv.weight", {2560, 10240, 1}, 2, format::Encoding::kQ8_0, 0x4ULL},
      Descriptor{"blk.{}.attn_qkv.weight",
                 {2560, 10240, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0x777777777773ULL},
      Descriptor{
          "blk.{}.attn_v.weight", {2560, 512, 1}, 2, format::Encoding::kQ6_K, 0x888888888888ULL},
      Descriptor{"blk.{}.ffn_down_exps.weight",
                 {640, 2560, 512},
                 3,
                 format::Encoding::kIQ4_NL,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.ffn_down_shexp.weight",
                 {640, 2560, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0xffffffffffffULL},
      Descriptor{
          "blk.{}.ffn_gate_exps.weight", {2560, 640, 512}, 3, format::Encoding::kIQ3_S, 0x4ULL},
      Descriptor{"blk.{}.ffn_gate_exps.weight",
                 {2560, 640, 512},
                 3,
                 format::Encoding::kIQ2_S,
                 0xfffffffffffbULL},
      Descriptor{"blk.{}.ffn_gate_inp.weight",
                 {2560, 512, 1},
                 2,
                 format::Encoding::kF32,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.ffn_gate_inp_shexp.weight",
                 {2560, 1, 1},
                 1,
                 format::Encoding::kF32,
                 0xffffffffffffULL},
      Descriptor{
          "blk.{}.ffn_gate_shexp.weight", {2560, 640, 1}, 2, format::Encoding::kQ8_0, 0x4ULL},
      Descriptor{"blk.{}.ffn_gate_shexp.weight",
                 {2560, 640, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0xfffffffffffbULL},
      Descriptor{
          "blk.{}.ffn_up_exps.weight", {2560, 640, 512}, 3, format::Encoding::kIQ3_S, 0x4ULL},
      Descriptor{"blk.{}.ffn_up_exps.weight",
                 {2560, 640, 512},
                 3,
                 format::Encoding::kIQ2_S,
                 0xfffffffffffbULL},
      Descriptor{"blk.{}.ffn_up_shexp.weight", {2560, 640, 1}, 2, format::Encoding::kQ8_0, 0x4ULL},
      Descriptor{"blk.{}.ffn_up_shexp.weight",
                 {2560, 640, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0xfffffffffffbULL},
      Descriptor{"blk.{}.hc_attn_down.weight",
                 {10240, 320, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.hc_attn_inject.weight",
                 {10240, 4, 1},
                 2,
                 format::Encoding::kF32,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.hc_attn_norm.weight",
                 {10240, 1, 1},
                 1,
                 format::Encoding::kF32,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.hc_attn_up.weight",
                 {320, 10240, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.hc_ffn_down.weight",
                 {10240, 320, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.hc_ffn_inject.weight",
                 {10240, 4, 1},
                 2,
                 format::Encoding::kF32,
                 0xffffffffffffULL},
      Descriptor{
          "blk.{}.hc_ffn_norm.weight", {10240, 1, 1}, 1, format::Encoding::kF32, 0xffffffffffffULL},
      Descriptor{"blk.{}.hc_ffn_up.weight",
                 {320, 10240, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0xffffffffffffULL},
      Descriptor{"blk.{}.indexer.k_norm.weight",
                 {128, 1, 1},
                 1,
                 format::Encoding::kF32,
                 0x888888888888ULL},
      Descriptor{"blk.{}.indexer.k_proj.weight",
                 {2560, 128, 1},
                 2,
                 format::Encoding::kBF16,
                 0x888888888888ULL},
      Descriptor{"blk.{}.indexer.q_norm.weight",
                 {128, 1, 1},
                 1,
                 format::Encoding::kF32,
                 0x888888888888ULL},
      Descriptor{"blk.{}.indexer.q_proj.weight",
                 {2560, 512, 1},
                 2,
                 format::Encoding::kBF16,
                 0x888888888888ULL},
      Descriptor{"blk.{}.ple_conv1d.weight", {4, 10240, 1}, 2, format::Encoding::kF32, 0x2ULL},
      Descriptor{"blk.{}.ple_key.weight", {2560, 10240, 1}, 2, format::Encoding::kQ8_0, 0x2ULL},
      Descriptor{"blk.{}.ple_norm_conv.weight", {10240, 1, 1}, 1, format::Encoding::kF32, 0x2ULL},
      Descriptor{"blk.{}.ple_norm_key.weight", {10240, 1, 1}, 1, format::Encoding::kF32, 0x2ULL},
      Descriptor{"blk.{}.ple_norm_query.weight", {10240, 1, 1}, 1, format::Encoding::kF32, 0x2ULL},
      Descriptor{"blk.{}.ple_value.weight", {2560, 2560, 1}, 2, format::Encoding::kQ8_0, 0x2ULL},
      Descriptor{"blk.{}.ssm_a", {48, 1, 1}, 1, format::Encoding::kF32, 0x777777777777ULL},
      Descriptor{
          "blk.{}.ssm_alpha.weight", {2560, 48, 1}, 2, format::Encoding::kF32, 0x777777777777ULL},
      Descriptor{
          "blk.{}.ssm_beta.weight", {2560, 48, 1}, 2, format::Encoding::kF32, 0x777777777777ULL},
      Descriptor{
          "blk.{}.ssm_conv1d.weight", {4, 10240, 1}, 2, format::Encoding::kF32, 0x777777777777ULL},
      Descriptor{"blk.{}.ssm_dt.bias", {48, 1, 1}, 1, format::Encoding::kF32, 0x777777777777ULL},
      Descriptor{
          "blk.{}.ssm_norm.weight", {128, 1, 1}, 1, format::Encoding::kF32, 0x777777777777ULL},
      Descriptor{
          "blk.{}.ssm_out.weight", {6144, 2560, 1}, 2, format::Encoding::kQ6_K, 0x777777777777ULL},
      Descriptor{"output.weight", {2560, 248320, 1}, 2, format::Encoding::kQ6_K, 0x0ULL},
      Descriptor{"output_hc_down.weight", {10240, 320, 1}, 2, format::Encoding::kQ8_0, 0x0ULL},
      Descriptor{"output_hc_norm.weight", {10240, 1, 1}, 1, format::Encoding::kF32, 0x0ULL},
      Descriptor{"output_hc_up.weight", {320, 10240, 1}, 2, format::Encoding::kQ8_0, 0x0ULL},
      Descriptor{
          "per_layer_token_embd.weight", {160, 320001536, 1}, 2, format::Encoding::kIQ4_NL, 0x0ULL},
      Descriptor{"token_embd.weight", {2560, 248320, 1}, 2, format::Encoding::kQ6_K, 0x0ULL},
  };
  for (const auto& d : descriptors) {
    const auto add = [&](std::string name) {
      const auto shape = format::ShapeOf(std::span(d.dims).first(d.rank));
      base::Check(shape.has_value(), "recorded GGUF fixture shape");
      const auto& info = format::Info(d.encoding);
      format::TensorFormat f{
          .encoding = d.encoding,
          .layout = info.quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense,
          .shape = *shape};
      const auto parts = format::PartsOf(f);
      base::Check(parts.has_value(), "recorded GGUF fixture format");
      const auto at = (out.files[0].bytes + 31) / 32 * 32;
      out.tensors.push_back(
          {.name = std::move(name),
           .format = f,
           .experts = d.rank == 3,
           .extent = {.offset = at, .bytes = (*parts)[0].bytes},
           .dtype = info.name,
           .dims = std::vector<std::uint64_t>(d.dims.begin(), d.dims.begin() + d.rank)});
      out.files[0].bytes = at + (*parts)[0].bytes;
    };
    if (!d.layers) {
      add(std::string(d.name));
      continue;
    }
    for (std::uint32_t layer = 0; layer < 48; ++layer) {
      if (!(d.layers & (std::uint64_t{1} << layer))) continue;
      const auto at = d.name.find("{}");
      add(std::string(d.name.substr(0, at)) + std::to_string(layer) +
          std::string(d.name.substr(at + 2)));
    }
  }
  base::Check(out.tensors.size() == 1224, "complete source metadata fixture");
  return out;
}
}  // namespace llmp::test_support
#endif  // LLMP_TESTS_UNIT_QWEN38_GGUF_NORMALIZATION_FIXTURE_H_
