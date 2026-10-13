// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_UNIT_DSV4_GGUF_NORMALIZATION_FIXTURE_H_
#define LLMP_TESTS_UNIT_DSV4_GGUF_NORMALIZATION_FIXTURE_H_

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
// Independent bounded header inventory of fbbb5b93 UD-Q2_K_XL, three shards.
// Header hashes de65c5bb / 294896f9 / a97eea9d; synthetic extents, no payload.
inline import::Checkpoint Dsv4GgufCheckpoint() {
  import::Checkpoint out{
      .container = "gguf", .architecture = "deepseek4", .files = {{.name = "memory", .bytes = 0}}};
  out.metadata.push_back({"general.architecture", std::string("deepseek4")});
  out.metadata.push_back(
      {"deepseek4.attention.compress_ratios",
       std::vector<std::int64_t>{0, 0,   4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,   0, 0}});
  out.metadata.push_back({"deepseek4.attention.compress_rope_freq_base", 160000.0});
  out.metadata.push_back({"deepseek4.attention.head_count", std::int64_t{64}});
  out.metadata.push_back({"deepseek4.attention.head_count_kv", std::int64_t{1}});
  out.metadata.push_back({"deepseek4.attention.indexer.head_count", std::int64_t{64}});
  out.metadata.push_back({"deepseek4.attention.indexer.key_length", std::int64_t{128}});
  out.metadata.push_back({"deepseek4.attention.indexer.top_k", std::int64_t{512}});
  out.metadata.push_back({"deepseek4.attention.key_length", std::int64_t{512}});
  out.metadata.push_back({"deepseek4.attention.layer_norm_rms_epsilon", 9.999999974752427e-07});
  out.metadata.push_back({"deepseek4.attention.output_group_count", std::int64_t{8}});
  out.metadata.push_back({"deepseek4.attention.output_lora_rank", std::int64_t{1024}});
  out.metadata.push_back({"deepseek4.attention.q_lora_rank", std::int64_t{1024}});
  out.metadata.push_back({"deepseek4.attention.sliding_window", std::int64_t{128}});
  out.metadata.push_back({"deepseek4.attention.value_length", std::int64_t{512}});
  out.metadata.push_back({"deepseek4.block_count", std::int64_t{43}});
  out.metadata.push_back({"deepseek4.context_length", std::int64_t{1048576}});
  out.metadata.push_back({"deepseek4.embedding_length", std::int64_t{4096}});
  out.metadata.push_back({"deepseek4.expert_count", std::int64_t{256}});
  out.metadata.push_back({"deepseek4.expert_feed_forward_length", std::int64_t{2048}});
  out.metadata.push_back({"deepseek4.expert_gating_func", std::int64_t{4}});
  out.metadata.push_back({"deepseek4.expert_shared_count", std::int64_t{1}});
  out.metadata.push_back({"deepseek4.expert_used_count", std::int64_t{6}});
  out.metadata.push_back({"deepseek4.expert_weights_norm", true});
  out.metadata.push_back({"deepseek4.expert_weights_scale", 1.5});
  out.metadata.push_back({"deepseek4.hash_layer_count", std::int64_t{3}});
  out.metadata.push_back({"deepseek4.hyper_connection.count", std::int64_t{4}});
  out.metadata.push_back({"deepseek4.hyper_connection.epsilon", 9.999999974752427e-07});
  out.metadata.push_back({"deepseek4.hyper_connection.sinkhorn_iterations", std::int64_t{20}});
  out.metadata.push_back({"deepseek4.rope.dimension_count", std::int64_t{64}});
  out.metadata.push_back({"deepseek4.rope.freq_base", 10000.0});
  out.metadata.push_back({"deepseek4.rope.scaling.factor", 16.0});
  out.metadata.push_back({"deepseek4.rope.scaling.original_context_length", std::int64_t{65536}});
  out.metadata.push_back({"deepseek4.rope.scaling.type", std::string("yarn")});
  out.metadata.push_back({"deepseek4.rope.scaling.yarn_beta_fast", 32.0});
  out.metadata.push_back({"deepseek4.rope.scaling.yarn_beta_slow", 1.0});
  out.metadata.push_back(
      {"deepseek4.swiglu_clamp_exp",
       std::vector<double>{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
  out.metadata.push_back(
      {"deepseek4.swiglu_clamp_shexp",
       std::vector<double>{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
  std::ranges::sort(out.metadata, {}, &import::MetaEntry::key);
  struct Descriptor {
    std::string_view name;
    std::array<std::uint64_t, 3> dims;
    std::uint8_t rank;
    format::Encoding encoding;
    std::uint64_t layers;
  };
  constexpr std::array descriptors{
      Descriptor{"blk.{}.attn_compressor_ape.weight",
                 {512, 128, 1},
                 2,
                 format::Encoding::kF32,
                 0x2aaaaaaaaa8ULL},
      Descriptor{"blk.{}.attn_compressor_ape.weight",
                 {1024, 4, 1},
                 2,
                 format::Encoding::kF32,
                 0x55555555554ULL},
      Descriptor{"blk.{}.attn_compressor_gate.weight",
                 {4096, 512, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x2aaaaaaaaa8ULL},
      Descriptor{"blk.{}.attn_compressor_gate.weight",
                 {4096, 1024, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x55555555554ULL},
      Descriptor{"blk.{}.attn_compressor_kv.weight",
                 {4096, 512, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x2aaaaaaaaa8ULL},
      Descriptor{"blk.{}.attn_compressor_kv.weight",
                 {4096, 1024, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x55555555554ULL},
      Descriptor{"blk.{}.attn_compressor_norm.weight",
                 {512, 1, 1},
                 1,
                 format::Encoding::kF32,
                 0x7fffffffffcULL},
      Descriptor{
          "blk.{}.attn_kv.weight", {4096, 512, 1}, 2, format::Encoding::kQ8_0, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.attn_kv_a_norm.weight", {512, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.attn_norm.weight", {4096, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{"blk.{}.attn_output_a.weight",
                 {4096, 8192, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x7ffffffffffULL},
      Descriptor{"blk.{}.attn_output_b.weight",
                 {8192, 4096, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.attn_q_a.weight", {4096, 1024, 1}, 2, format::Encoding::kQ5_K, 0x7fffbffffffULL},
      Descriptor{
          "blk.{}.attn_q_a.weight", {4096, 1024, 1}, 2, format::Encoding::kQ6_K, 0x4000000ULL},
      Descriptor{
          "blk.{}.attn_q_a_norm.weight", {1024, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.attn_q_b.weight", {1024, 32768, 1}, 2, format::Encoding::kQ8_0, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.attn_sinks.weight", {64, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.exp_probs_b.bias", {256, 1, 1}, 1, format::Encoding::kF32, 0x7fffffffff8ULL},
      Descriptor{"blk.{}.ffn_down_exps.weight",
                 {2048, 4096, 256},
                 3,
                 format::Encoding::kIQ3_XXS,
                 0x3fffbffffffULL},
      Descriptor{"blk.{}.ffn_down_exps.weight",
                 {2048, 4096, 256},
                 3,
                 format::Encoding::kMxfp4,
                 0x40004000000ULL},
      Descriptor{"blk.{}.ffn_down_shexp.weight",
                 {2048, 4096, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0x7fffbffffffULL},
      Descriptor{"blk.{}.ffn_down_shexp.weight",
                 {2048, 4096, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x4000000ULL},
      Descriptor{"blk.{}.ffn_gate_exps.weight",
                 {4096, 2048, 256},
                 3,
                 format::Encoding::kIQ2_XS,
                 0x7fffbffffffULL},
      Descriptor{"blk.{}.ffn_gate_exps.weight",
                 {4096, 2048, 256},
                 3,
                 format::Encoding::kIQ3_XXS,
                 0x4000000ULL},
      Descriptor{"blk.{}.ffn_gate_inp.weight",
                 {4096, 256, 1},
                 2,
                 format::Encoding::kBF16,
                 0x7ffffffffffULL},
      Descriptor{"blk.{}.ffn_gate_shexp.weight",
                 {4096, 2048, 1},
                 2,
                 format::Encoding::kQ5_K,
                 0x7fffbffffffULL},
      Descriptor{"blk.{}.ffn_gate_shexp.weight",
                 {4096, 2048, 1},
                 2,
                 format::Encoding::kQ6_K,
                 0x4000000ULL},
      Descriptor{
          "blk.{}.ffn_gate_tid2eid.weight", {6, 129280, 1}, 2, format::Encoding::kI32, 0x7ULL},
      Descriptor{
          "blk.{}.ffn_norm.weight", {4096, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{"blk.{}.ffn_up_exps.weight",
                 {4096, 2048, 256},
                 3,
                 format::Encoding::kIQ2_XS,
                 0x7fffbffffffULL},
      Descriptor{"blk.{}.ffn_up_exps.weight",
                 {4096, 2048, 256},
                 3,
                 format::Encoding::kIQ3_XXS,
                 0x4000000ULL},
      Descriptor{"blk.{}.ffn_up_shexp.weight",
                 {4096, 2048, 1},
                 2,
                 format::Encoding::kQ5_K,
                 0x7fffbffffffULL},
      Descriptor{
          "blk.{}.ffn_up_shexp.weight", {4096, 2048, 1}, 2, format::Encoding::kQ6_K, 0x4000000ULL},
      Descriptor{
          "blk.{}.hc_attn_base.weight", {24, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.hc_attn_fn.weight", {16384, 24, 1}, 2, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.hc_attn_scale.weight", {3, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.hc_ffn_base.weight", {24, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.hc_ffn_fn.weight", {16384, 24, 1}, 2, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{
          "blk.{}.hc_ffn_scale.weight", {3, 1, 1}, 1, format::Encoding::kF32, 0x7ffffffffffULL},
      Descriptor{"blk.{}.indexer.attn_q_b.weight",
                 {1024, 8192, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x55555555554ULL},
      Descriptor{
          "blk.{}.indexer.proj.weight", {4096, 64, 1}, 2, format::Encoding::kF32, 0x55555555554ULL},
      Descriptor{"blk.{}.indexer_compressor_ape.weight",
                 {256, 4, 1},
                 2,
                 format::Encoding::kF32,
                 0x55555555554ULL},
      Descriptor{"blk.{}.indexer_compressor_gate.weight",
                 {4096, 256, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x55555555554ULL},
      Descriptor{"blk.{}.indexer_compressor_kv.weight",
                 {4096, 256, 1},
                 2,
                 format::Encoding::kQ8_0,
                 0x55555555554ULL},
      Descriptor{"blk.{}.indexer_compressor_norm.weight",
                 {128, 1, 1},
                 1,
                 format::Encoding::kF32,
                 0x55555555554ULL},
      Descriptor{"output.weight", {4096, 129280, 1}, 2, format::Encoding::kQ4_K, 0x0ULL},
      Descriptor{"output_hc_base.weight", {4, 1, 1}, 1, format::Encoding::kF32, 0x0ULL},
      Descriptor{"output_hc_fn.weight", {16384, 4, 1}, 2, format::Encoding::kF32, 0x0ULL},
      Descriptor{"output_hc_scale.weight", {1, 1, 1}, 1, format::Encoding::kF32, 0x0ULL},
      Descriptor{"output_norm.weight", {4096, 1, 1}, 1, format::Encoding::kF32, 0x0ULL},
      Descriptor{"token_embd.weight", {4096, 129280, 1}, 2, format::Encoding::kQ5_K, 0x0ULL},
  };
  for (const auto& d : descriptors) {
    const auto add = [&](std::string name) {
      auto shape = format::ShapeOf(std::span(d.dims).first(d.rank));
      base::Check(shape.has_value(), "captured DS shape");
      format::TensorFormat f{.encoding = d.encoding,
                             .layout = format::Info(d.encoding).quantized
                                           ? format::Layout::kGgmlBlocks
                                           : format::Layout::kDense,
                             .shape = *shape};
      auto parts = format::PartsOf(f);
      base::Check(parts && parts->size() == 1, "captured DS part");
      auto at = (out.files[0].bytes + 31) / 32 * 32;
      out.tensors.push_back({.name = std::move(name),
                             .format = f,
                             .experts = d.rank == 3,
                             .extent = {.offset = at, .bytes = (*parts)[0].bytes},
                             .dtype = format::Info(d.encoding).name,
                             .dims = {d.dims.begin(), d.dims.begin() + d.rank}});
      out.files[0].bytes = at + (*parts)[0].bytes;
    };
    if (!d.layers)
      add(std::string(d.name));
    else
      for (unsigned layer = 0; layer < 43; ++layer)
        if (d.layers & (1ULL << layer)) add(std::vformat(d.name, std::make_format_args(layer)));
  }
  return out;
}
}  // namespace llmp::test_support
#endif
