// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's fixture imports as resource tables (families/qwen.h
// Qwen38Resource, shaped and typed as the M3 importers wrote them:
// docs/artifact-format.md), and as mapping sources
// (fixture_source.h FixtureSource, the ModelOpt pairs assembled): the
// ModelOpt import with its experts in GGML's NVFP4 blocks (67617f87) or the
// CUTLASS layout (c4fb47a9), a GGUF quantization's (of this fixture's
// types), and the MTP drafter's; and the native importer's ModelOpt and MLX
// targets.

#ifndef LLMP_TESTS_UNIT_QWEN38_FIXTURE_H_
#define LLMP_TESTS_UNIT_QWEN38_FIXTURE_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/json.h"
#include "artifact/representation.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "families/qwen.h"
#include "fixture_source.h"
#include "format/tensor_format.h"
#include "import/mlx_qwen38.h"
#include "import/modelopt_qwen38.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"

namespace llmp::test_support::qwen38 {

// The n-gram table's rows (16 heads of 20,000,096 values, qwen38_test's).
inline constexpr std::uint64_t kTableRows = 320001536;

enum class Form : std::uint8_t { kGgmlExperts, kCutlass, kGguf };

// The target's resources, in the importers' order, then its expert arrays.
inline std::vector<families::Qwen38Resource> Resources(const families::Qwen38Profile& p,
                                                       Form form) {
  std::vector<families::Qwen38Resource> r;
  std::vector<families::Qwen38Resource> arrays;
  const bool gguf = form == Form::kGguf;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto plain = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = true, .type = std::move(type), .ne = std::move(ne)});
  };
  // A matrix: MXFP8 codes and scales, or a GGUF's Q8_0.
  const auto mx = [&](const std::string& name, std::uint64_t k, std::uint64_t n) {
    if (gguf) {
      ggml(name + ".weight", "Q8_0", {k, n});
      return;
    }
    plain(name + ".weight", "F8_E4M3", {k, n});
    plain(name + ".weight_scale", "U8", {k / 32, n});
  };
  // A BF16 matrix of the ModelOpt import, a GGUF's Q8_0 (or F32 where `f32`).
  const auto bf = [&](const std::string& name, std::uint64_t k, std::uint64_t n, bool f32 = false) {
    ggml(name, gguf ? (f32 ? "F32" : "Q8_0") : "BF16", {k, n});
  };
  const std::uint64_t w = p.width;
  const std::uint64_t hw = p.hc_width();
  bf("token_embd.weight", w, p.vocab);
  bf("output.weight", w, p.vocab);
  ggml("output_hc_norm.weight", "F32", {hw});
  bf("output_hc_down.weight", hw, p.hc_rank);
  bf("output_hc_up.weight", p.hc_rank, hw);
  if (gguf) {
    ggml("per_layer_token_embd.weight", "F16", {p.ple_row, kTableRows});
  } else {
    plain("per_layer_token_embd.weight", "U8", {(p.ple_row / 2) + (p.ple_row / 16), kTableRows});
    plain("per_layer_token_embd.weight_scale_2", "F32", {1});
  }
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    for (const char* kind : {"attn", "ffn"}) {
      ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {hw});
      bf(std::format("{}hc_{}_down.weight", n, kind), hw, p.hc_rank);
      bf(std::format("{}hc_{}_up.weight", n, kind), p.hc_rank, hw);
      bf(std::format("{}hc_{}_inject.weight", n, kind), hw, p.hc, true);
    }
    if (p.linear(il)) {
      mx(n + "attn_qkv", w, p.conv_channels());
      mx(n + "attn_gate", w, p.lin_v_width());
      if (gguf) {
        ggml(n + "ssm_beta.weight", "F32", {w, p.lin_v_heads});
        ggml(n + "ssm_alpha.weight", "F32", {w, p.lin_v_heads});
      } else {
        mx(n + "ssm_beta", w, p.lin_v_heads);
        mx(n + "ssm_alpha", w, p.lin_v_heads);
      }
      ggml(n + "ssm_dt.bias", "F32", {p.lin_v_heads});
      ggml(n + "ssm_a", "F32", {p.lin_v_heads});
      ggml(n + "ssm_conv1d.weight", "F32", {p.conv, p.conv_channels()});
      ggml(n + "ssm_norm.weight", "F32", {p.lin_head_dim});
      mx(n + "ssm_out", p.lin_v_width(), w);
    } else {
      mx(n + "attn_q", w, 2 * std::uint64_t{p.heads} * p.head_dim);
      mx(n + "attn_k", w, std::uint64_t{p.kv_heads} * p.head_dim);
      mx(n + "attn_v", w, std::uint64_t{p.kv_heads} * p.head_dim);
      mx(n + "attn_output", std::uint64_t{p.heads} * p.head_dim, w);
      ggml(n + "attn_q_norm.weight", "F32", {p.head_dim});
      ggml(n + "attn_k_norm.weight", "F32", {p.head_dim});
      if (gguf) {
        ggml(n + "indexer.q_proj.weight", "BF16",
             {w, std::uint64_t{p.indexer_heads} * p.indexer_head_dim});
        ggml(n + "indexer.k_proj.weight", "BF16", {w, p.indexer_head_dim});
      } else {
        mx(n + "indexer.qk_proj", w, std::uint64_t{p.indexer_heads + 1} * p.indexer_head_dim);
      }
      ggml(n + "indexer.q_norm.weight", "F32", {p.indexer_head_dim});
      ggml(n + "indexer.k_norm.weight", "F32", {p.indexer_head_dim});
    }
    if (il == p.ple_layer) {
      bf(n + "ple_key.weight", p.ple_width(), hw);
      bf(n + "ple_value.weight", p.ple_width(), w);
      for (const char* part : {"key", "query", "conv"}) {
        ggml(std::format("{}ple_norm_{}.weight", n, part), "F32", {hw});
      }
      ggml(n + "ple_conv1d.weight", "F32", {p.ple_conv, hw});
      if (!gguf) {
        plain(n + "ple_multipliers", "I64", {p.ngram});
        plain(n + "ple_head_offsets", "I64", {p.ple_heads()});
        plain(n + "ple_head_vocab", "I64", {p.ple_heads()});
      }
    }
    bf(n + "ffn_gate_inp.weight", w, p.experts, true);
    ggml(n + "ffn_gate_inp_shexp.weight", gguf ? "F32" : "BF16", {w});
    mx(n + "ffn_gate_shexp", w, p.shared_ffn);
    mx(n + "ffn_up_shexp", w, p.shared_ffn);
    mx(n + "ffn_down_shexp", p.shared_ffn, w);
    const std::uint64_t f = p.expert_ffn;
    const auto array = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
      arrays.push_back({.roles = {std::move(name)},
                        .plain = false,
                        .type = std::move(type),
                        .ne = std::move(ne),
                        .expert_array = true,
                        .count = p.experts});
    };
    if (form == Form::kCutlass) {
      for (const char* proj : {"gate", "up", "down"}) {
        ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {p.experts});
      }
      const std::uint64_t gu_atoms = ((2 * f + 127) / 128) * (w / 64);
      const std::uint64_t d_atoms = ((w + 127) / 128) * (f / 64);
      array(n + "ffn_gate_up_exps.codes", "I8", {w / 2, 2 * f});
      array(n + "ffn_gate_up_exps.scales", "I8", {512, gu_atoms});
      array(n + "ffn_down_exps.codes", "I8", {f / 2, w});
      array(n + "ffn_down_exps.scales", "I8", {512, d_atoms});
      continue;
    }
    for (const char* proj : {"gate", "up", "down"}) {
      const bool down = std::string_view(proj) == "down";
      if (!gguf) ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {p.experts});
      array(std::format("{}ffn_{}_exps.weight", n, proj), gguf ? "Q4_0" : "NVFP4",
            down ? std::vector<std::uint64_t>{f, w} : std::vector<std::uint64_t>{w, f});
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// The MTP drafter's resources (its layer's BF16 linears, CUTLASS experts).
inline std::vector<families::Qwen38Resource> MtpResources(const families::Qwen38Profile& p,
                                                          std::uint64_t selected_rows = 0) {
  std::vector<families::Qwen38Resource> r;
  std::vector<families::Qwen38Resource> arrays;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  const std::uint64_t w = p.width;
  const std::uint64_t hw = p.hc_width();
  const std::uint64_t f = p.expert_ffn;
  ggml("fc_embd.weight", "BF16", {w, w});
  ggml("fc_hidden.weight", "BF16", {w, w});
  ggml("norm_embd.weight", "F32", {w});
  ggml("norm_hidden.weight", "F32", {hw});
  for (const char* kind : {"attn", "ffn"}) {
    ggml(std::format("blk.0.hc_{}_norm.weight", kind), "F32", {hw});
    ggml(std::format("blk.0.hc_{}_down.weight", kind), "BF16", {hw, p.hc_rank});
    ggml(std::format("blk.0.hc_{}_up.weight", kind), "BF16", {p.hc_rank, hw});
    ggml(std::format("blk.0.hc_{}_inject.weight", kind), "BF16", {hw, p.hc});
  }
  ggml("blk.0.attn_q.weight", "BF16", {w, 2 * std::uint64_t{p.heads} * p.head_dim});
  ggml("blk.0.attn_k.weight", "BF16", {w, std::uint64_t{p.kv_heads} * p.head_dim});
  ggml("blk.0.attn_v.weight", "BF16", {w, std::uint64_t{p.kv_heads} * p.head_dim});
  ggml("blk.0.attn_output.weight", "BF16", {std::uint64_t{p.heads} * p.head_dim, w});
  ggml("blk.0.attn_q_norm.weight", "F32", {p.head_dim});
  ggml("blk.0.attn_k_norm.weight", "F32", {p.head_dim});
  ggml("blk.0.indexer.qk_proj.weight", "BF16",
       {w, std::uint64_t{p.indexer_heads + 1} * p.indexer_head_dim});
  ggml("blk.0.indexer.q_norm.weight", "F32", {p.indexer_head_dim});
  ggml("blk.0.indexer.k_norm.weight", "F32", {p.indexer_head_dim});
  ggml("blk.0.ffn_gate_inp.weight", "BF16", {w, p.experts});
  ggml("blk.0.ffn_gate_inp_shexp.weight", "BF16", {w});
  ggml("blk.0.ffn_gate_shexp.weight", "BF16", {w, p.shared_ffn});
  ggml("blk.0.ffn_up_shexp.weight", "BF16", {w, p.shared_ffn});
  ggml("blk.0.ffn_down_shexp.weight", "BF16", {p.shared_ffn, w});
  for (const char* proj : {"gate", "up", "down"}) {
    ggml(std::format("blk.0.ffn_{}_exps.weight_scale_2", proj), "F32", {p.experts});
  }
  ggml("output_hc_norm.weight", "F32", {hw});
  ggml("output_hc_down.weight", "BF16", {hw, p.hc_rank});
  ggml("output_hc_up.weight", "BF16", {p.hc_rank, hw});
  if (selected_rows != 0) {
    ggml("draft_output.weight", "BF16", {w, selected_rows});
    ggml("draft_output.ids", "I32", {1, selected_rows});
  }
  const std::uint64_t gu_atoms = ((2 * f + 127) / 128) * (w / 64);
  const std::uint64_t d_atoms = ((w + 127) / 128) * (f / 64);
  const auto array = [&](std::string name, std::vector<std::uint64_t> ne) {
    arrays.push_back({.roles = {std::move(name)},
                      .plain = false,
                      .type = "I8",
                      .ne = std::move(ne),
                      .expert_array = true,
                      .count = p.experts});
  };
  array("blk.0.ffn_gate_up_exps.codes", {w / 2, 2 * f});
  array("blk.0.ffn_gate_up_exps.scales", {512, gu_atoms});
  array("blk.0.ffn_down_exps.codes", {f / 2, w});
  array("blk.0.ffn_down_exps.scales", {512, d_atoms});
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// A resource's v0 representation.
inline artifact::Representation RepresentationOf(const families::Qwen38Resource& r) {
  std::string ne;
  std::vector<std::uint64_t> dims = r.ne;
  if (r.plain) std::ranges::reverse(dims);  // a plain shape is row-major
  for (const auto d : dims) ne += (ne.empty() ? "" : ",") + std::to_string(d);
  const std::string text =
      r.plain ? std::format(R"({{"dtype":"{}","family":"plain","shape":[{}]}})", r.type, ne)
              : std::format(R"({{"family":"ggml","ne":[{}],"type":"{}"}})", ne, r.type);
  const auto doc = artifact::json::Parse(text);
  EXPECT_TRUE(doc.has_value()) << text;
  const auto rep = artifact::ParseRepresentation(doc->root());
  EXPECT_TRUE(rep.has_value()) << text;
  return rep.value_or(artifact::Representation{});
}

// The folds the native ModelOpt importer records (sorted), which a
// ModelOpt-form fixture's source carries; a GGUF's come from its mapping's
// fold rules.
inline std::vector<std::string> ModelOptFolds() {
  return {"gdn.value_heads_tiled", "norm.plus_one", "ssm.a_neg_exp"};
}

// The resources as a mapping's source (the ModelOpt pairs assembled), with
// `folds`.
inline spec::Source SourceOf(std::string architecture, std::string container,
                             const std::vector<families::Qwen38Resource>& resources,
                             std::vector<std::string> folds = {}) {
  std::deque<artifact::Representation> reps;
  std::vector<FixtureTensor> tensors;
  for (const families::Qwen38Resource& r : resources) {
    reps.push_back(RepresentationOf(r));
    tensors.push_back({.name = r.roles.front(),
                       .roles = r.roles,
                       .repr = &reps.back(),
                       .count = r.expert_array ? r.count : 0});
  }
  auto source = FixtureSource(std::move(architecture), std::move(container), tensors,
                              kQwen38ModelOptAssembly, std::move(folds));
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  return source.value_or(spec::Source{});
}

// The target's source in `form`: a GGUF's, or a ModelOpt import's with the
// importer's folds.
inline spec::Source TargetSource(const families::Qwen38Profile& p, Form form) {
  const bool gguf = form == Form::kGguf;
  return SourceOf("qwen4exp", gguf ? "gguf" : "safetensors", Resources(p, form),
                  gguf ? std::vector<std::string>{} : ModelOptFolds());
}

// The MTP drafter's source (a ModelOpt import's, with its folds).
inline spec::Source MtpSource(const families::Qwen38Profile& p, std::uint64_t selected_rows = 0) {
  return SourceOf("qwen4exp-mtp", "safetensors", MtpResources(p, selected_rows), ModelOptFolds());
}

// The target's values in `form` (qwen4exp's, Qwen38Mapping).
inline std::expected<std::vector<spec::SourceValue>, std::string> TargetValues(
    const families::Qwen38Profile& p, const families::Qwen38PleHash& hash, Form form) {
  if (form == Form::kGguf) return families::Qwen38GgufValues(p, hash);
  return families::Qwen38ModelOptValues(
      p, hash,
      form == Form::kCutlass ? families::Qwen38Experts::kCutlass : families::Qwen38Experts::kGgml);
}

struct NativeModelOptFixture {
  spec::Source source{};
  spec::ModelSpec model{};
  std::vector<spec::SourceValue> values{};
};

// The native ModelOpt target as its importer publishes it
// (import::ModelOptNativeTargetForm): relaid-out experts, folded norms and A.
inline NativeModelOptFixture NativeModelOpt() {
  NativeModelOptFixture result;
  auto form = import::ModelOptNativeTargetForm();
  EXPECT_TRUE(form.has_value());
  if (!form) return result;
  result.source = std::move(form->source);
  result.values = std::move(form->values);
  auto made = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), result.source, result.values);
  EXPECT_TRUE(made.has_value());
  if (made) result.model = std::move(*made);
  return result;
}

inline std::array<artifact::Source, 38> ModelOptOrigins() {
  constexpr std::array<std::tuple<std::string_view, std::uint64_t, std::string_view>, 38> kPins{
      {{"config.json", 109897ULL,
        "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
       {"hf_quant_config.json", 99088ULL,
        "d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7"},
       {"model-00001-of-00034.safetensors", 2841505880ULL,
        "3f870511a20ad170c997517765f03d475bcd364ae1a6011491085cab620d3c39"},
       {"model-00002-of-00034.safetensors", 1575009800ULL,
        "fe8f4bdbce28f4911f10460b75581b6c88a8ade9b836d2f1ec22c6bee6f976a8"},
       {"model-00003-of-00034.safetensors", 1575009808ULL,
        "2b7d1312a660cf17bb05c34e1ac79570820f86fb1e1115108ee40562e48acb5e"},
       {"model-00004-of-00034.safetensors", 1575009808ULL,
        "25ba4508c209a5cc54c04edb1edb8d0a0b5efa80e5f99637e2263fa78d6a25e8"},
       {"model-00005-of-00034.safetensors", 1575009808ULL,
        "a9848067bb49f44b8339bcdca9d60ca74d801a6df874c979c6f901c8da6ff545"},
       {"model-00006-of-00034.safetensors", 1575009808ULL,
        "66fc496b58675a90b94d9f36765fe67692dd8a4c966ec0692e076d751f835c8d"},
       {"model-00007-of-00034.safetensors", 1575009808ULL,
        "68f89e84be13069bd9f2928b44a9f9638d628fd62be5abb6a4fc956c1af25ea9"},
       {"model-00008-of-00034.safetensors", 1575009808ULL,
        "b8bdaa0a7b027dc0fa38c2e43bb31e8ddf9ba9288d3d0f28c422e6947241d96e"},
       {"model-00009-of-00034.safetensors", 1575009808ULL,
        "6c5a85990b16a93f0e7e28fb76f56713d5cbc05bef494bb15360188f575746e9"},
       {"model-00010-of-00034.safetensors", 1575009808ULL,
        "8734a1fc3b5857391fbf2252d4305391a9f77894b70577949f472fc75386582f"},
       {"model-00011-of-00034.safetensors", 1575009808ULL,
        "29792378d43c9f70d9f3fbdd07d0ee24b38f81fe27aff25a02ba0d3cde8df13c"},
       {"model-00012-of-00034.safetensors", 1575009808ULL,
        "b23e615ebefff4753d982d9ec6a8be6ad57353a8900146a3ec4e217f81c10e61"},
       {"model-00013-of-00034.safetensors", 1575009808ULL,
        "9da7b4570fc3f13cb781e5aca5e79d0edfc10be5555a18bc7bda8f47459d5acf"},
       {"model-00014-of-00034.safetensors", 1575009808ULL,
        "53e770d0bdf6261e30abaae2477aa0b50f6686511e0240514ded0fd65edbfeba"},
       {"model-00015-of-00034.safetensors", 1575009816ULL,
        "03440de56d127ff58e72963aafd0de0574b825667330c50b4441bb300dae9f06"},
       {"model-00016-of-00034.safetensors", 1575009824ULL,
        "ce2e0474ccb34c56003a61608b46d3a8f668c218a9993c1f861239df83afe77b"},
       {"model-00017-of-00034.safetensors", 1575009824ULL,
        "57f84845e0b8bcab738e70f0f880e941f1baa8da14663293d7d40481288c73a0"},
       {"model-00018-of-00034.safetensors", 1575009824ULL,
        "3477df0be60910576fc5a1f8159d8dbbf7847a03872b9cb53664808e1d5dfac8"},
       {"model-00019-of-00034.safetensors", 2438722732ULL,
        "43c22213b2fe0990c7fc2aa54c0ea65062eb93253b4ea419914d1a655e64b23a"},
       {"model-00020-of-00034.safetensors", 5133924592ULL,
        "b7143694bf8698c5b704ae9b3d492a0e9b8ef57bdda8ae6f3f29f264c183ad17"},
       {"model-00021-of-00034.safetensors", 5194891408ULL,
        "522bbbc894183865726e06ca44021d275d03cc0e8dcd60a2c7074d870e811d37"},
       {"model-00022-of-00034.safetensors", 5133933936ULL,
        "5a4789f6da6b386b09cb9093b0315fe12a816142958e4d8a4ae83e4fbaa066e7"},
       {"model-00023-of-00034.safetensors", 5194908864ULL,
        "cd0aaef1c8688cb4e35b488048ede2a6b653e1d558b20d4657567df24dc8be13"},
       {"model-00024-of-00034.safetensors", 5188755712ULL,
        "d07198edd38ae8740d556487729ae87be1e21e31a88cf6624c1b3f9d0fc6de76"},
       {"model-00025-of-00034.safetensors", 5133940040ULL,
        "c7d5e642968679ca14d07bdb1e8c2187c59898611d05a6909c209738642c9519"},
       {"model-00026-of-00034.safetensors", 5194907416ULL,
        "47a7c7a70188cb3dc318071a5278260c4b05d7ce1b69c3b99e7bf1c0e42c353b"},
       {"model-00027-of-00034.safetensors", 5133940216ULL,
        "a81fedf160434806e81f36731c853fe4e3f973dea2f0817043987f32b4915a54"},
       {"model-00028-of-00034.safetensors", 5194907672ULL,
        "91c5d479fe4de1035c430ad7fd3de9d15214a46fd786ca7909d1f2fa40ef603a"},
       {"model-00029-of-00034.safetensors", 5133939992ULL,
        "f31c6f61968f6f9e85d469f31c55c82ab6c5f2bc54c1d7a0cd6b23ac45f0ce15"},
       {"model-00030-of-00034.safetensors", 5194908960ULL,
        "df8627f5b8b0711b2ebf415c1f2549ef0b50f964e0babb560469b191b33c6a14"},
       {"model-00031-of-00034.safetensors", 5188755536ULL,
        "654c8f9ae1ab5de393cbcdd46a13bd0d3d1c1005e575879e0e660d40ad9ef425"},
       {"model-00032-of-00034.safetensors", 5133940208ULL,
        "7de2e680802ad92a0e019f875ebc07f2d5d82c8bca1c64b0f817d547bc3c9de7"},
       {"model-00033-of-00034.safetensors", 5021547836ULL,
        "0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11"},
       {"model-00034-of-00034.safetensors", 1597270800ULL,
        "fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203"},
       {"model-inputscales.safetensors", 9669936ULL,
        "49c7d04606ff7c8f103b4d1fc9742b44649a2937b25ecc896a5450ab1020f308"},
       {"model.safetensors.index.json", 33074286ULL,
        "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9"}}};
  std::array<artifact::Source, 38> out;
  for (std::size_t i = 0; i < 38; ++i) {
    const auto& [name, bytes, hex] = kPins[i];
    base::Sha256Digest digest{};
    for (std::size_t b = 0; b < digest.size(); ++b) {
      unsigned byte = 0;
      std::from_chars(hex.data() + 2 * b, hex.data() + 2 * b + 2, byte, 16);
      digest[b] = static_cast<std::uint8_t>(byte);
    }
    out[i] = {std::string(name), base::Bytes(bytes), digest};
  }
  return out;
}

struct NativeMlxFixture {
  spec::Source source{};
  spec::ModelSpec model{};
  std::vector<spec::SourceValue> values{};
};

inline NativeMlxFixture NativeMlx() {
  NativeMlxFixture result;
  result.source = import::ApprovedMlxTargetSource();
  const families::Qwen38PleHash hash{
      .multipliers = {23703573157769, 20109073645365, 8052911324071},
      .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                  160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                  300001275},
      .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}};
  families::Qwen38Profile p = families::Qwen38Flash();
  p.ple_row = 160;
  auto values = families::Qwen38GgufValues(p, hash);
  EXPECT_TRUE(values.has_value());
  if (values) {
    result.values = std::move(*values);
    result.values.push_back({"llmp.qwen4exp.indexer", std::string("fused")});
    result.values.push_back({"llmp.qwen4exp.experts_stacked", false});
    result.values.push_back({"llmp.qwen4exp.expert_scales", false});
  }
  auto made = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), result.source, result.values);
  EXPECT_TRUE(made.has_value());
  if (made) result.model = std::move(*made);
  return result;
}

inline std::array<artifact::Source, 24> MlxOrigins() {
  constexpr std::array<std::tuple<std::string_view, std::uint64_t, std::string_view>, 24> kPins{
      {{"config.json", 5361ULL, "4efd5f41c47e8de9f55c678df4a237cbbc78e4d517dad42a906889f7d94f79a2"},
       {"model-00001-of-00022.safetensors", 5283195887ULL,
        "308f20b7e35a3e525f36eca0df286f6d1043cfeb621e7bbcf372a834c99cd66d"},
       {"model-00002-of-00022.safetensors", 5250035137ULL,
        "8af1a616f2f39dbfe6ee92e959afb2414623fcb299aff1ced55984bfe810d608"},
       {"model-00003-of-00022.safetensors", 5250035154ULL,
        "519a0472252ac9a51af020686f462aa6be82ce655f948fd5f054d70949de17a6"},
       {"model-00004-of-00022.safetensors", 5250035160ULL,
        "b1ad7e951969eb8dc4e92762dc2e50bd47353c95990ae207db5da393d441400f"},
       {"model-00005-of-00022.safetensors", 5250035164ULL,
        "9ca797304c563bb65f2356b67d444511b11744bc5b5c8ec8345ae7d97d3e251c"},
       {"model-00006-of-00022.safetensors", 5250035184ULL,
        "8aab4c1dce7d8a0af645426b66d1caf30245d422d4c6b7029583959d675a1827"},
       {"model-00007-of-00022.safetensors", 5067850140ULL,
        "7908da37b2e3fa7f0de8b9734dad0759be7dc445716f3acd59c20c4842638db9"},
       {"model-00008-of-00022.safetensors", 5337131618ULL,
        "85beff20f250bfe63e95f12c0460512aab92bcde8f7408fe9c20e3de0321bc1d"},
       {"model-00009-of-00022.safetensors", 5337131558ULL,
        "f7c03ae640e9bf0f7813ef2b2980afa83a883c02b3988e814e9921055fc08a28"},
       {"model-00010-of-00022.safetensors", 4970130103ULL,
        "6897f52227533e720b243d6aef24c1293ad1eb02a1bb2ec6a6e4dc48d56d0c8f"},
       {"model-00011-of-00022.safetensors", 5341257482ULL,
        "905d27f9db887284289a043f9638020258c5f90d52122cdce2e7aef941a431f2"},
       {"model-00012-of-00022.safetensors", 4967968541ULL,
        "c4d2bb76fa4613f7386ddaa7045608c320f9736623211a814aa28fb370530dec"},
       {"model-00013-of-00022.safetensors", 5337131651ULL,
        "1f20a5e1f2256b2fb6a78fef52f9e66daa970f4318ab1be3abb32a1effedbbf3"},
       {"model-00014-of-00022.safetensors", 5337131799ULL,
        "860d12691792df5f08c9dfe245dc356773741f1edcc1fd430672ef06ac387635"},
       {"model-00015-of-00022.safetensors", 4970130195ULL,
        "259fb7577b8248f6dcbe6a932fe5d2f1c8869eda761ef3394fe6ad06065f216c"},
       {"model-00016-of-00022.safetensors", 5341257440ULL,
        "e931785f9ff4e0a567bf8c06df37639be46b4dae156e0f8b39b9d32a03d75792"},
       {"model-00017-of-00022.safetensors", 4967968481ULL,
        "4f4610dc4a862bf9d93d70cfd0ec037e8e7943a734794d34eb5c0d748b74747a"},
       {"model-00018-of-00022.safetensors", 5337131695ULL,
        "afdb4828920186600f9d61537d1465979fb36692f4eddb8c5a961a6b72964b77"},
       {"model-00019-of-00022.safetensors", 5337131801ULL,
        "869d34c6e8f7db79bd4e54b9833d5b0db45eefac860c9bd64fbd312bf862a667"},
       {"model-00020-of-00022.safetensors", 4970130135ULL,
        "5ca2f994fec1aa06fefef6646b9f072cf70f7f73aa8dab4c7c1a9a424afb4a52"},
       {"model-00021-of-00022.safetensors", 5341257436ULL,
        "58def6762f29cc798437e5649a56ee75d053faca69138a2454dbdb34bb8a37c3"},
       {"model-00022-of-00022.safetensors", 3715570974ULL,
        "30a3c82b573d813a1d2349f1c05c308c0d415f3ca1bb67caa0bfaad8a6afa7a3"},
       {"model.safetensors.index.json", 420813ULL,
        "8848503860c7da8619fdc8d5e65201868f8f1d21d69d2e21166c95522379ff77"}}};
  std::array<artifact::Source, 24> out;
  for (std::size_t i = 0; i < 24; ++i) {
    const auto& [name, bytes, hex] = kPins[i];
    base::Sha256Digest digest{};
    for (std::size_t b = 0; b < digest.size(); ++b) {
      unsigned byte = 0;
      std::from_chars(hex.data() + 2 * b, hex.data() + 2 * b + 2, byte, 16);
      digest[b] = static_cast<std::uint8_t>(byte);
    }
    out[i] = {std::string(name), base::Bytes(bytes), digest};
  }
  return out;
}

}  // namespace llmp::test_support::qwen38

#endif  // LLMP_TESTS_UNIT_QWEN38_FIXTURE_H_
