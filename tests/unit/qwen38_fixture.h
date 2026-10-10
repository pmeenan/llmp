// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's v0 artifacts as resource tables (model/qwen38.h Qwen38Resource,
// shaped and typed as the importers write them: docs/artifact-format.md),
// and as mapping sources (spec/mapping.h SourceOfV0Tensors, the mapping's
// pairs assembled): the ModelOpt import with its experts in GGML's NVFP4
// blocks (67617f87) or the CUTLASS layout (c4fb47a9), a GGUF
// quantization's (of this fixture's types), and the MTP drafter's.

#ifndef LLMP_TESTS_UNIT_QWEN38_FIXTURE_H_
#define LLMP_TESTS_UNIT_QWEN38_FIXTURE_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/json.h"
#include "artifact/representation.h"
#include "model/qwen38.h"
#include "spec/mapping.h"

namespace llmp::test_support::qwen38 {

// The n-gram table's rows (16 heads of 20,000,096 values, qwen38_test's).
inline constexpr std::uint64_t kTableRows = 320001536;

enum class Form : std::uint8_t { kGgmlExperts, kCutlass, kGguf };

// The target's resources, in the importers' order, then its expert arrays.
inline std::vector<model::Qwen38Resource> Resources(const model::Qwen38Profile& p, Form form) {
  std::vector<model::Qwen38Resource> r;
  std::vector<model::Qwen38Resource> arrays;
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
inline std::vector<model::Qwen38Resource> MtpResources(const model::Qwen38Profile& p,
                                                       std::uint64_t selected_rows = 0) {
  std::vector<model::Qwen38Resource> r;
  std::vector<model::Qwen38Resource> arrays;
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
inline artifact::Representation RepresentationOf(const model::Qwen38Resource& r) {
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

// The resources as a mapping's source (SourceOfV0's, its pairs assembled).
inline spec::Source SourceOf(std::string architecture, std::string container, std::string converter,
                             const std::vector<model::Qwen38Resource>& resources,
                             const spec::Mapping& mapping) {
  std::deque<artifact::Representation> reps;
  std::vector<spec::V0Tensor> tensors;
  for (const model::Qwen38Resource& r : resources) {
    reps.push_back(RepresentationOf(r));
    tensors.push_back({.name = r.roles.front(),
                       .roles = r.roles,
                       .repr = &reps.back(),
                       .count = r.expert_array ? r.count : 0});
  }
  auto source = spec::SourceOfV0Tensors(std::move(architecture), std::move(container),
                                        std::move(converter), tensors, &mapping);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  return source.value_or(spec::Source{});
}

// The converter identities of the importers (docs/artifact-layout).
inline constexpr std::string_view kModelOptConverter =
    "artifact-layout/import_m3.py m3-1+layout-a0d1980a9eddd1ad+modelopt_qwen38-031ed8750c9da55d";
inline constexpr std::string_view kGgufConverter =
    "artifact-layout/import_m3.py m3-1+layout-a0d1980a9eddd1ad";

}  // namespace llmp::test_support::qwen38

#endif  // LLMP_TESTS_UNIT_QWEN38_FIXTURE_H_
