// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's GGUF tensor tables as resource tables
// (tests/support/families/dsv4.h Dsv4Resource, shaped and typed as the
// GGUFs hold them), and as mapping sources (tests/support/fixture_source.h
// FixtureSource): the target's (every layer's window, compressor and
// indexer tensors by its compression ratio, its hash-routed layers' token
// tables) and the DSpark drafter's.

#ifndef LLMP_TESTS_UNIT_DSV4_FIXTURE_H_
#define LLMP_TESTS_UNIT_DSV4_FIXTURE_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "families/dsv4.h"
#include "fixture_source.h"
#include "spec/mapping.h"

namespace llmp::test_support::dsv4 {

using families::DsparkProfile;
using families::Dsv4Profile;
using families::Dsv4Resource;
using families::kDsv4CsaRatio;

inline void Add(std::vector<Dsv4Resource>& r, std::string name, std::string type,
                std::vector<std::uint64_t> ne) {
  r.push_back({.roles = {std::move(name)}, .type = std::move(type), .ne = std::move(ne)});
}

// A DeepSeek V4 block's tensors as its GGUFs have them, by the profile's
// ratio for the layer; expert arrays appended to `arrays`.
// The types a fixture's matrices take: the GGUFs' (dense Q8_0, the
// target's K-quant tables), or F16 throughout (small test profiles, whose
// widths are no whole K-quant blocks).
struct Types {
  std::string_view dense = "Q8_0";
  std::string_view small = "F16";  // the compressors' and the indexer's projections
  std::string_view router = "BF16";
  std::string_view table = "Q5_K";
  std::string_view output = "Q6_K";
  std::string_view experts = "IQ2_XS";
};
inline constexpr Types kF16Types{.dense = "F16",
                                 .small = "F16",
                                 .router = "F16",
                                 .table = "F16",
                                 .output = "F16",
                                 .experts = "F16"};

inline void Block(const Dsv4Profile& p, std::vector<Dsv4Resource>& r,
                  std::vector<Dsv4Resource>& arrays, std::uint32_t il, std::string_view experts,
                  const Types& t = {}) {
  const std::string dense(t.dense);
  const std::string small(t.small);
  const std::string n = std::format("blk.{}.", il);
  const std::uint64_t w = p.width;
  const std::uint64_t head = p.head_dim;
  Add(r, n + "attn_norm.weight", "F32", {w});
  Add(r, n + "attn_sinks.weight", "F32", {p.heads});
  Add(r, n + "attn_q_a.weight", dense, {w, p.q_lora});
  Add(r, n + "attn_q_a_norm.weight", "F32", {p.q_lora});
  Add(r, n + "attn_q_b.weight", dense, {p.q_lora, std::uint64_t{p.heads} * head});
  Add(r, n + "attn_kv.weight", dense, {w, head});
  Add(r, n + "attn_kv_a_norm.weight", "F32", {head});
  Add(r, n + "attn_output_a.weight", dense,
      {std::uint64_t{p.heads} * head / p.o_groups, std::uint64_t{p.o_lora} * p.o_groups});
  Add(r, n + "attn_output_b.weight", dense, {std::uint64_t{p.o_lora} * p.o_groups, w});
  for (const char* hc : {"hc_attn_", "hc_ffn_"}) {
    Add(r, n + hc + "fn.weight", "F32", {p.hc_width(), p.hc_mix()});
    Add(r, n + hc + "base.weight", "F32", {p.hc_mix()});
    Add(r, n + hc + "scale.weight", "F32", {3});
  }
  const std::uint32_t ratio = p.compress_ratios.at(il);
  if (ratio != 0) {
    const std::uint64_t coff = ratio == kDsv4CsaRatio ? 2 : 1;
    Add(r, n + "attn_compressor_kv.weight", small, {w, coff * head});
    Add(r, n + "attn_compressor_gate.weight", small, {w, coff * head});
    Add(r, n + "attn_compressor_ape.weight", "F32", {coff * head, ratio});
    Add(r, n + "attn_compressor_norm.weight", "F32", {head});
  }
  if (ratio == kDsv4CsaRatio) {
    const std::uint64_t ih = p.indexer_head_dim;
    Add(r, n + "indexer.attn_q_b.weight", dense, {p.q_lora, std::uint64_t{p.indexer_heads} * ih});
    Add(r, n + "indexer.proj.weight", small, {w, p.indexer_heads});
    Add(r, n + "indexer_compressor_kv.weight", small, {w, 2 * ih});
    Add(r, n + "indexer_compressor_gate.weight", small, {w, 2 * ih});
    Add(r, n + "indexer_compressor_ape.weight", "F32", {2 * ih, kDsv4CsaRatio});
    Add(r, n + "indexer_compressor_norm.weight", "F32", {ih});
  }
  Add(r, n + "ffn_norm.weight", "F32", {w});
  Add(r, n + "ffn_gate_inp.weight", std::string(t.router), {w, p.experts});
  if (il < p.hash_layers) {
    Add(r, n + "ffn_gate_tid2eid.weight", "I32", {p.experts_used, p.vocab});
  } else {
    Add(r, n + "exp_probs_b.bias", "F32", {p.experts});
  }
  const std::uint64_t shared = std::uint64_t{p.expert_ffn} * p.shared_experts;
  Add(r, n + "ffn_gate_shexp.weight", dense, {w, shared});
  Add(r, n + "ffn_up_shexp.weight", dense, {w, shared});
  Add(r, n + "ffn_down_shexp.weight", dense, {shared, w});
  for (const auto& [name, ne] :
       {std::pair{"ffn_gate_exps.weight", std::vector<std::uint64_t>{w, p.expert_ffn}},
        std::pair{"ffn_up_exps.weight", std::vector<std::uint64_t>{w, p.expert_ffn}},
        std::pair{"ffn_down_exps.weight", std::vector<std::uint64_t>{p.expert_ffn, w}}}) {
    arrays.push_back({.roles = {n + name},
                      .type = std::string(experts),
                      .ne = ne,
                      .expert_array = true,
                      .count = p.experts});
  }
}

// The target's resources, then its expert arrays.
inline std::vector<Dsv4Resource> Resources(const Dsv4Profile& p, const Types& t = {}) {
  std::vector<Dsv4Resource> r;
  std::vector<Dsv4Resource> arrays;
  Add(r, "token_embd.weight", std::string(t.table), {p.width, p.vocab});
  Add(r, "output_norm.weight", "F32", {p.width});
  Add(r, "output.weight", std::string(t.output), {p.width, p.vocab});
  Add(r, "output_hc_fn.weight", "F32", {p.hc_width(), p.hc});
  Add(r, "output_hc_base.weight", "F32", {p.hc});
  Add(r, "output_hc_scale.weight", "F32", {1});
  for (std::uint32_t il = 0; il < p.layers; ++il) Block(p, r, arrays, il, t.experts, t);
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// The DSpark drafter's, as its Q8_0 GGUF lists them (no table or head of
// its own).
inline std::vector<Dsv4Resource> DrafterResources(const DsparkProfile& d, const Types& t = {}) {
  const Dsv4Profile& p = d.blocks;
  std::vector<Dsv4Resource> r;
  std::vector<Dsv4Resource> arrays;
  const bool ggufs = t.experts == "IQ2_XS";
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    Block(p, r, arrays, il, ggufs ? "MXFP4" : t.experts, t);
  }
  Add(r, "enc.output_norm.weight", "F32", {p.width});
  Add(r, "fc.weight", std::string(t.dense),
      {std::uint64_t{p.width} * d.target_layers.size(), p.width});
  Add(r, "conf_proj.weight", std::string(t.router), {std::uint64_t{p.width} + d.markov_rank, 1});
  Add(r, "output_hc_base.weight", "F32", {p.hc});
  Add(r, "output_hc_fn.weight", "F32", {p.hc_width(), p.hc});
  Add(r, "output_hc_scale.weight", "F32", {1});
  Add(r, "markov_w1.weight", std::string(t.router), {d.markov_rank, p.vocab});
  Add(r, "markov_w2.weight", std::string(t.router), {d.markov_rank, p.vocab});
  Add(r, "output_norm.weight", "F32", {p.width});
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// The resources as a mapping's source (DeepSeek V4's GGUFs pair no parts
// and its mappings fold nothing; `mapping` is kept for its callers).
inline spec::Source SourceOf(std::string architecture, const std::vector<Dsv4Resource>& resources,
                             const spec::Mapping& /*mapping*/) {
  std::vector<FixtureTensor> tensors;
  for (const Dsv4Resource& r : resources) {
    tensors.push_back({.name = r.roles.front(),
                       .roles = r.roles,
                       .plain = false,
                       .type = r.type,
                       .ne = r.ne,
                       .count = r.expert_array ? r.count : 0});
  }
  auto source = FixtureSource(std::move(architecture), "gguf", tensors);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  return source.value_or(spec::Source{});
}

}  // namespace llmp::test_support::dsv4

#endif  // LLMP_TESTS_UNIT_DSV4_FIXTURE_H_
