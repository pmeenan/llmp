// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Immutable EXL3 profile/resource binding: every tensor is required with its
// exact source encoding and geometry; foreign resources and invalid profiles
// are refused. Shared selected-operation and component controls cover execution.

#include "model/qwen2_exl3.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "model/qwen2.h"

namespace {

using llmp::model::Exl3Resource;
using llmp::test_support::Failed;

const llmp::model::Qwen2Profile& Profile() { return llmp::model::Qwen25Instruct05BExl3(); }

Exl3Resource Resource(std::string name, std::string family, std::string dtype,
                      std::vector<std::uint64_t> shape, std::string role = "") {
  Exl3Resource r;
  r.name = std::move(name);
  r.family = std::move(family);
  r.dtype = std::move(dtype);
  r.shape = std::move(shape);
  r.role = std::move(role);
  return r;
}

void AddLinear(std::vector<Exl3Resource>& out, const std::string& name, std::uint64_t k,
               std::uint64_t n, std::uint32_t bits, bool bias) {
  Exl3Resource trellis =
      Resource(name + ".trellis", "exl3", "I16", {k / 16, n / 16, 16ULL * bits}, "trellis");
  trellis.k_bits = bits;
  trellis.in_features = k;
  trellis.out_features = n;
  trellis.codebook = "mcg";
  out.push_back(trellis);
  out.push_back(Resource(name + ".suh", "exl3", "F16", {k}, "suh"));
  out.push_back(Resource(name + ".svh", "exl3", "F16", {n}, "svh"));
  out.push_back(Resource(name + ".mcg", "exl3", "I32", {}, "mcg"));
  if (bias) {
    out.push_back(Resource(name + ".bias", "plain", "F16", {n}));
  }
}

std::vector<Exl3Resource> Resources() {
  const auto& p = Profile();
  std::vector<Exl3Resource> out;
  out.push_back(Resource("model.embed_tokens.weight", "plain", "BF16", {p.vocab, p.width}));
  out.push_back(Resource("model.norm.weight", "plain", "BF16", {p.width}));
  AddLinear(out, "lm_head", p.width, p.vocab, 8, false);
  for (std::uint32_t l = 0; l < p.layers; ++l) {
    const std::string b = std::format("model.layers.{}.", l);
    out.push_back(Resource(b + "input_layernorm.weight", "plain", "BF16", {p.width}));
    out.push_back(Resource(b + "post_attention_layernorm.weight", "plain", "BF16", {p.width}));
    const std::uint32_t bits = l < 10 ? 4 : 5;
    AddLinear(out, b + "self_attn.q_proj", p.width, p.width, bits, true);
    AddLinear(out, b + "self_attn.k_proj", p.width, p.kv_width(), bits, true);
    AddLinear(out, b + "self_attn.v_proj", p.width, p.kv_width(), bits, true);
    AddLinear(out, b + "self_attn.o_proj", p.width, p.width, bits, false);
    AddLinear(out, b + "mlp.gate_proj", p.width, p.ffn, bits, false);
    AddLinear(out, b + "mlp.up_proj", p.width, p.ffn, bits, false);
    AddLinear(out, b + "mlp.down_proj", p.ffn, p.width, bits, false);
  }
  return out;
}

llmp::model::Exl3Binding Binding() {
  const std::vector<Exl3Resource> resources = Resources();
  return llmp::model::BindQwen2Exl3(Profile(), "qwen2", resources).value();
}

TEST(Qwen2Exl3Test, BindsEveryTensorAndRefusesAnythingElse) {
  const auto binding = Binding();
  ASSERT_EQ(binding.layers.size(), 24U);
  EXPECT_EQ(binding.lm_head.n, 151936);
  EXPECT_EQ(binding.lm_head.bits, 8);
  EXPECT_TRUE(binding.layers[3].q.bias.has_value());
  EXPECT_FALSE(binding.layers[3].o.bias.has_value());
  EXPECT_EQ(binding.layers[0].down.k, 4864);

  const auto refused = [](std::vector<Exl3Resource> resources,
                          std::string_view architecture = "qwen2") {
    return Failed(llmp::model::BindQwen2Exl3(Profile(), architecture, resources));
  };
  EXPECT_TRUE(refused(Resources(), "llama").has_value());
  {
    auto r = Resources();
    r.push_back(Resource("model.layers.0.self_attn.o_proj.bias", "plain", "F16", {896}));
    EXPECT_NE(refused(r).value_or("").find("does not read"), std::string::npos);
  }
  for (const auto& change : std::vector<std::pair<std::string, std::function<void(Exl3Resource&)>>>{
           {"model.layers.2.mlp.up_proj.trellis", [](Exl3Resource& r) { r.k_bits = 3; }},
           {"model.layers.2.mlp.up_proj.trellis", [](Exl3Resource& r) { r.codebook = "3inst"; }},
           {"model.layers.2.mlp.up_proj.svh", [](Exl3Resource& r) { r.dtype = "F32"; }},
           {"model.layers.2.mlp.up_proj.mcg", [](Exl3Resource& r) { r.role = "mul1"; }},
           {"model.norm.weight", [](Exl3Resource& r) { r.dtype = "F32"; }},
           {"model.embed_tokens.weight", [](Exl3Resource& r) { r.shape = {151936, 1024}; }},
       }) {
    auto r = Resources();
    auto found = std::ranges::find(r, change.first, &Exl3Resource::name);
    ASSERT_NE(found, r.end());
    change.second(*found);
    EXPECT_TRUE(refused(r).has_value()) << change.first;
  }
  {
    auto r = Resources();
    std::erase_if(
        r, [](const Exl3Resource& x) { return x.name == "model.layers.7.self_attn.k_proj.bias"; });
    EXPECT_NE(refused(r).value_or("").find("k_proj.bias"), std::string::npos);
  }
}

// A profile with a zero count is refused with an error, never divided by.
TEST(Qwen2Exl3Test, RefusesAProfileWithAZeroCount) {
  const std::vector<Exl3Resource> resources = Resources();
  using Field = std::uint32_t llmp::model::Qwen2Profile::*;
  for (const auto& [name, field] : std::vector<std::pair<std::string, Field>>{
           {"kv_heads", &llmp::model::Qwen2Profile::kv_heads},
           {"heads", &llmp::model::Qwen2Profile::heads},
           {"head_dim", &llmp::model::Qwen2Profile::head_dim},
           {"width", &llmp::model::Qwen2Profile::width},
           {"ffn", &llmp::model::Qwen2Profile::ffn},
           {"vocab", &llmp::model::Qwen2Profile::vocab},
           {"layers", &llmp::model::Qwen2Profile::layers}}) {
    llmp::model::Qwen2Profile p = Profile();
    p.*field = 0;
    EXPECT_EQ(Failed(llmp::model::BindQwen2Exl3(p, "qwen2", resources)),
              "a profile the EXL3 plan cannot run")
        << name;
  }
}

}  // namespace
