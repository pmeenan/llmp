// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A Qwen3.8 wave's composition (engine/qwen38_wave_plan.h), planned on the
// host over a synthetic binding shaped as the artifact's (the CUTLASS
// expert layout, as qwen38_test.cc's) with a model of the device's choices:
// no device and no model files.
// - Consecutive compatible slots form one group while their rows fit
//   sixteen: each MXFP8 vector and routed product joins every member's
//   rows, ragged row counts included, and their three- or four-row full
//   heads join in one product with a view a slot; more slots form more
//   groups (eight of three rows: five, then three; sixteen of one: one).
// - A verify group also joins its HC products.
// - A slot whose kind differs from the one before it starts a new group.
// - Without pairing, every slot keeps its own products.

#include "engine/qwen38_wave_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "engine/qwen38_plan.h"
#include "engine/qwen38_runner.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/qwen38_graph.h"
#include "model/qwen38.h"

namespace {

namespace engine = jitllm::engine;
namespace md = jitllm::model;
namespace kg = jitllm::kernels::ggml;

constexpr std::uint64_t kTableRows = 320001536;
constexpr std::uint64_t kExpertStride = 2768976;

TEST(Qwen38StartupMeasurement, RejectsRuntimeStorageBeforeModelOrWaveAccess) {
  engine::Qwen38Model model;
  const engine::ActivationMeasurement measurement{256};
  const auto rejected = [](const auto& result) {
    EXPECT_FALSE(result);
    if (!result) EXPECT_NE(result.error().find("measurement-only"), std::string::npos);
  };
  rejected(engine::PlanQwen38Chunk(model, {}, {}, 256, 256, {}, {}, measurement));
  rejected(engine::PlanQwen38Mtp(model, {}, {}, 256, 256, measurement));
  rejected(engine::PlanQwen38TargetWave({}, {}, {.activations = 256, .bytes = 256}, measurement));
  rejected(engine::PlanQwen38DraftWave({}, {}, {.activations = 256, .bytes = 256}, measurement));
  EXPECT_TRUE(engine::Qwen38Options{}.startup_activation_threshold);
}

void SameStartupPlan(const engine::PlannedBase& ordinary, const engine::PlannedBase& measured) {
  ASSERT_EQ(ordinary.plan.steps.size(), measured.plan.steps.size());
  ASSERT_EQ(ordinary.plan.regions.size(), measured.plan.regions.size());
  for (std::size_t i = 0; i < ordinary.plan.regions.size(); ++i) {
    EXPECT_EQ(ordinary.plan.regions[i].first, measured.plan.regions[i].first);
    EXPECT_EQ(ordinary.plan.regions[i].last, measured.plan.regions[i].last);
  }
  for (std::size_t i = 0; i < ordinary.plan.steps.size(); ++i) {
    const auto& x = ordinary.plan.steps[i];
    const auto& y = measured.plan.steps[i];
    EXPECT_EQ(x.operation, y.operation);
    EXPECT_EQ(x.implementation, y.implementation);
    EXPECT_EQ(x.lane, y.lane);
    ASSERT_EQ(x.nodes.size(), y.nodes.size());
    for (std::size_t j = 0; j < x.nodes.size(); ++j) {
      EXPECT_STREQ(x.nodes[j]->name, y.nodes[j]->name);
      EXPECT_EQ(x.nodes[j]->op, y.nodes[j]->op);
      EXPECT_EQ(x.nodes[j]->type, y.nodes[j]->type);
      EXPECT_TRUE(std::ranges::equal(x.nodes[j]->ne, y.nodes[j]->ne));
    }
  }
  EXPECT_EQ(ordinary.inputs_bytes, measured.inputs_bytes);
  EXPECT_EQ(engine::PlannedHostBytes(ordinary), engine::PlannedHostBytes(measured));
  EXPECT_FALSE(ordinary.measurement_only);
  EXPECT_TRUE(measured.measurement_only);
}

TEST(Qwen38PlainTokens, DefaultPublicationKeepsLowLevelRowPlansExplicit) {
  EXPECT_TRUE(engine::Qwen38Options{}.device_tokens);
  EXPECT_FALSE(kg::Qwen38ChunkShape{}.token);
}

// The artifact's resources with the CUTLASS expert layout (as
// qwen38_test.cc's ArtifactLike(p, true)).
std::vector<md::Qwen38Resource> ArtifactLike(const md::Qwen38Profile& p) {
  std::vector<md::Qwen38Resource> r;
  std::vector<md::Qwen38Resource> arrays;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto plain = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = true, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto mx = [&](const std::string& name, std::uint64_t k, std::uint64_t n) {
    plain(name + ".weight", "F8_E4M3", {k, n});
    plain(name + ".weight_scale", "U8", {k / 32, n});
  };
  ggml("token_embd.weight", "BF16", {2560, 248320});
  ggml("output.weight", "BF16", {2560, 248320});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "BF16", {10240, 320});
  ggml("output_hc_up.weight", "BF16", {320, 10240});
  plain("per_layer_token_embd.weight", "U8", {90, kTableRows});
  plain("per_layer_token_embd.weight_scale_2", "F32", {1});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    for (const char* kind : {"attn", "ffn"}) {
      ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {10240});
      ggml(std::format("{}hc_{}_down.weight", n, kind), "BF16", {10240, 320});
      ggml(std::format("{}hc_{}_up.weight", n, kind), "BF16", {320, 10240});
      ggml(std::format("{}hc_{}_inject.weight", n, kind), "BF16", {10240, 4});
    }
    if (p.linear(il)) {
      mx(n + "attn_qkv", 2560, 10240);
      mx(n + "attn_gate", 2560, 6144);
      mx(n + "ssm_beta", 2560, 48);
      mx(n + "ssm_alpha", 2560, 48);
      ggml(n + "ssm_dt.bias", "F32", {48});
      ggml(n + "ssm_a", "F32", {48});
      ggml(n + "ssm_conv1d.weight", "F32", {4, 10240});
      ggml(n + "ssm_norm.weight", "F32", {128});
      mx(n + "ssm_out", 6144, 2560);
    } else {
      mx(n + "attn_q", 2560, 12288);
      mx(n + "attn_k", 2560, 512);
      mx(n + "attn_v", 2560, 512);
      mx(n + "attn_output", 6144, 2560);
      ggml(n + "attn_q_norm.weight", "F32", {256});
      ggml(n + "attn_k_norm.weight", "F32", {256});
      mx(n + "indexer.qk_proj", 2560, 640);
      ggml(n + "indexer.q_norm.weight", "F32", {128});
      ggml(n + "indexer.k_norm.weight", "F32", {128});
    }
    if (il == 1) {
      ggml(n + "ple_key.weight", "BF16", {2560, 10240});
      ggml(n + "ple_value.weight", "BF16", {2560, 2560});
      for (const char* part : {"key", "query", "conv"}) {
        ggml(std::format("{}ple_norm_{}.weight", n, part), "F32", {10240});
      }
      ggml(n + "ple_conv1d.weight", "F32", {4, 10240});
      plain(n + "ple_multipliers", "I64", {3});
      plain(n + "ple_head_offsets", "I64", {16});
      plain(n + "ple_head_vocab", "I64", {16});
    }
    ggml(n + "ffn_gate_inp.weight", "BF16", {2560, 512});
    ggml(n + "ffn_gate_inp_shexp.weight", "BF16", {2560});
    mx(n + "ffn_gate_shexp", 2560, 640);
    mx(n + "ffn_up_shexp", 2560, 640);
    mx(n + "ffn_down_shexp", 640, 2560);
    for (const char* proj : {"gate", "up", "down"}) {
      ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {512});
    }
    std::uint64_t group_offset = 0;
    for (const auto& [name, ne] :
         {std::pair{"gate_up_exps.codes", std::vector<std::uint64_t>{1280, 1280}},
          std::pair{"gate_up_exps.scales", std::vector<std::uint64_t>{512, 400}},
          std::pair{"down_exps.codes", std::vector<std::uint64_t>{320, 2560}},
          std::pair{"down_exps.scales", std::vector<std::uint64_t>{512, 200}}}) {
      arrays.push_back({.roles = {std::format("{}ffn_{}", n, name)},
                        .plain = false,
                        .type = "I8",
                        .ne = ne,
                        .expert_array = true,
                        .count = 512,
                        .group_offset = group_offset,
                        .readable = ne[0] * ne[1]});
      group_offset += ne[0] * ne[1];
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// GGUF fixture, the same UD-IQ3_XXS shapes as qwen38_test.cc.
std::uint64_t GgufReadable(std::string_view type, std::span<const std::uint64_t> ne) {
  ggml_type format = GGML_TYPE_F32;
  for (const auto& [name, candidate] :
       {std::pair{"F32", GGML_TYPE_F32}, std::pair{"BF16", GGML_TYPE_BF16},
        std::pair{"Q6_K", GGML_TYPE_Q6_K}, std::pair{"Q8_0", GGML_TYPE_Q8_0},
        std::pair{"IQ2_S", GGML_TYPE_IQ2_S}, std::pair{"IQ4_NL", GGML_TYPE_IQ4_NL}}) {
    if (type == name) {
      format = candidate;
      break;
    }
  }
  std::uint64_t bytes = ggml_row_size(format, static_cast<std::int64_t>(ne[0]));
  for (std::size_t i = 1; i < ne.size(); ++i) {
    bytes *= ne[i];
  }
  return bytes;
}

std::vector<md::Qwen38Resource> GgufLike(const md::Qwen38Profile& p) {
  std::vector<md::Qwen38Resource> r;
  std::vector<md::Qwen38Resource> arrays;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    const std::uint64_t readable = GgufReadable(type, ne);
    r.push_back({.roles = {std::move(name)},
                 .plain = false,
                 .type = std::move(type),
                 .ne = std::move(ne),
                 .readable = readable});
  };
  ggml("token_embd.weight", "Q6_K", {2560, 248320});
  ggml("output.weight", "Q6_K", {2560, 248320});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "Q8_0", {10240, 320});
  ggml("output_hc_up.weight", "Q8_0", {320, 10240});
  ggml("per_layer_token_embd.weight", "IQ4_NL", {160, kTableRows});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    for (const char* kind : {"attn", "ffn"}) {
      ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {10240});
      ggml(std::format("{}hc_{}_down.weight", n, kind), "Q8_0", {10240, 320});
      ggml(std::format("{}hc_{}_up.weight", n, kind), "Q8_0", {320, 10240});
      ggml(std::format("{}hc_{}_inject.weight", n, kind), "F32", {10240, 4});
    }
    if (p.linear(il)) {
      ggml(n + "attn_qkv.weight", "Q6_K", {2560, 10240});
      ggml(n + "attn_gate.weight", "Q6_K", {2560, 6144});
      ggml(n + "ssm_beta.weight", "F32", {2560, 48});
      ggml(n + "ssm_alpha.weight", "F32", {2560, 48});
      ggml(n + "ssm_dt.bias", "F32", {48});
      ggml(n + "ssm_a", "F32", {48});
      ggml(n + "ssm_conv1d.weight", "F32", {4, 10240});
      ggml(n + "ssm_norm.weight", "F32", {128});
      ggml(n + "ssm_out.weight", "Q6_K", {6144, 2560});
    } else {
      ggml(n + "attn_q.weight", "Q6_K", {2560, 12288});
      ggml(n + "attn_k.weight", "Q6_K", {2560, 512});
      ggml(n + "attn_v.weight", "Q6_K", {2560, 512});
      ggml(n + "attn_output.weight", "Q6_K", {6144, 2560});
      ggml(n + "attn_q_norm.weight", "F32", {256});
      ggml(n + "attn_k_norm.weight", "F32", {256});
      ggml(n + "indexer.q_proj.weight", "BF16", {2560, 512});
      ggml(n + "indexer.k_proj.weight", "BF16", {2560, 128});
      ggml(n + "indexer.q_norm.weight", "F32", {128});
      ggml(n + "indexer.k_norm.weight", "F32", {128});
    }
    if (il == 1) {
      ggml(n + "ple_key.weight", "Q8_0", {2560, 10240});
      ggml(n + "ple_value.weight", "Q8_0", {2560, 2560});
      for (const char* part : {"key", "query", "conv"}) {
        ggml(std::format("{}ple_norm_{}.weight", n, part), "F32", {10240});
      }
      ggml(n + "ple_conv1d.weight", "F32", {4, 10240});
    }
    ggml(n + "ffn_gate_inp.weight", "F32", {2560, 512});
    ggml(n + "ffn_gate_inp_shexp.weight", "F32", {2560});
    ggml(n + "ffn_gate_shexp.weight", "Q6_K", {2560, 640});
    ggml(n + "ffn_up_shexp.weight", "Q6_K", {2560, 640});
    ggml(n + "ffn_down_shexp.weight", "Q8_0", {640, 2560});
    std::uint64_t group_offset = 0;
    for (const auto& [proj, type, ne] :
         {std::tuple{"gate", "IQ2_S", std::vector<std::uint64_t>{2560, 640}},
          std::tuple{"up", "IQ2_S", std::vector<std::uint64_t>{2560, 640}},
          std::tuple{"down", "IQ4_NL", std::vector<std::uint64_t>{640, 2560}}}) {
      const std::uint64_t readable = GgufReadable(type, ne);
      arrays.push_back({.roles = {std::format("{}ffn_{}_exps.weight", n, proj)},
                        .plain = false,
                        .type = type,
                        .ne = ne,
                        .expert_array = true,
                        .count = 512,
                        .group_offset = group_offset,
                        .readable = readable});
      group_offset += (readable + 255) / 256 * 256;
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

std::vector<md::Qwen38Resource> MtpLike() {
  std::vector<md::Qwen38Resource> r;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  ggml("fc_embd.weight", "BF16", {2560, 2560});
  ggml("fc_hidden.weight", "BF16", {2560, 2560});
  ggml("norm_embd.weight", "F32", {2560});
  ggml("norm_hidden.weight", "F32", {10240});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "BF16", {10240, 320});
  ggml("output_hc_up.weight", "BF16", {320, 10240});
  for (const char* kind : {"attn", "ffn"}) {
    ggml(std::format("blk.0.hc_{}_norm.weight", kind), "F32", {10240});
    ggml(std::format("blk.0.hc_{}_down.weight", kind), "BF16", {10240, 320});
    ggml(std::format("blk.0.hc_{}_up.weight", kind), "BF16", {320, 10240});
    ggml(std::format("blk.0.hc_{}_inject.weight", kind), "BF16", {10240, 4});
  }
  ggml("blk.0.attn_q.weight", "BF16", {2560, 12288});
  ggml("blk.0.attn_k.weight", "BF16", {2560, 512});
  ggml("blk.0.attn_v.weight", "BF16", {2560, 512});
  ggml("blk.0.attn_output.weight", "BF16", {6144, 2560});
  ggml("blk.0.attn_q_norm.weight", "F32", {256});
  ggml("blk.0.attn_k_norm.weight", "F32", {256});
  ggml("blk.0.indexer.qk_proj.weight", "BF16", {2560, 640});
  ggml("blk.0.indexer.q_norm.weight", "F32", {128});
  ggml("blk.0.indexer.k_norm.weight", "F32", {128});
  ggml("blk.0.ffn_gate_inp.weight", "BF16", {2560, 512});
  ggml("blk.0.ffn_gate_inp_shexp.weight", "BF16", {2560});
  ggml("blk.0.ffn_gate_shexp.weight", "BF16", {2560, 640});
  ggml("blk.0.ffn_up_shexp.weight", "BF16", {2560, 640});
  ggml("blk.0.ffn_down_shexp.weight", "BF16", {640, 2560});
  for (const char* proj : {"gate", "up", "down"}) {
    ggml(std::format("blk.0.ffn_{}_exps.weight_scale_2", proj), "F32", {512});
  }
  std::uint64_t group_offset = 0;
  for (const auto& [name, ne] :
       {std::pair{"gate_up_exps.codes", std::vector<std::uint64_t>{1280, 1280}},
        std::pair{"gate_up_exps.scales", std::vector<std::uint64_t>{512, 400}},
        std::pair{"down_exps.codes", std::vector<std::uint64_t>{320, 2560}},
        std::pair{"down_exps.scales", std::vector<std::uint64_t>{512, 200}}}) {
    r.push_back({.roles = {std::format("blk.0.ffn_{}", name)},
                 .plain = false,
                 .type = "I8",
                 .ne = ne,
                 .expert_array = true,
                 .count = 512,
                 .group_offset = group_offset,
                 .readable = ne[0] * ne[1]});
    group_offset += ne[0] * ne[1];
  }
  return r;
}

// Hash constants of the checkpoint's shape: 16 heads of 20,000,008 rows.
md::Qwen38PleHash Hash() {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  std::vector<std::int64_t> m = {3, 5, 7};
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  return md::CheckQwen38PleHash(p, m, offsets, vocab, kTableRows).value();
}

// A model of the device's choices (as qwen38_test.cc's): float products of
// up to 16 columns on MMF tensor cores.
kg::DeviceChoices ModelDevice() {
  return {
      .mul_mat = [](const ggml_tensor* node) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
        const std::int64_t columns = node->src[1]->ne[1] * node->src[1]->ne[2];
        if (columns == 1) {
          return kg::MulMatPath::kVector;
        }
        return columns <= 16 ? kg::MulMatPath::kTensorCore : kg::MulMatPath::kCublas;
      },
      .vector_fusible = [](const ggml_tensor*) { return false; },
      .quant =
          [](const ggml_tensor* node) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
        const std::int64_t columns =
            node->op == GGML_OP_MUL_MAT_ID ? node->src[2]->ne[1] : node->src[1]->ne[1];
        return columns <= 8 ? kg::QuantMulMatPath::kVector : kg::QuantMulMatPath::kTile;
      },
      .dense_mmvq_shape = {}};
}

// Weights at distinct placeless addresses below the planner's own ranges:
// resources 2 GiB apart, expert arrays 4 GiB apart, the n-gram table, then
// each slot's state, drafter state and verify saves.
constexpr std::uint64_t kResources = std::uint64_t{1} << 36U;
constexpr std::uint64_t kArrays = std::uint64_t{1} << 44U;
constexpr std::uint64_t kPleTable = std::uint64_t{1} << 45U;
constexpr std::uint64_t kSlotPlaces = std::uint64_t{1} << 38U;

struct Request {
  std::uint32_t slot = 0;
  std::uint32_t n_past = 0;
  std::uint32_t rows = 0;
  engine::Qwen38ChunkKind kind{};
  bool token = false;
};

struct Products {
  std::vector<std::int64_t> mxfp8;   // each MXFP8 vector product's columns
  std::vector<std::int64_t> routed;  // each routed product's tokens
  std::vector<std::int64_t> hc;      // each HC candidate's columns (2 to 4)
  std::vector<std::int64_t> wide;    // each BF16 vector product past 4 columns
};

Products Of(std::span<ggml_tensor* const> nodes) {
  Products out;
  for (const ggml_tensor* t : nodes) {
    switch (kg::JitllmOpOf(t)) {
      case kg::JitllmOp::kMxfp8MulMatVec:
        out.mxfp8.push_back(t->src[2]->ne[1]);
        break;
      case kg::JitllmOp::kMoeGemv:
        out.routed.push_back(t->src[2]->ne[1]);
        break;
      default:
        if (engine::Qwen38HcPairCandidate(t)) {
          out.hc.push_back(t->ne[1]);
        } else if (kg::IsGemvBf16(t) && t->ne[1] > 4) {
          out.wide.push_back(t->ne[1]);
        }
        break;
    }
  }
  return out;
}

std::size_t Count(const std::vector<std::int64_t>& v, std::int64_t value) {
  return static_cast<std::size_t>(std::ranges::count(v, value));
}

class Qwen38WavePlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto binding = md::BindQwen38(p_, "qwen4exp", ArtifactLike(p_));
    ASSERT_TRUE(binding.has_value()) << binding.error();
    ASSERT_TRUE(binding->cutlass());
    binding_.emplace(std::move(*binding));
    auto state = md::Qwen38State(p_, 4096, 512, false);
    ASSERT_TRUE(state.has_value()) << state.error();
    state_.emplace(std::move(*state));
    auto mtp = md::Qwen38MtpStateOf(p_, *state_);
    ASSERT_TRUE(mtp.has_value()) << mtp.error();
    mtp_.emplace(*mtp);
    auto commit = md::Qwen38Commit(p_, 4);
    ASSERT_TRUE(commit.has_value()) << commit.error();
    commit_.emplace(std::move(*commit));
    for (std::uint32_t s = 0; s < engine::kQwen38WaveSlots; ++s) {
      const std::uint64_t base = kPleTable + (kSlotPlaces * (s + 1));
      models_[s] = {
          // Only compared (one common model), never read, on this path.
          .artifact = reinterpret_cast<const jitllm::artifact::Artifact*>(artifact_.data()),
          .profile = &p_,
          .binding = &*binding_,
          .state = &*state_,
          .places = {
              .resource = [](std::uint32_t i) { return kResources + (std::uint64_t{i} << 31U); },
              .array = [](std::uint32_t i) { return kArrays + (std::uint64_t{i} << 32U); },
              .stride = std::vector<std::uint64_t>(p_.layers, kExpertStride),
              .state = base,
              .ple_table = kPleTable,
              .mtp_state = base + (std::uint64_t{1} << 36U),
              .commit = base + (std::uint64_t{1} << 37U)},
          .fused = true,
          .exact = false,
          .cutlass = true,
          .mtp_state = &*mtp_,
          .commit = &*commit_};
    }
  }

  std::expected<std::unique_ptr<engine::Qwen38WavePlanned>, std::string> Plan(
      std::span<const Request> requests, bool paired = true, bool lanes = false,
      std::optional<engine::ActivationMeasurement> measurement = std::nullopt) {
    if (!state_.has_value()) {
      return std::unexpected(std::string("the fixture has no state layout"));
    }
    const md::Qwen38StateLayout& state = *state_;
    std::vector<engine::Qwen38TargetWaveInput> inputs;
    for (const Request& r : requests) {
      std::vector<std::int32_t> history(std::size_t{r.n_past} + r.rows, 1000);
      auto chunk = md::Qwen38Chunk(p_, state, hash_, history, r.n_past, r.rows, false,
                                   requests.size() > 1 ? 2048 : 256);
      if (!chunk) {
        return std::unexpected(chunk.error());
      }
      inputs.push_back({.slot = r.slot,
                        .model = &models_[r.slot],
                        .shape = kg::Qwen38ShapeOf(state, *chunk, r.rows),
                        .kind = r.kind});
      inputs.back().shape.token = r.token;
    }
    return engine::PlanQwen38TargetWave(
        inputs, ModelDevice(), {.paired = paired, .share_target_head = true, .lanes = lanes},
        measurement);
  }

  void CheckStartupMeasurement() {
    for (const auto rows : {1U, 3U}) {
      const std::array<Request, 4> requests = {Request{.slot = 0, .n_past = 100, .rows = rows},
                                               Request{.slot = 1, .n_past = 200, .rows = rows},
                                               Request{.slot = 2, .n_past = 3000, .rows = rows},
                                               Request{.slot = 3, .n_past = 4000, .rows = rows}};
      auto ordinary = Plan(requests, true, true);
      ASSERT_TRUE(ordinary) << ordinary.error();
      for (const auto ceiling : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
        auto measured = Plan(requests, true, true, engine::ActivationMeasurement{ceiling});
        ASSERT_TRUE(measured) << measured.error();
        SameStartupPlan(**ordinary, **measured);
        EXPECT_EQ((*ordinary)->host_bytes(), (*measured)->host_bytes());
        EXPECT_EQ((*measured)->placement.measurement_bound, ceiling != 0);
        if (ceiling == 0) EXPECT_EQ((*ordinary)->placement.extent, (*measured)->placement.extent);
        engine::StartupPlacementStats stats;
        stats.Account(**measured);
        EXPECT_EQ(stats.plans, 1U);
        EXPECT_EQ(stats.nodes, engine::PlannedNodes(**ordinary));
        EXPECT_EQ(stats.bounded + stats.exact, 1U);
        EXPECT_EQ(stats.max_inputs, (*ordinary)->inputs_bytes);
      }
    }
  }

  // A lone slot's products (the composition keeps them as they are).
  void CheckPlainTokens() {
    const std::array<Request, 2> token = {
        Request{.slot = 0, .n_past = 100, .rows = 1, .token = true},
        Request{.slot = 2, .n_past = 3000, .rows = 1, .token = true}};
    auto planned = Plan(token);
    ASSERT_TRUE(planned) << planned.error();
    const auto& wave = **planned;
    EXPECT_EQ(wave.active_slots(), 0b0101U);
    EXPECT_GT(wave.placement.extent, 0U);
    EXPECT_EQ(std::ranges::count_if(
                  wave.plan.steps,
                  [](const auto& step) { return step.implementation == kg::kArgmaxName; }),
              2);
    for (const auto slot : {0U, 2U}) {
      const auto* graph = wave.target(slot);
      ASSERT_NE(graph, nullptr);
      ASSERT_NE(graph->argmax, nullptr);
      EXPECT_EQ(graph->argmax->src[0], graph->logits);
      EXPECT_EQ(graph->argmax->type, GGML_TYPE_I32);
      EXPECT_EQ(graph->argmax->ne[0], 1);
      EXPECT_EQ(kg::JitllmOpInt(graph->argmax, 1),
                static_cast<std::int32_t>(kg::ArgmaxFlavor::kHostGreedy));
      EXPECT_TRUE(std::ranges::any_of(wave.placement.offsets,
                                      [&](const auto& at) { return at.first == graph->argmax; }));
    }
    auto row_requests = token;
    for (auto& r : row_requests) r.token = false;
    auto rows = Plan(row_requests);
    ASSERT_TRUE(rows) << rows.error();
    EXPECT_EQ((*rows)->target(0)->argmax, nullptr);
    auto bad = token;
    bad[0].token = false;
    EXPECT_FALSE(Plan(bad));
    bad = token;
    bad[0].rows = 2;
    EXPECT_FALSE(Plan(bad));
    bad = token;
    bad[0].kind = {.verify = true, .export_streams = true};
    EXPECT_FALSE(Plan(bad));
    bad = token;
    bad[0].kind.export_streams = true;
    EXPECT_FALSE(Plan(bad));
  }

  Products Lone(engine::Qwen38ChunkKind kind = {}) {
    const std::array<Request, 1> one = {Request{.slot = 0, .n_past = 100, .rows = 4, .kind = kind}};
    auto planned = Plan(one);
    EXPECT_TRUE(planned.has_value()) << (planned ? "" : planned.error());
    if (!planned) {
      return {};
    }
    EXPECT_EQ((*planned)->stats().paired_slots, 0U);
    return Of((*planned)->nodes());
  }

  const md::Qwen38Profile& p_ = md::Qwen38Flash();
  const md::Qwen38PleHash hash_ = Hash();
  alignas(std::max_align_t) std::array<std::byte, 64> artifact_{};
  std::optional<md::Qwen38Binding> binding_;
  std::optional<md::Qwen38StateLayout> state_;
  std::optional<md::Qwen38MtpState> mtp_;
  std::optional<md::Qwen38CommitLayout> commit_;
  std::array<engine::Qwen38Model, engine::kQwen38WaveSlots> models_{};
};

class Qwen38GgufWavePlanTest : public Qwen38WavePlanTest {
 protected:
  void SetUp() override {
    Qwen38WavePlanTest::SetUp();
    auto binding = md::BindQwen38(p_, "qwen4exp", GgufLike(p_));
    ASSERT_TRUE(binding.has_value()) << binding.error();
    ASSERT_TRUE(binding->gguf());
    binding_.emplace(std::move(*binding));
    for (auto& model : models_) {
      model.binding = &*binding_;
      model.mtp_state = nullptr;
      model.commit = nullptr;
      model.cutlass = false;
      model.places.stride.assign(p_.layers, 1977840);
    }
  }
};

TEST_F(Qwen38WavePlanTest, StartupMeasurementPreservesTargetSelectorsLanesAndHostOwnership) {
  CheckStartupMeasurement();
}

TEST_F(Qwen38GgufWavePlanTest, StartupMeasurementPreservesTargetSelectorsLanesAndHostOwnership) {
  CheckStartupMeasurement();
}

// Plain GGUF waves retain one-token reductions for dense, shared GLU,
// routed GLU and per-expert down products. Ten experts per token cap a
// group at twelve slots (120 pairs), even though dense products take sixteen.
TEST_F(Qwen38WavePlanTest, PlainTokensRetainIndependentHostCompatibleOutputs) {
  CheckPlainTokens();
}

TEST_F(Qwen38GgufWavePlanTest, PlainTokensRetainIndependentHostCompatibleOutputs) {
  CheckPlainTokens();
}

TEST_F(Qwen38GgufWavePlanTest, JoinsOneRowProductsWithinTheRoutedPairLimit) {
  for (const std::uint32_t count : {1U, 2U, 3U, 4U, 16U}) {
    std::vector<Request> requests;
    requests.reserve(count);
    for (std::uint32_t s = 0; s < count; ++s) {
      requests.push_back({.slot = s, .n_past = 100 + (s * 200), .rows = 1});
    }
    auto alone = Plan(std::span(requests).first(1));
    ASSERT_TRUE(alone.has_value()) << alone.error();
    const auto scalar = static_cast<std::uint64_t>(std::ranges::count_if(
        (*alone)->nodes(),
        [](const ggml_tensor* t) { return kg::JitllmOpOf(t) == kg::JitllmOp::kVecQ; }));
    ASSERT_GT(scalar, 0);
    auto planned = Plan(requests);
    ASSERT_TRUE(planned.has_value()) << planned.error();
    const auto& wave = **planned;
    const std::uint64_t groups = count > 12 ? 2 : 1;
    EXPECT_EQ(wave.stats().vecq_pairs, count == 1 ? 0 : groups * scalar);
    std::uint64_t products = 0;
    for (const ggml_tensor* t : wave.nodes()) {
      if (kg::JitllmOpOf(t) != kg::JitllmOp::kVecQ) {
        continue;
      }
      ++products;
      EXPECT_EQ(kg::VecQOneToken(t), count > 1);
      EXPECT_TRUE(kg::CheckVecQ(t).has_value());
      const bool routed = t->src[2] != nullptr && t->src[2]->type == GGML_TYPE_I32;
      const auto tokens = kg::JitllmOpInt(t, 0);
      EXPECT_LE(tokens, kg::kVecQMaxTokens);
      if (routed) {
        EXPECT_LE(tokens * t->src[2]->ne[0], 128);
      }
    }
    EXPECT_EQ(products, groups * scalar);
    EXPECT_EQ(wave.stats().paired_slots, count == 1 ? 0U : (1U << count) - 1U);
    EXPECT_EQ(wave.stats().full_head_pairs, 0U);
    // The quantized full head is also split, so a slot never reads another's logits.
    if (count > 1) {
      EXPECT_NE(wave.target(0)->logits->view_src, nullptr);
    }
    auto unpaired = Plan(requests, false);
    ASSERT_TRUE(unpaired.has_value()) << unpaired.error();
    EXPECT_EQ((*unpaired)->stats().vecq_pairs, 0U);
    EXPECT_EQ((*unpaired)->stats().paired_slots, 0U);
  }
}

// Multirow GGUF products use token-count-dependent reductions and stay scalar.
TEST_F(Qwen38GgufWavePlanTest, MultirowGgufProductsStayOriginal) {
  const std::array<Request, 2> requests = {Request{.slot = 0, .n_past = 100, .rows = 2},
                                           Request{.slot = 1, .n_past = 700, .rows = 2}};
  auto planned = Plan(requests);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  EXPECT_EQ((*planned)->stats().vecq_pairs, 0U);
  EXPECT_EQ((*planned)->stats().paired_slots, 0U);
}

// Four slots of four rows at different depths: one group whose every
// product reads its weights once for sixteen rows, and one full head of
// sixteen columns with each slot's logits a view of it.
TEST_F(Qwen38WavePlanTest, FourSlotsShareEveryProductAndTheirHeads) {
  const Products lone = Lone();
  ASSERT_FALSE(lone.mxfp8.empty());
  ASSERT_FALSE(lone.routed.empty());
  EXPECT_EQ(Count(lone.mxfp8, 4), lone.mxfp8.size());
  const std::array<Request, 4> four = {
      Request{.slot = 0, .n_past = 100, .rows = 4}, Request{.slot = 1, .n_past = 700, .rows = 4},
      Request{.slot = 2, .n_past = 2100, .rows = 4}, Request{.slot = 3, .n_past = 3000, .rows = 4}};
  auto planned = Plan(four);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.active_slots(), 0xFU);
  EXPECT_EQ(w.stats().paired_slots, 0xFU);
  EXPECT_EQ(w.stats().mxfp8_pairs, lone.mxfp8.size());
  EXPECT_EQ(w.stats().routed_pairs, lone.routed.size());
  const Products shared = Of(w.nodes());
  EXPECT_EQ(Count(shared.mxfp8, 16), lone.mxfp8.size());
  EXPECT_EQ(shared.mxfp8.size(), lone.mxfp8.size());
  EXPECT_EQ(Count(shared.routed, 16), lone.routed.size());
  EXPECT_EQ(shared.routed.size(), lone.routed.size());
  EXPECT_EQ(w.stats().full_head_pairs, 1U);
  ASSERT_EQ(w.head_products().size(), 1U);
  const auto& head = w.head_products().front();
  ASSERT_EQ(head.originals.size(), 4U);
  ASSERT_NE(head.together, nullptr);
  EXPECT_EQ(head.together->ne[1], 16);
  for (std::size_t s = 0; s < 4; ++s) {
    ASSERT_NE(w.target(s), nullptr) << s;
    const ggml_tensor* logits = w.target(s)->logits;
    EXPECT_EQ(logits->ne[1], 4) << s;
    EXPECT_EQ(logits->view_src, head.together) << s;
    EXPECT_EQ(logits->view_offs, s * 4 * head.together->nb[1]) << s;
  }
}

// With lanes, each slot's own operations between shared products run on a
// lane of its own (slots in wave order), each stretch a region; the shared
// products, their concatenations and views stay on the stream. The plan's
// steps are otherwise the plan without lanes', step for step.
TEST_F(Qwen38WavePlanTest, LanesCarryEachSlotsOwnOperations) {
  const std::array<Request, 4> four = {
      Request{.slot = 0, .n_past = 100, .rows = 4}, Request{.slot = 1, .n_past = 700, .rows = 4},
      Request{.slot = 2, .n_past = 2100, .rows = 4}, Request{.slot = 3, .n_past = 3000, .rows = 4}};
  auto plain = Plan(four);
  ASSERT_TRUE(plain.has_value()) << plain.error();
  auto laned = Plan(four, true, true);
  ASSERT_TRUE(laned.has_value()) << laned.error();
  const kg::GraphPlan& a = (*plain)->plan;
  const kg::GraphPlan& b = (*laned)->plan;
  EXPECT_TRUE(a.regions.empty());
  ASSERT_FALSE(b.regions.empty());
  ASSERT_EQ(a.steps.size(), b.steps.size());
  std::array<std::size_t, kg::kMaxLanes + 1> on{};
  for (std::size_t i = 0; i < b.steps.size(); ++i) {
    EXPECT_EQ(a.steps[i].implementation, b.steps[i].implementation) << i;
    ++on[b.steps[i].lane];
  }
  for (std::size_t lane = 0; lane <= kg::kMaxLanes; ++lane) {
    EXPECT_GT(on[lane], 0U) << lane;
  }
  // Every shared product is the stream's.
  const auto& w = **laned;
  for (const kg::PlanStep& step : b.steps) {
    for (const ggml_tensor* node : step.nodes) {
      if (kg::JitllmOpOf(node) == kg::JitllmOp::kMxfp8MulMatVec && node->src[2]->ne[1] == 16) {
        EXPECT_EQ(step.lane, 0U);
      }
    }
  }
  EXPECT_EQ(w.stats().paired_slots, 0xFU);
  // Fewer slots than kQwen38LaneSlots stay on the stream.
  const std::array<Request, 3> three = {Request{.slot = 0, .n_past = 100, .rows = 4},
                                        Request{.slot = 1, .n_past = 700, .rows = 4},
                                        Request{.slot = 2, .n_past = 2100, .rows = 4}};
  auto narrow = Plan(three, true, true);
  ASSERT_TRUE(narrow.has_value()) << narrow.error();
  EXPECT_TRUE((*narrow)->plan.regions.empty());
  EXPECT_EQ(engine::kQwen38LaneSlots, 4U);
}

// Three slots, not adjacent in slot number and of ragged rows (4, 3, 4),
// still join: eleven rows a product and an eleven-column head.
TEST_F(Qwen38WavePlanTest, ThreeRaggedSlotsShareAsOneGroup) {
  const Products lone = Lone();
  const std::array<Request, 3> three = {Request{.slot = 0, .n_past = 40, .rows = 4},
                                        Request{.slot = 2, .n_past = 1500, .rows = 3},
                                        Request{.slot = 3, .n_past = 2600, .rows = 4}};
  auto planned = Plan(three);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.stats().paired_slots, 0b1101U);
  const Products shared = Of(w.nodes());
  EXPECT_EQ(Count(shared.mxfp8, 11), lone.mxfp8.size());
  EXPECT_EQ(shared.mxfp8.size(), lone.mxfp8.size());
  EXPECT_EQ(Count(shared.routed, 11), lone.routed.size());
  ASSERT_EQ(w.head_products().size(), 1U);
  EXPECT_EQ(w.head_products().front().originals.size(), 3U);
  EXPECT_EQ(w.head_products().front().together->ne[1], 11);
  EXPECT_EQ(w.target(2)->logits->ne[1], 3);
  EXPECT_EQ(w.target(3)->logits->view_offs, 7 * w.head_products().front().together->nb[1]);
}

// A verify group of four also joins every multi-row HC product.
TEST_F(Qwen38WavePlanTest, AVerifyGroupJoinsItsHcProducts) {
  const engine::Qwen38ChunkKind verify{.verify = true};
  const Products lone = Lone(verify);
  ASSERT_FALSE(lone.hc.empty());
  EXPECT_EQ(Count(lone.hc, 4), lone.hc.size());
  std::array<Request, 4> four{};
  for (std::uint32_t s = 0; s < 4; ++s) {
    four[s] = {.slot = s, .n_past = 200 + (s * 900), .rows = 4, .kind = verify};
  }
  auto planned = Plan(four);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.stats().paired_slots, 0xFU);
  const Products shared = Of(w.nodes());
  EXPECT_TRUE(shared.hc.empty());
  EXPECT_EQ(Count(shared.wide, 16), lone.hc.size());
  EXPECT_EQ(shared.wide.size(), lone.hc.size());
  EXPECT_EQ(Count(shared.mxfp8, 16), lone.mxfp8.size());
  EXPECT_EQ(Count(shared.routed, 16), lone.routed.size());
}

// Eight slots of three rows (a shared wave's depth-two verifies): groups fill
// each joined product to at most sixteen rows, five slots (fifteen rows)
// then three (nine), and every slot joins one, with a head a group.
TEST_F(Qwen38WavePlanTest, EightSlotsFormGroupsWithinSixteenRows) {
  const engine::Qwen38ChunkKind verify{.verify = true};
  const Products lone = Lone(verify);
  std::array<Request, 8> eight{};
  for (std::uint32_t s = 0; s < 8; ++s) {
    eight[s] = {.slot = s, .n_past = 100 + (s * 450), .rows = 3, .kind = verify};
  }
  auto planned = Plan(eight);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.active_slots(), 0xFFU);
  EXPECT_EQ(w.stats().paired_slots, 0xFFU);
  EXPECT_EQ(w.stats().mxfp8_pairs, 2 * lone.mxfp8.size());
  EXPECT_EQ(w.stats().routed_pairs, 2 * lone.routed.size());
  const Products shared = Of(w.nodes());
  EXPECT_EQ(Count(shared.mxfp8, 15), lone.mxfp8.size());
  EXPECT_EQ(Count(shared.mxfp8, 9), lone.mxfp8.size());
  EXPECT_EQ(shared.mxfp8.size(), 2 * lone.mxfp8.size());
  EXPECT_EQ(Count(shared.routed, 15), lone.routed.size());
  EXPECT_EQ(Count(shared.routed, 9), lone.routed.size());
  ASSERT_EQ(w.head_products().size(), 2U);
  const auto [narrow, wide] =
      std::minmax(w.head_products()[0].together->ne[1], w.head_products()[1].together->ne[1]);
  EXPECT_EQ(narrow, 9);
  EXPECT_EQ(wide, 15);
}

// Sixteen one-row slots (a plain decode wave at the most slots) share every
// product as one group of sixteen rows.
TEST_F(Qwen38WavePlanTest, SixteenOneRowSlotsShareAsOneGroup) {
  const std::array<Request, 1> one = {Request{.slot = 0, .n_past = 100, .rows = 1}};
  auto alone = Plan(one);
  ASSERT_TRUE(alone.has_value()) << alone.error();
  const Products lone = Of((*alone)->nodes());
  ASSERT_FALSE(lone.mxfp8.empty());
  std::array<Request, engine::kQwen38WaveSlots> all{};
  for (std::uint32_t s = 0; s < all.size(); ++s) {
    all[s] = {.slot = s, .n_past = 100 + (s * 200), .rows = 1};
  }
  auto planned = Plan(all);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.active_slots(), 0xFFFFU);
  EXPECT_EQ(w.stats().paired_slots, 0xFFFFU);
  const Products shared = Of(w.nodes());
  EXPECT_EQ(Count(shared.mxfp8, 16), lone.mxfp8.size());
  EXPECT_EQ(shared.mxfp8.size(), lone.mxfp8.size());
  EXPECT_EQ(Count(shared.routed, 16), lone.routed.size());
  EXPECT_EQ(shared.routed.size(), lone.routed.size());
}

// A slot whose kind differs from the slot before it starts a new group:
// with kinds (plain, streams, plain, plain), only the last two join.
TEST_F(Qwen38WavePlanTest, AKindChangeStartsANewGroup) {
  const Products lone = Lone();
  const engine::Qwen38ChunkKind streams{.export_streams = true};
  const std::array<Request, 4> four = {
      Request{.slot = 0, .n_past = 100, .rows = 4},
      Request{.slot = 1, .n_past = 700, .rows = 4, .kind = streams},
      Request{.slot = 2, .n_past = 2100, .rows = 4}, Request{.slot = 3, .n_past = 3000, .rows = 4}};
  auto planned = Plan(four);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.stats().paired_slots, 0b1100U);
  EXPECT_EQ(w.stats().mxfp8_pairs, lone.mxfp8.size());
  const Products shared = Of(w.nodes());
  EXPECT_EQ(Count(shared.mxfp8, 8), lone.mxfp8.size());
  EXPECT_EQ(Count(shared.mxfp8, 4), shared.mxfp8.size() - lone.mxfp8.size());
  EXPECT_EQ(Count(shared.routed, 8), lone.routed.size());
  ASSERT_EQ(w.head_products().size(), 1U);
  EXPECT_EQ(w.head_products().front().originals.size(), 2U);
  EXPECT_EQ(w.target(0)->logits->view_src, nullptr);
  EXPECT_EQ(w.target(1)->logits->view_src, nullptr);
}

// Without pairing every slot keeps its own products and head.
TEST_F(Qwen38WavePlanTest, WithoutPairingEverySlotKeepsItsProducts) {
  const Products lone = Lone();
  const std::array<Request, 4> four = {
      Request{.slot = 0, .n_past = 100, .rows = 4}, Request{.slot = 1, .n_past = 700, .rows = 4},
      Request{.slot = 2, .n_past = 2100, .rows = 4}, Request{.slot = 3, .n_past = 3000, .rows = 4}};
  auto planned = Plan(four, false);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  const auto& w = **planned;
  EXPECT_EQ(w.stats().paired_slots, 0U);
  EXPECT_EQ(w.stats().mxfp8_pairs, 0U);
  EXPECT_EQ(w.stats().routed_pairs, 0U);
  EXPECT_TRUE(w.head_products().empty());
  const Products own = Of(w.nodes());
  EXPECT_EQ(Count(own.mxfp8, 4), 4 * lone.mxfp8.size());
  EXPECT_EQ(own.mxfp8.size(), 4 * lone.mxfp8.size());
  EXPECT_EQ(Count(own.routed, 4), 4 * lone.routed.size());
}

class Qwen38DraftWavePlanTest : public Qwen38WavePlanTest {
 protected:
  void BindDraft(bool selected, std::string_view head_type = "BF16") {
    auto resources = MtpLike();
    if (selected) {
      resources.push_back(
          {.roles = {"draft_output.weight"}, .type = std::string(head_type), .ne = {2560, 47172}});
      resources.push_back({.roles = {"draft_output.ids"}, .type = "I32", .ne = {1, 47172}});
    }
    auto binding = md::BindQwen38Mtp(p_, "qwen4exp-mtp", resources);
    ASSERT_TRUE(binding.has_value()) << binding.error();
    drafter_.emplace(std::move(*binding));
    for (auto& model : models_) {
      model.drafter = &*drafter_;
      model.mtp_stride = 2764800;
      model.places.mtp_resource = [](std::uint32_t i) {
        return (std::uint64_t{1} << 46U) + (std::uint64_t{i} << 31U);
      };
      model.places.mtp_array = [](std::uint32_t i) {
        return (std::uint64_t{1} << 47U) + (std::uint64_t{i} << 32U);
      };
    }
  }

  auto DraftPlan(bool paired = true, bool capture = false, std::int64_t head_rows = 65536,
                 bool confidence = false,
                 std::optional<engine::ActivationMeasurement> measurement = std::nullopt) {
    const kg::Qwen38MtpShape shape{.rows = 4,
                                   .passes = 3,
                                   .n_kv = 256,
                                   .cells = 4096,
                                   .head = true,
                                   .head_rows = head_rows,
                                   .confidence = confidence,
                                   .capture_head = capture,
                                   .hidden_rows = 513};
    const std::array<engine::Qwen38DraftWaveInput, 2> inputs = {
        engine::Qwen38DraftWaveInput{0, models_.data(), shape},
        engine::Qwen38DraftWaveInput{2, &models_[2], shape}};
    return engine::PlanQwen38DraftWave(inputs, ModelDevice(), {.paired = paired}, measurement);
  }

  std::optional<md::Qwen38MtpBinding> drafter_;
};

TEST_F(Qwen38DraftWavePlanTest, StartupMeasurementPreservesScalarAndJoinedSelectedHeads) {
  for (const auto type : {"BF16", "Q4_1"}) {
    BindDraft(true, type);
    auto ordinary = DraftPlan(true, true, 47171, true);
    ASSERT_TRUE(ordinary) << ordinary.error();
    for (const auto ceiling : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
      auto measured = DraftPlan(true, true, 47171, true, engine::ActivationMeasurement{ceiling});
      ASSERT_TRUE(measured) << measured.error();
      SameStartupPlan(**ordinary, **measured);
      EXPECT_EQ((*ordinary)->host_bytes(), (*measured)->host_bytes());
      EXPECT_EQ((*measured)->placement.measurement_bound, ceiling != 0);
    }
    const kg::Qwen38MtpShape shape{.rows = 4,
                                   .passes = 3,
                                   .n_kv = 256,
                                   .cells = 4096,
                                   .head = true,
                                   .head_rows = 47171,
                                   .confidence = true,
                                   .capture_head = true,
                                   .hidden_rows = 513};
    auto scalar = engine::PlanQwen38Mtp(models_[0], shape, ModelDevice(), 0, 0);
    auto measured = engine::PlanQwen38Mtp(models_[0], shape, ModelDevice(), 0, 0,
                                          engine::ActivationMeasurement{0});
    ASSERT_TRUE(scalar) << scalar.error();
    ASSERT_TRUE(measured) << measured.error();
    SameStartupPlan(**scalar, **measured);
    EXPECT_EQ((*scalar)->placement.extent, (*measured)->placement.extent);
  }
}

TEST_F(Qwen38DraftWavePlanTest, PrefixAndSelectedHeadsShareEveryPassWithVectorArithmetic) {
  for (const bool selected : {false, true}) {
    BindDraft(selected);
    const std::int64_t rows = selected ? 47172 : 65536;
    auto planned = DraftPlan(true, false, rows);
    ASSERT_TRUE(planned.has_value()) << planned.error();
    EXPECT_EQ((*planned)->stats().draft_head_pairs, 3U);
    EXPECT_EQ((*planned)->stats().full_head_pairs, 0U);
    unsigned heads = 0;
    for (const auto& step : (*planned)->plan.steps) {
      for (const ggml_tensor* node : step.nodes) {
        if (node->op == GGML_OP_MUL_MAT && node->ne[0] == rows) {
          EXPECT_EQ(node->ne[1], 2);
          EXPECT_EQ(step.implementation, kg::kMulMatVecFRows);
          ++heads;
        }
      }
    }
    EXPECT_EQ(heads, 3U);
  }
}

TEST_F(Qwen38DraftWavePlanTest, SelectedQ4HeadsShareOneTokenArithmeticAndFundTheirQ8Inputs) {
  BindDraft(true, "Q4_1");
  for (const auto& [paired, capture] :
       {std::pair{true, false}, std::pair{false, false}, std::pair{true, true}}) {
    auto planned = DraftPlan(paired, capture, 47172);
    ASSERT_TRUE(planned.has_value()) << planned.error();
    const bool joined = paired && !capture;
    EXPECT_EQ((*planned)->stats().vecq_pairs, joined ? 3U : 0U);
    EXPECT_EQ((*planned)->stats().draft_head_pairs, 0U);
    EXPECT_EQ((*planned)->stats().quantized_draft_head_pairs, joined ? 3U : 0U);
    unsigned heads = 0;
    for (const ggml_tensor* t : (*planned)->nodes()) {
      if (kg::JitllmOpOf(t) != kg::JitllmOp::kVecQ || t->ne[0] != 47172) continue;
      ++heads;
      EXPECT_EQ(t->ne[1], joined ? 2 : 1);
      EXPECT_EQ(t->src[0]->type, GGML_TYPE_Q4_1);
      EXPECT_EQ(t->src[0]->op, GGML_OP_NONE);
      EXPECT_EQ(t->src[0]->nb[1], 1600U);
      EXPECT_EQ(kg::VecQOneToken(t), joined);
      EXPECT_TRUE(kg::CheckVecQ(t).has_value());
      ASSERT_EQ(kg::JitllmOpOf(t->src[1]), kg::JitllmOp::kQuantizeQ8);
      EXPECT_EQ(ggml_nbytes(t->src[1]), 2880U * (joined ? 2U : 1U));
      EXPECT_NE(t->src[1]->data, nullptr);
    }
    EXPECT_EQ(heads, joined ? 3U : 6U);
  }
}

TEST_F(Qwen38DraftWavePlanTest, SelectedQ4PrefixesKeepOneTokenArithmeticAndTheirParentIdentity) {
  BindDraft(true, "Q4_1");
  for (const std::int64_t rows : {1, 31, 32, 33, 16385, 47171}) {
    for (const bool paired : {false, true}) {
      auto planned = DraftPlan(paired, false, rows);
      ASSERT_TRUE(planned.has_value()) << planned.error();
      EXPECT_EQ((*planned)->stats().vecq_pairs, paired ? 3U : 0U);
      EXPECT_EQ((*planned)->stats().quantized_draft_head_pairs, paired ? 3U : 0U);
      unsigned heads = 0;
      for (const ggml_tensor* t : (*planned)->nodes()) {
        if (kg::JitllmOpOf(t) != kg::JitllmOp::kVecQ || t->ne[0] != rows) continue;
        ++heads;
        ASSERT_EQ(t->src[0]->op, GGML_OP_VIEW);
        EXPECT_TRUE(engine::Qwen38SelectedQ4HeadPrefix(t->src[0], t->src[0]->view_src));
        EXPECT_EQ(t->src[0]->ne[1], rows);
        EXPECT_EQ(t->src[0]->view_src->ne[1], 47172);
        EXPECT_EQ(t->ne[1], paired ? 2 : 1);
        EXPECT_EQ(kg::VecQOneToken(t), paired);
        EXPECT_TRUE(kg::CheckVecQ(t));
        ASSERT_EQ(kg::JitllmOpOf(t->src[1]), kg::JitllmOp::kQuantizeQ8);
        EXPECT_EQ(ggml_nbytes(t->src[1]), 2880U * (paired ? 2U : 1U));
        EXPECT_NE(t->src[1]->data, nullptr);
      }
      EXPECT_EQ(heads, paired ? 3U : 6U);
    }
  }
}

TEST_F(Qwen38DraftWavePlanTest, ASelectedPrefixCannotBorrowAParentWithAMutableUnusedSuffix) {
  BindDraft(true, "Q4_1");
  const auto parent = models_[2].places.mtp_resource(drafter_->draft_output.index);
  // The requested 47171 rows are disjoint, but the complete immutable parent
  // includes its last 1600-byte row. A mutable alias there still forbids a join.
  models_[2].places.mtp_state = parent + 1600U * 47171;
  auto planned = DraftPlan(true, false, 47171);
  ASSERT_TRUE(planned.has_value()) << planned.error();
  EXPECT_EQ((*planned)->stats().quantized_draft_head_pairs, 0U);
  EXPECT_EQ((*planned)->stats().vecq_pairs, 0U);
}

TEST(Qwen38SelectedHeadTest, PrefixIdentityRejectsMalformedViewParentAddressAndSpan) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* parent = ggml_new_tensor_2d(c, GGML_TYPE_Q4_1, 2560, 47172);
  kg::TensorArena::Bind(parent, std::uint64_t{1} << 46U);
  auto* view = ggml_view_2d(c, parent, 2560, 47171, parent->nb[1], 0);
  ASSERT_TRUE(engine::Qwen38SelectedQ4HeadPrefix(parent, parent));
  ASSERT_TRUE(engine::Qwen38SelectedQ4HeadPrefix(view, parent));
  const auto original_parent = *parent, original_view = *view;
  const auto rejected = [&] {
    EXPECT_FALSE(engine::Qwen38SelectedQ4HeadPrefix(view, parent));
    *parent = original_parent;
    *view = original_view;
  };
  view->data = reinterpret_cast<void*>((std::uint64_t{1} << 46U) + 2);
  rejected();
  view->view_src = nullptr;
  rejected();
  view->src[0] = nullptr;
  rejected();
  view->src[1] = parent;
  rejected();
  view->view_offs = 1600;
  rejected();
  const std::size_t offset = 1600;
  std::memcpy(view->op_params, &offset, sizeof(offset));
  rejected();
  view->ne[1] = 47173;
  view->nb[2] = view->nb[3] = 1600U * 47173;
  rejected();
  view->nb[0] = 18;
  rejected();
  view->nb[1] += 20;
  rejected();
  view->nb[2] += 20;
  rejected();
  view->type = GGML_TYPE_Q4_0;
  rejected();
  parent->op = GGML_OP_ADD;
  rejected();
  parent->src[0] = view;
  rejected();
  parent->view_src = view;
  rejected();
  parent->ne[1] = 248321;
  parent->nb[2] = parent->nb[3] = 1600U * 248321;
  rejected();
  parent->data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1599);
  view->data = parent->data;
  rejected();
  EXPECT_TRUE(engine::Qwen38SelectedQ4HeadPrefix(view, parent));
}

TEST_F(Qwen38DraftWavePlanTest, ConfidenceOutputsRetainTheirPackedI32ProbabilityBits) {
  for (const std::string_view type : {"BF16", "Q4_1"}) {
    BindDraft(true, type);
    for (const bool paired : {false, true}) {
      auto planned = DraftPlan(paired, false, 47172, true);
      ASSERT_TRUE(planned.has_value()) << planned.error();
      for (const std::size_t slot : {0U, 2U}) {
        const auto* graph = (*planned)->draft(slot);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(graph->probabilities.size(), 3U);
        for (const auto* probability : graph->probabilities) {
          ASSERT_NE(probability, nullptr);
          EXPECT_EQ(probability->type, GGML_TYPE_I32);
          EXPECT_EQ(ggml_nbytes(probability), sizeof(float));
          EXPECT_TRUE(ggml_is_contiguous(probability));
          EXPECT_NE(probability->data, nullptr);
          ASSERT_NE(probability->view_src, nullptr);
          EXPECT_EQ(kg::JitllmOpOf(probability->view_src), kg::JitllmOp::kArgmax);
          EXPECT_EQ(probability->view_src->ne[0], 2);
          EXPECT_EQ(probability->view_offs, sizeof(std::int32_t));
          EXPECT_EQ(std::ranges::count((*planned)->nodes(), probability), 1);
        }
      }
    }
  }
}

TEST_F(Qwen38DraftWavePlanTest, UnpairedAndCapturedDraftHeadsRemainSeparate) {
  BindDraft(false);
  for (const auto& [paired, capture] : {std::pair{false, false}, std::pair{true, true}}) {
    auto planned = DraftPlan(paired, capture);
    ASSERT_TRUE(planned.has_value()) << planned.error();
    EXPECT_EQ((*planned)->stats().draft_head_pairs, 0U);
    unsigned heads = 0;
    for (const ggml_tensor* node : (*planned)->nodes()) {
      heads += node->op == GGML_OP_MUL_MAT && node->ne[0] == 65536 ? 1U : 0U;
    }
    EXPECT_EQ(heads, 6U);
  }
}

TEST_F(Qwen38DraftWavePlanTest, DifferentHeadBackingNeverShares) {
  BindDraft(false);
  if (!binding_.has_value()) {
    ADD_FAILURE() << "target binding is absent";
    return;
  }
  const auto head = binding_->output.index;
  models_[2].places.resource = [head](std::uint32_t i) {
    return kResources + (std::uint64_t{i} << 31U) + (i == head ? (std::uint64_t{1} << 30U) : 0);
  };
  auto planned = DraftPlan();
  ASSERT_TRUE(planned.has_value()) << planned.error();
  EXPECT_EQ((*planned)->stats().draft_head_pairs, 0U);
}

}  // namespace
