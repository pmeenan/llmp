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
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "engine/qwen38_plan.h"
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
      }};
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
      std::span<const Request> requests, bool paired = true) {
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
    }
    return engine::PlanQwen38TargetWave(inputs, ModelDevice(),
                                        {.paired = paired, .share_target_head = true});
  }

  // A lone slot's products (the composition keeps them as they are).
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

}  // namespace
