// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Independent physical counts and prelaunch contracts; fake addresses are
// never dereferenced. Content proof applies to the six selected values only.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/ggml/dsv4_ds4_moe.h"

namespace {
namespace kg = jitllm::kernels::ggml;
kg::Ds4CacheBuffer Buffer(std::uint64_t bytes, std::uint64_t slot) {
  return bytes == 0 ? kg::Ds4CacheBuffer{} : kg::Ds4CacheBuffer{(slot + 1) << 36, bytes};
}
kg::Ds4Moe Descriptor(kg::Ds4MoeShape shape, kg::Ds4MoeTier tier) {
  const auto plan = kg::Ds4MoeLayoutOf(shape, tier);
  if (!plan) return {};
  const auto& p = *plan;
  const auto rows = static_cast<std::uint64_t>(shape.rows);
  const auto pairs = rows * 6;
  const bool vector = tier == kg::Ds4MoeTier::kVector;
  const bool materialized =
      tier == kg::Ds4MoeTier::kMaterialized || tier == kg::Ds4MoeTier::kClassic;
  kg::Ds4Moe d;
  d.shape = shape;
  d.tier = tier;
  d.input = Buffer(rows * shape.input * 4, 0);
  d.gate_weights = Buffer(p.gate_weight_bytes, 1);
  d.up_weights = Buffer(p.gate_weight_bytes, 2);
  d.down_weights = Buffer(p.down_weight_bytes, 3);
  d.selected = Buffer(pairs * 4, 4);
  d.weights = Buffer(pairs * 4, 5);
  d.ids_source = Buffer(vector ? 0 : pairs * 4, 6);
  d.ids_destination = Buffer(vector ? 0 : pairs * 4, 7);
  d.expert_bounds = Buffer(vector ? 0 : 257 * 4, 8);
  d.work = Buffer(p.work_bytes, 9);
  d.input_quant = Buffer(p.input_quant_bytes, 10);
  d.down_quant = Buffer(p.down_quant_bytes, 11);
  d.gate = Buffer(materialized ? p.middle_bytes : 0, 12);
  d.up = Buffer(materialized ? p.middle_bytes : 0, 13);
  d.middle = Buffer((materialized || vector) ? p.middle_bytes : 0, 14);
  d.down = Buffer(p.down_bytes, 15);
  return d;
}
void Refused(const std::expected<void, kg::KernelFailure>& r) {
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().error, kg::KernelError::kRejected);
}
kg::Ds4Router Router() {
  return {.logits = Buffer(7ULL * 256 * 4, 0),
          .bias = {},
          .hash = {},
          .tokens = {},
          .selected = Buffer(7ULL * 6 * 4, 1),
          .weights = Buffer(7ULL * 6 * 4, 2),
          .probabilities = Buffer(7ULL * 256 * 4, 3),
          .rows = 7};
}

TEST(Ds4MoeValidate, FullModelPhysicalBytesAndConservativeArenasAreIndependent) {
  const auto p = kg::Ds4MoeLayoutOf({4096, 4096, 2048, 4096}, kg::Ds4MoeTier::kDirect);
  ASSERT_TRUE(p);
  EXPECT_EQ(p->gate_weight_bytes, 256ULL * 2048 * 16 * 66);
  EXPECT_EQ(p->down_weight_bytes, 256ULL * 2048 * 8 * 168);
  EXPECT_EQ(p->input_payload_bytes, 18874368U);
  EXPECT_EQ(p->input_quant_bytes, 113246208U + 18432U);
  EXPECT_EQ(p->down_payload_bytes, 56623104U);
  EXPECT_EQ(p->down_quant_bytes, 56623104U + 18432U);
  EXPECT_EQ(p->down_bytes, 402653184U);
  EXPECT_EQ(p->work_bytes, (768U + 256U + 1U) * 4U);
  const auto v = kg::Ds4MoeLayoutOf({8, 4096, 2048, 4096}, kg::Ds4MoeTier::kVector);
  ASSERT_TRUE(v);
  EXPECT_EQ(v->input_quant_bytes, 8U * 128U * 36U);
  EXPECT_EQ(v->down_quant_bytes, 48U * 64U * 36U);
  EXPECT_EQ(v->work_bytes, 0U);
}

TEST(Ds4MoeValidate, TiersRefuseUnqualifiedShapesAndNeedCompleteBoundedResources) {
  constexpr std::array tiers = {kg::Ds4MoeTier::kVector, kg::Ds4MoeTier::kDirect,
                                kg::Ds4MoeTier::kMaterialized, kg::Ds4MoeTier::kClassic};
  for (auto tier : tiers) {
    auto d = Descriptor({tier == kg::Ds4MoeTier::kVector ? 8U : 173U, 1024, 256, 130}, tier);
    ASSERT_TRUE(kg::CheckDs4Moe(d));
    for (auto member : {&kg::Ds4Moe::input, &kg::Ds4Moe::gate_weights, &kg::Ds4Moe::up_weights,
                        &kg::Ds4Moe::down_weights, &kg::Ds4Moe::selected, &kg::Ds4Moe::weights,
                        &kg::Ds4Moe::input_quant, &kg::Ds4Moe::down_quant, &kg::Ds4Moe::down}) {
      auto bad = d;
      --(bad.*member).bytes;
      Refused(kg::CheckDs4Moe(bad));
      bad = d;
      ++(bad.*member).address;
      Refused(kg::CheckDs4Moe(bad));
      bad = d;
      (bad.*member).address = 0;
      Refused(kg::CheckDs4Moe(bad));
      bad = d;
      (bad.*member).bytes = std::numeric_limits<std::uint64_t>::max();
      Refused(kg::CheckDs4Moe(bad));
    }
    auto bad = d;
    bad.shape.rows = 0;
    Refused(kg::CheckDs4Moe(bad));
    bad = d;
    bad.shape.input = 512;
    Refused(kg::CheckDs4Moe(bad));
    bad = d;
    bad.shape.middle = 128;
    Refused(kg::CheckDs4Moe(bad));
    bad = d;
    bad.shape.output = 129;
    Refused(kg::CheckDs4Moe(bad));
    bad = d;
    bad.down.address = bad.input.address;
    Refused(kg::CheckDs4Moe(bad));
    bad = d;
    bad.input_quant.address = bad.down_quant.address;
    Refused(kg::CheckDs4Moe(bad));
  }
  EXPECT_FALSE(kg::Ds4MoeLayoutOf({17, 1024, 256, 130}, kg::Ds4MoeTier::kVector));
  EXPECT_FALSE(kg::Ds4MoeLayoutOf({170, 1024, 256, 130}, kg::Ds4MoeTier::kDirect));
  EXPECT_FALSE(kg::Ds4MoeLayoutOf({4097, 1024, 256, 130}, kg::Ds4MoeTier::kClassic));
}

TEST(Ds4MoeValidate, DownD2rTailRequiresUniformWarpBarrierPathsBeforePacking) {
  constexpr std::array d2r = {kg::Ds4MoeTier::kDirect, kg::Ds4MoeTier::kMaterialized};
  for (auto tier : d2r) {
    for (std::uint32_t output : {2U, 14U, 128U, 130U, 142U, 256U, 258U, 4096U}) {
      const auto d = Descriptor({173, 1024, 256, output}, tier);
      EXPECT_TRUE(kg::CheckDs4Moe(d)) << output;
    }
    // These widths leave at least one full 16-row warp and one guarded
    // warp in the last 128-row CTA when an expert has 64 assignments.
    for (std::uint32_t output : {16U, 126U, 144U, 200U, 254U, 4094U}) {
      EXPECT_FALSE(kg::Ds4MoeLayoutOf({173, 1024, 256, output}, tier)) << output;
      auto bad = Descriptor({173, 1024, 256, 256}, tier);
      bad.shape.output = output;
      // Supply noncompact IDs: refusal must precede any packing launch.
      bad.selected_stride = 256;
      bad.selected.bytes = ((172ULL * 256ULL) + 6ULL) * 4ULL;
      bad.compact_ids = Buffer(173ULL * 6 * 4, 19);
      Refused(kg::CheckDs4Moe(bad));
    }
  }
  for (auto tier : {kg::Ds4MoeTier::kClassic, kg::Ds4MoeTier::kVector}) {
    for (std::uint32_t output : {16U, 144U, 200U, 4094U}) {
      const auto d =
          Descriptor({tier == kg::Ds4MoeTier::kVector ? 8U : 173U, 1024, 256, output}, tier);
      EXPECT_TRUE(kg::CheckDs4Moe(d)) << output;
    }
  }
}

TEST(Ds4MoeValidate, SixOf256StridesPackWithoutReadingPaddingAsExpertIds) {
  auto d = Descriptor({173, 1024, 256, 130}, kg::Ds4MoeTier::kDirect);
  d.selected_stride = 256;
  d.weight_stride = 256;
  d.selected.bytes = ((172ULL * 256ULL) + 6ULL) * 4ULL;
  d.weights.bytes = d.selected.bytes;
  d.compact_ids = Buffer(173ULL * 6 * 4, 19);
  d.compact_weights = Buffer(173ULL * 6 * 4, 20);
  EXPECT_TRUE(kg::CheckDs4Moe(d));
  auto bad = d;
  bad.compact_ids = {};
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  --bad.selected.bytes;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  bad.selected_stride = 257;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  bad.compact_ids.address = bad.selected.address;
  Refused(kg::CheckDs4Moe(bad));
  std::vector<std::int32_t> ids(2ULL * 256, -999);
  for (std::uint32_t t = 0; t < 2; ++t)
    for (std::uint32_t s = 0; s < 6; ++s)
      ids[(t * 256) + s] = static_cast<std::int32_t>((t * 6) + s);
  EXPECT_TRUE(kg::CheckDs4MoeIds(ids, 2, 256));
  ids[256 + 5] = ids[256];
  Refused(kg::CheckDs4MoeIds(ids, 2, 256));
  ids[256 + 5] = 256;
  Refused(kg::CheckDs4MoeIds(ids, 2, 256));
  ids[256 + 5] = -1;
  Refused(kg::CheckDs4MoeIds(ids, 2, 256));
}

TEST(Ds4MoeValidate, CurrentProducerRequiresSourceGenerationShapeLayoutAndSlack) {
  auto d = Descriptor({173, 1024, 256, 130}, kg::Ds4MoeTier::kDirect);
  d.generation = 7;
  const auto p = *kg::Ds4MoeLayoutOf(d.shape, d.tier);
  d.producer = {.storage = Buffer(p.input_payload_bytes + 18432, 21),
                .source_address = d.input.address,
                .generation = 7,
                .rows = 173,
                .width = 1024,
                .kind = kg::Ds4MoeQuant::kD4};
  EXPECT_TRUE(kg::CheckDs4Moe(d));
  auto bad = d;
  ++bad.producer.generation;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  bad.producer.source_address += 16;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  --bad.producer.rows;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  --bad.producer.storage.bytes;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  bad.producer.kind = kg::Ds4MoeQuant::kQ81;
  Refused(kg::CheckDs4Moe(bad));
  bad = d;
  bad.tier = kg::Ds4MoeTier::kMaterialized;
  Refused(kg::CheckDs4Moe(bad));
}

TEST(Ds4MoeValidate, RouterHashModeAndCooperativeBorrowedRangesAreExplicit) {
  auto r = Router();
  EXPECT_TRUE(kg::CheckDs4Router(r));
  auto bad = r;
  bad.weights.address = bad.logits.address;
  Refused(kg::CheckDs4Router(bad));
  bad = r;
  --bad.probabilities.bytes;
  Refused(kg::CheckDs4Router(bad));
  r.hash_rows = 3;
  r.hash = Buffer(3ULL * 6 * 4, 4);
  r.tokens = Buffer(7ULL * 4, 5);
  EXPECT_TRUE(kg::CheckDs4Router(r));
  bad = r;
  bad.bias = Buffer(256ULL * 4, 6);
  Refused(kg::CheckDs4Router(bad));
  bad = r;
  bad.tokens = {};
  Refused(kg::CheckDs4Router(bad));
  const kg::Ds4RouterCooperative coop{r, Buffer(7ULL * 4096 * 4, 7), Buffer(256ULL * 4096 * 2, 8),
                                      Buffer(7ULL * 256 * 8 * 4, 9)};
  EXPECT_TRUE(kg::CheckDs4RouterCooperative(coop));
  auto c = coop;
  c.router.rows = 9;
  Refused(kg::CheckDs4RouterCooperative(c));
  c = coop;
  c.partials.address = c.input.address;
  Refused(kg::CheckDs4RouterCooperative(c));
  c = coop;
  c.projection.address += 2;
  Refused(kg::CheckDs4RouterCooperative(c));
}

TEST(Ds4MoeValidate, SharedAndGuardedSixSumCannotAliasInputsOrEscapeBounds) {
  const kg::Ds4SharedSwiglu s{Buffer(7ULL * 129 * 4, 0), Buffer(7ULL * 129 * 4, 1),
                              Buffer(7ULL * 129 * 4, 2), 7, 129};
  EXPECT_TRUE(kg::CheckDs4SharedSwiglu(s));
  auto bad = s;
  bad.output.address = s.up.address;
  Refused(kg::CheckDs4SharedSwiglu(bad));
  bad = s;
  --bad.gate.bytes;
  Refused(kg::CheckDs4SharedSwiglu(bad));
  const kg::Ds4MoeSum sum{Buffer(7ULL * 131 * 6 * 4, 0), Buffer(7ULL * 131 * 4, 1), 7, 131};
  EXPECT_TRUE(kg::CheckDs4MoeSum(sum));
  auto invalid = sum;
  --invalid.slots.bytes;
  Refused(kg::CheckDs4MoeSum(invalid));
  invalid = sum;
  invalid.output.address = sum.slots.address;
  Refused(kg::CheckDs4MoeSum(invalid));
}
}  // namespace
