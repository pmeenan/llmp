// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Views are fake addresses, never dereferenced. These check refusal before
// submission, optional-output eligibility and complete accessed ranges.
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "kernels/ggml/dsv4_ds4_hc.h"

namespace {
namespace kg = jitllm::kernels::ggml;

TEST(Ds4HcValidate, RmsRejectsPartialAliasBadRangesAndProducerEligibility) {
  kg::Ds4Rms desc{.source = {0x100000, 3084}, .values = {0x200000, 3084}, .width = 257, .rows = 3};
  EXPECT_TRUE(kg::CheckDs4Rms(desc));
  desc.values = desc.source;
  EXPECT_TRUE(kg::CheckDs4Rms(desc));  // exact in-place is safe after the reduction
  desc.values.address += 4;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.values = {0x200000, 3083};
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.values = {0x200002, 3084};
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.values = {std::numeric_limits<std::uint64_t>::max() - 3, 3084};
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.values = {};
  desc.values_f16 = {0x200000, 1542};
  EXPECT_TRUE(kg::CheckDs4Rms(desc));
  desc.weights = {0x300000, 1028};
  EXPECT_FALSE(kg::CheckDs4Rms(desc));  // weighted F16 always has the F32 twin
  desc.values = {0x400000, 3084};
  EXPECT_TRUE(kg::CheckDs4Rms(desc));
  desc.values_f16.address = desc.values.address;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc = {.source = {0x100000, 65536},
          .weights = {0x200000, 1024},
          .values = {0x300000, 65536},
          .values_f16 = {0x400000, 32768},
          .q8_d4 = {0x500000, 18432},
          .width = 256,
          .rows = 64};
  EXPECT_TRUE(kg::CheckDs4Rms(desc));
  desc.rows = 63;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.rows = 64;
  desc.width = 255;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.width = 256;
  desc.values.address += 4;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));  // float4 producer read alignment
  desc.values.address -= 4;
  desc.q8_d4.bytes -= 1;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.q8_d4 = {0x500000, 18432};
  desc.epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
  desc.epsilon = 1e-6f;
  desc.rows = 4097;
  EXPECT_FALSE(kg::CheckDs4Rms(desc));
}

TEST(Ds4HcValidate, SplitRefusesIterationEpsilonAndOutputAlias) {
  kg::Ds4HcSplit desc{.mix = {0x100000, 1632},
                      .scale = {0x200000, 12},
                      .base = {0x300000, 96},
                      .split = {0x400000, 1632},
                      .rows = 17};
  EXPECT_TRUE(kg::CheckDs4HcSplit(desc));
  desc.iterations = 0;
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
  desc.iterations = 21;
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
  desc.iterations = 20;
  desc.epsilon = 0;
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
  desc.epsilon = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
  desc.epsilon = 1e-6f;
  desc.split = desc.mix;
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
  desc.split = {0x400000, 1632};
  desc.base.bytes = 92;
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
  desc.base.bytes = 96;
  desc.rows = 0;
  EXPECT_FALSE(kg::CheckDs4HcSplit(desc));
}

TEST(Ds4HcValidate, WeightedChecksLastCoefficientAndCrossStageAlias) {
  kg::Ds4HcWeighted weighted{.residual = {0x100000, 12336},
                             .weights = {0x200000, 208},
                             .values = {0x300000, 3084},
                             .width = 257,
                             .rows = 3};
  EXPECT_TRUE(kg::CheckDs4HcWeighted(weighted));
  weighted.weights.bytes = 204;
  EXPECT_FALSE(kg::CheckDs4HcWeighted(weighted));
  weighted.weights.bytes = 48;
  weighted.weight_stride = 4;
  EXPECT_TRUE(kg::CheckDs4HcWeighted(weighted));
  weighted.weight_stride = 8;
  EXPECT_FALSE(kg::CheckDs4HcWeighted(weighted));
  weighted.weight_stride = 4;
  weighted.values.address = weighted.residual.address;
  EXPECT_FALSE(kg::CheckDs4HcWeighted(weighted));
  kg::Ds4HcPre pre{.coefficients = {.mix = {0x100000, 288},
                                    .scale = {0x200000, 12},
                                    .base = {0x300000, 96},
                                    .split = {0x400000, 288},
                                    .rows = 3},
                   .residual = {0x500000, 12336},
                   .values = {0x600000, 3084},
                   .width = 257};
  EXPECT_TRUE(kg::CheckDs4HcPre(pre));
  pre.values.address = pre.coefficients.scale.address;
  EXPECT_FALSE(kg::CheckDs4HcPre(pre));
  pre.values.address = 0x600000;
  pre.residual.address = pre.coefficients.split.address;
  EXPECT_FALSE(kg::CheckDs4HcPre(pre));
}

TEST(Ds4HcValidate, ExpandRequiresExactFoldedShapeAndSixSlotExtent) {
  kg::Ds4HcExpand desc{.block = {0x100000, 147456},
                       .add = {0x200000, 147456},
                       .residual = {0x300000, 589824},
                       .split = {0x400000, 864},
                       .values = {0x1000000, 589824},
                       .values_f16 = {0x2000000, 294912},
                       .rows = 9};
  EXPECT_TRUE(kg::CheckDs4HcExpand(desc));
  desc.rows = 8;
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));
  desc.rows = 9;
  desc.width = 4095;
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));
  desc.width = 4096;
  desc.values_f16 = {};
  desc.width = 257;
  EXPECT_TRUE(kg::CheckDs4HcExpand(desc));  // ragged plain expand, no hidden fold
  desc.width = 4096;
  desc.values_f16 = {0x2000000, 294912};
  desc.moe_unsummed = {0x3000000, 884736};
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));  // cannot supply both block and slots
  desc.block = {};
  EXPECT_TRUE(kg::CheckDs4HcExpand(desc));
  desc.moe_unsummed.bytes -= 4;
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));
  desc.moe_unsummed.bytes += 4;
  desc.add = {};
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));
  desc.add = {0x200000, 147456};
  desc.values.address = desc.residual.address;
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));
  desc.values.address = 0x1000000;
  desc.values_f16.address = desc.values.address;
  EXPECT_FALSE(kg::CheckDs4HcExpand(desc));
}

TEST(Ds4HcValidate, HeadWeightsCheckCompleteCoefficientAndOutputRanges) {
  kg::Ds4HcHeadWeights desc{.pre = {0x100000, 144},
                            .scale = {0x200000, 4},
                            .base = {0x300000, 16},
                            .values = {0x400000, 144},
                            .rows = 9};
  EXPECT_TRUE(kg::CheckDs4HcHeadWeights(desc));
  desc.values.address = desc.pre.address;
  EXPECT_FALSE(kg::CheckDs4HcHeadWeights(desc));
  desc.values.address = 0x400000;
  desc.scale = {0, 4};
  EXPECT_FALSE(kg::CheckDs4HcHeadWeights(desc));
  desc.scale = {0x200000, 3};
  EXPECT_FALSE(kg::CheckDs4HcHeadWeights(desc));
}

}  // namespace
