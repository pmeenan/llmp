// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_product.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>

namespace {
namespace kg = jitllm::kernels::ggml;
constexpr std::uint64_t kBase = 1ULL << 40;
constexpr std::uint64_t kSlot = 1ULL << 36;

kg::Ds4ProductRead Read(std::uint64_t slot, std::uint64_t bytes) {
  return {.data = std::bit_cast<const void*>(kBase + (slot * kSlot)), .bytes = bytes};
}
kg::Ds4ProductWrite Write(std::uint64_t slot, std::uint64_t bytes) {
  return {.data = std::bit_cast<void*>(kBase + (slot * kSlot)), .bytes = bytes};
}
kg::Ds4ProductMatrix Matrix(std::uint64_t slot, std::uint32_t rows, std::uint32_t cols,
                            std::uint64_t element, std::uint64_t stride = 0) {
  if (stride == 0) stride = static_cast<std::uint64_t>(cols) * element;
  return {.storage = Read(slot, static_cast<std::uint64_t>(rows) * stride),
          .rows = rows,
          .columns = cols,
          .row_stride = stride};
}
kg::Ds4ProductOutput Output(std::uint64_t slot, std::uint32_t rows, std::uint32_t cols,
                            std::uint64_t stride = 0) {
  if (stride == 0) stride = static_cast<std::uint64_t>(cols) * 4;
  return {.storage = Write(slot, static_cast<std::uint64_t>(rows) * stride),
          .rows = rows,
          .columns = cols,
          .row_stride = stride};
}
kg::Ds4D4Sidecar Sidecar(const kg::Ds4ProductMatrix& x) {
  return {.storage = Write(5, *kg::Ds4D4Bytes(x.rows, x.columns)),
          .source = x.storage.data,
          .generation = 7,
          .rows = x.rows,
          .columns = x.columns};
}
kg::Ds4Q8Weights Weights(std::uint32_t m, std::uint32_t k) {
  return {.raw = Read(1, static_cast<std::uint64_t>(m) * (k / 32) * 34),
          .raw_row_stride = static_cast<std::uint64_t>(k / 32) * 34,
          .scales = Read(2, static_cast<std::uint64_t>(m) * (k / 32) * 2),
          .scale_row_stride = static_cast<std::uint64_t>(k / 32) * 2,
          .codes = Read(3, static_cast<std::uint64_t>(m) * k),
          .code_row_stride = k,
          .rows = m,
          .columns = k};
}
void Refused(const std::expected<void, kg::KernelFailure>& c) {
  ASSERT_FALSE(c);
  EXPECT_EQ(c.error().error, kg::KernelError::kRejected);
}

TEST(Dsv4Ds4Product, FrontendViewsRefusePartialAliasesMissingTokensAndOversizedGrids) {
  kg::Ds4Embedding embed{
      .tokens = Read(1, 13ULL * 4), .weights = Matrix(2, 17, 32, 2), .output = Output(3, 13, 128)};
  ASSERT_TRUE(kg::CheckDs4Embedding(embed));
  auto bad = embed;
  --bad.tokens.bytes;
  Refused(kg::CheckDs4Embedding(bad));
  bad = embed;
  bad.output.storage = Write(2, embed.output.storage.bytes);
  Refused(kg::CheckDs4Embedding(bad));
  bad = embed;
  bad.hyper_connections = 3;
  Refused(kg::CheckDs4Embedding(bad));
  kg::Ds4F16Conversion convert{
      .input = Matrix(1, 13, 32, 4),
      .output = {
          .storage = Write(2, 13ULL * 32 * 2), .rows = 13, .columns = 32, .row_stride = 32ULL * 2}};
  ASSERT_TRUE(kg::CheckDs4F16Conversion(convert));
  convert.output.storage = Write(1, 13ULL * 32 * 2);
  Refused(kg::CheckDs4F16Conversion(convert));
  convert.input = Matrix(1, 65535, 1U << 30, 4);
  convert.output = {.storage = Write(100000, 65535ULL * (1ULL << 30) * 2),
                    .rows = 65535,
                    .columns = 1U << 30,
                    .row_stride = 1ULL << 31};
  Refused(kg::CheckDs4F16Conversion(convert));
}

TEST(Dsv4Ds4Product, FusedQkvNormAcceptsIndependentExactAliasesAndRefusesCrossPlaneWrites) {
  kg::Ds4QkvNorm d{.query = Matrix(1, 13, 1024, 4),
                   .query_weight = Read(3, 1024ULL * 4),
                   .query_output = Output(4, 13, 1024),
                   .kv = Matrix(2, 13, 512, 4),
                   .kv_weight = Read(5, 512ULL * 4),
                   .kv_output = Output(6, 13, 512)};
  ASSERT_TRUE(kg::CheckDs4QkvNorm(d));
  d.query_output.storage = Write(1, d.query.storage.bytes);
  d.kv_output.storage = Write(2, d.kv.storage.bytes);
  ASSERT_TRUE(kg::CheckDs4QkvNorm(d));
  auto bad = d;
  bad.query_output.storage.data = std::bit_cast<void*>(kBase + kSlot + 4);
  Refused(kg::CheckDs4QkvNorm(bad));
  bad = d;
  bad.query_output.storage = Write(2, bad.query_output.storage.bytes);
  Refused(kg::CheckDs4QkvNorm(bad));
  bad = d;
  --bad.kv_weight.bytes;
  Refused(kg::CheckDs4QkvNorm(bad));
  bad = d;
  bad.epsilon = std::numeric_limits<float>::quiet_NaN();
  Refused(kg::CheckDs4QkvNorm(bad));
  bad = d;
  bad.query.rows = bad.query_output.rows = bad.kv.rows = bad.kv_output.rows = 8;
  Refused(kg::CheckDs4QkvNorm(bad));
}

TEST(Dsv4Ds4Product, WideF16HasExplicitPhysicalLeadingDimensionsAndNoSmallFallback) {
  kg::Ds4F16Product d{.weights = Matrix(1, 17, 32, 2, 80),
                      .input = Matrix(2, 9, 32, 2, 96),
                      .output = Output(3, 9, 17, 80)};
  ASSERT_TRUE(kg::CheckDs4F16Product(d));
  auto bad = d;
  bad.input.rows = bad.output.rows = 8;
  Refused(kg::CheckDs4F16Product(bad));
  bad = d;
  bad.weights.storage.bytes = ((17 - 1) * 80) + 64 - 1;
  Refused(kg::CheckDs4F16Product(bad));
  bad = d;
  bad.input.row_stride = 63;
  Refused(kg::CheckDs4F16Product(bad));
  bad = d;
  bad.output.storage = Write(1, d.output.storage.bytes);
  Refused(kg::CheckDs4F16Product(bad));
  bad = d;
  bad.output.columns = 18;
  Refused(kg::CheckDs4F16Product(bad));
}

TEST(Dsv4Ds4Product, D4RequiresGuardedTailAndCurrentSourceIdentity) {
  const auto x = Matrix(4, 9, 512, 4);
  const auto q = Sidecar(x);
  EXPECT_EQ(q.storage.bytes, ((9ULL * 4) + 256) * 144);
  ASSERT_TRUE(kg::CheckDs4D4(x, 7, q));
  auto bad = q;
  --bad.storage.bytes;
  Refused(kg::CheckDs4D4(x, 7, bad));
  bad = q;
  bad.generation = 8;
  Refused(kg::CheckDs4D4(x, 7, bad));
  bad = q;
  bad.source = Read(6, 16).data;
  Refused(kg::CheckDs4D4(x, 7, bad));
  bad = q;
  --bad.rows;
  Refused(kg::CheckDs4D4(x, 7, bad));
  bad = q;
  bad.storage.data = const_cast<void*>(x.storage.data);
  Refused(kg::CheckDs4D4(x, 7, bad));
  auto padded = x;
  padded.row_stride += 16;
  padded.storage.bytes += 9ULL * 16;
  Refused(kg::CheckDs4D4(padded, 7, q));
  EXPECT_FALSE(kg::Ds4D4Bytes(65536, 512));
  EXPECT_FALSE(kg::Ds4D4Bytes(1, 508));
  EXPECT_FALSE(kg::Ds4D4Bytes(65535, 32768));
}

TEST(Dsv4Ds4Product, D4RejectsGridYOverflowBeforeAnOtherwiseBoundedLaunch) {
  const auto x = Matrix(4, 1, 65535 * 512, 4);
  const auto q = Sidecar(x);
  ASSERT_TRUE(kg::CheckDs4D4(x, 7, q));
  auto bad_x = Matrix(4, 1, 65536 * 512, 4);
  auto bad_q = q;
  bad_q.source = bad_x.storage.data;
  bad_q.columns = bad_x.columns;
  bad_q.storage.bytes = (static_cast<std::uint64_t>(bad_x.columns / 128) + 256) * 144;
  Refused(kg::CheckDs4D4(bad_x, 7, bad_q));
  EXPECT_FALSE(kg::Ds4D4Bytes(bad_x.rows, bad_x.columns));
}

TEST(Dsv4Ds4Product, SmallF16PlansOriginalSplitTierWithoutRoundingActivations) {
  kg::Ds4F16Vector d{.weights = Matrix(1, 24, 16384, 2),
                     .input = Matrix(2, 4, 16384, 4),
                     .output = Output(3, 4, 24)};
  ASSERT_TRUE(kg::CheckDs4F16Vector(d));
  EXPECT_EQ(*kg::PlanDs4F16Vector(d), 24ULL * 32 * 4);
  auto full = d;
  full.weights = Matrix(1, 2048, 16384, 2);
  full.output = Output(3, 4, 2048);
  EXPECT_EQ(*kg::PlanDs4F16Vector(full), 0);
  auto scalar = d;
  scalar.weights = Matrix(1, 17, 1031, 2);
  scalar.input = Matrix(2, 4, 1031, 4);
  scalar.output = Output(3, 4, 17);
  EXPECT_EQ(*kg::PlanDs4F16Vector(scalar), 17ULL * 2 * 4);
  auto bad = d;
  bad.input.rows = bad.output.rows = 9;
  Refused(kg::CheckDs4F16Vector(bad));
  bad = d;
  bad.input.row_stride += 4;
  bad.input.storage.bytes += 4ULL * 4;
  Refused(kg::CheckDs4F16Vector(bad));
  bad = d;
  bad.output.storage.data = const_cast<void*>(bad.input.storage.data);
  Refused(kg::CheckDs4F16Vector(bad));
}

TEST(Dsv4Ds4Product, Q8LiteralPathsRefuseUnsupportedDenseAndRawWeightChoices) {
  const auto x = Matrix(4, 513, 2048, 4);
  kg::Ds4Q8Product d{.weights = Weights(2048, 2048),
                     .input = x,
                     .output = Output(6, x.rows, 2048),
                     .quantized = Sidecar(x),
                     .generation = 7,
                     .path = kg::Ds4Q8Path::kDenseD2r};
  ASSERT_TRUE(kg::CheckDs4Q8Product(d));
  d.prepared = true;
  ASSERT_TRUE(kg::CheckDs4Q8Product(d));
  auto bad = d;
  bad.weights.code_row_stride += 16;
  Refused(kg::CheckDs4Q8Product(bad));
  bad = d;
  --bad.weights.scales.bytes;
  Refused(kg::CheckDs4Q8Product(bad));
  bad = d;
  bad.weights.rows = bad.output.columns = 1024;
  bad.output.row_stride = 1024ULL * 4;
  Refused(kg::CheckDs4Q8Product(bad));
  d.path = kg::Ds4Q8Path::kMmq;
  ASSERT_TRUE(kg::CheckDs4Q8Product(d));
  bad = d;
  ++bad.weights.raw_row_stride;
  Refused(kg::CheckDs4Q8Product(bad));
  bad = d;
  bad.weights.raw.data = d.quantized.storage.data;
  Refused(kg::CheckDs4Q8Product(bad));
  bad = d;
  // Deliberately malformed descriptor; validate refusal rather than an enum consumer.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  bad.path = static_cast<kg::Ds4Q8Path>(255);
  Refused(kg::CheckDs4Q8Product(bad));
}

TEST(Dsv4Ds4Product, HeadRopeChecksTailCoordinatesAndPositionOwnership) {
  kg::Ds4HeadRope d{.input = Output(1, 13, 64 * 512),
                    .heads = 64,
                    .head_width = 512,
                    .rope = {.first = 1024,
                             .original_context = 131072,
                             .rotary = 64,
                             .base = 10000.0F,
                             .scale = 0.25F,
                             .extension = 1.0F},
                    .normalize = true};
  ASSERT_TRUE(kg::CheckDs4HeadRope(d));
  auto bad = d;
  bad.rope.first = std::numeric_limits<std::int32_t>::max() - 11;
  Refused(kg::CheckDs4HeadRope(bad));
  bad = d;
  bad.rope.rotary = std::numeric_limits<std::uint32_t>::max() - 1;
  Refused(kg::CheckDs4HeadRope(bad));
  bad = d;
  bad.rope.base = 1;
  Refused(kg::CheckDs4HeadRope(bad));
  bad = d;
  bad.epsilon = std::numeric_limits<float>::quiet_NaN();
  Refused(kg::CheckDs4HeadRope(bad));
  d.rope.positions = Read(2, 13ULL * 4);
  ASSERT_TRUE(kg::CheckDs4HeadRope(d));  // producer may supply negative I32 positions
  bad = d;
  bad.rope.positions.bytes = (13ULL * 4) - 1;
  Refused(kg::CheckDs4HeadRope(bad));
  bad = d;
  bad.rope.positions.data = d.input.storage.data;
  Refused(kg::CheckDs4HeadRope(bad));
}

TEST(Dsv4Ds4Product, MmqRejectsSignedOutputOverflowWithOtherwiseBoundedOperands) {
  auto x = Matrix(4, 32767, 512, 4);
  kg::Ds4Q8Product d{.weights = Weights(65536, 512),
                     .input = x,
                     .output = Output(6, x.rows, 65536),
                     .quantized = Sidecar(x),
                     .generation = 7,
                     .path = kg::Ds4Q8Path::kMmq};
  ASSERT_TRUE(kg::CheckDs4Q8Product(d));
  x = Matrix(4, 32768, 512, 4);
  d.input = x;
  d.output = Output(6, x.rows, 65536);
  d.quantized = Sidecar(x);
  Refused(kg::CheckDs4Q8Product(d));  // 2^31 outputs overflow original signed offsets.
}

TEST(Dsv4Ds4Product, MmqRejectsSignedRawBlockOverflowWithOtherwiseBoundedOperands) {
  const auto x = Matrix(4, 9, 1048576, 4);
  // Keep the output row stride 16-byte aligned while approaching the signed block limit.
  kg::Ds4Q8Product d{.weights = Weights(65532, 1048576),
                     .input = x,
                     .output = Output(6, x.rows, 65532),
                     .quantized = Sidecar(x),
                     .generation = 7,
                     .path = kg::Ds4Q8Path::kMmq};
  ASSERT_TRUE(kg::CheckDs4Q8Product(d));
  d.weights = Weights(65536, 1048576);
  d.output = Output(6, x.rows, 65536);
  Refused(kg::CheckDs4Q8Product(d));  // 2^31 Q8_0 blocks overflow original signed offsets.
}

TEST(Dsv4Ds4Product, SmallQ8SupportsFullHeadAndDistinctOriginalQ81Identity) {
  const auto x = Matrix(4, 1, 4096, 4);
  kg::Ds4Q8Vector d{.weights = Weights(129280, 4096),
                    .input = x,
                    .output = Output(6, 1, 129280),
                    .quantized = {.storage = Write(5, *kg::Ds4Q81Bytes(1, 4096)),
                                  .source = x.storage.data,
                                  .generation = 7,
                                  .rows = 1,
                                  .columns = 4096},
                    .generation = 7};
  ASSERT_TRUE(kg::CheckDs4Q8Vector(d));
  d.prepared = true;
  ASSERT_TRUE(kg::CheckDs4Q8Vector(d));
  d.path = kg::Ds4Q8VectorPath::kRaw;
  ASSERT_TRUE(kg::CheckDs4Q8Vector(d));
  auto bad = d;
  bad.quantized.generation = 8;
  Refused(kg::CheckDs4Q8Vector(bad));
  bad = d;
  --bad.quantized.storage.bytes;
  Refused(kg::CheckDs4Q8Vector(bad));
  bad = d;
  bad.quantized.source = Read(7, 16).data;
  Refused(kg::CheckDs4Q8Vector(bad));
  bad = d;
  bad.output.columns = 2;
  Refused(kg::CheckDs4Q8Vector(bad));
  bad = d;
  bad.quantized.storage.data = d.output.storage.data;
  Refused(kg::CheckDs4Q8Vector(bad));
  bad = d;
  // Deliberately malformed descriptor; validate refusal rather than an enum consumer.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  bad.path = static_cast<kg::Ds4Q8VectorPath>(255);
  Refused(kg::CheckDs4Q8Vector(bad));
  EXPECT_EQ(*kg::Ds4Q81Bytes(8, 256), 8ULL * 16 * 36);
  EXPECT_FALSE(kg::Ds4Q81Bytes(9, 4096));
  EXPECT_FALSE(kg::Ds4Q81Bytes(1, 1023));
  EXPECT_FALSE(kg::Ds4Q81Bytes(8, 2147483392));
}

TEST(Dsv4Ds4Product, SmallRawQ8RequiresReadableWholeRowTilesBeforeGuardedStores) {
  const auto x = Matrix(4, 1, 256, 4);
  kg::Ds4Q8Vector d{.weights = Weights(17, 256),
                    .input = x,
                    .output = Output(6, 1, 17),
                    .quantized = {.storage = Write(5, *kg::Ds4Q81Bytes(1, 256)),
                                  .source = x.storage.data,
                                  .generation = 7,
                                  .rows = 1,
                                  .columns = 256},
                    .generation = 7,
                    .path = kg::Ds4Q8VectorPath::kRaw};
  Refused(kg::CheckDs4Q8Vector(d));
  d.weights.raw.bytes = 20ULL * (256 / 32) * 34;
  ASSERT_TRUE(kg::CheckDs4Q8Vector(d));
  auto bad = d;
  --bad.weights.raw.bytes;
  Refused(kg::CheckDs4Q8Vector(bad));
  bad = d;
  bad.path = kg::Ds4Q8VectorPath::kAligned;
  Refused(kg::CheckDs4Q8Vector(bad));
  d.input = Matrix(4, 4, 256, 4);
  d.output = Output(6, 4, 17);
  d.quantized.rows = 4;
  d.quantized.storage.bytes = *kg::Ds4Q81Bytes(4, 256);
  d.weights.raw.bytes = 18ULL * (256 / 32) * 34;
  ASSERT_TRUE(kg::CheckDs4Q8Vector(d));
}

TEST(Dsv4Ds4Product, OwnOutAChargesWholeDummyReadsAndWmmaStoresAtRaggedTail) {
  kg::Ds4OutA d{.weights = Weights(8192, 4096),
                .heads = Matrix(4, 13, 32768, 4),
                .low = Output(6, 13, 8192),
                .rope_table = Write(7, 128ULL * 32 * 8),
                .rope = {.first = 101,
                         .original_context = 131072,
                         .rotary = 64,
                         .base = 10000.0F,
                         .scale = 0.25F,
                         .extension = 1.0F,
                         .inverse = true}};
  d.low.storage.bytes = 16ULL * 8192 * 4;
  ASSERT_TRUE(kg::CheckDs4OutA(d));
  auto bad = d;
  --bad.low.storage.bytes;
  Refused(kg::CheckDs4OutA(bad));
  bad = d;
  bad.low.storage.data = std::bit_cast<void*>(kBase + (6 * kSlot) + 16);
  Refused(kg::CheckDs4OutA(bad));  // WMMA needs 32-byte, not merely 16-byte alignment.
  bad = d;
  --bad.rope_table.bytes;
  Refused(kg::CheckDs4OutA(bad));
  bad = d;
  bad.rope.step = 2;
  Refused(kg::CheckDs4OutA(bad));
  bad = d;
  bad.quantized.rows = 1;  // absence must be a complete empty descriptor
  Refused(kg::CheckDs4OutA(bad));
  const kg::Ds4ProductMatrix low{.storage = Read(6, d.low.storage.bytes),
                                 .rows = 13,
                                 .columns = 8192,
                                 .row_stride = 8192ULL * 4};
  d.quantized = Sidecar(low);
  d.generation = 7;
  ASSERT_TRUE(kg::CheckDs4OutA(d));
  bad = d;
  bad.quantized.storage.data = d.rope_table.data;
  Refused(kg::CheckDs4OutA(bad));
}

TEST(Dsv4Ds4Product, ClaimedCapacitiesCannotWrapThePointerSpace) {
  auto x = Matrix(4, 9, 512, 4);
  auto q = Sidecar(x);
  x.storage.data = std::bit_cast<const void*>(std::numeric_limits<std::uintptr_t>::max() - 15);
  q.source = x.storage.data;
  Refused(kg::CheckDs4D4(x, 7, q));
  x = Matrix(4, 9, 512, 4);
  q = Sidecar(x);
  q.storage.bytes = std::numeric_limits<std::uint64_t>::max();
  Refused(kg::CheckDs4D4(x, 7, q));
}
}  // namespace
