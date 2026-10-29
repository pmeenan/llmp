// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <string>
#include <utility>

#include "execution/registry.h"
#include "kernels/ggml/measured_ranks.h"

namespace {
namespace ex = llmp::execution;
namespace ir = llmp::ir;
using ex::Registry;
using llmp::format::Encoding;
using llmp::format::Layout;
ex::Implementation Capable(std::string name, ex::Capability capability) {
  return {.name = std::move(name),
          .operation = capability.op,
          .source = "test",
          .revision = "test",
          .build = "test",
          .variant = "test",
          .capability = capability};
}
TEST(GgmlMeasuredRanksTest, DeviceRanksRequireEveryMeasuredFactAndUnbatchedOperands) {
  const auto data = llmp::kernels::ggml::MeasuredRanks();
  ASSERT_GE(data.records.size(), 2U);
  for (const auto& record : data.records) {
    SCOPED_TRACE(record.implementation);
    const bool product = record.op == ir::Primitive(ir::Op::kMatMul);
    std::array operands{
        ex::OperandFormat{.operand = 0,
                          .stacked = record.stacked,
                          .encoding = record.encoding,
                          .layout = record.layout,
                          .params = record.params,
                          .ne = {record.k, record.n, 1, 1}},
        ex::OperandFormat{
            .operand = 1,
            .encoding = product ? Encoding::kF32 : Encoding::kI32,
            .layout = Layout::kDense,
            .ne = {product ? record.k : record.rows_min, product ? record.rows_min : 1, 1, 1}},
        ex::OperandFormat{.operand = ex::kOutputOperand,
                          .encoding = Encoding::kF32,
                          .layout = Layout::kDense,
                          .ne = {product ? record.n : record.k, record.rows_min, 1, 1}},
    };
    ex::CapabilityQuery query{.op = record.op,
                              .operands = operands,
                              .rows = record.rows_min,
                              .k = record.k,
                              .n = record.n,
                              .arch = record.arch,
                              .device_class = record.device_class,
                              .device_model = record.device_model,
                              .multiprocessors = record.multiprocessors};
    ASSERT_EQ(ex::DeviceRank(query, record.implementation, data.records), 0U);
    EXPECT_FALSE(ex::DeviceRank(query, "fake.unmeasured", data.records));
    auto registry =
        Registry::Create({Capable(std::string(record.implementation), {.op = record.op}),
                          Capable("fake.alternative", {.op = record.op, .priority = 100})},
                         ex::RankData{{record}})
            .value();
    const auto unranked =
        Registry::Create({Capable(std::string(record.implementation), {.op = record.op}),
                          Capable("fake.alternative", {.op = record.op, .priority = 100})})
            .value();
    EXPECT_EQ(unranked.at(ex::Candidates(unranked, query).front()).name, "fake.alternative");
    EXPECT_EQ(registry.at(ex::Candidates(registry, query).front()).name, record.implementation);
    const auto original = query;
    const auto original_operands = operands;
    const auto misses = [&] {
      EXPECT_FALSE(ex::DeviceRank(query, record.implementation, data.records));
      query = original;
      operands = original_operands;
    };
    query.device_class = "sm_86";
    misses();
    query.device_model = "another device";
    misses();
    --query.arch;
    misses();
    --query.multiprocessors;
    misses();
    query.op = ir::Primitive(ir::Op::kAdd);
    misses();
    operands[0].encoding = Encoding::kBF16;
    misses();
    operands[0].layout = Layout::kDense;
    misses();
    operands[0].params.tensor_scale = !operands[0].params.tensor_scale;
    misses();
    ++operands[0].params.block_rows;
    misses();
    ++operands[0].params.block_cols;
    misses();
    auto block_rows_record = record;
    ++block_rows_record.params.block_rows;
    const auto block_rows_registry =
        Registry::Create({Capable(std::string(record.implementation), {.op = record.op}),
                          Capable("fake.alternative", {.op = record.op, .priority = 100})},
                         ex::RankData{{block_rows_record}})
            .value();
    EXPECT_NE(block_rows_registry.RankDigest(), registry.RankDigest());
    auto block_cols_record = record;
    ++block_cols_record.params.block_cols;
    const auto block_cols_registry =
        Registry::Create({Capable(std::string(record.implementation), {.op = record.op}),
                          Capable("fake.alternative", {.op = record.op, .priority = 100})},
                         ex::RankData{{block_cols_record}})
            .value();
    EXPECT_NE(block_cols_registry.RankDigest(), registry.RankDigest());
    ++operands[0].stacked;
    misses();
    // Outside the rows' band (the product's input carrying the rows too).
    for (const std::uint64_t rows : {record.rows_min - 1, record.rows_max + 1}) {
      query.rows = rows;
      if (product) operands[1].ne[1] = rows;
      misses();
    }
    query.rows = record.rows_max;
    if (product) operands[1].ne[1] = record.rows_max;
    EXPECT_EQ(ex::DeviceRank(query, record.implementation, data.records), 0U);
    query = original;
    operands = original_operands;
    ++query.k;
    misses();
    ++query.n;
    misses();
    ++operands[0].ne[0];
    misses();
    ++operands[0].ne[1];
    misses();
    ++operands[1].ne[product ? 1 : 0];
    misses();
    for (auto& operand : operands) {
      for (const std::size_t dimension : {2U, 3U}) {
        operand.ne[dimension] = 2;
        misses();
      }
    }
    query.operands = {};
    misses();
    query.operands = std::span(operands).first(1);
    misses();
    const std::array override{ex::MeasuredRank{.implementation = "fake.alternative", .rank = 0}};
    query.measured = override;
    EXPECT_EQ(registry.at(ex::Candidates(registry, query).front()).name, "fake.alternative");
  }
}

}  // namespace
