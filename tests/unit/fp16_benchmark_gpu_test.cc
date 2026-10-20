// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Active benchmark adapter: identical selected implementation before and after
// repetition and a real full weight eviction. No legacy numerical oracle.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/paged_weights.h"
#include "expected_error.h"
#include "fp16_runner.h"
#include "paged_node.h"
#include "v1_frozen.h"

namespace {
namespace jb = llmp::benchmarks;
namespace ts = llmp::test_support;
namespace fs = std::filesystem;

std::string Env(const char* name) {
  const auto* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value != nullptr ? value : "";
}

struct Lifetime {
  jb::Fp16Options options;
  ts::PagedNode node{{.slot_bytes = llmp::engine::kSlabSlotBytes}};
  std::string record;
  std::unique_ptr<jb::Fp16Runner> runner;
  std::vector<ts::PagedModel*> entered;
  std::array<std::vector<float>, 3> results;
};

void Exercise(bool heldout) {
  // LLMP_FP16_BENCH_ARTIFACT, else the published v1 FP16 fixture (the
  // store's v1-ids.json "qwen2-fp16").
  auto artifact = Env("LLMP_FP16_BENCH_ARTIFACT");
  if (artifact.empty()) {
    if (const auto id = ts::V1IdNamed("qwen2-fp16"); id) artifact = ts::V1Store() + "/" + *id;
  }
  auto tokens = Env(heldout ? "LLMP_QWEN2_FP16_HELDOUT_IDS" : "LLMP_QWEN2_FP16_TOKENS");
  if (tokens.empty()) {
    tokens = ts::Qwen2OraclePath(heldout ? "heldout-ids.i64le" : "control-tokens.txt");
  }
  if (artifact.empty() || !fs::is_regular_file(tokens))
    GTEST_FAIL() << "missing approved FP16 benchmark inputs: artifact=" << artifact
                 << " tokens=" << tokens;
  auto lifetime = std::make_unique<Lifetime>();
  auto& life = *lifetime;
  life.options.artifact = artifact;
  life.options.trajectory = heldout ? "heldout" : "control";
  life.options.tokens = tokens;
  auto scratch = Env("LLMP_TEST_SCRATCH");
  if (scratch.empty()) scratch = testing::TempDir();
  life.options.out =
      fs::path(scratch) / (heldout ? "fp16-benchmark-heldout" : "fp16-benchmark-control");
  life.runner =
      std::make_unique<jb::Fp16Runner>(life.node, life.options, 0, 0, nullptr, life.record);
  const auto run = [&]() -> ts::Status {
    if (auto setup = life.node.Open(); !setup) return setup;
    life.entered.push_back(life.runner.get());
    if (auto setup = life.runner->Setup(); !setup) return setup;
    if (auto workspace =
            life.node.MapWorkspace(life.runner->activations_needed(),
                                   std::max(life.runner->pool_needed(), ts::kPagedExtent));
        !workspace)
      return workspace;
    if (auto start = life.node.Start(llmp::base::Bytes((heldout ? 16ULL : 8ULL) << 30U)); !start)
      return start;
    if (auto registered = life.runner->Register(); !registered) return registered;
    if (auto bound = life.runner->Bind(); !bound) return bound;
    life.node.Run();
    for (int evaluation = 1; evaluation <= 3; ++evaluation) {
      if (evaluation == 3) {
        const auto weights = life.runner->weights();
        if (weights.empty()) return std::unexpected("adapter has no immutable weight closure");
        const auto backing = life.node.memory().backings();
        for (const auto id : weights) {
          const auto extent = life.node.catalog().Describe(id);
          if (!extent || extent->state != llmp::catalog::ExtentState::kResident ||
              extent->leases != 0 || extent->registrations != 0)
            return std::unexpected("adapter weight remains held before eviction");
        }
        if (auto evicted = life.node.Evict(weights); !evicted) return evicted;
        if (life.node.memory().backings() + weights.size() != backing)
          return std::unexpected("adapter eviction did not release complete weight backing");
        for (const auto id : weights) {
          const auto extent = life.node.catalog().Describe(id);
          if (!extent || extent->state != llmp::catalog::ExtentState::kNonresident)
            return std::unexpected("adapter eviction retained a weight extent");
        }
      }
      if (auto evaluated = life.runner->Evaluate(
              evaluation, life.results[static_cast<std::size_t>(evaluation - 1)]);
          !evaluated)
        return evaluated;
    }
    const auto expected_rows = heldout ? 577U : 76U;
    for (const auto& result : life.results) {
      if (result.size() != std::uint64_t{expected_rows} * 151936)
        return std::unexpected("adapter omitted full vocabulary heads");
      llmp::base::Sha256 hash;
      hash.Update(std::as_bytes(std::span(result)));
      testing::Test::RecordProperty("last_result_sha256", llmp::base::ToHex(hash.Finish()));
      for (std::size_t i = 0; i < result.size(); ++i)
        if (std::bit_cast<std::uint32_t>(result[i]) !=
            std::bit_cast<std::uint32_t>(life.results.front()[i]))
          return std::unexpected("same adapter changed after repetition or weight reload");
    }
    testing::Test::RecordProperty("compared_rows", std::to_string(expected_rows));
    testing::Test::RecordProperty("evaluations", "3");
    testing::Test::RecordProperty("weight_reload", "1");
    testing::Test::RecordProperty("weight_extents", std::to_string(life.runner->weights().size()));
    return {};
  }();
  const auto retired = life.node.TearDown(life.entered);
  if (!retired) {
    // Retain the node, inputs/results, configuration and complete GPU owner.
    (void)lifetime.release();
    FAIL() << "adapter retirement remains unproven: " << retired.error();
  }
  ASSERT_TRUE(run.has_value()) << ts::Failed(run).value_or("");
}

TEST(Fp16BenchmarkGpuTest, SharedControlRepeatsAndReloadsWeights) { Exercise(false); }
TEST(Fp16BenchmarkGpuTest, SharedHeldoutRepeatsAndReloadsWeights) { Exercise(true); }
TEST(Fp16BenchmarkGpuTest, InsufficientHostAllowanceRefusesBeforeConstruction) {
  Lifetime life;
  // These paths must never be read or parsed after the host inventory refuses.
  life.options.artifact = "/absent/adapter-artifact";
  life.options.tokens = "/absent/adapter-tokens";
  life.options.trajectory = "control";
  life.options.host_allowance = (6ULL << 30U) - 1;
  life.runner =
      std::make_unique<jb::Fp16Runner>(life.node, life.options, 0, 0, nullptr, life.record);
  const auto refused = life.runner->Setup();
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error(), "FP16 host allowance is insufficient");
  EXPECT_TRUE(life.runner->managed_extents().empty());
}
}  // namespace
