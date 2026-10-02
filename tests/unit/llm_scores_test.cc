// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Llm's real teacher-forcing and generation loops over a completed fake
// target/drafter, without a device. These controls test position, state,
// callback alignment and retirement, independently of HTTP serialization.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine/qwen38_wave_plan.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "runtime/serving.h"

namespace {

namespace rt = jitllm::runtime;
namespace engine = jitllm::engine;
namespace catalog = jitllm::catalog;
using ::testing::ElementsAre;

TEST(Qwen38WaveHead, AdaptiveRowsRequireContiguousFullBf16Geometry) {
  auto arena = jitllm::kernels::ggml::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* context = arena->context();
  auto* weight = ggml_new_tensor_2d(context, GGML_TYPE_BF16, 2560, 248320);
  for (const std::int64_t rows : {1, 2, 3, 4, 5}) {
    auto* input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2560, rows);
    auto* output = ggml_mul_mat(context, weight, input);
    const bool supported = rows == 3 || rows == 4;
    EXPECT_EQ(engine::Qwen38FullHeadPairCandidate(output), supported);
    if (!supported) {
      continue;
    }
    auto bad = *output;
    ++bad.ne[1];
    EXPECT_FALSE(engine::Qwen38FullHeadPairCandidate(&bad));
    auto strided = *input;
    strided.nb[1] += sizeof(float);
    bad = *output;
    bad.src[1] = &strided;
    EXPECT_FALSE(engine::Qwen38FullHeadPairCandidate(&bad));
    auto quantized = *weight;
    quantized.type = GGML_TYPE_Q8_0;
    bad = *output;
    bad.src[0] = &quantized;
    EXPECT_FALSE(engine::Qwen38FullHeadPairCandidate(&bad));
  }
  EXPECT_FALSE(engine::Qwen38FullHeadPairCandidate(nullptr));
}

TEST(Qwen38WaveHc, OnlyContiguousMultirowBf16ProductsAreCandidates) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = kg::TensorArena::Create(48);
  ASSERT_TRUE(arena);
  auto* context = arena->context();
  auto* weight = ggml_new_tensor_2d(context, GGML_TYPE_BF16, 2560, 128);
  for (const std::int64_t rows : {1, 2, 3, 4, 5}) {
    auto* input = ggml_new_tensor_2d(context, GGML_TYPE_BF16, 2560, rows);
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
      auto* output = kg::GemvBf16(context, weight, input, type);
      const bool supported = rows >= 2 && rows <= 4;
      EXPECT_EQ(engine::Qwen38HcPairCandidate(output), supported);
      if (!supported) {
        continue;
      }
      auto bad = *output;
      ++bad.ne[1];
      EXPECT_FALSE(engine::Qwen38HcPairCandidate(&bad));
      auto strided = *input;
      strided.nb[1] += sizeof(ggml_bf16_t);
      bad = *output;
      bad.src[1] = &strided;
      EXPECT_FALSE(engine::Qwen38HcPairCandidate(&bad));
      auto f32_input = *input;
      f32_input.type = GGML_TYPE_F32;
      bad = *output;
      bad.src[1] = &f32_input;
      EXPECT_FALSE(engine::Qwen38HcPairCandidate(&bad));
      auto quantized = *weight;
      quantized.type = GGML_TYPE_Q8_0;
      bad = *output;
      bad.src[0] = &quantized;
      EXPECT_FALSE(engine::Qwen38HcPairCandidate(&bad));
    }
  }
  EXPECT_FALSE(engine::Qwen38HcPairCandidate(nullptr));
}

class FakePaged final : public engine::PagedModel {
 public:
  std::uint32_t stream() const override { return 0; }
  const catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<catalog::ExtentId> managed_extents() const override { return {}; }
  engine::Status Release() override { return {}; }

 private:
  catalog::Closure closure_;
};

class FakeLlm : public rt::Llm {
  friend class NativeBranchesFake;

 public:
  explicit FakeLlm(bool speculative = false) {
    name_ = "fake";
    context_ = 32;
    max_rows_ = 8;
    speculate_ = speculative;
    stops_ = {7};
  }
  engine::PagedModel& paged() override { return paged_; }
  rt::Status Setup() override { return {}; }
  std::uint64_t activations_needed() const override { return 0; }
  std::uint64_t pool_needed() const override { return 0; }
  rt::Status Register() override { return {}; }
  rt::Status Bind() override { return {}; }
  std::vector<catalog::ExtentId> weights() const override { return {}; }
  std::vector<catalog::ExtentId> state() const override { return {}; }
  const catalog::Closure& everything() const override { return paged_.fence_closure(); }
  std::uint64_t weight_read_bytes() const override { return 0; }
  void Defaults(jitllm::chat::Conversation& /*conversation*/) const override {}
  void ConfigurePrefill(std::uint32_t context, std::uint32_t rows) {
    context_ = context;
    max_rows_ = rows;
  }

  static std::vector<float> Row(std::int32_t next) {
    std::vector<float> row(8, -2.0F);
    row[static_cast<std::size_t>(next)] = 3.0F;
    return row;
  }
  std::vector<std::int32_t> target;
  std::vector<std::int32_t> injection;
  unsigned chunks = 0;
  unsigned settlements = 0;
  unsigned clearings = 0;
  std::optional<unsigned> fail_chunk;
  bool usable = true;
  bool fail_settle = false;
  bool fail_clear = false;
  bool fail_prepare = false;
  bool pending = false;
  bool omit_row = false;
  std::vector<std::int32_t> step = {2, 3, 4};

 protected:
  rt::Status RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                      std::vector<float>& row) override {
    EXPECT_EQ(past, target.size());
    ++chunks;
    if (fail_chunk == chunks) {
      usable = false;  // an unknown submission cannot establish a prefix
      return std::unexpected("unproven fake completion");
    }
    const auto fresh = all.subspan(past);
    target.insert(target.end(), fresh.begin(), fresh.end());
    if (inject) {
      injection.insert(injection.end(), fresh.begin(), fresh.end());
    }
    row = Row((all.back() + 1) % 8);
    return {};
  }
  rt::Status SpecStep(std::span<const std::int32_t> all, std::uint32_t past, std::uint32_t left,
                      std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                      std::uint64_t& drafted) override {
    EXPECT_EQ(past, target.size());
    const auto count = std::min<std::size_t>(left, step.size());
    kept.assign(step.begin(), step.begin() + static_cast<std::ptrdiff_t>(count));
    target.push_back(all.back());
    target.insert(target.end(), kept.begin(), kept.end() - 1);
    if (logits != nullptr) {
      for (const auto token : kept) {
        logits->push_back(Row(token));
      }
      if (omit_row) {
        logits->pop_back();
      }
    }
    drafted += kept.size() - 1;
    pending = true;
    return {};
  }
  rt::Status Settle() override {
    ++settlements;
    if (fail_settle) {
      usable = false;
      return std::unexpected("unproven fake settlement");
    }
    pending = false;
    return {};
  }
  rt::Status ClearState() override {
    ++clearings;
    if (fail_clear) {
      return std::unexpected("fake clear refused before dispatch");
    }
    target.clear();
    injection.clear();
    usable = true;
    pending = false;
    return {};
  }
  bool StateUsable() const override { return usable; }
  rt::Status PrepareDecodeState(std::uint32_t /*pos*/, std::uint32_t /*left*/) override {
    if (fail_prepare) {
      return std::unexpected("fake decode preparation failed");
    }
    return {};
  }
  std::uint64_t target_state_base() const override { return 0; }
  std::uint64_t target_state_bytes() const override { return 0; }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return 0; }
  std::vector<engine::LiveState::Range> used_state_ranges() const override { return {}; }
  rt::Status SaveUsedState(void* /*host*/,
                           std::span<const engine::LiveState::Range> /*ranges*/) override {
    return {};
  }
  rt::Status RestoreUsedState(void* /*host*/,
                              std::span<const engine::LiveState::Range> /*ranges*/) override {
    return {};
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t /*positions*/) const override {
    return std::vector<engine::LiveState::Range>{};
  }
  rt::Status PrepareRestoreState(std::span<const engine::LiveState::Range> /*footprint*/) override {
    return {};
  }
  rt::Status CopyCheckpointState(void* /*host*/,
                                 std::span<const engine::LiveState::Range> /*ranges*/,
                                 bool /*to_host*/) override {
    return {};
  }

 private:
  FakePaged paged_;
};

class NativeBranchesFake final : public FakeLlm {
 public:
  explicit NativeBranchesFake(bool speculative = false, std::size_t wave_capacity = 4)
      : FakeLlm(speculative), wave_capacity_(wave_capacity) {
    for (auto& leaf : leaves_) {
      leaf = std::make_unique<FakeLlm>(speculative);
    }
    EXPECT_TRUE(PrepareBranches(4, 3).has_value());
  }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override { return wave_capacity_; }
  std::optional<std::uint32_t> nonfinite_wave_row;
  std::optional<std::uint32_t> failed_wave_judgement;
  FakeLlm& native_state(std::size_t slot) { return slot == 0 ? *this : *leaves_.at(slot - 1); }
  const auto& selected() const { return selected_; }
  jitllm::execution::AdaptiveDepth& policy(Branch& branch) { return BranchDecoding(branch); }
  std::uint32_t& pending_cursor(Branch& branch) { return cursors_[BranchIndex(branch)]; }
  rt::Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > kMaxBranches) {
      return std::unexpected("too many fake branches");
    }
    std::array<bool, kMaxBranches> selected{};
    for (const auto* branch : active) {
      if (branch == nullptr || &branch->model() != this) {
        return std::unexpected("foreign fake branch");
      }
      const auto slot = BranchIndex(*branch);
      if (selected[slot]) {
        return std::unexpected("duplicate fake branch");
      }
      selected[slot] = true;
    }
    selected_ = selected;
    return {};
  }

 protected:
  rt::Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    for (PreparedGeneration& unit : prepared) {
      const auto slot = BranchIndex(*unit.branch);
      if (failed_wave_judgement == slot) {
        // Fake completed verify/discard: the exact prior prefix is restored,
        // so its independent host error can preserve history at Step::position.
        unit.result = std::unexpected("fake completed judgement failed");
        unit.failed_prefix_valid = true;
        continue;
      }
      auto ran =
          unit.step.speculative
              ? SpecStepFor(*unit.branch, unit.step.all, unit.step.position, unit.step.left,
                            unit.kept, unit.step.need_logits ? &unit.logits : nullptr, unit.drafted)
              : RunChunkFor(*unit.branch, unit.step.all, unit.step.position, false, unit.row);
      if (!ran) {
        return ran;
      }
      if (nonfinite_wave_row == slot) {
        std::ranges::fill(unit.row, std::numeric_limits<float>::quiet_NaN());
      }
    }
    return {};
  }
  rt::Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                         bool inject, std::vector<float>& logits) override {
    return Native(branch).FakeLlm::RunChunk(all, n_past, inject, logits);
  }
  rt::Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                         std::uint32_t left, std::vector<std::int32_t>& kept,
                         std::vector<std::vector<float>>* logits, std::uint64_t& drafted) override {
    return Native(branch).FakeLlm::SpecStep(all, pos, left, kept, logits, drafted);
  }
  rt::Status SettleFor(Branch& branch) override { return Native(branch).FakeLlm::Settle(); }
  rt::Status ClearStateFor(Branch& branch) override { return Native(branch).FakeLlm::ClearState(); }
  bool StateUsableFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::StateUsable();
  }
  rt::Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override {
    return Native(branch).FakeLlm::PrepareDecodeState(pos, left);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::target_state_base();
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::target_state_bytes();
  }
  std::uint64_t DrafterStateBaseFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::drafter_state_base();
  }
  std::uint64_t DrafterStateBytesFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::drafter_state_bytes();
  }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::used_state_ranges();
  }
  rt::Status SaveUsedStateFor(Branch& branch, void* host,
                              std::span<const engine::LiveState::Range> ranges) override {
    auto saved = Native(branch).FakeLlm::SaveUsedState(host, ranges);
    if (saved) {
      saved_native_[BranchIndex(branch)] = {Native(branch).target, Native(branch).injection};
    }
    return saved;
  }
  rt::Status RestoreUsedStateFor(Branch& branch, void* host,
                                 std::span<const engine::LiveState::Range> ranges) override {
    auto restored = Native(branch).FakeLlm::RestoreUsedState(host, ranges);
    if (restored) {
      const auto& saved = saved_native_[BranchIndex(branch)];
      Native(branch).target = saved[0];
      Native(branch).injection = saved[1];
    }
    return restored;
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const override {
    return Native(branch).FakeLlm::CheckpointRanges(positions);
  }
  rt::Status PrepareRestoreStateFor(Branch& branch,
                                    std::span<const engine::LiveState::Range> footprint) override {
    return Native(branch).FakeLlm::PrepareRestoreState(footprint);
  }
  rt::Status CopyCheckpointStateFor(Branch& branch, void* host,
                                    std::span<const engine::LiveState::Range> ranges,
                                    bool to_host) override {
    return Native(branch).FakeLlm::CopyCheckpointState(host, ranges, to_host);
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return cursors_[BranchIndex(branch)];
  }
  void SetCursorFor(Branch& branch, std::uint32_t value) override {
    cursors_[BranchIndex(branch)] = value;
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  jitllm::execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch,
                                   const jitllm::execution::AdaptiveDepth& decoding) override {
    BranchDecoding(branch) = decoding;
  }

 private:
  const std::size_t wave_capacity_;
  FakeLlm& Native(Branch& branch) { return native_state(BranchIndex(branch)); }
  const FakeLlm& Native(const Branch& branch) const {
    const auto slot = BranchIndex(branch);
    return slot == 0 ? static_cast<const FakeLlm&>(*this) : *leaves_[slot - 1];
  }
  std::array<std::unique_ptr<FakeLlm>, kMaxBranches - 1> leaves_;
  std::array<std::array<std::vector<std::int32_t>, 2>, kMaxBranches> saved_native_{};
  std::array<std::uint32_t, kMaxBranches> cursors_{};
  std::array<bool, kMaxBranches> selected_{};
};

TEST(LlmScores, TeacherForcesStopIdsAndKeepsAllInjectionState) {
  FakeLlm model(true);
  const std::array<std::int32_t, 4> prompt = {0, 7, 2, 3};
  std::vector<float> last;
  std::vector<std::int32_t> scored;
  std::vector<std::uint32_t> sizes;
  rt::PrefillRun run;
  ASSERT_TRUE(model
                  .ScorePrompt(
                      prompt, last,
                      [&](std::int32_t id, std::span<const float> row) {
                        scored.push_back(id);
                        EXPECT_EQ(std::vector<float>(row.begin(), row.end()),
                                  FakeLlm::Row((prompt[scored.size() - 1] + 1) % 8));
                        return true;
                      },
                      [&](std::uint32_t rows) {
                        sizes.push_back(rows);
                        return true;
                      },
                      &run)
                  .has_value());
  EXPECT_THAT(scored, ElementsAre(7, 2, 3));
  EXPECT_THAT(sizes, ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(model.history(), ElementsAre(0, 7, 2, 3));
  EXPECT_EQ(model.target, model.injection);
  EXPECT_EQ(model.target, model.history());
  EXPECT_EQ(last, FakeLlm::Row(4));
  EXPECT_EQ(run.end, 4U);
  EXPECT_FALSE(run.stopped);
  EXPECT_EQ(model.settlements, 1U);
}

TEST(LlmScores, CancellationRetainsOnlyCompletedPrefixAndCanContinue) {
  FakeLlm model(true);
  const std::array<std::int32_t, 4> prompt = {0, 1, 2, 3};
  std::vector<float> last;
  rt::PrefillRun run;
  ASSERT_TRUE(
      model
          .ScorePrompt(
              prompt, last, {}, [&](std::uint32_t /*rows*/) { return model.chunks < 2; }, &run)
          .has_value());
  EXPECT_TRUE(run.stopped);
  EXPECT_TRUE(last.empty());
  EXPECT_THAT(model.history(), ElementsAre(0, 1));
  EXPECT_EQ(model.target, model.injection);
  ASSERT_TRUE(model.Prefill(std::span(prompt).subspan(2), last).has_value());
  EXPECT_THAT(model.history(), ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(model.target, model.injection);
  EXPECT_FALSE(model.ScorePrompt(prompt, last, {}).has_value());
}

TEST(LlmScores, UnknownCompletionInvalidatesHistoryBeforeFreshUse) {
  FakeLlm model;
  model.fail_chunk = 2;
  std::vector<float> last;
  const std::array<std::int32_t, 3> prompt = {0, 1, 2};
  EXPECT_FALSE(model.ScorePrompt(prompt, last, {}).has_value());
  EXPECT_TRUE(model.history().empty());
  EXPECT_TRUE(last.empty());
  model.fail_chunk.reset();
  ASSERT_TRUE(model.ScorePrompt(prompt, last, {}).has_value());
  EXPECT_EQ(model.clearings, 1U);
  EXPECT_THAT(model.target, ElementsAre(0, 1, 2));
}

TEST(LlmScores, GeneratedCallbacksMatchRowsAndStopWithinCompletedVerify) {
  FakeLlm model(true);
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  std::vector<std::int32_t> seen;
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.keep_logits = true;
  options.on_logits = [&](std::int32_t token, std::span<const float> row) {
    seen.push_back(token);
    EXPECT_EQ(std::vector<float>(row.begin(), row.end()), FakeLlm::Row(token));
    return true;
  };
  options.on_tokens = [](std::span<const std::int32_t> tokens) {
    return tokens.empty() || tokens.back() != 2;  // like a stop-string match
  };
  rt::Generation result;
  ASSERT_TRUE(model.Generate(last, options, result).has_value());
  EXPECT_THAT(result.tokens, ElementsAre(1, 2));
  EXPECT_THAT(seen, ElementsAre(1, 2));
  EXPECT_EQ(result.logits.size(), 2U);
  EXPECT_TRUE(result.cancelled);
  EXPECT_THAT(model.history(), ElementsAre(0, 1, 2, 3));  // full accepted verify retired
  EXPECT_EQ(model.target, model.history());
  EXPECT_FALSE(model.pending);
}

TEST(LlmScores, GeneratedStopHasScoreButNoVisibleTokenAndBudgetHasNoExtraRow) {
  FakeLlm model;
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {6};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 1;
  unsigned rows = 0;
  options.on_logits = [&](std::int32_t id, std::span<const float> row) {
    EXPECT_EQ(id, 7);
    EXPECT_EQ(row[7], 3.0F);
    ++rows;
    return true;
  };
  options.on_tokens = [](std::span<const std::int32_t> tokens) {
    EXPECT_TRUE(tokens.empty());
    return true;
  };
  rt::Generation result;
  ASSERT_TRUE(model.Generate(last, options, result).has_value());
  EXPECT_TRUE(result.stopped);
  EXPECT_THAT(result.tokens, ElementsAre(7));
  EXPECT_EQ(rows, 1U);
  EXPECT_EQ(model.chunks, 1U);
  options.max_tokens = 0;
  EXPECT_FALSE(model.Generate(last, options, result).has_value());
}

TEST(LlmScores, ScoreCallbackAndSettlementFailuresDoNotSubmitAnotherStep) {
  FakeLlm model;
  std::vector<float> last;
  const std::array<std::int32_t, 3> prompt = {0, 1, 2};
  rt::PrefillRun run;
  ASSERT_TRUE(model
                  .ScorePrompt(
                      prompt, last,
                      [](std::int32_t /*id*/, std::span<const float> /*row*/) { return false; }, {},
                      &run)
                  .has_value());
  EXPECT_TRUE(run.stopped);
  EXPECT_EQ(model.chunks, 1U);
  EXPECT_THAT(model.history(), ElementsAre(0));
  ASSERT_TRUE(model.Clear().has_value());
  model.fail_settle = true;
  EXPECT_FALSE(model.ScorePrompt(prompt, last, {}).has_value());
  EXPECT_TRUE(model.history().empty());
  EXPECT_TRUE(last.empty());
}

TEST(LlmScores, InconsistentVerifyRowsRefuseBeforeIndexingAndRetire) {
  FakeLlm model(true);
  model.omit_row = true;
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.keep_logits = true;
  rt::Generation result;
  EXPECT_FALSE(model.Generate(last, options, result).has_value());
  EXPECT_FALSE(model.pending);
  EXPECT_EQ(model.settlements, 1U);
  EXPECT_TRUE(model.history().empty());
}

TEST(LlmScores, InconsistentVerifyRowsPreserveUnprovenSettlementFailure) {
  FakeLlm model(true);
  model.omit_row = true;
  model.fail_settle = true;
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.keep_logits = true;
  rt::Generation result;
  const auto ran = model.Generate(last, options, result);
  ASSERT_FALSE(ran.has_value());
  EXPECT_THAT(ran.error(), ::testing::HasSubstr("inconsistent token/logit counts"));
  EXPECT_THAT(ran.error(), ::testing::HasSubstr("unproven fake settlement"));
  EXPECT_TRUE(model.pending);
  EXPECT_FALSE(model.usable);
  EXPECT_EQ(model.settlements, 1U);
  EXPECT_TRUE(model.history().empty());
}

TEST(LlmScores, DecodeFailurePreservesItsUnprovenSettlementFailure) {
  FakeLlm model;
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  model.fail_prepare = true;
  model.fail_settle = true;
  rt::GenerateOptions options;
  options.max_tokens = 2;
  rt::Generation result;
  const auto ran = model.Generate(last, options, result);
  ASSERT_FALSE(ran.has_value());
  EXPECT_THAT(ran.error(), ::testing::HasSubstr("fake decode preparation failed"));
  EXPECT_THAT(ran.error(), ::testing::HasSubstr("unproven fake settlement"));
  EXPECT_FALSE(model.usable);
  EXPECT_EQ(model.settlements, 1U);
  EXPECT_TRUE(model.history().empty());
}

TEST(LlmScores, OrdinaryKeepLogitsAndStoppedAcceptedTrajectoryStayAligned) {
  FakeLlm ordinary;
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(ordinary.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.keep_logits = true;
  rt::Generation baseline;
  ASSERT_TRUE(ordinary.Generate(last, options, baseline).has_value());
  EXPECT_THAT(baseline.tokens, ElementsAre(1, 2, 3, 4));
  ASSERT_EQ(baseline.logits.size(), baseline.tokens.size());
  for (std::size_t i = 0; i < baseline.tokens.size(); ++i) {
    EXPECT_EQ(baseline.logits[i], FakeLlm::Row(baseline.tokens[i]));
  }
  FakeLlm speculative(true);
  speculative.step = {2, 7, 4};
  ASSERT_TRUE(speculative.Prefill(prompt, last).has_value());
  std::vector<std::int32_t> seen;
  options.on_logits = [&](std::int32_t id, std::span<const float> /*row*/) {
    seen.push_back(id);
    return true;
  };
  rt::Generation stopped;
  ASSERT_TRUE(speculative.Generate(last, options, stopped).has_value());
  EXPECT_THAT(stopped.tokens, ElementsAre(1, 2, 7));
  EXPECT_THAT(seen, ElementsAre(1, 2, 7));
  EXPECT_EQ(stopped.logits.size(), 3U);
  EXPECT_TRUE(stopped.stopped);
  EXPECT_THAT(speculative.history(), ElementsAre(0, 1, 2, 7));
  EXPECT_EQ(speculative.target, speculative.history());
  EXPECT_FALSE(speculative.pending);
}

TEST(LlmScores, ResumablePlainAppliesCompletedRowsAndRetainsTheUnprocessedAnchor) {
  FakeLlm model;
  auto& branch = model.default_branch();
  EXPECT_EQ(&branch.model(), &model);
  EXPECT_EQ(&branch.history(), &model.history());
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(branch.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.keep_logits = true;
  std::vector<unsigned> calls;
  options.on_tokens = [&, count = 0U](std::span<const std::int32_t> /*tokens*/) mutable {
    calls.push_back(++count);
    return true;
  };
  rt::Generation result;
  auto opened = branch.BeginGeneration(last, options, result);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  EXPECT_THAT(result.tokens, ElementsAre(1));
  EXPECT_TRUE(model.HasRetainedState());
  rt::Generation refused;
  EXPECT_FALSE(model.BeginGeneration(last, options, refused).has_value());
  auto step = session.PrepareStep();
  ASSERT_TRUE(step.has_value());
  EXPECT_THAT(step->all, ElementsAre(0, 1));
  EXPECT_EQ(step->position, 1U);
  EXPECT_EQ(step->left, 2U);
  EXPECT_FALSE(step->speculative);
  EXPECT_TRUE(step->need_logits);
  // A completed external native job processed the anchor and returned its
  // row. Applying that result must not dispatch another chunk.
  model.target.push_back(step->all.back());
  ASSERT_TRUE(session.ApplyPlain(FakeLlm::Row(2)).has_value());
  EXPECT_EQ(model.chunks, 1U);
  EXPECT_FALSE(session.done());
  session.Cancel();
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_TRUE(result.cancelled);
  EXPECT_THAT(result.tokens, ElementsAre(1, 2));
  EXPECT_THAT(model.history(), ElementsAre(0, 1));
  EXPECT_EQ(model.target, model.history());
  EXPECT_EQ(result.logits.size(), 2U);
  EXPECT_EQ(model.settlements, 1U);

  rt::Generation next;
  auto again = model.BeginGeneration(FakeLlm::Row(2), options, next);
  ASSERT_TRUE(again.has_value());
  // Finishing an old session twice cannot release the new session's guard.
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_FALSE(branch.BeginGeneration(last, options, refused).has_value());
  EXPECT_THAT(calls, ElementsAre(1U, 2U, 3U));  // the original mutable callback
  (*again)->Cancel();
  ASSERT_TRUE((*again)->Finish().has_value());
  EXPECT_EQ(model.settlements, 2U);
  branch.Forget();
  EXPECT_FALSE(model.HasRetainedState());
  EXPECT_FALSE(branch.HasRetainedState());
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  EXPECT_EQ(model.clearings, 1U);
  EXPECT_EQ(branch.history(), model.target);
}

TEST(LlmScores, ResumableVerifySettlesTheFullAcceptedPrefixAfterVisibleStop) {
  FakeLlm model(true);
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.keep_logits = true;
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> visible;
  options.on_logits = [&](std::int32_t token, std::span<const float> row) {
    rows.push_back(token);
    EXPECT_EQ(std::vector<float>(row.begin(), row.end()), FakeLlm::Row(token));
    return true;
  };
  options.on_tokens = [&](std::span<const std::int32_t> fresh) {
    visible.insert(visible.end(), fresh.begin(), fresh.end());
    return true;
  };
  rt::Generation result;
  auto opened = model.BeginGeneration(last, options, result);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  auto step = session.PrepareStep();
  ASSERT_TRUE(step.has_value());
  EXPECT_TRUE(step->speculative);
  model.target = {0, 1, 2, 7};  // the anchor and accepted drafts, 4 still unprocessed
  model.pending = true;
  ASSERT_TRUE(
      session.ApplySpeculative({2, 7, 4}, {FakeLlm::Row(2), FakeLlm::Row(7), FakeLlm::Row(4)}, 2)
          .has_value());
  EXPECT_TRUE(session.done());
  EXPECT_TRUE(model.pending);
  EXPECT_THAT(result.tokens, ElementsAre(1, 2, 7));
  EXPECT_THAT(rows, ElementsAre(1, 2, 7));
  EXPECT_THAT(visible, ElementsAre(1, 2));
  EXPECT_EQ(result.drafted, 2U);
  EXPECT_EQ(result.accepted, 2U);
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_FALSE(model.pending);
  EXPECT_EQ(model.history(), model.target);
  EXPECT_EQ(model.settlements, 1U);
}

TEST(LlmScores, ResumableSamplingUsesAbsolutePositionKeysForCompletedRows) {
  FakeLlm model;
  std::vector<float> ignored;
  const std::array<std::int32_t, 3> prompt = {0, 1, 2};
  ASSERT_TRUE(model.Prefill(prompt, ignored).has_value());
  const std::vector<float> row(8, 0.0F);
  rt::GenerateOptions options;
  options.max_tokens = 2;
  options.stop = false;
  options.sampling =
      jitllm::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
  options.seed = 731;
  std::vector<jitllm::execution::SamplingCandidate> scratch;
  const auto first = jitllm::execution::Sample(
      row, *options.sampling, {.seed = options.seed, .stream = 0, .position = 3}, scratch);
  const auto second = jitllm::execution::Sample(
      row, *options.sampling, {.seed = options.seed, .stream = 0, .position = 4}, scratch);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  rt::Generation result;
  auto opened = model.BeginGeneration(row, options, result);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  auto step = session.PrepareStep();
  ASSERT_TRUE(step.has_value());
  model.target.push_back(step->all.back());
  ASSERT_TRUE(session.ApplyPlain(row).has_value());
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_THAT(result.tokens, ElementsAre(*first, *second));
  EXPECT_EQ(model.history(), model.target);
}

TEST(LlmScores, ResumableEarlyAndCompletedFailuresPreserveOnlyProvenHistory) {
  FakeLlm model;
  std::vector<float> last;
  const std::array<std::int32_t, 1> prompt = {0};
  ASSERT_TRUE(model.Prefill(prompt, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 2;
  options.stop = false;
  options.sampling =
      jitllm::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
  auto bad = last;
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  rt::Generation result;
  result.tokens = {6};
  EXPECT_FALSE(model.BeginGeneration(bad, options, result).has_value());
  EXPECT_THAT(result.tokens, ElementsAre(6));
  EXPECT_EQ(model.history(), model.target);
  auto opened = model.BeginGeneration(last, options, result);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  auto step = session.PrepareStep();
  ASSERT_TRUE(step.has_value());
  model.target.push_back(step->all.back());
  EXPECT_FALSE(session.ApplyPlain(bad).has_value());
  EXPECT_FALSE(session.Finish().has_value());
  EXPECT_THAT(model.history(), ElementsAre(0, 1));
  EXPECT_EQ(model.history(), model.target);
  EXPECT_EQ(model.settlements, 1U);
  EXPECT_TRUE(model.HasRetainedState());

  rt::Generation uncertain;
  auto again = model.BeginGeneration(FakeLlm::Row(2), options, uncertain);
  ASSERT_TRUE(again.has_value());
  ASSERT_TRUE((*again)->PrepareStep().has_value());
  model.usable = false;
  EXPECT_FALSE((*again)->FailStep("unproven external completion").has_value());
  EXPECT_FALSE((*again)->Finish().has_value());
  EXPECT_TRUE(model.history().empty());
  EXPECT_FALSE(model.HasRetainedState());
  EXPECT_EQ(model.settlements, 1U);  // uncertainty was not made safe by a destructor
}

TEST(LlmScores, NativeBranchesOwnTheirHistoriesAndRejectForeignSelection) {
  NativeBranchesFake model;
  FakeLlm serial;
  EXPECT_EQ(serial.branches(), 1U);
  EXPECT_FALSE(serial.branch(1).has_value());
  EXPECT_FALSE(model.branch(4).has_value());
  std::array<rt::Llm::Branch*, 4> branches{};
  std::array<std::vector<float>, 4> rows;
  for (std::size_t slot = 0; slot < branches.size(); ++slot) {
    auto branch = model.branch(slot);
    ASSERT_TRUE(branch.has_value());
    branches[slot] = *branch;
    EXPECT_EQ(&(*branch)->model(), &model);
    auto again = model.branch(slot);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(*again, *branch);
    const std::array<std::int32_t, 2> prompt = {static_cast<std::int32_t>(slot), 0};
    ASSERT_TRUE((*branch)->Prefill(prompt, rows[slot]).has_value());
    EXPECT_EQ((*branch)->history(), model.native_state(slot).target);
  }
  EXPECT_EQ(branches[0], &model.default_branch());
  ASSERT_TRUE(model.SelectBranches(branches).has_value());
  const auto original = model.selected();
  const std::array<rt::Llm::Branch*, 2> duplicates = {branches[0], branches[0]};
  EXPECT_FALSE(model.SelectBranches(duplicates).has_value());
  const std::array<rt::Llm::Branch*, 1> foreign = {&serial.default_branch()};
  EXPECT_FALSE(model.SelectBranches(foreign).has_value());
  const std::array<rt::Llm::Branch*, 1> null = {nullptr};
  EXPECT_FALSE(model.SelectBranches(null).has_value());
  EXPECT_EQ(model.selected(), original);
  ASSERT_TRUE(branches[1]->Clear().has_value());
  EXPECT_TRUE(branches[1]->history().empty());
  EXPECT_TRUE(model.native_state(1).target.empty());
  EXPECT_EQ(model.native_state(1).clearings, 1U);
  for (std::size_t slot : {0U, 2U, 3U}) {
    EXPECT_EQ(branches[slot]->history(), model.native_state(slot).target);
    EXPECT_EQ(model.native_state(slot).target.size(), 2U);
    EXPECT_EQ(model.native_state(slot).clearings, 0U);
  }
  branches[0]->Forget();
  EXPECT_TRUE(model.HasRetainedState());
  branches[2]->Forget();
  branches[3]->Forget();
  EXPECT_FALSE(model.HasRetainedState());
}

TEST(LlmScores, FourResumableBranchesInterleaveSamplingWithoutSharingKeysOrGuards) {
  NativeBranchesFake model;
  std::array<rt::Llm::Branch*, 4> branches{};
  std::array<rt::GenerateOptions, 4> options;
  std::array<rt::Generation, 4> results;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 4> sessions;
  std::array<std::array<std::int32_t, 3>, 4> expected{};
  const std::vector<float> row(8, 0.0F);
  for (std::size_t slot = 0; slot < branches.size(); ++slot) {
    auto branch = model.branch(slot);
    ASSERT_TRUE(branch.has_value());
    branches[slot] = *branch;
    const std::vector<std::int32_t> prompt(slot + 1, static_cast<std::int32_t>(slot));
    std::vector<float> ignored;
    ASSERT_TRUE((*branch)->Prefill(prompt, ignored).has_value());
    auto& option = options[slot];
    option.max_tokens = 3;
    option.stop = false;
    option.sampling =
        jitllm::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
    option.seed = 731 + slot;
    std::vector<jitllm::execution::SamplingCandidate> scratch;
    for (std::size_t generated = 0; generated < 3; ++generated) {
      auto token = jitllm::execution::Sample(
          row, *option.sampling,
          {.seed = option.seed, .stream = 0, .position = prompt.size() + generated}, scratch);
      ASSERT_TRUE(token.has_value());
      expected[slot][generated] = *token;
    }
    auto opened = (*branch)->BeginGeneration(row, option, results[slot]);
    ASSERT_TRUE(opened.has_value());
    sessions[slot] = std::move(*opened);
    rt::Generation refused;
    EXPECT_FALSE((*branch)->BeginGeneration(row, option, refused).has_value());
  }
  // Change execution order every wave; keys depend only on each branch's
  // seed and completed position. Cancel one at a completed boundary.
  for (std::size_t slot : {3U, 0U, 2U, 1U}) {
    auto step = sessions[slot]->PrepareStep();
    ASSERT_TRUE(step.has_value());
    model.native_state(slot).target.push_back(step->all.back());
    ASSERT_TRUE(sessions[slot]->ApplyPlain(row).has_value());
  }
  sessions[1]->Cancel();
  ASSERT_TRUE(sessions[1]->Finish().has_value());
  for (std::size_t slot : {2U, 3U, 0U}) {
    auto step = sessions[slot]->PrepareStep();
    ASSERT_TRUE(step.has_value());
    model.native_state(slot).target.push_back(step->all.back());
    ASSERT_TRUE(sessions[slot]->ApplyPlain(row).has_value());
    ASSERT_TRUE(sessions[slot]->Finish().has_value());
  }
  for (std::size_t slot = 0; slot < branches.size(); ++slot) {
    const auto count = slot == 1 ? 2U : 3U;
    EXPECT_EQ(results[slot].tokens,
              std::vector<std::int32_t>(expected[slot].begin(), expected[slot].begin() + count));
    EXPECT_EQ(results[slot].cancelled, slot == 1);
    EXPECT_EQ(branches[slot]->history(), model.native_state(slot).target);
    EXPECT_EQ(model.native_state(slot).settlements, 1U);
  }
}

TEST(LlmScores, BranchSnapshotsRestoreTheirOwnCursorAndAdaptivePolicy) {
  NativeBranchesFake model(true);
  auto a = model.branch(1);
  auto b = model.branch(2);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  std::vector<float> ignored;
  const std::array<std::int32_t, 2> first = {0, 1};
  const std::array<std::int32_t, 3> second = {2, 3, 4};
  ASSERT_TRUE((*a)->Prefill(first, ignored).has_value());
  ASSERT_TRUE((*b)->Prefill(second, ignored).has_value());
  model.pending_cursor(**a) = 2;
  model.pending_cursor(**b) = 4;
  for (unsigned i = 0; i < 4; ++i) {
    model.policy(**a).Observe(3, 1);
    model.policy(**b).Observe(3, 4);
  }
  const auto saved_a = model.policy(**a);
  const auto saved_b = model.policy(**b);
  ASSERT_NE(saved_a, saved_b);
  ASSERT_TRUE((*a)->SaveState(nullptr).has_value());
  ASSERT_TRUE((*b)->SaveState(nullptr).has_value());
  const std::array<std::int32_t, 1> tail = {5};
  ASSERT_TRUE((*a)->Prefill(tail, ignored).has_value());
  ASSERT_TRUE((*b)->Prefill(tail, ignored).has_value());
  model.pending_cursor(**a) = 6;
  model.policy(**a).Observe(2, 3);
  const auto changed_b = (*b)->history();
  ASSERT_TRUE((*a)->RestoreState(nullptr).has_value());
  EXPECT_EQ((*a)->history(), std::vector<std::int32_t>(first.begin(), first.end()));
  EXPECT_EQ((*a)->history(), model.native_state(1).target);
  EXPECT_EQ(model.pending_cursor(**a), 2U);
  EXPECT_EQ(model.policy(**a), saved_a);
  EXPECT_EQ((*b)->history(), changed_b);
  EXPECT_EQ(model.native_state(2).target, changed_b);
  EXPECT_EQ(model.pending_cursor(**b), 4U);
  EXPECT_EQ(model.policy(**b), saved_b);
}

TEST(LlmScores, ABranchPreparationRefusalLeavesPeerGenerationRunning) {
  NativeBranchesFake model;
  auto a = model.branch(1);
  auto b = model.branch(2);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  std::vector<float> last_a;
  std::vector<float> last_b;
  const std::array<std::int32_t, 1> first = {0};
  const std::array<std::int32_t, 1> second = {2};
  ASSERT_TRUE((*a)->Prefill(first, last_a).has_value());
  ASSERT_TRUE((*b)->Prefill(second, last_b).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  rt::Generation result_a;
  rt::Generation result_b;
  auto opened_a = (*a)->BeginGeneration(last_a, options, result_a);
  auto opened_b = (*b)->BeginGeneration(last_b, options, result_b);
  ASSERT_TRUE(opened_a.has_value());
  ASSERT_TRUE(opened_b.has_value());
  model.native_state(1).fail_prepare = true;  // proven host refusal before dispatch
  EXPECT_FALSE((*opened_a)->PrepareStep().has_value());
  EXPECT_FALSE((*opened_a)->Finish().has_value());
  EXPECT_EQ((*a)->history(), model.native_state(1).target);
  ASSERT_TRUE((*opened_b)->RunScalarStep().has_value());
  ASSERT_TRUE((*opened_b)->RunScalarStep().has_value());
  ASSERT_TRUE((*opened_b)->Finish().has_value());
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 4, 5));
  EXPECT_EQ((*b)->history(), model.native_state(2).target);
  EXPECT_EQ(model.native_state(2).settlements, 1U);
  EXPECT_EQ(model.native_state(0).chunks, 0U);
}

TEST(LlmScores, AGenerationWaveCapacityRefusalLeavesSparseActiveBranchesUntouched) {
  NativeBranchesFake model(false, 2);
  auto a = model.branch(0);
  auto b = model.branch(3);  // A high slot ID is valid with two active owners.
  auto c = model.branch(1);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(c.has_value());
  std::vector<float> last_a;
  std::vector<float> last_b;
  std::vector<float> last_c;
  ASSERT_TRUE((*a)->Prefill(std::array<std::int32_t, 1>{0}, last_a).has_value());
  ASSERT_TRUE((*b)->Prefill(std::array<std::int32_t, 1>{2}, last_b).has_value());
  ASSERT_TRUE((*c)->Prefill(std::array<std::int32_t, 1>{4}, last_c).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  rt::Generation result_a;
  rt::Generation result_b;
  rt::Generation result_c;
  auto opened_a = (*a)->BeginGeneration(last_a, options, result_a);
  auto opened_b = (*b)->BeginGeneration(last_b, options, result_b);
  auto opened_c = (*c)->BeginGeneration(last_c, options, result_c);
  ASSERT_TRUE(opened_a.has_value());
  ASSERT_TRUE(opened_b.has_value());
  ASSERT_TRUE(opened_c.has_value());
  model.native_state(1).fail_prepare = true;
  const std::array<rt::Llm::GenerationSession*, 3> too_many = {opened_a->get(), opened_b->get(),
                                                               opened_c->get()};
  EXPECT_FALSE(model.RunGenerationWave(too_many).has_value());
  EXPECT_FALSE((*opened_a)->done());
  EXPECT_FALSE((*opened_b)->done());
  EXPECT_FALSE((*opened_c)->done());  // No preparation reached its own refusal.
  EXPECT_THAT(result_a.tokens, ElementsAre(1));
  EXPECT_THAT(result_b.tokens, ElementsAre(3));
  EXPECT_THAT(result_c.tokens, ElementsAre(5));
  EXPECT_EQ(result_a.steps, 0U);
  EXPECT_EQ(result_b.steps, 0U);
  EXPECT_EQ(result_c.steps, 0U);
  EXPECT_EQ(model.native_state(0).chunks, 1U);
  EXPECT_EQ(model.native_state(3).chunks, 1U);
  EXPECT_EQ(model.native_state(1).chunks, 1U);
  const std::array<rt::Llm::GenerationSession*, 2> within_capacity = {opened_b->get(),
                                                                      opened_a->get()};
  ASSERT_TRUE(model.RunGenerationWave(within_capacity).has_value());
  ASSERT_TRUE(model.RunGenerationWave(within_capacity).has_value());
  ASSERT_TRUE((*opened_a)->Finish().has_value());
  ASSERT_TRUE((*opened_b)->Finish().has_value());
  EXPECT_THAT(result_a.tokens, ElementsAre(1, 2, 3));
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 4, 5));
  EXPECT_THAT(result_c.tokens, ElementsAre(5));
  (*opened_c)->Cancel();
  ASSERT_TRUE((*opened_c)->Finish().has_value());
  EXPECT_THAT((*c)->history(), ElementsAre(4));
  EXPECT_EQ(model.native_state(1).chunks, 1U);
}

TEST(LlmScores, AGenerationWavePreparationRefusalKeepsAnAlreadyPreparedPeer) {
  NativeBranchesFake model;
  auto a = model.branch(1);
  auto b = model.branch(2);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  std::vector<float> last_a;
  std::vector<float> last_b;
  ASSERT_TRUE((*a)->Prefill(std::array<std::int32_t, 1>{0}, last_a).has_value());
  ASSERT_TRUE((*b)->Prefill(std::array<std::int32_t, 1>{2}, last_b).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  rt::Generation result_a;
  rt::Generation result_b;
  auto opened_a = (*a)->BeginGeneration(last_a, options, result_a);
  auto opened_b = (*b)->BeginGeneration(last_b, options, result_b);
  ASSERT_TRUE(opened_a.has_value());
  ASSERT_TRUE(opened_b.has_value());
  model.native_state(1).fail_prepare = true;
  // The peer is already prepared when the second request refuses growth.
  const std::array<rt::Llm::GenerationSession*, 2> both = {opened_b->get(), opened_a->get()};
  ASSERT_TRUE(model.RunGenerationWave(both).has_value());
  EXPECT_TRUE((*opened_a)->done());
  EXPECT_FALSE((*opened_b)->done());
  EXPECT_THAT(result_a.tokens, ElementsAre(1));
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 4));
  EXPECT_FALSE((*opened_a)->Finish().has_value());
  EXPECT_EQ((*a)->history(), model.native_state(1).target);
  const std::array<rt::Llm::GenerationSession*, 1> peer = {opened_b->get()};
  ASSERT_TRUE(model.RunGenerationWave(peer).has_value());
  ASSERT_TRUE((*opened_b)->Finish().has_value());
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 4, 5));
  EXPECT_EQ((*b)->history(), model.native_state(2).target);
}

TEST(LlmScores, AGenerationWaveSamplingErrorPublishesTheCompletedPeer) {
  NativeBranchesFake model;
  auto a = model.branch(1);
  auto b = model.branch(2);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  std::vector<float> last_a;
  std::vector<float> last_b;
  ASSERT_TRUE((*a)->Prefill(std::array<std::int32_t, 1>{0}, last_a).has_value());
  ASSERT_TRUE((*b)->Prefill(std::array<std::int32_t, 1>{2}, last_b).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  options.sampling =
      jitllm::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
  rt::Generation result_a;
  rt::Generation result_b;
  auto opened_a = (*a)->BeginGeneration(last_a, options, result_a);
  auto opened_b = (*b)->BeginGeneration(last_b, options, result_b);
  ASSERT_TRUE(opened_a.has_value());
  ASSERT_TRUE(opened_b.has_value());
  model.nonfinite_wave_row = 1;
  const std::array<rt::Llm::GenerationSession*, 2> both = {opened_a->get(), opened_b->get()};
  ASSERT_TRUE(model.RunGenerationWave(both).has_value());
  EXPECT_TRUE((*opened_a)->done());
  EXPECT_FALSE((*opened_b)->done());
  EXPECT_FALSE((*opened_a)->Finish().has_value());
  // The failed choice follows a completed native anchor, which stays known.
  EXPECT_THAT((*a)->history(), ElementsAre(0, 1));
  EXPECT_EQ((*a)->history(), model.native_state(1).target);
  const std::array<rt::Llm::GenerationSession*, 1> peer = {opened_b->get()};
  ASSERT_TRUE(model.RunGenerationWave(peer).has_value());
  ASSERT_TRUE((*opened_b)->Finish().has_value());
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 4, 5));
  EXPECT_EQ((*b)->history(), model.native_state(2).target);
}

TEST(LlmScores, ADiscardedWaveJudgementPreservesOnlyItsOwnPriorPrefixAndCursor) {
  NativeBranchesFake model(true);
  auto a = model.branch(1);
  auto b = model.branch(2);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  std::vector<float> last_a;
  std::vector<float> last_b;
  ASSERT_TRUE((*a)->Prefill(std::array<std::int32_t, 1>{0}, last_a).has_value());
  ASSERT_TRUE((*b)->Prefill(std::array<std::int32_t, 1>{2}, last_b).has_value());
  const auto target_a = model.native_state(1).target;
  const auto drafter_a = model.native_state(1).injection;
  model.pending_cursor(**a) = 3;
  model.pending_cursor(**b) = 4;
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  rt::Generation result_a;
  rt::Generation result_b;
  auto opened_a = (*a)->BeginGeneration(last_a, options, result_a);
  auto opened_b = (*b)->BeginGeneration(last_b, options, result_b);
  ASSERT_TRUE(opened_a.has_value());
  ASSERT_TRUE(opened_b.has_value());
  model.failed_wave_judgement = 1;
  const std::array<rt::Llm::GenerationSession*, 2> both = {opened_a->get(), opened_b->get()};
  ASSERT_TRUE(model.RunGenerationWave(both).has_value());
  EXPECT_FALSE((*opened_a)->Finish().has_value());
  EXPECT_EQ((*a)->history(), target_a);
  EXPECT_EQ(model.native_state(1).target, target_a);
  EXPECT_EQ(model.native_state(1).injection, drafter_a);
  EXPECT_EQ(model.pending_cursor(**a), 3U);
  ASSERT_TRUE((*opened_b)->Finish().has_value());
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 2, 3, 4));
  EXPECT_EQ((*b)->history(), model.native_state(2).target);
  EXPECT_EQ(model.pending_cursor(**b), 4U);
  EXPECT_FALSE(model.native_state(2).pending);
}

TEST(LlmScores, ResumablePromptAdmissionIsHostOnlyAndOwnsItsPrompt) {
  FakeLlm model(true);
  std::vector<std::int32_t> prompt(19, 2);
  auto opened = model.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  EXPECT_EQ(model.clearings, 0U);
  EXPECT_EQ(model.chunks, 0U);
  EXPECT_TRUE(model.history().empty());
  auto next = session.NextUnit();
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->phase, rt::Llm::PromptSession::Phase::kReuse);
  EXPECT_EQ(next->rows, 0U);
  EXPECT_FALSE(model.default_branch().BeginPrompt(prompt).has_value());
  rt::GenerateOptions options;
  rt::Generation generation;
  EXPECT_FALSE(model.BeginGeneration(FakeLlm::Row(1), options, generation).has_value());
  EXPECT_FALSE(session.Finish().has_value());
  std::ranges::fill(prompt, 3);  // caller storage is not retained
  std::vector<std::uint32_t> units;
  while (!session.done()) {
    auto unit = session.NextUnit();
    ASSERT_TRUE(unit.has_value());
    units.push_back(unit->rows);
    ASSERT_TRUE(session.Advance().has_value());
  }
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_THAT(units, ElementsAre(0, 8, 8, 3));
  EXPECT_EQ(model.history(), std::vector<std::int32_t>(19, 2));
  EXPECT_EQ(model.history(), model.target);
  EXPECT_EQ(model.injection, model.target);
  EXPECT_EQ(model.clearings, 1U);
  EXPECT_EQ(session.run().end, 19U);
  EXPECT_EQ(session.run().chunks, 3U);
  EXPECT_EQ(session.last(), FakeLlm::Row(3));
}

TEST(LlmScores, ResumablePromptCancelsAtACompletePrefixAndCanResume) {
  FakeLlm model;
  std::vector<float> row;
  const std::array<std::int32_t, 2> prefix = {0, 1};
  ASSERT_TRUE(model.Prefill(prefix, row).has_value());
  std::vector<std::int32_t> prompt = {0, 1};
  prompt.resize(19, 2);
  auto opened = model.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  ASSERT_TRUE(session.Advance().has_value());
  EXPECT_EQ(session.reused(), 2U);
  ASSERT_TRUE(session.Advance().has_value());
  EXPECT_EQ(model.target.size(), 10U);
  session.Cancel();
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_TRUE(session.run().stopped);
  EXPECT_EQ(session.run().end, 10U);
  EXPECT_EQ(session.run().chunks, 1U);
  EXPECT_TRUE(session.last().empty());
  EXPECT_EQ(model.history(), model.target);
  EXPECT_EQ(model.clearings, 0U);
  std::uint32_t reused = 0;
  rt::PrefillRun finished;
  ASSERT_TRUE(model.PreparePrompt(prompt, 0, row, reused, {}, &finished).has_value());
  EXPECT_EQ(reused, 10U);
  EXPECT_EQ(model.target, prompt);
  EXPECT_EQ(model.history(), prompt);
  EXPECT_EQ(finished.chunks, 2U);
}

TEST(LlmScores, ResumablePromptKeepsTheTiledTailAndTurnBoundarySeparate) {
  FakeLlm model;
  model.ConfigurePrefill(8192, 4096);
  const std::vector<std::int32_t> prompt(2051, 1);
  auto opened = model.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  ASSERT_TRUE(session.Advance().has_value());
  auto first = session.NextUnit();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->rows, 2048U);
  ASSERT_TRUE(session.Advance().has_value());
  auto tail = session.NextUnit();
  ASSERT_TRUE(tail.has_value());
  EXPECT_EQ(tail->rows, 3U);
  ASSERT_TRUE(session.Advance().has_value());
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_EQ(model.history(), prompt);

  FakeLlm boundary_model;
  const std::vector<std::int32_t> boundary_prompt(17, 2);
  auto boundary = boundary_model.default_branch().BeginPrompt(boundary_prompt, 9);
  ASSERT_TRUE(boundary.has_value());
  ASSERT_TRUE((*boundary)->Advance().has_value());
  ASSERT_TRUE((*boundary)->Advance().has_value());
  auto before = (*boundary)->NextUnit();
  ASSERT_TRUE(before.has_value());
  EXPECT_EQ(before->rows, 1U);
  ASSERT_TRUE((*boundary)->Advance().has_value());
  auto checkpoint = (*boundary)->NextUnit();
  ASSERT_TRUE(checkpoint.has_value());
  EXPECT_EQ(checkpoint->phase, rt::Llm::PromptSession::Phase::kCheckpoint);
  EXPECT_EQ(checkpoint->rows, 0U);
  (*boundary)->Cancel();
  ASSERT_TRUE((*boundary)->Finish().has_value());
  EXPECT_EQ(boundary_model.history(), std::vector<std::int32_t>(9, 2));
  EXPECT_EQ(boundary_model.history(), boundary_model.target);
  EXPECT_EQ(boundary_model.turn_checkpoints(), 0U);
}

TEST(LlmScores, ResumablePromptFailuresDoNotPublishAnUnprovenChunk) {
  FakeLlm model;
  const std::vector<std::int32_t> prompt(19, 2);
  auto opened = model.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  ASSERT_TRUE(session.Advance().has_value());
  ASSERT_TRUE(session.Advance().has_value());
  model.fail_chunk = 2;
  EXPECT_FALSE(session.Advance().has_value());
  EXPECT_TRUE(session.done());
  EXPECT_FALSE(session.Finish().has_value());
  EXPECT_EQ(session.run().end, 8U);  // only one successfully completed chunk
  EXPECT_EQ(session.run().chunks, 1U);
  EXPECT_TRUE(session.last().empty());
  EXPECT_TRUE(model.history().empty());
  EXPECT_FALSE(model.HasRetainedState());
  std::vector<float> row;
  const std::array<std::int32_t, 1> restart = {3};
  ASSERT_TRUE(model.Prefill(restart, row).has_value());
  EXPECT_EQ(model.clearings, 2U);
  EXPECT_EQ(model.history(), model.target);
  EXPECT_THAT(model.history(), ElementsAre(3));
}

TEST(LlmScores, ResumablePromptKeepsARequiredClearAfterAProvenRefusal) {
  FakeLlm model;
  std::vector<float> row;
  const std::array<std::int32_t, 2> prefix = {0, 1};
  ASSERT_TRUE(model.Prefill(prefix, row).has_value());
  model.Forget();  // native state remains usable, but its host history was discarded
  model.fail_clear = true;
  const std::array<std::int32_t, 1> restart = {3};
  auto opened = model.default_branch().BeginPrompt(restart);
  ASSERT_TRUE(opened.has_value());
  EXPECT_FALSE((*opened)->Advance().has_value());
  EXPECT_FALSE((*opened)->Finish().has_value());
  EXPECT_TRUE(model.usable);
  EXPECT_EQ(model.clearings, 1U);
  EXPECT_TRUE(model.history().empty());
  EXPECT_THAT(model.target, ElementsAre(0, 1));
  model.fail_clear = false;
  ASSERT_TRUE(model.Prefill(restart, row).has_value());
  EXPECT_EQ(model.clearings, 2U);
  EXPECT_EQ(model.history(), model.target);
  EXPECT_THAT(model.history(), ElementsAre(3));
}

TEST(LlmScores, AResumablePromptDoesNotBlockOrCancelItsPeerGeneration) {
  NativeBranchesFake model;
  auto decoding = model.branch(0);
  auto prefilling = model.branch(1);
  ASSERT_TRUE(decoding.has_value());
  ASSERT_TRUE(prefilling.has_value());
  std::vector<float> last;
  const std::array<std::int32_t, 1> prefix = {0};
  ASSERT_TRUE((*decoding)->Prefill(prefix, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  rt::Generation out;
  auto generation = (*decoding)->BeginGeneration(last, options, out);
  ASSERT_TRUE(generation.has_value());
  const std::vector<std::int32_t> prompt(19, 2);
  auto prefill = (*prefilling)->BeginPrompt(prompt);
  ASSERT_TRUE(prefill.has_value());
  ASSERT_TRUE((*prefill)->Advance().has_value());
  ASSERT_TRUE((*generation)->RunScalarStep().has_value());
  ASSERT_TRUE((*prefill)->Advance().has_value());
  (*prefill)->Cancel();
  ASSERT_TRUE((*prefill)->Finish().has_value());
  ASSERT_TRUE((*generation)->RunScalarStep().has_value());
  ASSERT_TRUE((*generation)->Finish().has_value());
  EXPECT_THAT(out.tokens, ElementsAre(1, 2, 3));
  EXPECT_FALSE(out.cancelled);
  EXPECT_EQ((*decoding)->history(), model.native_state(0).target);
  EXPECT_EQ((*prefilling)->history(), std::vector<std::int32_t>(8, 2));
  EXPECT_EQ((*prefilling)->history(), model.native_state(1).target);
  EXPECT_EQ(model.native_state(2).chunks, 0U);
  EXPECT_EQ(model.native_state(3).chunks, 0U);
}

}  // namespace
