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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "runtime/serving.h"

namespace {

namespace rt = jitllm::runtime;
namespace engine = jitllm::engine;
namespace catalog = jitllm::catalog;
using ::testing::ElementsAre;

class FakePaged final : public engine::PagedModel {
 public:
  std::uint32_t stream() const override { return 0; }
  const catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<catalog::ExtentId> managed_extents() const override { return {}; }
  engine::Status Release() override { return {}; }

 private:
  catalog::Closure closure_;
};

class FakeLlm final : public rt::Llm {
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

}  // namespace
