// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Llm's real teacher-forcing and generation loops over a completed fake
// target/drafter, without a device. These controls test position, state,
// callback alignment and retirement, independently of HTTP serialization.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "chat/chat.h"
#include "engine/qwen38_wave_plan.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "memory/reclaim.h"
#include "runtime/cohort_schedule.h"
#include "runtime/serving.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace {

namespace rt = jitllm::runtime;
namespace engine = jitllm::engine;
namespace catalog = jitllm::catalog;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

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
  // A tokenizer and chat template, as registration gives a model them.
  rt::Status UseChat(jitllm::tokenizer::Tokenizer tokenizer, std::string_view chat_template) {
    tokenizer_ = std::make_unique<jitllm::tokenizer::Tokenizer>(std::move(tokenizer));
    return UseTemplate(chat_template);
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
  // A clean capacity refusal before dispatch (Llm::StateRefusedFor): the
  // state stays as it was. capacity_refused reports the last call's.
  bool refuse_capacity = false;
  bool capacity_refused = false;
  bool pending = false;
  bool omit_row = false;
  // Spilled (Llm::SpillIdle): its tokens kept, none of them resident.
  bool spilled = false;
  std::vector<std::int32_t> step = {2, 3, 4};

 protected:
  std::uint32_t StepTokenBoundFor(const Branch&, std::uint32_t left) const override {
    return std::min<std::uint32_t>(left, static_cast<std::uint32_t>(step.size()));
  }
  rt::Status RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                      std::vector<float>& row) override {
    EXPECT_EQ(past, target.size());
    capacity_refused = refuse_capacity;
    if (refuse_capacity) {
      return std::unexpected("loading would exceed the execution budget");
    }
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
    spilled = false;
    injection.clear();
    usable = true;
    pending = false;
    return {};
  }
  bool StateUsable() const override { return usable; }
  rt::Status PrepareDecodeState(std::uint32_t /*pos*/, std::uint32_t /*left*/) override {
    capacity_refused = refuse_capacity;
    if (refuse_capacity) {
      return std::unexpected("loading would exceed the execution budget");
    }
    if (fail_prepare) {
      return std::unexpected("fake decode preparation failed");
    }
    return {};
  }
  std::uint64_t target_state_base() const override { return 0; }
  std::uint64_t target_state_bytes() const override { return 0; }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return target.size(); }
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

class PlainDeviceFake final : public FakeLlm {
 public:
  bool enabled = true;
  bool scalar = false;
  bool fail = false;
  unsigned device_calls = 0;
  std::optional<std::int32_t> observed_chosen;
  std::pair<bool, bool> Eligibility(GenerationSession* session, bool speculative, bool rows,
                                    bool has_branch = true) {
    PreparedGeneration unit;
    unit.session = session;
    unit.branch = has_branch ? &default_branch() : nullptr;
    unit.step.speculative = speculative;
    unit.step.need_logits = rows;
    return {GreedyWithoutRows(unit), DeviceGreedy(unit)};
  }

 protected:
  std::optional<rt::Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                              std::uint32_t past, std::int32_t& token) override {
    if (!enabled) return std::nullopt;
    ++device_calls;
    token = 6;  // A failed implementation must not publish this value.
    if (fail) return std::unexpected("fake plain token refused before dispatch");
    std::vector<float> row;
    auto ran = RunChunkFor(branch, all, past, false, row);
    if (ran) token = static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
    return ran;
  }
  rt::Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    auto ran =
        scalar ? RunScalarGenerationUnits(prepared) : Llm::RunPreparedGenerationWave(prepared);
    observed_chosen = prepared.front().chosen;
    return ran;
  }
};

TEST(LlmScores, RowFreeGreedyEligibilityPreservesOrdinarySpeculationRefusal) {
  for (const bool sampled : {false, true}) {
    PlainDeviceFake model;
    std::vector<float> last;
    ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{0}, last));
    rt::GenerateOptions options;
    options.max_tokens = 2;
    if (sampled)
      options.sampling = jitllm::execution::SamplingParams{.temperature = 0.7F, .top_k = 4};
    rt::Generation out;
    auto session = model.BeginGeneration(last, options, out);
    ASSERT_TRUE(session);
    for (const bool spec : {false, true}) {
      for (const bool rows : {false, true}) {
        const bool row_free = !sampled && !rows;
        EXPECT_EQ(model.Eligibility(session->get(), spec, rows),
                  (std::pair{row_free, row_free && !spec}));
      }
    }
    EXPECT_EQ(model.Eligibility(nullptr, false, false), (std::pair{false, false}));
    EXPECT_EQ(model.Eligibility(session->get(), false, false, false), (std::pair{false, false}));
    (*session)->Cancel();
    EXPECT_TRUE((*session)->Finish());
  }
}

TEST(LlmScores, PreparedPlainAndScalarFallbackPublishOnlySuccessfulDeviceTokens) {
  for (const bool scalar : {false, true}) {
    PlainDeviceFake model;
    model.scalar = scalar;
    std::vector<float> last;
    ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{0}, last));
    rt::GenerateOptions options;
    options.max_tokens = 2;
    rt::Generation out;
    auto session = model.BeginGeneration(last, options, out);
    ASSERT_TRUE(session);
    const std::array<rt::Llm::GenerationSession*, 1> one = {session->get()};
    ASSERT_TRUE(model.RunGenerationWave(one));
    ASSERT_TRUE((*session)->Finish());
    EXPECT_EQ(model.device_calls, 1);
    EXPECT_EQ(model.observed_chosen, 2);
    EXPECT_THAT(out.tokens, ElementsAre(1, 2));
    EXPECT_THAT(model.target, ElementsAre(0, 1));
  }
}

TEST(LlmScores, PreparedPlainNulloptAndRequestedScoresRetainRows) {
  for (const bool scores : {false, true}) {
    PlainDeviceFake model;
    model.enabled = scores;  // Nullopt otherwise; scores must never ask.
    std::vector<float> last;
    ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{0}, last));
    rt::GenerateOptions options;
    options.max_tokens = 2;
    options.keep_logits = scores;
    rt::Generation out;
    auto session = model.BeginGeneration(last, options, out);
    ASSERT_TRUE(session);
    const std::array<rt::Llm::GenerationSession*, 1> one = {session->get()};
    ASSERT_TRUE(model.RunGenerationWave(one));
    ASSERT_TRUE((*session)->Finish());
    EXPECT_EQ(model.device_calls, 0);
    EXPECT_FALSE(model.observed_chosen);
    EXPECT_THAT(out.tokens, ElementsAre(1, 2));
  }
}

TEST(LlmScores, FailedPreparedDeviceTokenDoesNotPublishTheWrittenOutput) {
  for (const bool scalar : {false, true}) {
    PlainDeviceFake model;
    model.scalar = scalar;
    model.fail = true;
    std::vector<float> last;
    ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{0}, last));
    rt::GenerateOptions options;
    options.max_tokens = 2;
    rt::Generation out;
    auto session = model.BeginGeneration(last, options, out);
    ASSERT_TRUE(session);
    const std::array<rt::Llm::GenerationSession*, 1> one = {session->get()};
    // Independent scalar refusal belongs to its session; a shared C1 error
    // fails the outer wave. Neither path may publish the temporary token.
    EXPECT_EQ(model.RunGenerationWave(one).has_value(), scalar);
    EXPECT_FALSE(model.observed_chosen);
    EXPECT_THAT(out.tokens, ElementsAre(1));
    EXPECT_FALSE((*session)->Finish());
  }
}

class NativeBranchesFake final : public FakeLlm {
 public:
  explicit NativeBranchesFake(bool speculative = false, std::size_t wave_capacity = 4,
                              std::uint32_t branches = 4)
      : FakeLlm(speculative), wave_capacity_(wave_capacity) {
    for (auto& leaf : leaves_) {
      leaf = std::make_unique<FakeLlm>(speculative);
    }
    EXPECT_TRUE(PrepareBranches(branches, 3).has_value());
  }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override { return wave_capacity_; }
  std::size_t prefill_wave_capacity() const override { return prefill_waves ? wave_capacity_ : 1; }
  bool prefill_waves = false;
  std::function<void()> prefill_funding_hook;
  std::array<std::uint32_t, kMaxBranches> prefill_funded{};
  std::vector<std::vector<std::uint32_t>> prefill_dispatches;
  std::vector<std::pair<std::uint32_t, rt::PrefillHint>> delivered_prefill_hints;
  std::vector<PrefillHeadModes> future_prefill_heads;
  std::optional<std::uint32_t> nonfinite_wave_row;
  std::optional<std::uint32_t> failed_wave_judgement;
  std::optional<std::uint32_t> failed_scalar_judgement;
  bool scalar_units = false;
  std::optional<std::uint32_t> refuse_scalar_dispatch;
  // The execution budget in state tokens, shared by every branch: growth
  // past it is refused for capacity before dispatch (StateRefusedFor).
  std::optional<std::size_t> budget;
  // Whether a request leases the selected branches' state (BranchIdle).
  bool lease_held = false;
  unsigned spills = 0;
  unsigned restores = 0;
  FakeLlm& native_state(std::size_t slot) { return slot == 0 ? *this : *leaves_.at(slot - 1); }
  std::size_t held() {
    std::size_t tokens = 0;
    for (std::size_t slot = 0; slot < kMaxBranches; ++slot) {
      // Spilled state holds nothing resident.
      tokens += native_state(slot).spilled ? 0 : native_state(slot).target.size();
    }
    return tokens;
  }
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
  bool GenerationCohortUsable() const override {
    if (!prefill_waves) return true;
    if (!usable) return false;
    return std::ranges::all_of(leaves_, [](const auto& leaf) { return leaf->usable; });
  }
  rt::Status PreparePrefillStateFor(Branch& branch, std::uint32_t past,
                                    std::uint32_t rows) override {
    const auto id = BranchIndex(branch);
    auto& native = Native(branch);
    native.capacity_refused = false;
    std::size_t paid = 0;
    for (std::size_t i = 0; i < kMaxBranches; ++i)
      paid += std::max<std::size_t>(prefill_funded[i], native_state(i).target.size());
    paid -= std::max<std::size_t>(prefill_funded[id], native.target.size());
    if (paid + past + rows > budget.value_or(SIZE_MAX)) {
      native.capacity_refused = true;
      return std::unexpected("fake prefill funding would exceed capacity");
    }
    prefill_funded[id] = past + rows;  // backing grew; no KV row was dispatched
    if (prefill_funding_hook) prefill_funding_hook();
    return {};
  }
  rt::Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                std::uint32_t past, bool inject, bool want_head,
                                std::vector<float>& row, rt::PrefillHint next) override {
    delivered_prefill_hints.emplace_back(BranchIndex(branch), next);
    return FakeLlm::RunPrefillChunkFor(branch, all, past, inject, want_head, row, next);
  }
  rt::Status RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) override {
    future_prefill_heads.push_back(FuturePrefillHeads(prepared));
    std::vector<std::uint32_t> owners;
    for (const auto& unit : prepared) {
      owners.push_back(BranchIndex(*unit.branch));
      EXPECT_FALSE(unit.session->NextUnit().has_value());  // still borrowed
      EXPECT_EQ(unit.branch->history().size(), unit.past);
    }
    prefill_dispatches.push_back(std::move(owners));
    return Llm::RunPreparedPrefillWave(prepared);
  }
  rt::Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    if (scalar_units) {
      if (refuse_scalar_dispatch) native_state(*refuse_scalar_dispatch).refuse_capacity = true;
      return RunScalarGenerationUnits(prepared);
    }
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
  // Spill and restore (Llm::SpillIdle, Llm::SpillSetAside, a prompt's
  // restore): a spilled slot's tokens are kept but count against no budget;
  // its restore is refused for capacity like a growth past the budget.
  bool LeasedFor(const Branch& branch) const override {
    return lease_held && selected_[BranchIndex(branch)];
  }
  rt::Status SpillFor(Branch& branch) override {
    ++spills;
    Native(branch).spilled = true;
    return {};
  }
  rt::Status RestoreFor(Branch& branch) override {
    FakeLlm& native = Native(branch);
    native.capacity_refused = false;
    if (held() + native.target.size() > budget.value_or(SIZE_MAX)) {
      native.capacity_refused = true;
      return std::unexpected("loading would exceed the execution budget");
    }
    ++restores;
    native.spilled = false;
    return {};
  }
  bool SpilledFor(const Branch& branch) const override { return Native(branch).spilled; }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return Native(branch).spilled ? Native(branch).target.size() : 0;
  }
  rt::Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                         bool inject, std::vector<float>& logits) override {
    if (Native(branch).spilled) {
      ADD_FAILURE() << "a spilled state ran a chunk";
      return std::unexpected("a spilled state ran a chunk");
    }
    if (OverBudget(branch, all.size())) {
      return std::unexpected("loading would exceed the execution budget");
    }
    return Native(branch).FakeLlm::RunChunk(all, n_past, inject, logits);
  }
  rt::Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                         std::uint32_t left, std::vector<std::int32_t>& kept,
                         std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                         bool* prefix_kept = nullptr) override {
    if (failed_scalar_judgement == BranchIndex(branch)) {
      // Fake completed verify, judgement failed and the verify discarded.
      if (prefix_kept != nullptr) {
        *prefix_kept = true;
      }
      return std::unexpected("fake completed judgement failed");
    }
    return Native(branch).FakeLlm::SpecStep(all, pos, left, kept, logits, drafted);
  }
  rt::Status SettleFor(Branch& branch) override { return Native(branch).FakeLlm::Settle(); }
  rt::Status ClearStateFor(Branch& branch) override { return Native(branch).FakeLlm::ClearState(); }
  rt::Status ReleaseIdleStateFor(Branch& branch) override {
    return Native(branch).FakeLlm::ClearState();
  }
  bool StateUsableFor(const Branch& branch) const override {
    return Native(branch).FakeLlm::StateUsable();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return Native(branch).capacity_refused;
  }
  rt::Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override {
    if (OverBudget(branch, std::size_t{pos} + 1)) {
      return std::unexpected("loading would exceed the execution budget");
    }
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
  // Whether growing this branch's state to `tokens` would pass the budget
  // beside every other branch's: then refused for capacity, nothing run.
  bool OverBudget(Branch& branch, std::size_t tokens) {
    if (!budget.has_value()) {
      return false;
    }
    const std::size_t limit = *budget;
    FakeLlm& native = Native(branch);
    const bool over =
        held() - native.target.size() + std::max(tokens, native.target.size()) > limit;
    if (over) {
      native.capacity_refused = true;
    }
    return over;
  }
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

// GPT-2's byte alphabet, for a byte-level vocabulary without merges: every
// byte of text is a token.
std::string ByteText(unsigned b) {
  const auto printable = [](unsigned x) {
    return (x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || x >= 0xAE;
  };
  unsigned n = 0;
  for (unsigned x = 0; x < b; ++x) {
    n += printable(x) ? 0 : 1;
  }
  std::string out;
  jitllm::tokenizer::unicode::AppendUtf8(printable(b) ? b : 256 + n, out);
  return out;
}

jitllm::tokenizer::Tokenizer ByteTokenizer() {
  jitllm::tokenizer::TokenizerSpec spec;
  for (unsigned b = 0; b < 256; ++b) {
    spec.tokens.push_back(ByteText(b));
    spec.kinds.push_back(jitllm::tokenizer::TokenKind::kNormal);
  }
  for (const char* control : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}) {
    spec.tokens.emplace_back(control);
    spec.kinds.push_back(jitllm::tokenizer::TokenKind::kControl);
  }
  spec.eos = 258;
  auto t = jitllm::tokenizer::Tokenizer::Create(std::move(spec));
  EXPECT_TRUE(t.has_value()) << t.error().ToString();
  return std::move(*t);
}

// The prompt-size chain follows the model, not the tokenizer's library
// defaults (D-102): a chat over 4 MiB, more than 2^22 tokens, renders and
// tokenizes when the context holds it (the route's bodies and the model's
// context, not 4 MiB, bound it), and the context's bound is reported as
// such, before any model work.
TEST(LlmRender, AChatOverFourMiBRendersWithinTheContext) {
  FakeLlm model;
  model.ConfigurePrefill(8'000'000, 8);
  ASSERT_TRUE(model
                  .UseChat(ByteTokenizer(),
                           "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
                           "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>"
                           "assistant\n{% endif %}")
                  .has_value());
  // At least the floor; four times the context times the longest token.
  EXPECT_GE(model.render_bytes(), std::size_t{32} << 20U);
  const std::size_t bytes = (std::size_t{4} << 20U) + 300'000;  // past 2^22 tokens too
  jitllm::chat::Conversation c;
  c.messages.push_back({jitllm::chat::Role::kUser, std::string(bytes, 'x'), std::nullopt, {}});
  c.max_render_bytes = model.render_bytes();
  std::uint32_t boundary = 0;
  rt::ChatRenderFailure failure = rt::ChatRenderFailure::kOther;
  auto tokens = model.RenderChat(c, &boundary, {.max_tokens = model.usable_context()}, &failure);
  ASSERT_TRUE(tokens.has_value()) << tokens.error();
  EXPECT_GT(tokens->size(), std::size_t{1} << 22U);
  EXPECT_EQ(tokens->size(), bytes + 1 + 5 + 1 + 1 + 1 + 10);  // the template's tokens around it
  EXPECT_EQ(tokens->front(), 256);                            // <|im_start|>, a control token
  // Past the context: a typed failure, so the route answers 400
  // context_length_exceeded rather than an internal error.
  tokens = model.RenderChat(c, nullptr, {.max_tokens = 1000}, &failure);
  ASSERT_FALSE(tokens.has_value());
  EXPECT_EQ(failure, rt::ChatRenderFailure::kTooLong);
  // A cancellation ends an interpreted rendering (the request ended).
  const std::function<bool()> ended = [] { return true; };
  tokens = model.RenderChat(c, nullptr, {.cancelled = &ended}, &failure);
  ASSERT_FALSE(tokens.has_value());
  EXPECT_EQ(failure, rt::ChatRenderFailure::kCancelled);
}

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

TEST(LlmScores, ResumableScoringMatchesSerialRowsInjectionAndLastHead) {
  FakeLlm serial(true);
  FakeLlm stepped(true);
  const std::array<std::int32_t, 5> prompt{0, 7, 2, 3, 1};
  std::vector<std::pair<std::int32_t, std::vector<float>>> expected;
  std::vector<float> last;
  ASSERT_TRUE(serial
                  .ScorePrompt(prompt, last,
                               [&](std::int32_t id, std::span<const float> row) {
                                 expected.emplace_back(id,
                                                       std::vector<float>(row.begin(), row.end()));
                                 return true;
                               })
                  .has_value());
  std::vector<std::pair<std::int32_t, std::vector<float>>> actual;
  auto session = stepped.default_branch().BeginScoringPrompt(
      prompt, [&](std::int32_t id, std::span<const float> row) {
        actual.emplace_back(id, std::vector<float>(row.begin(), row.end()));
        return true;
      });
  ASSERT_TRUE(session.has_value());
  while (!(*session)->done()) {
    auto unit = (*session)->NextUnit();
    ASSERT_TRUE(unit.has_value());
    EXPECT_EQ(unit->rows, unit->phase == rt::Llm::PromptSession::Phase::kChunk ? 1U : 0U);
    ASSERT_TRUE((*session)->Advance().has_value());
  }
  ASSERT_TRUE((*session)->Finish().has_value());
  ASSERT_TRUE((*session)->Finish().has_value());
  EXPECT_EQ(actual, expected);
  EXPECT_EQ((*session)->last(), last);
  EXPECT_EQ(stepped.history(), serial.history());
  EXPECT_EQ(stepped.target, serial.target);
  EXPECT_EQ(stepped.injection, serial.injection);
  EXPECT_EQ(stepped.chunks, serial.chunks);
  EXPECT_EQ(stepped.settlements, 1U);
}

TEST(LlmScores, PurePromptSessionsMayFillTheEntireContextButNeverExceedIt) {
  for (const bool scoring : {false, true}) {
    FakeLlm model;
    const std::vector<std::int32_t> prompt(32, 1);
    std::vector<float> serial_last;
    ASSERT_TRUE(model.Prefill(prompt, serial_last).has_value());
    EXPECT_FALSE(model.Prefill(std::array<std::int32_t, 1>{1}, serial_last).has_value());
    unsigned reported = 0;
    auto opened = scoring ? model.default_branch().BeginScoringPrompt(
                                prompt,
                                [&](std::int32_t id, std::span<const float> row) {
                                  EXPECT_EQ(id, 1);
                                  EXPECT_EQ(row.size(), 8U);
                                  ++reported;
                                  return true;
                                })
                          : model.default_branch().BeginPrompt(prompt, 0, true);
    ASSERT_TRUE(opened.has_value());
    auto& session = **opened;
    while (!session.done()) {
      ASSERT_TRUE(session.Advance().has_value());
    }
    ASSERT_TRUE(session.Finish().has_value());
    EXPECT_EQ(session.run().end, 32U);
    EXPECT_EQ(model.default_branch().history(), prompt);
    EXPECT_EQ(reported, scoring ? 31U : 0U);
    EXPECT_FALSE(model.default_branch().BeginPrompt(std::vector<std::int32_t>(33, 1)).has_value());
  }
}

TEST(LlmScores, ResumableScoringSpillsAndContinuesOnlyTheRemainingRows) {
  NativeBranchesFake model(true);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch.has_value());
  const std::array<std::int32_t, 6> prompt{0, 7, 2, 3, 1, 4};
  std::vector<std::int32_t> scored;
  const auto on_row = [&](std::int32_t id, std::span<const float> row) {
    EXPECT_EQ(std::vector<float>(row.begin(), row.end()),
              FakeLlm::Row((prompt[scored.size()] + 1) % 8));
    scored.push_back(id);
    return true;
  };
  auto session = (*branch)->BeginScoringPrompt(prompt, on_row);
  ASSERT_TRUE(session.has_value());
  ASSERT_TRUE((*session)->Advance().has_value());  // fresh history
  for (unsigned i = 0; i < 2; ++i) {
    ASSERT_TRUE((*session)->Advance().has_value());
  }
  EXPECT_THAT(scored, ElementsAre(7, 2));
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish().has_value());
  session->reset();
  (*branch)->HoldContinuation();
  ASSERT_TRUE(model.SpillIdle(**branch).has_value());
  model.set_retention(rt::Clock::duration::zero());
  auto resumed = (*branch)->BeginScoringPrompt(prompt, on_row, true);
  ASSERT_TRUE(resumed.has_value());
  while (!(*resumed)->done()) {
    ASSERT_TRUE((*resumed)->Advance().has_value());
  }
  ASSERT_TRUE((*resumed)->Finish().has_value());
  EXPECT_THAT(scored, ElementsAre(7, 2, 3, 1, 4));
  EXPECT_EQ(model.native_state(1).chunks, prompt.size());
  EXPECT_EQ(model.native_state(1).target, (*branch)->history());
  EXPECT_EQ(model.native_state(1).injection, (*branch)->history());
  EXPECT_EQ((*resumed)->last(), FakeLlm::Row(5));
  EXPECT_EQ(model.restores, 1U);
  (*branch)->ReleaseContinuation();
}

TEST(LlmScores, AnUntouchedScorerDoesNotSettleOrDiscardASpilledContinuation) {
  NativeBranchesFake model(true);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch.has_value());
  const std::array<std::int32_t, 4> prompt{0, 1, 2, 3};
  auto session = (*branch)->BeginScoringPrompt(prompt, {});
  ASSERT_TRUE(session.has_value());
  ASSERT_TRUE((*session)->Advance().has_value());
  ASSERT_TRUE((*session)->Advance().has_value());
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish().has_value());
  session->reset();
  (*branch)->HoldContinuation();
  ASSERT_TRUE(model.SpillIdle(**branch).has_value());
  const auto prefix = (*branch)->history();
  const auto settlements = model.native_state(1).settlements;
  model.native_state(1).fail_settle = true;  // native rollback refuses an unrestored slot
  auto untouched = (*branch)->BeginScoringPrompt(prompt, {}, true);
  ASSERT_TRUE(untouched.has_value());
  (*untouched)->Cancel();
  ASSERT_TRUE((*untouched)->Finish().has_value());
  untouched->reset();
  EXPECT_EQ((*branch)->history(), prefix);
  EXPECT_TRUE(model.native_state(1).spilled);
  EXPECT_EQ(model.native_state(1).settlements, settlements);
  EXPECT_EQ(model.restores, 0U);
  model.native_state(1).fail_settle = false;
  auto continued = (*branch)->BeginScoringPrompt(prompt, {}, true);
  ASSERT_TRUE(continued.has_value());
  while (!(*continued)->done()) {
    ASSERT_TRUE((*continued)->Advance().has_value());
  }
  ASSERT_TRUE((*continued)->Finish().has_value());
  EXPECT_EQ((*branch)->history(), std::vector<std::int32_t>(prompt.begin(), prompt.end()));
  EXPECT_EQ(model.native_state(1).chunks, prompt.size());
  EXPECT_EQ(model.restores, 1U);
  (*branch)->ReleaseContinuation();
}

TEST(LlmScores, ResumableScoringRetriesCapacityBeforeAnyRowIsReported) {
  NativeBranchesFake model;
  const std::array<std::int32_t, 3> prompt{0, 1, 2};
  unsigned reported = 0;
  auto session =
      model.default_branch().BeginScoringPrompt(prompt, [&](std::int32_t, std::span<const float>) {
        ++reported;
        return true;
      });
  ASSERT_TRUE(session.has_value());
  ASSERT_TRUE((*session)->Advance().has_value());
  model.budget = 0;
  EXPECT_FALSE((*session)->Advance({}, true).has_value());
  EXPECT_TRUE((*session)->refused());
  EXPECT_EQ(reported, 0U);
  EXPECT_TRUE(model.history().empty());
  model.budget = 3;
  while (!(*session)->done()) {
    ASSERT_TRUE((*session)->Advance().has_value());
  }
  ASSERT_TRUE((*session)->Finish().has_value());
  EXPECT_EQ(reported, 2U);
  EXPECT_EQ(model.chunks, 3U);
}

TEST(LlmScores, ResumableScoringStopsAtItsCallbackAndPreservesSettlementFailure) {
  FakeLlm model;
  const std::array<std::int32_t, 3> prompt{0, 1, 2};
  auto session = model.default_branch().BeginScoringPrompt(
      prompt, [](std::int32_t, std::span<const float>) { return false; });
  ASSERT_TRUE(session.has_value());
  ASSERT_TRUE((*session)->Advance().has_value());
  ASSERT_TRUE((*session)->Advance().has_value());
  EXPECT_TRUE((*session)->done());
  EXPECT_TRUE((*session)->run().stopped);
  EXPECT_THAT(model.history(), ElementsAre(0));
  EXPECT_EQ(model.chunks, 1U);
  model.fail_settle = true;
  EXPECT_FALSE((*session)->Finish().has_value());
  EXPECT_TRUE(model.history().empty());
  EXPECT_TRUE((*session)->last().empty());
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

// A model of eight request slots (docs/runtime-serving.md#request-slots)
// runs every branch in one generation wave, each its own continuation;
// a ninth branch does not exist.
TEST(LlmScores, EightBranchesRunInOneGenerationWave) {
  constexpr std::size_t kSlots = 8;
  NativeBranchesFake model(false, kSlots, kSlots);
  EXPECT_EQ(model.branches(), kSlots);
  EXPECT_FALSE(model.branch(kSlots).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  std::array<std::vector<float>, kSlots> last;
  std::array<rt::Generation, kSlots> results;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, kSlots> sessions;
  std::array<rt::Llm::GenerationSession*, kSlots> wave{};
  for (std::size_t slot = 0; slot < kSlots; ++slot) {
    auto branch = model.branch(slot);
    ASSERT_TRUE(branch.has_value());
    const std::array<std::int32_t, 1> prompt = {static_cast<std::int32_t>(slot)};
    ASSERT_TRUE((*branch)->Prefill(prompt, last[slot]).has_value());
    auto opened = (*branch)->BeginGeneration(last[slot], options, results[slot]);
    ASSERT_TRUE(opened.has_value());
    sessions[slot] = std::move(*opened);
    wave[slot] = sessions[slot].get();
  }
  ASSERT_TRUE(model.RunGenerationWave(wave).has_value());
  ASSERT_TRUE(model.RunGenerationWave(wave).has_value());
  for (std::size_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_TRUE(sessions[slot]->done());
    ASSERT_TRUE(sessions[slot]->Finish().has_value());
    const auto at = [slot](std::size_t k) { return static_cast<std::int32_t>((slot + k) % 8); };
    EXPECT_THAT(results[slot].tokens, ElementsAre(at(1), at(2), at(3))) << slot;
    auto branch = model.branch(slot);
    ASSERT_TRUE(branch.has_value());
    EXPECT_EQ((*branch)->history(), model.native_state(slot).target) << slot;
  }
}

// Admission by memory (docs/runtime-serving.md#request-slots): its reclaim
// never spills the branch the request is about to continue or clear (its
// state counts as the request's own already); every other idle branch with
// resident state is a candidate, a leased or empty one is not.
TEST(LlmScores, AdmissionsReclaimSparesTheChosenBranch) {
  NativeBranchesFake model;
  std::vector<float> last;
  for (const std::size_t slot : {0U, 1U, 3U}) {
    auto branch = model.branch(slot);
    ASSERT_TRUE(branch.has_value());
    const std::array<std::int32_t, 2> prompt = {static_cast<std::int32_t>(slot), 1};
    ASSERT_TRUE((*branch)->Prefill(prompt, last).has_value());
  }
  const rt::IdleStateRates rates{.spill_rate = 11.0e9, .restore_rate = 14.5e9};
  const auto ids = [](const std::vector<jitllm::memory::ReclaimCandidate>& c) {
    std::vector<std::uint64_t> out;
    for (const auto& x : c) {
      EXPECT_EQ(x.kind, jitllm::memory::ReclaimKind::kIdleState);
      EXPECT_GT(x.bytes, 0U);
      out.push_back(x.id);
    }
    return out;
  };
  std::vector<jitllm::memory::ReclaimCandidate> all;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, rates, all);
  EXPECT_THAT(ids(all), ElementsAre(0U, 1U, 3U));  // slot 2 holds nothing
  auto chosen = model.branch(1);
  ASSERT_TRUE(chosen.has_value());
  std::vector<jitllm::memory::ReclaimCandidate> spared;
  rt::AddIdleStateCandidates(model, 0, true, *chosen, rates, spared);
  EXPECT_THAT(ids(spared), ElementsAre(0U, 3U));
  // A branch the request open on the stream leases is no candidate either.
  auto peer = model.branch(0);
  ASSERT_TRUE(peer.has_value());
  const std::array<rt::Llm::Branch*, 1> selected = {*peer};
  ASSERT_TRUE(model.SelectBranches(selected).has_value());
  model.lease_held = true;
  std::vector<jitllm::memory::ReclaimCandidate> leased;
  rt::AddIdleStateCandidates(model, 0, true, *chosen, rates, leased);
  EXPECT_THAT(ids(leased), ElementsAre(3U));
  model.lease_held = false;
  // A conversation a continuation holds is spilled, never dropped (its
  // drop is refused): with a spill budget of 0, or one smaller than its
  // state, it is no candidate; one nothing holds is, at its recomputation.
  auto held = model.branch(3);
  ASSERT_TRUE(held.has_value());
  (*held)->HoldContinuation();
  std::vector<jitllm::memory::ReclaimCandidate> spilled;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, rates, spilled);
  EXPECT_THAT(ids(spilled), ElementsAre(0U, 1U, 3U));
  for (const std::uint64_t budget : {std::uint64_t{0}, model.ResidentStateBytes(**held) - 1}) {
    rt::IdleStateRates small = rates;
    small.spill_budget = budget;
    std::vector<jitllm::memory::ReclaimCandidate> dropped;
    rt::AddIdleStateCandidates(model, 0, true, nullptr, small, dropped);
    EXPECT_THAT(ids(dropped), ElementsAre(0U, 1U)) << budget;
  }
  (*held)->ReleaseContinuation();
  rt::IdleStateRates none = rates;
  none.spill_budget = 0;
  std::vector<jitllm::memory::ReclaimCandidate> unheld;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, none, unheld);
  EXPECT_THAT(ids(unheld), ElementsAre(0U, 1U, 3U));
}

// Room for a request slot: free bytes, else a reclaim that frees all of the
// shortfall; an unreadable budget leaves it to the capacity policy.
TEST(LlmScores, RoomForAsksTheReclaimOnlyForTheShortfall) {
  std::uint64_t asked = 0;
  const auto reclaim = [&](std::uint64_t freed) {
    return [&asked, freed](std::uint64_t shortfall) {
      asked = shortfall;
      return freed;
    };
  };
  EXPECT_TRUE(rt::RoomFor(100, 150, reclaim(0)));
  EXPECT_EQ(asked, 0U);
  EXPECT_TRUE(rt::RoomFor(100, 60, reclaim(40)));
  EXPECT_EQ(asked, 40U);
  EXPECT_FALSE(rt::RoomFor(100, 60, reclaim(39)));
  EXPECT_TRUE(rt::RoomFor(100, std::nullopt, reclaim(0)));
  EXPECT_TRUE(rt::RoomFor(0, 0, reclaim(0)));
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

TEST(LlmScores, ADiscardedScalarJudgementKeepsTheBranchsPrefix) {
  NativeBranchesFake model(true);
  auto a = model.branch(1);
  ASSERT_TRUE(a.has_value());
  std::vector<float> last;
  ASSERT_TRUE((*a)->Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
  const auto target = model.native_state(1).target;
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  rt::Generation result;
  model.failed_scalar_judgement = 1;
  EXPECT_FALSE((*a)->Generate(last, options, result).has_value());
  // The scalar step's undone verify keeps the prefix, as a wave's does.
  EXPECT_EQ((*a)->history(), target);
  EXPECT_EQ(model.native_state(1).target, target);
}

class PrefillHeadFake : public FakeLlm {
 public:
  std::vector<bool> requested_heads;
  std::vector<rt::PrefillHint> requested_hints;

 protected:
  rt::Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                std::uint32_t past, bool inject, bool want_head,
                                std::vector<float>& row, rt::PrefillHint next) override {
    requested_heads.push_back(want_head);
    requested_hints.push_back(next);
    auto ran = FakeLlm::RunPrefillChunkFor(branch, all, past, inject, want_head, row, next);
    if (ran && !want_head) row.clear();
    return ran;
  }
};
TEST(LlmScores, NonFinalPromptHeadsAreOptionalButScoringAndDefaultFamiliesStayFull) {
  const std::vector<std::int32_t> prompt(19, 2);
  PrefillHeadFake legacy;
  std::vector<float> row;
  ASSERT_TRUE(legacy.Prefill(prompt, row));
  EXPECT_EQ(legacy.requested_heads, (std::vector<bool>{false, false, true}));
  ASSERT_EQ(legacy.requested_hints.size(), 3U);
  EXPECT_EQ(legacy.requested_hints[0].rows, 8U);
  EXPECT_FALSE(legacy.requested_hints[0].want_head);
  EXPECT_EQ(legacy.requested_hints[1].rows, 3U);
  EXPECT_TRUE(legacy.requested_hints[1].want_head);
  EXPECT_EQ(legacy.requested_hints[2].rows, 0U);
  EXPECT_FALSE(row.empty());
  PrefillHeadFake incremental;
  auto opened = incremental.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  ASSERT_TRUE((*opened)->Advance());
  ASSERT_TRUE((*opened)->Advance());
  EXPECT_TRUE((*opened)->last().empty());
  (*opened)->Cancel();
  ASSERT_TRUE((*opened)->Finish());
  EXPECT_EQ(incremental.history().size(), 8U);
  opened = incremental.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  while (!(*opened)->done()) ASSERT_TRUE((*opened)->Advance());
  ASSERT_TRUE((*opened)->Finish());
  EXPECT_EQ(incremental.requested_heads, (std::vector<bool>{false, false, true}));
  ASSERT_EQ(incremental.requested_hints.size(), 3U);
  EXPECT_EQ(incremental.requested_hints[0].rows, 8U);
  EXPECT_EQ(incremental.requested_hints[1].rows, 3U);
  EXPECT_EQ(incremental.requested_hints[2].rows, 0U);
  EXPECT_EQ((*opened)->last(), row);
  PrefillHeadFake scoring;
  unsigned scores = 0;
  auto scored = scoring.default_branch().BeginScoringPrompt(
      prompt, [&](std::int32_t, std::span<const float> head) {
        EXPECT_FALSE(head.empty());
        ++scores;
        return true;
      });
  ASSERT_TRUE(scored);
  while (!(*scored)->done()) ASSERT_TRUE((*scored)->Advance());
  ASSERT_TRUE((*scored)->Finish());
  EXPECT_EQ(scores, prompt.size() - 1);
  EXPECT_FALSE(scoring.requested_heads.empty());
  EXPECT_TRUE(std::ranges::all_of(scoring.requested_heads, [](bool head) { return head; }));
  FakeLlm unchanged;
  auto full = unchanged.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(full);
  ASSERT_TRUE((*full)->Advance());
  ASSERT_TRUE((*full)->Advance());
  EXPECT_FALSE((*full)->last().empty());
  (*full)->Cancel();
  ASSERT_TRUE((*full)->Finish());
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

TEST(LlmScores, JoinedHintsPreserveOwnerOrderAndIndependentlySuppressMixedFutureHeads) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  auto a = model.branch(0), b = model.branch(1);
  ASSERT_TRUE(a && b);
  const std::vector<std::int32_t> short_prompt(16, 2), long_prompt(24, 3);
  auto x = (*a)->BeginPrompt(short_prompt), y = (*b)->BeginPrompt(long_prompt);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  // Reverse admission order: prepared ownership and its hint must sort together.
  std::array<rt::Llm::PromptSession*, 2> sessions{y->get(), x->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  ASSERT_TRUE(model.RunPromptWave(sessions, callbacks, true));
  EXPECT_THAT(model.prefill_dispatches.back(), ElementsAre(0, 1));
  ASSERT_EQ(model.delivered_prefill_hints.size(), 2U);
  const auto& first = model.delivered_prefill_hints[0];
  const auto& second = model.delivered_prefill_hints[1];
  EXPECT_EQ(first.first, 0U);
  EXPECT_EQ(first.second.rows, 8U);
  EXPECT_TRUE(first.second.want_head);
  EXPECT_EQ(first.second.after_rows, 0U);
  EXPECT_EQ(second.first, 1U);
  EXPECT_EQ(second.second.rows, 8U);
  EXPECT_FALSE(second.second.want_head);
  EXPECT_EQ(second.second.after_rows, 8U);
  EXPECT_TRUE(second.second.after_want_head);
  ASSERT_EQ(model.future_prefill_heads.size(), 1U);
  EXPECT_FALSE(model.future_prefill_heads.back().next.has_value());
  ASSERT_TRUE(model.future_prefill_heads.back().after.has_value());
  EXPECT_TRUE(*model.future_prefill_heads.back().after);
  // Only current backing/history was prepared; the hints are descriptors.
  EXPECT_EQ(model.prefill_funded[0], 8U);
  EXPECT_EQ(model.prefill_funded[1], 8U);
  EXPECT_EQ((*a)->history().size(), 8U);
  EXPECT_EQ((*b)->history().size(), 8U);
  (*x)->Cancel();
  (*y)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
}

TEST(LlmScores, JoinedHintsStopAtEachOwnersCheckpointBoundary) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  auto a = model.branch(0), b = model.branch(1);
  ASSERT_TRUE(a && b);
  const std::vector<std::int32_t> prompt(24, 2);
  auto x = (*a)->BeginPrompt(prompt, 8), y = (*b)->BeginPrompt(prompt, 16);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  ASSERT_TRUE(model.RunPromptWave(sessions, callbacks, true));
  ASSERT_EQ(model.delivered_prefill_hints.size(), 2U);
  EXPECT_EQ(model.delivered_prefill_hints[0].second.rows, 0U);
  EXPECT_EQ(model.delivered_prefill_hints[0].second.after_rows, 0U);
  EXPECT_EQ(model.delivered_prefill_hints[1].second.rows, 8U);
  EXPECT_FALSE(model.delivered_prefill_hints[1].second.want_head);
  EXPECT_EQ(model.delivered_prefill_hints[1].second.after_rows, 0U);
  EXPECT_EQ((*x)->NextUnit()->phase, rt::Llm::PromptSession::Phase::kCheckpoint);
  ASSERT_TRUE(model.future_prefill_heads.back().next.has_value());
  EXPECT_FALSE(*model.future_prefill_heads.back().next);
  EXPECT_FALSE(model.future_prefill_heads.back().after.has_value());
  (*x)->Cancel();
  (*y)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
}

TEST(LlmScores, PromptWaveReadinessSettlesColdPeerWithoutReplacingDueDecode) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  auto first = model.branch(0), second = model.branch(1);
  ASSERT_TRUE(first && second);
  const std::vector<std::int32_t> a(19, 2), b(27, 3);
  auto x = (*first)->BeginPrompt(a), y = (*second)->BeginPrompt(b);
  ASSERT_TRUE(x && y);
  std::array<rt::Llm::PromptSession*, 3> sessions{x->get(), y->get(), nullptr};
  std::array<rt::ScheduledMember, 3> members{
      rt::ScheduledMember{.admitted = 1, .remaining = 19},
      rt::ScheduledMember{.admitted = 2, .remaining = 27},
      rt::ScheduledMember{
          .stage = rt::ScheduledMember::Stage::kGeneration, .admitted = 3, .waited = 1}};
  const auto choose = [&] {
    const auto choice = rt::NextCohortUnit(members);
    return rt::Llm::PromptSession::SelectForWave(choice.decode, choice.prompt, sessions);
  };
  EXPECT_FALSE(choose());  // a due decoder keeps its original priority
  members[2].waited = 0;
  ASSERT_EQ(choose(), 0U);  // the original shortest prompt settles its reuse
  ASSERT_TRUE((*x)->Advance());
  ASSERT_EQ((*x)->NextUnit()->phase, rt::Llm::PromptSession::Phase::kChunk);
  model.prefill_waves = false;
  EXPECT_EQ(choose(), 0U);  // nonparticipating families retain scalar scheduling
  model.prefill_waves = true;
  ASSERT_EQ(choose(), 1U);  // its longer peer is readied before any prompt rows run
  EXPECT_EQ((*x)->run().end, 0U);
  ASSERT_TRUE((*y)->Advance());
  EXPECT_TRUE((*x)->CanJoin(**y));
  ASSERT_EQ(choose(), 0U);  // the original shortest chunk remains the anchor
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  ASSERT_TRUE(model.RunPromptWave(std::span(sessions).first(2), callbacks, true));
  EXPECT_EQ((*x)->run().end, 8U);
  EXPECT_EQ((*y)->run().end, 8U);
  EXPECT_THAT(model.prefill_dispatches.back(), ElementsAre(0, 1));
  (*x)->Cancel();
  (*y)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
}

TEST(LlmScores, PromptWaveReadinessDoesNotBorrowAScorerOrForeignPeer) {
  NativeBranchesFake model, foreign;
  model.prefill_waves = true;
  auto a = model.branch(0), b = model.branch(1), c = foreign.branch(0);
  ASSERT_TRUE(a && b && c);
  const std::vector<std::int32_t> prompt(19, 2);
  auto x = (*a)->BeginPrompt(prompt);
  auto y = (*b)->BeginScoringPrompt(prompt, {});
  auto z = (*c)->BeginPrompt(prompt);
  ASSERT_TRUE(x && y && z);
  ASSERT_TRUE((*x)->Advance());
  std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
  EXPECT_EQ(rt::Llm::PromptSession::SelectForWave(false, 0, sessions), 0U);
  sessions[1] = z->get();
  EXPECT_EQ(rt::Llm::PromptSession::SelectForWave(false, 0, sessions), 0U);
  (*x)->Cancel();
  (*y)->Cancel();
  (*z)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
  ASSERT_TRUE((*z)->Finish());
}

TEST(LlmScores, PromptWaveExcludesDifferentFinalRowCounts) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  auto a = model.branch(0), b = model.branch(1);
  ASSERT_TRUE(a && b);
  const std::vector<std::int32_t> short_prompt(3, 2), long_prompt(5, 3);
  auto x = (*a)->BeginPrompt(short_prompt);
  auto y = (*b)->BeginPrompt(long_prompt);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
  EXPECT_EQ(rt::Llm::PromptSession::SelectForWave(false, 0, sessions), 0U);
  ASSERT_TRUE((*y)->Advance());
  ASSERT_TRUE((*x)->NextUnit()->want_head);
  ASSERT_TRUE((*y)->NextUnit()->want_head);
  EXPECT_FALSE((*x)->CanJoin(**y));
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  EXPECT_FALSE(model.RunPromptWave(sessions, callbacks, true));
  EXPECT_TRUE(model.prefill_dispatches.empty());
  EXPECT_EQ((*x)->run().end, 0U);
  EXPECT_EQ((*y)->run().end, 0U);
  (*x)->Cancel();
  (*y)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
}

TEST(LlmScores, OneRowCheckpointPrefillStaysScalarAndSingletonRemainsUsable) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  auto a = model.branch(0), b = model.branch(1);
  ASSERT_TRUE(a && b);
  const std::vector<std::int32_t> prompt(19, 2);
  auto x = (*a)->BeginPrompt(prompt, 1), y = (*b)->BeginPrompt(prompt, 1);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  EXPECT_EQ((*x)->NextUnit()->rows, 1U);
  EXPECT_FALSE((*x)->NextUnit()->want_head);
  EXPECT_FALSE((*x)->CanJoin(**y));
  std::array<rt::Llm::PromptSession*, 1> singleton{x->get()};
  const std::array<rt::PrefillGoOn, 1> callbacks{};
  ASSERT_TRUE(model.RunPromptWave(singleton, callbacks, true));
  EXPECT_EQ((*x)->run().end, 1U);
  EXPECT_EQ((*x)->NextUnit()->phase, rt::Llm::PromptSession::Phase::kCheckpoint);
  EXPECT_EQ((*y)->run().end, 0U);
  (*x)->Cancel();
  (*y)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
}

TEST(LlmScores, CompatiblePromptWavePublishesPeersOnlyAfterSharedCompletion) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  model.prefill_funding_hook = [] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); };
  auto first = model.branch(0), second = model.branch(1);
  ASSERT_TRUE(first && second);
  const std::vector<std::int32_t> a(19, 2), b(19, 3);
  auto x = (*first)->BeginPrompt(a), y = (*second)->BeginPrompt(b);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  ASSERT_TRUE((*x)->CanJoin(**y));
  std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  for (std::uint32_t at : {8U, 16U, 19U}) {
    ASSERT_TRUE(model.RunPromptWave(sessions, callbacks, true));
    EXPECT_GE((*x)->run().longest, 0.009);
    EXPECT_GE((*y)->run().longest, 0.009);
    EXPECT_EQ((*x)->run().end, at);
    EXPECT_EQ((*y)->run().end, at);
    EXPECT_EQ((*first)->history(), model.native_state(0).target);
    EXPECT_EQ((*second)->history(), model.native_state(1).target);
    EXPECT_TRUE((*x)->last_unit_result());
    EXPECT_TRUE((*y)->last_unit_result());
  }
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
  EXPECT_EQ((*first)->history(), a);
  EXPECT_EQ((*second)->history(), b);
  EXPECT_EQ(model.prefill_dispatches.size(), 3U);
  for (const auto& owners : model.prefill_dispatches) EXPECT_THAT(owners, ElementsAre(0, 1));
}

TEST(LlmScores, PromptWaveOmitsCancelledAndCapacityRefusedPeersBeforeDispatch) {
  for (bool cancelled : {false, true}) {
    NativeBranchesFake model;
    model.prefill_waves = true;
    if (!cancelled) model.budget = 12;
    auto first = model.branch(0), second = model.branch(1);
    ASSERT_TRUE(first && second);
    const std::vector<std::int32_t> prompt(19, 2);
    auto x = (*first)->BeginPrompt(prompt), y = (*second)->BeginPrompt(prompt);
    ASSERT_TRUE(x && y);
    ASSERT_TRUE((*x)->Advance());
    ASSERT_TRUE((*y)->Advance());
    std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
    const std::array<rt::PrefillGoOn, 2> callbacks = {
        rt::PrefillGoOn{}, [cancelled](std::uint32_t) { return !cancelled; }};
    ASSERT_TRUE(model.RunPromptWave(sessions, callbacks, true));
    EXPECT_THAT(model.prefill_dispatches.back(), ElementsAre(0));
    ASSERT_EQ(model.delivered_prefill_hints.size(), 1U);
    EXPECT_EQ(model.delivered_prefill_hints.front().first, 0U);
    EXPECT_EQ(model.delivered_prefill_hints.front().second.rows, 8U);
    EXPECT_EQ(model.delivered_prefill_hints.front().second.after_rows, 3U);
    EXPECT_EQ(model.prefill_funded[0], 8U);
    EXPECT_EQ((*first)->history(), std::vector<std::int32_t>(8, 2));
    EXPECT_TRUE((*second)->history().empty());
    EXPECT_TRUE(model.native_state(1).target.empty());
    EXPECT_EQ((*y)->refused(), !cancelled);
    EXPECT_EQ((*y)->done(), cancelled);
    EXPECT_EQ((*y)->last_unit_result().has_value(), cancelled);
    if (!cancelled) {
      model.budget.reset();
      ASSERT_TRUE((*y)->Advance());  // its unchanged first chunk is retryable
      EXPECT_EQ((*second)->history(), (*first)->history());
    }
    (*x)->Cancel();
    (*y)->Cancel();
    ASSERT_TRUE((*x)->Finish());
    ASSERT_TRUE((*y)->Finish());
  }
}

TEST(LlmScores, SharedPromptFailurePublishesNoPeerHistoryAndRequiresFreshClear) {
  NativeBranchesFake model;
  model.prefill_waves = true;
  auto first = model.branch(0), second = model.branch(1);
  ASSERT_TRUE(first && second);
  const std::vector<std::int32_t> prompt(19, 2);
  auto x = (*first)->BeginPrompt(prompt), y = (*second)->BeginPrompt(prompt);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  model.native_state(1).fail_chunk = 1;
  std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  EXPECT_FALSE(model.RunPromptWave(sessions, callbacks, true));
  EXPECT_TRUE((*first)->history().empty());
  EXPECT_TRUE((*second)->history().empty());
  EXPECT_EQ(model.native_state(0).target.size(), 8U);  // dispatch completed before peer failed
  EXPECT_FALSE((*x)->Finish());
  EXPECT_FALSE((*y)->Finish());
  model.native_state(1).fail_chunk.reset();
  auto fresh = (*first)->BeginPrompt(prompt);
  ASSERT_TRUE(fresh);
  ASSERT_TRUE((*fresh)->Advance());
  EXPECT_TRUE(model.native_state(0).target.empty());  // uncommitted native rows cleared
  (*fresh)->Cancel();
  ASSERT_TRUE((*fresh)->Finish());
}

TEST(LlmScores, PromptWaveRefusesMixedModesForeignDuplicatesAndCheckpointUnits) {
  NativeBranchesFake model, foreign;
  model.prefill_waves = true;
  auto a = model.branch(0), b = model.branch(1);
  ASSERT_TRUE(a && b);
  auto x = (*a)->BeginPrompt(std::vector<std::int32_t>(7, 2));
  auto y = (*b)->BeginPrompt(std::vector<std::int32_t>(19, 3), 9);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  std::array<rt::Llm::PromptSession*, 2> sessions{x->get(), y->get()};
  EXPECT_EQ(rt::Llm::PromptSession::SelectForWave(false, 0, sessions), 0U);
  ASSERT_TRUE((*y)->Advance());
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  EXPECT_FALSE((*x)->CanJoin(**y));
  EXPECT_FALSE(model.RunPromptWave(sessions, callbacks, true));
  sessions[1] = x->get();
  EXPECT_FALSE(model.RunPromptWave(sessions, callbacks, true));
  sessions[1] = nullptr;
  EXPECT_FALSE(model.RunPromptWave(sessions, callbacks, true));
  sessions[1] = y->get();
  EXPECT_FALSE(foreign.RunPromptWave(sessions, callbacks, true));
  EXPECT_TRUE((*a)->history().empty());
  EXPECT_TRUE((*b)->history().empty());
  ASSERT_TRUE((*y)->Advance());
  ASSERT_TRUE((*y)->Advance());
  ASSERT_EQ((*y)->NextUnit()->phase, rt::Llm::PromptSession::Phase::kCheckpoint);
  EXPECT_EQ(rt::Llm::PromptSession::SelectForWave(false, 0, sessions), 0U);
  EXPECT_EQ(rt::Llm::PromptSession::SelectForWave(false, 1, sessions), 1U);
  EXPECT_FALSE(model.RunPromptWave(sessions, callbacks, true));
  EXPECT_TRUE(model.prefill_dispatches.empty());
  (*x)->Cancel();
  (*y)->Cancel();
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
}

TEST(LlmScores, APausedPartialPromptKeepsItsExactSpillPastIdleRetention) {
  NativeBranchesFake model;
  model.ConfigurePrefill(64, 8);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch.has_value());
  const std::vector<std::int32_t> prompt(19, 2);
  auto opened = (*branch)->BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  ASSERT_TRUE((*opened)->Advance().has_value());  // reuse/clear unit
  ASSERT_TRUE((*opened)->Advance().has_value());  // first complete chunk
  ASSERT_EQ((*branch)->history().size(), 8U);
  (*opened)->Cancel();
  ASSERT_TRUE((*opened)->Finish().has_value());
  (*branch)->HoldContinuation();
  ASSERT_TRUE(model.SpillSetAside(**branch).has_value());
  model.set_retention(std::chrono::seconds(0));
  EXPECT_TRUE(model.BranchIdle(**branch));  // residency can give way; logical state cannot
  EXPECT_FALSE((*branch)->ReleaseIdleState().has_value());
  auto continued = (*branch)->BeginPrompt(prompt);
  ASSERT_TRUE(continued.has_value());
  while (!(*continued)->done()) {
    ASSERT_TRUE((*continued)->Advance().has_value());
  }
  ASSERT_TRUE((*continued)->Finish().has_value());
  EXPECT_EQ((*continued)->reused(), 8U);
  EXPECT_EQ(model.restores, 1U);
  EXPECT_EQ(model.native_state(1).chunks, 3U);
  EXPECT_EQ((*branch)->history(), prompt);
  EXPECT_EQ(model.native_state(1).target, prompt);
  (*branch)->ReleaseContinuation();
  EXPECT_TRUE(model.BranchIdle(**branch));
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

// State capacity in a cohort (runtime/cohort_capacity.h): a refusal the
// caller defers leaves the session resumable at its completed prefix.
TEST(LlmScores, ACapacityRefusedPromptChunkKeepsItsPrefixAndRetriesTheSameUnit) {
  NativeBranchesFake model(true);
  auto branch = model.branch(2);
  ASSERT_TRUE(branch.has_value());
  auto& native = model.native_state(2);
  const std::vector<std::int32_t> prompt(19, 2);
  auto opened = (*branch)->BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  auto& session = **opened;
  ASSERT_TRUE(session.Advance({}, true).has_value());  // reuse: nothing to reuse
  ASSERT_TRUE(session.Advance({}, true).has_value());  // rows [0, 8)
  native.refuse_capacity = true;
  unsigned asked = 0;
  const rt::PrefillGoOn go_on = [&asked](std::uint32_t /*rows*/) {
    ++asked;
    return true;
  };
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto refused = session.Advance(go_on, true);
    ASSERT_FALSE(refused.has_value());
    EXPECT_THAT(refused.error(), HasSubstr("the chunk at 8"));
    EXPECT_THAT(refused.error(), HasSubstr("execution budget"));
    EXPECT_TRUE(session.refused());
    EXPECT_FALSE(session.done());
    EXPECT_EQ((*branch)->history(), std::vector<std::int32_t>(8, 2));
    EXPECT_EQ(native.target, (*branch)->history());
    EXPECT_EQ(session.run().chunks, 1U);
    auto unit = session.NextUnit();
    ASSERT_TRUE(unit.has_value());
    EXPECT_EQ(unit->phase, rt::Llm::PromptSession::Phase::kChunk);
    EXPECT_EQ(unit->rows, 8U);  // the same unit again
  }
  EXPECT_EQ(asked, 2U);
  native.refuse_capacity = false;
  while (!session.done()) {
    ASSERT_TRUE(session.Advance({}, true).has_value());
    EXPECT_FALSE(session.refused());
  }
  ASSERT_TRUE(session.Finish().has_value());
  EXPECT_EQ((*branch)->history(), prompt);
  EXPECT_EQ(native.target, prompt);
  EXPECT_EQ(native.injection, prompt);
  EXPECT_EQ(session.run().chunks, 3U);
  EXPECT_EQ(session.last(), FakeLlm::Row(3));
}

TEST(LlmScores, AnUndeferredCapacityRefusalEndsThePromptAtItsProvenPrefix) {
  NativeBranchesFake model;
  auto branch = model.branch(1);
  ASSERT_TRUE(branch.has_value());
  const std::vector<std::int32_t> prompt(19, 2);
  auto opened = (*branch)->BeginPrompt(prompt);
  ASSERT_TRUE(opened.has_value());
  ASSERT_TRUE((*opened)->Advance().has_value());
  ASSERT_TRUE((*opened)->Advance().has_value());
  model.native_state(1).refuse_capacity = true;
  EXPECT_FALSE((*opened)->Advance().has_value());
  EXPECT_FALSE((*opened)->refused());
  EXPECT_TRUE((*opened)->done());
  EXPECT_FALSE((*opened)->Finish().has_value());
  // The usable state keeps the chunks that completed (as before deferral).
  EXPECT_EQ((*branch)->history(), std::vector<std::int32_t>(8, 2));
}

TEST(LlmScores, ACancelledWaitingPromptAndGenerationKeepTheirCompletedPrefixes) {
  NativeBranchesFake model;
  auto prefilling = model.branch(1);
  auto decoding = model.branch(2);
  ASSERT_TRUE(prefilling.has_value());
  ASSERT_TRUE(decoding.has_value());
  const std::vector<std::int32_t> prompt(19, 2);
  auto prompt_session = (*prefilling)->BeginPrompt(prompt);
  ASSERT_TRUE(prompt_session.has_value());
  ASSERT_TRUE((*prompt_session)->Advance({}, true).has_value());
  ASSERT_TRUE((*prompt_session)->Advance({}, true).has_value());
  model.native_state(1).refuse_capacity = true;
  EXPECT_FALSE((*prompt_session)->Advance({}, true).has_value());
  ASSERT_TRUE((*prompt_session)->refused());
  (*prompt_session)->Cancel();
  ASSERT_TRUE((*prompt_session)->Finish().has_value());
  EXPECT_TRUE((*prompt_session)->run().stopped);
  EXPECT_EQ((*prefilling)->history(), std::vector<std::int32_t>(8, 2));
  EXPECT_EQ((*prefilling)->history(), model.native_state(1).target);

  std::vector<float> last;
  ASSERT_TRUE((*decoding)->Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  rt::Generation out;
  auto generation = (*decoding)->BeginGeneration(last, options, out);
  ASSERT_TRUE(generation.has_value());
  const std::array<rt::Llm::GenerationSession*, 1> one = {generation->get()};
  ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
  model.native_state(2).refuse_capacity = true;
  ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
  ASSERT_TRUE((*generation)->refused());
  EXPECT_FALSE((*generation)->done());
  (*generation)->Cancel();
  ASSERT_TRUE((*generation)->Finish().has_value());
  EXPECT_THAT(out.tokens, ElementsAre(1, 2));
  EXPECT_THAT((*decoding)->history(), ElementsAre(0, 1));
  EXPECT_EQ((*decoding)->history(), model.native_state(2).target);
}

TEST(LlmScores, ACapacityRefusedGenerationWaitsWhileItsPeerRunsThenCompletes) {
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
  auto& native_a = model.native_state(1);
  native_a.refuse_capacity = true;
  const std::array<rt::Llm::GenerationSession*, 2> both = {opened_a->get(), opened_b->get()};
  for (int wave = 0; wave < 2; ++wave) {
    ASSERT_TRUE(model.RunGenerationWave(both, true).has_value());
    EXPECT_TRUE((*opened_a)->refused());
    EXPECT_THAT((*opened_a)->refusal(), HasSubstr("execution budget"));
    EXPECT_FALSE((*opened_a)->done());
    EXPECT_FALSE((*opened_b)->refused());
  }
  EXPECT_THAT(result_a.tokens, ElementsAre(1));
  EXPECT_EQ(result_a.steps, 0U);
  EXPECT_THAT(native_a.target, ElementsAre(0));
  EXPECT_THAT(result_b.tokens, ElementsAre(3, 4, 5));
  EXPECT_TRUE((*opened_b)->done());
  ASSERT_TRUE((*opened_b)->Finish().has_value());
  // The peer's state is free now: the waiting generation goes on.
  native_a.refuse_capacity = false;
  const std::array<rt::Llm::GenerationSession*, 1> alone = {opened_a->get()};
  while (!(*opened_a)->done()) {
    ASSERT_TRUE(model.RunGenerationWave(alone, true).has_value());
    EXPECT_FALSE((*opened_a)->refused());
  }
  ASSERT_TRUE((*opened_a)->Finish().has_value());
  EXPECT_THAT(result_a.tokens, ElementsAre(1, 2, 3));
  EXPECT_EQ((*a)->history(), native_a.target);
  EXPECT_EQ((*b)->history(), model.native_state(2).target);

  // Undeferred, the same refusal ends the generation, as before.
  std::vector<float> last_c;
  auto c = model.branch(3);
  ASSERT_TRUE(c.has_value());
  ASSERT_TRUE((*c)->Prefill(std::array<std::int32_t, 1>{4}, last_c).has_value());
  rt::Generation result_c;
  auto opened_c = (*c)->BeginGeneration(last_c, options, result_c);
  ASSERT_TRUE(opened_c.has_value());
  model.native_state(3).refuse_capacity = true;
  const std::array<rt::Llm::GenerationSession*, 1> undeferred = {opened_c->get()};
  ASSERT_TRUE(model.RunGenerationWave(undeferred).has_value());
  EXPECT_FALSE((*opened_c)->refused());
  EXPECT_TRUE((*opened_c)->done());
  EXPECT_FALSE((*opened_c)->Finish().has_value());
  EXPECT_THAT((*c)->history(), ElementsAre(4));
}

TEST(LlmScores, AnIdleBranchReleasesItsRetainedStateForItsPeers) {
  NativeBranchesFake model;
  auto idle = model.branch(2);
  ASSERT_TRUE(idle.has_value());
  std::vector<float> last;
  ASSERT_TRUE((*idle)->Prefill(std::array<std::int32_t, 3>{0, 1, 2}, last).has_value());
  ASSERT_TRUE((*idle)->HasRetainedState());
  ASSERT_TRUE((*idle)->ReleaseIdleState().has_value());
  EXPECT_FALSE((*idle)->HasRetainedState());
  EXPECT_TRUE(model.native_state(2).target.empty());
  EXPECT_EQ(model.native_state(2).clearings, 1U);
  // Its next turn prefills from the start, with no further clear owed.
  ASSERT_TRUE((*idle)->Prefill(std::array<std::int32_t, 1>{4}, last).has_value());
  EXPECT_THAT((*idle)->history(), ElementsAre(4));
  EXPECT_EQ(model.native_state(2).clearings, 1U);
  // A busy branch is not idle.
  auto busy = (*idle)->BeginPrompt(std::array<std::int32_t, 2>{4, 5});
  ASSERT_TRUE(busy.has_value());
  EXPECT_DEATH(EXPECT_FALSE((*idle)->ReleaseIdleState().has_value()), "another active prompt");
  (*busy)->Cancel();
  ASSERT_TRUE((*busy)->Finish().has_value());
  // A family without separate native slots refuses, owing a clear.
  FakeLlm serial;
  ASSERT_TRUE(serial.Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
  EXPECT_FALSE(serial.default_branch().ReleaseIdleState().has_value());
  EXPECT_FALSE(serial.HasRetainedState());
  ASSERT_TRUE(serial.Prefill(std::array<std::int32_t, 1>{3}, last).has_value());
  EXPECT_EQ(serial.clearings, 1U);
}

TEST(LlmScores, APreemptedGenerationResumesFromItsRebuiltStateWithoutRepeatingTokens) {
  for (const bool sampled : {false, true}) {
    NativeBranchesFake model;
    rt::GenerateOptions options;
    options.max_tokens = 6;
    options.stop = false;
    if (sampled) {
      options.sampling =
          jitllm::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
      options.seed = 913;
    }
    // The same request, uninterrupted, on another branch.
    auto reference = model.branch(3);
    ASSERT_TRUE(reference.has_value());
    std::vector<float> last;
    ASSERT_TRUE((*reference)->Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
    rt::Generation expected;
    ASSERT_TRUE((*reference)->Generate(last, options, expected).has_value());
    ASSERT_EQ(expected.tokens.size(), 6U);

    auto branch = model.branch(1);
    ASSERT_TRUE(branch.has_value());
    std::vector<std::int32_t> visible;
    rt::GenerateOptions streamed = options;
    streamed.on_tokens = [&visible](std::span<const std::int32_t> fresh) {
      visible.insert(visible.end(), fresh.begin(), fresh.end());
      return true;
    };
    ASSERT_TRUE((*branch)->Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
    rt::Generation out;
    auto opened = (*branch)->BeginGeneration(last, streamed, out);
    ASSERT_TRUE(opened.has_value());
    const std::array<rt::Llm::GenerationSession*, 1> one = {opened->get()};
    ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
    ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
    model.native_state(1).refuse_capacity = true;
    ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
    ASSERT_TRUE((*opened)->refused());
    // The preemption: the session ends at its completed boundary, the host
    // keeps the tokens the state held, and the state is discarded.
    (*opened)->Cancel();
    ASSERT_TRUE((*opened)->Finish().has_value());
    const std::vector<std::int32_t> held = (*branch)->history();
    ASSERT_EQ(held.size(), 3U);  // the prompt and two generated; the anchor waits
    ASSERT_TRUE((*branch)->Clear().has_value());
    EXPECT_TRUE(model.native_state(1).target.empty());
    model.native_state(1).refuse_capacity = false;
    // Later: the state is rebuilt from those tokens, and the generation goes on.
    auto rebuilt = (*branch)->BeginPrompt(held);
    ASSERT_TRUE(rebuilt.has_value());
    while (!(*rebuilt)->done()) {
      ASSERT_TRUE((*rebuilt)->Advance({}, true).has_value());
    }
    ASSERT_TRUE((*rebuilt)->Finish().has_value());
    EXPECT_EQ(model.native_state(1).target, held);
    rt::Generation unused;
    EXPECT_FALSE((*branch)->ResumeGeneration((*rebuilt)->last(), streamed, unused).has_value());
    auto resumed = (*branch)->ResumeGeneration((*rebuilt)->last(), streamed, out);
    ASSERT_TRUE(resumed.has_value());
    EXPECT_FALSE(out.cancelled);
    EXPECT_EQ(out.tokens.size(), 3U);
    const std::array<rt::Llm::GenerationSession*, 1> again = {resumed->get()};
    while (!(*resumed)->done()) {
      ASSERT_TRUE(model.RunGenerationWave(again, true).has_value());
    }
    ASSERT_TRUE((*resumed)->Finish().has_value());
    EXPECT_EQ(out.tokens, expected.tokens) << "sampled " << sampled;
    EXPECT_EQ(visible, expected.tokens) << "sampled " << sampled;  // each streamed once
    EXPECT_EQ((*branch)->history(), (*reference)->history());
    EXPECT_EQ((*branch)->history(), model.native_state(1).target);
  }
}

// Spill, not clear (Llm::SpillSetAside): a generation set aside for its
// peers keeps its state, spilled, and resumes from it restored, with no
// prefill and no token chosen or streamed again.
TEST(LlmScores, AGenerationSetAsideResumesFromItsSpilledStateWithoutPrefill) {
  for (const bool sampled : {false, true}) {
    NativeBranchesFake model;
    rt::GenerateOptions options;
    options.max_tokens = 6;
    options.stop = false;
    if (sampled) {
      options.sampling =
          jitllm::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
      options.seed = 913;
    }
    auto reference = model.branch(3);
    ASSERT_TRUE(reference.has_value());
    std::vector<float> last;
    ASSERT_TRUE((*reference)->Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
    rt::Generation expected;
    ASSERT_TRUE((*reference)->Generate(last, options, expected).has_value());

    auto branch = model.branch(1);
    ASSERT_TRUE(branch.has_value());
    std::vector<std::int32_t> visible;
    rt::GenerateOptions streamed = options;
    streamed.on_tokens = [&visible](std::span<const std::int32_t> fresh) {
      visible.insert(visible.end(), fresh.begin(), fresh.end());
      return true;
    };
    ASSERT_TRUE((*branch)->Prefill(std::array<std::int32_t, 1>{0}, last).has_value());
    rt::Generation out;
    auto opened = (*branch)->BeginGeneration(last, streamed, out);
    ASSERT_TRUE(opened.has_value());
    const std::array<rt::Llm::GenerationSession*, 1> one = {opened->get()};
    ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
    ASSERT_TRUE(model.RunGenerationWave(one, true).has_value());
    (*opened)->Cancel();
    ASSERT_TRUE((*opened)->Finish().has_value());
    const std::vector<std::int32_t> held = (*branch)->history();
    ASSERT_EQ(held.size(), 3U);
    // Selected and leased (a cohort member): set aside, it is spilled all
    // the same, its tokens kept.
    const std::array<rt::Llm::Branch*, 1> selected = {*branch};
    ASSERT_TRUE(model.SelectBranches(selected).has_value());
    model.lease_held = true;
    EXPECT_FALSE(model.SpillIdle(**branch).has_value());  // in use: not idle
    (*branch)->HoldContinuation();
    ASSERT_TRUE(model.SpillSetAside(**branch).has_value());
    model.lease_held = false;
    model.set_retention(std::chrono::seconds(0));
    EXPECT_TRUE(model.BranchIdle(**branch));
    EXPECT_FALSE((*branch)->ReleaseIdleState().has_value());
    EXPECT_TRUE((*branch)->spilled());
    EXPECT_EQ(model.native_state(1).target, held);
    const unsigned chunks = model.native_state(1).chunks;
    // Begun again with `resume`: its only unit restores the state.
    auto restored = (*branch)->BeginPrompt(held, 0, false, true);
    ASSERT_TRUE(restored.has_value());
    while (!(*restored)->done()) {
      ASSERT_TRUE((*restored)->Advance({}, true).has_value());
    }
    ASSERT_TRUE((*restored)->Finish().has_value());
    EXPECT_EQ((*restored)->reused(), 3U);
    EXPECT_EQ(model.native_state(1).chunks, chunks);  // no prefill
    EXPECT_EQ(model.restores, 1U);
    EXPECT_FALSE((*branch)->spilled());
    auto resumed = (*branch)->ResumeGeneration((*restored)->last(), streamed, out);
    ASSERT_TRUE(resumed.has_value());
    const std::array<rt::Llm::GenerationSession*, 1> again = {resumed->get()};
    while (!(*resumed)->done()) {
      ASSERT_TRUE(model.RunGenerationWave(again, true).has_value());
    }
    ASSERT_TRUE((*resumed)->Finish().has_value());
    EXPECT_EQ(out.tokens, expected.tokens) << "sampled " << sampled;
    EXPECT_EQ(visible, expected.tokens) << "sampled " << sampled;
    EXPECT_EQ((*branch)->history(), (*reference)->history());
    EXPECT_EQ((*branch)->history(), model.native_state(1).target);
    (*branch)->ReleaseContinuation();
    EXPECT_TRUE(model.BranchIdle(**branch));
  }
}

std::optional<std::size_t> LargestIdle(NativeBranchesFake& model, const rt::Llm::Branch& keep);

// An idle conversation spilled for a peer's capacity keeps its history and
// state; its next turn restores the state before reusing it (refused for
// capacity, deferred like a chunk: the serial reclaim makes room and the
// same unit runs again) and continues exactly, prefilling only its new
// tokens.
TEST(LlmScores, AnIdleConversationSpilledForItsPeersContinuesExactly) {
  NativeBranchesFake model;
  model.budget = 20;
  auto one = model.branch(1);
  ASSERT_TRUE(one.has_value());
  const std::vector<std::int32_t> first = {1, 2, 3, 4, 5, 6};
  std::vector<float> last;
  std::uint32_t reused = 0;
  ASSERT_TRUE((*one)->PreparePrompt(first, 0, last, reused).has_value());
  std::vector<std::size_t> spilled;
  model.set_capacity_reclaim([&model, &spilled](const rt::Llm::Branch& refused) {
    const auto idle = LargestIdle(model, refused);
    if (!idle) {
      return false;
    }
    auto branch = model.branch(*idle);
    if (!branch || !model.SpillIdle(**branch)) {
      return false;
    }
    spilled.push_back(*idle);
    return true;
  });
  // The default branch's 16 tokens beside slot 1's 6 pass 20: slot 1 spills.
  ASSERT_TRUE(model.Prefill(std::vector<std::int32_t>(16, 2), last).has_value());
  EXPECT_THAT(spilled, ElementsAre(1U));
  EXPECT_TRUE((*one)->spilled());
  EXPECT_EQ((*one)->history(), first);
  EXPECT_EQ(model.native_state(1).target, first);
  EXPECT_EQ(model.spills, 1U);

  // Its next turn: restoring 6 beside the default's 16 passes 20, so the
  // reclaim spills the default branch (idle now) and the restore runs again.
  std::vector<std::int32_t> next = first;
  next.insert(next.end(), {7, 0});
  const unsigned chunks = model.native_state(1).chunks;
  ASSERT_TRUE((*one)->PreparePrompt(next, 0, last, reused).has_value());
  EXPECT_THAT(spilled, ElementsAre(1U, 0U));
  EXPECT_EQ(reused, 6U);
  EXPECT_EQ(model.restores, 1U);
  EXPECT_FALSE((*one)->spilled());
  EXPECT_EQ(model.native_state(1).target, next);
  EXPECT_EQ(model.native_state(1).chunks, chunks + 1);  // only its two new tokens
  EXPECT_EQ(last, FakeLlm::Row(1));
  EXPECT_TRUE(model.default_branch().spilled());
  EXPECT_EQ(model.history().size(), 16U);

  // Without room even after a reclaim, the restore's refusal ends the turn
  // typed, the branch still spilled with its history.
  model.set_capacity_reclaim({});
  model.budget = 4;
  std::vector<std::int32_t> returning(16, 2);
  returning.push_back(5);
  auto refused = model.PreparePrompt(returning, 0, last, reused);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("execution budget"));
  EXPECT_TRUE(model.default_branch().capacity_refused());
  EXPECT_TRUE(model.default_branch().spilled());
  EXPECT_EQ(model.history().size(), 16U);
  // A turn that does not continue it discards the spilled state instead.
  model.budget = 20;
  ASSERT_TRUE(model.PreparePrompt(std::vector<std::int32_t>{3, 3}, 0, last, reused).has_value());
  EXPECT_EQ(reused, 0U);
  EXPECT_FALSE(model.default_branch().spilled());
  EXPECT_THAT(model.target, ElementsAre(3, 3));
}

// The idle branch other than `keep` holding the most resident state, if
// any: idle meaning no session open and no request leasing it
// (Llm::BranchIdle), spilled state not counted (Llm::ResidentStateBytes).
std::optional<std::size_t> LargestIdle(NativeBranchesFake& model, const rt::Llm::Branch& keep) {
  std::optional<std::size_t> idle;
  std::uint64_t most = 0;
  for (std::size_t slot = 0; slot < model.branches(); ++slot) {
    auto branch = model.branch(slot);
    if (!branch || *branch == &keep || !model.BranchIdle(**branch)) {
      continue;
    }
    if (const std::uint64_t bytes = model.ResidentStateBytes(**branch); bytes > most) {
      most = bytes;
      idle = slot;
    }
  }
  return idle;
}

// Serial state capacity (Llm::set_capacity_reclaim): a serial request
// refused for capacity frees idle branches' state (here the largest
// first, dropped: the retry's mechanics; the runtime's order and its
// spilling are Server::Reclaim's), and runs the same unit again; one it
// cannot relieve ends at its completed prefix, typed, for the request alone.
void ReclaimIdleLargestFirst(NativeBranchesFake& model, std::vector<std::size_t>& released) {
  model.set_capacity_reclaim([&model, &released](const rt::Llm::Branch& refused) {
    const auto idle = LargestIdle(model, refused);
    if (!idle) {
      return false;
    }
    auto branch = model.branch(*idle);
    if (!branch || !(*branch)->ReleaseIdleState()) {
      return false;
    }
    released.push_back(*idle);
    return true;
  });
}

void HoldIdle(NativeBranchesFake& model, std::size_t slot, std::size_t tokens) {
  auto branch = model.branch(slot);
  ASSERT_TRUE(branch.has_value());
  std::vector<float> last;
  ASSERT_TRUE((*branch)->Prefill(std::vector<std::int32_t>(tokens, 1), last).has_value());
}

// What a prompt would reuse of a branch (the chat route's choice of
// branch): its live history when continued, else nothing here (no turn
// checkpoint), and nothing past the retention or from a branch owing a clear.
TEST(LlmScores, AReusablePrefixIsTheLiveHistoryAPromptContinues) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 6);
  auto one = model.branch(1);
  auto two = model.branch(2);
  ASSERT_TRUE(one.has_value());
  ASSERT_TRUE(two.has_value());
  std::vector<std::int32_t> next(6, 1);
  next.push_back(2);
  EXPECT_EQ(model.ReusablePrefix(**one, next), 6U);
  EXPECT_EQ(model.ReusablePrefix(**two, next), 0U);                             // empty
  EXPECT_EQ(model.ReusablePrefix(**one, std::vector<std::int32_t>(6, 1)), 0U);  // nothing new
  EXPECT_EQ(model.ReusablePrefix(**one, std::vector<std::int32_t>{1, 1, 3, 4}), 0U);
  model.set_retention(std::chrono::seconds(0));
  EXPECT_EQ(model.ReusablePrefix(**one, next), 0U);  // past its retention
}

TEST(LlmScores, AnIdleBranchHasNoSessionAndNoLeaseAndCountsOnlyResidentState) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 6);
  HoldIdle(model, 2, 9);
  HoldIdle(model, 3, 4);
  auto zero = model.branch(0);
  auto two = model.branch(2);
  ASSERT_TRUE(zero.has_value());
  ASSERT_TRUE(two.has_value());
  EXPECT_EQ(LargestIdle(model, **zero), 2U);
  EXPECT_EQ(LargestIdle(model, **two), 1U);
  // A branch with an open session is not idle.
  auto busy = (*two)->BeginPrompt(std::array<std::int32_t, 1>{4});
  ASSERT_TRUE(busy.has_value());
  EXPECT_FALSE(model.BranchIdle(**two));
  EXPECT_EQ(LargestIdle(model, **zero), 1U);
  (*busy)->Cancel();
  ASSERT_TRUE((*busy)->Finish().has_value());
  // Nor is one whose state a request leases.
  const std::array<rt::Llm::Branch*, 1> selected = {*two};
  ASSERT_TRUE(model.SelectBranches(selected).has_value());
  model.lease_held = true;
  EXPECT_FALSE(model.BranchIdle(**two));
  model.lease_held = false;
  EXPECT_TRUE(model.BranchIdle(**two));
  // A spilled branch holds nothing resident.
  ASSERT_TRUE(model.SpillIdle(**two).has_value());
  EXPECT_EQ(model.ResidentStateBytes(**two), 0U);
  EXPECT_EQ(model.SpilledStateBytes(**two), 9U);
  EXPECT_EQ(model.StateBytes(**two), 9U);
  EXPECT_EQ(LargestIdle(model, **zero), 1U);
  // No branch retaining state, or none but the refused one: none.
  NativeBranchesFake empty;
  auto first = empty.branch(0);
  ASSERT_TRUE(first.has_value());
  EXPECT_FALSE(LargestIdle(empty, **first).has_value());
  // A family without separate native slots has no idle branch apart.
  FakeLlm serial;
  std::vector<float> last;
  ASSERT_TRUE(serial.Prefill(std::array<std::int32_t, 2>{0, 1}, last).has_value());
  EXPECT_FALSE(serial.BranchIdle(serial.default_branch()));
}

TEST(LlmScores, AHeldUnclaimedConversationCanSpillForARestoredPeersGrowth) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 9);
  HoldIdle(model, 2, 3);
  auto paused = model.branch(1);
  auto running = model.branch(0);
  ASSERT_TRUE(paused.has_value());
  ASSERT_TRUE(running.has_value());
  const auto exact = (*paused)->history();
  (*paused)->HoldContinuation();
  model.budget = 12;
  unsigned reclaimed = 0;
  model.set_capacity_reclaim([&](const rt::Llm::Branch& keep) {
    const auto chosen = LargestIdle(model, keep);
    if (!chosen || *chosen != 1) {
      return false;
    }
    ++reclaimed;
    return model.SpillIdle(**paused).has_value();
  });
  std::vector<float> last;
  const std::array<std::int32_t, 4> growth{0, 1, 2, 3};
  ASSERT_TRUE((*running)->Prefill(growth, last).has_value());
  EXPECT_EQ(reclaimed, 1U);
  EXPECT_TRUE((*paused)->spilled());
  EXPECT_EQ((*paused)->history(), exact);
  EXPECT_EQ(model.native_state(1).target, exact);
  EXPECT_FALSE((*paused)->ReleaseIdleState().has_value());
  model.budget = 32;
  auto restored = (*paused)->BeginPrompt(exact, 0, false, true);
  ASSERT_TRUE(restored.has_value());
  while (!(*restored)->done()) {
    ASSERT_TRUE((*restored)->Advance().has_value());
  }
  ASSERT_TRUE((*restored)->Finish().has_value());
  EXPECT_EQ((*restored)->reused(), exact.size());
  EXPECT_EQ((*paused)->history(), exact);
  EXPECT_EQ(model.native_state(1).target, exact);
  (*paused)->ReleaseContinuation();
}

TEST(LlmScores, ASerialPrefillReleasesIdleStateLargestFirstAndRetriesTheRefusedChunk) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 6);
  HoldIdle(model, 2, 9);
  std::vector<std::size_t> released;
  ReclaimIdleLargestFirst(model, released);
  model.budget = 20;
  const std::vector<std::int32_t> prompt(10, 2);
  std::vector<float> last;
  std::vector<std::uint32_t> asked;
  rt::PrefillRun run;
  ASSERT_TRUE(model.Clear().has_value());
  ASSERT_TRUE(model
                  .Prefill(
                      prompt, last,
                      [&asked](std::uint32_t rows) {
                        asked.push_back(rows);
                        return true;
                      },
                      &run)
                  .has_value());
  EXPECT_THAT(released, ElementsAre(2U));   // 15 + 8 > 20; then 6 + 8 and 6 + 10 fit
  EXPECT_THAT(asked, ElementsAre(8U, 2U));  // each chunk declared once
  EXPECT_EQ(model.history(), prompt);
  EXPECT_EQ(model.target, prompt);
  EXPECT_FALSE(model.default_branch().capacity_refused());
  EXPECT_EQ(run.chunks, 2U);
  EXPECT_TRUE(model.native_state(2).target.empty());
  EXPECT_EQ(model.native_state(1).target.size(), 6U);  // not needed: kept
  auto one = model.branch(1);
  ASSERT_TRUE(one.has_value());
  EXPECT_EQ((*one)->history().size(), 6U);
}

TEST(LlmScores, ASerialPrefillThatCannotFitAloneEndsTypedAtItsCompletedPrefix) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 6);
  HoldIdle(model, 2, 9);
  std::vector<std::size_t> released;
  ReclaimIdleLargestFirst(model, released);
  model.budget = 20;
  const std::vector<std::int32_t> prompt(25, 2);
  std::vector<float> last;
  auto refused = model.Prefill(prompt, last);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("the chunk at 16"));
  EXPECT_THAT(refused.error(), HasSubstr("execution budget"));
  EXPECT_THAT(released, ElementsAre(2U, 1U));
  EXPECT_TRUE(model.default_branch().capacity_refused());
  EXPECT_TRUE(last.empty());
  // The state is usable and holds the chunks that ran; nothing owes a clear.
  EXPECT_EQ(model.history(), std::vector<std::int32_t>(16, 2));
  EXPECT_EQ(model.target, model.history());
  const unsigned clearings = model.clearings;
  ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{3}, last).has_value());
  EXPECT_EQ(model.clearings, clearings);
  EXPECT_FALSE(model.default_branch().capacity_refused());

  // Without a reclaim the same refusal ends the prefill at once, typed.
  model.set_capacity_reclaim({});
  ASSERT_TRUE(model.Clear().has_value());
  HoldIdle(model, 3, 5);
  EXPECT_FALSE(model.Prefill(std::vector<std::int32_t>(20, 2), last).has_value());
  EXPECT_TRUE(model.default_branch().capacity_refused());
  EXPECT_EQ(model.history(), std::vector<std::int32_t>(8, 2));
  EXPECT_EQ(model.native_state(3).target.size(), 5U);
  EXPECT_THAT(released, ElementsAre(2U, 1U));
}

TEST(LlmScores, ASerialNonCapacityFailureIsNeitherReclaimedNorTyped) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 6);
  std::vector<std::size_t> released;
  ReclaimIdleLargestFirst(model, released);
  model.fail_chunk = 2;
  std::vector<float> last;
  EXPECT_FALSE(model.Prefill(std::vector<std::int32_t>(12, 2), last).has_value());
  EXPECT_TRUE(released.empty());
  EXPECT_FALSE(model.default_branch().capacity_refused());
  EXPECT_TRUE(model.history().empty());  // unknown completion: a clear is owed
  EXPECT_EQ(model.native_state(1).target.size(), 6U);
}

TEST(LlmScores, ASerialPromptReleasesIdleStateAndRetriesOrEndsTyped) {
  NativeBranchesFake model;
  HoldIdle(model, 3, 7);
  std::vector<std::size_t> released;
  ReclaimIdleLargestFirst(model, released);
  model.budget = 18;
  const std::vector<std::int32_t> turn(16, 2);
  std::vector<float> last;
  std::uint32_t reused = 0;
  unsigned asked = 0;
  const rt::PrefillGoOn go_on = [&asked](std::uint32_t /*rows*/) {
    ++asked;
    return true;
  };
  ASSERT_TRUE(model.PreparePrompt(turn, 0, last, reused, go_on).has_value());
  EXPECT_THAT(released, ElementsAre(3U));  // 7 + 16 > 18
  EXPECT_EQ(model.history(), turn);
  EXPECT_EQ(last, FakeLlm::Row(3));
  EXPECT_FALSE(model.default_branch().capacity_refused());
  EXPECT_EQ(asked, 4U);  // reuse, two chunks, and the refused chunk declared again

  // Alone, a longer turn that cannot fit ends at its completed prefix.
  std::vector<std::int32_t> longer = turn;
  longer.resize(28, 3);
  auto refused = model.PreparePrompt(longer, 0, last, reused, go_on);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("execution budget"));
  EXPECT_TRUE(model.default_branch().capacity_refused());
  EXPECT_EQ(reused, 16U);
  EXPECT_EQ(model.history(), turn);
  EXPECT_TRUE(last.empty());
  EXPECT_THAT(released, ElementsAre(3U));
}

TEST(LlmScores, ASerialScoreReleasesIdleStateAndReportsEachRowOnce) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 4);
  std::vector<std::size_t> released;
  ReclaimIdleLargestFirst(model, released);
  model.budget = 8;
  const std::array<std::int32_t, 6> prompt = {0, 1, 2, 3, 4, 5};
  std::vector<float> last;
  std::vector<std::int32_t> scored;
  ASSERT_TRUE(model
                  .ScorePrompt(prompt, last,
                               [&scored](std::int32_t id, std::span<const float> /*row*/) {
                                 scored.push_back(id);
                                 return true;
                               })
                  .has_value());
  EXPECT_THAT(released, ElementsAre(1U));  // the fifth row: 4 + 5 > 8
  EXPECT_THAT(scored, ElementsAre(1, 2, 3, 4, 5));
  EXPECT_THAT(model.history(), ElementsAre(0, 1, 2, 3, 4, 5));
  EXPECT_FALSE(model.default_branch().capacity_refused());

  ASSERT_TRUE(model.Clear().has_value());
  model.budget = 3;
  scored.clear();
  auto refused =
      model.ScorePrompt(prompt, last, [&scored](std::int32_t id, std::span<const float> /*row*/) {
        scored.push_back(id);
        return true;
      });
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("fake's scoring: the row at 3: "));
  EXPECT_TRUE(model.default_branch().capacity_refused());
  EXPECT_THAT(scored, ElementsAre(1, 2, 3));  // the fourth row: 4 > 3, nothing idle
  EXPECT_THAT(model.history(), ElementsAre(0, 1, 2));
}

TEST(LlmScores, ASerialGenerationReleasesIdleStateThenEndsTypedWhenAlone) {
  NativeBranchesFake model;
  HoldIdle(model, 1, 5);
  std::vector<std::size_t> released;
  ReclaimIdleLargestFirst(model, released);
  model.budget = 12;
  std::vector<float> last;
  ASSERT_TRUE(model.Prefill(std::vector<std::int32_t>(6, 0), last).has_value());
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  std::vector<std::int32_t> visible;
  options.on_tokens = [&visible](std::span<const std::int32_t> fresh) {
    visible.insert(visible.end(), fresh.begin(), fresh.end());
    return true;
  };
  rt::Generation out;
  ASSERT_TRUE(model.Generate(last, options, out).has_value());
  EXPECT_THAT(released, ElementsAre(1U));  // the second step: 5 + 8 > 12
  EXPECT_THAT(out.tokens, ElementsAre(1, 2, 3, 4));
  EXPECT_EQ(visible, out.tokens);  // each streamed once
  EXPECT_FALSE(model.default_branch().capacity_refused());
  EXPECT_EQ(model.history().size(), 9U);

  // Alone, past the budget: ends at its completed prefix, settled.
  ASSERT_TRUE(model.Clear().has_value());
  model.budget = 8;
  ASSERT_TRUE(model.Prefill(std::vector<std::int32_t>(6, 0), last).has_value());
  rt::Generation alone;
  visible.clear();
  auto refused = model.Generate(last, options, alone);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("fake's decode step at 8: "));
  EXPECT_THAT(refused.error(), HasSubstr("execution budget"));
  EXPECT_TRUE(model.default_branch().capacity_refused());
  EXPECT_THAT(alone.tokens, ElementsAre(1, 2, 3));
  EXPECT_EQ(visible, alone.tokens);
  EXPECT_EQ(model.history().size(), 8U);  // the prompt and two processed tokens
  EXPECT_EQ(model.target, model.history());
  EXPECT_EQ(model.settlements, 2U);  // each generation's end

  // A refused generation whose settling then fails is not capacity's: the
  // branch owes a clear and the failure is untyped (the service stops).
  ASSERT_TRUE(model.Clear().has_value());
  ASSERT_TRUE(model.Prefill(std::vector<std::int32_t>(6, 0), last).has_value());
  model.fail_settle = true;
  rt::Generation unsettled;
  auto failed = model.Generate(last, options, unsettled);
  ASSERT_FALSE(failed.has_value());
  EXPECT_THAT(failed.error(), HasSubstr("settling failed"));
  EXPECT_FALSE(model.default_branch().capacity_refused());
  EXPECT_TRUE(model.history().empty());
  model.fail_settle = false;
}

}  // namespace

TEST(TokenHistory, PromptAdmissionAndRetirementChargeActualCapacity) {
  rt::RequestMemory memory(1024);
  FakeLlm model;
  model.SetTokenMemory(memory);
  const std::vector<std::int32_t> prompt(16, 2);
  auto session = model.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(session);
  EXPECT_EQ(memory.used(), prompt.size() * sizeof(std::int32_t));
  while (!(*session)->done()) {
    ASSERT_TRUE((*session)->Advance());
  }
  EXPECT_EQ(memory.used(), prompt.size() * sizeof(std::int32_t) + model.token_history_bytes());
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(memory.used(), model.history().capacity() * sizeof(std::int32_t));
  ASSERT_TRUE(model.Clear());
  EXPECT_EQ(model.history().capacity(), 0U);
  EXPECT_EQ(memory.used(), 0U);
}

TEST(TokenHistory, RefusedAdmissionPreservesAliasedCompletedPrefix) {
  rt::RequestMemory memory(64);
  FakeLlm model;
  model.SetTokenMemory(memory);
  std::vector<float> last;
  const std::vector<std::int32_t> prompt(16, 2);
  ASSERT_TRUE(model.Prefill(prompt, last));
  const auto before = model.history();
  auto refused = model.default_branch().BeginPrompt(model.history(), 0, false, true);
  EXPECT_FALSE(refused);
  EXPECT_EQ(model.history(), before);
  EXPECT_EQ(model.target, before);
  EXPECT_EQ(memory.used(), model.history().capacity() * sizeof(std::int32_t));
  ASSERT_TRUE(model.Clear());
  EXPECT_EQ(memory.used(), 0U);
}

TEST(TokenHistory, ConcurrentSessionsCannotSpendOneAnothersCapacity) {
  rt::RequestMemory memory(128);
  NativeBranchesFake model;
  model.SetTokenMemory(memory);
  auto first = model.branch(1);
  auto second = model.branch(2);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const std::vector<std::int32_t> prompt(16, 2);
  auto active = (*first)->BeginPrompt(prompt);
  ASSERT_TRUE(active);
  auto peer = (*second)->BeginPrompt(prompt);
  ASSERT_TRUE(peer);
  ASSERT_TRUE((*active)->Advance());
  auto refused = (*active)->Advance({}, true);
  EXPECT_FALSE(refused);
  EXPECT_TRUE((*active)->refused());
  EXPECT_EQ(model.native_state(1).chunks, 0U);
  (*peer)->Cancel();
  ASSERT_TRUE((*peer)->Finish());
  ASSERT_TRUE((*active)->Advance());
  (*active)->Cancel();
  ASSERT_TRUE((*active)->Finish());
  EXPECT_EQ(memory.used(), (*first)->history().capacity() * sizeof(std::int32_t));
  ASSERT_TRUE((*first)->ReleaseIdleState());
  EXPECT_EQ(memory.used(), 0U);
}

TEST(TokenHistory, GenerationRefusalStartsNoNativeWorkAndKeepsPrefix) {
  rt::RequestMemory memory(36);
  FakeLlm model;
  model.SetTokenMemory(memory);
  std::vector<float> last;
  const std::array<std::int32_t, 4> prompt{0, 1, 2, 3};
  ASSERT_TRUE(model.Prefill(prompt, last));
  rt::Generation out;
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  auto session = model.BeginGeneration(last, options, out);
  ASSERT_TRUE(session);
  const auto chunks = model.chunks;
  auto refused = (*session)->RunScalarStep(true);
  EXPECT_FALSE(refused);
  EXPECT_TRUE((*session)->refused());
  EXPECT_EQ(model.chunks, chunks);
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(model.history(), std::vector<std::int32_t>(prompt.begin(), prompt.end()));
  EXPECT_EQ(model.target, model.history());
  EXPECT_EQ(memory.used(), model.history().capacity() * sizeof(std::int32_t));
  model.Forget();
  EXPECT_EQ(memory.used(), 0U);
}

TEST(TokenHistory, SnapshotsOwnIndependentCapacityUntilInvalidation) {
  rt::RequestMemory memory(512);
  NativeBranchesFake model;
  model.SetTokenMemory(memory);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch);
  const std::vector<std::int32_t> prompt(8, 2);
  std::vector<float> last;
  ASSERT_TRUE((*branch)->Prefill(prompt, last));
  ASSERT_TRUE((*branch)->SaveState(reinterpret_cast<void*>(1)));
  EXPECT_EQ(memory.used(), 2 * prompt.size() * sizeof(std::int32_t));
  EXPECT_FALSE(model.HistoryReclaimable(**branch));
  ASSERT_TRUE((*branch)->Clear());
  EXPECT_EQ((*branch)->history().capacity(), 0U);
  EXPECT_EQ(memory.used(), prompt.size() * sizeof(std::int32_t));
  ASSERT_TRUE((*branch)->RestoreState(reinterpret_cast<void*>(1)));
  EXPECT_EQ((*branch)->history(), prompt);
  EXPECT_EQ(model.native_state(1).target, prompt);
  EXPECT_EQ(memory.used(), 2 * prompt.size() * sizeof(std::int32_t));
  (*branch)->InvalidateStateSnapshot();
  EXPECT_TRUE(model.HistoryReclaimable(**branch));
  EXPECT_EQ(memory.used(), prompt.size() * sizeof(std::int32_t));
  ASSERT_TRUE((*branch)->ReleaseIdleState());
  EXPECT_EQ(memory.used(), 0U);
}

TEST(TokenHistory, SpilledContinuationsRetainTokensUntilTheirOwnerRetires) {
  rt::RequestMemory memory(256);
  NativeBranchesFake model;
  model.SetTokenMemory(memory);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch);
  const std::vector<std::int32_t> prompt(8, 2);
  std::vector<float> last;
  ASSERT_TRUE((*branch)->Prefill(prompt, last));
  (*branch)->HoldContinuation();
  EXPECT_FALSE(model.HistoryReclaimable(**branch));
  ASSERT_TRUE(model.SpillIdle(**branch));
  EXPECT_EQ(memory.used(), (*branch)->history().capacity() * sizeof(std::int32_t));
  EXPECT_FALSE((*branch)->ReleaseIdleState());
  auto resumed = (*branch)->BeginPrompt((*branch)->history(), 0, false, true);
  ASSERT_TRUE(resumed);
  ASSERT_TRUE((*resumed)->Advance());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(model.native_state(1).target, prompt);
  (*branch)->ReleaseContinuation();
  ASSERT_TRUE((*branch)->ReleaseIdleState());
  EXPECT_EQ(memory.used(), 0U);
}

class HistoryLlm final : public FakeLlm {
 protected:
  bool LeasedFor(const Branch&) const override { return false; }
};

TEST(TokenHistory, ARepeatedLongRequestLibraryReclaimsIdleCapacity) {
  rt::RequestMemory memory(0, 4096, true);
  std::vector<std::unique_ptr<HistoryLlm>> models;
  unsigned reclaimed = 0;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    if (memory.used() + incoming > 320) {
      for (const auto& model : models) {
        if (model->HistoryReclaimable(model->default_branch()) && !model->history().empty()) {
          const auto cleared = model->Clear();
          EXPECT_TRUE(cleared);
          if (!cleared) {
            return false;
          }
          ++reclaimed;
          if (memory.used() + incoming <= 320) {
            break;
          }
        }
      }
    }
    return memory.used() + incoming <= 320;
  });
  for (unsigned index = 0; index < 24; ++index) {
    models.push_back(std::make_unique<HistoryLlm>());
    auto& model = *models.back();
    model.SetTokenMemory(memory);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      const std::vector<std::int32_t> prompt(24, static_cast<std::int32_t>(repeat));
      auto session = model.default_branch().BeginPrompt(prompt, 0, true);
      ASSERT_TRUE(session);
      while (!(*session)->done()) {
        ASSERT_TRUE((*session)->Advance());
      }
      ASSERT_TRUE((*session)->Finish());
      EXPECT_EQ(model.target, prompt);
      std::uint64_t actual = 0;
      for (const auto& item : models) {
        actual += item->history().capacity() * sizeof(std::int32_t);
      }
      EXPECT_EQ(memory.used(), actual);
      EXPECT_LE(memory.used(), 320U);
    }
  }
  EXPECT_GT(reclaimed, 0U);
  models.clear();
  EXPECT_EQ(memory.used(), 0U);
}

TEST(TokenHistory, NearBudgetGrowthRefusesInsteadOfCopyingEachGeneratedToken) {
  rt::RequestMemory memory(0, 4096, true);
  unsigned growth_requests = 0;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    if (incoming != 0) {
      ++growth_requests;
    }
    return memory.used() + incoming <= 1024;
  });
  FakeLlm model;
  model.ConfigurePrefill(512, 64);
  model.SetTokenMemory(memory);
  const std::vector<std::int32_t> prompt(64, 2);
  std::vector<float> last;
  ASSERT_TRUE(model.Prefill(prompt, last));
  rt::GenerateOptions options;
  options.max_tokens = 100;
  options.stop = false;
  rt::Generation out;
  auto session = model.BeginGeneration(last, options, out);
  ASSERT_TRUE(session);
  const auto before = growth_requests;
  EXPECT_FALSE((*session)->RunScalarStep(true));
  EXPECT_TRUE((*session)->refused());
  EXPECT_EQ(growth_requests - before, 1U);  // no repeated exact-size fallback
  EXPECT_EQ(model.chunks, 1U);
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(model.history(), prompt);
}

TEST(LlmScores, ScalarUnitPublicationKeepsEarlierSuccessOnLaterCleanDispatchRefusal) {
  NativeBranchesFake model;
  model.scalar_units = true;
  const auto a = *model.branch(1), b = *model.branch(2), reference = *model.branch(3);
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  options.seed = 73;
  options.sampling = jitllm::execution::SamplingParams{.temperature = 0.7F, .top_k = 4};
  std::vector<float> last;
  ASSERT_TRUE(reference->Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::Generation expected;
  ASSERT_TRUE(reference->Generate(last, options, expected));
  ASSERT_TRUE(a->Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::Generation out_a, out_b;
  std::vector<std::int32_t> visible;
  auto streamed = options;
  streamed.on_tokens = [&](std::span<const std::int32_t> ids) {
    visible.insert(visible.end(), ids.begin(), ids.end());
    return true;
  };
  auto sa = a->BeginGeneration(last, options, out_a);
  ASSERT_TRUE(sa);
  ASSERT_TRUE(b->Prefill(std::array<std::int32_t, 1>{0}, last));
  auto sb = b->BeginGeneration(last, streamed, out_b);
  ASSERT_TRUE(sb);
  model.refuse_scalar_dispatch = 2;
  const std::array<rt::Llm::GenerationSession*, 2> both{sa->get(), sb->get()};
  const auto ran = model.RunGenerationWave(both);
  EXPECT_TRUE(ran);
  EXPECT_EQ(out_a.tokens.size(), 2U);
  EXPECT_EQ(out_b.tokens.size(), 1U);
  EXPECT_EQ(model.native_state(1).chunks, 2U);
  EXPECT_EQ(model.native_state(2).chunks, 1U);
  EXPECT_TRUE((*sb)->done());
  EXPECT_FALSE((*sb)->Finish());
  sb->reset();
  EXPECT_EQ(b->history(), model.native_state(2).target);
  model.refuse_scalar_dispatch.reset();
  model.native_state(2).refuse_capacity = false;
  auto resumed = b->ResumeGeneration({}, streamed, out_b);
  ASSERT_TRUE(resumed);
  const std::array<rt::Llm::GenerationSession*, 2> again{sa->get(), resumed->get()};
  while (!(*sa)->done() && !(*resumed)->done()) EXPECT_TRUE(model.RunGenerationWave(again));
  const std::array<rt::Llm::GenerationSession*, 1> remaining{resumed->get()};
  while (!(*resumed)->done()) EXPECT_TRUE(model.RunGenerationWave(remaining));
  EXPECT_TRUE((*sa)->Finish());
  EXPECT_TRUE((*resumed)->Finish());
  EXPECT_EQ(out_a.tokens, expected.tokens);
  EXPECT_EQ(out_b.tokens, expected.tokens);
  EXPECT_EQ(visible, expected.tokens);
  EXPECT_EQ(a->history(), b->history());
}

TEST(LlmScores, PerRequestChatStopsDoNotChangeLiteralStops) {
  FakeLlm model;
  std::vector<float> last;
  const std::array<std::int32_t, 1> channel{1};
  rt::GenerateOptions chat;
  chat.max_tokens = 3;
  chat.extra_stops = channel;
  ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::Generation stopped;
  ASSERT_TRUE(model.Generate(last, chat, stopped));
  EXPECT_TRUE(stopped.stopped);
  EXPECT_EQ(stopped.steps, 0U);
  ASSERT_TRUE((*model.branch(0))->Clear());
  ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::GenerateOptions literal;
  literal.max_tokens = 3;
  rt::Generation ordinary;
  ASSERT_TRUE(model.Generate(last, literal, ordinary));
  EXPECT_FALSE(ordinary.stopped);
  EXPECT_THAT(ordinary.tokens, ElementsAre(1, 2, 3));
}
