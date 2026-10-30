// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The LLM serving adapter (runtime/serving.h) on the fake device
// backend: Llm's own conversation code (prompt sessions, generation
// sessions, waves, spills, turn checkpoints, kept conversations) over the
// shared runner, its step programs the fake step source's in its logits
// mode (tests/support/fake_step_source.h). A head row there names a token
// that follows from the row's token and position alone, but only while the
// target's state holds every earlier position; so an uninterrupted chain
// of choices is what every generation must reproduce exactly: plain,
// greedy on the device, speculative with two drafters (an MTP-like one
// that catches up and a DSpark-like block one, cascaded), joined in waves,
// after a spill, a turn checkpoint's restore, a recovery, and a restart
// that adopts the conversation from its kept record (D-105, version 2: a
// spill file a stage), where an old record of version 1 is refused.

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>

#include "llm_fixture.h"

namespace {
using namespace llmp::test_support::llm_fixture;
class EngineLlmTest : public LlmFixture {};

// Snapshot payload and variable metadata have one catalog charge. These are
// integration controls over the real slot copy/restore path on the fake node;
// no test receives a writable pointer into the opaque snapshot.
TEST_F(EngineLlmTest, MixedVerifyKeepsEachOwnersDeviceOrSampledOutputAndCompletedChain) {
  native_outputs_ = true;
  mixed_outputs_ = true;
  Start(true, true);
  const std::array prompts{Prompt(17, 3), Prompt(23, 5)};
  Open({&BranchAt(0), &BranchAt(1)});
  std::array<rt::Generation, 2> outputs;
  std::array<rt::GenerateOptions, 2> owned_options{};
  const auto finish = [](rt::Llm::GenerationSession* session) {
    if (!session) return;
    if (!session->done()) session->Cancel();
    (void)session->Finish();
    delete session;
  };
  std::array<std::unique_ptr<rt::Llm::GenerationSession, decltype(finish)>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    std::vector<float> head;
    std::uint32_t reused = 0;
    ASSERT_TRUE(BranchAt(i).PreparePrompt(prompts[i], 0, head, reused));
    auto& options = owned_options[i];
    options.max_tokens = 16;
    options.stop = false;
    if (i == 1)
      options.sampling =
          llmp::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
    auto opened = BranchAt(i).BeginGeneration(head, options, outputs[i]);
    ASSERT_TRUE(opened);
    sessions[i].reset(opened->release());
  }
  for (unsigned guard = 0; guard < 32; ++guard) {
    std::vector<rt::Llm::GenerationSession*> active;
    for (auto& session : sessions)
      if (!session->done()) active.push_back(session.get());
    if (active.empty()) break;
    ASSERT_TRUE(llm_->RunGenerationWave(active));
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    ASSERT_TRUE(sessions[i]->done());
    ASSERT_TRUE(sessions[i]->Finish());
    EXPECT_EQ(outputs[i].tokens, Chain(prompts[i], 16));
  }
  EXPECT_TRUE(std::ranges::any_of(source_->built(), [](const auto& built) {
    return !built.measuring && built.shape.phase == pl::Phase::kVerify &&
           built.shape.segments.size() == 2 &&
           built.shape.segments[0].greedy != built.shape.segments[1].greedy;
  }));
  Close();
}

TEST_F(EngineLlmTest, MixedCausalUnitKeepsEachRoleResultHistoryAndNextStep) {
  mixed_causal_ = mixed_outputs_ = true;
  Start(false, false);
  Open({&BranchAt(0), &BranchAt(1)});
  auto ready = ReadyCausal();
  ASSERT_TRUE(ready) << Failed(ready).value_or("");
  auto& sessions = **ready;
  const auto before = source_->executed();
  const std::array prompts{sessions.prompt.get()};
  const std::array generations{sessions.generation.get()};
  const std::array<rt::PrefillGoOn, 1> callbacks{};
  ASSERT_TRUE(llm_->RunCausalWave(prompts, callbacks, generations));
  EXPECT_EQ(source_->executed(), before + 1);
  EXPECT_EQ(llm_->counts().joined_causal_steps, 1U);
  EXPECT_EQ(llm_->counts().joined_causal_rows, sessions.incoming.size() + 1);
  EXPECT_EQ(BranchAt(1).history(), sessions.incoming);
  EXPECT_EQ(sessions.out.tokens, Chain(sessions.original, 2));
  EXPECT_EQ(llm_->runner().slots().positions(0), sessions.original.size() + 1);
  EXPECT_EQ(llm_->runner().slots().positions(1), sessions.incoming.size());
  EXPECT_TRUE(std::ranges::any_of(source_->built(), [](const auto& built) {
    return !built.measuring && built.shape.phase == pl::Phase::kPrefill &&
           built.shape.segments.size() == 2 &&
           built.shape.segments[0].origin_phase == pl::Phase::kDecode &&
           built.shape.segments[0].greedy && !built.shape.segments[1].greedy;
  }));
  ASSERT_TRUE(sessions.prompt->done());
  const auto last = sessions.prompt->last();
  ASSERT_TRUE(sessions.prompt->Finish());
  while (!sessions.generation->done()) ASSERT_TRUE(llm_->RunGenerationWave(generations));
  ASSERT_TRUE(sessions.generation->Finish());
  EXPECT_EQ(sessions.out.tokens, Chain(sessions.original, sessions.options.max_tokens));
  rt::Generation next;
  ASSERT_TRUE(BranchAt(1).Generate(last, sessions.options, next));
  EXPECT_EQ(next.tokens, Chain(sessions.incoming, sessions.options.max_tokens));
  Close();
}

TEST_F(EngineLlmTest, MixedCausalPromptCancellationLeavesItsPeerAndCursorExact) {
  mixed_causal_ = mixed_outputs_ = true;
  Start(false, false);
  Open({&BranchAt(0), &BranchAt(1)});
  auto ready = ReadyCausal();
  ASSERT_TRUE(ready) << Failed(ready).value_or("");
  auto& sessions = **ready;
  const auto before = source_->executed();
  unsigned calls = 0;
  const std::array<rt::PrefillGoOn, 1> callbacks{[&](std::uint32_t rows) {
    EXPECT_EQ(rows, sessions.incoming.size());
    ++calls;
    return false;
  }};
  const std::array prompts{sessions.prompt.get()};
  const std::array generations{sessions.generation.get()};
  ASSERT_TRUE(llm_->RunCausalWave(prompts, callbacks, generations));
  EXPECT_EQ(calls, 1U);
  EXPECT_EQ(source_->executed(), before + 1);
  EXPECT_TRUE(sessions.prompt->done());
  EXPECT_EQ(llm_->runner().slots().positions(1), 0U);
  EXPECT_TRUE(BranchAt(1).history().empty());
  EXPECT_EQ(sessions.out.tokens, Chain(sessions.original, 2));
  ASSERT_TRUE(sessions.prompt->Finish());
  while (!sessions.generation->done()) ASSERT_TRUE(llm_->RunGenerationWave(generations));
  ASSERT_TRUE(sessions.generation->Finish());
  EXPECT_EQ(sessions.out.tokens, Chain(sessions.original, sessions.options.max_tokens));
  EXPECT_EQ(calls, 1U);
  Close();
}

TEST_F(EngineLlmTest, MixedCausalUnknownWorkPublishesNeitherOwnersOutputs) {
  mixed_causal_ = mixed_outputs_ = true;
  Start(false, false);
  Open({&BranchAt(0), &BranchAt(1)});
  auto ready = ReadyCausal();
  ASSERT_TRUE(ready) << Failed(ready).value_or("");
  auto& sessions = **ready;
  source_->options().positioned = true;
  source_->options().fail_run = kn::StepError::kUnknown;
  source_->options().fail_phase = pl::Phase::kPrefill;
  const auto before = source_->executed();
  const auto output_before = sessions.out.tokens;
  const std::array prompts{sessions.prompt.get()};
  const std::array generations{sessions.generation.get()};
  const std::array<rt::PrefillGoOn, 1> callbacks{};
  (void)llm_->RunCausalWave(prompts, callbacks, generations);
  EXPECT_EQ(source_->executed(), before + 1);  // Real queued numeric fault fired.
  EXPECT_FALSE(sessions.prompt->last_unit_result());
  EXPECT_EQ(sessions.out.tokens, output_before);
  EXPECT_FALSE(llm_->runner().slots().usable(0));
  EXPECT_FALSE(llm_->runner().slots().usable(1));
  EXPECT_TRUE(sessions.prompt->done());
  EXPECT_TRUE(sessions.generation->done());
  EXPECT_FALSE(sessions.prompt->Finish());
  EXPECT_FALSE(sessions.generation->Finish());
  sessions.prompt.reset();
  sessions.generation.reset();
  source_->options().fail_run.reset();
  std::ignore = node_->EndRequest(0);
  ASSERT_TRUE(node_->Job(
      llmp::catalog::Closure{},
      [](llmp::providers::NativeStream) { return llmp::scheduler::JobResult::kQueued; },
      "fencing the test's stream", 0));
  ASSERT_TRUE(llm_->RecoverInPlace());
}

TEST_F(EngineLlmTest, ScoringBatchesGrowPastDecodeCapacityWithoutLosingAnyLikelihoodRow) {
  private_score_ = true;
  Start(false, false);
  const auto prompt = Prompt(2 * kChunk + 5, 3);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> last;
  std::size_t scores = 0;
  rt::PrefillRun run;
  const auto scored = branch.ScorePrompt(
      prompt, last,
      [&](std::int32_t token, std::span<const float> row) {
        EXPECT_EQ(token, prompt[scores + 1]);
        EXPECT_EQ(ArgmaxOf(row),
                  ts::FakeLogit(prompt[scores], static_cast<std::int32_t>(scores), kWidth));
        ++scores;
        return true;
      },
      {}, &run);
  ASSERT_TRUE(scored) << Why(scored);
  EXPECT_EQ(scores, prompt.size() - 1);
  EXPECT_EQ(run.end, prompt.size());
  EXPECT_EQ(run.published, prompt.size() - 1);
  EXPECT_LT(run.chunks, prompt.size());
  EXPECT_EQ(branch.history(), prompt);
  EXPECT_EQ(ArgmaxOf(last),
            ts::FakeLogit(prompt.back(), static_cast<std::int32_t>(prompt.size() - 1), kWidth));
  EXPECT_TRUE(std::ranges::any_of(source_->built(), [](const auto& build) {
    return build.shape.phase == pl::Phase::kScore && build.shape.segments.front().rows > 1;
  }));
  Close();
}

TEST_F(EngineLlmTest, ScoringCapacityBackoffRetriesWholeUntouchedBatchAndPublishesAllRows) {
  private_score_ = true;
  score_workspace_ = true;
  // Actual F32 head bytes, not an injected refusal or an invented row quota.
  // The ordinary one-row-per-owner setup fits; larger batches must compete
  // for the same 128-extent node budget as device, DMA and owned outputs.
  score_width_ = 1U << 20;
  Start(false, false);
  const auto prompt = Prompt(kChunk + 5, 9);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::size_t scores = 0;
  auto session =
      branch.BeginScoringPrompt(prompt, [&](std::int32_t token, std::span<const float> row) {
        EXPECT_EQ(token, prompt[scores + 1]);
        EXPECT_EQ(row.size(), score_width_);
        EXPECT_EQ(ArgmaxOf(row),
                  ts::FakeLogit(prompt[scores], static_cast<std::int32_t>(scores), score_width_));
        ++scores;
        return true;
      });
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->Advance());  // empty-state reuse
  const auto queued = source_->queued();
  ASSERT_TRUE((*session)->Advance()) << Why((*session)->last_unit_result());
  std::uint32_t measured = 0;
  for (const auto& built : source_->built())
    if (built.shape.phase == pl::Phase::kScore)
      measured = std::max(measured, built.shape.segments.front().rows);
  EXPECT_GT(measured, (*session)->run().end);
  EXPECT_GT((*session)->run().end, 1U);
  EXPECT_EQ(source_->queued(), queued + 1);  // refused candidates had no numeric child
  EXPECT_EQ(source_->executed(), queued + 1);
  while (!(*session)->done()) ASSERT_TRUE((*session)->Advance());
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(scores, prompt.size() - 1);
  EXPECT_EQ(branch.history(), prompt);
  EXPECT_FALSE(branch.capacity_refused());
  Close();
}

TEST_F(EngineLlmTest, ScoringSourceRefusalIsNotRetriedAsCapacityOrPublished) {
  private_score_ = true;
  Start(false, false);
  const auto prompt = Prompt(kChunk + 5, 4);
  auto& branch = BranchAt(0);
  Open({&branch});
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.phase == pl::Phase::kScore;
  };
  std::uint32_t published = 0;
  std::vector<float> last;
  const auto queued = source_->queued();
  auto result = branch.ScorePrompt(prompt, last, [&](std::int32_t, std::span<const float>) {
    ++published;
    return true;
  });
  ASSERT_FALSE(result);
  EXPECT_EQ(published, 0U);
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_TRUE(branch.history().empty());
  EXPECT_FALSE(branch.capacity_refused());
  EXPECT_EQ(std::ranges::count_if(
                source_->built(),
                [](const auto& built) { return built.shape.phase == pl::Phase::kScore; }),
            1);
  source_->options().refuse_build = {};
  ASSERT_TRUE(branch.Prefill(prompt, last));
  EXPECT_EQ(branch.history(), prompt);  // old model closure is still usable
  Close();
}

TEST_F(EngineLlmTest, ScoringCallbackStopKeepsRetiredRowsThroughSpillAndResumedPublication) {
  private_score_ = true;
  Start(false, false);
  const auto prompt = Prompt(2 * kChunk + 5, 7);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<std::int32_t> ids;
  const auto callback = [&](std::int32_t token, std::span<const float> row) {
    EXPECT_EQ(token, prompt[ids.size() + 1]);
    EXPECT_EQ(ArgmaxOf(row),
              ts::FakeLogit(prompt[ids.size()], static_cast<std::int32_t>(ids.size()), kWidth));
    ids.push_back(token);
    return ids.size() != 2;
  };
  auto first = branch.BeginScoringPrompt(prompt, callback);
  ASSERT_TRUE(first) << ts::Failed(first).value_or("");
  while (!(*first)->done()) ASSERT_TRUE((*first)->Advance());
  ASSERT_TRUE((*first)->Finish());
  EXPECT_TRUE((*first)->run().stopped);
  EXPECT_EQ((*first)->run().published, 2U);
  EXPECT_GT((*first)->run().end, 2U);
  const auto end = (*first)->run().end;
  const auto queued = source_->queued();
  EXPECT_EQ(branch.history().size(), end);
  first->reset();
  Close();
  branch.HoldContinuation();
  ASSERT_TRUE(llm_->SpillIdle(branch));
  Open({&branch});
  auto resumed = branch.BeginScoringPrompt(prompt, callback, true);
  ASSERT_TRUE(resumed);
  ASSERT_TRUE((*resumed)->Advance());  // actual state restore/reuse
  const auto after_restore = source_->queued();
  ASSERT_FALSE((*resumed)->done());
  const auto next = (*resumed)->NextUnit();
  ASSERT_TRUE(next);
  EXPECT_EQ(next->phase, rt::Llm::PromptSession::Phase::kScores);
  ASSERT_TRUE((*resumed)->Advance());  // publish remaining owned rows, no native work
  EXPECT_EQ(source_->queued(), after_restore);
  EXPECT_GE(after_restore, queued);
  while (!(*resumed)->done()) ASSERT_TRUE((*resumed)->Advance());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(ids.size(), prompt.size() - 1);
  EXPECT_EQ(branch.history(), prompt);
  Close();
  branch.ReleaseContinuation();
}

TEST_F(EngineLlmTest, AnUnbuiltStateEstimateIsAnErrorNotZeroBytes) {
  en::PagedNode node(en::NodeSettings{});
  rt::Llm model(node, entry_, Settings(false), roles_, 0, Build(false));
  auto bytes = model.StateBytesThrough(100);
  ASSERT_FALSE(bytes.has_value());
  EXPECT_NE(bytes.error().find("not built"), std::string::npos);
}

TEST_F(EngineLlmTest, PostFactoryExecutionSealGrowthRemainsInStartupAccounting) {
  class SealingSource final : public en::StepSource {
   public:
    SealingSource(std::unique_ptr<en::StepSource> source, std::function<en::Status()> seal)
        : source_(std::move(source)), seal_(std::move(seal)) {}
    pl::PlanKey KeyBase() const override { return source_->KeyBase(); }
    en::Status PrepareExecution() override {
      if (auto charged = seal_(); !charged) return charged;
      return source_->PrepareExecution();
    }
    std::expected<std::optional<llmp::base::Sha256Digest>, std::string> ExecutionIdentity()
        const override {
      return source_->ExecutionIdentity();
    }
    void EmitShapeParameters(const pl::StepShape& shape,
                             llmp::model::ShapeParameterSink& sink) const override {
      source_->EmitShapeParameters(shape, sink);
    }
    std::expected<en::SpecializationChange, en::SpecializationError> NextSpecializationChange(
        const en::SpecializationRequest& request) const override {
      return source_->NextSpecializationChange(request);
    }

    std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(
        const pl::StepShape& shape, const kn::Addresses& addresses) override {
      return source_->Build(shape, addresses);
    }

   private:
    std::unique_ptr<en::StepSource> source_;
    std::function<en::Status()> seal_;
  };
  node_ = std::make_unique<en::PagedNode>(
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
  ASSERT_TRUE(node_->Open());
  auto paid = en::ExecutionMetadataAllowance::AdmitWithin(
      node_->catalog(), node_->domain(), kExtent, 128 * kExtent, 128 * kExtent, kExtent);
  ASSERT_TRUE(paid);
  auto payment = *paid;
  constexpr std::uint64_t fixed_host = 4096;
  const auto before_factory = fixed_host + payment->capacity() + payment->OverheadBytes();
  std::uint64_t factory_bytes = 0;
  auto build = [&, original = Build(false)](const rt::ModelSettings& settings) mutable
      -> std::expected<rt::LlmBuildParts, std::string> {
    auto parts = original(settings);
    if (!parts) return parts;
    parts->startup_host_bytes = fixed_host;
    parts->execution_metadata = payment;
    parts->source = std::make_unique<SealingSource>(std::move(parts->source), [&, payment] {
      // Production source sealing happens here, after the factory returns.
      return payment->ChargeOrGrow(node_->catalog(), node_->domain(), 2 * kExtent, 128 * kExtent,
                                   128 * kExtent, kExtent);
    });
    factory_bytes = fixed_host + payment->capacity() + payment->OverheadBytes();
    return parts;
  };
  llm_ = std::make_unique<rt::Llm>(*node_, entry_, Settings(false), roles_, 0, std::move(build));
  ASSERT_TRUE(llm_->Setup());
  EXPECT_EQ(factory_bytes, before_factory);
  EXPECT_GT(llm_->startup_host_bytes(), factory_bytes);
  EXPECT_EQ(llm_->startup_host_bytes(),
            fixed_host + payment->capacity() + payment->OverheadBytes());
  const auto extents = node_->catalog().ExtentsOf(node_->domain()).size();
  ASSERT_TRUE(payment->ChargeOrGrow(node_->catalog(), node_->domain(), kExtent, 128 * kExtent,
                                    128 * kExtent, kExtent));
  EXPECT_EQ(node_->catalog().ExtentsOf(node_->domain()).size(), extents);
  EXPECT_EQ(llm_->startup_host_bytes(),
            fixed_host + payment->capacity() + payment->OverheadBytes());
  Stop();
}

TEST_F(EngineLlmTest, OpaqueSnapshotRestoresCapturedBytesAndHistoryWithOneCharge) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  const auto prompt = Prompt(17, 3);
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(prompt, logits));
  const auto occupancy = [&] {
    std::uint64_t value = 0;
    EXPECT_TRUE(node_->Call(
        [&]() -> rt::Status {
          value = node_->catalog().OccupancyOf(node_->domain()).Total().value();
          return {};
        },
        "snapshot test occupancy"));
    return value;
  };
  const auto before = occupancy();
  {
    rt::SnapshotStore store(*node_, 8 * kExtent);
    auto saved = branch.CaptureState(store);
    ASSERT_TRUE(saved) << Failed(saved).value_or("");
    auto original = store.Digest(*saved);
    ASSERT_TRUE(original) << Failed(original).value_or("");
    EXPECT_EQ(occupancy() - before, store.capacity());
    const auto tail = Prompt(3, 7);
    ASSERT_TRUE(branch.Prefill(tail, logits));
    EXPECT_NE(branch.history(), prompt);
    ASSERT_TRUE(branch.RestoreState(store, *saved));
    EXPECT_EQ(branch.history(), prompt);
    EXPECT_EQ(llm_->runner().slots().positions(0), prompt.size());
    // Capture again through the same owned allocation: old capability expires,
    // while the actual copied bytes match the first completed capture.
    auto again = branch.CaptureState(store);
    ASSERT_TRUE(again) << Failed(again).value_or("");
    EXPECT_FALSE(store.Digest(*saved));
    auto repeated = store.Digest(*again);
    ASSERT_TRUE(repeated) << Failed(repeated).value_or("");
    EXPECT_EQ(*repeated, *original);
  }
  // Plans/state may grow with the tail, so only the snapshot's exact charge is
  // checked while its lifetime is isolated, above. Continuation checks actual
  // logical state rather than only the diagnostic byte hash.
  rt::GenerateOptions options;
  options.max_tokens = 4;
  rt::Generation generated;
  // The captured frontier's logits are reconstructed by a one-row score-free
  // continuation prompt; the state prefix remains the captured history.
  const auto next = Prompt(1, 9);
  ASSERT_TRUE(branch.Prefill(next, logits));
  auto continued = prompt;
  continued.insert(continued.end(), next.begin(), next.end());
  ASSERT_TRUE(branch.Generate(logits, options, generated));
  EXPECT_EQ(generated.tokens, Chain(continued, 4));
  Close();
}

TEST_F(EngineLlmTest, RequiredExecutionIdentityRefusesBeforeSnapshotEffects) {
  const auto original = llmp::base::Sha256().Update(std::string_view("owned fake recipe")).Finish();
  execution_identity_ = std::optional{original};
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(9, 2), logits));
  rt::SnapshotStore store(*node_, 8 * kExtent);
  auto saved = branch.CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  auto digest = store.Digest(*saved);
  ASSERT_TRUE(digest);
  const auto history = branch.history();
  const auto position = llm_->runner().slots().positions(0);
  const auto capacity = store.capacity();
  for (int mutation = 0; mutation < 3; ++mutation) {
    SCOPED_TRACE(mutation);
    if (mutation == 0) {
      auto changed = original;
      changed[0] ^= 1;
      execution_identity_ = std::optional{changed};
    } else if (mutation == 1) {
      execution_identity_ = std::nullopt;
    } else {
      execution_identity_ = std::unexpected("backend is not ready");
    }
    EXPECT_FALSE(llm_->ExecutionIdentity());
    EXPECT_FALSE(branch.CaptureState(store));
    EXPECT_FALSE(branch.RestoreState(store, *saved));
    EXPECT_EQ(branch.history(), history);
    EXPECT_EQ(llm_->runner().slots().positions(0), position);
    EXPECT_EQ(store.capacity(), capacity);
    EXPECT_EQ(store.Digest(*saved), digest);
  }
  execution_identity_ = std::optional{original};
  ASSERT_TRUE(branch.RestoreState(store, *saved));
  auto repeated = branch.CaptureState(store);
  ASSERT_TRUE(repeated);
  EXPECT_EQ(store.Digest(*repeated), digest);
  Close();
}

TEST_F(EngineLlmTest, OpaqueSnapshotRejectsForeignStoreAndOwnerBeforeAnyRestore) {
  Start(false, false);
  auto& a = BranchAt(0);
  auto& b = BranchAt(1);
  Open({&a, &b});
  std::vector<float> logits;
  ASSERT_TRUE(a.Prefill(Prompt(7, 1), logits));
  ASSERT_TRUE(b.Prefill(Prompt(7, 8), logits));
  rt::SnapshotStore first(*node_, 8 * kExtent);
  rt::SnapshotStore second(*node_, 8 * kExtent);
  auto saved = a.CaptureState(first);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto a_history = a.history();
  const auto b_history = b.history();
  const auto a_position = llm_->runner().slots().positions(0);
  const auto b_position = llm_->runner().slots().positions(1);
  EXPECT_FALSE(second.Digest(*saved));
  EXPECT_FALSE(a.RestoreState(second, *saved));
  EXPECT_FALSE(b.RestoreState(first, *saved));
  EXPECT_EQ(a.history(), a_history);
  EXPECT_EQ(b.history(), b_history);
  EXPECT_EQ(llm_->runner().slots().positions(0), a_position);
  EXPECT_EQ(llm_->runner().slots().positions(1), b_position);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
  EXPECT_FALSE(llm_->runner().slots().owed(1));
  EXPECT_TRUE(first.Digest(*saved));
  Close();
}

TEST_F(EngineLlmTest, OpaqueSnapshotCanRecaptureAfterIssuerExpiryWithoutRevivingOldCapability) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(9, 2), logits));
  rt::SnapshotStore store(*node_, 8 * kExtent);
  auto saved = branch.CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto capacity = store.capacity();
  branch.InvalidateStateSnapshot();
  EXPECT_FALSE(store.Digest(*saved));
  EXPECT_FALSE(branch.RestoreState(store, *saved));
  auto current = branch.CaptureState(store);
  ASSERT_TRUE(current) << Failed(current).value_or("");
  EXPECT_EQ(store.capacity(), capacity);
  EXPECT_TRUE(store.Digest(*current));
  EXPECT_FALSE(store.Digest(*saved));
  Close();
}

TEST_F(EngineLlmTest, OpaqueSnapshotStoreAndCapabilityMayOutliveNodeAndBranch) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(9, 2), logits));
  auto store = std::make_unique<rt::SnapshotStore>(*node_, 8 * kExtent);
  auto saved = branch.CaptureState(*store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  Close();
  const auto observer = node_->staging_lifetime().lock();
  ASSERT_TRUE(observer);
  Stop();  // node and branch retire while the non-owning capability remains
  EXPECT_FALSE(observer->live());
  EXPECT_FALSE(store->usable());
  EXPECT_FALSE(store->Digest(*saved));
  store.reset();  // must neither call the expired node nor reread freed bytes
}

TEST_F(EngineLlmTest, OpaqueSnapshotCeilingIncludesMetadataAndAllocatesNothingOnRefusal) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(7, 4), logits));
  rt::SnapshotStore store(*node_, 1);
  const auto history = branch.history();
  auto refused = branch.CaptureState(store);
  EXPECT_FALSE(refused);
  EXPECT_EQ(store.capacity(), 0U);
  EXPECT_EQ(branch.history(), history);
  EXPECT_EQ(llm_->runner().slots().positions(0), history.size());
  Close();
}

// Prompts in chunks (state-only cuts, the last headed) and plain decoding,
// its greedy tokens chosen on the device or, with logits kept, by the host
// from each row; the next turn reuses the whole live history.
TEST_F(EngineLlmTest, PromptUnitsAndPlainGenerationFollowTheChain) {
  Start(/*drafters=*/false, /*speculate=*/false);
  EXPECT_EQ(llm_->KeptVersion(), kept::kStageFilesVersion);
  EXPECT_TRUE(llm_->KeptLayout().starts_with("llmp-state-v2:"));
  EXPECT_FALSE(llm_->speculative());
  const auto prompt = Prompt(150, 3);
  const Turned first = Turn(0, prompt, 20);
  EXPECT_EQ(first.reused, 0U);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 20));
  EXPECT_GE(llm_->counts().greedy_tokens, 19U);
  EXPECT_EQ(BranchAt(0).history().size(), prompt.size() + 19);

  const auto next = Continue(0, first.out, Prompt(30, 7));
  const std::size_t history = BranchAt(0).history().size();
  const Turned second = Turn(0, next, 12, /*keep_logits=*/true);
  EXPECT_EQ(second.reused, history);
  EXPECT_EQ(second.out.tokens, Chain(next, 12));
  ASSERT_EQ(second.out.logits.size(), second.out.tokens.size());
  for (std::size_t i = 0; i < second.out.tokens.size(); ++i) {
    EXPECT_EQ(ArgmaxOf(second.out.logits[i]), second.out.tokens[i]) << i;
  }
  // Prefill was planned and run as steps of the runner.
  EXPECT_GT(llm_->runner().graph_stats().eager + llm_->runner().graph_stats().replayed, 0U);
}

// Two conversations joined: their prompts' chunks in joined prompt waves
// (with the next chunks as futures), their decode rows in generation waves;
// each follows its own chain.
TEST_F(EngineLlmTest, PromptAndGenerationWavesJoinTheSlots) {
  Start(false, false);
  EXPECT_EQ(llm_->prefill_wave_capacity(), kSlots);
  const std::array<std::vector<std::int32_t>, 2> prompts = {Prompt(128, 1), Prompt(128, 29)};
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompts[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    sessions[i] = std::move(*begun);
    // Each one's reuse first, alone.
    while (!sessions[i]->done() && sessions[i]->NextUnit().has_value() &&
           sessions[i]->NextUnit()->phase != rt::Llm::PromptSession::Phase::kChunk) {
      auto advanced = sessions[i]->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  const std::array<rt::PrefillGoOn, 2> go_on{};
  while (!sessions[0]->done() && !sessions[1]->done()) {
    const std::array<rt::Llm::PromptSession*, 2> both = {sessions[0].get(), sessions[1].get()};
    auto ran = llm_->RunPromptWave(both, go_on);
    ASSERT_TRUE(ran) << Why(ran);
  }
  EXPECT_TRUE(sessions[0]->done() && sessions[1]->done());
  EXPECT_GE(llm_->counts().joined_prefill_steps, 1U);
  std::array<std::vector<float>, 2> lasts = {sessions[0]->last(), sessions[1]->last()};
  for (auto& session : sessions) {
    auto finished = session->Finish();
    ASSERT_TRUE(finished) << Why(finished);
  }
  rt::GenerateOptions options;
  options.max_tokens = 16;
  std::array<rt::Generation, 2> outs;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> generations;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginGeneration(lasts[i], options, outs[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    generations[i] = std::move(*begun);
  }
  for (int guard = 0; guard < 64; ++guard) {
    std::vector<rt::Llm::GenerationSession*> active;
    for (auto& g : generations) {
      if (!g->done()) active.push_back(g.get());
    }
    if (active.empty()) break;
    auto ran = llm_->RunGenerationWave(active);
    ASSERT_TRUE(ran) << Why(ran);
  }
  for (std::size_t i = 0; i < 2; ++i) {
    auto finished = generations[i]->Finish();
    ASSERT_TRUE(finished) << Why(finished);
    EXPECT_EQ(outs[i].tokens, Chain(prompts[i], 16)) << i;
  }
  Close();
  EXPECT_GE(llm_->counts().joined_waves, 1U);
}

// A joined prompt step whose owners' last chunks differ in rows (they step
// apart) builds both of those steps while it runs, so neither is planned
// on its own first run; the prompts still end on the chain.
TEST_F(EngineLlmTest, UnequalLastChunksAreBuiltAhead) {
  equal_prefill_ = true;
  Start(false, false);
  const std::array<std::vector<std::int32_t>, 2> prompts = {Prompt(kChunk + 20, 3),
                                                            Prompt(kChunk + 40, 17)};
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompts[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    sessions[i] = std::move(*begun);
    while (!sessions[i]->done() && sessions[i]->NextUnit().has_value() &&
           sessions[i]->NextUnit()->phase != rt::Llm::PromptSession::Phase::kChunk) {
      auto advanced = sessions[i]->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  const std::uint64_t before = llm_->runner().lookahead_stats().built;
  // The first chunks joined; then each prompt's last chunk alone.
  const std::array<rt::Llm::PromptSession*, 2> both = {sessions[0].get(), sessions[1].get()};
  const std::array<rt::PrefillGoOn, 2> go_on{};
  auto ran = llm_->RunPromptWave(both, go_on);
  ASSERT_TRUE(ran) << Why(ran);
  EXPECT_EQ(llm_->counts().joined_prefill_steps, 1U);
  EXPECT_EQ(llm_->runner().lookahead_stats().built - before, 2U);
  const std::size_t built = source_->built().size();
  for (auto& session : sessions) {
    while (!session->done()) {
      auto advanced = session->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  // Nothing more was built: both last chunks ran on plans built ahead.
  EXPECT_EQ(source_->built().size(), built);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(ArgmaxOf(sessions[i]->last()), Chain(prompts[i], 1).front()) << i;
    auto finished = sessions[i]->Finish();
    ASSERT_TRUE(finished) << Why(finished);
  }
  Close();
}

// A joined prompt wave ran with both owners' next chunks predicted (built,
// captured and their state prepared ahead); then the first owner's
// callback stops it. Its peer runs on alone and ends on its chain; the
// stopped owner keeps the prefix its completed chunk wrote (an abandoned
// prediction leaves no pending preparation), and its next turn reuses that
// prefix and follows its own chain.
TEST_F(EngineLlmTest, AStoppedOwnerKeepsItsCompletedPrefixAndItsPeerRunsOn) {
  Start(false, false);
  const std::array<std::vector<std::int32_t>, 2> prompts = {Prompt(3 * kChunk, 5),
                                                            Prompt(3 * kChunk, 23)};
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompts[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    sessions[i] = std::move(*begun);
    while (!sessions[i]->done() && sessions[i]->NextUnit().has_value() &&
           sessions[i]->NextUnit()->phase != rt::Llm::PromptSession::Phase::kChunk) {
      auto advanced = sessions[i]->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  const std::array<rt::Llm::PromptSession*, 2> both = {sessions[0].get(), sessions[1].get()};
  const std::array<rt::PrefillGoOn, 2> go_on{};
  auto ran = llm_->RunPromptWave(both, go_on);
  ASSERT_TRUE(ran) << Why(ran);
  EXPECT_EQ(llm_->counts().joined_prefill_steps, 1U);
  EXPECT_GE(llm_->runner().lookahead_stats().built, 1U);
  ASSERT_EQ(a.history().size(), kChunk);
  const std::array<rt::PrefillGoOn, 2> stop_first = {[](std::uint32_t) { return false; },
                                                     rt::PrefillGoOn{}};
  ran = llm_->RunPromptWave(both, stop_first);
  ASSERT_TRUE(ran) << Why(ran);
  EXPECT_TRUE(sessions[0]->done());
  EXPECT_FALSE(node_->has_pending_state_preparation());
  auto finished = sessions[0]->Finish();
  ASSERT_TRUE(finished) << Why(finished);
  EXPECT_EQ(a.history(),
            std::vector<std::int32_t>(prompts[0].begin(), prompts[0].begin() + kChunk));
  while (!sessions[1]->done()) {
    const std::array<rt::Llm::PromptSession*, 1> alone = {sessions[1].get()};
    const std::array<rt::PrefillGoOn, 1> on{};
    ran = llm_->RunPromptWave(alone, on);
    ASSERT_TRUE(ran) << Why(ran);
  }
  EXPECT_EQ(ArgmaxOf(sessions[1]->last()), Chain(prompts[1], 1).front());
  finished = sessions[1]->Finish();
  ASSERT_TRUE(finished) << Why(finished);
  EXPECT_EQ(b.history(), prompts[1]);
  EXPECT_FALSE(node_->has_pending_state_preparation());
  Close();
  const Turned next = Turn(0, prompts[0], 6);
  EXPECT_EQ(next.reused, kChunk);
  EXPECT_EQ(next.out.tokens, Chain(prompts[0], 6));
}

TEST_F(EngineLlmTest, ConfiguredPlainCohortKeepsCompanionStateAndSpeculativeSessionResults) {
  using Wave = llmp::execution::AdaptiveWaveMode;
  wave_mode_.emplace(Wave::Costs{}, Wave::Force::kPlain);
  Start(true, true);
  const std::array prompts{Prompt(32, 5), Prompt(32, 17)};
  std::array<rt::Generation, 2> outputs;
  JoinedTurn(prompts, outputs);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(outputs[i].tokens, Chain(prompts[i], 16));
    ASSERT_EQ(outputs[i].logits.size(), 16U);
    for (std::size_t row = 0; row < 16; ++row)
      EXPECT_EQ(ArgmaxOf(outputs[i].logits[row]), outputs[i].tokens[row]);
  }
  EXPECT_EQ(llm_->counts().speculative_units, 0U);
  // A lone owner still speculates and reads the companion state written by
  // the preceding ordinary target waves.
  const auto continuation = Continue(0, outputs[0], {});
  const auto next = Turn(0, continuation, 8);
  EXPECT_EQ(next.out.tokens, Chain(continuation, 8));
  EXPECT_GT(llm_->counts().drafted, 0U);
}

TEST_F(EngineLlmTest, ExplicitSpeculativeWaveKeepsTheExistingJoinedMode) {
  using Wave = llmp::execution::AdaptiveWaveMode;
  wave_mode_.emplace(Wave::Costs{}, Wave::Force::kSpeculative);
  Start(true, true);
  const std::array prompts{Prompt(32, 5), Prompt(32, 17)};
  std::array<rt::Generation, 2> outputs;
  JoinedTurn(prompts, outputs);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(outputs[i].tokens, Chain(prompts[i], 16));
    EXPECT_EQ(outputs[i].logits.size(), 16U);
  }
  EXPECT_GT(llm_->counts().speculative_units, 0U);
  EXPECT_GT(llm_->counts().drafted, 0U);
}

// Speculation with two drafters in a cascade: the MTP-like drafter's two
// proposals and the block's first follow the chain, the block's second
// does not, so a verify keeps four of its five rows; the tokens are the
// plain chain's all the same, alone and in a joined wave, with their rows.
TEST_F(EngineLlmTest, SpeculationWithTwoDraftersKeepsThePlainChain) {
  Start(/*drafters=*/true, /*speculate=*/true);
  EXPECT_TRUE(llm_->speculative());
  const auto prompt = Prompt(200, 5);
  const Turned first = Turn(0, prompt, 40);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 40));
  EXPECT_GT(first.out.drafted, 0U);
  EXPECT_GT(first.out.accepted, 0U);
  EXPECT_LT(first.out.accepted, first.out.drafted);  // the block's fill rows miss
  EXPECT_GT(llm_->counts().speculative_units, 0U);

  const auto other = Prompt(90, 17);
  const Turned rows = Turn(1, other, 25, /*keep_logits=*/true);
  EXPECT_EQ(rows.out.tokens, Chain(other, 25));
  ASSERT_EQ(rows.out.logits.size(), rows.out.tokens.size());
  for (std::size_t i = 0; i < rows.out.tokens.size(); ++i) {
    EXPECT_EQ(ArgmaxOf(rows.out.logits[i]), rows.out.tokens[i]) << i;
  }

  // Both continued in speculative waves.
  const auto next0 = Continue(0, first.out, Prompt(10, 2));
  const auto next1 = Continue(1, rows.out, Prompt(12, 4));
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::vector<float>, 2> lasts;
  for (std::size_t i = 0; i < 2; ++i) {
    std::uint32_t reused = 0;
    auto prepared = BranchAt(i).PreparePrompt(i == 0 ? next0 : next1, 0, lasts[i], reused);
    ASSERT_TRUE(prepared) << Why(prepared);
    EXPECT_GT(reused, 0U);
  }
  rt::GenerateOptions options;
  options.max_tokens = 20;
  std::array<rt::Generation, 2> outs;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> generations;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginGeneration(lasts[i], options, outs[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    generations[i] = std::move(*begun);
  }
  const std::uint64_t waves = llm_->counts().joined_waves;
  for (int guard = 0; guard < 64; ++guard) {
    std::vector<rt::Llm::GenerationSession*> active;
    for (auto& g : generations) {
      if (!g->done()) active.push_back(g.get());
    }
    if (active.empty()) break;
    auto ran = llm_->RunGenerationWave(active);
    ASSERT_TRUE(ran) << Why(ran);
  }
  for (std::size_t i = 0; i < 2; ++i) {
    auto finished = generations[i]->Finish();
    ASSERT_TRUE(finished) << Why(finished);
  }
  Close();
  EXPECT_EQ(outs[0].tokens, Chain(next0, 20));
  EXPECT_EQ(outs[1].tokens, Chain(next1, 20));
  EXPECT_GT(llm_->counts().joined_waves, waves);
}

TEST_F(EngineLlmTest, DecodeObservationPreservesMappedTwoOwnerRunAndMarksOverflow) {
  struct Run {
    std::array<rt::Generation, 3> outputs;
    std::array<std::vector<std::int32_t>, 3> histories;
    std::array<std::vector<std::byte>, 3> images;
    std::uint64_t drafted = 0;
    std::uint64_t accepted = 0;
    std::size_t events_before_restore = 0;
    std::size_t events_after_restore = 0;
    std::unique_ptr<rt::LlmDecodeObservation> trace;
  };
  const auto exercise = [this](bool observe, std::size_t event_limit) {
    Run run;
    Start(/*drafters=*/true, /*speculate=*/true);
    if (!llm_) {
      ADD_FAILURE() << "fake speculative model did not start";
      return run;
    }
    auto mapped = rt::LlmTestAccess::SetMappedDrafterHeadForTest(*llm_, 1);
    if (!mapped) {
      ADD_FAILURE() << Why(mapped);
      Stop();
      return run;
    }
    if (observe) {
      const auto bytes = rt::LlmDecodeObservation::BoundedBytes(
          event_limit, rt::LlmDecodeObservation::kMaxTokensPerEvent);
      if (!node_->ChargeHost(bytes, false)) {
        ADD_FAILURE() << "cannot fund bounded decode trace";
        Stop();
        return run;
      }
      run.trace = std::make_unique<rt::LlmDecodeObservation>(
          event_limit, rt::LlmDecodeObservation::kMaxTokensPerEvent);
      auto installed = rt::LlmTestAccess::InstallDecodeObservation(*llm_, *run.trace, false);
      if (!installed) {
        ADD_FAILURE() << Why(installed);
        Stop();
        return run;
      }
    }
    const std::array prompts{Prompt(200, 5), Prompt(90, 17)};
    for (std::size_t owner = 0; owner < prompts.size(); ++owner) {
      const Turned turn = Turn(owner, prompts[owner], owner == 0 ? 40 : 25,
                               /*keep_logits=*/true);
      run.outputs[owner] = turn.out;
      EXPECT_EQ(turn.out.tokens, Chain(prompts[owner], owner == 0 ? 40 : 25));
      EXPECT_EQ(turn.out.logits.size(), turn.out.tokens.size());
      run.histories[owner] = BranchAt(owner).history();
      run.images[owner] = NativeImage(static_cast<std::uint32_t>(owner));
    }
    if (run.trace) run.events_before_restore = run.trace->count;
    const auto depth_before_spill = Depth(0);
    auto spilled = llm_->SpillIdle(BranchAt(0));
    if (!spilled) {
      ADD_FAILURE() << Why(spilled);
      Stop();
      return run;
    }
    Open({&BranchAt(0)});
    auto restore = BranchAt(0).BeginPrompt(BranchAt(0).history());
    if (!restore) {
      ADD_FAILURE() << restore.error();
      Close();
      Stop();
      return run;
    }
    while (!(*restore)->done()) {
      auto advanced = (*restore)->Advance();
      if (!advanced) {
        ADD_FAILURE() << Why(advanced);
        if (!(*restore)->done()) (*restore)->Cancel();
        (void)(*restore)->Finish();
        restore->reset();
        Close();
        Stop();
        return run;
      }
    }
    auto restored = (*restore)->Finish();
    if (!restored) {
      ADD_FAILURE() << Why(restored);
      restore->reset();
      Close();
      Stop();
      return run;
    }
    restore->reset();
    EXPECT_EQ(Depth(0), depth_before_spill);
    Close();
    const auto continuation = Continue(0, run.outputs[0], Prompt(10, 2));
    const Turned resumed = Turn(0, continuation, 12, /*keep_logits=*/true);
    run.outputs[2] = resumed.out;
    EXPECT_EQ(resumed.reused, run.histories[0].size());
    EXPECT_EQ(resumed.out.tokens, Chain(continuation, 12));
    run.histories[2] = BranchAt(0).history();
    run.images[2] = NativeImage(0);
    if (run.trace) run.events_after_restore = run.trace->count;
    run.drafted = llm_->counts().drafted;
    run.accepted = llm_->counts().accepted;
    EXPECT_GT(run.drafted, 0U);
    EXPECT_GT(run.accepted, 0U);
    Stop();  // destroys the orchestrator before the returned trace is inspected
    return run;
  };

  const Run plain = exercise(false, 0);
  const Run traced = exercise(true, rt::LlmDecodeObservation::kMaxEvents);
  const Run overflow = exercise(true, 1);
  ASSERT_TRUE(traced.trace);
  ASSERT_TRUE(overflow.trace);
  for (std::size_t owner = 0; owner < plain.outputs.size(); ++owner) {
    for (const Run* compared : {&traced, &overflow}) {
      EXPECT_EQ(compared->outputs[owner].tokens, plain.outputs[owner].tokens);
      EXPECT_EQ(compared->outputs[owner].logits, plain.outputs[owner].logits);
      EXPECT_EQ(compared->outputs[owner].drafted, plain.outputs[owner].drafted);
      EXPECT_EQ(compared->outputs[owner].accepted, plain.outputs[owner].accepted);
      EXPECT_EQ(compared->outputs[owner].steps, plain.outputs[owner].steps);
      EXPECT_EQ(compared->histories[owner], plain.histories[owner]);
      EXPECT_EQ(compared->images[owner], plain.images[owner]);
    }
  }
  EXPECT_EQ(traced.drafted, plain.drafted);
  EXPECT_EQ(traced.accepted, plain.accepted);
  EXPECT_EQ(overflow.drafted, plain.drafted);
  EXPECT_EQ(overflow.accepted, plain.accepted);

  ASSERT_FALSE(traced.trace->incomplete);
  ASSERT_GT(traced.trace->count, 0U);
  bool saw_first_drafter = false, saw_mapped_drafter = false;
  bool saw_discarded_suffix = false, saw_mapped_change = false;
  std::array<bool, 2> saw_slots{};
  std::array<std::uint32_t, 2> next_ordinal{};
  std::vector<std::int32_t> mapped_ids(kWidth);
  for (std::size_t row = 0; row < mapped_ids.size(); ++row)
    mapped_ids[row] = static_cast<std::int32_t>(mapped_ids.size() - row - 1);
  const std::string mapped_digest =
      llmp::base::ToHex(llmp::base::Sha256().Update(std::as_bytes(std::span(mapped_ids))).Finish());
  for (std::size_t i = 0; i < traced.trace->count; ++i) {
    const auto& event = traced.trace->events[i];
    ASSERT_LT(event.slot, saw_slots.size());
    saw_slots[event.slot] = true;
    EXPECT_EQ(event.ordinal, next_ordinal[event.slot]++);
    if (event.verify) {
      EXPECT_EQ(event.actual_rows, event.requested_rows);
      EXPECT_EQ(event.input_tokens.size(), event.requested_rows);
      EXPECT_FALSE(event.head_values.empty());  // all target verify rows, including suffix
      saw_discarded_suffix |= event.keep < event.requested_rows;
    } else if (event.drafter == 0) {
      saw_first_drafter = true;
      EXPECT_TRUE(event.head_map.empty());
    } else if (event.drafter == 1) {
      saw_mapped_drafter = true;
      EXPECT_EQ(event.head_map, mapped_digest);
      ASSERT_EQ(event.proposal_rows.size(), event.result_tokens.size());
      for (std::size_t row = 0; row < event.proposal_rows.size(); ++row) {
        ASSERT_LT(event.proposal_rows[row], mapped_ids.size());
        EXPECT_EQ(event.result_tokens[row], mapped_ids[event.proposal_rows[row]]);
        saw_mapped_change |=
            event.result_tokens[row] != static_cast<std::int32_t>(event.proposal_rows[row]);
      }
    }
  }
  EXPECT_TRUE(saw_slots[0]);
  EXPECT_TRUE(saw_slots[1]);
  EXPECT_TRUE(saw_first_drafter);
  EXPECT_TRUE(saw_mapped_drafter);
  EXPECT_TRUE(saw_discarded_suffix);
  EXPECT_TRUE(saw_mapped_change);
  EXPECT_GT(traced.events_after_restore, traced.events_before_restore);
  EXPECT_TRUE(overflow.trace->incomplete);
  EXPECT_EQ(overflow.trace->count, 1U);
}

TEST_F(EngineLlmTest, FullHeadDiagnosticMatchesConfiguredNativeTokenOutput) {
  struct Run {
    rt::Generation output;
    std::vector<std::int32_t> history;
    std::vector<std::byte> image;
    std::uint64_t drafted = 0;
    std::uint64_t accepted = 0;
    std::unique_ptr<rt::LlmDecodeObservation> trace;
  };
  const auto exercise = [this](bool observe, bool full_head) {
    Run run;
    native_outputs_ = true;
    Start(/*drafters=*/true, /*speculate=*/true);
    if (!llm_) {
      ADD_FAILURE() << "fake native-token model did not start";
      return run;
    }
    if (observe) {
      const auto bytes = rt::LlmDecodeObservation::BoundedBytes(
          rt::LlmDecodeObservation::kMaxEvents, rt::LlmDecodeObservation::kMaxTokensPerEvent);
      if (!node_->ChargeHost(bytes, false)) {
        ADD_FAILURE() << "cannot fund bounded decode trace";
        Stop();
        return run;
      }
      run.trace = std::make_unique<rt::LlmDecodeObservation>();
      auto installed = rt::LlmTestAccess::InstallDecodeObservation(*llm_, *run.trace, full_head);
      if (!installed) {
        ADD_FAILURE() << Why(installed);
        Stop();
        return run;
      }
    }
    const auto prompt = Prompt(120, 11);
    const Turned turn = Turn(0, prompt, 32, /*keep_logits=*/true);
    run.output = turn.out;
    EXPECT_EQ(run.output.tokens, Chain(prompt, 32));
    EXPECT_EQ(run.output.logits.size(), run.output.tokens.size());
    run.history = BranchAt(0).history();
    run.image = NativeImage(0);
    run.drafted = llm_->counts().drafted;
    run.accepted = llm_->counts().accepted;
    Stop();
    return run;
  };

  const Run plain = exercise(false, false);
  const Run configured = exercise(true, false);
  const Run full_head = exercise(true, true);
  ASSERT_TRUE(configured.trace);
  ASSERT_TRUE(full_head.trace);
  for (const Run* compared : {&configured, &full_head}) {
    EXPECT_EQ(compared->output.tokens, plain.output.tokens);
    EXPECT_EQ(compared->output.logits, plain.output.logits);
    EXPECT_EQ(compared->output.drafted, plain.output.drafted);
    EXPECT_EQ(compared->output.accepted, plain.output.accepted);
    EXPECT_EQ(compared->output.steps, plain.output.steps);
    EXPECT_EQ(compared->history, plain.history);
    EXPECT_EQ(compared->image, plain.image);
    EXPECT_EQ(compared->drafted, plain.drafted);
    EXPECT_EQ(compared->accepted, plain.accepted);
  }
  bool saw_native = false, saw_head = false;
  for (std::size_t i = 0; i < configured.trace->count; ++i) {
    const auto& event = configured.trace->events[i];
    if (!event.verify) {
      saw_native |= event.kind == rt::LlmDecodeObservation::Kind::kDeviceTokens;
      EXPECT_TRUE(event.proposal_rows.empty());
    }
  }
  for (std::size_t i = 0; i < full_head.trace->count; ++i) {
    const auto& event = full_head.trace->events[i];
    if (!event.verify) {
      saw_head |= event.kind == rt::LlmDecodeObservation::Kind::kHeadRows;
      EXPECT_EQ(event.head_width, kWidth);
      EXPECT_EQ(event.proposal_rows.size(), event.result_tokens.size());
      EXPECT_FALSE(event.head_values.empty());
    }
  }
  EXPECT_TRUE(saw_native);
  EXPECT_TRUE(saw_head);
  EXPECT_FALSE(configured.trace->incomplete);
  EXPECT_FALSE(full_head.trace->incomplete);
}

TEST_F(EngineLlmTest, DecodeObservationRefusesAnActivePromptWithoutAbandoningIt) {
  Start(/*drafters=*/true, /*speculate=*/true);
  ASSERT_NE(llm_, nullptr);
  auto& branch = BranchAt(0);
  Open({&branch});
  auto prompt = Prompt(12, 7);
  auto opened = branch.BeginPrompt(prompt);
  ASSERT_TRUE(opened) << opened.error();
  const auto bytes = rt::LlmDecodeObservation::BoundedBytes(
      rt::LlmDecodeObservation::kMaxEvents, rt::LlmDecodeObservation::kMaxTokensPerEvent);
  ASSERT_TRUE(node_->ChargeHost(bytes, false));
  auto observer = std::make_unique<rt::LlmDecodeObservation>();
  const auto refused = rt::LlmTestAccess::InstallDecodeObservation(*llm_, *observer, false);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error(), "observation requires an empty default branch");
  if (!(*opened)->done()) (*opened)->Cancel();
  auto finished = (*opened)->Finish();
  ASSERT_TRUE(finished) << Why(finished);
  opened->reset();
  Close();
  Stop();
}

// An idle conversation spilled (every stage's file) and restored by its
// next turn's first unit: the turn reuses the whole history and continues
// the chain.
TEST_F(EngineLlmTest, ASpilledConversationComesBackExactly) {
  Start(true, true);
  const auto prompt = Prompt(180, 9);
  const Turned first = Turn(0, prompt, 12);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 12));
  rt::Llm::Branch& branch = BranchAt(0);
  const std::size_t history = branch.history().size();
  auto spilled = llm_->SpillIdle(branch);
  ASSERT_TRUE(spilled) << Why(spilled);
  EXPECT_TRUE(branch.spilled());
  EXPECT_GT(llm_->SpilledStateBytes(branch), 0U);
  EXPECT_EQ(llm_->ResidentStateBytes(branch), 0U);
  const auto next = Continue(0, first.out, Prompt(20, 1));
  const Turned second = Turn(0, next, 15);
  EXPECT_EQ(second.reused, history);
  EXPECT_EQ(second.out.tokens, Chain(next, 15));
  EXPECT_FALSE(branch.spilled());
  EXPECT_EQ(llm_->spill_stats().spills, 1U);
  EXPECT_EQ(llm_->spill_stats().restores, 1U);
}

// A turn checkpoint at the renderer's boundary, restored when a later
// prompt diverges after it: the lifecycle's three-step restore with its
// positions and footprint, every stage's pages.
TEST_F(EngineLlmTest, ATurnCheckpointRestoresItsPrefix) {
  Start(true, true);
  const auto prompt = Prompt(300, 4);
  const Turned first = Turn(0, prompt, 8, false, /*boundary=*/200);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 8));
  ASSERT_EQ(BranchAt(0).turn_checkpoints(), 1U);
  std::vector<std::int32_t> diverged(prompt.begin(), prompt.begin() + 250);
  const auto tail = Prompt(30, 11);
  diverged.insert(diverged.end(), tail.begin(), tail.end());
  const Turned second = Turn(0, diverged, 10);
  EXPECT_EQ(second.reused, 200U);
  EXPECT_EQ(second.out.tokens, Chain(diverged, 10));
}

// A turn checkpoint records the drafter taps' bookkeeping beside the state
// it saves (the taps' cells are state it holds), and its restore puts it
// back, so the next draft catches up from the captured features rather than
// starting without them (only acceptance, never tokens, would differ).
TEST_F(EngineLlmTest, ATurnCheckpointRestoresItsDrafterTapRows) {
  Start(true, true);
  const auto prompt = Prompt(300, 4);
  const Turned first = Turn(0, prompt, 8, false, /*boundary=*/200);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 8));
  rt::Llm::Branch& branch = BranchAt(0);
  ASSERT_EQ(branch.turn_checkpoints(), 1U);
  const auto captured = rt::LlmTestAccess::CheckpointTaps(branch, 0);
  ASSERT_NE(captured.stages, 0U);
  EXPECT_EQ(captured.positions, 200U);
  // This fixture's taps are position rings, which keep no per-row
  // bookkeeping: what a per-row tap would hold by now (a later chunk's rows)
  // stands in for the taps having moved on (the engine's own restore of
  // them: ChunkFeatureTest.CheckpointRestoreWithItsTapRowsReadsTheCapturedFeatures).
  auto moved = rt::LlmTestAccess::Taps(*llm_, branch);
  ASSERT_EQ(moved.positions, branch.history().size());
  moved.held[kMtp] = {.first = 290, .count = 18, .preceding = 4};
  ASSERT_TRUE(rt::LlmTestAccess::SetTaps(*llm_, branch, moved));
  ASSERT_NE(rt::LlmTestAccess::Taps(*llm_, branch), captured);
  std::vector<std::int32_t> diverged(prompt.begin(), prompt.begin() + 250);
  const auto tail = Prompt(30, 11);
  diverged.insert(diverged.end(), tail.begin(), tail.end());
  Open({&branch});
  auto session = branch.BeginPrompt(diverged);
  ASSERT_TRUE(session) << session.error();
  // The first unit is the reuse: the checkpoint's restore.
  ASSERT_TRUE((*session)->Advance());
  EXPECT_EQ((*session)->reused(), 200U);
  EXPECT_EQ(rt::LlmTestAccess::Taps(*llm_, branch), captured);
  if (!(*session)->done()) (*session)->Cancel();
  auto finished = (*session)->Finish();
  ASSERT_TRUE(finished) << Why(finished);
  session->reset();
  Close();
  // The continuation from the restored checkpoint is the chain.
  const Turned second = Turn(0, diverged, 10);
  EXPECT_EQ(second.out.tokens, Chain(diverged, 10));
}

// Turn checkpoints are bounded by real resources (retention, the spill
// budget's disk, the kept record's format), not a count: a conversation
// keeps one a turn, and the spill budget's trim drops the oldest first.
TEST_F(EngineLlmTest, TurnCheckpointsAreBoundByResourcesNotACount) {
  Start(true, true);
  Turned turn = Turn(0, Prompt(300, 4), 8, false, /*boundary=*/200);
  for (int i = 0; i < 2; ++i) {
    const auto next = Continue(0, turn.out, Prompt(40, 5 + i));
    turn = Turn(0, next, 8, false, static_cast<std::uint32_t>(next.size() - 10));
  }
  rt::Llm::Branch& branch = BranchAt(0);
  ASSERT_EQ(branch.turn_checkpoints(), 3U);  // past the count of two it once kept
  const std::uint64_t bytes = branch.turn_checkpoint_bytes();
  ASSERT_TRUE(llm_->OldestTurnCheckpoint(branch).has_value());
  const std::uint64_t dropped = llm_->DropOldestTurnCheckpoint(branch);
  EXPECT_GT(dropped, 0U);
  EXPECT_EQ(branch.turn_checkpoints(), 2U);
  EXPECT_EQ(branch.turn_checkpoint_bytes(), bytes - dropped);
}

TEST_F(EngineLlmTest, ACheckpointIsAdmittedBeforeItWritesAndRefusalKeepsTheConversation) {
  Start(true, true);
  rt::Llm::Branch& branch = BranchAt(0);
  std::uint64_t budget = 0;
  std::uint64_t asked = 0;
  std::size_t attempts = 0;
  llm_->set_checkpoint_admission([&](std::uint64_t bytes) {
    ++attempts;
    asked = bytes;
    // Capture runs while the prompt holds its branch: those checkpoints
    // cannot be trimmed, but their complete files still count.
    EXPECT_FALSE(llm_->OldestTurnCheckpoint(branch));
    return rt::TurnCheckpointFits(budget, 0, branch.turn_checkpoint_bytes(), bytes);
  });
  Turned turn = Turn(0, Prompt(300, 4), 8, false, /*boundary=*/200);
  EXPECT_EQ(branch.turn_checkpoints(), 0U);
  EXPECT_EQ(attempts, 1U);
  EXPECT_GT(asked, 0U);
  EXPECT_EQ(asked % kept::kFileAlignment, 0U);
  EXPECT_EQ(turn.out.tokens, Chain(Prompt(300, 4), 8));
  budget = asked;
  auto next = Continue(0, turn.out, Prompt(40, 5));
  turn = Turn(0, next, 8, false, static_cast<std::uint32_t>(next.size() - 10));
  ASSERT_EQ(branch.turn_checkpoints(), 1U);
  EXPECT_LE(branch.turn_checkpoint_bytes(), budget);
  const auto held = branch.turn_checkpoint_bytes();
  next = Continue(0, turn.out, Prompt(40, 6));
  turn = Turn(0, next, 8, false, static_cast<std::uint32_t>(next.size() - 10));
  EXPECT_EQ(branch.turn_checkpoints(), 1U);
  EXPECT_EQ(branch.turn_checkpoint_bytes(), held);
  EXPECT_EQ(attempts, 3U);
  EXPECT_EQ(turn.out.tokens, Chain(next, 8));
}

TEST_F(EngineLlmTest, TrimmingAKeptCheckpointCanSpillAnotherConversationSynchronously) {
  const fs::path directory = root_ / "trim-kept";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  rt::RequestMemory memory(0, 128 * kExtent, true);
  rt::StateKeeper keeper(1, {});
  const auto model = keeper.AddModel(::dup(dir));
  Start(false, false, [&](rt::Llm& llm) {
    llm.SetTokenMemory(memory);
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = {.build = "test build",
                               .artifact = "abcdef",
                               .drafter = "",
                               .layout = llm.KeptLayout()},
                  .adopt = {}});
  });
  bool trimming = false, reentered = false;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    if (trimming && incoming != 0 && !reentered) {
      reentered = true;
      // Server's token grower can reclaim idle state this way while the
      // outer DropOldestTurnCheckpoint is updating its kept record.
      EXPECT_TRUE(llm_->SpillIdle(BranchAt(1)));
    }
    return true;
  });
  Turn(0, Prompt(300, 4), 8, false, 200);
  Turn(1, Prompt(40, 5), 8);
  auto& branch = BranchAt(0);
  ASSERT_EQ(branch.turn_checkpoints(), 1U);
  ASSERT_TRUE(llm_->SpillIdle(branch));
  ASSERT_TRUE(keeper.Drain(rt::Clock::now() + std::chrono::seconds(30)));
  ASSERT_TRUE(keeper.Kept(model, 0));
  // Drain reports completed publication before the worker destroys its
  // Job and returns the record's token charge. Once that charge is gone,
  // shrink the exact grant so the replacement record must call Grow.
  const auto released_by = rt::Clock::now() + std::chrono::seconds(30);
  while (memory.used() != llm_->token_history_bytes() && rt::Clock::now() < released_by) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(memory.used(), llm_->token_history_bytes());
  memory.Settle(0, std::chrono::milliseconds{0});
  ASSERT_EQ(memory.grant(), memory.used());
  const auto spilled = [&] {
    return llm_->SpilledStateBytes(branch) + llm_->SpilledStateBytes(BranchAt(1));
  };
  const auto checkpoints = [&] { return branch.turn_checkpoint_bytes(); };
  const auto extra = checkpoints();
  ASSERT_GT(extra, 0U);
  const auto initially_spilled = spilled();
  const auto newly_spilled = llm_->ResidentStateBytes(BranchAt(1));
  ASSERT_GT(newly_spilled, 0U);
  // Both the checkpoint and the smaller second conversation must exceed
  // the available room. Their file sizes need not be the same.
  const auto budget = initially_spilled + extra + std::min(extra, newly_spilled) - 1;
  ASSERT_FALSE(rt::TurnCheckpointFits(budget, initially_spilled, checkpoints(), extra));
  ASSERT_TRUE(rt::TurnCheckpointFits(budget, initially_spilled, 0, extra));
  trimming = true;
  const bool fits = rt::TrimTurnCheckpointBudget(budget, extra, spilled, checkpoints, [&] {
    return llm_->DropOldestTurnCheckpoint(branch) != 0;
  });
  trimming = false;
  EXPECT_TRUE(reentered);
  EXPECT_TRUE(BranchAt(1).spilled());
  EXPECT_GT(spilled(), initially_spilled);
  EXPECT_EQ(checkpoints(), 0U);
  EXPECT_FALSE(fits);
  EXPECT_FALSE(rt::TurnCheckpointFits(budget, spilled(), checkpoints(), extra));
  ASSERT_TRUE(keeper.Drain(rt::Clock::now() + std::chrono::seconds(30)));
  Stop();
  ::close(dir);
}

TEST_F(EngineLlmTest, CheckpointCapacityIsHeldWhileFallbackStagingReclaims) {
  Start(false, false);  // no reserved staging: Capture uses its fallback
  Turn(1, Prompt(40, 5), 8);
  auto& branch = BranchAt(0);
  Open({&branch});
  const auto tokens = Prompt(300, 4);
  auto begun = branch.BeginPrompt(tokens, 200);
  ASSERT_TRUE(begun) << Failed(begun).value_or("");
  auto prompt = std::move(*begun);
  for (;;) {
    auto next = prompt->NextUnit();
    ASSERT_TRUE(next) << Failed(next).value_or("");
    if (next->phase == rt::Llm::PromptSession::Phase::kCheckpoint) break;
    ASSERT_TRUE(prompt->Advance());
  }
  const auto occupied = node_->catalog().OccupancyOf(node_->domain()).Total().value();
  ASSERT_GT(node_->budget().value(), occupied + kExtent);
  std::vector<llmp::catalog::ExtentId> staging;
  auto pad = node_->Pinned(node_->budget().value() - occupied - kExtent, 0, staging);
  ASSERT_TRUE(pad) << Failed(pad).value_or("");
  const auto victim = llm_->ResidentStateBytes(BranchAt(1));
  ASSERT_GT(victim, 0U);
  std::uint64_t budget = 0, admitted = 0;
  llm_->set_checkpoint_admission([&](std::uint64_t bytes) {
    admitted = bytes;
    budget = std::max(bytes, victim);  // either fits alone, never both
    return rt::TurnCheckpointFits(budget, 0, branch.turn_checkpoint_bytes(), bytes);
  });
  bool reclaimed = false;
  node_->SetReclaimer([&](std::uint64_t, en::PagedNode::ReclaimFor what,
                          std::span<const llmp::catalog::ExtentId>) -> std::uint64_t {
    EXPECT_EQ(what, en::PagedNode::ReclaimFor::kStaging);
    reclaimed = true;
    EXPECT_GT(admitted, 0U);
    EXPECT_EQ(branch.turn_checkpoint_bytes(), admitted);
    EXPECT_FALSE(llm_->OldestTurnCheckpoint(branch));
    // The server's spill admission counts the capture's capacity even
    // though its file has not yet been published on the branch.
    if (!rt::TurnCheckpointFits(budget, 0, branch.turn_checkpoint_bytes(), victim)) return 0;
    EXPECT_TRUE(llm_->SpillIdle(BranchAt(1)));
    return victim;
  });
  ASSERT_TRUE(prompt->Advance());  // staging refuses this optional save
  EXPECT_TRUE(reclaimed);
  EXPECT_FALSE(BranchAt(1).spilled());
  EXPECT_EQ(branch.turn_checkpoints(), 0U);
  EXPECT_EQ(branch.turn_checkpoint_bytes(), 0U);  // failed capture returned its capacity
  node_->SetReclaimer({});
  ASSERT_TRUE(node_->FreePinned(*pad));
  while (!prompt->done()) ASSERT_TRUE(prompt->Advance());
  ASSERT_TRUE(prompt->Finish());
  prompt.reset();
  Close();
  // A later successful capture transfers the reservation to its file,
  // counted once, and still produces the uninterrupted token chain.
  const auto turn = Turn(0, Prompt(320, 7), 8, false, 220);
  EXPECT_EQ(turn.out.tokens, Chain(Prompt(320, 7), 8));
  EXPECT_EQ(branch.turn_checkpoints(), 1U);
  EXPECT_EQ(branch.turn_checkpoint_bytes(), admitted);
}

TEST_F(EngineLlmTest, CancelledCheckpointCaptureReturnsItsDiskCapacity) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  auto begun = branch.BeginPrompt(Prompt(300, 4), 200);
  ASSERT_TRUE(begun) << Failed(begun).value_or("");
  auto prompt = std::move(*begun);
  for (;;) {
    auto next = prompt->NextUnit();
    ASSERT_TRUE(next) << Failed(next).value_or("");
    if (next->phase == rt::Llm::PromptSession::Phase::kCheckpoint) break;
    ASSERT_TRUE(prompt->Advance());
  }
  std::uint64_t admitted = 0;
  llm_->set_checkpoint_admission([&](std::uint64_t bytes) {
    admitted = bytes;
    return true;
  });
  unsigned calls = 0;
  ASSERT_TRUE(prompt->Advance([&](std::uint32_t) {
    if (++calls == 1) return true;  // before capture admission
    EXPECT_GT(admitted, 0U);
    EXPECT_EQ(branch.turn_checkpoint_bytes(), admitted);
    return false;  // Capture's first cancellation checkpoint
  }));
  EXPECT_EQ(calls, 2U);
  EXPECT_TRUE(prompt->done());
  EXPECT_EQ(branch.turn_checkpoints(), 0U);
  EXPECT_EQ(branch.turn_checkpoint_bytes(), 0U);
  ASSERT_TRUE(prompt->Finish());
  prompt.reset();
  Close();
}

// The speed gate's diagnostic (ServingOptions::diagnostic_skip_turn_
// checkpoints): the turn saves no checkpoint and is otherwise the same (its
// tokens and the steps it ran); a later divergent prompt then has none to
// restore and prefills, with the same reply.
TEST_F(EngineLlmTest, TheGateDiagnosticSkipsOnlyTheTurnCheckpointSave) {
  const auto prompt = Prompt(300, 4);
  std::vector<std::int32_t> diverged(prompt.begin(), prompt.begin() + 250);
  const auto tail = Prompt(30, 11);
  diverged.insert(diverged.end(), tail.begin(), tail.end());
  struct Seen {
    std::vector<std::int32_t> tokens;
    std::uint64_t queued = 0, executed = 0;
    std::size_t checkpoints = 0;
    std::uint32_t reused = 0;
  };
  std::array<Seen, 2> seen;
  for (const bool save : {true, false}) {
    save_checkpoints_ = save;
    Start(false, false);
    Seen& s = seen.at(save ? 0 : 1);
    const Turned first = Turn(0, prompt, 8, false, /*boundary=*/200);
    s.tokens = first.out.tokens;
    s.queued = source_->queued();
    s.executed = source_->executed();
    s.checkpoints = BranchAt(0).turn_checkpoints();
    const Turned second = Turn(0, diverged, 10);
    EXPECT_EQ(second.out.tokens, Chain(diverged, 10));
    s.reused = second.reused;
    Stop();
  }
  EXPECT_EQ(seen[0].tokens, Chain(prompt, 8));
  EXPECT_EQ(seen[1].tokens, seen[0].tokens);
  EXPECT_EQ(seen[1].queued, seen[0].queued);
  EXPECT_EQ(seen[1].executed, seen[0].executed);
  EXPECT_EQ(seen[0].checkpoints, 1U);
  EXPECT_EQ(seen[1].checkpoints, 0U);
  EXPECT_EQ(seen[0].reused, 200U);
  EXPECT_EQ(seen[1].reused, 0U);
}

// D-102's rung 2: a step that failed after it queued its work faults the
// cohort; recovery discards what the failure may have touched (the resident
// conversation, forgotten) and keeps the spilled one, which continues.
TEST_F(EngineLlmTest, RecoveryKeepsTheSpilledConversationAndForgetsTheRest) {
  Start(false, false);
  const auto kept_prompt = Prompt(100, 2);
  const Turned kept_turn = Turn(1, kept_prompt, 6);
  rt::Llm::Branch& spilled = BranchAt(1);
  ASSERT_TRUE(llm_->SpillIdle(spilled));
  const std::size_t history = spilled.history().size();

  rt::Llm::Branch& failing = BranchAt(0);
  Open({&failing});
  std::vector<float> last;
  std::uint32_t reused = 0;
  ASSERT_TRUE(failing.PreparePrompt(Prompt(70, 3), 0, last, reused));
  source_->options().fail_run = kn::StepError::kUnknown;
  rt::GenerateOptions options;
  options.max_tokens = 4;
  rt::Generation out;
  EXPECT_FALSE(failing.Generate(last, options, out));
  source_->options().fail_run.reset();
  std::ignore = node_->EndRequest(0);
  // The stream fenced, as the server's recovery does first.
  ASSERT_TRUE(node_->Job(
      llmp::catalog::Closure{},
      [](llmp::providers::NativeStream) { return llmp::scheduler::JobResult::kQueued; },
      "fencing the test's stream", 0));
  auto recovered = llm_->RecoverInPlace();
  ASSERT_TRUE(recovered) << Why(recovered);
  EXPECT_TRUE(failing.history().empty());
  EXPECT_TRUE(spilled.spilled());
  EXPECT_EQ(spilled.history().size(), history);

  const auto next = Continue(1, kept_turn.out, Prompt(9, 8));
  const Turned resumed = Turn(1, next, 7);
  EXPECT_EQ(resumed.reused, history);
  EXPECT_EQ(resumed.out.tokens, Chain(next, 7));
  const auto fresh = Prompt(40, 21);
  const Turned again = Turn(0, fresh, 5);
  EXPECT_EQ(again.out.tokens, Chain(fresh, 5));
}

// F15a: a chunk bound that is not a whole number of row tiles cuts tiled
// chunks (PrefillRows: 1024 of 1030), which are still full chunks, so the
// next one is recorded beside the current run and replays.
TEST_F(EngineLlmTest, ATiledFullChunkIsRecordedAheadOfItsUse) {
  context_rows_ = 4096;
  chunk_rows_ = 1030;
  Start(false, false);
  const auto prompt = Prompt(3 * 1024 + 100, 7);
  const Turned turn = Turn(0, prompt, 2);
  EXPECT_EQ(turn.out.tokens, Chain(prompt, 2));
  EXPECT_GE(llm_->runner().lookahead_stats().captured_ahead, 1U);
}

// F15a: the drafter's catch-up after a prompt chunk (CatchUpPrompt) is a
// step of its own. One that fails after it queued its work faults the
// cohort although the chunk itself retired: the prompt fails, and after the
// fence the model recovers, prefills it again and drafts.
TEST_F(EngineLlmTest, APromptCatchUpOfUnknownEffectFailsThePromptAndRecovers) {
  native_outputs_ = true;
  Start(true, true);
  const auto prompt = Prompt(2 * kChunk + 5, 3);
  rt::Llm::Branch& branch = BranchAt(0);
  Open({&branch});
  source_->options().positioned = true;  // eager runs, so the fault reaches them
  source_->options().fail_run = kn::StepError::kUnknown;
  source_->options().fail_phase = pl::Phase::kDraft;
  const auto before = source_->executed();
  std::vector<float> last;
  std::uint32_t reused = 0;
  EXPECT_FALSE(branch.PreparePrompt(prompt, 0, last, reused));
  EXPECT_GT(source_->executed(), before);  // the queued catch-up ran
  EXPECT_FALSE(llm_->runner().slots().usable(0));
  source_->options().fail_run.reset();
  source_->options().fail_phase.reset();
  std::ignore = node_->EndRequest(0);
  ASSERT_TRUE(node_->Job(
      llmp::catalog::Closure{},
      [](llmp::providers::NativeStream) { return llmp::scheduler::JobResult::kQueued; },
      "fencing the test's stream", 0));
  auto recovered = llm_->RecoverInPlace();
  ASSERT_TRUE(recovered) << Why(recovered);
  EXPECT_TRUE(branch.history().empty());
  const Turned again = Turn(0, prompt, 12);
  EXPECT_EQ(again.reused, 0U);
  EXPECT_EQ(again.out.tokens, Chain(prompt, 12));
  EXPECT_GT(again.out.drafted, 0U);
  EXPECT_GT(again.out.accepted, 0U);
}

// Kept across a restart (D-105) with a spill file a stage (record version
// 2): the first process spills a conversation with a turn checkpoint and
// its record is written; the second checks the record and every stage's
// file, adopts it, and its next turn restores the state from the files and
// continues the chain. A record of version 1 for the other slot is refused
// and removed.
TEST_F(EngineLlmTest, AConversationIsKeptAcrossARestartInStageFiles) {
  const auto recipe = llmp::base::Sha256().Update("qualified kept execution").Finish();
  execution_identity_ = std::optional{recipe};
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  std::vector<std::int32_t> history;
  rt::Generation reply;
  std::string layout;
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::Llm& llm) {
      layout = llm.KeptLayout();
      identity.layout = layout;
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    const auto prompt = Prompt(220, 6);
    const Turned first = Turn(0, prompt, 10, false, /*boundary=*/150);
    EXPECT_EQ(first.out.tokens, Chain(prompt, 10));
    reply = first.out;
    rt::Llm::Branch& branch = BranchAt(0);
    ASSERT_TRUE(llm_->SpillIdle(branch));
    history = branch.history();
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    EXPECT_TRUE(keeper.Kept(model, 0));
    llm_->PreserveKeptFiles();
    Stop();
  }
  for (const char* name :
       {"slot-0.state", "slot-0.stage-1.state", "slot-0.stage-2.state", "slot-0.record"}) {
    EXPECT_TRUE(fs::exists(directory / name)) << name;
  }
  auto text = pf::ReadPrivateFile(dir, "slot-0.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record.has_value()) << record.error();
  EXPECT_EQ(record->version, kept::kStageFilesVersion);
  ASSERT_EQ(record->files.size(), 3U);
  EXPECT_EQ(record->files[2].name, "slot-0.stage-2.state");
  EXPECT_EQ(record->tokens, history);
  EXPECT_EQ(record->checkpoints.size(), 1U);

  // A fresh owner with another backend/tuning/domain identity (or the legacy
  // null identity) must refuse the same saved record before adopting any state.
  // It uses ordinary scratch files: the authentic saved files stay untouched.
  for (int mutation = 0; mutation < 4; ++mutation) {
    SCOPED_TRACE(mutation);
    auto changed = recipe;
    if (mutation < 3) {
      changed[static_cast<std::size_t>(mutation)] ^= 1;
      execution_identity_ = std::optional{changed};
    } else {
      execution_identity_ = std::nullopt;
    }
    Start(true, true);
    EXPECT_NE(llm_->KeptLayout(), layout);
    EXPECT_EQ(llm_->KeptLayout(), llm_->runner().slots().kept_layout_id());
    auto expected_identity = identity;
    expected_identity.layout = llm_->KeptLayout();
    kept::Expected expected{.version = llm_->KeptVersion(),
                            .identity = expected_identity,
                            .slot = 0,
                            .regions = llm_->KeptRegions(),
                            .layouts = llm_->KeptLayouts(),
                            .context = llm_->usable_context(),
                            .vocabulary = kWidth,
                            .now_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count(),
                            .retention_ms = 24LL * 3600 * 1000};
    auto candidate = *record;
    auto checked = kept::Check(candidate, expected);
    ASSERT_FALSE(checked);
    EXPECT_EQ(checked.error(),
              "a kept conversation's record: its state layout is not this runner's");
    std::vector<en::LiveState::Range> footprint;
    const auto layouts = llm_->KeptLayouts();
    for (const auto& extent : record->extents) {
      const auto offset = std::uint64_t{extent.index} * kExtent;
      ASSERT_LT(extent.region, layouts.size());
      ASSERT_LT(offset, layouts[extent.region]);
      footprint.push_back({.region = extent.region,
                           .offset = offset,
                           .bytes = std::min(kExtent, layouts[extent.region] - offset)});
    }
    auto& slots = llm_->runner().slots();
    ASSERT_TRUE(slots.ValidateFootprint(record->cursor, footprint));
    const auto before = NativeImage(0);
    const auto executed = source_->executed();
    auto adopt = slots.Adopt(0, record->cursor, footprint, layout);
    ASSERT_FALSE(adopt);
    EXPECT_NE(adopt.error().find("layout differs"), std::string::npos);
    EXPECT_FALSE(slots.adopted(0));
    EXPECT_EQ(NativeImage(0), before);
    EXPECT_TRUE(BranchAt(0).history().empty());
    Open({&BranchAt(0)});
    auto restore = slots.PrepareRestore(0, record->cursor, footprint, layout);
    ASSERT_FALSE(restore);
    EXPECT_NE(restore.error().find("source layout differs"), std::string::npos);
    EXPECT_EQ(NativeImage(0), before);
    EXPECT_EQ(source_->executed(), executed);
    Close();
    Stop();
    EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
  }
  execution_identity_ = std::optional{recipe};

  // The next process: the record checked against what it would write, its
  // files hashed, then adopted.
  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  kept::Record old;
  Start(true, true, [&](rt::Llm& llm) {
    EXPECT_EQ(llm.KeptLayout(), layout);
    kept::Expected expected{.version = llm.KeptVersion(),
                            .identity = identity,
                            .slot = 0,
                            .regions = llm.KeptRegions(),
                            .layouts = llm.KeptLayouts(),
                            .context = llm.usable_context(),
                            .vocabulary = kWidth,
                            .now_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count(),
                            .retention_ms = 24LL * 3600 * 1000};
    auto checked = kept::Check(*record, expected);
    ASSERT_TRUE(checked.has_value()) << checked.error();
    EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
    for (const kept::Checkpoint& c : record->checkpoints) {
      EXPECT_EQ(
          rt::VerifyKeptFile(dir, c.file, c.id, c.file_bytes, kept::CheckpointPlaces(c), c.digests),
          "");
    }
    EXPECT_EQ(rt::EmptyUnlisted(dir, *record), "");
    // The other slot's record of the old version: decoded, refused.
    old = *record;
    old.version = 1;  // the retired M3 runners' one spill file a slot
    old.slot = 1;
    old.files = {{.name = kept::StateFileName(1), .id = record->files[0].id, .bytes = 0}};
    old.checkpoints.clear();
    for (const std::uint64_t bytes : old.regions) old.files[0].bytes += bytes;
    expected.slot = 1;
    auto refused = kept::Check(old, expected);
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().find("format version 1"), std::string::npos) << refused.error();
    std::vector<kept::FileId> ids;
    for (const kept::StateFile& file : record->files) ids.push_back(file.id);
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0},
                  .files = {ids}});
  });
  rt::MemoryCharge charge;
  rt::Llm::Branch& branch = BranchAt(0);
  auto adopted = llm_->Adopt(branch, *record, charge);
  ASSERT_TRUE(adopted) << Why(adopted);
  EXPECT_EQ(branch.history(), history);
  EXPECT_TRUE(branch.spilled());
  EXPECT_EQ(branch.turn_checkpoints(), 1U);
  EXPECT_TRUE(keeper.Kept(model, 0));

  // The old record, written where the other slot's would be: no longer
  // read, and refused at adoption too (as decoded before), and removed.
  ASSERT_TRUE(pf::ReplacePrivateFile(dir, kept::RecordFileName(1).c_str(), kept::Encode(old)));
  auto old_text = pf::ReadPrivateFile(dir, kept::RecordFileName(1).c_str(), kept::kMostRecordBytes);
  ASSERT_TRUE(old_text.has_value());
  auto old_record = kept::Decode(*old_text);
  ASSERT_FALSE(old_record.has_value());
  EXPECT_NE(old_record.error().find("version"), std::string::npos) << old_record.error();
  rt::MemoryCharge old_charge;
  auto old_adopted = llm_->Adopt(BranchAt(1), old, old_charge);
  ASSERT_FALSE(old_adopted);
  EXPECT_NE(Why(old_adopted).find("format version"), std::string::npos) << Why(old_adopted);
  EXPECT_FALSE(fs::exists(directory / kept::RecordFileName(1)));
  EXPECT_TRUE(BranchAt(1).history().empty());
  // Only that slot's record went: the adopted slot's record and stage files
  // stay as they were.
  for (const char* name :
       {"slot-0.state", "slot-0.stage-1.state", "slot-0.stage-2.state", "slot-0.record"}) {
    EXPECT_TRUE(fs::exists(directory / name)) << name;
  }
  EXPECT_TRUE(keeper.Kept(model, 0));

  // Its next turn restores every stage from its files and continues.
  std::vector<std::int32_t> next = history;
  next.push_back(reply.tokens.back());
  const auto more = Prompt(15, 2);
  next.insert(next.end(), more.begin(), more.end());
  const Turned resumed = Turn(0, next, 12);
  EXPECT_EQ(resumed.reused, history.size());
  EXPECT_EQ(resumed.out.tokens, Chain(next, 12));
  EXPECT_FALSE(keeper.Kept(model, 0));  // live again: its record went
  Stop();
  (void)::close(dir);
}

// A file put under a kept name after the start checked it (between
// VerifyKeptState and the reopens that adopt it) is not the one its record
// names: a stage's spill file found so refuses the slot's adoption and a
// turn checkpoint found so is left out, and the file reached through the
// name is never adopted, written nor emptied (D-105 as amended).
TEST_F(EngineLlmTest, AFileSwappedInAfterTheChecksIsNeverAdoptedNorWritten) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::Llm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    for (const std::size_t slot : {std::size_t{0}, std::size_t{1}}) {
      const auto prompt = Prompt(220, 6 + static_cast<std::int32_t>(slot));
      (void)Turn(slot, prompt, 10, false, /*boundary=*/150);
      ASSERT_TRUE(llm_->SpillIdle(BranchAt(slot)));
    }
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    llm_->PreserveKeptFiles();
    Stop();
  }
  std::vector<kept::Record> records;
  for (const std::uint32_t slot : {0U, 1U}) {
    auto text =
        pf::ReadPrivateFile(dir, kept::RecordFileName(slot).c_str(), kept::kMostRecordBytes);
    ASSERT_TRUE(text.has_value()) << slot;
    auto record = kept::Decode(*text);
    ASSERT_TRUE(record.has_value()) << record.error();
    ASSERT_EQ(record->checkpoints.size(), 1U);
    // The start's checks pass: the files are the ones named.
    EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
    records.push_back(std::move(*record));
  }
  // Then each of two names becomes a link to a copy: the same bytes and
  // size, another file.
  const auto contents = [](const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  const auto swap = [&](const std::string& name) {
    const fs::path foreign = root_ / ("foreign-" + name);
    fs::copy_file(directory / name, foreign);
    fs::remove(directory / name);
    fs::create_symlink(foreign, directory / name);
    return foreign;
  };
  const fs::path stage = swap("slot-0.stage-1.state");
  const fs::path checkpoint = swap(records[1].checkpoints[0].file);
  const std::string stage_bytes = contents(stage);
  const std::string checkpoint_bytes = contents(checkpoint);
  ASSERT_FALSE(stage_bytes.empty());
  ASSERT_FALSE(checkpoint_bytes.empty());

  // The next process reopens the files the start checked (PrepareKept
  // passes their identities).
  std::vector<std::vector<kept::FileId>> files;
  for (const kept::Record& record : records) {
    files.emplace_back();
    for (const kept::StateFile& file : record.files) files.back().push_back(file.id);
  }
  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::Llm& llm) {
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0, 1},
                  .files = files});
  });
  rt::MemoryCharge first_charge;
  auto first = llm_->Adopt(BranchAt(0), records[0], first_charge);
  EXPECT_FALSE(first);
  if (!first) {
    EXPECT_NE(Why(first).find("not the one its record names"), std::string::npos) << Why(first);
    EXPECT_TRUE(BranchAt(0).ReleaseIdleState());  // as the server does with one refused
  }
  EXPECT_FALSE(fs::is_symlink(directory / "slot-0.stage-1.state"));
  rt::MemoryCharge second_charge;
  auto second = llm_->Adopt(BranchAt(1), records[1], second_charge);
  ASSERT_TRUE(second) << Why(second);
  EXPECT_EQ(BranchAt(1).turn_checkpoints(), 0U);
  // Both slots used and spilled again: neither file reached through a
  // swapped name changes.
  for (const std::size_t slot : {std::size_t{0}, std::size_t{1}}) {
    const auto prompt = Prompt(90, 30 + static_cast<std::int32_t>(slot));
    const Turned turned = Turn(slot, prompt, 6);
    EXPECT_EQ(turned.out.tokens, Chain(prompt, 6));
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(slot)));
  }
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_TRUE(contents(stage) == stage_bytes);
  EXPECT_TRUE(contents(checkpoint) == checkpoint_bytes);
  Stop();
  (void)::close(dir);
}

// A slot set to be adopted without its record's file identities keeps
// none of its files: it gets new empty ones, so a file put under a name
// since the process before (here a link to a copy of the same size) is
// never used, nor emptied when the adoption is refused.
TEST_F(EngineLlmTest, AKeptSlotWithoutItsFileIdentitiesStartsFresh) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::Llm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    (void)Turn(0, Prompt(220, 6), 10, false);
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(0)));
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    llm_->PreserveKeptFiles();
    Stop();
  }
  auto text = pf::ReadPrivateFile(dir, kept::RecordFileName(0).c_str(), kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record.has_value()) << record.error();
  const std::string name = kept::StateFileName(0, 0);
  const fs::path foreign = root_ / "foreign";
  fs::copy_file(directory / name, foreign);
  fs::remove(directory / name);
  fs::create_symlink(foreign, directory / name);
  const auto contents = [](const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  const std::string bytes = contents(foreign);
  ASSERT_FALSE(bytes.empty());
  ASSERT_NE(bytes.find_first_not_of('\0'), std::string::npos);

  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::Llm& llm) {
    llm.set_kept(
        {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {0}});
  });
  rt::MemoryCharge charge;
  auto adopted = llm_->Adopt(BranchAt(0), *record, charge);
  ASSERT_FALSE(adopted);
  EXPECT_NE(Why(adopted).find("not the one its record names"), std::string::npos) << Why(adopted);
  EXPECT_TRUE(BranchAt(0).ReleaseIdleState());  // as the server does with one refused
  EXPECT_FALSE(fs::is_symlink(directory / name));
  EXPECT_TRUE(contents(foreign) == bytes);
  Stop();
  (void)::close(dir);
}

// A model's kept directory made unwritable after the start's checks, and
// an adopted slot's name made a link to nothing meanwhile: the model still
// registers, each slot spilling to an unnamed file (said, and kept for
// this run only), the adoption is refused and nothing in the directory is
// touched; the slots serve as fresh ones.
TEST_F(EngineLlmTest, ADirectoryUnwritableAfterTheChecksKeepsNothingAndStarts) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::Llm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    (void)Turn(0, Prompt(220, 6), 10, false);
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(0)));
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    llm_->PreserveKeptFiles();
    Stop();
  }
  auto text = pf::ReadPrivateFile(dir, kept::RecordFileName(0).c_str(), kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record.has_value()) << record.error();
  EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
  std::vector<kept::FileId> ids;
  for (const kept::StateFile& file : record->files) ids.push_back(file.id);
  // After the checks: the stage-0 name a link to nothing, the directory
  // closed to writes.
  const std::string name = kept::StateFileName(0, 0);
  fs::remove(directory / name);
  fs::create_symlink(root_ / "nothing", directory / name);
  const auto listing = [&] {
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(directory)) {
      names.push_back(e.path().filename().string());
    }
    std::ranges::sort(names);
    return names;
  };
  const std::vector<std::string> before = listing();

  // Writable again however the test ends, so TearDown removes it.
  struct Reopen {
    fs::path path;
    explicit Reopen(fs::path p) : path(std::move(p)) {}
    Reopen(const Reopen&) = delete;
    Reopen& operator=(const Reopen&) = delete;
    Reopen(Reopen&&) = delete;
    Reopen& operator=(Reopen&&) = delete;
    ~Reopen() {
      std::error_code ignored;
      fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace, ignored);
    }
  } const reopen(directory);
  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::Llm& llm) {
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0},
                  .files = {ids}});
    fs::permissions(directory, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace);
  });
  const std::vector<std::string> unkept = llm_->UnkeptSlots();
  ASSERT_EQ(unkept.size(), std::size_t{kSlots});
  EXPECT_NE(unkept[0].find("slot 0: the spill file"), std::string::npos) << unkept[0];
  rt::MemoryCharge charge;
  auto adopted = llm_->Adopt(BranchAt(0), *record, charge);
  ASSERT_FALSE(adopted);
  EXPECT_NE(Why(adopted).find("not the one its record names"), std::string::npos) << Why(adopted);
  EXPECT_TRUE(BranchAt(0).ReleaseIdleState());  // as the server does with one refused
  for (const std::size_t slot : {std::size_t{0}, std::size_t{1}}) {
    const auto prompt = Prompt(90, 30 + static_cast<std::int32_t>(slot));
    const Turned turned = Turn(slot, prompt, 6);
    EXPECT_EQ(turned.out.tokens, Chain(prompt, 6));
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(slot)));
  }
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace);
  EXPECT_EQ(listing(), before);
  EXPECT_TRUE(fs::is_symlink(directory / name));
  Stop();
  (void)::close(dir);
}

// Setup-only controls still own real runner mappings, staging and callbacks.
// Retire through the node while the serving adapter remains alive, including
// an early ASSERT return; dropping the adapter never authenticates a fence.
class SetupNodeRetirement {
 public:
  explicit SetupNodeRetirement(en::PagedNode& owner) : node_(owner) {}
  ~SetupNodeRetirement() {
    const auto retired = node_.TearDown({});
    EXPECT_TRUE(retired) << Why(retired);
  }

 private:
  en::PagedNode& node_;
};

// An adapter without its parts, or whose parts are refused, refuses Setup
// with its reason; nothing else runs.
TEST_F(EngineLlmTest, SetupRefusesWhatItCannotBuild) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
  ASSERT_TRUE(node.Open());
  // The adapters below hold no runner, so retiring after them is safe.
  const SetupNodeRetirement retirement(node);
  rt::Llm unbuilt(node, entry_, Settings(false), roles_, 0, {});
  EXPECT_FALSE(unbuilt.Setup());
  rt::Llm refused(node, entry_, Settings(false), roles_, 0,
                  [](const rt::ModelSettings&) -> std::expected<rt::LlmBuildParts, std::string> {
                    return std::unexpected(std::string("no step source yet"));
                  });
  auto set = refused.Setup();
  ASSERT_FALSE(set);
  EXPECT_NE(Why(set).find("no step source yet"), std::string::npos);
  EXPECT_FALSE(refused.SelectBranches({}));
  // Speculation on with parts that have no drafter: refused, never plain
  // in its place (M3's classes refused it too).
  rt::Llm plain(node, entry_, Settings(true), roles_, 0, Build(false));
  auto speculating = plain.Setup();
  ASSERT_FALSE(speculating);
  EXPECT_NE(Why(speculating).find("speculation is on but its parts have no drafter"),
            std::string::npos);
  // No runner: a server stopping after this failed start leaves it out of
  // the node's teardown (Server::TearDown) instead of asking for it.
  EXPECT_FALSE(unbuilt.has_paged());
  EXPECT_FALSE(refused.has_paged());
}

// A request's thinking when it names none: the slice's default (DeepSeek
// V4's on, as its M3 class served it), the template's own when unset, and
// off where thinking is not served; a request's own choice is kept.
TEST_F(EngineLlmTest, ConversationDefaultsFollowTheChatSlice) {
  struct Case {
    bool thinking;
    std::optional<bool> default_thinking;
    std::optional<bool> asked;
    std::optional<bool> expected;
  };
  for (const auto c :
       {Case{true, std::nullopt, std::nullopt, std::nullopt}, Case{true, true, std::nullopt, true},
        Case{true, false, std::nullopt, false}, Case{true, true, false, false},
        Case{true, false, true, true}, Case{false, std::nullopt, std::nullopt, false},
        Case{false, true, std::nullopt, false}, Case{false, true, true, true}}) {
    SCOPED_TRACE(
        std::format("{} {} {}", c.thinking, c.default_thinking.has_value(), c.asked.has_value()));
    en::PagedNode node(en::NodeSettings{
        .compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
    ASSERT_TRUE(node.Open());
    auto build = Build(false);
    rt::Llm llm(
        node, entry_, Settings(false), roles_, 0,
        [build,
         c](const rt::ModelSettings& settings) -> std::expected<rt::LlmBuildParts, std::string> {
          auto parts = build(settings);
          if (!parts) return parts;
          parts->chat.thinking = c.thinking;
          parts->chat.default_thinking = c.default_thinking;
          return parts;
        });
    const SetupNodeRetirement retirement(node);
    const auto set = llm.Setup();
    ASSERT_TRUE(set) << Why(set);
    llmp::chat::Conversation conversation;
    conversation.enable_thinking = c.asked;
    llm.Defaults(conversation);
    EXPECT_EQ(conversation.enable_thinking, c.expected);
  }
}

TEST_F(EngineLlmTest, CompleteDraftScheduleReservesOnlyItsEnabledDepth) {
  struct Case {
    bool enabled;
    std::uint32_t depth;
    bool complete;
    bool adaptive;
    std::uint32_t target_context;
    std::uint32_t usable;
  };
  for (const auto c :
       {Case{true, 2, true, false, kContext, 1022}, Case{true, 3, true, true, kContext, 1021},
        Case{false, 3, true, true, kContext, 1024}, Case{true, 3, false, true, kContext, 1024},
        Case{true, 2, true, false, 2, 0}}) {
    SCOPED_TRACE(c.enabled);
    SCOPED_TRACE(c.depth);
    SCOPED_TRACE(c.target_context);
    en::PagedNode node(en::NodeSettings{
        .compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
    ASSERT_TRUE(node.Open());
    auto build = Build(true);
    rt::Llm llm(
        node, entry_, Settings(c.enabled), roles_, 0,
        [build,
         c](const rt::ModelSettings& settings) -> std::expected<rt::LlmBuildParts, std::string> {
          auto parts = build(settings);
          if (!parts) return parts;
          parts->mode.drafters.resize(1);
          auto& draft = parts->mode.drafters.front();
          draft.max_depth = c.depth;
          draft.draft_before_limit = c.complete;
          draft.adaptive = c.adaptive;
          parts->branch_depth = 3;
          parts->config.slots.stages.front().layout.context = c.target_context;
          return parts;
        });
    const SetupNodeRetirement retirement(node);
    const auto set = llm.Setup();
    if (c.usable == 0) {
      ASSERT_FALSE(set);
      EXPECT_NE(Why(set).find("complete draft schedule does not fit"), std::string::npos);
      EXPECT_FALSE(llm.has_paged());
    } else {
      ASSERT_TRUE(set) << Why(set);
      EXPECT_EQ(llm.usable_context(), c.usable);
    }
  }
}

TEST_F(EngineLlmTest, NativeVerifyTokenRangeRefusesBeforeAcceptance) {
  native_outputs_ = true;
  Start(true, true);
  Open({&BranchAt(0)});
  const auto prompt = Prompt(37, 7);
  std::vector<float> head;
  ASSERT_TRUE(BranchAt(0).Prefill(prompt, head));
  source_->options().native_token_phase = pl::Phase::kVerify;
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  for (const auto token : {-1, static_cast<std::int32_t>(kWidth)}) {
    source_->options().native_token = token;
    rt::Generation bad;
    const auto run = BranchAt(0).Generate(head, options, bad);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("outside the target vocabulary"), std::string::npos);
    EXPECT_EQ(bad.tokens.size(), 1U);  // only the already valid prefill choice
    EXPECT_EQ(llm_->runner().slots().positions(0), prompt.size());
    EXPECT_TRUE(llm_->runner().slots().usable(0));
  }
  source_->options().native_token.reset();
  rt::Generation good;
  ASSERT_TRUE(BranchAt(0).Generate(head, options, good));
  EXPECT_EQ(good.tokens, Chain(prompt, 8));
  Close();
}

// Native proposal tokens (followed by next features) and multirow verify
// tokens use the ordinary serving sampler/state path, including a truncated
// final verify and the branch-owned adaptive state preserved by snapshots.
TEST_F(EngineLlmTest, NativeDraftAndVerifyTokensPreserveRowsAndSnapshotContinuation) {
  native_outputs_ = true;
  Start(true, true);
  const auto prompt = Prompt(37, 7);
  const auto tokens = Turn(0, prompt, 18);
  EXPECT_EQ(tokens.out.tokens, Chain(prompt, 18));
  EXPECT_GT(tokens.out.drafted, 0U);
  const auto rows = Turn(1, prompt, 18, true);
  EXPECT_EQ(rows.out.tokens, tokens.out.tokens);
  ASSERT_EQ(rows.out.logits.size(), 18U);
  for (std::size_t i = 0; i < rows.out.tokens.size(); ++i)
    EXPECT_EQ(ArgmaxOf(rows.out.logits[i]), rows.out.tokens[i]);
  Open({&BranchAt(0)});
  rt::SnapshotStore store(*node_, 8 * kExtent);
  auto captured = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(captured) << captured.error();
  const auto original = BranchAt(0).history();
  auto next = original;
  next.push_back(tokens.out.tokens.back());
  std::vector<float> head;
  std::uint32_t reused = 0;
  ASSERT_TRUE(BranchAt(0).PreparePrompt(next, 0, head, reused));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  rt::Generation first;
  ASSERT_TRUE(BranchAt(0).Generate(head, options, first));
  ASSERT_TRUE(BranchAt(0).RestoreState(store, *captured));
  EXPECT_EQ(BranchAt(0).history(), original);
  ASSERT_TRUE(BranchAt(0).PreparePrompt(next, 0, head, reused));
  rt::Generation again;
  ASSERT_TRUE(BranchAt(0).Generate(head, options, again));
  EXPECT_EQ(again.tokens, first.tokens);
  EXPECT_EQ(again.drafted, first.drafted);
  EXPECT_EQ(again.accepted, first.accepted);
  Close();
  // Changed prompt inputs still use the captured step programs.
  const auto changed = Turn(0, Prompt(37, 11), 18);
  EXPECT_EQ(changed.out.tokens, Chain(Prompt(37, 11), 18));
  EXPECT_NE(changed.out.tokens, tokens.out.tokens);
}

TEST_F(EngineLlmTest, FreshPromptPredictionUsesFutureSamplingAndReplaysTheCorrectDecodePlan) {
  Start(false, false);
  Open({&BranchAt(0), &BranchAt(1)});
  for (std::uint32_t owner = 0; owner < 2; ++owner) {
    const bool sampled = owner == 1;
    rt::GenerateOptions options;
    options.max_tokens = 3;
    options.stop = false;
    if (sampled)
      options.sampling =
          llmp::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
    auto prompt = BranchAt(owner).BeginPrompt(Prompt(kChunk, static_cast<std::int32_t>(owner + 1)));
    ASSERT_TRUE(prompt) << Failed(prompt).value_or("");
    (*prompt)->SetContinuationRows(rt::PromptTargetRows(options.max_tokens, 0, false), sampled);
    while (!(*prompt)->done()) ASSERT_TRUE((*prompt)->Advance());
    auto head = (*prompt)->last();
    ASSERT_TRUE((*prompt)->Finish());
    bool predicted = false;
    for (const auto& built : source_->built()) {
      if (built.measuring || built.shape.phase != pl::Phase::kDecode ||
          built.shape.segments.size() != 1 || built.shape.segments[0].slot != owner)
        continue;
      predicted = true;
      EXPECT_EQ(built.shape.segments[0].greedy, !sampled);
    }
    ASSERT_TRUE(predicted);
    const auto before = source_->built().size();
    rt::Generation out;
    auto generation = BranchAt(owner).BeginGeneration(head, options, out);
    ASSERT_TRUE(generation) << Failed(generation).value_or("");
    const std::array<rt::Llm::GenerationSession*, 1> one{generation->get()};
    ASSERT_TRUE(llm_->RunGenerationWave(one));
    EXPECT_EQ(source_->built().size(), before);
    if (!(*generation)->done()) (*generation)->Cancel();
    ASSERT_TRUE((*generation)->Finish());
  }
  Close();
}

TEST_F(EngineLlmTest, PromptPolicyObservesTheActivePeersBudgetAfterItsCompletedApply) {
  Start(false, false);
  Open({&BranchAt(0), &BranchAt(1)});
  std::vector<float> head;
  ASSERT_TRUE(BranchAt(0).Prefill(Prompt(kChunk, 1), head));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  rt::Generation out;
  auto generation = BranchAt(0).BeginGeneration(head, options, out);
  ASSERT_TRUE(generation) << Failed(generation).value_or("");
  const std::array<rt::Llm::GenerationSession*, 1> one{generation->get()};
  ASSERT_TRUE(llm_->RunGenerationWave(one));
  ASSERT_EQ(out.tokens.size(), 2U);  // Two target rows remain before PrepareStep is called again.
  auto prompt = BranchAt(1).BeginPrompt(Prompt(kChunk, 2));
  ASSERT_TRUE(prompt) << Failed(prompt).value_or("");
  (*prompt)->SetContinuationRows(5, false);
  while (!(*prompt)->done()) ASSERT_TRUE((*prompt)->Advance());
  ASSERT_TRUE((*prompt)->Finish());
  bool joined = false, actual_tail = false;
  for (const auto& built : source_->built()) {
    if (built.measuring || built.shape.phase != pl::Phase::kDecode) continue;
    if (built.shape.segments.size() == 2) {
      joined = true;
      EXPECT_EQ(built.shape.segments[0].past, kChunk + 1);
      EXPECT_EQ(built.shape.segments[1].past, kChunk);
    } else if (built.shape.segments.size() == 1 && built.shape.segments[0].slot == 1 &&
               built.shape.segments[0].past == kChunk + 2) {
      actual_tail = true;
    }
  }
  EXPECT_TRUE(joined);
  EXPECT_TRUE(actual_tail);
  (*generation)->Cancel();
  ASSERT_TRUE((*generation)->Finish());
  Close();
}

TEST_F(EngineLlmTest, OneTokenFreshPromptDoesNotBuildUnusedPostPrefillWork) {
  Start(false, false);
  Open({&BranchAt(0)});
  auto prompt = BranchAt(0).BeginPrompt(Prompt(kChunk, 1));
  ASSERT_TRUE(prompt) << Failed(prompt).value_or("");
  (*prompt)->SetContinuationRows(rt::PromptTargetRows(1, 0, false), false);
  while (!(*prompt)->done()) ASSERT_TRUE((*prompt)->Advance());
  ASSERT_TRUE((*prompt)->Finish());
  for (const auto& built : source_->built()) {
    if (!built.measuring) EXPECT_NE(built.shape.phase, pl::Phase::kDecode);
  }
  Close();
}

TEST_F(EngineLlmTest, CapabilityAdmittedRaggedPromptSessionsUseOneActualSharedStep) {
  Start(false, false);
  Open({&BranchAt(0), &BranchAt(1)});
  EXPECT_TRUE(llm_->CompatiblePrefill(0, 1, 0, 2));
  EXPECT_FALSE(llm_->CompatiblePrefill(0, 1, 0, 2, true, false));
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> prompts;
  const std::array<std::uint32_t, 2> lengths{3, 5};
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    auto begun = BranchAt(i).BeginPrompt(Prompt(lengths[i], static_cast<std::int32_t>(i + 1)));
    ASSERT_TRUE(begun);
    prompts[i] = std::move(*begun);
    while (!prompts[i]->done()) {
      auto next = prompts[i]->NextUnit();
      ASSERT_TRUE(next);
      if (next->phase == rt::Llm::PromptSession::Phase::kChunk) break;
      ASSERT_TRUE(prompts[i]->Advance());
    }
  }
  ASSERT_TRUE(prompts[0]->CanJoin(*prompts[1]));
  const std::array<rt::Llm::PromptSession*, 2> both{prompts[0].get(), prompts[1].get()};
  const std::array<rt::PrefillGoOn, 2> go_on{};
  ASSERT_TRUE(llm_->RunPromptWave(both, go_on));
  EXPECT_EQ(llm_->counts().joined_prefill_steps, 1U);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    EXPECT_TRUE(prompts[i]->done());
    EXPECT_EQ(llm_->runner().slots().positions(static_cast<std::uint32_t>(i)), lengths[i]);
    ASSERT_TRUE(prompts[i]->Finish());
  }
  Close();
}

TEST(EngineLlmPartsTest, UninstalledPartsDestroyConfigBorrowersBeforeTheirSourceOwner) {
  struct Owner {
    explicit Owner(bool* value) : alive(value) { *alive = true; }
    ~Owner() { *alive = false; }
    bool* alive;
  };
  struct Borrower {
    const bool* alive;
    bool* observed;
    ~Borrower() { *observed = *alive; }
  };
  bool owner_lived = false, alive = false;
  {
    llmp::runtime::LlmBuildParts parts;
    auto owner = std::make_shared<Owner>(&alive);
    auto borrowed = std::make_shared<Borrower>(&alive, &owner_lived);
    parts.keep = owner;
    parts.config.retire = [borrowed] {};
  }
  EXPECT_TRUE(owner_lived);
}

TEST_F(EngineLlmTest, CertainSpanTruncatesBeforeStopTokenAndPreservesBranchStateInvariants) {
  Start(false, false);
  Open({&BranchAt(0)});
  const auto prompt = Prompt(20, 3);
  std::vector<float> head;
  ASSERT_TRUE(BranchAt(0).Prefill(prompt, head));

  llmp::tokenizer::TokenizerSpec spec;
  spec.pre_tokenizer = llmp::tokenizer::PreTokenizer::kRawBpe;
  constexpr std::string_view hex = "0123456789ABCDEF";
  for (unsigned b = 0; b < 256; ++b) {
    std::string value = "<0x00>";
    value[3] = hex[b >> 4U];
    value[4] = hex[b & 15U];
    spec.tokens.push_back(std::move(value));
    spec.kinds.push_back(llmp::tokenizer::TokenKind::kByte);
  }
  for (char ch = 'a'; ch <= 'z'; ++ch) {
    spec.tokens.emplace_back(1, ch);
    spec.kinds.push_back(llmp::tokenizer::TokenKind::kNormal);
  }
  spec.tokens.emplace_back("▁");
  spec.kinds.push_back(llmp::tokenizer::TokenKind::kNormal);
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();

  const llmp::base::json::Limits limits{
      .max_bytes = 65536, .max_depth = 32, .max_values = 65536, .max_string_bytes = 65536};
  const std::vector<llmp::chat::ToolDefinition> defs{
      {.name = "lookup", .description = "find", .parameters_json = R"({"type":"object"})"}};
  auto request = llmp::chat::ToolRequest::Create(defs, {}, true, limits);
  ASSERT_TRUE(request) << request.error();
  auto tool_sampling =
      std::make_shared<llmp::chat::ToolSampling>(*tokenizer, llmp::chat::QwenToolCalls(), *request,
                                                 limits, false, std::span<const std::int32_t>{});

  const auto chain = Chain(prompt, 5);
  const std::array<std::int32_t, 1> extra_stops{chain[2]};
  rt::GenerateOptions options;
  options.max_tokens = 10;
  options.stop = true;
  options.extra_stops = extra_stops;
  options.tool_sampling = tool_sampling;
  rt::Generation out;
  auto generation = BranchAt(0).BeginGeneration(head, options, out);
  ASSERT_TRUE(generation) << Failed(generation).value_or("");
  ASSERT_EQ(out.tokens, (std::vector<std::int32_t>{chain[0]}));

  // Queue a multi-token certain span containing a stop token (chain[2]) followed by chain[3].
  tool_sampling->Certain({chain[1], chain[2], chain[3]});
  const std::array<rt::Llm::GenerationSession*, 1> one{generation->get()};
  ASSERT_TRUE(llm_->RunGenerationWave(one));
  EXPECT_TRUE((*generation)->done());
  EXPECT_TRUE(out.stopped);
  EXPECT_EQ(out.tokens, (std::vector<std::int32_t>{chain[0], chain[1], chain[2]}));
  // Unconsumed certain tokens after the stop token remain preserved on tool_sampling.
  ASSERT_EQ(tool_sampling->certain().size(), 1U);
  EXPECT_EQ(tool_sampling->certain()[0], chain[3]);
  ASSERT_TRUE((*generation)->Finish());

  // Branch history and KV slot positions hold prompt + {chain[0], chain[1]}, while chain[2]
  // is the final anchor not yet committed into KV.
  EXPECT_EQ(BranchAt(0).history().size(), prompt.size() + 2);
  EXPECT_EQ(llm_->runner().slots().positions(0), prompt.size() + 2);
  Close();

  const auto continued = Continue(0, out, {});
  const Turned next = Turn(0, continued, 4);
  EXPECT_EQ(next.reused, prompt.size() + 2);
  EXPECT_EQ(next.out.tokens, Chain(continued, 4));
}

}  // namespace
