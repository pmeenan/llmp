// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Concrete Llm teacher-forcing and generation on the real runner and fake
// device/step source. These controls test position, state,
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
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/work_pulse.h"
#include "chat/chat.h"
#include "llm_fixture.h"
#include "memory/reclaim.h"
#include "providers/fake/fake_device_runtime.h"
#include "runtime/service_request.h"
#include "runtime/serving.h"
#include "scheduler/cohort_policy.h"
#include "scheduler/cohort_schedule.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"
#include "tokenizer_fixtures.h"

namespace {
using namespace llmp::test_support::llm_fixture;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

std::vector<float> Row(std::int32_t token) {
  std::vector<float> row(kWidth, 0.0F);
  row.at(static_cast<std::size_t>(token)) = 1.0F;
  return row;
}

class LlmScores : public LlmFixture {
 protected:
  void StartScores(bool speculative = false, std::uint32_t slots = 4, std::uint32_t context = 64,
                   std::uint32_t chunk = 1) {
    context_rows_ = context;
    chunk_rows_ = chunk;
    slot_count_ = slots;
    native_outputs_ = speculative;
    Start(speculative, speculative);
    ASSERT_NE(llm_, nullptr);
    std::vector<rt::Llm::Branch*> branches;
    for (std::size_t i = 0; i < llm_->branches(); ++i) branches.push_back(&BranchAt(i));
    Open(std::move(branches));
  }
  std::vector<std::int32_t> State(std::size_t slot, std::size_t stage = 0) {
    const auto& slots = llm_->runner().slots();
    std::vector<std::int32_t> tokens;
    const auto& item = slots.layout(stage).items.at(stage == 0 ? 0 : 1);
    for (std::uint32_t i = 0; i < slots.positions(static_cast<std::uint32_t>(slot)); ++i) {
      ts::FakeCell cell;
      const auto address = slots.live(static_cast<std::uint32_t>(slot), stage).base(0) +
                           item.offset + (i % item.cells) * item.contract.cell_bytes;
      std::memcpy(&cell, reinterpret_cast<const void*>(address), sizeof(cell));
      EXPECT_EQ(cell.magic, ts::kFakeMagic);
      EXPECT_EQ(cell.position, i);
      EXPECT_EQ(cell.slot, slot);
      tokens.push_back(cell.token);
    }
    return tokens;
  }
  void Recover() {
    source_->options().fail_run.reset();
    std::ignore = node_->EndRequest(0);
    ASSERT_TRUE(node_->Job(
        llmp::catalog::Closure{},
        [](llmp::providers::NativeStream) { return llmp::scheduler::JobResult::kQueued; },
        "fencing a faulted test stream", 0));
    ASSERT_TRUE(llm_->RecoverInPlace());
  }
  void Room(std::uint64_t extents) {
    const auto stats = node_->Stats();
    ASSERT_TRUE(stats);
    auto set = node_->Call(
        [&]() -> rt::Status {
          (void)node_->ApplyBudget(
              Bytes(stats->catalog_occupancy_bytes + extents * kExtent + kExtent / 2));
          return {};
        },
        "setting the score test budget");
    ASSERT_TRUE(set);
  }
  rt::RequestMemory& Fund(std::uint64_t bytes) {
    memory_ = std::make_unique<rt::RequestMemory>(bytes);
    llm_->SetTokenMemory(*memory_);
    return *memory_;
  }
  std::unique_ptr<rt::RequestMemory> memory_;
};
class TokenHistory : public LlmScores {};
class LlmRender : public LlmFixture {};

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
  llmp::tokenizer::unicode::AppendUtf8(printable(b) ? b : 256 + n, out);
  return out;
}

llmp::tokenizer::TokenizerSpec ByteTokenizerSpec() {
  llmp::tokenizer::TokenizerSpec spec;
  for (unsigned b = 0; b < 256; ++b) {
    spec.tokens.push_back(ByteText(b));
    spec.kinds.push_back(llmp::tokenizer::TokenKind::kNormal);
  }
  for (const char* control : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}) {
    spec.tokens.emplace_back(control);
    spec.kinds.push_back(llmp::tokenizer::TokenKind::kControl);
  }
  spec.eos = 258;
  return spec;
}
llmp::tokenizer::Tokenizer ByteTokenizer() {
  auto spec = ByteTokenizerSpec();
  auto t = llmp::tokenizer::Tokenizer::Create(std::move(spec));
  EXPECT_TRUE(t.has_value()) << t.error().ToString();
  return std::move(*t);
}

TEST_F(LlmRender, CompactSourceOutputRetainsItsExactFundedCapacityThroughCacheOwnership) {
  auto spec = ByteTokenizerSpec();
  spec.has_post_processor = true;
  spec.post_processor.ids.assign(512, 257);
  spec.post_processor.prefix.assign(256, {0, 512});
  spec.post_processor.prefix.push_back({0, 1});
  spec.bos = 258;
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(tokenizer);
  en::PagedNode node(en::NodeSettings{});
  rt::Llm model(node, entry_, Settings(false), roles_, 0, {});
  ASSERT_TRUE(
      rt::LlmTestAccess::UseChat(model, std::move(*tokenizer), "{{ messages[0].content }}"));
  const auto capacity = model.tokenizer().OutputTokensBound(1, true) + 1;
  const auto output_bytes = capacity * sizeof(std::int32_t);
  const auto need = model.tokenizer().WorkingBytes(1) + output_bytes;
  rt::RequestMemory too_small(need - 1);
  std::uint64_t reported = 0;
  const auto refused = model.EncodeText("a", &too_small, &reported);
  EXPECT_FALSE(refused);
  EXPECT_EQ(reported, need);
  EXPECT_EQ(too_small.used(), 0U);

  rt::RequestMemory memory(need);
  {
    auto encoded = model.EncodeText("a", &memory);
    ASSERT_TRUE(encoded) << (encoded ? "" : encoded.error());
    EXPECT_EQ(encoded->tokens.capacity(), capacity);
    EXPECT_EQ(encoded->tokens.size(), 131075U);  // endpoint BOS + source prefix + a
    EXPECT_EQ(encoded->tokens.front(), 258);
    EXPECT_EQ(encoded->tokens.back(), 'a');
    EXPECT_EQ(encoded->charge.bytes(), output_bytes);
    EXPECT_EQ(memory.used(), output_bytes);  // temporary work was released
    std::map<std::string, rt::EncodedText> cache;
    encoded->tokens.resize(16);
    cache.emplace("model", std::move(*encoded));
    EXPECT_EQ(memory.used(), output_bytes);
    const std::span<const std::int32_t> borrowed = cache.at("model").tokens;
    EXPECT_EQ(borrowed.data(), cache.at("model").tokens.data());
    EXPECT_EQ(borrowed.size(), 16U);
    cache.clear();
    EXPECT_EQ(memory.used(), 0U);
  }
  EXPECT_EQ(memory.used(), 0U);
  const auto invalid = model.EncodeText(std::string(1, static_cast<char>(0xff)), &memory);
  EXPECT_FALSE(invalid);
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(LlmRender, AChatOverFourMiBRendersWithinTheContext) {
  context_rows_ = 8'000'000;
  en::PagedNode node(en::NodeSettings{});
  rt::Llm model(node, entry_, Settings(false), roles_, 0, {});

  ASSERT_TRUE(rt::LlmTestAccess::UseChat(
                  model, ByteTokenizer(),
                  "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
                  "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>"
                  "assistant\n{% endif %}")
                  .has_value());
  // At least the floor; four times the context times the longest token.
  EXPECT_GE(model.render_bytes(), std::size_t{32} << 20U);
  const std::size_t bytes = (std::size_t{4} << 20U) + 300'000;  // past 2^22 tokens too
  llmp::chat::Conversation c;
  c.messages.push_back({llmp::chat::Role::kUser, std::string(bytes, 'x'), std::nullopt, {}});
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

TEST_F(LlmRender, ContextTextReadCannotGrowPastTheFundedFileSize) {
  // Larger than the string's inline storage, small enough to isolate the
  // actual fixed-buffer contract without a model or filesystem race.
  std::string storage(33, '\0');
  const auto capacity = storage.capacity();
  const auto* address = storage.data();
  const std::string input(33, 'a');
  std::istringstream exact(input);
  ASSERT_TRUE(rt::ReadContextText(exact, std::span(storage)));
  EXPECT_EQ(storage, input);
  EXPECT_EQ(storage.capacity(), capacity);
  EXPECT_EQ(storage.data(), address);
  for (const auto& changed : {std::string(32, 'a'), std::string(34, 'a')}) {
    std::istringstream stream(changed);
    EXPECT_FALSE(rt::ReadContextText(stream, std::span(storage)));
    EXPECT_EQ(storage.capacity(), capacity);
    EXPECT_EQ(storage.data(), address);
  }
  std::istringstream broken(input);
  broken.setstate(std::ios::badbit);
  EXPECT_FALSE(rt::ReadContextText(broken, std::span(storage)));
  std::istringstream empty;
  EXPECT_FALSE(rt::ReadContextText(empty, {}));
  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  std::istringstream cancelled(input);
  const auto result = rt::ReadContextText(cancelled, std::span(storage));
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_FALSE(result);
  EXPECT_EQ(cancelled.tellg(), std::streampos{0});
}

TEST(LlmReclaim, SpillRoomCostFollowsTheDeletionOrder) {
  const auto at = [](int seconds) { return rt::Clock::time_point(std::chrono::seconds(seconds)); };
  const std::vector<rt::SpillVictim> victims = {
      {.conversation = true, .used = at(20), .counted = 30, .seconds = 7.0},
      {.conversation = false, .used = at(30), .counted = 10, .seconds = 0.0},
      {.conversation = true, .used = at(10), .counted = 30, .seconds = 3.0},
  };
  // 70 spilled within 100: 30 more fit as they are.
  EXPECT_EQ(rt::SpillRoomCost(victims, 70, 0, 100, 30), 0.0);
  // 40 more: the retained file goes first, at no cost.
  EXPECT_EQ(rt::SpillRoomCost(victims, 70, 0, 100, 40), 0.0);
  // 50 more: then the older conversation (3 s), not the newer.
  EXPECT_EQ(rt::SpillRoomCost(victims, 70, 0, 100, 50), 3.0);
  // 90 more: both conversations.
  EXPECT_EQ(rt::SpillRoomCost(victims, 70, 0, 100, 90), 10.0);
  // Held checkpoints stay: 95 cannot fit beside 10 of them.
  EXPECT_EQ(rt::SpillRoomCost(victims, 70, 10, 100, 95), std::nullopt);
  EXPECT_EQ(rt::SpillRoomCost({}, 0, 0, 100, 101), std::nullopt);
}

TEST(LlmReclaim, RoomForAsksTheReclaimOnlyForTheShortfall) {
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

TEST_F(LlmScores, RowFreeGreedyEligibilityAndRequestedScoresUseTheActualDispatch) {
  StartScores();
  auto& branch = BranchAt(0);
  for (const bool scalar : {false, true}) {
    for (const bool scores : {false, true}) {
      for (const bool sampling : {false, true}) {
        ASSERT_TRUE(branch.Clear());
        std::vector<float> last;
        const auto prompt = Prompt(1, 3);
        ASSERT_TRUE(branch.Prefill(prompt, last));
        rt::GenerateOptions options;
        options.max_tokens = 2;
        options.keep_logits = scores;
        if (sampling) options.sampling = llmp::execution::SamplingParams{.temperature = 0.5F};
        const auto before = llm_->counts().greedy_tokens;
        rt::Generation out;
        auto session = branch.BeginGeneration(last, options, out);
        ASSERT_TRUE(session);
        if (scalar)
          ASSERT_TRUE((*session)->RunScalarStep());
        else {
          const std::array<rt::Llm::GenerationSession*, 1> one{session->get()};
          ASSERT_TRUE(llm_->RunGenerationWave(one));
        }
        ASSERT_TRUE((*session)->Finish());
        EXPECT_EQ(llm_->counts().greedy_tokens - before, !scores && !sampling ? 1U : 0U);
        EXPECT_EQ(out.tokens.size(), 2U);
        EXPECT_EQ(out.logits.size(), scores ? 2U : 0U);
        EXPECT_EQ(State(0), branch.history());
        if (!sampling) EXPECT_EQ(out.tokens, Chain(prompt, 2));
      }
    }
  }
}

TEST_F(LlmScores, PlainWithoutDeviceOutputRetainsTheCompleteRowPath) {
  device_greedy_ = false;
  StartScores();
  const auto prompt = Prompt(1, 3);
  std::vector<float> last;
  ASSERT_TRUE(llm_->Prefill(prompt, last));
  rt::GenerateOptions options;
  options.max_tokens = 2;
  rt::Generation out;
  ASSERT_TRUE(llm_->Generate(last, options, out));
  EXPECT_EQ(llm_->counts().greedy_tokens, 0U);
  EXPECT_EQ(out.tokens, Chain(prompt, 2));
  EXPECT_EQ(State(0), llm_->history());
}

TEST_F(LlmScores, FailedPreparedDeviceTokenDoesNotPublishTheWrittenOutput) {
  for (const bool scalar : {false, true}) {
    StartScores();
    const auto prompt = Prompt(1, 3);
    std::vector<float> last;
    ASSERT_TRUE(llm_->Prefill(prompt, last));
    rt::GenerateOptions options;
    options.max_tokens = 2;
    rt::Generation out;
    auto session = llm_->BeginGeneration(last, options, out);
    ASSERT_TRUE(session);
    const auto before = source_->executed();
    source_->options().fail_run = kn::StepError::kUnknown;
    if (scalar)
      EXPECT_FALSE((*session)->RunScalarStep());
    else {
      const std::array<rt::Llm::GenerationSession*, 1> one{session->get()};
      // A completed failure belongs to this owner; the wave itself remains usable.
      EXPECT_TRUE(llm_->RunGenerationWave(one));
    }
    EXPECT_EQ(source_->executed(), before + 1);
    EXPECT_EQ(out.tokens, Chain(prompt, 1));
    EXPECT_FALSE((*session)->Finish());
    EXPECT_TRUE(llm_->history().empty());
    EXPECT_FALSE(llm_->runner().slots().usable(0));
    session->reset();
    Recover();
    Stop();
  }
}

TEST_F(LlmScores, TeacherForcesStopIdsAndKeepsAllInjectionState) {
  StartScores(true);
  rt::LlmTestAccess::Stops(*llm_, {7});
  const std::array<std::int32_t, 4> prompt{0, 7, 2, 3};
  std::vector<float> last;
  std::vector<std::int32_t> scored;
  std::vector<std::uint32_t> sizes;
  rt::PrefillRun run;
  ASSERT_TRUE(llm_->ScorePrompt(
      prompt, last,
      [&](std::int32_t id, std::span<const float> row) {
        const auto position = scored.size();
        EXPECT_EQ(ArgmaxOf(row),
                  ts::FakeLogit(prompt[position], static_cast<std::int32_t>(position), kWidth));
        scored.push_back(id);
        return true;
      },
      [&](std::uint32_t rows) {
        sizes.push_back(rows);
        return true;
      },
      &run));
  EXPECT_THAT(scored, ElementsAre(7, 2, 3));
  EXPECT_THAT(sizes, ElementsAre(1U, 1U, 1U, 1U));
  EXPECT_EQ(llm_->history(), (std::vector<std::int32_t>{0, 7, 2, 3}));
  EXPECT_EQ(State(0), llm_->history());
  EXPECT_EQ(State(0, 1), llm_->history());
  EXPECT_EQ(ArgmaxOf(last), ts::FakeLogit(3, 3, kWidth));
  EXPECT_EQ(run.end, 4U);
  EXPECT_FALSE(run.stopped);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
}

TEST_F(LlmScores, CancellationRetainsOnlyCompletedPrefixAndCanContinue) {
  StartScores(true);
  const std::array<std::int32_t, 4> prompt{0, 1, 2, 3};
  std::vector<float> last;
  rt::PrefillRun run;
  unsigned chunks = 0;
  ASSERT_TRUE(
      llm_->ScorePrompt(prompt, last, {}, [&](std::uint32_t) { return chunks++ < 2; }, &run));
  EXPECT_TRUE(run.stopped);
  EXPECT_TRUE(last.empty());
  EXPECT_THAT(llm_->history(), ElementsAre(0, 1));
  EXPECT_EQ(State(0), State(0, 1));
  ASSERT_TRUE(llm_->Prefill(std::span(prompt).subspan(2), last));
  EXPECT_THAT(llm_->history(), ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(State(0), State(0, 1));
  EXPECT_FALSE(llm_->ScorePrompt(prompt, last, {}));
}

TEST_F(LlmScores, ResumableScoringMatchesSerialRowsInjectionAndLastHead) {
  StartScores(true);
  const auto prompt = Prompt(5, 9);
  using Scored = std::pair<std::int32_t, std::vector<float>>;
  std::vector<Scored> expected, actual;
  std::vector<float> last;
  ASSERT_TRUE(
      BranchAt(0).ScorePrompt(prompt, last, [&](std::int32_t id, std::span<const float> row) {
        expected.emplace_back(id, std::vector<float>(row.begin(), row.end()));
        return true;
      }));
  auto session =
      BranchAt(1).BeginScoringPrompt(prompt, [&](std::int32_t id, std::span<const float> row) {
        actual.emplace_back(id, std::vector<float>(row.begin(), row.end()));
        return true;
      });
  ASSERT_TRUE(session);
  while (!(*session)->done()) {
    auto next = (*session)->NextUnit();
    ASSERT_TRUE(next);
    EXPECT_EQ(next->rows, next->phase == rt::Llm::PromptSession::Phase::kChunk ? 1U : 0U);
    ASSERT_TRUE((*session)->Advance());
  }
  ASSERT_TRUE((*session)->Finish());
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(actual, expected);
  EXPECT_EQ((*session)->last(), last);
  EXPECT_EQ(BranchAt(0).history(), BranchAt(1).history());
  EXPECT_EQ(State(0), State(1));
  EXPECT_EQ(State(0, 1), State(1, 1));
}

// F15a: a prompt that will generate writes its catch-up drafter's rows as
// each chunk commits (CatchUpPrompt: every row but each chunk's first, which
// would read a carried feature); a score that generates nothing leaves them.
TEST_F(LlmScores, OnlyAPromptThatMayGenerateWritesItsDraftersPromptRows) {
  StartScores(true, 4, 64, 8);
  const auto prompt = Prompt(20, 9);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).ScorePrompt(prompt, last,
                                      [](std::int32_t, std::span<const float>) { return true; }));
  std::uint32_t reused = 0;
  ASSERT_TRUE(BranchAt(1).PreparePrompt(prompt, 0, last, reused));
  const auto& slots = llm_->runner().slots();
  const auto& kv = slots.layout(1).items.at(0);
  const auto cell = [&](std::uint32_t slot, std::uint32_t q) {
    ts::FakeCell out;
    std::memcpy(&out,
                reinterpret_cast<const void*>(slots.live(slot, 1).base(0) + kv.offset +
                                              q * kv.contract.cell_bytes),
                sizeof(out));
    return out;
  };
  for (std::uint32_t q = 0; q < prompt.size(); ++q) {
    EXPECT_EQ(cell(0, q).magic, 0) << "scored row " << q;
    const auto drafted = cell(1, q);
    if (q % 8 == 0) {
      EXPECT_EQ(drafted.magic, 0) << "chunk's first row " << q;
      continue;
    }
    EXPECT_EQ(drafted.magic, ts::kFakeMagic) << "row " << q;
    EXPECT_EQ(drafted.position, static_cast<std::int32_t>(q));
    EXPECT_EQ(drafted.slot, 1);
    EXPECT_EQ(drafted.token, prompt[q]);
  }
}

TEST_F(LlmScores, PurePromptSessionsMayFillTheEntireContextButNeverExceedIt) {
  StartScores();
  for (const bool scoring : {false, true}) {
    auto& branch = BranchAt(scoring ? 1 : 0);
    const std::vector<std::int32_t> prompt(context_rows_, 1);
    std::vector<float> serial_last;
    ASSERT_TRUE(branch.Prefill(prompt, serial_last));
    EXPECT_FALSE(branch.Prefill(std::array<std::int32_t, 1>{1}, serial_last));
    unsigned reported = 0;
    auto opened = scoring
                      ? branch.BeginScoringPrompt(prompt,
                                                  [&](std::int32_t id, std::span<const float> row) {
                                                    EXPECT_EQ(id, 1);
                                                    EXPECT_EQ(row.size(), kWidth);
                                                    ++reported;
                                                    return true;
                                                  })
                      : branch.BeginPrompt(prompt, 0, true);
    ASSERT_TRUE(opened);
    while (!(*opened)->done()) ASSERT_TRUE((*opened)->Advance());
    ASSERT_TRUE((*opened)->Finish());
    EXPECT_EQ((*opened)->run().end, context_rows_);
    EXPECT_EQ(branch.history(), prompt);
    EXPECT_EQ(reported, scoring ? context_rows_ - 1 : 0);
    EXPECT_FALSE(branch.BeginPrompt(std::vector<std::int32_t>(context_rows_ + 1, 1)));
  }
}

TEST_F(LlmScores, ResumableScoringSpillsAndContinuesOnlyTheRemainingRows) {
  StartScores(true);
  auto& branch = BranchAt(1);
  const auto prompt = Prompt(6, 3);
  std::vector<std::int32_t> scored;
  const auto row = [&](std::int32_t id, std::span<const float> logits) {
    EXPECT_EQ(ArgmaxOf(logits), ts::FakeLogit(prompt[scored.size()],
                                              static_cast<std::int32_t>(scored.size()), kWidth));
    scored.push_back(id);
    return true;
  };
  auto session = branch.BeginScoringPrompt(prompt, row);
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->Advance());
  for (unsigned i = 0; i < 2; ++i) ASSERT_TRUE((*session)->Advance());
  EXPECT_EQ(scored.size(), 2U);
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  session->reset();
  branch.HoldContinuation();
  Close();
  ASSERT_TRUE(llm_->SpillIdle(branch));
  EXPECT_TRUE(branch.spilled());
  llm_->set_retention(rt::Clock::duration::zero());
  Open({&branch});
  auto resumed = branch.BeginScoringPrompt(prompt, row, true);
  ASSERT_TRUE(resumed);
  while (!(*resumed)->done()) ASSERT_TRUE((*resumed)->Advance());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(scored, (std::vector<std::int32_t>(prompt.begin() + 1, prompt.end())));
  EXPECT_EQ(State(1), prompt);
  EXPECT_EQ(State(1, 1), prompt);
  EXPECT_EQ(ArgmaxOf((*resumed)->last()),
            ts::FakeLogit(prompt.back(), static_cast<std::int32_t>(prompt.size() - 1), kWidth));
  branch.ReleaseContinuation();
}

TEST_F(LlmScores, AnUntouchedScorerDoesNotSettleOrDiscardASpilledContinuation) {
  StartScores();
  auto& branch = BranchAt(1);
  const auto prompt = Prompt(4, 3);
  auto session = branch.BeginScoringPrompt(prompt, {});
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->Advance());
  ASSERT_TRUE((*session)->Advance());
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  session->reset();
  branch.HoldContinuation();
  Close();
  ASSERT_TRUE(llm_->SpillIdle(branch));
  const auto prefix = branch.history();
  const auto executed = source_->executed();
  auto untouched = branch.BeginScoringPrompt(prompt, {}, true);
  ASSERT_TRUE(untouched);
  (*untouched)->Cancel();
  ASSERT_TRUE((*untouched)->Finish());
  untouched->reset();
  EXPECT_EQ(branch.history(), prefix);
  EXPECT_TRUE(branch.spilled());
  EXPECT_EQ(source_->executed(), executed);
  Open({&branch});
  auto continued = branch.BeginScoringPrompt(prompt, {}, true);
  ASSERT_TRUE(continued);
  while (!(*continued)->done()) ASSERT_TRUE((*continued)->Advance());
  ASSERT_TRUE((*continued)->Finish());
  EXPECT_EQ(branch.history(), prompt);
  EXPECT_EQ(State(1), prompt);
  branch.ReleaseContinuation();
}

TEST_F(LlmScores, ResumableScoringStopsAtItsCallbackAndPreservesSettlementFailure) {
  StartScores();
  auto session = llm_->default_branch().BeginScoringPrompt(
      Prompt(3, 3), [](std::int32_t, std::span<const float>) { return false; });
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->Advance());
  ASSERT_TRUE((*session)->Advance());
  EXPECT_TRUE((*session)->done());
  EXPECT_TRUE((*session)->run().stopped);
  EXPECT_EQ(llm_->history().size(), 1U);
  llm_->runner().slots().Fault();
  EXPECT_FALSE((*session)->Finish());
  EXPECT_TRUE(llm_->history().empty());
  EXPECT_TRUE((*session)->last().empty());
  session->reset();
  Recover();
}

TEST_F(LlmScores, UnknownCompletionInvalidatesHistoryBeforeFreshUse) {
  StartScores();
  const auto prompt = Prompt(3, 3);
  std::vector<float> last;
  unsigned chunks = 0;
  auto failed = llm_->ScorePrompt(prompt, last, {}, [&](std::uint32_t) {
    if (chunks++ == 1) source_->options().fail_run = kn::StepError::kUnknown;
    return true;
  });
  EXPECT_FALSE(failed);
  EXPECT_TRUE(llm_->history().empty());
  EXPECT_TRUE(last.empty());
  Recover();
  Open({&BranchAt(0)});
  ASSERT_TRUE(llm_->ScorePrompt(prompt, last, {}));
  EXPECT_EQ(State(0), prompt);
}
TEST_F(LlmScores, GeneratedCallbacksMatchRowsAndStopWithinCompletedVerify) {
  StartScores(true);
  const auto prompt = Prompt(1, 0);
  const auto chain = Chain(prompt, 8);
  std::vector<float> last;
  ASSERT_TRUE(llm_->Prefill(prompt, last));
  std::vector<std::int32_t> seen, visible;
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.keep_logits = true;
  options.on_logits = [&](std::int32_t token, std::span<const float> row) {
    seen.push_back(token);
    EXPECT_EQ(ArgmaxOf(row), token);
    return true;
  };
  options.on_tokens = [&](std::span<const std::int32_t> tokens) {
    visible.insert(visible.end(), tokens.begin(), tokens.end());
    return visible.size() < 2;
  };
  rt::Generation out;
  ASSERT_TRUE(llm_->Generate(last, options, out));
  EXPECT_EQ(out.tokens, (std::vector<std::int32_t>(chain.begin(), chain.begin() + 2)));
  EXPECT_EQ(seen, out.tokens);
  EXPECT_EQ(visible, out.tokens);
  EXPECT_EQ(out.logits.size(), 2U);
  EXPECT_TRUE(out.cancelled);
  EXPECT_EQ(State(0), llm_->history());
  EXPECT_GE(llm_->history().size(), prompt.size() + 1);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
}

TEST_F(LlmScores, GeneratedStopHasScoreButNoVisibleTokenAndBudgetHasNoExtraRow) {
  StartScores();
  const auto prompt = Prompt(1, 0);
  const auto next = Chain(prompt, 1).front();
  rt::LlmTestAccess::Stops(*llm_, {next});
  std::vector<float> last;
  ASSERT_TRUE(llm_->Prefill(prompt, last));
  rt::GenerateOptions options;
  options.max_tokens = 1;
  unsigned rows = 0, visible = 0;
  options.on_logits = [&](std::int32_t id, std::span<const float> row) {
    EXPECT_EQ(id, next);
    EXPECT_EQ(ArgmaxOf(row), next);
    ++rows;
    return true;
  };
  options.on_tokens = [&](std::span<const std::int32_t> tokens) {
    visible += tokens.size();
    return true;
  };
  const auto executed = source_->executed();
  rt::Generation out;
  ASSERT_TRUE(llm_->Generate(last, options, out));
  EXPECT_TRUE(out.stopped);
  EXPECT_EQ(out.tokens, (std::vector<std::int32_t>{next}));
  EXPECT_EQ(rows, 1U);
  EXPECT_EQ(visible, 0U);
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_EQ(llm_->history(), prompt);
}

TEST_F(LlmScores, ScoreCallbackAndSettlementFailuresDoNotSubmitAnotherStep) {
  StartScores();
  const auto prompt = Prompt(3, 3);
  std::vector<float> last;
  const auto before = source_->executed();
  auto ran = llm_->ScorePrompt(prompt, last, [&](std::int32_t, std::span<const float>) {
    llm_->runner().slots().Fault();
    return false;
  });
  EXPECT_FALSE(ran);
  EXPECT_EQ(source_->executed(), before + 1);
  EXPECT_TRUE(llm_->history().empty());
  EXPECT_TRUE(last.empty());
  Recover();
}

TEST_F(LlmScores, InconsistentVerifyRowsRefuseBeforeIndexingAndPreserveSettlementFailure) {
  for (const bool faulted : {false, true}) {
    StartScores(true);
    std::vector<float> last;
    ASSERT_TRUE(llm_->Prefill(Prompt(1, 3), last));
    rt::GenerateOptions options;
    options.max_tokens = 4;
    options.keep_logits = true;
    rt::Generation out;
    auto session = llm_->BeginGeneration(last, options, out);
    ASSERT_TRUE(session);
    ASSERT_TRUE((*session)->PrepareStep());
    if (faulted) llm_->runner().slots().Fault();
    // The public external-completion boundary must reject malformed results
    // before indexing, even when its native lifecycle also cannot settle.
    const auto applied = (*session)->ApplySpeculative({2, 3}, {Row(2)}, 1);
    ASSERT_FALSE(applied);
    EXPECT_THAT(applied.error(), HasSubstr("inconsistent token/logit counts"));
    if (faulted) EXPECT_THAT(applied.error(), HasSubstr("settling failed"));
    EXPECT_FALSE((*session)->Finish());
    EXPECT_EQ(out.tokens.size(), 1U);
    EXPECT_TRUE(llm_->history().empty());
    session->reset();
    if (faulted) Recover();
    Stop();
  }
}

TEST_F(LlmScores, DecodeFailurePreservesItsUnprovenSettlementFailure) {
  StartScores();
  std::vector<float> last;
  ASSERT_TRUE(llm_->Prefill(Prompt(1, 3), last));
  rt::GenerateOptions options;
  options.max_tokens = 2;
  rt::Generation out;
  auto session = llm_->BeginGeneration(last, options, out);
  ASSERT_TRUE(session);
  execution_identity_ = std::unexpected("execution identity unavailable");
  EXPECT_FALSE((*session)->RunScalarStep());
  const auto finished = (*session)->Finish();
  ASSERT_FALSE(finished);
  EXPECT_THAT(finished.error(), HasSubstr("execution identity unavailable"));
  EXPECT_THAT(finished.error(), HasSubstr("settling failed"));
  EXPECT_TRUE(llm_->history().empty());
  execution_identity_ = std::nullopt;
}

TEST_F(LlmScores, OrdinaryKeepLogitsAndStoppedAcceptedTrajectoryStayAligned) {
  for (const bool speculate : {false, true}) {
    StartScores(speculate);
    const auto prompt = Prompt(1, 0);
    const auto chain = Chain(prompt, 5);
    std::vector<float> last;
    ASSERT_TRUE(llm_->Prefill(prompt, last));
    rt::GenerateOptions options;
    options.max_tokens = 5;
    options.keep_logits = true;
    rt::Generation baseline;
    ASSERT_TRUE(llm_->Generate(last, options, baseline));
    EXPECT_EQ(baseline.tokens, chain);
    ASSERT_EQ(baseline.logits.size(), baseline.tokens.size());
    for (std::size_t i = 0; i < baseline.tokens.size(); ++i)
      EXPECT_EQ(ArgmaxOf(baseline.logits[i]), baseline.tokens[i]);
    ASSERT_TRUE(llm_->Clear());
    ASSERT_TRUE(llm_->Prefill(prompt, last));
    rt::LlmTestAccess::Stops(*llm_, {chain[2]});
    std::vector<std::int32_t> seen, visible;
    options.on_logits = [&](std::int32_t token, std::span<const float>) {
      seen.push_back(token);
      return true;
    };
    options.on_tokens = [&](std::span<const std::int32_t> fresh) {
      visible.insert(visible.end(), fresh.begin(), fresh.end());
      return true;
    };
    rt::Generation stopped;
    ASSERT_TRUE(llm_->Generate(last, options, stopped));
    EXPECT_EQ(stopped.tokens, (std::vector<std::int32_t>(chain.begin(), chain.begin() + 3)));
    EXPECT_EQ(seen, stopped.tokens);
    EXPECT_EQ(visible, (std::vector<std::int32_t>(chain.begin(), chain.begin() + 2)));
    EXPECT_EQ(stopped.logits.size(), 3U);
    EXPECT_TRUE(stopped.stopped);
    EXPECT_EQ(State(0), llm_->history());
    EXPECT_FALSE(llm_->runner().slots().owed(0));
    Stop();
  }
}

TEST_F(LlmScores, ResumablePlainAppliesCompletedRowsAndRetainsTheUnprocessedAnchor) {
  StartScores();
  auto& branch = BranchAt(0);
  EXPECT_EQ(&branch.model(), llm_.get());
  EXPECT_EQ(&branch.history(), &llm_->history());
  const auto prompt = Prompt(1, 0);
  const auto chain = Chain(prompt, 3);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(prompt, last));
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.keep_logits = true;
  std::vector<unsigned> calls;
  options.on_tokens = [&, count = 0U](std::span<const std::int32_t>) mutable {
    calls.push_back(++count);
    return true;
  };
  rt::Generation out, refused;
  auto session = branch.BeginGeneration(last, options, out);
  ASSERT_TRUE(session);
  EXPECT_EQ(out.tokens, (std::vector<std::int32_t>{chain[0]}));
  EXPECT_TRUE(llm_->HasRetainedState());
  EXPECT_FALSE(branch.BeginGeneration(last, options, refused));
  auto step = (*session)->PrepareStep();
  ASSERT_TRUE(step);
  EXPECT_THAT(step->all, ElementsAre(prompt[0], chain[0]));
  EXPECT_EQ(step->position, 1U);
  EXPECT_EQ(step->left, 2U);
  EXPECT_FALSE(step->speculative);
  EXPECT_TRUE(step->need_logits);
  std::vector<float> row;
  ASSERT_TRUE(rt::LlmTestAccess::ExecuteChunk(*llm_, branch, *step, row));
  const auto before = source_->executed();
  ASSERT_TRUE((*session)->ApplyPlain(row));
  EXPECT_EQ(source_->executed(), before);
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  EXPECT_TRUE(out.cancelled);
  EXPECT_EQ(out.tokens, (std::vector<std::int32_t>(chain.begin(), chain.begin() + 2)));
  EXPECT_THAT(branch.history(), ElementsAre(prompt[0], chain[0]));
  EXPECT_EQ(State(0), branch.history());
  EXPECT_EQ(out.logits.size(), 2U);
  rt::Generation next;
  auto again = branch.BeginGeneration(row, options, next);
  ASSERT_TRUE(again);
  ASSERT_TRUE((*session)->Finish());
  EXPECT_FALSE(branch.BeginGeneration(last, options, refused));
  EXPECT_THAT(calls, ElementsAre(1U, 2U, 3U));
  (*again)->Cancel();
  ASSERT_TRUE((*again)->Finish());
  branch.Forget();
  EXPECT_FALSE(llm_->HasRetainedState());
  ASSERT_TRUE(branch.Prefill(prompt, last));
  EXPECT_EQ(State(0), prompt);
}

TEST_F(LlmScores, ResumableSamplingUsesAbsolutePositionKeysForCompletedRows) {
  StartScores();
  auto& branch = BranchAt(0);
  const auto prompt = Prompt(3, 3);
  std::vector<float> ignored;
  ASSERT_TRUE(branch.Prefill(prompt, ignored));
  const std::vector<float> uniform(kWidth, 0);
  rt::GenerateOptions options;
  options.max_tokens = 2;
  options.stop = false;
  options.sampling =
      llmp::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
  options.seed = 731;
  std::vector<llmp::execution::SamplingCandidate> scratch;
  const auto first = llmp::execution::Sample(
      uniform, *options.sampling, {.seed = options.seed, .stream = 0, .position = 3}, scratch);
  const auto second = llmp::execution::Sample(
      uniform, *options.sampling, {.seed = options.seed, .stream = 0, .position = 4}, scratch);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  rt::Generation out;
  auto session = branch.BeginGeneration(uniform, options, out);
  ASSERT_TRUE(session);
  auto step = (*session)->PrepareStep();
  ASSERT_TRUE(step);
  ASSERT_TRUE(rt::LlmTestAccess::ExecuteChunk(*llm_, branch, *step, ignored));
  ASSERT_TRUE((*session)->ApplyPlain(uniform));
  ASSERT_TRUE((*session)->Finish());
  EXPECT_THAT(out.tokens, ElementsAre(*first, *second));
  EXPECT_EQ(State(0), branch.history());
}

TEST_F(LlmScores, ResumableEarlyAndCompletedFailuresPreserveOnlyProvenHistory) {
  StartScores();
  auto& branch = BranchAt(0);
  const auto prompt = Prompt(1, 0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(prompt, last));
  rt::GenerateOptions options;
  options.max_tokens = 2;
  options.stop = false;
  options.sampling = llmp::execution::SamplingParams{.temperature = 1, .top_k = 1};
  auto bad = last;
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  rt::Generation out;
  out.tokens = {6};
  EXPECT_FALSE(branch.BeginGeneration(bad, options, out));
  EXPECT_THAT(out.tokens, ElementsAre(6));
  EXPECT_EQ(State(0), branch.history());
  auto session = branch.BeginGeneration(last, options, out);
  ASSERT_TRUE(session);
  auto step = (*session)->PrepareStep();
  ASSERT_TRUE(step);
  ASSERT_TRUE(rt::LlmTestAccess::ExecuteChunk(*llm_, branch, *step, last));
  EXPECT_FALSE((*session)->ApplyPlain(bad));
  EXPECT_FALSE((*session)->Finish());
  EXPECT_EQ(State(0), branch.history());
  EXPECT_EQ(branch.history().size(), 2U);
  EXPECT_TRUE(branch.HasRetainedState());
  rt::Generation uncertain;
  auto again = branch.BeginGeneration(last, options, uncertain);
  ASSERT_TRUE(again);
  ASSERT_TRUE((*again)->PrepareStep());
  llm_->runner().slots().Fault();
  EXPECT_FALSE((*again)->FailStep("unproven external completion"));
  EXPECT_FALSE((*again)->Finish());
  EXPECT_TRUE(branch.history().empty());
  EXPECT_FALSE(branch.HasRetainedState());
  again->reset();
  session->reset();
  Recover();
}
TEST_F(LlmScores, NativeBranchesOwnTheirHistoriesAndRejectForeignSelection) {
  StartScores();
  rt::Llm foreign(*node_, entry_, Settings(false), roles_, 1, {});
  EXPECT_FALSE(llm_->branch(4));
  std::array<rt::Llm::Branch*, 4> branches{};
  for (std::size_t i = 0; i < branches.size(); ++i) {
    branches[i] = &BranchAt(i);
    EXPECT_EQ(&branches[i]->model(), llm_.get());
    EXPECT_EQ(*llm_->branch(i), branches[i]);
    std::vector<float> row;
    ASSERT_TRUE(branches[i]->Prefill(Prompt(2, static_cast<std::int32_t>(i)), row));
    EXPECT_EQ(State(i), branches[i]->history());
  }
  EXPECT_EQ(branches[0], &llm_->default_branch());
  Close();
  ASSERT_TRUE(llm_->SelectBranches(branches));
  const auto selected = llm_->runner().slots().active();
  const std::array<rt::Llm::Branch*, 2> duplicate{branches[0], branches[0]};
  const std::array<rt::Llm::Branch*, 1> other{&foreign.default_branch()}, null{nullptr};
  EXPECT_FALSE(llm_->SelectBranches(duplicate));
  EXPECT_FALSE(llm_->SelectBranches(other));
  EXPECT_FALSE(llm_->SelectBranches(null));
  EXPECT_EQ(llm_->runner().slots().active(), selected);
  Open({branches[0], branches[1], branches[2], branches[3]});
  ASSERT_TRUE(branches[1]->Clear());
  EXPECT_TRUE(branches[1]->history().empty());
  EXPECT_EQ(llm_->runner().slots().positions(1), 0U);
  for (std::size_t i : {0U, 2U, 3U}) EXPECT_EQ(State(i), branches[i]->history());
  branches[0]->Forget();
  EXPECT_TRUE(llm_->HasRetainedState());
  branches[2]->Forget();
  branches[3]->Forget();
  EXPECT_FALSE(llm_->HasRetainedState());
}

TEST_F(LlmScores, FourResumableBranchesInterleaveSamplingWithoutSharingKeysOrGuards) {
  StartScores();
  source_->options().transform_logits = [](pl::Phase, std::int32_t, std::int32_t,
                                           std::span<float> row) { std::ranges::fill(row, 0.0F); };
  std::array<rt::GenerateOptions, 4> options;
  std::array<rt::Generation, 4> results;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 4> sessions;
  std::array<std::vector<std::int32_t>, 4> expected;
  const std::vector<float> row(kWidth, 0);
  for (std::size_t slot = 0; slot < 4; ++slot) {
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(slot).Prefill(Prompt(slot + 1, static_cast<std::int32_t>(slot)), last));
    auto& option = options[slot];
    option.max_tokens = 3;
    option.stop = false;
    option.sampling =
        llmp::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
    option.seed = 731 + slot;
    std::vector<llmp::execution::SamplingCandidate> scratch;
    for (unsigned i = 0; i < 3; ++i) {
      auto token = llmp::execution::Sample(
          row, *option.sampling, {.seed = option.seed, .stream = 0, .position = slot + 1 + i},
          scratch);
      ASSERT_TRUE(token);
      expected[slot].push_back(*token);
    }
    auto opened = BranchAt(slot).BeginGeneration(last, option, results[slot]);
    ASSERT_TRUE(opened);
    sessions[slot] = std::move(*opened);
    rt::Generation refused;
    EXPECT_FALSE(BranchAt(slot).BeginGeneration(last, option, refused));
  }
  for (auto slot : {3U, 0U, 2U, 1U}) ASSERT_TRUE(sessions[slot]->RunScalarStep());
  sessions[1]->Cancel();
  ASSERT_TRUE(sessions[1]->Finish());
  expected[1].resize(2);
  for (auto slot : {2U, 3U, 0U}) {
    ASSERT_TRUE(sessions[slot]->RunScalarStep());
    ASSERT_TRUE(sessions[slot]->Finish());
  }
  for (unsigned slot = 0; slot < 4; ++slot) {
    EXPECT_EQ(results[slot].tokens, expected[slot]);
    EXPECT_EQ(results[slot].cancelled, slot == 1);
    EXPECT_EQ(State(slot), BranchAt(slot).history());
  }
}

TEST_F(LlmScores, BranchSnapshotsRestoreTheirOwnCursorAndAdaptivePolicy) {
  StartScores(true);
  const auto first = Prompt(2, 1), second = Prompt(3, 2);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(1).Prefill(first, last));
  ASSERT_TRUE(BranchAt(2).Prefill(second, last));
  for (unsigned i = 0; i < 4; ++i) {
    ObserveDepth(1, 3, 1);
    ObserveDepth(2, 3, 4);
  }
  const auto saved_a = Depth(1), saved_b = Depth(2);
  ASSERT_NE(saved_a, saved_b);
  std::vector<std::byte> a(BranchAt(1).state_snapshot_bytes());
  std::vector<std::byte> b(BranchAt(2).state_snapshot_bytes());
  ASSERT_TRUE(BranchAt(1).SaveState(a.data()));
  ASSERT_TRUE(BranchAt(2).SaveState(b.data()));
  ASSERT_TRUE(BranchAt(1).Prefill(std::array<std::int32_t, 1>{5}, last));
  ASSERT_TRUE(BranchAt(2).Prefill(std::array<std::int32_t, 1>{5}, last));
  ObserveDepth(1, 2, 3);
  const auto changed_b = NativeImage(2);
  ASSERT_TRUE(BranchAt(1).RestoreState(a.data()));
  EXPECT_EQ(BranchAt(1).history(), first);
  EXPECT_EQ(State(1), first);
  EXPECT_EQ(llm_->runner().slots().positions(1), first.size());
  EXPECT_EQ(Depth(1), saved_a);
  EXPECT_EQ(NativeImage(2), changed_b);
  EXPECT_EQ(llm_->runner().slots().positions(2), second.size() + 1);
  EXPECT_EQ(Depth(2), saved_b);
}

TEST_F(LlmScores, AGenerationWaveCapacityRefusalLeavesSparseActiveBranchesUntouched) {
  StartScores();
  ASSERT_EQ(llm_->generation_wave_capacity(), 4U);
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  std::array<rt::Generation, 3> results;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 3> sessions;
  const std::array<unsigned, 3> slots{0, 3, 1};
  for (std::size_t i = 0; i < slots.size(); ++i) {
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(slots[i]).Prefill(Prompt(1, static_cast<std::int32_t>(i)), last));
    auto begun = BranchAt(slots[i]).BeginGeneration(last, options, results[i]);
    ASSERT_TRUE(begun);
    sessions[i] = std::move(*begun);
  }
  const auto before = source_->executed();
  // The model has four real request slots. Reject the oversized submission
  // before validating owners or preparing any of its existing sessions.
  const std::array<rt::Llm::GenerationSession*, 5> too_many{sessions[0].get(), sessions[1].get(),
                                                            sessions[2].get(), sessions[0].get(),
                                                            sessions[1].get()};
  EXPECT_FALSE(llm_->RunGenerationWave(too_many));
  EXPECT_EQ(source_->executed(), before);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(sessions[i]->done());
    EXPECT_EQ(results[i].tokens.size(), 1U);
    EXPECT_EQ(results[i].steps, 0U);
  }
  const std::array<rt::Llm::GenerationSession*, 2> both{sessions[1].get(), sessions[0].get()};
  ASSERT_TRUE(llm_->RunGenerationWave(both));
  ASSERT_TRUE(llm_->RunGenerationWave(both));
  for (std::size_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(sessions[i]->Finish());
    EXPECT_EQ(results[i].tokens, Chain(Prompt(1, static_cast<std::int32_t>(i)), 3));
    EXPECT_EQ(State(slots[i]), BranchAt(slots[i]).history());
  }
  sessions[2]->Cancel();
  ASSERT_TRUE(sessions[2]->Finish());
  EXPECT_EQ(BranchAt(1).history(), Prompt(1, 2));
}

TEST_F(LlmScores, EightBranchesRunInOneGenerationWave) {
  StartScores(false, 8);
  EXPECT_FALSE(llm_->branch(8));
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  std::array<rt::Generation, 8> results;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 8> sessions;
  std::array<rt::Llm::GenerationSession*, 8> wave{};
  for (std::size_t slot = 0; slot < 8; ++slot) {
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(slot).Prefill(Prompt(1, static_cast<std::int32_t>(slot)), last));
    auto begun = BranchAt(slot).BeginGeneration(last, options, results[slot]);
    ASSERT_TRUE(begun);
    sessions[slot] = std::move(*begun);
    wave[slot] = sessions[slot].get();
  }
  const auto before = source_->executed();
  ASSERT_TRUE(llm_->RunGenerationWave(wave));
  ASSERT_TRUE(llm_->RunGenerationWave(wave));
  EXPECT_EQ(source_->executed() - before, 2U);
  for (std::size_t slot = 0; slot < 8; ++slot) {
    ASSERT_TRUE(sessions[slot]->done());
    ASSERT_TRUE(sessions[slot]->Finish());
    EXPECT_EQ(results[slot].tokens, Chain(Prompt(1, static_cast<std::int32_t>(slot)), 3));
    EXPECT_EQ(State(slot), BranchAt(slot).history());
  }
}

TEST_F(LlmScores, AGenerationWaveSamplingErrorPublishesTheCompletedPeer) {
  StartScores();
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  options.sampling = llmp::execution::SamplingParams{.temperature = 1, .top_k = 1};
  std::array<rt::Generation, 2> results;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> sessions;
  for (unsigned i = 0; i < 2; ++i) {
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(i).Prefill(Prompt(1, static_cast<std::int32_t>(i)), last));
    auto opened = BranchAt(i).BeginGeneration(last, options, results[i]);
    ASSERT_TRUE(opened);
    sessions[i] = std::move(*opened);
  }
  source_->options().transform_logits = [](pl::Phase, std::int32_t slot, std::int32_t,
                                           std::span<float> row) {
    if (slot == 0) row[0] = std::numeric_limits<float>::quiet_NaN();
  };
  const std::array<rt::Llm::GenerationSession*, 2> both{sessions[0].get(), sessions[1].get()};
  ASSERT_TRUE(llm_->RunGenerationWave(both));
  EXPECT_TRUE(sessions[0]->done());
  EXPECT_FALSE(sessions[1]->done());
  EXPECT_FALSE(sessions[0]->Finish());
  EXPECT_EQ(BranchAt(0).history().size(), 2U);
  EXPECT_EQ(State(0), BranchAt(0).history());
  const std::array<rt::Llm::GenerationSession*, 1> peer{sessions[1].get()};
  ASSERT_TRUE(llm_->RunGenerationWave(peer));
  ASSERT_TRUE(sessions[1]->Finish());
  EXPECT_EQ(results[1].tokens, Chain(Prompt(1, 1), 3));
  EXPECT_EQ(State(1), BranchAt(1).history());
}

TEST_F(LlmScores, ADiscardedJudgementKeepsOnlyItsOwnPrefixAndCompletedCursor) {
  for (const bool wave : {false, true}) {
    StartScores(true);
    rt::GenerateOptions options;
    options.max_tokens = 5;
    options.stop = false;
    options.sampling = llmp::execution::SamplingParams{.temperature = 1, .top_k = 1};
    std::array<rt::Generation, 2> results;
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> sessions;
    for (unsigned i = 0; i < 2; ++i) {
      std::vector<float> last;
      ASSERT_TRUE(BranchAt(i).Prefill(Prompt(1, static_cast<std::int32_t>(i)), last));
      auto begun = BranchAt(i).BeginGeneration(last, options, results[i]);
      ASSERT_TRUE(begun);
      sessions[i] = std::move(*begun);
    }
    const auto target_before = State(0), injection_before = State(0, 1);
    source_->options().transform_logits = [](pl::Phase phase, std::int32_t slot, std::int32_t,
                                             std::span<float> row) {
      if (phase == pl::Phase::kVerify && slot == 0)
        row[0] = std::numeric_limits<float>::quiet_NaN();
    };
    if (wave) {
      const std::array<rt::Llm::GenerationSession*, 2> both{sessions[0].get(), sessions[1].get()};
      ASSERT_TRUE(llm_->RunGenerationWave(both));
    } else {
      EXPECT_FALSE(sessions[0]->RunScalarStep());
      ASSERT_TRUE(sessions[1]->RunScalarStep());
    }
    EXPECT_FALSE(sessions[0]->Finish());
    EXPECT_EQ(BranchAt(0).history(), Prompt(1, 0));
    // Truncated KV backing may still contain rejected rows beyond the cursor.
    // The committed target prefix and target-written drafter tap must survive.
    EXPECT_EQ(State(0), target_before);
    EXPECT_EQ(State(0, 1), injection_before);
    EXPECT_EQ(llm_->runner().slots().positions(0), 1U);
    while (!sessions[1]->done()) ASSERT_TRUE(sessions[1]->RunScalarStep());
    ASSERT_TRUE(sessions[1]->Finish());
    EXPECT_EQ(results[1].tokens, Chain(Prompt(1, 1), 5));
    EXPECT_EQ(State(1), BranchAt(1).history());
    EXPECT_FALSE(llm_->runner().slots().owed(1));
    sessions = {};
    Stop();
  }
}

TEST_F(LlmScores, AdmissionsReclaimSparesTheChosenBranch) {
  StartScores();
  auto& model = *llm_;
  std::vector<float> last;
  for (const std::size_t slot : {0U, 1U, 3U}) {
    auto branch = model.branch(slot);
    ASSERT_TRUE(branch.has_value());
    const std::array<std::int32_t, 2> prompt = {static_cast<std::int32_t>(slot), 1};
    ASSERT_TRUE((*branch)->Prefill(prompt, last).has_value());
  }
  Close();
  const rt::IdleStateRates rates{.spill_rate = 11.0e9, .restore_rate = 14.5e9};
  const auto ids = [](const std::vector<llmp::memory::ReclaimCandidate>& c) {
    std::vector<std::uint64_t> out;
    for (const auto& x : c) {
      EXPECT_EQ(x.kind, llmp::memory::ReclaimKind::kIdleState);
      EXPECT_GT(x.bytes, 0U);
      out.push_back(x.id);
    }
    return out;
  };
  std::vector<llmp::memory::ReclaimCandidate> all;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, rates, all);
  EXPECT_THAT(ids(all), ElementsAre(0U, 1U, 3U));  // slot 2 holds nothing
  auto chosen = model.branch(1);
  ASSERT_TRUE(chosen.has_value());
  std::vector<llmp::memory::ReclaimCandidate> spared;
  rt::AddIdleStateCandidates(model, 0, true, *chosen, rates, spared);
  EXPECT_THAT(ids(spared), ElementsAre(0U, 3U));
  // A branch the request open on the stream leases is no candidate either.
  auto peer = model.branch(0);
  ASSERT_TRUE(peer.has_value());
  const std::array<rt::Llm::Branch*, 1> selected = {*peer};
  ASSERT_TRUE(model.SelectBranches(selected).has_value());
  Open({*peer});
  std::vector<llmp::memory::ReclaimCandidate> leased;
  rt::AddIdleStateCandidates(model, 0, true, *chosen, rates, leased);
  EXPECT_THAT(ids(leased), ElementsAre(3U));
  Close();
  // A conversation a continuation holds is spilled, never dropped (its
  // drop is refused): with a spill budget of 0, or one smaller than its
  // state, it is no candidate; one nothing holds is, at its recomputation.
  auto held = model.branch(3);
  ASSERT_TRUE(held.has_value());
  (*held)->HoldContinuation();
  std::vector<llmp::memory::ReclaimCandidate> spilled;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, rates, spilled);
  EXPECT_THAT(ids(spilled), ElementsAre(0U, 1U, 3U));
  for (const std::uint64_t budget : {std::uint64_t{0}, model.ResidentStateBytes(**held) - 1}) {
    rt::IdleStateRates small = rates;
    small.spill_budget = budget;
    std::vector<llmp::memory::ReclaimCandidate> dropped;
    rt::AddIdleStateCandidates(model, 0, true, nullptr, small, dropped);
    EXPECT_THAT(ids(dropped), ElementsAre(0U, 1U)) << budget;
  }
  (*held)->ReleaseContinuation();
  rt::IdleStateRates none = rates;
  none.spill_budget = 0;
  std::vector<llmp::memory::ReclaimCandidate> unheld;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, none, unheld);
  EXPECT_THAT(ids(unheld), ElementsAre(0U, 1U, 3U));
  // Priced as what its take does: a spill the budget cannot make room for
  // even past every deletable spill is a drop (a held one no candidate), and
  // one it can is charged the recomputation of the spills it deletes.
  rt::IdleStateRates full = rates;
  full.spill_room = [](const rt::Llm::Branch&, std::uint64_t) { return std::optional<double>(); };
  (*held)->HoldContinuation();
  std::vector<llmp::memory::ReclaimCandidate> no_room;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, full, no_room);
  EXPECT_THAT(ids(no_room), ElementsAre(0U, 1U));
  (*held)->ReleaseContinuation();
  rt::IdleStateRates deleting = rates;
  deleting.spill_room = [](const rt::Llm::Branch&, std::uint64_t) {
    return std::optional<double>(5.0);
  };
  std::vector<llmp::memory::ReclaimCandidate> charged;
  rt::AddIdleStateCandidates(model, 0, true, nullptr, deleting, charged);
  ASSERT_EQ(charged.size(), spilled.size());
  for (std::size_t i = 0; i < charged.size(); ++i) {
    EXPECT_DOUBLE_EQ(charged[i].restore_seconds, spilled[i].restore_seconds + 5.0);
  }
}

TEST_F(TokenHistory, PromptAdmissionAndRetirementChargeActualCapacity) {
  StartScores();
  auto& memory = Fund(1024);
  auto& model = *llm_;
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

TEST_F(TokenHistory, RefusedAdmissionPreservesAliasedCompletedPrefix) {
  StartScores();
  auto& memory = Fund(64);
  auto& model = *llm_;
  model.SetTokenMemory(memory);
  std::vector<float> last;
  const std::vector<std::int32_t> prompt(16, 2);
  ASSERT_TRUE(model.Prefill(prompt, last));
  const auto before = model.history();
  auto refused = model.default_branch().BeginPrompt(model.history(), 0, false, true);
  EXPECT_FALSE(refused);
  EXPECT_EQ(model.history(), before);
  EXPECT_EQ(State(0), before);
  EXPECT_EQ(memory.used(), model.history().capacity() * sizeof(std::int32_t));
  ASSERT_TRUE(model.Clear());
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(TokenHistory, ConcurrentSessionsCannotSpendOneAnothersCapacity) {
  StartScores();
  auto& memory = Fund(128);
  auto& model = *llm_;
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
  EXPECT_EQ(source_->executed(), 0U);
  (*peer)->Cancel();
  ASSERT_TRUE((*peer)->Finish());
  ASSERT_TRUE((*active)->Advance());
  (*active)->Cancel();
  ASSERT_TRUE((*active)->Finish());
  EXPECT_EQ(memory.used(), (*first)->history().capacity() * sizeof(std::int32_t));
  Close();
  ASSERT_TRUE((*first)->ReleaseIdleState());
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(TokenHistory, GenerationRefusalStartsNoNativeWorkAndKeepsPrefix) {
  StartScores();
  auto& memory = Fund(36);
  auto& model = *llm_;
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
  const auto chunks = source_->executed();
  auto refused = (*session)->RunScalarStep(true);
  EXPECT_FALSE(refused);
  EXPECT_TRUE((*session)->refused());
  EXPECT_EQ(source_->executed(), chunks);
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(model.history(), std::vector<std::int32_t>(prompt.begin(), prompt.end()));
  EXPECT_EQ(State(0), model.history());
  EXPECT_EQ(memory.used(), model.history().capacity() * sizeof(std::int32_t));
  model.Forget();
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(TokenHistory, SnapshotsOwnIndependentCapacityUntilInvalidation) {
  StartScores();
  auto& memory = Fund(512);
  auto& model = *llm_;
  model.SetTokenMemory(memory);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch);
  const std::vector<std::int32_t> prompt(8, 2);
  std::vector<float> last;
  ASSERT_TRUE((*branch)->Prefill(prompt, last));
  std::vector<std::byte> saved((*branch)->state_snapshot_bytes());
  ASSERT_TRUE((*branch)->SaveState(saved.data()));
  EXPECT_EQ(memory.used(), 2 * prompt.size() * sizeof(std::int32_t));
  EXPECT_FALSE(model.HistoryReclaimable(**branch));
  ASSERT_TRUE((*branch)->Clear());
  EXPECT_EQ((*branch)->history().capacity(), 0U);
  EXPECT_EQ(memory.used(), prompt.size() * sizeof(std::int32_t));
  ASSERT_TRUE((*branch)->RestoreState(saved.data()));
  EXPECT_EQ((*branch)->history(), prompt);
  EXPECT_EQ(State(1), prompt);
  EXPECT_EQ(memory.used(), 2 * prompt.size() * sizeof(std::int32_t));
  (*branch)->InvalidateStateSnapshot();
  Close();
  EXPECT_TRUE(model.HistoryReclaimable(**branch));
  EXPECT_EQ(memory.used(), prompt.size() * sizeof(std::int32_t));
  ASSERT_TRUE((*branch)->ReleaseIdleState());
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(TokenHistory, SpilledContinuationsRetainTokensUntilTheirOwnerRetires) {
  StartScores();
  auto& memory = Fund(256);
  auto& model = *llm_;
  model.SetTokenMemory(memory);
  auto branch = model.branch(1);
  ASSERT_TRUE(branch);
  const std::vector<std::int32_t> prompt(8, 2);
  std::vector<float> last;
  ASSERT_TRUE((*branch)->Prefill(prompt, last));
  (*branch)->HoldContinuation();
  EXPECT_FALSE(model.HistoryReclaimable(**branch));
  Close();
  ASSERT_TRUE(model.SpillIdle(**branch));
  EXPECT_EQ(memory.used(), (*branch)->history().capacity() * sizeof(std::int32_t));
  EXPECT_FALSE((*branch)->ReleaseIdleState());
  Open({*branch});
  auto resumed = (*branch)->BeginPrompt((*branch)->history(), 0, false, true);
  ASSERT_TRUE(resumed);
  ASSERT_TRUE((*resumed)->Advance());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(State(1), prompt);
  (*branch)->ReleaseContinuation();
  Close();
  ASSERT_TRUE((*branch)->ReleaseIdleState());
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(TokenHistory, NearBudgetGrowthRefusesInsteadOfCopyingEachGeneratedToken) {
  StartScores(false, 4, 512, 64);
  memory_ = std::make_unique<rt::RequestMemory>(0, 4096, true);
  auto& memory = *memory_;
  auto& model = *llm_;
  unsigned growth_requests = 0;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    if (incoming != 0) {
      ++growth_requests;
    }
    return memory.used() + incoming <= 1024;
  });
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
  EXPECT_EQ(source_->executed(), 1U);
  (*session)->Cancel();
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(model.history(), prompt);
}

TEST_F(LlmScores, PerRequestChatStopsDoNotChangeLiteralStops) {
  StartScores();
  auto& model = *llm_;
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
  EXPECT_THAT(ordinary.tokens, ElementsAre(1, 11, 24));
}

TEST(SwapContinuation, RowsAreTheModelsLogitWidthNotItsTokenizers) {
  const std::size_t width = kWidth;
  const std::vector<float> last = Row(3);
  rt::Generation reference;
  reference.logits = {Row(1), Row(2)};
  rt::Generation continued = reference;
  EXPECT_TRUE(rt::CompleteContinuationRows(last, reference, continued, 2, width));
  // A narrower tokenizer's size is not the rows' width.
  EXPECT_FALSE(rt::CompleteContinuationRows(last, reference, continued, 2, width - 2));
  EXPECT_FALSE(rt::CompleteContinuationRows(last, reference, continued, 2, 0));
  EXPECT_FALSE(rt::CompleteContinuationRows(last, reference, continued, 3, width));
  EXPECT_FALSE(rt::CompleteContinuationRows({}, reference, continued, 2, width));
  continued.logits[1].pop_back();
  EXPECT_FALSE(rt::CompleteContinuationRows(last, reference, continued, 2, width));
  continued.logits[1] = Row(2);
  continued.logits[1][5] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(rt::CompleteContinuationRows(last, reference, continued, 2, width));
  continued.logits[1][5] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(rt::CompleteContinuationRows(last, reference, continued, 2, width));
}

TEST_F(LlmScores, ResumableScoringRetriesCapacityBeforeAnyRowIsReported) {
  cell_bytes_ = kExtent / 256;
  StartScores(false, 4, 1024, 1);
  unsigned reported = 0;
  auto session =
      BranchAt(0).BeginScoringPrompt(Prompt(3, 3), [&](std::int32_t, std::span<const float>) {
        ++reported;
        return true;
      });
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->Advance());
  Room(0);
  const auto before = source_->executed();
  EXPECT_FALSE((*session)->Advance({}, true));
  EXPECT_TRUE((*session)->refused());
  EXPECT_EQ(reported, 0U);
  EXPECT_TRUE(llm_->history().empty());
  EXPECT_EQ(source_->executed(), before);
  Room(1);
  while (!(*session)->done()) ASSERT_TRUE((*session)->Advance());
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(reported, 2U);
  EXPECT_EQ(State(0), Prompt(3, 3));
}

TEST_F(LlmScores, ABranchPreparationRefusalKeepsItsPrefixAndAnAlreadyPreparedPeer) {
  for (const bool wave : {false, true}) {
    cell_bytes_ = kExtent / 256;
    StartScores(false, 4, 1024, 256);
    std::vector<float> a, b;
    ASSERT_TRUE(BranchAt(0).Prefill(Prompt(256, 3), a));
    ASSERT_TRUE(BranchAt(1).Prefill(Prompt(1, 4), b));
    rt::GenerateOptions options;
    options.max_tokens = 3;
    options.stop = false;
    rt::Generation out_a, out_b;
    auto sa = BranchAt(0).BeginGeneration(a, options, out_a);
    auto sb = BranchAt(1).BeginGeneration(b, options, out_b);
    ASSERT_TRUE(sa);
    ASSERT_TRUE(sb);
    Room(0);
    if (wave) {
      const std::array<rt::Llm::GenerationSession*, 2> both{sb->get(), sa->get()};
      ASSERT_TRUE(llm_->RunGenerationWave(both));
    } else {
      EXPECT_FALSE((*sa)->RunScalarStep());
      ASSERT_TRUE((*sb)->RunScalarStep());
    }
    EXPECT_TRUE((*sa)->done());
    EXPECT_FALSE((*sb)->done());
    EXPECT_EQ(out_a.tokens.size(), 1U);
    EXPECT_EQ(out_b.tokens.size(), 2U);
    EXPECT_FALSE((*sa)->Finish());
    EXPECT_EQ(BranchAt(0).history(), Prompt(256, 3));
    EXPECT_EQ(State(0), BranchAt(0).history());
    ASSERT_TRUE((*sb)->RunScalarStep());
    ASSERT_TRUE((*sb)->Finish());
    EXPECT_EQ(out_b.tokens, Chain(Prompt(1, 4), 3));
    EXPECT_EQ(State(1), BranchAt(1).history());
    sa->reset();
    sb->reset();
    Stop();
  }
}

TEST_F(LlmScores, ScalarUnitPublicationKeepsEarlierSuccessOnLaterCleanDispatchRefusal) {
  StartScores();
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  options.seed = 73;
  options.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 4};
  const auto prompt = Prompt(1, 0);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(3).Prefill(prompt, last));
  rt::Generation expected;
  ASSERT_TRUE(BranchAt(3).Generate(last, options, expected));
  std::vector<std::int32_t> visible;
  auto streamed = options;
  streamed.on_tokens = [&](std::span<const std::int32_t> ids) {
    visible.insert(visible.end(), ids.begin(), ids.end());
    return true;
  };
  rt::Generation out_a, out_b;
  ASSERT_TRUE(BranchAt(0).Prefill(prompt, last));
  auto sa = BranchAt(0).BeginGeneration(last, options, out_a);
  ASSERT_TRUE(sa);
  ASSERT_TRUE(BranchAt(1).Prefill(prompt, last));
  auto sb = BranchAt(1).BeginGeneration(last, streamed, out_b);
  ASSERT_TRUE(sb);
  ASSERT_TRUE((*sa)->RunScalarStep());
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.phase == pl::Phase::kDecode && shape.segments.front().slot == 1;
  };
  EXPECT_FALSE((*sb)->RunScalarStep());
  EXPECT_EQ(out_a.tokens.size(), 2U);
  EXPECT_EQ(out_b.tokens.size(), 1U);
  EXPECT_FALSE((*sb)->Finish());
  sb->reset();
  EXPECT_EQ(BranchAt(1).history(), prompt);
  EXPECT_EQ(State(1), prompt);
  source_->options().refuse_build = {};
  auto resumed = BranchAt(1).ResumeGeneration({}, streamed, out_b);
  ASSERT_TRUE(resumed);
  while (!(*sa)->done()) ASSERT_TRUE((*sa)->RunScalarStep());
  while (!(*resumed)->done()) ASSERT_TRUE((*resumed)->RunScalarStep());
  ASSERT_TRUE((*sa)->Finish());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(out_a.tokens, expected.tokens);
  EXPECT_EQ(out_b.tokens, expected.tokens);
  EXPECT_EQ(visible, expected.tokens);
  EXPECT_EQ(BranchAt(0).history(), BranchAt(1).history());
}

TEST_F(TokenHistory, ARepeatedLongRequestLibraryReclaimsIdleCapacity) {
  compute_streams_ = 25;
  StartScores(false, 4, 64, 8);
  Close();
  memory_ = std::make_unique<rt::RequestMemory>(0, 4096, true);
  auto& memory = *memory_;
  std::vector<std::unique_ptr<rt::Llm>> models;
  unsigned reclaimed = 0;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    if (memory.used() + incoming > 320) {
      for (const auto& model : models) {
        if (model->HistoryReclaimable(model->default_branch()) && !model->history().empty()) {
          const auto cleared = model->Clear();
          EXPECT_TRUE(cleared);
          if (!cleared) return false;
          ++reclaimed;
          if (memory.used() + incoming <= 320) break;
        }
      }
    }
    return memory.used() + incoming <= 320;
  });
  for (unsigned index = 0; index < 24; ++index) {
    auto entry = entry_;
    entry.name = std::format("library{}", index);
    auto model = std::make_unique<rt::Llm>(*node_, entry, Settings(false), roles_,
                                           static_cast<int>(index + 1), Build(false));
    ASSERT_TRUE(model->Setup());
    ASSERT_TRUE(model->Register());
    ASSERT_TRUE(model->Bind());
    model->SetTokenMemory(memory);
    models.push_back(std::move(model));
    auto& current = *models.back();
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      const std::vector<std::int32_t> prompt(24, static_cast<std::int32_t>(repeat));
      ASSERT_TRUE(current.PrepareDefaultRequest());
      ASSERT_TRUE(
          node_->BeginRequest(current.paged().stream(), current.request_closure(), "library test"));
      auto session = current.default_branch().BeginPrompt(prompt, 0, true);
      ASSERT_TRUE(session);
      while (!(*session)->done()) ASSERT_TRUE((*session)->Advance());
      ASSERT_TRUE((*session)->Finish());
      ASSERT_TRUE(node_->EndRequest(current.paged().stream()));
      EXPECT_EQ(current.history(), prompt);
      EXPECT_EQ(current.runner().slots().positions(0), prompt.size());
      std::uint64_t actual = 0;
      for (const auto& item : models) actual += item->history().capacity() * sizeof(std::int32_t);
      EXPECT_EQ(memory.used(), actual);
      EXPECT_LE(memory.used(), 320U);
    }
  }
  EXPECT_GT(reclaimed, 0U);
  // Wrapper destruction releases host history; the node retains each runner
  // until its authenticated teardown drains and releases native resources.
  models.clear();
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(LlmScores, NonFinalPromptHeadsAreOptionalButScoringKeepsEveryRequiredRow) {
  StartScores(false, 4, 64, 8);
  const auto prompt = Prompt(19, 2);
  std::vector<float> serial;
  ASSERT_TRUE(BranchAt(0).Prefill(prompt, serial));
  EXPECT_EQ(ArgmaxOf(serial), ts::FakeLogit(prompt.back(), 18, kWidth));
  bool state_only = false, final_head = false;
  for (const auto& built : source_->built()) {
    if (built.measuring || built.shape.phase != pl::Phase::kPrefill) continue;
    for (const auto& segment : built.shape.segments) {
      if (segment.slot != 0) continue;
      state_only |= segment.rows == 8 && !segment.want_head;
      final_head |= segment.rows == 3 && segment.want_head;
    }
  }
  EXPECT_TRUE(state_only);
  EXPECT_TRUE(final_head);
  auto incremental = BranchAt(1).BeginPrompt(prompt);
  ASSERT_TRUE(incremental);
  ASSERT_TRUE((*incremental)->Advance());
  auto hint = rt::LlmTestAccess::Hint(**incremental, 8);
  EXPECT_EQ(hint.rows, 8U);
  EXPECT_FALSE(hint.want_head);
  ASSERT_TRUE((*incremental)->Advance());
  EXPECT_TRUE((*incremental)->last().empty());
  (*incremental)->Cancel();
  ASSERT_TRUE((*incremental)->Finish());
  EXPECT_EQ(BranchAt(1).history().size(), 8U);
  incremental = BranchAt(1).BeginPrompt(prompt);
  ASSERT_TRUE(incremental);
  while (!(*incremental)->done()) ASSERT_TRUE((*incremental)->Advance());
  ASSERT_TRUE((*incremental)->Finish());
  EXPECT_EQ((*incremental)->last(), serial);
  EXPECT_EQ(State(1), prompt);
  unsigned scores = 0;
  auto scoring =
      BranchAt(2).BeginScoringPrompt(prompt, [&](std::int32_t token, std::span<const float> row) {
        EXPECT_FALSE(row.empty());
        EXPECT_EQ(token, prompt[scores + 1]);
        EXPECT_EQ(ArgmaxOf(row),
                  ts::FakeLogit(prompt[scores], static_cast<std::int32_t>(scores), kWidth));
        ++scores;
        return true;
      });
  ASSERT_TRUE(scoring);
  while (!(*scoring)->done()) ASSERT_TRUE((*scoring)->Advance());
  ASSERT_TRUE((*scoring)->Finish());
  EXPECT_EQ(scores, prompt.size() - 1);
  EXPECT_EQ((*scoring)->last(), serial);
}

TEST_F(LlmScores, ResumableVerifySettlesTheFullAcceptedPrefixAfterVisibleStop) {
  StartScores(true);
  const auto prompt = Prompt(1, 0);
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.keep_logits = true;
  options.stop = false;
  rt::Generation baseline;
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(prompt, last));
  auto reference = BranchAt(0).BeginGeneration(last, options, baseline);
  ASSERT_TRUE(reference);
  ASSERT_TRUE((*reference)->RunScalarStep());
  (*reference)->Cancel();
  ASSERT_TRUE((*reference)->Finish());
  const auto retired = BranchAt(0).history();
  ASSERT_GT(retired.size(), 2U);  // a real accepted verify includes unseen suffix rows
  std::vector<std::int32_t> rows, visible;
  options.on_logits = [&](std::int32_t token, std::span<const float> row) {
    rows.push_back(token);
    EXPECT_EQ(ArgmaxOf(row), token);
    return true;
  };
  options.on_tokens = [&](std::span<const std::int32_t> fresh) {
    visible.insert(visible.end(), fresh.begin(), fresh.end());
    return visible.size() < 2;
  };
  ASSERT_TRUE(BranchAt(1).Prefill(prompt, last));
  rt::Generation out;
  auto session = BranchAt(1).BeginGeneration(last, options, out);
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->RunScalarStep());
  EXPECT_TRUE((*session)->done());
  ASSERT_TRUE((*session)->Finish());
  EXPECT_EQ(out.tokens.size(), 2U);
  EXPECT_EQ(rows, out.tokens);
  EXPECT_EQ(visible, out.tokens);
  EXPECT_GT(llm_->counts().speculative_units, 0U);
  EXPECT_EQ(out.accepted, baseline.accepted);
  EXPECT_EQ(out.drafted, baseline.drafted);
  EXPECT_EQ(BranchAt(1).history(), retired);
  EXPECT_EQ(State(1), retired);
  EXPECT_FALSE(llm_->runner().slots().owed(1));
}

}  // namespace

class LlmTools : public LlmScores {
 protected:
  void QwenChat() {
    score_width_ = 265;
    StartScores(false, 1, 4096, 32);
    const auto text = ts::ReadFile(ts::DataDir() + "/chat/templates/qwen3.8.jinja");
    ASSERT_TRUE(text);
    auto spec = ByteTokenizerSpec();
    for (const auto* marker : {"<think>", "</think>", "<tool_call>", "</tool_call>",
                               "<tool_response>", "</tool_response>"}) {
      spec.tokens.emplace_back(marker);
      spec.kinds.push_back(llmp::tokenizer::TokenKind::kUserDefined);
    }
    auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(spec));
    ASSERT_TRUE(tokenizer);
    ASSERT_TRUE(rt::LlmTestAccess::UseChat(*llm_, std::move(*tokenizer), *text));
  }
  std::shared_ptr<llmp::chat::ToolSampling> Constraint(bool speculative = false) {
    score_width_ = 259;
    StartScores(speculative, 1, 128, 32);
    EXPECT_TRUE(rt::LlmTestAccess::UseChat(*llm_, ByteTokenizer(), "{{ messages[0].content }}"));
    const std::array<llmp::chat::ToolDefinition, 1> definitions{
        {{.name = "lookup", .description = {}}}};
    auto request = llmp::chat::ToolRequest::Create(
        definitions, {llmp::chat::ToolChoiceKind::kNamed, "lookup"}, true, {});
    EXPECT_TRUE(request);
    return std::make_shared<llmp::chat::ToolSampling>(
        llm_->tokenizer(), llmp::chat::QwenToolCalls(), std::move(*request),
        llmp::base::json::Limits{}, false, std::span<const std::int32_t>{});
  }
};

// A tools request keeps the device's greedy choice wherever its constraint
// restricts nothing (B3): every unconstrained auto step outside a call; a
// named choice forces its opener on the host.
class LlmToolDevice : public LlmTools {
 protected:
  void Run(bool named) {
    score_width_ = 259;
    StartScores(false, 1, 128, 32);
    ASSERT_TRUE(rt::LlmTestAccess::UseChat(*llm_, ByteTokenizer(), "{{ messages[0].content }}"));
    source_->options().transform_logits = [](pl::Phase, std::int32_t, std::int32_t,
                                             std::span<float> row) {
      std::ranges::fill(row, 0.0F);
      row['a'] = 9;
    };
    source_->options().transform_native_token = [](pl::Phase, std::int32_t, std::int32_t,
                                                   std::int32_t) { return std::int32_t{'a'}; };
    const std::array<llmp::chat::ToolDefinition, 1> definitions{
        {{.name = "lookup", .description = {}}}};
    auto request = llmp::chat::ToolRequest::Create(
        definitions,
        named ? llmp::chat::ToolChoice{llmp::chat::ToolChoiceKind::kNamed, "lookup"}
              : llmp::chat::ToolChoice{},
        true, {});
    ASSERT_TRUE(request);
    auto constraint = std::make_shared<llmp::chat::ToolSampling>(
        llm_->tokenizer(), llmp::chat::QwenToolCalls(), std::move(*request),
        llmp::base::json::Limits{}, false, std::span<const std::int32_t>{});
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
    rt::GenerateOptions options;
    options.max_tokens = 4;
    options.stop = false;
    options.tool_sampling = constraint;
    const auto before = llm_->counts().greedy_tokens;
    rt::Generation output;
    ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
    ASSERT_EQ(output.tokens.size(), 4U);
    if (named) {
      EXPECT_EQ(llm_->counts().greedy_tokens, before);  // forced on the host
      std::string text;
      ASSERT_TRUE(llm_->tokenizer().Decode(output.tokens, {}, text));
      EXPECT_EQ(text, "<too");
    } else {
      // The first token comes from the prompt's own row; the rest are the
      // device's choices.
      EXPECT_EQ(llm_->counts().greedy_tokens - before, 3U);
      EXPECT_EQ(output.tokens, (std::vector<std::int32_t>(4, 'a')));
    }
    EXPECT_FALSE(constraint->refused());
  }
};
TEST_F(LlmToolDevice, UnrestrictedAutoStepsKeepTheDeviceGreedyChoice) { Run(false); }
TEST_F(LlmToolDevice, ANamedChoiceForcesItsOpenerOnTheHost) { Run(true); }

// RE-050: a head wider than the tokenizer (six padding rows past the byte
// tokenizer's 259 tokens) whose device argmax picks a padding row never
// emits it nor fails the step: the host chooses again from the read-back
// row where the plan retains head rows, and without them chooses every
// token itself.
class LlmPaddedHead : public LlmTools, public ::testing::WithParamInterface<bool> {};
TEST_P(LlmPaddedHead, APaddingRowTheDeviceChoosesIsChosenAgainOnTheHost) {
  retained_rows_ = GetParam();
  score_width_ = 265;
  StartScores(false, 1, 128, 32);
  ASSERT_TRUE(rt::LlmTestAccess::UseChat(*llm_, ByteTokenizer(), "{{ messages[0].content }}"));
  ASSERT_EQ(llm_->tokenizer().size(), 259U);
  source_->options().transform_logits = [](pl::Phase, std::int32_t, std::int32_t,
                                           std::span<float> row) {
    std::ranges::fill(row, 0.0F);
    row['a'] = 9;
    row[262] = 20;  // a padding row wins the whole head
  };
  source_->options().transform_native_token = [](pl::Phase, std::int32_t, std::int32_t,
                                                 std::int32_t) { return std::int32_t{262}; };
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  const auto before = llm_->counts().greedy_tokens;
  rt::Generation output;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
  EXPECT_EQ(output.tokens, (std::vector<std::int32_t>(4, 'a')));
  EXPECT_EQ(llm_->counts().greedy_tokens, before);  // no device choice kept
}
INSTANTIATE_TEST_SUITE_P(RetainedRows, LlmPaddedHead, ::testing::Bool());

// RE-050's fix: a build that read the chat assets bounds the device's greedy
// choices at the tokenizer's vocabulary and hands Setup the assets, which
// adopts them. The device never picks a padding row of the 265-row head, so
// its greedy choices stand without retained head rows; a bound that is not
// the tokenizer's is refused.
TEST_F(LlmTools, ABoundHeadKeepsTheDeviceGreedyChoiceWithoutRetainedRows) {
  score_width_ = 265;
  choice_vocabulary_ = 259;
  chat_assets_ = rt::ChatAssets{
      .spec = ByteTokenizerSpec(), .chat_template = "{{ messages[0].content }}", .from = "test"};
  StartScores(false, 1, 128, 32);
  ASSERT_EQ(llm_->tokenizer().size(), 259U);
  EXPECT_EQ(llm_->logit_width(), 265U);
  source_->options().transform_logits = [](pl::Phase, std::int32_t, std::int32_t,
                                           std::span<float> row) {
    std::ranges::fill(row, 0.0F);
    row['a'] = 9;
  };
  // The bounded device argmax: the best of the tokenizer's rows, never a
  // padding one.
  source_->options().transform_native_token = [](pl::Phase, std::int32_t, std::int32_t,
                                                 std::int32_t) { return std::int32_t{'a'}; };
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  const auto before = llm_->counts().greedy_tokens;
  rt::Generation output;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
  EXPECT_EQ(output.tokens, (std::vector<std::int32_t>(4, 'a')));
  // The first token comes from the prompt's own row; the rest are the
  // device's choices.
  EXPECT_EQ(llm_->counts().greedy_tokens - before, 3U);
}

// The host's proposals from drafters' head rows (no native draft tokens)
// read only the tokenizer's rows, as a bounded device argmax does: though a
// padding row wins every draft row, each proposal is the target's choice
// and is kept, where a padding proposal would be refused by every verdict.
TEST_F(LlmTools, HostDraftsOfABoundHeadProposeNoPaddingRow) {
  score_width_ = 265;
  choice_vocabulary_ = 259;
  chat_assets_ = rt::ChatAssets{
      .spec = ByteTokenizerSpec(), .chat_template = "{{ messages[0].content }}", .from = "test"};
  context_rows_ = 128;
  chunk_rows_ = 32;
  slot_count_ = 1;
  Start(true, true);  // host proposals from F32 draft rows, host verdicts
  ASSERT_NE(llm_, nullptr);
  Open({&BranchAt(0)});
  ASSERT_EQ(llm_->tokenizer().size(), 259U);
  source_->options().transform_logits = [](pl::Phase phase, std::int32_t, std::int32_t,
                                           std::span<float> row) {
    std::ranges::fill(row, 0.0F);
    row['a'] = 9;
    if (phase == pl::Phase::kDraft) row[262] = 20;  // a padding row wins every draft
  };
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  rt::Generation output;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
  EXPECT_EQ(output.tokens, (std::vector<std::int32_t>(8, 'a')));
  EXPECT_GT(output.drafted, 0U);
  EXPECT_GT(output.accepted, 0U);
}

TEST_F(LlmRender, AHeadBoundAtAnotherVocabularyThanItsTokenizerIsRefused) {
  score_width_ = 265;
  choice_vocabulary_ = 258;
  chat_assets_ = rt::ChatAssets{
      .spec = ByteTokenizerSpec(), .chat_template = "{{ messages[0].content }}", .from = "test"};
  node_ = std::make_unique<en::PagedNode>(
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
  ASSERT_TRUE(node_->Open());
  llm_ = std::make_unique<rt::Llm>(*node_, entry_, Settings(false), roles_, 0, Build(false));
  const auto set = llm_->Setup();
  ASSERT_FALSE(set);
  EXPECT_NE(set.error().find("tokenizer's 259 tokens"), std::string::npos) << set.error();
  llm_.reset();  // never set up: no runner for the node's teardown
}

// A device verify's verdict holds while the constraint allows each row's
// device choice; the first refused one ends it before that row (B3).
TEST_F(LlmTools, DeviceVerifyEndsBeforeTheFirstRowTheConstraintRefuses) {
  QwenChat();
  const auto opener = llm_->tokenizer().Find("<tool_call>");
  ASSERT_TRUE(opener);
  const std::array<std::int32_t, 1> terminals{'#'};
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(definitions, {}, true, {});
  ASSERT_TRUE(request);
  llmp::chat::ToolSampling sampling(llm_->tokenizer(), llmp::chat::QwenToolCalls(), *request,
                                    llmp::base::json::Limits{}, false, terminals);
  sampling.Budget(64, 64);
  ASSERT_TRUE(sampling.Identity());
  rt::LlmTestAccess::UseTools(BranchAt(0), &sampling);
  const auto judge = [&](std::vector<std::int32_t> verified, std::vector<std::int32_t> tokens) {
    return rt::LlmTestAccess::Judge(*llm_, 0, true, verified, std::as_bytes(std::span(tokens)));
  };
  // Answer text: every row is unrestricted, so the device verdict stands.
  auto text = judge({'a', 'b', 'c', 'd'}, {'b', 'c', 'x', 'y'});
  ASSERT_TRUE(text) << text.error();
  EXPECT_EQ(text->keep, 3U);
  EXPECT_EQ(text->tokens, (std::vector<std::int32_t>{'b', 'c', 'x'}));
  // The kept opener starts a block, where the terminal '#' is refused: the
  // verdict keeps the opener (the next unit chooses its row on the host).
  auto call = judge({'a', *opener, '#', 'b'}, {*opener, '#', 'b', 'c'});
  ASSERT_TRUE(call) << call.error();
  EXPECT_EQ(call->keep, 1U);
  EXPECT_EQ(call->tokens, (std::vector<std::int32_t>{*opener}));
  // Allowed device choices inside the block are still exact.
  auto inside = judge({'a', *opener, '\n', '<'}, {*opener, '\n', '<', 'f'});
  ASSERT_TRUE(inside) << inside.error();
  EXPECT_EQ(inside->keep, 4U);
  EXPECT_EQ(inside->tokens, (std::vector<std::int32_t>{*opener, '\n', '<', 'f'}));
  EXPECT_FALSE(sampling.refused());
  EXPECT_TRUE(sampling.Identity());  // the judge worked on a copy
  rt::LlmTestAccess::UseTools(BranchAt(0), nullptr);
}

// A greedy step's device choice that the tool constraint refuses is made on
// the host from the head row the runner reads back (DeviceChooses): under
// tool_choice none the device's call opener gives way to the row's best
// allowed token, and an allowed device choice stands.
TEST_F(LlmTools, ARefusedDeviceChoiceIsMadeFromItsReadBackRow) {
  retained_rows_ = true;
  QwenChat();
  const auto opener = llm_->tokenizer().Find("<tool_call>");
  ASSERT_TRUE(opener);
  const std::int32_t call = *opener;
  std::int32_t native = call;
  source_->options().transform_logits = [call](pl::Phase, std::int32_t, std::int32_t,
                                               std::span<float> row) {
    std::ranges::fill(row, 0.0F);
    row[static_cast<std::size_t>(call)] = 9;
    row['a'] = 5;
  };
  source_->options().transform_native_token = [&native](pl::Phase, std::int32_t, std::int32_t,
                                                        std::int32_t) { return native; };
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(definitions,
                                                 {llmp::chat::ToolChoiceKind::kNone, {}}, true, {});
  ASSERT_TRUE(request);
  const auto generate = [&](std::vector<std::int32_t>& tokens) {
    auto constraint = std::make_shared<llmp::chat::ToolSampling>(
        llm_->tokenizer(), llmp::chat::QwenToolCalls(), *request, llmp::base::json::Limits{}, false,
        std::span<const std::int32_t>{});
    ASSERT_FALSE(constraint->Identity());
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
    rt::GenerateOptions options;
    options.max_tokens = 4;
    options.stop = false;
    options.tool_sampling = constraint;
    rt::Generation output;
    ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
    EXPECT_FALSE(constraint->refused());
    tokens = output.tokens;
  };
  // The first token comes from the prompt's row on the host; every later
  // one is the device's choice, refused (the opener) and made from its row.
  auto greedy = llm_->counts().greedy_tokens;
  auto read = llm_->counts().head_rows;
  std::vector<std::int32_t> tokens;
  generate(tokens);
  EXPECT_EQ(tokens, (std::vector<std::int32_t>(4, 'a')));
  EXPECT_EQ(llm_->counts().head_rows - read, 3U);
  EXPECT_EQ(llm_->counts().greedy_tokens, greedy);
  // An allowed device choice is kept as it is, with nothing read back.
  native = 'b';
  greedy = llm_->counts().greedy_tokens;
  read = llm_->counts().head_rows;
  generate(tokens);
  EXPECT_EQ(tokens, (std::vector<std::int32_t>{'a', 'b', 'b', 'b'}));
  EXPECT_EQ(llm_->counts().head_rows, read);
  EXPECT_EQ(llm_->counts().greedy_tokens - greedy, 3U);
}

// A device verify under a restricting constraint (B3): its readback names the
// first row whose device choice the constraint refuses, and its verdict makes
// that row's choice on the host from the read-back row, keeping the rows
// before it.
TEST_F(LlmTools, ADeviceVerifyMakesTheRefusedRowsChoiceFromItsReadBackRow) {
  retained_rows_ = true;
  QwenChat();
  const auto opener = llm_->tokenizer().Find("<tool_call>");
  ASSERT_TRUE(opener);
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(definitions,
                                                 {llmp::chat::ToolChoiceKind::kNone, {}}, true, {});
  ASSERT_TRUE(request);
  llmp::chat::ToolSampling sampling(llm_->tokenizer(), llmp::chat::QwenToolCalls(), *request,
                                    llmp::base::json::Limits{}, false,
                                    std::span<const std::int32_t>{});
  sampling.Budget(64, 64);
  ASSERT_FALSE(sampling.Identity());
  rt::LlmTestAccess::UseTools(BranchAt(0), &sampling);
  const auto judge = [&](std::vector<std::int32_t> verified, std::vector<std::int32_t> tokens) {
    return rt::LlmTestAccess::Judge(*llm_, 0, true, verified, std::as_bytes(std::span(tokens)));
  };
  std::vector<float> row(llm_->tokenizer().size(), 0.0F);
  row[static_cast<std::size_t>(*opener)] = 9;
  row['z'] = 5;
  // Every choice allowed: nothing to read back, the device verdict stands.
  const std::vector<std::int32_t> verified{'a', 'b', 'c', 'd'};
  EXPECT_FALSE(rt::LlmTestAccess::Readback(*llm_, 0, verified,
                                           std::vector<std::int32_t>{'b', 'c', 'x', 'y'}));
  // Row 2 chose the opener: the readback names it, the verdict keeps rows 0
  // to 2 and the host's choice from its row.
  const std::vector<std::int32_t> at2{'b', 'c', *opener, 'y'};
  EXPECT_EQ(rt::LlmTestAccess::Readback(*llm_, 0, verified, at2), 2U);
  rt::LlmTestAccess::ReadRow(*llm_, 0, 2, row);
  auto made = judge(verified, at2);
  ASSERT_TRUE(made) << made.error();
  EXPECT_EQ(made->keep, 3U);
  EXPECT_EQ(made->tokens, (std::vector<std::int32_t>{'b', 'c', 'z'}));
  // Without its row the verdict ends before it.
  rt::LlmTestAccess::ClearRow(*llm_, 0);
  auto cut = judge(verified, at2);
  ASSERT_TRUE(cut) << cut.error();
  EXPECT_EQ(cut->keep, 2U);
  EXPECT_EQ(cut->tokens, (std::vector<std::int32_t>{'b', 'c'}));
  // Row 0 refused: only its read-back row makes a verdict.
  const std::vector<std::int32_t> at0{*opener, 'c', 'd', 'e'};
  EXPECT_EQ(rt::LlmTestAccess::Readback(*llm_, 0, verified, at0), 0U);
  EXPECT_FALSE(judge(verified, at0));
  rt::LlmTestAccess::ReadRow(*llm_, 0, 0, row);
  auto first = judge(verified, at0);
  ASSERT_TRUE(first) << first.error();
  EXPECT_EQ(first->keep, 1U);
  EXPECT_EQ(first->tokens, (std::vector<std::int32_t>{'z'}));
  // A plain step is a verify of its anchor alone.
  EXPECT_EQ(rt::LlmTestAccess::Readback(*llm_, 0, std::vector<std::int32_t>{'a'},
                                        std::vector<std::int32_t>{*opener}),
            0U);
  EXPECT_FALSE(rt::LlmTestAccess::Readback(*llm_, 0, std::vector<std::int32_t>{'a'},
                                           std::vector<std::int32_t>{'q'}));
  rt::LlmTestAccess::ClearRow(*llm_, 0);
  EXPECT_FALSE(sampling.refused());
  rt::LlmTestAccess::UseTools(BranchAt(0), nullptr);
}

// The lazy constrained choices are the masked row's (B4): greedy takes an
// allowed argmax, top-k keeps exactly the masked row's k best, and every
// sampled choice is a token the constraint allows.
TEST_F(LlmTools, LazyChoicesMatchTheMaskedRow) {
  auto constraint = Constraint();
  constraint->Budget(64, 64);
  auto& branch = BranchAt(0);
  rt::LlmTestAccess::UseTools(branch, constraint.get());
  // Named and forcing "<tool_call><function=lookup>": only a few tokens pass.
  std::vector<float> row(259), scratch;
  std::uint32_t x = 7;
  for (auto& v : row) {
    x = x * 1103515245U + 12345U;
    v = static_cast<float>((x >> 8U) % 1000U) / 100.0F;
  }
  const auto masked = constraint->Mask(row, scratch);
  std::vector<std::int32_t> allowed;
  for (std::size_t i = 0; i < masked.size(); ++i)
    if (std::isfinite(masked[i])) allowed.push_back(static_cast<std::int32_t>(i));
  ASSERT_FALSE(allowed.empty());
  ASSERT_LT(allowed.size(), row.size());
  // Ranked as top-k ranks: logit, then the lower ID.
  auto ranked = allowed;
  std::ranges::sort(ranked, [&](std::int32_t a, std::int32_t b) {
    return row[static_cast<std::size_t>(a)] > row[static_cast<std::size_t>(b)] ||
           (row[static_cast<std::size_t>(a)] == row[static_cast<std::size_t>(b)] && a < b);
  });
  // A k above the allowed count keeps them all, as the masked row's top-k.
  for (const std::size_t k : {std::size_t{1}, ranked.size(), ranked.size() + 3}) {
    auto kept = rt::LlmTestAccess::LeadingAllowed(*llm_, branch, row, *constraint, k);
    ASSERT_TRUE(kept) << k;
    std::vector<std::int32_t> best(
        ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(std::min(k, ranked.size())));
    std::ranges::sort(best);
    EXPECT_EQ(*kept, best) << k;
  }
  // Greedy: the masked argmax, however the row's own argmax is placed.
  auto greedy = rt::LlmTestAccess::Choose(*llm_, branch, row, 1);
  ASSERT_TRUE(greedy) << greedy.error();
  EXPECT_EQ(*greedy, ranked.front());
  // The row's own argmax a token the constraint refuses.
  auto refused = row;
  std::size_t outside = 0;
  while (std::isfinite(masked[outside])) ++outside;
  refused[outside] = 100.0F;
  greedy = rt::LlmTestAccess::Choose(*llm_, branch, refused, 1);
  ASSERT_TRUE(greedy) << greedy.error();
  EXPECT_EQ(*greedy, ranked.front());
  // From the row's leading tokens, with no mask of the whole row built.
  EXPECT_LT(rt::LlmTestAccess::ConstrainedLogits(branch), row.size());
  std::int32_t next = -1;
  auto kept = rt::LlmTestAccess::Keep(*llm_, branch, refused, ranked.front(), 1, next);
  ASSERT_TRUE(kept) << kept.error();
  EXPECT_TRUE(*kept);
  EXPECT_EQ(next, ranked.front());
  kept =
      rt::LlmTestAccess::Keep(*llm_, branch, refused, static_cast<std::int32_t>(outside), 1, next);
  ASSERT_TRUE(kept) << kept.error();
  EXPECT_FALSE(*kept);
  EXPECT_EQ(next, ranked.front());
  // Allowed tokens tied below the refused argmax: the lower ID, as the
  // masked row's Greedy; and the same when every allowed token ranks last,
  // past the leading tokens first judged.
  if (allowed.size() >= 2) {
    auto tied = refused;
    tied[static_cast<std::size_t>(allowed[1])] = 50.0F;
    tied[static_cast<std::size_t>(allowed[0])] = 50.0F;
    greedy = rt::LlmTestAccess::Choose(*llm_, branch, tied, 1);
    ASSERT_TRUE(greedy) << greedy.error();
    EXPECT_EQ(*greedy, allowed[0]);
    kept = rt::LlmTestAccess::Keep(*llm_, branch, tied, allowed[1], 1, next);
    ASSERT_TRUE(kept) << kept.error();
    EXPECT_FALSE(*kept);
    EXPECT_EQ(next, allowed[0]);
  }
  auto last = refused;
  for (const auto token : allowed) last[static_cast<std::size_t>(token)] = -50.0F;
  last[static_cast<std::size_t>(allowed.back())] = -40.0F;
  greedy = rt::LlmTestAccess::Choose(*llm_, branch, last, 1);
  ASSERT_TRUE(greedy) << greedy.error();
  EXPECT_EQ(*greedy, allowed.back());
  // Sampled (top-k, then temperature alone): always an allowed token.
  for (const auto params :
       {llmp::execution::SamplingParams{.temperature = 1, .top_k = 3, .top_p = 1, .min_p = 0},
        llmp::execution::SamplingParams{.temperature = 2, .top_k = 0, .top_p = 1, .min_p = 0}}) {
    rt::LlmTestAccess::UseSampling(branch, params, 99);
    for (std::uint64_t position = 0; position < 64; ++position) {
      auto token = rt::LlmTestAccess::Choose(*llm_, branch, refused, position);
      ASSERT_TRUE(token) << token.error();
      EXPECT_NE(std::ranges::find(allowed, *token), allowed.end());
    }
  }
  rt::LlmTestAccess::UseSampling(branch, std::nullopt);
  // Waiting for reasoning to end before a required call: everything but the
  // terminal is allowed, and the terminal leads the row.
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(
      definitions, {llmp::chat::ToolChoiceKind::kRequired, {}}, true, {});
  ASSERT_TRUE(request);
  const std::array<std::int32_t, 1> terminals{'#'};
  llmp::chat::ToolSampling waiting(llm_->tokenizer(), llmp::chat::QwenToolCalls(), *request,
                                   llmp::base::json::Limits{}, true, terminals);
  waiting.Budget(64, 64);
  rt::LlmTestAccess::UseTools(branch, &waiting);
  auto leading = row;
  leading['#'] = 100.0F;
  std::vector<std::int32_t> order(259);
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int32_t>(i);
  std::erase(order, '#');
  std::ranges::sort(order, [&](std::int32_t a, std::int32_t b) {
    return leading[static_cast<std::size_t>(a)] > leading[static_cast<std::size_t>(b)] ||
           (leading[static_cast<std::size_t>(a)] == leading[static_cast<std::size_t>(b)] && a < b);
  });
  auto top = rt::LlmTestAccess::LeadingAllowed(*llm_, branch, leading, waiting, 3);
  ASSERT_TRUE(top);
  std::vector<std::int32_t> three(order.begin(), order.begin() + 3);
  std::ranges::sort(three);
  EXPECT_EQ(*top, three);
  auto chosen = rt::LlmTestAccess::Choose(*llm_, branch, leading, 1);
  ASSERT_TRUE(chosen) << chosen.error();
  EXPECT_EQ(*chosen, order.front());
  rt::LlmTestAccess::UseTools(branch, nullptr);
  EXPECT_FALSE(constraint->refused());
  EXPECT_FALSE(waiting.refused());
}

TEST_F(LlmTools, PlainKnownOpeningRunsWithItsSelectingAnchorInOneCausalUnit) {
  auto constraint = Constraint();
  auto& branch = BranchAt(0);
  const std::array<std::int32_t, 1> prompt{'x'};
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(prompt, last));
  const auto before = source_->built().size();
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  std::vector<std::int32_t> reported;
  options.on_tokens = [&](auto fresh) {
    reported.insert(reported.end(), fresh.begin(), fresh.end());
    return true;
  };
  rt::Generation output;
  ASSERT_TRUE(branch.Generate(last, options, output));
  EXPECT_EQ(output.steps, 1U);
  EXPECT_EQ(output.tokens.size(), 8U);
  EXPECT_EQ(reported, output.tokens);
  std::string text;
  ASSERT_TRUE(llm_->tokenizer().Decode(output.tokens, {}, text));
  EXPECT_EQ(text, "<tool_ca");
  auto expected = std::vector<std::int32_t>(prompt.begin(), prompt.end());
  expected.insert(expected.end(), output.tokens.begin(), output.tokens.end() - 1);
  EXPECT_EQ(State(0), expected);
  bool multirow = false;
  for (std::size_t i = before; i < source_->built().size(); ++i) {
    const auto& shape = source_->built()[i].shape;
    multirow = multirow || (shape.phase == pl::Phase::kPrefill && shape.segments.size() == 1 &&
                            shape.segments[0].rows == 7);
  }
  EXPECT_TRUE(multirow);
}

class LlmToolRouteStops : public LlmScores,
                          public testing::WithParamInterface<std::tuple<bool, bool>> {};
TEST_P(LlmToolRouteStops, ExactGrammarRouteStopContinuesThroughCommonGeneration) {
  const auto [speculative, reasoning] = GetParam();
  score_width_ = 262;
  StartScores(speculative, 1, 256, 32);
  auto spec = ByteTokenizerSpec();
  spec.tokens.emplace_back("<|tool_call>");
  spec.kinds.push_back(llmp::tokenizer::TokenKind::kControl);
  spec.tokens.emplace_back("<tool_call|>");
  spec.kinds.push_back(llmp::tokenizer::TokenKind::kControl);
  spec.tokens.emplace_back("<channel|>");
  spec.kinds.push_back(llmp::tokenizer::TokenKind::kControl);
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(tokenizer);
  ASSERT_TRUE(
      rt::LlmTestAccess::UseChat(*llm_, std::move(*tokenizer), "{{ messages[0].content }}"));
  rt::LlmTestAccess::Stops(*llm_, {258});
  std::vector<std::int32_t> script;
  if (reasoning) script = {'r', 261};
  script.push_back(259);
  for (const char ch : std::string("call:lookup{}")) script.push_back(ch);
  script.push_back(260);
  script.push_back(258);
  source_->options().transform_logits = [script](pl::Phase, std::int32_t, std::int32_t position,
                                                 std::span<float> row) {
    std::ranges::fill(row, -std::numeric_limits<float>::infinity());
    const auto at = std::min(static_cast<std::size_t>(position), script.size() - 1);
    row[static_cast<std::size_t>(script[at])] = 10;
    // A genuine EOS is always preferred, but cannot end an open required call.
    row[258] = 20;
  };
  source_->options().transform_native_token = [script](pl::Phase phase, std::int32_t,
                                                       std::int32_t position,
                                                       std::int32_t selected) -> std::int32_t {
    if (phase != pl::Phase::kDraft) return selected;
    return script[std::min(static_cast<std::size_t>(position), script.size() - 1)];
  };
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(
      definitions, {llmp::chat::ToolChoiceKind::kNamed, "lookup"}, true, {});
  ASSERT_TRUE(request);
  const std::array<std::int32_t, 1> stops{258}, route{259};
  auto constraint = std::make_shared<llmp::chat::ToolSampling>(
      llm_->tokenizer(), llmp::chat::GemmaToolCalls(), std::move(*request),
      llmp::base::json::Limits{}, reasoning, stops, route);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 128;
  options.tool_sampling = constraint;
  options.extra_stops = route;
  if (speculative)
    options.sampling =
        llmp::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
  std::vector<std::int32_t> visible;
  options.on_tokens = [&](auto fresh) {
    visible.insert(visible.end(), fresh.begin(), fresh.end());
    return true;
  };
  rt::Generation generated;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, generated));
  EXPECT_TRUE(generated.stopped);
  EXPECT_EQ(generated.tokens, script);
  script.pop_back();
  EXPECT_EQ(visible, script);
  std::vector<std::int32_t> state{'x'};
  state.insert(state.end(), script.begin(), script.end());
  EXPECT_EQ(BranchAt(0).history(), state);
  EXPECT_EQ(State(0), state);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
  if (speculative) EXPECT_GT(llm_->counts().speculative_units, 0U);
}
INSTANTIATE_TEST_SUITE_P(PlainAndSampledSpeculative, LlmToolRouteStops,
                         testing::Combine(testing::Bool(), testing::Bool()));

TEST_F(LlmTools, AcceptedSpeculativeEosEndsJudgingBeforeAnUnpublishedToolOpener) {
  (void)Constraint(true);
  // A one-byte declared opener makes the post-stop lexical transition
  // unambiguous without adding another canonical segmentation of the prefix.
  auto format = llmp::chat::GemmaToolCalls();
  format.call_begin = "[";
  format.call_end = "]";
  format.continued_route_stops.clear();
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(
      definitions, {llmp::chat::ToolChoiceKind::kNamed, "lookup"}, true, {});
  ASSERT_TRUE(request);
  const std::array<std::int32_t, 1> stops{258};
  auto constraint = std::make_shared<llmp::chat::ToolSampling>(
      llm_->tokenizer(), format, std::move(*request), llmp::base::json::Limits{}, false, stops);
  rt::LlmTestAccess::Stops(*llm_, {258});
  std::vector<std::int32_t> script;
  // The forced span's held '{' is its only candidate, so it is certain and
  // the first verify anchor is '}': five padding bytes then align ']' as a
  // verify anchor at depth two; its draft proposals are then EOS and the next
  // '[' in the same unit.
  for (const char ch : std::string("[lookup{}     ]")) script.push_back(ch);
  script.push_back(258);
  const auto expected = script;
  script.push_back('[');  // accepted draft after EOS would reopen the parser
  script.push_back(258);  // only finite candidate while that call is incomplete
  source_->options().transform_logits = [script](pl::Phase, std::int32_t, std::int32_t position,
                                                 std::span<float> row) {
    std::ranges::fill(row, -std::numeric_limits<float>::infinity());
    const auto at = std::min(static_cast<std::size_t>(position), script.size() - 1);
    row[static_cast<std::size_t>(script[at])] = 10;
  };
  source_->options().transform_native_token = [script](pl::Phase phase, std::int32_t,
                                                       std::int32_t position,
                                                       std::int32_t selected) -> std::int32_t {
    if (phase != pl::Phase::kDraft) return selected;
    return script[std::min(static_cast<std::size_t>(position), script.size() - 1)];
  };
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 64;
  options.tool_sampling = constraint;
  options.keep_logits = true;
  options.sampling =
      llmp::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
  std::vector<std::int32_t> visible;
  options.on_tokens = [&](auto fresh) {
    visible.insert(visible.end(), fresh.begin(), fresh.end());
    return true;
  };
  std::vector<std::int32_t> scored;
  options.on_logits = [&](std::int32_t token, std::span<const float> row) {
    scored.push_back(token);
    EXPECT_EQ(row.size(), 259U);
    return true;
  };
  rt::Generation output;
  const auto generated = BranchAt(0).Generate(last, options, output);
  ASSERT_TRUE(generated) << (generated ? std::string{} : generated.error());
  EXPECT_TRUE(output.stopped);
  EXPECT_EQ(output.tokens, expected);
  EXPECT_EQ(scored, expected);
  EXPECT_EQ(output.logits.size(), expected.size());
  std::vector<std::int32_t> state{'x'};
  state.insert(state.end(), expected.begin(), expected.end() - 1);
  EXPECT_EQ(visible, (std::vector<std::int32_t>{expected.begin(), expected.end() - 1}));
  EXPECT_EQ(BranchAt(0).history(), state);
  EXPECT_EQ(State(0), state);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
  EXPECT_GT(llm_->counts().speculative_units, 0U);
  EXPECT_GT(output.accepted, 0U);
  EXPECT_GT(output.drafted, output.accepted);
  EXPECT_FALSE(constraint->refused());
}

TEST_F(LlmTools, CertainSpanScoresEveryPublishedTokenFromItsActualCausalRow) {
  auto constraint = Constraint();
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  options.keep_logits = true;
  std::vector<std::int32_t> scored;
  options.on_logits = [&](std::int32_t token, std::span<const float> row) {
    const auto previous = scored.empty() ? 'x' : scored.back();
    EXPECT_EQ(ArgmaxOf(row),
              ts::FakeLogit(previous, static_cast<std::int32_t>(scored.size()), 259));
    scored.push_back(token);
    return true;
  };
  rt::Generation output;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
  EXPECT_EQ(output.steps, 1U);
  EXPECT_EQ(scored, output.tokens);
  EXPECT_EQ(output.logits.size(), output.tokens.size());
  EXPECT_EQ(output.tokens.size(), 8U);
}

TEST_F(LlmTools, EmptyConstraintCandidatesRetireOnlyTheirRequestAndKeepNativeFailuresDistinct) {
  auto constraint = Constraint();
  auto native_failure = std::make_shared<llmp::chat::ToolSampling>(*constraint);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  const auto queued = source_->queued();
  std::ranges::fill(last, -std::numeric_limits<float>::infinity());
  last['z'] = 1;  // finite native output, excluded by the required opener
  rt::Generation refused;
  EXPECT_FALSE(BranchAt(0).Generate(last, options, refused));
  EXPECT_TRUE(constraint->refused());
  EXPECT_TRUE(llm_->generation_cohort_usable());
  EXPECT_TRUE(refused.tokens.empty());
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_EQ(State(0), (std::vector<std::int32_t>{'x'}));
  options.tool_sampling = native_failure;
  last['z'] = -std::numeric_limits<float>::infinity();
  rt::Generation invalid;
  EXPECT_FALSE(BranchAt(0).Generate(last, options, invalid));
  EXPECT_FALSE(native_failure->refused());
  EXPECT_EQ(source_->queued(), queued);
  options.tool_sampling.reset();
  last['a'] = 1;
  options.max_tokens = 1;
  rt::Generation next;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, next));
  EXPECT_EQ(next.tokens, (std::vector<std::int32_t>{'a'}));
}

TEST_F(LlmTools, ChooseValidatesExcludedNativeLogitsBeforeMaskingOrRefusal) {
  auto initial = Constraint();
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  const auto queued = source_->queued();
  for (const bool sampled : {false, true}) {
    for (const bool allowed : {false, true}) {
      for (const float invalid :
           {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto constraint = std::make_shared<llmp::chat::ToolSampling>(*initial);
        std::ranges::fill(last, -std::numeric_limits<float>::infinity());
        last[allowed ? '<' : 'z'] = 1;
        last[258] = invalid;  // excluded by the named opener
        rt::GenerateOptions options;
        options.max_tokens = 1;
        options.stop = false;
        options.tool_sampling = constraint;
        if (sampled) options.sampling = llmp::execution::SamplingParams{};
        rt::Generation output;
        auto generated = BranchAt(0).Generate(last, options, output);
        ASSERT_FALSE(generated);
        EXPECT_THAT(generated.error(), HasSubstr("invalid-logits"));
        EXPECT_FALSE(constraint->refused());
        EXPECT_TRUE(output.tokens.empty());
        EXPECT_EQ(source_->queued(), queued);
        EXPECT_EQ(State(0), (std::vector<std::int32_t>{'x'}));
      }
    }
  }
  // Negative infinity is legal padding, including at excluded IDs.
  std::ranges::fill(last, -std::numeric_limits<float>::infinity());
  last['<'] = 1;
  rt::GenerateOptions options;
  options.max_tokens = 1;
  options.stop = false;
  options.tool_sampling = initial;
  rt::Generation output;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, output));
  EXPECT_EQ(output.tokens, (std::vector<std::int32_t>{'<'}));
}

TEST_F(LlmTools, CertainSpanCannotBypassDecodeExecutionQualification) {
  auto constraint = Constraint();
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  rt::Generation output;
  auto session = BranchAt(0).BeginGeneration(last, options, output);
  ASSERT_TRUE(session);
  ASSERT_EQ(output.tokens.size(), 1U);
  const auto queued = source_->queued();
  source_->options().check_execution_use = [](en::NumericalUse use) -> en::Status {
    if (use == en::NumericalUse::kDecode) return std::unexpected("decode qualification withdrawn");
    return {};
  };
  EXPECT_FALSE((*session)->RunScalarStep());
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_EQ(output.tokens.size(), 1U);
  EXPECT_EQ(State(0), (std::vector<std::int32_t>{'x'}));
  EXPECT_FALSE((*session)->Finish());
}

TEST_F(LlmTools, CertainSpanSkipsModelDrafterAndPreservesRealTargetWrittenTap) {
  auto constraint = Constraint(true);
  auto& branch = BranchAt(0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{'x'}, last));
  const auto before = source_->built().size();
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  rt::Generation output;
  ASSERT_TRUE(branch.Generate(last, options, output));
  EXPECT_EQ(output.steps, 1U);
  EXPECT_EQ(output.drafted, 0U);
  for (std::size_t i = before; i < source_->built().size(); ++i)
    EXPECT_NE(source_->built()[i].shape.phase, pl::Phase::kDraft);
  EXPECT_EQ(State(0), State(0, kMtp));
}

TEST_F(LlmTools, CancelledPublicAnchorResumesItsPendingSpanWithoutReplay) {
  auto constraint = Constraint();
  auto& branch = BranchAt(0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  std::vector<std::int32_t> reported;
  bool pause = true;
  options.on_tokens = [&](auto fresh) {
    reported.insert(reported.end(), fresh.begin(), fresh.end());
    return !pause;
  };
  rt::Generation output;
  ASSERT_TRUE(branch.Generate(last, options, output));
  ASSERT_EQ(output.tokens.size(), 1U);
  ASSERT_FALSE(constraint->certain().empty());
  EXPECT_EQ(reported, output.tokens);
  pause = false;
  auto resumed = branch.ResumeGeneration({}, options, output);
  ASSERT_TRUE(resumed);
  while (!(*resumed)->done()) ASSERT_TRUE((*resumed)->RunScalarStep());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(output.tokens.size(), 8U);
  EXPECT_EQ(reported, output.tokens);
  EXPECT_EQ(output.steps, 1U);
}

namespace {
class ToolExchange : public rt::api::Exchange {
 public:
  bool Admit(const Admission&) override { return true; }
  bool Next(rt::Phase, std::uint64_t) override { return true; }
  bool Reasoning(std::string_view value) override {
    reasoning += value;
    return !paused;
  }
  bool Content(std::string_view value) override {
    text += value;
    if (pause_text) paused = true;
    return !paused;
  }
  bool ToolCall(const llmp::chat::ToolUsePart& call) override {
    calls.push_back(call);
    paused = true;
    return !serial;
  }
  bool Continue() override { return !paused; }
  bool Paused() const override { return paused && !serial; }
  bool Yielding() const override { return paused && serial; }
  bool paused = false, serial = false, pause_text = false;
  std::string text, reasoning;
  std::vector<llmp::chat::ToolUsePart> calls;
};
}  // namespace

TEST_F(LlmTools, PrepareRejectsUnrepresentableActiveNamesBeforeNativeWork) {
  QwenChat();
  rt::RequestMemory memory(64 * 1024 * 1024);
  ToolExchange exchange;
  const auto queued = source_->queued();
  for (const auto name : {"a>b", " padded", "padded\t", "lookup"}) {
    for (const auto choice : {"auto", "required", "named", "none"}) {
      std::string quoted;
      llmp::base::json::AppendQuoted(name, quoted);
      std::string body =
          R"({"model":"m","max_tokens":8,"messages":[{"role":"user","content":"read"}],"functions":[{"name":)" +
          quoted + "}],\"function_call\":";
      if (std::string_view(choice) == "named")
        body += "{\"name\":" + quoted + "}";
      else
        llmp::base::json::AppendQuoted(choice, body);
      body += '}';
      {
        auto request = rt::api::ParseChatRequest(body, &memory);
        ASSERT_TRUE(request);
        auto prepared = rt::service_detail::PrepareChat(*llm_, *request, exchange, memory);
        if (std::string_view(choice) == "none" || std::string_view(name) == "lookup") {
          ASSERT_TRUE(prepared) << prepared.error().message;
        } else {
          ASSERT_FALSE(prepared) << body;
          EXPECT_EQ(prepared.error().status, 400);
          EXPECT_EQ(prepared.error().code, "invalid_tool_name");
        }
      }
      EXPECT_EQ(source_->queued(), queued);
      EXPECT_EQ(memory.used(), 0U);
    }
  }
}

TEST_F(LlmTools, PreparedOutputPaymentFollowsPausedStateAfterRequestRetires) {
  QwenChat();
  rt::RequestMemory memory(64 * 1024 * 1024);
  ToolExchange exchange;
  exchange.serial = true;
  std::shared_ptr<rt::service_detail::ChatOutput::State> saved;
  std::uint64_t held = 0;
  {
    auto request = rt::api::ParseChatRequest(R"({"model":"m","max_tokens":128,
      "messages":[{"role":"user","content":"read"}],
      "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]})",
                                             &memory);
    ASSERT_TRUE(request);
    auto prepared = rt::service_detail::PrepareChat(*llm_, *request, exchange, memory);
    ASSERT_TRUE(prepared) << prepared.error().message;
    ASSERT_TRUE(prepared->tools);
    rt::service_detail::ChatOutput output(*llm_, exchange, false, prepared->tools,
                                          prepared->output_charge);
    const std::string generated = "<function=lookup><parameter=s>x</parameter></function>tail";
    const std::vector<std::int32_t> ids(generated.begin(), generated.end());
    ASSERT_FALSE(output.Push(ids));
    ASSERT_TRUE(output.pending());
    // The output's share grew with the call's bytes as they arrived.
    held = prepared->tools->charge.bytes() + prepared->tools->options->charge.bytes() +
           prepared->output_charge->bytes();
    saved = output.Save();
  }
  EXPECT_EQ(memory.used(), held);
  EXPECT_GT(held, 0U);
  {
    rt::service_detail::ChatOutput resumed(*llm_, exchange, false);
    resumed.Restore(std::move(saved));
    exchange.paused = false;
    ASSERT_TRUE(resumed.Drain());
    ASSERT_TRUE(resumed.Finish());
    EXPECT_EQ(exchange.calls.size(), 1U);
    EXPECT_EQ(exchange.text, "tail");
    EXPECT_EQ(memory.used(), held);
  }
  EXPECT_EQ(memory.used(), 0U);
}

TEST_F(LlmTools, ToolOutputMemoryFollowsGeneratedBytesNotMaxTokens) {
  QwenChat();
  // A small request pool: the old prepayment (about 1,200 x the longest token
  // x max_tokens, and max_tokens omitted is the rest of the context) refused
  // this request or grew the pool by reclaiming other models (B2).
  rt::RequestMemory memory(std::uint64_t{32} << 20U);
  ToolExchange exchange;
  exchange.serial = true;
  auto request = rt::api::ParseChatRequest(R"({"model":"m",
      "messages":[{"role":"user","content":"read"}],
      "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]})",
                                           &memory);
  ASSERT_TRUE(request);
  auto prepared = rt::service_detail::PrepareChat(*llm_, *request, exchange, memory);
  ASSERT_TRUE(prepared) << prepared.error().message;
  ASSERT_TRUE(prepared->tools);
  EXPECT_GT(prepared->max_tokens, 2000U);  // the rest of the context
  const auto initial = prepared->tools->charge.bytes();
  EXPECT_LT(initial, std::uint64_t{1} << 20U);
  rt::service_detail::ChatOutput output(*llm_, exchange, false, prepared->tools,
                                        prepared->output_charge);
  const std::string value(20000, 'v');
  const std::string generated = "<function=lookup><parameter=s>" + value;
  const std::vector<std::int32_t> ids(generated.begin(), generated.end());
  ASSERT_TRUE(output.Push(ids));
  const auto grown = prepared->tools->charge.bytes();
  EXPECT_GE(grown, *llmp::chat::ToolOutputBytes(prepared->tools->format, value.size()));
  EXPECT_GT(grown, initial);
  // Past the request pool's capacity the growth is refused for this request alone.
  const std::vector<std::int32_t> more(1'000'000, 'v');
  EXPECT_FALSE(output.Push(more));
  ASSERT_TRUE(output.error());
  EXPECT_LE(memory.used(), memory.capacity());
}

TEST_F(LlmTools, AcceptedCallPauseOwnsUnreadTokenBatchAcrossBothContinuationAdapters) {
  auto sampling = Constraint();
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(definitions, {}, true, {});
  ASSERT_TRUE(request);
  auto options = std::make_shared<rt::api::ToolOptions>();
  options->request = std::move(*request);
  auto tools = std::make_shared<rt::service_detail::ChatTools>();
  tools->options = options;
  tools->format = llmp::chat::QwenToolCalls();
  tools->sampling = sampling;
  const std::string generated =
      "before<function=lookup><parameter=s>value</parameter></function>after";
  const std::vector<std::int32_t> ids(generated.begin(), generated.end());
  for (const bool serial : {false, true}) {
    ToolExchange exchange;
    exchange.serial = serial;
    rt::service_detail::ChatOutput first(*llm_, exchange, false, tools);
    EXPECT_EQ(first.Push(ids), !serial);
    ASSERT_TRUE(first.pending());
    ASSERT_EQ(exchange.calls.size(), 1U);
    EXPECT_EQ(exchange.text, "before");
    auto saved = first.Save();
    rt::service_detail::ChatOutput resumed(*llm_, exchange, false, tools);
    resumed.Restore(std::move(saved));
    exchange.paused = false;
    ASSERT_TRUE(resumed.Drain());
    ASSERT_TRUE(resumed.Finish());
    EXPECT_TRUE(resumed.done());
    EXPECT_FALSE(resumed.error());
    EXPECT_EQ(exchange.calls.size(), 1U);
    EXPECT_EQ(exchange.calls[0].input_json, R"({"s":"value"})");
    EXPECT_EQ(exchange.text, "beforeafter");
  }
}

TEST_F(LlmTools, FinalBareJsonTextPauseResumesWithoutReplayingOrFinishingTwice) {
  auto sampling = Constraint();
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(definitions, {}, true, {});
  ASSERT_TRUE(request);
  auto options = std::make_shared<rt::api::ToolOptions>();
  options->request = std::move(*request);
  auto tools = std::make_shared<rt::service_detail::ChatTools>();
  tools->options = options;
  tools->format = llmp::chat::JsonToolCalls();
  tools->sampling = sampling;
  ToolExchange exchange;
  exchange.serial = true;
  exchange.pause_text = true;
  rt::service_detail::ChatOutput output(*llm_, exchange, false, tools);
  const std::string generated = R"({"answer":"unfinished)";
  const std::vector<std::int32_t> ids(generated.begin(), generated.end());
  ASSERT_TRUE(output.Push(ids));
  EXPECT_FALSE(output.Finish(false));
  EXPECT_TRUE(output.pending());
  exchange.paused = false;
  exchange.pause_text = false;
  ASSERT_TRUE(output.Drain());
  EXPECT_TRUE(output.done());
  EXPECT_FALSE(output.error());
  EXPECT_EQ(exchange.text, generated);
}

namespace {
// A switch yield (Exchange::Yielding) as both paths see it: every event is
// accepted and then pauses its sender; nothing is refused.
class SwitchExchange : public rt::api::Exchange {
 public:
  bool Admit(const Admission&) override { return true; }
  bool Next(rt::Phase, std::uint64_t) override { return true; }
  bool Reasoning(std::string_view value) override {
    reasoning += value;
    return !yielding && !backpressure;
  }
  bool Content(std::string_view value) override {
    text += value;
    ++events;
    return !yielding && !backpressure;
  }
  bool ToolCall(const llmp::chat::ToolUsePart& call) override {
    calls.push_back(call);
    ++events;
    if (pause_after_call) backpressure = true;
    return !yielding && !backpressure;
  }
  bool Continue() override { return !yielding && !backpressure; }
  bool Paused() const override { return backpressure && !yielding; }
  bool Yielding() const override { return yielding; }
  bool yielding = false, backpressure = false, pause_after_call = false;
  std::string text, reasoning;
  std::vector<llmp::chat::ToolUsePart> calls;
  int events = 0;
};
}  // namespace

// The end of a chat output on either path (FinishOrCarry, A11): a complete
// generation's output is finished even when the request yields its place
// for a switch; what the yield pauses is carried and completed exactly once
// by the resumed output-only frame, never dropped as a finished reply.
TEST_F(LlmTools, ASwitchYieldAtTheEndCarriesTheFinalCallAndFlushAndCompletesThemOnce) {
  auto sampling = Constraint();
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(definitions, {}, true, {});
  ASSERT_TRUE(request);
  auto options = std::make_shared<rt::api::ToolOptions>();
  options->request = std::move(*request);
  auto tools = std::make_shared<rt::service_detail::ChatTools>();
  tools->options = options;
  tools->format = llmp::chat::QwenToolCalls();
  tools->sampling = sampling;
  // Held at the generation's end: the call's trailing text (its reader
  // behind when the yield came) and a marker prefix only the finish flushes.
  const std::string generated =
      "before<function=lookup><parameter=s>value</parameter></function>after<fun";
  const std::vector<std::int32_t> ids(generated.begin(), generated.end());
  for (const bool behind : {false, true}) {
    SCOPED_TRACE(behind);
    SwitchExchange exchange;
    exchange.pause_after_call = behind;
    rt::service_detail::ChatOutput output(*llm_, exchange, false, tools);
    (void)output.Push(ids);
    ASSERT_EQ(exchange.calls.size(), 1U);
    EXPECT_EQ(output.pending(), behind);
    // The yield, between the final decode and the output unit.
    exchange.yielding = true;
    EXPECT_TRUE(rt::service_detail::FinishOrCarry(output, true, true, true));
    EXPECT_FALSE(output.done());
    EXPECT_FALSE(output.error());
    auto saved = output.Save();
    // Taken up again, output-only: it drains and finishes what was carried.
    exchange.yielding = false;
    exchange.backpressure = false;
    exchange.pause_after_call = false;
    rt::service_detail::ChatOutput resumed(*llm_, exchange, false, tools);
    resumed.Restore(std::move(saved));
    (void)resumed.Drain();
    EXPECT_FALSE(rt::service_detail::FinishOrCarry(resumed, true, true, false));
    EXPECT_TRUE(resumed.done());
    EXPECT_FALSE(resumed.error());
    EXPECT_EQ(exchange.calls.size(), 1U);
    EXPECT_EQ(exchange.calls[0].input_json, R"({"s":"value"})");
    EXPECT_EQ(exchange.text, "beforeafter<fun");
  }
}

TEST_F(LlmTools, AGenerationUnderWayIsCarriedUnfinishedAndOneNotYieldingFinishes) {
  (void)Constraint();  // the fixture's model
  SwitchExchange exchange;
  rt::service_detail::ChatOutput carried(*llm_, exchange, false);
  const std::vector<std::int32_t> ids{'a', 'b', 'c'};
  ASSERT_TRUE(carried.Push(ids));
  exchange.yielding = true;
  // Its generation continues: carried as it is, nothing finished or flushed.
  EXPECT_TRUE(rt::service_detail::FinishOrCarry(carried, false, false, true));
  EXPECT_FALSE(carried.done());
  // Not yielding (a request that ends, its client gone or its allowance
  // spent): finished, whatever its generation.
  exchange.yielding = false;
  rt::service_detail::ChatOutput ended(*llm_, exchange, false);
  ASSERT_TRUE(ended.Push(ids));
  EXPECT_FALSE(rt::service_detail::FinishOrCarry(ended, false, false, false));
  EXPECT_TRUE(ended.done());
  EXPECT_EQ(exchange.text, "abcabc");
}

TEST_F(LlmTools, MidCertainBatchYieldPreservesUnreportedKnownSuffixForRestoredPublicPrefix) {
  auto constraint = Constraint();
  auto& branch = BranchAt(0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{'x'}, last));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  options.tool_sampling = constraint;
  std::vector<std::int32_t> reported;
  options.on_tokens = [&](auto fresh) {
    reported.insert(reported.end(), fresh.begin(), fresh.end());
    return reported.size() < 3;
  };
  rt::Generation output;
  ASSERT_TRUE(branch.Generate(last, options, output));
  ASSERT_EQ(output.tokens.size(), 3U);
  ASSERT_FALSE(constraint->certain().empty());
  std::vector<std::int32_t> public_prefix{'x'};
  public_prefix.insert(public_prefix.end(), output.tokens.begin(), output.tokens.end() - 1);
  std::uint32_t reused = 0;
  ASSERT_TRUE(branch.PreparePrompt(public_prefix, 0, last, reused, {}, nullptr, false, true));
  options.on_tokens = [&](auto fresh) {
    reported.insert(reported.end(), fresh.begin(), fresh.end());
    return true;
  };
  auto resumed = branch.ResumeGeneration(last, options, output);
  ASSERT_TRUE(resumed);
  while (!(*resumed)->done()) ASSERT_TRUE((*resumed)->RunScalarStep());
  ASSERT_TRUE((*resumed)->Finish());
  EXPECT_EQ(output.tokens.size(), 8U);
  EXPECT_EQ(reported, output.tokens);
  EXPECT_EQ(llm_->Detokenize(output.tokens), "<tool_ca");
}

class LlmToolInvalidVerify : public LlmTools,
                             public ::testing::WithParamInterface<std::tuple<bool, bool, bool>> {};
TEST_P(LlmToolInvalidVerify, ExcludedNativeInvalidityCannotDisappearInKeep) {
  const auto [sampled, allowed, nan] = GetParam();
  (void)Constraint(true);
  bool verified = false;
  source_->options().transform_logits = [&](pl::Phase phase, std::int32_t, std::int32_t,
                                            std::span<float> row) {
    std::ranges::fill(row, -std::numeric_limits<float>::infinity());
    row[phase == pl::Phase::kVerify ? (allowed ? 'a' : 257) : 'r'] = 1;
    if (phase == pl::Phase::kVerify) {
      verified = true;
      row[258] =
          nan ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    }
  };
  source_->options().transform_native_token = [](pl::Phase phase, std::int32_t, std::int32_t,
                                                 std::int32_t token) -> std::int32_t {
    return phase == pl::Phase::kDraft ? 'a' : token;
  };
  const std::array<llmp::chat::ToolDefinition, 1> definitions{
      {{.name = "lookup", .description = {}}}};
  auto request = llmp::chat::ToolRequest::Create(
      definitions, {llmp::chat::ToolChoiceKind::kNamed, "lookup"}, true, {});
  ASSERT_TRUE(request);
  const std::array<std::int32_t, 2> stops{257, 258};
  auto constraint = std::make_shared<llmp::chat::ToolSampling>(
      llm_->tokenizer(), llmp::chat::QwenToolCalls(), std::move(*request),
      llmp::base::json::Limits{}, true, stops);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{'x'}, last));
  const auto target = State(0), tap = State(0, kMtp);
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.tool_sampling = constraint;
  if (sampled) options.sampling = llmp::execution::SamplingParams{};
  std::vector<std::int32_t> reported;
  options.on_tokens = [&](auto tokens) {
    reported.insert(reported.end(), tokens.begin(), tokens.end());
    return true;
  };
  rt::Generation output;
  auto generated = BranchAt(0).Generate(last, options, output);
  ASSERT_FALSE(generated);
  EXPECT_THAT(generated.error(), HasSubstr("invalid-logits"));
  EXPECT_TRUE(verified);
  EXPECT_FALSE(constraint->refused());
  EXPECT_EQ(reported, (std::vector<std::int32_t>{'r'}));
  EXPECT_EQ(output.tokens, reported);
  EXPECT_EQ(output.accepted, 0U);
  EXPECT_EQ(State(0), target);
  EXPECT_EQ(State(0, kMtp), tap);
  EXPECT_EQ(BranchAt(0).history(), target);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
}
INSTANTIATE_TEST_SUITE_P(GreedyAndSampled, LlmToolInvalidVerify,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                            ::testing::Bool()));

class LlmToolVerify : public LlmTools {
 protected:
  void Boundary(std::uint32_t preceding, bool sampled) {
    (void)Constraint(true);
    source_->options().transform_logits = [preceding](pl::Phase phase, std::int32_t,
                                                      std::int32_t position, std::span<float> row) {
      // Negative-infinity padding remains legal through both Choose and Keep.
      std::ranges::fill(row, -std::numeric_limits<float>::infinity());
      const auto selected = position == 0                                      ? 'r'
                            : position <= static_cast<std::int32_t>(preceding) ? 'a'
                                                                               : '~';
      row[static_cast<std::size_t>(selected)] = 90;
      // The target would prefer EOS. Both Choose and Keep must remove it
      // during reasoning before applying the identical greedy/sampled filter.
      if (phase != pl::Phase::kDraft) row[258] = 100;
    };
    source_->options().transform_native_token = [preceding](pl::Phase phase, std::int32_t,
                                                            std::int32_t position,
                                                            std::int32_t token) -> std::int32_t {
      if (phase != pl::Phase::kDraft) return token;
      return position <= static_cast<std::int32_t>(preceding) ? 'a' : '~';
    };
    const std::array<llmp::chat::ToolDefinition, 1> definitions{
        {{.name = "lookup", .description = {}}}};
    auto request = llmp::chat::ToolRequest::Create(
        definitions, {llmp::chat::ToolChoiceKind::kNamed, "lookup"}, true, {});
    ASSERT_TRUE(request);
    auto format = llmp::chat::QwenToolCalls();
    format.reasoning_end = "~";
    const std::array<std::int32_t, 2> stops{'~', 258};
    auto constraint = std::make_shared<llmp::chat::ToolSampling>(
        llm_->tokenizer(), format, std::move(*request), llmp::base::json::Limits{}, true, stops);
    rt::LlmTestAccess::Stops(*llm_, {stops.begin(), stops.end()});
    auto& branch = BranchAt(0);
    std::vector<float> last;
    ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{'x'}, last));
    rt::GenerateOptions options;
    options.max_tokens = 24;
    options.tool_sampling = constraint;
    if (sampled)
      options.sampling =
          llmp::execution::SamplingParams{.temperature = 1, .top_k = 1, .top_p = 1, .min_p = 0};
    std::vector<std::int32_t> reported;
    options.on_tokens = [&](auto fresh) {
      reported.insert(reported.end(), fresh.begin(), fresh.end());
      return reported.empty() || reported.back() != '~';
    };
    rt::Generation output;
    ASSERT_TRUE(branch.Generate(last, options, output));
    std::vector<std::int32_t> expected{'r'};
    expected.insert(expected.end(), preceding, 'a');
    expected.push_back('~');
    EXPECT_EQ(output.tokens, expected);
    EXPECT_EQ(reported, expected);
    EXPECT_FALSE(output.stopped);  // reasoning-end/terminal overlap is resolved
    EXPECT_TRUE(output.cancelled);
    EXPECT_EQ(output.accepted, preceding);
    EXPECT_GT(output.drafted, 0U);
    EXPECT_FALSE(constraint->certain().empty());
    std::vector<std::int32_t> state{'x'};
    state.insert(state.end(), expected.begin(), expected.end() - 1);
    EXPECT_EQ(State(0), state);
    EXPECT_FALSE(llm_->runner().slots().owed(0));
  }
};
TEST_F(LlmToolVerify, RowZeroBoundaryStopsAcceptanceBeforePublishingTheCertainSpan) {
  Boundary(0, false);
}
TEST_F(LlmToolVerify, LaterSampledBoundaryRetainsOnlyTheAuthenticatedEarlierPrefix) {
  Boundary(1, true);
}
