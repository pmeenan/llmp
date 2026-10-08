// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <print>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "engine/checkpoint_file.h"
#include "engine/gemma2_runner.h"
#include "engine/gemma3_runner.h"
#include "plain_token_serving_checks.h"
#include "platform/crash_policy.h"
#include "platform/kept_files.h"
#include "runtime/serving.h"

namespace rt = jitllm::runtime;
namespace en = jitllm::engine;
namespace cfg = jitllm::config;

// Exercise the production adapters through actual PromptSession admission and
// RunPromptWave, rather than handing precomputed hints directly to a runner.
class GemmaPrefillServingGpu : public ::testing::TestWithParam<std::uint32_t> {
 protected:
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  rt::Llm* model = nullptr;
  rt::SwapParts first_activation;
  std::filesystem::path scratch;
  bool retirement_failed = false;
  void SetUp() override {
    std::string name = "/home/pmeenan/.cache/jitllm-gemma-prefill-XXXXXX";
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    life->roles.installed =
        GetParam() == 2 ? "/home/pmeenan/.local/share/jitllm/gemma2-import-20261007/artifacts"
                        : "/home/pmeenan/.local/share/jitllm/gemma3-import-20261007/artifacts";
    life->roles.spill = scratch / "spill";
    life->roles.state = scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      ASSERT_TRUE(std::filesystem::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    cfg::ModelEntry entry;
    entry.name = "gemma";
    entry.artifact = GetParam() == 2
                         ? "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870"
                         : "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb";
    entry.overrides["context"] = std::int64_t{GetParam() == 2 ? 8192 : 4096};
    entry.overrides["prefill_chunk"] = std::int64_t{128};
    entry.overrides["max_slots"] = std::int64_t{2};
    life->config.models.push_back(entry);
  }
  rt::Status Start(bool ahead, bool owner_prefill = true, std::size_t branch_count = 2) {
    if (branch_count == 0 || branch_count > 2)
      return std::unexpected("invalid fixture branch count");
    life->options.gemma2_owner_prefill = owner_prefill;
    life->options.gemma3_owner_prefill = owner_prefill;
    life->options.gemma2_flexible_owner_prefill = owner_prefill;
    life->options.gemma3_flexible_owner_prefill = owner_prefill;
    life->options.gemma2_prefill_lookahead = ahead;
    life->options.gemma2_capture_ahead = ahead;
    life->options.gemma3_prefill_lookahead = ahead;
    life->options.gemma3_capture_ahead = ahead;
    life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, stderr);
    if (auto r = life->server->Start(true); !r) return r;
    model = dynamic_cast<rt::Llm*>(life->server->Find("gemma"));
    if (!model) return std::unexpected("configured Gemma adapter missing");
    first_activation = {};
    if (auto r = life->server->Activate(*model, first_activation); !r) return r;
    std::array<rt::Llm::Branch*, 2> branches{};
    for (std::size_t id = 0; id < branch_count; ++id) {
      auto branch = model->branch(id);
      if (!branch) return std::unexpected(branch.error());
      branches[id] = *branch;
    }
    return life->server->SelectRequestBranches(*model, std::span(branches).first(branch_count));
  }
  rt::Status Retire() {
    if (retirement_failed) return std::unexpected("prior server retirement unproven");
    if (!life->server) return {};
    const auto r = life->server->TearDown();
    retirement_failed = !r;
    if (r) {
      life->server.reset();
      model = nullptr;
    }
    return r;
  }
  void TearDown() override {
    const auto r = Retire();
    EXPECT_TRUE(r) << (r ? "" : r.error());
    if (!r) {
      std::ignore = life.release();
      return;
    }
    std::filesystem::remove_all(scratch);
  }
  template <class Runner>
  std::expected<jitllm::base::Sha256Digest, std::string> StateHash(
      Runner& runner, std::uint32_t id, en::PagedNode* direct = nullptr) {
    const auto slot = runner.request_slot(id);
    if (!slot) return std::unexpected(slot.error());
    auto& node = direct ? *direct : life->server->node();
    std::vector<jitllm::catalog::ExtentId> staging;
    auto buffer = node.Pinned(1U << 20U, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    jitllm::base::Sha256 hash;
    for (const auto& range : (*slot)->state().used_ranges())
      for (std::uint64_t at = 0; at < range.bytes; at += 1U << 20U) {
        const en::LiveState::Range part{range.region, range.offset + at,
                                        std::min<std::uint64_t>(1U << 20U, range.bytes - at)};
        en::LiveState::CopyRetirement retired = en::LiveState::CopyRetirement::kUnproven;
        const auto copied = runner.CopyState(id, *buffer, std::span(&part, 1), &retired);
        if (retired == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(*buffer);
        if (!copied) {
          if (retired == en::LiveState::CopyRetirement::kProven) (void)node.FreePinned(*buffer);
          return std::unexpected(copied.error());
        }
        hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
      }
    if (auto r = node.FreePinned(*buffer); !r) return std::unexpected(r.error());
    return hash.Finish();
  }
  template <class Runner>
  rt::Status Prompt(Runner& runner, std::array<std::vector<float>, 2>& heads,
                    std::array<jitllm::base::Sha256Digest, 2>& states, double& seconds,
                    std::uint32_t extra_rows = 0) {
    constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
    std::array<std::vector<std::int32_t>, 2> tokens;
    for (std::size_t id = 0; id < 2; ++id) {
      // With five-row tails, retain a 128-row skew within the same padded
      // KV width. A 256-row skew would correctly take the scalar fallback.
      tokens[id].resize((id == 0 ? 1280 : (extra_rows == 0 ? 1536 : 1408)) + extra_rows);
      for (std::size_t i = 0; i < tokens[id].size(); ++i)
        tokens[id][i] = seed[(i + id * 3 + i / 7) % seed.size()];
      // A short completed warmup loads weights without paid prefill plans.
      auto branch = model->branch(static_cast<std::uint32_t>(id));
      if (!branch) return std::unexpected(branch.error());
      std::vector<float> warm;
      if (auto r = (*branch)->Prefill(std::span(tokens[id]).first(3), warm); !r) return r;
      if (auto r = (*branch)->Clear(); !r) return r;
    }
    runner.DropPlans();
    struct Sessions {
      std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> owned;
      ~Sessions() {
        for (auto& session : owned)
          if (session) {
            session->Cancel();
            const auto r = session->Finish();
            EXPECT_TRUE(r) << (r ? "" : r.error());
          }
      }
    } sessions;
    for (std::uint32_t id = 0; id < 2; ++id) {
      auto begun = (*model->branch(id))->BeginPrompt(tokens[id]);
      if (!begun) return std::unexpected(begun.error());
      sessions.owned[id] = std::move(*begun);
      if (auto r = sessions.owned[id]->Advance(); !r) return r;  // scalar reuse
    }
    const auto begin = std::chrono::steady_clock::now();
    while (!sessions.owned[0]->done() || !sessions.owned[1]->done()) {
      // Reverse incoming order, as independent endpoint clients may arrive.
      std::array<rt::Llm::PromptSession*, 2> active{};
      std::size_t count = 0;
      for (const auto id : {1U, 0U})
        if (!sessions.owned[id]->done() && (count == 0 || active[0]->CanJoin(*sessions.owned[id])))
          active[count++] = sessions.owned[id].get();
      const std::array<rt::PrefillGoOn, 2> callbacks{};
      if (auto r = model->RunPromptWave(std::span(active).first(count),
                                        std::span(callbacks).first(count));
          !r)
        return r;
      for (const auto* session : std::span(active).first(count))
        if (!session->last_unit_result()) return session->last_unit_result();
    }
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    for (std::uint32_t id = 0; id < 2; ++id) {
      heads[id] = sessions.owned[id]->last();
      EXPECT_EQ((*model->branch(id))->history(), tokens[id]);
      auto state = StateHash(runner, id);
      if (!state) return std::unexpected(state.error());
      states[id] = *state;
    }
    return {};
  }
  void CheckConfiguredRoots(std::uint32_t rows, std::uint32_t chunks, bool prepare_state = false);
};

TEST_P(GemmaPrefillServingGpu, PlainRuntimeTokensPreserveRowsStateAndRestoredContinuations) {
  ASSERT_TRUE(Start(false));
  const auto exercise = [&](auto& runner) {
    return jitllm::test::CheckPlainServing(
        *life->server, *model, runner.profile().vocab,
        [&](std::uint32_t id) { return StateHash(runner, id); },
        [&] { return runner.greedy_tokens(); });
  };
  const auto r = GetParam() == 2 ? exercise(dynamic_cast<en::Gemma2Runner&>(model->paged()))
                                 : exercise(dynamic_cast<en::Gemma3Runner&>(model->paged()));
  ASSERT_TRUE(r) << (r ? "" : r.error());
}

TEST_P(GemmaPrefillServingGpu, DefaultPreparationPublishesOnlyActualPromptProgress) {
  EXPECT_TRUE(en::Gemma2Options{}.prepare_state);
  EXPECT_TRUE(rt::ServingOptions{}.gemma2_prepare_state);
  EXPECT_TRUE(en::Gemma3Options{}.prepare_state);
  EXPECT_TRUE(rt::ServingOptions{}.gemma3_prepare_state);
  std::array<std::vector<float>, 2> expected;
  std::array<jitllm::base::Sha256Digest, 2> expected_state{};
  std::array<std::int64_t, 2> expected_joined{};
  for (const bool ordinary : {false, true}) {
    life->options = rt::ServingOptions{};
    if (!ordinary) {
      life->options.gemma2_prepare_state = false;
      life->options.gemma3_prepare_state = false;
    }
    ASSERT_TRUE(Start(true));
    const auto exercise = [&](auto& runner) {
      const bool gemma2 = GetParam() == 2;
      ASSERT_EQ(runner.layout().local_cells, gemma2 ? 4352U : 1280U);
      const auto before = runner.state_preparation_stats();
      std::array<std::vector<float>, 2> heads;
      std::array<jitllm::base::Sha256Digest, 2> states{};
      double unused_seconds = 0;
      const auto result = Prompt(runner, heads, states, unused_seconds, gemma2 ? 3072U : 0U);
      ASSERT_TRUE(result) << (result ? "" : result.error());
      const auto prepared = runner.state_preparation_stats();
      if (ordinary) {
        EXPECT_GT(prepared.submitted, before.submitted);
        EXPECT_GT(prepared.completed_extents, before.completed_extents);
        EXPECT_GT(prepared.adopted_extents, before.adopted_extents);
      } else {
        EXPECT_EQ(prepared.attempted, before.attempted);
        EXPECT_EQ(prepared.submitted, before.submitted);
        EXPECT_EQ(prepared.completed_extents, before.completed_extents);
        EXPECT_EQ(prepared.adopted_extents, before.adopted_extents);
      }
      EXPECT_EQ(prepared.failed, before.failed);
      EXPECT_EQ(prepared.refused, before.refused);
      EXPECT_TRUE(runner.cohort_usable());
      EXPECT_FALSE(life->server->node().has_pending_state_preparation());
      for (std::uint32_t owner = 0; owner < 2; ++owner) {
        ASSERT_EQ(heads[owner].size(), gemma2 ? 256000U : 262208U);
        EXPECT_TRUE(std::ranges::all_of(heads[owner], [](float v) { return std::isfinite(v); }));
        EXPECT_TRUE((*runner.request_slot(owner))->state_usable());
        EXPECT_EQ((*runner.request_slot(owner))->completed_positions(),
                  gemma2 ? (owner == 0 ? 4352U : 4480U) : (owner == 0 ? 1280U : 1536U));
      }
      const auto extra = jitllm::base::json::Parse(model->extra());
      ASSERT_TRUE(extra);
      std::array<std::int64_t, 2> joined{};
      for (std::size_t i = 0; i < joined.size(); ++i) {
        const auto field =
            extra->root().find(i == 0 ? "joined_prefill_groups" : "joined_prefill_rows");
        ASSERT_TRUE(field);
        joined[i] = field->int64().value_or(0);
        EXPECT_GT(joined[i], 0);
      }
      if (!ordinary) {
        expected = heads;
        expected_state = states;
        expected_joined = joined;
      } else {
        for (std::size_t owner = 0; owner < heads.size(); ++owner)
          EXPECT_EQ(std::memcmp(heads[owner].data(), expected[owner].data(),
                                heads[owner].size() * sizeof(float)),
                    0);
        EXPECT_EQ(states, expected_state);
        EXPECT_EQ(joined, expected_joined);
        std::println(
            "GEMMA_DEFAULT_PREPARATION_RUNTIME completed={} adopted={} state_exact=1 "
            "head_exact=1 history_exact=1 joined_groups={} joined_rows={}",
            prepared.completed_extents - before.completed_extents,
            prepared.adopted_extents - before.adopted_extents, joined[0], joined[1]);
      }
    };
    if (GetParam() == 2)
      exercise(dynamic_cast<en::Gemma2Runner&>(model->paged()));
    else
      exercise(dynamic_cast<en::Gemma3Runner&>(model->paged()));
    if (HasFatalFailure()) return;
    ASSERT_TRUE(Retire());
  }
}

TEST_P(GemmaPrefillServingGpu, Admitted8448ScalarPreparationPreservesPromptAndContinuation) {
  if (GetParam() != 3) GTEST_SKIP() << "Gemma3 admitted scalar context";
  EXPECT_TRUE(en::Gemma3Options{}.prepare_state);
  EXPECT_TRUE(rt::ServingOptions{}.gemma3_prepare_state);
  const auto input =
      std::filesystem::path("/home/pmeenan/.local/share/jitllm/references/gemma3-depth/ids.i32");
  ASSERT_EQ(std::filesystem::file_size(input), 33024U);
  std::vector<std::int32_t> ids(8256);
  std::ifstream stream(input, std::ios::binary);
  ASSERT_TRUE(stream.read(reinterpret_cast<char*>(ids.data()),
                          static_cast<std::streamsize>(ids.size() * sizeof(ids[0]))));
  jitllm::base::Sha256 hash;
  ASSERT_EQ(jitllm::base::ToHex(hash.Update(std::as_bytes(std::span(ids))).Finish()),
            "44196b939c8b53b535a59e1c139f6dc4a7959f7880cd8c8a9d91688818f5ad67");
  ASSERT_TRUE(std::ranges::all_of(ids, [](std::int32_t id) { return id >= 0 && id < 262208; }));
  ASSERT_EQ(ids.front(), 2);
  ASSERT_EQ(ids[8192], 496);
  const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + 8192);
  std::vector<float> expected_head, expected_next;
  jitllm::base::Sha256Digest expected_state{}, expected_next_state{};
  life->config.models.front().overrides["context"] = std::int64_t{8448};
  life->config.models.front().overrides["max_slots"] = std::int64_t{1};
  for (const bool ordinary : {false, true}) {
    life->options = rt::ServingOptions{};
    if (!ordinary) life->options.gemma3_prepare_state = false;
    ASSERT_TRUE(Start(true, true, 1));
    auto& runner = dynamic_cast<en::Gemma3Runner&>(model->paged());
    ASSERT_EQ(runner.layout().context, 8448U);
    ASSERT_EQ(runner.layout().max_rows, 128U);
    ASSERT_EQ(runner.layout().local_cells, 1280U);
    ASSERT_FALSE(model->branch(1));
    auto branch = model->branch(0);
    ASSERT_TRUE(branch);
    std::vector<float> warm;
    ASSERT_TRUE((*branch)->Prefill(std::span(prompt).first(3), warm));
    ASSERT_TRUE((*branch)->Clear());
    runner.DropPlans();
    const auto before = runner.state_preparation_stats();
    auto begun = (*branch)->BeginPrompt(prompt);
    ASSERT_TRUE(begun);
    rt::Status run;
    std::uint32_t chunks = 0;
    while (!(*begun)->done()) {
      const auto prior = (*runner.request_slot(0))->completed_positions();
      run = (*begun)->Advance();
      if (!run) break;
      const auto current = (*runner.request_slot(0))->completed_positions();
      EXPECT_EQ(current, (*branch)->history().size());
      EXPECT_LE(current, 8192U);
      EXPECT_LE(current - prior, 128U);
      chunks += current > prior;
    }
    if (!run) (*begun)->Cancel();
    const auto finished = (*begun)->Finish();
    ASSERT_TRUE(run) << (run ? "" : run.error());
    ASSERT_TRUE(finished) << (finished ? "" : finished.error());
    EXPECT_EQ(chunks, 64U);
    const auto head = (*begun)->last();
    begun->reset();
    ASSERT_EQ(head.size(), 262208U);
    ASSERT_TRUE(std::ranges::all_of(head, [](float v) { return std::isfinite(v); }));
    EXPECT_EQ((*branch)->history(), prompt);
    EXPECT_EQ((*runner.request_slot(0))->completed_positions(), 8192U);
    auto state = StateHash(runner, 0);
    ASSERT_TRUE(state);
    const auto prepared = runner.state_preparation_stats();
    if (ordinary) {
      EXPECT_GT(prepared.submitted, before.submitted);
      EXPECT_GT(prepared.completed_extents, before.completed_extents);
      EXPECT_GT(prepared.adopted_extents, before.adopted_extents);
    } else {
      EXPECT_EQ(prepared.attempted, before.attempted);
      EXPECT_EQ(prepared.submitted, before.submitted);
      EXPECT_EQ(prepared.completed_extents, before.completed_extents);
      EXPECT_EQ(prepared.adopted_extents, before.adopted_extents);
    }
    EXPECT_EQ(prepared.refused, before.refused);
    EXPECT_EQ(prepared.failed, before.failed);
    EXPECT_TRUE(runner.cohort_usable());
    EXPECT_TRUE((*runner.request_slot(0))->state_usable());
    EXPECT_FALSE(life->server->node().has_pending_state_preparation());
    std::vector<float> next;
    ASSERT_TRUE((*branch)->Prefill(std::span(ids).subspan(8192, 1), next));
    ASSERT_EQ(next.size(), 262208U);
    ASSERT_TRUE(std::ranges::all_of(next, [](float v) { return std::isfinite(v); }));
    auto continued = prompt;
    continued.push_back(496);
    EXPECT_EQ((*branch)->history(), continued);
    EXPECT_EQ((*runner.request_slot(0))->completed_positions(), 8193U);
    auto next_state = StateHash(runner, 0);
    ASSERT_TRUE(next_state);
    EXPECT_TRUE(runner.cohort_usable());
    EXPECT_TRUE((*runner.request_slot(0))->state_usable());
    EXPECT_FALSE(life->server->node().has_pending_state_preparation());
    const auto extra = jitllm::base::json::Parse(model->extra());
    ASSERT_TRUE(extra);
    const auto joined = extra->root().find("joined_prefill_groups");
    ASSERT_TRUE(joined);
    EXPECT_EQ(joined->int64().value_or(-1), 0);
    EXPECT_EQ(runner.state_preparation_stats().failed, before.failed);
    EXPECT_EQ(runner.state_preparation_stats().refused, before.refused);
    if (!ordinary) {
      expected_head = head;
      expected_state = *state;
      expected_next = next;
      expected_next_state = *next_state;
    } else {
      EXPECT_EQ(std::memcmp(head.data(), expected_head.data(), head.size() * sizeof(float)), 0);
      EXPECT_EQ(*state, expected_state);
      EXPECT_EQ(std::memcmp(next.data(), expected_next.data(), next.size() * sizeof(float)), 0);
      EXPECT_EQ(*next_state, expected_next_state);
      std::println(
          "GEMMA3_ADMITTED_STATE_PREPARATION context=8448 owners=1 prompt=8192 continuation=8193 "
          "completed={} adopted={} chunks={} state_exact=1 head_exact=1 history_exact=1 "
          "continuation_exact=1 registry_drained=1",
          prepared.completed_extents - before.completed_extents,
          prepared.adopted_extents - before.adopted_extents, chunks);
    }
    ASSERT_TRUE(Retire());
  }
}

TEST_P(GemmaPrefillServingGpu, ActualJoinedHintsPreserveHeadsAndState) {
  std::array<std::vector<float>, 2> expected;
  std::array<jitllm::base::Sha256Digest, 2> expected_state{};
  // Short fresh-server bookends, not sustained performance qualification.
  std::uint32_t pass = 0;
  for (const bool ahead : {false, true, true, false}) {
    ASSERT_TRUE(Start(ahead));
    std::array<std::vector<float>, 2> heads;
    std::array<jitllm::base::Sha256Digest, 2> states{};
    double seconds = 0;
    const auto exercise = [&](auto& runner) {
      const auto r = Prompt(runner, heads, states, seconds);
      EXPECT_TRUE(r) << (r ? "" : r.error());
      const auto& stats = runner.lookahead_stats();
      if (ahead) {
        EXPECT_GT(stats.built, 0U);
        EXPECT_EQ(stats.built, stats.cached);
        EXPECT_GT(stats.captured_ahead, 0U);
      } else {
        EXPECT_EQ(stats.built, 0U);
        EXPECT_EQ(stats.captured_ahead, 0U);
      }
      EXPECT_EQ(stats.dropped_ahead, 0U);
      EXPECT_EQ(runner.coverage().violations, 0U);
    };
    if (GetParam() == 2)
      exercise(dynamic_cast<en::Gemma2Runner&>(model->paged()));
    else
      exercise(dynamic_cast<en::Gemma3Runner&>(model->paged()));
    const auto parsed = jitllm::base::json::Parse(model->extra());
    ASSERT_TRUE(parsed);
    const auto joined = parsed->root().find("joined_prefill_groups");
    ASSERT_TRUE(joined);
    EXPECT_GT(joined->int64().value_or(0), 0);
    for (const auto& head : heads) {
      ASSERT_EQ(head.size(), GetParam() == 2 ? 256000U : 262208U);
      ASSERT_TRUE(std::ranges::all_of(head, [](float value) { return std::isfinite(value); }));
    }
    if (pass++ == 0) {
      expected = heads;
      expected_state = states;
    } else {
      for (std::size_t id = 0; id < 2; ++id) {
        ASSERT_EQ(heads[id].size(), expected[id].size());
        EXPECT_EQ(
            std::memcmp(heads[id].data(), expected[id].data(), heads[id].size() * sizeof(float)),
            0);
      }
      EXPECT_EQ(states, expected_state);
    }
    std::println("GEMMA_PREFILL_ADAPTER family={} ahead={} seconds={} extra={}", GetParam(), ahead,
                 seconds, model->extra());
    ASSERT_TRUE(Retire());
  }
}

TEST_P(GemmaPrefillServingGpu, ActualOwnerPrefillPreservesHeadsStateAndRestart) {
  life->options.keep_conversations = true;
  std::array<std::vector<float>, 2> expected, expected_next;
  std::array<jitllm::base::Sha256Digest, 2> expected_state{};
  std::uint32_t pass = 0;
  for (const bool owners : {false, true, true, false}) {
    ASSERT_TRUE(Start(true, owners));
    std::array<std::vector<float>, 2> heads;
    std::array<jitllm::base::Sha256Digest, 2> states{};
    double seconds = 0;
    const auto exercise = [&](auto& runner) {
      const auto r = Prompt(runner, heads, states, seconds, 5);
      EXPECT_TRUE(r) << (r ? "" : r.error());
      const auto& selected = runner.plan_selections();
      EXPECT_EQ(selected.flexible_owner_prefill_attention > 0, owners);
      EXPECT_GT(owners ? selected.owner_prefill_attention : selected.packed_prefill_attention, 0U);
      EXPECT_EQ(owners ? selected.packed_prefill_attention : selected.owner_prefill_attention, 0U);
      EXPECT_GT(runner.lookahead_stats().captured_ahead, 0U);
      EXPECT_EQ(runner.coverage().violations, 0U);
    };
    if (GetParam() == 2)
      exercise(dynamic_cast<en::Gemma2Runner&>(model->paged()));
    else
      exercise(dynamic_cast<en::Gemma3Runner&>(model->paged()));
    const auto parsed = jitllm::base::json::Parse(model->extra());
    ASSERT_TRUE(parsed);
    const auto selected = parsed->root().find("bound_owner_prefill_attention");
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->int64().value_or(0) > 0, owners);
    const auto flexible = parsed->root().find("bound_flexible_owner_prefill_attention");
    ASSERT_TRUE(flexible);
    EXPECT_EQ(flexible->int64().value_or(0) > 0, owners);
    for (const auto& head : heads) {
      ASSERT_EQ(head.size(), GetParam() == 2 ? 256000U : 262208U);
      ASSERT_TRUE(std::ranges::all_of(head, [](float value) { return std::isfinite(value); }));
    }
    if (pass == 0) {
      expected = heads;
      expected_state = states;
    } else {
      EXPECT_EQ(states, expected_state);
      for (std::size_t id = 0; id < 2; ++id)
        EXPECT_EQ(
            std::memcmp(heads[id].data(), expected[id].data(), heads[id].size() * sizeof(float)),
            0);
    }
    std::array<std::vector<std::int32_t>, 2> histories;
    for (std::uint32_t id = 0; id < 2; ++id) histories[id] = (*model->branch(id))->history();
    const auto retired = life->server->RetireRequestBranches(*model, true);
    ASSERT_TRUE(retired.result);
    ASSERT_TRUE(retired.references_retired);
    for (std::uint32_t id = 0; id < 2; ++id) {
      ASSERT_TRUE(model->SpillIdle(**model->branch(id)));
      EXPECT_TRUE((*model->branch(id))->spilled());
    }
    life->server->Persist(rt::Clock::now() + std::chrono::seconds(30), {});
    ASSERT_TRUE(Retire());
    ASSERT_TRUE(Start(true, owners));
    for (std::uint32_t id = 0; id < 2; ++id) {
      auto* branch = *model->branch(id);
      ASSERT_EQ(branch->history(), histories[id]);
      auto restoring = branch->BeginPrompt(histories[id], 0, false, true);
      ASSERT_TRUE(restoring);
      rt::Status run;
      while (!(*restoring)->done()) {
        run = (*restoring)->Advance();
        if (!run) break;
      }
      if (!run) (*restoring)->Cancel();
      const auto finished = (*restoring)->Finish();
      ASSERT_TRUE(run);
      ASSERT_TRUE(finished);
      EXPECT_EQ((*restoring)->reused(), histories[id].size());
      EXPECT_TRUE((*restoring)->last().empty());
      restoring->reset();
      auto restored = GetParam() == 2
                          ? StateHash(dynamic_cast<en::Gemma2Runner&>(model->paged()), id)
                          : StateHash(dynamic_cast<en::Gemma3Runner&>(model->paged()), id);
      ASSERT_TRUE(restored);
      EXPECT_EQ(*restored, states[id]);
      constexpr std::array<std::int32_t, 1> anchor{563};
      std::vector<float> next;
      ASSERT_TRUE(branch->Prefill(anchor, next));
      ASSERT_EQ(next.size(), GetParam() == 2 ? 256000U : 262208U);
      ASSERT_TRUE(std::ranges::all_of(next, [](float value) { return std::isfinite(value); }));
      if (pass == 0)
        expected_next[id] = next;
      else
        EXPECT_EQ(std::memcmp(next.data(), expected_next[id].data(), next.size() * sizeof(float)),
                  0);
    }
    std::println("GEMMA_OWNER_PREFILL_ADAPTER family={} owners={} seconds={}", GetParam(), owners,
                 seconds);
    ASSERT_TRUE(Retire());
    // The next factor arm starts with no kept metadata or spill backing.
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      std::filesystem::remove_all(path);
      ASSERT_TRUE(std::filesystem::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    ++pass;
  }
}

// Public Gemma chunk admission remains 128. Qualify the explicitly configured
// larger runner and its kept-file layout without bypassing that API clamp.
void GemmaPrefillServingGpu::CheckConfiguredRoots(std::uint32_t kRows, std::uint32_t kChunks,
                                                  bool prepare_state) {
  const std::uint32_t kPositions = kRows * kChunks;
  const auto exercise = [&]<class Runner, class Options>() -> en::Status {
    EXPECT_EQ(Options{}.prefill_lookahead_capacity, 2U);
    struct DirectLife {
      en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
      std::unique_ptr<Runner> runner;
      std::vector<en::PagedModel*> entered;
      bool retired = false;
    };
    const auto directory = scratch / "larger-runner";
    if (!std::filesystem::create_directory(directory) || chmod(directory.c_str(), 0700) != 0)
      return std::unexpected("larger-runner private directory");
    auto opened = jitllm::platform::OpenPrivateDirectory(-1, directory.c_str());
    if (!opened) return std::unexpected("larger-runner directory open");
    const int spill_dir = *opened;
    auto primary = std::make_unique<DirectLife>(), restarted = std::make_unique<DirectLife>();
    auto packed = std::make_unique<DirectLife>();
    const auto retire = [&](DirectLife& target) -> en::Status {
      if (target.retired) return {};
      target.retired = true;
      const auto result = target.node.TearDown(target.entered);
      if (!result) retirement_failed = true;
      return result;
    };
    const auto start = [&](DirectLife& target, bool keep, bool owners) -> en::Status {
      Options options;
      options.artifact = life->roles.installed / *life->config.models[0].artifact;
      options.out = directory;
      options.context = GetParam() == 2 ? 8192 : 4096;
      options.max_rows = kRows;
      options.slots = 2;
      options.max_wave_rows = 2 * kRows;
      options.max_head_rows = 2;
      options.owner_decode = true;
      options.packed_prefill = true;
      options.owner_prefill = owners;
      options.flexible_owner_prefill = owners;
      options.bounded_roots = true;
      options.device_masks = true;
      options.fuse_norms = true;
      options.fuse_quant_glu = true;
      options.fuse_norm_rope = GetParam() == 3;
      options.fuse_norm_add = true;
      options.prefill_lookahead = true;
      options.capture_ahead = true;
      options.prefill_lookahead_capacity = 2;
      if constexpr (requires { options.prepare_state; })
        options.prepare_state = prepare_state && owners;
      options.spill_place = [spill_dir, keep, owners](std::uint32_t slot) {
        return en::LiveState::SpillPlace{
            .directory = {},
            .dir = spill_dir,
            .name = std::format("{}slot{}.kv", owners ? "" : "packed-", slot),
            .keep = keep};
      };
      auto& node = target.node;
      target.runner = std::make_unique<Runner>(node, std::move(options), 0, 0);
      auto& runner = *target.runner;
      if (auto r = node.Open(); !r) return r;
      target.entered.push_back(&runner);
      if (auto r = runner.Setup(); !r) return r;
      if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r)
        return r;
      const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
      node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (16ULL << 20U));
      if (auto r = node.Start(jitllm::base::Bytes(
              fixed + runner.weights().size() * en::kPagedExtent + 4 * node.StateCapacity()));
          !r)
        return r;
      if (auto r = runner.Register(); !r) return r;
      if (auto r = runner.Bind(); !r) return r;
      node.Run();
      return runner.SelectSlots(std::array<std::uint32_t, 2>{0, 1});
    };
    const auto vocab = GetParam() == 2 ? 256000U : 262208U;
    const auto valid = [&](const std::vector<float>& head) {
      return head.size() == vocab &&
             std::ranges::all_of(head, [](float value) { return std::isfinite(value); });
    };
    std::vector<en::LiveState::Range> footprint;
    std::vector<float> expected_next;
    jitllm::base::Sha256Digest expected_state{};
    std::string source_layout;
    std::uint64_t selected_owner_steps = 0;
    std::array<std::vector<float>, 2> packed_heads;
    std::array<jitllm::base::Sha256Digest, 2> packed_states;
    const auto prefill = [&](Runner& runner, std::array<std::vector<float>, 2>& heads,
                             bool hinted) -> en::Status {
      constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
      std::vector<std::int32_t> ids(kPositions);
      for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = seed[i % seed.size()];
      const auto before = runner.lookahead_stats();
      const auto replayed = runner.graph_stats().replayed;
      en::LiveState::PreparationStats preparation_before;
      if constexpr (requires { runner.state_preparation_stats(); })
        preparation_before = runner.state_preparation_stats();
      const bool require_wrap = kRows == 512 || prepare_state;
      if (prepare_state &&
          (kRows != 128 || runner.layout().local_cells != (GetParam() == 2 ? 4352U : 1280U) ||
           kPositions <= runner.layout().local_cells))
        return std::unexpected("prepared 128-row ring wrap not exercised");
      if (kRows == 512 && (runner.layout().local_cells != (GetParam() == 2 ? 4608U : 1536U) ||
                           kPositions <= runner.layout().local_cells))
        return std::unexpected("configured 512-row ring wrap not exercised");
      std::uint32_t wrapped_root_waves = 0, wrapped_replays = 0;
      for (std::uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        const auto tokens = std::span(ids).subspan(chunk * kRows, kRows);
        std::array<typename Runner::Work, 2> work{
            {{0, chunk * kRows, tokens, &heads[0]}, {1, chunk * kRows, tokens, &heads[1]}}};
        const bool last = chunk + 1 == kChunks;
        const std::array<typename Runner::PrefillNext, 2> next{
            {{0, last ? 0U : kRows, chunk + 2 < kChunks ? kRows : 0U},
             {1, last ? 0U : kRows, chunk + 2 < kChunks ? kRows : 0U}}};
        const auto replayed_before = runner.graph_stats().replayed;
        if (auto r = runner.WavePrefill(work, last, std::span(next).first(hinted ? 2U : 0U),
                                        chunk + 2 == kChunks, chunk + 3 == kChunks);
            !r)
          return r;
        if (hinted && require_wrap && (chunk + 1) * kRows > runner.layout().local_cells) {
          const auto& selected = runner.plan_selections();
          if (selected.owner_prefill_attention == 0 || selected.packed_prefill_attention != 0 ||
              selected.largest_owner_prefill_rows != kRows)
            return std::unexpected("wrapped root-only plans not selected");
          ++wrapped_root_waves;
          wrapped_replays += runner.graph_stats().replayed > replayed_before;
        }
        for (std::uint32_t owner = 0; owner < 2; ++owner)
          if ((*runner.request_slot(owner))->completed_positions() != (chunk + 1) * kRows)
            return std::unexpected("lookahead advanced a future state cursor");
        if (!last && (!heads[0].empty() || !heads[1].empty()))
          return std::unexpected("state-only larger chunk published a head");
      }
      if (!valid(heads[0]) || !valid(heads[1]))
        return std::unexpected("larger independent owner heads incomplete");
      const auto& after = runner.lookahead_stats();
      if (hinted) {
        if (kRows == 256) {
          if (after.built_pairs - before.built_pairs != 1 ||
              after.cached_pairs - before.cached_pairs != 1 || after.built - before.built != 2 ||
              after.cached - before.cached != 2 ||
              after.captured_ahead - before.captured_ahead != 1 ||
              runner.graph_stats().replayed - replayed != 1)
            return std::unexpected("larger two-future construction/capture/replay not executed");
        } else if ((!prepare_state && (after.built_pairs <= before.built_pairs ||
                                       after.cached_pairs <= before.cached_pairs)) ||
                   after.built - before.built < 2 ||
                   after.cached - before.cached != after.built - before.built ||
                   after.captured_ahead <= before.captured_ahead ||
                   runner.graph_stats().replayed <= replayed || after.refused != before.refused ||
                   after.dropped_ahead != before.dropped_ahead || wrapped_root_waves == 0 ||
                   wrapped_replays == 0) {
          return std::unexpected("wrapped root capture/replay/lifetime path not executed");
        }
      } else if (after.built != before.built || after.cached != before.cached) {
        return std::unexpected("unhinted packed control built a future");
      }
      if constexpr (requires { runner.state_preparation_stats(); }) {
        const auto prepared = runner.state_preparation_stats();
        if (prepare_state && hinted) {
          if (prepared.submitted <= preparation_before.submitted ||
              prepared.completed_extents <= preparation_before.completed_extents ||
              prepared.adopted_extents <= preparation_before.adopted_extents ||
              prepared.failed != preparation_before.failed ||
              prepared.refused != preparation_before.refused)
            return std::unexpected("actual 128-row preparation/adoption not executed");
          std::println("GEMMA_PREPARED_STATE_LIFETIME rows={} positions={} completed={} adopted={}",
                       kRows, kPositions,
                       prepared.completed_extents - preparation_before.completed_extents,
                       prepared.adopted_extents - preparation_before.adopted_extents);
        } else if (prepared.submitted != preparation_before.submitted) {
          return std::unexpected("unhinted baseline prepared future state");
        }
      }
      return {};
    };
    const auto checked = [&]() -> en::Status {
      if (auto r = start(*packed, false, false); !r) return r;
      auto control = packed->node.WithRequest(
          0, packed->runner->closure(), "larger identical-input packed control",
          [&]() -> en::Status {
            if (auto r = prefill(*packed->runner, packed_heads, false); !r) return r;
            for (std::uint32_t owner = 0; owner < 2; ++owner) {
              const auto state = StateHash(*packed->runner, owner, &packed->node);
              if (!state) return std::unexpected("larger packed state read: " + state.error());
              packed_states[owner] = *state;
            }
            const auto selection = packed->runner->plan_selections();
            if (selection.owner_prefill_attention || !selection.packed_prefill_attention ||
                packed->runner->coverage().violations)
              return std::unexpected("larger packed control path not selected");
            return {};
          });
      if (!control) return control;
      if (auto r = retire(*packed); !r) return r;
      if (auto r = start(*primary, false, true); !r) return r;
      auto& runner = *primary->runner;
      source_layout = runner.CheckpointLayoutId();
      auto produced = primary->node.WithRequest(
          0, runner.closure(), "larger owner restart", [&]() -> en::Status {
            std::array<std::vector<float>, 2> heads;
            if (auto r = prefill(runner, heads, true); !r) return r;
            if (primary->node.has_pending_state_preparation())
              return std::unexpected("prepared state owner still linked before checkpoint");
            for (std::size_t owner = 0; owner < 2; ++owner)
              if (std::memcmp(heads[owner].data(), packed_heads[owner].data(),
                              vocab * sizeof(float)))
                return std::unexpected("larger identical-input root differs from packed owner");
            std::uint64_t differing = 0;
            double total_difference = 0, max_difference = 0;
            for (std::size_t index = 0; index < vocab; ++index) {
              const double difference = std::abs(double(heads[0][index]) - double(heads[1][index]));
              differing += heads[0][index] != heads[1][index];
              total_difference += difference;
              max_difference = std::max(max_difference, difference);
            }
            const auto best = [](const auto& head) {
              return std::distance(head.begin(), std::max_element(head.begin(), head.end()));
            };
            std::println(
                "GEMMA_LARGER_IDENTICAL_INPUT family={} rows={} chunks={} initialized_rows={} "
                "packed_root_exact=1 "
                "cross_owner_differing={} max_abs={} mean_abs={} argmax0={} argmax1={}",
                GetParam(), kRows, kChunks, kPositions, differing, max_difference,
                total_difference / vocab, best(heads[0]), best(heads[1]));
            selected_owner_steps = runner.plan_selections().owner_prefill_attention;
            if (selected_owner_steps == 0 || runner.coverage().violations != 0)
              return std::unexpected("larger actual-root path not selected");
            const auto first = StateHash(runner, 0, &primary->node);
            const auto peer = StateHash(runner, 1, &primary->node);
            if (!first) return std::unexpected("larger owner state read: " + first.error());
            if (!peer) return std::unexpected("larger peer state read: " + peer.error());
            if (*first != packed_states[0] || *peer != packed_states[1])
              return std::unexpected("larger independent root state differs from packed owner");
            expected_state = *first;
            footprint = (*runner.request_slot(0))->state().used_ranges();
            if (footprint.size() < 2) return std::unexpected("larger state footprint missing");
            // A same-owner checkpoint supplies the uninterrupted restart oracle;
            // it requires no cross-owner equality assumption.
            auto checkpoint = en::CheckpointFile::Capture(
                primary->node, directory, footprint,
                [&](void* host, std::span<const en::LiveState::Range> ranges) {
                  return runner.CopyState(0, host, ranges);
                });
            if (!checkpoint) return std::unexpected(checkpoint.error().detail);
            constexpr std::array<std::int32_t, 1> anchor{563};
            const typename Runner::Work next{0, kPositions, anchor, &expected_next};
            if (auto r = runner.Wave(std::span(&next, 1)); !r) return r;
            if (!valid(expected_next))
              return std::unexpected("larger live control head incomplete");
            const auto restored_checkpoint = checkpoint->Restore(
                primary->node,
                [&] { return runner.PrepareRestore(0, kPositions, footprint, source_layout); },
                [&](void* host, std::span<const en::LiveState::Range> ranges) {
                  return runner.CopyState(0, host, ranges, false);
                });
            if (!restored_checkpoint) return std::unexpected(restored_checkpoint.error().detail);
            if (auto r = runner.CompleteRestore(0, kPositions); !r) return r;
            const auto rewound = StateHash(runner, 0, &primary->node);
            const auto unchanged_peer = StateHash(runner, 1, &primary->node);
            if (!rewound)
              return std::unexpected("larger checkpoint state read: " + rewound.error());
            if (!unchanged_peer)
              return std::unexpected("larger checkpoint peer read: " + unchanged_peer.error());
            if (*rewound != expected_state || *unchanged_peer != *peer ||
                (*runner.request_slot(1))->completed_positions() != kPositions)
              return std::unexpected("larger checkpoint or peer state changed");
            if (auto r = runner.Spill(0); !r) return r;
            if (!(*runner.request_slot(0))->is_spilled())
              return std::unexpected("larger owner was not spilled");
            return {};
          });
      if (!produced) return produced;
      if (auto r = retire(*primary); !r) return r;
      if (auto r = start(*restarted, true, true); !r) return r;
      auto& restored = *restarted->runner;
      if (restored.CheckpointLayoutId() != source_layout)
        return std::unexpected("larger restart layout changed");
      auto incomplete = footprint;
      incomplete.pop_back();
      if (restored.Adopt(0, kPositions, footprint, "wrong-larger-layout") ||
          restored.Adopt(0, kPositions, incomplete, source_layout))
        return std::unexpected("larger malformed adoption accepted");
      if (auto r = restored.Adopt(0, kPositions, footprint, source_layout); !r) return r;
      if (restored.Adopt(0, kPositions, footprint, source_layout))
        return std::unexpected("larger duplicate adoption accepted");
      if (auto r = restored.Restore(0); !r) return r;
      return restarted->node.WithRequest(
          0, restored.closure(), "larger adopted continuation", [&]() -> en::Status {
            const auto state = StateHash(restored, 0, &restarted->node);
            if (!state) return std::unexpected("larger adopted state read: " + state.error());
            if (*state != expected_state)
              return std::unexpected("larger adopted state bytes changed");
            constexpr std::array<std::int32_t, 1> anchor{563};
            std::vector<float> head;
            const typename Runner::Work next{0, kPositions, anchor, &head};
            if (auto r = restored.Wave(std::span(&next, 1)); !r) return r;
            if (!valid(head) ||
                std::memcmp(head.data(), expected_next.data(), vocab * sizeof(float)))
              return std::unexpected(
                  "larger adopted continuation differs from uninterrupted owner");
            if ((*restored.request_slot(0))->completed_positions() != kPositions + 1 ||
                (*restored.request_slot(1))->completed_positions() != 0)
              return std::unexpected("larger adopted continuation changed a peer cursor");
            return {};
          });
    }();
    const auto packed_retired = retire(*packed);
    const auto first_retired = retire(*primary), second_retired = retire(*restarted);
    if (retirement_failed) {
      std::ignore = packed.release();
      std::ignore = primary.release();
      std::ignore = restarted.release();
      return std::unexpected("larger runner retirement unproven; files and directory held");
    }
    close(spill_dir);
    if (!packed_retired) return packed_retired;
    if (!first_retired) return first_retired;
    if (!second_retired) return second_retired;
    if (!checked) return checked;
    std::println(
        "GEMMA_LARGER_OWNER_RESTART family={} rows={} initialized_rows={} selected_owner_steps={} "
        "state_exact=1 continuation_exact=1",
        GetParam(), kRows, kPositions, selected_owner_steps);
    return {};
  };
  const auto result = GetParam() == 2
                          ? exercise.template operator()<en::Gemma2Runner, en::Gemma2Options>()
                          : exercise.template operator()<en::Gemma3Runner, en::Gemma3Options>();
  ASSERT_TRUE(result) << (result ? "" : result.error());
}

TEST_P(GemmaPrefillServingGpu, Prepared128RootsPreserveStateAndRestartContinuation) {
  // Both actual profiles cross their own authenticated local ring under a
  // real joined wave, independently of the unequal-prefix timing schedule.
  if (GetParam() == 2) EXPECT_TRUE(en::Gemma2Options{}.prepare_state);
  CheckConfiguredRoots(128, GetParam() == 2 ? 36 : 12, true);
}

TEST_P(GemmaPrefillServingGpu, UnprovenPreparationStopsServerBeforeOwnerDestruction) {
  if (GetParam() != 3) GTEST_SKIP() << "Gemma3 preparation adapter only";
  // This test is launched alone. Parent SetUp only creates paths/configuration;
  // CUDA/node threads are first created inside the isolated child. No fork of
  // an active provider and no production fault hook are involved.
  EXPECT_EXIT(
      ([&] {
        if (!jitllm::platform::InstallCrashPolicy("prepared-owner-control")) std::_Exit(61);
        if (!Start(false)) std::_Exit(62);
        auto& node = life->server->node();
        auto& runner = dynamic_cast<en::Gemma3Runner&>(model->paged());
        // An independently owned state uses the actual node registry without
        // exposing a runner's private state or adding a production fault hook.
        // It and the Server remain alive until the guard terminates this child.
        auto owned_state = std::make_unique<en::LiveState>("unknown-owner-control");
        auto& state = *owned_state;
        if (!node.Call(
                [&]() -> en::Status {
                  if (auto r = state.AddGrowing(node, "unknown owner", en::kPagedExtent, 0); !r)
                    return r;
                  return state.RegisterSpill(node, scratch);
                },
                "register independent preparation owner"))
          std::_Exit(71);
        const std::array ranges = {en::LiveState::Range{0, 0, 16}};
        if (!state.Prepare(node, ranges).value_or(false) || !node.has_pending_state_preparation())
          std::_Exit(63);
        const auto charge = node.host_counted();
        const auto id = state.reserved_extents()[0];
        bool resident = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!resident && std::chrono::steady_clock::now() < deadline) {
          if (!node.Call(
                  [&]() -> en::Status {
                    resident = node.catalog().Describe(id)->state ==
                               jitllm::catalog::ExtentState::kResident;
                    return {};
                  },
                  "waiting for prepared owner"))
            std::_Exit(64);
          if (!resident) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!resident) std::_Exit(65);
        // Synthetic unknown CATALOG outcome, with mapped backing retained.
        // No provider/service eviction is issued or claimed to have retired.
        if (!node.Call(
                [&]() -> en::Status {
                  if (!node.catalog().InvalidateContents(id)) return std::unexpected("invalidate");
                  auto eviction = node.catalog().BeginEvict(id);
                  if (!eviction || !node.catalog().QuarantineEviction(*eviction))
                    return std::unexpected("unknown eviction");
                  return {};
                },
                "retain synthetic unknown prepared backing"))
          std::_Exit(66);
        if (state.FinishPreparation() || !state.preparing() ||
            !node.has_pending_state_preparation() || node.host_counted() != charge ||
            !state.extents().empty())
          std::_Exit(67);
        const std::array<en::PagedModel*, 1> models = {&runner};
        if (node.TearDown(models) || !state.preparing() || !node.has_pending_state_preparation() ||
            node.host_counted() != charge || !node.threaded())
          std::_Exit(68);
        bool owned = false;
        if (!node.Call(
                [&]() -> en::Status {
                  const auto view = node.catalog().Describe(id);
                  owned = view && view->state == jitllm::catalog::ExtentState::kQuarantined &&
                          node.memory()
                              .MappedAt(node.scheduler().SourceOf(id)->backing->reservation,
                                        node.scheduler().SourceOf(id)->backing->offset)
                              .has_value();
                  return {};
                },
                "prove unknown prepared backing remains owned") ||
            !owned)
          std::_Exit(69);
        // Actual Server guard must stop before returning to any destructor.
        (void)life->server->TearDown();
        std::_Exit(70);
      }()),
      ::testing::ExitedWithCode(jitllm::platform::kFatalSignalExitBase + SIGABRT),
      "unretired state preparation; aborting before owner destruction");
}

TEST_P(GemmaPrefillServingGpu, ConfiguredLargerRootsPreserveStateAndRestartContinuation) {
  CheckConfiguredRoots(256, 3);
}

TEST_P(GemmaPrefillServingGpu, Wrapped512RootsPreserveStateAndRestartContinuation) {
  // Both owners cross the actual local ring before a full-state checkpoint,
  // protected-peer restore and kept restart continuation.
  CheckConfiguredRoots(512, GetParam() == 2 ? 10 : 5);
}

TEST_P(GemmaPrefillServingGpu, DiagnosticBudgetCapLeavesDefaultsAndChecksActualAdmission) {
  EXPECT_FALSE(life->options.partial_weight_eviction);
  EXPECT_FALSE(life->options.diagnostic_budget_cap_bytes);
  ASSERT_TRUE(Start(true));
  EXPECT_EQ(life->server->budget(), life->server->dynamic_budget());
  EXPECT_FALSE(first_activation.diagnostic.enabled);
  EXPECT_FALSE(life->server->node().backing_create_stats().timing.enabled);
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
  std::vector<float> expected;
  ASSERT_TRUE((*model->branch(0))->Prefill(prompt, expected));
  ASSERT_FALSE(expected.empty());
  ASSERT_TRUE(Retire());

  constexpr std::uint64_t kCap = std::uint64_t{16} << 30U;
  life->options.diagnostic_budget_cap_bytes = kCap;
  ASSERT_TRUE(Start(true));
  EXPECT_EQ(life->server->budget(), kCap);
  EXPECT_GE(life->server->dynamic_budget(), kCap);
  ASSERT_TRUE(first_activation.diagnostic.enabled);
  EXPECT_FALSE(first_activation.diagnostic.eviction_split_available);  // direct first Load
  EXPECT_GT(first_activation.diagnostic.ready.room_seconds,
            first_activation.diagnostic.requested.room_seconds);
  const auto& diagnostic = first_activation.diagnostic;
  EXPECT_FALSE(diagnostic.creates_loaded.timing.enabled);
  EXPECT_EQ(diagnostic.creates_loaded.timing.ordinary.completed +
                diagnostic.creates_loaded.timing.reserve.completed +
                diagnostic.creates_loaded.timing.park.completed,
            0U);
  EXPECT_EQ(diagnostic.creates_loaded.timing.ordinary.total_ns +
                diagnostic.creates_loaded.timing.reserve.total_ns +
                diagnostic.creates_loaded.timing.park.total_ns,
            0U);
  EXPECT_GT(
      diagnostic.creates_loaded.reserve_attempts + diagnostic.creates_loaded.ordinary_attempts,
      diagnostic.creates_requested.reserve_attempts +
          diagnostic.creates_requested.ordinary_attempts);
  EXPECT_GE(diagnostic.creates_ready.reserve_attempts, diagnostic.creates_loaded.reserve_attempts);
  EXPECT_EQ(diagnostic.creates_ready.reserve_failures, 0U);
  EXPECT_EQ(diagnostic.creates_ready.ordinary_failures, 0U);
  std::vector<float> actual;
  ASSERT_TRUE((*model->branch(0))->Prefill(prompt, actual));
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
  ASSERT_TRUE(Retire());

  life->options.diagnostic_budget_cap_bytes = 1;
  const auto too_small = Start(true);
  EXPECT_FALSE(too_small);
  if (!too_small)
    EXPECT_NE(too_small.error().find("required startup footprint"), std::string::npos);
  ASSERT_TRUE(Retire());
  life->options.diagnostic_budget_cap_bytes = ~std::uint64_t{0};
  const auto too_large = Start(true);
  EXPECT_FALSE(too_large);
  if (!too_large) EXPECT_NE(too_large.error().find("ordinary dynamic budget"), std::string::npos);
}

// Small real adapters exercise Server's shared transaction, not a second
// synthetic approximation of its request and restart-record behavior.
TEST_P(GemmaPrefillServingGpu, PartialCachedSwitchReleasesRequestAndRetainsOtherWeights) {
  life->roles.installed = "/home/pmeenan/.local/share/jitllm/m3-artifacts";
  auto other = life->config.models.front();
  other.name = "other";
  other.artifact = GetParam() == 2
                       ? "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb"
                       : "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870";
  other.overrides["context"] = std::int64_t{4096};
  life->config.models.push_back(other);
  life->options.partial_weight_eviction = true;
  life->options.diagnostic_budget_cap_bytes = std::uint64_t{16} << 30U;
  ASSERT_TRUE(Start(true));
  ASSERT_TRUE(first_activation.diagnostic.eviction_split_available);  // empty-victim Swap
  auto* peer = dynamic_cast<rt::Llm*>(life->server->Find("other"));
  ASSERT_NE(peer, nullptr);
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
  std::vector<float> expected;
  ASSERT_TRUE((*model->branch(0))->Prefill(prompt, expected));
  const auto first_weight_tick = model->weight_tick();
  std::uint64_t incoming_bytes = 0;
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        for (const auto extent : peer->weights())
          incoming_bytes +=
              life->server->node().catalog().Describe(extent)->descriptor.size.value();
        return {};
      },
      "binding the zero-victim fixture's actual missing weight size"));
  const auto free = life->server->node().FreeBytes();
  ASSERT_TRUE(free);
  ASSERT_GE(*free, incoming_bytes);
  rt::SwapParts outgoing;
  ASSERT_TRUE(life->server->Activate(*peer, outgoing));
  ASSERT_TRUE(outgoing.diagnostic.eviction_split_available);
  EXPECT_LE(outgoing.diagnostic.requested_ns, outgoing.diagnostic.scheduler_started_ns);
  EXPECT_LE(outgoing.diagnostic.scheduler_started_ns, outgoing.diagnostic.scheduler_evicted_ns);
  EXPECT_LE(outgoing.diagnostic.scheduler_evicted_ns, outgoing.diagnostic.scheduler_loaded_ns);
  EXPECT_FALSE(outgoing.diagnostic.creates_loaded.timing.enabled);
  EXPECT_EQ(outgoing.diagnostic.creates_loaded.timing.create_serial, 0U);
  EXPECT_EQ(outgoing.diagnostic.creates_loaded.timing.park.completed, 0U);
  EXPECT_EQ(outgoing.diagnostic.creates_loaded.timing.park.total_ns, 0U);
  EXPECT_EQ(outgoing.diagnostic.creates_loaded.timing.park_metadata.completed +
                outgoing.diagnostic.creates_loaded.timing.park_stash.completed +
                outgoing.diagnostic.creates_loaded.timing.park_publication.completed,
            0U);
  EXPECT_GT(outgoing.diagnostic.eviction_prepare_seconds, 0);
  EXPECT_GE(outgoing.diagnostic.eviction_retire_seconds, 0);
  EXPECT_NEAR(
      outgoing.diagnostic.eviction_prepare_seconds + outgoing.diagnostic.eviction_retire_seconds,
      outgoing.evict, 1e-9);
  ASSERT_FALSE(life->server->node().InRequest(model->paged().stream()));
  EXPECT_GT(model->weight_tick(), first_weight_tick);
  EXPECT_EQ(outgoing.evicted_weight_bytes, 0U);
  EXPECT_GT(outgoing.retained_outgoing_weight_bytes, 0U);
  ASSERT_TRUE(life->server->FinishSwap(outgoing));
  const std::array<rt::Llm::Branch*, 1> one{*peer->branch(0)};
  ASSERT_TRUE(life->server->SelectRequestBranches(*peer, one));
  std::vector<float> peer_head;
  ASSERT_TRUE(one[0]->Prefill(prompt, peer_head));
  rt::SwapParts incoming;
  ASSERT_TRUE(life->server->Activate(*model, incoming));
  ASSERT_TRUE(incoming.diagnostic.eviction_split_available);
  EXPECT_NEAR(
      incoming.diagnostic.eviction_prepare_seconds + incoming.diagnostic.eviction_retire_seconds,
      incoming.evict, 1e-9);
  ASSERT_FALSE(life->server->node().InRequest(peer->paged().stream()));
  EXPECT_EQ(incoming.evicted_weight_bytes, 0U);
  EXPECT_GT(incoming.retained_incoming_weight_bytes, 0U);
  EXPECT_GT(incoming.retained_outgoing_weight_bytes, 0U);
  ASSERT_TRUE(life->server->FinishSwap(incoming));
  const std::array<rt::Llm::Branch*, 1> resumed{*model->branch(0)};
  ASSERT_TRUE(life->server->SelectRequestBranches(*model, resumed));
  ASSERT_TRUE(resumed[0]->Clear());
  std::vector<float> actual;
  ASSERT_TRUE(resumed[0]->Prefill(prompt, actual));
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
  EXPECT_TRUE(life->server->RetireRequestBranches(*model, true).references_retired);
  model->DropPlans();
  peer->DropPlans();
  auto protected_kept = model->kept_state();
  const auto peer_kept = peer->kept_state();
  protected_kept.insert(protected_kept.end(), peer_kept.begin(), peer_kept.end());
  EXPECT_EQ(life->server->Reclaim(2U << 20U, false, "optional cache eligibility control", nullptr,
                                  std::nullopt, false, nullptr, 0, protected_kept, nullptr, false),
            0U);
  // The same catalog population is eligible for a genuine load need.
  EXPECT_EQ(life->server->Reclaim(2U << 20U, false, "materialization eligibility control", nullptr,
                                  std::nullopt, false, nullptr, 0, protected_kept),
            2U << 20U);
}

TEST_P(GemmaPrefillServingGpu, FailedPartialLoadPreservesCachesAndRecreatesSavedRecord) {
  life->roles.installed = "/home/pmeenan/.local/share/jitllm/m3-artifacts";
  auto other = life->config.models.front();
  other.name = "other";
  other.artifact = GetParam() == 2
                       ? "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb"
                       : "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870";
  other.overrides["context"] = std::int64_t{4096};
  life->config.models.push_back(other);
  auto inactive_entry = other;
  inactive_entry.name = "inactive";
  inactive_entry.artifact = "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3";
  life->config.models.push_back(inactive_entry);
  life->options.partial_weight_eviction = true;
  life->options.keep_conversations = true;
  life->config.memory.keep_across_restart = true;
  ASSERT_TRUE(Start(true));
  auto* peer = dynamic_cast<rt::Llm*>(life->server->Find("other"));
  ASSERT_NE(peer, nullptr);
  ASSERT_TRUE(peer->keeps());
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
  std::vector<float> expected;
  ASSERT_TRUE((*model->branch(0))->Prefill(prompt, expected));
  rt::SwapParts swap;
  auto* inactive = life->server->Find("inactive");
  ASSERT_NE(inactive, nullptr);
  ASSERT_TRUE(life->server->Activate(*inactive, swap));
  ASSERT_TRUE(life->server->FinishSwap(swap));
  ASSERT_TRUE(life->server->Activate(*peer, swap));
  ASSERT_TRUE(life->server->FinishSwap(swap));
  const std::array<rt::Llm::Branch*, 1> selected{*peer->branch(0)};
  ASSERT_TRUE(life->server->SelectRequestBranches(*peer, selected));
  std::vector<float> peer_head;
  ASSERT_TRUE(selected[0]->Prefill(prompt, peer_head));
  ASSERT_TRUE(life->server->Activate(*model, swap));
  ASSERT_TRUE(life->server->FinishSwap(swap));
  ASSERT_TRUE(life->server->DrainKept(rt::Clock::now() + std::chrono::seconds(10)));
  std::filesystem::path record;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(life->roles.spill))
    if (entry.path().filename() == "slot-0.record" &&
        entry.path().parent_path().filename() == *other.artifact)
      record = entry.path();
  ASSERT_FALSE(record.empty());
  const auto read = [&]() {
    std::ifstream file(record, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  };
  const auto before_record = read();
  ASSERT_FALSE(before_record.empty());
  const auto weights = peer->weights();
  ASSERT_GT(weights.size(), 1U);
  std::vector<jitllm::catalog::ExtentId> inactive_cached;
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        for (const auto extent : inactive->weights())
          if (life->server->node().catalog().Describe(extent)->state ==
              jitllm::catalog::ExtentState::kResident)
            inactive_cached.push_back(extent);
        return {};
      },
      "snapshotting unrelated retained cache before the failed switch"));
  ASSERT_FALSE(inactive_cached.empty());
  const auto missing = weights.back();
  ASSERT_TRUE(life->server->node().Evict({missing}));
  jitllm::scheduler::PageSource original;
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        original = *life->server->node().scheduler().SourceOf(missing);
        auto bad = original;
        bad.read.offset = std::uint64_t{1} << 60U;  // a clean short-read failure
        if (!life->server->node().scheduler().SetSource(missing, bad))
          return std::unexpected("the test read failure source was refused");
        return {};
      },
      "injecting one missing weight read failure"));
  const auto failed = life->server->Activate(*peer, swap);
  EXPECT_FALSE(failed);
  EXPECT_EQ(life->server->resident(), model);
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        for (std::size_t i = 0; i + 1 < weights.size(); ++i)
          EXPECT_EQ(life->server->node().catalog().Describe(weights[i])->state,
                    jitllm::catalog::ExtentState::kResident);
        for (const auto extent : model->weights())
          EXPECT_EQ(life->server->node().catalog().Describe(extent)->state,
                    jitllm::catalog::ExtentState::kResident);
        for (const auto extent : inactive_cached)
          EXPECT_EQ(life->server->node().catalog().Describe(extent)->state,
                    jitllm::catalog::ExtentState::kResident);
        if (!life->server->node().scheduler().SetSource(missing, original))
          return std::unexpected("the original weight source could not be restored");
        return {};
      },
      "checking rollback cache preservation and repairing the injected source"));
  ASSERT_TRUE(life->server->DrainKept(rt::Clock::now() + std::chrono::seconds(10)));
  EXPECT_EQ(read(), before_record);  // Unkeep was followed by proven saved generations.
  const std::array<rt::Llm::Branch*, 1> resumed{*model->branch(0)};
  ASSERT_TRUE(life->server->SelectRequestBranches(*model, resumed));
  ASSERT_TRUE(resumed[0]->Clear());
  std::vector<float> actual;
  ASSERT_TRUE(resumed[0]->Prefill(prompt, actual));
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
  EXPECT_TRUE(life->server->RetireRequestBranches(*model, true).references_retired);
  // A later successful return uses the repaired missing source and retained cache.
  ASSERT_TRUE(life->server->Activate(*peer, swap));
  EXPECT_GT(swap.retained_incoming_weight_bytes, 0U);
  ASSERT_TRUE(life->server->FinishSwap(swap));
  ASSERT_TRUE(life->server->Activate(*model, swap));
  ASSERT_TRUE(life->server->FinishSwap(swap));
  ASSERT_TRUE(life->server->DrainKept(rt::Clock::now() + std::chrono::seconds(10)));
  const auto before_late_record = read();
  ASSERT_FALSE(before_late_record.empty());
  const auto unpinned = weights.front();
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        life->server->node().scheduler().UnpinPlaces(std::span(&unpinned, 1));
        if (life->server->node().scheduler().PlacePinned(unpinned))
          return std::unexpected("the test weight still has another place pin");
        return {};
      },
      "injecting a completed-transfer readiness failure"));
  const auto late_failed = life->server->Activate(*peer, swap);
  EXPECT_FALSE(late_failed);
  EXPECT_EQ(life->server->resident(), model);
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        if (!life->server->node().scheduler().PinPlaces(std::span(&unpinned, 1)))
          return std::unexpected("the readiness test's place pin could not be restored");
        return {};
      },
      "repairing the readiness-only failure"));
  ASSERT_TRUE(life->server->DrainKept(rt::Clock::now() + std::chrono::seconds(10)));
  EXPECT_EQ(read(), before_late_record);
  const auto outgoing_record =
      record.parent_path().parent_path() / *life->config.models.front().artifact / "slot-0.record";
  EXPECT_FALSE(std::filesystem::exists(outgoing_record));  // outgoing is live again
  ASSERT_TRUE(life->server->Activate(*peer, swap));
  EXPECT_GT(swap.restore, 0);  // incoming's spilled membership survived late rollback
  ASSERT_TRUE(life->server->FinishSwap(swap));
  ASSERT_TRUE(life->server->Activate(*model, swap));
  ASSERT_TRUE(life->server->FinishSwap(swap));
  const auto hash_state = [&]() {
    if (GetParam() == 2) return StateHash(dynamic_cast<en::Gemma2Runner&>(model->paged()), 0);
    return StateHash(dynamic_cast<en::Gemma3Runner&>(model->paged()), 0);
  };
  const auto saved_hash = hash_state();
  ASSERT_TRUE(saved_hash);
  // Two independent failures: the incoming readiness check, followed by
  // a missing outgoing weight's short read during undo. No request runs
  // against either state between their completed spill and restoration.
  const auto outgoing_missing = model->weights().back();
  ASSERT_TRUE(life->server->node().Evict({outgoing_missing}));
  jitllm::scheduler::PageSource outgoing_source;
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        outgoing_source = *life->server->node().scheduler().SourceOf(outgoing_missing);
        auto bad = outgoing_source;
        bad.read.offset = std::uint64_t{1} << 60U;
        if (!life->server->node().scheduler().SetSource(outgoing_missing, bad))
          return std::unexpected("the undo read failure source was refused");
        life->server->node().scheduler().UnpinPlaces(std::span(&unpinned, 1));
        return {};
      },
      "injecting independent late readiness and undo-read failures"));
  const auto twice_failed = life->server->Activate(*peer, swap);
  EXPECT_FALSE(twice_failed);
  EXPECT_EQ(life->server->resident(), nullptr);
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        if (!life->server->node().scheduler().SetSource(outgoing_missing, outgoing_source) ||
            !life->server->node().scheduler().PinPlaces(std::span(&unpinned, 1)))
          return std::unexpected("the double-failure sources could not be repaired");
        return {};
      },
      "repairing both completed failure controls"));
  ASSERT_TRUE(life->server->DrainKept(rt::Clock::now() + std::chrono::seconds(10)));
  EXPECT_TRUE(std::filesystem::exists(outgoing_record));
  EXPECT_FALSE(read().empty());
  ASSERT_TRUE(life->server->Activate(*model, swap));
  EXPECT_GT(swap.restore, 0);
  ASSERT_TRUE(life->server->FinishSwap(swap));
  const auto restored_hash = hash_state();
  ASSERT_TRUE(restored_hash);
  EXPECT_EQ(*restored_hash, *saved_hash);
}

INSTANTIATE_TEST_SUITE_P(ApprovedFamilies, GemmaPrefillServingGpu, ::testing::Values(2U, 3U));
