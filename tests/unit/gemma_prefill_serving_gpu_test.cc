// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <print>
#include <tuple>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "engine/gemma2_runner.h"
#include "engine/gemma3_runner.h"
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
  rt::Status Start(bool ahead, bool owner_prefill = true) {
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
    rt::SwapParts parts;
    if (auto r = life->server->Activate(*model, parts); !r) return r;
    std::array<rt::Llm::Branch*, 2> branches{*model->branch(0), *model->branch(1)};
    return life->server->SelectRequestBranches(*model, branches);
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
  std::expected<jitllm::base::Sha256Digest, std::string> StateHash(Runner& runner,
                                                                   std::uint32_t id) {
    const auto slot = runner.request_slot(id);
    if (!slot) return std::unexpected(slot.error());
    auto& node = life->server->node();
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
};

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

INSTANTIATE_TEST_SUITE_P(ApprovedFamilies, GemmaPrefillServingGpu, ::testing::Values(2U, 3U));
