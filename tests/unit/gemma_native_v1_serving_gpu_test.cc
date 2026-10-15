// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Published native Gemma 3/4 artifacts against the pinned v0 imports, through
// the same component engine and settings. This is a short acceptance screen,
// not a migration performance gate or permission for another artifact family.
// Both arms are also asserted against the frozen digests under
// tests/unit/data/v1_frozen (tests/support/v1_frozen.h), so the v1 proof
// outlives the v0 artifacts: without the v0 artifact (or with
// LLMP_TEST_V1_FROZEN_ONLY set) only the v1 arm runs, against them.
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "runtime/engine_llm.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace cfg = llmp::config;
namespace en = llmp::engine;
namespace rt = llmp::runtime;

constexpr std::uint64_t kResultBytes = 32ULL << 20U;
constexpr std::uint64_t kSpillCacheBytes = 4ULL << 30U;
constexpr std::uint64_t kExecutionCap = 32ULL << 30U;
// Two arms * two owners * (prefill + four generated rows + continuation).
static_assert(24ULL * 262208 * sizeof(float) + (2ULL << 20U) <= kResultBytes);

struct Profile {
  const char* name;
  const char* legacy_store;
  const char* legacy_id;
  const char* native_id_env;
  const char* native_store_env;
  std::uint32_t vocab;
  std::uint32_t prefill;
  bool joined_prefill;
};
constexpr std::array<Profile, 3> kProfiles{{
    {"Gemma3", "gemma3-import-20261007/artifacts",
     "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb", "LLMP_TEST_GEMMA3_V1_ID",
     "LLMP_TEST_GEMMA3_V1_STORE", 262208, 128, true},
    {"Gemma26", "m3-artifacts", "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3",
     "LLMP_TEST_GEMMA26_V1_ID", "LLMP_TEST_GEMMA26_V1_STORE", 262144, 1024, false},
    {"Gemma31", "m3-artifacts", "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08",
     "LLMP_TEST_GEMMA31_V1_ID", "LLMP_TEST_GEMMA31_V1_STORE", 262144, 256, false},
}};

bool Finite(std::span<const float> row, std::size_t width) {
  return row.size() == width && std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
}
bool Same(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}
std::vector<std::int32_t> Tokens(std::uint32_t owner, std::size_t rows) {
  constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
  std::vector<std::int32_t> out(rows);
  for (std::size_t i = 0; i < rows; ++i) out[i] = seed[(i + owner * 3 + i / 7) % seed.size()];
  return out;
}

class GemmaNativeV1ServingGpu : public ::testing::TestWithParam<std::size_t> {
 protected:
  struct State {
    std::vector<std::array<std::uint64_t, 3>> ranges;
    llmp::base::Sha256Digest digest{};
    std::uint64_t bytes = 0;
    std::uint32_t positions = 0;
    bool operator==(const State&) const = default;
  };
  struct Result {
    std::array<std::vector<float>, 2> heads, continued;
    std::array<rt::Generation, 2> generated;
    std::array<std::vector<std::int32_t>, 2> histories, continued_histories;
    std::array<State, 2> state, restored, resumed;
  };
  struct CloseFile {
    void operator()(std::FILE* file) const {
      if (file != nullptr) (void)std::fclose(file);
    }
  };
  struct Lifetime {
    // Declaration order keeps log/config/options alive through server destruction,
    // and destroys retained observations before their last startup payment.
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<std::FILE, CloseFile> log;
    fs::path scratch, legacy_store, native_store;
    std::string native_id;
    std::unique_ptr<rt::Server> retired_server, server;
    std::array<Result, 2> result;
    rt::GenerateOptions generate;
    std::array<std::vector<std::int32_t>, 2> prompts;
    std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> generations;
    std::array<std::vector<std::int32_t>, 2> continuation;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  rt::Llm* model = nullptr;
  bool uncertain = false;
  const Profile& profile() const { return kProfiles.at(GetParam()); }
  bool have_v0 = true;
  rt::Llm::Branch& Branch(std::uint32_t id) { return **model->branch(id); }
  rt::EngineLlm& Engine() { return dynamic_cast<rt::EngineLlm&>(*model); }

  void SetUp() override {
    const char* store = std::getenv(profile().native_store_env);  // NOLINT(concurrency-mt-unsafe)
    if (store == nullptr)
      store = std::getenv("LLMP_TEST_GEMMA_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
    if (store == nullptr)
      store = std::getenv("LLMP_TEST_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
    const char* id = std::getenv(profile().native_id_env);  // NOLINT(concurrency-mt-unsafe)
    std::string mapped_id;
    if (id == nullptr) {
      // The store's v1-ids.json names the v1 ID of each pinned v0 artifact.
      if (const auto mapped = llmp::test_support::V1IdFor(profile().legacy_id); mapped) {
        mapped_id = *mapped;
        id = mapped_id.c_str();
      }
    }
    if (store == nullptr || id == nullptr) GTEST_SKIP() << "published native store/ID not supplied";
    life->native_id = id;
    ASSERT_EQ(life->native_id.size(), 64U);
    ASSERT_TRUE(std::ranges::all_of(
        life->native_id, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
    life->native_store = store;
    life->legacy_store = fs::path(llmp::test_support::ModelsDir()) / profile().legacy_store;
    std::error_code error;
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    have_v0 = std::getenv("LLMP_TEST_V1_FROZEN_ONLY") == nullptr &&
              fs::is_directory(life->legacy_store / profile().legacy_id, error);
    ASSERT_TRUE(fs::is_directory(life->native_store / life->native_id, error));
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path root = base != nullptr ? base : ::testing::TempDir();
    fs::create_directories(root, error);
    ASSERT_FALSE(error);
    std::string name = (root / "gemma-native-v1-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    life->scratch = name;
    life->roles.spill = life->scratch / "spill";
    life->roles.state = life->scratch / "state";
    ClearStorage();
    ASSERT_FALSE(HasFatalFailure());
    cfg::ModelEntry entry;
    entry.name = "gemma";
    entry.overrides["context"] = std::int64_t{4096};
    entry.overrides["max_slots"] = std::int64_t{2};
    // In particular, keep the approved 26B/31B default prefill recipes.
    life->config.models.push_back(std::move(entry));
    life->options.diagnostic_budget_cap_bytes = kExecutionCap;
    // Both arms' full rows and a conservative spill page-cache bound are
    // paid before Start opens either artifact. There is no external parse.
    // Normal runtime admission still pays compiler, device and pinned copies.
    life->options.diagnostic_startup_host_bytes = kResultBytes + kSpillCacheBytes;
    life->generate.max_tokens = 4;
    life->generate.stop = false;
    life->generate.keep_logits = true;
    life->log.reset(std::tmpfile());
    ASSERT_NE(life->log, nullptr);
  }
  void ClearStorage() {
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      fs::remove_all(path);
      ASSERT_TRUE(fs::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
  }
  rt::Status FinishSessions() {
    for (auto& session : life->sessions) {
      if (!session) continue;
      if (!session->done()) session->Cancel();
      if (auto finished = session->Finish(); !finished) {
        uncertain = true;
        return finished;
      }
      session.reset();
    }
    for (auto& session : life->generations) {
      if (!session) continue;
      if (!session->done()) session->Cancel();
      if (auto finished = session->Finish(); !finished) {
        uncertain = true;
        return finished;
      }
      session.reset();
    }
    return {};
  }
  void TearDown() override {
    if (!uncertain) {
      const auto finished = FinishSessions();
      EXPECT_TRUE(finished) << (finished ? "" : finished.error());
    }
    if (!uncertain && life->server) {
      const auto retired = life->server->TearDown();
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      uncertain = !retired;
    }
    if (uncertain) {
      // Assertions and failed setup must not destroy a submitted operation's
      // buffers, prompt sessions, state, log, catalog or funding owners.
      std::ignore = life.release();
      return;
    }
    const fs::path scratch = life->scratch;
    life.reset();
    if (!scratch.empty()) fs::remove_all(scratch);
  }
  rt::Status Select() {
    std::array<rt::Llm::Branch*, 2> branches{};
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      auto branch = model->branch(owner);
      if (!branch) return std::unexpected(branch.error());
      branches[owner] = *branch;
    }
    return life->server->SelectRequestBranches(*model, branches);
  }
  rt::Status Start(std::size_t arm) {
    life->roles.installed = arm == 0 ? life->legacy_store : life->native_store;
    life->config.models.front().artifact = arm == 0 ? profile().legacy_id : life->native_id;
    life->server =
        std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log.get());
    if (auto started = life->server->Start(true); !started) return started;
    // Transfer payment only after the new node has admitted the whole allowance.
    life->retired_server.reset();
    model = dynamic_cast<rt::Llm*>(life->server->Find("gemma"));
    if (model == nullptr || dynamic_cast<rt::EngineLlm*>(model) == nullptr)
      return std::unexpected("published artifact did not use the common EngineLlm");
    if (model->logit_width() != profile().vocab ||
        model->settings().prefill_chunk.value != profile().prefill ||
        model->settings().context.value != 4096 || model->settings().max_slots.value != 2)
      return std::unexpected("published artifact changed the approved execution recipe");
    rt::SwapParts activation;
    if (auto active = life->server->Activate(*model, activation); !active) return active;
    const auto state_bound = model->StateBytesThrough(4096);
    if (state_bound == 0 || state_bound > kSpillCacheBytes / 2)
      return std::unexpected("two owners exceed the prepaid spill-cache envelope");
    return Select();
  }
  std::uint64_t ExecutedSteps() {
    const auto& stats = Engine().runner().graph_stats();
    return stats.eager + stats.captured + stats.replayed;
  }
  rt::Status Prompts(Result& out) {
    // Preserve the approved policy: Gemma3 joins prompts; Gemma4 requires
    // prompts alone. Both must later execute actual joined decode steps.
    if (model->prefill_wave_capacity() != (profile().joined_prefill ? 2U : 1U))
      return std::unexpected("approved prompt wave capacity changed");
    const auto before = Engine().counts();
    const auto executed = ExecutedSteps();
    life->prompts = {Tokens(0, 7), Tokens(1, 7)};
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      if (auto cleared = Branch(owner).Clear(); !cleared) return cleared;
      auto begun = Branch(owner).BeginPrompt(life->prompts[owner]);
      if (!begun) return std::unexpected(begun.error());
      life->sessions[owner] = std::move(*begun);
      if (auto advanced = life->sessions[owner]->Advance(); !advanced) {
        uncertain = true;
        return advanced;
      }
    }
    while (!life->sessions[0]->done() || !life->sessions[1]->done()) {
      std::array<rt::Llm::PromptSession*, 2> active{};
      std::size_t count = 0;
      for (const auto owner : {1U, 0U}) {
        auto* session = life->sessions[owner].get();
        if (!session->done() && (count == 0 || active[0]->CanJoin(*session)))
          active[count++] = session;
      }
      const std::array<rt::PrefillGoOn, 2> callbacks{};
      if (auto run = model->RunPromptWave(std::span(active).first(count),
                                          std::span(callbacks).first(count));
          !run) {
        uncertain = true;
        return run;
      }
      for (const auto* session : std::span(active).first(count))
        if (!session->last_unit_result()) {
          uncertain = true;
          return session->last_unit_result();
        }
    }
    const auto after = Engine().counts();
    const auto joined = profile().joined_prefill ? 1U : 0U;
    if (after.joined_prefill_steps != before.joined_prefill_steps + joined ||
        after.joined_prefill_rows != before.joined_prefill_rows + 14U * joined ||
        ExecutedSteps() != executed + (profile().joined_prefill ? 1U : 2U))
      return std::unexpected("the two prompts did not execute their approved batching recipe");
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      out.heads[owner] = life->sessions[owner]->last();
      if (Branch(owner).history() != life->prompts[owner])
        return std::unexpected("prefill history differs");
    }
    return FinishSessions();
  }
  rt::Status GenerateJoined(Result& out) {
    if (!model->supports_generation_waves() || model->generation_wave_capacity() != 2)
      return std::unexpected("approved joined generation capacity changed");
    const auto before = Engine().counts();
    const auto executed = ExecutedSteps();
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      auto begun =
          Branch(owner).BeginGeneration(out.heads[owner], life->generate, out.generated[owner]);
      if (!begun) return std::unexpected(begun.error());
      life->generations[owner] = std::move(*begun);
    }
    // The first token comes from prefill. Exactly three completed native
    // two-owner steps produce the remaining three tokens for each owner.
    // A wave can report per-owner failure through its session rather than its
    // return status. Retain the complete owner until both sessions settle.
    uncertain = true;
    for (std::uint32_t wave = 0; wave < 3; ++wave) {
      for (const auto& session : life->generations)
        if (session->done()) return std::unexpected("joined generation ended before three steps");
      const std::array<rt::Llm::GenerationSession*, 2> both{life->generations[1].get(),
                                                            life->generations[0].get()};
      if (auto run = model->RunGenerationWave(both); !run) return run;
      for (const auto& result : out.generated)
        if (result.tokens.size() != wave + 2 || result.steps != wave + 1)
          return std::unexpected("joined generation did not complete both owners' exact unit");
    }
    for (const auto& session : life->generations)
      if (!session->done()) return std::unexpected("joined generation exceeded four tokens");
    const auto after = Engine().counts();
    if (after.joined_waves != before.joined_waves + 3 ||
        after.joined_units != before.joined_units + 6 || ExecutedSteps() != executed + 3)
      return std::unexpected("three two-owner requests were not three actual joined decode steps");
    if (auto finished = FinishSessions(); !finished) return finished;
    uncertain = false;
    return {};
  }
  rt::Status Continue(std::uint32_t owner, Result& out) {
    auto& prompt = life->continuation[owner];
    prompt = out.histories[owner];
    prompt.push_back(out.generated[owner].tokens.back());
    auto begun = Branch(owner).BeginPrompt(prompt);
    if (!begun) return std::unexpected(begun.error());
    auto& session = life->sessions[owner];
    session = std::move(*begun);
    while (!session->done()) {
      if (auto run = session->Advance(); !run) {
        uncertain = true;
        return run;
      }
    }
    if (session->reused() != out.histories[owner].size())
      return std::unexpected("continuation recomputed history");
    out.continued[owner] = session->last();
    return FinishSessions();
  }
  std::expected<State, std::string> StateImage(std::uint32_t owner) {
    auto& slots = Engine().runner().slots();
    auto& node = life->server->node();
    State state;
    state.positions = slots.positions(owner);
    // Record every initialized page's coordinates, including initialized tails.
    // Metadata is inside the prepaid 2 MiB slack; no complete payload copy.
    for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
      for (const auto& range : slots.live(owner, stage).used_ranges()) {
        if (state.ranges.size() == 1024 || range.bytes > kSpillCacheBytes - state.bytes)
          return std::unexpected("initialized state exceeds the fixed observation domain");
        state.ranges.push_back({stage, range.offset, range.bytes});
        state.bytes += range.bytes;
      }
    }
    if (state.bytes == 0) return std::unexpected("no initialized state");
    constexpr std::uint64_t kPart = 1ULL << 20U;
    std::vector<llmp::catalog::ExtentId> staging;
    auto buffer = node.Pinned(kPart, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    llmp::base::Sha256 hash;
    for (const auto& range : state.ranges) {
      for (std::uint64_t at = 0; at < range[2]; at += kPart) {
        const en::LiveState::Range part{
            .region = range[0], .offset = range[1] + at, .bytes = std::min(kPart, range[2] - at)};
        en::LiveState::CopyRetirement retired = en::LiveState::CopyRetirement::kUnproven;
        const auto copied = slots.CopyState(owner, *buffer, std::span(&part, 1), true, &retired);
        if (retired == en::LiveState::CopyRetirement::kUnproven) {
          node.KeepPinned(*buffer);
          uncertain = true;
          return std::unexpected(copied ? "state copy retirement is unproven" : copied.error());
        }
        if (!copied) {
          (void)node.FreePinned(*buffer);
          return std::unexpected(copied.error());
        }
        hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
      }
    }
    if (auto freed = node.FreePinned(*buffer); !freed) return std::unexpected(freed.error());
    state.digest = hash.Finish();
    return state;
  }
  rt::Status Restore(std::uint32_t owner, const std::vector<std::int32_t>& history) {
    const auto before = model->graphs();
    auto begun = Branch(owner).BeginPrompt(history, 0, false, true);
    if (!begun) return std::unexpected(begun.error());
    auto& session = life->sessions[owner];
    session = std::move(*begun);
    while (!session->done()) {
      if (auto run = session->Advance(); !run) {
        uncertain = true;
        return run;
      }
    }
    if (session->reused() != history.size())
      return std::unexpected("restore recomputed prompt tokens");
    if (auto finished = FinishSessions(); !finished) return finished;
    const auto after = model->graphs();
    if (after.eager != before.eager || after.captured != before.captured ||
        after.replayed != before.replayed)
      return std::unexpected("restore executed a model step");
    if (Branch(owner).spilled() || Branch(owner).history() != history)
      return std::unexpected("restore changed history or left spilled state");
    return {};
  }
};

TEST_P(GemmaNativeV1ServingGpu, PublishedMatchesV0FullRowsStateAndSpillContinuation) {
  for (std::size_t arm = 0; arm < life->result.size(); ++arm) {
    if (arm == 0 && !have_v0) continue;
    SCOPED_TRACE(arm);
    const auto started = Start(arm);
    ASSERT_TRUE(started) << (started ? "" : started.error());
    auto& out = life->result[arm];
    const auto prompted = Prompts(out);
    ASSERT_TRUE(prompted) << (prompted ? "" : prompted.error());
    const auto generated = GenerateJoined(out);
    ASSERT_TRUE(generated) << (generated ? "" : generated.error());
    std::uint64_t spill_bytes = 0;
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      ASSERT_EQ(out.generated[owner].tokens.size(), 4U);
      ASSERT_EQ(out.generated[owner].logits.size(), 4U);
      ASSERT_EQ(out.generated[owner].steps, 3U);
      out.histories[owner] = Branch(owner).history();
      auto state = StateImage(owner);
      ASSERT_TRUE(state) << (state ? "" : state.error());
      out.state[owner] = std::move(*state);
      // Bound all bytes the spill can preserve before issuing either write.
      const auto bytes = model->StateBytes(Branch(owner));
      ASSERT_LE(bytes, kSpillCacheBytes - spill_bytes);
      ASSERT_GE(bytes, out.state[owner].bytes);
      spill_bytes += bytes;
    }
    ASSERT_TRUE(life->server->EndRequestBranches(*model));
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      const auto spilled = model->SpillIdle(Branch(owner));
      ASSERT_TRUE(spilled) << (spilled ? "" : spilled.error());
      ASSERT_TRUE(Branch(owner).spilled());
    }
    ASSERT_TRUE(Select());
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      const auto restored = Restore(owner, out.histories[owner]);
      ASSERT_TRUE(restored) << (restored ? "" : restored.error());
      auto state = StateImage(owner);
      ASSERT_TRUE(state) << (state ? "" : state.error());
      out.restored[owner] = std::move(*state);
      ASSERT_EQ(out.restored[owner], out.state[owner]);
      const auto continued = Continue(owner, out);
      ASSERT_TRUE(continued) << (continued ? "" : continued.error());
      out.continued_histories[owner] = Branch(owner).history();
      auto expected = out.histories[owner];
      expected.push_back(out.generated[owner].tokens.back());
      ASSERT_EQ(out.continued_histories[owner], expected);
      state = StateImage(owner);
      ASSERT_TRUE(state) << (state ? "" : state.error());
      out.resumed[owner] = std::move(*state);
    }
    EXPECT_EQ(Engine().runner().coverage().violations, 0U);
    ASSERT_TRUE(life->server->EndRequestBranches(*model));
    if (arm == 0) {
      const auto stopped = life->server->TearDown();
      uncertain = !stopped;
      ASSERT_TRUE(stopped) << (stopped ? "" : stopped.error());
      life->retired_server = std::move(life->server);
      model = nullptr;
      ClearStorage();
      ASSERT_FALSE(HasFatalFailure());
    }
  }
  // The frozen digests: both arms (the v0 one while it exists) against what
  // the pinned imports produced when they were recorded.
  llmp::test_support::FrozenDigests frozen(std::string("gemma_") + profile().name);
  for (std::size_t arm = have_v0 ? 0 : 1; arm < life->result.size(); ++arm) {
    SCOPED_TRACE(arm == 0 ? "v0 against frozen" : "v1 against frozen");
    const auto& r = life->result[arm];
    const auto state_text = [](const State& state) {
      return llmp::base::ToHex(state.digest) + ':' + std::to_string(state.bytes) + ':' +
             std::to_string(state.positions);
    };
    for (std::uint32_t owner = 0; owner < 2; ++owner) {
      const auto o = std::to_string(owner);
      using llmp::test_support::HexOfFloats;
      using llmp::test_support::HexOfInts;
      using llmp::test_support::HexOfRows;
      frozen.Expect("heads." + o, HexOfFloats(r.heads[owner]));
      frozen.Expect("tokens." + o, llmp::test_support::JoinTokens(r.generated[owner].tokens));
      frozen.Expect("logits." + o, HexOfRows(r.generated[owner].logits));
      frozen.Expect("history." + o, HexOfInts(r.histories[owner]));
      frozen.Expect("state." + o, state_text(r.state[owner]));
      frozen.Expect("restored." + o, state_text(r.restored[owner]));
      frozen.Expect("continued." + o, HexOfFloats(r.continued[owner]));
      frozen.Expect("resumed." + o, state_text(r.resumed[owner]));
    }
    if (frozen.recording()) break;  // one arm records
  }
  if (!have_v0) return;
  const auto& a = life->result[0];
  const auto& b = life->result[1];
  for (std::uint32_t owner = 0; owner < 2; ++owner) {
    for (const auto& arm : life->result) {
      ASSERT_TRUE(Finite(arm.heads[owner], profile().vocab));
      ASSERT_TRUE(Finite(arm.continued[owner], profile().vocab));
      for (const auto& row : arm.generated[owner].logits) ASSERT_TRUE(Finite(row, profile().vocab));
    }
    EXPECT_TRUE(Same(a.heads[owner], b.heads[owner]));
    EXPECT_TRUE(Same(a.continued[owner], b.continued[owner]));
    EXPECT_EQ(a.generated[owner].tokens, b.generated[owner].tokens);
    EXPECT_EQ(a.generated[owner].steps, b.generated[owner].steps);
    EXPECT_EQ(a.histories[owner], b.histories[owner]);
    EXPECT_EQ(a.continued_histories[owner], b.continued_histories[owner]);
    EXPECT_EQ(a.state[owner], b.state[owner]);
    EXPECT_EQ(a.restored[owner], b.restored[owner]);
    EXPECT_EQ(a.resumed[owner], b.resumed[owner]);
    for (std::size_t row = 0; row < 4; ++row)
      EXPECT_TRUE(Same(a.generated[owner].logits[row], b.generated[owner].logits[row]));
  }
}
INSTANTIATE_TEST_SUITE_P(Approved, GemmaNativeV1ServingGpu, ::testing::Values(0U, 1U, 2U),
                         [](const auto& info) { return kProfiles.at(info.param).name; });
}  // namespace
