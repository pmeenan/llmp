// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 2 and Gemma 3 served by the shared engine (D-107; runtime/
// engine_llm.h over the component step source) on the approved artifacts
// (label `models`), through the runtime's server as the service drives it:
//
// - plain generation's device tokens and logit rows agree, alone and
//   joined, through an idle spill and a restored continuation (the
//   plain-token oracle, plain_token_serving_checks.h);
// - a lone prompt's head, sixteen greedy tokens with their logit rows, a
//   continuation's head and two owners' joined prompt heads are the same
//   from one server to the next (run-to-run exactness);
// - its state: a snapshot taken and restored, an idle spill and its
//   restore, a turn checkpoint's restore, and the conversation kept across
//   a restart with its turn checkpoint, each exactly the state and
//   continuation it left;
// - two owners' joined prompts of unequal lengths across the local ring's
//   wrap, built ahead, with the state prepared ahead: heads and states
//   exact from one server to the next, and through a spill, a kept
//   restart and a restore, with the continuation's head;
// - Gemma 3 at an admitted 8,448 context: an 8,192-token prompt in 128-row
//   chunks and its continuation, exact through a spill and restore;
// - the server's swap transactions on these models: the diagnostic budget
//   cap's admission, a partial cached switch that retains the other
//   model's weights, a failed partial load that preserves caches and
//   recreates the saved record, and a state preparation the server must
//   stop on rather than destroy (Gemma 3).
//
// What the retired M3 classes' controls covered beyond these now lives in
// the shared pieces' own tests: the slot lifecycle's restore, adoption,
// selection and quarantine rules (slot_lifecycle_test), the runner's waves,
// lookahead and capture (model_runner_test, engine_llm_test) and the
// kept-record versions (kept_state_test, engine_llm_test).

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "plain_token_serving_checks.h"
#include "platform/crash_policy.h"
#include "runtime/engine_llm.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"

namespace {

namespace rt = llmp::runtime;
namespace en = llmp::engine;
namespace cfg = llmp::config;
namespace fs = std::filesystem;

constexpr const char* kGemma2 = "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870";
constexpr const char* kGemma3 = "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb";
constexpr const char* kGemma4 = "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3";

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool Finite(const std::vector<float>& row, std::size_t width) {
  return row.size() == width && std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
}

bool Present(const fs::path& path) {
  std::error_code error;
  return fs::exists(path, error);
}

// The prompt of `owner`, `rows` tokens long (a repeated seed, as the
// serving fixtures use).
std::vector<std::int32_t> Tokens(std::uint32_t owner, std::size_t rows, std::size_t salt = 0) {
  constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
  std::vector<std::int32_t> out(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    out[i] = seed[(i + (owner * 3) + (i / 7) + salt) % seed.size()];
  }
  return out;
}

class GemmaEngineServingGpu : public ::testing::TestWithParam<std::uint32_t> {
 protected:
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  rt::Llm* model = nullptr;
  fs::path scratch;
  std::FILE* log = nullptr;
  bool retirement_failed = false;
  // The server's log to the process's own stderr (a death test matches it),
  // or to a file the test reads back.
  bool log_to_stderr = false;
  rt::SwapParts first_activation;

  std::uint32_t Vocab() const { return GetParam() == 2 ? 256000U : 262208U; }
  // A ring cell count that two owners' prompts pass: the local layers'
  // ring (window and chunk, padded to 256 cells).
  std::uint32_t Ring() const { return GetParam() == 2 ? 4352U : 1280U; }
  static fs::path Models() { return llmp::test_support::ModelsDir(); }

  void SetUp() override {
    const fs::path installed = Models() / (GetParam() == 2 ? "gemma2-import-20261007/artifacts"
                                                           : "gemma3-import-20261007/artifacts");
    std::error_code error;
    if (!fs::exists(installed / (GetParam() == 2 ? kGemma2 : kGemma3), error)) {
      GTEST_SKIP() << "no Gemma" << GetParam() << " artifact in " << installed;
    }
    // Spill files need direct I/O: scratch in the build tree.
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path root = base != nullptr ? base : ::testing::TempDir();
    fs::create_directories(root, error);
    std::string name = (root / "gemma-engine-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    life->roles.installed = installed;
    life->roles.spill = scratch / "spill";
    life->roles.state = scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      ASSERT_TRUE(fs::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    cfg::ModelEntry entry;
    entry.name = "gemma";
    entry.artifact = GetParam() == 2 ? kGemma2 : kGemma3;
    entry.overrides["context"] = std::int64_t{GetParam() == 2 ? 8192 : 4096};
    entry.overrides["prefill_chunk"] = std::int64_t{128};
    entry.overrides["max_slots"] = std::int64_t{2};
    life->config.models.push_back(entry);
    log = std::tmpfile();
    ASSERT_NE(log, nullptr);
  }

  // A server with the family on the engine, `owners` request slots
  // selected.
  rt::Status Start(std::size_t owners = 2) {
    life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options,
                                                log_to_stderr ? stderr : log);
    if (auto r = life->server->Start(true); !r) return r;
    model = dynamic_cast<rt::Llm*>(life->server->Find("gemma"));
    if (model == nullptr) return std::unexpected("configured Gemma model missing");
    if (dynamic_cast<rt::EngineLlm*>(model) == nullptr) {
      return std::unexpected("the family is not on the component engine");
    }
    first_activation = {};
    if (auto r = life->server->Activate(*model, first_activation); !r) return r;
    return Select(owners);
  }
  rt::Status Select(std::size_t count) {
    std::array<rt::Llm::Branch*, 2> branches{};
    for (std::size_t id = 0; id < count; ++id) {
      auto branch = model->branch(static_cast<std::uint32_t>(id));
      if (!branch) return std::unexpected(branch.error());
      branches[id] = *branch;
    }
    return life->server->SelectRequestBranches(*model, std::span(branches).first(count));
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
    if (log != nullptr) (void)std::fclose(log);
    if (!scratch.empty()) fs::remove_all(scratch);
  }
  // The next arm starts with no kept metadata or spill backing.
  void ClearStorage() {
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      fs::remove_all(path);
      ASSERT_TRUE(fs::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
  }
  std::string Log() const {
    std::string text;
    (void)std::fflush(log);
    std::rewind(log);
    std::array<char, 4096> buffer{};
    for (std::size_t n = 0; (n = std::fread(buffer.data(), 1, buffer.size(), log)) > 0;) {
      text.append(buffer.data(), n);
    }
    return text;
  }
  rt::Llm::Branch& Branch(std::uint32_t id) { return **model->branch(id); }
  rt::EngineLlm& Engine() { return dynamic_cast<rt::EngineLlm&>(*model); }

  // Every stage's used state of `slot`, hashed (the engine's slots).
  std::expected<llmp::base::Sha256Digest, std::string> StateHash(std::uint32_t slot) {
    en::SlotLifecycle& slots = Engine().runner().slots();
    auto& node = life->server->node();
    constexpr std::uint64_t kPart = 1ULL << 20U;
    std::vector<llmp::catalog::ExtentId> staging;
    auto buffer = node.Pinned(kPart, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    llmp::base::Sha256 hash;
    std::uint64_t bytes = 0;
    for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
      for (const en::LiveState::Range& range : slots.live(slot, stage).used_ranges()) {
        for (std::uint64_t at = 0; at < range.bytes; at += kPart) {
          const en::LiveState::Range part{.region = stage,
                                          .offset = range.offset + at,
                                          .bytes = std::min(kPart, range.bytes - at)};
          en::LiveState::CopyRetirement retired = en::LiveState::CopyRetirement::kUnproven;
          const auto copied = slots.CopyState(slot, *buffer, std::span(&part, 1), true, &retired);
          if (retired == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(*buffer);
          if (!copied) {
            if (retired == en::LiveState::CopyRetirement::kProven) (void)node.FreePinned(*buffer);
            return std::unexpected(copied.error());
          }
          hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
          bytes += part.bytes;
        }
      }
    }
    if (auto r = node.FreePinned(*buffer); !r) return std::unexpected(r.error());
    if (bytes == 0) return std::unexpected("no state");
    return hash.Finish();
  }

  // Two owners' prompts of `rows` tokens in joined prompt waves (as the
  // service's driver runs them, the second owner first as independent
  // clients may arrive): each owner's head.
  rt::Status JoinedPrompts(std::array<std::uint32_t, 2> rows,
                           std::array<std::vector<float>, 2>& heads) {
    std::array<std::vector<std::int32_t>, 2> tokens{Tokens(0, rows[0]), Tokens(1, rows[1])};
    struct Sessions {
      std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> owned;
      ~Sessions() {
        for (auto& session : owned) {
          if (session) {
            session->Cancel();
            const auto r = session->Finish();
            EXPECT_TRUE(r) << (r ? "" : r.error());
          }
        }
      }
    } sessions;
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (auto r = Branch(id).Clear(); !r) return r;
      auto begun = Branch(id).BeginPrompt(tokens[id]);
      if (!begun) return std::unexpected(begun.error());
      sessions.owned[id] = std::move(*begun);
      if (auto r = sessions.owned[id]->Advance(); !r) return r;
    }
    while (!sessions.owned[0]->done() || !sessions.owned[1]->done()) {
      std::array<rt::Llm::PromptSession*, 2> active{};
      std::size_t count = 0;
      for (const auto id : {1U, 0U}) {
        if (!sessions.owned[id]->done() &&
            (count == 0 || active[0]->CanJoin(*sessions.owned[id]))) {
          active[count++] = sessions.owned[id].get();
        }
      }
      const std::array<rt::PrefillGoOn, 2> callbacks{};
      if (auto r = model->RunPromptWave(std::span(active).first(count),
                                        std::span(callbacks).first(count));
          !r) {
        return r;
      }
      for (const auto* session : std::span(active).first(count)) {
        if (!session->last_unit_result()) return session->last_unit_result();
      }
    }
    for (std::uint32_t id = 0; id < 2; ++id) {
      heads[id] = sessions.owned[id]->last();
      if (Branch(id).history() != tokens[id]) return std::unexpected("a joined history differs");
      if (auto r = sessions.owned[id]->Finish(); !r) return r;
      sessions.owned[id].reset();
    }
    return {};
  }

  // A lone owner's prompt, sixteen greedy tokens with their rows, and the
  // continuation's head.
  struct Lone {
    std::vector<float> head;
    rt::Generation generation;
    std::vector<float> continued;
  };
  rt::Status LonePrompt(Lone& out) {
    if (auto r = Select(1); !r) return r;
    if (auto r = Branch(0).Clear(); !r) return r;
    const auto tokens = Tokens(0, 1285);
    if (auto r = Branch(0).Prefill(tokens, out.head); !r) return r;
    rt::GenerateOptions options;
    options.max_tokens = 16;
    options.stop = false;
    options.keep_logits = true;
    if (auto r = Branch(0).Generate(out.head, options, out.generation); !r) return r;
    if (out.generation.tokens.size() != 16) return std::unexpected("short generation");
    const std::array<std::int32_t, 1> anchor{out.generation.tokens.back()};
    if (auto r = Branch(0).Prefill(anchor, out.continued); !r) return r;
    return Select(2);
  }

  // A restore of `branch`'s spilled history, run as the server's driver
  // does (a prompt that reuses all of it).
  static rt::Status RestoreBranch(rt::Llm::Branch& branch,
                                  const std::vector<std::int32_t>& tokens) {
    auto restoring = branch.BeginPrompt(tokens, 0, false, true);
    if (!restoring) return std::unexpected(restoring.error());
    rt::Status run;
    while (!(*restoring)->done() && (run = (*restoring)->Advance())) {
    }
    if (!run) (*restoring)->Cancel();
    const auto finished = (*restoring)->Finish();
    if (!run) return run;
    if (!finished) return finished;
    if ((*restoring)->reused() != tokens.size()) return std::unexpected("a restore reused less");
    return {};
  }

  // Two owners' prompts passing the local ring, hashed, and the
  // continuation heads they give.
  struct Wrapped {
    std::array<std::vector<float>, 2> heads, next;
    std::array<llmp::base::Sha256Digest, 2> state{};
    std::array<std::vector<std::int32_t>, 2> history;
  };
  std::array<std::uint32_t, 2> WrapRows() const { return {Ring() + 256U, Ring() + 384U}; }
  rt::Status WrappedPrompts(Wrapped& out) {
    if (auto r = JoinedPrompts(WrapRows(), out.heads); !r) return r;
    const auto& runner = Engine().runner();
    // The joined prompts were built ahead and the state prepared ahead, and
    // nothing is left pending.
    if (runner.lookahead_stats().built == 0 || runner.lookahead_stats().captured_ahead == 0) {
      return std::unexpected("no step was built and captured ahead");
    }
    if (runner.coverage().violations != 0) return std::unexpected("uncovered memory");
    if (life->server->node().has_pending_state_preparation()) {
      return std::unexpected("a state preparation is still linked");
    }
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (!Finite(out.heads[id], Vocab())) return std::unexpected("a head is incomplete");
      if (Engine().runner().slots().positions(id) != WrapRows()[id]) {
        return std::unexpected("a completed cursor differs");
      }
      out.history[id] = Branch(id).history();
      auto hashed = StateHash(id);
      if (!hashed) return std::unexpected(hashed.error());
      out.state[id] = *hashed;
    }
    return {};
  }
};

// The tests of Gemma 3 alone: a suite of its own, so that no test of another
// family skips beside them (CTest reports a test that prints a skip as
// skipped whatever else it printed).
class GemmaEngineGemma3Gpu : public GemmaEngineServingGpu {};

TEST_P(GemmaEngineServingGpu, PlainTokensAndRowsAgreeOnTheEngine) {
  const auto started = Start();
  ASSERT_TRUE(started) << (started ? "" : started.error());
  EXPECT_EQ(model->logit_width(), Vocab());
  const auto r = llmp::test::CheckPlainServing(
      *life->server, *model, Vocab(), [&](std::uint32_t id) { return StateHash(id); },
      [&] { return Engine().counts().greedy_tokens; });
  ASSERT_TRUE(r) << (r ? "" : r.error());
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

TEST_P(GemmaEngineServingGpu, HeadsTokensAndRowsAreTheSameFromOneServerToTheNext) {
  Lone want, got;
  std::array<std::vector<float>, 2> want_joined, got_joined;
  for (const bool second : {false, true}) {
    const auto started = Start();
    ASSERT_TRUE(started) << (started ? "" : started.error());
    Lone& lone = second ? got : want;
    auto& joined = second ? got_joined : want_joined;
    const auto ran = LonePrompt(lone);
    ASSERT_TRUE(ran) << (ran ? "" : ran.error());
    const auto waves = JoinedPrompts({1285, 1413}, joined);
    ASSERT_TRUE(waves) << (waves ? "" : waves.error());
    std::println("GEMMA_ENGINE_CLASS family={} extra={}", GetParam(), model->extra());
    ASSERT_TRUE(Retire());
  }
  ASSERT_TRUE(Finite(want.head, Vocab()));
  EXPECT_TRUE(Same(got.head, want.head));
  EXPECT_EQ(got.generation.tokens, want.generation.tokens);
  // The sixteen greedy tokens of this prompt, as the M3 classes gave them
  // (the migration's gate held the engine's heads and tokens bit for bit to
  // theirs): the repeated seed's own pattern.
  const std::vector<std::int32_t> expected =
      GetParam() == 2
          ? std::vector<std::int32_t>(16, 235256)
          : std::vector<std::int32_t>{236743, 107,    818, 236743, 107,    818, 236743, 107,
                                      818,    236743, 107, 818,    236743, 107, 818,    236743};
  EXPECT_EQ(want.generation.tokens, expected);
  ASSERT_EQ(got.generation.logits.size(), want.generation.logits.size());
  for (std::size_t i = 0; i < want.generation.logits.size(); ++i) {
    ASSERT_TRUE(Finite(want.generation.logits[i], Vocab())) << "row " << i;
    EXPECT_TRUE(Same(got.generation.logits[i], want.generation.logits[i])) << "row " << i;
  }
  EXPECT_TRUE(Finite(want.continued, Vocab()));
  EXPECT_TRUE(Same(got.continued, want.continued));
  for (std::size_t id = 0; id < 2; ++id) {
    ASSERT_TRUE(Finite(want_joined[id], Vocab()));
    EXPECT_TRUE(Same(got_joined[id], want_joined[id])) << "owner " << id;
  }
}
TEST_P(GemmaEngineServingGpu, SnapshotSpillCheckpointAndRestartKeepTheState) {
  life->options.keep_conversations = true;
  ASSERT_TRUE(Start());
  // Owner 0: a prompt whose first 1,280 tokens are its turn's stable
  // prefix (a turn checkpoint there), then its continuation's head from a
  // snapshot taken and restored around it.
  const auto prefix = Tokens(0, 1280);
  auto first = prefix;
  const auto tail_a = Tokens(0, 9, 1), tail_c = Tokens(0, 7, 2);
  first.insert(first.end(), tail_a.begin(), tail_a.end());
  auto other = prefix;
  other.insert(other.end(), tail_c.begin(), tail_c.end());
  std::vector<float> last;
  std::uint32_t reused = 0;
  ASSERT_TRUE(Branch(0).PreparePrompt(first, 1280, last, reused));
  ASSERT_EQ(Branch(0).turn_checkpoints(), 1U);
  // The other prompt from the checkpoint, then the first again.
  std::vector<float> expected_other;
  ASSERT_TRUE(Branch(0).PreparePrompt(other, 1280, expected_other, reused));
  EXPECT_EQ(reused, 1280U);
  ASSERT_TRUE(Finite(expected_other, Vocab()));
  ASSERT_TRUE(Branch(0).PreparePrompt(first, 1280, last, reused));
  EXPECT_EQ(reused, 1280U);
  {
    // The same prompt prefilled from nothing: the same head.
    ASSERT_TRUE(Select(2));
    std::vector<float> fresh;
    ASSERT_TRUE(Branch(1).Prefill(other, fresh));
    EXPECT_TRUE(Same(fresh, expected_other));
    ASSERT_TRUE(Branch(1).Clear());
  }
  const auto state = StateHash(0);
  ASSERT_TRUE(state) << state.error();
  constexpr std::array<std::int32_t, 1> anchor{563};
  std::vector<float> expected_next;
  {
    auto& node = life->server->node();
    std::vector<llmp::catalog::ExtentId> staging;
    auto snapshot = node.Pinned(Branch(0).state_snapshot_bytes(), 0, staging);
    ASSERT_TRUE(snapshot);
    ASSERT_TRUE(Branch(0).SaveState(*snapshot));
    ASSERT_TRUE(Branch(0).Prefill(anchor, expected_next));
    ASSERT_TRUE(Finite(expected_next, Vocab()));
    ASSERT_TRUE(Branch(0).RestoreState(*snapshot));
    ASSERT_TRUE(node.FreePinned(*snapshot));
  }
  EXPECT_EQ(Branch(0).history(), first);
  const auto restored = StateHash(0);
  ASSERT_TRUE(restored) << restored.error();
  EXPECT_EQ(*restored, *state);
  // A diagnostic snapshot's restore drops the turn checkpoints
  // (Llm::RestoreState); the other prompt makes its prefix's again, and
  // the first prompt from it is the state it was.
  EXPECT_EQ(Branch(0).turn_checkpoints(), 0U);
  {
    std::vector<float> again;
    ASSERT_TRUE(Branch(0).PreparePrompt(other, 1280, again, reused));
    EXPECT_TRUE(Same(again, expected_other));
    ASSERT_EQ(Branch(0).turn_checkpoints(), 1U);
    ASSERT_TRUE(Branch(0).PreparePrompt(first, 1280, last, reused));
    EXPECT_EQ(reused, 1280U);
    const auto back = StateHash(0);
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(*back, *state);
  }
  // An idle spill and its restore.
  {
    const auto retired = life->server->RetireRequestBranches(*model, true);
    ASSERT_TRUE(retired.result);
    ASSERT_TRUE(model->SpillIdle(Branch(0)));
    EXPECT_TRUE(Branch(0).spilled());
    ASSERT_TRUE(Select(2));
    auto restoring = Branch(0).BeginPrompt(first, 0, false, true);
    ASSERT_TRUE(restoring);
    rt::Status run;
    while (!(*restoring)->done() && (run = (*restoring)->Advance())) {
    }
    ASSERT_TRUE(run) << (run ? "" : run.error());
    ASSERT_TRUE((*restoring)->Finish());
    EXPECT_EQ((*restoring)->reused(), first.size());
    restoring->reset();
    const auto back = StateHash(0);
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(*back, *state);
  }
  // Kept across a restart, the turn checkpoint too.
  {
    const auto retired = life->server->RetireRequestBranches(*model, true);
    ASSERT_TRUE(retired.result);
    ASSERT_TRUE(model->SpillIdle(Branch(0)));
    life->server->Persist(rt::Clock::now() + std::chrono::seconds(30), {});
    ASSERT_TRUE(Retire());
  }
  ASSERT_TRUE(Start());
  ASSERT_EQ(Branch(0).history(), first);
  EXPECT_EQ(Branch(0).turn_checkpoints(), 1U) << Log();
  {
    auto restoring = Branch(0).BeginPrompt(first, 0, false, true);
    ASSERT_TRUE(restoring);
    rt::Status run;
    while (!(*restoring)->done() && (run = (*restoring)->Advance())) {
    }
    ASSERT_TRUE(run) << (run ? "" : run.error());
    ASSERT_TRUE((*restoring)->Finish());
    EXPECT_EQ((*restoring)->reused(), first.size());
  }
  const auto kept = StateHash(0);
  ASSERT_TRUE(kept) << kept.error();
  EXPECT_EQ(*kept, *state);
  std::vector<float> next;
  ASSERT_TRUE(Branch(0).Prefill(anchor, next));
  EXPECT_TRUE(Same(next, expected_next));
  // The kept turn checkpoint restores the other prompt's prefix.
  std::vector<float> from_kept;
  ASSERT_TRUE(Branch(0).PreparePrompt(other, 1280, from_kept, reused));
  EXPECT_EQ(reused, 1280U);
  EXPECT_TRUE(Same(from_kept, expected_other));
  const std::string text = Log();
  EXPECT_EQ(text.find("kept conversation was refused"), std::string::npos) << text;
  std::println("GEMMA_ENGINE_STATE family={} snapshot=1 spill=1 checkpoint=1 kept=1 extra={}",
               GetParam(), model->extra());
}

TEST_P(GemmaEngineServingGpu, JoinedPromptsPastTheRingKeepHeadsStateAndRestartExactly) {
  life->options.keep_conversations = true;
  constexpr std::array<std::int32_t, 1> anchor{563};
  Wrapped want, got;
  // The uninterrupted owners: the prompts, then their continuations' heads.
  ASSERT_TRUE(Start());
  {
    const auto r = WrappedPrompts(want);
    ASSERT_TRUE(r) << (r ? "" : r.error());
    for (std::uint32_t id = 0; id < 2; ++id) {
      ASSERT_TRUE(Branch(id).Prefill(anchor, want.next[id]));
      ASSERT_TRUE(Finite(want.next[id], Vocab()));
    }
  }
  ASSERT_TRUE(Retire());
  ASSERT_NO_FATAL_FAILURE(ClearStorage());
  // The same prompts on a new server, then spilled, kept, restarted and
  // restored.
  ASSERT_TRUE(Start());
  {
    const auto r = WrappedPrompts(got);
    ASSERT_TRUE(r) << (r ? "" : r.error());
  }
  for (std::uint32_t id = 0; id < 2; ++id) {
    EXPECT_TRUE(Same(got.heads[id], want.heads[id])) << "owner " << id;
    EXPECT_EQ(got.state[id], want.state[id]) << "owner " << id;
  }
  {
    const auto retired = life->server->RetireRequestBranches(*model, true);
    ASSERT_TRUE(retired.result);
    ASSERT_TRUE(retired.references_retired);
    for (std::uint32_t id = 0; id < 2; ++id) {
      ASSERT_TRUE(model->SpillIdle(Branch(id)));
      EXPECT_TRUE(Branch(id).spilled());
    }
    life->server->Persist(rt::Clock::now() + std::chrono::seconds(30), {});
    ASSERT_TRUE(Retire());
  }
  ASSERT_TRUE(Start());
  for (std::uint32_t id = 0; id < 2; ++id) {
    ASSERT_EQ(Branch(id).history(), got.history[id]);
    const auto restored = RestoreBranch(Branch(id), got.history[id]);
    ASSERT_TRUE(restored) << (restored ? "" : restored.error());
    const auto state = StateHash(id);
    ASSERT_TRUE(state) << state.error();
    EXPECT_EQ(*state, want.state[id]) << "owner " << id;
    std::vector<float> next;
    ASSERT_TRUE(Branch(id).Prefill(anchor, next));
    EXPECT_TRUE(Same(next, want.next[id])) << "owner " << id;
  }
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
  const std::string text = Log();
  EXPECT_EQ(text.find("kept conversation was refused"), std::string::npos) << text;
  std::println("GEMMA_ENGINE_RING family={} rows={},{} state_exact=1 head_exact=1 restart_exact=1",
               GetParam(), WrapRows()[0], WrapRows()[1]);
}

// Gemma 3 at an admitted 8,448 context: an 8,192-token prompt in 128-row
// chunks (the public chunk), its continuation, and both through a spill and
// a restore, the same from one server to the next.
TEST_P(GemmaEngineGemma3Gpu, Admitted8448ContextKeepsAPromptItsContinuationAndASpillExact) {
  const auto input = Models() / "references/gemma3-depth/ids.i32";
  std::error_code error;
  const auto bytes = fs::file_size(input, error);
  if (error) GTEST_SKIP() << "no Gemma3 depth reference at " << input;
  ASSERT_EQ(bytes, 33024U);
  std::vector<std::int32_t> ids(8256);
  std::ifstream stream(input, std::ios::binary);
  ASSERT_TRUE(stream.read(reinterpret_cast<char*>(ids.data()),
                          static_cast<std::streamsize>(ids.size() * sizeof(ids[0]))));
  llmp::base::Sha256 hash;
  ASSERT_EQ(llmp::base::ToHex(hash.Update(std::as_bytes(std::span(ids))).Finish()),
            "44196b939c8b53b535a59e1c139f6dc4a7959f7880cd8c8a9d91688818f5ad67");
  ASSERT_TRUE(std::ranges::all_of(ids, [](std::int32_t id) { return id >= 0 && id < 262208; }));
  ASSERT_EQ(ids.front(), 2);
  ASSERT_EQ(ids[8192], 496);
  const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + 8192);
  auto continued = prompt;
  continued.push_back(496);
  life->config.models.front().overrides["context"] = std::int64_t{8448};
  life->config.models.front().overrides["max_slots"] = std::int64_t{1};
  constexpr std::array<std::int32_t, 1> anchor{563};
  std::vector<float> want_head, want_next, want_after;
  llmp::base::Sha256Digest want_state{}, want_next_state{};
  for (const bool second : {false, true}) {
    ASSERT_TRUE(Start(1));
    ASSERT_FALSE(model->branch(1));
    auto& branch = Branch(0);
    std::vector<float> warm;
    ASSERT_TRUE(branch.Prefill(std::span(prompt).first(3), warm));
    ASSERT_TRUE(branch.Clear());
    model->DropPlans();
    auto begun = branch.BeginPrompt(prompt);
    ASSERT_TRUE(begun);
    rt::Status run;
    std::uint32_t chunks = 0;
    while (!(*begun)->done()) {
      const auto prior = Engine().runner().slots().positions(0);
      run = (*begun)->Advance();
      if (!run) break;
      const auto current = Engine().runner().slots().positions(0);
      EXPECT_EQ(current, branch.history().size());
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
    ASSERT_TRUE(Finite(head, 262208U));
    EXPECT_EQ(branch.history(), prompt);
    EXPECT_EQ(Engine().runner().slots().positions(0), 8192U);
    const auto state = StateHash(0);
    ASSERT_TRUE(state) << state.error();
    EXPECT_FALSE(life->server->node().has_pending_state_preparation());
    EXPECT_EQ(Engine().counts().joined_prefill_steps, 0U);
    std::vector<float> next;
    ASSERT_TRUE(branch.Prefill(std::span(ids).subspan(8192, 1), next));
    ASSERT_TRUE(Finite(next, 262208U));
    EXPECT_EQ(branch.history(), continued);
    EXPECT_EQ(Engine().runner().slots().positions(0), 8193U);
    const auto next_state = StateHash(0);
    ASSERT_TRUE(next_state) << next_state.error();
    // An idle spill and its restore keep the continued state exactly, and
    // the conversation goes on from it.
    {
      const auto retired = life->server->RetireRequestBranches(*model, true);
      ASSERT_TRUE(retired.result);
      ASSERT_TRUE(model->SpillIdle(branch));
      EXPECT_TRUE(branch.spilled());
      ASSERT_TRUE(Select(1));
      const auto restored = RestoreBranch(branch, continued);
      ASSERT_TRUE(restored) << (restored ? "" : restored.error());
      const auto back = StateHash(0);
      ASSERT_TRUE(back) << back.error();
      EXPECT_EQ(*back, *next_state);
    }
    std::vector<float> after;
    ASSERT_TRUE(branch.Prefill(anchor, after));
    ASSERT_TRUE(Finite(after, 262208U));
    EXPECT_EQ(Engine().runner().coverage().violations, 0U);
    if (!second) {
      want_head = head;
      want_state = *state;
      want_next = next;
      want_next_state = *next_state;
      want_after = after;
    } else {
      EXPECT_TRUE(Same(head, want_head));
      EXPECT_EQ(*state, want_state);
      EXPECT_TRUE(Same(next, want_next));
      EXPECT_EQ(*next_state, want_next_state);
      EXPECT_TRUE(Same(after, want_after));
      std::println(
          "GEMMA3_ADMITTED_8448 context=8448 owners=1 prompt=8192 continuation=8193 chunks={} "
          "state_exact=1 head_exact=1 continuation_exact=1 spill_exact=1",
          chunks);
    }
    ASSERT_TRUE(Retire());
  }
}

TEST_P(GemmaEngineServingGpu, DiagnosticBudgetCapLeavesDefaultsAndChecksActualAdmission) {
  EXPECT_FALSE(life->options.partial_weight_eviction);
  EXPECT_FALSE(life->options.diagnostic_budget_cap_bytes);
  ASSERT_TRUE(Start());
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
  ASSERT_TRUE(Start());
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
  const auto too_small = Start();
  EXPECT_FALSE(too_small);
  if (!too_small)
    EXPECT_NE(too_small.error().find("required startup footprint"), std::string::npos);
  ASSERT_TRUE(Retire());
  life->options.diagnostic_budget_cap_bytes = ~std::uint64_t{0};
  const auto too_large = Start();
  EXPECT_FALSE(too_large);
  if (!too_large) EXPECT_NE(too_large.error().find("ordinary dynamic budget"), std::string::npos);
}

// Small real adapters exercise Server's shared transaction, not a second
// synthetic approximation of its request and restart-record behavior.
TEST_P(GemmaEngineServingGpu, PartialCachedSwitchReleasesRequestAndRetainsOtherWeights) {
  life->roles.installed = Models() / "m3-artifacts";
  if (!Present(life->roles.installed / kGemma2) || !Present(life->roles.installed / kGemma3)) {
    GTEST_SKIP() << "no Gemma2 and Gemma3 artifacts in " << life->roles.installed;
  }
  auto other = life->config.models.front();
  other.name = "other";
  other.artifact = GetParam() == 2 ? kGemma3 : kGemma2;
  other.overrides["context"] = std::int64_t{4096};
  life->config.models.push_back(other);
  life->options.partial_weight_eviction = true;
  life->options.diagnostic_budget_cap_bytes = std::uint64_t{16} << 30U;
  ASSERT_TRUE(Start());
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

TEST_P(GemmaEngineServingGpu, FailedPartialLoadPreservesCachesAndRecreatesSavedRecord) {
  life->roles.installed = Models() / "m3-artifacts";
  if (!Present(life->roles.installed / kGemma2) || !Present(life->roles.installed / kGemma3)) {
    GTEST_SKIP() << "no Gemma2 and Gemma3 artifacts in " << life->roles.installed;
  }
  auto other = life->config.models.front();
  other.name = "other";
  other.artifact = GetParam() == 2 ? kGemma3 : kGemma2;
  other.overrides["context"] = std::int64_t{4096};
  life->config.models.push_back(other);
  auto inactive_entry = other;
  inactive_entry.name = "inactive";
  inactive_entry.artifact = kGemma4;
  if (!Present(life->roles.installed / *inactive_entry.artifact)) {
    GTEST_SKIP() << "no Gemma 4 artifact in " << life->roles.installed;
  }
  life->config.models.push_back(inactive_entry);
  life->options.partial_weight_eviction = true;
  life->options.keep_conversations = true;
  life->config.memory.keep_across_restart = true;
  ASSERT_TRUE(Start());
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
  std::vector<llmp::catalog::ExtentId> inactive_cached;
  ASSERT_TRUE(life->server->node().Call(
      [&]() -> rt::Status {
        for (const auto extent : inactive->weights())
          if (life->server->node().catalog().Describe(extent)->state ==
              llmp::catalog::ExtentState::kResident)
            inactive_cached.push_back(extent);
        return {};
      },
      "snapshotting unrelated retained cache before the failed switch"));
  ASSERT_FALSE(inactive_cached.empty());
  const auto missing = weights.back();
  ASSERT_TRUE(life->server->node().Evict({missing}));
  llmp::scheduler::PageSource original;
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
                    llmp::catalog::ExtentState::kResident);
        for (const auto extent : model->weights())
          EXPECT_EQ(life->server->node().catalog().Describe(extent)->state,
                    llmp::catalog::ExtentState::kResident);
        for (const auto extent : inactive_cached)
          EXPECT_EQ(life->server->node().catalog().Describe(extent)->state,
                    llmp::catalog::ExtentState::kResident);
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
  const auto hash_state = [&]() { return StateHash(0); };
  const auto saved_hash = hash_state();
  ASSERT_TRUE(saved_hash);
  // Two independent failures: the incoming readiness check, followed by
  // a missing outgoing weight's short read during undo. No request runs
  // against either state between their completed spill and restoration.
  const auto outgoing_missing = model->weights().back();
  ASSERT_TRUE(life->server->node().Evict({outgoing_missing}));
  llmp::scheduler::PageSource outgoing_source;
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

TEST_P(GemmaEngineGemma3Gpu, UnprovenPreparationStopsServerBeforeOwnerDestruction) {
  // This test is launched alone. Parent SetUp only creates paths/configuration;
  // CUDA/node threads are first created inside the isolated child. No fork of
  // an active provider and no production fault hook are involved.
  EXPECT_EXIT(
      ([&] {
        if (!llmp::platform::InstallCrashPolicy("prepared-owner-control")) std::_Exit(61);
        log_to_stderr = true;
        if (!Start()) std::_Exit(62);
        auto& node = life->server->node();
        en::PagedModel& runner = model->paged();
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
                    resident =
                        node.catalog().Describe(id)->state == llmp::catalog::ExtentState::kResident;
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
                  owned = view && view->state == llmp::catalog::ExtentState::kQuarantined &&
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
      ::testing::ExitedWithCode(llmp::platform::kFatalSignalExitBase + SIGABRT),
      "unretired state preparation; aborting before owner destruction");
}

INSTANTIATE_TEST_SUITE_P(Gemma3, GemmaEngineGemma3Gpu, ::testing::Values(3U),
                         [](const ::testing::TestParamInfo<std::uint32_t>& info) {
                           return std::format("gemma{}", info.param);
                         });

INSTANTIATE_TEST_SUITE_P(Families, GemmaEngineServingGpu, ::testing::Values(2U, 3U),
                         [](const ::testing::TestParamInfo<std::uint32_t>& info) {
                           return std::format("gemma{}", info.param);
                         });

}  // namespace
