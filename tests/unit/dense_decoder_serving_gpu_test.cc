// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Dense decoder models (Llama 3.2 1B Instruct Q8_0 and Q4_K_M, Qwen2.5 0.5B
// Instruct FP16 GGUF and EXL3 4.0 bpw) served through the runtime's server
// over the shared component engine (runtime/serving.h) under the default
// serving recipe (spec/mapping.h kDefaultServingRecipe):
// - admission and default settings (prefill_chunk = 4,096);
// - multi-row prompt scoring through BeginScoringPrompt and RunPromptWave,
//   verifying published row indices advance per row;
// - prompt prefill and greedy generation (two-owner joined decode waves for
//   Llama 3.2 1B, single-owner waves for Qwen2);
// - idle state spill, zero-step restore, bit-identical state image before
//   spill and after restore, and continuation without recomputing history.

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
#include "runtime/serving.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace cfg = llmp::config;
namespace en = llmp::engine;
namespace rt = llmp::runtime;

constexpr std::uint64_t kResultBytes = 64ULL << 20U;
constexpr std::uint64_t kSpillCacheBytes = 4ULL << 30U;
constexpr std::uint64_t kExecutionCap = 32ULL << 30U;

struct Profile {
  const char* name;
  const char* v1_name;
  std::uint32_t vocab;
  std::uint32_t slots;
  bool joined_decode;
  // The rows device choices read below a wider head: its tokenizer's (RE-050).
  std::uint32_t choices = 0;
};

constexpr std::array<Profile, 4> kProfiles{{
    {"Llama32_1B_Q8_0", "llama32-1b-q8_0", 128256, 2, true, 0},
    {"Llama32_1B_Q4_K_M", "llama32-1b-q4_k_m", 128256, 2, true, 0},
    {"Qwen2_FP16", "qwen2-fp16", 151936, 1, false, 0},
    {"Qwen2_EXL3_4_0", "qwen2-exl3-4.0", 151936, 1, false, 151665},
}};

bool Finite(std::span<const float> row, std::size_t width) {
  return row.size() == width && std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
}

std::vector<std::int32_t> Tokens(std::uint32_t owner, std::size_t rows) {
  constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
  std::vector<std::int32_t> out(rows);
  for (std::size_t i = 0; i < rows; ++i) out[i] = seed[(i + owner * 3 + i / 7) % seed.size()];
  return out;
}

class DenseDecoderServingGpu : public ::testing::TestWithParam<std::size_t> {
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
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<std::FILE, CloseFile> log;
    fs::path scratch, native_store;
    std::string native_id;
    std::unique_ptr<rt::Server> server;
    Result result;
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
  rt::Llm::Branch& Branch(std::uint32_t id) { return **model->branch(id); }
  rt::Llm& Engine() { return dynamic_cast<rt::Llm&>(*model); }

  void SetUp() override {
    const auto id = llmp::test_support::V1IdNamed(profile().v1_name);
    if (!id) {
      GTEST_FAIL() << "no published v1 " << profile().v1_name << " in "
                   << llmp::test_support::V1Store();
    }
    life->native_id = *id;
    ASSERT_EQ(life->native_id.size(), 64U);
    life->native_store = llmp::test_support::V1Store();
    std::error_code error;
    ASSERT_TRUE(fs::is_directory(life->native_store / life->native_id, error));
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path root = base != nullptr ? base : ::testing::TempDir();
    fs::create_directories(root, error);
    ASSERT_FALSE(error);
    std::string name = (root / "dense-decoder-v1-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    life->scratch = name;
    life->roles.spill = life->scratch / "spill";
    life->roles.state = life->scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      fs::remove_all(path);
      ASSERT_TRUE(fs::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    cfg::ModelEntry entry;
    entry.name = "dense";
    entry.overrides["context"] = std::int64_t{4096};
    entry.overrides["max_slots"] = static_cast<std::int64_t>(profile().slots);
    life->config.models.push_back(std::move(entry));
    life->options.diagnostic_budget_cap_bytes = kExecutionCap;
    life->options.diagnostic_startup_host_bytes = kResultBytes + kSpillCacheBytes;
    life->generate.max_tokens = 4;
    life->generate.stop = false;
    life->generate.keep_logits = true;
    life->log.reset(std::tmpfile());
    ASSERT_NE(life->log, nullptr);
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
      std::ignore = life.release();
      return;
    }
    const fs::path scratch = life->scratch;
    life.reset();
    if (!scratch.empty()) fs::remove_all(scratch);
  }

  rt::Status Select() {
    std::array<rt::Llm::Branch*, 2> branches{};
    for (std::uint32_t owner = 0; owner < profile().slots; ++owner) {
      auto branch = model->branch(owner);
      if (!branch) return std::unexpected(branch.error());
      branches[owner] = *branch;
    }
    return life->server->SelectRequestBranches(*model, std::span(branches).first(profile().slots));
  }

  rt::Status Start() {
    life->roles.installed = life->native_store;
    life->config.models.front().artifact = life->native_id;
    life->server =
        std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log.get());
    if (auto started = life->server->Start(true); !started) return started;
    model = dynamic_cast<rt::Llm*>(life->server->Find("dense"));
    if (model == nullptr || dynamic_cast<rt::Llm*>(model) == nullptr)
      return std::unexpected("published artifact did not use the common Llm");
    if (model->logit_width() != profile().vocab ||
        model->choice_vocabulary() != profile().choices ||
        model->settings().prefill_chunk.value != 4096U ||
        model->settings().context.value != 4096U ||
        model->settings().max_slots.value != profile().slots)
      return std::unexpected("published artifact did not receive kDefaultServingRecipe settings");
    rt::SwapParts activation;
    if (auto active = life->server->Activate(*model, activation); !active) return active;
    const auto state_bound = model->StateBytesThrough(4096);
    if (!state_bound || *state_bound == 0 || *state_bound > kSpillCacheBytes / profile().slots)
      return std::unexpected("slots exceed the prepaid spill-cache envelope");
    return Select();
  }

  std::uint64_t ExecutedSteps() {
    const auto& stats = Engine().runner().graph_stats();
    return stats.eager + stats.captured + stats.replayed;
  }

  rt::Status CheckMultiRowScoring() {
    if (auto cleared = Branch(0).Clear(); !cleared) return cleared;
    const auto prompt = Tokens(0, 12);
    std::vector<std::uint32_t> published_positions;
    std::vector<std::int32_t> scored_ids;
    auto begun = Branch(0).BeginScoringPrompt(
        prompt,
        [&](std::int32_t id, std::span<const float> row) {
          if (!Finite(row, profile().vocab)) return false;
          published_positions.push_back(life->sessions[0]->run().published);
          scored_ids.push_back(id);
          return true;
        },
        false);
    if (!begun) return std::unexpected(begun.error());
    life->sessions[0] = std::move(*begun);
    while (!life->sessions[0]->done()) {
      if (auto advanced = life->sessions[0]->Advance(); !advanced) {
        uncertain = true;
        return advanced;
      }
    }
    if (published_positions.size() != prompt.size() - 1)
      return std::unexpected("multi-row scoring did not publish every target row");
    for (std::size_t i = 0; i < published_positions.size(); ++i) {
      if (published_positions[i] != i + 1 || scored_ids[i] != prompt[i + 1])
        return std::unexpected("multi-row scoring published wrong row index or token id");
    }
    if (!Finite(life->sessions[0]->last(), profile().vocab))
      return std::unexpected("multi-row scoring frontier logits are not finite");
    return FinishSessions();
  }

  rt::Status Prompts(Result& out) {
    const std::uint32_t slots = profile().slots;
    for (std::uint32_t owner = 0; owner < slots; ++owner) {
      life->prompts[owner] = Tokens(owner, 7);
      if (auto cleared = Branch(owner).Clear(); !cleared) return cleared;
      auto begun = Branch(owner).BeginPrompt(life->prompts[owner]);
      if (!begun) return std::unexpected(begun.error());
      life->sessions[owner] = std::move(*begun);
      if (auto advanced = life->sessions[owner]->Advance(); !advanced) {
        uncertain = true;
        return advanced;
      }
    }
    while (std::ranges::any_of(std::span(life->sessions).first(slots),
                               [](const auto& s) { return !s->done(); })) {
      std::array<rt::Llm::PromptSession*, 2> active{};
      std::size_t count = 0;
      for (std::uint32_t owner = slots; owner > 0; --owner) {
        auto* session = life->sessions[owner - 1].get();
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
    for (std::uint32_t owner = 0; owner < slots; ++owner) {
      out.heads[owner] = life->sessions[owner]->last();
      if (Branch(owner).history() != life->prompts[owner])
        return std::unexpected("prefill history differs");
    }
    return FinishSessions();
  }

  rt::Status GenerateWaves(Result& out) {
    const std::uint32_t slots = profile().slots;
    if (profile().joined_decode &&
        (!model->supports_generation_waves() || model->generation_wave_capacity() != slots))
      return std::unexpected("expected joined generation wave capacity");
    const auto before = Engine().counts();
    const auto executed = ExecutedSteps();
    for (std::uint32_t owner = 0; owner < slots; ++owner) {
      auto begun =
          Branch(owner).BeginGeneration(out.heads[owner], life->generate, out.generated[owner]);
      if (!begun) return std::unexpected(begun.error());
      life->generations[owner] = std::move(*begun);
    }
    uncertain = true;
    for (std::uint32_t wave = 0; wave < 3; ++wave) {
      std::array<rt::Llm::GenerationSession*, 2> active{};
      for (std::uint32_t i = 0; i < slots; ++i) {
        if (life->generations[i]->done())
          return std::unexpected("generation ended before three steps");
        active[i] = life->generations[slots - 1 - i].get();
      }
      if (auto run = model->RunGenerationWave(std::span(active).first(slots)); !run) return run;
      for (std::uint32_t owner = 0; owner < slots; ++owner)
        if (out.generated[owner].tokens.size() != wave + 2 ||
            out.generated[owner].steps != wave + 1)
          return std::unexpected("generation wave did not complete exact unit");
    }
    for (std::uint32_t owner = 0; owner < slots; ++owner)
      if (!life->generations[owner]->done())
        return std::unexpected("generation exceeded four tokens");
    const auto after = Engine().counts();
    if (profile().joined_decode &&
        (after.joined_waves != before.joined_waves + 3 ||
         after.joined_units != before.joined_units + 3 * slots || ExecutedSteps() != executed + 3))
      return std::unexpected("two-owner requests were not three actual joined decode steps");
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

TEST_P(DenseDecoderServingGpu, ServesPromptsScoringWavesSpillAndRestoreOnSharedEngine) {
  const auto started = Start();
  ASSERT_TRUE(started) << (started ? "" : started.error());
  const auto scored = CheckMultiRowScoring();
  ASSERT_TRUE(scored) << (scored ? "" : scored.error());
  auto& out = life->result;
  const auto prompted = Prompts(out);
  ASSERT_TRUE(prompted) << (prompted ? "" : prompted.error());
  const auto generated = GenerateWaves(out);
  ASSERT_TRUE(generated) << (generated ? "" : generated.error());
  const std::uint32_t slots = profile().slots;
  std::uint64_t spill_bytes = 0;
  for (std::uint32_t owner = 0; owner < slots; ++owner) {
    ASSERT_EQ(out.generated[owner].tokens.size(), 4U);
    ASSERT_EQ(out.generated[owner].logits.size(), 4U);
    ASSERT_EQ(out.generated[owner].steps, 3U);
    out.histories[owner] = Branch(owner).history();
    auto state = StateImage(owner);
    ASSERT_TRUE(state) << (state ? "" : state.error());
    out.state[owner] = std::move(*state);
    const auto bytes = model->StateBytes(Branch(owner));
    ASSERT_LE(bytes, kSpillCacheBytes - spill_bytes);
    ASSERT_GE(bytes, out.state[owner].bytes);
    spill_bytes += bytes;
  }
  ASSERT_TRUE(life->server->EndRequestBranches(*model));
  for (std::uint32_t owner = 0; owner < slots; ++owner) {
    const auto spilled = model->SpillIdle(Branch(owner));
    ASSERT_TRUE(spilled) << (spilled ? "" : spilled.error());
    ASSERT_TRUE(Branch(owner).spilled());
  }
  ASSERT_TRUE(Select());
  for (std::uint32_t owner = 0; owner < slots; ++owner) {
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
  for (std::uint32_t owner = 0; owner < slots; ++owner) {
    ASSERT_TRUE(Finite(out.heads[owner], profile().vocab));
    ASSERT_TRUE(Finite(out.continued[owner], profile().vocab));
    for (const auto& row : out.generated[owner].logits) ASSERT_TRUE(Finite(row, profile().vocab));
  }
}

INSTANTIATE_TEST_SUITE_P(Canaries, DenseDecoderServingGpu, ::testing::Values(0U, 1U, 2U, 3U),
                         [](const auto& info) { return kProfiles.at(info.param).name; });
}  // namespace
