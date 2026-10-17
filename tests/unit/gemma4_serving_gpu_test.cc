// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The production adapter and driver over the approved native artifacts:
// since M3.6 (slice G2c) Gemma 4 serves on the shared engine
// (runtime/engine_llm.h), so these controls, taken on the M3 class, now hold
// for it (the M3 class's own internals, its preparation diagnostic and its
// kept tags' profile prefix, are gone from them).
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/paged_node.h"
#include "runtime/engine_llm.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"
#include "v1_frozen.h"

namespace rt = llmp::runtime;
namespace en = llmp::engine;
namespace cfg = llmp::config;

namespace {
// A published v1 artifact's ID by the store's v1-ids.json name, empty when
// absent (tests/support/v1_frozen.h).
std::string Published(const char* name) { return llmp::test_support::V1IdNamed(name).value_or(""); }
// The model's and the other profile's targets.
const char* TargetName(std::uint32_t profile) { return profile == 26 ? "gemma26" : "gemma31"; }
const char* OtherName(std::uint32_t profile) { return profile == 26 ? "gemma31" : "gemma26"; }
}  // namespace

class Gemma4ServingGpu : public ::testing::TestWithParam<std::uint32_t> {
 protected:
  // The approved 8K/C4 envelope (context 8,192, four slots, the profile's
  // prefill chunk) instead of the 4K fixture with twelve slots.
  virtual bool Envelope() const { return false; }
  // The profile's assistant, as the model's `drafter`.
  virtual bool Assistant() const { return false; }
  void SetUp() override {
    const std::filesystem::path installed = llmp::test_support::V1Store();
    const std::string artifact = Published(TargetName(GetParam()));
    if (artifact.empty()) GTEST_FAIL() << "no published v1 Gemma 4 in " << installed;
    // Spill files need direct I/O: scratch in the build tree.
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path root = base != nullptr ? base : ::testing::TempDir();
    std::error_code error;
    std::filesystem::create_directories(root, error);
    std::string name = (root / "gemma-serving-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    roles.installed = installed;
    roles.spill = scratch / "spill";
    roles.state = scratch / "state";
    std::filesystem::create_directory(roles.spill);
    std::filesystem::create_directory(roles.state);
    ASSERT_EQ(chmod(roles.spill.c_str(), 0700), 0);
    ASSERT_EQ(chmod(roles.state.c_str(), 0700), 0);
    cfg::ModelEntry entry;
    entry.name = "gemma";
    entry.artifact = artifact;
    if (Assistant()) {
      const std::string assistant =
          Published(GetParam() == 26 ? "gemma26-assistant" : "gemma31-assistant");
      if (assistant.empty()) GTEST_FAIL() << "no published v1 Gemma 4 assistant in " << installed;
      entry.drafter = assistant;
    }
    entry.overrides["context"] = std::int64_t{Envelope() ? 8192 : 4096};
    entry.overrides["prefill_chunk"] =
        std::int64_t{Envelope() ? (GetParam() == 26 ? 1024 : 256) : 16};
    entry.overrides["max_slots"] = std::int64_t{Envelope() ? 4 : 12};
    config.models.push_back(entry);
    options.keep_conversations = true;
    ASSERT_TRUE(Start());
  }
  bool Start() {
    ++starts;
    server = std::make_unique<rt::Server>(config, roles, options, stderr);
    auto r = server->Start(true);
    if (!r) {
      ADD_FAILURE() << r.error();
      return false;
    }
    model = dynamic_cast<rt::Llm*>(server->Find("gemma"));
    if (model == nullptr) {
      ADD_FAILURE() << "configured Gemma model missing";
      return false;
    }
    rt::SwapParts parts;
    r = server->Activate(*model, parts);
    if (!r) {
      ADD_FAILURE() << r.error();
      return false;
    }
    return true;
  }
  rt::Status RetireServer() {
    // Server teardown is one-shot: a repeated no-op success cannot prove
    // completion after an earlier failed fence. Keep that failure sticky.
    if (retirement_failed) return std::unexpected("server retirement was previously unproven");
    if (!server) return {};
    auto retired = server->TearDown();
    retirement_failed = !retired;
    return retired;
  }
  void TearDown() override {
    if (server) {
      const auto retired = RetireServer();
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      if (!retired) {
        // A failed teardown is not retirement proof. Retain node/catalog,
        // contexts, borrowed configuration and state files until process exit.
        std::ignore = lifetime.release();
        return;
      }
    }
    if (!scratch.empty()) std::filesystem::remove_all(scratch);
  }
  // The test's own host copies, charged to the current server's node. The
  // charge ends before that server does: returned after a restart, it would
  // write the destroyed node's memory, which the next server's allocations
  // may hold by then. One that outlives its server fails the test instead.
  struct HostCopies {
    HostCopies(Gemma4ServingGpu& fixture, std::uint64_t capacity)
        : owner(fixture),
          start(fixture.starts),
          node(fixture.server->node()),
          bytes(capacity),
          funded(node.ChargeHost(capacity, false)) {}
    ~HostCopies() {
      if (!funded) return;
      if (owner.starts != start || !owner.server) {
        ADD_FAILURE() << "a test host charge outlived the server it was charged to";
        return;
      }
      node.UnchargeHost(bytes);
    }
    HostCopies(const HostCopies&) = delete;
    HostCopies& operator=(const HostCopies&) = delete;
    Gemma4ServingGpu& owner;
    std::uint64_t start;
    en::PagedNode& node;
    std::uint64_t bytes;
    bool funded;
  };
  // The kept layout identity a server of the other profile reports on these
  // settings (its own scratch roles; started and retired here, with no
  // server of this profile running).
  std::string OtherProfileLayout() {
    const std::string artifact = Published(OtherName(GetParam()));
    if (artifact.empty()) {
      ADD_FAILURE() << "no published v1 " << OtherName(GetParam());
      return {};
    }
    auto bundle = std::make_unique<Lifetime>();
    Lifetime& other = *bundle;
    other.config = config;
    other.config.models[0].artifact = artifact;
    other.roles = roles;
    other.roles.spill = scratch / "other-spill";
    other.roles.state = scratch / "other-state";
    std::filesystem::create_directory(other.roles.spill);
    std::filesystem::create_directory(other.roles.state);
    if (chmod(other.roles.spill.c_str(), 0700) != 0 ||
        chmod(other.roles.state.c_str(), 0700) != 0) {
      ADD_FAILURE() << "no scratch roles for the other profile";
      return {};
    }
    other.options.keep_conversations = false;
    other.server = std::make_unique<rt::Server>(other.config, other.roles, other.options, stderr);
    std::string layout;
    auto r = other.server->Start(true);
    if (!r) {
      ADD_FAILURE() << r.error();
    } else if (auto* llm = dynamic_cast<rt::Llm*>(other.server->Find("gemma")); llm == nullptr) {
      ADD_FAILURE() << "the other profile's model is missing";
    } else {
      rt::SwapParts parts;
      if (auto activated = other.server->Activate(*llm, parts); !activated) {
        ADD_FAILURE() << activated.error();
      } else {
        layout = llm->KeptLayout();
      }
    }
    const auto retired = other.server->TearDown();
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    // An unproven retirement keeps the bundle the server borrows.
    if (!retired) std::ignore = bundle.release();
    return layout;
  }
  rt::Llm::Branch& Branch(std::uint32_t i) { return **model->branch(i); }
  rt::Status Select(std::uint32_t count) {
    std::array<rt::Llm::Branch*, 12> active{};
    for (std::uint32_t i = 0; i < count; ++i) active[i] = &Branch(i);
    return server->SelectRequestBranches(*model, std::span(active).first(count));
  }
  std::filesystem::path scratch;
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  // Server borrows these objects. Allocate them together before construction
  // and retain the complete stable bundle if retirement is unproven.
  std::unique_ptr<Lifetime> lifetime = std::make_unique<Lifetime>();
  cfg::NodeConfig& config = lifetime->config;
  cfg::RuntimeRoles& roles = lifetime->roles;
  rt::ServingOptions& options = lifetime->options;
  std::unique_ptr<rt::Server>& server = lifetime->server;
  bool retirement_failed = false;
  std::uint64_t starts = 0;  // servers started (HostCopies)
  rt::Llm* model = nullptr;
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
};

// Four owners at the approved 8K/C4 envelope, their prompts advanced a chunk
// at a time in turn while the selection shrinks as each departs (prompt
// waves are one owner's; the others stay live and idle).
class Gemma4EnvelopeServingGpu : public Gemma4ServingGpu {
 protected:
  bool Envelope() const override { return true; }
  struct Run {
    std::array<std::vector<float>, 4> heads;
    std::array<std::vector<std::int32_t>, 4> continuation;
    std::array<std::uint64_t, 4> units{}, rows{};
    bool continued_after_departure = false;
  };
  // Every prompt through its chunks: `interleaved` takes one chunk of each
  // live owner in turn, otherwise each owner's whole prompt in turn. Then
  // the four owners' joined generation from their heads.
  void RunFourOwners(std::span<const std::vector<std::int32_t>, 4> tokens, bool interleaved,
                     Run& run) {
    struct Sessions {
      std::array<std::unique_ptr<rt::Llm::PromptSession>, 4> owned;
      ~Sessions() {
        for (auto& session : owned)
          if (session) {
            session->Cancel();
            const auto r = session->Finish();
            EXPECT_TRUE(r) << (r ? "" : r.error());
          }
      }
    } sessions;
    ASSERT_TRUE(Select(4));
    for (std::uint32_t id = 0; id < 4; ++id) ASSERT_TRUE(Branch(id).Clear());
    for (std::uint32_t id = 0; id < 4; ++id) {
      auto begun = Branch(id).BeginPrompt(tokens[id]);
      ASSERT_TRUE(begun) << (begun ? "" : begun.error());
      sessions.owned[id] = std::move(*begun);
      const auto settled = sessions.owned[id]->Advance();
      ASSERT_TRUE(settled) << (settled ? "" : settled.error());
    }
    std::uint32_t remaining = 4;
    const auto step = [&](std::uint32_t id) {
      std::array<rt::Llm::Branch*, 4> selected{};
      std::size_t count = 0;
      for (std::uint32_t peer = 0; peer < 4; ++peer)
        if (sessions.owned[peer]) selected[count++] = &Branch(peer);
      ASSERT_TRUE(server->SelectRequestBranches(*model, std::span(selected).first(count)));
      auto unit = sessions.owned[id]->NextUnit();
      ASSERT_TRUE(unit) << (unit ? "" : unit.error());
      ASSERT_EQ(unit->phase, rt::Llm::PromptSession::Phase::kChunk);
      EXPECT_LE(unit->rows, model->settings().prefill_chunk.value);
      rt::Llm::PromptSession* session = sessions.owned[id].get();
      const std::array<rt::PrefillGoOn, 1> callbacks{};
      const auto ran = model->RunPromptWave(std::span(&session, 1), callbacks);
      ASSERT_TRUE(ran) << (ran ? "" : ran.error());
      ASSERT_TRUE(session->last_unit_result());
      ++run.units[id];
      run.rows[id] += unit->rows;
      run.continued_after_departure |= remaining < 4;
      if (!session->done()) return;
      run.heads[id] = session->last();
      ASSERT_EQ(run.heads[id].size(), 262144U);
      ASSERT_TRUE(std::ranges::all_of(run.heads[id], [](float v) { return std::isfinite(v); }));
      EXPECT_EQ(Branch(id).history(), tokens[id]);
      ASSERT_TRUE(session->Finish());
      sessions.owned[id].reset();
      --remaining;
    };
    if (interleaved) {
      while (remaining != 0) {
        for (std::uint32_t id = 0; id < 4; ++id) {
          if (!sessions.owned[id]) continue;
          step(id);
          if (::testing::Test::HasFatalFailure()) return;
        }
      }
    } else {
      for (std::uint32_t id = 0; id < 4; ++id) {
        while (sessions.owned[id]) {
          step(id);
          if (::testing::Test::HasFatalFailure()) return;
        }
      }
    }
    // All prompt work has ended; the departed owners continue together.
    ASSERT_TRUE(Select(4));
    for (std::uint32_t id = 0; id < 4; ++id) {
      EXPECT_EQ(Branch(id).history(), tokens[id]);
      EXPECT_EQ(run.rows[id], tokens[id].size());
    }
    std::array<rt::GenerateOptions, 4> generate;
    std::array<rt::Generation, 4> output;
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 4> session;
    std::array<rt::Llm::GenerationSession*, 4> work{};
    const auto continued = [&]() -> rt::Status {
      for (std::uint32_t id = 0; id < 4; ++id) {
        generate[id].max_tokens = 5;
        generate[id].stop = false;
        auto began = Branch(id).BeginGeneration(run.heads[id], generate[id], output[id]);
        if (!began) return std::unexpected(began.error());
        session[id] = std::move(*began);
        work[id] = session[id].get();
      }
      for (std::uint32_t step = 0; step < 4; ++step)
        if (auto r = model->RunGenerationWave(work); !r) return r;
      return {};
    }();
    for (std::uint32_t id = 0; id < 4; ++id) {
      if (!session[id]) continue;
      session[id]->Cancel();
      EXPECT_TRUE(session[id]->Finish());
      session[id].reset();
      run.continuation[id] = output[id].tokens;
    }
    ASSERT_TRUE(continued) << (continued ? "" : continued.error());
  }
};

TEST_P(Gemma4EnvelopeServingGpu,
       FourOwnersInterleavedPromptsAndDeparturesKeepHeadsAndContinuations) {
  ASSERT_EQ(model->settings().context.value, 8192U);
  ASSERT_EQ(model->settings().prefill_chunk.value, GetParam() == 26 ? 1024U : 256U);
  ASSERT_EQ(model->settings().max_slots.value, 4U);
  ASSERT_EQ(model->branches(), 4U);
  ASSERT_EQ(model->generation_wave_capacity(), 4U);
  // Prompts of different lengths, so the owners depart in turn at either
  // profile's chunk (the 26B-A4B's 1,024 rows: 2, 3, 5 and 7 chunks; the
  // 31B's 256: 6, 12, 18 and 25), the longest past the sliding layers' ring.
  constexpr std::array<std::uint32_t, 4> lengths{1300, 2900, 4600, 6300};
  std::array<std::vector<std::int32_t>, 4> tokens;
  for (std::uint32_t id = 0; id < 4; ++id) {
    tokens[id].resize(lengths[id]);
    for (std::size_t i = 0; i < tokens[id].size(); ++i)
      tokens[id][i] = prompt[(i + id * 3 + i / 7) % prompt.size()];
  }
  // Eight retained full heads and bounded token metadata.
  HostCopies copies(*this, 8ULL * 262144 * sizeof(float) + (512U << 10U));
  ASSERT_TRUE(copies.funded);
  Run turns;
  RunFourOwners(tokens, true, turns);
  if (HasFatalFailure()) return;
  EXPECT_TRUE(turns.continued_after_departure);
  Run alone;
  RunFourOwners(tokens, false, alone);
  if (HasFatalFailure()) return;
  for (std::uint32_t id = 0; id < 4; ++id) {
    // The same chunks, the same rows, whatever the peers did between them.
    EXPECT_EQ(turns.units[id], alone.units[id]);
    EXPECT_EQ(turns.units[id], (lengths[id] + model->settings().prefill_chunk.value - 1) /
                                   model->settings().prefill_chunk.value);
    EXPECT_EQ(turns.rows[id], lengths[id]);
    EXPECT_EQ(alone.rows[id], lengths[id]);
    ASSERT_EQ(turns.heads[id].size(), alone.heads[id].size());
    EXPECT_EQ(std::memcmp(turns.heads[id].data(), alone.heads[id].data(),
                          turns.heads[id].size() * sizeof(float)),
              0)
        << "owner " << id << ": another head";
    EXPECT_EQ(turns.continuation[id].size(), 5U);
    EXPECT_EQ(turns.continuation[id], alone.continuation[id]) << "owner " << id;
  }
  EXPECT_TRUE(model->violations().empty());
  const auto retired = server->RetireRequestBranches(*model, true);
  ASSERT_TRUE(retired.result);
  ASSERT_TRUE(retired.references_retired);
  std::println(
      "GEMMA4_FOUR_OWNER_PROMPTS profile={} context8192 slots4 chunk={} units={},{},{},{} "
      "exact_heads_continuations owner_departure",
      GetParam(), model->settings().prefill_chunk.value, turns.units[0], turns.units[1],
      turns.units[2], turns.units[3]);
}

INSTANTIATE_TEST_SUITE_P(ApprovedProfiles, Gemma4EnvelopeServingGpu, ::testing::Values(26U, 31U),
                         [](const auto& info) { return "Gemma" + std::to_string(info.param); });

// The profile's assistant served as the model's drafter at the envelope: it
// drafts from the target's cache and final-norm features, the target
// verifies, and the drafts it keeps agree with the target's own greedy choice,
// so a wrong assistant (weights, features, borrowed cache or positions)
// keeps none. The generation's tokens are the target's greedy choices; the
// quality of every draft is the verify's.
class Gemma4AssistantServingGpu : public Gemma4EnvelopeServingGpu {
 protected:
  bool Assistant() const override { return true; }
};

TEST_P(Gemma4AssistantServingGpu, DraftsFromTheTargetsStateAreVerifiedAndKept) {
  ASSERT_TRUE(model->speculative());
  ASSERT_TRUE(Select(1));
  // A repeating prompt, whose greedy continuation the assistant can follow.
  std::vector<std::int32_t> tokens;
  for (int i = 0; i < 24; ++i) tokens.insert(tokens.end(), prompt.begin(), prompt.end());
  std::vector<float> last;
  ASSERT_TRUE(Branch(0).Prefill(tokens, last));
  rt::GenerateOptions options;
  options.max_tokens = 48;
  options.stop = false;
  rt::Generation speculative;
  ASSERT_TRUE(Branch(0).Generate(last, options, speculative));
  ASSERT_EQ(speculative.tokens.size(), 48U);
  EXPECT_GT(speculative.drafted, 0U);
  EXPECT_GT(speculative.accepted, 0U);
  EXPECT_LE(speculative.accepted, speculative.drafted);
  EXPECT_LT(speculative.steps, 48U);  // verifies of several rows each
  const auto& engine = dynamic_cast<const rt::EngineLlm&>(*model);
  EXPECT_GT(engine.counts().speculative_units, 0U);
  EXPECT_EQ(engine.counts().discarded, 0U);
  EXPECT_TRUE(model->violations().empty());
  // The same prompt again, from a clear slot: the same drafts and tokens.
  ASSERT_TRUE(Branch(0).Clear());
  ASSERT_TRUE(Branch(0).Prefill(tokens, last));
  rt::Generation again;
  ASSERT_TRUE(Branch(0).Generate(last, options, again));
  EXPECT_EQ(again.tokens, speculative.tokens);
  EXPECT_EQ(again.drafted, speculative.drafted);
  EXPECT_EQ(again.accepted, speculative.accepted);
  std::println("GEMMA4_ASSISTANT profile={} drafted={} accepted={} steps={} tokens=48", GetParam(),
               speculative.drafted, speculative.accepted, speculative.steps);
}

INSTANTIATE_TEST_SUITE_P(ApprovedProfiles, Gemma4AssistantServingGpu, ::testing::Values(26U, 31U),
                         [](const auto& info) { return "Gemma" + std::to_string(info.param); });

TEST_P(Gemma4ServingGpu, PinnedPublicationCapacityMatchesServingSlotsNotInputChunkRows) {
  const auto row_bytes = std::uint64_t{262144} * sizeof(float);
  std::uint64_t staging = 0;
  bool output = false;
  for (const auto& [extent, generation] : model->everything().extents) {
    (void)generation;
    auto view = server->node().catalog().Describe(extent);
    ASSERT_TRUE(view);
    if (view->descriptor.memory_class != llmp::catalog::MemoryClass::kStaging) continue;
    EXPECT_EQ(view->state, llmp::catalog::ExtentState::kResident);
    EXPECT_EQ(view->descriptor.recovery, llmp::catalog::Recovery::kPinned);
    staging += view->descriptor.size.value();
    output |= view->descriptor.size.value() == 12 * row_bytes;
  }
  // Outside the envelope (twelve slots) every step is one owner's: the
  // runner's pinned output staging holds one row and the setup's margin,
  // not a row a slot.
  EXPECT_FALSE(output);
  EXPECT_GE(staging, row_bytes);
  EXPECT_LT(staging, 4 * row_bytes);
}
TEST_P(Gemma4ServingGpu, IndependentScalarCohortsOneTwoFourEightTwelveReplaySeededTokens) {
  EXPECT_EQ(model->generation_wave_capacity(), 12U);
  EXPECT_EQ(model->branches(), 12U);
  EXPECT_GE(server->host_input_bytes(),
            12ULL * 262144 * (2 * sizeof(float) + 2 * sizeof(llmp::execution::SamplingCandidate)));
  HostCopies copies(*this, 13ULL * 262144 * sizeof(float));
  ASSERT_TRUE(copies.funded);
  std::array<rt::GenerateOptions, 12> generate;
  for (std::uint32_t i = 0; i < 12; ++i) {
    generate[i].max_tokens = 4;
    generate[i].stop = false;
    generate[i].sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
    generate[i].seed = 2718 + i;
  }
  std::array<std::vector<std::int32_t>, 12> expected;
  std::array<std::vector<float>, 12> frontiers;
  for (const auto count : {1U, 2U, 4U, 8U, 12U}) {
    ASSERT_TRUE(Select(count));
    for (std::uint32_t i = 0; i < count; ++i) {
      ASSERT_TRUE(Branch(i).Clear());
      std::vector<float> last;
      ASSERT_TRUE(Branch(i).Prefill(prompt, last));
      ASSERT_EQ(last.size(), 262144U);
      frontiers[i] = last;
      rt::Generation out;
      auto r = Branch(i).Generate(last, generate[i], out);
      ASSERT_TRUE(r) << (r ? "" : r.error());
      expected[i] = out.tokens;
    }
    std::array<rt::Generation, 12> output;
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 12> session;
    std::array<rt::Llm::GenerationSession*, 12> work{};
    auto exercised = [&]() -> rt::Status {
      for (std::uint32_t i = 0; i < count; ++i) {
        if (auto r = Branch(i).Clear(); !r) return r;
        std::vector<float> last;
        if (auto r = Branch(i).Prefill(prompt, last); !r) return r;
        EXPECT_EQ(last, frontiers[i]);
        auto began = Branch(i).BeginGeneration(last, generate[i], output[i]);
        if (!began) return std::unexpected(began.error());
        session[i] = std::move(*began);
        work[i] = session[i].get();
      }
      for (std::uint32_t step = 0; step < 3; ++step)
        if (auto r = model->RunGenerationWave(std::span(work).first(count)); !r) return r;
      return {};
    }();
    for (std::uint32_t i = 0; i < count; ++i) {
      if (!session[i]) continue;
      session[i]->Cancel();
      EXPECT_TRUE(session[i]->Finish());
      session[i].reset();
      EXPECT_EQ(output[i].tokens, expected[i]);
      EXPECT_EQ(Branch(i).history().size(), prompt.size() + 3);
    }
    ASSERT_TRUE(exercised) << (exercised ? "" : exercised.error());
    const auto retired = server->RetireRequestBranches(*model, true);
    ASSERT_TRUE(retired.result);
    ASSERT_TRUE(retired.references_retired);
  }
  EXPECT_GT(model->graphs().replayed, 0U);
  EXPECT_TRUE(model->violations().empty());
}
TEST_P(Gemma4ServingGpu, OwnedSnapshotTurnRollbackSpillAndRestartContinueExactly) {
  ASSERT_TRUE(Select(2));
  std::vector<float> first, peer;
  ASSERT_TRUE(Branch(0).Prefill(prompt, first));
  ASSERT_TRUE(Branch(1).Prefill(prompt, peer));
  ASSERT_TRUE(server->SaveSnapshot(*model));
  rt::GenerateOptions gen;
  gen.max_tokens = 4;
  gen.stop = false;
  // With rows, the host chooses each greedy token; without (from here on),
  // the device does (Llm::DeviceGreedy), to the same tokens.
  std::optional<HostCopies> copies;
  copies.emplace(*this, 4ULL * 262144 * sizeof(float));
  ASSERT_TRUE(copies->funded);
  auto rows = gen;
  rows.keep_logits = true;
  rt::Generation expected, restored;
  ASSERT_TRUE(Branch(0).Generate(first, rows, expected));
  ASSERT_EQ(expected.logits.size(), 4U);
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& row = expected.logits[i];
    EXPECT_EQ(expected.tokens[i], std::max_element(row.begin(), row.end()) - row.begin());
  }
  ASSERT_TRUE(server->RestoreSnapshot(*model));
  const auto& engine = dynamic_cast<const rt::EngineLlm&>(*model);
  const std::uint64_t greedy_before = engine.counts().greedy_tokens;
  ASSERT_TRUE(Branch(0).Generate(first, gen, restored));
  EXPECT_EQ(expected.tokens, restored.tokens);
  // The first is the prompt row's.
  EXPECT_EQ(engine.counts().greedy_tokens - greedy_before, 3U);
  // Peer state survives destination rollback.
  rt::Generation peer_out;
  ASSERT_TRUE(Branch(1).Generate(peer, gen, peer_out));
  EXPECT_EQ(expected.tokens, peer_out.tokens);
  // A prompt checkpoint replaces a later occupied state using its owned
  // logical boundary, not the initialized extent byte count.
  ASSERT_TRUE(Branch(0).Clear());
  std::vector<std::int32_t> conversation(prompt.begin(), prompt.end());
  conversation.push_back(563);
  std::uint32_t reused = 0;
  ASSERT_TRUE(Branch(0).PreparePrompt(conversation, 6, first, reused, {}, nullptr, true));
  EXPECT_EQ(Branch(0).turn_checkpoints(), 1U);
  rt::Generation later;
  ASSERT_TRUE(Branch(0).Generate(first, gen, later));
  std::vector<float> rollback;
  ASSERT_TRUE(Branch(0).PreparePrompt(conversation, 6, rollback, reused));
  EXPECT_EQ(reused, 6U);
  // End leases, spill and keep the exact settled ledger/history.
  auto retired = server->RetireRequestBranches(*model, true);
  ASSERT_TRUE(retired.result);
  ASSERT_TRUE(retired.references_retired);
  ASSERT_TRUE(model->SpillIdle(Branch(0)));
  EXPECT_TRUE(Branch(0).spilled());
  const auto history = Branch(0).history();
  server->Persist(rt::Clock::now() + std::chrono::seconds(30), {});
  // The rows the copies' charge funds go with it, before its server does.
  expected.logits = {};
  copies.reset();
  ASSERT_TRUE(RetireServer());
  server.reset();
  const auto record_path =
      roles.spill / "conversations" / *config.models[0].artifact / rt::kept::RecordFileName(0);
  std::ifstream saved(record_path);
  const std::string text{std::istreambuf_iterator<char>(saved), {}};
  auto record = rt::kept::Decode(text);
  ASSERT_TRUE(record);
  ASSERT_EQ(record->checkpoints.size(), 1U);
  EXPECT_EQ(record->checkpoints[0].position, 6U);
  EXPECT_EQ(record->checkpoints[0].cursor, 6U);
  // Its well-formed, redigested metadata fits the same 256-cell padded
  // state bytes, but contradicts the owned logical checkpoint boundary.
  record->checkpoints[0].cursor = 5;
  std::ofstream(record_path) << rt::kept::Encode(*record);
  const auto peer_record_path = record_path.parent_path() / rt::kept::RecordFileName(1);
  std::ifstream peer_saved(peer_record_path);
  const std::string peer_text{std::istreambuf_iterator<char>(peer_saved), {}};
  auto peer_record = rt::kept::Decode(peer_text);
  ASSERT_TRUE(peer_record);
  ASSERT_EQ(peer_record->cursor, peer_record->tokens.size());
  --peer_record->cursor;
  // A plain Gemma kept conversation also has no speculative pending row.
  std::ofstream(peer_record_path) << rt::kept::Encode(*peer_record);
  ASSERT_TRUE(Start());
  EXPECT_TRUE(Branch(1).history().empty());
  EXPECT_FALSE(Branch(1).spilled());
  EXPECT_EQ(Branch(0).history(), history);
  EXPECT_EQ(Branch(0).turn_checkpoints(), 0U);
  ASSERT_TRUE(Select(1));
  std::vector<float> ignored;
  auto began = Branch(0).BeginPrompt(history, 0, false, true);
  ASSERT_TRUE(began);
  auto r = (*began)->Advance();
  while (r && !(*began)->done()) r = (*began)->Advance();
  const auto finished = (*began)->Finish();
  ASSERT_TRUE(r) << (r ? "" : r.error());
  ASSERT_TRUE(finished);
  EXPECT_EQ(Branch(0).history(), history);
  EXPECT_FALSE(Branch(0).spilled());
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}
TEST_P(Gemma4ServingGpu, LiteralTeacherForcingAndChatRefusalsUseTheTargetRoute) {
  auto literal = model->EncodeText("Hello, café. 世界");
  ASSERT_TRUE(literal);
  EXPECT_EQ(literal->front(), 2);
  ASSERT_TRUE(Select(1));
  std::vector<float> last;
  std::size_t at = 1;
  auto scored =
      Branch(0).ScorePrompt(*literal, last, [&](std::int32_t id, std::span<const float> row) {
        EXPECT_EQ(id, (*literal)[at++]);
        EXPECT_EQ(row.size(), 262144U);
        return true;
      });
  ASSERT_TRUE(scored) << (scored ? "" : scored.error());
  EXPECT_EQ(at, literal->size());
  llmp::chat::Conversation c;
  c.messages.push_back({.role = llmp::chat::Role::kUser,
                        .content = "Hello",
                        .reasoning_content = {},
                        .tool_calls = {}});
  model->Defaults(c);
  ASSERT_TRUE(model->RenderChat(c));
  c.enable_thinking = true;
  EXPECT_FALSE(model->RenderChat(c));
  c.enable_thinking = false;
  auto tool = llmp::base::json::Parse(R"({"type":"function","function":{"name":"test"}})");
  ASSERT_TRUE(tool);
  c.tools.push_back(tool->root());
  EXPECT_FALSE(model->RenderChat(c));
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, ActualPinnedTemplateNativeInterpreterAndRendererAgree) {
  auto artifact = llmp::artifact::Artifact::Open(roles.installed / *config.models[0].artifact);
  ASSERT_TRUE(artifact);
  auto assets = rt::ReadChatAssets(*artifact, config.models[0]);
  ASSERT_TRUE(assets);
  ASSERT_TRUE(assets->chat_template);
  const auto& text = *assets->chat_template;
  auto fixture = llmp::test_support::LoadJson("chat/gemma-26-serving.json");
  const auto root = fixture.root();
  EXPECT_EQ(llmp::base::ToHex(llmp::base::Sha256{}.Update(text).Finish()),
            llmp::test_support::Get(root, "template_sha256").string());
  const auto facts = llmp::chat::TokenFacts::From(model->tokenizer());
  auto native = llmp::chat::ChatTemplate::ForText(text, facts);
  ASSERT_TRUE(native);
  EXPECT_EQ(native->name(), "gemma-4-unsloth");
  EXPECT_EQ(native->how(), llmp::chat::ChatTemplate::How::kNativeByProbe);
  auto program = llmp::chat::jinja::Template::Parse(text);
  ASSERT_TRUE(program);
  const auto cases = llmp::test_support::Get(root, "cases");
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto c = cases.at(i);
    SCOPED_TRACE(llmp::test_support::Get(c, "name").string());
    std::deque<llmp::base::json::Document> arguments;
    const auto conversation = llmp::test_support::ConversationFrom(c, &arguments);
    const auto a = native->Render(conversation);
    const auto b = llmp::chat::RenderInterpreted(*program, conversation, facts, std::nullopt);
    if (const auto expected = c.find("sha256")) {
      ASSERT_TRUE(a) << (a ? "" : a.error().ToString());
      ASSERT_TRUE(b) << (b ? "" : b.error().ToString());
      EXPECT_EQ(a->text, b->text);
      std::vector<llmp::tokenizer::TokenId> native_ids, interpreted_ids;
      EXPECT_TRUE(model->tokenizer().EncodeMarked(a->text, a->specials, {}, native_ids));
      EXPECT_TRUE(model->tokenizer().EncodeMarked(b->text, b->specials, {}, interpreted_ids));
      EXPECT_EQ(native_ids, interpreted_ids);
      EXPECT_FALSE(native_ids.empty());
      if (!native_ids.empty()) EXPECT_EQ(native_ids.front(), 2);
      EXPECT_EQ(llmp::base::ToHex(llmp::base::Sha256{}.Update(a->text).Finish()),
                expected->string());
    } else {
      EXPECT_FALSE(a);
      EXPECT_FALSE(b);
    }
  }
}

TEST_P(Gemma4ServingGpu, MaximumLiteralScoringRowsInterleaveAndResumeWithoutRepeatingScores) {
  ASSERT_TRUE(Select(2));
  rt::GenerateOptions gen;
  gen.max_tokens = 4;
  gen.stop = false;
  gen.seed = 991;
  gen.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
  std::vector<float> last;
  ASSERT_TRUE(Branch(1).Prefill(prompt, last));
  rt::Generation expected;
  ASSERT_TRUE(Branch(1).Generate(last, gen, expected));
  ASSERT_TRUE(Branch(1).Clear());
  ASSERT_TRUE(Branch(1).Prefill(prompt, last));
  rt::Generation output;
  auto peer = Branch(1).BeginGeneration(last, gen, output);
  ASSERT_TRUE(peer);
  std::vector<std::int32_t> tokens(128);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
  std::size_t at = 1;
  const auto score = [&](std::int32_t id, std::span<const float> row) {
    EXPECT_EQ(id, tokens[at++]);
    EXPECT_EQ(row.size(), 262144U);
    EXPECT_TRUE(llmp::execution::ScoreToken(row, id, 16));
    return true;
  };
  auto scorer = Branch(0).BeginScoringPrompt(tokens, score);
  ASSERT_TRUE(scorer);
  auto exercised = [&]() -> rt::Status {
    while (Branch(0).history().size() < 64) {
      if (auto r = (*scorer)->Advance({}, true); !r) return r;
      if (!(*peer)->done() && Branch(0).history().size() % 16 == 0) {
        const std::array<rt::Llm::GenerationSession*, 1> one{peer->get()};
        if (auto r = model->RunGenerationWave(one); !r) return r;
      }
    }
    (*scorer)->Cancel();
    if (auto r = (*scorer)->Finish(); !r) return r;
    scorer->reset();
    auto resumed = Branch(0).BeginScoringPrompt(tokens, score, true);
    if (!resumed) return std::unexpected(resumed.error());
    *scorer = std::move(*resumed);
    while (!(*scorer)->done())
      if (auto r = (*scorer)->Advance({}, true); !r) return r;
    return {};
  }();
  if (*scorer) {
    (*scorer)->Cancel();
    EXPECT_TRUE((*scorer)->Finish());
  }
  (*peer)->Cancel();
  EXPECT_TRUE((*peer)->Finish());
  EXPECT_TRUE(exercised) << (exercised ? "" : exercised.error());
  EXPECT_EQ(at, tokens.size());
  EXPECT_EQ(Branch(0).history(), tokens);
  EXPECT_EQ(output.tokens, expected.tokens);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, SeededStreamResumesAfterStateSpillWhilePeerContinues) {
  ASSERT_TRUE(Select(2));
  rt::GenerateOptions options;
  options.max_tokens = 6;
  options.stop = false;
  options.seed = 881;
  options.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
  std::vector<float> last;
  ASSERT_TRUE(Branch(0).Prefill(prompt, last));
  rt::Generation expected;
  ASSERT_TRUE(Branch(0).Generate(last, options, expected));
  ASSERT_TRUE(Branch(0).Clear());
  ASSERT_TRUE(Branch(0).Prefill(prompt, last));
  rt::Generation out;
  std::vector<std::int32_t> visible;
  auto streamed = options;
  streamed.on_tokens = [&](std::span<const std::int32_t> ids) {
    visible.insert(visible.end(), ids.begin(), ids.end());
    return true;
  };
  auto paused = Branch(0).BeginGeneration(last, streamed, out);
  ASSERT_TRUE(paused);
  const std::array<rt::Llm::GenerationSession*, 1> one{paused->get()};
  const auto first = model->RunGenerationWave(one);
  (*paused)->Cancel();
  const auto settled = (*paused)->Finish();
  paused->reset();
  ASSERT_TRUE(first);
  ASSERT_TRUE(settled);
  const auto held = Branch(0).history();
  Branch(0).HoldContinuation();
  auto exercised = [&]() -> rt::Status {
    if (auto r = server->RetireRequestBranches(*model, true); !r.result) return r.result;
    if (auto r = model->SpillSetAside(Branch(0)); !r) return r;
    EXPECT_TRUE(Branch(0).spilled());
    if (auto r = Select(2); !r) return r;
    // A different owner progresses while the continuation's KV is absent.
    std::vector<float> peer_last;
    if (auto r = Branch(1).Prefill(prompt, peer_last); !r) return r;
    rt::Generation peer;
    if (auto r = Branch(1).Generate(peer_last, options, peer); !r) return r;
    EXPECT_EQ(peer.tokens, expected.tokens);
    auto restored = Branch(0).BeginPrompt(held, 0, false, true);
    if (!restored) return std::unexpected(restored.error());
    auto result = [&]() -> rt::Status {
      while (!(*restored)->done())
        if (auto r = (*restored)->Advance({}, true); !r) return r;
      return {};
    }();
    if (!result) (*restored)->Cancel();
    const auto finished = (*restored)->Finish();
    if (!result) return result;
    if (!finished) return finished;
    EXPECT_TRUE((*restored)->last().empty());
    auto resumed = Branch(0).ResumeGeneration((*restored)->last(), streamed, out);
    if (!resumed) return std::unexpected(resumed.error());
    const std::array<rt::Llm::GenerationSession*, 1> again{resumed->get()};
    auto continued = [&]() -> rt::Status {
      while (!(*resumed)->done())
        if (auto r = model->RunGenerationWave(again); !r) return r;
      return {};
    }();
    if (!continued) (*resumed)->Cancel();
    const auto ended = (*resumed)->Finish();
    return continued ? ended : continued;
  }();
  Branch(0).ReleaseContinuation();
  EXPECT_TRUE(exercised) << (exercised ? "" : exercised.error());
  EXPECT_EQ(out.tokens, expected.tokens);
  EXPECT_EQ(visible, expected.tokens);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, LargeTopKSamplingCapacityIsFundedAndRetiresOnEveryClose) {
  ASSERT_TRUE(Select(2));
  std::vector<float> last;
  ASSERT_TRUE(Branch(0).Prefill(prompt, last));
  for (const auto top_k : {200000U, 262143U}) {
    ASSERT_TRUE(Branch(0).Clear());
    ASSERT_TRUE(Branch(0).Prefill(prompt, last));
    rt::GenerateOptions options;
    options.max_tokens = 2;
    options.stop = false;
    options.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = top_k};
    options.seed = 456;
    rt::Generation out;
    auto opened = Branch(0).BeginGeneration(last, options, out);
    ASSERT_TRUE(opened);
    const auto bytes = 2ULL * 262144 * sizeof(llmp::execution::SamplingCandidate);
    EXPECT_EQ(Branch(0).sampling_scratch_bytes(), bytes);
    const std::array<rt::Llm::GenerationSession*, 1> one{opened->get()};
    const auto ran = model->RunGenerationWave(one);
    (*opened)->Cancel();
    const auto finished = (*opened)->Finish();
    EXPECT_TRUE(ran);
    EXPECT_TRUE(finished);
    EXPECT_EQ(Branch(0).sampling_scratch_bytes(), 0U);
    EXPECT_EQ(Branch(1).sampling_scratch_bytes(), 0U);
    // Cancellation after first sampling retires the same fixed capacity.
    ASSERT_TRUE(Branch(0).Clear());
    ASSERT_TRUE(Branch(0).Prefill(prompt, last));
    auto cancelled = Branch(0).BeginGeneration(last, options, out);
    ASSERT_TRUE(cancelled);
    EXPECT_EQ(Branch(0).sampling_scratch_bytes(), bytes);
    (*cancelled)->Cancel();
    EXPECT_TRUE((*cancelled)->Finish());
    EXPECT_EQ(Branch(0).sampling_scratch_bytes(), 0U);
    // Invalid first-row arithmetic is rejected after prepare reserved its
    // capacity; failed Begin must retire it without changing the KV ledger.
    auto bad = last;
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    const auto history = Branch(0).history();
    EXPECT_FALSE(Branch(0).BeginGeneration(bad, options, out));
    EXPECT_EQ(Branch(0).sampling_scratch_bytes(), 0U);
    EXPECT_EQ(Branch(0).history(), history);
  }
  EXPECT_GE(server->host_input_bytes(),
            12ULL * 262144 * (2 * sizeof(float) + 2 * sizeof(llmp::execution::SamplingCandidate)));
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, CompleteLikelihoodRowsMatchOwnedOneTokenFrontiersAndStopAtPrefix) {
  ASSERT_TRUE(Select(2));
  HostCopies copies(*this, 6ULL * 262144 * sizeof(float));
  ASSERT_TRUE(copies.funded);
  std::array<std::vector<float>, 6> rows;
  for (std::size_t i = 0; i < prompt.size(); ++i)
    ASSERT_TRUE(Branch(1).Prefill(std::span(prompt).subspan(i, 1), rows[i]));
  const auto peer_history = Branch(1).history();
  const auto peer_bytes = model->ResidentStateBytes(Branch(1));
  std::size_t at = 1;
  rt::PrefillRun run;
  std::vector<float> stopped;
  const auto score = Branch(0).ScorePrompt(
      prompt, stopped,
      [&](std::int32_t id, std::span<const float> row) {
        EXPECT_EQ(id, prompt[at]);
        EXPECT_TRUE(std::ranges::equal(row, rows[at - 1]));
        const auto actual = llmp::execution::ScoreToken(row, id, 16);
        const auto expected = llmp::execution::ScoreToken(rows[at - 1], prompt[at], 16);
        EXPECT_TRUE(actual);
        EXPECT_TRUE(expected);
        if (actual && expected) EXPECT_EQ(actual->logprob, expected->logprob);
        return ++at < 4;
      },
      {}, &run);
  ASSERT_TRUE(score);
  EXPECT_EQ(at, 4U);
  EXPECT_EQ(run.end, 3U);
  EXPECT_TRUE(run.stopped);
  EXPECT_TRUE(stopped.empty());
  EXPECT_EQ(Branch(0).history(), (std::vector<std::int32_t>(prompt.begin(), prompt.begin() + 3)));
  EXPECT_EQ(Branch(1).history(), peer_history);
  EXPECT_EQ(model->ResidentStateBytes(Branch(1)), peer_bytes);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, CallbackStopPublishesCompletedPeerAndReleasesEachSampler) {
  ASSERT_TRUE(Select(2));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  options.seed = 883;
  options.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
  std::vector<float> baseline;
  ASSERT_TRUE(Branch(1).Prefill(prompt, baseline));
  rt::Generation expected;
  ASSERT_TRUE(Branch(1).Generate(baseline, options, expected));
  ASSERT_TRUE(Branch(1).Clear());
  std::array<std::vector<float>, 2> last;
  std::array<rt::Generation, 2> output;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> sessions;
  std::vector<std::int32_t> visible;
  auto stopping = options;
  stopping.on_tokens = [&](std::span<const std::int32_t> ids) {
    visible.insert(visible.end(), ids.begin(), ids.end());
    return visible.size() < 2;
  };
  const auto exercised = [&]() -> rt::Status {
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      if (auto r = Branch(slot).Prefill(prompt, last[slot]); !r) return r;
      auto began =
          Branch(slot).BeginGeneration(last[slot], slot == 0 ? stopping : options, output[slot]);
      if (!began) return std::unexpected(began.error());
      sessions[slot] = std::move(*began);
    }
    std::array<rt::Llm::GenerationSession*, 2> work{sessions[0].get(), sessions[1].get()};
    if (auto r = model->RunGenerationWave(work); !r) return r;
    EXPECT_TRUE(sessions[0]->done());
    EXPECT_EQ(output[0].tokens.size(), 2U);
    // History publishes only when this completed session is finished.
    EXPECT_EQ(Branch(0).history().size(), prompt.size());
    const std::array<rt::Llm::GenerationSession*, 1> peer{sessions[1].get()};
    while (!sessions[1]->done())
      if (auto r = model->RunGenerationWave(peer); !r) return r;
    return {};
  }();
  for (auto& session : sessions)
    if (session) {
      session->Cancel();
      EXPECT_TRUE(session->Finish());
    }
  EXPECT_TRUE(exercised) << (exercised ? "" : exercised.error());
  EXPECT_EQ(Branch(0).history().size(), prompt.size() + 1);
  EXPECT_EQ(visible, output[0].tokens);
  EXPECT_EQ(output[1].tokens, expected.tokens);
  EXPECT_EQ(Branch(0).sampling_scratch_bytes(), 0U);
  EXPECT_EQ(Branch(1).sampling_scratch_bytes(), 0U);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, CrossVariantKeptTagIsRejectedBeforeAdoptionWhilePeerRestores) {
  ASSERT_TRUE(Select(2));
  std::vector<float> first, peer;
  ASSERT_TRUE(Branch(0).Prefill(prompt, first));
  ASSERT_TRUE(Branch(1).Prefill(prompt, peer));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  rt::Generation expected;
  ASSERT_TRUE(Branch(1).Generate(peer, options, expected));
  ASSERT_TRUE(Branch(1).Clear());
  ASSERT_TRUE(Branch(1).Prefill(prompt, peer));
  const auto peer_history = Branch(1).history();
  const auto layout = model->KeptLayout();
  ASSERT_FALSE(layout.empty());
  const auto retired = server->RetireRequestBranches(*model, true);
  ASSERT_TRUE(retired.result);
  ASSERT_TRUE(retired.references_retired);
  ASSERT_TRUE(model->SpillIdle(Branch(0)));
  ASSERT_TRUE(model->SpillIdle(Branch(1)));
  server->Persist(rt::Clock::now() + std::chrono::seconds(30), {});
  ASSERT_TRUE(RetireServer());
  server.reset();
  const auto path =
      roles.spill / "conversations" / *config.models[0].artifact / rt::kept::RecordFileName(0);
  std::ifstream input(path);
  auto record = rt::kept::Decode(std::string{std::istreambuf_iterator<char>(input), {}});
  ASSERT_TRUE(record);
  ASSERT_EQ(record->identity.layout, layout);
  // The other profile's real kept identity, taken from a server of its own
  // started on the same settings.
  const auto other = OtherProfileLayout();
  ASSERT_FALSE(other.empty());
  ASSERT_NE(other, layout);
  record->identity.layout = other;
  std::ofstream(path) << rt::kept::Encode(*record);
  ASSERT_TRUE(Start());
  EXPECT_TRUE(Branch(0).history().empty());
  EXPECT_FALSE(Branch(0).spilled());
  EXPECT_EQ(Branch(1).history(), peer_history);
  EXPECT_EQ(model->KeptLayout(), layout);
  ASSERT_TRUE(Select(2));
  auto restoring = Branch(1).BeginPrompt(peer_history, 0, false, true);
  ASSERT_TRUE(restoring);
  const auto restored = [&]() -> rt::Status {
    while (!(*restoring)->done())
      if (auto r = (*restoring)->Advance(); !r) return r;
    return {};
  }();
  if (!restored) (*restoring)->Cancel();
  const auto finished = (*restoring)->Finish();
  EXPECT_EQ((*restoring)->reused(), peer_history.size());
  restoring->reset();
  ASSERT_TRUE(restored) << (restored ? "" : restored.error());
  ASSERT_TRUE(finished) << (finished ? "" : finished.error());
  rt::Generation actual;
  ASSERT_TRUE(Branch(1).Generate(peer, options, actual));
  EXPECT_EQ(actual.tokens, expected.tokens);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, CapacityRefusalPreservesCompletedPrefixAndPeerProgress) {
  ASSERT_TRUE(Select(2));
  // The profiles have different KV row widths: these completed boundaries
  // each require new state extents for the next scalar decode.
  const std::uint32_t boundary = GetParam() == 26 ? 512 : 256;
  std::vector<std::int32_t> long_prompt(boundary);
  const auto through = model->StateBytesThrough(boundary);
  const auto past = model->StateBytesThrough(boundary + 1);
  ASSERT_TRUE(through && past);
  ASSERT_GT(*past, *through + (16U << 20U));
  for (std::size_t i = 0; i < long_prompt.size(); ++i) long_prompt[i] = prompt[i % prompt.size()];
  std::array<std::vector<float>, 2> last;
  ASSERT_TRUE(Branch(0).Prefill(long_prompt, last[0]));
  ASSERT_TRUE(Branch(1).Prefill(prompt, last[1]));
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  options.seed = 909;
  options.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
  rt::Generation peer_expected;
  ASSERT_TRUE(Branch(1).Generate(last[1], options, peer_expected));
  ASSERT_TRUE(Branch(1).Clear());
  ASSERT_TRUE(Branch(1).Prefill(prompt, last[1]));
  const auto bytes = model->ResidentStateBytes(Branch(0));
  std::array<rt::Generation, 2> output;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> sessions;
  const auto exercised = [&]() -> rt::Status {
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      auto began = Branch(slot).BeginGeneration(last[slot], options, output[slot]);
      if (!began) return std::unexpected(began.error());
      sessions[slot] = std::move(*began);
    }
    std::array<rt::Llm::GenerationSession*, 2> work{sessions[0].get(), sessions[1].get()};
    std::vector<llmp::memory::ReclaimCandidate> candidates;
    model->ReclaimCandidates(0, true, candidates);
    for (const auto& c : candidates) {
      if (c.kind == llmp::memory::ReclaimKind::kGraph) {
        model->Reclaim(c.kind, c.id);
      }
    }
    const auto available = server->node().FreeBytes();
    if (!available || *available <= (16U << 20U))
      return std::unexpected("test capacity unavailable");
    const auto pressure = *available - (16U << 20U);
    if (!server->node().ChargeHost(pressure, false))
      return std::unexpected("test pressure refused");
    const auto pressured = model->RunGenerationWave(work, true);
    server->node().UnchargeHost(pressure);
    if (!pressured) return pressured;
    EXPECT_TRUE(sessions[0]->refused());
    EXPECT_FALSE(sessions[0]->done());
    EXPECT_EQ(Branch(0).history(), long_prompt);
    EXPECT_EQ(model->ResidentStateBytes(Branch(0)), bytes);
    EXPECT_EQ(output[1].tokens.size(), 2U);
    while (!sessions[0]->done() || !sessions[1]->done()) {
      std::array<rt::Llm::GenerationSession*, 2> active{};
      std::size_t count = 0;
      for (const auto& session : sessions)
        if (!session->done()) active[count++] = session.get();
      if (auto r = model->RunGenerationWave(std::span(active).first(count), true); !r) return r;
    }
    return {};
  }();
  for (auto& session : sessions)
    if (session) {
      session->Cancel();
      EXPECT_TRUE(session->Finish());
    }
  EXPECT_TRUE(exercised) << (exercised ? "" : exercised.error());
  EXPECT_EQ(output[1].tokens, peer_expected.tokens);
  ASSERT_TRUE(Branch(0).Clear());
  ASSERT_TRUE(Branch(0).Prefill(long_prompt, last[0]));
  rt::Generation target_expected;
  ASSERT_TRUE(Branch(0).Generate(last[0], options, target_expected));
  EXPECT_EQ(output[0].tokens, target_expected.tokens);
  EXPECT_EQ(Branch(0).sampling_scratch_bytes(), 0U);
  EXPECT_EQ(Branch(1).sampling_scratch_bytes(), 0U);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

TEST_P(Gemma4ServingGpu, PausedGenerationSurvivesOppositeProfileSwitchAndOwnedOutput) {
  auto opposite = config.models[0];
  opposite.name = "other";
  opposite.artifact = Published(OtherName(GetParam()));
  if (opposite.artifact->empty()) GTEST_FAIL() << "no published v1 " << OtherName(GetParam());
  ASSERT_TRUE(RetireServer());
  server.reset();
  opposite.overrides["max_slots"] = std::int64_t{2};
  config.models.push_back(opposite);
  ASSERT_TRUE(Start());
  auto* other = dynamic_cast<rt::Llm*>(server->Find("other"));
  ASSERT_NE(other, nullptr);
  EXPECT_NE(model->KeptLayout(), other->KeptLayout());
  ASSERT_TRUE(Select(1));
  rt::GenerateOptions options;
  options.max_tokens = 6;
  options.stop = false;
  options.seed = 1909;
  options.sampling = llmp::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
  std::vector<float> last;
  ASSERT_TRUE(Branch(0).Prefill(prompt, last));
  rt::Generation expected;
  ASSERT_TRUE(Branch(0).Generate(last, options, expected));
  ASSERT_TRUE(Branch(0).Clear());
  ASSERT_TRUE(Branch(0).Prefill(prompt, last));
  rt::Generation output;
  std::vector<std::int32_t> visible;
  auto streamed = options;
  streamed.on_tokens = [&](std::span<const std::int32_t> tokens) {
    visible.insert(visible.end(), tokens.begin(), tokens.end());
    return true;
  };
  auto paused = Branch(0).BeginGeneration(last, streamed, output);
  ASSERT_TRUE(paused);
  const std::array<rt::Llm::GenerationSession*, 1> one{paused->get()};
  const auto ran = model->RunGenerationWave(one);
  (*paused)->Cancel();
  const auto ended = (*paused)->Finish();
  paused->reset();
  ASSERT_TRUE(ran);
  ASSERT_TRUE(ended);
  const auto held = Branch(0).history();
  Branch(0).HoldContinuation();
  const auto exercised = [&]() -> rt::Status {
    const auto retired = server->RetireRequestBranches(*model, true);
    if (!retired.result) return retired.result;
    if (!retired.references_retired) return std::unexpected("references remain unproven");
    rt::SwapParts outgoing;
    if (auto r = server->Activate(*other, outgoing); !r) return r;
    if (auto r = server->FinishSwap(outgoing); !r) return r;
    auto other_branch = other->branch(0);
    if (!other_branch) return std::unexpected(other_branch.error());
    const std::array<rt::Llm::Branch*, 1> selected{*other_branch};
    if (auto r = server->SelectRequestBranches(*other, selected); !r) return r;
    std::vector<float> other_last;
    if (auto r = (*other_branch)->Prefill(prompt, other_last); !r) return r;
    rt::Generation other_output;
    if (auto r = (*other_branch)->Generate(other_last, options, other_output); !r) return r;
    const auto other_retired = server->RetireRequestBranches(*other, true);
    if (!other_retired.result) return other_retired.result;
    if (!other_retired.references_retired) return std::unexpected("opposite references unproven");
    rt::SwapParts incoming;
    if (auto r = server->Activate(*model, incoming); !r) return r;
    if (auto r = server->FinishSwap(incoming); !r) return r;
    if (auto r = Select(1); !r) return r;
    auto prompt_session = Branch(0).BeginPrompt(held, 0, false, true);
    if (!prompt_session) return std::unexpected(prompt_session.error());
    auto restored = [&]() -> rt::Status {
      while (!(*prompt_session)->done())
        if (auto r = (*prompt_session)->Advance({}, true); !r) return r;
      return {};
    }();
    if (!restored) (*prompt_session)->Cancel();
    const auto finished = (*prompt_session)->Finish();
    if (!restored) return restored;
    if (!finished) return finished;
    auto resumed = Branch(0).ResumeGeneration((*prompt_session)->last(), streamed, output);
    if (!resumed) return std::unexpected(resumed.error());
    const std::array<rt::Llm::GenerationSession*, 1> active{resumed->get()};
    auto continued = [&]() -> rt::Status {
      while (!(*resumed)->done())
        if (auto r = model->RunGenerationWave(active); !r) return r;
      return {};
    }();
    if (!continued) (*resumed)->Cancel();
    const auto closed = (*resumed)->Finish();
    return continued ? closed : continued;
  }();
  Branch(0).ReleaseContinuation();
  EXPECT_TRUE(exercised) << (exercised ? "" : exercised.error());
  EXPECT_EQ(output.tokens, expected.tokens);
  EXPECT_EQ(visible, expected.tokens);
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}

INSTANTIATE_TEST_SUITE_P(ApprovedProfiles, Gemma4ServingGpu, ::testing::Values(26U, 31U),
                         [](const auto& info) { return "Gemma" + std::to_string(info.param); });
