// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The production adapter and driver over the approved native artifact.
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/gemma4_runner.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"

namespace rt = jitllm::runtime;
namespace en = jitllm::engine;
namespace cfg = jitllm::config;
class GemmaJoinedGpu : public ::testing::TestWithParam<std::uint32_t> {
 protected:
  void SetUp() override {
    std::string name = "/home/pmeenan/.cache/jitllm-gemma-serving-XXXXXX";
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    roles.installed = "/home/pmeenan/.local/share/jitllm/m3-artifacts";
    roles.spill = scratch / "spill";
    roles.state = scratch / "state";
    std::filesystem::create_directory(roles.spill);
    std::filesystem::create_directory(roles.state);
    ASSERT_EQ(chmod(roles.spill.c_str(), 0700), 0);
    ASSERT_EQ(chmod(roles.state.c_str(), 0700), 0);
    cfg::ModelEntry entry;
    entry.name = "gemma";
    entry.artifact = GetParam() == 26
                         ? "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3"
                         : "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08";
    entry.overrides["context"] = std::int64_t{4096};
    entry.overrides["prefill_chunk"] = std::int64_t{16};
    entry.overrides["max_slots"] = std::int64_t{12};
    config.models.push_back(entry);
    options.keep_conversations = true;
    options.gemma_joined = true;
    options.gemma_row_invariant = true;
    ASSERT_TRUE(Start());
  }
  bool Start() {
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
    std::filesystem::remove_all(scratch);
  }
  struct HostCopies {
    HostCopies(en::PagedNode& owner, std::uint64_t capacity)
        : node(owner), bytes(capacity), funded(owner.ChargeHost(capacity, false)) {}
    ~HostCopies() {
      if (funded) node.UnchargeHost(bytes);
    }
    HostCopies(const HostCopies&) = delete;
    HostCopies& operator=(const HostCopies&) = delete;
    en::PagedNode& node;
    std::uint64_t bytes;
    bool funded;
  };
  en::Gemma4Runner& Runner() { return dynamic_cast<en::Gemma4Runner&>(model->paged()); }
  rt::Status StateBytes(std::uint32_t slot, std::uint32_t past, std::vector<std::byte>& bytes) {
    auto ranges = Runner().CheckpointRanges(past);
    if (!ranges) return std::unexpected(ranges.error());
    std::uint64_t count = 0;
    for (const auto& r : *ranges) count += r.bytes;
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = server->node().Pinned(count, 0, staging);
    if (!pinned) return std::unexpected(pinned.error());
    auto copied = Runner().CopyState(slot, *pinned, *ranges, true);
    if (copied) {
      const auto* begin = static_cast<const std::byte*>(*pinned);
      bytes.assign(begin, begin + count);
    }
    auto freed = server->node().FreePinned(*pinned);
    return copied ? freed : copied;
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
  rt::Llm* model = nullptr;
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
};
TEST_P(GemmaJoinedGpu, BoundedUnequalOwnersMatchSoloCompleteHeadsAndInitializedState) {
  constexpr std::array<std::uint32_t, 12> ids{1, 4, 7, 11, 0, 3, 6, 10, 2, 5, 8, 9};
  for (const std::size_t count : {1U, 2U, 4U, 8U, 12U}) {
    std::array<rt::Llm::Branch*, 12> active{};
    for (std::size_t i = 0; i < count; ++i) active[i] = &Branch(ids[i]);
    ASSERT_TRUE(server->SelectRequestBranches(*model, std::span(active).first(count)));
    auto ranges = Runner().CheckpointRanges(20);
    ASSERT_TRUE(ranges);
    std::uint64_t state_capacity = 0;
    for (const auto& range : *ranges) state_capacity += range.bytes;
    // Two sets of 48 retained heads, twelve working frontiers and one
    // temporary frontier; twelve saved initialized states and one current copy.
    HostCopies copies(server->node(), 109ULL * 262144 * sizeof(float) + 13 * state_capacity);
    ASSERT_TRUE(copies.funded);
    rt::GenerateOptions generate;
    generate.max_tokens = 4;
    generate.stop = false;
    generate.keep_logits = true;
    std::array<std::vector<std::int32_t>, 12> prompts;
    std::array<rt::Generation, 12> expected;
    std::array<std::vector<std::byte>, 12> states;
    for (std::size_t i = 0; i < count; ++i) {
      prompts[i].assign(prompt.begin(), prompt.end());
      for (std::size_t j = 0; j < i; ++j) prompts[i].push_back(563);
      ASSERT_TRUE(Branch(ids[i]).Clear());
      std::vector<float> last;
      ASSERT_TRUE(Branch(ids[i]).Prefill(prompts[i], last));
      ASSERT_TRUE(Branch(ids[i]).Generate(last, generate, expected[i]));
      ASSERT_EQ(expected[i].logits.size(), 4U);
      ASSERT_TRUE(StateBytes(ids[i], static_cast<std::uint32_t>(prompts[i].size()) + 3, states[i]));
    }
    for (const auto repeat : {0U, 1U}) {
      std::array<rt::Generation, 12> output;
      std::array<std::unique_ptr<rt::Llm::GenerationSession>, 12> sessions;
      std::array<rt::Llm::GenerationSession*, 12> work{};
      auto exercised = [&]() -> rt::Status {
        for (std::size_t i = 0; i < count; ++i) {
          if (auto r = Branch(ids[i]).Clear(); !r) return r;
          std::vector<float> last;
          if (auto r = Branch(ids[i]).Prefill(prompts[i], last); !r) return r;
          auto began = Branch(ids[i]).BeginGeneration(last, generate, output[i]);
          if (!began) return std::unexpected(began.error());
          sessions[i] = std::move(*began);
          work[i] = sessions[i].get();
        }
        for (unsigned step = 0; step < 3; ++step) {
          if (auto r = model->RunGenerationWave(std::span(work).first(count)); !r) return r;
          EXPECT_EQ(Runner().last_built_policy().rows, count == 12 ? 4U : count);
          EXPECT_EQ(Runner().last_built_policy().segments, count == 12 ? 4U : count);
          EXPECT_GT(Runner().last_built_policy().row_products, 0U);
        }
        return {};
      }();
      for (std::size_t i = 0; i < count; ++i) {
        if (!sessions[i]) continue;
        sessions[i]->Cancel();
        EXPECT_TRUE(sessions[i]->Finish());
        sessions[i].reset();
        EXPECT_EQ(output[i].tokens, expected[i].tokens)
            << "owner=" << ids[i] << " repeat=" << repeat;
        ASSERT_EQ(output[i].logits.size(), expected[i].logits.size());
        for (std::size_t row = 0; row < output[i].logits.size(); ++row) {
          ASSERT_EQ(output[i].logits[row].size(), expected[i].logits[row].size());
          EXPECT_EQ(std::memcmp(output[i].logits[row].data(), expected[i].logits[row].data(),
                                expected[i].logits[row].size() * sizeof(float)),
                    0)
              << "owner=" << ids[i] << " row=" << row << " repeat=" << repeat;
        }
        std::vector<std::byte> state;
        ASSERT_TRUE(StateBytes(ids[i], static_cast<std::uint32_t>(prompts[i].size()) + 3, state));
        EXPECT_EQ(state, states[i]) << "owner=" << ids[i] << " repeat=" << repeat;
      }
      ASSERT_TRUE(exercised) << (exercised ? "" : exercised.error());
    }
    const auto retired = server->RetireRequestBranches(*model, true);
    ASSERT_TRUE(retired.result);
    ASSERT_TRUE(retired.references_retired);
  }
  EXPECT_GT(model->graphs().captured, 0U);
  EXPECT_GT(model->graphs().replayed, 0U);
  EXPECT_TRUE(model->violations().empty());
}
TEST_P(GemmaJoinedGpu, CallbackStopAndIndependentCancelPreserveOtherOwnersAndReleaseSamplers) {
  ASSERT_TRUE(Select(12));
  HostCopies copies(server->node(), 13ULL * 262144 * sizeof(float));
  ASSERT_TRUE(copies.funded);
  rt::GenerateOptions gen;
  gen.max_tokens = 4;
  gen.stop = false;
  gen.seed = 883;
  gen.sampling = jitllm::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
  std::vector<float> baseline;
  ASSERT_TRUE(Branch(11).Prefill(prompt, baseline));
  rt::Generation expected;
  ASSERT_TRUE(Branch(11).Generate(baseline, gen, expected));
  ASSERT_TRUE(Branch(11).Clear());
  std::array<std::vector<float>, 12> last;
  std::array<rt::Generation, 12> output;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 12> sessions;
  std::vector<std::int32_t> visible;
  auto stopping = gen;
  stopping.on_tokens = [&](std::span<const std::int32_t> ids) {
    visible.insert(visible.end(), ids.begin(), ids.end());
    return visible.size() < 2;
  };
  const auto exercised = [&]() -> rt::Status {
    for (std::uint32_t i = 0; i < 12; ++i) {
      if (auto r = Branch(i).Prefill(prompt, last[i]); !r) return r;
      auto began = Branch(i).BeginGeneration(last[i], i == 0 ? stopping : gen, output[i]);
      if (!began) return std::unexpected(began.error());
      sessions[i] = std::move(*began);
    }
    std::array<rt::Llm::GenerationSession*, 12> work{};
    for (std::size_t i = 0; i < 12; ++i) work[i] = sessions[i].get();
    if (auto r = model->RunGenerationWave(work); !r) return r;
    EXPECT_TRUE(sessions[0]->done());
    EXPECT_EQ(output[0].tokens.size(), 2U);
    EXPECT_EQ(Branch(0).history().size(), prompt.size());
    // Remove an owner from the second subwave independently after a completed
    // unit. Both the stopped owner and canceled owner retain their own prefix.
    sessions[8]->Cancel();
    for (;;) {
      std::size_t count = 0;
      for (std::size_t i = 1; i < 12; ++i)
        if (i != 8 && !sessions[i]->done()) work[count++] = sessions[i].get();
      if (count == 0) break;
      if (auto r = model->RunGenerationWave(std::span(work).first(count)); !r) return r;
    }
    return {};
  }();
  for (auto& session : sessions)
    if (session) {
      session->Cancel();
      EXPECT_TRUE(session->Finish());
      session.reset();
    }
  ASSERT_TRUE(exercised) << (exercised ? "" : exercised.error());
  EXPECT_EQ(visible, output[0].tokens);
  EXPECT_EQ(output[8].tokens.size(), 2U);
  for (std::uint32_t i = 0; i < 12; ++i) {
    EXPECT_EQ(Branch(i).sampling_scratch_bytes(), 0U);
    EXPECT_EQ(Branch(i).history().size(), prompt.size() + ((i == 0 || i == 8) ? 1 : 3));
    if (i != 0 && i != 8) EXPECT_EQ(output[i].tokens, expected.tokens) << i;
  }
  const auto retired = server->RetireRequestBranches(*model, true);
  ASSERT_TRUE(retired.result);
  ASSERT_TRUE(retired.references_retired);
}
TEST_P(GemmaJoinedGpu, CompleteLikelihoodRowsMatchOwnedOneTokenFrontiersAndStopAtPrefix) {
  ASSERT_TRUE(Select(2));
  HostCopies copies(server->node(), 6ULL * 262144 * sizeof(float));
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
        const auto actual = jitllm::execution::ScoreToken(row, id, 16);
        const auto expected = jitllm::execution::ScoreToken(rows[at - 1], prompt[at], 16);
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

TEST_P(GemmaJoinedGpu, JoinedSnapshotSpillAndRestartPreserveExactOwnedContinuations) {
  ASSERT_TRUE(Select(2));
  auto ranges = Runner().CheckpointRanges(12);
  ASSERT_TRUE(ranges);
  std::uint64_t state_capacity = 0;
  for (const auto& range : *ranges) state_capacity += range.bytes;
  auto copies = std::make_unique<HostCopies>(server->node(),
                                             20ULL * 262144 * sizeof(float) + 3 * state_capacity);
  ASSERT_TRUE(copies->funded);
  std::array<std::vector<float>, 2> last;
  for (std::uint32_t i = 0; i < 2; ++i) ASSERT_TRUE(Branch(i).Prefill(prompt, last[i]));
  struct Snapshots {
    en::PagedNode& node;
    std::array<void*, 2> buffers{};
    std::array<std::uint64_t, 2> capacities{};
    ~Snapshots() {
      for (void* buffer : buffers)
        if (buffer) EXPECT_TRUE(node.FreePinned(buffer));
    }
  };
  auto snapshots = std::make_unique<Snapshots>(server->node());
  for (std::uint32_t i = 0; i < 2; ++i) {
    std::vector<jitllm::catalog::ExtentId> staging;
    auto buffer = server->node().Pinned(Branch(i).state_snapshot_bytes(), 0, staging);
    ASSERT_TRUE(buffer);
    snapshots->buffers[i] = *buffer;
    snapshots->capacities[i] = Branch(i).state_snapshot_bytes();
    ASSERT_TRUE(Branch(i).SaveState(*buffer));
  }
  rt::GenerateOptions gen;
  gen.max_tokens = 4;
  gen.stop = false;
  gen.keep_logits = true;
  std::array<std::vector<std::int32_t>, 2> resumed_publications;
  const auto joined = [&](std::array<rt::Generation, 2>& output,
                          bool resume = false) -> rt::Status {
    std::array<rt::GenerateOptions, 2> options{gen, gen};
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> sessions;
    std::array<rt::Llm::GenerationSession*, 2> work{};
    const auto run = [&]() -> rt::Status {
      for (std::uint32_t i = 0; i < 2; ++i) {
        if (resume)
          options[i].on_tokens = [&, i](std::span<const std::int32_t> tokens) {
            resumed_publications[i].insert(resumed_publications[i].end(), tokens.begin(),
                                           tokens.end());
            return true;
          };
        auto began = resume ? Branch(i).ResumeGeneration(last[i], options[i], output[i])
                            : Branch(i).BeginGeneration(last[i], options[i], output[i]);
        if (!began) return std::unexpected(began.error());
        sessions[i] = std::move(*began);
        work[i] = sessions[i].get();
      }
      for (unsigned step = 0; step < 3; ++step)
        if (auto r = model->RunGenerationWave(work); !r) return r;
      return {};
    }();
    rt::Status finished;
    for (auto& session : sessions)
      if (session) {
        session->Cancel();
        auto r = session->Finish();
        if (!r) finished = r;
      }
    return run ? finished : run;
  };
  std::array<rt::Generation, 2> first, restored;
  ASSERT_TRUE(joined(first));
  std::array<std::vector<std::byte>, 2> states;
  for (std::uint32_t i = 0; i < 2; ++i) ASSERT_TRUE(StateBytes(i, 9, states[i]));
  for (std::uint32_t i = 0; i < 2; ++i) ASSERT_TRUE(Branch(i).RestoreState(snapshots->buffers[i]));
  ASSERT_TRUE(joined(restored));
  for (std::uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(first[i].tokens, restored[i].tokens);
    ASSERT_EQ(first[i].logits.size(), restored[i].logits.size());
    for (std::size_t row = 0; row < first[i].logits.size(); ++row)
      EXPECT_EQ(std::memcmp(first[i].logits[row].data(), restored[i].logits[row].data(),
                            262144 * sizeof(float)),
                0);
    std::vector<std::byte> state;
    ASSERT_TRUE(StateBytes(i, 9, state));
    EXPECT_EQ(state, states[i]);
  }
  std::array<std::vector<std::int32_t>, 2> held;
  std::array<rt::Generation, 2> expected, continued;
  for (std::uint32_t i = 0; i < 2; ++i) {
    ASSERT_LE(Branch(i).state_snapshot_bytes(), snapshots->capacities[i]);
    ASSERT_TRUE(Branch(i).SaveState(snapshots->buffers[i]));
    ASSERT_FALSE(restored[i].tokens.empty());
    continued[i].tokens = {restored[i].tokens.back()};
  }
  gen.keep_logits = false;
  for (std::uint32_t i = 0; i < 2; ++i) {
    held[i] = Branch(i).history();
    last[i] = restored[i].logits.back();
    ASSERT_TRUE(Branch(i).Generate(last[i], gen, expected[i]));
  }
  for (std::uint32_t i = 0; i < 2; ++i) ASSERT_TRUE(Branch(i).RestoreState(snapshots->buffers[i]));
  const auto retired = server->RetireRequestBranches(*model, true);
  ASSERT_TRUE(retired.result);
  ASSERT_TRUE(retired.references_retired);
  for (std::uint32_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(model->SpillIdle(Branch(i)));
    EXPECT_TRUE(Branch(i).spilled());
  }
  server->Persist(rt::Clock::now() + std::chrono::seconds(30), {});
  // Retained host rows belong to this node budget. Release their storage and
  // grant before replacing the node, then fund the new frontier copies.
  first = {};
  restored = {};
  states = {};
  last = {};
  copies.reset();
  snapshots.reset();
  ASSERT_TRUE(RetireServer());
  server.reset();
  ASSERT_TRUE(Start());
  copies = std::make_unique<HostCopies>(server->node(), 2ULL * 262144 * sizeof(float));
  ASSERT_TRUE(copies->funded);
  ASSERT_TRUE(Select(2));
  for (std::uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(Branch(i).history(), held[i]);
    auto restoring = Branch(i).BeginPrompt(held[i], 0, false, true);
    ASSERT_TRUE(restoring);
    const auto run = [&]() -> rt::Status {
      while (!(*restoring)->done())
        if (auto r = (*restoring)->Advance(); !r) return r;
      return {};
    }();
    if (!run) (*restoring)->Cancel();
    const auto finished = (*restoring)->Finish();
    ASSERT_TRUE(run);
    ASSERT_TRUE(finished);
    EXPECT_EQ((*restoring)->reused(), held[i].size());
    EXPECT_TRUE((*restoring)->last().empty());
    last[i] = (*restoring)->last();
    restoring->reset();
  }
  const auto ran = joined(continued, true);
  ASSERT_TRUE(ran) << (ran ? "" : ran.error());
  for (std::uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(continued[i].tokens, expected[i].tokens);
    EXPECT_EQ(resumed_publications[i], (std::vector<std::int32_t>(expected[i].tokens.begin() + 1,
                                                                  expected[i].tokens.end())));
  }
  EXPECT_TRUE(server->RetireRequestBranches(*model, true).references_retired);
}
TEST_P(GemmaJoinedGpu, IndependentPrepareRefusalPreservesCompletedPrefixAndPeerProgress) {
  ASSERT_TRUE(Select(2));
  // The profiles have different KV row widths: these completed boundaries
  // each require new state extents for the next scalar decode.
  const std::uint32_t boundary = GetParam() == 26 ? 512 : 256;
  std::vector<std::int32_t> long_prompt(boundary);
  ASSERT_GT(model->StateBytesThrough(boundary + 1),
            model->StateBytesThrough(boundary) + (16U << 20U));
  for (std::size_t i = 0; i < long_prompt.size(); ++i) long_prompt[i] = prompt[i % prompt.size()];
  std::array<std::vector<float>, 2> last;
  ASSERT_TRUE(Branch(0).Prefill(long_prompt, last[0]));
  ASSERT_TRUE(Branch(1).Prefill(prompt, last[1]));
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  options.seed = 909;
  options.sampling = jitllm::execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
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

INSTANTIATE_TEST_SUITE_P(ApprovedProfiles, GemmaJoinedGpu, ::testing::Values(26U, 31U));
