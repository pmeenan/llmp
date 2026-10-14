// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/dsv4_runner.h"
#include "model/dsv4.h"
#include "plain_token_serving_checks.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"

namespace rt = llmp::runtime;
namespace en = llmp::engine;
namespace cfg = llmp::config;

class PlainTokensServingGpu : public ::testing::TestWithParam<std::uint32_t> {
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
  void SetUp() override {
    // Case 4 (FourSlotDsparkGpu): DeepSeek with DSpark at four request slots
    // and a context whose end is past HCA's 256 compressed cells.
    const bool adaptive = GetParam() >= 3, four = GetParam() == 4;
    const std::filesystem::path models = llmp::test_support::ModelsDir();
    const auto installed = models / "m3-artifacts";
    const char* artifact = "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
    const char* drafter = "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5";
    std::error_code error;
    if (!std::filesystem::exists(installed / artifact, error) ||
        (adaptive && !std::filesystem::exists(installed / drafter, error))) {
      GTEST_SKIP() << "no artifact for case " << GetParam() << " in " << models;
    }
    // Spill files need direct I/O: scratch in the build tree.
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path root = base != nullptr ? base : ::testing::TempDir();
    std::filesystem::create_directories(root, error);
    std::string name = (root / "plain-tokens-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    life->roles.installed = installed;
    life->roles.spill = scratch / "spill";
    life->roles.state = scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      ASSERT_TRUE(std::filesystem::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    cfg::ModelEntry entry;
    entry.name = "target";
    entry.artifact = artifact;
    if (adaptive) {
      entry.drafter = drafter;
      entry.overrides["wave_form"] = std::string("plain");
    }
    entry.overrides["context"] = std::int64_t{four ? 65536 : 8704};
    entry.overrides["prefill_chunk"] = std::int64_t{4096};
    entry.overrides["max_slots"] = std::int64_t{four ? 4 : 2};
    life->config.models.push_back(entry);
    life->options.plain = !adaptive;
    ASSERT_FALSE(life->options.diagnostic_plain_device_tokens.has_value());
    life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, stderr);
    const auto started = life->server->Start(true);
    ASSERT_TRUE(started) << (started ? "" : started.error());
    model = dynamic_cast<rt::Llm*>(life->server->Find("target"));
    ASSERT_NE(model, nullptr);
    rt::SwapParts parts;
    const auto activated = life->server->Activate(*model, parts);
    ASSERT_TRUE(activated) << (activated ? "" : activated.error());
  }
  void TearDown() override {
    if (life->server) {
      const auto retired = life->server->TearDown();
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      if (!retired) {
        std::ignore = life.release();
        return;
      }
      life->server.reset();
    }
    if (!scratch.empty()) std::filesystem::remove_all(scratch);
  }
  template <class Runner>
  std::expected<llmp::base::Sha256Digest, std::string> StateHash(Runner& runner, std::uint32_t id) {
    auto slot = runner.request_slot(id);
    if (!slot) return std::unexpected(slot.error());
    auto& node = life->server->node();
    constexpr std::uint64_t capacity = 1ULL << 20U;
    std::vector<llmp::catalog::ExtentId> staging;
    auto buffer = node.Pinned(capacity, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    llmp::base::Sha256 hash;
    std::uint64_t bytes = 0;
    for (const auto& range : (*slot)->used_state_ranges()) {
      bytes += range.bytes;
      for (std::uint64_t at = 0; at < range.bytes; at += capacity) {
        const en::LiveState::Range part{range.region, range.offset + at,
                                        std::min(capacity, range.bytes - at)};
        if (auto r = (*slot)->SaveUsedState(*buffer, std::span(&part, 1)); !r) {
          // An error is not completion proof; keep its host destination.
          node.KeepPinned(*buffer);
          return std::unexpected(r.error());
        }
        hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
      }
    }
    if (auto r = node.FreePinned(*buffer); !r) return std::unexpected(r.error());
    if (bytes == 0) return std::unexpected("complete initialized state required");
    return hash.Finish();
  }
};

TEST_P(PlainTokensServingGpu, ScalarJoinedMixedAndSpilledRowsMatchDeviceTokens) {
  const auto exercise = [&](auto& runner) {
    return llmp::test::CheckPlainServing(
        *life->server, *model, runner.vocab(),
        [&](std::uint32_t id) { return StateHash(runner, id); },
        [&] { return runner.device_token_outputs(); },
        GetParam() == 3 ? std::array<std::uint64_t, 2>{0, 4} : std::array<std::uint64_t, 2>{3, 5});
  };
  const auto r = exercise(dynamic_cast<en::Dsv4Runner&>(model->paged()));
  ASSERT_TRUE(r) << (r ? "" : r.error());
  if (GetParam() == 3) {
    // The ordinary scalar path still speculates after C2 -> C1 departure.
    // Now consume features injected by token-only C2 through a real joined
    // draft/verify, comparing complete state (including the wrapped ring).
    auto& runner = dynamic_cast<en::Dsv4Runner&>(model->paged());
    ASSERT_TRUE(runner.speculative());
    ASSERT_GT(runner.drafter_state_bytes(), 0U);
    ASSERT_TRUE(life->server->node().ChargeHost(64ULL << 20U, false));
    struct Charge {
      en::PagedNode& node;
      ~Charge() { node.UnchargeHost(64ULL << 20U); }
    } charge{life->server->node()};
    std::array<en::Dsv4Runner::Slot*, 2> slots{};
    std::array<rt::Llm::Branch*, 2> branches{};
    for (std::uint32_t i = 0; i < 2; ++i) {
      auto slot = runner.request_slot(i);
      ASSERT_TRUE(slot);
      slots[i] = *slot;
      auto branch = model->branch(i);
      ASSERT_TRUE(branch);
      branches[i] = *branch;
    }
    ASSERT_TRUE(life->server->SelectRequestBranches(*model, branches));
    std::array<std::vector<std::int32_t>, 2> expected_drafts;
    std::array<std::vector<float>, 2> expected_heads;
    std::array<llmp::base::Sha256Digest, 2> expected_state{}, expected_settled{};
    const auto finite = [&](const std::vector<float>& rows, std::uint32_t count) {
      return rows.size() == std::size_t{count} * runner.vocab() &&
             std::ranges::all_of(rows, [](float x) { return std::isfinite(x); });
    };
    for (const bool device : {false, true}) {
      std::array<std::uint32_t, 2> past{257, 258};
      std::array<std::int32_t, 2> anchors{};
      std::array<std::vector<float>, 2> rows;
      for (std::uint32_t i = 0; i < 2; ++i) {
        ASSERT_TRUE(slots[i]->Clear());
        std::vector<std::int32_t> seed(past[i]);
        constexpr std::array<std::int32_t, 7> actual{2, 818, 5279, 529, 7001, 563, 42};
        for (std::size_t j = 0; j < seed.size(); ++j) seed[j] = actual[(j + i) % actual.size()];
        ASSERT_TRUE(slots[i]->Chunk(0, seed, rows[i], en::Dsv4ChunkKind::kInject));
        ASSERT_TRUE(finite(rows[i], 1));
        anchors[i] = static_cast<std::int32_t>(std::ranges::max_element(rows[i]) - rows[i].begin());
      }
      const auto before = runner.device_token_outputs();
      const auto replayed = runner.wave_stats().replayed;
      for (unsigned step = 0; step < 3; ++step) {
        std::array<std::int32_t, 2> chosen{};
        std::array<en::Dsv4Runner::WaveWork, 2> work;
        for (std::uint32_t i = 0; i < 2; ++i)
          work[i] = {.slot = slots[i],
                     .pos = past[i],
                     .anchor = anchors[i],
                     .logits = device ? nullptr : &rows[i],
                     .token = device ? &chosen[i] : nullptr};
        ASSERT_TRUE(runner.DecodeWave(work));
        for (std::uint32_t i = 0; i < 2; ++i) {
          if (!device) {
            ASSERT_TRUE(finite(rows[i], 1));
            chosen[i] =
                static_cast<std::int32_t>(std::ranges::max_element(rows[i]) - rows[i].begin());
          }
          anchors[i] = chosen[i];
          ++past[i];
        }
      }
      EXPECT_EQ(runner.device_token_outputs() - before, device ? 6U : 0U);
      EXPECT_GT(runner.wave_stats().replayed, replayed);
      std::array<std::vector<std::int32_t>, 2> drafts;
      for (std::uint32_t i = 0; i < 2; ++i) {
        auto state = StateHash(runner, i);
        ASSERT_TRUE(state);
        if (device)
          EXPECT_EQ(*state, expected_state[i]);
        else
          expected_state[i] = *state;
      }
      const auto drafts_before = runner.draft_stats();
      std::array<en::Dsv4Runner::WaveWork, 2> verify;
      for (std::uint32_t i = 0; i < 2; ++i)
        verify[i] = {.slot = slots[i],
                     .pos = past[i],
                     .anchor = anchors[i],
                     .rows = 3,
                     .drafts = &drafts[i],
                     .logits = &rows[i]};
      ASSERT_TRUE(runner.DraftVerifyWave(verify));
      EXPECT_GT(runner.draft_stats().eager + runner.draft_stats().captured +
                    runner.draft_stats().replayed,
                drafts_before.eager + drafts_before.captured + drafts_before.replayed);
      for (std::uint32_t i = 0; i < 2; ++i) {
        ASSERT_EQ(drafts[i].size(), runner.draft_rows());
        ASSERT_TRUE(finite(rows[i], 3));
        if (device) {
          EXPECT_EQ(drafts[i], expected_drafts[i]);
          EXPECT_EQ(rows[i], expected_heads[i]);
        } else {
          expected_drafts[i] = drafts[i];
          expected_heads[i] = rows[i];
        }
        ASSERT_TRUE(slots[i]->Accept(1));
        ASSERT_TRUE(slots[i]->Rollback());
        auto state = StateHash(runner, i);
        ASSERT_TRUE(state);
        if (device)
          EXPECT_EQ(*state, expected_settled[i]);
        else
          expected_settled[i] = *state;
      }
    }
  }
}

// A logit row is the model's logit width (the swap table's continuation
// check, rt::CompleteContinuationRows), which for Qwen3.8's tokenizer.json is
// wider than the tokenizer.
TEST_P(PlainTokensServingGpu, LogitRowsAreTheModelsWidthNotTheTokenizers) {
  const std::uint32_t vocab = dynamic_cast<en::Dsv4Runner&>(model->paged()).vocab();
  ASSERT_EQ(model->logit_width(), vocab);
  ASSERT_TRUE(life->server->node().ChargeHost(64ULL << 20U, false));
  struct Charge {
    en::PagedNode& node;
    ~Charge() { node.UnchargeHost(64ULL << 20U); }
  } charge{life->server->node()};
  auto branch = model->branch(0);
  ASSERT_TRUE(branch);
  const std::array<rt::Llm::Branch*, 1> active{*branch};
  ASSERT_TRUE(life->server->SelectRequestBranches(*model, active));
  constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
  std::vector<float> last;
  std::array<rt::Generation, 2> generated;
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  options.keep_logits = true;
  for (auto& out : generated) {
    ASSERT_TRUE((*branch)->Clear());
    ASSERT_TRUE((*branch)->Prefill(seed, last));
    const auto r = (*branch)->Generate(last, options, out);
    ASSERT_TRUE(r) << (r ? "" : r.error());
  }
  EXPECT_TRUE(
      rt::CompleteContinuationRows(last, generated[0], generated[1], 3, model->logit_width()));
  if (GetParam() == 0) {
    EXPECT_FALSE(rt::CompleteContinuationRows(last, generated[0], generated[1], 3,
                                              model->tokenizer().size()));
  }
}

// Case 4 alone: its server is DeepSeek with DSpark at four slots.
class FourSlotDsparkGpu : public PlainTokensServingGpu {};

// Four slots' DSpark verify waves (and their joined drafts) at every
// alignment of their compressors' blocks and mixed rows a slot. A wave of
// 9 to 15 rows plans more graph nodes than the start's widest waves (1 and
// 4 rows a slot); each must fit the graph-reader allowance (planned.h
// GraphIndexAllowance), as the chat route's four-request speculative waves
// need.
TEST_P(FourSlotDsparkGpu, SpeculativeWavesFitTheStartupAllowance) {
  auto& runner = dynamic_cast<en::Dsv4Runner&>(model->paged());
  ASSERT_TRUE(runner.speculative());
  ASSERT_EQ(runner.wave_capacity(), 4U);
  ASSERT_TRUE(life->server->node().ChargeHost(64ULL << 20U, false));
  struct Charge {
    en::PagedNode& node;
    ~Charge() { node.UnchargeHost(64ULL << 20U); }
  } charge{life->server->node()};
  constexpr std::size_t kSlots = 4;
  std::array<en::Dsv4Runner::Slot*, kSlots> slots{};
  std::array<rt::Llm::Branch*, kSlots> branches{};
  for (std::uint32_t i = 0; i < kSlots; ++i) {
    auto slot = runner.request_slot(i);
    ASSERT_TRUE(slot);
    slots[i] = *slot;
    auto branch = model->branch(i);
    ASSERT_TRUE(branch);
    branches[i] = *branch;
  }
  ASSERT_TRUE(life->server->SelectRequestBranches(*model, branches));
  const auto finite = [&](const std::vector<float>& rows, std::uint32_t count) {
    return rows.size() == std::size_t{count} * runner.vocab() &&
           std::ranges::all_of(rows, [](float x) { return std::isfinite(x); });
  };
  const auto argmax = [&](const std::vector<float>& rows, std::uint32_t row) {
    const auto first = rows.begin() + static_cast<std::ptrdiff_t>(row * runner.vocab());
    return static_cast<std::int32_t>(
        std::max_element(first, first + static_cast<std::ptrdiff_t>(runner.vocab())) - first);
  };
  // Whole-chunk prompts ending at each alignment of a CSA block, one at an
  // HCA block's end.
  std::array<std::uint32_t, kSlots> past{4093, 4094, 4095, 4096};
  std::array<std::int32_t, kSlots> anchors{};
  std::array<std::vector<float>, kSlots> rows;
  for (std::uint32_t i = 0; i < kSlots; ++i) {
    ASSERT_TRUE(slots[i]->Clear());
    std::vector<std::int32_t> seed(past[i]);
    constexpr std::array<std::int32_t, 7> actual{2, 818, 5279, 529, 7001, 563, 42};
    for (std::size_t j = 0; j < seed.size(); ++j) seed[j] = actual[(j + i) % actual.size()];
    const auto chunk = slots[i]->Chunk(0, seed, rows[i], en::Dsv4ChunkKind::kInject);
    ASSERT_TRUE(chunk) << (chunk ? "" : chunk.error());
    ASSERT_TRUE(finite(rows[i], 1));
    anchors[i] = argmax(rows[i], 0);
  }
  constexpr std::array<std::array<std::uint32_t, kSlots>, 3> kRows{
      {{4, 4, 4, 4}, {1, 2, 3, 4}, {3, 1, 4, 2}}};
  const auto stats = runner.draft_stats();
  for (std::uint32_t wave = 0; wave < 12; ++wave) {
    std::array<std::vector<std::int32_t>, kSlots> drafts;
    std::array<en::Dsv4Runner::WaveWork, kSlots> work;
    for (std::uint32_t i = 0; i < kSlots; ++i) {
      // As serving's VerifyRows: no verify across a step's mask widths.
      std::uint32_t verify = kRows[wave % kRows.size()][i];
      while (verify > 1 && !llmp::model::Dsv4SameWidths(runner.state_layout(), past[i], verify))
        --verify;
      work[i] = {.slot = slots[i],
                 .pos = past[i],
                 .anchor = anchors[i],
                 .rows = verify,
                 .drafts = &drafts[i],
                 .logits = &rows[i]};
    }
    const auto verified = runner.DraftVerifyWave(work);
    ASSERT_TRUE(verified) << "wave " << wave << ": " << (verified ? "" : verified.error());
    for (std::uint32_t i = 0; i < kSlots; ++i) {
      ASSERT_EQ(drafts[i].size(), runner.draft_rows());
      ASSERT_TRUE(finite(rows[i], work[i].rows));
      // The anchor's row decides the next token; the slot moves on by one.
      ASSERT_TRUE(slots[i]->Accept(1));
      anchors[i] = argmax(rows[i], 0);
      ++past[i];
    }
  }
  EXPECT_GT(
      runner.draft_stats().eager + runner.draft_stats().captured + runner.draft_stats().replayed,
      stats.eager + stats.captured + stats.replayed);
}

INSTANTIATE_TEST_SUITE_P(ApprovedTargets, PlainTokensServingGpu, ::testing::Values(2U, 3U));
INSTANTIATE_TEST_SUITE_P(DeepSeekFourSlots, FourSlotDsparkGpu, ::testing::Values(4U));
