// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <system_error>
#include <tuple>
#include <vector>

#include "artifact/gguf_metadata.h"
#include "base/sha256.h"
#include "engine/gemma4_assistant.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"
#include "execution/sampling.h"
#include "tokenizer/gguf.h"
#include "tokenizer_fixtures.h"

namespace en = jitllm::engine;
namespace base = jitllm::base;
namespace ar = jitllm::artifact;
namespace tk = jitllm::tokenizer;
namespace md = jitllm::model;
struct Vocabulary {
  tk::GgufTokenizer tokenizer;
  ar::GgufMetadata raw;
  md::Gemma4AssistantVocabulary view() const {
    return {tokenizer.spec.tokens, tokenizer.spec.merges, raw.at("tokenizer.ggml.scores").reals,
            raw.at("tokenizer.ggml.token_type").integers};
  }
};
std::expected<Vocabulary, std::string> Decode(const ar::Artifact& artifact,
                                              std::string_view filename) {
  const auto listed = std::ranges::find_if(artifact.files(), [filename](const auto& file) {
    return file.role == ar::FileRole::kSourceMetadata &&
           file.path == "meta/" + std::string(filename);
  });
  if (listed == artifact.files().end() || listed->bytes.value() > (16ULL << 20U))
    return en::support::Error("fixture vocabulary exceeds funded metadata envelope");
  auto bytes = artifact.ReadMetadata(filename);
  if (!bytes) return en::support::Error(bytes.error().ToString());
  auto input = std::as_bytes(std::span(bytes->data(), bytes->size()));
  auto tokenizer = tk::ReadGgufTokenizer(input);
  constexpr std::array<std::string_view, 2> keys{"tokenizer.ggml.scores",
                                                 "tokenizer.ggml.token_type"};
  auto raw = ar::ReadGgufMetadata(input, keys);
  if (!tokenizer || !raw || !raw->contains(keys[0]) || !raw->contains(keys[1]))
    return en::support::Error("fixture vocabulary parse failed");
  return Vocabulary{std::move(*tokenizer), std::move(*raw)};
}

class Gemma4GreedyGpu : public ::testing::TestWithParam<en::Gemma4Variant> {
 protected:
  std::filesystem::path Artifact() const {
    return std::filesystem::path(jitllm::test_support::ModelsDir()) / "m3-artifacts" /
           (GetParam() == en::Gemma4Variant::k31B
                ? "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
                : "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3");
  }
  std::filesystem::path AssistantArtifact() const {
    return Artifact().parent_path() /
           (GetParam() == en::Gemma4Variant::k31B
                ? "447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120"
                : "1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42");
  }
  void SetUp() override {
    if (std::error_code error; !std::filesystem::exists(Artifact(), error) ||
                               !std::filesystem::exists(AssistantArtifact(), error)) {
      GTEST_SKIP() << "no Gemma 4 target and assistant in " << Artifact().parent_path();
    }
  }
  void Start() {
    const auto artifact = Artifact();
    ASSERT_TRUE(std::filesystem::exists(artifact));
    // Spill files need direct I/O: scratch in the build tree.
    const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path root = scratch != nullptr ? scratch : ::testing::TempDir();
    en::Gemma4Options options{.artifact = artifact,
                              .out = root / "gemma4-greedy-control",
                              .variant = GetParam(),
                              .context = 1536,
                              .max_rows = 16,
                              .slots = 2,
                              .max_head_rows = 4,
                              .max_verify_rows = 4,
                              .retain_features = true};
    runner = std::make_unique<en::Gemma4Runner>(node, std::move(options), 0, 0);
    ASSERT_TRUE(node.Open());
    entered.push_back(runner.get());
    // All caller output vectors and small state/range containers are funded
    // before growth. State payload diagnostics use separately cataloged pinned
    // memory, not uncharged vector copies.
    ASSERT_TRUE(runner->Setup());
    const auto assistant_path = AssistantArtifact();
    // Actual token/merge/score/type containers are funded before parsing and
    // destroyed before this admission grant is returned. No raw checkpoint.
    {
      constexpr auto admission_bytes = 512ULL << 20U;
      ASSERT_TRUE(node.ChargeHost(admission_bytes, false));
      struct Grant {
        en::PagedNode& node;
        ~Grant() { node.UnchargeHost(admission_bytes); }
      } grant{node};
      auto ta = ar::Artifact::Open(artifact), aa = ar::Artifact::Open(assistant_path);
      ASSERT_TRUE(ta && aa);
      auto target_tokens = Decode(*ta, GetParam() == en::Gemma4Variant::k31B
                                           ? "gemma-4-31B-it-UD-Q4_K_XL.kv.gguf"
                                           : "gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf");
      auto assistant_tokens =
          Decode(*aa, GetParam() == en::Gemma4Variant::k31B ? "mtp-gemma-4-31B-it.kv.gguf"
                                                            : "mtp-gemma-4-26B-A4B-it.kv.gguf");
      ASSERT_TRUE(target_tokens && assistant_tokens);
      auto created =
          runner->SetupAssistant(assistant_path, target_tokens->view(), assistant_tokens->view());
      ASSERT_TRUE(created) << (created ? "" : created.error());
      assistant = *created;
    }
    ASSERT_TRUE(node.MapWorkspace(runner->activations_needed(), runner->pool_needed()));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes());
    ASSERT_TRUE(node.Start(base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                       2 * node.StateCapacity())));
    ASSERT_TRUE(runner->Register());
    ASSERT_TRUE(runner->Bind());
    node.Run();
    ASSERT_TRUE(node.ChargeHost(64ULL << 20, false));
    ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
    const auto capacity = 4 * runner->layout().tensors.size();
    EXPECT_EQ((*runner->request_slot(0))->state().saved().capacity(), capacity);
  }
  void TearDown() override {
    auto retired = node.TearDown(entered);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    if (!retired) std::ignore = lifetime.release();
  }
  en::Status Held(const std::function<en::Status()>& body) {
    return node.WithRequest(0, runner->closure(), "Gemma4 greedy control", body);
  }
  en::Status Single(std::uint32_t slot, std::uint32_t past, std::span<const std::int32_t> tokens,
                    std::vector<float>& heads) {
    const en::Gemma4Runner::Work work{slot, past, tokens, &heads};
    return runner->Wave(std::span(&work, 1));
  }
  en::Status Prefix(std::uint32_t slot, std::uint32_t rows, std::vector<float>& head) {
    std::array<std::int32_t, 16> tokens{};
    for (std::uint32_t past = 0; past < rows;) {
      const auto count = std::min(16U, rows - past);
      for (std::uint32_t i = 0; i < count; ++i) tokens[i] = past + i == 0 ? 2 : 818;
      if (auto r = Single(slot, past, std::span(tokens).first(count), head); !r) return r;
      past += count;
    }
    return {};
  }
  std::expected<base::Sha256Digest, std::string> StateHash(std::uint32_t slot,
                                                           std::uint32_t positions) {
    auto ranges = runner->CheckpointRanges(positions);
    if (!ranges) return std::unexpected(ranges.error());
    return RangeHash(slot, *ranges);
  }
  std::expected<base::Sha256Digest, std::string> RangeHash(
      std::uint32_t slot, std::span<const en::LiveState::Range> ranges) {
    std::uint64_t bytes = 0;
    for (const auto& r : ranges) bytes += r.bytes;
    std::vector<jitllm::catalog::ExtentId> staging;
    if (bytes == 0) return base::Sha256{}.Finish();
    auto pinned = node.Pinned(bytes, 0, staging);
    if (!pinned) return std::unexpected(pinned.error());
    auto copied = runner->CopyState(slot, *pinned, ranges, true);
    if (!copied) {
      node.KeepPinned(*pinned);
      return std::unexpected(copied.error());
    }
    base::Sha256 hash;
    hash.Update(std::span(static_cast<const std::byte*>(*pinned), bytes));
    const auto digest = hash.Finish();
    if (auto r = node.FreePinned(*pinned); !r) return std::unexpected(r.error());
    return digest;
  }
  en::Status Feature(std::uint32_t slot, std::uint32_t position, std::vector<float>& out) {
    const auto bytes = std::uint64_t{runner->profile().width} * sizeof(float);
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = node.Pinned(bytes, 0, staging);
    if (!pinned) return en::support::Error(pinned.error());
    if (auto r = runner->CopyFeatures(slot, position, 1, *pinned); !r) {
      node.KeepPinned(*pinned);
      return r;
    }
    const auto* values = static_cast<const float*>(*pinned);
    out.assign(values, values + runner->profile().width);
    return node.FreePinned(*pinned);
  }
  void Exact(std::span<const float> a, std::span<const float> b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size_bytes()), 0);
    EXPECT_TRUE(std::ranges::all_of(a, [](float x) { return std::isfinite(x); }));
  }
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> lifetime = std::make_unique<Lifetime>();
  en::PagedNode& node = lifetime->node;
  std::unique_ptr<en::Gemma4Runner>& runner = lifetime->runner;
  std::vector<en::PagedModel*>& entered = lifetime->entered;
  en::Gemma4Assistant* assistant = nullptr;
  void Fund(en::Gemma4GreedyWorkspace& workspace, en::Gemma4GreedyResult& result) {
    const auto vocab = runner->profile().vocab, width = runner->profile().width;
    workspace.draft_head.reserve(vocab);
    workspace.verify_heads.reserve(4 * vocab);
    workspace.verify_features.reserve(4 * width);
    result.head.reserve(vocab);
    result.feature.reserve(width);
  }
  en::Status Proposals(std::span<const float> head, std::array<std::int32_t, 4>& proposal,
                       en::Gemma4GreedyWorkspace& workspace) {
    auto anchor = jitllm::execution::Greedy(head);
    if (!anchor) return en::support::Error("initial target head refused");
    proposal[0] = *anchor;
    auto borrow = runner->BorrowFrozen(0, *anchor);
    if (!borrow) return en::support::Error(borrow.error());
    const std::array<const en::Gemma4Runner::FrozenBorrow*, 1> owners{&*borrow};
    const std::array<std::vector<float>*, 1> outputs{&workspace.draft_head};
    for (std::uint32_t i = 0; i < 3; ++i) {
      const std::array<std::int32_t, 1> token{proposal[i]};
      if (auto r = assistant->Step(owners, token, i == 0, outputs); !r) return r;
      auto next = jitllm::execution::Greedy(workspace.draft_head);
      if (!next) return en::support::Error("draft head refused");
      proposal[i + 1] = *next;
      EXPECT_EQ(borrow->prefix(), 6U);
      if (auto r = runner->CheckBorrow(*borrow); !r) return r;
    }
    return {};
  }
  std::expected<std::vector<en::LiveState::Range>, std::string> Writes(std::uint32_t past,
                                                                       std::uint32_t rows) {
    auto writes = md::Gemma4ChunkWrites(runner->profile(), runner->layout(), past, rows);
    if (!writes) return en::support::Error(writes.error());
    std::vector<en::LiveState::Range> result;
    for (const auto& range : *writes)
      result.push_back({.region = 0, .offset = range.offset, .bytes = range.bytes});
    return result;
  }
};

TEST_P(Gemma4GreedyGpu, RealDraftAcceptEqualsIndependentFourRowTargetAndContinuation) {
  Start();
  ASSERT_TRUE(assistant);
  auto done = Held([&]() -> en::Status {
    en::Gemma4GreedyWorkspace workspace;
    en::Gemma4GreedyResult result;
    Fund(workspace, result);
    std::vector<float> head, control, expected, selected, before_feature, after_feature;
    std::array<std::int32_t, 4> proposal{};
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      if (auto r = runner->Clear(0); !r) return r;
      if (auto r = runner->Clear(1); !r) return r;
      if (auto r = Prefix(0, 6, head); !r) return r;
      if (auto r = Prefix(1, 6, control); !r) return r;
      if (auto r = runner->ReserveStateThrough(0, 10); !r) return r;
      auto immutable = StateHash(0, 10);
      if (!immutable) return en::support::Error(immutable.error());
      if (auto r = Feature(0, 5, before_feature); !r) return r;
      if (auto r = Proposals(head, proposal, workspace); !r) return r;
      auto after_draft = StateHash(0, 10);
      if (!after_draft) return en::support::Error(after_draft.error());
      EXPECT_EQ(*immutable, *after_draft);
      if (auto r = Feature(0, 5, after_feature); !r) return r;
      Exact(before_feature, after_feature);
      const en::Gemma4Runner::Work work{1, 6, proposal, &expected};
      if (auto r = runner->Wave(std::span(&work, 1), true, true); !r) return r;
      auto decision =
          en::JudgeGemma4Greedy(std::span(proposal).subspan(1), expected, runner->profile().vocab);
      if (!decision) return en::support::Error(decision.error());
      std::vector<en::LiveState::Range> rejected;
      if (decision->keep < 4) {
        auto ranges = Writes(6 + decision->keep, 4 - decision->keep);
        if (!ranges) return en::support::Error(ranges.error());
        rejected = std::move(*ranges);
      }
      auto rejected_before = RangeHash(0, rejected);
      if (!rejected_before) return en::support::Error(rejected_before.error());
      if (auto r = assistant->GreedyUnit(0, 6, 3, head, workspace, result); !r) return r;
      EXPECT_EQ(result.count, decision->keep);
      EXPECT_EQ(result.past, 6U + decision->keep);
      EXPECT_EQ(result.next_anchor, decision->next_anchor);
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), result.past);
      EXPECT_TRUE((*runner->request_slot(0))->state_usable());
      for (std::uint32_t i = 0; i < 4; ++i)
        EXPECT_EQ(result.committed[i], i < result.count ? proposal[i] : 0);
      // Pending next anchor is not added to cursor or committed token count.
      Exact(workspace.verify_heads, expected);
      Exact(result.head, std::span(expected).subspan((result.count - 1) * runner->profile().vocab,
                                                     runner->profile().vocab));
      for (std::uint32_t row = 0; row < 4; ++row) {
        if (auto r = Feature(1, 6 + row, selected); !r) return r;
        Exact(selected, std::span(workspace.verify_features)
                            .subspan(row * runner->profile().width, runner->profile().width));
      }
      if (auto r = Feature(0, result.past - 1, selected); !r) return r;
      Exact(result.feature, selected);
      Exact(result.feature,
            std::span(workspace.verify_features)
                .subspan((result.count - 1) * runner->profile().width, runner->profile().width));
      auto prefix = Writes(0, result.past);
      if (!prefix) return en::support::Error(prefix.error());
      auto state = RangeHash(0, *prefix), peer = RangeHash(1, *prefix);
      if (!state || !peer) return en::support::Error("semantic accepted prefix copy failed");
      EXPECT_EQ(*state, *peer);
      auto rejected_after = RangeHash(0, rejected);
      if (!rejected_after) return en::support::Error(rejected_after.error());
      EXPECT_EQ(*rejected_before, *rejected_after);
      if (result.count < 4) {
        auto ranges = runner->CheckpointRanges(result.past);
        if (!ranges) return en::support::Error(ranges.error());
        std::uint64_t bytes = 0;
        for (const auto& range : *ranges) bytes += range.bytes;
        std::vector<jitllm::catalog::ExtentId> staging;
        auto saved = node.Pinned(bytes, 0, staging);
        if (!saved) return en::support::Error(saved.error());
        if (auto r = runner->CopyState(1, *saved, *ranges, true); !r) {
          node.KeepPinned(*saved);
          return r;
        }
        if (auto r = runner->Clear(1); !r) return r;
        if (auto r = runner->RestoreCheckpoint(1, result.past, *saved, *ranges,
                                               runner->CheckpointLayoutId());
            !r) {
          node.KeepPinned(*saved);
          return r;
        }
        if (auto r = node.FreePinned(*saved); !r) return r;
      }
      const std::array<std::int32_t, 1> next{result.next_anchor};
      if (auto r = Single(0, result.past, next, head); !r) return r;
      if (auto r = Single(1, result.past, next, control); !r) return r;
      Exact(head, control);
      if (auto r = Feature(0, result.past, before_feature); !r) return r;
      if (auto r = Feature(1, result.past, after_feature); !r) return r;
      Exact(before_feature, after_feature);
      // A following real unit reinitializes assistant recurrence from the
      // newly completed target feature; the previous draft tail is discarded.
      if (auto r = assistant->GreedyUnit(0, result.past + 1, 1, head, workspace, result); !r)
        return r;
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), result.past);
    }
    return {};
  });
  ASSERT_TRUE(done) << (done ? "" : done.error());
  EXPECT_GT(assistant->graph_stats().replayed, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
}
TEST_P(Gemma4GreedyGpu, BorrowBarrierAndWholeDiscardRestoreTargetBytesAndFeature) {
  Start();
  ASSERT_TRUE(assistant);
  auto done = Held([&]() -> en::Status {
    std::vector<float> head, before_feature, after_feature, first_heads, first_features;
    en::Gemma4GreedyWorkspace workspace;
    en::Gemma4GreedyResult result;
    Fund(workspace, result);
    if (auto r = Prefix(0, 6, head); !r) return r;
    if (auto r = runner->ReserveStateThrough(0, 10); !r) return r;
    auto before = StateHash(0, 10);
    if (!before) return en::support::Error(before.error());
    if (auto r = Feature(0, 5, before_feature); !r) return r;
    std::array<std::int32_t, 4> proposal{};
    if (auto r = Proposals(head, proposal, workspace); !r) return r;
    {
      auto borrow = runner->BorrowFrozen(0, proposal[0]);
      if (!borrow) return en::support::Error(borrow.error());
      std::vector<float> sentinel_heads{17}, sentinel_features{19};
      EXPECT_FALSE(runner->Verify(0, 6, proposal, sentinel_heads, sentinel_features));
      EXPECT_FALSE(assistant->GreedyUnit(0, 6, 3, head, workspace, result));
      EXPECT_EQ(sentinel_heads, std::vector<float>{17});
      EXPECT_EQ(sentinel_features, std::vector<float>{19});
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 6U);
    }
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      if (auto r =
              runner->Verify(0, 6, proposal, workspace.verify_heads, workspace.verify_features);
          !r)
        return r;
      if (repeat == 0) {
        first_heads = workspace.verify_heads;
        first_features = workspace.verify_features;
      } else {
        Exact(first_heads, workspace.verify_heads);
        Exact(first_features, workspace.verify_features);
      }
      // Completed verification cancelled before acceptance: no tokens commit.
      if (auto r = runner->DiscardVerify(0); !r) return r;
      auto restored = StateHash(0, 10);
      if (!restored) return en::support::Error(restored.error());
      EXPECT_EQ(*before, *restored);
      if (auto r = Feature(0, 5, after_feature); !r) return r;
      Exact(before_feature, after_feature);
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 6U);
      EXPECT_TRUE((*runner->request_slot(0))->state_usable());
    }
    return {};
  });
  ASSERT_TRUE(done) << (done ? "" : done.error());
}
TEST_P(Gemma4GreedyGpu, CapacityCursorAndDepthRefusalDoNotPublishOrDispatch) {
  Start();
  ASSERT_TRUE(assistant);
  auto done = Held([&]() -> en::Status {
    std::vector<float> head;
    if (auto r = Prefix(0, 6, head); !r) return r;
    en::Gemma4GreedyWorkspace workspace;
    en::Gemma4GreedyResult result;
    Fund(workspace, result);
    result.committed = {11, 12, 13, 14};
    result.count = 3;
    result.past = 42;
    result.next_anchor = 18;
    result.head.assign(1, 21);
    result.feature.assign(1, 23);
    auto before = StateHash(0, 6);
    if (!before) return en::support::Error(before.error());
    const auto executed = assistant->graph_stats().eager + assistant->graph_stats().captured +
                          assistant->graph_stats().replayed;
    EXPECT_FALSE(assistant->GreedyUnit(0, 6, 0, head, workspace, result));
    EXPECT_FALSE(assistant->GreedyUnit(0, 6, 4, head, workspace, result));
    EXPECT_FALSE(assistant->GreedyUnit(0, 5, 3, head, workspace, result));
    EXPECT_FALSE(assistant->GreedyUnit(1, 0, 3, head, workspace, result));
    en::Gemma4GreedyWorkspace absent;
    EXPECT_FALSE(assistant->GreedyUnit(0, 6, 3, head, absent, result));
    en::Gemma4GreedyResult unfunded;
    EXPECT_FALSE(assistant->GreedyUnit(0, 6, 3, head, workspace, unfunded));
    const float old = head[0];
    head[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(assistant->GreedyUnit(0, 6, 3, head, workspace, result));
    head[0] = old;
    EXPECT_EQ(result.committed, (std::array<std::int32_t, 4>{11, 12, 13, 14}));
    EXPECT_EQ(result.count, 3U);
    EXPECT_EQ(result.past, 42U);
    EXPECT_EQ(result.next_anchor, 18);
    EXPECT_EQ(result.head, std::vector<float>{21});
    EXPECT_EQ(result.feature, std::vector<float>{23});
    auto after = StateHash(0, 6);
    if (!after) return en::support::Error(after.error());
    EXPECT_EQ(*before, *after);
    EXPECT_EQ(executed, assistant->graph_stats().eager + assistant->graph_stats().captured +
                            assistant->graph_stats().replayed);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 6U);
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 0U);
    EXPECT_TRUE((*runner->request_slot(0))->state_usable());
    EXPECT_TRUE((*runner->request_slot(1))->state_usable());
    return {};
  });
  ASSERT_TRUE(done) << (done ? "" : done.error());
}
INSTANTIATE_TEST_SUITE_P(ApprovedProfiles, Gemma4GreedyGpu,
                         ::testing::Values(en::Gemma4Variant::k26BA4B, en::Gemma4Variant::k31B));
