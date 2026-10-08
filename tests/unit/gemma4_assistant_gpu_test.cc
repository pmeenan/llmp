// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

#include "artifact/gguf_metadata.h"
#include "base/sha256.h"
#include "engine/gemma4_assistant.h"
#include "engine/support.h"
#include "tokenizer/gguf.h"

namespace {
namespace en = jitllm::engine;
namespace md = jitllm::model;
namespace ar = jitllm::artifact;
namespace tk = jitllm::tokenizer;
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
class Gemma4AssistantGpu : public ::testing::Test {
 protected:
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> lifetime = std::make_unique<Lifetime>();
  en::PagedNode& node = lifetime->node;
  std::unique_ptr<en::Gemma4Runner>& runner = lifetime->runner;
  en::Gemma4Assistant* assistant = nullptr;
  std::expected<jitllm::base::Sha256Digest, std::string> Witness(std::uint32_t slot,
                                                                 std::uint32_t prefix) {
    auto ranges = runner->CheckpointRanges(prefix);
    if (!ranges) return en::support::Error(ranges.error());
    std::uint64_t bytes = 0;
    for (const auto& range : *ranges) bytes += range.bytes;
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = node.Pinned(std::max(bytes, std::uint64_t{2816} * sizeof(float)), 0, staging);
    if (!pinned) return en::support::Error(pinned.error());
    // Failed or unknown copies leave the pinned owner registered until the
    // fixture proves its final fence; no destructor guesses retirement.
    if (auto copied = runner->CopyState(slot, *pinned, *ranges, true); !copied)
      return en::support::Error(copied.error());
    jitllm::base::Sha256 digest;
    digest.Update(std::span(static_cast<const std::byte*>(*pinned), bytes));
    if (auto copied = runner->CopyFeatures(slot, prefix - 1, 1, *pinned); !copied)
      return en::support::Error(copied.error());
    digest.Update(std::span(static_cast<const std::byte*>(*pinned), 2816 * sizeof(float)));
    if (auto freed = node.FreePinned(*pinned); !freed) return en::support::Error(freed.error());
    return digest.Finish();
  }
  virtual bool FailedSetup() const { return false; }
  virtual std::uint32_t MaxRows() const { return 128; }
  virtual bool CaptureAhead() const { return false; }
  void SetUp() override {
    const auto store = std::filesystem::path("/home/pmeenan/.local/share/jitllm/m3-artifacts");
    const auto target = store / "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3";
    const auto path = store / "1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42";
    ASSERT_TRUE(node.Open());
    runner = std::make_unique<en::Gemma4Runner>(
        node,
        en::Gemma4Options{.artifact = target,
                          .out = "/tmp/jitllm-gemma-assistant-control",
                          .max_rows = MaxRows(),
                          .slots = 2,
                          .retain_features = true,
                          .capture_ahead = CaptureAhead(),
                          .prefill_lookahead_capacity = CaptureAhead() ? 2U : 1U},
        0, 0);
    lifetime->entered.push_back(runner.get());
    ASSERT_TRUE(runner->Setup());
    // Fixed approved kept-metadata inputs, each <=16MiB: actual strings,
    // token/merge containers and raw arrays are covered before any parsing.
    constexpr auto admission_bytes = 512ULL << 20U;
    ASSERT_TRUE(node.ChargeHost(admission_bytes, false));
    struct Grant {
      en::PagedNode& node;
      ~Grant() { node.UnchargeHost(admission_bytes); }
    } grant{node};
    const auto ta = ar::Artifact::Open(target), aa = ar::Artifact::Open(path);
    ASSERT_TRUE(ta && aa);
    auto t = Decode(*ta, "gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf");
    auto a = Decode(*aa, "mtp-gemma-4-26B-A4B-it.kv.gguf");
    ASSERT_TRUE(t && a);
    const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
    auto tokens = a->view();
    tokens.tokens = {};
    EXPECT_FALSE(runner->SetupAssistant(path, t->view(), tokens));
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupancy);
    if (FailedSetup()) {
      EXPECT_FALSE(runner->SetupAssistant(store / "absent-assistant", t->view(), a->view()));
      return;
    }
    auto created = runner->SetupAssistant(path, t->view(), a->view());
    ASSERT_TRUE(created) << (created ? "" : created.error());
    assistant = *created;
    ASSERT_TRUE(node.MapWorkspace(runner->activations_needed(), runner->pool_needed()));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() +
                      (CaptureAhead() ? 2U : 1U) * runner->plan_floor_bytes());
    ASSERT_TRUE(node.Start(jitllm::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                               2 * node.StateCapacity())));
    ASSERT_TRUE(runner->Register());
    ASSERT_TRUE(runner->Bind());
    node.Run();
  }
  void TearDown() override {
    auto retired = node.TearDown(lifetime->entered);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    if (!retired) std::ignore = lifetime.release();
  }
};
TEST_F(Gemma4AssistantGpu, EndogenousThreeStepRepeatsAndIndependentFrozenOwners) {
  const auto bytes = std::uint64_t{16} * (262144U + 2816U) * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, bytes};
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
  std::array<std::vector<float>, 2> target_heads;
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  auto r = node.WithRequest(
      0, runner->closure(), "assistant frozen independent owners", [&]() -> en::Status {
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
          const std::array<std::int32_t, 5> peer{2, 818, 5279, 529, 818};
          const auto tokens = slot == 0 ? std::span<const std::int32_t>(prompt)
                                        : std::span<const std::int32_t>(peer);
          const en::Gemma4Runner::Work work{slot, 0, tokens, &target_heads[slot]};
          if (auto done = runner->Wave(std::span(&work, 1)); !done) return done;
        }
        std::array<jitllm::base::Sha256Digest, 2> immutable;
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
          auto witness = Witness(slot, slot == 0 ? 6U : 5U);
          if (!witness) return en::support::Error(witness.error());
          immutable[slot] = *witness;
        }
        std::array<en::Gemma4Runner::FrozenBorrow, 2> borrows;
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
          auto borrow = runner->BorrowFrozen(
              slot, static_cast<std::int32_t>(
                        std::max_element(target_heads[slot].begin(), target_heads[slot].end()) -
                        target_heads[slot].begin()));
          if (!borrow) return en::support::Error(borrow.error());
          borrows[slot] = std::move(*borrow);
        }
        const std::array<const en::Gemma4Runner::FrozenBorrow*, 2> owners{&borrows[0], &borrows[1]};
        std::array<std::vector<float>, 2> heads, features;
        const std::array<std::vector<float>*, 2> head_out{&heads[0], &heads[1]},
            feature_out{&features[0], &features[1]};
        std::array<std::int32_t, 2> anchor{borrows[0].anchor(), borrows[1].anchor()};
        std::array<std::array<std::vector<float>, 2>, 3> expected_heads, expected_features;
        // C1 first, followed by independently repeated joined C2. Each policy
        // resets to its protected native target feature, not stock outputs.
        for (const std::size_t count : {1U, 2U}) {
          for (unsigned repeat = 0; repeat < 2; ++repeat) {
            anchor = {borrows[0].anchor(), borrows[1].anchor()};
            for (unsigned step = 0; step < 3; ++step) {
              if (auto done = assistant->Step(
                      std::span(owners).first(count), std::span(anchor).first(count), step == 0,
                      std::span(head_out).first(count), std::span(feature_out).first(count));
                  !done)
                return done;
              for (std::uint32_t slot = 0; slot < count; ++slot) {
                if (auto valid = runner->CheckBorrow(borrows[slot]); !valid) return valid;
                EXPECT_EQ((*runner->request_slot(slot))->completed_positions(),
                          slot == 0 ? 6U : 5U);
                EXPECT_EQ(heads[slot].size(), 262144U);
                EXPECT_EQ(features[slot].size(), 2816U);
                EXPECT_TRUE(
                    std::ranges::all_of(heads[slot], [](float v) { return std::isfinite(v); }));
                EXPECT_TRUE(
                    std::ranges::all_of(features[slot], [](float v) { return std::isfinite(v); }));
                if (repeat == 0) {
                  expected_heads[step][slot] = heads[slot];
                  expected_features[step][slot] = features[slot];
                } else {
                  EXPECT_EQ(std::memcmp(expected_heads[step][slot].data(), heads[slot].data(),
                                        heads[slot].size() * sizeof(float)),
                            0);
                  EXPECT_EQ(std::memcmp(expected_features[step][slot].data(), features[slot].data(),
                                        features[slot].size() * sizeof(float)),
                            0);
                }
                anchor[slot] = static_cast<std::int32_t>(
                    std::max_element(heads[slot].begin(), heads[slot].end()) - heads[slot].begin());
              }
            }
            for (std::uint32_t slot = 0; slot < 2; ++slot) {
              auto witness = Witness(slot, slot == 0 ? 6U : 5U);
              if (!witness) return en::support::Error(witness.error());
              EXPECT_EQ(*witness, immutable[slot]);
            }
          }
        }
        EXPECT_GT(assistant->graph_stats().replayed, 0U);
        // An alien/default ticket cannot dispatch or publish any owner.
        const en::Gemma4Runner::FrozenBorrow absent;
        const std::array<const en::Gemma4Runner::FrozenBorrow*, 1> invalid{&absent};
        const std::array<std::int32_t, 1> token{2};
        const std::array<std::vector<float>*, 1> out{&heads[0]};
        EXPECT_FALSE(assistant->Step(invalid, token, true, out));
        return {};
      });
  ASSERT_TRUE(r) << (r ? "" : r.error());
}
class Gemma4AssistantCapturedGpu : public Gemma4AssistantGpu {
 protected:
  std::uint32_t MaxRows() const override { return 256; }
  bool CaptureAhead() const override { return true; }
};
TEST_F(Gemma4AssistantCapturedGpu, ThreeEndogenousStepsMatchAfterCapturedTargetFrontier) {
  const auto bytes = std::uint64_t{16} * (262144U + 2816U) * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, bytes};
  const std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
  std::array<std::int32_t, 768> prompt{};
  for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = seed[i % seed.size()];
  const std::array<std::int32_t, 5> peer{2, 818, 5279, 529, 818};
  std::array<std::array<std::vector<float>, 2>, 3> expected_heads, expected_features;
  std::array<jitllm::base::Sha256Digest, 2> expected_target;
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  auto ran = node.WithRequest(
      0, runner->closure(), "assistant consumes captured target frontier", [&]() -> en::Status {
        for (const bool hinted : {false, true}) {
          for (const auto slot : {0U, 1U})
            if (auto r = runner->Clear(slot); !r) return r;
          // Clears both target and assistant graph/plan caches. first=true
          // below resets assistant recurrent input to the newly borrowed target.
          runner->DropPlans();
          const auto before = runner->lookahead_stats();
          const auto graphs = runner->graph_stats();
          std::array<std::vector<float>, 2> target_heads;
          for (std::uint32_t chunk = 0; chunk < 3; ++chunk) {
            const auto past = chunk * 256;
            if (auto r = runner->ChunkPrefill(
                    past, std::span(prompt).subspan(past, 256), target_heads[0], true,
                    hinted && chunk < 2 ? 256U : 0U, true, hinted && chunk == 0 ? 256U : 0U, true);
                !r)
              return r;
          }
          const auto after = runner->lookahead_stats();
          EXPECT_EQ(after.refused, before.refused);
          EXPECT_EQ(after.dropped_ahead, before.dropped_ahead);
          if (hinted) {
            EXPECT_GT(after.built_pairs, before.built_pairs);
            EXPECT_GT(after.cached_pairs, before.cached_pairs);
            EXPECT_GT(after.captured_ahead, before.captured_ahead);
            EXPECT_GT(runner->graph_stats().replayed, graphs.replayed);
          } else {
            EXPECT_EQ(after.captured_ahead, before.captured_ahead);
            EXPECT_EQ(runner->graph_stats().replayed, graphs.replayed);
          }
          const en::Gemma4Runner::Work work{1, 0, peer, &target_heads[1]};
          if (auto r = runner->Wave(std::span(&work, 1)); !r) return r;
          std::array<en::Gemma4Runner::FrozenBorrow, 2> borrows;
          for (const auto slot : {0U, 1U}) {
            EXPECT_EQ(target_heads[slot].size(), 262144U);
            auto witness = Witness(slot, slot == 0 ? 768U : 5U);
            if (!witness) return en::support::Error(witness.error());
            if (!hinted)
              expected_target[slot] = *witness;
            else
              EXPECT_EQ(*witness, expected_target[slot]);
            const auto anchor = static_cast<std::int32_t>(
                std::max_element(target_heads[slot].begin(), target_heads[slot].end()) -
                target_heads[slot].begin());
            auto borrowed = runner->BorrowFrozen(slot, anchor);
            if (!borrowed) return en::support::Error(borrowed.error());
            borrows[slot] = std::move(*borrowed);
          }
          const std::array<const en::Gemma4Runner::FrozenBorrow*, 2> owners{&borrows[0],
                                                                            &borrows[1]};
          std::array<std::int32_t, 2> anchors{borrows[0].anchor(), borrows[1].anchor()};
          std::array<std::vector<float>, 2> heads, features;
          const std::array<std::vector<float>*, 2> head_out{&heads[0], &heads[1]},
              feature_out{&features[0], &features[1]};
          for (std::uint32_t step = 0; step < 3; ++step) {
            if (auto r = assistant->Step(owners, anchors, step == 0, head_out, feature_out); !r)
              return r;
            for (const auto slot : {0U, 1U}) {
              if (auto r = runner->CheckBorrow(borrows[slot]); !r) return r;
              EXPECT_EQ((*runner->request_slot(slot))->completed_positions(),
                        slot == 0 ? 768U : 5U);
              EXPECT_EQ(heads[slot].size(), 262144U);
              EXPECT_EQ(features[slot].size(), 2816U);
              EXPECT_TRUE(
                  std::ranges::all_of(heads[slot], [](float v) { return std::isfinite(v); }));
              EXPECT_TRUE(
                  std::ranges::all_of(features[slot], [](float v) { return std::isfinite(v); }));
              if (!hinted) {
                expected_heads[step][slot] = heads[slot];
                expected_features[step][slot] = features[slot];
              } else {
                if (heads[slot].size() != expected_heads[step][slot].size() ||
                    features[slot].size() != expected_features[step][slot].size())
                  return en::support::Error("assistant full output shapes differ");
                EXPECT_EQ(std::memcmp(heads[slot].data(), expected_heads[step][slot].data(),
                                      heads[slot].size() * sizeof(float)),
                          0);
                EXPECT_EQ(std::memcmp(features[slot].data(), expected_features[step][slot].data(),
                                      features[slot].size() * sizeof(float)),
                          0);
              }
              anchors[slot] = static_cast<std::int32_t>(
                  std::max_element(heads[slot].begin(), heads[slot].end()) - heads[slot].begin());
            }
          }
          for (const auto slot : {0U, 1U}) {
            auto witness = Witness(slot, slot == 0 ? 768U : 5U);
            if (!witness) return en::support::Error(witness.error());
            EXPECT_EQ(*witness, expected_target[slot]);
          }
        }
        return {};
      });
  ASSERT_TRUE(ran) << (ran ? "" : ran.error());
}
class Gemma4AssistantFailedSetupGpu : public Gemma4AssistantGpu {
 protected:
  bool FailedSetup() const override { return true; }
};
TEST_F(Gemma4AssistantFailedSetupGpu, PartialSetupCannotRegisterBindOrPublish) {
  const auto before = node.catalog().OccupancyOf(node.domain()).Total().value();
  EXPECT_FALSE(runner->Register());
  EXPECT_FALSE(runner->Bind());
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), before);
  std::vector<float> sentinel{123};
  const std::array<std::int32_t, 1> tokens{2};
  const en::Gemma4Runner::Work work{0, 0, tokens, &sentinel};
  EXPECT_FALSE(runner->Wave(std::span(&work, 1)));
  EXPECT_EQ(sentinel, (std::vector<float>{123}));
  // No successful component pointer escaped admission, so no Step entry is
  // available. A retained failed attempt also cannot be replaced in place.
  EXPECT_FALSE(runner->BorrowFrozen(0, 2));
}
}  // namespace
