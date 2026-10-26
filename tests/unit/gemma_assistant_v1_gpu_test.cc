// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 4 26B-A4B and 31B served with their assistants as the model's
// drafter, from the published v1 artifacts (label `models`): the server,
// approved settings and speculative path, one owner.
//
// The run prefills a repeating prompt and generates 48 tokens speculatively
// with every logit row, takes the complete state (every stage, the
// assistant's included), spills it, restores it with no model work, checks
// the state is the same, continues from the restored history with 16 more
// speculative tokens, and takes the state again. The observations are
// digests (full prefill head, tokens, drafted and accepted counts, logit
// rows, histories, state images), asserted against tests/unit/data/v1_frozen
// (tests/support/v1_frozen.h), which hold what the pinned v0 imports
// produced bit for bit before serving became v1-only (D-118).
//
// LLMP_TEST_V1_STORE names the v1 store and holds v1-ids.json.
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
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
namespace ts = llmp::test_support;

struct Profile {
  const char* name;
  const char* target;  // the store's v1-ids.json names
  const char* assistant;
  std::uint32_t prefill_chunk;
};
constexpr std::array<Profile, 2> kProfiles{{
    {"Gemma26", "gemma26", "gemma26-assistant", 1024},
    {"Gemma31", "gemma31", "gemma31-assistant", 256},
}};
constexpr std::uint32_t kVocab = 262144;
constexpr std::uint32_t kContext = 8192, kSlots = 4;
constexpr std::uint32_t kFirst = 48, kSecond = 16;

using Observations = std::vector<std::pair<std::string, std::string>>;

struct Arm {
  fs::path store;
  std::string target, assistant;
};

struct Owner {
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  std::FILE* log = nullptr;
  fs::path scratch;
  std::unique_ptr<rt::Server> server;
  ~Owner() {
    if (log != nullptr) std::fclose(log);
    if (!scratch.empty()) {
      std::error_code ignored;
      fs::remove_all(scratch, ignored);
    }
  }
};

// The complete used state of slot 0, every stage, hashed through pinned host
// memory the test funds beside the node's catalog.
std::expected<std::string, std::string> StateImage(Owner& own, rt::Llm& model) {
  auto& slots = model.runner().slots();
  auto& node = own.server->node();
  constexpr std::uint64_t kPart = 1ULL << 20U;
  constexpr std::uint64_t kFunding = 64ULL << 20U;
  if (!node.ChargeHost(kFunding, false)) return std::unexpected("cannot fund state observation");
  struct Uncharge {
    en::PagedNode& node;
    ~Uncharge() { node.UnchargeHost(kFunding); }
  } uncharge{node};
  std::vector<llmp::catalog::ExtentId> staging;
  auto buffer = node.Pinned(kPart, 0, staging);
  if (!buffer) return std::unexpected(buffer.error());
  llmp::base::Sha256 hash;
  std::uint64_t bytes = 0, ranges = 0;
  for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
    for (const en::LiveState::Range& range : slots.live(0, stage).used_ranges()) {
      ++ranges;
      // Coordinates are part of the image, as in the Gemma and Qwen proofs.
      const std::array<std::uint64_t, 3> where{stage, range.offset, range.bytes};
      hash.Update(std::as_bytes(std::span(where)));
      for (std::uint64_t at = 0; at < range.bytes; at += kPart) {
        const en::LiveState::Range part{.region = stage,
                                        .offset = range.offset + at,
                                        .bytes = std::min(kPart, range.bytes - at)};
        auto retired = en::LiveState::CopyRetirement::kUnproven;
        const auto copied = slots.CopyState(0, *buffer, std::span(&part, 1), true, &retired);
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
  if (auto freed = node.FreePinned(*buffer); !freed) return std::unexpected(freed.error());
  if (bytes == 0) return std::unexpected("no initialized state");
  const auto digest = llmp::base::ToHex(hash.Finish());
  const auto positions = slots.positions(0);
  return digest + ':' + std::to_string(bytes) + ':' + std::to_string(ranges) + ':' +
         std::to_string(positions);
}

bool Finite(std::span<const float> row) {
  return row.size() == kVocab && std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
}

std::expected<Observations, std::string> RunArm(const Profile& profile, const Arm& arm) {
  auto own = std::make_unique<Owner>();
  Observations out;
  std::error_code error;
  const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const fs::path root = base != nullptr ? base : ::testing::TempDir();
  fs::create_directories(root, error);
  std::string name = (root / "gemma-assistant-v1-XXXXXX").string();
  if (mkdtemp(name.data()) == nullptr) return std::unexpected("creating test scratch");
  own->scratch = name;
  own->roles.installed = arm.store;
  own->roles.spill = own->scratch / "spill";
  own->roles.state = own->scratch / "state";
  for (const auto& path : {own->roles.spill, own->roles.state}) {
    fs::create_directory(path, error);
    if (error || chmod(path.c_str(), 0700) != 0) return std::unexpected("protecting test state");
  }
  cfg::ModelEntry entry;
  entry.name = "gemma";
  entry.artifact = arm.target;
  entry.drafter = arm.assistant;
  entry.overrides["context"] = std::int64_t{kContext};
  entry.overrides["prefill_chunk"] = std::int64_t{profile.prefill_chunk};
  entry.overrides["max_slots"] = std::int64_t{kSlots};
  own->config.models.push_back(std::move(entry));
  own->log = std::tmpfile();
  if (own->log == nullptr) return std::unexpected("opening test log");
  own->server = std::make_unique<rt::Server>(own->config, own->roles, own->options, own->log);

  const auto note = [&out](std::string key, std::string value) {
    out.emplace_back(std::move(key), std::move(value));
  };
  const auto completed = [&]() -> rt::Status {
    if (auto started = own->server->Start(true); !started) return started;
    auto* model = dynamic_cast<rt::Llm*>(own->server->Find("gemma"));
    if (model == nullptr || dynamic_cast<rt::Llm*>(model) == nullptr)
      return std::unexpected("the pair is not on the component engine");
    if (!model->speculative()) return std::unexpected("the assistant is not the model's drafter");
    if (model->logit_width() != kVocab || model->settings().context.value != kContext ||
        model->settings().prefill_chunk.value != profile.prefill_chunk ||
        model->settings().max_slots.value != kSlots)
      return std::unexpected("the pair changed the approved execution recipe");
    auto& engine = dynamic_cast<rt::Llm&>(*model);
    rt::SwapParts parts;
    if (auto activated = own->server->Activate(*model, parts); !activated) return activated;
    auto branch = model->branch(0);
    if (!branch) return std::unexpected(branch.error());
    std::array<rt::Llm::Branch*, 1> active{*branch};
    if (auto selected = own->server->SelectRequestBranches(*model, active); !selected)
      return selected;

    constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
    std::vector<std::int32_t> prompt;
    for (int i = 0; i < 24; ++i) prompt.insert(prompt.end(), seed.begin(), seed.end());
    std::vector<float> head;
    if (auto prefill = (*branch)->Prefill(prompt, head); !prefill) return prefill;
    if (!Finite(head)) return std::unexpected("prefill head is not finite");
    note("heads.0", ts::HexOfFloats(head));

    rt::GenerateOptions options;
    options.stop = false;
    options.keep_logits = true;
    options.max_tokens = kFirst;
    rt::Generation first;
    if (auto generated = (*branch)->Generate(head, options, first); !generated) return generated;
    if (first.tokens.size() != kFirst || first.drafted == 0 || first.accepted == 0 ||
        first.logits.empty())
      return std::unexpected("the first generation did not speculate completely");
    for (const auto& row : first.logits)
      if (!Finite(row)) return std::unexpected("a generation logit row is not finite");
    note("tokens.0", ts::JoinTokens(first.tokens));
    note("rows.0", std::to_string(first.logits.size()) + ':' + ts::HexOfRows(first.logits));
    note("drafted.0", std::to_string(first.drafted));
    note("accepted.0", std::to_string(first.accepted));
    note("steps.0", std::to_string(first.steps));
    const auto history = (*branch)->history();
    note("history.0", ts::HexOfInts(history));
    auto state = StateImage(*own, engine);
    if (!state) return std::unexpected(state.error());
    note("state.0", *state);

    // Spill the idle conversation, restore it through the prompt interface
    // with no model work, and find exactly the same state.
    if (auto ended = own->server->EndRequestBranches(*model); !ended) return ended;
    if (auto spilled = model->SpillIdle(**branch); !spilled) return spilled;
    if (!(*branch)->spilled()) return std::unexpected("the conversation did not spill");
    if (auto selected = own->server->SelectRequestBranches(*model, active); !selected)
      return selected;
    const auto before = model->graphs();
    auto begun = (*branch)->BeginPrompt(history, 0, false, true);
    if (!begun) return std::unexpected(begun.error());
    auto restoration = std::move(*begun);
    while (!restoration->done())
      if (auto advanced = restoration->Advance(); !advanced) return advanced;
    if (restoration->reused() != history.size())
      return std::unexpected("restore recomputed its history");
    if (auto finished = restoration->Finish(); !finished) return finished;
    restoration.reset();
    const auto after = model->graphs();
    if (before.eager != after.eager || before.captured != after.captured ||
        before.replayed != after.replayed || (*branch)->spilled())
      return std::unexpected("restore executed model work or left spilled state");
    auto restored = StateImage(*own, engine);
    if (!restored) return std::unexpected(restored.error());
    if (*restored != *state) return std::unexpected("restored state differs from the spilled");
    note("restored.0", *restored);

    // Continue from the restored history: the last sampled token, then 16 more.
    auto next = history;
    next.push_back(first.tokens.back());
    std::uint32_t reused = 0;
    std::vector<float> continued;
    if (auto prepared = (*branch)->PreparePrompt(next, 0, continued, reused); !prepared)
      return prepared;
    if (reused + 1 != next.size()) return std::unexpected("continuation lost its prefix");
    if (!Finite(continued)) return std::unexpected("continuation head is not finite");
    note("heads.1", ts::HexOfFloats(continued));
    options.max_tokens = kSecond;
    rt::Generation second;
    if (auto generated = (*branch)->Generate(continued, options, second); !generated)
      return generated;
    if (second.tokens.size() != kSecond || second.logits.empty())
      return std::unexpected("the continuation did not generate completely");
    for (const auto& row : second.logits)
      if (!Finite(row)) return std::unexpected("a continuation logit row is not finite");
    note("tokens.1", ts::JoinTokens(second.tokens));
    note("rows.1", std::to_string(second.logits.size()) + ':' + ts::HexOfRows(second.logits));
    note("drafted.1", std::to_string(second.drafted));
    note("accepted.1", std::to_string(second.accepted));
    note("history.1", ts::HexOfInts((*branch)->history()));
    auto settled = StateImage(*own, engine);
    if (!settled) return std::unexpected(settled.error());
    note("state.1", *settled);
    if (engine.runner().coverage().violations != 0 || !model->violations().empty())
      return std::unexpected("the run used uncovered memory or recorded violations");
    return own->server->EndRequestBranches(*model);
  }();
  if (!completed) {
    const auto reason = completed.error();
    std::ignore = own.release();  // an unproven owner is kept, never destroyed
    return std::unexpected(reason + "; complete failed owner retained");
  }
  if (auto retired = own->server->TearDown(); !retired) {
    const auto reason = retired.error();
    std::ignore = own.release();
    return std::unexpected(reason + "; unproven owner retained");
  }
  return out;
}

class GemmaAssistantV1Gpu : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GemmaAssistantV1Gpu, PublishedPairMatchesTheFrozenDigests) {
  const Profile& profile = kProfiles.at(GetParam());
  const auto store = ts::V1Store();
  const auto target = ts::V1IdNamed(profile.target);
  const auto assistant = ts::V1IdNamed(profile.assistant);
  if (!target || !assistant) {
    GTEST_FAIL() << "no published v1 " << profile.target << " and its assistant in " << store;
  }
  const auto v1 = RunArm(profile, {store, *target, *assistant});
  ASSERT_TRUE(v1) << (v1 ? "" : v1.error());
  ts::FrozenDigests frozen(std::string("gemma_assistant_") + profile.name);
  for (const auto& [key, value] : *v1) frozen.Expect(key, value);
  RecordProperty("v1_target", *target);
  RecordProperty("v1_assistant", *assistant);
}

INSTANTIATE_TEST_SUITE_P(Approved, GemmaAssistantV1Gpu, ::testing::Values(0U, 1U),
                         [](const auto& info) { return kProfiles.at(info.param).name; });
}  // namespace
