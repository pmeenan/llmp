// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 on the component engine's shared route: the derived serving
// domain's arithmetic, the automatic serving envelope (the route's only one:
// every other request is refused before its source is read), and the
// publication screens (Published*) of the published v1 artifacts (the target
// d2046349 and the DSpark drafter e14bf29f; LLMP_DSV4_TARGET_STORE, else
// tests/support/v1_frozen.h V1Store) against their frozen digests
// (tests/unit/data/v1_frozen/dsv4_*.json). The serving tests
// (dsv4_engine_serving_gpu_test.cc) cover the rest of its service.

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/component_source.h"
#include "expected_error.h"
#include "model/state_contract.h"
#include "platform/host_probe.h"
#include "runtime/component_build.h"
#include "runtime/memory_guard.h"
#include "runtime/serving.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace en = llmp::engine;
namespace rt = llmp::runtime;
namespace pl = llmp::planner;
using Status = std::expected<void, std::string>;
using llmp::test_support::Failed;
constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
constexpr std::uint64_t kCap = 110 * kGiB;
constexpr std::uint32_t kContext = 3072, kChunk = 512, kPrefix = 2560;
constexpr std::uint32_t kVocab = 129280;

constexpr std::string_view kArtifact =
    "d2046349e81da7019c9d787c2eb92521b7a3f4ca9d09c398448da1a99f8eef3f";

constexpr std::string_view kDsparkArtifact =
    "e14bf29f28c408a742ba6ab19879e415b1ec13019c0ed33b87b7802232a0bd90";

// The store holding both: LLMP_DSV4_TARGET_STORE, else the published v1
// store.
std::string Store() {
  const char* store = std::getenv("LLMP_DSV4_TARGET_STORE");  // NOLINT(concurrency-mt-unsafe)
  return store != nullptr ? std::string(store) : llmp::test_support::V1Store();
}

rt::ModelSettings Settings() {
  rt::ModelSettings s;
  s.name = "deepseek-component-target";
  s.architecture = "deepseek4";
  s.container = "gguf";
  s.context.value = kContext;
  s.prefill_chunk.value = kChunk;
  s.max_slots.value = 1;
  s.wave_read_align.value = 256;
  return s;
}
// Bounded streaming checks before JSON/tokenizer construction. Artifact::Open
// authenticates every prepared header again; no full weight hash pass is added.
Status File(const fs::path& path, std::uint64_t bytes, std::string_view digest) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return std::unexpected("missing fixed DeepSeek metadata: " + path.string());
  llmp::base::Sha256 hash;
  std::array<std::byte, 65536> buffer{};
  std::uint64_t read = 0;
  while (read < bytes) {
    const auto wanted =
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes - read, buffer.size()));
    const auto got = std::fread(buffer.data(), 1, wanted, f);
    if (got != wanted) {
      std::fclose(f);
      return std::unexpected("short fixed DeepSeek metadata");
    }
    hash.Update(std::span(buffer).first(got));
    read += got;
  }
  const bool eof = std::fgetc(f) == EOF && !std::ferror(f);
  std::fclose(f);
  if (!eof || llmp::base::ToHex(hash.Finish()) != digest)
    return std::unexpected("fixed DeepSeek metadata identity differs");
  return {};
}
Status Metadata(const fs::path& artifact) {
  if (auto s = File(artifact / "manifest.json", 14663, kArtifact); !s) return s;
  if (auto s = File(artifact / "index.json", 5058896,
                    "4f84772b3cfa5912e6a5ece00f438b9b8ac330679dd848a1896810f7d48a8e80");
      !s)
    return s;
  return File(artifact / "meta/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.kv.gguf", 5257664,
              "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b");
}

TEST(Dsv4EngineTargetInputs, ServingDomainUsesActualBlockStateAndOwnsItsPayment) {
  namespace md = llmp::model;
  using C = md::StateTensorContract;
  llmp::spec::ModelSpec spec;
  spec.pipeline = llmp::spec::PipelineKind::kLlm;
  spec.stages.resize(2);
  spec.stages[0].name = "target";
  spec.stages[0].role = llmp::spec::StageRole::kTarget;
  spec.stages[1].name = "block";
  spec.stages[1].role = llmp::spec::StageRole::kDrafter;
  const std::vector<std::pair<std::uint32_t, md::StateContract>> target = {
      {0, {.tensors = {C{.name = "kv", .cell_bytes = 256}}}}};
  const std::vector<std::pair<std::uint32_t, md::StateContract>> block = {
      {0,
       {.tensors = {C{.name = "ring",
                      .kind = md::StateKind::kDrafterRing,
                      .cell_bytes = 256,
                      .cells = md::CellRule::kPositionRing,
                      .fixed_cells = 256,
                      .rollback = md::Rollback::kSnapshot,
                      .target_writes = true}}}}};
  auto a = md::Layout(target, 8192, 512);
  auto b = md::Layout(block, 8192, 512);
  ASSERT_TRUE(a) << Failed(a).value_or("");
  ASSERT_TRUE(b) << Failed(b).value_or("");
  const std::array layouts{*a, *b};
  const std::array batching{md::Batching{.max_segments = 4}, md::Batching{.max_segments = 4}};
  en::DecodingMode mode{
      .kind = en::DecodingMode::Kind::kSpeculative,
      .drafters = {{.stage = 1, .shape = en::DrafterSpec::Shape::kBlock, .max_depth = 3}},
      .combine = en::DecodingMode::Drafters::kCascade,
      .row_invariant = false};
  llmp::catalog::Catalog catalog;
  auto allowance =
      en::ExecutionMetadataAllowance::Admit(catalog, catalog.AddDomain("serving domain"), 2 * kMiB,
                                            2 * kMiB + rt::kUncountedMargin, rt::kUncountedMargin);
  ASSERT_TRUE(allowance) << Failed(allowance).value_or("");
  for (const auto owners : {1U, 4U}) {
    auto domain = rt::ComponentServingDomain(spec, layouts, batching, owners, mode, *allowance);
    ASSERT_TRUE(domain) << Failed(domain).value_or("");
    EXPECT_GT((*allowance)->used(), 0U);
    EXPECT_TRUE((*domain)->CheckMode(en::NumericalUse::kPrompt));
    EXPECT_TRUE((*domain)->CheckMode(en::NumericalUse::kSpeculative));
    EXPECT_EQ((*domain)->stages()[0].phases[2].max_owners, owners);
    EXPECT_EQ((*domain)->stages()[0].phases[2].max_owner_rows, 4U);
  }
  EXPECT_EQ((*allowance)->used(), 0U);
  auto invalid = mode;
  invalid.drafters[0].stage = 0;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, batching, 4, invalid, *allowance));
  invalid = mode;
  invalid.drafters[0].max_depth = 512;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, batching, 4, invalid, *allowance));
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, batching, 4, mode, nullptr));
  EXPECT_EQ((*allowance)->used(), 0U);
  auto block_shared = mode;
  block_shared.drafters[0].shared_wave_depth = 2;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, batching, 4, block_shared, *allowance));
  spec.stages.resize(1);
  auto plain = rt::ComponentServingDomain(spec, std::span(layouts).first(1),
                                          std::span(batching).first(1), 4, {}, *allowance);
  ASSERT_TRUE(plain) << Failed(plain).value_or("");
  EXPECT_FALSE((*plain)->CheckMode(en::NumericalUse::kSpeculative));
}

TEST(Dsv4EngineTargetInputs, SpeculationWithoutAnOwnedDrafterRefusesBeforeSourceRead) {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  llmp::config::ModelEntry entry;
  entry.name = Settings().name;
  entry.artifact = std::string(kArtifact);
  auto settings = Settings();
  settings.speculation.value = true;
  auto build = rt::ComponentLlmBuild(node, "/nonexistent-serving-source", entry, 0);
  const auto refused = build(settings);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error(), "component speculation needs an owned drafter artifact");
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), 0U);
}

namespace native_target {
// Reuse the complete initialized-state reader and stopped-owner payment
// handoff from the Qwen native publication proof, over this family's Server.
namespace cfg = llmp::config;
constexpr std::uint64_t kObservations = 128 * kMiB;
// Both arms: 36 full rows, growth capacity, six coordinate inventories,
// histories and the copy window. The catalog charge precedes every result.
static_assert(3 * 36ULL * kVocab * sizeof(float) + 6 * 4096 * 24 + 4 * kMiB < kObservations);
struct StateImage {
  std::vector<std::array<std::uint64_t, 3>> ranges;
  llmp::base::Sha256Digest digest{};
  std::uint32_t positions = 0;
  bool operator==(const StateImage&) const = default;
};
struct Result {
  std::uint64_t selected_budget = 0;
  std::array<StateImage, 3> states;
  std::array<std::vector<float>, 2> heads;
  std::array<rt::Generation, 2> generations;
  std::array<std::vector<std::int32_t>, 2> history;
};
struct Owner {
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  std::unique_ptr<rt::Server> server;
  std::shared_ptr<en::ExecutionMetadataAllowance> observations;
  std::unique_ptr<rt::Llm::PromptSession> restoration;
  Result result;
  fs::path scratch;
  std::FILE* log = nullptr;
  ~Owner() {
    if (log) std::fclose(log);
  }
};
std::expected<StateImage, std::string> ReadState(Owner& life, rt::Llm& model) {
  StateImage state;
  auto& slots = model.runner().slots();
  auto& node = life.server->node();
  state.positions = slots.positions(0);
  for (std::size_t stage = 0; stage < slots.stages(); ++stage)
    for (const auto& range : slots.live(0, stage).used_ranges()) {
      if (state.ranges.size() == 4096)
        return std::unexpected("state observation exceeds prepaid coordinates");
      state.ranges.push_back({stage, range.offset, range.bytes});
    }
  if (state.ranges.empty()) return std::unexpected("state observation is empty");
  constexpr std::uint64_t part_bytes = 1ULL << 20U;
  std::vector<llmp::catalog::ExtentId> extents;
  auto buffer = node.Pinned(part_bytes, 0, extents);
  if (!buffer) return std::unexpected(buffer.error());
  llmp::base::Sha256 hash;
  for (const auto& range : state.ranges)
    for (std::uint64_t at = 0; at < range[2]; at += part_bytes) {
      const en::LiveState::Range part{.region = range[0],
                                      .offset = range[1] + at,
                                      .bytes = std::min(part_bytes, range[2] - at)};
      auto retired = en::LiveState::CopyRetirement::kUnproven;
      auto copied = slots.CopyState(0, *buffer, std::span(&part, 1), true, &retired);
      if (retired == en::LiveState::CopyRetirement::kUnproven) {
        node.KeepPinned(*buffer);
        return std::unexpected("state copy retirement is unproven");
      }
      if (!copied) {
        (void)node.FreePinned(*buffer);
        return std::unexpected(copied.error());
      }
      hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
    }
  if (auto freed = node.FreePinned(*buffer); !freed) return std::unexpected(freed.error());
  state.digest = hash.Finish();
  return state;
}

std::expected<std::unique_ptr<Owner>, std::string> Run(const fs::path& store, std::string_view id,
                                                       Owner* previous = nullptr,
                                                       std::string_view drafter = {}) {
  const auto* scratch = std::getenv("LLMP_TEST_SCRATCH");
  if (!scratch) return std::unexpected("missing publication scratch");
  auto life = std::make_unique<Owner>();
  std::error_code ec;
  fs::create_directories(scratch, ec);
  if (ec) return std::unexpected(ec.message());
  auto path = (fs::path(scratch) / "ds-native-XXXXXX").string();
  if (!mkdtemp(path.data())) return std::unexpected("creating native target scratch");
  life->scratch = path;
  life->roles.installed = store;
  life->roles.spill = life->scratch / "spill";
  life->roles.state = life->scratch / "state";
  for (const auto& dir : {life->roles.spill, life->roles.state}) {
    fs::create_directory(dir, ec);
    if (ec || chmod(dir.c_str(), 0700)) return std::unexpected("protecting native target state");
  }
  cfg::ModelEntry entry;
  entry.name = "deepseek";
  entry.artifact = id;
  if (!drafter.empty()) entry.drafter = std::string(drafter);
  entry.overrides["context"] = std::int64_t{kContext};
  entry.overrides["prefill_chunk"] = std::int64_t{kChunk};
  entry.overrides["max_slots"] = std::int64_t{1};
  life->config.models.push_back(std::move(entry));
  life->options.diagnostic_auto_startup = true;
  // DSpark's weights and state need the node's full 112 GiB (the speed gate's cap).
  life->options.diagnostic_budget_cap_bytes = drafter.empty() ? kCap : 120259084288ULL;
  life->log = std::tmpfile();
  if (!life->log) return std::unexpected("opening native target log");
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log);
  auto completed = [&]() -> Status {
    if (auto started = life->server->Start(true); !started) return started;
    auto& node = life->server->node();
    life->result.selected_budget = node.budget().value();
    if (auto paid = node.Call(
            [&]() -> en::Status {
              const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
              const auto cap = node.budget().value();
              const auto charge = kObservations + en::ExecutionMetadataAllowance::OverheadBytes();
              if (occupied > cap || charge > cap - occupied)
                return std::unexpected("native target observations exceed node capacity");
              auto admitted = en::ExecutionMetadataAllowance::Admit(
                  node.catalog(), node.domain(), charge,
                  llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
              if (!admitted) return std::unexpected(admitted.error());
              life->observations = std::move(*admitted);
              if (!life->observations->Charge(kObservations))
                return std::unexpected("native target observations lack payment");
              return {};
            },
            "pay native DS observations");
        !paid)
      return paid;
    // Only the new paid owner may take over both arms' retained host results.
    if (previous) previous->server.reset();
    auto* model = dynamic_cast<rt::Llm*>(life->server->Find("deepseek"));
    if (!model || model->logit_width() != kVocab || model->generation_wave_capacity() != 1)
      return std::unexpected("native target selected the wrong source or serving envelope");
    rt::SwapParts parts;
    if (auto activated = life->server->Activate(*model, parts); !activated) return activated;
    auto branch = model->branch(0);
    if (!branch) return std::unexpected(branch.error());
    std::array<rt::Llm::Branch*, 1> active{*branch};
    if (auto selected = life->server->SelectRequestBranches(*model, active); !selected)
      return selected;
    std::array<std::int32_t, kPrefix> prompt{};
    for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 100 + (i * 7919U + 17U) % 30000U;
    if (auto prefill = (*branch)->Prefill(prompt, life->result.heads[0]); !prefill) return prefill;
    for (unsigned turn = 0; turn != 2; ++turn) {
      rt::GenerateOptions options;
      options.max_tokens = 8;
      options.stop = false;
      options.keep_logits = true;
      auto& generation = life->result.generations[turn];
      if (auto generated = (*branch)->Generate(life->result.heads[turn], options, generation);
          !generated)
        return generated;
      if (generation.tokens.size() != 8 || generation.logits.size() != 8 ||
          (drafter.empty() ? generation.drafted != 0 : generation.drafted == 0) ||
          generation.stopped || generation.cancelled)
        return std::unexpected(drafter.empty()
                                   ? "native target returned incomplete plain generation"
                                   : "native target did not speculate with its drafter");
      life->result.history[turn] = (*branch)->history();
      auto state = ReadState(*life, *model);
      if (!state) return std::unexpected(state.error());
      life->result.states[turn == 0 ? 0 : 2] = std::move(*state);
      if (turn) continue;
      if (auto ended = life->server->EndRequestBranches(*model); !ended) return ended;
      if (auto spilled = model->SpillIdle(**branch); !spilled) return spilled;
      if (!(*branch)->spilled()) return std::unexpected("native target did not spill");
      if (auto selected = life->server->SelectRequestBranches(*model, active); !selected)
        return selected;
      const auto before = model->graphs();
      auto begun = (*branch)->BeginPrompt(life->result.history[0], 0, false, true);
      if (!begun) return std::unexpected(begun.error());
      life->restoration = std::move(*begun);
      while (!life->restoration->done())
        if (auto advanced = life->restoration->Advance(); !advanced) return advanced;
      if (life->restoration->reused() != life->result.history[0].size())
        return std::unexpected("native target restore recomputed its history");
      if (auto finished = life->restoration->Finish(); !finished) return finished;
      life->restoration.reset();
      const auto after = model->graphs();
      if (before.eager != after.eager || before.captured != after.captured ||
          before.replayed != after.replayed || (*branch)->spilled())
        return std::unexpected("native target restoration executed model work");
      auto restored = ReadState(*life, *model);
      if (!restored) return std::unexpected(restored.error());
      life->result.states[1] = std::move(*restored);
      if (life->result.states[0] != life->result.states[1])
        return std::unexpected("native target restored different state");
      auto next = (*branch)->history();
      next.push_back(generation.tokens.back());
      std::uint32_t reused = 0;
      if (auto continued = (*branch)->PreparePrompt(next, 0, life->result.heads[1], reused);
          !continued)
        return continued;
      if (reused + 1 != next.size())
        return std::unexpected("native target continuation lost prefix");
    }
    if (model->runner().coverage().violations)
      return std::unexpected("native target used uncovered memory");
    return life->server->EndRequestBranches(*model);
  }();
  if (!completed) {
    const auto error = completed.error();
    std::ignore = life.release();
    return std::unexpected(error + "; complete failed owner retained");
  }
  if (auto retired = life->server->TearDown(); !retired) {
    const auto error = retired.error();
    std::ignore = life.release();
    return std::unexpected(error + "; unproven owner retained");
  }
  return life;
}
// The published v1 target (and DSpark) against the frozen digests: heads,
// greedy tokens with their logit rows, the state after each turn and after a
// spill and restore. LLMP_TEST_DSV4_V1_STORE (or LLMP_DSV4_TARGET_STORE, else
// the v1 store) and LLMP_TEST_DSV4_V1_ID name another store and target.
void PublishedScreen(bool dspark) {
  const char* native_env = std::getenv("LLMP_TEST_DSV4_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
  const fs::path store = native_env != nullptr ? fs::path(native_env) : fs::path(Store());
  const char* given = std::getenv("LLMP_TEST_DSV4_V1_ID");  // NOLINT(concurrency-mt-unsafe)
  const std::string id = given != nullptr ? given : std::string(kArtifact);
  ASSERT_EQ(id.size(), 64U);
  ASSERT_TRUE(std::ranges::all_of(
      id, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
  const std::string drafter = dspark ? std::string(kDsparkArtifact) : std::string();
  std::error_code ec;
  if (!fs::exists(store / id, ec) || (dspark && !fs::exists(store / drafter, ec)))
    GTEST_FAIL() << "no DeepSeek V4 v1 artifact " << id << (dspark ? " or DSpark " : "") << drafter
                 << " in " << store;
  if (id == kArtifact) {
    auto checked = Metadata(store / kArtifact);
    ASSERT_TRUE(checked) << Failed(checked).value_or("");
  }
  auto native = native_target::Run(store, id, nullptr, drafter);
  ASSERT_TRUE(native) << Failed(native).value_or("");
  llmp::test_support::FrozenDigests frozen(dspark ? "dsv4_dspark" : "dsv4_target");
  const auto& r = (*native)->result;
  using llmp::test_support::HexOfFloats;
  using llmp::test_support::HexOfInts;
  using llmp::test_support::HexOfRows;
  for (unsigned turn = 0; turn != 2; ++turn) {
    const auto t = std::to_string(turn);
    frozen.Expect("heads." + t, HexOfFloats(r.heads[turn]));
    frozen.Expect("tokens." + t, llmp::test_support::JoinTokens(r.generations[turn].tokens));
    frozen.Expect("logits." + t, HexOfRows(r.generations[turn].logits));
    frozen.Expect("history." + t, HexOfInts(r.history[turn]));
    if (dspark) {
      frozen.Expect("drafted." + t, std::to_string(r.generations[turn].drafted));
      frozen.Expect("accepted." + t, std::to_string(r.generations[turn].accepted));
    }
  }
  // States: 0 after the first turn, 1 after spill and restore, 2 after the second.
  for (std::size_t i = 0; i < r.states.size(); ++i)
    frozen.Expect("state." + std::to_string(i), llmp::base::ToHex(r.states[i].digest) + ':' +
                                                    std::to_string(r.states[i].positions) + ':' +
                                                    std::to_string(r.states[i].ranges.size()));
  const auto finite = [](std::span<const float> row) {
    EXPECT_EQ(row.size(), kVocab);
    EXPECT_TRUE(std::ranges::all_of(row, [](float f) { return std::isfinite(f); }));
  };
  for (unsigned turn = 0; turn != 2; ++turn) {
    finite(r.heads[turn]);
    for (const auto& row : r.generations[turn].logits) finite(row);
  }
  testing::Test::RecordProperty("complete_head_rows", 18);
  testing::Test::RecordProperty("head_values_per_row", kVocab);
  testing::Test::RecordProperty("complete_state_images", 3);
  testing::Test::RecordProperty("restored_owners", 1);
  testing::Test::RecordProperty("prompt_tokens", kPrefix);
  testing::Test::RecordProperty("native_artifact", id);
  if (dspark) testing::Test::RecordProperty("native_drafter", drafter);
  testing::Test::RecordProperty("v1_selected_node_budget", std::to_string(r.selected_budget));
  fs::remove_all((*native)->scratch);
}
TEST(Dsv4EngineTargetGpu, PublishedGgufMatchesFrozenHeadsStateAndSpill) { PublishedScreen(false); }
TEST(Dsv4EngineTargetGpu, PublishedDsparkMatchesFrozenHeadsStateAndSpill) { PublishedScreen(true); }

// ---- DeepSeek V4 community IQ2_XXS ------------------------------------
// antirez/deepseek-v4-gguf@f71f23d5's chat-v2 imatrix GGUF (IQ2_XXS
// gate/up, Q2_K down, F16 compressor APE and mixing tables), its published
// v1 import ("deepseek-iq2xxs" in the pinned v1 IDs) served plain greedy by
// the same screen as the UD target: the prompt's head, two eight-token turns
// with their logit rows, the state after each turn and after a spill and
// restore, against its frozen digests (tests/unit/data/v1_frozen/
// dsv4_iq2xxs.json). Fails when the store does not hold it.
TEST(Dsv4EngineTargetGpu, PublishedIq2xxsMatchesFrozenHeadsStateAndSpill) {
  const auto id = llmp::test_support::V1IdNamed("deepseek-iq2xxs");
  ASSERT_TRUE(id) << "no deepseek-iq2xxs in " << llmp::test_support::V1IdsPath();
  const fs::path store = Store();
  std::error_code ec;
  if (!fs::exists(store / *id / "manifest.json", ec))
    GTEST_FAIL() << "no DeepSeek V4 community IQ2_XXS v1 artifact " << *id << " in " << store;
  auto native = native_target::Run(store, *id);
  ASSERT_TRUE(native) << Failed(native).value_or("");
  llmp::test_support::FrozenDigests frozen("dsv4_iq2xxs");
  const auto& r = (*native)->result;
  for (unsigned turn = 0; turn != 2; ++turn) {
    const auto t = std::to_string(turn);
    frozen.Expect("heads." + t, llmp::test_support::HexOfFloats(r.heads[turn]));
    frozen.Expect("tokens." + t, llmp::test_support::JoinTokens(r.generations[turn].tokens));
    frozen.Expect("logits." + t, llmp::test_support::HexOfRows(r.generations[turn].logits));
    frozen.Expect("history." + t, llmp::test_support::HexOfInts(r.history[turn]));
    EXPECT_EQ(r.heads[turn].size(), kVocab);
    EXPECT_TRUE(std::ranges::all_of(r.heads[turn], [](float f) { return std::isfinite(f); }));
  }
  // States: 0 after the first turn, 1 after spill and restore, 2 after the second.
  for (std::size_t i = 0; i < r.states.size(); ++i)
    frozen.Expect("state." + std::to_string(i), llmp::base::ToHex(r.states[i].digest) + ':' +
                                                    std::to_string(r.states[i].positions) + ':' +
                                                    std::to_string(r.states[i].ranges.size()));
  testing::Test::RecordProperty("native_artifact", *id);
  testing::Test::RecordProperty("prompt_tokens", kPrefix);
  fs::remove_all((*native)->scratch);
}
// ---- end DeepSeek V4 community IQ2_XXS --------------------------------
}  // namespace native_target

}  // namespace
