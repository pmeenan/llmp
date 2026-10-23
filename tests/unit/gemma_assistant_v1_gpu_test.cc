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
#include <format>
#include <memory>
#include <optional>
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
#include "llm_test_access.h"
#include "runtime/serving.h"
#include "u4_diagnostic_export.h"
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

struct ArmDiagnostics {
  std::optional<std::string> target_policy;
  std::optional<std::string> drafter_policy;
  std::string expected_target_policy;
  std::string expected_drafter_policy;
  bool observe_decode = false;
  std::size_t observation_events = rt::LlmDecodeObservation::kMaxEvents;
  std::size_t observation_tokens = rt::LlmDecodeObservation::kMaxTokensPerEvent;
  bool full_head = false;
  std::optional<fs::path> export_directory;
  std::string model_name, target_id, drafter_id, transport;
};

struct Owner {
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  std::FILE* log = nullptr;
  fs::path scratch;
  // Keep the trace alive until the server and its wrapped policy callbacks
  // have been destroyed. Its bounded allocation is charged to that node.
  std::unique_ptr<rt::LlmDecodeObservation> observation;
  std::uint64_t observation_charge = 0;
  std::optional<ts::U4JsonlFile> diagnostic_log;
  std::unique_ptr<rt::Server> server;
  ~Owner() {
    if (log != nullptr && !diagnostic_log) std::fclose(log);
    if (!scratch.empty()) {
      std::error_code ignored;
      fs::remove_all(scratch, ignored);
    }
  }
};

// The complete used state of slot 0, every stage, hashed through pinned host
// memory the test funds beside the node's catalog.
std::expected<std::string, std::string> StateImage(Owner& own, rt::Llm& model,
                                                   ts::U4JsonlFile* export_file = nullptr,
                                                   std::string_view phase = {}) {
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
  if (export_file &&
      !export_file->Write(std::format("{{\"type\":\"state\",\"phase\":{},\"ranges\":[",
                                      std::format("\"{}\"", phase))))
    return std::unexpected("writing Gemma state header failed");
  for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
    for (const en::LiveState::Range& range : slots.live(0, stage).used_ranges()) {
      ++ranges;
      if (export_file) {
        if (ranges > 1 && !export_file->Write(","))
          return std::unexpected("writing Gemma state separator failed");
        if (!export_file->Write(std::format("[{}, {}, {}]", stage, range.offset, range.bytes)))
          return std::unexpected("writing Gemma state coordinate failed");
      }
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
  if (export_file &&
      !export_file->Line(std::format("],\"sha256\":\"{}\",\"bytes\":{},\"positions\":{}}}", digest,
                                     bytes, positions)))
    return std::unexpected("writing Gemma state completion failed");
  return digest + ':' + std::to_string(bytes) + ':' + std::to_string(ranges) + ':' +
         std::to_string(positions);
}

bool Finite(std::span<const float> row) {
  return row.size() == kVocab && std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
}

std::expected<Observations, std::string> RunArm(const Profile& profile, const Arm& arm,
                                                ArmDiagnostics diagnostic = {}) {
  auto own = std::make_unique<Owner>();
  Observations out;
  std::optional<ts::U4DiagnosticExport> export_to;
  std::optional<ts::U4JsonlFile> result_export;
  std::error_code error;
  const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const fs::path root = base != nullptr ? base : ::testing::TempDir();
  fs::create_directories(root, error);
  std::string name = (root / "gemma-assistant-v1-XXXXXX").string();
  if (mkdtemp(name.data()) == nullptr) return std::unexpected("creating test scratch");
  own->scratch = name;
  own->roles.installed = arm.store;
  own->options.diagnostic_target_policy = diagnostic.target_policy;
  own->options.diagnostic_drafter_policy = diagnostic.drafter_policy;
  own->options.debug_plans = diagnostic.observe_decode;
  if (diagnostic.observe_decode) {
    if (!diagnostic.export_directory || diagnostic.model_name.empty() ||
        diagnostic.target_id.empty() || diagnostic.drafter_id.empty() ||
        diagnostic.transport.empty())
      return std::unexpected("diagnostic run lacks its explicit enrollment");
    auto created = ts::U4DiagnosticExport::Create(*diagnostic.export_directory);
    if (!created) return std::unexpected(created.error());
    export_to.emplace(std::move(*created));
    auto enrolled = ts::U4WriteEnrollment(*export_to, diagnostic.model_name, arm.store.string(),
                                          diagnostic.target_id, diagnostic.drafter_id,
                                          diagnostic.transport, diagnostic.expected_target_policy,
                                          diagnostic.expected_drafter_policy);
    if (!enrolled) return std::unexpected(enrolled.error());
    auto result_file = export_to->Open("results.jsonl");
    if (!result_file) return std::unexpected(result_file.error());
    result_export.emplace(std::move(*result_file));
    auto runtime_file = export_to->Open("runtime.log");
    if (!runtime_file) return std::unexpected(runtime_file.error());
    own->diagnostic_log.emplace(std::move(*runtime_file));
    own->log = own->diagnostic_log->Raw();
  }
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
  if (own->log == nullptr) own->log = std::tmpfile();
  if (own->log == nullptr) return std::unexpected("opening test log");
  own->server = std::make_unique<rt::Server>(own->config, own->roles, own->options, own->log);

  bool result_export_failed = false;
  const auto note = [&out, &result_export, &result_export_failed](std::string key,
                                                                  std::string value) {
    if (result_export &&
        (!result_export->Write("{\"key\":") || !ts::U4WriteJsonString(*result_export, key) ||
         !result_export->Write(",\"value\":") || !ts::U4WriteJsonString(*result_export, value) ||
         !result_export->Line("}")))
      result_export_failed = true;
    out.emplace_back(std::move(key), std::move(value));
  };
  if (diagnostic.observe_decode) {
    if (diagnostic.expected_target_policy.empty() || diagnostic.expected_drafter_policy.empty())
      return std::unexpected("diagnostic enrollment needs the registered expected policy texts");
    note("decode.transport", diagnostic.full_head ? "full-head" : "configured-default");
  }
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
    if (export_to &&
        !result_export->Line(std::format("{{\"type\":\"envelope\",\"context\":{},"
                                         "\"prefill_chunk\":{},\"max_slots\":{},"
                                         "\"vocabulary\":{}}}",
                                         kContext, profile.prefill_chunk, kSlots, kVocab)))
      return std::unexpected("writing Gemma envelope record failed");
    if (diagnostic.observe_decode) {
      own->observation_charge = rt::LlmDecodeObservation::BoundedBytes(
          diagnostic.observation_events, diagnostic.observation_tokens);
      if (own->observation_charge == 0 ||
          !own->server->node().ChargeHost(own->observation_charge, false))
        return std::unexpected("cannot fund bounded decode observation");
      own->observation = std::make_unique<rt::LlmDecodeObservation>(diagnostic.observation_events,
                                                                    diagnostic.observation_tokens);
    }
    auto& engine = dynamic_cast<rt::Llm&>(*model);
    rt::SwapParts parts;
    if (auto activated = own->server->Activate(*model, parts); !activated) return activated;
    auto branch = model->branch(0);
    if (!branch) return std::unexpected(branch.error());
    if (diagnostic.observe_decode) {
      const auto installed = rt::LlmTestAccess::InstallDecodeObservation(engine, *own->observation,
                                                                         diagnostic.full_head);
      if (!installed) return installed;
    } else if (diagnostic.full_head) {
      return std::unexpected("full-head diagnostic requires decode observation");
    }
    std::array<rt::Llm::Branch*, 1> active{*branch};
    if (auto selected = own->server->SelectRequestBranches(*model, active); !selected)
      return selected;

    constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
    std::vector<std::int32_t> prompt;
    for (int i = 0; i < 24; ++i) prompt.insert(prompt.end(), seed.begin(), seed.end());
    if (diagnostic.observe_decode) note("prompt", ts::HexOfInts(prompt));
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
    if (diagnostic.observe_decode && first.logits.size() != kFirst)
      return std::unexpected("the first diagnostic generation lacks complete kept rows");
    for (const auto& row : first.logits)
      if (!Finite(row)) return std::unexpected("a generation logit row is not finite");
    note("tokens.0", ts::JoinTokens(first.tokens));
    note("rows.0", std::to_string(first.logits.size()) + ':' + ts::HexOfRows(first.logits));
    note("drafted.0", std::to_string(first.drafted));
    note("accepted.0", std::to_string(first.accepted));
    note("steps.0", std::to_string(first.steps));
    const auto history = (*branch)->history();
    note("history.0", ts::HexOfInts(history));
    auto state =
        StateImage(*own, engine, result_export ? &*result_export : nullptr, "after-turn-0");
    if (!state) return std::unexpected(state.error());
    note("state.0", *state);
    if (export_to) {
      auto snapshot = ts::U4WritePlanSnapshot(engine, *export_to, "turn0");
      if (!snapshot) return std::unexpected(snapshot.error());
    }

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
    auto restored =
        StateImage(*own, engine, result_export ? &*result_export : nullptr, "after-spill-restore");
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
    if (diagnostic.observe_decode && second.logits.size() != kSecond)
      return std::unexpected("the continuation diagnostic lacks complete kept rows");
    for (const auto& row : second.logits)
      if (!Finite(row)) return std::unexpected("a continuation logit row is not finite");
    note("tokens.1", ts::JoinTokens(second.tokens));
    note("rows.1", std::to_string(second.logits.size()) + ':' + ts::HexOfRows(second.logits));
    note("drafted.1", std::to_string(second.drafted));
    note("accepted.1", std::to_string(second.accepted));
    note("history.1", ts::HexOfInts((*branch)->history()));
    auto settled =
        StateImage(*own, engine, result_export ? &*result_export : nullptr, "after-turn-1");
    if (!settled) return std::unexpected(settled.error());
    note("state.1", *settled);
    if (diagnostic.observe_decode) {
      const auto& trace = *own->observation;
      if (trace.incomplete || trace.count == 0 || trace.count > trace.events.size())
        return std::unexpected("decode observation was incomplete or empty");
      auto observation =
          ts::U4WriteObservation(*export_to, diagnostic.model_name, diagnostic.transport, trace);
      if (!observation) return std::unexpected(observation.error());
    }
    if (export_to) {
      auto snapshot = ts::U4WritePlanSnapshot(engine, *export_to, "turn1");
      if (!snapshot) return std::unexpected(snapshot.error());
    }
    if (result_export_failed) return std::unexpected("writing result digest export failed");
    if (result_export) {
      auto closed = result_export->Close();
      if (!closed) return closed;
    }
    if (own->diagnostic_log && !own->diagnostic_log->Flush())
      return std::unexpected("flushing full runtime diagnostic log failed");
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
  if (own->diagnostic_log) {
    own->server.reset();
    auto closed = own->diagnostic_log->Close();
    if (!closed) return std::unexpected(closed.error());
    own->log = nullptr;
    auto marked = export_to->Complete();
    if (!marked) return std::unexpected(marked.error());
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

TEST(GemmaAssistantV1Gpu, DISABLED_U4DiagnosticGemma26) {
  auto enrollment = ts::U4ManualEnrollment::Read("gemma26", true, true);
  ASSERT_TRUE(enrollment) << (enrollment ? "" : enrollment.error());
  ArmDiagnostics diagnostic;
  diagnostic.observe_decode = true;
  diagnostic.full_head = enrollment->transport == "full-head";
  diagnostic.export_directory = enrollment->export_directory;
  diagnostic.model_name = enrollment->model;
  diagnostic.target_id = enrollment->target_id;
  diagnostic.drafter_id = enrollment->drafter_id;
  diagnostic.transport = enrollment->transport;
  diagnostic.expected_target_policy = enrollment->expected_target_policy;
  diagnostic.expected_drafter_policy = enrollment->expected_drafter_policy;
  const auto run =
      RunArm(kProfiles[0], {enrollment->store, enrollment->target_id, enrollment->drafter_id},
             std::move(diagnostic));
  ASSERT_TRUE(run) << (run ? "" : run.error());
  RecordProperty("diagnostic_export", enrollment->export_directory.string());
}

TEST(GemmaAssistantV1Gpu, DISABLED_U4DiagnosticGemma31) {
  auto enrollment = ts::U4ManualEnrollment::Read("gemma31", true, true);
  ASSERT_TRUE(enrollment) << (enrollment ? "" : enrollment.error());
  ArmDiagnostics diagnostic;
  diagnostic.observe_decode = true;
  diagnostic.full_head = enrollment->transport == "full-head";
  diagnostic.export_directory = enrollment->export_directory;
  diagnostic.model_name = enrollment->model;
  diagnostic.target_id = enrollment->target_id;
  diagnostic.drafter_id = enrollment->drafter_id;
  diagnostic.transport = enrollment->transport;
  diagnostic.expected_target_policy = enrollment->expected_target_policy;
  diagnostic.expected_drafter_policy = enrollment->expected_drafter_policy;
  const auto run =
      RunArm(kProfiles[1], {enrollment->store, enrollment->target_id, enrollment->drafter_id},
             std::move(diagnostic));
  ASSERT_TRUE(run) << (run ? "" : run.error());
  RecordProperty("diagnostic_export", enrollment->export_directory.string());
}
INSTANTIATE_TEST_SUITE_P(Approved, GemmaAssistantV1Gpu, ::testing::Values(0U, 1U),
                         [](const auto& info) { return kProfiles.at(info.param).name; });
}  // namespace
