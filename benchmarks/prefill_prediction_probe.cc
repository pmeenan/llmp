// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Actual scalar production prefill, through PromptSession's hint delivery.
// Both arms provision the same optional plans/backing; off withholds hints.
// Optional headed|state-only argument instead compares output-intent delivery
// under identical headed+state-only Setup provisioning; hints stay independent.
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "engine/dsv4_runner.h"
#include "engine/qwen38_runner.h"
#include "runtime/serving.h"

namespace {
namespace rt = jitllm::runtime;
namespace en = jitllm::engine;
namespace cfg = jitllm::config;
namespace ba = jitllm::base;
using rt::Status;

struct Lifetime {
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  std::unique_ptr<rt::Server> server;
};

struct HostCharge {
  en::PagedNode& node;
  bool funded;
  explicit HostCharge(en::PagedNode& n) : node(n), funded(n.ChargeHost(64ULL << 20U, false)) {}
  ~HostCharge() {
    if (funded) node.UnchargeHost(64ULL << 20U);
  }
};

template <typename T>
Status Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::out | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.close();
  if (!file) return std::unexpected(std::format("writing {}", path.string()));
  return {};
}

std::expected<std::vector<std::int32_t>, std::string> Tokens(const char* path,
                                                             std::uint32_t count) {
  std::error_code ec;
  if (std::filesystem::file_size(path, ec) != std::uint64_t{count} * sizeof(std::int32_t) || ec)
    return std::unexpected("literal token extent differs");
  std::vector<std::int32_t> tokens(count);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
  if (!file) return std::unexpected("reading literal token extent");
  return tokens;
}

struct Snapshot {
  std::string sha;
  std::uint64_t bytes = 0;
  std::string ranges;
};

template <typename Runner>
std::expected<Snapshot, std::string> State(rt::Server& server, Runner& runner, std::uint32_t id) {
  auto slot = runner.request_slot(id);
  if (!slot) return std::unexpected(slot.error());
  auto& node = server.node();
  constexpr std::uint64_t capacity = 1ULL << 20U;
  std::vector<jitllm::catalog::ExtentId> staging;
  auto buffer = node.Pinned(capacity, 0, staging);
  if (!buffer) return std::unexpected(buffer.error());
  ba::Sha256 hash;
  Snapshot out{.sha = {}, .bytes = 0, .ranges = "["};
  for (const auto& range : (*slot)->used_state_ranges()) {
    out.bytes += range.bytes;
    out.ranges += std::format("[{},{},{}],", range.region, range.offset, range.bytes);
    for (std::uint64_t at = 0; at < range.bytes; at += capacity) {
      const en::LiveState::Range part{range.region, range.offset + at,
                                      std::min(capacity, range.bytes - at)};
      if (auto r = (*slot)->SaveUsedState(*buffer, std::span(&part, 1)); !r) {
        node.KeepPinned(*buffer);  // no completion proof: retain DMA destination
        return std::unexpected(r.error());
      }
      hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
    }
  }
  if (auto r = node.FreePinned(*buffer); !r) return std::unexpected(r.error());
  if (out.bytes == 0) return std::unexpected("complete initialized state required");
  out.ranges.back() = ']';
  out.sha = ba::ToHex(hash.Finish());
  return out;
}

std::string Json(const Snapshot& s) {
  return std::format(R"({{"sha256":"{}","bytes":{},"ranges":{}}})", s.sha, s.bytes, s.ranges);
}
std::string Json(const en::PrefillLookaheadStats& s) {
  return std::format(
      R"({{"attempted":{},"built":{},"cached":{},"refused":{},"mtp_built":{},"mtp_cached":{},"build_seconds":{:.9f}}})",
      s.attempted, s.built, s.cached, s.refused, s.mtp_built, s.mtp_cached, s.build_seconds);
}
std::string Json(const en::LiveState::PreparationStats& s) {
  return std::format(
      R"({{"attempted":{},"submitted":{},"refused":{},"failed":{},"completed_extents":{},"adopted_extents":{},"cancel_requested":{}}})",
      s.attempted, s.submitted, s.refused, s.failed, s.completed_extents, s.adopted_extents,
      s.cancel_requested);
}

std::string Json(const en::PrefillOutputStats& s) {
  return std::format(
      R"({{"headed":{},"state_only":{},"tail_cut":{},"rows":{},"selected_nodes":{}}})", s.headed,
      s.state_only, s.tail_cut, s.rows, s.selected_nodes);
}
std::string Json(const en::StartupPlacementStats& s) {
  return std::format(
      R"({{"plans":{},"nodes":{},"bounded":{},"exact":{},"max_activations":{},"max_inputs":{},"max_host":{},"max_nodes":{}}})",
      s.plans, s.nodes, s.bounded, s.exact, s.max_activations, s.max_inputs, s.max_host,
      s.max_nodes);
}

template <typename Runner>
Status Exercise(Lifetime& life, rt::Llm& model, Runner& runner,
                const std::array<std::vector<std::int32_t>, 2>& tokens,
                const std::filesystem::path& out, std::string_view recipe, bool hints,
                std::optional<bool> output_cut) {
  auto& server = *life.server;
  const bool ds = recipe == "ds" || recipe == "dspark";
  const bool spec = recipe == "dspark" || recipe == "qmtp";
  const auto& settings = model.settings();
  const std::uint32_t context = ds ? 16384U : 2048U;
  const std::uint32_t chunk = ds ? 4096U : 512U;
  if (settings.context.value != context || model.context() != context ||
      settings.prefill_chunk.value != chunk || model.configured_rows() != chunk ||
      model.max_rows() != chunk || settings.max_slots.value != 2 ||
      settings.speculation.value != spec || model.speculative() != spec ||
      runner.speculative() != spec || model.prefill_wave_capacity() != 1)
    return std::unexpected("resolved scalar prefill recipe differs");
  HostCharge charge(server.node());
  if (!charge.funded) return std::unexpected("bounded observation funding refused");
  const auto vocab = runner.vocab();
  const auto finite = [vocab](const std::vector<float>& row) {
    return row.size() == vocab &&
           std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
  };
  std::array<rt::Llm::Branch*, 2> branches{};
  for (std::uint32_t id = 0; id < 2; ++id) {
    auto branch = model.branch(id);
    if (!branch) return std::unexpected(branch.error());
    branches[id] = *branch;
    if (!std::ranges::all_of(tokens[id],
                             [vocab](auto t) { return t >= 0 && std::uint32_t(t) < vocab; }))
      return std::unexpected("literal token outside actual vocabulary");
  }
  if (auto r = server.SelectRequestBranches(model, branches); !r) return r;
  for (std::uint32_t id = 0; id < 2; ++id) {
    std::vector<float> warm;
    if (auto r = branches[id]->Prefill(std::span(tokens[id]).first(3), warm); !r) return r;
    if (!finite(warm)) return std::unexpected("warm frontier incomplete");
    if (auto r = branches[id]->Clear(); !r) return r;
  }
  runner.DropPlans();
  const auto graph_before = model.graphs();
  const auto plans_before = runner.prefill_stats();
  const auto prep_before = runner.state_preparation_stats();
  const auto output_before = runner.prefill_output_stats();
  std::array<std::vector<float>, 2> heads;
  std::array<Snapshot, 2> initial, generated, continued;
  std::array<rt::PrefillRun, 2> runs;
  struct Prompts {
    std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> owned;
    ~Prompts() {
      for (auto& s : owned)
        if (s) {
          s->Cancel();
          if (auto r = s->Finish(); !r)
            std::println(stderr, "prompt cleanup failed: {}", r.error());
        }
    }
  } prompts;
  for (std::uint32_t id = 0; id < 2; ++id) {
    auto opened = branches[id]->BeginPrompt(tokens[id]);
    if (!opened) return std::unexpected(opened.error());
    prompts.owned[id] = std::move(*opened);
    if (auto r = prompts.owned[id]->Advance(); !r) return r;  // pure scalar reuse
  }
  // First traversal: warm weights, fresh prompt plans, no observations in paid work.
  const auto start = rt::Clock::now();
  std::uint32_t units = 0;
  while (!prompts.owned[0]->done() || !prompts.owned[1]->done()) {
    for (const auto id : {1U, 0U})
      if (!prompts.owned[id]->done()) {
        if (auto r = prompts.owned[id]->Advance(); !r) return r;
        ++units;
      }
  }
  const double seconds = std::chrono::duration<double>(rt::Clock::now() - start).count();
  const auto graph_after = model.graphs();
  const auto plans = runner.prefill_stats();
  const auto prep = runner.state_preparation_stats();
  const auto output_after = runner.prefill_output_stats();
  if (output_cut) {
    const std::uint64_t chunks = ds ? 4 : 6;
    const std::uint64_t no_head = *output_cut ? chunks - 2 : 0;
    const std::uint64_t cut = spec ? 0 : no_head;
    if (output_after.headed - output_before.headed != chunks - no_head ||
        output_after.state_only - output_before.state_only != no_head ||
        output_after.tail_cut - output_before.tail_cut != cut ||
        output_after.rows - output_before.rows != tokens[0].size() + tokens[1].size() ||
        output_after.selected_nodes <= output_before.selected_nodes)
      return std::unexpected("actual scalar output intent/dependency cut was not exercised");
  }
  if (server.node().has_pending_state_preparation())
    return std::unexpected("undrained prefill ticket");
  if (!std::isfinite(seconds) || seconds <= 0 || plans.refused != plans_before.refused ||
      prep.failed != prep_before.failed || prep.refused != prep_before.refused)
    return std::unexpected("prefill timing/optional-work guard failed");
  if (hints && (plans.built <= plans_before.built || plans.cached <= plans_before.cached))
    return std::unexpected("actual hinted CPU plan build/install not exercised");
  if (!hints &&
      (plans.attempted != plans_before.attempted || prep.attempted != prep_before.attempted))
    return std::unexpected("withheld hints still submitted work");
  if (hints && (prep.submitted <= prep_before.submitted ||
                prep.adopted_extents <= prep_before.adopted_extents))
    return std::unexpected("short boundary did not prepare fresh backing");
  if constexpr (std::is_same_v<Runner, en::Qwen38Runner>) {
    if (hints && runner.speculative() && plans.mtp_cached <= plans_before.mtp_cached)
      return std::unexpected("injected MTP companion prediction not exercised");
  }
  for (std::uint32_t id = 0; id < 2; ++id) {
    heads[id] = prompts.owned[id]->last();
    runs[id] = prompts.owned[id]->run();
    if (!finite(heads[id]) || branches[id]->history() != tokens[id] ||
        runs[id].end != tokens[id].size())
      return std::unexpected("completed prefill publication differs");
    if (auto r = prompts.owned[id]->Finish(); !r) return r;
    prompts.owned[id].reset();
    auto state = State(server, runner, id);
    if (!state) return std::unexpected(state.error());
    initial[id] = std::move(*state);
    if (auto r = Write(out / std::format("prefill{}.f32", id), std::span<const float>(heads[id]));
        !r)
      return r;
  }
  // Off-paid actual joined draft/verify (where configured) and C2 -> C1 departure.
  std::array<rt::GenerateOptions, 2> options;
  std::array<rt::Generation, 2> generation;
  struct Generations {
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> owned;
    ~Generations() {
      for (auto& s : owned)
        if (s) {
          s->Cancel();
          if (auto r = s->Finish(); !r)
            std::println(stderr, "generation cleanup failed: {}", r.error());
        }
    }
  } sessions;
  for (std::uint32_t id = 0; id < 2; ++id) {
    options[id].max_tokens = id == 0 ? 9 : 3;
    options[id].stop = false;
    options[id].keep_logits = true;
    auto opened = branches[id]->BeginGeneration(heads[id], options[id], generation[id]);
    if (!opened) return std::unexpected(opened.error());
    sessions.owned[id] = std::move(*opened);
  }
  std::array<bool, 2> finished{};
  bool departed = false;
  std::uint64_t joined_calls = 0;
  std::uint64_t paired_slots = 0;
  while (!sessions.owned[0]->done() || !sessions.owned[1]->done()) {
    std::array<rt::Llm::GenerationSession*, 2> active{};
    std::array<rt::Llm::Branch*, 2> selected{};
    std::size_t count = 0;
    for (std::uint32_t id = 0; id < 2; ++id)
      if (!sessions.owned[id]->done()) {
        selected[count] = branches[id];
        active[count++] = sessions.owned[id].get();
      }
    const bool solo = count == 1 && sessions.owned[1]->done();
    std::optional<Snapshot> peer;
    if (solo) {
      auto state = State(server, runner, 1);
      if (!state) return std::unexpected(state.error());
      peer = std::move(*state);
    }
    if (auto r = server.SelectRequestBranches(model, std::span(selected).first(count)); !r)
      return r;
    if (auto r = model.RunGenerationWave(std::span(active).first(count)); !r) return r;
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (sessions.owned[id]->done() && !finished[id]) {
        if (auto r = sessions.owned[id]->Finish(); !r) return r;
        finished[id] = true;
      }
    }
    if (count == 2) {
      ++joined_calls;
      if constexpr (std::is_same_v<Runner, en::Qwen38Runner>)
        paired_slots |= runner.last_wave().paired_slots;
    }
    if (solo) {
      // Observation borrows peer1 again only after owner0 work retired. The
      // next work call reinstates its actual one-owner cohort.
      if (auto r = server.SelectRequestBranches(model, branches); !r) return r;
      auto state = State(server, runner, 1);
      if (!state || state->sha != peer->sha) return std::unexpected("departed owner state changed");
      departed = true;
    }
  }
  if (!departed || joined_calls == 0)
    return std::unexpected("actual joined/owner departure missing");
  if constexpr (std::is_same_v<Runner, en::Qwen38Runner>) {
    if (paired_slots != 3) return std::unexpected("actual joined Qwen products missing");
  } else {
    const auto& wave = runner.wave_stats();
    if (wave.eager + wave.captured + wave.replayed == 0)
      return std::unexpected("actual joined DeepSeek work missing");
  }
  for (std::uint32_t id = 0; id < 2; ++id) {
    if (!finished[id]) return std::unexpected("finished generation settlement missing");
    sessions.owned[id].reset();
    if (generation[id].tokens.size() != options[id].max_tokens ||
        generation[id].logits.size() != options[id].max_tokens)
      return std::unexpected("complete generation rows missing");
    for (std::size_t row = 0; row < generation[id].logits.size(); ++row) {
      if (!finite(generation[id].logits[row])) return std::unexpected("nonfinite generation row");
      if (auto r = Write(out / std::format("generation{}-{}.f32", id, row),
                         std::span<const float>(generation[id].logits[row]));
          !r)
        return r;
    }
    if (auto r = Write(out / std::format("choices{}.i32", id),
                       std::span<const std::int32_t>(generation[id].tokens));
        !r)
      return r;
    auto state = State(server, runner, id);
    if (!state) return std::unexpected(state.error());
    generated[id] = std::move(*state);
  }
  // Retire the cohort, spill owner0, restore both and consume pending anchors.
  const auto retired = server.RetireRequestBranches(model, true);
  if (!retired.result) return retired.result;
  if (!retired.references_retired) return std::unexpected("request retirement unproven");
  if (auto r = model.SpillIdle(*branches[0]); !r) return r;
  if (!branches[0]->spilled()) return std::unexpected("actual spill missing");
  if (auto r = server.SelectRequestBranches(model, branches); !r) return r;
  for (std::uint32_t id = 0; id < 2; ++id) {
    auto history = branches[id]->history();
    const auto stable = static_cast<std::uint32_t>(history.size());
    history.push_back(generation[id].tokens.back());
    std::vector<float> last;
    std::uint32_t reused = 0;
    if (auto r = branches[id]->PreparePrompt(history, stable, last, reused); !r) return r;
    if (reused != stable || !finite(last) || branches[id]->history() != history ||
        branches[id]->turn_checkpoints() == 0)
      return std::unexpected("restored continuation/checkpoint publication differs");
    if (auto r = Write(out / std::format("continued{}.f32", id), std::span<const float>(last)); !r)
      return r;
    if (auto r =
            Write(out / std::format("history{}.i32", id), std::span<const std::int32_t>(history));
        !r)
      return r;
    auto state = State(server, runner, id);
    if (!state) return std::unexpected(state.error());
    continued[id] = std::move(*state);
  }
  std::string output_proof = "null";
  if (output_cut) {
    // A future stable boundary cuts one nonfinal chunk, then actually settles
    // MTP/DSpark before checkpointing and runs the final headed continuation.
    auto history = branches[0]->history();
    const auto old_end = static_cast<std::uint32_t>(history.size());
    history.insert(history.end(), tokens[0].begin(), tokens[0].begin() + 32);
    const auto stable = old_end + 16;
    const auto checkpoints = branches[0]->turn_checkpoints();
    auto opened = branches[0]->BeginPrompt(history, stable);
    if (!opened) return std::unexpected(opened.error());
    prompts.owned[0] = std::move(*opened);
    bool nonfinal_seen = false;
    while (!prompts.owned[0]->done()) {
      if (auto r = prompts.owned[0]->Advance(); !r) return r;
      if (prompts.owned[0]->run().end == stable) {
        nonfinal_seen = true;
        if (*output_cut ? !prompts.owned[0]->last().empty() : !finite(prompts.owned[0]->last()))
          return std::unexpected("nonfinal checkpoint boundary output intent differs");
      }
    }
    if (!nonfinal_seen || !finite(prompts.owned[0]->last()) || branches[0]->history() != history ||
        branches[0]->turn_checkpoints() <= checkpoints)
      return std::unexpected("actual nonfinal checkpoint/continuation publication differs");
    if (auto r = Write(out / "checkpointed0.f32", std::span<const float>(prompts.owned[0]->last()));
        !r)
      return r;
    if (auto r = Write(out / "checkpointed-history0.i32", std::span<const std::int32_t>(history));
        !r)
      return r;
    if (auto r = prompts.owned[0]->Finish(); !r) return r;
    prompts.owned[0].reset();
    auto checkpointed = State(server, runner, 0);
    if (!checkpointed) return std::unexpected(checkpointed.error());
    const auto score_before = runner.prefill_output_stats();
    std::uint32_t score_rows = 0;
    auto scoring = branches[1]->BeginScoringPrompt(
        std::span(tokens[1]).first(5), [&](std::int32_t token, std::span<const float> row) {
          if (score_rows >= 4 || row.size() != vocab ||
              !std::ranges::all_of(row, [](float x) { return std::isfinite(x); }) ||
              token != tokens[1][score_rows + 1])
            return false;
          const auto wrote = Write(out / std::format("scoring1-{}.f32", score_rows), row);
          if (!wrote) return false;
          ++score_rows;
          return true;
        });
    if (!scoring) return std::unexpected(scoring.error());
    prompts.owned[1] = std::move(*scoring);
    while (!prompts.owned[1]->done())
      if (auto r = prompts.owned[1]->Advance(); !r) return r;
    const auto score_after = runner.prefill_output_stats();
    if (score_rows != 4 || !finite(prompts.owned[1]->last()) ||
        branches[1]->history() != std::vector(tokens[1].begin(), tokens[1].begin() + 5) ||
        score_after.state_only != score_before.state_only ||
        score_after.tail_cut != score_before.tail_cut ||
        score_after.headed - score_before.headed != 5)
      return std::unexpected("actual scoring did not retain every head");
    if (auto r = Write(out / "scoring1.f32", std::span<const float>(prompts.owned[1]->last())); !r)
      return r;
    if (auto r = prompts.owned[1]->Finish(); !r) return r;
    prompts.owned[1].reset();
    auto scored_state = State(server, runner, 1);
    if (!scored_state) return std::unexpected(scored_state.error());
    auto protected_peer = State(server, runner, 0);
    if (!protected_peer) return std::unexpected(protected_peer.error());
    if (protected_peer->sha != checkpointed->sha || protected_peer->bytes != checkpointed->bytes ||
        protected_peer->ranges != checkpointed->ranges)
      return std::unexpected("scoring changed its checkpointed peer");
    output_proof = std::format(
        R"({{"stable_boundary":{},"nonfinal_seen":true,"checkpointed":{},"scoring_rows":{},"scored":{}}})",
        stable, Json(*checkpointed), score_rows, Json(*scored_state));
  }
  std::string owners = "[";
  for (std::uint32_t id = 0; id < 2; ++id) {
    owners += std::format(
        R"({{"prompt_rows":{},"prefill_end":{},"prefill_chunks":{},"initial":{},"generated":{},"continued":{},"steps":{},"drafted":{},"accepted":{}}},)",
        tokens[id].size(), runs[id].end, runs[id].chunks, Json(initial[id]), Json(generated[id]),
        Json(continued[id]), generation[id].steps, generation[id].drafted, generation[id].accepted);
  }
  owners.back() = ']';
  const auto floor = runner.plan_floor_bytes();
  const auto temporary = runner.prefill_temporary_plan_bytes();
  std::string plan_report;
  ba::json::AppendQuoted(runner.plan_report(), plan_report);
  auto allocation = model.allocation_report();
  if (allocation.empty()) allocation = "null";
  if (!model.violations().empty()) return std::unexpected(model.violations());
  std::string startup;
  if constexpr (std::is_same_v<Runner, en::Dsv4Runner>) {
    startup = Json(runner.startup_placement_stats());
  } else {
    startup =
        std::format(R"({{"scalar":{},"wave":{}}})", Json(runner.scalar_startup_placement_stats()),
                    Json(runner.wave_startup_placement_stats()));
  }
  const auto result = std::format(
      R"({{"recipe":"{}","hints":{},"state_only_provisioned":{},"output_intent":{},"setup_seconds":{:.9f},"startup":{},"output_before":{},"output_after":{},"output_proof":{},"context":{},"chunk":{},"slots":2,"speculative":{},"prefill_capacity":1,"prefill_seconds":{:.9f},"paid_units":{},"vocab":{},"plan_floor":{},"temporary_plan_floor":{},"derived_ordinary_plan_floor":{},"activations":{},"scratch":{},"host_inputs":{},"graphs":[{},{},{}],"plans_before":{},"plans":{},"preparation_before":{},"preparation":{},"owners":{},"allocation":{},"plan_report":{},"joined_generation_calls":{},"paired_slots":{},"model_extra":{},"departed":true,"spilled":true,"checkpointed":true}})",
      recipe, hints, output_cut.has_value(), output_cut.value_or(false), runner.setup_seconds(),
      startup, Json(output_before), Json(output_after), output_proof, context, chunk, spec, seconds,
      units, vocab, floor, temporary, floor - temporary, runner.activations_needed(),
      runner.pool_needed(), runner.host_input_bytes(), graph_after.eager - graph_before.eager,
      graph_after.captured - graph_before.captured, graph_after.replayed - graph_before.replayed,
      Json(plans_before), Json(plans), Json(prep_before), Json(prep), owners, allocation,
      plan_report, joined_calls, paired_slots, model.extra());
  std::ofstream file(out / "result.json", std::ios::out | std::ios::noreplace);
  file << result << '\n';
  file.close();
  if (!file) return std::unexpected("writing result receipt");
  return {};
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 10 && argc != 11) {
    std::println(stderr,
                 "usage: prefill_prediction_probe ds|dspark|qn|qmtp|qg STORE TARGET DRAFTER|- "
                 "TOKENIZER_DIR|- IDS0 IDS1 NEW_OUT off|on [headed|state-only]");
    return 2;
  }
  const std::string_view recipe = argv[1], policy = argv[9];
  const bool ds = recipe == "ds" || recipe == "dspark";
  const bool spec = recipe == "dspark" || recipe == "qmtp";
  if ((!ds && recipe != "qn" && recipe != "qmtp" && recipe != "qg") ||
      (policy != "off" && policy != "on") || spec == (std::string_view(argv[4]) == "-"))
    return 2;
  std::optional<bool> output_cut;
  if (argc == 11) {
    const std::string_view intent = argv[10];
    if (intent != "headed" && intent != "state-only") return 2;
    output_cut = intent == "state-only";
  }
  const std::filesystem::path out = argv[8];
  std::error_code ec;
  if (!std::filesystem::create_directory(out, ec) || ec || chmod(out.c_str(), 0700) != 0) return 2;
  const std::array<std::uint32_t, 2> counts =
      ds ? std::array<std::uint32_t, 2>{4352, 4608} : std::array<std::uint32_t, 2>{1536, 1280};
  std::array<std::vector<std::int32_t>, 2> tokens;
  for (std::uint32_t id = 0; id < 2; ++id) {
    auto read = Tokens(argv[6 + id], counts[id]);
    if (!read) {
      std::println(stderr, "{}", read.error());
      return 2;
    }
    tokens[id] = std::move(*read);
  }
  auto life = std::make_unique<Lifetime>();
  life->roles.installed = argv[2];
  life->roles.spill = out / "spill";
  life->roles.state = out / "state";
  for (const auto& path : {life->roles.spill, life->roles.state})
    if (!std::filesystem::create_directory(path, ec) || ec || chmod(path.c_str(), 0700) != 0)
      return 2;
  cfg::ModelEntry entry;
  entry.name = "target";
  entry.artifact = argv[3];
  if (spec) entry.drafter = argv[4];
  // Off-paid companion proof must exercise DSpark joined verification even
  // when its public adaptive policy would choose a plain wave.
  if (recipe == "dspark") entry.overrides["wave_form"] = std::string("speculative");
  if (!ds) {
    entry.tokenizer = std::filesystem::path(argv[5]) / "tokenizer.json";
    entry.chat_template = std::filesystem::path(argv[5]) / "chat_template.jinja";
  }
  entry.overrides["context"] = std::int64_t{ds ? 16384 : 2048};
  entry.overrides["prefill_chunk"] = std::int64_t{ds ? 4096 : 512};
  entry.overrides["max_slots"] = std::int64_t{2};
  life->config.models.push_back(entry);
  life->options.plain = !spec;
  life->options.dsv4_prefill_lookahead = true;
  life->options.dsv4_prepare_state = true;
  life->options.qwen38_prefill_lookahead = true;
  life->options.qwen38_prepare_state = true;
  life->options.predicted_prefill_hints = policy == "on";
  life->options.dsv4_state_only_prefill = output_cut.has_value();
  life->options.qwen38_state_only_prefill = output_cut.has_value();
  life->options.prefill_output_intent = output_cut.value_or(true);
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, stderr);
  Status ran = life->server->Start(true);
  rt::Llm* model = nullptr;
  if (ran) {
    model = dynamic_cast<rt::Llm*>(life->server->Find("target"));
    if (!model) ran = std::unexpected("configured LLM adapter missing");
  }
  if (ran) {
    rt::SwapParts parts;
    ran = life->server->Activate(*model, parts);
  }
  if (ran) {
    ran = ds ? Exercise(*life, *model, dynamic_cast<en::Dsv4Runner&>(model->paged()), tokens, out,
                        recipe, policy == "on", output_cut)
             : Exercise(*life, *model, dynamic_cast<en::Qwen38Runner&>(model->paged()), tokens, out,
                        recipe, policy == "on", output_cut);
  }
  if (!ran) std::println(stderr, "probe failed: {}", ran.error());
  const auto retired = life->server->TearDown();
  if (!retired) {
    std::println(stderr, "retirement unproven: {}", retired.error());
    std::ignore = life.release();
    return 1;
  }
  if (!ran) return 1;
  std::println("PREFILL_PREDICTION_RETIRED recipe={} hints={}", recipe, policy == "on");
  return 0;
}
