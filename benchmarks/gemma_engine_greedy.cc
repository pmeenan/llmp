// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 4 31B assistant's repeated greedy screen on the shared engine
// (M3.6's gate for the assistant, docs/experiments/m36-baselines): the
// work llmp_gemma_greedy_reference pays on the M3 runner, paid by serving's
// component build (runtime/component_build.h) and the decoding modes
// (engine/decoding.h) instead. The 31B target and its assistant, C1 at
// context 4,096 and 128-row chunks, the canonical P=64 inputs, the M3
// screen's arithmetic (its explicit norm-rope and norm-add chains, no
// quantized GLU, no owner attention, no narrowed head: a policy given in
// place of the mapping's), and 32 committed greedy tokens a round: units of
// min(3, remaining - 1) drafts and one verify, and a plain decode when one
// token remains. One warm round, two quality rounds (their tokens, every
// verify's heads, each unit's last draft head and the final head
// archived) and two timed rounds, each from a fresh 64-token prefix; the
// prefix and the state's growth through 96 positions are outside the
// clock. Its output lines are the M3 screen's, so one harness reads both.
// Manual, never CTest.
//
// usage: llmp_gemma_engine_greedy TARGET ASSISTANT OUT 31 INPUTS unit32|plain32
//   (TARGET and ASSISTANT installed artifacts of one store)

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "config/node_config.h"
#include "engine/decoding.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/support.h"
#include "engine/wave_composer.h"
#include "runtime/component_build.h"
#include "runtime/model_settings.h"

namespace {

namespace en = llmp::engine;
namespace pl = llmp::planner;
namespace rt = llmp::runtime;
using en::support::Error;

constexpr std::uint32_t kVocab = 262144, kPast = 64, kEmitted = 32;
constexpr std::uint64_t kHost = 256ULL << 20U;
// The M3 screen's arithmetic on the target (the retired
// gemma_greedy_reference.cc's Gemma4Options: the plain, norm-rope and
// norm-add chains, nothing else; its retained features keep every row to
// the head).
constexpr std::string_view kScreenPolicy =
    "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\ndisable narrow_final\n"
    "disable owner_attention\ndisable prepared_inputs\ndisable rope_store\ndisable route_reduce\n";

template <class T>
en::Status Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  if (!file) return Error("exclusive output creation failed");
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("complete output write failed");
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}
std::span<const float> Row(std::span<const std::byte> bytes, std::size_t row) {
  return {reinterpret_cast<const float*>(bytes.data()) + (row * kVocab), kVocab};
}

struct Unit {
  std::uint32_t past = 0, depth = 0, rows = 0, keep = 0;
};

class Screen {
 public:
  Screen(en::ModelRunner& runner, const en::WaveComposer& composer,
         std::span<const std::int32_t> ids, std::filesystem::path out)
      : runner_(runner), composer_(composer), ids_(ids), out_(std::move(out)) {}

  en::Status Run(bool speculate, std::int32_t eos) {
    auto plain = en::DecodingOrchestrator::Make(runner_, composer_,
                                                {.kind = en::DecodingMode::Kind::kPlain}, {});
    if (!plain) return Error(plain.error());
    std::optional<en::DecodingOrchestrator> assisted;
    if (speculate) {
      auto made = en::DecodingOrchestrator::Make(
          runner_, composer_,
          {.kind = en::DecodingMode::Kind::kSpeculative,
           .drafters = {{.stage = 1,
                         .shape = en::DrafterSpec::Shape::kAutoregressive,
                         .max_depth = 3,
                         .borrows_target = true,
                         .anchored = true,
                         .features = true}},
           .combine = en::DecodingMode::Drafters::kCascade,
           .row_invariant = false},
          Policy());
      if (!made) return Error(made.error());
      assisted.emplace(std::move(*made));
    }
    head_.reserve(kVocab);
    draft_head_.reserve(kVocab);
    verify_heads_.reserve(std::size_t{kEmitted} * 4 * kVocab);
    last_draft_heads_.reserve(std::size_t{kEmitted} * kVocab);
    for (unsigned round = 0; round < 5; ++round) {
      const bool quality = round == 1 || round == 2, timed = round >= 3;
      const char* name = round == 0   ? "warm"
                         : round == 1 ? "quality-first"
                         : round == 2 ? "quality-repeat"
                         : round == 3 ? "timed-first"
                                      : "timed-repeat";
      const auto dir = out_ / name;
      std::error_code ec;
      if (!std::filesystem::create_directory(dir, ec) || ec) {
        return Error("bounded32 chronology directory must be new");
      }
      archive_ = quality;
      verify_heads_.clear();
      last_draft_heads_.clear();
      // The prefix and the state's growth, outside the clock.
      if (auto r = Prefix(); !r) return r;
      if (auto r = runner_.slots().ReserveStateThrough(0, kPast + kEmitted); !r) return r;
      std::vector<std::int32_t> history(ids_.begin(), ids_.begin() + kPast);
      history.push_back(Argmax(head_));
      std::array<std::int32_t, kEmitted> tokens{};
      std::vector<Unit> units;
      std::uint32_t count = 0, verified = 0, drafted = 0, eos_count = 0;
      double seconds = 0;
      const auto started = std::chrono::steady_clock::now();
      while (count < kEmitted) {
        const std::uint32_t remaining = kEmitted - count;
        const std::uint32_t past = kPast + count;
        const std::uint32_t depth = speculate && remaining >= 2 ? std::min(3U, remaining - 1) : 0;
        std::uint32_t keep = 1;
        const std::array<en::DecodeWork, 1> work = {
            en::DecodeWork{.slot = 0, .history = history, .left = depth + 1}};
        if (depth != 0) {
          const en::DecodeReport report = assisted->Step(work);
          const en::DecodeOutcome& out = report.outcomes.front();
          if (out.refusal) return Error(out.refusal->what);
          keep = out.kept;
          if (keep == 0 || keep > depth + 1 || out.proposed.size() != depth) {
            return Error("bounded32 completed unit envelope changed");
          }
          // The anchor and the accepted drafts commit; the target's choice
          // after them is the next anchor.
          history.insert(history.end(), out.tokens.begin(), out.tokens.end());
          for (std::uint32_t row = 0; row < keep; ++row) {
            tokens.at(count + row) = history.at(past + row);
          }
        } else {
          const en::DecodeReport report = plain->Step(work);
          const en::DecodeOutcome& out = report.outcomes.front();
          if (out.refusal) return Error(out.refusal->what);
          const auto row = Row(out.outputs, 0);
          head_.assign(row.begin(), row.end());
          if (archive_) verify_heads_.insert(verify_heads_.end(), row.begin(), row.end());
          tokens.at(count) = history[past];
          history.push_back(Argmax(row));
        }
        for (std::uint32_t row = 0; row < keep; ++row) eos_count += tokens.at(count + row) == eos;
        units.push_back({.past = past, .depth = depth, .rows = depth + 1, .keep = keep});
        count += keep;
        verified += depth + 1;
        drafted += depth;
      }
      const auto finished = std::chrono::steady_clock::now();
      if (timed) seconds = en::support::Seconds(finished - started);
      if (runner_.slots().positions(0) != kPast + kEmitted || !runner_.slots().usable(0)) {
        return Error("bounded32 completed endpoint was not published");
      }
      if (round == 0) continue;
      if (auto r = Write<std::int32_t>(dir / "tokens.i32", tokens); !r) return r;
      if (auto r = Write<float>(dir / "final-head.f32", head_); !r) return r;
      if (quality) {
        if (auto r = Write<float>(dir / "verify-heads.f32", verify_heads_); !r) return r;
        if (speculate) {
          if (auto r = Write<float>(dir / "last-draft-heads.f32", last_draft_heads_); !r) return r;
        }
      }
      std::string list = "[";
      for (std::size_t i = 0; i < units.size(); ++i) {
        const Unit& u = units[i];
        if (i != 0) list += ',';
        list += std::format(R"({{"past":{},"depth":{},"verified_rows":{},"keep":{}}})", u.past,
                            u.depth, u.rows, u.keep);
      }
      list += ']';
      const std::string metadata = std::format(
          R"({{"mode":"{}","phase":"{}","emitted":32,"endpoint":96,"prefill_queries":64,)"
          R"("max_rows":128,"context":4096,"eog_ignored":true,"eos_id":{},"eos_occurrences":{},)"
          R"("verified_rows":{},"drafted_rows":{},"seconds":{},"units":{}}})"
          "\n",
          speculate ? "unit32" : "plain32", quality ? "quality" : "timed", eos, eos_count, verified,
          drafted, seconds, list);
      if (auto r = Write<char>(dir / "completion.json", metadata); !r) return r;
      std::cout << "GREEDY32_COMPLETED mode=" << (speculate ? "unit32" : "plain32")
                << " phase=" << name << " emitted=32 endpoint=96 verified=" << verified
                << " drafted=" << drafted << " seconds=" << seconds << '\n';
    }
    return {};
  }

 private:
  // A fresh slot's 64-token prefix, its head the first anchor's.
  en::Status Prefix() {
    if (auto r = runner_.slots().Clear(0); !r) return r;
    const pl::StepShape shape{.stage = 0,
                              .phase = pl::Phase::kPrefill,
                              .segments = {{.slot = 0,
                                            .rows = kPast,
                                            .past = 0,
                                            .read_cells = 256,
                                            .outputs = 1,
                                            .group = 0,
                                            .want_head = true,
                                            .greedy = false}},
                              .row_invariant = false,
                              .read_align = 256};
    const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = ids_.first(kPast)}};
    auto ran = runner_.RunStep(
        {.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}});
    if (!ran) return Error(ran.error().what);
    const auto row = Row(ran->outputs.front(), 0);
    head_.assign(row.begin(), row.end());
    return {};
  }

  // Greedy proposals and the greedy verdict (the M3 screen's
  // JudgeGemma4Greedy): each draft kept while it is the target's choice,
  // then the target's choice after the last kept row, whose head is the
  // unit's carry.
  en::DecodingPolicy Policy() {
    en::DecodingPolicy policy;
    policy.propose =
        [this](std::uint32_t, std::size_t, std::span<const std::byte> outputs,
               std::uint32_t want) -> std::expected<std::vector<std::int32_t>, std::string> {
      if (outputs.size() < kVocab * sizeof(float) || want == 0) {
        return Error("a draft without its head");
      }
      const auto row = Row(outputs, 0);
      if (archive_) draft_head_.assign(row.begin(), row.end());
      return std::vector<std::int32_t>{Argmax(row)};
    };
    policy.judge =
        [this](std::uint32_t, std::span<const std::int32_t> verified,
               std::span<const std::byte> outputs) -> std::expected<en::Verdict, std::string> {
      if (outputs.size() != verified.size() * kVocab * sizeof(float)) {
        return Error("a verify without its heads");
      }
      std::uint32_t keep = 1;
      while (keep < verified.size() && Argmax(Row(outputs, keep - 1)) == verified[keep]) ++keep;
      const auto head = Row(outputs, keep - 1);
      head_.assign(head.begin(), head.end());
      en::Verdict verdict{.keep = keep, .tokens = {}};
      verdict.tokens.assign(verified.begin() + 1, verified.begin() + keep);
      verdict.tokens.push_back(Argmax(head));
      if (archive_) {
        for (std::size_t r = 0; r < verified.size(); ++r) {
          const auto row = Row(outputs, r);
          verify_heads_.insert(verify_heads_.end(), row.begin(), row.end());
        }
        last_draft_heads_.insert(last_draft_heads_.end(), draft_head_.begin(), draft_head_.end());
      }
      return verdict;
    };
    return policy;
  }

  en::ModelRunner& runner_;
  const en::WaveComposer& composer_;
  std::span<const std::int32_t> ids_;
  std::filesystem::path out_;
  bool archive_ = false;
  std::vector<float> head_, draft_head_, verify_heads_, last_draft_heads_;
};

}  // namespace

int main(int argc, char** argv) {
  umask(0077);
  if (argc != 7 || std::string_view(argv[4]) != "31" ||
      (std::string_view(argv[6]) != "plain32" && std::string_view(argv[6]) != "unit32")) {
    std::cerr << "usage: llmp_gemma_engine_greedy TARGET ASSISTANT OUT 31 INPUTS unit32|plain32\n";
    return 2;
  }
  const bool speculate = std::string_view(argv[6]) == "unit32";
  std::array<std::int32_t, 1024> ids{};
  std::ifstream input(argv[5], std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(ids.data()), sizeof(ids)) ||
      input.peek() != std::char_traits<char>::eof() || ids[0] != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < static_cast<int>(kVocab); }) ||
      llmp::base::ToHex(llmp::base::Sha256{}.Update(std::as_bytes(std::span(ids))).Finish()) !=
          "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610") {
    std::cerr << "the inputs are not the canonical P=64 query inputs\n";
    return 2;
  }
  const std::filesystem::path target = argv[1], assistant = argv[2], out = argv[3];
  if (target.parent_path() != assistant.parent_path()) {
    std::cerr << "the target and its assistant are artifacts of one store\n";
    return 2;
  }
  std::error_code ec;
  if (!std::filesystem::create_directory(out, ec) || ec) return 2;
  if (!Write<std::int32_t>(out / "inputs.i32", ids)) return 2;

  llmp::config::ModelEntry entry;
  entry.name = "gemma31";
  entry.artifact = target.filename().string();
  if (speculate) entry.drafter = assistant.filename().string();
  rt::ModelSettings settings;
  settings.name = entry.name;
  settings.architecture = "gemma4";
  settings.container = "gguf";
  settings.context.value = 4096;
  settings.prefill_chunk.value = 128;
  settings.max_slots.value = 1;
  settings.speculation.value = speculate;
  settings.wave_read_align.value = 256;

  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  // What the runner's sources and launch borrow, kept until it is gone.
  std::shared_ptr<void> keep;
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    const rt::EngineBuild build = rt::ComponentLlmBuild(
        node, target.parent_path(), entry, 0, {.target_policy = std::string(kScreenPolicy)});
    auto parts = build(settings);
    if (!parts) return Error(parts.error());
    keep = std::move(parts->keep);
    parts->config.slots.spill_directory = out;
    auto created =
        en::ModelRunner::Create(node, std::move(parts->config), std::move(parts->source), 0, 0);
    if (!created) return Error(created.error());
    runner = std::move(*created);
    entered.push_back(runner.get());
    if (auto r = runner->Setup(); !r) return r;
    if (auto r =
            node.MapWorkspace(runner->activations_needed(),
                              std::max<std::uint64_t>(runner->pool_needed(), en::kPagedExtent));
        !r) {
      return r;
    }
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + kHost);
    if (auto r =
            node.Start(llmp::base::Bytes(fixed + (runner->weights().size() * en::kPagedExtent) +
                                         (2 * node.StateCapacity()) + kHost));
        !r) {
      return r;
    }
    if (auto r = runner->Register(); !r) return r;
    node.Run();
    if (auto r = runner->Bind(); !r) return r;
    if (!node.ChargeHost(kHost, false)) return Error("caller publication funding refused");
    struct Grant {
      en::PagedNode& node;
      ~Grant() { node.UnchargeHost(kHost); }
    } grant{node};
    if (auto r = runner->slots().SelectSlots(std::array<std::uint32_t, 1>{0}); !r) return r;
    auto composer = en::WaveComposer::For(parts->caps, runner->slots());
    if (!composer) return Error(composer.error());
    Screen screen(*runner, *composer, ids, out);
    // The target's EOS (Gemma 4's <turn|>), counted, never stopping a round.
    constexpr std::int32_t kEos = 106;
    return node.WithRequest(0, runner->slots().execution(), "Gemma C1/P64 engine greedy screen",
                            [&] { return screen.Run(speculate, kEos); });
  };
  const auto done = execute();
  const auto retired = node.TearDown(entered);
  if (!done || !retired) {
    std::cerr << "REFUSAL " << (!done ? done.error() : retired.error()) << '\n';
    if (!retired) std::ignore = runner.release();
    return 1;
  }
  runner.reset();
  std::cout << "GREEDY_ENGINE_RETIRED profile=31 mode=" << argv[6]
            << " C=1 P=64 prefill_query=64 emitted=32 endpoint=96 warm=1 quality=2 timed=2\n";
  return 0;
}
