// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded internal probe, not a serving interface. See docs/gemma2.md.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/gemma2_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace en = jitllm::engine;
namespace fs = std::filesystem;
using en::support::Error;
constexpr std::uint32_t kVocab = 256000, kPrompt = 256, kWarm = 3, kSteps = 32;
constexpr std::uint32_t kInput = kPrompt + kWarm + kSteps;
std::expected<std::string, std::string> Read(const fs::path& path, std::uint64_t cap) {
  std::error_code error;
  const auto bytes = fs::file_size(path, error);
  if (error || bytes > cap) return Error("bounded input size refused");
  std::string data(static_cast<std::size_t>(bytes), '\0');
  std::ifstream file(path, std::ios::binary);
  if (!file.read(data.data(), static_cast<std::streamsize>(data.size())) ||
      file.peek() != std::char_traits<char>::eof())
    return Error("input read changed or failed");
  return data;
}
template <class T>
en::Status Write(const fs::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("exclusive complete output publication failed");
}
std::expected<std::int32_t, std::string> Greedy(std::span<const float> row) {
  if (row.size() != kVocab ||
      !std::ranges::all_of(row, [](float value) { return std::isfinite(value); }))
    return Error("nonfinite or incomplete head");
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}
en::Status Prepare(const char* metadata_path, const char* text_path, const char* output_path) {
  auto metadata = Read(metadata_path, 32ULL << 20U);
  auto text = Read(text_path, 65536);
  if (!metadata || !text) return Error("preparation input refused");
  auto parsed = jitllm::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*metadata)));
  if (!parsed) return Error(parsed.error().ToString());
  if (parsed->spec.tokens.size() != kVocab || parsed->spec.bos != 2 || !parsed->spec.add_bos ||
      parsed->spec.add_eos)
    return Error("Gemma2 vocabulary/BOS contract differs");
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(parsed->spec));
  if (!tokenizer) return Error(tokenizer.error().ToString());
  std::vector<jitllm::tokenizer::TokenId> ids;
  if (auto r = tokenizer->Encode(*text, {.add_bos_eos = true, .max_tokens = 8192}, ids); !r)
    return Error(r.error().ToString());
  if (ids.size() < kInput || ids.front() != 2 || std::ranges::any_of(ids, [](auto id) {
        return id < 0 || std::cmp_greater_equal(id, kVocab);
      }))
    return Error("preparation needs 291 valid actual tokens including BOS");
  if (!fs::create_directory(output_path)) return Error("preparation output must be new");
  const auto kept = std::span(ids).first(kInput);
  if (auto r = Write<std::int32_t>(fs::path(output_path) / "ids.i32", kept); !r) return r;
  std::cout << "GEMMA2_INPUT rows=" << kInput << " sha256="
            << jitllm::base::ToHex(jitllm::base::Sha256{}.Update(std::as_bytes(kept)).Finish())
            << " text_sha256=" << jitllm::base::ToHex(jitllm::base::Sha256{}.Update(*text).Finish())
            << '\n';
  return {};
}
}  // namespace
int main(int argc, char** argv) {
  if (auto r = jitllm::platform::InstallCrashPolicy("gemma2-probe"); !r) return 2;
  if (argc == 5 && std::string_view(argv[1]) == "prepare") {
    const auto r = Prepare(argv[2], argv[3], argv[4]);
    if (!r) std::cerr << r.error() << '\n';
    return r ? 0 : 1;
  }
  if (argc != 7) return 2;
  const std::string_view mode = argv[4], policy = argv[5], graphs = argv[6];
  if ((mode != "teacher" && mode != "cycle" && mode != "lifetime" && mode != "greedy-own" &&
       mode != "greedy-cycle") ||
      (policy != "primitive" && policy != "optimized") || (graphs != "0" && graphs != "1"))
    return 2;
  const bool greedy_own = mode == "greedy-own", greedy_cycle = mode == "greedy-cycle";
  if ((greedy_own || greedy_cycle) && policy != "optimized") return 2;
  auto raw = Read(argv[2], kInput * sizeof(std::int32_t));
  if (!raw || raw->size() != kInput * sizeof(std::int32_t)) return 2;
  std::array<std::int32_t, kInput> ids{};
  std::memcpy(ids.data(), raw->data(), raw->size());
  if (ids.front() != 2 || std::ranges::any_of(ids, [](auto id) {
        return id < 0 || std::cmp_greater_equal(id, kVocab);
      }))
    return 2;
  const fs::path out = argv[3];
  if (!fs::create_directory(out)) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma2Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto& node = lifetime->node;
  lifetime->runner =
      std::make_unique<en::Gemma2Runner>(node,
                                         en::Gemma2Options{.artifact = argv[1],
                                                           .out = out,
                                                           .slots = 1,
                                                           .max_head_rows = 1,
                                                           .graphs = graphs == "1",
                                                           .fuse_norms = policy == "optimized",
                                                           .fuse_quant_glu = policy == "optimized",
                                                           .fuse_norm_add = policy == "optimized"},
                                         0, 0);
  auto& runner = *lifetime->runner;
  std::vector<float> logits;
  logits.reserve(kVocab);
  std::uint32_t past = 0;
  void* state_buffer = nullptr;
  constexpr std::uint64_t kStateBuffer = 8ULL << 20U;
  const auto snapshot = [&]() -> std::expected<std::string, std::string> {
    auto ranges = runner.CheckpointRanges(past);
    if (!ranges) return Error(ranges.error());
    jitllm::base::Sha256 hash;
    for (const auto& range : *ranges) {
      for (std::uint64_t at = 0; at < range.bytes;) {
        auto part = range;
        part.offset += at;
        part.bytes = std::min(kStateBuffer, range.bytes - at);
        en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
        const auto copied = runner.CopyState(0, state_buffer, std::span(&part, 1), &retirement);
        if (retirement == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(state_buffer);
        if (!copied) return Error(copied.error());
        hash.Update(std::span(static_cast<const std::byte*>(state_buffer),
                              static_cast<std::size_t>(part.bytes)));
        at += part.bytes;
      }
    }
    return jitllm::base::ToHex(hash.Finish());
  };
  const auto chunk = [&](std::span<const std::int32_t> tokens, bool head) -> en::Status {
    const en::Gemma2Runner::Work work{0, past, tokens, &logits};
    if (auto r = runner.WavePrefill(std::span(&work, 1), head); !r) return r;
    past += static_cast<std::uint32_t>(tokens.size());
    if (head) {
      auto finite = Greedy(logits);
      if (!finite) return Error(finite.error());
    } else if (!logits.empty())
      return Error("state-only prefill published a stale head");
    return {};
  };
  const auto prompt = [&](bool state_only) -> en::Status {
    if (auto r = chunk(std::span(ids).first(128), !state_only); !r) return r;
    return chunk(std::span(ids).subspan(128, 128), true);
  };
  const auto warm = [&]() -> en::Status {
    for (std::uint32_t i = 0; i < kWarm; ++i)
      if (auto r = chunk(std::span(ids).subspan(kPrompt + i, 1), true); !r) return r;
    return {};
  };
  std::int32_t selected_token = -1;
  const auto token_chunk = [&](std::span<const std::int32_t> tokens) -> en::Status {
    const en::Gemma2Runner::Work work{0, past, tokens, nullptr, &selected_token};
    if (auto r = runner.Wave(std::span(&work, 1)); !r) return r;
    past += static_cast<std::uint32_t>(tokens.size());
    return {};
  };
  const auto token_prompt = [&]() -> en::Status {
    if (auto r = chunk(std::span(ids).first(128), false); !r) return r;
    return token_chunk(std::span(ids).subspan(128, 128));
  };
  const auto token_warm = [&]() -> en::Status {
    for (std::uint32_t i = 0; i < kWarm; ++i)
      if (auto r = token_chunk(std::span(ids).subspan(kPrompt + i, 1)); !r) return r;
    return {};
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    lifetime->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (mode == "lifetime" || greedy_own) {
      std::vector<jitllm::catalog::ExtentId> staging;
      auto pinned = node.Pinned(kStateBuffer, 0, staging);
      if (!pinned) return Error(pinned.error());
      state_buffer = *pinned;
    }
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (8ULL << 20U));
    if (auto r = node.Start(jitllm::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                                2 * node.StateCapacity()));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    std::cout << "GEMMA2_ENVELOPE activations=" << runner.activations_needed()
              << " scratch=" << runner.pool_needed() << " host_inputs=" << runner.host_input_bytes()
              << " plan_floor=" << runner.plan_floor_bytes() << '\n';
    return node.WithRequest(0, runner.closure(), "Gemma2 first screen", [&]() -> en::Status {
      if (!node.InRequest(0)) return Error("probe must hold the direct-step request");
      if (mode == "cycle" || greedy_cycle) {
        if (auto r = greedy_cycle ? token_prompt() : prompt(true); !r) return r;
        if (auto r = greedy_cycle ? token_warm() : warm(); !r) return r;
        for (std::uint32_t i = 0; i < 8; ++i) {
          auto next = greedy_cycle && i < 6
                          ? std::expected<std::int32_t, std::string>(selected_token)
                          : Greedy(logits);
          if (!next) return Error(next.error());
          if (auto r = greedy_cycle && i < 5 ? token_chunk(std::span(&*next, 1))
                                             : chunk(std::span(&*next, 1), true);
              !r)
            return r;
        }
        if (auto r = runner.Clear(); !r) return r;
        past = 0;
      }
      const auto prefill_start = std::chrono::steady_clock::now();
      if (auto r = greedy_cycle ? token_prompt() : prompt(true); !r) return r;
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - prefill_start);
      if (auto r = greedy_cycle ? token_warm() : warm(); !r) return r;
      if (mode == "lifetime") {
        const auto before = logits;
        const auto before_state = snapshot();
        if (!before_state) return Error(before_state.error());
        const auto positions = (*runner.request_slot(0))->completed_positions();
        const std::int32_t invalid = -1;
        const en::Gemma2Runner::Work wrong_past{0, past + 1, std::span(ids).first(1), &logits};
        const en::Gemma2Runner::Work wrong_token{0, past, std::span(&invalid, 1), &logits};
        const en::Gemma2Runner::Work wide_head{0, past, std::span(ids).first(2), &logits};
        if (runner.Wave(std::span(&wrong_past, 1)) || runner.Wave(std::span(&wrong_token, 1)) ||
            runner.Wave(std::span(&wide_head, 1), true) ||
            (*runner.request_slot(0))->completed_positions() != positions)
          return Error("malformed work changed the completed prefix");
        const auto after_refusal = snapshot();
        if (!after_refusal || *after_refusal != *before_state)
          return Error("malformed work changed initialized state");
        const auto kept = runner.state();
        if (auto r = runner.Clear(); !r) return r;
        if (runner.kept_state() != kept)
          return Error("Clear did not retain the resident state backing");
        past = 0;
        if (auto r = prompt(false); !r) return r;
        if (auto r = warm(); !r) return r;
        if (before.size() != logits.size() ||
            std::memcmp(before.data(), logits.data(), logits.size() * sizeof(float)) != 0)
          return Error("full/state-only prefill or Clear changed the completed head");
        const auto after_clear = snapshot();
        if (!after_clear || *after_clear != *before_state)
          return Error("Clear and full/state-only prefill changed initialized state");
        if (auto r = runner.Spill(0); !r) return r;
        if (!(*runner.request_slot(0))->is_spilled() ||
            (*runner.request_slot(0))->completed_positions() != positions)
          return Error("spill changed the logical prefix");
        if (auto r = runner.Restore(0); !r) return r;
        const auto after_restore = snapshot();
        if (!after_restore || *after_restore != *before_state)
          return Error("spill/restore changed initialized state");
        std::cout
            << "GEMMA2_CLEAR refusals_unchanged=1 state_only_equal=1 restored_equal=1 kept_extents="
            << kept.size() << '\n';
      }
      std::ofstream heads;
      if (mode == "teacher" || greedy_own) {
        heads.open(out / "heads.f32", std::ios::binary | std::ios::noreplace);
        if (!heads) return Error("exclusive teacher output refused");
        heads.write(reinterpret_cast<const char*>(logits.data()), kVocab * sizeof(float));
      }
      std::array<std::int32_t, kSteps> chosen{};
      const auto decode_start = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < kSteps; ++i) {
        auto next = greedy_cycle ? std::expected<std::int32_t, std::string>(selected_token)
                                 : Greedy(logits);
        if (!next) return Error(next.error());
        chosen[i] = *next;
        const auto token = mode == "cycle" || greedy_cycle ? *next : ids[kPrompt + kWarm + i];
        if (auto r = greedy_cycle && i + 1 != kSteps ? token_chunk(std::span(&token, 1))
                                                     : chunk(std::span(&token, 1), true);
            !r)
          return r;
        if (mode == "teacher" || greedy_own)
          heads.write(reinterpret_cast<const char*>(logits.data()), kVocab * sizeof(float));
      }
      const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_start);
      if (mode == "teacher" || greedy_own) {
        heads.flush();
        if (!heads) return Error("complete teacher output failed");
      }
      if (greedy_own) {
        const auto full_choices = chosen;
        const auto full_final = logits;
        const auto full_state = snapshot();
        if (!full_state) return Error(full_state.error());
        const auto kept = runner.state();
        if (auto r = runner.Clear(); !r) return r;
        if (runner.kept_state() != kept) return Error("greedy Clear did not retain state backing");
        past = 0;
        const auto before_graphs = runner.graph_stats();
        if (auto r = token_prompt(); !r) return r;
        if (graphs == "1" && runner.graph_stats().eager <= before_graphs.eager)
          return Error("greedy frontier reused the full-head shape key");
        if (auto r = token_warm(); !r) return r;
        if (graphs == "1" && (runner.graph_stats().captured <= before_graphs.captured ||
                              runner.graph_stats().replayed <= before_graphs.replayed))
          return Error("greedy shape did not capture and replay fresh scalar inputs");
        const auto before_refusal = snapshot();
        if (!before_refusal) return Error(before_refusal.error());
        const auto slot0 = *runner.request_slot(0);
        const auto positions0 = slot0->completed_positions();
        const auto bytes0 = slot0->used_state_bytes();
        const auto untouched = selected_token;
        const en::Gemma2Runner::Work aliased0{0, past, std::span(ids).first(1), nullptr,
                                              &selected_token};
        const std::array aliased{aliased0, aliased0};
        const en::Gemma2Runner::Work mixed{0, past, std::span(ids).first(1), &logits,
                                           &selected_token};
        if (runner.Wave(aliased) || runner.Wave(std::span(&mixed, 1)) ||
            runner.Wave(std::span(&aliased0, 1), true) ||
            runner.WavePrefill(std::span(&aliased0, 1), false) || selected_token != untouched ||
            slot0->completed_positions() != positions0 || slot0->used_state_bytes() != bytes0)
          return Error(
              "greedy duplicate/mixed/publication refusal changed state metadata or output");
        const auto after_refusal = snapshot();
        if (!after_refusal || *after_refusal != *before_refusal)
          return Error("greedy publication refusal changed initialized state");
        for (std::uint32_t i = 0; i < kSteps; ++i) {
          chosen[i] = selected_token;
          if (chosen[i] != full_choices[i])
            return Error("device greedy choice differs from full-head argmax");
          const auto token = ids[kPrompt + kWarm + i];
          if (auto r = i + 1 == kSteps ? chunk(std::span(&token, 1), true)
                                       : token_chunk(std::span(&token, 1));
              !r)
            return r;
        }
        const auto device_state = snapshot();
        if (!device_state || *device_state != *full_state || logits.size() != full_final.size() ||
            std::memcmp(logits.data(), full_final.data(), logits.size() * sizeof(float)) != 0)
          return Error(
              "device greedy final head or initialized state differs from full-head execution");
        if (runner.greedy_tokens() != 35)
          return Error("greedy own control did not publish 35 GPU tokens");
        const auto state_record = std::format(
            "{{\"reference\":\"{}\",\"device\":\"{}\",\"choices_checked\":32,"
            "\"refusals_unchanged\":true,\"kept_extents\":{},\"allocated_slots\":1,\"executed_"
            "slots\":1}}\n",
            *full_state, *device_state, kept.size());
        if (auto r = Write<char>(out / "state.json", std::span(state_record)); !r) return r;
        std::cout
            << "GEMMA2_GREEDY_OWN choices_equal=32 final_equal=1 state_equal=1 refusals_unchanged=1"
               " kept_extents="
            << kept.size() << " allocated_slots=1 executed_slots=1\n";
      }
      if (auto r = Write<std::int32_t>(out / "chosen.i32", chosen); !r) return r;
      if (auto r = Write<float>(out / "final.f32", logits); !r) return r;
      const auto& stats = runner.graph_stats();
      const auto& selections = runner.plan_selections();
      if (graphs == "1" && (stats.captured == 0 || stats.replayed == 0))
        return Error("graph probe did not execute capture and replay");
      if (policy == "optimized" && (selections.norm_mul == 0 || selections.quant_geglu == 0 ||
                                    selections.norm_add == 0 || selections.norm_rope != 0))
        return Error("optimized probe did not select precisely the eligible fusion families");
      std::cout << "GEMMA2_PROBE mode=" << mode << " policy=" << policy << " context=4096 slots=1"
                << " chunk=128 prompt_rows=" << kPrompt << " untimed_rows=" << kWarm
                << " decode_rows=" << kSteps << " past=" << past << " prefill_seconds=" << prefill
                << " decode_seconds=" << decode << " eager=" << stats.eager
                << " captured=" << stats.captured << " replayed=" << stats.replayed
                << " refused=" << stats.refused << " plans_built=" << selections.plans
                << " selected_steps=" << selections.steps
                << " selected_norm_mul=" << selections.norm_mul
                << " selected_quant_geglu=" << selections.quant_geglu
                << " selected_norm_rope=" << selections.norm_rope
                << " selected_norm_add=" << selections.norm_add
                << " gpu_tokens=" << runner.greedy_tokens()
                << " coverage_violations=" << runner.coverage().violations << '\n';
      return {};
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(lifetime->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = lifetime.release();
  }
  if (ran && retired) std::cout << "GEMMA2_RETIRED\n";
  return ran && retired ? 0 : 1;
}
