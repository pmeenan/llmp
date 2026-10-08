// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Internal C2 decode screen: independently-prefilled real slots, not joint prefill.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/gemma2_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace en = llmp::engine;
namespace fs = std::filesystem;
using en::support::Error;
constexpr std::uint32_t kVocab = 256000, kInput = 4387, kSteps = 32;
constexpr std::uint64_t kCopy = 8ULL << 20U;
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
en::Status Save(const fs::path& path, std::span<const T> data) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("exclusive complete output refused");
}
std::expected<std::int32_t, std::string> Best(const std::vector<float>& row) {
  if (row.size() != kVocab || !std::ranges::all_of(row, [](float x) { return std::isfinite(x); }))
    return Error("finite complete head required");
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}
en::Status Prepare(const char* metadata_path, const char* text_path, const char* output_path) {
  auto metadata = Read(metadata_path, 32ULL << 20U);
  auto text = Read(text_path, 65536);
  if (!metadata || !text) return Error("preparation input refused");
  auto parsed = llmp::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*metadata)));
  if (!parsed) return Error(parsed.error().ToString());
  if (parsed->spec.tokens.size() != kVocab || parsed->spec.bos != 2 || !parsed->spec.add_bos ||
      parsed->spec.add_eos)
    return Error("Gemma2 vocabulary/BOS contract differs");
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(parsed->spec));
  if (!tokenizer) return Error(tokenizer.error().ToString());
  std::vector<llmp::tokenizer::TokenId> ids;
  if (auto r = tokenizer->Encode(*text, {.add_bos_eos = true, .max_tokens = 8192}, ids); !r)
    return Error(r.error().ToString());
  if (ids.size() < kInput || ids.front() != 2 || std::ranges::any_of(ids, [](auto id) {
        return id < 0 || std::cmp_greater_equal(id, kVocab);
      }))
    return Error("preparation needs 4387 valid actual tokens including BOS");
  if (!fs::create_directory(output_path)) return Error("preparation output must be new");
  const auto kept = std::span(ids).first(kInput);
  if (auto r = Save<std::int32_t>(fs::path(output_path) / "ids.i32", kept); !r) return r;
  std::cout << "GEMMA2_INPUT rows=" << kInput << " sha256="
            << llmp::base::ToHex(llmp::base::Sha256{}.Update(std::as_bytes(kept)).Finish())
            << " text_sha256=" << llmp::base::ToHex(llmp::base::Sha256{}.Update(*text).Finish())
            << '\n';
  return {};
}
}  // namespace
int main(int argc, char** argv) {
  if (!llmp::platform::InstallCrashPolicy("gemma2-c2-probe")) return 2;
  if (argc == 5 && std::string_view(argv[1]) == "prepare") {
    auto r = Prepare(argv[2], argv[3], argv[4]);
    if (!r) std::cerr << r.error() << '\n';
    return r ? 0 : 1;
  }
  if (argc != 6) return 2;
  const std::string mode = argv[5];
  const bool ring = mode == "ring";
  const bool eager = mode == "eager-own";
  const bool primitive = mode == "ordinary-own";
  const bool own = mode == "own" || ring || eager || primitive, cycle = mode == "cycle";
  const std::uint32_t prefix = ring ? 4352 : 256;
  if (!own && !cycle) return 2;
  std::array<std::array<std::int32_t, kInput>, 2> ids{};
  for (std::size_t s = 0; s < 2; ++s) {
    std::error_code ec;
    if (fs::file_size(argv[2 + s], ec) != kInput * 4 || ec) return 2;
    std::ifstream file(argv[2 + s], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(ids[s].data()), kInput * 4) ||
        file.peek() != std::char_traits<char>::eof() || ids[s][0] != 2 ||
        !std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < std::int32_t(kVocab); }))
      return 2;
  }
  if (ids[0] == ids[1]) return 2;
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma2Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner = std::make_unique<en::Gemma2Runner>(node,
                                                    en::Gemma2Options{.artifact = argv[1],
                                                                      .out = out,
                                                                      .context = 8192,
                                                                      .slots = 2,
                                                                      .max_head_rows = 2,
                                                                      .graphs = !eager,
                                                                      .owner_decode = !primitive,
                                                                      .fuse_norms = true,
                                                                      .fuse_quant_glu = true,
                                                                      .fuse_norm_rope = false,
                                                                      .fuse_norm_add = true},
                                                    0, 0);
  auto& runner = *life->runner;
  std::array<std::vector<float>, 2> heads, quality_final;
  for (auto& head : heads) head.reserve(kVocab);
  std::array<std::uint32_t, 2> past{};
  std::array<std::int32_t, 2> selected{-1, -1};
  void* pinned = nullptr;
  const auto snapshot = [&]() -> std::expected<std::array<std::string, 2>, std::string> {
    std::array<std::string, 2> result;
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      auto ranges = runner.CheckpointRanges(past[slot]);
      if (!ranges) return Error(ranges.error());
      llmp::base::Sha256 hash;
      for (const auto& range : *ranges)
        for (std::uint64_t at = 0; at < range.bytes;) {
          auto part = range;
          part.offset += at;
          part.bytes = std::min(kCopy, range.bytes - at);
          en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
          auto copied = runner.CopyState(slot, pinned, std::span(&part, 1), &retirement);
          if (retirement == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(pinned);
          if (!copied) return Error(copied.error());
          hash.Update(std::span(static_cast<const std::byte*>(pinned), std::size_t(part.bytes)));
          at += part.bytes;
        }
      result[slot] = llmp::base::ToHex(hash.Finish());
    }
    return result;
  };
  const auto independent = [&](std::uint32_t s, std::span<const std::int32_t> tokens, bool head,
                               bool token) -> en::Status {
    const en::Gemma2Runner::Work work{s, past[s], tokens, token ? nullptr : &heads[s],
                                      token ? &selected[s] : nullptr};
    if (auto r = runner.WavePrefill(std::span(&work, 1), head); !r) return r;
    past[s] += static_cast<std::uint32_t>(tokens.size());
    if (head && !token && !Best(heads[s])) return Error("bad independent head");
    if (!head && !heads[s].empty()) return Error("state-only head leak");
    return {};
  };
  const auto prompt = [&](bool token) -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s)
      for (std::uint32_t at = 0; at < prefix; at += 128)
        if (auto r = independent(s, std::span(ids[s]).subspan(at, 128), at + 128 == prefix,
                                 token && at + 128 == prefix);
            !r)
          return r;
    return {};
  };
  const auto warm = [&](bool token) -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s)
      for (std::uint32_t i = 0; i < 3; ++i)
        if (auto r = independent(s, std::span(ids[s]).subspan(prefix + i, 1), true, token); !r)
          return r;
    return {};
  };
  const auto joined = [&](const std::array<std::int32_t, 2>& tokens, bool token) -> en::Status {
    std::array<en::Gemma2Runner::Work, 2> work;
    for (std::uint32_t s = 0; s < 2; ++s)
      work[s] = {s, past[s], std::span(&tokens[s], 1), token ? nullptr : &heads[s],
                 token ? &selected[s] : nullptr};
    if (auto r = runner.Wave(work); !r) return r;
    for (std::uint32_t s = 0; s < 2; ++s) {
      ++past[s];
      if (!token && !Best(heads[s])) return Error("bad joined head");
    }
    return {};
  };
  const auto clear = [&]() -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s)
      if (auto r = runner.Clear(s); !r) return r;
    past = {};
    return {};
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (own) {
      std::vector<llmp::catalog::ExtentId> extents;
      auto allocation = node.Pinned(kCopy, 0, extents);
      if (!allocation) return Error(allocation.error());
      pinned = *allocation;
    }
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (16ULL << 20U));
    if (auto r = node.Start(llmp::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                              4 * node.StateCapacity()));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(0, runner.closure(), "Gemma2 C2 decode screen", [&]() -> en::Status {
      if (!node.InRequest(0)) return Error("held direct request required");
      const std::array<std::uint32_t, 2> slots{0, 1};
      if (auto r = runner.SelectSlots(slots); !r) return r;
      if (cycle) {
        if (auto r = prompt(true); !r) return r;
        if (auto r = warm(true); !r) return r;
        for (std::uint32_t i = 0; i < 8; ++i) {
          std::array<std::int32_t, 2> next;
          for (std::uint32_t s = 0; s < 2; ++s) next[s] = i < 6 ? selected[s] : *Best(heads[s]);
          if (auto r = joined(next, i < 5); !r) return r;
        }
        if (auto r = clear(); !r) return r;
      }
      const auto begin = std::chrono::steady_clock::now();
      if (auto r = prompt(cycle); !r) return r;
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - begin);
      if (auto r = warm(cycle); !r) return r;
      std::ofstream rows;
      const auto write_rows = [&]() {
        for (const auto& head : heads)
          rows.write(reinterpret_cast<const char*>(head.data()), kVocab * 4);
      };
      if (own) {
        rows.open(out / "heads.f32", std::ios::binary | std::ios::noreplace);
        if (!rows) return Error("teacher rows must be new");
        write_rows();
      }
      std::array<std::int32_t, kSteps * 2> choices{};
      const auto decode_begin = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < kSteps; ++i) {
        std::array<std::int32_t, 2> tokens;
        for (std::uint32_t s = 0; s < 2; ++s) {
          choices[i * 2 + s] = cycle ? selected[s] : *Best(heads[s]);
          tokens[s] = cycle ? choices[i * 2 + s] : ids[s][prefix + 3 + i];
        }
        if (auto r = joined(tokens, cycle && i + 1 != kSteps); !r) return r;
        if (own) write_rows();
      }
      const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_begin);
      if (own) {
        rows.flush();
        if (!rows) return Error("complete teacher rows failed");
        const auto full_heads = heads;
        quality_final = full_heads;
        const auto full_state = snapshot();
        if (!full_state) return Error(full_state.error());
        const auto kept = runner.state();
        if (auto r = clear(); !r) return r;
        if (runner.kept_state() != kept) return Error("Clear discarded resident backing");
        if (auto r = prompt(true); !r) return r;
        if (auto r = warm(true); !r) return r;
        const auto before = snapshot();
        if (!before) return Error(before.error());
        std::array<std::uint64_t, 2> bytes;
        for (std::uint32_t s = 0; s < 2; ++s)
          bytes[s] = (*runner.request_slot(s))->used_state_bytes();
        const auto untouched = selected;
        const std::array<en::Gemma2Runner::Work, 2> alias{
            {{0, past[0], std::span(ids[0]).first(1), nullptr, &selected[0]},
             {1, past[1], std::span(ids[1]).first(1), nullptr, &selected[0]}}};
        auto mixed = alias;
        mixed[1].token = nullptr;
        mixed[1].logits = &heads[1];
        auto wrong = alias;
        wrong[1].token = &selected[1];
        ++wrong[1].n_past;
        if (runner.Wave(alias) || runner.Wave(mixed) || runner.Wave(wrong) || selected != untouched)
          return Error("C2 malformed publication accepted or output changed");
        for (std::uint32_t s = 0; s < 2; ++s)
          if ((*runner.request_slot(s))->completed_positions() != past[s] ||
              (*runner.request_slot(s))->used_state_bytes() != bytes[s])
            return Error("C2 refusal changed prefix metadata");
        const auto after = snapshot();
        if (!after || *after != *before) return Error("C2 refusal changed initialized state");
        for (std::uint32_t i = 0; i < kSteps; ++i) {
          std::array<std::int32_t, 2> tokens;
          for (std::uint32_t s = 0; s < 2; ++s) {
            if (selected[s] != choices[i * 2 + s]) return Error("C2 GPU/full-head choices differ");
            tokens[s] = ids[s][prefix + 3 + i];
          }
          if (auto r = joined(tokens, i + 1 != kSteps); !r) return r;
        }
        for (std::uint32_t s = 0; s < 2; ++s)
          if (heads[s].size() != full_heads[s].size() ||
              std::memcmp(heads[s].data(), full_heads[s].data(), kVocab * 4) != 0)
            return Error("C2 GPU/full-head final differs");
        const auto device_state = snapshot();
        if (!device_state || *device_state != *full_state) return Error("C2 GPU state differs");
        for (std::uint32_t s = 0; s < 2; ++s) {
          if (auto r = runner.Spill(s); !r) return r;
          if (auto r = runner.Restore(s); !r) return r;
        }
        const auto restored = snapshot();
        if (!restored || *restored != *full_state) return Error("C2 spill/restore state differs");
        // A restored ring must produce the same next head/state as a fresh
        // uninterrupted replay of exactly the same supplied history.
        std::array<std::int32_t, 2> extra{ids[0][prefix + 2], ids[1][prefix + 2]};
        if (auto r = joined(extra, false); !r) return r;
        const auto restored_heads = heads;
        const auto restored_next = snapshot();
        if (!restored_next) return Error(restored_next.error());
        if (auto r = clear(); !r) return r;
        if (auto r = prompt(false); !r) return r;
        if (auto r = warm(false); !r) return r;
        for (std::uint32_t i = 0; i < kSteps; ++i) {
          const std::array<std::int32_t, 2> supplied{ids[0][prefix + 3 + i],
                                                     ids[1][prefix + 3 + i]};
          if (auto r = joined(supplied, false); !r) return r;
        }
        if (auto r = joined(extra, false); !r) return r;
        const auto uninterrupted = snapshot();
        if (!uninterrupted || *uninterrupted != *restored_next)
          return Error("C2 restored next state differs from uninterrupted history");
        for (std::uint32_t s = 0; s < 2; ++s)
          if (std::memcmp(heads[s].data(), restored_heads[s].data(), kVocab * 4) != 0)
            return Error("C2 restored next head differs from uninterrupted history");
        const auto record =
            "{\"slot0\":\"" + (*full_state)[0] + "\",\"slot1\":\"" + (*full_state)[1] +
            "\",\"gpu_equal\":true,\"restore_equal\":true,\"refusals_unchanged\":true}\n";
        if (auto r = Save<char>(out / "state.json", record); !r) return r;
        if (runner.greedy_tokens() != 70) return Error("C2 own GPU publication count differs");
      }
      if (cycle && runner.greedy_tokens() != 88)
        return Error("C2 cycle GPU publication count differs");
      if (auto r = Save<std::int32_t>(out / "chosen.i32", choices); !r) return r;
      std::vector<float> final;
      final.reserve(kVocab * 2);
      for (const auto& head : own ? quality_final : heads)
        final.insert(final.end(), head.begin(), head.end());
      if (auto r = Save<float>(out / "final.f32", final); !r) return r;
      const auto& stats = runner.graph_stats();
      const auto& bound = runner.plan_selections();
      if ((!eager && (!stats.captured || !stats.replayed)) ||
          (eager && (stats.captured || stats.replayed)) ||
          (primitive ? bound.owner_attention != 0 : bound.owner_attention == 0) ||
          bound.norm_rope || !bound.norm_add || runner.coverage().violations)
        return Error("C2 did not select/replay checked implementation families");
      std::cout << "GEMMA2_C2 mode=" << mode << " slots=2 context_per_slot=8192 chunk=128"
                << " independent_prefill=1 prompt_rows_per_slot=" << prefix
                << " untimed_rows_per_slot=3"
                << " decode_steps=32 paid_generated_tokens=64 past0=" << past[0]
                << " past1=" << past[1] << " prefill_seconds=" << prefill
                << " decode_seconds=" << decode << " eager=" << stats.eager
                << " captured=" << stats.captured << " replayed=" << stats.replayed
                << " selected_owner=" << bound.owner_attention
                << " selected_norm_mul=" << bound.norm_mul
                << " selected_quant_geglu=" << bound.quant_geglu
                << " selected_norm_rope=" << bound.norm_rope
                << " selected_norm_add=" << bound.norm_add
                << " gpu_tokens=" << runner.greedy_tokens() << '\n';
      return {};
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(life->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = life.release();
  }
  if (ran && retired) std::cout << "GEMMA2_C2_RETIRED\n";
  return ran && retired ? 0 : 1;
}
