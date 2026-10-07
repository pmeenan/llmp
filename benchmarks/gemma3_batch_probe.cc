// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Paired prefill, four independent states and matched C4/solo decode controls.
#include <algorithm>
#include <array>
#include <charconv>
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
#include "engine/checkpoint_file.h"
#include "engine/gemma3_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace en = jitllm::engine;
namespace fs = std::filesystem;
using en::support::Error;
constexpr std::uint32_t kVocab = 262208, kSteps = 32, kTail = 4, kOwners = 4;
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
en::Status Prepare(const char* metadata_path, const char* text_path, const char* output_path,
                   std::uint32_t input_rows) {
  auto metadata = Read(metadata_path, 32ULL << 20U);
  auto text = Read(text_path, 65536);
  if (!metadata || !text) return Error("preparation input refused");
  auto parsed = jitllm::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*metadata)));
  if (!parsed) return Error(parsed.error().ToString());
  if (parsed->spec.tokens.size() != kVocab || parsed->spec.bos != 2 || !parsed->spec.add_bos ||
      parsed->spec.add_eos)
    return Error("Gemma3 vocabulary/BOS contract differs");
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(parsed->spec));
  if (!tokenizer) return Error(tokenizer.error().ToString());
  std::vector<jitllm::tokenizer::TokenId> ids;
  if (auto r = tokenizer->Encode(*text, {.add_bos_eos = true, .max_tokens = 8192}, ids); !r)
    return Error(r.error().ToString());
  if (ids.size() < input_rows || ids.front() != 2 || std::ranges::any_of(ids, [](auto id) {
        return id < 0 || std::cmp_greater_equal(id, kVocab);
      }))
    return Error("preparation needs 295 valid actual tokens including BOS");
  if (!fs::create_directory(output_path)) return Error("preparation output must be new");
  const auto kept = std::span(ids).first(input_rows);
  if (auto r = Save<std::int32_t>(fs::path(output_path) / "ids.i32", kept); !r) return r;
  std::cout << "GEMMA3_INPUT rows=" << input_rows << " sha256="
            << jitllm::base::ToHex(jitllm::base::Sha256{}.Update(std::as_bytes(kept)).Finish())
            << " text_sha256=" << jitllm::base::ToHex(jitllm::base::Sha256{}.Update(*text).Finish())
            << '\n';
  return {};
}
}  // namespace
int main(int argc, char** argv) {
  if (!jitllm::platform::InstallCrashPolicy("gemma3-batch-probe") || (argc != 6 && argc != 7))
    return 2;
  if (std::string_view(argv[1]) == "prepare") {
    if (argc != 6) return 2;
    std::uint32_t count = 0;
    const std::string_view number = argv[4];
    const auto parsed = std::from_chars(number.data(), number.data() + number.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || count < 295 ||
        count > 3968)
      return 2;
    const auto status = Prepare(argv[2], argv[3], argv[5], count);
    if (!status) std::cerr << status.error() << '\n';
    return status ? 0 : 1;
  }
  const std::string_view policy = argc == 7 ? argv[6] : "joined";
  const bool solo = policy == "solo", eager = policy == "eager";
  if (policy != "joined" && !solo && !eager) return 2;
  const std::string mode = argv[5];
  const bool own = mode == "own", cycle = mode == "cycle";
  if ((!own && !cycle) || (cycle && (solo || eager))) return 2;
  std::array<std::vector<std::int32_t>, kOwners> ids;
  std::array<std::uint32_t, kOwners> prefix{};
  for (std::size_t s = 0; s < kOwners; ++s) {
    std::error_code ec;
    const auto bytes = fs::file_size(argv[2 + s % 2], ec);
    if (ec || bytes % 4 || bytes < 295 * 4 || bytes > 3968 * 4) return 2;
    ids[s].resize(bytes / 4);
    prefix[s] = static_cast<std::uint32_t>(ids[s].size()) - 3 - kSteps - kTail;
    std::ifstream file(argv[2 + s % 2], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(ids[s].data()),
                   static_cast<std::streamsize>(ids[s].size() * 4)) ||
        file.peek() != std::char_traits<char>::eof() || ids[s][0] != 2 ||
        !std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < std::int32_t(kVocab); }))
      return 2;
  }
  if (ids[0] == ids[1] || prefix[0] != 256 || prefix[1] != 256) return 2;
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma3Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner = std::make_unique<en::Gemma3Runner>(node,
                                                    en::Gemma3Options{.artifact = argv[1],
                                                                      .out = out,
                                                                      .context = 4096,
                                                                      .max_rows = 128,
                                                                      .slots = kOwners,
                                                                      .max_wave_rows = 256,
                                                                      .max_head_rows = kOwners,
                                                                      .graphs = !eager,
                                                                      .owner_decode = true,
                                                                      .packed_prefill = true,
                                                                      .bounded_roots = true,
                                                                      .fuse_norms = true,
                                                                      .fuse_quant_glu = true,
                                                                      .fuse_norm_rope = true,
                                                                      .fuse_norm_add = true},
                                                    0, 0);
  auto& runner = *life->runner;
  std::array<std::vector<float>, kOwners> heads;
  for (auto& head : heads) head.reserve(kVocab);
  std::array<std::uint32_t, kOwners> past{};
  std::array<std::int32_t, kOwners> selected{-1, -1, -1, -1};
  void* pinned = nullptr;
  const auto snapshot = [&]() -> std::expected<std::array<std::string, kOwners>, std::string> {
    std::array<std::string, kOwners> result;
    for (std::uint32_t slot = 0; slot < kOwners; ++slot) {
      auto ranges = runner.CheckpointRanges(past[slot]);
      if (!ranges) return Error(ranges.error());
      jitllm::base::Sha256 hash;
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
      result[slot] = jitllm::base::ToHex(hash.Finish());
    }
    return result;
  };
  const auto independent = [&](std::uint32_t s, std::span<const std::int32_t> tokens, bool head,
                               bool token) -> en::Status {
    const en::Gemma3Runner::Work work{s, past[s], tokens, token ? nullptr : &heads[s],
                                      token ? &selected[s] : nullptr};
    if (auto r = runner.WavePrefill(std::span(&work, 1), head); !r) return r;
    past[s] += static_cast<std::uint32_t>(tokens.size());
    if (head && !token && !Best(heads[s])) return Error("bad independent head");
    if (!head && !heads[s].empty()) return Error("state-only head leak");
    return {};
  };
  std::uint64_t prefill_groups = 0, prefill_rows = 0;
  const auto prompt = [&](bool token) -> en::Status {
    // Preserve the qualified paired-prefill capacity and 256-row wave. Four
    // decode owners never imply four 128-row chunks in one prompt wave.
    for (std::uint32_t first = 0; first < kOwners; first += 2) {
      while (past[first] < prefix[first]) {
        const auto n = std::min(128U, prefix[first] - past[first]);
        const bool head = past[first] + n == prefix[first];
        std::array<en::Gemma3Runner::Work, 2> work;
        for (std::uint32_t i = 0; i < 2; ++i) {
          const auto s = first + i;
          if (past[s] != past[first] || prefix[s] != prefix[first])
            return Error("paired prompt geometry differs");
          work[i] = {s, past[s], std::span(ids[s]).subspan(past[s], n),
                     token && head ? nullptr : &heads[s], token && head ? &selected[s] : nullptr};
        }
        if (auto r = runner.WavePrefill(work, head); !r) return r;
        for (const auto& unit : work) {
          past[unit.slot] += n;
          if (head && !token && !Best(heads[unit.slot])) return Error("bad paired prefill head");
          if (!head && !heads[unit.slot].empty()) return Error("paired state-only head leak");
        }
        ++prefill_groups;
        prefill_rows += n * 2;
      }
    }
    return {};
  };
  const auto warm = [&](bool token) -> en::Status {
    for (std::uint32_t s = 0; s < kOwners; ++s)
      for (std::uint32_t i = 0; i < 3; ++i)
        if (auto r = independent(s, std::span(ids[s]).subspan(prefix[s] + i, 1), true, token); !r)
          return r;
    return {};
  };
  const auto joined = [&](const std::array<std::int32_t, kOwners>& tokens,
                          bool token) -> en::Status {
    std::array<en::Gemma3Runner::Work, kOwners> work;
    for (std::uint32_t s = 0; s < kOwners; ++s)
      work[s] = {s, past[s], std::span(&tokens[s], 1), token ? nullptr : &heads[s],
                 token ? &selected[s] : nullptr};
    if (solo) {
      for (const auto& unit : work)
        if (auto r = runner.Wave(std::span(&unit, 1)); !r) return r;
    } else if (auto r = runner.Wave(work); !r) {
      return r;
    }
    for (std::uint32_t s = 0; s < kOwners; ++s) {
      ++past[s];
      if (!token && !Best(heads[s])) return Error("bad joined head");
    }
    return {};
  };
  const auto clear = [&]() -> en::Status {
    for (std::uint32_t s = 0; s < kOwners; ++s)
      if (auto r = runner.Clear(s); !r) return r;
    past = {};
    return {};
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (own) {
      std::vector<jitllm::catalog::ExtentId> extents;
      auto allocation = node.Pinned(kCopy, 0, extents);
      if (!allocation) return Error(allocation.error());
      pinned = *allocation;
    }
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (16ULL << 20U));
    if (auto r = node.Start(jitllm::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                                4 * node.StateCapacity()));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(0, runner.closure(), "Gemma3 C4 decode screen", [&]() -> en::Status {
      if (!node.InRequest(0)) return Error("held direct request required");
      const std::array<std::uint32_t, kOwners> slots{0, 1, 2, 3};
      if (auto r = runner.SelectSlots(slots); !r) return r;
      if (cycle) {
        if (auto r = prompt(true); !r) return r;
        if (auto r = warm(true); !r) return r;
        for (std::uint32_t i = 0; i < 8; ++i) {
          std::array<std::int32_t, kOwners> next;
          for (std::uint32_t s = 0; s < kOwners; ++s)
            next[s] = i < 6 ? selected[s] : *Best(heads[s]);
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
      std::vector<std::int32_t> choices((kSteps + (own ? kTail : 0U)) * kOwners);
      const auto decode_begin = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < kSteps; ++i) {
        std::array<std::int32_t, kOwners> tokens;
        for (std::uint32_t s = 0; s < kOwners; ++s) {
          choices[i * kOwners + s] = cycle ? selected[s] : *Best(heads[s]);
          tokens[s] = cycle ? choices[i * kOwners + s] : ids[s][prefix[s] + 3 + i];
        }
        if (auto r = joined(tokens, cycle && i + 1 != kSteps); !r) return r;
        if (own) write_rows();
      }
      if (own) {
        // One owner leaves the joined cohort; the peer's prefix remains untouched.
        for (std::uint32_t s = 0; s < kOwners; ++s) {
          const auto peer_prefix = past;
          for (std::uint32_t i = 0; i < kTail; ++i) {
            choices[kSteps * kOwners + s * kTail + i] = *Best(heads[s]);
            if (auto r = independent(s, std::span(ids[s]).subspan(past[s], 1), true, false); !r)
              return r;
            rows.write(reinterpret_cast<const char*>(heads[s].data()), kVocab * 4);
          }
          for (std::uint32_t peer = 0; peer < kOwners; ++peer)
            if (peer != s &&
                (past[peer] != peer_prefix[peer] ||
                 (*runner.request_slot(peer))->completed_positions() != peer_prefix[peer]))
              return Error("partial departure changed peer cursor");
        }
      }
      const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_begin);
      if (own) {
        rows.flush();
        if (!rows) return Error("complete teacher rows failed");
        const auto full_heads = heads;
        const auto full_state = snapshot();
        if (!full_state) return Error(full_state.error());
        const auto kept = runner.state();
        if (auto r = clear(); !r) return r;
        if (runner.kept_state() != kept) return Error("Clear discarded resident backing");
        if (auto r = prompt(true); !r) return r;
        if (auto r = warm(true); !r) return r;
        const auto before = snapshot();
        if (!before) return Error(before.error());
        std::array<std::uint64_t, kOwners> bytes;
        for (std::uint32_t s = 0; s < kOwners; ++s)
          bytes[s] = (*runner.request_slot(s))->used_state_bytes();
        const auto untouched = selected;
        std::array<en::Gemma3Runner::Work, kOwners> alias;
        for (std::uint32_t s = 0; s < kOwners; ++s)
          alias[s] = {s, past[s], std::span(ids[s]).first(1), nullptr, &selected[s]};
        alias[1].token = &selected[0];
        auto mixed = alias;
        mixed[1].token = nullptr;
        mixed[1].logits = &heads[1];
        auto wrong = alias;
        wrong[1].token = &selected[1];
        ++wrong[1].n_past;
        if (runner.Wave(alias) || runner.Wave(mixed) || runner.Wave(wrong) || selected != untouched)
          return Error("C4 malformed publication accepted or output changed");
        for (std::uint32_t s = 0; s < kOwners; ++s)
          if ((*runner.request_slot(s))->completed_positions() != past[s] ||
              (*runner.request_slot(s))->used_state_bytes() != bytes[s])
            return Error("C4 refusal changed prefix metadata");
        const auto after = snapshot();
        if (!after || *after != *before) return Error("C4 refusal changed initialized state");
        for (std::uint32_t i = 0; i < kSteps; ++i) {
          std::array<std::int32_t, kOwners> tokens;
          for (std::uint32_t s = 0; s < kOwners; ++s) {
            if (selected[s] != choices[i * kOwners + s])
              return Error("C4 GPU/full-head choices differ");
            tokens[s] = ids[s][prefix[s] + 3 + i];
          }
          if (auto r = joined(tokens, true); !r) return r;
        }
        for (std::uint32_t s = 0; s < kOwners; ++s)
          for (std::uint32_t i = 0; i < kTail; ++i) {
            if (selected[s] != choices[kSteps * kOwners + s * kTail + i])
              return Error("partial GPU/full-head choice differs");
            if (auto r =
                    independent(s, std::span(ids[s]).subspan(past[s], 1), true, i + 1 != kTail);
                !r)
              return r;
          }
        for (std::uint32_t s = 0; s < kOwners; ++s)
          if (heads[s].size() != full_heads[s].size() ||
              std::memcmp(heads[s].data(), full_heads[s].data(), kVocab * 4) != 0)
            return Error("C4 GPU/full-head final differs");
        const auto device_state = snapshot();
        if (!device_state || *device_state != *full_state) return Error("C4 GPU state differs");
        for (std::uint32_t s = 0; s < kOwners; ++s) {
          if (auto r = runner.Spill(s); !r) return r;
          if (auto r = runner.Restore(s); !r) return r;
        }
        const auto restored = snapshot();
        if (!restored || *restored != *full_state) return Error("C4 spill/restore state differs");
        std::array<std::optional<en::CheckpointFile>, kOwners> checkpoints;
        std::array<std::vector<en::LiveState::Range>, kOwners> footprints;
        const auto boundary = past;
        for (std::uint32_t s = 0; s < kOwners; ++s) {
          footprints[s] = (*runner.request_slot(s))->state().used_ranges();
          auto checkpoint = en::CheckpointFile::Capture(
              node, out, footprints[s],
              [&](void* host, std::span<const en::LiveState::Range> ranges) {
                return runner.CopyState(s, host, ranges);
              });
          if (!checkpoint) return Error(checkpoint.error().detail);
          checkpoints[s] = std::move(*checkpoint);
        }
        std::array<std::vector<float>, kOwners> continued;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
          for (std::uint32_t s = 0; s < kOwners; ++s) {
            if (auto x = independent(s, std::span(ids[s]).first(128), true, false); !x) return x;
            if (repeat == 0)
              continued[s] = heads[s];
            else if (heads[s].size() != continued[s].size() ||
                     std::memcmp(heads[s].data(), continued[s].data(), kVocab * 4))
              return Error("checkpoint continuation differs");
          }
          for (std::uint32_t s = 0; s < kOwners; ++s) {
            auto restored_checkpoint = checkpoints[s]->Restore(
                node,
                [&] {
                  return runner.PrepareRestore(s, boundary[s], footprints[s],
                                               runner.CheckpointLayoutId());
                },
                [&](void* host, std::span<const en::LiveState::Range> ranges) {
                  return runner.CopyState(s, host, ranges, false);
                });
            if (!restored_checkpoint) return Error(restored_checkpoint.error().detail);
            if (auto x = runner.CompleteRestore(s, boundary[s]); !x) return x;
            past[s] = boundary[s];
          }
          const auto checkpoint_state = snapshot();
          if (!checkpoint_state || *checkpoint_state != *full_state)
            return Error("ring checkpoint restored state differs");
        }
        heads = full_heads;
        std::string record = "{";
        for (std::uint32_t s = 0; s < kOwners; ++s)
          record += (s == 0 ? "" : ",") + std::string("\"slot") + std::to_string(s) + "\":\"" +
                    (*full_state)[s] + "\"";
        record += ",\"gpu_equal\":true,\"restore_equal\":true,\"refusals_unchanged\":true," +
                  std::string("\"joined_prefill_groups\":") + std::to_string(prefill_groups) +
                  ",\"joined_prefill_rows\":" + std::to_string(prefill_rows) + "}\n";
        if (auto r = Save<char>(out / "state.json", record); !r) return r;
        if (runner.greedy_tokens() != 39 * kOwners)
          return Error("C4 own GPU publication count differs");
      }
      if (cycle && runner.greedy_tokens() != 44 * kOwners)
        return Error("C4 cycle GPU publication count differs");
      if (auto r = Save<std::int32_t>(out / "chosen.i32", choices); !r) return r;
      std::vector<float> final;
      final.reserve(kVocab * kOwners);
      for (const auto& head : heads) final.insert(final.end(), head.begin(), head.end());
      if (auto r = Save<float>(out / "final.f32", final); !r) return r;
      const auto& stats = runner.graph_stats();
      const auto& bound = runner.plan_selections();
      if (!prefill_groups || !bound.packed_prefill_attention || (!solo && !bound.owner_attention) ||
          (!eager && (!stats.captured || !stats.replayed)) || !bound.norm_rope || !bound.norm_add ||
          runner.coverage().violations)
        return Error("C4 did not select/replay checked implementation families");
      if (bound.bounded_owner_attention != 0)
        return Error("equal-width C4 selected unequal bounded roots");
      std::cout << "GEMMA3_C4_BATCH mode=" << mode
                << " slots=4 context_per_slot=4096 chunk=128 input_identities=2"
                << " compatible_prefill=1 max_wave_rows=256 prompt_rows0=" << prefix[0]
                << " prompt_rows1=" << prefix[1] << " untimed_rows_per_slot=3"
                << " decode_steps=32 departure_steps=" << (own ? kTail : 0U)
                << " paid_generated_tokens=128 past0=" << past[0] << " past1=" << past[1]
                << " past2=" << past[2] << " past3=" << past[3] << " prefill_seconds=" << prefill
                << " decode_seconds=" << decode << " eager=" << stats.eager
                << " captured=" << stats.captured << " replayed=" << stats.replayed
                << " selected_owner=" << bound.owner_attention
                << " selected_packed_prefill=" << bound.packed_prefill_attention
                << " policy=" << policy << " graphs=" << !eager
                << " selected_bounded_owner=" << bound.bounded_owner_attention
                << " selected_norm_mul=" << bound.norm_mul
                << " selected_quant_geglu=" << bound.quant_geglu
                << " selected_norm_rope=" << bound.norm_rope
                << " selected_norm_add=" << bound.norm_add
                << " joined_prefill_groups=" << prefill_groups
                << " joined_prefill_rows=" << prefill_rows
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
  if (ran && retired) std::cout << "GEMMA3_C4_BATCH_RETIRED\n";
  return ran && retired ? 0 : 1;
}
