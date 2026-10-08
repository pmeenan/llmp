// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Compatible two-owner prefill, ring/departure/restore and matched C2 decode.
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
#include <optional>
#include <span>
#include <string>
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
constexpr std::uint32_t kVocab = 262208, kSteps = 32, kTail = 4;
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
    return Error("preparation needs 291 valid actual tokens including BOS");
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
  if (!jitllm::platform::InstallCrashPolicy("gemma3-joint-prefill-probe") ||
      (argc < 6 || argc > 13))
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
  bool bounded = false, owner_prefill = false;
  bool flexible = false, have_chunk = false, stock_ring = false;
  bool device_masks = false, prefill_ahead = false;
  std::uint32_t chunk = 128;
  for (int i = 6; i < argc; ++i) {
    const std::string_view flag = argv[i];
    if (flag == "bounded-roots" && !bounded)
      bounded = true;
    else if (flag == "owner-prefill" && !owner_prefill)
      owner_prefill = true;
    else if (flag == "flexible-owner-prefill" && !flexible)
      flexible = true;
    else if (flag == "stock-ring" && !stock_ring)
      stock_ring = true;
    else if (flag == "device-masks" && !device_masks)
      device_masks = true;
    else if (flag == "prefill-ahead" && !prefill_ahead)
      prefill_ahead = true;
    else if (flag.starts_with("chunk=") && !have_chunk) {
      const auto number = flag.substr(6);
      const auto parsed = std::from_chars(number.data(), number.data() + number.size(), chunk);
      if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || chunk < 2 ||
          chunk > 256)
        return 2;
      have_chunk = true;
    } else
      return 2;
  }
  if (flexible && !owner_prefill) return 2;
  if (stock_ring && chunk != 256) return 2;
  // Match stock's joined ubatch allowance for this explicit comparison only.
  const auto state_rows = stock_ring ? 2 * chunk : chunk;
  const std::string mode = argv[5];
  const bool own = mode == "own", cycle = mode == "cycle";
  if (!own && !cycle) return 2;
  std::array<std::vector<std::int32_t>, 2> ids;
  std::array<std::uint32_t, 2> prefix{};
  for (std::size_t s = 0; s < 2; ++s) {
    std::error_code ec;
    const auto bytes = fs::file_size(argv[2 + s], ec);
    if (ec || bytes % 4 || bytes < 295 * 4 || bytes > 3968 * 4) return 2;
    ids[s].resize(bytes / 4);
    prefix[s] = static_cast<std::uint32_t>(ids[s].size()) - 3 - kSteps - kTail;
    std::ifstream file(argv[2 + s], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(ids[s].data()),
                   static_cast<std::streamsize>(ids[s].size() * 4)) ||
        file.peek() != std::char_traits<char>::eof() || ids[s][0] != 2 ||
        !std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < std::int32_t(kVocab); }))
      return 2;
  }
  if (ids[0] == ids[1]) return 2;
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma3Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner =
      std::make_unique<en::Gemma3Runner>(node,
                                         en::Gemma3Options{.artifact = argv[1],
                                                           .out = out,
                                                           .max_rows = state_rows,
                                                           .slots = 2,
                                                           .max_wave_rows = 2 * chunk,
                                                           .max_head_rows = 2,
                                                           .owner_decode = true,
                                                           .packed_prefill = true,
                                                           .owner_prefill = owner_prefill,
                                                           .flexible_owner_prefill = flexible,
                                                           .bounded_roots = bounded,
                                                           .device_masks = device_masks,
                                                           .fuse_norms = true,
                                                           .fuse_quant_glu = true,
                                                           .fuse_norm_rope = true,
                                                           .fuse_norm_add = true,
                                                           .prefill_lookahead = prefill_ahead,
                                                           .capture_ahead = prefill_ahead},
                                         0, 0);
  auto& runner = *life->runner;
  std::array<std::vector<float>, 2> heads;
  for (auto& head : heads) head.reserve(kVocab);
  std::array<std::uint32_t, 2> past{};
  std::array<std::int32_t, 2> selected{-1, -1};
  void* pinned = nullptr;
  const auto snapshot = [&]() -> std::expected<std::array<std::string, 2>, std::string> {
    std::array<std::string, 2> result;
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
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
    while (past[0] < prefix[0] || past[1] < prefix[1]) {
      const std::uint32_t first = past[0] == prefix[0]                         ? 1U
                                  : past[1] == prefix[1]                       ? 0U
                                  : prefix[0] - past[0] <= prefix[1] - past[1] ? 0U
                                                                               : 1U;
      const auto first_rows = std::min(chunk, prefix[first] - past[first]);
      const bool head = past[first] + first_rows == prefix[first];
      std::array<en::Gemma3Runner::Work, 2> work;
      std::size_t count = 0;
      std::uint32_t rows = 0;
      for (const std::uint32_t s : {first, 1U - first}) {
        if (past[s] == prefix[s]) continue;
        const auto n = std::min(chunk, prefix[s] - past[s]);
        if ((past[s] + n == prefix[s]) != head) continue;
        work[count++] = {s, past[s], std::span(ids[s]).subspan(past[s], n),
                         token && head ? nullptr : &heads[s],
                         token && head ? &selected[s] : nullptr};
        rows += n;
      }
      std::array<en::Gemma3Runner::PrefillNext, 2> hints{};
      std::optional<bool> next_head, after_head;
      bool mixed_next = false, mixed_after = false;
      bool token_next = false, token_after = false;
      const auto mode = [](std::optional<bool>& out, bool& mixed, bool value) {
        if (out.has_value() && *out != value)
          mixed = true;
        else if (!out.has_value())
          out = value;
      };
      if (prefill_ahead)
        for (std::size_t i = 0; i < count; ++i) {
          const auto& unit = work[i];
          const auto end = unit.n_past + static_cast<std::uint32_t>(unit.tokens.size());
          auto& hint = hints[i];
          hint.slot = unit.slot;
          hint.rows = std::min(chunk, prefix[unit.slot] - end);
          if (hint.rows != 0) {
            const bool next_final = end + hint.rows == prefix[unit.slot];
            mode(next_head, mixed_next, next_final);
            token_next |= token && next_final;
            hint.after = std::min(chunk, prefix[unit.slot] - end - hint.rows);
            if (hint.after != 0) {
              const bool after_final = end + hint.rows + hint.after == prefix[unit.slot];
              mode(after_head, mixed_after, after_final);
              token_after |= token && after_final;
            }
          }
        }
      // PrefillNext describes full heads or state-only work, not GPU-token heads.
      // Keep row descriptors for later positions while suppressing unsupported stages.
      if (mixed_next || token_next) next_head.reset();
      if (mixed_after || token_after) after_head.reset();
      if (auto r = runner.WavePrefill(std::span(work).first(count), head,
                                      std::span(hints).first(prefill_ahead ? count : 0U), next_head,
                                      after_head);
          !r)
        return r;
      for (const auto& unit : std::span(work).first(count)) {
        past[unit.slot] += static_cast<std::uint32_t>(unit.tokens.size());
        if (head && !token && !Best(heads[unit.slot])) return Error("bad joined prefill head");
        if (!head && !heads[unit.slot].empty()) return Error("joined state-only head leak");
      }
      if (count == 2) {
        ++prefill_groups;
        prefill_rows += rows;
      }
    }
    return {};
  };
  const auto warm = [&](bool token) -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s)
      for (std::uint32_t i = 0; i < 3; ++i)
        if (auto r = independent(s, std::span(ids[s]).subspan(prefix[s] + i, 1), true, token); !r)
          return r;
    return {};
  };
  const auto joined = [&](const std::array<std::int32_t, 2>& tokens, bool token) -> en::Status {
    std::array<en::Gemma3Runner::Work, 2> work;
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
    if (stock_ring && runner.layout().local_cells != 1536)
      return Error("stock ring diagnostic requires 1536 local cells");
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
    return node.WithRequest(0, runner.closure(), "Gemma3 C2 decode screen", [&]() -> en::Status {
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
        const auto warm_state = runner.state();
        const auto warm_plans = runner.plans_bytes();
        const auto warm_graphs = runner.graph_count();
        if (auto r = clear(); !r) return r;
        if (warm_state.empty() || warm_plans == 0 || warm_graphs == 0 ||
            runner.kept_state() != warm_state || runner.plans_bytes() != warm_plans ||
            runner.graph_count() != warm_graphs)
          return Error("warm cycle Clear changed retained backing, plans or graphs");
        std::cout << "warm_retained_state_extents=" << warm_state.size()
                  << " warm_retained_plan_bytes=" << warm_plans
                  << " warm_retained_graphs=" << warm_graphs << '\n';
      }
      const auto graph_before = runner.graph_stats();
      const auto ahead_before = runner.lookahead_stats();
      const auto begin = std::chrono::steady_clock::now();
      if (auto r = prompt(cycle); !r) return r;
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - begin);
      const auto graph_prefill = runner.graph_stats();
      const auto ahead_prefill = runner.lookahead_stats();
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
      std::vector<std::int32_t> choices((kSteps + (own ? kTail : 0U)) * 2);
      const auto decode_begin = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < kSteps; ++i) {
        std::array<std::int32_t, 2> tokens;
        for (std::uint32_t s = 0; s < 2; ++s) {
          choices[i * 2 + s] = cycle ? selected[s] : *Best(heads[s]);
          tokens[s] = cycle ? choices[i * 2 + s] : ids[s][prefix[s] + 3 + i];
        }
        if (auto r = joined(tokens, cycle && i + 1 != kSteps); !r) return r;
        if (own) write_rows();
      }
      if (own) {
        // One owner leaves the joined cohort; the peer's prefix remains untouched.
        for (std::uint32_t s = 0; s < 2; ++s) {
          const auto peer_prefix = past[1 - s];
          for (std::uint32_t i = 0; i < kTail; ++i) {
            choices[kSteps * 2 + s * kTail + i] = *Best(heads[s]);
            if (auto r = independent(s, std::span(ids[s]).subspan(past[s], 1), true, false); !r)
              return r;
            rows.write(reinterpret_cast<const char*>(heads[s].data()), kVocab * 4);
          }
          if (past[1 - s] != peer_prefix ||
              (*runner.request_slot(1 - s))->completed_positions() != peer_prefix)
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
        std::array<std::uint64_t, 2> bytes;
        for (std::uint32_t s = 0; s < 2; ++s)
          bytes[s] = (*runner.request_slot(s))->used_state_bytes();
        const auto untouched = selected;
        const std::array<en::Gemma3Runner::Work, 2> alias{
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
            tokens[s] = ids[s][prefix[s] + 3 + i];
          }
          if (auto r = joined(tokens, true); !r) return r;
        }
        for (std::uint32_t s = 0; s < 2; ++s)
          for (std::uint32_t i = 0; i < kTail; ++i) {
            if (selected[s] != choices[kSteps * 2 + s * kTail + i])
              return Error("partial GPU/full-head choice differs");
            if (auto r =
                    independent(s, std::span(ids[s]).subspan(past[s], 1), true, i + 1 != kTail);
                !r)
              return r;
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
        std::array<std::optional<en::CheckpointFile>, 2> checkpoints;
        std::array<std::vector<en::LiveState::Range>, 2> footprints;
        const auto boundary = past;
        for (std::uint32_t s = 0; s < 2; ++s) {
          footprints[s] = (*runner.request_slot(s))->state().used_ranges();
          auto checkpoint = en::CheckpointFile::Capture(
              node, out, footprints[s],
              [&](void* host, std::span<const en::LiveState::Range> ranges) {
                return runner.CopyState(s, host, ranges);
              });
          if (!checkpoint) return Error(checkpoint.error().detail);
          checkpoints[s] = std::move(*checkpoint);
        }
        std::array<std::vector<float>, 2> continued;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
          for (std::uint32_t s = 0; s < 2; ++s) {
            if (auto x = independent(s, std::span(ids[s]).first(chunk), true, false); !x) return x;
            if (repeat == 0)
              continued[s] = heads[s];
            else if (heads[s].size() != continued[s].size() ||
                     std::memcmp(heads[s].data(), continued[s].data(), kVocab * 4))
              return Error("checkpoint continuation differs");
          }
          for (std::uint32_t s = 0; s < 2; ++s) {
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
        const auto record =
            "{\"slot0\":\"" + (*full_state)[0] + "\",\"slot1\":\"" + (*full_state)[1] +
            "\",\"gpu_equal\":true,\"restore_equal\":true,\"refusals_unchanged\":true," +
            "\"joined_prefill_groups\":" + std::to_string(prefill_groups) +
            ",\"joined_prefill_rows\":" + std::to_string(prefill_rows) + "}\n";
        if (auto r = Save<char>(out / "state.json", record); !r) return r;
        if (runner.greedy_tokens() != 78) return Error("C2 own GPU publication count differs");
      }
      if (cycle && runner.greedy_tokens() != 88)
        return Error("C2 cycle GPU publication count differs");
      if (auto r = Save<std::int32_t>(out / "chosen.i32", choices); !r) return r;
      std::vector<float> final;
      final.reserve(kVocab * 2);
      for (const auto& head : heads) final.insert(final.end(), head.begin(), head.end());
      if (auto r = Save<float>(out / "final.f32", final); !r) return r;
      const auto& stats = runner.graph_stats();
      const auto& bound = runner.plan_selections();
      if (!prefill_groups ||
          (owner_prefill ? !bound.owner_prefill_attention : !bound.packed_prefill_attention) ||
          !bound.owner_attention ||
          (device_masks ? !bound.device_masks : bound.device_masks != 0) ||
          (prefill_ahead && (!runner.lookahead_stats().cached ||
                             (chunk <= 128 && !runner.lookahead_stats().captured_ahead))) ||
          !stats.captured || !stats.replayed || !bound.norm_rope || !bound.norm_add ||
          runner.coverage().violations)
        return Error("C2 did not select/replay checked implementation families");
      if (bounded && prefix[0] != prefix[1] && !bound.bounded_owner_attention)
        return Error("bounded roots did not select actual unequal owner attention");
      std::cout
          << "GEMMA3_JOINT_PREFILL mode=" << mode
          << " slots=2 context_per_slot=4096 chunk=" << chunk
          << " compatible_prefill=1 max_wave_rows=" << 2 * chunk << " state_max_rows=" << state_rows
          << " local_cache_cells=" << runner.layout().local_cells << " stock_ring=" << stock_ring
          << " prompt_rows0=" << prefix[0] << " prompt_rows1=" << prefix[1]
          << " untimed_rows_per_slot=3"
          << " decode_steps=32 departure_steps=" << (own ? kTail : 0U)
          << " paid_generated_tokens=64 past0=" << past[0] << " past1=" << past[1]
          << " prefill_seconds=" << prefill << " decode_seconds=" << decode
          << " paid_prefill_eager=" << graph_prefill.eager - graph_before.eager
          << " paid_prefill_captured=" << graph_prefill.captured - graph_before.captured
          << " paid_prefill_replayed=" << graph_prefill.replayed - graph_before.replayed
          << " paid_prefill_built=" << ahead_prefill.built - ahead_before.built
          << " paid_prefill_cached=" << ahead_prefill.cached - ahead_before.cached
          << " paid_prefill_first=" << ahead_prefill.captured_first - ahead_before.captured_first
          << " paid_prefill_ahead=" << ahead_prefill.captured_ahead - ahead_before.captured_ahead
          << " eager=" << stats.eager << " captured=" << stats.captured
          << " replayed=" << stats.replayed << " selected_owner=" << bound.owner_attention
          << " selected_packed_prefill=" << bound.packed_prefill_attention
          << " owner_prefill=" << owner_prefill << " flexible_owner_prefill=" << flexible
          << " selected_owner_prefill=" << bound.owner_prefill_attention
          << " device_masks=" << device_masks << " selected_device_masks=" << bound.device_masks
          << " prefill_ahead=" << prefill_ahead
          << " lookahead_built=" << runner.lookahead_stats().built
          << " lookahead_cached=" << runner.lookahead_stats().cached
          << " lookahead_captured_ahead=" << runner.lookahead_stats().captured_ahead
          << " bounded_roots=" << bounded
          << " selected_bounded_owner=" << bound.bounded_owner_attention
          << " selected_norm_mul=" << bound.norm_mul
          << " selected_quant_geglu=" << bound.quant_geglu
          << " selected_norm_rope=" << bound.norm_rope << " selected_norm_add=" << bound.norm_add
          << " joined_prefill_groups=" << prefill_groups << " joined_prefill_rows=" << prefill_rows
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
  if (ran && retired) std::cout << "GEMMA3_JOINT_PREFILL_RETIRED\n";
  return ran && retired ? 0 : 1;
}
