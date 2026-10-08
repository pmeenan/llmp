// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Existing-model plain-token factor. Full initialized states and an actual
// full-row continuation are observed after the paid endpoint, in both arms.
// Usage: ARTIFACT IDS0.i32 IDS1.i32 NEW_OUT off|on [c1] [gguf]
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/qwen38_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"

namespace {
namespace en = jitllm::engine;
namespace fs = std::filesystem;
using en::support::Error;
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kSteps = 32;
constexpr std::uint64_t kCopy = 8ULL << 20U;
constexpr std::array<std::uint32_t, 2> kPrefix{1536, 1280};
template <class T>
en::Status Save(const fs::path& path, std::span<const T> data) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("exclusive complete output refused");
}
std::string Hash(std::span<const std::byte> bytes) {
  return jitllm::base::ToHex(jitllm::base::Sha256{}.Update(bytes).Finish());
}
std::int32_t Choose(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 6 || argc > 8 || !jitllm::platform::InstallCrashPolicy("plain-token-probe")) return 2;
  bool scalar = false, gguf = false;
  for (int i = 6; i < argc; ++i) {
    const std::string_view flag = argv[i];
    if (flag == "c1" && !scalar)
      scalar = true;
    else if (flag == "gguf" && !gguf)
      gguf = true;
    else
      return 2;
  }
  const std::uint32_t owners = scalar ? 1 : 2;
  const std::string_view mode = argv[5];
  if (mode != "off" && mode != "on") return 2;
  const bool device = mode == "on";
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  std::array<std::vector<std::int32_t>, 2> input;
  for (std::size_t i = 0; i < input.size(); ++i) {
    std::error_code ec;
    if (fs::file_size(argv[i + 2], ec) != kPrefix[i] * 4 || ec) return 2;
    input[i].resize(kPrefix[i]);
    std::ifstream file(argv[i + 2], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input[i].data()), kPrefix[i] * 4) ||
        file.peek() != std::char_traits<char>::eof())
      return 2;
    std::cout << "PLAIN_TOKEN_INPUT slot=" << i << " rows=" << kPrefix[i]
              << " sha256=" << Hash(std::as_bytes(std::span(input[i]))) << '\n';
  }
  if (input[0] == input[1]) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Qwen38Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  en::Qwen38Options options{.artifact = argv[1],
                            .out = out,
                            .context = 2048,
                            .max_rows = 512,
                            .drafter = {},
                            .wave_slots = owners,
                            .request_slots = owners,
                            .spill_place = {},
                            .device_tokens = device};
  life->runner = std::make_unique<en::Qwen38Runner>(node, options, 0, 0);
  auto& runner = *life->runner;
  std::array<en::Qwen38Runner::Slot*, 2> slots{};
  std::array<std::vector<std::int32_t>, 2> history;
  std::array<std::vector<float>, 2> heads;
  std::array<std::int32_t, 2> next{};
  void* pinned = nullptr;
  std::array<std::string, 2> states;
  std::array<std::uint64_t, 2> state_bytes{};
  const auto snapshot = [&]() -> en::Status {
    std::ofstream file(out / "state-ranges.txt", std::ios::noreplace);
    for (std::size_t i = 0; i < owners; ++i) {
      jitllm::base::Sha256 hash;
      for (const auto& range : slots[i]->used_state_ranges()) {
        if (range.bytes == 0) return Error("empty initialized state range");
        file << i << ' ' << range.region << ' ' << range.offset << ' ' << range.bytes << '\n';
        state_bytes[i] += range.bytes;
        for (std::uint64_t at = 0; at < range.bytes;) {
          auto part = range;
          part.offset += at;
          part.bytes = std::min(kCopy, range.bytes - at);
          if (auto copied = slots[i]->SaveUsedState(pinned, std::span(&part, 1)); !copied) {
            node.KeepPinned(pinned);
            return copied;
          }
          hash.Update(std::span(static_cast<const std::byte*>(pinned), std::size_t{part.bytes}));
          at += part.bytes;
        }
      }
      if (state_bytes[i] == 0) return Error("initialized target state required");
      states[i] = jitllm::base::ToHex(hash.Finish());
    }
    file.flush();
    return file ? en::Status{} : Error("state range persistence failed");
  };
  const auto traversal = [&](double& prefill, double& decode) -> en::Status {
    const auto begin = Clock::now();
    for (std::size_t i = 0; i < owners; ++i) {
      if (auto r = slots[i]->Clear(); !r) return r;
      history[i].assign(input[i].begin(), input[i].end());
      std::uint32_t past = 0;
      while (past < kPrefix[i]) {
        const auto rows = std::min(512U, kPrefix[i] - past);
        if (auto r = slots[i]->Chunk(std::span(history[i]).first(past + rows), past, heads[i]); !r)
          return r;
        past += rows;
      }
      if (heads[i].size() != runner.vocab()) return Error("complete prefill head required");
      history[i].push_back(Choose(heads[i]));
    }
    const auto prefill_end = Clock::now();
    for (std::uint32_t step = 0; step < kSteps; ++step) {
      if (scalar) {
        if (device) {
          if (auto r = slots[0]->GreedyChunk(history[0], kPrefix[0] + step, next[0]); !r) return r;
        } else {
          if (auto r = slots[0]->Chunk(history[0], kPrefix[0] + step, heads[0]); !r) return r;
          if (heads[0].size() != runner.vocab())
            return Error("complete scalar decode head required");
          next[0] = Choose(heads[0]);
        }
        history[0].push_back(next[0]);
        continue;
      }
      std::array<en::Qwen38Runner::ChunkWork, 2> work{};
      for (std::size_t i = 0; i < owners; ++i)
        work[i] = {.slot = slots[i],
                   .history = history[i],
                   .n_past = kPrefix[i] + step,
                   .logits = device ? nullptr : &heads[i],
                   .token = device ? &next[i] : nullptr};
      if (auto r = runner.ChunkWave(work); !r) return r;
      for (std::size_t i = 0; i < owners; ++i) {
        if (!device) {
          if (heads[i].size() != runner.vocab()) return Error("complete decode head required");
          next[i] = Choose(heads[i]);
        }
        history[i].push_back(next[i]);
      }
    }
    const auto end = Clock::now();
    prefill = en::support::Seconds(prefill_end - begin);
    decode = en::support::Seconds(end - prefill_end);
    return {};
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (runner.speculative() || runner.wave_capacity() != owners || runner.vocab() != 248320)
      return Error("target-only plain envelope required");
    for (std::uint32_t i = 0; i < owners; ++i) {
      auto slot = runner.request_slot(i);
      if (!slot) return Error(slot.error());
      slots[i] = *slot;
      if (!std::ranges::all_of(
              input[i], [&](auto id) { return id >= 0 && std::cmp_less(id, runner.vocab()); }))
        return Error("input IDs exceed the vocabulary");
    }
    std::vector<jitllm::catalog::ExtentId> extents;
    auto allocation = node.Pinned(kCopy, 0, extents);
    if (!allocation) return Error(allocation.error());
    pinned = *allocation;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (64ULL << 20U));
    for (std::size_t i = 0; i < owners; ++i) history[i].reserve(kPrefix[i] + kSteps + 2);
    const auto budget =
        fixed + runner.weights().size() * en::kPagedExtent + 4 * node.StateCapacity();
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    std::vector<en::LoadStats> loads;
    if (auto r = node.Load(runner.weights(), "plain-token factor weights", loads); !r) return r;
    if (auto r = runner.ReadPleHash(); !r) return r;
    if (auto r = runner.SelectSlots(std::span(slots).first(owners)); !r) return r;
    const auto setup = runner.setup_budget();
    std::cout << "PLAIN_TOKEN_SETUP budget=" << budget << " fixed=" << fixed
              << " activations=" << setup.activations << " scratch=" << setup.scratch
              << " pinned=" << setup.pinned << " host_inputs=" << setup.host_inputs
              << " wave_output_pinned=" << setup.wave_output_pinned << '\n';
    return node.WithRequest(
        0, runner.execution_closure(), "plain-token factor", [&]() -> en::Status {
          double warm_prefill = 0, warm_decode = 0;
          if (auto r = traversal(warm_prefill, warm_decode); !r) return r;
          const auto before = runner.graph_stats();
          const auto before_tokens = runner.device_token_outputs();
          double prefill = 0, decode = 0;
          if (auto r = traversal(prefill, decode); !r) return r;
          const auto after = runner.graph_stats();
          const auto selected_tokens = runner.device_token_outputs() - before_tokens;
          const auto paired = runner.last_wave();
          const auto completed = after.eager - before.eager + after.captured - before.captured +
                                 after.replayed - before.replayed;
          if (selected_tokens != (device ? owners * kSteps : 0) ||
              after.replayed <= before.replayed || before.captured == 0 ||
              completed != 3 * owners + kSteps ||
              (!scalar && (paired.paired_slots != 0b11 ||
                           (gguf ? paired.vecq_pairs == 0 : paired.mxfp8_pairs == 0))) ||
              runner.coverage_violations() != 0 || prefill <= 0 || decode <= 0 ||
              !std::isfinite(prefill) || !std::isfinite(decode))
            return Error(
                "actual selected token, capture/replay, pairing or coverage witness refused");
          if (auto r = snapshot(); !r) return r;
          for (std::size_t i = 0; i < owners; ++i) {
            if (auto r =
                    Save<std::int32_t>(out / ("history" + std::to_string(i) + ".i32"), history[i]);
                !r)
              return r;
            // Consume the identical next anchor as an actual row-producing continuation.
            // This observation and its state mutation occur after the paid state snapshot.
            if (auto r = slots[i]->Chunk(history[i], kPrefix[i] + kSteps, heads[i]); !r) return r;
            if (heads[i].size() != runner.vocab() ||
                !std::ranges::all_of(heads[i], [](float value) { return std::isfinite(value); }))
              return Error("finite complete continuation head required");
            if (auto r = Save<float>(out / ("head" + std::to_string(i) + ".f32"), heads[i]); !r)
              return r;
          }
          std::cout << std::setprecision(17) << "PLAIN_TOKEN_FACTOR mode=" << mode
                    << " context=2048 chunk=512 slots=" << owners
                    << " format=" << (gguf ? "gguf" : "native") << " decode_units=" << kSteps
                    << " device_tokens=" << selected_tokens << " prefill_seconds=" << prefill
                    << " decode_seconds=" << decode << " paid_seconds=" << prefill + decode
                    << " warm_prefill_seconds=" << warm_prefill
                    << " warm_decode_seconds=" << warm_decode
                    << " eager=" << after.eager - before.eager
                    << " captured=" << after.captured - before.captured
                    << " replayed=" << after.replayed - before.replayed
                    << " mxfp8_pairs=" << paired.mxfp8_pairs << " vecq_pairs=" << paired.vecq_pairs;
          for (std::size_t i = 0; i < owners; ++i)
            std::cout << " state" << i << "_bytes=" << state_bytes[i] << " state" << i << '='
                      << states[i];
          std::cout << '\n';
          return {};
        });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(life->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = life.release();  // Unknown work retains every native owner.
  }
  if (ran && retired) std::cout << "PLAIN_TOKEN_FACTOR_RETIRED\n";
  return ran && retired ? 0 : 1;
}
