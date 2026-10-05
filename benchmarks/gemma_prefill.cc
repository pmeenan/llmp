// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// First paid 8K prefill and fixed-prefix decode screen. No quality claim.
// ARTIFACT IDS_I32 NEW_OUTPUT_DIR [26|31] [ordinary|both]. Defaults remain off.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
using en::support::Error;
int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) return 2;
  const std::string_view variant = argc >= 5 ? argv[4] : "26";
  const std::string_view policy = argc == 6 ? argv[5] : "ordinary";
  if (variant != "26" && variant != "31") return 2;
  if (policy != "ordinary" && policy != "both") return 2;
  constexpr std::uint32_t kPrefill = 8192, kWarm = 3, kSteps = 32;
  constexpr std::uint32_t kInput = kPrefill + kWarm + kSteps, kVocab = 262144;
  std::error_code error;
  if (std::filesystem::file_size(argv[2], error) != kInput * 4 || error) return 2;
  std::array<std::int32_t, kInput> ids{};
  std::ifstream file(argv[2], std::ios::binary);
  file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
  if (!file || ids.front() != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < 262144; }))
    return 2;
  const std::filesystem::path out(argv[3]);
  if (!std::filesystem::create_directory(out, error) || error) return 2;
  en::PagedNode node({.slot_bytes = en::kSlabSlotBytes});
  en::Gemma4Runner runner(
      node,
      {.artifact = argv[1],
       .out = out,
       .variant = variant == "31" ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
       .context = 16384,
       .fuse_norm_rope = policy == "both",
       .fuse_norm_add = policy == "both"},
      0, 0);
  std::vector<en::PagedModel*> entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint64_t kOutputBytes = kVocab * 4;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    // Every call can introduce at most one plan and one captured graph.
    // PlannedHostBytes includes kPlanNodeHostBytes per launched node, so
    // floor/kPlanNodeHostBytes bounds its graph's catalog charge. This
    // derives a conservative retention budget from the exact call count,
    // rather than consuming state funding with accumulated plan charges.
    constexpr std::uint64_t kCalls = 1 + kPrefill / 128 + kWarm + kSteps;
    constexpr auto kGraphRatio = en::kGraphNodeHostBytes / en::kPlanNodeHostBytes;
    const auto floor = runner.plan_floor_bytes();
    if (floor > std::numeric_limits<std::uint64_t>::max() / kCalls / (1 + kGraphRatio))
      return Error("plan/graph budget overflow");
    const auto retention = kCalls * floor * (1 + kGraphRatio);
    const auto base = fixed + runner.weights().size() * en::kPagedExtent +
                      2 * node.StateCapacity() + kOutputBytes;
    if (retention > std::numeric_limits<std::uint64_t>::max() - base)
      return Error("execution budget overflow");
    const auto budget = base + retention;
    std::cout << "PREFILL_BUDGET fixed=" << fixed
              << " weights=" << runner.weights().size() * en::kPagedExtent
              << " state_capacity=" << node.StateCapacity() << " publication=" << kOutputBytes
              << " call_bound=" << kCalls << " plan_floor=" << floor
              << " plan_graph_capacity=" << retention << " total=" << budget << '\n';
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes());
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(0, runner.closure(), "Gemma4 paid 8K screen", [&]() -> en::Status {
      if (!node.ChargeHost(kOutputBytes, false)) return Error("publication capacity refused");
      struct Charge {
        en::PagedNode& node;
        std::uint64_t bytes;
        ~Charge() { node.UnchargeHost(bytes); }
      } charged{node, kOutputBytes};
      std::vector<float> logits;
      logits.reserve(kVocab);
      if (logits.capacity() != kVocab) return Error("unexpected publication vector capacity");
      // Page all weights before the paid fresh request, matching the already
      // loaded comparator. These six rows are discarded on both engines.
      if (auto r = runner.Chunk(0, std::span(ids).first(6), logits); !r) return r;
      if (auto r = runner.Clear(); !r) return r;
      const auto started = std::chrono::steady_clock::now();
      for (std::uint32_t first = 0; first < kPrefill; first += 128) {
        if (auto r = runner.Chunk(first, std::span(ids).subspan(first, 128), logits); !r) return r;
      }
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - started);
      if (logits.size() != kVocab) return Error("missing final prefill head");
      const auto save = [&](const char* name) -> en::Status {
        std::ofstream output(out / name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(logits.data()), kOutputBytes);
        output.flush();
        return output ? en::Status{} : en::Status(Error("writing completed head failed"));
      };
      if (auto r = save("prefill.f32"); !r) return r;
      for (std::uint32_t i = 0; i < kWarm; ++i) {
        const auto past = kPrefill + i;
        if (auto r = runner.Chunk(past, std::span(ids).subspan(past, 1), logits); !r) return r;
        std::cout << "PREFILL_TIMED_PREFIX appended=" << ids[past] << " past=" << past + 1 << '\n';
      }
      std::array<std::int32_t, kSteps> chosen{};
      const auto decode_started = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < kSteps; ++i) {
        chosen[i] = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                              logits.begin());
        const auto past = kPrefill + kWarm + i;
        if (auto r = runner.Chunk(past, std::span(ids).subspan(past, 1), logits); !r) return r;
      }
      const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_started);
      if (auto r = save("final.f32"); !r) return r;
      auto slot = runner.request_slot(0);
      if (!slot || (*slot)->completed_positions() != kInput)
        return Error("completed positions differ from paid work");
      const auto& policy = runner.last_built_policy();
      std::cout << "PREFILL_NATIVE prefill_seconds=" << prefill << " prefill_rows=" << kPrefill
                << " prefill_chunks=64 intermediate_heads=63 decode_seconds=" << decode
                << " decode_chunks=" << kSteps << " timed_start=" << kPrefill + kWarm
                << " completed=" << kInput << " context=16384 chunk=128 masks=device"
                << " shared_vecq=" << policy.shared_vecq << " row_products=" << policy.row_products
                << " norm_fused=" << policy.norm_fused << " rope_store=" << policy.rope_store
                << " norm_rope=" << policy.norm_rope << " norm_add=" << policy.norm_add
                << " captures=" << runner.graph_stats().captured
                << " replays=" << runner.graph_stats().replayed << '\n';
      for (std::uint32_t i = 0; i < kSteps; ++i)
        std::cout << "PREFILL_TOKEN step=" << i << " argmax=" << chosen[i]
                  << " forced=" << ids[kPrefill + kWarm + i] << '\n';
      return {};
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) std::cerr << retired.error() << '\n';
  return ran && retired ? 0 : 1;
}
