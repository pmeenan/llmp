// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Separate fixed-policy C4 owner-root factor; historical packed helper is unchanged.
// ARTIFACT OUTPUT_DIR 26|31 4 joined norm|compound IDS_I32 [8k]
// Bounded common-prefix units: every completed owner head is published/charged.
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
using en::support::Error;
int main(int argc, char** argv) {
  if ((argc != 8 && argc != 9) || std::string_view(argv[4]) != "4" ||
      std::string_view(argv[5]) != "joined" ||
      !((std::string_view(argv[3]) == "31" && std::string_view(argv[6]) == "norm") ||
        (std::string_view(argv[3]) == "26" && std::string_view(argv[6]) == "compound")))
    return 2;
  if (argc == 9 && std::string_view(argv[8]) != "8k") return 2;
  const bool long_context = argc == 9;
  const std::uint32_t context = long_context ? 16384 : 256;
  const std::size_t input_rows = long_context ? 8227 : 1024;
  const char* factor = std::getenv("JITLLM_GEMMA_OWNER_C4");
  if (!factor || (std::string_view(factor) != "packed" && std::string_view(factor) != "owners"))
    return 2;
  const char* normmul_option = std::getenv("JITLLM_GEMMA_C4_NORMMUL");
  if (normmul_option && std::string_view(normmul_option) != "0" &&
      std::string_view(normmul_option) != "1")
    return 2;
  // Preserve the dense31 default; the closed26 compound transfer defaults on.
  const bool compound = std::string_view(argv[3]) == "26";
  const bool normmul = normmul_option ? std::string_view(normmul_option) == "1" : compound;
  const char* phases_option = std::getenv("JITLLM_GEMMA_C4_PHASES");
  if (phases_option && std::string_view(phases_option) != "0" &&
      std::string_view(phases_option) != "1")
    return 2;
  const bool account_phases = phases_option && std::string_view(phases_option) == "1";
  std::vector<std::int32_t> supplied;
  {
    supplied.resize(input_rows);
    std::ifstream file(argv[7], std::ios::binary);
    file.read(reinterpret_cast<char*>(supplied.data()),
              static_cast<std::streamsize>(input_rows * sizeof(std::int32_t)));
    if (!file || file.peek() != std::char_traits<char>::eof() || supplied[0] != 2 ||
        !std::ranges::all_of(supplied, [](auto id) { return id >= 0 && id < 262144; }))
      return 2;
  }
  jitllm::base::Sha256 input_hash;
  input_hash.Update(std::as_bytes(std::span(supplied)));
  if (jitllm::base::ToHex(input_hash.Finish()) !=
      (long_context ? "6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b"
                    : "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"))
    return 2;
  const std::uint32_t prompt_rows = long_context ? 8188 : 64;
  const std::string_view variant = argv[3], mode = argv[5], policy = argv[6];
  std::uint32_t count = 0;
  const std::string_view number = argv[4];
  const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), count);
  if (error != std::errc{} || end != number.data() + number.size() || count == 0 || count > 12 ||
      (variant != "26" && variant != "31") || (mode != "scalar" && mode != "joined") ||
      (policy != "norm" && policy != "compound"))
    return 2;
  if (count != 4 || mode != "joined" || supplied.empty() ||
      (compound ? policy != "compound" : policy != "norm"))
    return 2;
  umask(0077);
  const std::filesystem::path out = argv[2];
  std::error_code file_error;
  if (!std::filesystem::create_directory(out, file_error) || file_error) return 2;
  if (!supplied.empty()) {
    std::ofstream file(out / "inputs.i32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(supplied.data()),
               static_cast<std::streamsize>(input_rows * sizeof(std::int32_t)));
    file.flush();
    if (!file) return 2;
  }
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto& node = lifetime->node;
  lifetime->runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{
          .artifact = argv[1],
          .out = out,
          .variant = variant == "26" ? en::Gemma4Variant::k26BA4B : en::Gemma4Variant::k31B,
          .context = context,
          .max_rows = 128,
          .slots = count,
          .fuse_norms = normmul,
          .row_invariant = policy == "rows" || policy == "rows-norm",
          .fuse_norm_rope = true,
          .fuse_norm_add = true,
          .fuse_gemma_route = compound,
          .fuse_gemma_reduce = compound,
          .owner_attention = std::string_view(factor) == "owners"},
      0, 0);
  auto& runner = *lifetime->runner;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    lifetime->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint32_t steps = 32, vocab = 262144;
    // One reusable node-owned pinned witness, funded before admission.
    // Each successful CopyState proves completion before this buffer is reused.
    const auto largest_positions = prompt_rows + count - 1 + 3 + steps;
    auto witness_ranges = runner.CheckpointRanges(largest_positions);
    if (!witness_ranges) return Error(witness_ranges.error());
    std::uint64_t witness_bytes = 0;
    for (const auto& range : *witness_ranges) witness_bytes += range.bytes;
    std::vector<jitllm::catalog::ExtentId> witness_extents;
    auto witness = node.Pinned(witness_bytes, 0, witness_extents);
    if (!witness) return Error(witness.error());
    const auto heap = std::uint64_t{count} * (steps + 2) * vocab * sizeof(float);
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto weight_count = runner.weights().size();
    auto budget = fixed + weight_count * en::kPagedExtent + 2 * node.StateCapacity() + heap;
    auto host_floor = runner.plan_floor_bytes() + runner.host_input_bytes() + heap;
    if (long_context && compound) {
      // Cache identity excludes n_past: two 128-row chunks share each 256-cell
      // bucket. Four slots each have 32 state-only buckets and one final tail;
      // the C4 decode has five width vectors across the 8192/8448 boundary.
      // Startup probes are temporary. This grants capacity, not occupancy.
      const std::uint64_t scalar_buckets = (prompt_rows + count - 1 + 255) / 256;
      const std::uint64_t unique_keys = count * (scalar_buckets + 1) + count + 1;
      constexpr auto ratio = en::kGraphNodeHostBytes / en::kPlanNodeHostBytes;
      constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
      const auto floor = runner.plan_floor_bytes();
      if (unique_keys != 137 || floor > maximum / unique_keys / (1 + ratio) ||
          weight_count > maximum / en::kPagedExtent || node.StateCapacity() > maximum / 2)
        return Error("long26 plan/graph budget overflow");
      const auto retention = unique_keys * floor * (1 + ratio);
      const auto scratch = en::ScratchArenaBytes();
      budget = 0;
      for (const auto bytes : {fixed, weight_count * en::kPagedExtent, 2 * node.StateCapacity(),
                               heap, scratch, retention}) {
        if (bytes > maximum - budget) return Error("long26 execution budget overflow");
        budget += bytes;
      }
      host_floor = 0;
      for (const auto bytes : {floor, runner.host_input_bytes(), heap, scratch}) {
        if (bytes > maximum - host_floor) return Error("long26 host floor overflow");
        host_floor += bytes;
      }
      std::cout << "OWNER_C4_BUDGET fixed=" << fixed
                << " weights=" << weight_count * en::kPagedExtent
                << " state_capacity=" << node.StateCapacity() << " publication=" << heap
                << " unique_key_bound=" << unique_keys << " plan_floor=" << floor
                << " planning_scratch=" << scratch << " plan_graph_capacity=" << retention
                << " host_floor=" << host_floor << " total=" << budget << '\n';
      // Reject an oversized capacity envelope for this 128 GB manual target.
      if (budget > 128'000'000'000ULL)
        return Error("long26 capacity grant exceeds the Spark physical budget");
    }
    node.SetHostFloor(host_floor);
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    std::array<std::uint32_t, 12> ids{};
    for (std::uint32_t i = 0; i < count; ++i) ids[i] = i;
    if (auto r = runner.SelectSlots(std::span(ids).first(count)); !r) return r;
    if (!node.ChargeHost(heap, false)) return Error("owner publication buffers do not fit");
    struct HostGrant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~HostGrant() { node.UnchargeHost(bytes); }
    } grant{node, heap};
    return node.WithRequest(0, runner.closure(), "Gemma joined fixed-prefix screen", [&]() {
      return [&]() -> en::Status {
        std::array<std::vector<float>, 12> heads;
        std::array<std::uint32_t, 12> past{};
        std::array<std::int32_t, 12> anchors{};
        constexpr std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
        constexpr std::array<std::int32_t, 3> seeds{45518, 107, 101};
        std::vector<float> published(std::size_t{steps} * count * vocab);
        std::array<std::array<std::int32_t, 12>, steps> chosen{};
        const auto report_policy = [&](std::string_view phase, std::uint32_t owner,
                                       std::uint32_t rows, std::uint32_t segments) -> en::Status {
          const auto& p = runner.last_built_policy();
          // Compound26 must hold in scalar prefill as well as C4 decode;
          // retain the existing31 check at its C4 first-built plan.
          if ((compound || (rows == 4 && segments == 4)) &&
              (p.norm_rope != (compound ? 60U : 120U) || p.norm_add != (compound ? 90U : 120U) ||
               p.row_products != 0 || p.norm_fused != (normmul ? 121 : 0) || p.rope_store != 0 ||
               p.shared_vecq != 0 || p.gemma_route != (compound ? 30U : 0U) ||
               p.gemma_reduce != (compound ? 30U : 0U)))
            return Error("owner screen prefill/decode policy counts differ");
          if (p.rows != rows || p.segments != segments)
            return Error("fresh selected policy does not match requested shape");
          std::cout << "JOINED_SELECTED phase=" << phase << " owner=" << owner << " rows=" << p.rows
                    << " segments=" << p.segments << " row_products=" << p.row_products
                    << " norm_rope=" << p.norm_rope << " norm_add=" << p.norm_add
                    << " norm_fused=" << p.norm_fused << " rope_store=" << p.rope_store
                    << " shared_vecq=" << p.shared_vecq << " gemma_route=" << p.gemma_route
                    << " gemma_reduce=" << p.gemma_reduce << " lane_steps=" << p.lane_steps << '\n';
          return {};
        };
        const auto prefill = [&](bool fresh) -> en::Status {
          for (std::uint32_t i = 0; i < count; ++i) {
            if (auto r = runner.Clear(i); !r) return r;
            std::vector<std::int32_t> tokens;
            if (supplied.empty()) {
              tokens.assign(prompt.begin(), prompt.end());
              tokens.insert(tokens.end(), i, 563);
            } else {
              tokens.assign(supplied.begin(), supplied.begin() + prompt_rows + i);
            }
            past[i] = static_cast<std::uint32_t>(tokens.size());
            std::uint32_t at = 0;
            while (at < tokens.size()) {
              const auto rows =
                  std::min<std::uint32_t>(128, static_cast<std::uint32_t>(tokens.size()) - at);
              const bool final = at + rows == tokens.size();
              const en::Gemma4Runner::Work work{i, at, std::span(tokens).subspan(at, rows),
                                                &heads[i]};
              if (long_context) {
                if (auto r = runner.WavePrefill(std::span(&work, 1), final); !r) return r;
              } else if (auto r = runner.Wave(std::span(&work, 1)); !r)
                return r;
              if (fresh && final)
                if (auto r = report_policy("prefill-first-build", i, rows, 1); !r) return r;
              at += rows;
            }
          }
          return {};
        };
        bool fresh_decode = true;
        const auto wave = [&](std::uint32_t step) -> en::Status {
          std::array<en::Gemma4Runner::Work, 12> work{};
          for (std::uint32_t i = 0; i < count; ++i) {
            anchors[i] = supplied.empty() ? seeds[(step + i) % seeds.size()]
                                          : supplied[prompt_rows + i + step];
            work[i] = {i, past[i], std::span(&anchors[i], 1), &heads[i]};
          }
          if (mode == "joined") {
            for (std::uint32_t first = 0; first < count; first += en::kGemma4InvariantWaveRows) {
              const auto rows = std::min(en::kGemma4InvariantWaveRows, count - first);
              if (auto r = runner.Wave(std::span(work).subspan(first, rows)); !r) return r;
              if (fresh_decode)
                if (auto r = report_policy("decode-first-build", first, rows, rows); !r) return r;
            }
          } else {
            for (std::uint32_t i = 0; i < count; ++i) {
              if (auto r = runner.Wave(std::span(work).subspan(i, 1)); !r) return r;
              if (fresh_decode)
                if (auto r = report_policy("decode-first-build", i, 1, 1); !r) return r;
            }
          }
          fresh_decode = false;
          for (std::uint32_t i = 0; i < count; ++i) ++past[i];
          return {};
        };
        if (auto r = prefill(true); !r) return r;
        for (std::uint32_t step = 0; step < 8; ++step)
          if (auto r = wave(step); !r) return r;
        if (auto r = prefill(false); !r) return r;
        for (std::uint32_t step = 0; step < 3; ++step)
          if (auto r = wave(step); !r) return r;
        const auto before = runner.graph_stats();
        if (account_phases) {
          runner.EnablePhaseAccounting();
          (void)runner.TakePhaseAccounting();
          (void)node.TakeTimes(0);
        }
        const auto started = std::chrono::steady_clock::now();
        for (std::uint32_t step = 0; step < steps; ++step) {
          if (auto r = wave(step + 3); !r) return r;
          for (std::uint32_t i = 0; i < count; ++i) {
            if (heads[i].size() != vocab) return Error("incomplete owner head publication");
            chosen[step][i] = static_cast<std::int32_t>(
                std::max_element(heads[i].begin(), heads[i].end()) - heads[i].begin());
            std::copy(heads[i].begin(), heads[i].end(),
                      published.data() + (std::size_t{step} * count + i) * vocab);
          }
        }
        const auto elapsed = en::support::Seconds(std::chrono::steady_clock::now() - started);
        if (account_phases) {
          // Only the paid waves: exclude warm-up, finite scans and state copies.
          // Bind/coverage/cache are nested; stream time includes submission gaps.
          const auto phases = runner.TakePhaseAccounting();
          const auto jobs = node.TakeTimes(0);
          std::cout << "C4_PHASE planned_calls=" << phases.planned_calls << " hits=" << phases.hits
                    << " misses=" << phases.misses;
          constexpr std::array names{"checks",
                                     "inputs",
                                     "state",
                                     "required_planning",
                                     "staging",
                                     "execution",
                                     "publication_cleanup"};
          for (std::size_t i = 0; i < names.size(); ++i)
            std::cout << ' ' << names[i] << "_seconds=" << phases.seconds[i];
          std::cout << " bind_seconds=" << phases.bind_seconds
                    << " coverage_seconds=" << phases.coverage_seconds
                    << " cache_seconds=" << phases.cache_seconds << " job_steps=" << jobs.steps
                    << " job_wall=" << jobs.wall << " dispatch=" << jobs.dispatch
                    << " submission=" << jobs.job << " after=" << jobs.after
                    << " stream_elapsed=" << jobs.device << '\n';
        }
        // Whole-array finite admission is deliberately OUTSIDE the paid interval,
        // matching the unchanged original client. Publication/argmax were paid.
        if (!std::ranges::all_of(published, [](float x) { return std::isfinite(x); }))
          return Error("nonfinite completed full head");
        for (std::uint32_t owner = 0; owner < count; ++owner) {
          auto slot = runner.request_slot(owner);
          if (!slot || (*slot)->completed_positions() != past[owner])
            return Error("completed owner state endpoint mismatch");
          auto ranges = runner.CheckpointRanges(past[owner]);
          if (!ranges) return Error(ranges.error());
          std::uint64_t bytes = 0;
          for (const auto& range : *ranges) bytes += range.bytes;
          if (bytes > witness_bytes) return Error("state witness funding envelope");
          if (auto copied = runner.CopyState(owner, *witness, *ranges, true); !copied)
            return copied;
          std::ofstream state(out / std::format("owner-{}.state", owner), std::ios::binary);
          state.write(static_cast<const char*>(*witness), static_cast<std::streamsize>(bytes));
          state.flush();
          if (!state) return Error("writing initialized state witness failed");
          std::cout << "PACKED_STATE owner=" << owner << " positions=" << past[owner]
                    << " bytes=" << bytes << " ranges=" << ranges->size()
                    << " layout=" << runner.CheckpointLayoutId() << '\n';
        }
        if (auto freed = node.FreePinned(*witness); !freed) return freed;
        std::ofstream file(out / "heads.f32", std::ios::binary);
        file.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(published.size() * sizeof(float)));
        file.flush();
        if (!file) return Error("writing completed owner heads failed");
        const auto& p = runner.last_built_policy();
        std::cout << "JOINED_NATIVE variant=" << variant << " owners=" << count << " mode=" << mode
                  << " policy=" << policy
                  << " native_owner_optin=" << (std::string_view(factor) == "owners")
                  << " normmul=" << (normmul ? "on" : "off") << " seconds=" << elapsed
                  << " completed_waves=" << steps << " completed_units=" << steps * count
                  << " paid_gpu_groups="
                  << steps * (mode == "scalar" ? count
                                               : (count + en::kGemma4InvariantWaveRows - 1) /
                                                     en::kGemma4InvariantWaveRows)
                  << " max_shared_rows=" << en::kGemma4InvariantWaveRows
                  << " first_past=" << prompt_rows + 3
                  << " input_mode=" << (supplied.empty() ? "synthetic" : "supplied")
                  << " unequal_past=" << (count > 1) << " context=" << context << " max_rows=128"
                  << " captured_delta=" << runner.graph_stats().captured - before.captured
                  << " replayed_delta=" << runner.graph_stats().replayed - before.replayed
                  << " rows=" << p.rows << " segments=" << p.segments
                  << " row_products=" << p.row_products << " norm_rope=" << p.norm_rope
                  << " norm_add=" << p.norm_add << " norm_fused=" << p.norm_fused
                  << " rope_store=" << p.rope_store << " shared_vecq=" << p.shared_vecq
                  << " gemma_route=" << p.gemma_route << " gemma_reduce=" << p.gemma_reduce
                  << " policy_basis=last-built lane_steps=" << p.lane_steps
                  << " heap_funded=" << heap << '\n';
        for (std::uint32_t step = 0; step < steps; ++step)
          for (std::uint32_t i = 0; i < count; ++i)
            std::cout << "JOINED_TOKEN step=" << step << " owner=" << i
                      << " argmax=" << chosen[step][i] << " forced="
                      << (supplied.empty() ? seeds[(step + 3 + i) % seeds.size()]
                                           : supplied[prompt_rows + i + step + 3])
                      << '\n';
        return {};
      }();
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(lifetime->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = lifetime.release();
  }
  return ran && retired ? 0 : 1;
}
