// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Actual runner startup only: no Register, Start, Load, binding or model dispatch.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>

#include "engine/dsv4_runner.h"
#include "engine/paged_node.h"
#include "engine/planned.h"
#include "engine/qwen38_runner.h"

namespace {
namespace en = jitllm::engine;

std::string Stats(const en::StartupPlacementStats& s) {
  return std::format(
      R"({{"plans":{},"nodes":{},"bounded":{},"exact":{},"max_activations":{},"max_inputs":{},"max_host":{},"max_nodes":{}}})",
      s.plans, s.nodes, s.bounded, s.exact, s.max_activations, s.max_inputs, s.max_host,
      s.max_nodes);
}

std::string Budget(const en::Qwen38SetupBudget& b) {
  std::string out = "{";
#define BUDGET_FIELD(name) out += std::format("\"" #name "\":{},", b.name)
  BUDGET_FIELD(scalar_activations);
  BUDGET_FIELD(scalar_scratch);
  BUDGET_FIELD(scalar_staging_inputs);
  BUDGET_FIELD(scalar_host_inputs);
  BUDGET_FIELD(scalar_ple_mapped);
  BUDGET_FIELD(scalar_pinned);
  BUDGET_FIELD(scalar_snapshot_per_branch);
  BUDGET_FIELD(activations);
  BUDGET_FIELD(scratch);
  BUDGET_FIELD(staging_inputs);
  BUDGET_FIELD(host_inputs);
  BUDGET_FIELD(ple_mapped);
  BUDGET_FIELD(pinned);
  BUDGET_FIELD(wave_output_pinned);
  BUDGET_FIELD(snapshot_per_branch);
  BUDGET_FIELD(runner_mapped);
  BUDGET_FIELD(target_per_branch);
  BUDGET_FIELD(drafter_per_branch);
  BUDGET_FIELD(virtual_per_branch);
  BUDGET_FIELD(initialized_logical);
  BUDGET_FIELD(initialized_extent_bytes);
#undef BUDGET_FIELD
  out.back() = '}';
  return out;
}

template <typename Runner>
struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  Runner runner;
  template <typename Options>
  explicit Lifetime(const Options& options) : runner(node, options, 0, 0) {}
};

template <typename Runner>
int Measure(std::unique_ptr<Lifetime<Runner>> owner, const std::filesystem::path& result,
            std::string_view recipe, bool threshold) {
  auto& node = owner->node;
  auto& runner = owner->runner;
  if (auto opened = node.Open(); !opened) {
    std::println(stderr, "node open failed: {}", opened.error());
    // A partial provider open has no positive retirement proof either.
    std::ignore = owner.release();
    return 1;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto setup = runner.Setup();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::string budget;
  std::string scalar;
  std::string wave = "null";
  if (setup) {
    if constexpr (std::is_same_v<Runner, en::Dsv4Runner>) {
      budget = std::format(
          R"({{"activations":{},"scratch":{},"lane_scratch":{},"host_inputs":{},"plan_floor":{},"scratch_arena":{},"initialized_logical":{}}})",
          runner.activations_needed(), runner.pool_needed(), runner.lane_pool_needed(),
          runner.host_input_bytes(), runner.plan_floor_bytes(), en::ScratchArenaBytes(),
          runner.used_state_bytes());
      scalar = Stats(runner.startup_placement_stats());
      std::println("plan budget: {}", runner.plan_report());
    } else {
      budget = Budget(runner.setup_budget());
      // The host graph-index allowance and node-based plan floor are separate
      // from the setup budget's device and staging bounds.
      budget.pop_back();
      budget += std::format(",\"scratch_arena\":{},\"plan_floor\":{}}}", en::ScratchArenaBytes(),
                            runner.plan_floor_bytes());
      scalar = Stats(runner.scalar_startup_placement_stats());
      wave = Stats(runner.wave_startup_placement_stats());
      std::println("plan budget: {}", runner.plan_report());
    }
  } else {
    std::println(stderr, "setup failed: {}", setup.error());
  }
  // Setup opens the cuBLAS context and allocates runner resources. Its teardown
  // still requires the node's explicit stream completion and release contract.
  const std::array<en::PagedModel*, 1> entered = {&runner};
  const auto retired = node.TearDown(entered);
  if (!retired) {
    std::println(stderr, "teardown failed: {}", retired.error());
    // Retain every CUDA/provider/resource owner until process exit; destructors
    // cannot establish completion or safely free backing after this failure.
    std::ignore = owner.release();
    return 1;
  }
  if (!setup) return 1;
  std::ofstream out(result, std::ios::out | std::ios::noreplace);
  out << std::format(
             R"({{"recipe":"{}","threshold":{},"setup_seconds":{:.9f},"budget":{},"scalar":{},"wave":{},"retired":true}})",
             recipe, threshold, seconds, budget, scalar, wave)
      << '\n';
  out.close();
  if (!out) return 1;
  std::println("STARTUP_MEASUREMENT_RETIRED recipe={} threshold={}", recipe, threshold);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::println(stderr,
                 "usage: jitllm_startup_measurement_probe ds|dspark|qn|qmtp|qg TARGET DRAFTER|- "
                 "NEW_OUT off|on");
    return 2;
  }
  const std::string_view recipe = argv[1];
  const std::string_view policy = argv[5];
  if ((recipe != "ds" && recipe != "dspark" && recipe != "qn" && recipe != "qmtp" &&
       recipe != "qg") ||
      (policy != "off" && policy != "on"))
    return 2;
  const bool drafted = recipe == "dspark" || recipe == "qmtp";
  if (drafted == (std::string_view(argv[3]) == "-")) return 2;
  std::error_code ec;
  const std::filesystem::path out = argv[4];
  if (!std::filesystem::create_directory(out, ec) || ec) return 2;
  const bool threshold = policy == "on";
  if (recipe == "ds" || recipe == "dspark") {
    en::Dsv4Options options;
    options.artifact = argv[2];
    options.out = out;
    options.context = 8704;
    options.max_rows = 512;
    options.wave_slots = 2;
    options.frontier_head = true;
    options.prefill_outa_hca = true;
    options.prefill_outa_hca_partial = true;
    if (drafted) options.drafter = argv[3];
    options.startup_activation_threshold = threshold;
    return Measure(std::make_unique<Lifetime<en::Dsv4Runner>>(options), out / "result.json", recipe,
                   threshold);
  }
  en::Qwen38Options options;
  options.artifact = argv[2];
  options.out = out;
  options.context = 8192;
  options.max_rows = 512;
  options.wave_slots = 4;
  options.request_slots = 4;
  if (drafted) options.drafter = argv[3];
  options.startup_activation_threshold = threshold;
  return Measure(std::make_unique<Lifetime<en::Qwen38Runner>>(options), out / "result.json", recipe,
                 threshold);
}
