// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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

namespace {
namespace en = llmp::engine;

std::string Stats(const en::StartupPlacementStats& s) {
  return std::format(
      R"({{"plans":{},"nodes":{},"bounded":{},"exact":{},"max_activations":{},"max_inputs":{},"max_host":{},"max_nodes":{}}})",
      s.plans, s.nodes, s.bounded, s.exact, s.max_activations, s.max_inputs, s.max_host,
      s.max_nodes);
}

struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  en::Dsv4Runner runner;
  explicit Lifetime(const en::Dsv4Options& options) : runner(node, options, 0, 0) {}
};

int Measure(std::unique_ptr<Lifetime> owner, const std::filesystem::path& result,
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
    budget = std::format(
        R"({{"activations":{},"scratch":{},"lane_scratch":{},"host_inputs":{},"plan_floor":{},"scratch_arena":{},"initialized_logical":{}}})",
        runner.activations_needed(), runner.pool_needed(), runner.lane_pool_needed(),
        runner.host_input_bytes(), runner.plan_floor_bytes(), en::ScratchArenaBytes(),
        runner.used_state_bytes());
    scalar = Stats(runner.startup_placement_stats());
    std::println("plan budget: {}", runner.plan_report());
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
                 "usage: llmp_startup_measurement_probe ds|dspark TARGET DRAFTER|- "
                 "NEW_OUT off|on");
    return 2;
  }
  const std::string_view recipe = argv[1];
  const std::string_view policy = argv[5];
  if ((recipe != "ds" && recipe != "dspark") || (policy != "off" && policy != "on")) return 2;
  const bool drafted = recipe == "dspark";
  if (drafted == (std::string_view(argv[3]) == "-")) return 2;
  std::error_code ec;
  const std::filesystem::path out = argv[4];
  if (!std::filesystem::create_directory(out, ec) || ec) return 2;
  const bool threshold = policy == "on";
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
  return Measure(std::make_unique<Lifetime>(options), out / "result.json", recipe, threshold);
}
