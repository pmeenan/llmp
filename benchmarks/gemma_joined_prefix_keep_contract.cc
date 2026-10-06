// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// No launch: stub the real planner to verify exact forwarding and keep lifetime.
#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "gemma_joined_prefix_keep.h"
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
namespace {
const en::Gemma4Model* seen_model = nullptr;
const kg::Gemma4ChunkShape* seen_shape = nullptr;
const kg::DeviceChoices* seen_choices = nullptr;
std::uint64_t seen_address = 0, seen_bytes = 0;
std::vector<std::string> seen_keep;
}  // namespace
GemmaPrefixPlanResult PrefixRealPlan(const en::Gemma4Model& model,
                                     const kg::Gemma4ChunkShape& shape,
                                     const kg::DeviceChoices& choices, std::uint64_t address,
                                     std::uint64_t bytes, std::span<const std::string> keep) {
  seen_model = &model;
  seen_shape = &shape;
  seen_choices = &choices;
  seen_address = address;
  seen_bytes = bytes;
  seen_keep.assign(keep.begin(), keep.end());
  return std::unique_ptr<en::Gemma4Planned>{};
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string_view expected = argv[1];
  if (expected != "off" && expected != "on" && expected != "invalid") return 2;
  const auto profile = jitllm::model::Gemma4_26BA4B();
  jitllm::model::Gemma4StateLayout state;
  state.context = 4096;
  state.max_rows = 1024;
  en::Gemma4Model model;
  model.profile = &profile;
  model.state = &state;
  model.slots.resize(5);
  kg::DeviceChoices choices;
  const std::array<std::string, 1> keep{"existing_output"};
  kg::Gemma4ChunkShape shape;
  shape.segments.push_back({.slot = 4, .rows = 992});
  const auto check = [&](bool eligible, std::uint64_t address) {
    seen_model = nullptr;
    const auto result = PrefixWrapPlan(model, shape, choices, address, 32768, keep);
    if (expected == "invalid") return !result && seen_model == nullptr;
    const bool appended = eligible && expected == "on";
    return result && seen_model == &model && seen_shape == &shape && seen_choices == &choices &&
           seen_address == address && seen_bytes == 32768 &&
           seen_keep.size() == (appended ? 2U : 1U) && seen_keep.front() == keep.front() &&
           (!appended || seen_keep.back() == "blk.28.router_probabilities");
  };
  // Startup/first pass and bound activation pass preserve all planner arguments.
  if (!check(true, 0) || !check(true, 65536)) return 1;
  // The five-query decode, C1 one-query, feature and unsupported contracts forward.
  shape.segments.clear();
  for (std::uint32_t slot = 0; slot < 5; ++slot)
    shape.segments.push_back({.slot = slot, .rows = 1});
  if (!check(false, 65536)) return 1;
  shape.segments.resize(1);
  if (!check(false, 65536)) return 1;
  shape.segments.front().rows = 992;
  shape.feature_outputs = 1;
  if (!check(false, 65536)) return 1;
  shape.feature_outputs = 0;
  state.context = 256;
  if (!check(false, 65536)) return 1;
  state.context = 4096;
  state.max_rows = 128;
  if (!check(false, 65536)) return 1;
  state.max_rows = 1024;
  model.slots.resize(1);
  if (!check(false, 65536)) return 1;
  model.slots.resize(5);
  model.options.first_layer = 1;
  if (!check(false, 65536)) return 1;
  model.options.first_layer = 0;
  model.profile = nullptr;
  if (!check(false, 65536)) return 1;
  std::cout << "PREFIX_KEEP_CONTRACT mode=" << expected << " cases=10 no_launch=1 PASS\n";
  return 0;
}
