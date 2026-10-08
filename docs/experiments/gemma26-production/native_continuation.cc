// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// ARTIFACT INPUT_I32 NEW_OUT [production|primitive-route|full-final-ffn] [gemma26|gemma31]. Each
// diagnostic changes one option of the historical narrow-final serving baseline.
// Corrected bounded serving selects full-final-ffn; "production" names the old baseline.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = llmp::engine;
using en::support::Error;
int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) return 2;
  const std::string_view profile = argc == 6 ? argv[5] : "gemma26";
  if (profile != "gemma26" && profile != "gemma31") return 2;
  const bool dense = profile == "gemma31";
  const std::uint32_t prefix = dense ? 128 : 256, count = prefix + 4;
  const std::string_view mode = argc >= 5 ? argv[4] : "production";
  if (mode != "production" && mode != "primitive-route" && mode != "full-final-ffn") return 2;
  const bool primitive_route = mode == "primitive-route";
  const bool full_final_ffn = mode == "full-final-ffn";
  constexpr std::uint32_t kVocab = 262144;
  constexpr std::uint64_t kPublication = std::uint64_t{4} * kVocab * sizeof(float);
  const std::filesystem::path artifact(argv[1]), out(argv[3]);
  if (artifact.filename() !=
      (dense ? "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
             : "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3"))
    return 2;
  std::error_code error;
  if (std::filesystem::file_size(argv[2], error) != count * sizeof(std::int32_t) || error) return 2;
  std::vector<std::int32_t> input(count);
  std::ifstream file(argv[2], std::ios::binary);
  file.read(reinterpret_cast<char*>(input.data()),
            static_cast<std::streamsize>(input.size() * sizeof(std::int32_t)));
  if (!file || input.front() != 2 || !std::ranges::all_of(input, [](auto token) {
        return token >= 0 && token < static_cast<std::int32_t>(kVocab);
      }))
    return 2;
  if (!std::filesystem::create_directory(out, error) || error) return 2;
  std::filesystem::permissions(out, std::filesystem::perms::owner_all, error);
  if (error) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto& node = lifetime->node;
  lifetime->runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{.artifact = artifact,
                        .out = out,
                        .variant = dense ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
                        .context = 8192,
                        .max_rows = dense ? 256U : 1024U,
                        .slots = 4,
                        .max_head_rows = 4,
                        .frontier_head = !full_final_ffn,
                        .fuse_norm_rope = true,
                        .fuse_norm_add = true,
                        .fuse_gemma_route = !dense && !primitive_route,
                        .fuse_gemma_reduce = !dense,
                        .fuse_quant_glu = dense,
                        .owner_attention = true},
      0, 0);
  auto& runner = *lifetime->runner;
  auto& entered = lifetime->entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    std::cout << "NATIVE_CONTROL primitive_route=" << primitive_route << " profile=" << profile
              << " full_final_ffn=" << full_final_ffn << '\n';
    if (runner.profile().vocab != kVocab || runner.profile().layers != (dense ? 60U : 30U))
      return Error("approved Gemma4 shape differs");
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget = fixed + runner.weights().size() * en::kPagedExtent +
                        2 * node.StateCapacity() + kPublication;
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes());
    if (auto r = node.Start(llmp::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(
        0, runner.closure(), "Gemma4 continuation control", [&]() -> en::Status {
          if (!node.ChargeHost(kPublication, false))
            return Error("head publication capacity refused");
          struct Charge {
            en::PagedNode& node;
            ~Charge() { node.UnchargeHost(kPublication); }
          } charged{node};
          std::vector<float> logits;
          logits.reserve(kPublication / sizeof(float));
          if (logits.capacity() * sizeof(float) != kPublication)
            return Error("head publication capacity differs");
          const auto check = [&]() -> en::Status {
            if (logits.size() != kVocab ||
                !std::ranges::all_of(logits, [](float x) { return std::isfinite(x); }))
              return Error("nonfinite or incomplete native head");
            return {};
          };
          const auto capture = [&](int repeat, const std::string& name) -> en::Status {
            if (auto r = check(); !r) return r;
            std::ofstream head(out / (std::to_string(repeat) + "-" + name), std::ios::binary);
            head.write(reinterpret_cast<const char*>(logits.data()),
                       static_cast<std::streamsize>(logits.size() * sizeof(float)));
            head.flush();
            if (!head) return Error("native head publication failed");
            const auto& policy = runner.last_built_policy();
            std::cout << "NATIVE_HEAD repeat=" << repeat << " name=" << name << " choice="
                      << std::max_element(logits.begin(), logits.end()) - logits.begin()
                      << " norm_fused=" << policy.norm_fused << " norm_rope=" << policy.norm_rope
                      << " norm_add=" << policy.norm_add << " route=" << policy.gemma_route
                      << " reduce=" << policy.gemma_reduce << '\n';
            return {};
          };
          for (int repeat = 0; repeat < 2; ++repeat) {
            if (auto r = runner.Clear(); !r) return r;
            if (auto r = runner.Chunk(0, std::span(input).first(prefix), logits); !r) return r;
            if (auto r = capture(repeat, "prefix" + std::to_string(prefix) + ".f32"); !r) return r;
            for (std::uint32_t i = prefix; i < count; ++i) {
              if (auto r = check(); !r) return r;
              const auto choice = std::max_element(logits.begin(), logits.end()) - logits.begin();
              std::cout << "NATIVE_SUPPLIED_STEP repeat=" << repeat << " position=" << i
                        << " natural_match=" << (choice == input[i]) << '\n';
              if (auto r = runner.Chunk(i, std::span(input).subspan(i, 1), logits); !r) return r;
            }
            if (auto r = capture(repeat, "scalar" + std::to_string(count) + ".f32"); !r) return r;
            if (auto r = runner.Clear(); !r) return r;
            if (auto r = runner.Chunk(0, input, logits); !r) return r;
            if (auto r = capture(repeat, "fresh" + std::to_string(count) + ".f32"); !r) return r;
          }
          return {};
        });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = lifetime.release();
  }
  if (ran && retired) std::cout << "NATIVE_CONTINUATION_RETIRED\n";
  return ran && retired ? 0 : 1;
}
