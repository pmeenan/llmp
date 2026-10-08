// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded full-vocabulary teacher forcing on exact shared integer IDs.
// ARTIFACT IDS_I32 OUTPUT_DIR CHUNK ordinary|norm|both|norm_rope|norm_add|moe|all
// [26|31] [normmul-off|normmul-on]. Without the final override, only the
// historical norm policy selects plain norm fusion. No support claim.
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = llmp::engine;
using en::support::Error;
int main(int argc, char** argv) {
  if (argc < 6 || argc > 8) return 2;
  const std::string_view variant = argc >= 7 ? argv[6] : "26";
  if (variant != "26" && variant != "31") return 2;
  std::uint32_t chunk = 0;
  const std::string_view number(argv[4]), policy(argv[5]);
  const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), chunk);
  if (error != std::errc{} || end != number.data() + number.size() || chunk == 0 || chunk > 1024 ||
      (policy != "ordinary" && policy != "norm" && policy != "both" && policy != "norm_rope" &&
       policy != "norm_add" && policy != "moe" && policy != "all"))
    return 2;
  const std::string_view normmul =
      argc == 8 ? argv[7] : (policy == "norm" ? "normmul-on" : "normmul-off");
  if (normmul != "normmul-off" && normmul != "normmul-on") return 2;
  std::error_code file_error;
  if (std::filesystem::file_size(argv[2], file_error) != 4096 || file_error) return 2;
  std::array<std::int32_t, 1024> ids{};
  std::ifstream file(argv[2], std::ios::binary);
  file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
  if (!file || ids.front() != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < 262144; }))
    return 2;
  const std::filesystem::path out(argv[3]);
  if (!std::filesystem::create_directory(out, file_error) || file_error) return 2;
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
          .variant = variant == "31" ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
          .max_rows = chunk,
          .fuse_norms = normmul == "normmul-on",
          .fuse_norm_rope = policy == "both" || policy == "norm_rope" || policy == "all",
          .fuse_norm_add = policy == "both" || policy == "norm_add" || policy == "all",
          .fuse_gemma_route = policy == "moe" || policy == "all",
          .fuse_gemma_reduce = policy == "moe" || policy == "all"},
      0, 0);
  auto& runner = *lifetime->runner;
  auto& entered = lifetime->entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    std::cout << "QUALITY_PROFILE variant=" << variant << " layers=" << runner.profile().layers
              << " experts=" << runner.profile().experts << " policy=" << policy
              << " normmul=" << normmul << " context=4096 max_rows=" << chunk
              << " input_rows=" << ids.size() << " scored_targets=" << ids.size() - 1
              << " all_outputs=1\n";
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(chunk) * 262144 * 4;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget = fixed + runner.weights().size() * en::kPagedExtent +
                        2 * node.StateCapacity() + output_bytes;
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes());
    if (auto r = node.Start(llmp::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(
        0, runner.closure(), "Gemma4 fixed teacher forcing", [&]() -> en::Status {
          // Charge the maximum caller publication vector before allocation. Its
          // destruction precedes the charge guard, even on failed model work.
          if (!node.ChargeHost(output_bytes, false))
            return Error("likelihood publication capacity refused");
          struct Charge {
            en::PagedNode& node;
            std::uint64_t bytes;
            ~Charge() { node.UnchargeHost(bytes); }
          } charged{node, output_bytes};
          std::vector<float> logits;
          logits.reserve(static_cast<std::size_t>(output_bytes / 4));
          if (logits.capacity() * 4 != output_bytes)
            return Error("unexpected publication vector capacity");
          std::cout << "QUALITY_PUBLICATION charged_capacity_bytes=" << output_bytes
                    << " capacity_rows=" << chunk << " vocab=" << runner.profile().vocab
                    << " total_head_bytes=" << std::uint64_t{ids.size()} * 262144 * 4 << '\n';
          std::ofstream heads(out / "logits.f32", std::ios::binary);
          for (std::uint32_t first = 0; first < ids.size(); first += chunk) {
            const auto rows = std::min(chunk, static_cast<std::uint32_t>(ids.size()) - first);
            if (auto r = runner.Chunk(first, std::span(ids).subspan(first, rows), logits, true); !r)
              return r;
            if (logits.size() != static_cast<std::size_t>(rows) * 262144)
              return Error("likelihood output shape differs");
            heads.write(reinterpret_cast<const char*>(logits.data()),
                        static_cast<std::streamsize>(logits.size() * 4));
            if (!heads) return Error("writing likelihood rows failed");
            const auto& p = runner.last_built_policy();
            std::cout << "QUALITY_CHUNK first=" << first << " rows=" << rows
                      << " completed=" << first + rows << " norm_fused=" << p.norm_fused
                      << " norm_rope=" << p.norm_rope << " norm_add=" << p.norm_add
                      << " gemma_route=" << p.gemma_route << " gemma_reduce=" << p.gemma_reduce
                      << " shared_vecq=" << p.shared_vecq << " row_products=" << p.row_products
                      << '\n';
          }
          heads.flush();
          return heads ? en::Status{} : en::Status(Error("flushing likelihood rows failed"));
        });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    // Quarantine the node, runner and graphs until process exit.
    std::ignore = lifetime.release();
  }
  return ran && retired ? 0 : 1;
}
