// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Fixed-capacity Gemma26 teacher forcing. ARTIFACT IDS_I32 NEW_OUTPUT_DIR
// TEACHER_CHUNK (128|1024). Both schedules use max_rows=1024/context=4096,
// full heads and the same all-policy fusions; only the Chunk schedule varies.
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

namespace en = jitllm::engine;
using en::support::Error;
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  constexpr std::string_view variant = "26", policy = "all", normmul = "normmul-on";
  constexpr std::uint32_t max_rows = 1024;
  std::uint32_t chunk = 0;
  const std::string_view number(argv[4]);
  const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), chunk);
  if (error != std::errc{} || end != number.data() + number.size() ||
      (chunk != 128 && chunk != 1024))
    return 2;
  std::error_code file_error;
  if (std::filesystem::file_size(argv[2], file_error) != 4096 || file_error) return 2;
  std::array<std::int32_t, 1024> ids{};
  std::ifstream file(argv[2], std::ios::binary);
  file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
  if (!file || ids.front() != 2 || std::ranges::count(ids, 2) != 1 ||
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
  lifetime->runner =
      std::make_unique<en::Gemma4Runner>(node,
                                         en::Gemma4Options{.artifact = argv[1],
                                                           .out = out,
                                                           .variant = en::Gemma4Variant::k26BA4B,
                                                           .context = 4096,
                                                           .max_rows = max_rows,
                                                           .slots = 1,
                                                           .max_head_rows = 0,
                                                           .fuse_norms = true,
                                                           .fuse_norm_rope = true,
                                                           .fuse_norm_add = true,
                                                           .fuse_gemma_route = true,
                                                           .fuse_gemma_reduce = true},
                                         0, 0);
  auto& runner = *lifetime->runner;
  auto& entered = lifetime->entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (runner.layout().max_rows != max_rows || runner.layout().context != 4096 ||
        runner.layout().local_cells != 2048 || runner.layout().global_cells != 4096)
      return Error("fixed teacher state capacity differs");
    std::cout << "QUALITY_PROFILE variant=" << variant << " layers=" << runner.profile().layers
              << " experts=" << runner.profile().experts << " policy=" << policy
              << " normmul=" << normmul << " context=4096 max_rows=" << max_rows
              << " teacher_chunk=" << chunk << " local_cells=" << runner.layout().local_cells
              << " global_cells=" << runner.layout().global_cells << " input_rows=" << ids.size()
              << " scored_targets=" << ids.size() - 1 << " all_outputs=1\n";
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(max_rows) * 262144 * 4;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget = fixed + runner.weights().size() * en::kPagedExtent +
                        2 * node.StateCapacity() + output_bytes;
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes());
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
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
                    << " capacity_rows=" << max_rows << " vocab=" << runner.profile().vocab
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
            if (p.norm_fused != 121 || p.norm_rope != 60 || p.norm_add != 90 ||
                p.gemma_route != 30 || p.gemma_reduce != 30 || p.shared_vecq != 0 ||
                p.row_products != 0)
              return Error("fixed all-policy teacher plan differs");
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
