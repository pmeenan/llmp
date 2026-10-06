// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded full-vocabulary teacher forcing on exact shared integer IDs.
// Closed diagnostic: ARTIFACT IDS_I32 OUTPUT_DIR. Required process environment
// JITLLM_GEMMA_KEEP28_ROUTING=0|1 is immutable across sizing and execution.
// One 26B/C1 1024-row all-output chunk, all policy plus plain norm. No support claim.
#include "gemma_quality_keep28.h"

#include <algorithm>
#include <array>
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
  if (argc != 4) return 2;
  const auto keep28 = en::diagnostic::Keep28Routing();
  if (!keep28) return 2;
  constexpr std::string_view variant = "26", policy = "all", normmul = "normmul-on";
  constexpr std::uint32_t chunk = 1024;
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
          .context = 4096,
          .max_rows = chunk,
          .slots = 1,
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
              << " all_outputs=1 keep28_routing=" << *keep28 << '\n';
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(chunk) * 262144 * 4;
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
            if (p.norm_fused != 121 || p.norm_rope != 60 || p.norm_add != 90 ||
                p.gemma_route != (*keep28 ? 29U : 30U) || p.gemma_reduce != 30 ||
                p.shared_vecq != 0 || p.row_products != 0)
              return Error("closed keep28 policy counts differ");
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
