// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// ARTIFACT NEW_OUTPUT IDS_I32 capture|control. Untimed scalar P67 acquisition.
#include "gemma_dense_ffn_capture.h"

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>

#include "engine/gemma4_runner.h"
#include "engine/support.h"
namespace en = jitllm::engine;
using en::support::Error;
int main(int argc, char** argv) {
  umask(0077);
  if (argc != 5 ||
      (std::string_view(argv[4]) != "capture" && std::string_view(argv[4]) != "control"))
    return 2;
  std::vector<std::int32_t> ids(1024);
  std::ifstream input(argv[3], std::ios::binary);
  input.read(reinterpret_cast<char*>(ids.data()), 4096);
  if (!input || input.peek() != std::char_traits<char>::eof() || ids[0] != 2 ||
      !std::ranges::all_of(ids, [](auto x) { return x >= 0 && x < 262144; }))
    return 2;
  std::filesystem::path out = argv[2];
  std::error_code e;
  if (!std::filesystem::create_directory(out, e) || e) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
    jitllm::benchmark::DenseFfnCapture capture;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner =
      std::make_unique<en::Gemma4Runner>(node,
                                         en::Gemma4Options{.artifact = argv[1],
                                                           .out = out,
                                                           .variant = en::Gemma4Variant::k31B,
                                                           .context = 256,
                                                           .max_rows = 128,
                                                           .slots = 1,
                                                           .graphs = false,
                                                           .row_invariant = false,
                                                           .fuse_norm_rope = true,
                                                           .fuse_norm_add = true},
                                         0, 0);
  auto& runner = *life->runner;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    constexpr auto metadata = jitllm::benchmark::DenseFfnCapture::kHostBytes;
    if (!node.ChargeHost(metadata, false)) return Error("capture metadata charge");
    struct Grant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~Grant() { node.UnchargeHost(bytes); }
    } mg{node, metadata};
    life->entered.push_back(&runner);
    if (auto r = life->capture.Setup(node, argv[1], std::string_view(argv[4]) == "capture"); !r)
      return r;
    jitllm::benchmark::DenseFfnCapture::active = &life->capture;
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint64_t heap = 3 * 262144 * 4 + 4096;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes() + heap + metadata);
    if (auto r = node.Start(jitllm::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                                2 * node.StateCapacity() + heap + metadata));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    if (!node.ChargeHost(heap, false)) return Error("publication host charge");
    Grant hg{node, heap};
    return node.WithRequest(
        0, runner.closure(), "dense31 scalar FFN protected acquisition", [&]() -> en::Status {
          std::vector<float> head;
          const auto prefill = [&]() -> en::Status {
            life->capture.Arm(0);
            if (auto r = runner.Clear(); !r) return r;
            en::Gemma4Runner::Work w{0, 0, std::span(ids).first(64), &head};
            return runner.Wave(std::span(&w, 1));
          };
          const auto step = [&](std::uint32_t p, bool arm) -> en::Status {
            life->capture.Arm(arm ? p : 0);
            en::Gemma4Runner::Work w{0, p, std::span(ids).subspan(p, 1), &head};
            return runner.Wave(std::span(&w, 1));
          };
          if (auto r = prefill(); !r) return r;
          for (std::uint32_t p = 64; p < 72; ++p)
            if (auto r = step(p, false); !r) return r;
          if (auto r = prefill(); !r) return r;
          for (std::uint32_t p = 64; p < 67; ++p)
            if (auto r = step(p, false); !r) return r;
          if (auto r = step(67, true); !r) return r;
          auto slot = runner.request_slot(0);
          if (!slot || (*slot)->completed_positions() != 68 || head.size() != 262144 ||
              !ffn_replay::Finite(head))
            return Error("incomplete final scalar query");
          if (!ffn_replay::Write(out / "heads.f32", head)) return Error("head write");
          std::ofstream f(out / "inputs.i32", std::ios::binary | std::ios::noreplace);
          f.write(reinterpret_cast<const char*>(ids.data()), 4096);
          f.flush();
          if (!f) return Error("ID write");
          if (auto r = life->capture.Save(out); !r) return r;
          const auto& p = runner.last_built_policy();
          std::cout << "FFN_ACQUISITION untimed=1 variant=31 owners=1 position=67 completed=68 "
                       "graphs=0 context=256 max_rows=128 row_products="
                    << p.row_products << " norm_rope=" << p.norm_rope << " norm_add=" << p.norm_add
                    << " norm_fused=" << p.norm_fused << " rope_store=" << p.rope_store
                    << " shared_vecq=" << p.shared_vecq << " gemma_route=" << p.gemma_route
                    << " gemma_reduce=" << p.gemma_reduce << " rows=" << p.rows
                    << " segments=" << p.segments << '\n';
          return {};
        });
  };
  auto result = execute();
  auto retired = node.TearDown(life->entered);
  if (!result) std::cerr << result.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    (void)life.release();
  } else
    jitllm::benchmark::DenseFfnCapture::active = nullptr;
  return result && retired ? 0 : 1;
}
