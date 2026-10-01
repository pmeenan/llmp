// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P2, oracle rungs 4 and 5 (docs/backend-proof.md), and P4's
// paging cases for FP16: the Qwen2.5-0.5B FP16 fixture from its v0
// prepared artifact, alone on a paged node (tests/support/paged_node.h),
// paged into jitLLM's device VMM through the host-VMM landing zone (D-081)
// by the scheduler and its lanes, and executed as device jobs that hold
// leases on everything they touch (docs/experiments/backend-proof-p2/
// README.md). What each option does is fp16_runner.h's.
//
//   jitllm_fp16_paged --artifact DIR --trajectory control|heldout --tokens FILE
//                     --fusion on|off --out DIR [--restores N] [--relocate]
//                     [--partial] [--spill premapped|managed]
//                     [--embeddings duplicated|shared]
//                     [--lanes threads|inline] [--record] [--coalesce on|off]
//   jitllm_fp16_paged --artifact DIR ... --out DIR --load-only N
//                     [--weights device|host] [--backing managed|premapped]
//                     [--slots N] [--coalesce on|off]
//
// - --coalesce on reads chunks that wait behind the four in flight and
//   continue one another in a shard as one vectored request (BP-P1,
//   D-056); off, the reader's default, one request per 2 MiB chunk. Each
//   load reports the requests its reads took and the chunks (pieces) they
//   carried.
// - --lanes inline drives the scheduler and every lane from this thread,
//   in turns; --record needs it, since the launch recorder sees only its
//   own thread's calls (plan_compare.py). --lanes threads (the default) runs
//   each on its own thread, as a program wires them.
//
// Every evaluation's logits must equal the first's bit for bit; the first's
// are written, with a summary of the page-ins (bytes, times) and the
// coverage check.
//
// --load-only N measures page-in alone, through the same scheduler and
// lanes: the device weights loaded and evicted N times, nothing evaluated.
// --weights host reads them in place into host VMM instead (D-034's direct
// path, which D-081 replaced for execution); --backing premapped maps every
// extent's backing at setup instead of on each load, so an eviction is the
// catalog's alone and a load only reads (and copies). Together they separate
// the zone's copy and D-033's per-load backing from the reads themselves.
// --slots N sizes the zone other than D-081's 2 x depth, to see what the
// copy's hand-off costs.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/bytes.h"
#include "fp16_runner.h"
#include "launch_recorder.h"
#include "paged_node.h"
#include "plan_record.h"

namespace {

namespace ts = jitllm::test_support;
using Status = ts::Status;

struct Options {
  jitllm::benchmarks::Fp16Options model;
  bool inline_lanes = false;
  bool record = false;
  std::size_t slots = ts::kPagedSlots;
  bool coalesce = false;  // BP-P1's coalesced reads: the reader's option, off by default
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options options;
  jitllm::benchmarks::Fp16Options& o = options.model;
  bool fusion_set = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "--record") {
      options.record = true;
      continue;
    }
    if (a == "--relocate") {
      o.relocate = true;
      continue;
    }
    if (a == "--partial") {
      o.partial = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      return std::unexpected(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    if (a == "--spill") {
      if (v != "premapped" && v != "managed") {
        return std::unexpected("--spill is premapped or managed");
      }
      o.spill = v;
    } else if (a == "--embeddings") {
      if (v != "duplicated" && v != "shared") {
        return std::unexpected("--embeddings is duplicated or shared");
      }
      o.shared_embeddings = v == "shared";
    } else if (a == "--artifact") {
      o.artifact = v;
    } else if (a == "--trajectory") {
      o.trajectory = v;
    } else if (a == "--tokens") {
      o.tokens = v;
    } else if (a == "--fusion") {
      o.fusion = v == "on";
      fusion_set = v == "on" || v == "off";
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--restores") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.restores).ec != std::errc{} ||
          o.restores < 0) {
        return std::unexpected("--restores takes a count");
      }
    } else if (a == "--load-only") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.load_only).ec != std::errc{} ||
          o.load_only < 1) {
        return std::unexpected("--load-only takes a count");
      }
    } else if (a == "--slots") {
      if (std::from_chars(v.data(), v.data() + v.size(), options.slots).ec != std::errc{} ||
          options.slots < 1 || options.slots > 64) {
        return std::unexpected("--slots takes a count from 1 to 64");
      }
    } else if (a == "--weights") {
      if (v != "device" && v != "host") {
        return std::unexpected("--weights is device or host");
      }
      o.weights_host = v == "host";
    } else if (a == "--backing") {
      if (v != "managed" && v != "premapped") {
        return std::unexpected("--backing is managed or premapped");
      }
      o.premapped = v == "premapped";
    } else if (a == "--lanes") {
      if (v != "inline" && v != "threads") {
        return std::unexpected("--lanes is inline or threads");
      }
      options.inline_lanes = v == "inline";
    } else if (a == "--coalesce") {
      if (v != "on" && v != "off") {
        return std::unexpected("--coalesce is on or off");
      }
      options.coalesce = v == "on";
    } else {
      return std::unexpected(std::format("unknown argument {}", a));
    }
  }
  if (o.artifact.empty() || o.trajectory.empty() || o.tokens.empty() || o.out.empty() ||
      !fusion_set || (options.record && !options.inline_lanes) ||
      ((o.weights_host || o.premapped || options.slots != ts::kPagedSlots) && o.load_only == 0) ||
      ((o.partial || !o.spill.empty() || o.shared_embeddings) && o.load_only > 0)) {
    return std::unexpected(
        "usage: jitllm_fp16_paged --artifact DIR --trajectory control|heldout --tokens FILE "
        "--fusion on|off --out DIR [--restores N] [--relocate] [--partial] "
        "[--spill premapped|managed] [--embeddings duplicated|shared] [--lanes threads|inline] "
        "[--record (with --lanes inline)] [--coalesce on|off] | --load-only N "
        "[--weights device|host] [--backing managed|premapped] [--slots N] [--coalesce on|off]");
  }
  return options;
}

// The node with the one model on stream 0, set up, run and torn down.
Status Run(jitllm::benchmarks::Fp16Runner& runner, ts::PagedNode& node,
           std::vector<ts::PagedModel*>& entered_models) {
  if (auto r = node.Open(); !r) {
    return r;
  }
  entered_models.push_back(&runner);
  if (auto r = runner.Setup(); !r) {
    return r;
  }
  if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) {
    return r;
  }
  if (auto r = node.Start(jitllm::base::Bytes(std::uint64_t{64} << 30U)); !r) {
    return r;
  }
  if (auto r = runner.Register(); !r) {
    return r;
  }
  if (auto r = runner.Bind(); !r) {
    return r;
  }
  node.Run();
  return runner.RunAlone();
}

}  // namespace

int main(int argc, char** argv) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  // The recording starts before anything touches CUDA (--record).
  const bool record =
      std::ranges::any_of(args, [](const char* a) { return std::string_view(a) == "--record"; });
  std::unique_ptr<ts::Recording> recording;
  if (record) {
    recording = std::make_unique<ts::Recording>();
  }
  const auto options = Parse(args);
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  std::string lines;
  Status ran;
  {
    ts::PagedNode node({.compute_streams = 1,
                        .slots = options->slots,
                        .inline_lanes = options->inline_lanes,
                        .coalesce = options->coalesce});
    jitllm::benchmarks::Fp16Runner runner(node, options->model, 0, 0, recording.get(), lines);
    std::vector<ts::PagedModel*> entered_models;
    ran = Run(runner, node, entered_models);
    if (auto finished = node.TearDown(entered_models); !finished) {
      if (!ran) std::println(stderr, "FAILED: {}", ran.error());
      std::println(stderr, "retirement failed: {}", finished.error());
      std::abort();
    }
  }
  if (recording) {
    for (const auto& event : recording->Take()) {
      lines += ts::EventLine(event);
    }
    std::filesystem::create_directories(options->model.out);
    std::ofstream file(options->model.out / "recording.jsonl");
    file << ts::HeaderLine(std::format("jitllm_fp16_paged {} fusion {}", options->model.trajectory,
                                       options->model.fusion ? "on" : "off"),
                           ts::LoadedCublas())
         << lines;
  }
  if (!ran) {
    std::println(stderr, "{}", ran.error());
    return 1;
  }
  return 0;
}
