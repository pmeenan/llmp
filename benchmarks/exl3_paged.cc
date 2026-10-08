// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P3, oracle rungs 4 and 5 for EXL3 (docs/backend-proof.md),
// and P4-P5's paging and lifetime cases: an EXL3 fixture from its v0
// prepared artifact, alone on a paged node (tests/support/paged_node.h),
// paged into llmpalooza's device VMM through the host-VMM landing zone (D-081)
// by the scheduler and its lanes, and run through the native operation
// plan as device jobs that hold leases on everything they touch
// (docs/experiments/backend-proof-p3/README.md). What each option does is
// exl3_runner.h's.
//
//   llmp_exl3_paged --artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O
//                     --plan PLAN.txt --ids FILE --out DIR
//                     [--prefixes 32,144,145,1023,1024] [--restores N]
//                     [--relocate] [--partial] [--spill premapped|managed]
//                     [--cancel-in-flight]
//                     [--lanes threads|inline] [--record] [--coalesce on|off]
//
// --coalesce on reads chunks that wait behind the four in flight and
// continue one another in a shard as one vectored request (BP-P1); off,
// the reader's default, one request per chunk.
//
// Every evaluation's logits must equal the first's bit for bit; the first's
// are written as exl3_exec.cc writes them (.npy per prefix), with a summary
// of the page-ins and the coverage check. Rung 4 against rung 3 is the
// comparison of those files with exl3_exec.cc's.

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <format>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/bytes.h"
#include "exl3_runner.h"
#include "paged_node.h"

namespace {

namespace ts = llmp::test_support;
using llmp::benchmarks::Exl3Options;
using Status = ts::Status;

struct Options {
  Exl3Options model;
  bool inline_lanes = false;
  bool coalesce = false;  // BP-P1's coalesced reads: the reader's option, off by default
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options options;
  Exl3Options& o = options.model;
  bool arm = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "--record") {
      o.record = true;
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
    if (a == "--cancel-in-flight") {
      o.cancel = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      return std::unexpected(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    if (a == "--artifact") {
      o.artifact = v;
    } else if (a == "--spill") {
      if (v != "premapped" && v != "managed") {
        return std::unexpected("--spill is premapped or managed");
      }
      o.spill = v;
    } else if (a == "--fixture") {
      o.fixture = v;
    } else if (a == "--arm") {
      if (v != "G" && v != "O") {
        return std::unexpected("--arm is G or O");
      }
      o.arm = v == "G" ? llmp::model::Exl3Arm::kG : llmp::model::Exl3Arm::kO;
      arm = true;
    } else if (a == "--plan") {
      o.plan = v;
    } else if (a == "--ids") {
      o.ids = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--prefixes") {
      o.prefixes.clear();
      std::istringstream list{std::string(v)};
      for (std::string item; std::getline(list, item, ',');) {
        int prefix = 0;
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), prefix);
        if (error != std::errc() || end != item.data() + item.size() || prefix <= 0) {
          return std::unexpected("--prefixes takes positive integers");
        }
        o.prefixes.push_back(prefix);
      }
    } else if (a == "--restores") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.restores).ec != std::errc{} ||
          o.restores < 0 || o.restores > 8) {
        return std::unexpected("--restores takes a count up to 8");
      }
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
  if (o.artifact.empty() || (o.fixture != "4.0bpw" && o.fixture != "4.5bpw") || !arm ||
      o.plan.empty() || o.ids.empty() || o.out.empty() || o.prefixes.empty() ||
      (o.record && !options.inline_lanes) || (o.cancel && options.inline_lanes)) {
    return std::unexpected(
        "usage: llmp_exl3_paged --artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O --plan PLAN.txt "
        "--ids FILE --out DIR [--prefixes LIST] [--restores N] [--relocate] [--partial] "
        "[--spill premapped|managed] [--cancel-in-flight] [--lanes threads|inline] "
        "[--record (with --lanes inline)] [--coalesce on|off]");
  }
  return options;
}

// The node with the one model on stream 0, set up and run.
Status Run(llmp::benchmarks::Exl3Runner& runner, ts::PagedNode& node,
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
  if (auto r = node.Start(llmp::base::Bytes(std::uint64_t{64} << 30U)); !r) {
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
  const auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Status ran;
  {
    ts::PagedNode node({.compute_streams = 1,
                        .slots = ts::kPagedSlots,
                        .inline_lanes = options->inline_lanes,
                        .coalesce = options->coalesce});
    llmp::benchmarks::Exl3Runner runner(node, options->model, 0, 0);
    std::vector<ts::PagedModel*> entered_models;
    ran = Run(runner, node, entered_models);
    if (auto finished = node.TearDown(entered_models); !finished) {
      if (!ran) std::println(stderr, "FAILED: {}", ran.error());
      std::println(stderr, "retirement failed: {}", finished.error());
      std::abort();
    }
  }
  if (!ran) {
    std::println(stderr, "FAILED: {}", ran.error());
    return 1;
  }
  std::println("DONE {}", options->model.out.string());
  return 0;
}
