// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The retained-backing comparison's deterministic replay (part (a),
// docs/backend-proof.md#retained-backing-comparison): replays each budget
// file of a swap trace through each design on the address-only fake
// provider, twice, and prints one JSON line per design and budget with the
// frozen deterministic metrics and whether the two replays agreed exactly.
// A disagreement voids the run (exit status 1). The report and the
// criteria's application are in docs/experiments/retained-backing/.
//
//   llmp_rb_replay [--threads N] [--check-every N] [--role ROLE]
//                    [--file NAME]... TRACE_DIR [DESIGN]...
//
// ROLE is primary (the default) or confirmation; the confirmation seed is
// reserved for the winner (D-079). With no DESIGN, all 13 run, in the
// criteria's order; with no --file, the three budget files, 5/4 first.

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "retained_backing/designs.h"
#include "retained_backing/replay.h"
#include "retained_backing/trace.h"

namespace {

using llmp::rb::DesignSpec;
using llmp::rb::Metrics;

std::optional<std::uint64_t> Number(std::string_view text) {
  std::uint64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

int Usage() {
  std::println(stderr,
               "usage: llmp_rb_replay [--threads N] [--check-every N] [--role ROLE] "
               "[--file NAME]... TRACE_DIR [DESIGN]...");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  std::uint64_t threads = 4;
  llmp::rb::ReplayOptions options;
  std::string role = "primary";
  std::vector<std::string> files;
  std::optional<std::string> dir;
  std::vector<DesignSpec> designs;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    const bool has_value = i + 1 < args.size();
    if (arg == "--threads" && has_value) {
      const auto n = Number(args[++i]);
      if (!n || *n == 0) {
        return Usage();
      }
      threads = *n;
    } else if (arg == "--check-every" && has_value) {
      const auto n = Number(args[++i]);
      if (!n) {
        return Usage();
      }
      options.check_every = *n;
    } else if (arg == "--role" && has_value) {
      role = args[++i];
    } else if (arg == "--file" && has_value) {
      files.emplace_back(args[++i]);
    } else if (arg.starts_with("--")) {
      return Usage();
    } else if (!dir) {
      dir = std::string(arg);
    } else if (const auto spec = llmp::rb::FindDesign(arg)) {
      designs.push_back(*spec);
    } else {
      std::println(stderr, "unknown design {}", arg);
      return 2;
    }
  }
  if (!dir) {
    return Usage();
  }
  if (designs.empty()) {
    designs = llmp::rb::AllDesigns();
  }
  if (files.empty()) {
    files.assign(llmp::rb::kTraceFiles.begin(), llmp::rb::kTraceFiles.end());
  }

  bool agreed = true;
  for (const std::string& file : files) {
    auto trace = llmp::rb::LoadTrace(*dir, file, role);
    if (!trace) {
      std::println(stderr, "{}", trace.error());
      return 1;
    }
    std::println(stderr, "{}: sha256 {}, budget {} bytes, {} groups, {} events", file,
                 trace->sha256, trace->budget, trace->groups.size(), trace->events.size());
    // Every design twice, each replay on its own fresh fake provider.
    std::vector<Metrics> results(designs.size() * 2);
    std::atomic<std::size_t> next{0};
    {
      std::vector<std::jthread> workers;
      workers.reserve(threads);
      for (std::uint64_t t = 0; t < threads; ++t) {
        workers.emplace_back([&] {
          for (std::size_t task = next++; task < results.size(); task = next++) {
            const auto start = std::chrono::steady_clock::now();
            results[task] = llmp::rb::Replay(*trace, designs[task / 2], options);
            const std::chrono::duration<double> took = std::chrono::steady_clock::now() - start;
            std::println(stderr, "{} {} replay {}: {:.1f} s", file, designs[task / 2].name,
                         (task % 2) + 1, took.count());
          }
        });
      }
    }
    for (std::size_t d = 0; d < designs.size(); ++d) {
      const Metrics& first = results[d * 2];
      const bool same = first == results[(d * 2) + 1];
      agreed = agreed && same;
      std::string line = llmp::rb::ToJson(first);
      line.pop_back();  // the closing brace
      std::println(R"({},"role":"{}","trace_sha256":"{}","replays_agree":{}}})", line, role,
                   trace->sha256, same);
      (void)std::fflush(stdout);
    }
  }
  return agreed ? 0 : 1;
}
