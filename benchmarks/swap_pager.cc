// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The production runtime's swap-table with the pager's options chosen:
// `jitllm_swap_pager ARM [--budget-bytes N] RUNTIME-ARGUMENTS...`, the rest exactly as
// jitllm-runtime takes them (`--config ... swap-table ...`). ARM `on` is
// the default; `off` turns lazy handoff off (ServingOptions::lazy_handoff);
// `plain` keeps it but turns off state zero-filling and the handle reserve
// (zero_state, handle_reserve). Matched controls for options configuration
// and the CLI do not expose.
#include <signal.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

#include "runtime/commands.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"

namespace {
std::string_view g_arm;
std::optional<std::uint64_t> g_budget;

int Serve(const jitllm::config::NodeConfig& config, const jitllm::config::RuntimeRoles& roles,
          const jitllm::runtime::CommandOptions& command, std::FILE* out, std::FILE* log) {
  auto chosen = command;
  chosen.serving.diagnostic_budget_cap_bytes = g_budget;
  chosen.serving.lazy_handoff = g_arm != "off";
  if (g_arm == "partial" || g_arm == "full")
    chosen.serving.partial_weight_eviction = g_arm == "partial";
  if (g_arm == "plain") {
    chosen.serving.zero_state = false;
    chosen.serving.handle_reserve = 0;
  }
  (void)std::fprintf(
      log,
      "jitllm_swap_pager: lazy_handoff=%s zero_state=%s handle_reserve=%s partial_weights=%s\n",
      chosen.serving.lazy_handoff ? "on" : "off", chosen.serving.zero_state ? "on" : "off",
      chosen.serving.handle_reserve.has_value() ? "none" : "default",
      chosen.serving.partial_weight_eviction ? "on" : "off");
  return jitllm::runtime::RunServing(config, roles, chosen, out, log);
}
}  // namespace

int main(int argc, char** argv) {
  g_arm = argc < 2 ? std::string_view() : std::string_view(argv[1]);
  if (g_arm != "on" && g_arm != "off" && g_arm != "plain" && g_arm != "full" &&
      g_arm != "partial") {
    (void)std::fprintf(stderr,
                       "usage: jitllm_swap_pager on|off|plain|full|partial [--budget-bytes N] "
                       "RUNTIME-ARGUMENTS...\n");
    return 2;
  }
  sigset_t stop;
  (void)::sigemptyset(&stop);
  (void)::sigaddset(&stop, SIGTERM);
  (void)::sigaddset(&stop, SIGINT);
  (void)::sigaddset(&stop, SIGHUP);
  (void)::sigaddset(&stop, SIGCHLD);
  (void)::pthread_sigmask(SIG_BLOCK, &stop, nullptr);
  (void)::signal(SIGPIPE, SIG_IGN);
  (void)::setenv("CUDA_CACHE_DISABLE", "1", 1);  // NOLINT(concurrency-mt-unsafe)
  const std::span<char*> all(argv, static_cast<std::size_t>(argc));
  std::size_t first = 2;
  if (all.size() > first && std::string_view(all[first]) == "--budget-bytes") {
    if (all.size() <= first + 1) return 2;
    const std::string_view text(all[first + 1]);
    std::uint64_t cap = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), cap);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || cap == 0) return 2;
    g_budget = cap;
    first += 2;
  }
  const std::vector<std::string_view> args(all.begin() + static_cast<std::ptrdiff_t>(first),
                                           all.end());
  return jitllm::runtime::Run(args, stderr, &Serve);
}
