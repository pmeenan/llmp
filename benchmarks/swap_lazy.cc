// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The production runtime's swap-table with lazy handoff chosen:
// `jitllm_swap_lazy on|off RUNTIME-ARGUMENTS...`, the rest exactly as
// jitllm-runtime takes them (`--config ... swap-table ...`). A matched
// control for ServingOptions::lazy_handoff, which configuration and the
// CLI do not expose.
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

#include "runtime/commands.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"

namespace {
bool g_lazy = false;

int Serve(const jitllm::config::NodeConfig& config, const jitllm::config::RuntimeRoles& roles,
          const jitllm::runtime::CommandOptions& command, std::FILE* out, std::FILE* log) {
  auto chosen = command;
  chosen.serving.lazy_handoff = g_lazy;
  (void)std::fprintf(log, "jitllm_swap_lazy: lazy_handoff=%s\n", g_lazy ? "on" : "off");
  return jitllm::runtime::RunServing(config, roles, chosen, out, log);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || (std::string_view(argv[1]) != "on" && std::string_view(argv[1]) != "off")) {
    (void)std::fprintf(stderr, "usage: jitllm_swap_lazy on|off RUNTIME-ARGUMENTS...\n");
    return 2;
  }
  g_lazy = std::string_view(argv[1]) == "on";
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
  const std::vector<std::string_view> args(all.begin() + 2, all.end());
  return jitllm::runtime::Run(args, stderr, &Serve);
}
