// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/gemma4_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"
#include "runtime/commands.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"

namespace {
namespace rt = jitllm::runtime;
namespace en = jitllm::engine;
namespace fs = std::filesystem;
using en::support::Error;
constexpr std::uint32_t kVocab = 262144, kPrefix = 8063, kRows = 8192, kOutputs = 129;
constexpr std::uint64_t kHost = 64ULL << 20U, kPinned = 16ULL << 20U;
struct Proof {
  std::string_view mode;
  std::uint32_t owners = 0;
  fs::path input, out;
};
std::unique_ptr<Proof> proof;

template <class T>
rt::Status Write(const fs::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  if (!file) return Error("exclusive proof output refused");
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  return file ? rt::Status{} : Error("complete proof publication failed");
}
struct ServingProof {
  jitllm::config::NodeConfig config;
  jitllm::config::RuntimeRoles roles;
  rt::ServingOptions options;
  std::unique_ptr<rt::Server> server;
  std::vector<std::int32_t> input;
  std::array<rt::Llm::Branch*, 4> branches{};
  std::array<std::vector<float>, 4> frontier, final_head;
  std::array<rt::Generation, 4> output;
  std::array<rt::GenerateOptions, 4> generation;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 4> sessions;
  std::array<std::ofstream, 4> heads;
  std::array<std::uint32_t, 4> rows{};
  rt::Llm* model = nullptr;
  en::Gemma4Runner* runner = nullptr;
  void* pinned = nullptr;
  std::vector<jitllm::catalog::ExtentId> staging;
  std::uint32_t owners = 0;
  bool publication_failed = false;

  rt::Status Export(const fs::path& directory, std::uint32_t positions) {
    auto ranges = runner->CheckpointRanges(positions);
    if (!ranges) return Error(ranges.error());
    if (auto r = Write<char>(directory / "layout.txt", runner->CheckpointLayoutId()); !r) return r;
    for (std::uint32_t owner = 0; owner < owners; ++owner) {
      std::ofstream file(directory / ("state-" + std::to_string(owner) + ".bin"),
                         std::ios::binary | std::ios::noreplace);
      if (!file) return Error("exclusive state proof refused");
      std::uint64_t bytes = 0;
      for (const auto& source : *ranges) {
        for (std::uint64_t at = 0; at < source.bytes;) {
          auto part = source;
          part.offset += at;
          part.bytes = std::min(kPinned, source.bytes - at);
          en::LiveState::CopyRetirement completion = en::LiveState::CopyRetirement::kUnproven;
          auto copied = runner->CopyState(owner, pinned, std::span(&part, 1), true, &completion);
          if (!copied) {
            if (completion == en::LiveState::CopyRetirement::kUnproven)
              server->node().KeepPinned(pinned);
            return copied;
          }
          file.write(static_cast<const char*>(pinned), static_cast<std::streamsize>(part.bytes));
          if (!file) return Error("complete state proof publication failed");
          bytes += part.bytes;
          at += part.bytes;
        }
      }
      file.flush();
      if (!file) return Error("complete state proof flush failed");
      std::fprintf(stdout, "SERVING_STATE owner=%u positions=%u bytes=%llu\n", owner, positions,
                   static_cast<unsigned long long>(bytes));
    }
    return {};
  }

  rt::Status Corpus(std::string_view phase) {
    const fs::path directory = proof->out / phase;
    std::error_code error;
    if (!fs::create_directory(directory, error) || error || chmod(directory.c_str(), 0700) != 0)
      return Error("exclusive corpus phase refused");
    if (auto r = server->SelectRequestBranches(*model, std::span(branches).first(1)); !r) return r;
    if (auto r = branches[0]->Clear(); !r) return r;
    std::ofstream heads_file(directory / "heads-0.f32", std::ios::binary | std::ios::noreplace);
    if (!heads_file) return Error("exclusive corpus heads refused");
    std::uint32_t at = 1;
    rt::PrefillRun run;
    const auto scored = branches[0]->ScorePrompt(
        std::span(input).first(1024), frontier[0],
        [&](std::int32_t token, std::span<const float> row) {
          if (at >= 1024 || input[at] != token || row.size() != kVocab ||
              !std::ranges::all_of(row, [](float value) { return std::isfinite(value); })) {
            publication_failed = true;
            return false;
          }
          heads_file.write(reinterpret_cast<const char*>(row.data()),
                           static_cast<std::streamsize>(row.size_bytes()));
          ++at;
          if (!heads_file) {
            publication_failed = true;
            return false;
          }
          return true;
        },
        {}, &run);
    if (!scored) return scored;
    if (publication_failed || at != 1024 || run.end != 1024 || run.stopped ||
        branches[0]->history().size() != 1024 || frontier[0].size() != kVocab)
      return Error("corpus scorer did not complete exactly 1024 supplied positions");
    heads_file.write(reinterpret_cast<const char*>(frontier[0].data()),
                     static_cast<std::streamsize>(frontier[0].size() * sizeof(float)));
    heads_file.flush();
    if (!heads_file) return Error("corpus final frontier publication failed");
    if (auto r = Export(directory, 1024); !r) return r;
    const auto retired = server->RetireRequestBranches(*model, true);
    if (!retired.result) return retired.result;
    if (!retired.references_retired) return Error("corpus retirement unproven");
    std::fprintf(stdout,
                 "SERVING_CORPUS phase=%.*s owners=1 query_rows=1 rows=1024 targets=1023 "
                 "cursor=1024 extra=%s\n",
                 static_cast<int>(phase.size()), phase.data(), model->extra().c_str());
    return {};
  }

  rt::Status Cycle(std::string_view phase, bool trace, bool timed) {
    const fs::path directory = proof->out / phase;
    std::error_code error;
    if (!fs::create_directory(directory, error) || error || chmod(directory.c_str(), 0700) != 0)
      return Error("exclusive phase directory refused");
    std::array<rt::Llm::GenerationSession*, 4> active{};
    const auto cycle_start = rt::Clock::now();
    if (auto r = server->SelectRequestBranches(*model, std::span(branches).first(owners)); !r)
      return r;
    for (std::uint32_t owner = 0; owner < owners; ++owner) {
      if (auto r = branches[owner]->Clear(); !r) return r;
      const auto tokens = std::span(input).subspan(std::size_t{owner} * kRows, kPrefix);
      if (auto r = branches[owner]->Prefill(tokens, frontier[owner]); !r) return r;
      if (frontier[owner].size() != kVocab) return Error("incomplete serving frontier");
      generation[owner] = {};
      generation[owner].max_tokens = kOutputs;
      generation[owner].stop = false;
      output[owner] = {};
      rows[owner] = 0;
      if (trace) {
        heads[owner].open(directory / ("heads-" + std::to_string(owner) + ".f32"),
                          std::ios::binary | std::ios::noreplace);
        if (!heads[owner]) return Error("exclusive head proof refused");
      }
      generation[owner].on_logits = [this, owner, trace](std::int32_t, std::span<const float> row) {
        if (row.size() != kVocab || (trace && !std::ranges::all_of(row, [](float value) {
                                       return std::isfinite(value);
                                     }))) {
          publication_failed = true;
          return false;
        }
        ++rows[owner];
        if (rows[owner] == kOutputs) final_head[owner].assign(row.begin(), row.end());
        if (trace) {
          heads[owner].write(reinterpret_cast<const char*>(row.data()),
                             static_cast<std::streamsize>(row.size_bytes()));
          if (!heads[owner]) {
            publication_failed = true;
            return false;
          }
        }
        return true;
      };
    }
    const auto prefilled = rt::Clock::now();
    const auto& prefill_policy = runner->last_built_policy();
    if (!timed)
      std::fprintf(stdout,
                   "SERVING_PREFILL phase=%.*s policy_basis=last-built rows=%u segments=%u "
                   "norm_rope=%u norm_add=%u owner=%u\n",
                   static_cast<int>(phase.size()), phase.data(), prefill_policy.rows,
                   prefill_policy.segments, prefill_policy.norm_rope, prefill_policy.norm_add,
                   prefill_policy.owner_attention_steps);
    for (std::uint32_t owner = 0; owner < owners; ++owner) {
      auto begun =
          branches[owner]->BeginGeneration(frontier[owner], generation[owner], output[owner]);
      if (!begun) return Error(begun.error());
      sessions[owner] = std::move(*begun);
      active[owner] = sessions[owner].get();
    }
    for (std::uint32_t wave = 0; wave < 128; ++wave) {
      if (auto r = model->RunGenerationWave(std::span(active).first(owners)); !r) return r;
      if (publication_failed) return Error("natural serving head publication failed");
    }
    for (std::uint32_t owner = 0; owner < owners; ++owner) {
      if (!sessions[owner]->done()) return Error("serving output count did not finish");
      if (auto r = sessions[owner]->Finish(); !r) return r;
      sessions[owner].reset();
    }
    const auto completed = server->RetireRequestBranches(*model, true);
    if (!completed.result) return completed.result;
    if (!completed.references_retired) return Error("serving cycle retirement unproven");
    const auto end = rt::Clock::now();
    const auto& decode_policy = runner->last_built_policy();
    std::fprintf(
        stdout,
        "SERVING_CYCLE phase=%.*s owners=%u emitted=%u waves=128 cursor=8191 pending_position=8191 "
        "timed=%d cycle_seconds=%.9f prefill_seconds=%.9f decode_seconds=%.9f rows=%u segments=%u "
        "norm_rope=%u norm_add=%u owner=%u policy_basis=last-built extra=%s\n",
        static_cast<int>(phase.size()), phase.data(), owners, kOutputs * owners, timed,
        en::support::Seconds(end - cycle_start), en::support::Seconds(prefilled - cycle_start),
        en::support::Seconds(end - prefilled), decode_policy.rows, decode_policy.segments,
        decode_policy.norm_rope, decode_policy.norm_add, decode_policy.owner_attention_steps,
        model->extra().c_str());
    for (std::uint32_t owner = 0; owner < owners; ++owner) {
      if (output[owner].tokens.capacity() > 2 * kOutputs || rows[owner] != kOutputs ||
          output[owner].tokens.size() != kOutputs || output[owner].stopped ||
          output[owner].cancelled || output[owner].steps != 128 ||
          branches[owner]->history().size() != 8191 ||
          !std::ranges::equal(std::span(branches[owner]->history()).first(kPrefix),
                              std::span(input).subspan(std::size_t{owner} * kRows, kPrefix)))
        return Error("serving completed history/output counts differ");
      if (trace) {
        heads[owner].flush();
        heads[owner].close();
        if (heads[owner].fail()) return Error("serving trace flush failed");
      }
      if (auto r = Write<std::int32_t>(directory / ("tokens-" + std::to_string(owner) + ".i32"),
                                       output[owner].tokens);
          !r)
        return r;
      if (auto r = Write<std::int32_t>(directory / ("history-" + std::to_string(owner) + ".i32"),
                                       branches[owner]->history());
          !r)
        return r;
    }
    if (!trace && !timed) return {};
    if (auto r = server->SelectRequestBranches(*model, std::span(branches).first(owners)); !r)
      return r;
    for (std::uint32_t owner = 0; owner < owners; ++owner) {
      if (auto r = Write<float>(directory / ("final-head-" + std::to_string(owner) + ".f32"),
                                final_head[owner]);
          !r)
        return r;
    }
    if (auto r = Export(directory, 8191); !r) return r;
    const auto retired = server->RetireRequestBranches(*model, true);
    return retired.references_retired ? retired.result : Error("final serving references unproven");
  }
};

int ProofServing(const jitllm::config::NodeConfig& config,
                 const jitllm::config::RuntimeRoles& roles, const rt::CommandOptions&, std::FILE*,
                 std::FILE*) {
  auto lifetime = std::make_unique<ServingProof>();
  lifetime->config = config;
  lifetime->roles = roles;
  lifetime->options.plain = true;
  lifetime->options.gemma31_production = true;
  lifetime->owners = proof->owners;
  lifetime->server =
      std::make_unique<rt::Server>(lifetime->config, lifetime->roles, lifetime->options, stderr);
  const auto ran = [&]() -> rt::Status {
    if (config.models.size() != 1) return Error("proof needs exactly one checked target");
    if (auto r = lifetime->server->Start(false); !r) return r;
    lifetime->model = dynamic_cast<rt::Llm*>(lifetime->server->Find(config.models[0].name));
    // `cycles` also measures other Gemma recipes (Gemma26 at its own prefill
    // chunk); the proofs keep the closed dense31 recipe.
    const bool any_recipe = proof->mode == "cycles";
    if (!lifetime->model ||
        (!any_recipe && (!lifetime->model->settings().gemma31_production ||
                         lifetime->model->settings().prefill_chunk.value != 256)) ||
        lifetime->model->settings().context.value != kRows ||
        lifetime->model->settings().max_slots.value != proof->owners)
      return Error("actual serving recipe differs from the closed proof");
    lifetime->runner = dynamic_cast<en::Gemma4Runner*>(&lifetime->model->paged());
    if (!lifetime->runner) return Error("proof did not bind the actual Gemma runner");
    if (!lifetime->server->node().ChargeHost(kHost, false)) return Error("caller funding refused");
    for (std::uint32_t owner = 0; owner < proof->owners; ++owner) {
      lifetime->frontier[owner].reserve(kVocab);
      lifetime->final_head[owner].reserve(kVocab);
      if (lifetime->frontier[owner].capacity() > kVocab ||
          lifetime->final_head[owner].capacity() > kVocab)
        return Error("bounded caller head reservation grew beyond its funding");
    }
    auto pinned = lifetime->server->node().Pinned(kPinned, 0, lifetime->staging);
    if (!pinned) return Error(pinned.error());
    lifetime->pinned = *pinned;
    lifetime->input.resize(4 * kRows);
    if (lifetime->input.capacity() > 4 * kRows)
      return Error("bounded input reservation grew beyond its funding");
    std::ifstream file(proof->input, std::ios::binary);
    file.read(reinterpret_cast<char*>(lifetime->input.data()),
              static_cast<std::streamsize>(lifetime->input.size() * sizeof(std::int32_t)));
    if (!file || file.peek() != std::char_traits<char>::eof() ||
        !std::ranges::all_of(lifetime->input, [](auto id) { return id >= 0 && id < 262144; }))
      return Error("invalid bounded authenticated input carrier");
    for (std::uint32_t owner = 0; owner < 4; ++owner) {
      const auto sequence = std::span(lifetime->input).subspan(std::size_t{owner} * kRows, kRows);
      if (sequence[0] != 2 || std::ranges::count(sequence, 2) != 1)
        return Error("each independent input needs exactly one BOS");
    }
    std::error_code error;
    if (!fs::create_directory(proof->out, error) || error || chmod(proof->out.c_str(), 0700) != 0)
      return Error("exclusive private proof root refused");
    rt::SwapParts swap;
    if (auto r = lifetime->server->Activate(*lifetime->model, swap); !r) return r;
    for (std::uint32_t owner = 0; owner < proof->owners; ++owner) {
      auto branch = lifetime->model->branch(owner);
      if (!branch) return Error(branch.error());
      lifetime->branches[owner] = *branch;
    }
    const auto phase = [&](std::string_view name, bool trace, bool timed) -> rt::Status {
      return lifetime->Cycle(name, trace, timed);
    };
    if (proof->mode == "corpus") {
      if (auto r = lifetime->Corpus("first"); !r) return r;
      if (auto r = lifetime->Corpus("repeat"); !r) return r;
    } else if (proof->mode == "quality") {
      if (auto r = phase("first", true, false); !r) return r;
      if (auto r = phase("repeat", true, false); !r) return r;
    } else if (proof->mode == "cycles") {
      for (const char* name : {"warm", "second", "third"}) {
        const auto before = lifetime->runner->graph_stats();
        if (auto r = phase(name, false, false); !r) return r;
        const auto& after = lifetime->runner->graph_stats();
        std::fprintf(
            stdout,
            "SERVING_GRAPHS phase=%s eager=%llu captured=%llu replayed=%llu capture_s=%.6f "
            "instantiate_s=%.6f\n",
            name, static_cast<unsigned long long>(after.eager - before.eager),
            static_cast<unsigned long long>(after.captured - before.captured),
            static_cast<unsigned long long>(after.replayed - before.replayed),
            after.capture_seconds - before.capture_seconds,
            after.instantiate_seconds - before.instantiate_seconds);
      }
    } else {
      if (auto r = phase("warm", false, false); !r) return r;
      if (auto r = phase("paid", false, true); !r) return r;
    }
    if (auto r = lifetime->server->node().FreePinned(lifetime->pinned); !r) return r;
    lifetime->pinned = nullptr;
    std::vector<std::int32_t>().swap(lifetime->input);
    std::vector<jitllm::catalog::ExtentId>().swap(lifetime->staging);
    for (std::uint32_t owner = 0; owner < proof->owners; ++owner) {
      std::vector<float>().swap(lifetime->frontier[owner]);
      std::vector<float>().swap(lifetime->final_head[owner]);
      lifetime->output[owner] = {};
      lifetime->generation[owner] = {};
    }
    lifetime->server->node().UnchargeHost(kHost);
    if (auto r = lifetime->server->TearDown(); !r) return r;
    return {};
  }();
  if (!ran) {
    std::fprintf(stderr, "serving proof failed: %s\n", ran.error().c_str());
    // A failed fence may retain GPU pointers and borrowed configuration.
    // Retain sessions, callbacks, node, state and host owners until process exit.
    (void)lifetime.release();
    return rt::kExitFailure;
  }
  std::fprintf(stdout, "SERVING_PROOF_RETIRED\n");
  return rt::kExitOk;
}

// Before this program's other initializers: non-dumpable early, in case
// something faults before main() installs the handlers. Shared libraries'
// initializers (the NVIDIA driver's libcuda.so.1) and the loader still run
// before it. Global constructors in jitLLM do no work that can fault
// (architecture.md, layers: nothing at static initialization).
// A failure here is repeated, and reported, by InstallCrashPolicy.
[[gnu::constructor(101)]] void NonDumpableFromTheStart() {
  if (!jitllm::platform::MarkNonDumpable()) {
    return;
  }
}

int ProductionServing(const jitllm::config::NodeConfig& config,
                      const jitllm::config::RuntimeRoles& roles,
                      const jitllm::runtime::CommandOptions& command, std::FILE* out,
                      std::FILE* log) {
  if (command.command == jitllm::runtime::Command::kService)
    return jitllm::runtime::RunService(config, roles, log, false, false, true);
  // Candidate only: the production executable keeps this control false.
  auto selected = command;
  selected.serving.gemma31_production = true;
  return jitllm::runtime::RunServing(config, roles, selected, out, log);
}

}  // namespace

int main(int argc, char** argv) {
  // First: a crash leaves no core image (D-014).
  if (auto policy = jitllm::platform::InstallCrashPolicy("jitllm-runtime"); !policy) {
    (void)std::fprintf(stderr, "jitllm-runtime: %s\n", policy.error().c_str());
    return jitllm::runtime::kExitFailure;
  }
  // A closed standard descriptor would be taken by the next file opened.
  for (int fd = 0; fd <= 2; ++fd) {
    if (::fcntl(fd, F_GETFD) == -1 && errno == EBADF &&
        ::open("/dev/null", O_RDWR | O_NOCTTY) != fd) {
      return jitllm::runtime::kExitFailure;
    }
  }
  // The stop signals are waited for, not handled; blocked before any
  // thread starts, so every thread inherits the mask.
  sigset_t stop;
  (void)::sigemptyset(&stop);
  (void)::sigaddset(&stop, SIGTERM);
  (void)::sigaddset(&stop, SIGINT);
  (void)::sigaddset(&stop, SIGHUP);
  (void)::sigaddset(&stop, SIGCHLD);
  (void)::pthread_sigmask(SIG_BLOCK, &stop, nullptr);
  (void)::signal(SIGPIPE, SIG_IGN);
  // The runtime compiles nothing on the GPU (its code is SASS), so the
  // driver has no reason to keep a JIT cache in the service user's home,
  // the data directory. Nothing has started a thread yet.
  (void)::setenv("CUDA_CACHE_DISABLE", "1", 1);  // NOLINT(concurrency-mt-unsafe)
  if (argc >= 6 && std::string_view(argv[1]) == "proof") {
    std::uint32_t owners = 0;
    const std::string_view supplied(argv[3]);
    const auto [last, error] =
        std::from_chars(supplied.data(), supplied.data() + supplied.size(), owners);
    if (error != std::errc{} || last != supplied.data() + supplied.size() ||
        (owners != 1 && owners != 4) ||
        (std::string_view(argv[2]) != "quality" && std::string_view(argv[2]) != "cycle" &&
         std::string_view(argv[2]) != "corpus" && std::string_view(argv[2]) != "cycles") ||
        (std::string_view(argv[2]) == "corpus" && owners != 1))
      return rt::kExitUsage;
    proof = std::make_unique<Proof>(Proof{argv[2], owners, argv[4], argv[5]});
    const std::span<char*> remaining(argv + 6, static_cast<std::size_t>(argc - 6));
    const std::vector<std::string_view> parsed(remaining.begin(), remaining.end());
    return rt::Run(parsed, stderr, &ProofServing);
  }
  const std::span<char*> all(argv, static_cast<std::size_t>(argc));
  const std::vector<std::string_view> args(all.begin() + (argc > 0 ? 1 : 0), all.end());
  return jitllm::runtime::Run(args, stderr, &ProductionServing);
}
