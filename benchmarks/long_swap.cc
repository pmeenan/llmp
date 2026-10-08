// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Replay the same long prefill, then compare uninterrupted and swapped
// continuations. No whole-state pinned diagnostic copy is needed at the ceiling.
#include <unistd.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "runtime/serving.h"

namespace {
namespace rt = llmp::runtime;
namespace cf = llmp::config;
using rt::Status;

double Seconds(rt::Clock::duration elapsed) {
  return std::chrono::duration<double>(elapsed).count();
}

bool Same(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

bool SameGeneration(const rt::Generation& a, const rt::Generation& b) {
  if (a.tokens != b.tokens || a.logits.size() != b.logits.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.logits.size(); ++i) {
    if (!Same(a.logits[i], b.logits[i])) {
      return false;
    }
  }
  return true;
}

std::string Hash(std::span<const float> row) {
  return llmp::base::ToHex(llmp::base::Sha256().Update(std::as_bytes(row)).Finish());
}

std::string Parts(const rt::SwapParts& p) {
  return std::format(R"({{"total_s":{:.6f},"evict_s":{:.6f},"restore_s":{:.6f},"page_in_s":{:.6f},)"
                     R"("setup_s":{:.6f},"release_s":{:.6f},"spilled_bytes":{},"read_bytes":{},)"
                     R"("loaded_extents":{},"evicted_extents":{}}})",
                     p.total, p.evict, p.restore, p.page_in, p.setup, p.release, p.spilled_bytes,
                     p.read_bytes, p.loaded, p.evicted);
}

Status Run(std::span<const std::string_view> args) {
  std::uint32_t count = 0;
  const auto number = std::from_chars(args[4].data(), args[4].data() + args[4].size(), count);
  if (number.ec != std::errc{} || number.ptr != args[4].data() + args[4].size() || count == 0) {
    return std::unexpected("invalid prompt token count");
  }
  auto config = cf::LoadNodeConfig({.main_file = std::filesystem::path(args[0]),
                                    .main_file_optional = false,
                                    .anchor = std::filesystem::path(args[1]),
                                    .trusted_uid = getuid()});
  if (!config) {
    std::string errors;
    for (const auto& error : config.error()) {
      errors += cf::FormatDiagnostic(error) + "\n";
    }
    return std::unexpected(errors);
  }
  auto roles = cf::PrepareRuntimeRoles(config->storage, getuid(), std::filesystem::path(args[1]));
  if (!roles) {
    return std::unexpected(roles.error().front());
  }
  rt::ServingOptions serving;
  rt::Server server(*config, *roles, serving, stderr);
  if (auto started = server.Start(false); !started) {
    return started;
  }
  auto* target = server.Find(args[2]);
  auto* other = server.Find(args[5]);
  if (target == nullptr || other == nullptr || !target->llm() || !other->llm() || target == other) {
    return std::unexpected("two different configured LLMs are required");
  }
  auto& llm = static_cast<rt::Llm&>(*target);
  auto& alternate = static_cast<rt::Llm&>(*other);
  constexpr std::uint32_t kOutputs = 64;
  if (llm.usable_context() <= kOutputs || count > llm.usable_context() - kOutputs) {
    return std::unexpected("prompt leaves too little usable context for the continuation");
  }
  // Check the size before allocating the file's contents.
  std::error_code file_error;
  const auto bytes = std::filesystem::file_size(std::filesystem::path(args[3]), file_error);
  if (file_error || bytes == 0 || bytes > (std::uintmax_t{32} << 20U)) {
    return std::unexpected("prompt text is empty, unavailable or exceeds 32 MiB");
  }
  std::ifstream file(std::filesystem::path(args[3]), std::ios::binary);
  std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file.eof() && file.fail()) {
    return std::unexpected("cannot read prompt text");
  }
  auto input = llm.EncodeText(text);
  if (!input || input->size() < count) {
    return std::unexpected("prompt text does not supply enough tokens");
  }
  input->resize(count);
  const rt::GenerateOptions options{.max_tokens = kOutputs,
                                    .stop = false,
                                    .keep_logits = true,
                                    .sampling = {},
                                    .seed = 0,
                                    .on_tokens = {}};
  // Prepare the alternate once so the long return tests a prepared swap.
  rt::SwapParts warm;
  if (auto active = server.Activate(alternate, warm); !active) {
    return active;
  }
  auto short_input = alternate.EncodeText("What is the capital of France?");
  if (!short_input) {
    return std::unexpected(short_input.error());
  }
  if (auto request =
          server.InRequest(alternate,
                           [&]() -> Status {
                             if (auto clear = alternate.Clear(); !clear) {
                               return clear;
                             }
                             std::vector<float> last;
                             if (auto pref = alternate.Prefill(*short_input, last); !pref) {
                               return pref;
                             }
                             rt::Generation ignored;
                             const rt::GenerateOptions short_options{.max_tokens = 8,
                                                                     .stop = false,
                                                                     .keep_logits = false,
                                                                     .sampling = {},
                                                                     .seed = 0,
                                                                     .on_tokens = {}};
                             return alternate.Generate(last, short_options, ignored);
                           });
      !request) {
    return request;
  }
  if (auto finished = server.FinishSwap(warm); !finished) {
    return finished;
  }
  // Drop the partner's tiny conversation before loading the long target.
  alternate.Forget();
  rt::SwapParts load;
  if (auto active = server.Activate(llm, load); !active) {
    return active;
  }
  std::vector<float> control_last;
  rt::Generation control;
  double control_prefill = 0;
  rt::PrefillRun control_run;
  if (auto request = server.InRequest(
          llm,
          [&]() -> Status {
            if (auto clear = llm.Clear(); !clear) {
              return clear;
            }
            const auto start = rt::Clock::now();
            if (auto pref = llm.Prefill(*input, control_last, {}, &control_run); !pref) {
              return pref;
            }
            control_prefill = Seconds(rt::Clock::now() - start);
            return llm.Generate(control_last, options, control);
          });
      !request) {
    return request;
  }
  if (auto finished = server.FinishSwap(load); !finished) {
    return finished;
  }
  std::vector<float> repeat_last;
  double repeat_prefill = 0;
  rt::PrefillRun repeat_run;
  if (auto request = server.InRequest(
          llm,
          [&]() -> Status {
            if (auto clear = llm.Clear(); !clear) {
              return clear;
            }
            const auto start = rt::Clock::now();
            if (auto pref = llm.Prefill(*input, repeat_last, {}, &repeat_run); !pref) {
              return pref;
            }
            repeat_prefill = Seconds(rt::Clock::now() - start);
            return {};
          });
      !request) {
    return request;
  }
  const std::string saved = llm.extra();
  rt::SwapParts away;
  if (auto active = server.Activate(alternate, away); !active) {
    return active;
  }
  if (auto finished = server.FinishSwap(away); !finished) {
    return finished;
  }
  rt::SwapParts back;
  if (auto active = server.Activate(llm, back); !active) {
    return active;
  }
  rt::Generation restored;
  if (auto request = server.InRequest(
          llm, [&]() -> Status { return llm.Generate(repeat_last, options, restored); });
      !request) {
    return request;
  }
  if (auto finished = server.FinishSwap(back); !finished) {
    return finished;
  }
  const bool prefill_exact = Same(control_last, repeat_last);
  const bool continuation_exact = SameGeneration(control, restored);
  (void)std::fprintf(
      stdout, "%s\n",
      std::format(R"({{"model":"{}","partner":"{}","prompt_tokens":{},"completion_tokens":{},)"
                  R"("speculative":{},"control_prefill_s":{:.6f},"repeat_prefill_s":{:.6f},)"
                  R"("control_chunks":{},"repeat_chunks":{},"control_longest_chunk_s":{:.6f},)"
                  R"("repeat_longest_chunk_s":{:.6f},)"
                  R"("control_decode_s":{:.6f},"restored_decode_s":{:.6f},"prefill_exact":{},)"
                  R"("continuation_exact":{},"control_logits_sha":"{}","repeat_logits_sha":"{}",)"
                  R"("fixed_bytes":{},"host_input_bytes":{},"budget_bytes":{},"peak_bytes":{},)"
                  R"("minimum_available_bytes":{},"saved":{},"load":{},"away":{},"back":{}}})",
                  llm.name(), alternate.name(), count, restored.tokens.size(), llm.speculative(),
                  control_prefill, repeat_prefill, control_run.chunks, repeat_run.chunks,
                  control_run.longest, repeat_run.longest, control.decode_seconds,
                  restored.decode_seconds, prefill_exact, continuation_exact, Hash(control_last),
                  Hash(repeat_last), server.fixed_bytes(), server.host_input_bytes(),
                  server.budget(), server.memory().peak(), server.memory().all(), saved,
                  Parts(load), Parts(away), Parts(back))
          .c_str());
  (void)std::fflush(stdout);
  if (!prefill_exact || !continuation_exact || restored.tokens.size() != kOutputs) {
    return std::unexpected("long swap or repeated prefill differs from uninterrupted control");
  }
  return server.TearDown();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    (void)std::fprintf(stderr, "usage: long_swap CONFIG ANCHOR MODEL TEXT TOKENS SWAP_MODEL\n");
    return 2;
  }
  std::vector<std::string_view> args;
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  auto result = Run(args);
  if (!result) {
    (void)std::fprintf(stderr, "%s\n", result.error().c_str());
    return 1;
  }
  return 0;
}
