// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Native turn rollback with identical prefill chunk boundaries as its fresh
// control. Raw output stays external; see docs/experiments/turn-reuse/.
#include "runtime/turn_reuse.h"

#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "chat/chat.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "runtime/serving.h"

namespace {
namespace rt = llmp::runtime;
namespace cf = llmp::config;
namespace ch = llmp::chat;
using rt::Status;

double Seconds(rt::Clock::duration elapsed) {
  return std::chrono::duration<double>(elapsed).count();
}

std::string Hash(std::span<const float> row) {
  return llmp::base::ToHex(llmp::base::Sha256().Update(std::as_bytes(row)).Finish());
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
  auto* model = server.Find(args[2]);
  if (model == nullptr || !model->llm()) {
    return std::unexpected("requested model is not a configured LLM");
  }
  auto& llm = static_cast<rt::Llm&>(*model);
  if (count >= llm.usable_context() - 256) {
    return std::unexpected("prompt leaves too little context for the test turns");
  }
  std::ifstream file(std::filesystem::path(args[3]), std::ios::binary);
  if (!file) {
    return std::unexpected("cannot open prompt text");
  }
  std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (text.empty() || text.size() > (std::size_t{16} << 20U)) {
    return std::unexpected("prompt text is empty or exceeds 16 MiB");
  }
  auto input = llm.EncodeText(text);
  if (!input || input->size() < count) {
    return std::unexpected("prompt text does not supply enough tokens");
  }
  const std::string content = llm.Detokenize(std::span(*input).first(count));
  ch::Conversation first;
  llm.Defaults(first);
  first.messages.push_back(
      {.role = ch::Role::kUser, .content = content, .reasoning_content = {}, .tool_calls = {}});
  std::uint32_t first_boundary = 0;
  auto first_tokens = llm.RenderChat(first, &first_boundary);
  ch::Conversation second = first;
  // The API client returns only the visible answer, without reasoning.
  second.messages.push_back({.role = ch::Role::kAssistant,
                             .content = "I inspected the code.",
                             .reasoning_content = {},
                             .tool_calls = {}});
  second.messages.push_back({.role = ch::Role::kUser,
                             .content = "List three concrete checks for the implementation.",
                             .reasoning_content = {},
                             .tool_calls = {}});
  std::uint32_t second_boundary = 0;
  auto second_tokens = llm.RenderChat(second, &second_boundary);
  if (!first_tokens || !second_tokens || first_boundary == 0 || second_boundary <= first_boundary ||
      rt::CommonPrefix(*first_tokens, *second_tokens) < first_boundary) {
    return std::unexpected("renderer did not supply the shared stable turn boundary");
  }
  rt::SwapParts load;
  if (auto activated = server.Activate(llm, load); !activated) {
    return activated;
  }
  const rt::GenerateOptions options{.max_tokens = 32,
                                    .stop = false,
                                    .keep_logits = true,
                                    .sampling = {},
                                    .seed = 0,
                                    .on_tokens = {}};
  std::vector<float> control_logits;
  rt::Generation control;
  double fresh_seconds = 0;
  auto fresh = server.InRequest(llm, [&]() -> Status {
    if (auto cleared = llm.Clear(); !cleared) {
      return cleared;
    }
    const auto start = rt::Clock::now();
    if (auto pref = llm.Prefill(std::span(*second_tokens).first(first_boundary), control_logits);
        !pref) {
      return pref;
    }
    if (auto pref = llm.Prefill(
            std::span(*second_tokens).subspan(first_boundary, second_boundary - first_boundary),
            control_logits);
        !pref) {
      return pref;
    }
    if (auto pref = llm.Prefill(std::span(*second_tokens).subspan(second_boundary), control_logits);
        !pref) {
      return pref;
    }
    fresh_seconds = Seconds(rt::Clock::now() - start);
    return llm.Generate(control_logits, options, control);
  });
  if (!fresh) {
    return fresh;
  }
  std::uint32_t first_reused = 0;
  double initial_seconds = 0;
  auto initial = server.InRequest(llm, [&]() -> Status {
    std::vector<float> last;
    const auto start = rt::Clock::now();
    if (auto pref =
            llm.PreparePrompt(*first_tokens, first_boundary, last, first_reused, {}, nullptr, true);
        !pref) {
      return pref;
    }
    initial_seconds = Seconds(rt::Clock::now() - start);
    rt::Generation ignored;
    return llm.Generate(last, options, ignored);
  });
  if (!initial) {
    return initial;
  }
  const auto checkpoint_bytes = llm.turn_checkpoint_bytes();
  if (auto finished = server.FinishSwap(load); !finished) {
    return finished;
  }
  std::vector<float> restored_logits;
  rt::Generation restored;
  std::uint32_t reused = 0;
  double reuse_seconds = 0;
  auto resumed = server.InRequest(llm, [&]() -> Status {
    const auto start = rt::Clock::now();
    if (auto pref = llm.PreparePrompt(*second_tokens, second_boundary, restored_logits, reused);
        !pref) {
      return pref;
    }
    reuse_seconds = Seconds(rt::Clock::now() - start);
    return llm.Generate(restored_logits, options, restored);
  });
  if (!resumed) {
    return resumed;
  }
  bool swapped = false;
  if (args.size() == 6) {
    auto* other = server.Find(args[5]);
    if (other == nullptr || !other->llm() || other == model) {
      return std::unexpected("swap partner must be a different configured LLM");
    }
    auto& alternate = static_cast<rt::Llm&>(*other);
    rt::SwapParts away;
    if (auto activated = server.Activate(alternate, away); !activated) {
      return activated;
    }
    auto other_tokens = alternate.EncodeText("What is the capital of France?");
    if (!other_tokens) {
      return std::unexpected(other_tokens.error());
    }
    if (auto pref = server.InRequest(alternate,
                                     [&]() -> Status {
                                       if (auto cleared = alternate.Clear(); !cleared) {
                                         return cleared;
                                       }
                                       std::vector<float> last;
                                       return alternate.Prefill(*other_tokens, last);
                                     });
        !pref) {
      return pref;
    }
    if (auto finished = server.FinishSwap(away); !finished) {
      return finished;
    }
    rt::SwapParts back;
    if (auto activated = server.Activate(llm, back); !activated) {
      return activated;
    }
    if (auto finished = server.FinishSwap(back); !finished) {
      return finished;
    }
    swapped = true;
  }
  // Repeating the second request must choose its newer checkpoint, including
  // after a full swap, and preserve both speculation and recurrent state.
  std::vector<float> repeated_logits;
  rt::Generation repeated;
  std::uint32_t repeated_reused = 0;
  double repeated_seconds = 0;
  auto repeat = server.InRequest(llm, [&]() -> Status {
    const auto start = rt::Clock::now();
    if (auto pref =
            llm.PreparePrompt(*second_tokens, second_boundary, repeated_logits, repeated_reused);
        !pref) {
      return pref;
    }
    repeated_seconds = Seconds(rt::Clock::now() - start);
    return llm.Generate(repeated_logits, options, repeated);
  });
  if (!repeat) {
    return repeat;
  }
  const bool exact = Same(control_logits, restored_logits) && SameGeneration(control, restored);
  const bool repeat_exact =
      Same(restored_logits, repeated_logits) && SameGeneration(restored, repeated);
  const std::string report = std::format(
      R"({{"model":"{}","first_tokens":{},"second_tokens":{},"stable_boundary":{},)"
      R"("reused":{},"repeat_reused":{},"fresh_seconds":{:.6f},"initial_seconds":{:.6f},)"
      R"("reuse_seconds":{:.6f},"repeat_seconds":{:.6f},"checkpoint_bytes":{},)"
      R"("checkpoints":{},"total_checkpoint_bytes":{},"fresh_logits_sha":"{}",)"
      R"("restored_logits_sha":"{}","exact":{},"repeat_exact":{},"swapped":{},)"
      R"("peak_gib":{:.6f}}})",
      llm.name(), first_tokens->size(), second_tokens->size(), first_boundary, reused,
      repeated_reused, fresh_seconds, initial_seconds, reuse_seconds, repeated_seconds,
      checkpoint_bytes, llm.turn_checkpoints(), llm.turn_checkpoint_bytes(), Hash(control_logits),
      Hash(restored_logits), exact ? "true" : "false", repeat_exact ? "true" : "false",
      swapped ? "true" : "false", static_cast<double>(server.memory().peak()) / (1ULL << 30U));
  (void)std::fprintf(stdout, "%s\n", report.c_str());
  (void)std::fflush(stdout);
  if (!exact || !repeat_exact || first_reused != 0 || reused != first_boundary ||
      repeated_reused != second_boundary || llm.turn_checkpoints() != 2) {
    return std::unexpected(
        "turn checkpoint differs from its fresh replay or nearest-boundary control");
  }
  return server.TearDown();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    (void)std::fprintf(stderr, "usage: turn_reuse CONFIG ANCHOR MODEL TEXT TOKENS [SWAP_MODEL]\n");
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
