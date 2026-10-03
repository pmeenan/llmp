// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The runtime's serving commands (D-096; docs/runtime-serving.md): after the
// startup steps (runtime.h), instead of waiting as the service does, the
// process registers the configured models on its node and runs one command
// in its own process, then stops:
//
//   chat        each --turn MODEL TEXT in order: the model made resident
//               (a full swap when another is), the text rendered as the next
//               user message of that model's conversation with its chat
//               template, and a reply generated (speculatively where the
//               model has a drafter), printed with the swap's parts and the
//               turn's speeds;
//   swap-table  M3's swap table (docs/plan.md): every ordered pair of the
//               configured models, or those named, A→B→A in this process,
//               first use and prepared, with 8K and 0 tokens of saved
//               context, each part of each swap timed, the endpoints and
//               checks of plan.md's table.
//
// Both are local: they run in the runtime's own process, holding its
// process lock, and open no listener (D-014). Parsing is vendor-free; the
// commands themselves need a CUDA build (serving.h).

#ifndef JITLLM_RUNTIME_COMMANDS_H_
#define JITLLM_RUNTIME_COMMANDS_H_

#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config/node_config.h"
#include "config/storage_roles.h"

namespace jitllm::runtime {

// The commands' turns, texts and pairs have no count or byte caps of their
// own (D-102): the command line's own limits bound them (Linux: an argument
// of at most 128 KiB, all of them at most a quarter of the stack limit),
// and --max-tokens and --context-tokens are checked against the model's
// context when they run. The image prompt keeps its bound until the image
// pipeline's own sizes are settled (M3.5's image work).
inline constexpr std::size_t kMaxImagePromptBytes = std::size_t{64} << 10U;
inline constexpr std::string_view kDefaultImagePrompt =
    "A red ceramic teapot on a plain wooden table, soft daylight, no text.";
inline constexpr std::string_view kDefaultShortPrompt =
    "What is the capital of France? Answer in one sentence.";

enum class Command : std::uint8_t { kService, kChat, kSwapTable };

struct Turn {
  std::string model;
  std::string text;
};

// What every serving command shares.
struct ServingOptions {
  // Decode plainly even where a model has a drafter.
  bool plain = false;
  // The image pipeline's generation, fixed when it registers (this slice's
  // image runner takes its prompt at setup): its prompt, and its initial
  // latents (BF16, the reference's for its seed), without which an image
  // model is not registered.
  std::string image_prompt{kDefaultImagePrompt};
  std::filesystem::path image_noise;
  // A JSON report of every number, written at the end (optional). It holds
  // prompts' token counts and the generated text only for chat.
  std::filesystem::path report;
};

struct ChatOptions {
  std::vector<Turn> turns;
  std::uint32_t max_tokens = 256;
  bool ignore_stop = false;  // generate max_tokens whatever the stop tokens
  bool fresh = false;        // every turn starts a new conversation
};

struct SwapTableOptions {
  // Ordered pairs of configured models' names; empty: every ordered pair.
  std::vector<std::pair<std::string, std::string>> pairs;
  std::filesystem::path context_text;  // an LLM A's context, tokenized by A
  std::uint32_t context_tokens = 8192;
  std::uint32_t continue_tokens = 16;
  std::uint32_t cycles = 2;  // the first is first use, the rest prepared
  bool zero_context = true;
  bool handoff = true;
  std::string short_prompt{kDefaultShortPrompt};  // an LLM B's, from a cleared state
  std::string image_expect;                       // the image's pixels' SHA-256, if known
};

struct CommandOptions {
  Command command = Command::kService;
  ServingOptions serving;
  ChatOptions chat;
  SwapTableOptions table;
};

// Parses a command's arguments (after its name).
std::expected<CommandOptions, std::string> ParseCommand(std::string_view name,
                                                        std::span<const std::string_view> args);

// The commands' usage text.
extern const std::string_view kCommandUsage;

// Runs a serving command once the startup steps have passed: registers the
// configured models on the node, runs the command (its output to `out`,
// the runtime's log lines to `log`), and tears the node down. Returns the
// process's exit status (runtime.h). A build without CUDA refuses.
int RunServing(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const CommandOptions& command, std::FILE* out, std::FILE* log);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_COMMANDS_H_
