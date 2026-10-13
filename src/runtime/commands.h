// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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
//
//   settings    every configured model's settings as registration resolves
//               them (D-103, model_settings.h), each with its source; with
//               --json one JSON object. It only reads the configuration,
//               the installed artifacts and the calibration records: no
//               process lock (so it runs beside the service), no device
//               context or memory, any build.

#ifndef LLMP_RUNTIME_COMMANDS_H_
#define LLMP_RUNTIME_COMMANDS_H_

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

namespace llmp::runtime {

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

enum class Command : std::uint8_t { kService, kChat, kSwapTable, kSettings };

struct Turn {
  std::string model;
  std::string text;
};

// What every serving command shares.
struct ServingOptions {
  // Decode plainly even where a model has a drafter.
  bool plain = false;
  // Internal Qwen component migration screen, including declared composed
  // sources with multiple owners. Only an explicitly funded service diagnostic
  // can select it through HTTP; normal configuration/defaults are unchanged.
  bool diagnostic_component_solo = false;
  // Internal default-off experiment, only with the paid shared service route.
  bool diagnostic_capture_first_verify = false;
  // Prepaid host compiler/result allowance for bounded startup screens.
  // Cataloged before artifact metadata or Setup, retained until node death.
  // Requires a known physical budget cap; zero leaves ordinary starts alone.
  std::uint64_t diagnostic_startup_host_bytes = 0;
  // Internal auto mode derives a parser loan and actual source construction.
  bool diagnostic_auto_startup = false;
  // Internal matched plain-token qualification; absent from CLI/TOML/HTTP.
  // Unset uses each runner's ordinary policy.
  std::optional<bool> diagnostic_plain_device_tokens;
  // The image pipeline's generation, fixed when it registers (this slice's
  // image runner takes its prompt at setup): its prompt, and its initial
  // latents (BF16, the reference's for its seed), without which an image
  // model is not registered.
  std::string image_prompt{kDefaultImagePrompt};
  std::filesystem::path image_noise;
  // A JSON report of every number, written at the end (optional). It holds
  // prompts' token counts and the generated text only for chat.
  std::filesystem::path report;
  // Conversations kept across a restart (D-105): the service keeps them;
  // the commands, which measure from a known start, leave what the service
  // kept alone and spill to unnamed files as before.
  bool keep_conversations = false;
  // A test hook (D-102's hang recovery; engine/paged_node.h
  // CountingStorage): reads held by Server::HoldReads complete as
  // cancelled when the lane cancels them. Otherwise, as a drive's, they
  // do not.
  bool hold_cancellable = false;
  // Internal maximum-context qualification only; no config/CLI/HTTP route.
  bool gemma3_trained_max = false;
  // Swaps' handoff evictions keep backing mapped until a load takes it
  // (engine::NodeSettings::lazy_handoff); off is an internal override for
  // matched controls only.
  bool lazy_handoff = true;
  // Internal full/partial swap factor; remains off until measured qualification.
  bool partial_weight_eviction = false;
  // Internal same-capacity swap controls only; no production config/CLI option.
  // Refused unless it fits the ordinary startup guard and required footprint.
  std::optional<std::uint64_t> diagnostic_budget_cap_bytes;
  // engine::NodeSettings::zero_state and handle_reserve; internal
  // overrides for matched controls only.
  bool zero_state = true;
  std::optional<std::size_t> handle_reserve;  // unset: the node's default
  // Selected scalar DS/Qwen prediction; no public setting or output-policy
  // change. Both mechanisms have independent internal matched controls.
  bool dsv4_prefill_lookahead = true, qwen38_prefill_lookahead = true;
  bool dsv4_prepare_state = true, qwen38_prepare_state = true;
  // Unset selects ordinary non-speculative prefill only; required-stream
  // drafter consumers remain explicit opt-ins.
  std::optional<bool> dsv4_state_only_prefill, qwen38_state_only_prefill;
  // Hint-delivery control for matched DS/Qwen tests with identical provisioning.
  // It changes no required work.
  bool predicted_prefill_hints = true;
  // Internal same-provisioning T54 control: false forces current/future heads.
  // Scoring remains headed and public settings never expose this override.
  bool prefill_output_intent = true;
  // Gate-only (owner, 2026-10-09): the component engine's models save no
  // turn checkpoint (Llm::set_turn_checkpoints), so M3.6's speed gate
  // compares equal work with a baseline whose M3 classes never saved one
  // (docs/experiments/m36-baselines/README.md). Kept for later families'
  // gates. Absent from configuration; the service sets it from
  // LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1, as its test hooks.
  bool diagnostic_skip_turn_checkpoints = false;
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
  bool json = false;  // settings --json
};

// Internal supervised service qualification only. The three environment values
// are all absent or all valid; an optional first-Verify capture selector requires
// that complete tuple. Refusal leaves options unchanged. Funding is
// derived and source/config-bound by the caller before starting the process.
std::expected<void, std::string> ApplyServiceDiagnostics(
    ServingOptions& options, std::optional<std::string_view> component,
    std::optional<std::string_view> startup_bytes, std::optional<std::string_view> node_cap,
    std::optional<std::string_view> capture_first_verify = std::nullopt);

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

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_COMMANDS_H_
