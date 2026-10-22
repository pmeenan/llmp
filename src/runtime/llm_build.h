// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_RUNTIME_LLM_BUILD_H_
#define LLMP_RUNTIME_LLM_BUILD_H_
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "engine/decoding.h"
#include "engine/model_runner.h"
#include "engine/prefill.h"
#include "execution/adaptive_wave_mode.h"
#include "model/batching.h"
#include "runtime/model_settings.h"

namespace llmp::runtime {

// What an engine-backed LLM runs, made from its settings.
struct LlmBuildParts {
  // What the step source and the launch borrow (the spec, the artifact's
  // tensors, a kernel source's backend): kept until the runner is gone.
  std::shared_ptr<void> keep;
  // The runner: the slots (each stage's layout, sized by the settings'
  // context and prefill chunk; the bound spec's digest, D-105's layout
  // identity), the stages' weights, setup shapes, launch, verify rows and
  // commits. Llm sets the slots' model name, count (max_slots), spill
  // directory and places, and chains its own invalidation.
  engine::RunnerConfig config;
  std::unique_ptr<engine::StepSource> source;
  model::Batching caps;  // the components' batching capability
  // The decoding mode, with its drafters when the model has any (they
  // draft only with the speculation setting on); kPlain otherwise.
  engine::DecodingMode mode;
  // Optional existing counted wave-form policy. Unset preserves the source
  // recipe: every speculative session uses its declared drafter schedule.
  std::optional<execution::AdaptiveWaveMode> wave_mode{};
  std::uint32_t logit_width = 0;  // a head row's F32 logits
  // A plain greedy step returns its tokens chosen on the device.
  bool device_greedy = false;
  // A greedy verify may return one native I32 token per output row.
  bool device_verify = false;
  // The sole drafter's adaptive schedule lives with the conversation's
  // existing persisted AdaptiveDepth, rather than the orchestrator instance.
  std::uint32_t branch_depth = 0;  // persisted policy ceiling; 0 uses the ordinary bound
  // Lookahead, capture ahead and prepare ahead of prompt chunks (the chunk
  // rows are the settings').
  engine::PrefillOptions prefill;
  // Its family's name for the report ("gemma2").
  std::string architecture;
  // What of its chat template the chat route serves (the model's qualified
  // slice): thought-channel output off by default and refused on, tool
  // declarations refused, and control tokens that end a chat reply beside
  // the template's own stops (Gemma 4's channel and tool-call openers,
  // which no parser here reads). `default_thinking` is a request's
  // thinking when it names none and thinking is served (DeepSeek V4's on,
  // as llama-server's /apply-template rendered its references' prompts);
  // unset, the template's own default.
  struct ChatSlice {
    bool thinking = true;
    bool tools = true;
    std::vector<std::string> stops;
    std::optional<bool> default_thinking;
  };
  ChatSlice chat;
  std::shared_ptr<const artifact::Artifact> target_artifact{};
  std::uint64_t startup_host_bytes = 0;
  std::uint64_t startup_cap = 0;
  // Remains live through source sealing; its counted high-water may grow
  // after the parts factory returns and must enter startup loan retirement.
  std::shared_ptr<engine::ExecutionMetadataAllowance> execution_metadata{};
};

// Makes a model's parts from its settings: the component engine's
// composer in serving (the spec, components, planner and lowering), a fake
// in the tests.
using LlmBuild =
    std::function<std::expected<LlmBuildParts, std::string>(const ModelSettings& settings)>;

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_LLM_BUILD_H_
