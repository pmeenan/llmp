// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The component engine's parts of an LLM served by EngineLlm (D-107;
// docs/runtime-serving.md, "Engine-backed models"; runtime/engine_llm.h):
// from a v1 artifact (and its drafter's) to the spec, the stages' state
// layouts, the batching and composition, the component step sources over
// GGML's registry and lowering, the weights' places and the runner's own
// memory. Serving's build (Server::Make) and the gate harnesses that pay
// serving's work (benchmarks/gemma_engine_greedy.cc) share it. CUDA builds
// only (llmp_runtime_serving).

#ifndef LLMP_RUNTIME_COMPONENT_BUILD_H_
#define LLMP_RUNTIME_COMPONENT_BUILD_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "config/node_config.h"
#include "engine/component_source.h"
#include "engine/execution_contract.h"
#include "engine/paged_node.h"
#include "kernels/common/lower.h"
#include "runtime/component_admission.h"
#include "runtime/engine_llm.h"
#include "spec/spec.h"

namespace llmp::runtime {

// What a build may change of its family's recipe (a harness's matched
// arithmetic): the target stage's planning policy, in its canonical text
// (planner/policy.h), in place of its mapping's default.
struct ComponentBuildOptions {
  std::optional<std::string> target_policy;
  // The same for a drafter stage (a diagnostic: its mapping's default
  // otherwise).
  std::optional<std::string> drafter_policy{};
  // On the shared route (the Server sets it for the architectures whose
  // serving recipe names one, ServingRecipe::shared_route): its plain,
  // joined MTP and automatic serving envelopes. Explicit diagnostic domains
  // retain their closed envelopes (a route without qualified envelopes has
  // none: it serves only its automatic shape). Automatic service admission
  // derives the domain and payment from its actual source, state, decoding
  // mode and configured owners.
  bool shared_route = false;
  // Default-off capture scheduling experiment; numerical policy is unchanged.
  bool capture_first_verify = false;
  // Nonzero only under the Server's admitted metadata bootstrap loan (its
  // default automatic startup): the node's startup execution cap. Derive and
  // pin this source's construction budget before graph construction.
  std::uint64_t automatic_startup_cap = 0;
  // One driver-owned transient compiler payment across automatic aliases.
  // Sources retain this owner; per-alias source/spec/chat storage stays separate.
  std::shared_ptr<engine::SharedCompilerAllowance> shared_compiler{};
  bool prefill_lookahead = true;
  bool prepare_state = true;
  std::optional<bool> state_only_prefill{};
  // Internal owned source qualification. No command/settings flag activates
  // this. An explicit complete domain requires the real backend declaration;
  // supplied tuning owns all names and never supplies provider device facts.
  std::shared_ptr<const engine::NumericalDomain> execution_domain{};
  // Derive the complete ordinary or composed domain from the actual spec,
  // state layouts and configured envelope (Server's default for Qwen3.8 and
  // DeepSeek V4). The existing metadata allowance pays before construction;
  // an explicit domain cannot be combined with this request.
  bool derive_execution_domain = false;
  std::shared_ptr<const planner::OwnedDeviceFacts> tuning{};
  // Optional cap on this source's one growing metadata payment, including
  // accounting overhead. Zero uses the node's startup/running budget and
  // physical headroom. Admission pays actual live demand, never this cap.
  std::uint64_t execution_metadata_cap_bytes = 0;
  // Read-only diagnostic validation of the real source's lowering. Captures
  // are part of the harness's prepaid lifetime, never an alternate backend.
  kernels::ProgramValidator validate_program{};
};

// The parts of `entry`'s model (its artifact and drafter in `store`), made
// at its Setup from its settings: the spec its v1 artifact stores (D-113,
// D-118), composed with its drafter's when it speculates
// (spec::ComposeDrafter: the drafter a stage of its own that reads the
// target's state and taps), each stage's state layout (each tensor on its
// own paged extents), the target's batching, and a component step source a
// stage (one StagedStepSource over them) planned under its approval's
// policy for the serving envelope (its context and owners:
// ComponentAdmissionPolicy, the family's recipe as data, with how serving
// composes its steps) over GGML's registry and lowered onto its
// graphs, on the model's own stream (`index`: its registration's), with a
// cuBLAS handle over a workspace the runner maps and charges as its own; a
// layer's routed experts one slab, its pitch the lowering's expert stride
// (engine/expert_slab.h ExpertSlabsOf). A head row is the head's vocabulary
// of logits; a greedy step returns each owner's token chosen on the
// device. An architecture whose mapping has no serving recipe is refused
// at Setup, naming why.
// The paid internal serving domain, shared by ordinary and block/replay
// compositions. The retained domain owns its allowance charge; this does not
// approve a source or replace ComponentStepSource's complete sealing checks.
std::expected<std::shared_ptr<const engine::NumericalDomain>, std::string> ComponentServingDomain(
    const spec::ModelSpec& model, std::span<const model::StateLayoutDerived> layouts,
    std::uint32_t slots, const engine::DecodingMode& mode,
    const std::shared_ptr<engine::ExecutionMetadataAllowance>& allowance,
    std::function<bool(std::uint64_t)> charge = {});

EngineBuild ComponentLlmBuild(engine::PagedNode& node, std::filesystem::path store,
                              const config::ModelEntry& entry, int index,
                              ComponentBuildOptions options = {});

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_COMPONENT_BUILD_H_
