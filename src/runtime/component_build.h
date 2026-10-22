// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The component engine's parts of an LLM served by Llm (D-107;
// docs/runtime-serving.md, "Engine-backed models"; runtime/serving.h):
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
#include <vector>

#include "config/node_config.h"
#include "engine/component_source.h"
#include "engine/execution_contract.h"
#include "engine/paged_node.h"
#include "kernels/common/lower.h"
#include "runtime/component_admission.h"
#include "runtime/component_domain.h"
#include "runtime/llm_build.h"
#include "runtime/model_settings.h"
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
  // Default-off capture scheduling experiment; numerical policy is unchanged.
  bool capture_first_verify = false;
  // Internal common-path measurement candidate. No CLI/settings activation;
  // production remains separate until actual matched evidence selects it.
  bool trial_mixed_outputs = false;
  bool trial_mixed_causal = false;
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
  // Internal diagnostics own/prepay their supplied inventory just like the
  // validation callback. Build prepays its retained copy before allocation.
  // Every shape must fit the actual layout/backend and explicit domain; this
  // measures only the supplied shapes, never grants wider execution support.
  std::vector<planner::StepShape> diagnostic_setup_shapes{};
  // With no explicit internal domain the complete mechanical envelope is
  // derived from the admitted source/state/mode. Neither form grants numerical
  // transformations; actual policy evidence and source sealing still apply.
  std::shared_ptr<const planner::OwnedDeviceFacts> tuning{};
  // Optional cap on this source's one growing metadata payment, including
  // accounting overhead. Zero uses the node's startup/running budget and
  // physical headroom. Admission pays actual live demand, never this cap.
  std::uint64_t execution_metadata_cap_bytes = 0;
  // Read-only diagnostic validation of the real source's lowering. Captures
  // are part of the harness's prepaid lifetime, never an alternate backend.
  kernels::ProgramValidator validate_program{};
  // Already-opened artifacts and admissions from ReadArtifactFacts (Start),
  // retained across build retries. Each build funds its own mutable copies;
  // shared metadata and its payment live until the last factory/runner releases them.
  // Effective draft vocabulary is fixed by this alias preparation; changing
  // that width requires a separately prepared alias/factory. Equal effective
  // widths (including zero meaning all available rows) can reuse it.
  std::shared_ptr<const OpenedModelArtifacts> prepared{};
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

LlmBuild ComponentLlmBuild(engine::PagedNode& node, std::filesystem::path store,
                           const config::ModelEntry& entry, int index,
                           ComponentBuildOptions options = {});

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_COMPONENT_BUILD_H_
