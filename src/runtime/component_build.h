// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The component engine's parts of an LLM served by EngineLlm (D-107;
// docs/runtime-serving.md, "Engine-backed models"; runtime/engine_llm.h):
// from a v0 artifact (and its drafter's) to the spec, the stages' state
// layouts, the batching and composition, the component step sources over
// GGML's registry and lowering, the weights' places and the runner's own
// memory. Serving's build (Server::Make) and the gate harnesses that pay
// serving's work (benchmarks/gemma_engine_greedy.cc) share it. CUDA builds
// only (llmp_runtime_serving).

#ifndef LLMP_RUNTIME_COMPONENT_BUILD_H_
#define LLMP_RUNTIME_COMPONENT_BUILD_H_

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "config/node_config.h"
#include "engine/execution_contract.h"
#include "engine/paged_node.h"
#include "kernels/common/lower.h"
#include "runtime/engine_llm.h"

namespace llmp::runtime {

// The architectures served by the component engine's adapter for their
// pipeline kind (engine_llm.h, engine_image.h; D-107): Gemma 2 and Gemma 3
// since their migration, Gemma 4 with its assistant since its. A family
// joins when its migration passes its gate.
bool OnComponentEngine(std::string_view architecture);

// What a build may change of its family's recipe (a harness's matched
// arithmetic): the target stage's planning policy, in its canonical text
// (planner/policy.h), in place of its mapping's default.
struct ComponentBuildOptions {
  std::optional<std::string> target_policy;
  // Internal qualification only; Qwen and DeepSeek remain off in OnComponentEngine.
  // Other slot counts refuse before opening artifacts, except the closed
  // target-only C4 scoring envelope, a declared repeat-pending composition,
  // or a derived ordinary target-only domain. Public admission stays unchanged.
  bool diagnostic_solo = false;
  // Nonzero only under the Server's admitted metadata bootstrap loan.
  // Derive and pin this source's construction budget before graph construction.
  std::uint64_t automatic_startup_cap = 0;
  // Complete catalog-backed construction allowance for the closed DeepSeek
  // diagnostic. Admitted by its owner before constructing EngineLlm; the
  // factory charges before reading/parsing its prepared source and retains
  // the charge with every returned source/plan borrower. Not a paid flag.
  std::shared_ptr<engine::ExecutionMetadataAllowance> construction_metadata{};
  bool prefill_lookahead = true;
  bool prepare_state = true;
  std::optional<bool> state_only_prefill{};
  // Internal owned source qualification. No command/settings flag activates
  // this. An explicit complete domain requires the real backend declaration;
  // supplied tuning owns all names and never supplies provider device facts.
  std::shared_ptr<const engine::NumericalDomain> execution_domain{};
  // Internal serving qualification: derive the complete ordinary or composed domain from
  // the actual spec, state layouts and configured envelope. The existing
  // metadata allowance pays before construction; an explicit domain cannot
  // be combined with this request. No new public setting or default admission.
  bool derive_execution_domain = false;
  std::shared_ptr<const planner::OwnedDeviceFacts> tuning{};
  // Paid once in the node catalog, before declaration construction. The build
  // retains it across retries; no host-floor/pre-Start counter is authority.
  std::uint64_t execution_metadata_bytes = 1U << 20;
  // Read-only diagnostic validation of the real source's lowering. Captures
  // are part of the harness's prepaid lifetime, never an alternate backend.
  kernels::ProgramValidator validate_program{};
};

// The parts of `entry`'s model (its artifact and drafter in `store`), made
// at its Setup from its settings: its spec from its architecture's mapping
// (a v0 artifact's, with its compiled profile's values, D-109's
// synthesis), composed with its drafter's when it speculates
// (spec::ComposeDrafterV0: the drafter a stage of its own that reads the
// target's state and taps), each stage's state layout (each tensor on its
// own paged extents), the target's batching, and a component step source a
// stage (one StagedStepSource over them) planned under each mapping's
// default policy for those values and the serving envelope (its context
// and owners: spec::DefaultRule, the family's recipe as data, with how
// serving composes its steps) over GGML's registry and lowered onto its
// graphs, on the model's own stream (`index`: its registration's), with a
// cuBLAS handle over a workspace the runner maps and charges as its own; a
// layer's routed experts one slab, its pitch the lowering's expert stride
// (engine/expert_slab.h ExpertSlabsOf). A head row is the head's vocabulary
// of logits; a greedy step returns each owner's token chosen on the
// device. Gemma 2, Gemma 3 and Gemma 4 (with its assistant: an anchored
// drafter reading the target's tapped features, three drafts a unit), by
// their compiled profiles' values; any other architecture is refused at
// Setup, naming why.
EngineBuild ComponentLlmBuild(engine::PagedNode& node, std::filesystem::path store,
                              const config::ModelEntry& entry, int index,
                              ComponentBuildOptions options = {});

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_COMPONENT_BUILD_H_
