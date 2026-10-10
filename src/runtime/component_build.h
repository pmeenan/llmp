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
#include <optional>
#include <string>
#include <string_view>

#include "config/node_config.h"
#include "engine/paged_node.h"
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
