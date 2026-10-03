// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash's chunk planning (docs/experiments/dsv4-native/,
// docs/experiments/fast-swap/): each chunk's graph built, bound, planned and
// placed as the resident harness (benchmarks/dsv4_exec.cc) first did it,
// with the weights' and state's addresses given by the caller, so the paged
// runner (dsv4_runner.h) plans over device VMM exactly as the resident one
// plans over cudaMalloc; and each chunk's host-built inputs. CUDA builds
// only.

#ifndef JITLLM_ENGINE_DSV4_PLAN_H_
#define JITLLM_ENGINE_DSV4_PLAN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "engine/planned.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/tensors.h"
#include "model/dspark.h"
#include "model/dsv4.h"

namespace jitllm::engine {

// Where a DeepSeek model's weights and state live.
struct Dsv4Places {
  // A resource's device address (the token table's group has none: it is
  // read on the host).
  std::function<std::uint64_t(std::uint32_t resource)> resource;
  // An expert array's view: expert 0's slice, the others at the layer's
  // stride.
  std::function<std::uint64_t(std::uint32_t array)> array;
  std::vector<std::uint64_t> stride;  // each layer's expert stride
  std::uint64_t state = 0;            // the state region (model/dsv4.h)
};

struct Dsv4Model {
  const artifact::Artifact* artifact = nullptr;
  const model::Dsv4Profile* profile = nullptr;
  const model::Dsv4Binding* binding = nullptr;
  const model::Dsv4StateLayout* state = nullptr;
  Dsv4Places places;
  std::vector<float> rot;  // the indexer's Hadamard matrix
  // The reference mode (the owner's policy, 2026-09-28: speed first): the
  // graph node for node as llama.cpp builds it, planned unfused, so its
  // logits equal llama.cpp's with fusion off bit for bit, and a verify's
  // row-invariant plan (D-092). Off (the default): jitLLM's fused plan and
  // a batched verify, judged coarsely against llama.cpp.
  bool exact = false;
  // The output-A/HCA prefill combination on full 4,096-row chunks (the
  // qualified shape; serving's default, docs/experiments/ds4-output-prefix):
  // OutA keeps its descriptor fallback; HCA requires all layers' OutA and
  // the 256-compressed shape. Other rows retain ordinary math.
  bool prefill_outa_hca = false;
  // And on every prefill chunk of kDsv4StageMinRows to 4,096 rows, a
  // prompt's last, partial one included. Internal and off: it failed the
  // registered 32K fixed-history bound (docs/experiments/ds4-output-prefix,
  // "Default-on acceptance").
  bool prefill_outa_hca_partial = false;
};

// DeepSeek's DSpark drafter beside its target (model/dspark.h): its places
// (its own resources, expert arrays and strides, and its ring as
// places.state) and its target's resources, whose head it reads.
struct DsparkModel {
  const artifact::Artifact* artifact = nullptr;
  const model::DsparkProfile* profile = nullptr;
  const model::DsparkBinding* binding = nullptr;
  const model::DsparkStateLayout* state = nullptr;
  Dsv4Places places;
  std::function<std::uint64_t(std::uint32_t resource)> target_resource;
  bool exact = false;  // the reference mode, as Dsv4Model::exact
};

// How a target chunk runs beside a drafter.
struct Dsv4Speculation {
  // A speculative verify: every row's logits; in the exact mode
  // (Dsv4Model::exact) the row-invariant plan (D-092; graph_plan.h
  // DeviceChoices::row_invariant, dsv4_graph.h Dsv4GraphOptions), else the
  // batched plan, each routed expert read once for every row that selects it.
  bool verify = false;
  // With a drafter: the chunk's features and its last `inject_rows` rows'
  // KV injection into the drafter's ring (dsv4_graph.h Dsv4Injection).
  const DsparkModel* drafter = nullptr;
  std::int64_t inject_rows = 0;
};

// One chunk shape's graph, plan, placement and bound implementations.
using Dsv4Planned = PlannedGraph<kernels::ggml::Dsv4Graph>;

// Binds the graph's weights and state at the model's places.
void BindDsv4Weights(const Dsv4Model& m, kernels::ggml::Dsv4Graph& g);
// Binds only the state tensors of `g` at a state region `base` of `m`'s
// layout (a wave slot's).
void BindDsv4State(const Dsv4Model& m, std::uint64_t base, kernels::ggml::Dsv4Graph& g);

// Whether a chunk of this shape takes the combined output-A/HCA prefill's
// HCA (with Dsv4Model::prefill_outa_hca, the fast plan): a full 4,096-row
// chunk (with prefill_outa_hca_partial, any prefill chunk of
// kDsv4HcaMinRows to kDsv4HcaMaxRows rows) while the HCA layers'
// compressed cells are 256 wide. Its plan then holds the chunk's first
// position (PlanDsv4Chunk `first_position`).
bool Dsv4PrefillHca(const Dsv4Model& m, const kernels::ggml::Dsv4ChunkShape& shape);

// Builds, binds, plans and places one chunk shape's graph: every computed
// tensor first at its own address, then placed in `activations` (0 to
// measure only), planned again, which must give the same plan. `keep`
// names llama.cpp callback tensors to keep alive ("*": every named one).
// Shape outputs can narrow a fast chunk's trailing head rows; reference,
// verify and named-diagnostic plans require all rows.
// Not bound to the registry.
std::expected<std::unique_ptr<Dsv4Planned>, std::string> PlanDsv4Chunk(
    const Dsv4Model& m, const kernels::ggml::Dsv4ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::span<const std::string> keep,
    std::uint64_t activations, std::uint64_t activation_bytes,
    const Dsv4Speculation& speculation = {},
    std::optional<std::uint32_t> first_position = std::nullopt);

// Sets an HCA prefill plan's chunk position: its HCA nodes' host scalar and
// the position its inputs are authenticated against (BuildDsv4Inputs), so a
// chunk of the same shape at another position runs the same plan. HCA
// reads its position only at launch, so such a plan must run launch by
// launch: refused when `captured` (the plan has a graph, whose replay would
// keep the captured position), for a plan without HCA, or for a chunk past
// the context.
std::expected<void, std::string> SetDsv4HcaFirstPosition(const Dsv4Model& m,
                                                         kernels::ggml::Dsv4Graph& g,
                                                         std::uint32_t first, bool captured);

// Binds a chunk graph's DSpark injection: the drafter's fc, norms and wkv,
// and its ring.
void BindDsparkInjection(const DsparkModel& d, kernels::ggml::Dsv4Graph& g);

// A wave's graph (kernels/ggml/dsv4_graph.h Dsv4WaveGraph), plan,
// placement and bound implementations.
using Dsv4WavePlanned = PlannedGraph<kernels::ggml::Dsv4WaveGraph>;

// Builds, binds, plans and places a wave of the fast plan: `m`'s weights,
// each slot's state at `states` (in wave order), and with `drafter` its
// injection's weights and each slot's ring at `rings`. As PlanDsv4Chunk
// (activations 0: measure only); refused for the reference mode.
std::expected<std::unique_ptr<Dsv4WavePlanned>, std::string> PlanDsv4Wave(
    const Dsv4Model& m, std::span<const std::uint64_t> states,
    const kernels::ggml::Dsv4WaveShape& shape, const kernels::ggml::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes, const DsparkModel* drafter = nullptr,
    std::span<const std::uint64_t> rings = {});

// A draft block's graph, plan, placement and bound implementations.
using DsparkPlanned = PlannedGraph<kernels::ggml::DsparkGraph>;

// Builds, binds, plans and places a draft block of `rows` rows, as
// PlanDsv4Chunk does a chunk (activations 0: measure only).
std::expected<std::unique_ptr<DsparkPlanned>, std::string> PlanDsparkDraft(
    const DsparkModel& d, std::int64_t rows, const kernels::ggml::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes);

// A chunk's host-built inputs, in the graph's copy order: each input
// tensor and the bytes it takes (owned here, so they live as long as this).
struct Dsv4HostInputs {
  std::vector<float> embd;
  std::vector<std::int32_t> tokens;
  std::vector<std::int32_t> out_ids;
  std::vector<std::uint16_t> zeros;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};

// The inputs of `tokens` at `in`'s positions: the embedding rows
// dequantized on the host from `table` (the token table group's bytes, in
// host memory, as llama.cpp's CPU backend looks them up), and the chunk
// plan's indices and masks. Refused for a token outside the vocabulary.
std::expected<void, std::string> BuildDsv4Inputs(
    const Dsv4Model& m, const kernels::ggml::Dsv4Graph& g, const model::Dsv4ChunkInputs& in,
    std::span<const std::int32_t> tokens, std::span<const std::byte> table, Dsv4HostInputs& out,
    std::span<const std::int64_t> inject_cells = {});

// A wave's host-built inputs (Dsv4WaveGraph::inputs' order).
struct Dsv4WaveHostInputs {
  std::vector<float> embd;
  std::vector<std::int32_t> tokens;
  std::vector<std::int32_t> positions;
  std::array<std::vector<std::int32_t>, 3> state_pos;  // CSA, HCA, the indexer
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};

// One wave slot's host inputs: its chunk's (model/dsv4.h Dsv4Chunk, without
// masks), its tokens and its injection's cells (empty without one).
struct Dsv4WaveSlotInputs {
  const model::Dsv4ChunkInputs* chunk = nullptr;
  std::span<const std::int32_t> tokens;
  std::span<const std::int64_t> inject_cells;
};

// The inputs of a wave, each slot's rows joined in wave order where the
// graph joins them; refused for a token outside the vocabulary or inputs
// that are not the graph's slots'. `slots` and what they name must outlive
// `out`'s use.
std::expected<void, std::string> BuildDsv4WaveInputs(const Dsv4Model& m,
                                                     const kernels::ggml::Dsv4WaveGraph& g,
                                                     std::span<const Dsv4WaveSlotInputs> slots,
                                                     std::span<const std::byte> table,
                                                     Dsv4WaveHostInputs& out);

// The embedding rows of `tokens`, dequantized on the host from `table`
// (the token table group's bytes) as every chunk looks them up.
std::expected<void, std::string> Dsv4EmbeddingRows(const Dsv4Model& m,
                                                   std::span<const std::int32_t> tokens,
                                                   std::span<const std::byte> table,
                                                   std::vector<float>& embd);

// A draft block's inputs: its tokens' embedding rows from the target's
// table (`m`'s, on the host), and the block's positions, ring cells and
// mask.
std::expected<void, std::string> BuildDsparkInputs(const Dsv4Model& m,
                                                   const kernels::ggml::DsparkGraph& g,
                                                   const model::DsparkBlockInputs& in,
                                                   std::span<const std::byte> table,
                                                   Dsv4HostInputs& out);

// The row's greedy token: its largest logit's index, the lowest among
// equals (as the harnesses and llama.cpp's greedy sampler choose).
std::int32_t Argmax(std::span<const float> row);

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_DSV4_PLAN_H_
