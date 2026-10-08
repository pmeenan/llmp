// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 Flash Next's chunk planning (docs/experiments/qwen38-native/,
// docs/experiments/fast-swap/): each chunk's graph built, bound, planned and
// placed, with the weights' and state's addresses given by the caller, so
// the paged runner (qwen38_runner.h) plans over device VMM exactly as the
// resident harness (benchmarks/qwen38_exec.cc) plans over cudaMalloc, both
// through this one path; and each chunk's host-built inputs in the graph's
// copy order. The paged runner's logits are checked equal to the resident
// harness's outputs bit for bit. CUDA builds only.

#ifndef LLMP_ENGINE_QWEN38_PLAN_H_
#define LLMP_ENGINE_QWEN38_PLAN_H_

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
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen38.h"

namespace llmp::engine {

// Where a Qwen3.8 model's weights and state live.
struct Qwen38Places {
  // A resource's device address.
  std::function<std::uint64_t(std::uint32_t resource)> resource;
  // An expert array's view: expert 0's slice, the others at the layer's
  // stride.
  std::function<std::uint64_t(std::uint32_t array)> array;
  std::vector<std::uint64_t> stride;  // each layer's expert stride
  std::uint64_t state = 0;            // the state region (model/qwen38.h)
  // Where the n-gram table's rows are read: the table's own address, or
  // (row paging) the chunk's row slots, which `binding`'s table then
  // describes (its rows the slots').
  std::uint64_t ple_table = 0;
  // Speculation (docs/experiments/qwen38-mtp/): the MTP drafter's
  // resources and expert arrays (as `resource` and `array`, in its own
  // artifact), its state (model/qwen38.h Qwen38MtpState) and a verify's
  // saves (Qwen38CommitLayout).
  // NOLINTNEXTLINE(readability-redundant-member-init): designated initializers may omit it
  std::function<std::uint64_t(std::uint32_t resource)> mtp_resource = {};
  // NOLINTNEXTLINE(readability-redundant-member-init): designated initializers may omit it
  std::function<std::uint64_t(std::uint32_t array)> mtp_array = {};
  std::uint64_t mtp_state = 0;
  std::uint64_t commit = 0;
};

struct Qwen38Model {
  const artifact::Artifact* artifact = nullptr;
  const model::Qwen38Profile* profile = nullptr;
  // The binding the graph is built from: the artifact's, or with row
  // paging a copy whose n-gram table has the row slots' rows.
  const model::Qwen38Binding* binding = nullptr;
  const model::Qwen38StateLayout* state = nullptr;
  Qwen38Places places;
  // The graph (qwen38_graph.h Qwen38GraphOptions): llmpalooza's fusions, and
  // whether each layer's expert slots hold the CUTLASS layout (the
  // artifact's, binding->cutlass(), or converted at load by the resident
  // harness) rather than GGML's.
  bool fused = true;
  // The fused graph's reference form (Qwen38GraphOptions::exact) rather
  // than the fast one.
  bool exact = false;
  bool cutlass = false;
  // Speculation: the drafter's binding, state and expert stride, and the
  // verify's commit layout (null without a drafter).
  const model::Qwen38MtpBinding* drafter = nullptr;
  const model::Qwen38MtpState* mtp_state = nullptr;
  std::uint64_t mtp_stride = 0;
  const model::Qwen38CommitLayout* commit = nullptr;
  bool device_masks = false;
};

// What a target chunk computes beside its own rows' work.
struct Qwen38ChunkKind {
  bool verify = false;               // a speculative verify (Qwen38GraphOptions::verify)
  bool export_streams = false;       // its streams for the drafter
  std::uint64_t capture_routed = 0;  // benchmark-only retained down operands
  bool state_only = false;           // required state/export, without a head
  bool operator==(const Qwen38ChunkKind&) const = default;
};

// One chunk shape's graph, plan, placement and bound implementations.
using Qwen38Planned = PlannedGraph<kernels::ggml::Qwen38Graph>;

// Binds the graph's weights and state at the model's places.
void BindQwen38Weights(const Qwen38Model& m, kernels::ggml::Qwen38Graph& g);

// Builds, binds, plans and places one chunk shape's graph: every computed
// tensor first at its own address, then placed in `activations` (0 to
// measure only), planned again, which must give the same plan; the named
// intermediates `keep` stay live to the end (the resident harness's
// dumps). Not bound to the registry.
std::expected<std::unique_ptr<Qwen38Planned>, std::string> PlanQwen38Chunk(
    const Qwen38Model& m, const kernels::ggml::Qwen38ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, std::span<const std::string> keep = {},
    Qwen38ChunkKind kind = {});

// Startup-only measurement; the returned plan cannot be bound.
std::expected<std::unique_ptr<Qwen38Planned>, std::string> PlanQwen38Chunk(
    const Qwen38Model& m, const kernels::ggml::Qwen38ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, std::span<const std::string> keep, Qwen38ChunkKind kind,
    std::optional<ActivationMeasurement> measurement);

// A chunk's host-built inputs in the graph's copy order (the resident
// harness's): each input tensor and its bytes, which `in`, `out_ids` and
// `zeros` own. `ple_rows` replaces in.ple_rows when not empty (the row
// slots' indices). Exported streams go to rows stream_row0 ...
struct Qwen38HostInputs {
  std::vector<std::int32_t> out_ids;
  std::int64_t zero_row = 0;
  std::int32_t zero_index = 0;
  std::vector<std::int64_t> row_ids;
  std::vector<std::int64_t> stream_rows;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};
void Qwen38Sources(const kernels::ggml::Qwen38Graph& g, const model::Qwen38ChunkInputs& in,
                   std::uint32_t outputs, std::span<const std::int32_t> ple_rows,
                   Qwen38HostInputs& out, std::int64_t stream_row0 = 1);

// The MTP drafter's graph (kernels/ggml/qwen38_graph.h BuildQwen38MtpGraph)
// built, bound at the model's places, planned and placed as a chunk's.
using Qwen38MtpPlanned = PlannedGraph<kernels::ggml::Qwen38MtpGraph>;
std::expected<std::unique_ptr<Qwen38MtpPlanned>, std::string> PlanQwen38Mtp(
    const Qwen38Model& m, const kernels::ggml::Qwen38MtpShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes);

// Startup-only measurement; the returned plan cannot be bound.
std::expected<std::unique_ptr<Qwen38MtpPlanned>, std::string> PlanQwen38Mtp(
    const Qwen38Model& m, const kernels::ggml::Qwen38MtpShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, std::optional<ActivationMeasurement> measurement);

// A drafter pass's host-built inputs: `in` the pass's positions (tokens
// only for pass 0), in the graph's copy order.
struct Qwen38MtpHostInputs {
  std::int64_t zero_row = 0;
  std::int32_t zero_index = 0;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};
void Qwen38MtpSources(const kernels::ggml::Qwen38MtpGraph& g,
                      std::span<const model::Qwen38ChunkInputs> passes,
                      std::span<const std::int32_t> tokens, Qwen38MtpHostInputs& out);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_QWEN38_PLAN_H_
