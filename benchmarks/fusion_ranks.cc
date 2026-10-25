// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Own-decomposition paired screen. Protocol and prospective bounds:
// docs/experiments/fusion-ranks/README.md. No checkpoints or global rank pins.
#include "execution/fusion_ranks.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/bytes.h"
#include "base/json.h"
#include "components/graph_io.h"
#include "components/spatial.h"
#include "engine/comparison_pair.h"
#include "engine/graph_plan.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/op_traits.h"
#include "kernels/common/comparison.h"
#include "kernels/common/device_backend.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"
#include "planner/alternative.h"
#include "planner/pattern.h"
#include "providers/cuda/cuda_device_execution.h"
#include "runtime/component_admission.h"
#include "runtime/component_image_build.h"
#include "spec/image_spatial.h"

namespace {
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
namespace kn = llmp::kernels;
namespace en = llmp::engine;
namespace co = llmp::components;
namespace ref = llmp::kernels::reference;
using llmp::base::Bytes;
using llmp::format::Encoding;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kPool = 16 * kMiB;
constexpr std::uint64_t kActivations = 32 * kMiB;
constexpr std::uint64_t kMetadata = 64 * kMiB;
constexpr int kSamples = 9, kWarmups = 4, kReplays = 128;

std::string Quote(std::string_view text);
std::string_view phase = "setup";
void Receipt(std::string_view event, std::string_view fields) {
  std::println("{{\"event\":{},\"phase\":{},{} }}", Quote(event), Quote(phase), fields);
  std::fflush(stdout);
}
[[noreturn]] void Fail(std::string_view message) {
  Receipt("refused", std::format("\"reason\":{}", Quote(message)));
  std::println(stderr, "fusion screen failed: {}", message);
  std::abort();  // supervised process exit retains every owner until context teardown
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) Fail(cudaGetErrorString(error));
}
template <class T, class E>
T Must(std::expected<T, E> result) {
  if (!result) {
    if constexpr (requires { pl::PlanErrorName(result.error().error); })
      Fail(std::format("{}: {}", pl::PlanErrorName(result.error().error), result.error().detail));
    else if constexpr (requires { result.error().detail; })
      Fail(result.error().detail);
    else if constexpr (requires { std::string_view(result.error()); })
      Fail(result.error());
    else
      Fail("provider operation refused");
  }
  if constexpr (!std::is_void_v<T>) return std::move(*result);
}
std::string Quote(std::string_view text) {
  std::string out;
  llmp::base::json::AppendQuoted(text, out);
  return out;
}
std::string Identity(const ex::MeasuredImplementation& value) {
  return std::format("{{\"name\":{},\"identity\":{}}}", Quote(value.name),
                     Quote(llmp::base::ToHex(value.identity)));
}

enum class ReadyArithmetic {
  kMatMulGlu,
  kNormAdd,
  kMatMulAdd,
  kSplitResidual,
  kSplitModulate,
  kSplitJoined,
  kSpatialNorm,
  kHcPostNorm,
  kHcCombine,
  kGdnGates
};

enum class FixtureKind { kNormMul, kNormGate, kNormRope, kMatMulGlu };

struct Fixture {
  FixtureKind kind;
  std::string_view name;
  std::string_view family;
  ir::OpKey operation;
  std::array<std::string_view, 2> patterns;
  std::size_t pattern_count;
};

struct FixtureGraph {
  ir::Graph graph;
  std::vector<llmp::components::StepWeight> weights;
};

constexpr std::array kFixtures = {
    Fixture{FixtureKind::kNormMul,
            "rms_norm_mul",
            "norm/gate/rope",
            ir::Fused(ir::FusedKind::kRmsNormMul),
            {"rms_norm_mul", {}},
            1},
    Fixture{FixtureKind::kNormGate,
            "norm_gate",
            "norm/gate/rope",
            ir::Fused(ir::FusedKind::kNormGate),
            {"gdn_norm_gate", {}},
            1},
    Fixture{FixtureKind::kNormRope,
            "rms_norm_mul_rope",
            "norm/gate/rope",
            ir::Fused(ir::FusedKind::kRmsNormMulRope),
            {"interleaved_norm_rope", "rms_norm_mul_rope"},
            2},
    Fixture{FixtureKind::kMatMulGlu,
            "matmul_glu",
            "dense/routed products",
            ir::Fused(ir::FusedKind::kMatMulGlu),
            {"float_mul_mat_glu", "mul_mat_glu"},
            2},
};

const Fixture* FindFixture(std::string_view name) {
  const auto found = std::ranges::find(kFixtures, name, &Fixture::name);
  return found == kFixtures.end() ? nullptr : &*found;
}

FixtureGraph BuildFixture(const Fixture& fixture, std::uint64_t width, std::uint64_t rows) {
  ir::Builder b;
  b.SetOwner(0);
  std::vector<llmp::components::StepWeight> weights;
  const auto x = b.Input(ir::Dense(Encoding::kF32, fixture.kind == FixtureKind::kNormRope
                                                       ? ir::ShapeOf({width, 1, rows})
                                                       : ir::ShapeOf({width, rows})));
  if (fixture.kind == FixtureKind::kNormMul) {
    const ir::WeightRef ref{.role = 1};
    const auto w = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({width})), ref, "norm.weight");
    weights.push_back(
        {.tensor = w, .ref = ref, .parts = {{"norm.weight", llmp::format::PartRole::kData}}});
    b.Keep(x);
    const std::array sources{x, w};
    b.Output(b.Fused(ir::FusedKind::kRmsNormMul, sources, ir::NormParams{.eps = 1e-6F})[0]);
  } else if (fixture.kind == FixtureKind::kNormGate) {
    const ir::WeightRef ref{.role = 1};
    const auto w = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({width})), ref, "norm.weight");
    weights.push_back(
        {.tensor = w, .ref = ref, .parts = {{"norm.weight", llmp::format::PartRole::kData}}});
    b.Keep(x);
    const auto gate = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})));
    b.Keep(gate);
    const std::array sources{x, w, gate};
    b.Output(b.Fused(ir::FusedKind::kNormGate, sources,
                     ir::NormGateParams{.eps = 1e-6F, .gate = ir::NormGateFn::kSigmoid})[0]);
  } else if (fixture.kind == FixtureKind::kNormRope) {
    const ir::WeightRef ref{.role = 1};
    const auto w = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({width})), ref, "norm.weight");
    weights.push_back(
        {.tensor = w, .ref = ref, .parts = {{"norm.weight", llmp::format::PartRole::kData}}});
    const auto positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})));
    b.Keep(x);
    b.Keep(positions);
    const std::array sources{x, w, positions};
    ir::NormRopeParams params{.eps = 1e-6F};
    params.rope.mode = ir::RopeMode::kNeox;
    params.rope.dims = static_cast<std::uint32_t>(width);
    params.rope.original_context = 4096;
    b.Output(b.Fused(ir::FusedKind::kRmsNormMulRope, sources, params)[0]);
  } else {
    const ir::WeightRef gate_ref{.role = 1}, up_ref{.role = 2};
    const auto gate =
        b.Weight(ir::Dense(Encoding::kF16, ir::ShapeOf({width, width})), gate_ref, "gate.weight");
    const auto up =
        b.Weight(ir::Dense(Encoding::kF16, ir::ShapeOf({width, width})), up_ref, "up.weight");
    weights.push_back({.tensor = gate,
                       .ref = gate_ref,
                       .parts = {{"gate.weight", llmp::format::PartRole::kData}}});
    weights.push_back(
        {.tensor = up, .ref = up_ref, .parts = {{"up.weight", llmp::format::PartRole::kData}}});
    b.Keep(x);
    const std::array sources{gate, up, x};
    b.Output(b.Fused(ir::FusedKind::kMatMulGlu, sources,
                     ir::GluParams{.kind = ir::GluKind::kSwiGlu})[0]);
  }
  return {.graph = Must(std::move(b).Finish()), .weights = std::move(weights)};
}

std::string_view PatternKindName(pl::PatternKind kind) {
  switch (kind) {
    case pl::PatternKind::kRewrite:
      return "rewrite";
    case pl::PatternKind::kBatch:
      return "batch";
    case pl::PatternKind::kComposite:
      return "composite";
    case pl::PatternKind::kOption:
      return "option";
  }
  return "unknown";
}
struct CapturedInput {
  std::size_t logical_index = 0;
  std::vector<std::byte> bytes;
};
struct CapturedWeight {
  ir::WeightRef ref;
  std::vector<std::byte> bytes;
};
struct Arm {
  std::unique_ptr<kn::StepProgram> program;
  std::unique_ptr<kn::Recorded> recorded;
  std::vector<ex::MeasuredImplementation> implementations;
  std::vector<ex::MeasuredImplementation> physical_recipes;
  std::vector<float> values;
  std::vector<CapturedInput> host_inputs;
  std::vector<CapturedWeight> host_weights;
  std::vector<double> samples;
  std::optional<llmp::base::Sha256Digest> final_query;
  std::optional<ir::OpKey> selected_fusion;
  std::optional<kn::ComparisonRecipeSeal> comparison_recipe;
  bool grouped_or_batched = false;
};

struct ReferenceError {
  double maximum = 0;
  double max_absolute = 0;
  std::size_t index = 0;
  double expected = 0;
  double actual = 0;
};

class ScreenMetadataFunder final : public ir::StorageFunder {
 public:
  ScreenMetadataFunder(std::uint64_t& used, std::uint64_t capacity)
      : used_(used), capacity_(capacity) {}
  bool Charge(std::uint64_t bytes) const override {
    if (bytes > capacity_ - used_) {
      Receipt("metadata_admission_refused",
              std::format("\"used_bytes\":{},\"requested_bytes\":{},\"budget_bytes\":{}", used_,
                          bytes, capacity_));
      return false;
    }
    used_ += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const override {
    if (bytes > used_) Fail("comparison metadata accounting underflow");
    used_ -= bytes;
  }

 private:
  std::uint64_t& used_;
  std::uint64_t capacity_;
};

float ReadF32(std::span<const std::byte> bytes, std::size_t index) {
  float value = 0;
  const auto offset = index * sizeof(value);
  if (offset > bytes.size() || bytes.size() - offset < sizeof(value))
    Fail("product reference F32 read exceeds captured input bytes");
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

float ReadF16(std::span<const std::byte> bytes, std::size_t index) {
  __half value{};
  const auto offset = index * sizeof(value);
  if (offset > bytes.size() || bytes.size() - offset < sizeof(value))
    Fail("product reference F16 read exceeds captured weight bytes");
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return __half2float(value);
}

ReferenceError ProductReferenceError(const Arm& arm, std::uint64_t width, std::uint64_t rows) {
  const auto input =
      std::ranges::find(arm.host_inputs, std::size_t{0}, &CapturedInput::logical_index);
  const auto gate = std::ranges::find(arm.host_weights, std::uint16_t{1},
                                      [](const CapturedWeight& weight) { return weight.ref.role; });
  const auto up = std::ranges::find(arm.host_weights, std::uint16_t{2},
                                    [](const CapturedWeight& weight) { return weight.ref.role; });
  if (input == arm.host_inputs.end() || gate == arm.host_weights.end() ||
      up == arm.host_weights.end())
    Fail("product reference is missing captured activation or immutable weights");
  if (arm.values.size() != width * rows || input->bytes.size() != width * rows * sizeof(float) ||
      gate->bytes.size() != width * width * sizeof(__half) ||
      up->bytes.size() != width * width * sizeof(__half))
    Fail("product reference fixture byte extents differ from the typed graph");
  ReferenceError error;
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t output = 0; output < width; ++output) {
      double gate_sum = 0, up_sum = 0;
      for (std::uint64_t k = 0; k < width; ++k) {
        const auto input_value = static_cast<double>(ReadF32(input->bytes, row * width + k));
        const auto gate_value = static_cast<double>(ReadF16(gate->bytes, output * width + k));
        const auto up_value = static_cast<double>(ReadF16(up->bytes, output * width + k));
        gate_sum += gate_value * input_value;
        up_sum += up_value * input_value;
      }
      const double expected = (gate_sum / (1.0 + std::exp(-gate_sum))) * up_sum;
      const std::size_t index = row * width + output;
      const double actual = arm.values[index];
      if (!std::isfinite(expected) || !std::isfinite(actual))
        Fail("product reference or device output is nonfinite");
      const double relative = std::abs(actual - expected) / (1.0 + std::abs(expected));
      error.max_absolute = std::max(error.max_absolute, std::abs(actual - expected));
      if (relative > error.maximum) {
        error.maximum = relative;
        error.index = index;
        error.expected = expected;
        error.actual = actual;
      }
    }
  }
  return error;
}

std::string Identities(std::span<const ex::MeasuredImplementation> values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out += ',';
    out += Identity(values[i]);
  }
  out += ']';
  return out;
}

std::string AppliedSites(const pl::StepPlan& plan) {
  std::string out = "[";
  for (std::size_t i = 0; i < plan.applied.size(); ++i) {
    if (i != 0) out += ',';
    const auto& applied = plan.applied[i];
    out += std::format("{{\"pattern\":{},\"replaced_source_nodes\":[", Quote(applied.pattern));
    for (std::size_t j = 0; j < applied.replaced.size(); ++j) {
      if (j != 0) out += ',';
      out += std::to_string(applied.replaced[j].v);
    }
    out += "]}";
  }
  out += ']';
  return out;
}

std::string_view ProofStateName(kn::ComparisonProofState state) {
  switch (state) {
    case kn::ComparisonProofState::kUnproved:
      return "unproved";
    case kn::ComparisonProofState::kDescriptorVerified:
      return "descriptor_verified";
    case kn::ComparisonProofState::kBoundVerified:
      return "bound_verified";
  }
  return "unknown";
}

std::string_view ComparisonMissingName(kn::ComparisonMissing missing) {
  switch (missing) {
    case kn::ComparisonMissing::kNone:
      return "none";
    case kn::ComparisonMissing::kBoundary:
      return "boundary";
    case kn::ComparisonMissing::kPhysicalClosure:
      return "physical_closure";
    case kn::ComparisonMissing::kOperationSchema:
      return "operation_schema";
    case kn::ComparisonMissing::kSelectedCall:
      return "selected_call";
    case kn::ComparisonMissing::kBindings:
      return "bindings";
    case kn::ComparisonMissing::kSourceIdentity:
      return "source_identity";
  }
  return "unknown";
}

struct AdmittedImageBlockSource {
  // The caller's existing paid source/reference allowance precedes all
  // payloads.
  std::shared_ptr<const void> source_payment;
  std::shared_ptr<const llmp::artifact::Composition> composition;
  std::shared_ptr<const llmp::runtime::ImageFamily> family;
  llmp::model::ImageDenoiserProfile profile;
  std::vector<llmp::model::ImageTensor> source_roles;
  std::vector<llmp::components::BoundTensor> tensors;
  llmp::spec::SpatialComposition mapping;
  std::optional<llmp::components::SpatialGraph> spatial;
  llmp::planner::Policy policy;
  // graph.weights() order; BF16 native bytes, also bound directly to original
  // CPU IR.
  std::vector<std::vector<std::uint16_t>> weights;
  // graph.inputs() order, exact native F32/I32 bytes. Both frames remain owned
  // until every upload and execution fence has completed.
  std::array<std::vector<std::vector<std::byte>>, 2> inputs;

  llmp::components::GraphBuildIO io() const {
    const auto& graph = spatial->graph();
    return {graph, graph.inputs(), graph.outputs(), spatial->weights()};
  }
};

// Caller has already paid the actual opened composition/components and the
// source mapper's kVaeSpatialSpecBytes, payload and CPU-reference inventory.
// Spatial graph and numerical projection use their normal real funders below.
// Reduced geometry and deterministic BF16 payloads are explicitly synthetic
// arithmetic data; family/source identity and numerical admission are genuine.
std::expected<std::unique_ptr<AdmittedImageBlockSource>, std::string> MakeAdmittedImageBlockSource(
    std::shared_ptr<const llmp::artifact::Composition> composition,
    std::span<const llmp::artifact::Artifact* const> components,
    std::shared_ptr<llmp::ir::StorageLoan> source_payment,
    llmp::components::SpatialPayment graph_payment,
    std::shared_ptr<const llmp::kernels::ConstructionFunder> numerical_funder,
    const llmp::planner::DeviceFacts& device, std::uint32_t rows, bool prepare_payloads) {
  namespace co = llmp::components;
  namespace fm = llmp::format;
  namespace ir = llmp::ir;
  namespace mo = llmp::model;
  if (!composition || !source_payment || !numerical_funder)
    return std::unexpected("image block needs actual source and paid construction owners");
  auto family = llmp::runtime::ReadImageFamily(*composition, components);
  if (!family) return std::unexpected(family.error());
  if (!(*family)->mapping || !(*family)->mapping->serving ||
      !(*family)->mapping->serving->serving_rule)
    return std::unexpected("image block has no admitted serving policy");
  auto source = std::make_unique<AdmittedImageBlockSource>();
  source->source_payment = source_payment;
  source->composition = std::move(composition);
  source->family = std::move(*family);
  source->profile = source->family->profile.denoiser;
  auto& profile = source->profile;
  if (profile.head_dim != 128 || profile.rope_axes != std::array<std::uint32_t, 3>{16, 56, 56})
    return std::unexpected("image block does not have the qualified source head geometry");
  profile.blocks = 1;
  profile.heads = 2;
  profile.width = 256;
  profile.mlp = 512;
  profile.context = 256;
  // Keep source timestep/epsilon/RoPE semantics unchanged.
  source->source_roles = mo::DenoiserTensors(profile, source->family->mapping->names.denoiser);
  for (const auto& tensor : source->source_roles) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    source->tensors.push_back({tensor.name, ir::Dense(fm::Encoding::kBF16, shape)});
  }
  // Exact reduced flow cells used by image_fusion_gpu_test; R10 retains
  // the constructor's rectangular source fixture; R11 adds one text row only
  // for the unmeasured installed-data miss control.
  const std::uint32_t text_rows = rows == 10   ? 2
                                  : rows == 11 ? 3
                                  : rows == 17 ? 1
                                  : rows == 77 ? 13
                                               : 8;
  const std::uint32_t grid_height = rows == 10 || rows == 11 ? 2
                                    : rows == 17             ? 4
                                    : rows == 77             ? 8
                                                             : 16;
  const std::uint32_t grid_width = rows == 10 || rows == 11 ? 4 : grid_height;
  if ((rows != 10 && rows != 11 && rows != 17 && rows != 77 && rows != 264) ||
      rows != text_rows + grid_height * grid_width)
    return std::unexpected("image block has no source-backed flow geometry");
  auto mapping = llmp::spec::ImageTransformerBlockSpec(profile, 0, text_rows, source->source_roles);
  if (!mapping) return std::unexpected(mapping.error());
  source->mapping = std::move(*mapping);
  auto policy = llmp::planner::ParsePolicy(source->family->mapping->serving->serving_rule->text);
  if (!policy) return std::unexpected(policy.error());
  if (!policy->numerical_candidates.empty())
    return std::unexpected("image serving policy unexpectedly grants diagnostic candidates");
  auto step_policy = llmp::planner::ForStep(*policy, rows, 1);
  if (!step_policy) return std::unexpected(step_policy.error());
  source->policy = std::move(*step_policy);
  const std::array<std::span<const co::BoundTensor>, 1> tables{source->tensors};
  const std::array owners{co::SpatialOwner{0, {1, rows, profile.width, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {1, 2, profile.timestep_dim, 1}},
                         co::SpatialInput{0, 1, {rows, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 2, {rows, rows, 1, 1}}};
  auto built_graph = co::BuildSpatialStep(
      source->mapping.spec, tables, owners, std::move(graph_payment),
      {.links = source->mapping.links,
       .reference_buffers = false,
       .exports = source->mapping.exports,
       .inputs = ports,
       .sink_layouts = !llmp::planner::Disabled(source->policy, "layout_sinking"),
       .stage = 1});
  if (!built_graph) return std::unexpected(built_graph.error());
  source->spatial.emplace(std::move(*built_graph));
  auto admitted = llmp::runtime::AdmittedImageNumerics(
      source->family->spec, source->family->approval, *source->composition,
      source->spatial->graph(), source->mapping.spec, 1, device, std::move(numerical_funder));
  if (!admitted) return std::unexpected(admitted.error());
  if (!*admitted)
    return std::unexpected("the actual source/device supplied no image numerical evidence");
  source->policy.numerical = std::move(*admitted);
  source->policy.active_stage = 1;
  llmp::planner::Canonicalize(source->policy);
  if (!prepare_payloads) return source;

  // Pay native payloads, readback and complete original-IR reference storage
  // before creating the vectors. Graph metadata is paid by graph_payment.
  std::uint64_t payload_bytes = 0;
  const auto add_payload = [&](std::uint64_t bytes, std::uint64_t copies) {
    std::uint64_t total = 0;
    return !__builtin_mul_overflow(bytes, copies, &total) &&
           !__builtin_add_overflow(payload_bytes, total, &payload_bytes);
  };
  const auto& graph = source->spatial->graph();
  std::uint64_t largest_temporary = 0;
  for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) {
    const llmp::ir::TensorId id{i};
    if (graph.root(id) != id) continue;
    const auto bytes = llmp::ir::StorageBytes(graph.tensor(id));
    if (!bytes || !add_payload(*bytes, 2))
      return std::unexpected("image encoded reference root inventory overflows");
  }
  for (const auto& node : graph.nodes()) {
    std::uint64_t elements = 0;
    for (const auto values : {node.sources(), node.outputs()})
      for (const auto id : values)
        if (__builtin_add_overflow(elements, llmp::ir::Elements(graph.tensor(id).shape()),
                                   &elements))
          return std::unexpected("image reference operation inventory overflows");
    std::uint64_t temporary = 0;
    if (__builtin_mul_overflow(elements, std::uint64_t{2 * sizeof(double)}, &temporary))
      return std::unexpected("image reference double temporary inventory overflows");
    largest_temporary = std::max(largest_temporary, temporary);
  }
  if (!add_payload(largest_temporary, 1))
    return std::unexpected("image reference temporary budget overflows");
  for (const auto& weight : source->spatial->weights()) {
    const auto parts = llmp::format::PartsOf(graph.tensor(weight.tensor).format);
    if (!parts) return std::unexpected("image native weight has no readable part inventory");
    for (const auto& part : parts->parts())
      if (!add_payload(part.readable, 2))
        return std::unexpected("image native payload inventory overflows");
  }
  for (const auto input : graph.inputs()) {
    const auto bytes = llmp::ir::StorageBytes(graph.tensor(input));
    if (!bytes || !add_payload(*bytes, 2))
      return std::unexpected("image changed original input inventory overflows");
  }
  for (const auto output : graph.outputs()) {
    const auto bytes = llmp::ir::StorageBytes(graph.tensor(output));
    if (!bytes || !add_payload(*bytes, 6))
      return std::unexpected("image complete output control inventory overflows");
  }
  // Retained two-frame conditioning model/bounds. Its temporary source
  // readbacks reuse the already paid largest reference-operation envelope.
  if (!add_payload(std::uint64_t{profile.width} * 2, 2 * (sizeof(float) + sizeof(double))))
    return std::unexpected("conditioning reference inventory overflows");
  if (!source_payment->Grow(payload_bytes))
    return std::unexpected("image native payload and reference budget is exhausted");

  const auto to_bf16 = [](float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
  };
  const auto& original = source->spatial->graph();
  for (std::size_t w = 0; w < source->spatial->weights().size(); ++w) {
    const auto& weight = source->spatial->weights()[w];
    const auto& tensor = original.tensor(weight.tensor);
    const auto bytes = ir::StorageBytes(tensor);
    if (!bytes || tensor.format.encoding != fm::Encoding::kBF16 || *bytes % 2 != 0)
      return std::unexpected("image source weight has an unexpected native format");
    auto& values = source->weights.emplace_back(static_cast<std::size_t>(*bytes / 2));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto centered = static_cast<int>((i * 17 + w * 13) % 41) - 20;
      const float value = static_cast<float>(centered) / 1024.F;
      values[i] = to_bf16((tensor.shape().rank == 1 ? 1.F : 0.F) + value);
    }
  }
  if (original.inputs().size() != 4 || original.outputs().size() != 2)
    return std::unexpected("image block changed its declared complete port interface");
  for (std::size_t frame = 0; frame < source->inputs.size(); ++frame) {
    const auto sinusoid = mo::ImageTimestepSinusoid(profile, frame == 0 ? 750.F : 375.F);
    for (std::size_t port = 0; port < original.inputs().size(); ++port) {
      const auto& tensor = original.tensor(original.inputs()[port]);
      const auto bytes = ir::StorageBytes(tensor);
      if (!bytes) return std::unexpected("image input byte extent is unavailable");
      auto& payload = source->inputs[frame].emplace_back(static_cast<std::size_t>(*bytes));
      const auto put = [&](std::size_t i, auto value) {
        std::memcpy(payload.data() + i * sizeof(value), &value, sizeof(value));
      };
      const auto exact = [&](fm::Encoding encoding, std::uint64_t elements) {
        return tensor.format.encoding == encoding && *bytes == elements * 4;
      };
      if (port == 0 && exact(fm::Encoding::kF32, rows * profile.width)) {
        for (std::size_t i = 0; i < *bytes / 4; ++i)
          put(i, static_cast<float>(static_cast<int>((i * 11 + frame * 29) % 97) - 48) / 64.F);
      } else if (port == 1 && exact(fm::Encoding::kF32, 2 * profile.timestep_dim) &&
                 sinusoid.size() == 2 * profile.timestep_dim) {
        // Same WHCN transpose as runtime/component_image_build.cc, not the
        // row-major BF16 array returned by ImageTimestepSinusoid.
        for (std::uint32_t c = 0; c < profile.timestep_dim; ++c)
          for (std::uint32_t row = 0; row < 2; ++row)
            put(2 * c + row, std::bit_cast<float>(
                                 std::uint32_t{sinusoid[row * profile.timestep_dim + c]} << 16U));
      } else if (port == 2 && exact(fm::Encoding::kI32, 3 * rows)) {
        for (std::uint32_t row = 0; row < rows; ++row) {
          const auto position = mo::ImageDenoiserPosition(text_rows, grid_height, grid_width, row);
          for (std::uint32_t axis = 0; axis < 3; ++axis) put(row + rows * axis, position[axis]);
        }
      } else if (port == 3 && exact(fm::Encoding::kF32, rows * rows)) {
        for (std::uint32_t q = 0; q < rows; ++q)
          for (std::uint32_t k = 0; k < rows; ++k)
            put(k + rows * q,
                q < text_rows && k > q ? -std::numeric_limits<float>::infinity() : 0.F);
      } else {
        return std::unexpected("image input port does not match its source-defined payload");
      }
    }
  }
  return source;
}

// The extra block export is the unchanged two-row timestep norm projection,
// not the final flow output covered by its relative-RMS quality gate. Model
// the actual four-product conditioning prefix without changing its payload,
// original graph, native policy or original F32 oracle.
struct ImageConditioningModel {
  std::vector<float> rounded;
  std::vector<double> bound;
};
ImageConditioningModel ImageConditioningForwardModel(const AdmittedImageBlockSource& source,
                                                     ref::Executor& cpu, std::size_t frame) {
  using Global = llmp::model::DenoiserGlobal;
  constexpr std::array roles{Global::kTimeLinear1Weight, Global::kTimeLinear2Weight,
                             Global::kModulation, Global::kNormOutWeight};
  constexpr double ub = 0x1p-8, uf = 0x1p-24, tiny = 0x1p-126;
  const auto& graph = source.spatial->graph();
  std::array<std::vector<float>, 2> activated, reference_activated;
  std::array<std::vector<double>, 2> activated_bound;
  std::vector<float> last_reference, last_model;
  std::vector<double> last_bound;
  const auto rounded = [](float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto bf16 = (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U;
    return std::bit_cast<float>(bf16 << 16U);
  };
  for (std::size_t stage = 0; stage < roles.size(); ++stage) {
    const auto& name = source.source_roles.at(static_cast<std::size_t>(roles[stage])).name;
    const auto weight = std::ranges::find_if(source.spatial->weights(), [&](const auto& item) {
      return item.parts.size() == 1 && item.parts[0].first == name;
    });
    if (weight == source.spatial->weights().end())
      Fail("conditioning model lost its actual source weight role");
    const ir::Node* product = nullptr;
    for (const auto& node : graph.nodes()) {
      if (node.key != ir::Primitive(ir::Op::kMatMul) || node.src[0] != weight->tensor) continue;
      if (product) Fail("conditioning source role has more than one original product");
      product = &node;
    }
    if (!product) Fail("conditioning source role has no original product");
    const auto& wd = graph.tensor(weight->tensor);
    const auto& xd = graph.tensor(product->src[1]);
    const auto k = xd.ne(0), n = wd.ne(1), rows = xd.ne(1);
    if (wd.format.encoding != Encoding::kBF16 || xd.format.encoding != Encoding::kF32 ||
        rows != 2 || wd.ne(0) != k || xd.ne(2) != 1 || xd.ne(3) != 1 || k == 0 ||
        k > UINT64_MAX / rows || n > UINT64_MAX / rows || n > UINT64_MAX / k ||
        2.0 * static_cast<double>(k) * uf >= 1)
      Fail("conditioning model has an unsupported actual source extent");
    const auto w = Must(cpu.Values(weight->tensor));
    const auto x = Must(cpu.Values(product->src[1]));
    const auto reference = Must(cpu.Values(product->out[0]));
    if (x.size() != k * rows || reference.size() != n * rows || w.size() != k * n)
      Fail("conditioning model omitted complete source values");
    const std::size_t previous = stage == 0 ? 0 : stage == 1 ? 0 : 1;
    if (stage != 0 && x != reference_activated[previous])
      Fail("conditioning metadata changed its original channel-major product input");
    const auto& input = stage == 0 ? x : activated[previous];
    std::vector<float> model(reference.size());
    std::vector<double> bound(reference.size());
    const double gamma =
        (2.0 * static_cast<double>(k) * uf) / (1.0 - 2.0 * static_cast<double>(k) * uf);
    constexpr double ud = 0x1p-53;
    const double gamma_double =
        (2.0 * static_cast<double>(k) * ud) / (1.0 - 2.0 * static_cast<double>(k) * ud);
    const double reference_rounding = uf + gamma_double * (1 + uf);
    double squared_gap = 0, reference_squared = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
      for (std::uint64_t j = 0; j < n; ++j) {
        double value = 0, inherited = 0, magnitude = 0, reference_magnitude = 0;
        for (std::uint64_t i = 0; i < k; ++i) {
          const auto at = row * k + i;
          const auto weight_value = static_cast<double>(w[j * k + i]);
          const auto input_error = stage == 0 ? 0 : activated_bound[previous][at];
          const auto delta = (1 + ub) * input_error + ub * std::abs(x[at]) + tiny;
          constexpr auto max_bf16 = std::bit_cast<float>(std::uint32_t{0x7f7f0000});
          if (!std::isfinite(delta) || (1 + ub) * (std::abs(x[at]) + input_error) + tiny > max_bf16)
            Fail("conditioning bound cannot certify finite BF16 input rounding");
          const auto aw = std::abs(weight_value);
          value += weight_value * rounded(input[at]);
          inherited += aw * delta;
          magnitude += aw * (std::abs(x[at]) + delta);
          reference_magnitude += aw * std::abs(x[at]);
        }
        const auto at = row * n + j;
        if (!std::isfinite(magnitude) ||
            (1 + gamma) * magnitude > std::numeric_limits<float>::max())
          Fail("conditioning bound cannot certify a finite native dot accumulation");
        model[at] = static_cast<float>(value);
        bound[at] = inherited + gamma * magnitude + reference_rounding * reference_magnitude + tiny;
        if (!std::isfinite(model[at]) || !std::isfinite(reference[at]) ||
            !std::isfinite(bound[at]) || bound[at] < 0)
          Fail("conditioning forward model or error bound is nonfinite");
        const auto gap = static_cast<double>(model[at]) - reference[at];
        squared_gap += gap * gap;
        reference_squared += static_cast<double>(reference[at]) * reference[at];
      }
    }
    Receipt("whole_image_conditioning_rounding",
            std::format("\"frame\":{},\"projection\":{},\"elements\":{},"
                        "\"original_vs_rounded_relative_rms_diagnostic\":{},"
                        "\"rounding\":\"native_bf16_input_rn_each_product\","
                        "\"model_accumulation\":\"double_then_f32_store\"",
                        frame, stage, model.size(),
                        std::sqrt(squared_gap / std::max(reference_squared, 1e-300))));
    if (stage < 2) {
      activated[stage].resize(model.size());
      reference_activated[stage].resize(model.size());
      activated_bound[stage].resize(model.size());
      for (std::size_t i = 0; i < model.size(); ++i) {
        activated[stage][i] = static_cast<float>(model[i] / (1.0 + std::exp(-double{model[i]})));
        const double original = reference[i] / (1.0 + std::exp(-double{reference[i]}));
        reference_activated[stage][i] = static_cast<float>(original);
        activated_bound[stage][i] =
            1.2 * bound[i] + 0x1p-20 * (std::abs(original) + 1.2 * bound[i]) + 1e-30;
        if (!std::isfinite(activated[stage][i]) || !std::isfinite(activated_bound[stage][i]))
          Fail("conditioning SiLU forward bound is nonfinite");
      }
    }
    if (stage == 3) {
      last_reference = reference;
      last_model = std::move(model);
      last_bound = std::move(bound);
    }
  }
  const auto output = graph.outputs()[1];
  const auto& desc = graph.tensor(output);
  const auto original = Must(cpu.Values(output));
  const auto width = source.profile.width;
  if (desc.ne(0) != 1 || desc.ne(1) != 2 || desc.ne(2) != width || desc.ne(3) != 1 ||
      original.size() != std::uint64_t{width} * 2 || last_reference.size() != original.size())
    Fail("conditioning export no longer has its original two-row WHCN layout");
  ImageConditioningModel result{.rounded = std::vector<float>(original.size()),
                                .bound = std::vector<double>(original.size())};
  for (std::size_t c = 0; c < width; ++c) {
    for (std::size_t row = 0; row < 2; ++row) {
      if (original[c * 2 + row] != last_reference[row * width + c])
        Fail("conditioning export differs from its authentic original projection");
      result.rounded[c * 2 + row] = last_model[row * width + c];
      result.bound[c * 2 + row] = last_bound[row * width + c];
    }
  }
  return result;
}

// Reference use in the existing benchmark (no alternate graph or policy):
// Executor::Create(source.spatial->graph()); bind every source.weights entry
// as BF16 bytes to spatial->weights()[i].tensor, and each source.inputs[frame]
// byte vector to original.inputs()[i], with PartView{kData, bytes}. Run the
// COMPLETE original graph and read BOTH original.outputs() without trimming.
// Restage the exact same original payloads before each ordinary/captured arm;
// poison all owned activations, preserve immutable native weights/readable
// tails, and use the existing whole-program key path (source stage1,
// ComponentPorts). Numerical bounds must come from the accepted image evidence
// contract before native execution, not from this constructor or observed
// fixture errors.

// Discovery retains committed intermediate rewrites. Selection additionally
// requires their exact outputs to survive at a scheduled registered result;
// inherited lineage alone cannot distinguish a query from a regenerated key.
struct ImageSelectedResult {
  ir::NodeId node{};
  const pl::Selected* step = nullptr;
};
ImageSelectedResult SelectedImageResult(const pl::ComparisonDiscoveryTicket& ticket,
                                        pl::DiscoveredComparisonPlan& discovered,
                                        const ir::Graph& source) {
  const auto& plan = discovered.incumbent.plan;
  const auto& graph = plan.graph;
  const auto mapping = discovered.discovery.source_to_planned();
  const auto lineage = discovered.discovery.planned_node_origins();
  const auto output_producer = [&](const pl::ComparisonSitePortRef& port) {
    if (port.member + 1 >= ticket.member_offsets.size() || !port.original_tensor.valid() ||
        port.original_tensor.v >= source.tensor_count() ||
        source.root(port.original_tensor) != port.original_root ||
        source.tensor(port.original_tensor).view_offset != port.view_offset)
      Fail("image ticket output lost its original source association");
    const auto original_producer = source.producer(port.original_tensor);
    const auto group =
        std::span(ticket.member_origins)
            .subspan(ticket.member_offsets[port.member],
                     ticket.member_offsets[port.member + 1] - ticket.member_offsets[port.member]);
    if (!std::ranges::contains(group, original_producer) ||
        port.original_tensor.v >= mapping.size())
      Fail("image ticket output is not produced by its authenticated member");
    auto tensor = mapping[port.original_tensor.v];
    // Only metadata aliases are transparent; Cont and conversions execute.
    for (std::size_t walk = 0; tensor.valid() && walk < graph.nodes().size(); ++walk) {
      if (tensor.v >= graph.tensor_count()) Fail("image output mapping is out of range");
      const auto at = graph.producer(tensor);
      if (!at.valid() || at.v >= graph.nodes().size())
        Fail("image mapped output has no actual producer");
      const auto& node = graph.node(at);
      if (node.key != ir::Primitive(ir::Op::kReshape) && node.key != ir::Primitive(ir::Op::kView) &&
          node.key != ir::Primitive(ir::Op::kPermute))
        return at;
      if (node.nsrc != 1) Fail("image output alias has an unexpected source contract");
      tensor = node.src[0];
    }
    if (tensor.valid()) Fail("image output alias mapping did not terminate");
    return ir::NodeId{};
  };
  const pl::Applied* applied = nullptr;
  for (const auto& candidate : plan.applied) {
    if (candidate.pattern != ticket.pattern ||
        candidate.replaced.size() + 1 != ticket.member_offsets.size())
      continue;
    bool match = true;
    for (std::size_t member = 0; member < candidate.replaced.size(); ++member) {
      const auto group =
          std::span(ticket.member_origins)
              .subspan(ticket.member_offsets[member],
                       ticket.member_offsets[member + 1] - ticket.member_offsets[member]);
      if (!std::ranges::contains(group, candidate.replaced[member])) match = false;
    }
    if (!match) continue;
    if (applied) Fail("image ticket matches ambiguous committed rewrites");
    applied = &candidate;
  }
  if (!applied || !ticket.boundary_representable) return {};
  ImageSelectedResult selected;
  for (const auto made : applied->made) {
    if (!made.valid() || made.v >= graph.nodes().size() ||
        graph.node(made).key != ticket.registered_operation)
      continue;
    bool has_output = false, exact_outputs = true;
    for (const auto& port : ticket.boundary_ports) {
      if (port.kind != pl::ComparisonSitePortKind::kOutput) continue;
      has_output = true;
      if (output_producer(port) != made) exact_outputs = false;
    }
    if (!has_output || !exact_outputs) continue;
    if (made.v >= lineage.size()) Fail("image primary result has no retained lineage");
    for (const auto origin : ticket.member_origins)
      if (!discovered.discovery.origins().Contains(lineage[made.v], origin))
        Fail("image primary result lost authenticated member lineage");
    for (const auto& step : plan.steps) {
      if (!std::ranges::contains(step.nodes, made)) continue;
      if (selected.step) Fail("image ticket has ambiguous scheduled primary results");
      selected = {made, &step};
    }
  }
  return selected;
}

const pl::ComparisonDiscoveryTicket* ConsumingImageTicket(
    const pl::ComparisonDiscoveryTicket& ticket, pl::DiscoveredComparisonPlan& discovered,
    const ir::Graph& source) {
  const pl::ComparisonDiscoveryTicket* consumer = nullptr;
  const auto& graph = discovered.incumbent.plan.graph;
  const auto mapping = discovered.discovery.source_to_planned();
  for (const auto& later : discovered.discovery.tickets()) {
    if (later.kind != pl::ComparisonSiteKind::kRewrite || later.ordinal <= ticket.ordinal ||
        later.generation < ticket.generation ||
        !std::ranges::all_of(
            ticket.member_origins,
            [&](ir::NodeId origin) { return std::ranges::contains(later.member_origins, origin); }))
      continue;
    const auto primary = SelectedImageResult(later, discovered, source);
    if (!primary.step) continue;
    bool has_output = false, consumed_outputs = true;
    for (const auto& port : ticket.boundary_ports) {
      if (port.kind != pl::ComparisonSitePortKind::kOutput) continue;
      has_output = true;
      if (!port.original_tensor.valid() || port.original_tensor.v >= mapping.size())
        Fail("image consumed ticket output has no original mapping");
      const auto mapped = mapping[port.original_tensor.v];
      // An absorbed result is elided, or becomes an actual output of the
      // consumer's registered primary. A surviving helper is not that primary.
      if (mapped.valid() && !std::ranges::contains(graph.node(primary.node).outputs(), mapped))
        consumed_outputs = false;
    }
    if (!has_output || !consumed_outputs) continue;
    if (consumer) Fail("image intermediate ticket has ambiguous final consumers");
    consumer = &later;
  }
  return consumer;
}

class Screen {
 public:
  explicit Screen(bool defer_launch = false, std::uint64_t metadata_capacity = kMetadata)
      : metadata_capacity_(metadata_capacity) {
    if (metadata_capacity_ != kMetadata) {
      std::ifstream meminfo("/proc/meminfo");
      std::string line;
      std::uint64_t available = 0;
      while (std::getline(meminfo, line)) {
        if (!std::string_view(line).starts_with("MemAvailable:")) continue;
        const auto first = line.find_first_of("0123456789");
        if (first == std::string::npos || line.find("kB", first) == std::string::npos)
          Fail("image metadata allowance has no typed unified memory availability");
        const auto [end, error] =
            std::from_chars(line.data() + first, line.data() + line.size(), available);
        (void)end;
        if (error != std::errc{} ||
            __builtin_mul_overflow(available, std::uint64_t{1024}, &available))
          Fail("image unified memory availability overflows");
        break;
      }
      constexpr std::uint64_t headroom = 6 * 1024 * kMiB;
      std::uint64_t required = 0;
      if (__builtin_add_overflow(metadata_capacity_, headroom, &required) || available < required)
        Fail("image metadata allowance exceeds available unified memory with 6GiB headroom");
      Receipt(
          "metadata_allowance",
          std::format(
              "\"budget_bytes\":{},\"available_unified_bytes\":{},\"required_headroom_bytes\":{}",
              metadata_capacity_, available, headroom));
    }
    execution_ = Must(llmp::providers::cuda::OpenDeviceExecution(0));
    stream_ = Must(execution_->CreateStream());
    native_ = Must(execution_->Submission(stream_));
    backend_ = Must(kn::DeviceBackend::Create(0, *execution_, stream_));
    const auto charge = [this](std::uint64_t bytes) {
      if (bytes > metadata_capacity_ - metadata_used_) return false;
      metadata_used_ += bytes;
      return true;
    };
    const auto refund = [this](std::uint64_t bytes) {
      if (bytes > metadata_used_) Fail("metadata accounting underflow");
      metadata_used_ -= bytes;
    };
    Must(backend_->FundConstruction(charge, refund));
    // Use the serving backend's declaration and workspace ownership sequence:
    // its library sizing is a device fact, and binding seals the declaration.
    Must(backend_->Declare({}, charge, refund));
    std::vector<std::uint64_t> workspaces{Allocate(backend_->library_workspace_bytes())};
    if (backend_->persistent_bytes() != 0)
      workspaces.push_back(Allocate(backend_->persistent_bytes()));
    Must(backend_->BindWorkspaces(workspaces));
    if (!defer_launch)
      launch_ = Must(backend_->Launch({Allocate(kPool), Bytes(kPool)}, /*allow_lanes=*/false));
    Receipt("backend_ready",
            std::format("\"library_workspace_bytes\":{},\"persistent_bytes\":{},\"pool_bytes\":{}",
                        backend_->library_workspace_bytes(), backend_->persistent_bytes(),
                        defer_launch ? 0 : kPool));
  }
  ~Screen() {
    Finish();
    arms_ = {};
    launch_.reset();
    backend_->ReleaseWorkspaces();
    backend_.reset();
    if (start_) Check(cudaEventDestroy(start_));
    if (stop_) Check(cudaEventDestroy(stop_));
    Must(execution_->DestroyStream(stream_));
    for (void* allocation : allocations_) Check(cudaFree(allocation));
  }
  void Run(std::uint64_t width, std::uint64_t rows, bool validation_only) {
    Run(*FindFixture("rms_norm_mul"), width, rows, validation_only);
  }
  void RunFixture(std::string_view name, std::uint64_t width, std::uint64_t rows,
                  bool validation_only) {
    const auto* fixture = FindFixture(name);
    if (!fixture) Fail("unknown fixture name");
    if (!validation_only && fixture->kind != FixtureKind::kNormMul)
      Fail("new fixtures are validation-only until U8 source freeze");
    Run(*fixture, width, rows, validation_only);
  }

  void RunReadyNorm(std::uint64_t width, std::uint64_t rows, std::string_view installed = {}) {
    RunReadyNormPair(width, rows, false, installed);
  }
  void RunReadyMatMulGlu(std::uint64_t width, std::uint64_t rows, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kMatMulGlu, width, rows, 0, false, timing);
  }
  void RunReadyNormAdd(std::uint64_t width, std::uint64_t rows, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kNormAdd, width, rows, 0, false, timing);
  }
  void RunReadyMatMulAdd(std::uint64_t width, std::uint64_t rows, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kMatMulAdd, width, rows, 0, false, timing);
  }
  void RunReadySplit(ReadyArithmetic fixture, std::uint64_t width, std::uint64_t rows,
                     std::uint32_t split, bool timing = false) {
    RunReadyArithmeticPair(fixture, width, rows, split, false, timing);
  }
  void RunReadyGdnGates(std::uint64_t rows, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kGdnGates, 64, rows, 0, false, timing);
  }
  void RunReadyHcCombine(std::uint64_t rows, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kHcCombine, 4096, rows, 0, false, timing);
  }
  void RunReadyHcPostNorm(std::uint64_t rows, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kHcPostNorm, 4096, rows, 0, false, timing);
  }
  void RunReadySpatialNorm(std::uint64_t channels, bool silu, bool timing = false) {
    RunReadyArithmeticPair(ReadyArithmetic::kSpatialNorm, channels, 37 * 3, 0, silu, timing);
  }
  void ValidateInstalledArithmetic(ReadyArithmetic fixture, std::uint64_t width, std::uint64_t rows,
                                   std::uint32_t split, bool silu, std::string_view installed) {
    RunReadyArithmeticPair(fixture, width, rows, split, silu, false, installed);
  }
  void RunReadyNormGate(std::uint64_t dim, std::uint64_t heads, std::uint64_t rows,
                        bool timing = false) {
    RunReadyNormGatePair(dim, heads, rows, timing);
  }
  void TimeReadyNorm(std::uint64_t width, std::uint64_t rows) {
    RunReadyNormPair(width, rows, true);
  }

  void RunReadyNormRope(std::uint64_t width, std::uint64_t rows, bool timing = false) {
    RunReadyNormRopePair(width, rows, timing);
  }

  void RunReadyNormRopeSections(std::uint64_t width, std::uint64_t rows, std::uint32_t sections,
                                bool bf16_gamma, bool timing = false) {
    const std::array one{64U};
    const std::array two{32U, 32U};
    const std::array three{16U, 56U, 56U};
    const std::span<const std::uint32_t> dims = sections == 1 ? std::span<const std::uint32_t>(one)
                                                : sections == 2
                                                    ? std::span<const std::uint32_t>(two)
                                                    : std::span<const std::uint32_t>(three);
    RunReadyNormRopePair(width, rows, timing,
                         sections == 3 ? ir::RopeMode::kNeox : ir::RopeMode::kNormal, dims,
                         bf16_gamma, 3);
  }

  // The ordinary production EXL3 pair: no Ready adapter or capture promotion.
  std::unique_ptr<AdmittedImageBlockSource> MakeWholeImageSource(
      const std::filesystem::path& composition_path, std::uint32_t rows, bool prepare_payloads) {
    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    auto loan = ir::AcquireStorageLoan(funder, llmp::spec::kVaeSpatialSpecBytes + 65536);
    if (!loan) Fail("image source mapper metadata budget is exhausted");
    auto payment = std::make_shared<ir::StorageLoan>(std::move(*loan));
    // These are source metadata files only; weight shards are not read here.
    // The parser's retained strings/containers have a bounded envelope over
    // the actual encoded documents, paid before opening them.
    const auto pay_documents = [&](const std::filesystem::path& path) {
      std::uint64_t bytes = 0;
      std::error_code error;
      for (const auto name : {"manifest.json", "spec.json", "index.json"}) {
        const auto file = path / name;
        if (!std::filesystem::exists(file, error)) {
          if (error) Fail("image metadata file inspection failed");
          continue;
        }
        const auto size = std::filesystem::file_size(file, error);
        if (error || __builtin_add_overflow(bytes, size, &bytes))
          Fail("image metadata document inventory overflows");
      }
      if (__builtin_mul_overflow(bytes, std::uint64_t{8}, &bytes))
        Fail("image parsed document metadata inventory overflows");
      const auto meta = path / "meta";
      if (std::filesystem::exists(meta, error)) {
        auto entry = std::filesystem::recursive_directory_iterator(meta, error);
        const auto end = std::filesystem::recursive_directory_iterator{};
        while (!error && entry != end) {
          if (entry->is_regular_file(error)) {
            const auto size = entry->file_size(error);
            if (error || __builtin_add_overflow(bytes, size, &bytes))
              Fail("image kept metadata inventory overflows");
          }
          entry.increment(error);
        }
        if (error) Fail("image kept metadata inspection failed");
      } else if (error) {
        Fail("image kept metadata inspection failed");
      }
      if (!payment->Grow(bytes)) Fail("image source document metadata budget is exhausted");
    };
    pay_documents(composition_path);
    auto opened = llmp::artifact::OpenComposition(composition_path);
    if (!opened) Fail(opened.error().ToString());
    auto composition = std::make_shared<const llmp::artifact::Composition>(std::move(*opened));
    std::vector<llmp::artifact::Artifact> artifacts;
    std::vector<const llmp::artifact::Artifact*> components;
    artifacts.reserve(composition->components().size());
    components.reserve(composition->components().size());
    for (const auto& component : composition->components()) {
      const auto path = composition_path.parent_path() / component.artifact;
      pay_documents(path);
      auto artifact = llmp::artifact::Artifact::Open(path, {.expected_id = component.artifact});
      if (!artifact) Fail(artifact.error().ToString());
      if (artifact->model().architecture != component.architecture)
        Fail("image component architecture differs from its composition");
      artifacts.push_back(std::move(*artifact));
      components.push_back(&artifacts.back());
    }
    const co::SpatialPayment graph_payment{
        [funder](std::uint64_t bytes) { return funder->Charge(bytes); },
        [funder](std::uint64_t bytes) { funder->Release(bytes); }};
    auto numerical = std::make_shared<const kn::ConstructionFunder>(
        kn::ConstructionFunder{graph_payment.charge, graph_payment.release});
    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("image backend owner is not ready");
    const auto& table = owned.owner->backend();
    auto source =
        Must(MakeAdmittedImageBlockSource(composition, components, payment, graph_payment,
                                          numerical, table.device, rows, prepare_payloads));
    return source;
  }

  void PreflightWholeImageBlock(const std::filesystem::path& composition_path, std::uint32_t rows) {
    auto source = MakeWholeImageSource(composition_path, rows, false);
    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    const auto owned = backend_->Backend();
    const auto& table = owned.owner->backend();
    const auto io = source->io();
    const auto external = SourceExternalBindings(io);
    const co::ComparisonBoundaryView view{.io = io, .active_stage = 1, .external = external};
    kn::Addresses alignment{};
    alignment.activation_alignment = 256;
    const auto options = Must(en::PrepareComponentPlanOptions(io, table, alignment));
    std::vector<ir::OpKey> filter;
    for (const auto& pattern : pl::Patterns())
      if (pattern.kind == pl::PatternKind::kRewrite &&
          !std::ranges::contains(filter, pattern.result))
        filter.push_back(pattern.result);
    auto discovered = Must(pl::DiscoverComparisonPlan(view, filter, false, funder, *table.registry,
                                                      table.device, source->policy, options));
    for (const auto& ticket : discovered.discovery.tickets()) {
      std::string groups = "[";
      for (std::size_t g = 0; g + 1 < ticket.member_offsets.size(); ++g) {
        if (g) groups += ',';
        groups += '[';
        for (auto i = ticket.member_offsets[g]; i < ticket.member_offsets[g + 1]; ++i) {
          if (i != ticket.member_offsets[g]) groups += ',';
          groups += std::format("{}", ticket.member_origins[i].v);
        }
        groups += ']';
      }
      groups += ']';
      Receipt("whole_image_site",
              std::format("\"pattern\":{},\"version\":{},\"kind\":{},\"boundary_representable\":{},"
                          "\"actual_original_groups\":{}",
                          Quote(ticket.pattern), ticket.version, static_cast<unsigned>(ticket.kind),
                          ticket.boundary_representable, groups));
    }
    std::string calls = "[";
    for (const auto& step : discovered.incumbent.plan.steps) {
      if (calls.size() > 1) calls += ',';
      calls += Quote(table.registry->at(step.implementation).name);
    }
    calls += ']';
    Receipt(
        "whole_image_preflight",
        std::format(
            "\"composition\":{},\"source_graph\":{},\"rows\":{},\"width\":256,\"heads\":2,"
            "\"head_dim\":128,\"mlp\":512,\"inputs\":4,\"outputs\":2,"
            "\"synthetic_native_bf16_payloads\":true,\"default_implementations\":{},"
            "\"complete_output_relative_rms_limit\":0.005,\"native_whole_proof_qualified\":false,"
            "\"ranking_samples_collected\":false",
            Quote(source->composition->id()), Quote(llmp::base::ToHex(io.graph.Digest())), rows,
            calls));
  }

  void RunWholeImageBlock(const std::filesystem::path& composition_path, std::uint32_t rows,
                          bool timing = false, std::string_view installed = {}) {
    auto source = MakeWholeImageSource(composition_path, rows, true);
    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    const auto io = source->io();
    const auto owned = backend_->Backend();
    const auto& table = owned.owner->backend();
    auto external = SourceExternalBindings(io);
    std::vector<std::uint64_t> weight_addresses;
    for (std::size_t w = 0; w < io.weights.size(); ++w) {
      const auto parts = Must(llmp::format::PartsOf(io.graph.tensor(io.weights[w].tensor).format));
      if (parts.size() != 1 || parts[0].role != llmp::format::PartRole::kData ||
          parts[0].bytes != source->weights[w].size() * sizeof(std::uint16_t) ||
          parts[0].readable != parts[0].bytes)
        Fail("image native BF16 weight has an unexpected readable part extent");
      const auto bytes = std::as_bytes(std::span(source->weights[w]));
      weight_addresses.push_back(Allocate(parts[0].readable));
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(weight_addresses.back()), bytes.data(),
                            bytes.size(), cudaMemcpyHostToDevice, CudaStream()));
      const auto fact = std::ranges::find(external, io.weights[w].tensor,
                                          &co::ComparisonExternalBinding::original_root);
      if (fact == external.end() || fact->part != llmp::format::PartRole::kData)
        Fail("image actual BF16 source part lost its boundary identity");
      // These are deterministic native arithmetic-fixture bytes, not the
      // imported model's weight bytes. Both identities remain explicit.
      fact->resource_part_identity =
          llmp::base::Sha256{}
              .Update("llmp-admitted-image-native-fixture-v1;stage=1;")
              .Update(source->composition->id())
              .Update(std::format("{}:{}:{};", io.weights[w].ref.stage, io.weights[w].ref.component,
                                  io.weights[w].ref.role))
              .Update(bytes)
              .Finish();
    }
    Finish();
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole role) {
      const auto at = std::ranges::find(io.weights, ref, &co::StepWeight::ref);
      return at != io.weights.end() && role == llmp::format::PartRole::kData
                 ? weight_addresses[static_cast<std::size_t>(at - io.weights.begin())]
                 : 0;
    };
    addresses.weight_range = [&](const ir::WeightRef& ref,
                                 llmp::format::PartRole role) -> std::optional<kn::BoundedAddress> {
      const auto at = std::ranges::find(io.weights, ref, &co::StepWeight::ref);
      if (at == io.weights.end() || role != llmp::format::PartRole::kData) return {};
      const auto w = static_cast<std::size_t>(at - io.weights.begin());
      const auto fact =
          std::ranges::find(external, at->tensor, &co::ComparisonExternalBinding::original_root);
      if (fact == external.end() || !fact->resource_part_identity)
        Fail("image actual weight range lost its source content identity");
      return kn::BoundedAddress{.base = weight_addresses[w],
                                .bytes = source->weights[w].size() * sizeof(std::uint16_t),
                                .alignment = 256,
                                .content_identity = fact->resource_part_identity};
    };
    addresses.weight_alignment = [](const ir::WeightRef&, llmp::format::PartRole) {
      return std::uint64_t{256};
    };
    addresses.activation_alignment = 256;
    const auto options = Must(en::PrepareComponentPlanOptions(io, table, addresses));
    // The measurement and normal runtime derive the identical real source
    // boundary from the same IO and owner-bounded part tuple.
    auto source_bindings = Must(en::ComparisonSourceBindings::Create(funder, io, 1));
    {
      auto initial = Must(en::PlanComponentGraph(io, table, source->policy, addresses, options));
      addresses.activations = Allocate(initial.placement.extent);
      addresses.activation_bytes = initial.placement.extent;
      auto initial_ranges = Must(source_bindings.Resolve(initial, addresses, table.written));
    }
    const auto source_view = source_bindings.boundary_view();
    if (!source_view) Fail("image source boundary changed after resolving actual ranges");
    const auto view = *source_view;
    std::vector<ir::OpKey> filter;
    for (const auto& pattern : pl::Patterns())
      if (pattern.kind == pl::PatternKind::kRewrite &&
          !std::ranges::contains(filter, pattern.result))
        filter.push_back(pattern.result);
    auto discovered = Must(pl::DiscoverComparisonPlan(view, filter, false, funder, *table.registry,
                                                      table.device, source->policy, options));
    // Complete original CPU outputs are shared by every accepted comparison.
    auto cpu = Must(ref::Executor::Create(io.graph));
    for (std::size_t w = 0; w < io.weights.size(); ++w) {
      const std::array parts{ref::PartView{.role = llmp::format::PartRole::kData,
                                           .bytes = std::as_bytes(std::span(source->weights[w]))}};
      Must(cpu.Bind(io.weights[w].tensor, parts));
    }
    std::array<std::vector<std::vector<float>>, 2> expected;
    std::array<ImageConditioningModel, 2> conditioning_models;
    for (std::size_t frame = 0; frame < 2; ++frame) {
      for (std::size_t input = 0; input < io.inputs.size(); ++input) {
        const std::array parts{ref::PartView{.role = llmp::format::PartRole::kData,
                                             .bytes = source->inputs[frame][input]}};
        Must(cpu.Bind(io.inputs[input], parts));
      }
      Must(cpu.Run());
      for (const auto output : io.outputs) expected[frame].push_back(Must(cpu.Values(output)));
      conditioning_models[frame] = ImageConditioningForwardModel(*source, cpu, frame);
      if (expected[frame].size() != 2) Fail("image CPU omitted an original output");
      if (frame != 0)
        for (std::size_t output = 0; output < io.outputs.size(); ++output)
          if (expected[frame][output] == expected[0][output])
            Fail("image changed source frame did not change a complete CPU output");
    }
    // One existing bounded launch serves each independent complete program.
    launch_ = Must(backend_->Launch({Allocate(kPool), Bytes(kPool)}, false));
    std::size_t occurrence = 0;
    bool production_checked = false;
    for (const auto& ticket : discovered.discovery.tickets()) {
      if (ticket.kind != pl::ComparisonSiteKind::kRewrite) continue;
      phase = "whole_image_classify";
      const auto selected = SelectedImageResult(ticket, discovered, io.graph);
      if (!selected.step) {
        const auto* consumer = ConsumingImageTicket(ticket, discovered, io.graph);
        if (!consumer) Fail("image accepted ticket has no exact final selection or consumer");
        const auto primary = SelectedImageResult(*consumer, discovered, io.graph);
        std::string groups = "[";
        for (std::size_t member = 0; member + 1 < ticket.member_offsets.size(); ++member) {
          if (member) groups += ',';
          groups += '[';
          for (auto i = ticket.member_offsets[member]; i < ticket.member_offsets[member + 1]; ++i) {
            if (i != ticket.member_offsets[member]) groups += ',';
            groups += std::format("{}", ticket.member_origins[i].v);
          }
          groups += ']';
        }
        groups += ']';
        Receipt(
            "whole_image_consumed_ticket",
            std::format(
                "\"ticket_ordinal\":{},\"pattern\":{},\"actual_original_groups\":{},"
                "\"consumer_ordinal\":{},\"consumer_pattern\":{},\"consumer_operation\":{},"
                "\"consumer_implementation\":{},\"reason\":\"original output elided or replaced by "
                "the scheduled consumer primary\",\"independent_comparison\":false",
                ticket.ordinal, Quote(ticket.pattern), groups, consumer->ordinal,
                Quote(consumer->pattern),
                Quote(ir::OpKeyName(discovered.incumbent.plan.graph.node(primary.node).key)),
                Quote(table.registry->at(primary.step->implementation).name)));
        continue;
      }
      phase = "whole_image_prepare";
      Receipt("whole_image_comparison",
              std::format("\"occurrence\":{},\"pattern\":{},\"rows\":{},\"active_stage\":1,"
                          "\"synthetic_native_bf16_payloads\":true",
                          occurrence, Quote(ticket.pattern), rows));
      pl::ComparisonSiteSelector selector;
      selector.source_graph_identity = io.graph.identity();
      selector.kind = ticket.kind;
      selector.pattern = ticket.pattern;
      selector.version = ticket.version;
      selector.member_owner_segments = ticket.owner_segments;
      selector.boundary_ports = ticket.boundary_ports;
      for (std::size_t g = 0; g + 1 < ticket.member_offsets.size(); ++g)
        selector.member_origins.emplace_back(
            ticket.member_origins.begin() + ticket.member_offsets[g],
            ticket.member_origins.begin() + ticket.member_offsets[g + 1]);
      phase = "whole_image_bound_comparison";
      Receipt("whole_image_metadata", std::format("\"live_bytes\":{},\"budget_bytes\":{}",
                                                  metadata_used_, metadata_capacity_));
      auto inventory = Must(pl::BoundComparison(view, selector, funder, *table.registry,
                                                table.device, source->policy, options));
      phase = "whole_image_make_alternative";
      auto alternative = Must(pl::MakeAlternative(inventory));
      phase = "whole_image_plan_alternative";
      auto candidate = Must(pl::PlanAlternative(inventory, std::move(alternative), *table.registry,
                                                table.device, source->policy, options));
      std::vector<std::uint32_t> offsets{0};
      std::vector<ir::NodeId> members;
      for (const auto& group : inventory.site.accepted_member_origins) {
        members.insert(members.end(), group.begin(), group.end());
        offsets.push_back(static_cast<std::uint32_t>(members.size()));
      }
      phase = "whole_image_site_boundary";
      auto region = Must(co::MakeComparisonRegionBoundary(
          {.source = view, .member_offsets = offsets, .members = members}, funder));
      const std::array<const pl::OwnedComparisonPlan*, 2> plans{&inventory.incumbent, &candidate};
      const auto capacity =
          std::max(plans[0]->plan.placement.extent, plans[1]->plan.placement.extent);
      addresses.activations = Allocate(capacity);
      addresses.activation_bytes = capacity;
      std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
      std::array<kn::ComparisonRecipeSeal, 2> seals;
      for (std::size_t arm = 0; arm < 2; ++arm) {
        std::vector<kn::ComparisonBindingRange> bindings;
        for (const auto& fact : inventory.source_boundary.external()) {
          const auto& root = inventory.source_boundary.roots()[fact.canonical_root];
          std::uint64_t address = 0;
          if (fact.part) {
            const auto w =
                std::ranges::find(io.weights, root.original_root, &co::StepWeight::tensor);
            if (w == io.weights.end()) Fail("image whole binding lost a weight source");
            address = addresses.weight(w->ref, *fact.part);
          } else {
            const auto mapped = plans[arm]->plan.tensors.at(root.original_root.v);
            if (!mapped.valid()) Fail("image original port lost its planned tensor");
            const auto planned_root = plans[arm]->plan.graph.root(mapped);
            const auto at = std::ranges::find(plans[arm]->plan.placement.offsets, planned_root,
                                              &std::pair<ir::TensorId, std::uint64_t>::first);
            if (at == plans[arm]->plan.placement.offsets.end())
              Fail("image original port has no actual planned placement");
            address = addresses.activations + at->second;
          }
          bindings.push_back({.original_root = root.original_root,
                              .part = fact.part,
                              .allocation_class = fact.canonical_allocation_class,
                              .allocation_bytes = fact.allocation_bytes,
                              .offset = fact.offset,
                              .bytes = fact.bytes,
                              .alignment = fact.alignment,
                              .base = address,
                              .resource_part_identity = fact.resource_part_identity});
        }
        const kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                            .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                            .bindings = bindings,
                                            .active_stage = 1,
                                            .expected_boundary = &inventory.source_boundary};
        const kn::LowerInput input(plans[arm]->plan, io, addresses, std::nullopt, &request);
        phase = arm == 0 ? "whole_image_lower_incumbent" : "whole_image_lower_alternative";
        programs[arm] = Must(table.lower(input));
        std::string selected_calls = "[";
        for (const auto& step : plans[arm]->plan.steps) {
          if (selected_calls.size() > 1) selected_calls += ',';
          selected_calls += Quote(table.registry->at(step.implementation).name);
        }
        selected_calls += ']';
        Receipt("whole_image_selected_plan",
                std::format("\"occurrence\":{},\"arm\":{},\"implementations\":{}", occurrence, arm,
                            selected_calls));
        seals[arm] = WholeDescriptor(*programs[arm], *plans[arm], io.graph,
                                     inventory.source_boundary, *table.registry);
        if (programs[arm]->pool_bytes() > kPool || programs[arm]->private_pool_bytes() > kPool ||
            programs[arm]->scratch_bytes() > capacity)
          Fail("image complete program exceeds its actual owned workspace");
        Must(programs[arm]->Bind(*launch_));
        const auto bound = programs[arm]->comparison_recipe();
        if (!bound || !en::SameWholeProgramRecipe(seals[arm], *bound))
          Fail("image whole bound seal differs from its descriptor");
        const auto* calls = programs[arm]->bound();
        if (!calls) Fail("image bound program omitted its call inventory");
        std::string inventory_json = "[";
        for (std::size_t call = 0; call < calls->size(); ++call) {
          if (call) inventory_json += ',';
          inventory_json += Quote(calls->at(call).name);
        }
        inventory_json += ']';
        Receipt("whole_image_arm",
                std::format("\"occurrence\":{},\"arm\":{},\"whole_bound_verified\":true,"
                            "\"actual_calls\":{},\"scheduled_nodes\":{}",
                            occurrence, arm, inventory_json, seals[arm].scheduled_calls));
      }
      const bool ordinary =
          seals[0].execution_method == kn::ComparisonExecutionMethod::kOrdinaryOnly ||
          seals[1].execution_method == kn::ComparisonExecutionMethod::kOrdinaryOnly;
      const auto key = en::RecipeComparisonKeyForWholePrograms(
          inventory.source_boundary, region, inventory.site, ex::Phase::kPrefill,
          table.registry->SelectionDigest(), seals[0], seals[1]);
      if (!key) Fail("image accepted whole pair has no authenticated key");
      Receipt(
          "whole_image_key",
          std::format("\"occurrence\":{},\"recipe_key\":{},\"registered_rows\":{},\"matched\":{},"
                      "\"ordinary_run_only\":{}",
                      occurrence, RecipeKeyJson(*key),
                      table.registry->ranks().recipe_comparisons.size(),
                      table.registry->FindRecipeComparison(*key) != nullptr, ordinary));
      const auto stage = [&](std::size_t arm, std::size_t frame) {
        Check(cudaMemsetAsync(reinterpret_cast<void*>(addresses.activations), 0xff, capacity,
                              CudaStream()));
        const auto slots = programs[arm]->inputs();
        if (slots.size() != io.inputs.size()) Fail("image full program omitted an original input");
        for (std::size_t input = 0; input < io.inputs.size(); ++input) {
          const auto slot = std::ranges::find(slots, io.inputs[input], &kn::InputSlot::tensor);
          if (slot == slots.end() || slot->kind != kn::InputKind::kStepBytes ||
              slot->bytes != source->inputs[frame][input].size())
            Fail("image original typed input identity or extent changed");
          Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot->device_address),
                                source->inputs[frame][input].data(), slot->bytes,
                                cudaMemcpyHostToDevice, CudaStream()));
        }
        Finish();
      };
      const auto validate = [&](std::size_t arm, std::size_t frame, bool captured) {
        const auto outputs = programs[arm]->outputs();
        if (outputs.size() != io.outputs.size()) Fail("image full program omitted an output");
        for (std::size_t output = 0; output < outputs.size(); ++output) {
          const auto& want = expected[frame][output];
          if (outputs[output].tensor != io.outputs[output] ||
              outputs[output].bytes != want.size() * sizeof(float))
            Fail("image complete output identity or extent changed");
          std::vector<float> actual(want.size());
          Check(cudaMemcpy(actual.data(),
                           reinterpret_cast<const void*>(outputs[output].device_address),
                           outputs[output].bytes, cudaMemcpyDeviceToHost));
          double reference_squared = 0, gap_squared = 0, maximum_gap = 0;
          double maximum_forward_ratio = 0, rounded_gap_squared = 0;
          for (std::size_t i = 0; i < want.size(); ++i) {
            if (!std::isfinite(want[i]) || !std::isfinite(actual[i]))
              Fail("image full output is nonfinite");
            const double gap = static_cast<double>(actual[i]) - want[i];
            gap_squared += gap * gap;
            reference_squared += static_cast<double>(want[i]) * want[i];
            maximum_gap = std::max(maximum_gap, std::abs(gap));
            if (output == 1) {
              const auto& conditioning = conditioning_models[frame];
              if (conditioning.bound.size() != want.size() ||
                  conditioning.rounded.size() != want.size() ||
                  !std::isfinite(conditioning.bound[i]) || conditioning.bound[i] <= 0)
                Fail("image conditioning error bound lost its complete source extent");
              maximum_forward_ratio =
                  std::max(maximum_forward_ratio, std::abs(gap) / conditioning.bound[i]);
              const double rounded_gap = static_cast<double>(actual[i]) - conditioning.rounded[i];
              rounded_gap_squared += rounded_gap * rounded_gap;
            }
          }
          const auto relative_rms = std::sqrt(gap_squared / std::max(reference_squared, 1e-300));
          const auto reference_rms =
              std::sqrt(reference_squared / static_cast<double>(want.size()));
          if (!std::isfinite(relative_rms)) Fail("image complete-output error is nonfinite");
          Receipt(
              "whole_image_output",
              std::format(
                  "\"occurrence\":{},\"arm\":{},\"frame\":{},\"output\":{},\"captured\":{},"
                  "\"elements\":{},\"relative_rms\":{},\"relative_rms_limit\":{},"
                  "\"max_absolute_over_reference_rms_diagnostic\":{},"
                  "\"maximum_conditioning_forward_error_ratio\":{},"
                  "\"gpu_vs_bf16_rounded_prefix_gap_over_original_rms_diagnostic\":{}",
                  occurrence, arm, frame, output, captured, want.size(), relative_rms,
                  output == 0 ? "0.005" : "null", maximum_gap / std::max(reference_rms, 1e-300),
                  output == 1 ? std::format("{}", maximum_forward_ratio) : "null",
                  output == 1 ? std::format("{}", std::sqrt(rounded_gap_squared /
                                                            std::max(reference_squared, 1e-300)))
                              : "null"));
          if ((output == 0 && relative_rms > 5e-3) || (output == 1 && maximum_forward_ratio > 1))
            Fail("image complete output exceeds its prospective source error contract");
        }
        for (std::size_t w = 0; w < io.weights.size(); ++w) {
          const auto want = std::as_bytes(std::span(source->weights[w]));
          std::vector<std::byte> after(want.size());
          Check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(weight_addresses[w]),
                           after.size(), cudaMemcpyDeviceToHost));
          if (!std::ranges::equal(after, want))
            Fail("image execution changed immutable native BF16 parts/tails");
        }
      };
      std::array<std::unique_ptr<kn::Recorded>, 2> recorded;
      const auto run_frame = [&](std::size_t arm, std::size_t frame, bool captured, bool measure) {
        stage(arm, frame);
        if (captured) {
          Must(launch_->Launch(native_, *recorded[arm]));
        } else {
          if (measure) Check(cudaEventRecord(start_, CudaStream()));
          Must(programs[arm]->Run(*launch_, native_));
          if (measure) Check(cudaEventRecord(stop_, CudaStream()));
        }
        Finish();
        double duration = 0;
        if (captured) {
          duration = Must(recorded[arm]->gpu_duration_ns());
        } else if (measure) {
          float milliseconds = 0;
          Check(cudaEventElapsedTime(&milliseconds, start_, stop_));
          duration = static_cast<double>(milliseconds) * 1'000'000.0;
        }
        if ((captured || measure) &&
            (!std::isfinite(duration) || duration < 0.5 ||
             duration >= static_cast<double>(std::numeric_limits<long long>::max())))
          Fail("image duration cannot be represented as positive nanoseconds");
        validate(arm, frame, captured);
        if (captured)
          Receipt("whole_image_capture",
                  std::format("\"occurrence\":{},\"arm\":{},\"frame\":{},\"nodes\":{},"
                              "\"body_invocations\":1,\"duration_checked\":true",
                              occurrence, arm, frame, recorded[arm]->nodes()));
        return captured || measure ? static_cast<std::uint64_t>(std::llround(duration)) : 0;
      };
      for (std::size_t frame = 0; frame < 2; ++frame)
        for (std::size_t arm = 0; arm < 2; ++arm) (void)run_frame(arm, frame, false, false);
      if (!ordinary) {
        for (std::size_t arm = 0; arm < 2; ++arm) {
          stage(arm, 0);
          Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
          auto capture =
              launch_->Capture(native_, [&] { return programs[arm]->Run(*launch_, native_); });
          auto disabled = launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled);
          Must(std::move(disabled));
          recorded[arm] = Must(std::move(capture));
          if (recorded[arm]->nodes() <= 2) Fail("image capture omitted its complete ordinary body");
        }
        for (std::size_t frame = 0; frame < 2; ++frame)
          for (std::size_t arm = 0; arm < 2; ++arm) (void)run_frame(arm, frame, true, false);
      }
      const auto* installed_row = table.registry->FindRecipeComparison(*key);
      if (installed == "miss" && installed_row)
        Fail("installed-row miss control unexpectedly matched this exact source");
      if (!installed.empty() && !production_checked && (installed == "miss" || installed_row)) {
        // No injected ranks: this is the ordinary component entry used by
        // StagedExecution, over the currently generated rank table.
        std::vector<std::pair<const kn::StepProgram*, llmp::base::Sha256Digest>> lowered_graphs;
        const kn::ProgramValidator observe =
            [&](const kn::LowerInput& input,
                const kn::StepProgram& program) -> std::expected<void, std::string> {
          if (&input.io.graph != &io.graph || input.token)
            return std::unexpected("installed image changed its original component source domain");
          const auto previous = std::ranges::find(
              lowered_graphs, &program.comparison_program(),
              &std::pair<const kn::StepProgram*, llmp::base::Sha256Digest>::first);
          if (previous == lowered_graphs.end())
            lowered_graphs.emplace_back(&program.comparison_program(), input.plan.graph.Digest());
          else
            previous->second = input.plan.graph.Digest();
          return {};
        };
        auto production = Must(en::BuildComponentProgram(
            io, table, source->policy, addresses, options, std::nullopt, observe, owned.owner));
        const auto preference = installed_row ? ex::RecipeComparisonPreference(*installed_row)
                                              : ex::RecipePreference::kIncumbent;
        const auto chosen = preference == ex::RecipePreference::kAlternative ? 1U : 0U;
        const auto actual_graph =
            std::ranges::find(lowered_graphs, &production->comparison_program(),
                              &std::pair<const kn::StepProgram*, llmp::base::Sha256Digest>::first);
        const auto descriptor = production->comparison_recipe();
        if (!descriptor && (actual_graph == lowered_graphs.end() ||
                            actual_graph->second != seals[chosen].planned_graph))
          Fail("normal component selection changed its exact planned graph");
        const auto expected_plan = programs[chosen]->bound()->identity();
        const bool expected_capturable = programs[chosen]->capturable();
        if ((installed_row && !descriptor) || (descriptor && *descriptor != seals[chosen]))
          Fail("normal component selection differs from its first matching whole disposition");
        Must(production->Bind(*launch_));
        const auto bound = production->comparison_recipe();
        const bool whole_bound = bound && en::SameWholeProgramRecipe(seals[chosen], *bound);
        if ((installed_row || descriptor) && !whole_bound)
          Fail("installed component recipe changed at Bind");
        if (!production->bound() || production->bound()->identity() != expected_plan ||
            (!installed_row && production->capturable() != expected_capturable))
          Fail(
              "normal component selection changed its ordered bound implementation plan or method");
        recorded = {};
        programs[chosen] = std::move(production);
        std::string calls = "[";
        if (!programs[chosen]->bound()) Fail("installed component omitted actual bound calls");
        for (std::size_t call = 0; call < programs[chosen]->bound()->size(); ++call) {
          if (call) calls += ',';
          calls += Quote(programs[chosen]->bound()->at(call).name);
        }
        calls += ']';
        for (std::size_t frame = 0; frame < 2; ++frame)
          (void)run_frame(chosen, frame, false, false);
        if (!ordinary) {
          stage(chosen, 0);
          Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
          auto capture =
              launch_->Capture(native_, [&] { return programs[chosen]->Run(*launch_, native_); });
          auto disabled = launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled);
          Must(std::move(disabled));
          recorded[chosen] = Must(std::move(capture));
          for (std::size_t frame = 0; frame < 2; ++frame)
            (void)run_frame(chosen, frame, true, false);
        }
        Receipt(
            "whole_image_installed_selection",
            std::format("\"expected\":{},\"matched\":{},\"record_key\":{},"
                        "\"preferred_arm\":{},\"whole_bound_verified\":{},\"bound_selected_plan_"
                        "verified\":true,\"frames\":2,"
                        "\"normal_component_entry\":true,\"actual_calls\":{},"
                        "\"losing_fusion_veto_verified\":{},\"ranking_samples_collected\":false",
                        Quote(installed), installed_row != nullptr,
                        installed_row ? RecipeKeyJson(installed_row->key) : "null", chosen,
                        whole_bound, calls, installed_row && chosen == 1));
        production_checked = true;
      }
      if (timing) {
        const auto protocol = std::format(
            "llmp-whole-admitted-image-event-timing-v1;prep={};"
            "frames=2-complete-original-ir-outputs-validated-before-events;"
            "input=frame0-all-original-f32-i32-restage-every-invocation;arena=nan-poison;"
            "weights=synthetic-native-bf16-composition-stage-bound-immutable-readable-tails;"
            "body=one-complete-whole-program;body-invocations=1;"
            "events={};pair-method={};"
            "outputs=complete-flow-relative-rms-5e-3:conditioning-elementwise-forward-bound-v1;"
            "fixed-warmup-pairs=4:incumbent-then-alternative;"
            "bookends=9-alternative-before-and-after:median;"
            "alternating-warmup-pairs=4:start-incumbent-then-alternate;"
            "alternating-sample-pairs=9:start-incumbent-then-alternate;"
            "observations=52;bookend-drift=10pct;no-filtering;no-auto-retry",
            llmp::base::ToHex(key->preparation_domain),
            ordinary ? "external-start-run-stop-on-owning-stream"
                     : "captured-external-start-body-stop",
            ordinary ? "ordinary-run-for-both-arms" : "captured-single-body-for-both-arms");
        const auto shape = std::format("whole.image.block.bf16.width256.rows{}.{}.site{}", rows,
                                       ticket.pattern, occurrence);
        MeasureReadyBody(
            *key, protocol, shape,
            [&](en::ComparisonPairArm arm) {
              return run_frame(static_cast<std::size_t>(arm), 0, !ordinary, true);
            },
            /*captured_events=*/!ordinary, /*bookend_observations=*/9);
      }
      Finish();
      recorded = {};
      programs = {};
      Receipt(
          "whole_image_validated",
          std::format("\"occurrence\":{},\"pattern\":{},\"frames\":2,\"outputs\":2,"
                      "\"weights_and_readable_tails_unchanged\":true,\"single_invocation\":true,"
                      "\"captured\":{},\"ranking_samples_collected\":{}",
                      occurrence, Quote(ticket.pattern), !ordinary, timing));
      ++occurrence;
    }
    if (!installed.empty() && !production_checked)
      Fail("installed-row hit control found no actual matching row");
    if (occurrence == 0) Fail("image source has no accepted original rewrite comparison");
  }

  void RunWholeGluProducts(std::uint32_t rows, const std::filesystem::path& artifact_root,
                           bool timing = false, bool f32_input = false, bool q8 = false,
                           std::string_view installed = {}) {
    if (q8 && (rows != 1 || !f32_input))
      Fail("Q8 whole GLU requires its single R1 F32 source cell");
    namespace ar = llmp::artifact;
    using llmp::format::PartRole;
    constexpr std::uint64_t k = 896, n = 4864;
    std::optional<ar::Artifact> opened;
    if (!q8) {
      auto artifact = ar::Artifact::Open(artifact_root);
      if (!artifact) Fail(artifact.error().ToString());
      opened.emplace(std::move(*artifact));
      if (opened->format_version() != 1) Fail("whole GLU requires the published v1 artifact");
    }
    constexpr std::array exl3_roles{PartRole::kCodes, PartRole::kInputSigns, PartRole::kOutputSigns,
                                    PartRole::kCodebookMarker};
    constexpr std::array q8_roles{PartRole::kData};
    const auto roles =
        q8 ? std::span<const PartRole>(q8_roles) : std::span<const PartRole>(exl3_roles);
    const auto event = [q8](std::string_view exl3, std::string_view quantized) {
      return q8 ? quantized : exl3;
    };
    constexpr std::array suffixes{".trellis", ".suh", ".svh", ".mcg"};
    const std::array<std::string, 2> names{"model.layers.0.mlp.gate_proj",
                                           "model.layers.0.mlp.up_proj"};
    std::array<std::array<std::uint32_t, 4>, 2> resources{};
    std::array<llmp::format::TensorFormat, 2> formats;
    std::vector<co::StepWeight> weights;
    ir::Builder builder;
    builder.SetOwner(0);
    const auto input_encoding = f32_input ? Encoding::kF32 : Encoding::kF16;
    const auto input_bytes = k * rows * (f32_input ? sizeof(float) : sizeof(__half));
    const auto target_pattern = q8          ? "mul_mat_glu"
                                : f32_input ? "exl3_products"
                                            : "paired_glu_products";
    const auto x = builder.Input(ir::Dense(input_encoding, ir::ShapeOf({k, rows})),
                                 q8 ? "q8-glu-input" : "exl3-input");
    std::array<ir::TensorId, 2> weight_tensors, products;
    for (std::size_t w = 0; w < 2; ++w) {
      if (q8) {
        formats[w] = {.encoding = Encoding::kQ8_0,
                      .layout = llmp::format::Layout::kGgmlBlocks,
                      .shape = ir::ShapeOf({k, n})};
      } else
        for (std::size_t p = 0; p < roles.size(); ++p) {
          const auto found = opened->FindResource(names[w] + suffixes[p]);
          if (!found) Fail("published EXL3 artifact is missing a layer-zero gate/up part");
          resources[w][p] = *found;
          const auto& resource = opened->resources()[*found];
          if (resource.part != roles[p] || !resource.format)
            Fail("published EXL3 resource has a different physical role or no format");
          if (p == 0) formats[w] = *resource.format;
          if (*resource.format != formats[w]) Fail("whole GLU multipart formats disagree");
        }
      const auto& format = formats[w];
      if (!q8 && (format.encoding != Encoding::kExl3Trellis ||
                  format.layout != llmp::format::Layout::kExl3Tiles || format.params.k_bits != 4 ||
                  format.params.codebook != llmp::format::Exl3Codebook::kMcg ||
                  format.shape != ir::ShapeOf({k, n})))
        Fail("published EXL3 gate/up does not match the existing 896x4864 MCG4 cell");
      const ir::WeightRef ref{.stage = 0, .component = 1, .role = static_cast<std::uint16_t>(w)};
      weight_tensors[w] = builder.Weight(format, ref, names[w]);
      co::StepWeight weight{.tensor = weight_tensors[w], .ref = ref, .parts = {}};
      for (std::size_t p = 0; p < roles.size(); ++p)
        weight.parts.emplace_back(q8 ? names[w] : names[w] + suffixes[p], roles[p]);
      weights.push_back(std::move(weight));
      products[w] = builder.MatMul(weight_tensors[w], x);
    }
    const auto y = builder.Glu(products[0], products[1], {.kind = ir::GluKind::kSwiGlu});
    builder.Output(y);
    auto graph = Must(std::move(builder).Finish());
    const en::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), weights};
    std::array<std::array<std::vector<std::byte>, 4>, 2> host_parts;
    if (q8) {
      // Real native Q8_0 bytes. The original CPU graph decodes exactly these
      // bytes; the initializer's unquantized values are never the oracle.
      std::vector<float> values(k * n);
      for (std::size_t w = 0; w < weights.size(); ++w) {
        for (std::size_t i = 0; i < values.size(); ++i) {
          std::uint64_t mixed = i + 8191 + w * 1000003ULL + 0x9E3779B97F4A7C15ULL;
          mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
          mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
          mixed ^= mixed >> 31U;
          values[i] = static_cast<float>(static_cast<int>(mixed >> 53U) - 1024) / 8192.0F;
        }
        const auto parts = Must(llmp::format::PartsOf(formats[w]));
        if (parts.size() != 1 || parts[0].role != PartRole::kData)
          Fail("Q8 original weight must have exactly one native data part");
        host_parts[w][0].resize(parts[0].readable);
        Must(ref::EncodeRows(formats[w], 0, 0, values, host_parts[w][0]));
      }
    }
    auto external = SourceExternalBindings(io);
    for (auto& fact : external) {
      if (!fact.part) {
        fact.alignment = 256;
        continue;
      }
      const auto weight = std::ranges::find(weights, fact.original_root, &co::StepWeight::tensor);
      if (weight == weights.end()) Fail("whole GLU boundary names an unknown weight");
      const auto part =
          std::ranges::find(weight->parts, *fact.part, &std::pair<std::string, PartRole>::second);
      if (part == weight->parts.end()) Fail("whole GLU boundary names an unknown part");
      if (q8) {
        const auto w = static_cast<std::size_t>(weight - weights.begin());
        fact.resource_part_identity = llmp::base::Sha256{}
                                          .Update("llmp-u6-native-q8-glu-fixture-v1;")
                                          .Update(part->first)
                                          .Update(host_parts[w][0])
                                          .Finish();
      } else {
        fact.resource_part_identity =
            llmp::base::Sha256{}.Update(opened->id()).Update(part->first).Finish();
      }
    }
    const pl::Policy policy{.active_stage = 0};
    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    const co::ComparisonBoundaryView view{.io = io, .external = external};
    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("whole GLU backend owner is not ready");
    const auto& table = owned.owner->backend();
    if (!table.device.preferred.empty())
      Fail("whole GLU must use the ordinary unpreferred backend");
    const std::array filter{q8          ? ir::Fused(ir::FusedKind::kMatMulGlu)
                            : f32_input ? ir::Primitive(ir::Op::kConvert)
                                        : ir::Fused(ir::FusedKind::kPreparedProducts)};
    kn::Addresses alignment{};
    alignment.activation_alignment = 256;
    const auto options = Must(en::PrepareComponentPlanOptions(io, table, alignment));
    auto discovered = Must(pl::DiscoverComparisonPlan(view, filter, false, funder, *table.registry,
                                                      table.device, policy, options));
    if (q8)
      Receipt("whole_q8_contract",
              "\"weight_encoding\":\"q8_0\",\"synthetic_native_payloads\":true,"
              "\"numerical_grants\":0,\"cpu_metric\":\"nmse\","
              "\"cpu_bound\":0.0005,\"peer_normalized_max_bound\":0.00001");
    const pl::ComparisonDiscoveryTicket* ticket = nullptr;
    for (const auto& item : discovered.discovery.tickets()) {
      if (item.pattern != target_pattern || item.kind != pl::ComparisonSiteKind::kRewrite) continue;
      if (ticket) Fail("whole GLU default source has ambiguous target rewrite tickets");
      ticket = &item;
    }
    std::string selected = "[";
    for (const auto& step : discovered.incumbent.plan.steps) {
      if (selected.size() > 1) selected += ',';
      selected += Quote(table.registry->at(step.implementation).name);
    }
    selected += ']';
    Receipt(event("whole_exl3_preflight", "whole_q8_preflight"),
            std::format("\"artifact\":{},\"rows\":{},\"k\":896,\"n\":4864,"
                        "\"native_weight_encoding\":{},\"preferred_implementations\":0,"
                        "\"target_pattern\":{},\"input_encoding\":{},\"target_site_present\":{},"
                        "\"default_implementations\":{},"
                        "\"ranking_samples_collected\":false",
                        q8 ? "null" : Quote(opened->id()), rows, Quote(q8 ? "q8_0" : "exl3-mcg4"),
                        Quote(target_pattern), Quote(f32_input ? "f32" : "f16"), ticket != nullptr,
                        selected));
    const std::string_view default_product = rows == 1 ? "exl3.pinned.40g.multi_linear.mgemm"
                                             : rows == 145
                                                 ? "exl3.pinned.40g.linear.reconstruct"
                                                 : "exl3.pinned.40g.linear.reconstruct_fused";
    const auto& default_steps = discovered.incumbent.plan.steps;
    if (q8 &&
        (default_steps.size() != 1 ||
         table.registry->at(default_steps[0].implementation).name != "ggml.mul_mat_glu.mmvq_fused"))
      Fail("Q8 default source did not select its genuine ungranted mul_mat_glu");
    if (!f32_input &&
        (default_steps.size() != (rows == 1 ? 2U : 3U) ||
         table.registry->at(default_steps.front().implementation).name != default_product ||
         table.registry->at(default_steps.back().implementation).name != "ggml.swiglu" ||
         (rows != 1 &&
          table.registry->at(default_steps[1].implementation).name != default_product)))
      Fail("whole GLU ordinary default inventory differs from the source-backed cell");
    if (rows != 1) {
      if (ticket) Fail("whole GLU nonselection control unexpectedly has a paired default site");
      Receipt("whole_exl3_nonselection_validated",
              "\"native_whole_proof_qualified\":false,\"payload_allocated\":false,"
              "\"cpu_execution_performed\":false,\"capture_promoted\":false");
      return;
    }
    if (!ticket || !ticket->boundary_representable)
      Fail("whole GLU R1 default source has no complete accepted target rewrite site");
    pl::ComparisonSiteSelector selector;
    selector.source_graph_identity = graph.identity();
    selector.kind = ticket->kind;
    selector.pattern = ticket->pattern;
    selector.version = ticket->version;
    selector.member_owner_segments = ticket->owner_segments;
    selector.boundary_ports = ticket->boundary_ports;
    for (std::size_t g = 0; g + 1 < ticket->member_offsets.size(); ++g)
      selector.member_origins.emplace_back(
          ticket->member_origins.begin() + ticket->member_offsets[g],
          ticket->member_origins.begin() + ticket->member_offsets[g + 1]);
    auto inventory = Must(pl::BoundComparison(view, selector, funder, *table.registry, table.device,
                                              policy, options));
    auto alternative = Must(pl::MakeAlternative(inventory));
    auto candidate = Must(pl::PlanAlternative(inventory, std::move(alternative), *table.registry,
                                              table.device, policy, options));
    std::vector<std::uint32_t> offsets{0};
    std::vector<ir::NodeId> members;
    for (const auto& group : inventory.site.accepted_member_origins) {
      members.insert(members.end(), group.begin(), group.end());
      offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    auto site_boundary = Must(co::MakeComparisonRegionBoundary(
        {.source = view, .member_offsets = offsets, .members = members}, funder));
    // Only R1 reads payload. R145/R1024 above are genuine default nonselection controls.
    std::array<std::array<kn::BoundedAddress, 4>, 2> addresses_parts;
    for (std::size_t w = 0; w < 2; ++w) {
      const auto parts = Must(llmp::format::PartsOf(formats[w]));
      for (std::size_t p = 0; p < roles.size(); ++p) {
        const auto extent =
            std::ranges::find(parts.parts(), roles[p], &llmp::format::PartExtent::role);
        if (extent == parts.parts().end()) Fail("whole GLU format omitted an actual artifact part");
        auto& bytes = host_parts[w][p];
        if (!q8) {
          const auto placement = Must(opened->ResourcePlacement(resources[w][p]));
          if (placement.bytes.value() != extent->bytes || extent->readable < extent->bytes ||
              extent->readable >
                  static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
            Fail("whole GLU artifact part extent differs from its format");
          const auto& group = opened->groups()[placement.group];
          const auto& shard = opened->shards()[group.shard];
          std::uint64_t at = shard.data_offset.value();
          for (auto add : {group.offset.value(), placement.offset.value()}) {
            if (add > UINT64_MAX - at) Fail("whole GLU payload offset overflows");
            at += add;
          }
          if (at > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
            Fail("whole GLU payload offset exceeds the file interface");
          bytes.resize(static_cast<std::size_t>(extent->readable));
          std::ifstream input(artifact_root / shard.path, std::ios::binary);
          input.seekg(static_cast<std::streamoff>(at));
          input.read(reinterpret_cast<char*>(bytes.data()),
                     static_cast<std::streamsize>(extent->bytes));
          if (!input || input.gcount() != static_cast<std::streamsize>(extent->bytes))
            Fail("published EXL3 part read is incomplete");
        }
        const auto fact = std::ranges::find_if(external, [&](const auto& binding) {
          return binding.original_root == weight_tensors[w] && binding.part == roles[p];
        });
        if (fact == external.end() || !fact->resource_part_identity)
          Fail("whole GLU part lost its original content identity");
        addresses_parts[w][p] = {.base = Allocate(extent->readable),
                                 .bytes = extent->readable,
                                 .alignment = 16,
                                 .content_identity = fact->resource_part_identity};
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(addresses_parts[w][p].base), bytes.data(),
                              bytes.size(), cudaMemcpyHostToDevice, CudaStream()));
        Receipt(event("whole_exl3_part", "whole_q8_part"),
                std::format("\"resource\":{},\"bytes\":{},\"readable\":{},\"sha256\":{}",
                            Quote(q8 ? names[w] : opened->resources()[resources[w][p]].name),
                            extent->bytes, extent->readable,
                            Quote(llmp::base::ToHex(llmp::base::Sha256{}.Update(bytes).Finish()))));
      }
    }
    Finish();
    const std::array<const pl::OwnedComparisonPlan*, 2> plans{&inventory.incumbent, &candidate};
    const auto capacity =
        std::max(plans[0]->plan.placement.extent, plans[1]->plan.placement.extent);
    const auto arena = Allocate(capacity);
    kn::Addresses addresses;
    addresses.activations = arena;
    addresses.activation_bytes = capacity;
    addresses.activation_alignment = 256;
    const auto part_address = [&](const ir::WeightRef& ref, PartRole role) -> kn::BoundedAddress {
      const auto w = std::ranges::find(weights, ref, &co::StepWeight::ref);
      const auto p = std::ranges::find(roles, role);
      if (w == weights.end() || p == roles.end()) return {};
      return addresses_parts[static_cast<std::size_t>(w - weights.begin())]
                            [static_cast<std::size_t>(p - roles.begin())];
    };
    addresses.weight = [&](const ir::WeightRef& ref, PartRole role) {
      return part_address(ref, role).base;
    };
    addresses.weight_range = [&](const ir::WeightRef& ref,
                                 PartRole role) -> std::optional<kn::BoundedAddress> {
      const auto found = part_address(ref, role);
      return found.base ? std::optional(found) : std::nullopt;
    };
    std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
    std::array<kn::ComparisonRecipeSeal, 2> seals;
    std::array<std::uint64_t, 2> input_addresses{};
    std::uint64_t pool_bytes = 256;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto& plan = plans[arm]->plan;
      std::vector<kn::ComparisonBindingRange> bindings;
      for (const auto& fact : inventory.source_boundary.external()) {
        const auto& root = inventory.source_boundary.roots()[fact.canonical_root];
        std::uint64_t address = 0;
        if (fact.part) {
          const auto w = std::ranges::find(weights, root.original_root, &co::StepWeight::tensor);
          if (w == weights.end()) Fail("whole GLU whole binding lost a weight root");
          address = part_address(w->ref, *fact.part).base;
        } else {
          const auto mapped = plan.tensors.at(root.original_root.v);
          if (!mapped.valid()) Fail("whole GLU original input lost its mapped tensor");
          const auto mapped_root = plan.graph.root(mapped);
          const auto placed = std::ranges::find_if(plan.placement.offsets, [&](const auto& item) {
            return plan.graph.root(item.first) == mapped_root;
          });
          if (placed == plan.placement.offsets.end())
            Fail("whole GLU input has no actual placement");
          address = arena + placed->second;
          if (root.original_root != x || fact.offset != 0 || fact.bytes != input_bytes)
            Fail("whole GLU original input binding has a different identity or extent");
          input_addresses[arm] = address;
        }
        bindings.push_back({.original_root = root.original_root,
                            .part = fact.part,
                            .allocation_class = fact.canonical_allocation_class,
                            .allocation_bytes = fact.allocation_bytes,
                            .offset = fact.offset,
                            .bytes = fact.bytes,
                            .alignment = fact.alignment,
                            .base = address,
                            .resource_part_identity = fact.resource_part_identity});
      }
      const kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                          .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                          .bindings = bindings,
                                          .expected_boundary = &inventory.source_boundary};
      const kn::LowerInput input(plan, io, addresses, std::nullopt, &request);
      auto lowered = table.lower(input);
      if (!lowered) Fail(lowered.error());
      programs[arm] = std::move(*lowered);
      seals[arm] = WholeDescriptor(*programs[arm], *plans[arm], graph, inventory.source_boundary,
                                   *table.registry);
      pool_bytes =
          std::max({pool_bytes, programs[arm]->pool_bytes(), programs[arm]->private_pool_bytes()});
    }
    launch_ = Must(backend_->Launch({Allocate(pool_bytes), Bytes(pool_bytes)}, false));
    for (std::size_t arm = 0; arm < 2; ++arm) {
      Must(programs[arm]->Bind(*launch_));
      const auto bound = programs[arm]->comparison_recipe();
      if (!bound || !en::SameWholeProgramRecipe(seals[arm], *bound))
        Fail("whole GLU whole bound seal differs from its descriptor");
      std::string calls = "[";
      const auto* actual = programs[arm]->bound();
      if (!actual || actual->size() == 0 || (!f32_input && actual->size() != (arm == 0 ? 2U : 3U)))
        Fail(
            "whole GLU bound call inventory is not its complete pair or two-product decomposition");
      const auto paired_calls =
          std::ranges::count_if(plans[arm]->plan.steps, [&](const auto& step) {
            return table.registry->at(step.implementation).operation ==
                   ir::Fused(ir::FusedKind::kPreparedProducts);
          });
      if (!f32_input && paired_calls != (arm == 0 ? 1 : 0))
        Fail("whole GLU own alternative did not remove the accepted paired-product operation");
      for (std::size_t i = 0; i < actual->size(); ++i) {
        if (i) calls += ',';
        calls += Quote(actual->at(i).name);
      }
      calls += ']';
      Receipt(event("whole_exl3_arm", "whole_q8_arm"),
              std::format(
                  "\"arm\":{},\"whole_bound_verified\":true,"
                  "\"execution_method\":{},\"actual_calls\":{},"
                  "\"program_pool_bytes\":{},\"private_pool_bytes\":{},\"launch_pool_bytes\":{},"
                  "\"activation_capacity\":{}",
                  arm,
                  Quote(seals[arm].execution_method == kn::ComparisonExecutionMethod::kOrdinaryOnly
                            ? "ordinary_only"
                            : "captured"),
                  calls, programs[arm]->pool_bytes(), programs[arm]->private_pool_bytes(),
                  pool_bytes, capacity));
    }
    const bool ordinary =
        seals[0].execution_method == kn::ComparisonExecutionMethod::kOrdinaryOnly ||
        seals[1].execution_method == kn::ComparisonExecutionMethod::kOrdinaryOnly;
    if (!q8 && seals[0].execution_method != kn::ComparisonExecutionMethod::kOrdinaryOnly)
      Fail("whole GLU default pinned pair did not declare its ordinary-only method");
    if (q8 && ordinary) Fail("Q8 whole GLU requires both ordinary and captured validation");
    auto key = en::RecipeComparisonKeyForWholePrograms(
        inventory.source_boundary, site_boundary, inventory.site, ex::Phase::kPrefill,
        table.registry->SelectionDigest(), seals[0], seals[1]);
    if (!key) Fail("whole GLU ordinary pair has no authenticated whole key");
    Receipt(event("whole_exl3_key", "whole_q8_key"),
            std::format("\"recipe_key\":{},\"registered_rows\":{},\"matched\":{},"
                        "\"ordinary_run_only\":{},\"capture_promoted\":false",
                        RecipeKeyJson(*key), table.registry->ranks().recipe_comparisons.size(),
                        table.registry->FindRecipeComparison(*key) != nullptr, ordinary));
    auto cpu = Must(ref::Executor::Create(graph));
    for (std::size_t w = 0; w < 2; ++w) {
      std::array<ref::PartView, 4> parts;
      for (std::size_t p = 0; p < roles.size(); ++p)
        parts[p] = {.role = roles[p], .bytes = host_parts[w][p]};
      Must(cpu.Bind(weight_tensors[w], std::span(parts).first(roles.size())));
    }
    std::vector<float> first_reference;
    std::array<std::vector<float>, 2> first_actual;
    std::vector<float> current_peer;
    std::array<std::unique_ptr<kn::Recorded>, 2> recorded;
    const auto run_frame = [&](std::uint32_t frame, std::size_t arm, bool measure,
                               bool captured = false) -> std::uint64_t {
      std::vector<__half> half_values(k * rows);
      std::vector<float> float_values(f32_input ? k * rows : 0);
      for (std::size_t i = 0; i < half_values.size(); ++i) {
        std::uint64_t mixed = i + 4241 + frame * 1000003ULL + 0x9E3779B97F4A7C15ULL;
        mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
        mixed ^= mixed >> 31U;
        const auto value = static_cast<float>(static_cast<int>(mixed >> 53U) - 1024) /
                           (f32_input ? 1000.0F : 1024.0F);
        half_values[i] = __float2half_rn(value);
        if (f32_input) float_values[i] = value;
      }
      const auto original_bytes = f32_input ? std::as_bytes(std::span(float_values))
                                            : std::as_bytes(std::span(half_values));
      const std::array<ref::PartView, 1> input_parts{
          {{.role = PartRole::kData, .bytes = original_bytes}}};
      Must(cpu.Bind(x, input_parts));
      Must(cpu.Run());
      const auto copy_values = [&](ir::TensorId tensor) {
        const auto values = Must(cpu.Values(tensor));
        return std::vector<float>(values.begin(), values.end());
      };
      const auto gate = copy_values(products[0]);
      const auto up = copy_values(products[1]);
      const auto expected = copy_values(y);
      if (expected.size() != n * rows || gate.size() != expected.size() ||
          up.size() != expected.size())
        Fail("whole GLU CPU reference omitted complete original outputs");
      if (frame != 0 && std::ranges::equal(expected, first_reference))
        Fail("whole GLU changed original input did not change the CPU output");
      if (frame == 0) first_reference.assign(expected.begin(), expected.end());
      std::vector<double> gate_rms(rows), up_rms(rows);
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t j = 0; j < n; ++j) {
          gate_rms[row] += static_cast<double>(gate[row * n + j]) * gate[row * n + j];
          up_rms[row] += static_cast<double>(up[row * n + j]) * up[row * n + j];
        }
        gate_rms[row] = std::sqrt(gate_rms[row] / static_cast<double>(n));
        up_rms[row] = std::sqrt(up_rms[row] / static_cast<double>(n));
        if (!(gate_rms[row] > 0 && up_rms[row] > 0))
          Fail("whole GLU CPU product row is degenerate");
      }
      if (f32_input && !q8) {
        // Keep the original F32 graph as the quality oracle. Separately expose
        // the mandatory native input rounding; neither oracle nor bound is
        // changed to make the preparation look exact.
        std::vector<float> rounded_values(half_values.size());
        for (std::size_t i = 0; i < half_values.size(); ++i)
          rounded_values[i] = __half2float(half_values[i]);
        const std::array<ref::PartView, 1> rounded_parts{
            {{.role = PartRole::kData, .bytes = std::as_bytes(std::span(rounded_values))}}};
        Must(cpu.Bind(x, rounded_parts));
        Must(cpu.Run());
        for (std::size_t product = 0; product < products.size(); ++product) {
          const auto rounded = Must(cpu.Values(products[product]));
          const auto& original = product == 0 ? gate : up;
          double squared = 0, reference_squared = 0, maximum = 0;
          for (std::size_t i = 0; i < original.size(); ++i) {
            const auto gap = static_cast<double>(rounded[i]) - original[i];
            squared += gap * gap;
            reference_squared += static_cast<double>(original[i]) * original[i];
            maximum = std::max(maximum, std::abs(gap));
          }
          const auto rounding_rms = std::sqrt(squared / std::max(reference_squared, 1e-300));
          if (!std::isfinite(rounding_rms)) Fail("whole GLU mandatory input rounding is nonfinite");
          Receipt("whole_exl3_input_rounding",
                  std::format("\"arm\":{},\"frame\":{},\"product\":{},"
                              "\"relative_rms_diagnostic\":{},\"maximum_gap_diagnostic\":{},"
                              "\"quality_oracle\":\"original_f32\"",
                              arm, frame, product, rounding_rms, maximum));
        }
      }
      Check(cudaMemsetAsync(reinterpret_cast<void*>(arena), 0xff, capacity, CudaStream()));
      const auto slots = programs[arm]->inputs();
      if (slots.size() != 1 || slots[0].tensor != x || slots[0].kind != kn::InputKind::kStepBytes ||
          slots[0].device_address != input_addresses[arm] || slots[0].bytes != input_bytes)
        Fail("whole GLU original input slot or extent changed");
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(slots[0].device_address), original_bytes.data(),
                            slots[0].bytes, cudaMemcpyHostToDevice, CudaStream()));
      Finish();
      if (measure && !captured) Check(cudaEventRecord(start_, CudaStream()));
      if (captured) {
        if (ordinary) Fail("ordinary-only whole pair cannot be promoted to capture");
        if (!recorded[arm]) {
          Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
          auto result =
              launch_->Capture(native_, [&] { return programs[arm]->Run(*launch_, native_); });
          auto disabled = launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled);
          Must(std::move(disabled));
          recorded[arm] = Must(std::move(result));
          if (recorded[arm]->nodes() <= 2) Fail("Q8 capture omitted its complete ordinary body");
        }
        Must(launch_->Launch(native_, *recorded[arm]));
      } else {
        Must(programs[arm]->Run(*launch_, native_));
      }
      if (measure && !captured) Check(cudaEventRecord(stop_, CudaStream()));
      Finish();
      std::uint64_t elapsed_ns = 0;
      if (captured) {
        const auto duration = Must(recorded[arm]->gpu_duration_ns());
        if (!std::isfinite(duration) || duration <= 0) Fail("Q8 capture duration is invalid");
        if (measure) {
          if (duration < 0.5 ||
              duration >= static_cast<double>(std::numeric_limits<long long>::max()))
            Fail("captured Q8 duration cannot be represented as positive nanoseconds");
          elapsed_ns = static_cast<std::uint64_t>(std::llround(duration));
        }
      }
      if (measure && !captured) {
        float milliseconds = 0;
        Check(cudaEventElapsedTime(&milliseconds, start_, stop_));
        const double duration = static_cast<double>(milliseconds) * 1'000'000.0;
        if (!std::isfinite(duration) || duration < 0.5 ||
            duration >= static_cast<double>(std::numeric_limits<long long>::max()))
          Fail("ordinary EXL3 duration cannot be represented as positive nanoseconds");
        elapsed_ns = static_cast<std::uint64_t>(std::llround(duration));
      }
      const auto outputs = programs[arm]->outputs();
      if (outputs.size() != 1 || outputs[0].bytes != expected.size() * sizeof(float))
        Fail("whole GLU complete ordinary output extent changed");
      std::vector<float> actual(expected.size());
      Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(outputs[0].device_address),
                       outputs[0].bytes, cudaMemcpyDeviceToHost));
      std::array<bool, 2> product_readback{};
      for (std::size_t product = 0; !q8 && product < products.size(); ++product) {
        const auto& plan = plans[arm]->plan;
        const auto mapped = plan.tensors.at(products[product].v);
        if (!mapped.valid()) Fail("whole GLU original product lost its mapped tensor");
        auto root = plan.graph.root(mapped);
        std::uint64_t offset = plan.graph.tensor(mapped).view_offset;
        const auto within = table.within ? table.within(plan.graph, root) : ir::TensorId{};
        if (within.valid()) {
          const auto& made = plan.graph.node(plan.graph.producer(root));
          if (made.key != ir::Fused(ir::FusedKind::kPreparedProducts) || made.nout != 2 ||
              root != made.out[1] || within != made.out[0])
            Fail("whole GLU product has an unexpected shared-output storage contract");
          // The actual selected pair's second view begins at one complete
          // output plane (lower.cc's MakeSelected, checked by whole proof).
          offset += plan.graph.tensor(mapped).nb[2];
          root = within;
        }
        const auto placed = std::ranges::find(plan.placement.offsets, root,
                                              &std::pair<ir::TensorId, std::uint64_t>::first);
        if (placed == plan.placement.offsets.end() || placed->second > capacity ||
            offset > capacity - placed->second ||
            outputs[0].bytes > capacity - placed->second - offset)
          Fail("whole GLU complete product lies outside its actual arena");
        const auto address = arena + placed->second + offset;
        const auto output_address = outputs[0].device_address;
        const bool overlaps = address <= output_address
                                  ? output_address - address < outputs[0].bytes
                                  : address - output_address < outputs[0].bytes;
        // Dead products may be overwritten by the final GLU. No Keep or
        // source output is added merely to preserve diagnostic readback.
        if (overlaps) continue;
        product_readback[product] = true;
        std::vector<float> observed(expected.size());
        Check(cudaMemcpy(observed.data(), reinterpret_cast<const void*>(address), outputs[0].bytes,
                         cudaMemcpyDeviceToHost));
        const auto& reference_product = product == 0 ? gate : up;
        const auto& reference_rms = product == 0 ? gate_rms : up_rms;
        for (std::size_t row = 0; row < rows; ++row) {
          double squared_gap = 0, largest_gap = 0;
          for (std::size_t j = 0; j < n; ++j) {
            const auto i = row * n + j;
            if (!std::isfinite(observed[i])) Fail("whole GLU retained product is nonfinite");
            const auto gap = std::abs(static_cast<double>(observed[i]) - reference_product[i]);
            squared_gap += gap * gap;
            largest_gap = std::max(largest_gap, gap);
          }
          if (std::sqrt(squared_gap / static_cast<double>(n)) / reference_rms[row] > 0x1p-8 ||
              largest_gap / reference_rms[row] > 0x1p-5)
            Fail("whole GLU retained product exceeds the existing per-row CPU reference bounds");
        }
      }
      double maximum_ratio = 0;
      if (q8) {
        double squared = 0, energy = 0, peer_gap = 0, peer_scale = 1;
        for (std::size_t i = 0; i < actual.size(); ++i) {
          if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            Fail("Q8 complete output is nonfinite");
          const double gap = static_cast<double>(actual[i]) - expected[i];
          squared += gap * gap;
          energy += static_cast<double>(expected[i]) * expected[i];
          if (arm != 0) {
            if (current_peer.size() != actual.size()) Fail("Q8 production peer is absent");
            peer_gap =
                std::max(peer_gap, std::abs(static_cast<double>(actual[i]) - current_peer[i]));
            peer_scale = std::max(peer_scale, std::abs(static_cast<double>(current_peer[i])));
          }
        }
        const double nmse = squared / std::max(energy, 1e-300);
        Receipt("whole_q8_quality",
                std::format("\"arm\":{},\"frame\":{},\"captured\":{},"
                            "\"nmse\":{},\"cpu_nmse_bound\":0.0005,\"peer_normalized_max\":{},"
                            "\"peer_bound\":0.00001",
                            arm, frame, captured, nmse, peer_gap / peer_scale));
        if (nmse > 5e-4 || (arm != 0 && peer_gap / peer_scale > 1e-5))
          Fail("Q8 whole GLU exceeds its prospective CPU or strict peer bound");
        // Peer comparison uses the matching current frame/method, never an old stop/input.
        if (arm == 0) current_peer = actual;
      } else
        for (std::size_t i = 0; i < actual.size(); ++i) {
          const auto row = i / n;
          const double g = gate[i], u = up[i], bg = 0x1p-5 * gate_rms[row],
                       bu = 0x1p-5 * up_rms[row];
          const double silu = g / (1.0 + std::exp(-g));
          const double tolerance = 1.2 * bg * (std::abs(u) + bu) + std::abs(silu) * bu +
                                   0x1p-20 * std::abs(static_cast<double>(expected[i])) + 1e-30;
          if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            Fail("whole GLU output is nonfinite");
          maximum_ratio = std::max(
              maximum_ratio, std::abs(static_cast<double>(actual[i]) - expected[i]) / tolerance);
        }
      if (maximum_ratio > 1)
        Fail("whole GLU complete SwiGLU output exceeded its prospective propagated bound");
      if (frame != 0 && actual == first_actual[arm])
        Fail("whole GLU changed input did not change its ordinary output");
      if (frame == 0) first_actual[arm] = actual;
      for (std::size_t w = 0; w < 2; ++w)
        for (std::size_t p = 0; p < roles.size(); ++p) {
          std::vector<std::byte> after(host_parts[w][p].size());
          Check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(addresses_parts[w][p].base),
                           after.size(), cudaMemcpyDeviceToHost));
          if (after != host_parts[w][p])
            Fail("whole GLU ordinary execution changed a weight part or readable tail");
        }
      if (q8) {
        Receipt(
            "whole_q8_validated",
            std::format(
                "\"arm\":{},\"frame\":{},\"captured\":{},"
                "\"elements\":{},\"weight_parts_and_tails_unchanged\":true,"
                "\"body_invocations\":1,\"capture_nodes\":{},\"ranking_samples_collected\":false",
                arm, frame, captured, actual.size(), recorded[arm] ? recorded[arm]->nodes() : 0));
      } else
        Receipt("whole_exl3_validated",
                std::format("\"arm\":{},\"frame\":{},\"elements\":{},"
                            "\"maximum_propagated_error_ratio\":{},"
                            "\"weight_parts_and_tails_unchanged\":true,"
                            "\"ordinary_invocations\":1,\"product_readback\":[{},{}],"
                            "\"product_relative_rms_limit\":0.00390625,"
                            "\"product_max_error_over_reference_rms_limit\":0.03125,"
                            "\"timing_observation\":{}",
                            arm, frame, actual.size(), maximum_ratio, product_readback[0],
                            product_readback[1], measure));
      return elapsed_ns;
    };
    for (std::uint32_t frame = 0; frame < 2; ++frame)
      for (std::size_t arm = 0; arm < 2; ++arm) (void)run_frame(frame, arm, false);
    if (q8 && !ordinary) {
      for (std::uint32_t frame = 0; frame < 2; ++frame)
        for (std::size_t arm = 0; arm < 2; ++arm) (void)run_frame(frame, arm, false, true);
    }
    if (!installed.empty()) {
      if (q8 || !ordinary) Fail("installed EXL3 validation requires the actual OrdinaryOnly pair");
      const auto* row = table.registry->FindRecipeComparison(*key);
      if ((installed == "hit") != (row != nullptr))
        Fail("installed EXL3 row lookup differs from the requested hit or miss");
      const auto preference =
          row ? ex::RecipeComparisonPreference(*row) : ex::RecipePreference::kIncumbent;
      const std::size_t chosen = preference == ex::RecipePreference::kAlternative ? 1 : 0;
      const auto expected_plan = programs[chosen]->bound()->identity();
      std::vector<std::pair<const kn::StepProgram*, llmp::base::Sha256Digest>> lowered_graphs;
      const kn::ProgramValidator observe =
          [&](const kn::LowerInput& input,
              const kn::StepProgram& program) -> std::expected<void, std::string> {
        if (&input.io.graph != &io.graph || input.token)
          return std::unexpected("installed EXL3 changed its original ComponentPorts source");
        const auto previous =
            std::ranges::find(lowered_graphs, &program.comparison_program(),
                              &std::pair<const kn::StepProgram*, llmp::base::Sha256Digest>::first);
        if (previous == lowered_graphs.end())
          lowered_graphs.emplace_back(&program.comparison_program(), input.plan.graph.Digest());
        else
          previous->second = input.plan.graph.Digest();
        return {};
      };
      auto production = Must(en::BuildComponentProgram(io, table, policy, addresses, options,
                                                       std::nullopt, observe, owned.owner));
      const auto actual_graph =
          std::ranges::find(lowered_graphs, &production->comparison_program(),
                            &std::pair<const kn::StepProgram*, llmp::base::Sha256Digest>::first);
      const auto descriptor = production->comparison_recipe();
      if ((!descriptor && (actual_graph == lowered_graphs.end() ||
                           actual_graph->second != seals[chosen].planned_graph)) ||
          (row && !descriptor) || (descriptor && *descriptor != seals[chosen]))
        Fail(
            "normal EXL3 construction changed its exact preferred whole descriptor or planned "
            "graph");
      Must(production->Bind(*launch_));
      const auto bound = production->comparison_recipe();
      const bool whole_bound = bound && en::SameWholeProgramRecipe(seals[chosen], *bound);
      if (((row || descriptor) && !whole_bound) || !production->bound() ||
          production->bound()->identity() != expected_plan || production->capturable())
        Fail("installed EXL3 identity or pairwide OrdinaryOnly method changed at Bind");
      const auto slots = production->inputs();
      if (slots.size() != 1 || slots[0].tensor != x || slots[0].bytes != input_bytes)
        Fail("installed EXL3 changed the actual original input port");
      input_addresses[chosen] = slots[0].device_address;
      programs[chosen] = std::move(production);
      // Reuse the exact original-input/CPU-product/SwiGLU/immutable-part
      // callback. An OrdinaryOnly pair never records or launches a capture.
      for (std::uint32_t frame = 0; frame < 2; ++frame)
        (void)run_frame(frame, chosen, false, false);
      Receipt("whole_exl3_installed_selection",
              std::format("\"expected\":{},\"matched\":{},\"record_key\":{},"
                          "\"preferred_arm\":{},\"whole_bound_verified\":{},"
                          "\"bound_selected_plan_verified\":true,"
                          "\"normal_component_entry\":true,\"ordinary_only\":true,"
                          "\"capture_promoted\":false,\"frames\":2,"
                          "\"losing_fusion_veto_verified\":{},\"ranking_samples_collected\":false",
                          Quote(installed), row != nullptr, row ? RecipeKeyJson(row->key) : "null",
                          chosen, whole_bound, row && chosen == 1));
    }
    if (timing) {
      // Either ordinary-only seal requires ordinary execution for BOTH arms.
      // The capturable Q8 pair instead measures its complete captured body.
      const auto protocol = std::format(
          "{};prep={};"
          "frames=2-complete-original-ir-validated-before-events;"
          "input=frame0-original-{}-restage-before-every-invocation;arena=nan-poison;"
          "weights={};"
          "body={};body-invocations=1;"
          "outputs={};"
          "events={};"
          "fixed-warmup-pairs=4:incumbent-then-alternative;"
          "bookends=9-alternative-before-and-after:median;"
          "alternating-warmup-pairs=4:start-incumbent-then-alternate;"
          "alternating-sample-pairs=9:start-incumbent-then-alternate;"
          "observations=52;bookend-drift=10pct;no-filtering;no-auto-retry",
          q8 ? "llmp-whole-q8-captured-event-timing-v1"
             : "llmp-whole-exl3-ordinary-event-timing-v1",
          llmp::base::ToHex(key->preparation_domain), f32_input ? "f32" : "f16",
          q8 ? "native-q8_0-encoded-fixture-immutable-readable-tails"
             : "published-layer0-multipart-mcg4-immutable-readable-tails",
          q8 ? "one-complete-captured-program-for-both-arms"
             : "one-complete-ordinary-program-run-for-both-arms",
          q8 ? "complete-swiglu-cpu-nmse-5e-4-and-peer-max-1e-5-checked"
             : "both-live-products-and-complete-swiglu-checked",
          q8 ? "internal-captured-start-body-stop-on-owning-stream"
             : "external-start-run-stop-on-owning-stream");
      MeasureReadyBody(
          *key, protocol,
          q8 ? std::format("whole.q8_0.k896.n4864.rows1.f32.{}", target_pattern)
             : std::format("whole.exl3.mcg4.k896.n4864.rows1.{}.{}", f32_input ? "f32" : "f16",
                           target_pattern),
          [&](en::ComparisonPairArm arm) {
            return run_frame(0, static_cast<std::size_t>(arm), true, q8 && !ordinary);
          },
          /*captured_events=*/q8 && !ordinary, /*bookend_observations=*/9);
    }
    Finish();
    recorded = {};
    programs = {};
  }

  void Inventory() const {
    const auto& registry = backend_->registry();
    const auto& device = backend_->device();
    std::println(
        "{{\"inventory_version\":1,\"device\":{{\"model\":{},\"class\":{},"
        "\"arch\":{},\"features\":{},\"multiprocessors\":{}}},"
        "\"patterns\":[",
        Quote(device.device_model), Quote(device.device_class), device.arch, device.features,
        device.multiprocessors);
    bool first = true;
    for (const auto& pattern : pl::Patterns()) {
      if (!first) std::print(",");
      first = false;
      const auto* trait = ir::Traits(pattern.result);
      std::println(
          "{{\"name\":{},\"kind\":{},\"result\":{},\"round\":{},"
          "\"has_decomposition\":{},\"written_sources\":{},\"aliases\":[{},{}]}}",
          Quote(pattern.name), Quote(PatternKindName(pattern.kind)),
          Quote(ir::OpKeyName(pattern.result)), pattern.round, trait && trait->decomposition,
          trait ? trait->written_sources : 0,
          trait ? static_cast<int>(trait->aliased_outputs[0]) : -1,
          trait ? static_cast<int>(trait->aliased_outputs[1]) : -1);
    }
    std::print("],\"fused_traits\":[");
    first = true;
    for (const auto& trait : ir::FusedTraits()) {
      if (!first) std::print(",");
      first = false;
      std::print(
          "{{\"op\":{},\"decomposition\":{},\"written_sources\":{},"
          "\"aliases\":[{},{}],\"candidates\":[",
          Quote(ir::OpKeyName(trait.key)), trait.decomposition, trait.written_sources,
          static_cast<int>(trait.aliased_outputs[0]), static_cast<int>(trait.aliased_outputs[1]));
      bool candidate_first = true;
      for (std::size_t i = 0; i < registry.size(); ++i) {
        const auto& implementation = registry.at(i);
        if (implementation.operation != trait.key || !implementation.capability) continue;
        const auto& capability = *implementation.capability;
        if (!candidate_first) std::print(",");
        candidate_first = false;
        const bool device_range = device.arch >= capability.device.min_arch &&
                                  device.arch <= capability.device.max_arch &&
                                  (capability.device.needs & ~device.features) == 0;
        std::print(
            "{{\"name\":{},\"source\":{},\"device_range_match\":{},"
            "\"predicate_evaluated\":false,\"fusion_needs_measurement\":{},"
            "\"standalone_comparator\":{},\"group\":{},\"batch\":{},"
            "\"needs_state_span\":{},\"pitched_output\":{},\"encodings\":[",
            Quote(implementation.name), Quote(implementation.source), device_range,
            capability.fusion_needs_measurement, capability.standalone_comparator,
            Quote(capability.group), capability.batch, capability.needs_state_span,
            capability.pitched_output);
        bool rule_first = true;
        for (const auto& rule : capability.operands) {
          if (!rule_first) std::print(",");
          rule_first = false;
          std::print("{{\"operand\":{},\"accepted\":[", rule.operand);
          for (std::size_t e = 0; e < rule.encodings.size(); ++e) {
            if (e != 0) std::print(",");
            std::print("{}", Quote(llmp::format::Info(rule.encodings[e]).name));
          }
          std::print("],\"any_encoding\":{}}}", rule.encodings.empty());
        }
        std::print("]}}");
      }
      std::print("]}}");
    }
    std::print("],\"fixtures\":[");
    first = true;
    for (const auto& fixture : kFixtures) {
      if (!first) std::print(",");
      first = false;
      std::print("{{\"name\":{},\"family\":{},\"operation\":{},\"patterns\":[", Quote(fixture.name),
                 Quote(fixture.family), Quote(ir::OpKeyName(fixture.operation)));
      for (std::size_t p = 0; p < fixture.pattern_count; ++p) {
        if (p != 0) std::print(",");
        const auto* pattern = pl::FindPattern(fixture.patterns[p]);
        if (!pattern) Fail("fixture mapping names an unregistered planner pattern");
        if (pattern->result != fixture.operation)
          Fail("fixture mapping disagrees with planner pattern result");
        std::print("{}", Quote(fixture.patterns[p]));
      }
      std::print("],\"registered_device_range_candidates\":[");
      bool candidate_first = true;
      for (std::size_t i = 0; i < registry.size(); ++i) {
        const auto& implementation = registry.at(i);
        if (implementation.operation != fixture.operation || !implementation.capability) continue;
        const auto& capability = *implementation.capability;
        const bool device_range = device.arch >= capability.device.min_arch &&
                                  device.arch <= capability.device.max_arch &&
                                  (capability.device.needs & ~device.features) == 0;
        if (!device_range) continue;
        if (!candidate_first) std::print(",");
        candidate_first = false;
        std::print("{}", Quote(implementation.name));
      }
      std::print("]}}");
    }
    std::println("]}}");
  }

 private:
  std::string RecipeArmJson(const ex::RecipeArmIdentity& arm) const {
    return std::format(
        "{{\"planned_region\":{},\"physical_recipe\":{},"
        "\"backend_recipe\":{},\"backend_build\":{},"
        "\"backend_selection\":{}}}",
        Quote(llmp::base::ToHex(arm.planned_region)), Quote(llmp::base::ToHex(arm.physical_recipe)),
        Quote(llmp::base::ToHex(arm.backend_recipe)), Quote(llmp::base::ToHex(arm.backend_build)),
        Quote(llmp::base::ToHex(arm.backend_selection)));
  }

  std::string RecipeKeyJson(const ex::RecipeComparisonKey& key) const {
    const auto operation =
        key.discovery_operation.op == ir::Op::kFused
            ? std::format("fused.{}", ir::FusedKindName(key.discovery_operation.fused))
            : std::string(ir::OpName(key.discovery_operation.op));
    return std::format(
        "{{\"version\":{},\"region_canonicalizer\":{},\"discovery_operation\":{},"
        "\"mode\":{},\"subject\":{},\"boundary\":{},\"policy\":{},\"device\":{},"
        "\"selection_environment\":{},\"preparation_domain\":{},\"incumbent\":{},"
        "\"alternative\":{}}}",
        key.version, key.region_canonicalizer, Quote(operation),
        Quote(key.mode == ex::RecipeRunMode::kFull ? "full" : "warm"),
        Quote(llmp::base::ToHex(key.subject)), Quote(llmp::base::ToHex(key.boundary)),
        Quote(llmp::base::ToHex(key.policy)), Quote(llmp::base::ToHex(key.device)),
        Quote(llmp::base::ToHex(key.selection_environment)),
        Quote(llmp::base::ToHex(key.preparation_domain)), RecipeArmJson(key.incumbent),
        RecipeArmJson(key.alternative));
  }

  std::uint64_t TimeReadyBody(en::ComparisonPair& pair, en::ComparisonPairArm arm,
                              std::span<const float> host_input, std::span<const float> host_weight,
                              std::uint64_t weight_address, std::uint64_t weight_bytes,
                              ir::TensorId input, ir::TensorId prefix) {
    const auto input_slots = pair.input_slots(arm);
    if (input_slots.size() != 1 || input_slots.front().tensor != input ||
        input_slots.front().kind != kn::InputKind::kStepBytes ||
        input_slots.front().bytes != host_input.size_bytes())
      Fail("ready-norm timed arm has an invalid original input slot");
    Check(cudaMemcpyAsync(reinterpret_cast<void*>(input_slots.front().device_address),
                          host_input.data(), host_input.size_bytes(), cudaMemcpyHostToDevice,
                          CudaStream()));
    Finish();
    auto prepared = pair.PrepareFrame(arm, *launch_, native_);
    if (!prepared) Fail("ready-norm timed prefix preparation failed");
    Finish();

    std::vector<std::byte> prefix_before;
    for (const auto& port : pair.value_ports(arm)) {
      if (port.kind != kn::ComparisonRegionValueKind::kInput || port.source_tensor != prefix)
        continue;
      if (!prefix_before.empty() || port.bytes == 0)
        Fail("ready-norm timed prefix operand is malformed");
      prefix_before.resize(static_cast<std::size_t>(port.bytes));
      Check(cudaMemcpy(prefix_before.data(), reinterpret_cast<const void*>(port.address),
                       prefix_before.size(), cudaMemcpyDeviceToHost));
    }
    if (prefix_before.empty()) Fail("ready-norm timed prefix operand is missing");
    std::vector<std::byte> weight_before(static_cast<std::size_t>(weight_bytes));
    Check(cudaMemcpy(weight_before.data(), reinterpret_cast<const void*>(weight_address),
                     weight_before.size(), cudaMemcpyDeviceToHost));
    if (weight_before.size() != std::span(host_weight).size_bytes() ||
        !std::ranges::equal(weight_before, std::as_bytes(std::span(host_weight))))
      Fail("ready-norm timed immutable weight differs from its source bytes");

    Check(cudaEventRecord(start_, CudaStream()));
    auto replayed = pair.ReplayCapturedRegion(arm, *launch_, native_);
    if (!replayed) Fail("ready-norm timed body replay failed");
    Check(cudaEventRecord(stop_, CudaStream()));
    Check(cudaEventSynchronize(stop_));
    float milliseconds = 0;
    Check(cudaEventElapsedTime(&milliseconds, start_, stop_));
    if (!std::isfinite(milliseconds) || milliseconds <= 0)
      Fail("ready-norm body timing returned an invalid interval");
    Finish();

    std::vector<std::byte> prefix_after(prefix_before.size());
    bool found_prefix = false;
    for (const auto& port : pair.value_ports(arm)) {
      if (port.kind != kn::ComparisonRegionValueKind::kInput || port.source_tensor != prefix)
        continue;
      if (found_prefix || port.bytes != prefix_after.size())
        Fail("ready-norm timed prefix operand changed its descriptor");
      Check(cudaMemcpy(prefix_after.data(), reinterpret_cast<const void*>(port.address),
                       prefix_after.size(), cudaMemcpyDeviceToHost));
      found_prefix = true;
    }
    if (!found_prefix || prefix_after != prefix_before)
      Fail("ready-norm repeated body changed its prepared prefix operand");
    std::vector<std::byte> weight_after(weight_before.size());
    Check(cudaMemcpy(weight_after.data(), reinterpret_cast<const void*>(weight_address),
                     weight_after.size(), cudaMemcpyDeviceToHost));
    if (weight_after != weight_before)
      Fail("ready-norm repeated body modified immutable weight storage");
    const auto inventory = pair.captured_region_inventory(arm, *launch_);
    if (!inventory || inventory->repetitions != kReplays)
      Fail("ready-norm timed body recording changed its authenticated inventory");
    const auto ns_per_body =
        static_cast<double>(milliseconds) * 1'000'000.0 / inventory->repetitions;
    if (!std::isfinite(ns_per_body) || ns_per_body < 0.5 ||
        ns_per_body >= static_cast<double>(std::numeric_limits<long long>::max()))
      Fail("ready-norm body interval cannot be represented as positive nanoseconds");
    const auto rounded_ns = std::llround(ns_per_body);
    if (rounded_ns <= 0) Fail("ready-norm body interval rounded to zero nanoseconds");
    return static_cast<std::uint64_t>(rounded_ns);
  }

  void MeasureReadyNormBody(en::ComparisonPair& pair, const ex::RecipeComparisonKey& key,
                            std::uint64_t width, std::uint64_t rows, std::span<const float> input,
                            std::span<const float> weight, std::uint64_t weight_address,
                            std::uint64_t weight_bytes, ir::TensorId input_tensor,
                            ir::TensorId prefix) {
    const auto protocol_text = std::format(
        "llmp-ready-body-timing-v1;prep={};frames=2-original-ir-validated-before-events;"
        "input=restage-original-before-each-block;prefix=prepare-once-before-events;"
        "body=captured-complete-local-region;body-replays={};"
        "fixed-warmup-pairs={}:incumbent-then-alternative;"
        "bookends=alternative-before-and-after;"
        "alternating-warmup-pairs={}:start-incumbent-then-alternate;"
        "alternating-sample-pairs={}:start-incumbent-then-alternate;"
        "events=one-start-stop-per-captured-block;bookend-drift=10pct",
        llmp::base::ToHex(key.preparation_domain), kReplays, kWarmups, kWarmups, kSamples);
    const auto shape_class = std::format("ready.rms_norm_mul.f32.width{}.rows{}", width, rows);
    MeasureReadyBody(key, protocol_text, shape_class, [&](en::ComparisonPairArm arm) {
      return TimeReadyBody(pair, arm, input, weight, weight_address, weight_bytes, input_tensor,
                           prefix);
    });
  }

  template <typename TimeBody>
  void MeasureReadyBody(const ex::RecipeComparisonKey& key, std::string_view protocol_text,
                        std::string_view shape_class, TimeBody&& time_body,
                        bool captured_events = false, std::uint32_t bookend_observations = 1) {
    if (!captured_events && !start_) {
      Check(cudaEventCreate(&start_));
      Check(cudaEventCreate(&stop_));
    }
    std::array<std::vector<std::uint64_t>, 2> samples;
    std::string raw_receipt;
    const auto time = [&](std::size_t arm, std::string_view stage, std::int32_t pair_index,
                          std::uint32_t order) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      phase = stage;
      const auto ns = time_body(which);
      const auto event =
          std::format("{{\"stage\":{},\"pair\":{},\"order\":{},\"arm\":{},\"ns_per_body\":{}}}",
                      Quote(stage), pair_index, order, arm, ns);
      if (!raw_receipt.empty()) raw_receipt += ',';
      raw_receipt += event;
      phase = stage;
      Receipt("ready_body_sample", std::format("\"stage\":{},\"pair\":{},\"order\":{},"
                                               "\"arm\":{},\"ns_per_body\":{}",
                                               Quote(stage), pair_index, order, arm, ns));
      return ns;
    };
    for (std::int32_t pair_index = 0; pair_index < kWarmups; ++pair_index) {
      for (std::size_t arm = 0; arm < 2; ++arm)
        (void)time(arm, "fixed_warmup", pair_index, static_cast<std::uint32_t>(arm));
    }
    const auto median = [](std::vector<std::uint64_t> values) {
      std::ranges::sort(values);
      return values.at(values.size() / 2);
    };
    phase = "ready_body_bookend_before";
    std::vector<std::uint64_t> before_observations;
    for (std::uint32_t order = 0; order < bookend_observations; ++order)
      before_observations.push_back(time(1, "bookend_before", -1, order));
    const auto before = median(before_observations);
    for (std::int32_t pair_index = -kWarmups; pair_index < kSamples; ++pair_index) {
      const auto stage =
          pair_index < 0 ? std::string_view("paired_warmup") : std::string_view("timing");
      const auto first_arm = static_cast<std::size_t>((pair_index + kWarmups) % 2);
      for (std::uint32_t order = 0; order < 2; ++order) {
        const auto arm = (first_arm + order) % 2;
        const auto ns = time(arm, stage, pair_index, order);
        if (pair_index >= 0) samples[arm].push_back(ns);
      }
    }
    phase = "ready_body_bookend_after";
    std::vector<std::uint64_t> after_observations;
    for (std::uint32_t order = 0; order < bookend_observations; ++order)
      after_observations.push_back(time(1, "bookend_after", -1, order));
    const auto after = median(after_observations);
    if (static_cast<double>(std::max(before, after)) >
        1.1 * static_cast<double>(std::min(before, after)))
      Fail("ready-norm body timing bookends drifted by more than 10%");
    const auto incumbent_ns = median(samples[0]);
    const auto alternative_ns = median(samples[1]);
    const auto preference = ex::ClassifyRecipeLatency(incumbent_ns, alternative_ns);
    const auto protocol = llmp::base::Sha256{}.Update(protocol_text).Finish();
    const auto raw_receipt_json = std::format("[{}]", raw_receipt);
    const auto receipt = llmp::base::Sha256{}.Update(raw_receipt_json).Finish();
    const auto preference_name = preference == ex::RecipePreference::kAlternative ? "alternative"
                                 : preference == ex::RecipePreference::kIncumbent ? "incumbent"
                                                                                  : "neutral";
    std::println("{{\"ready_body_receipt\":{{\"sha256\":{},\"events\":{}}}}}",
                 Quote(llmp::base::ToHex(receipt)), raw_receipt_json);
    std::println(
        "{{\"recipe_comparison_candidate\":{{\"key\":{},\"incumbent_ns\":{},"
        "\"alternative_ns\":{},\"samples\":{},\"protocol\":{},\"receipt\":{},"
        "\"provenance\":\"UNREVIEWED\",\"shape_class\":{}}}}}",
        RecipeKeyJson(key), incumbent_ns, alternative_ns, kSamples,
        Quote(llmp::base::ToHex(protocol)), Quote(llmp::base::ToHex(receipt)), Quote(shape_class));
    std::println("{{\"recipe_comparison_decision\":{{\"value\":{}}}}}", Quote(preference_name));
    std::println("{{\"ready_body_protocol\":{{\"sha256\":{},\"text\":{}}}}}",
                 Quote(llmp::base::ToHex(protocol)), Quote(protocol_text));
    std::fflush(stdout);
  }

  void RunReadyNormPair(std::uint64_t width, std::uint64_t rows, bool timing,
                        std::string_view installed = {}) {
    if (rows > std::numeric_limits<std::uint32_t>::max())
      Fail("ready-norm row count exceeds the planner segment range");
    ir::Builder builder;
    builder.SetComponent(0);
    builder.SetOwner(0);
    const auto x =
        builder.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})), "ready-norm-input");
    const ir::WeightRef weight_ref{.stage = 0, .component = 0, .role = 1, .layer = 0};
    const auto w = builder.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({width})), weight_ref,
                                  "ready-norm.weight");
    builder.SetJoined();
    const auto prefix = builder.RmsNorm(x, 1e-6F);
    const auto normalized = builder.RmsNorm(prefix, 1e-6F);
    const auto scaled = builder.Mul(normalized, w);
    builder.SetOwner(0);
    const auto output = builder.Concat(scaled, prefix, 0);
    builder.Output(output);
    auto source_graph = Must(std::move(builder).Finish());

    const auto norm_node = source_graph.producer(normalized);
    const auto mul_node = source_graph.producer(scaled);
    const auto x_bytes = ir::StorageBytes(source_graph.tensor(x));
    const auto w_parts = llmp::format::PartsOf(source_graph.tensor(w).format);
    if (!x_bytes || !w_parts || w_parts->size() != 1)
      Fail("ready-norm source has no bounded input or weight storage");
    const auto weight_part = w_parts->parts().front();

    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    auto funded_graph = source_graph.Clone(funder);
    if (!funded_graph) Fail("ready-norm source graph admission was refused");
    const auto source_identity = funded_graph->identity();
    co::StepGraph source{.graph = std::move(*funded_graph),
                         .inputs = {{.tensor = x,
                                     .kind = co::InputKind::kStream,
                                     .component = 0,
                                     .what = "ready-norm-input"}},
                         .weights = {{.tensor = w,
                                      .ref = weight_ref,
                                      .parts = {{"ready-norm.weight", weight_part.role}}}},
                         .state = {},
                         .logits = {},
                         .greedy = {},
                         .features = {},
                         .saves = {},
                         .owner_products = {}};
    pl::StepShape shape{.stage = 0,
                        .segments = {{.slot = 0,
                                      .rows = static_cast<std::uint32_t>(rows),
                                      .outputs = static_cast<std::uint32_t>(rows)}}};
    std::vector<co::ComparisonExternalBinding> external{{.original_root = x,
                                                         .allocation_class = 1,
                                                         .allocation_bytes = *x_bytes,
                                                         .bytes = *x_bytes,
                                                         .alignment = 256},
                                                        {.original_root = w,
                                                         .part = weight_part.role,
                                                         .allocation_class = 2,
                                                         .allocation_bytes = weight_part.readable,
                                                         .bytes = weight_part.bytes,
                                                         .alignment = 16}};

    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    const pl::Policy policy{.active_stage = 0};
    const auto* pattern = pl::FindPattern("rms_norm_mul");
    if (!pattern) Fail("ready-norm planner pattern is unavailable");
    const auto activation_base = Allocate(kActivations);
    const auto weight_address = Allocate(weight_part.readable);
    const auto weight_alignment = std::uint64_t{16};
    kn::Addresses addresses{
        .weight =
            [weight_ref, weight_address](const ir::WeightRef& ref, llmp::format::PartRole part) {
              return ref == weight_ref && part == llmp::format::PartRole::kData ? weight_address
                                                                                : 0;
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation_base,
        .activation_bytes = kActivations,
        .activation_alignment = 256,
        .weight_alignment =
            [weight_ref](const ir::WeightRef& ref, llmp::format::PartRole part) {
              return ref == weight_ref && part == llmp::format::PartRole::kData ? 16 : 0;
            }};
    addresses.weight_range = [&](const ir::WeightRef& ref,
                                 llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
      if (ref != weight_ref || part != weight_part.role) return {};
      return kn::BoundedAddress{weight_address, weight_part.readable, weight_alignment};
    };
    const auto region_external = external;
    auto made =
        en::ComparisonPair::Create(funder, std::move(source), std::move(shape), std::move(external),
                                   0, 0, en::ComparisonInputDomain::kComponentPorts);
    if (!made) Fail(made.error().detail);
    auto pair = std::move(*made);
    pl::ComparisonSiteSelector selector{.source_graph_identity = source_identity,
                                        .kind = pl::ComparisonSiteKind::kRewrite,
                                        .pattern = "rms_norm_mul",
                                        .version = pattern->version,
                                        .member_origins = {{norm_node}, {mul_node}},
                                        .member_owner_segments = {},
                                        .boundary_ports = {}};
    const auto options =
        Must(en::PrepareComponentPlanOptions(co::GraphIO(pair->source()), table, addresses,
                                             {.capturable = true, .within = table.within}));
    auto planned = pair->Plan(selector, *table.registry, table.device, policy, options);
    if (!planned) Fail(planned.error().detail);
    const auto* incumbent_plan = pair->plan(en::ComparisonPairArm::kIncumbent);
    const auto* alternative_plan = pair->plan(en::ComparisonPairArm::kAlternative);
    if (!incumbent_plan || !alternative_plan ||
        std::ranges::none_of(
            incumbent_plan->plan.applied,
            [](const pl::Applied& applied) { return applied.pattern == "rms_norm_mul"; }) ||
        std::ranges::any_of(alternative_plan->plan.applied, [](const pl::Applied& applied) {
          return applied.pattern == "rms_norm_mul";
        }))
      Fail("ready-norm pair did not retain the actual fused and own-decomposition plans");

    std::array<std::vector<kn::ComparisonBindingRange>, 2> binding_storage;
    for (std::size_t arm = 0; arm < binding_storage.size(); ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      const auto* plan = pair->plan(which);
      if (plan == nullptr || x.v >= plan->plan.tensors.size())
        Fail("ready-norm plan omitted its original activation mapping");
      const auto mapped = plan->plan.tensors[x.v];
      if (!mapped.valid()) Fail("ready-norm plan removed its staged input");
      const auto root = plan->plan.graph.root(mapped);
      const auto placement = std::ranges::find_if(
          plan->plan.placement.offsets,
          [&](const auto& item) { return plan->plan.graph.root(item.first) == root; });
      if (placement == plan->plan.placement.offsets.end() || placement->second > kActivations ||
          *x_bytes > kActivations - placement->second)
        Fail("ready-norm input placement exceeds its admitted activation storage");
      binding_storage[arm].push_back({.original_root = x,
                                      .allocation_class = 1,
                                      .allocation_bytes = *x_bytes,
                                      .bytes = *x_bytes,
                                      .alignment = 256,
                                      .base = activation_base + placement->second});
      binding_storage[arm].push_back({.original_root = w,
                                      .part = weight_part.role,
                                      .allocation_class = 2,
                                      .allocation_bytes = weight_part.readable,
                                      .bytes = weight_part.bytes,
                                      .alignment = weight_alignment,
                                      .base = weight_address});
    }
    const std::array<std::span<const kn::ComparisonBindingRange>, 2> binding_views{
        binding_storage[0], binding_storage[1]};
    auto installed_bindings = pair->InstallBindings(binding_views);
    if (!installed_bindings) Fail(installed_bindings.error().detail);
    const co::ComparisonBoundaryView region_view{
        .io = co::GraphIO(pair->source()), .active_stage = 0, .external = region_external};
    std::vector<std::uint32_t> region_offsets{0};
    std::vector<ir::NodeId> region_members;
    for (const auto& group : pair->inventory()->site.accepted_member_origins) {
      region_members.insert(region_members.end(), group.begin(), group.end());
      region_offsets.push_back(static_cast<std::uint32_t>(region_members.size()));
    }
    auto installed_region = Must(
        co::MakeComparisonRegionBoundary({region_view, region_offsets, region_members}, funder));
    std::array<kn::ComparisonRegionRecipeSeal, 2> ready_descriptors;
    std::array<const kn::StepProgram*, 2> native_programs{};
    std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
    for (std::size_t arm = 0; arm < programs.size(); ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      auto lower = pair->LoweringInput(which, addresses);
      if (!lower) Fail(lower.error().detail);
      auto lowered = table.lower(*lower);
      if (!lowered) Fail(lowered.error());
      if ((*lowered)->scratch_bytes() > kActivations || (*lowered)->pool_bytes() > kPool)
        Fail("ready-norm program exceeds Screen storage");
      const auto certificate = (*lowered)->ProjectComparisonRegionCertificate(
          installed_region, kn::ComparisonCertificatePhase::kDescriptor);
      if (certificate.status != kn::ComparisonRegionStatus::kClosed)
        Fail("ready-norm original pair lacks its actual descriptor applicability certificate");
      ready_descriptors[arm] = certificate.seal;
      native_programs[arm] = lowered->get();
      programs[arm] = std::move(*lowered);
    }
    auto installed_programs =
        pair->InstallDescriptorPrograms(std::move(programs[0]), std::move(programs[1]));
    if (!installed_programs) Fail(installed_programs.error().detail);
    phase = "ready_norm_descriptor";
    auto described = en::DescribeRecipeComparison(
        *pair, *table.registry, en::ComparisonRecipePreparation::kStatelessReadyBoundaryOperands);
    if (!described || !*described) Fail("ready-norm pair did not produce a closed recipe key");
    const auto key = **described;
    const auto* measured = table.registry->FindRecipeComparison(key);
    const auto preference =
        measured ? ex::ClassifyRecipeLatency(measured->incumbent_ns, measured->alternative_ns)
                 : ex::RecipePreference::kNeutral;
    const auto preference_name =
        !measured                                          ? std::string_view("unmeasured")
        : preference == ex::RecipePreference::kAlternative ? std::string_view("alternative")
        : preference == ex::RecipePreference::kIncumbent   ? std::string_view("incumbent")
                                                           : std::string_view("neutral");
    const auto shape_class = measured ? Quote(measured->shape_class) : std::string("null");
    Receipt(
        "ready_norm_rank_lookup",
        std::format("\"recipe_key_digest\":{},\"registered_rows\":{},"
                    "\"matched\":{},\"shape_class\":{},\"preference\":{}",
                    Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key))),
                    table.registry->ranks().recipe_comparisons.size(),
                    measured == nullptr ? "false" : "true", shape_class, Quote(preference_name)));
    auto bound = pair->BindPrograms(*launch_, *launch_);
    if (!bound) Fail(bound.error().detail);
    auto closed = en::CloseComparisonPair(std::move(pair));
    if (!closed) Fail(closed.error().detail);
    pair = std::move(*closed);
    if (!pair->closed()) Fail("ready-norm pair did not close after descriptor binding");

    std::array<std::size_t, 2> captured_nodes{};
    for (std::size_t arm_index = 0; arm_index < 2; ++arm_index) {
      const auto arm =
          arm_index == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      auto captured = pair->CaptureRegion(arm, *launch_, native_, kReplays);
      if (!captured) Fail("ready-norm body capture failed");
      const auto inventory = pair->captured_region_inventory(arm, *launch_);
      const auto expected_nodes = kReplays * (arm_index == 0 ? 1U : 2U);
      if (!inventory || inventory->repetitions != kReplays ||
          inventory->recorded_nodes != expected_nodes)
        Fail("ready-norm repeated body inventory differs from its fused/decomposed schedule");
      captured_nodes[arm_index] = inventory->recorded_nodes;
    }

    std::vector<float> host_weight(width);
    for (std::size_t i = 0; i < host_weight.size(); ++i)
      host_weight[i] = 0.5F + static_cast<float>(i % 7) * 0.0625F;
    if (weight_part.bytes != host_weight.size() * sizeof(float))
      Fail("ready-norm CPU reference weight extent differs from the actual weight part");
    Check(cudaMemcpyAsync(reinterpret_cast<void*>(weight_address), host_weight.data(),
                          static_cast<std::size_t>(weight_part.bytes), cudaMemcpyHostToDevice,
                          CudaStream()));
    Finish();

    auto reference = ref::Executor::Create(source_graph);
    if (!reference) Fail(reference.error().detail);
    std::array<double, 2> arm_max_error{};
    std::array<std::vector<float>, 2> first_frame_outputs;
    std::vector<float> measurement_input;
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      std::vector<float> host_input(width * rows);
      for (std::size_t i = 0; i < host_input.size(); ++i) {
        const float centered = static_cast<float>(static_cast<int>((i * 17) % 101) - 50);
        host_input[i] = centered * (frame == 0 ? 0.03125F : 0.015625F) +
                        (frame == 0 ? 0.125F : static_cast<float>(i % 11) * 0.0078125F);
      }
      if (frame == 1) measurement_input = host_input;
      auto ref_input = reference->BindValues(x, host_input);
      if (!ref_input) Fail(ref_input.error().detail);
      auto ref_weight = reference->BindValues(w, host_weight);
      if (!ref_weight) Fail(ref_weight.error().detail);
      auto ref_run = reference->Run();
      if (!ref_run) Fail(ref_run.error().detail);
      auto expected = reference->Values(scaled);
      if (!expected) Fail(expected.error().detail);
      for (const auto arm :
           {en::ComparisonPairArm::kIncumbent, en::ComparisonPairArm::kAlternative}) {
        const auto inputs = pair->input_slots(arm);
        if (inputs.size() != 1 || inputs.front().tensor != x)
          Fail("ready-norm pair has an unexpected staged input inventory");
        const auto& slot = inputs.front();
        if (slot.kind != kn::InputKind::kStepBytes ||
            slot.bytes != host_input.size() * sizeof(float))
          Fail("ready-norm source input is not the expected typed component port");
        bool poisoned_prefix = false;
        bool poisoned_output = false;
        for (const auto& port : pair->value_ports(arm)) {
          if (port.kind == kn::ComparisonRegionValueKind::kInput && port.source_tensor == prefix) {
            if (port.role != ir::Role::kActivation)
              Fail("ready-norm pure-prefix input is not an activation value");
            Check(cudaMemsetAsync(reinterpret_cast<void*>(port.address), 0xA5,
                                  static_cast<std::size_t>(port.bytes), CudaStream()));
            poisoned_prefix = true;
          }
          if (port.kind == kn::ComparisonRegionValueKind::kOutput && port.source_tensor == scaled) {
            if (port.role != ir::Role::kActivation)
              Fail("ready-norm body output is not an activation value");
            Check(cudaMemsetAsync(reinterpret_cast<void*>(port.address), 0xA5,
                                  static_cast<std::size_t>(port.bytes), CudaStream()));
            poisoned_output = true;
          }
        }
        if (!poisoned_prefix || !poisoned_output)
          Fail("ready-norm boundary omitted its pure-prefix input or body output");
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), host_input.data(),
                              static_cast<std::size_t>(slot.bytes), cudaMemcpyHostToDevice,
                              CudaStream()));
        Finish();
        auto prepared = pair->PrepareFrame(arm, *launch_, native_);
        if (!prepared) Fail("ready-norm pure-prefix preparation failed");
        Finish();
        std::vector<std::byte> prefix_before;
        for (const auto& port : pair->value_ports(arm)) {
          if (port.kind != kn::ComparisonRegionValueKind::kInput || port.source_tensor != prefix)
            continue;
          if (!prefix_before.empty() || port.bytes == 0)
            Fail("ready-norm pure-prefix value port is malformed");
          prefix_before.resize(static_cast<std::size_t>(port.bytes));
          Check(cudaMemcpy(prefix_before.data(), reinterpret_cast<const void*>(port.address),
                           prefix_before.size(), cudaMemcpyDeviceToHost));
        }
        if (prefix_before.empty()) Fail("ready-norm prefix value was not snapshotted");
        auto ran = pair->ReplayCapturedRegion(arm, *launch_, native_);
        if (!ran) Fail("ready-norm prefix/body replay failed");
        Finish();
        for (const auto& port : pair->value_ports(arm)) {
          if (port.kind == kn::ComparisonRegionValueKind::kInput && port.source_tensor == prefix) {
            std::vector<std::byte> prefix_after(prefix_before.size());
            Check(cudaMemcpy(prefix_after.data(), reinterpret_cast<const void*>(port.address),
                             prefix_after.size(), cudaMemcpyDeviceToHost));
            if (prefix_after != prefix_before)
              Fail("ready-norm repeated body changed its prepared prefix operand");
          }
        }
        bool found = false;
        for (const auto& port : pair->value_ports(arm)) {
          if (port.kind != kn::ComparisonRegionValueKind::kOutput || port.source_tensor != scaled)
            continue;
          if (found || port.bytes != expected->size() * sizeof(float))
            Fail("ready-norm body output port is malformed");
          found = true;
          std::vector<float> actual(expected->size());
          Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(port.address), port.bytes,
                           cudaMemcpyDeviceToHost));
          for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite((*expected)[i]))
              Fail("ready-norm CPU/GPU comparison produced a nonfinite value");
            const double error = std::abs(static_cast<double>(actual[i]) - (*expected)[i]);
            arm_max_error[arm == en::ComparisonPairArm::kIncumbent ? 0 : 1] =
                std::max(arm_max_error[arm == en::ComparisonPairArm::kIncumbent ? 0 : 1], error);
            if (error > 1e-5 * (1.0 + std::abs((*expected)[i])))
              Fail("ready-norm arm exceeded the existing normalized RMS reference bound");
          }
          if (frame == 0) {
            first_frame_outputs[arm == en::ComparisonPairArm::kIncumbent ? 0 : 1] = actual;
          } else {
            const auto index = arm == en::ComparisonPairArm::kIncumbent ? 0U : 1U;
            const auto& prior = first_frame_outputs[index];
            if (prior.size() != actual.size()) Fail("ready-norm frame output size changed");
            double changed = 0;
            for (std::size_t i = 0; i < actual.size(); ++i)
              changed = std::max(changed, std::abs(static_cast<double>(actual[i]) - prior[i]));
            if (changed <= 1e-4)
              Fail("ready-norm changed-input replay produced no materially changed output");
          }
        }
        if (!found) Fail("ready-norm closed region omitted its checked output");
        std::vector<float> weight_after(width);
        Check(cudaMemcpy(weight_after.data(), reinterpret_cast<const void*>(weight_address),
                         static_cast<std::size_t>(weight_part.bytes), cudaMemcpyDeviceToHost));
        if (weight_after != host_weight)
          Fail("ready-norm execution modified immutable weight storage");
      }
    }
    Finish();
    Receipt(
        "ready_norm_validated",
        std::format("\"preparation_domain\":{},\"recipe_key\":{},"
                    "\"incumbent_identity\":{},\"alternative_identity\":{},"
                    "\"recorded_nodes\":[{},{}],\"recording_repetitions\":{},"
                    "\"frames\":2,\"max_error\":[{},{}],\"timing_performed\":false",
                    Quote(llmp::base::ToHex(key.preparation_domain)),
                    Quote(llmp::base::ToHex(llmp::execution::RecipeComparisonKeyDigest(key))),
                    Quote(llmp::base::ToHex(key.incumbent.physical_recipe)),
                    Quote(llmp::base::ToHex(key.alternative.physical_recipe)), captured_nodes[0],
                    captured_nodes[1], kReplays, arm_max_error[0], arm_max_error[1]));
    phase = "ready_norm_validated";
    if (!installed.empty()) {
      if ((installed == "hit") != (measured != nullptr) ||
          (measured && measured->kind != ex::RecipeComparisonKind::kLatency))
        Fail("installed local row differs from the requested hit or miss");
      const std::size_t chosen =
          measured && preference == ex::RecipePreference::kAlternative ? 1 : 0;
      const auto expected_plan = native_programs[chosen]->bound()->identity();
      const auto& expected_graph =
          chosen == 0 ? incumbent_plan->plan.graph : alternative_plan->plan.graph;
      struct Observed {
        const kn::StepProgram* native;
        std::uint64_t identity;
        llmp::base::Sha256Digest graph;
        kn::ComparisonRegionDescriptorResult local;
      };
      std::vector<Observed> observed;
      const kn::ProgramValidator observe =
          [&](const kn::LowerInput& lower,
              const kn::StepProgram& program) -> std::expected<void, std::string> {
        if (&lower.io.graph != &pair->source().graph || lower.token)
          return std::unexpected(
              "installed local comparison changed its real ComponentPorts source");
        auto local = program.ProjectComparisonRegionCertificate(
            installed_region, kn::ComparisonCertificatePhase::kDescriptor);
        // Keep no borrowed expansion spans across the next backend query.
        local.required_members = {};
        const auto* native = &program.comparison_program();
        const Observed current{native, lower.plan.graph.identity(), lower.plan.graph.Digest(),
                               local};
        const auto previous = std::ranges::find(observed, native, &Observed::native);
        if (previous == observed.end())
          observed.push_back(current);
        else
          *previous = current;
        return {};
      };
      auto production =
          Must(en::BuildComponentProgram(co::GraphIO(pair->source()), table, policy, addresses,
                                         options, std::nullopt, observe, owned.owner));
      const auto identity = production->comparison_plan_identity();
      if (identity && (identity->source_graph_identity != pair->source().graph.identity() ||
                       std::ranges::count(observed, identity->planned_graph_identity,
                                          &Observed::identity) != 1))
        Fail("normal local selection has no unique retained source and plan observation");
      if (!identity && !measured &&
          std::ranges::count(observed, &production->comparison_program(), &Observed::native) != 1)
        Fail("normal local miss has no unique retained native observation");
      const auto at =
          identity
              ? std::ranges::find(observed, identity->planned_graph_identity, &Observed::identity)
          : !measured
              ? std::ranges::find(observed, &production->comparison_program(), &Observed::native)
              : observed.end();
      if (at == observed.end() || at->graph != expected_graph.Digest())
        Fail("normal local adoption changed its exact complete planned graph");
      if (at->local.status == kn::ComparisonRegionStatus::kClosed) {
        auto actual = ready_descriptors;
        actual[chosen] = at->local.seal;
        const auto actual_key = en::RecipeComparisonKeyForReadyCertificates(
            installed_region, pair->inventory()->site.registered_result, pair->shape().phase,
            table.registry->SelectionDigest(), actual[0], actual[1]);
        if (!actual_key || *actual_key != key)
          Fail("normal local selection differs from its authentic installed Ready key");
      } else if (measured) {
        Fail("installed Ready hit lacks its chosen descriptor certificate");
      }
      Must(production->Bind(*launch_));
      // The wrapper deliberately denies external local projections. This
      // borrowed native observation is joined to its retained plan identity;
      // Bind and execution still go through the normal production owner.
      bool local_bound = false;
      if (at->local.status == kn::ComparisonRegionStatus::kClosed) {
        const auto bound_certificate = at->native->ProjectComparisonRegionCertificate(
            installed_region, kn::ComparisonCertificatePhase::kBound);
        const auto persistent = [](const kn::ComparisonRegionRecipeSeal& seal) {
          return std::tie(seal.version, seal.missing, seal.subject, seal.boundary,
                          seal.planned_region, seal.physical_recipe, seal.policy, seal.registry,
                          seal.backend_recipe, seal.backend_build, seal.backend_selection,
                          seal.backend_device);
        };
        local_bound = bound_certificate.status == kn::ComparisonRegionStatus::kClosed &&
                      at->local.seal.state == kn::ComparisonProofState::kDescriptorVerified &&
                      bound_certificate.seal.state == kn::ComparisonProofState::kBoundVerified &&
                      bound_certificate.seal.missing == kn::ComparisonMissing::kNone &&
                      persistent(at->local.seal) == persistent(bound_certificate.seal);
        if (!local_bound) Fail("normal local chosen certificate changed from Descriptor to Bound");
      } else if (measured) {
        Fail("installed Ready hit lost its retained local certificate");
      }
      if (!production->bound() || production->bound()->identity() != expected_plan)
        Fail("normal local selection changed its complete ordered bound implementation plan");
      for (std::uint32_t frame = 0; frame < 2; ++frame) {
        std::vector<float> input(width * rows);
        for (std::size_t i = 0; i < input.size(); ++i) {
          const float centered = static_cast<float>(static_cast<int>((i * 17) % 101) - 50);
          input[i] = centered * (frame == 0 ? 0.03125F : 0.015625F) +
                     (frame == 0 ? 0.125F : static_cast<float>(i % 11) * 0.0078125F);
        }
        Must(reference->BindValues(x, input));
        Must(reference->BindValues(w, host_weight));
        Must(reference->Run());
        const auto expected = Must(reference->Values(output));
        if (production->inputs().size() != 1 || production->inputs()[0].tensor != x ||
            production->inputs()[0].kind != kn::InputKind::kStepBytes ||
            production->inputs()[0].bytes != input.size() * sizeof(float))
          Fail("installed local program changed the actual original input port");
        Check(cudaMemsetAsync(reinterpret_cast<void*>(activation_base), 0xff, kActivations,
                              CudaStream()));
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(production->inputs()[0].device_address),
                              input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice,
                              CudaStream()));
        Must(production->Run(*launch_, native_));
        Finish();
        if (production->outputs().size() != 1 || production->outputs()[0].tensor != output ||
            production->outputs()[0].bytes != expected.size() * sizeof(float))
          Fail("installed local program lost the complete original Concat output");
        std::vector<float> actual(expected.size()), weight_after(width);
        Check(cudaMemcpy(actual.data(),
                         reinterpret_cast<const void*>(production->outputs()[0].device_address),
                         actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < actual.size(); ++i)
          if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
              std::abs(static_cast<double>(actual[i]) - expected[i]) >
                  1e-5 * (1 + std::abs(expected[i])))
            Fail("installed local full original output exceeds its unchanged CPU bound");
        Check(cudaMemcpy(weight_after.data(), reinterpret_cast<const void*>(weight_address),
                         weight_part.bytes, cudaMemcpyDeviceToHost));
        if (weight_after != host_weight) Fail("installed local program modified immutable gamma");
      }
      Receipt("ready_norm_installed_selection",
              std::format("\"expected\":{},\"matched\":{},\"record_key\":{},"
                          "\"preferred_arm\":{},\"normal_component_entry\":true,\"frames\":2,"
                          "\"full_original_output\":true,\"local_bound_verified\":{},"
                          "\"bound_plan_verified\":true,"
                          "\"losing_fusion_veto_verified\":{},\"ranking_samples_collected\":false",
                          Quote(installed), measured != nullptr,
                          measured ? RecipeKeyJson(measured->key) : "null", chosen, local_bound,
                          measured && chosen == 1));
    }
    if (timing)
      MeasureReadyNormBody(*pair, **described, width, rows, std::span(measurement_input),
                           std::span(host_weight), weight_address, weight_part.bytes, x, prefix);
  }

  void RunReadyArithmeticPair(ReadyArithmetic fixture, std::uint64_t width, std::uint64_t rows,
                              std::uint32_t split = 0, bool silu = false, bool timing = false,
                              std::string_view installed_selection = {}) {
    // main() bounds original matrix/row backing before Screen construction.
    const bool gdn = fixture == ReadyArithmetic::kGdnGates;
    const bool hc = fixture == ReadyArithmetic::kHcPostNorm;
    const bool hc_combine = fixture == ReadyArithmetic::kHcCombine;
    const bool hc_family = hc || hc_combine;
    const bool spatial = fixture == ReadyArithmetic::kSpatialNorm;
    const bool whole_control = spatial || hc_family || gdn;
    const bool norm_add = fixture == ReadyArithmetic::kNormAdd;
    const bool matmul_glu = fixture == ReadyArithmetic::kMatMulGlu;
    const bool split_gate = fixture == ReadyArithmetic::kSplitResidual;
    const bool split_mod = fixture == ReadyArithmetic::kSplitModulate;
    const bool split_joined = fixture == ReadyArithmetic::kSplitJoined;
    const bool split_fixture = split_gate || split_mod || split_joined;
    std::string_view event_prefix, pattern, descriptor_phase;
    ir::FusedKind operation_kind = ir::FusedKind::kMatMulGlu;
    switch (fixture) {
      case ReadyArithmetic::kGdnGates:
        event_prefix = "ready_gdn_gates";
        pattern = "gdn_gates";
        descriptor_phase = "ready_gdn_gates_descriptor";
        operation_kind = ir::FusedKind::kGdnGates;
        break;
      case ReadyArithmetic::kHcCombine:
        event_prefix = "ready_hc_combine";
        pattern = "hc_combine";
        descriptor_phase = "ready_hc_combine_descriptor";
        operation_kind = ir::FusedKind::kHcCombine;
        break;
      case ReadyArithmetic::kHcPostNorm:
        event_prefix = "ready_hc_post_norm";
        pattern = "hc_post_flat_norm";
        descriptor_phase = "ready_hc_post_norm_descriptor";
        operation_kind = ir::FusedKind::kHcNormPost;
        break;
      case ReadyArithmetic::kSpatialNorm:
        event_prefix = "ready_spatial_channel_norm";
        pattern = "spatial_channel_norm";
        descriptor_phase = "ready_spatial_channel_norm_descriptor";
        operation_kind = ir::FusedKind::kSpatialChannelNorm;
        break;
      case ReadyArithmetic::kMatMulGlu:
        event_prefix = "ready_matmul_glu";
        pattern = "float_mul_mat_glu";
        descriptor_phase = "ready_matmul_glu_descriptor";
        operation_kind = ir::FusedKind::kMatMulGlu;
        break;
      case ReadyArithmetic::kNormAdd:
        event_prefix = "ready_norm_add";
        pattern = "rms_norm_mul_add";
        descriptor_phase = "ready_norm_add_descriptor";
        operation_kind = ir::FusedKind::kRmsNormMulAdd;
        break;
      case ReadyArithmetic::kMatMulAdd:
        event_prefix = "ready_matmul_add";
        pattern = "float_mul_mat_add";
        descriptor_phase = "ready_matmul_add_descriptor";
        operation_kind = ir::FusedKind::kMatMulAdd;
        break;
      case ReadyArithmetic::kSplitResidual:
        event_prefix = "ready_split_gated_residual";
        pattern = "split_gated_residual";
        descriptor_phase = "ready_split_gated_residual_descriptor";
        operation_kind = ir::FusedKind::kGatedResidualSplit;
        break;
      case ReadyArithmetic::kSplitModulate:
        event_prefix = "ready_split_modulate";
        pattern = "split_modulate";
        descriptor_phase = "ready_split_modulate_descriptor";
        operation_kind = ir::FusedKind::kLayerNormModulateSplit;
        break;
      case ReadyArithmetic::kSplitJoined:
        event_prefix = "ready_split_gated_residual_modulate";
        pattern = "split_gated_residual_modulate";
        descriptor_phase = "ready_split_gated_residual_modulate_descriptor";
        operation_kind = ir::FusedKind::kGatedResidualModulateSplit;
        break;
    }
    const auto operation = ir::Fused(operation_kind);
    const std::uint64_t cells = (gdn ? 48 : width) * rows * (hc_family ? 4 : 1);
    const std::uint64_t bytes = cells * sizeof(float);
    ir::Builder builder;
    builder.SetComponent(17);
    builder.SetOwner(0);
    const auto input =
        builder.Input(ir::Dense(Encoding::kF32, spatial      ? ir::ShapeOf({37, 3, width})
                                                : hc_combine ? ir::ShapeOf({width, 4, rows})
                                                             : ir::ShapeOf({width, rows})),
                      gdn             ? "ready-gdn-token-features"
                      : hc_family     ? "ready-hc-stream"
                      : spatial       ? "ready-spatial-stream"
                      : split_fixture ? "ready-split-stream"
                                      : "ready-matmul-glu-features");
    std::vector<ir::TensorId> dynamic_inputs{input}, weights;
    std::vector<ir::WeightRef> weight_refs;
    std::vector<std::string> names;
    std::vector<ir::TensorId> member_tensors, prepared_prefixes;
    auto weight = [&](Encoding encoding, const llmp::format::Shape& shape, std::string name) {
      const ir::WeightRef ref{.stage = 0,
                              .component = 17,
                              .role = static_cast<std::uint16_t>(weights.size() + 1),
                              .layer = 0};
      const auto tensor = builder.Weight(ir::Dense(encoding, shape), ref, name);
      weights.push_back(tensor);
      weight_refs.push_back(ref);
      names.push_back(std::move(name));
      return tensor;
    };
    ir::TensorId output;
    std::vector<ir::TensorId> original_outputs;
    if (gdn) {
      const auto wa = weight(Encoding::kF32, ir::ShapeOf({width, 48}), "gdn.alpha");
      const auto wb = weight(Encoding::kF32, ir::ShapeOf({width, 48}), "gdn.beta");
      const auto dt = weight(Encoding::kF32, ir::ShapeOf({48}), "gdn.dt_bias");
      const auto a = weight(Encoding::kF32, ir::ShapeOf({48}), "gdn.a");
      // Both authentic products are the pure prefix; the seven pointwise
      // members below are the planner's complete original gates group.
      const auto alpha_product = builder.MatMul(wa, input);
      const auto beta_product = builder.MatMul(wb, input);
      prepared_prefixes = {alpha_product, beta_product};
      const auto alpha = builder.Reshape(alpha_product, ir::ShapeOf({48, rows, 1}));
      const auto shifted = builder.Add(alpha, dt);
      const auto softplus = builder.Unary(shifted, ir::UnaryFn::kSoftplus);
      const auto scaled = builder.Mul(softplus, a);
      const auto gate = builder.Reshape(scaled, ir::ShapeOf({1, 48, rows, 1}));
      const auto beta_rows = builder.Reshape(beta_product, ir::ShapeOf({1, 48, rows, 1}));
      output = builder.Unary(beta_rows, ir::UnaryFn::kSigmoid);
      member_tensors = {alpha, shifted, softplus, scaled, gate, beta_rows, output};
      original_outputs.push_back(gate);
    } else if (hc_combine) {
      // CombineEmit's actual pointwise DAG after its logits product: the
      // three original inputs have independent, source-backed extents.
      const auto block = builder.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})),
                                       "ready-hc-block-output");
      const auto logits = builder.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({4, rows})),
                                        "ready-hc-combine-logits");
      dynamic_inputs.insert(dynamic_inputs.end(), {block, logits});
      const auto expanded = builder.Repeat(builder.Reshape(block, ir::ShapeOf({width, 1, rows})),
                                           ir::ShapeOf({width, 4, rows}));
      const auto sigmoid = builder.Unary(builder.Scale(logits, 0.25F), ir::UnaryFn::kSigmoid);
      const auto coefficients =
          builder.Reshape(builder.Scale(sigmoid, 2.0F), ir::ShapeOf({1, 4, rows}));
      output = builder.Add(input, builder.Mul(expanded, coefficients));
    } else if (hc) {
      const auto side = [&](const llmp::format::Shape& shape, std::string_view name) {
        const auto tensor = builder.Input(ir::Dense(Encoding::kF32, shape), name);
        dynamic_inputs.push_back(tensor);
        return tensor;
      };
      const auto residual = side(ir::ShapeOf({width, 4, rows}), "ready-hc-residual");
      const auto post_weights = side(ir::ShapeOf({4, rows}), "ready-hc-post");
      const auto comb = side(ir::ShapeOf({4, 4, rows}), "ready-hc-comb");
      const auto post = builder.HcPost(input, residual, post_weights, comb, {.streams = 4});
      const auto flat = builder.Reshape(post, ir::ShapeOf({width * 4, rows}));
      output = builder.RmsNorm(flat, 1e-6F);
      original_outputs.push_back(post);
    } else if (spatial) {
      const auto gamma = weight(Encoding::kF32, ir::ShapeOf({width}), "spatial.gamma");
      const auto permuted = builder.Permute(input, {1, 2, 0, 3});
      const auto contiguous = builder.Cont(permuted);
      const auto normalized = builder.L2Norm(contiguous, 1e-12F);
      const auto scaled = builder.Scale(normalized, std::sqrt(static_cast<float>(width)));
      auto value = builder.Mul(scaled, gamma);
      if (silu) value = builder.Unary(value, ir::UnaryFn::kSilu);
      output = builder.Cont(builder.Permute(value, {2, 0, 1, 3}));
    } else if (split_fixture) {
      // The existing image primitive graphs: independent [C] conditioning
      // vectors select the two row ranges. An identity repeat stays absent.
      const auto source_input = [&](const llmp::format::Shape& shape) {
        const auto tensor = builder.Input(ir::Dense(Encoding::kF32, shape));
        dynamic_inputs.push_back(tensor);
        return tensor;
      };
      const auto side = [&] {
        const auto first = source_input(ir::ShapeOf({width}));
        const auto second = source_input(ir::ShapeOf({width}));
        const auto repeated = [&](ir::TensorId tensor, std::uint64_t count) {
          return count == 1 ? tensor : builder.Repeat(tensor, ir::ShapeOf({width, count}));
        };
        const auto first_rows = repeated(first, split);
        const auto second_rows = repeated(second, rows - split);
        return builder.Concat(first_rows, second_rows, 1);
      };
      auto value = input;
      if (!split_mod) {
        const auto branch = source_input(ir::ShapeOf({width, rows}));
        value = builder.Add(input, builder.Mul(branch, builder.Unary(side(), ir::UnaryFn::kTanh)));
        if (split_joined) original_outputs.push_back(value);
      }
      if (split_gate) {
        output = value;
      } else {
        const auto conditioning = side();
        const auto normalized = builder.LayerNorm(value, 1e-6F);
        output = builder.Mul(normalized, builder.Scale(conditioning, 1.0F, 1.0F));
      }
    } else if (norm_add) {
      const auto gamma = weight(Encoding::kF32, ir::ShapeOf({width}), "norm.weight");
      const auto residual = builder.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})),
                                          "ready-norm-add-residual");
      dynamic_inputs.push_back(residual);
      const auto normalized = builder.RmsNorm(input, 1e-6F);
      const auto scaled = builder.Mul(normalized, gamma);
      output = builder.Add(scaled, residual);
      member_tensors = {normalized, scaled, output};
    } else {
      const auto matrix = weight(Encoding::kF16, ir::ShapeOf({width, width}),
                                 matmul_glu ? "gate.weight" : "matrix.weight");
      const auto second =
          weight(matmul_glu ? Encoding::kF16 : Encoding::kF32,
                 matmul_glu ? ir::ShapeOf({width, width}) : ir::ShapeOf({width, rows}),
                 matmul_glu ? "up.weight" : "bias.weight");
      const auto product = builder.MatMul(matrix, input);
      if (matmul_glu) {
        const auto up = builder.MatMul(second, input);
        output = builder.Glu(product, up, ir::GluParams{.kind = ir::GluKind::kSwiGlu});
        member_tensors = {product, up, output};
      } else {
        output = builder.Add(product, second);
        member_tensors = {product, output};
      }
    }
    original_outputs.push_back(output);
    for (const auto tensor : original_outputs) builder.Output(tensor);
    auto source_graph = Must(std::move(builder).Finish());
    std::vector<ir::NodeId> expected_members;
    if (split_fixture || spatial || hc_family) {
      for (std::uint32_t i = 0; i < source_graph.nodes().size(); ++i)
        if (source_graph.node(ir::NodeId{i}).key.op != ir::Op::kInput &&
            source_graph.node(ir::NodeId{i}).key.op != ir::Op::kWeight)
          expected_members.push_back(ir::NodeId{i});
    } else {
      for (const auto tensor : member_tensors)
        expected_members.push_back(source_graph.producer(tensor));
    }
    std::vector<std::uint64_t> input_bytes;
    for (const auto tensor : dynamic_inputs) {
      const auto storage = ir::StorageBytes(source_graph.tensor(tensor));
      if (!storage || *storage == 0 || *storage > kActivations)
        Fail("ready arithmetic original input storage is incomplete");
      input_bytes.push_back(*storage);
    }
    std::vector<llmp::format::PartExtent> parts;
    for (const auto tensor : weights) {
      const auto layout = llmp::format::PartsOf(source_graph.tensor(tensor).format);
      const auto logical = ir::StorageBytes(source_graph.tensor(tensor));
      if (!layout || layout->size() != 1 || !logical || layout->parts()[0].bytes != *logical ||
          layout->parts()[0].readable < *logical || layout->parts()[0].readable > kActivations)
        Fail("ready arithmetic weight exceeds its admitted typed backing");
      parts.push_back(layout->parts()[0]);
    }
    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    auto funded_graph = source_graph.Clone(funder);
    if (!funded_graph) Fail("ready arithmetic graph admission was refused");
    const auto source_identity = funded_graph->identity();
    co::StepGraph source{.graph = std::move(*funded_graph),
                         .inputs = {},
                         .weights = {},
                         .state = {},
                         .logits = gdn ? ir::TensorId{} : output,
                         .greedy = {},
                         .features = {},
                         .saves = {},
                         .owner_products = {}};
    for (std::size_t i = 0; i < dynamic_inputs.size(); ++i)
      source.inputs.push_back({.tensor = dynamic_inputs[i],
                               .kind = gdn      ? co::InputKind::kFeatures
                                       : i == 0 ? co::InputKind::kStream
                                                : co::InputKind::kSide,
                               .segment = 0,
                               .component = 17,
                               .what = gdn         ? "ready-gdn-token-features"
                                       : hc_family ? (i == 0 ? "ready-hc-stream" : "ready-hc-side")
                                       : spatial   ? "ready-spatial-stream"
                                       : split_fixture
                                           ? (i == 0 ? "ready-split-stream" : "ready-split-side")
                                       : i == 0 ? "ready-matmul-glu-features"
                                                : "ready-norm-add-residual"});
    for (std::size_t i = 0; i < weights.size(); ++i)
      source.weights.push_back(
          {.tensor = weights[i], .ref = weight_refs[i], .parts = {{names[i], parts[i].role}}});
    pl::StepShape shape{.stage = 0,
                        .phase = gdn ? ex::Phase::kVerify : ex::Phase::kPrefill,
                        .segments = {{.slot = 0,
                                      .rows = static_cast<std::uint32_t>(rows),
                                      .outputs = static_cast<std::uint32_t>(rows)}}};
    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    const pl::Policy base_policy{.active_stage = 0};
    const auto policy =
        gdn ? Must(pl::ForStep(base_policy, shape.rows(), shape.segments.size())) : base_policy;
    const auto activation_base = Allocate(kActivations);
    std::vector<std::uint64_t> weight_addresses;
    for (const auto& part : parts) weight_addresses.push_back(Allocate(part.readable));
    kn::Addresses addresses{
        .weight =
            [weight_refs, parts, weight_addresses](const ir::WeightRef& ref,
                                                   llmp::format::PartRole part) {
              for (std::size_t i = 0; i < weight_refs.size(); ++i)
                if (ref == weight_refs[i] && part == parts[i].role) return weight_addresses[i];
              return std::uint64_t{0};
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation_base,
        .activation_bytes = kActivations,
        .activation_alignment = 256,
        .weight_alignment =
            [weight_refs, parts](const ir::WeightRef& ref, llmp::format::PartRole part) {
              for (std::size_t i = 0; i < weight_refs.size(); ++i)
                if (ref == weight_refs[i] && part == parts[i].role) return std::uint64_t{16};
              return std::uint64_t{0};
            }};
    addresses.weight_range = [weight_refs, parts, weight_addresses](
                                 const ir::WeightRef& ref,
                                 llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
      for (std::size_t i = 0; i < weight_refs.size(); ++i)
        if (ref == weight_refs[i] && part == parts[i].role)
          return kn::BoundedAddress{weight_addresses[i], parts[i].readable, 16};
      return {};
    };
    // Actual token and staged-component constructors supply these same input
    // sets and backend storage/stream callbacks. No token or route is invented.
    const auto io = co::GraphIO(source);
    auto options = Must(en::PrepareComponentPlanOptions(
        io, table, addresses,
        pl::PlanOptions{.row_invariant = gdn && shape.row_invariant,
                        .capturable = true,
                        .staged = std::vector(io.inputs.begin(), io.inputs.end()),
                        .on_stream = table.on_stream,
                        .written = table.written,
                        .within = table.within,
                        .shape = gdn ? &shape : nullptr,
                        .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())}));
    std::vector<co::ComparisonExternalBinding> external;
    {
      // Resolve the real original source through the same paid path used by
      // production. This owner borrows source and dies before its move below.
      auto source_bindings =
          gdn ? Must(en::ComparisonSourceBindings::Create(funder, source, 0))
              : Must(en::ComparisonSourceBindings::Create(funder, co::GraphIO(source), 0));
      auto original = Must(source.graph.Clone(funder));
      auto baseline =
          Must(pl::Plan(std::move(original), *table.registry, table.device, policy, options));
      auto ranges = Must(source_bindings.Resolve(baseline, addresses, table.written));
      const auto view = source_bindings.boundary_view();
      if (!view || !view->unstored_inputs.empty())
        Fail("ready arithmetic source has no complete ordinary staged input boundary");
      external.assign(view->external.begin(), view->external.end());
    }
    const co::ComparisonBoundaryView source_view{.io = co::GraphIO(source),
                                                 .state = source.state,
                                                 .active_stage = 0,
                                                 .external = external,
                                                 .unstored_inputs = {}};
    const std::array filter{operation};
    pl::ComparisonSiteSelector selector;
    {
      auto discovered = pl::DiscoverComparisonPlan(source_view, filter, false, funder,
                                                   *table.registry, table.device, policy, options);
      if (!discovered) Fail(discovered.error().detail);
      const auto& discovery = discovered->discovery;
      const pl::ComparisonDiscoveryTicket* selected = nullptr;
      for (const auto& ticket : discovery.tickets()) {
        if (ticket.pattern != pattern || ticket.registered_operation != filter.front() ||
            !ticket.boundary_representable || ticket.member_offsets.size() < 2)
          continue;
        // A rewrite ticket must cover exactly the original arithmetic DAG.
        // Joined round-one fusion retains its two authenticated source groups.
        if (split_fixture || spatial || hc_family || gdn) {
          if (ticket.kind != pl::ComparisonSiteKind::kRewrite) continue;
          auto origins = ticket.member_origins;
          std::ranges::sort(origins, {}, &ir::NodeId::v);
          if (origins != expected_members) continue;
        } else {
          if (ticket.member_offsets.size() != expected_members.size() + 1 ||
              ticket.member_origins.size() != expected_members.size())
            continue;
          bool original_group = true;
          for (std::size_t i = 0; i < expected_members.size(); ++i)
            original_group = original_group && ticket.member_offsets[i] == i &&
                             ticket.member_origins[i] == expected_members[i];
          if (!original_group || ticket.member_offsets.back() != expected_members.size()) continue;
        }
        if (selected) Fail("ready arithmetic discovery found multiple complete original sites");
        selected = &ticket;
      }
      if (!selected || discovery.source_identity() != source_identity)
        Fail("ready arithmetic original source has no complete default float fusion site");
      selector.source_graph_identity = discovery.source_identity();
      selector.kind = selected->kind;
      selector.pattern = std::string(selected->pattern);
      selector.version = selected->version;
      selector.member_owner_segments = selected->owner_segments;
      selector.boundary_ports = selected->boundary_ports;
      if (selected->member_offsets.front() != 0 ||
          selected->member_offsets.back() != selected->member_origins.size())
        Fail("ready arithmetic discovery returned malformed source grouping");
      for (std::size_t i = 0; i + 1 < selected->member_offsets.size(); ++i) {
        const auto first = selected->member_offsets[i], last = selected->member_offsets[i + 1];
        if (first >= last || last > selected->member_origins.size())
          Fail("ready arithmetic discovery returned an invalid source group");
        selector.member_origins.emplace_back(selected->member_origins.begin() + first,
                                             selected->member_origins.begin() + last);
      }
    }
    std::vector<ir::NodeId> selected_origins;
    for (const auto& group : selector.member_origins)
      selected_origins.insert(selected_origins.end(), group.begin(), group.end());
    std::ranges::sort(selected_origins, {}, &ir::NodeId::v);
    if (selected_origins != expected_members)
      Fail("ready arithmetic selected group differs from its complete original source");
    std::optional<co::OwnedComparisonRegionBoundary> retained_boundary;
    {
      std::vector<std::uint32_t> offsets{0};
      std::vector<ir::NodeId> members;
      for (const auto& group : selector.member_origins) {
        members.insert(members.end(), group.begin(), group.end());
        offsets.push_back(static_cast<std::uint32_t>(members.size()));
      }
      auto boundary = co::MakeComparisonRegionBoundary({source_view, offsets, members}, funder);
      if (!boundary) Fail(boundary.error().detail);
      retained_boundary.emplace(std::move(*boundary));
    }
    auto made = en::ComparisonPair::Create(
        funder, std::move(source), std::move(shape), std::move(external), 0, 0,
        gdn ? en::ComparisonInputDomain::kTokenStep : en::ComparisonInputDomain::kComponentPorts);
    if (!made) Fail(made.error().detail);
    auto pair = std::move(*made);
    if (gdn) options.shape = &pair->shape();
    auto planned = pair->Plan(selector, *table.registry, table.device, policy, options);
    if (!planned) Fail(planned.error().detail);
    const auto* incumbent = pair->plan(en::ComparisonPairArm::kIncumbent);
    const auto* alternative = pair->plan(en::ComparisonPairArm::kAlternative);
    if (!incumbent || !alternative ||
        std::ranges::none_of(incumbent->plan.applied,
                             [&](const auto& applied) { return applied.pattern == pattern; }) ||
        std::ranges::any_of(alternative->plan.applied,
                            [&](const auto& applied) { return applied.pattern == pattern; }))
      Fail("ready arithmetic pair did not preserve the complete ordinary fusion/decomposition");
    auto source_bindings =
        gdn ? Must(en::ComparisonSourceBindings::Create(funder, pair->source(), 0))
            : Must(en::ComparisonSourceBindings::Create(funder, co::GraphIO(pair->source()), 0));
    std::array<std::optional<en::ComparisonResolvedBindings>, 2> resolved;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      auto ranges =
          Must(source_bindings.Resolve(pair->plan(arm == 0 ? en::ComparisonPairArm::kIncumbent
                                                           : en::ComparisonPairArm::kAlternative)
                                           ->plan,
                                       addresses, table.written));
      // The move-only paid range owner remains alive through InstallBindings.
      resolved[arm].emplace(std::move(ranges));
    }
    const std::array<std::span<const kn::ComparisonBindingRange>, 2> bindings{
        resolved[0]->ranges(), resolved[1]->ranges()};
    auto installed_bindings = pair->InstallBindings(bindings);
    if (!installed_bindings) Fail(installed_bindings.error().detail);
    std::array<kn::ComparisonRegionRecipeSeal, 2> ready_descriptors;
    std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
    std::array<kn::StepProgram*, 2> live_programs{};
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      auto lower = pair->LoweringInput(which, addresses);
      if (!lower) Fail(lower.error().detail);
      auto lowered = table.lower(*lower);
      if (!lowered) Fail(lowered.error());
      if ((*lowered)->scratch_bytes() > kActivations || (*lowered)->pool_bytes() > kPool)
        Fail("ready arithmetic lowered storage exceeds admitted Screen backing");
      if (whole_control) {
        const auto whole = (*lowered)->comparison_recipe();
        if (!whole || whole->state != kn::ComparisonProofState::kDescriptorVerified ||
            whole->missing != kn::ComparisonMissing::kNone) {
          if (gdn)
            (*lowered)->StreamPlanReport(
                [](std::string_view fragment) { std::print("{}", fragment); });
          Fail(
              std::format("ready arithmetic whole descriptor proof is incomplete: arm {}, "
                          "state {}, missing {}",
                          arm, whole ? static_cast<unsigned>(whole->state) : 255,
                          whole ? static_cast<unsigned>(whole->missing) : 255));
        }
      }
      const auto local = (*lowered)->ProjectComparisonRegionCertificate(
          *retained_boundary, kn::ComparisonCertificatePhase::kDescriptor);
      if (local.status != kn::ComparisonRegionStatus::kClosed)
        Fail("ready arithmetic lacks its exact descriptor applicability certificate");
      ready_descriptors[arm] = local.seal;
      programs[arm] = std::move(*lowered);
      live_programs[arm] = programs[arm].get();
    }
    auto installed_programs =
        pair->InstallDescriptorPrograms(std::move(programs[0]), std::move(programs[1]));
    if (!installed_programs) Fail(installed_programs.error().detail);
    phase = descriptor_phase;
    auto described = en::DescribeRecipeComparison(
        *pair, *table.registry, en::ComparisonRecipePreparation::kStatelessReadyBoundaryOperands);
    if (!described) Fail(described.error().detail);
    if (!*described) {
      if (retained_boundary) {
        for (std::size_t arm = 0; arm < 2; ++arm) {
          for (const bool certificate : {false, true}) {
            const auto result =
                certificate
                    ? live_programs[arm]->ProjectComparisonRegionCertificate(
                          *retained_boundary, kn::ComparisonCertificatePhase::kDescriptor)
                    : live_programs[arm]->ProjectComparisonRegionDescriptor(*retained_boundary);
            std::string required = "[";
            for (const auto member : result.required_members) {
              if (required.size() > 1) required += ',';
              required += std::to_string(member.v);
            }
            required += ']';
            Receipt(
                std::format("{}_descriptor_refusal", event_prefix),
                std::format(
                    "\"arm\":{},\"purpose\":{},\"status\":{},\"missing\":{},"
                    "\"reason\":{},\"required_members\":{}",
                    arm, Quote(certificate ? "applicability" : "executable"),
                    Quote(result.status == kn::ComparisonRegionStatus::kClosed   ? "closed"
                          : result.status == kn::ComparisonRegionStatus::kExpand ? "expand"
                                                                                 : "unproved"),
                    static_cast<unsigned>(result.seal.missing), Quote(result.reason), required));
          }
        }
      }
      Fail("ready arithmetic actual recipes did not produce a closed Ready descriptor key");
    }
    const auto key = **described;
    const auto* ranked = table.registry->FindRecipeComparison(key);
    Receipt(
        std::format("{}_rank_lookup", event_prefix),
        std::format("\"recipe_key_digest\":{},\"registered_rows\":{},\"matched\":{}",
                    Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key))),
                    table.registry->ranks().recipe_comparisons.size(), ranked ? "true" : "false"));
    auto bound = pair->BindPrograms(*launch_, *launch_);
    if (!bound) Fail(bound.error().detail);
    if (whole_control) {
      for (const auto* program : live_programs) {
        const auto whole = program->comparison_recipe();
        if (!whole || whole->state != kn::ComparisonProofState::kBoundVerified ||
            whole->missing != kn::ComparisonMissing::kNone)
          Fail("ready arithmetic whole bound proof is incomplete");
        const auto remainder = program->CompareComparisonRemainder(
            *program, *retained_boundary, kn::ComparisonCertificatePhase::kBound);
        if (remainder.status != kn::ComparisonRemainderStatus::kSame) Fail(remainder.reason);
      }
      // CloseComparisonPair below reacquires all borrowed region ports after
      // the self-remainder analyses have rebuilt the program-owned tables.
    }
    std::array<std::vector<std::string_view>, 2> expected_calls =
        norm_add ? std::array{std::vector<std::string_view>{"ggml.rms_norm_mul_add.fused"},
                              std::vector<std::string_view>{"ggml.rms_norm_mul.fused", "ggml.add"}}
        : matmul_glu ? std::array{std::vector<std::string_view>{"ggml.mul_mat_glu.mmvf_fused"},
                                  std::vector<std::string_view>{"ggml.mul_mat.mmvf",
                                                                "ggml.mul_mat.mmvf", "ggml.swiglu"}}
                     : std::array{std::vector<std::string_view>{"ggml.mul_mat_add.mmvf_fused"},
                                  std::vector<std::string_view>{"ggml.mul_mat.mmvf", "ggml.add"}};
    if (gdn) {
      const std::string_view product = rows <= 3 ? "ggml.mul_mat.mmvf" : "ggml.mul_mat.cublas";
      expected_calls[0] = {product, product, "llmp.gdn.gates"};
      expected_calls[1] = {product, product, "ggml.add", "ggml.unary", "ggml.mul", "ggml.unary"};
    } else if (hc_combine) {
      expected_calls[0] = {"llmp.hc.combine"};
      expected_calls[1] = {"ggml.repeat", "ggml.scale", "ggml.unary",
                           "ggml.scale",  "ggml.mul",   "ggml.add"};
    } else if (hc) {
      expected_calls[0] = {"llmp.dsv4.hc_post_norm_f16"};
      expected_calls[1] = {"ggml.dsv4_hc_post", "ggml.rms_norm"};
    } else if (spatial) {
      expected_calls[0] = {"llmp.image.spatial_channel_norm"};
      expected_calls[1] = {"ggml.cont", "ggml.l2_norm", "ggml.scale", "ggml.mul"};
      if (silu) expected_calls[1].push_back("ggml.unary");
      expected_calls[1].push_back("ggml.cont");
    } else if (split_fixture) {
      expected_calls[0] = {split_gate  ? "llmp.image.gated_residual_split"
                           : split_mod ? "llmp.image.layer_norm_modulate_split"
                                       : "llmp.image.gated_residual_modulate_split"};
      if (split_joined) {
        expected_calls[1] = {"llmp.image.gated_residual_split",
                             "llmp.image.layer_norm_modulate_split"};
      } else {
        expected_calls[1].clear();
        if (split != 1) expected_calls[1].push_back("ggml.repeat");
        if (rows - split != 1) expected_calls[1].push_back("ggml.repeat");
        expected_calls[1].push_back("ggml.concat");
        if (split_gate) {
          expected_calls[1].insert(expected_calls[1].end(), {"ggml.unary", "ggml.mul", "ggml.add"});
        } else {
          expected_calls[1].insert(expected_calls[1].end(),
                                   {"ggml.norm", "ggml.scale", "ggml.mul"});
        }
      }
    }
    std::array<std::uint64_t, 2> body_launches{}, program_pool{}, private_pool{};
    std::string actual_calls = "[";
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto* actual = live_programs[arm]->bound();
      if (!actual) Fail("ready arithmetic bound recipe is absent");
      std::vector<std::string_view> call_names;
      for (std::size_t i = 0; i < actual->size(); ++i) call_names.push_back(actual->at(i).name);
      if (hc_combine) {
        std::string inventory = "[";
        for (std::size_t i = 0; i < call_names.size(); ++i) {
          if (i) inventory += ',';
          inventory += Quote(call_names[i]);
        }
        inventory += ']';
        Receipt("ready_hc_combine_actual_inventory",
                std::format("\"arm\":{},\"calls\":{}", arm, inventory));
      }
      bool accepted = call_names == expected_calls[arm];
      if (norm_add && arm == 1)
        accepted =
            accepted ||
            call_names == std::vector<std::string_view>{"ggml.rms_norm_mul.unfused", "ggml.add"} ||
            call_names == std::vector<std::string_view>{"ggml.rms_norm", "ggml.mul", "ggml.add"};
      if (!accepted)
        Fail("ready arithmetic actual calls differ from the ordinary source-backed recipe");
      // These checked calls each launch one kernel, except RMSMul's explicit
      // unfused implementation (norm plus multiply). StepProgram::launches()
      // is a conservative whole-program bound, not the actual fused count.
      body_launches[arm] = call_names.size() - (gdn ? 2 : 0);
      program_pool[arm] = live_programs[arm]->pool_bytes();
      private_pool[arm] = live_programs[arm]->private_pool_bytes();
      if (norm_add && arm == 1 && call_names.front() == "ggml.rms_norm_mul.unfused")
        ++body_launches[arm];
      if (arm) actual_calls += ',';
      actual_calls += '[';
      for (std::size_t i = 0; i < call_names.size(); ++i) {
        if (i) actual_calls += ',';
        actual_calls += Quote(call_names[i]);
      }
      actual_calls += ']';
    }
    actual_calls += ']';
    auto closed = en::CloseComparisonPair(std::move(pair));
    if (!closed) Fail(closed.error().detail);
    pair = std::move(*closed);
    bool repeat_refusal_checked = false;
    std::array<std::size_t, 2> captured_nodes{};
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      if (gdn && rows == 1 && arm == 1) {
        const auto repeated = pair->CaptureRegion(which, *launch_, native_, 2);
        Finish();
        if (repeated)
          Fail("GDN clobbered prepared inputs unexpectedly allowed two body repetitions");
        if (pair->captured_region_inventory(which, *launch_))
          Fail("GDN refused repetition left an unexpected recording");
        repeat_refusal_checked = true;
        Receipt("ready_gdn_unsafe_repeat_refused",
                "\"arm\":1,\"requested_repetitions\":2,\"recording_absent_after_refusal\":true");
      }
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
      auto captured = pair->CaptureRegion(which, *launch_, native_, 1);
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
      if (!captured) Fail(captured.error().detail);
      const auto inventory = pair->captured_region_inventory(which, *launch_);
      if (!inventory || inventory->repetitions != 1 ||
          inventory->recorded_nodes != body_launches[arm] + 2)
        Fail("ready arithmetic capture is not one complete body plus two event nodes");
      captured_nodes[arm] = inventory->recorded_nodes;
    }
    std::vector<std::vector<std::byte>> weight_bytes(weights.size());
    std::vector<std::vector<float>> rounded_weights(weights.size());
    for (std::size_t w = 0; w < weights.size(); ++w) {
      weight_bytes[w].resize(static_cast<std::size_t>(parts[w].readable));
      const auto encoding = source_graph.tensor(weights[w]).format.encoding;
      const auto unit = encoding == Encoding::kF16 ? sizeof(__half) : sizeof(float);
      rounded_weights[w].resize(static_cast<std::size_t>(parts[w].bytes / unit));
      for (std::size_t i = 0; i < rounded_weights[w].size(); ++i) {
        const float original =
            encoding == Encoding::kF16
                ? static_cast<float>(static_cast<int>((i * 37U + w * 13U) % 251U) - 125) /
                      (8193.0F + static_cast<float>(w) * 16.0F)
            : gdn
                ? (w < 2 ? static_cast<float>(static_cast<int>((i * 37U + w * 13U) % 251U) - 125) /
                               8193.0F
                   : w == 2 ? -0.2F + static_cast<float>(i % 7U) * 0.07F
                            : -0.8F - static_cast<float>(i % 5U) * 0.03F)
            : (norm_add || spatial) ? 0.75F + static_cast<float>(i % 9U) * 0.03125F
                                    : static_cast<float>(static_cast<int>(i % 11U) - 5) * 0.015625F;
        if (encoding == Encoding::kF16) {
          const auto half = __float2half_rn(original);
          std::memcpy(weight_bytes[w].data() + i * sizeof(half), &half, sizeof(half));
          rounded_weights[w][i] = __half2float(half);
        } else {
          std::memcpy(weight_bytes[w].data() + i * sizeof(float), &original, sizeof(float));
          rounded_weights[w][i] = original;
        }
      }
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(weight_addresses[w]), weight_bytes[w].data(),
                            weight_bytes[w].size(), cudaMemcpyHostToDevice, CudaStream()));
    }
    Finish();
    auto reference = ref::Executor::Create(source_graph);
    if (!reference) Fail(reference.error().detail);
    for (std::size_t w = 0; w < weights.size(); ++w)
      if (!reference->BindValues(weights[w], rounded_weights[w]))
        Fail("ready arithmetic CPU reference refused actual immutable weight values");
    const std::array<double, 2> reference_limits =
        gdn                      ? std::array{1e-4, 1e-4}
        : (spatial || hc_family) ? std::array{1e-5, 1e-5}
        : split_fixture          ? std::array{split_mod ? 1e-5 : 1e-4, split_mod ? 1e-5 : 1e-4}
        : norm_add               ? std::array{1e-5, 1e-5}
        : matmul_glu             ? std::array{1e-4, 2e-2}
                                 : std::array{2e-2, 2e-2};
    std::array<double, 2> max_relative{}, max_absolute{};
    std::array<double, 2> prefix_max_relative{};
    std::array<bool, 2> dead_private_prefix_reuse{};
    // RMS uses the existing normalized ReadyNorm bound. F16 bias uses the
    // symmetric DEFAULT-product bound. GLU retains its existing double-product
    // convention in addition to the complete original-IR CPU output.
    Arm product_reference;
    if (matmul_glu) {
      product_reference.host_inputs.push_back(
          {.logical_index = 0, .bytes = std::vector<std::byte>(bytes)});
      for (std::size_t w = 0; w < weights.size(); ++w) {
        const auto logical =
            std::span(weight_bytes[w]).first(static_cast<std::size_t>(parts[w].bytes));
        product_reference.host_weights.push_back(
            {.ref = weight_refs[w],
             .bytes = std::vector<std::byte>(logical.begin(), logical.end())});
      }
    }
    std::array<std::vector<std::vector<float>>, 2> first_outputs;
    std::vector<std::vector<float>> first_expected;
    // Functional controls and optional samples use this exact invocation:
    // fresh original inputs and prefix, followed by one captured body. CPU,
    // readback and immutable-resource checks remain outside its event nodes.
    std::unique_ptr<kn::StepProgram> production;
    const auto run_frame = [&](std::uint32_t frame, std::optional<std::size_t> measured_arm,
                               bool normal_program = false) {
      std::uint64_t duration_ns = 0;
      std::vector<std::vector<float>> frame_inputs(dynamic_inputs.size());
      for (std::size_t input_index = 0; input_index < frame_inputs.size(); ++input_index) {
        frame_inputs[input_index].resize(
            static_cast<std::size_t>(input_bytes[input_index] / sizeof(float)));
        for (std::size_t i = 0; i < frame_inputs[input_index].size(); ++i) {
          const auto stride =
              (split_fixture || hc_family) ? 17U + input_index * 12U : (input_index ? 29U : 13U);
          const float centered = static_cast<float>(static_cast<int>((i * stride) % 97U) - 48);
          const float distinct_source =
              (split_fixture || hc_family) ? static_cast<float>(input_index) : 0.0F;
          frame_inputs[input_index][i] =
              centered * (frame == 0 ? 0.0234375F : 0.015625F) +
              (frame == 0 ? 0.125F + distinct_source * 0.03125F
                          : static_cast<float>(i % 7U) * 0.01171875F - distinct_source * 0.015625F);
        }
        if (spatial) {
          // Every channel shares the same spatial index. Exercise exact zero
          // and the L2 epsilon clamp without changing the source geometry.
          for (std::size_t channel = 0; channel < width; ++channel) {
            frame_inputs[input_index][channel * 111] = 0.0F;
            frame_inputs[input_index][channel * 111 + 1] =
                (frame == 0 ? 1e-16F : -1.5e-16F) * static_cast<float>(1 + channel % 3);
          }
        }
        if (!reference->BindValues(dynamic_inputs[input_index], frame_inputs[input_index]))
          Fail("ready arithmetic CPU reference refused an original component input");
      }
      Must(reference->Run());
      std::vector<std::vector<float>> expected;
      for (const auto tensor : original_outputs) {
        auto values = reference->Values(tensor);
        if (!values || values->size() != cells)
          Fail("ready arithmetic CPU reference omitted a complete original-IR output");
        expected.emplace_back(values->begin(), values->end());
      }
      std::vector<std::vector<float>> expected_prefix;
      for (const auto tensor : prepared_prefixes) {
        auto values = reference->Values(tensor);
        if (!values || values->size() != cells)
          Fail("GDN CPU reference omitted a complete original prefix product");
        expected_prefix.emplace_back(values->begin(), values->end());
      }
      if (frame == 0)
        first_expected = expected;
      else
        for (std::size_t i = 0; i < expected.size(); ++i)
          if (expected[i] == first_expected[i])
            Fail("ready arithmetic changed source frame did not change a CPU output");
      if (matmul_glu)
        std::memcpy(product_reference.host_inputs[0].bytes.data(), frame_inputs[0].data(),
                    static_cast<std::size_t>(bytes));
      for (std::size_t arm = 0; arm < 2; ++arm) {
        if (measured_arm && arm != *measured_arm) continue;
        const auto which =
            arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
        const auto slots = normal_program ? production->inputs() : pair->input_slots(which);
        // Every ordinary F32 stream/side port is restaged after NaN poisoning.
        if (slots.size() != dynamic_inputs.size())
          Fail("ready arithmetic component-port inventory changed");
        std::vector<bool> seen(dynamic_inputs.size(), false);
        Check(cudaMemsetAsync(reinterpret_cast<void*>(activation_base), 0xFF,
                              static_cast<std::size_t>(kActivations), CudaStream()));
        for (const auto& slot : slots) {
          const auto found = std::ranges::find(dynamic_inputs, slot.tensor);
          if (found == dynamic_inputs.end() ||
              slot.kind != (gdn ? kn::InputKind::kFeatures : kn::InputKind::kStepBytes))
            Fail("ready arithmetic actual component port differs from its original tensor/extent");
          const auto index = static_cast<std::size_t>(found - dynamic_inputs.begin());
          if (slot.bytes != input_bytes[index])
            Fail("ready arithmetic actual input extent changed");
          if (seen[index]) Fail("ready arithmetic component port is duplicated");
          seen[index] = true;
          Check(cudaMemcpyAsync(
              reinterpret_cast<void*>(slot.device_address), frame_inputs[index].data(),
              static_cast<std::size_t>(input_bytes[index]), cudaMemcpyHostToDevice, CudaStream()));
        }
        Finish();
        if (normal_program) {
          Must(production->Run(*launch_, native_));
          Finish();
        } else {
          Must(pair->PrepareFrame(which, *launch_, native_));
          Finish();
          // A frame prepares the actual original products once, then consumes
          // them in one body replay. Dead private product storage may be reused
          // later in that body; check its complete value before consumption.
          for (std::size_t i = 0; i < prepared_prefixes.size(); ++i) {
            bool found = false;
            for (const auto& port : pair->value_ports(which)) {
              if (port.kind != kn::ComparisonRegionValueKind::kInput ||
                  port.source_tensor != prepared_prefixes[i])
                continue;
              if (found || port.bytes != bytes ||
                  port.role != source_graph.tensor(prepared_prefixes[i]).role)
                Fail("GDN prepared prefix port differs from its original complete product");
              found = true;
              std::vector<float> actual(static_cast<std::size_t>(cells));
              Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(port.address),
                               static_cast<std::size_t>(port.bytes), cudaMemcpyDeviceToHost));
              double gap = 0.0, scale = 1.0;
              for (std::size_t j = 0; j < actual.size(); ++j) {
                if (!std::isfinite(actual[j]) || !std::isfinite(expected_prefix[i][j]))
                  Fail("GDN prepared prefix or its original CPU reference is nonfinite");
                gap =
                    std::max(gap, std::abs(static_cast<double>(actual[j]) - expected_prefix[i][j]));
                scale = std::max(scale, std::abs(static_cast<double>(expected_prefix[i][j])));
              }
              // Prospectively use the same complete-tensor convention and
              // 1e-4 bound as the existing full-GDN original-IR reference check.
              prefix_max_relative[arm] = std::max(prefix_max_relative[arm], gap / scale);
              if (gap / scale > reference_limits[arm])
                Fail("GDN prepared prefix exceeded its prospective full-GDN reference bound");
              for (const auto& output_port : pair->value_ports(which)) {
                if (output_port.kind != kn::ComparisonRegionValueKind::kOutput) continue;
                const bool overlaps = port.address <= output_port.address
                                          ? output_port.address - port.address < port.bytes
                                          : port.address - output_port.address < output_port.bytes;
                dead_private_prefix_reuse[arm] = dead_private_prefix_reuse[arm] || overlaps;
              }
            }
            if (!found) Fail("GDN complete prepared prefix product is absent");
          }
          Must(pair->ReplayCapturedRegion(which, *launch_, native_));
          Finish();
          const double duration = Must(pair->captured_region_duration_ns(which, *launch_));
          if (!std::isfinite(duration) || duration <= 0 ||
              duration >= static_cast<double>(std::numeric_limits<long long>::max()))
            Fail("ready arithmetic completed capture duration is invalid");
          duration_ns = static_cast<std::uint64_t>(std::llround(duration));
          if (duration_ns == 0) Fail("ready arithmetic duration rounds to zero nanoseconds");
        }
        std::vector<std::vector<float>> actual_outputs;
        for (std::size_t output_index = 0; output_index < original_outputs.size(); ++output_index) {
          const auto checked_output = original_outputs[output_index];
          std::vector<float> actual(static_cast<std::size_t>(cells));
          bool found_output = false;
          if (normal_program) {
            for (const auto& port : production->outputs()) {
              if (port.tensor != checked_output) continue;
              if (found_output || port.bytes != bytes)
                Fail("normal arithmetic program changed a complete original output");
              found_output = true;
              Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(port.device_address),
                               static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost));
            }
          } else {
            for (const auto& port : pair->value_ports(which)) {
              if (port.kind != kn::ComparisonRegionValueKind::kOutput ||
                  port.source_tensor != checked_output)
                continue;
              if (found_output || port.bytes != bytes ||
                  port.role != source_graph.tensor(checked_output).role)
                Fail("ready arithmetic complete output port is malformed");
              found_output = true;
              Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(port.address),
                               static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost));
            }
          }
          if (!found_output) Fail("ready arithmetic original-IR output is absent");
          double output_gap = 0.0, output_scale = 1.0;
          for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[output_index][i]))
              Fail("ready arithmetic reference or device output is nonfinite");
            const double gap = std::abs(static_cast<double>(actual[i]) - expected[output_index][i]);
            const double magnitude = std::abs(static_cast<double>(expected[output_index][i]));
            const double relative = gap / (spatial         ? std::max(magnitude, 1e-3)
                                           : split_fixture ? std::max(magnitude, 1.0)
                                                           : 1.0 + magnitude);
            output_gap = std::max(output_gap, gap);
            output_scale = std::max(output_scale, magnitude);
            if (!matmul_glu && !gdn) {
              max_absolute[arm] = std::max(max_absolute[arm], gap);
              max_relative[arm] = std::max(max_relative[arm], relative);
            }
            if (!gdn && (hc_family ? gap > reference_limits[arm] * (1.0 + magnitude)
                         : (split_fixture || spatial) ? relative > reference_limits[arm]
                                                      : relative >= reference_limits[arm]))
              Fail("ready arithmetic exceeded its existing source-backed numerical bound");
          }
          if (gdn) {
            const double normalized_gap = output_gap / output_scale;
            max_absolute[arm] = std::max(max_absolute[arm], output_gap);
            max_relative[arm] = std::max(max_relative[arm], normalized_gap);
            if (normalized_gap > reference_limits[arm])
              Fail("GDN complete original CPU output exceeded its full-GDN numerical bound");
          }
          if (matmul_glu) {
            product_reference.values = actual;
            const auto product_error = ProductReferenceError(product_reference, width, rows);
            if (product_error.maximum >= reference_limits[arm])
              Fail("ready arithmetic exceeded the existing double product-reference bound");
            max_relative[arm] = std::max(max_relative[arm], product_error.maximum);
            max_absolute[arm] = std::max(max_absolute[arm], product_error.max_absolute);
          }
          if (frame != 0 && actual == first_outputs[arm][output_index])
            Fail("ready arithmetic changed component input did not change a captured output");
          actual_outputs.push_back(std::move(actual));
        }
        if (frame == 0) first_outputs[arm] = std::move(actual_outputs);
        for (std::size_t w = 0; w < weights.size(); ++w) {
          std::vector<std::byte> after(weight_bytes[w].size());
          Check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(weight_addresses[w]),
                           after.size(), cudaMemcpyDeviceToHost));
          if (after != weight_bytes[w])
            Fail("ready arithmetic changed immutable weight bytes or readable tails");
        }
        if (!normal_program) {
          const auto inventory = pair->captured_region_inventory(which, *launch_);
          if (!inventory || inventory->recorded_nodes != captured_nodes[arm] ||
              inventory->repetitions != 1)
            Fail("ready arithmetic capture inventory changed between restaged frames");
        }
      }
      return duration_ns;
    };
    for (std::uint32_t frame = 0; frame < 2; ++frame)
      for (std::size_t arm = 0; arm < 2; ++arm) (void)run_frame(frame, arm);
    Finish();
    // Audit the actual production construction before replacement samples.
    // An absent operation in the current table legitimately takes ordinary
    // fallback; it provides complete plan identity, not a fabricated local seal.
    // Like the existing RMS installed-row callback, this observer borrows only
    // the uniquely retained chosen native program, then verifies it through Bind.
    {
      if (!installed_selection.empty() &&
          ((installed_selection == "hit") != (ranked != nullptr) ||
           (ranked && ranked->kind != ex::RecipeComparisonKind::kLatency)))
        Fail("installed arithmetic row differs from the requested hit or miss");
      const auto preference =
          ranked ? ex::ClassifyRecipeLatency(ranked->incumbent_ns, ranked->alternative_ns)
                 : ex::RecipePreference::kNeutral;
      const std::size_t chosen = ranked && preference == ex::RecipePreference::kAlternative ? 1 : 0;
      const auto expected_plan = live_programs[chosen]->bound()->identity();
      const auto& expected_graph = chosen == 0 ? incumbent->plan.graph : alternative->plan.graph;
      struct Observed {
        const kn::StepProgram* native;
        std::uint64_t identity;
        llmp::base::Sha256Digest graph;
        kn::ComparisonRegionDescriptorResult local;
      };
      std::vector<Observed> observed;
      const kn::ProgramValidator observe =
          [&](const kn::LowerInput& lower,
              const kn::StepProgram& program) -> std::expected<void, std::string> {
        if (&lower.io.graph != &pair->source().graph || lower.token.has_value() != gdn ||
            (gdn && (&lower.token->step != &pair->source() ||
                     &lower.token->shape != &pair->shape() || lower.token->context != 0)))
          return std::unexpected(
              "installed local comparison changed its real ComponentPorts source");
        auto local = program.ProjectComparisonRegionCertificate(
            *retained_boundary, kn::ComparisonCertificatePhase::kDescriptor);
        // Keep no borrowed expansion spans across the next backend query.
        local.required_members = {};
        const auto* native = &program.comparison_program();
        const Observed current{native, lower.plan.graph.identity(), lower.plan.graph.Digest(),
                               local};
        const auto previous = std::ranges::find(observed, native, &Observed::native);
        if (previous == observed.end())
          observed.push_back(current);
        else
          *previous = current;
        return {};
      };
      production = gdn ? Must(en::BuildCumulativeRecipeProgram(
                             co::GraphIO(pair->source()), pair->source(), pair->shape(), 0, 0,
                             table, policy, addresses, options, observe, owned.owner))
                       : Must(en::BuildComponentProgram(co::GraphIO(pair->source()), table, policy,
                                                        addresses, options, std::nullopt, observe,
                                                        owned.owner));
      const auto identity = production->comparison_plan_identity();
      if (identity && (identity->source_graph_identity != pair->source().graph.identity() ||
                       std::ranges::count(observed, identity->planned_graph_identity,
                                          &Observed::identity) != 1))
        Fail("normal local selection has no unique retained source and plan observation");
      if (!identity && !ranked &&
          std::ranges::count(observed, &production->comparison_program(), &Observed::native) != 1)
        Fail("normal local miss has no unique retained native observation");
      const auto at =
          identity
              ? std::ranges::find(observed, identity->planned_graph_identity, &Observed::identity)
          : !ranked
              ? std::ranges::find(observed, &production->comparison_program(), &Observed::native)
              : observed.end();
      if (at == observed.end() || at->graph != expected_graph.Digest())
        Fail("normal local adoption changed its exact complete planned graph");
      bool query_key_checked = false;
      if (at->local.status == kn::ComparisonRegionStatus::kClosed) {
        auto actual = ready_descriptors;
        actual[chosen] = at->local.seal;
        const auto actual_key = en::RecipeComparisonKeyForReadyCertificates(
            *retained_boundary, pair->inventory()->site.registered_result, pair->shape().phase,
            table.registry->SelectionDigest(), actual[0], actual[1]);
        if (!actual_key || *actual_key != key)
          Fail("normal local selection differs from its authentic installed Ready key");
        query_key_checked = true;
      } else if (ranked) {
        Fail("normal arithmetic hit lacks its chosen descriptor certificate");
      }
      Must(production->Bind(*launch_));
      // The wrapper deliberately denies external local projections. This
      // borrowed native observation is joined to its retained plan identity;
      // Bind and execution still go through the normal production owner.
      bool local_bound = false;
      if (at->local.status == kn::ComparisonRegionStatus::kClosed) {
        const auto bound_certificate = at->native->ProjectComparisonRegionCertificate(
            *retained_boundary, kn::ComparisonCertificatePhase::kBound);
        const auto persistent = [](const kn::ComparisonRegionRecipeSeal& seal) {
          return std::tie(seal.version, seal.missing, seal.subject, seal.boundary,
                          seal.planned_region, seal.physical_recipe, seal.policy, seal.registry,
                          seal.backend_recipe, seal.backend_build, seal.backend_selection,
                          seal.backend_device);
        };
        local_bound = bound_certificate.status == kn::ComparisonRegionStatus::kClosed &&
                      at->local.seal.state == kn::ComparisonProofState::kDescriptorVerified &&
                      bound_certificate.seal.state == kn::ComparisonProofState::kBoundVerified &&
                      bound_certificate.seal.missing == kn::ComparisonMissing::kNone &&
                      persistent(at->local.seal) == persistent(bound_certificate.seal);
        if (!local_bound) Fail("normal local chosen certificate changed from Descriptor to Bound");
      } else if (ranked) {
        Fail("normal arithmetic hit lost its retained local certificate");
      }
      if (!production->bound() || production->bound()->identity() != expected_plan)
        Fail("normal local selection changed its complete ordered bound implementation plan");
      Finish();

      if (!installed_selection.empty()) {
        for (std::uint32_t frame = 0; frame < 2; ++frame) (void)run_frame(frame, chosen, true);
      }
      Receipt(
          std::format("{}_production_selection", event_prefix),
          std::format(
              "\"expected\":{},\"matched\":{},\"record_key\":{},"
              "\"preferred_arm\":{},\"normal_component_entry\":true,"
              "\"token_step\":{},\"key_proof_available\":{},\"key_equal\":{},"
              "\"local_bound_verified\":{},"
              "\"bound_plan_verified\":true,\"frames_executed\":{},"
              "\"full_original_outputs\":{},\"immutable_weights_and_tails\":{},"
              "\"losing_fusion_veto_verified\":{},\"ranking_samples_collected\":false",
              installed_selection.empty() ? Quote("canonical-query") : Quote(installed_selection),
              ranked != nullptr, ranked ? RecipeKeyJson(ranked->key) : "null", chosen, gdn,
              query_key_checked, query_key_checked, local_bound,
              installed_selection.empty() ? 0 : 2, !installed_selection.empty(),
              !installed_selection.empty(), ranked && chosen == 1 && !installed_selection.empty()));
    }

    std::string input_extents = "[";
    for (std::size_t i = 0; i < input_bytes.size(); ++i) {
      if (i) input_extents += ',';
      input_extents += std::to_string(input_bytes[i]);
    }
    input_extents += ']';
    const std::string split_details =
        gdn          ? std::format(
                           ",\"heads\":48,\"execution_phase\":\"verify\",\"prefix_calls_per_arm\":2,"
                           "\"body_calls\":[1,4],\"prepared_prefix_products\":2,"
                           "\"prepared_prefix_bytes_each\":{},\"prefix_compared_before_each_body\":true,"
                           "\"fresh_original_inputs_and_prefix_each_invocation\":true,"
                           "\"body_repetitions\":1,\"unsafe_repeat2_refusal_checked\":{},"
                           "\"dead_private_prefix_reuse\":[{},{}],"
                           "\"prefix_max_relative\":[{},{}],\"prefix_reference_limit\":1e-4,"
                           "\"library_workspace_bytes\":{},\"program_pool_bytes\":[{},{}],"
                           "\"private_pool_bytes\":[{},{}],"
                           "\"whole_descriptor_verified\":true,\"whole_bound_verified\":true,"
                           "\"bound_self_remainder_same\":true,"
                           "\"error_metric\":\"max_abs_error/max(1,max_abs_reference)\"",
                           bytes, repeat_refusal_checked, dead_private_prefix_reuse[0],
                           dead_private_prefix_reuse[1], prefix_max_relative[0], prefix_max_relative[1],
                           backend_->library_workspace_bytes(), program_pool[0], program_pool[1],
                           private_pool[0], private_pool[1])
        : hc_combine ? std::string(
                           ",\"streams\":4,\"whole_descriptor_verified\":true,"
                           "\"whole_bound_verified\":true,\"bound_self_remainder_same\":true,"
                           "\"error_metric\":\"abs(error)/(1+abs(reference))\"")
        : hc         ? std::string(
                           ",\"streams\":4,\"eps\":1e-6,\"whole_descriptor_verified\":true,"
                           "\"whole_bound_verified\":true,\"bound_self_remainder_same\":true,"
                           "\"error_metric\":\"abs(error)/(1+abs(reference))\"")
        : spatial
            ? std::format(
                  ",\"image_width\":37,\"image_height\":3,\"channels\":{},\"silu\":{},"
                  "\"eps\":1e-12,\"scale\":{},"
                  "\"whole_descriptor_verified\":true,\"whole_bound_verified\":true,"
                  "\"bound_self_remainder_same\":true,\"zero_and_epsilon_clamped_pixels\":true,"
                  "\"error_metric\":\"abs(error)/max(abs(reference),1e-3)\"",
                  width, silu, std::sqrt(static_cast<float>(width)))
        : split_fixture
            ? std::format(
                  ",\"split\":{},\"eps\":{},\"error_metric\":\"abs(error)/max(abs(reference),1)\"",
                  split, split_gate ? 0.0 : 1e-6)
            : std::string{};
    Receipt(std::format("{}_validated", event_prefix),
            std::format("\"width\":{},\"rows\":{},\"weight_encoding\":{},"
                        "\"input_encoding\":\"f32\",\"output_encoding\":\"f32\","
                        "\"input_domain\":{},\"dynamic_inputs\":{},\"input_bytes\":{},"
                        "\"original_outputs\":{},"
                        "\"discovery_pattern\":{},\"recipe_key\":{},\"recipe_key_digest\":{},"
                        "\"actual_calls\":{},\"frames\":2,\"recorded_nodes\":[{},{}],"
                        "\"recording_repetitions\":[1,1],\"max_relative\":[{},{}],"
                        "\"max_absolute\":[{},{}],\"reference_limits\":[{},{}],"
                        "\"weight_bytes_and_tails_unchanged\":true,"
                        "\"capture_measurement\":\"gpu_duration\",\"duration_checked\":true,"
                        "\"ranking_samples_collected\":false,\"timing_callback_validated\":true,"
                        "\"production_applicability_claim\":false{}",
                        width, rows,
                        Quote(weights.empty()                ? "none"
                              : (norm_add || spatial || gdn) ? "f32"
                                                             : "f16"),
                        Quote(gdn ? "token_features" : "component_ports"), dynamic_inputs.size(),
                        input_extents, original_outputs.size(), Quote(pattern), RecipeKeyJson(key),
                        Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key))), actual_calls,
                        captured_nodes[0], captured_nodes[1], max_relative[0], max_relative[1],
                        max_absolute[0], max_absolute[1], reference_limits[0], reference_limits[1],
                        split_details));
    if (timing) {
      int driver = 0, runtime = 0;
      Check(cudaDriverGetVersion(&driver));
      Check(cudaRuntimeGetVersion(&runtime));
      Receipt("ready_arithmetic_timing_environment",
              std::format("\"device\":{},\"driver\":{},\"runtime\":{},\"pattern\":{},"
                          "\"recorded_nodes\":[{},{}],\"recording_repetitions\":[1,1]",
                          Quote(backend_->device().device_model), driver, runtime, Quote(pattern),
                          captured_nodes[0], captured_nodes[1]));
      const auto protocol_text = std::format(
          "llmp-ready-arithmetic-in-capture-single-body-timing-v1;prep={};"
          "frames=2-original-ir-validated-before-samples;sample-input=second-original-frame;"
          "input=nan-poison-arena-then-restage-all-original-f32-component-ports;"
          "prefix=fresh-prepare-and-finish-before-every-body:complete-products-cpu-checked;"
          "body=captured-complete-local-region;body-replays=1;"
          "fixed-warmup-pairs=4:incumbent-then-alternative;"
          "bookends=alternative-nine-before-and-nine-after:median-of-nine;"
          "alternating-warmup-pairs=4:start-incumbent-then-alternate;"
          "alternating-sample-pairs=9:start-incumbent-then-alternate;"
          "events=two-external-event-nodes-inside-original-body-capture;"
          "all-original-cpu-outputs-and-immutable-weights-tails-and-inventory-check=every-arm;"
          "dead-private-prefix-reuse=allowed-after-consumption:fresh-prefix-every-invocation;"
          "ordered-retained-records=52;filter=none;bookend-drift=10pct;retry=none",
          llmp::base::ToHex(key.preparation_domain));
      const auto shape_class =
          std::format("ready.{}.width{}.rows{}.split{}.silu{}", pattern, width, rows, split, silu);
      MeasureReadyBody(
          key, protocol_text, shape_class,
          [&](en::ComparisonPairArm arm) {
            return run_frame(1, arm == en::ComparisonPairArm::kIncumbent ? 0U : 1U);
          },
          true, 9);
    }
  }

  void RequireNormalReadyContext(const en::ComparisonPair& pair,
                                 const co::OwnedComparisonRegionBoundary& region,
                                 const std::array<kn::ComparisonRegionRecipeSeal, 2>& descriptors,
                                 const ex::RecipeComparisonKey& key,
                                 const std::array<kn::StepProgram*, 2>& measured,
                                 const kn::StepBackend& table,
                                 std::shared_ptr<const kn::BackendOwner> owner,
                                 const pl::Policy& policy, const kn::Addresses& addresses,
                                 pl::PlanOptions options, bool token) {
    struct Observed {
      std::uint64_t identity;
      llmp::base::Sha256Digest graph;
      kn::ComparisonRegionStatus status;
      kn::ComparisonRegionRecipeSeal seal;
    };
    std::vector<Observed> observed;
    const kn::ProgramValidator observe =
        [&](const kn::LowerInput& lower,
            const kn::StepProgram& program) -> std::expected<void, std::string> {
      if (&lower.io.graph != &pair.source().graph || static_cast<bool>(lower.token) != token ||
          (token && (&lower.token->step != &pair.source() || &lower.token->shape != &pair.shape() ||
                     lower.token->context != 0)))
        return std::unexpected("normal ready validation changed its source or token context");
      const auto local = program.ProjectComparisonRegionCertificate(
          region, kn::ComparisonCertificatePhase::kDescriptor);
      // Only persistent values survive this synchronous callback; candidate
      // programs and any borrowed expansion spans may disappear after lookup.
      observed.push_back(
          {lower.plan.graph.identity(), lower.plan.graph.Digest(), local.status, local.seal});
      return {};
    };
    auto normal =
        token ? Must(en::BuildCumulativeRecipeProgram(co::GraphIO(pair.source()), pair.source(),
                                                      pair.shape(), 0, 0, table, policy, addresses,
                                                      options, observe, owner))
              : Must(en::BuildComponentProgram(co::GraphIO(pair.source()), table, policy, addresses,
                                               options, std::nullopt, observe, owner));
    if (observed.empty()) Fail("normal ready construction omitted its source observer");
    const auto* ranked = table.registry->FindRecipeComparison(key);
    if (ranked && ranked->kind != ex::RecipeComparisonKind::kLatency)
      Fail("normal Ready context cannot consume whole quality disposition");
    const std::size_t chosen =
        ranked && ex::ClassifyRecipeLatency(ranked->incumbent_ns, ranked->alternative_ns) ==
                      ex::RecipePreference::kAlternative
            ? 1
            : 0;
    const auto* expected = pair.plan(chosen == 0 ? en::ComparisonPairArm::kIncumbent
                                                 : en::ComparisonPairArm::kAlternative);
    if (!expected) Fail("normal ready comparison omitted its expected owned plan");
    const auto identity = normal->comparison_plan_identity();
    const Observed* final = &observed.back();
    if (identity) {
      if (identity->source_graph_identity != pair.source().graph.identity())
        Fail("normal ready owner changed its source identity");
      const auto found =
          std::ranges::find(observed, identity->planned_graph_identity, &Observed::identity);
      if (found == observed.end()) Fail("normal ready owner has no retained plan observation");
      final = &*found;
    }
    if (final->graph != expected->plan.graph.Digest())
      Fail("normal ready construction changed its complete chosen graph");
    const bool operation_in_filter = std::ranges::any_of(
        table.registry->ranks().recipe_comparisons,
        [&](const auto& row) { return row.key.discovery_operation == key.discovery_operation; });
    bool exact_key = false;
    for (const auto& candidate : observed) {
      if (candidate.graph != final->graph ||
          candidate.status != kn::ComparisonRegionStatus::kClosed)
        continue;
      auto actual = descriptors;
      actual[chosen] = candidate.seal;
      const auto actual_key = en::RecipeComparisonKeyForReadyCertificates(
          region, pair.inventory()->site.registered_result, pair.shape().phase,
          table.registry->SelectionDigest(), actual[0], actual[1]);
      if (actual_key && *actual_key == key) exact_key = true;
    }
    if ((operation_in_filter || ranked) && !exact_key)
      Fail("normal ready lookup did not reconstruct the measured exact key");
    // With no installed operation row, ordinary lowering legitimately carries
    // no local certificate. Its canonical graph/context/native plan still match;
    // exact hit closure is required after real rows are installed.
    Must(normal->Bind(*launch_));
    if (!normal->bound() || !measured[chosen]->bound() ||
        normal->bound()->identity() != measured[chosen]->bound()->identity())
      Fail("normal ready construction changed its complete native implementation identity");
    // Bind may stage parameters asynchronously; retire them before this temporary
    // normal owner is released, even though this preflight does not execute it.
    Finish();
    Receipt("ready_normal_context",
            std::format("\"token_step\":{},\"execution_phase\":{},\"exact_key_reconstructed\":{},"
                        "\"operation_in_installed_filter\":{},\"matched\":{},"
                        "\"native_identity_equal\":true,\"recipe_key_digest\":{}",
                        token ? "true" : "false", static_cast<unsigned>(pair.shape().phase),
                        exact_key ? "true" : "false", operation_in_filter ? "true" : "false",
                        ranked ? "true" : "false",
                        Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key)))));
  }

  void RunReadyNormGatePair(std::uint64_t dim, std::uint64_t heads, std::uint64_t rows,
                            bool timing) {
    // main() checks these products before Screen (and therefore before any
    // graph or device allocation) is constructed.
    const std::uint64_t width = dim * heads;
    const std::uint64_t cells = width * rows;
    const std::uint64_t bytes = cells * sizeof(float);
    if (rows > std::numeric_limits<std::uint32_t>::max())
      Fail("ready-norm-gate rows exceed the planner segment range");

    ir::Builder builder;
    builder.SetComponent(17);
    builder.SetOwner(0);
    const auto out_input =
        builder.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({dim, heads, rows, 1})),
                      "ready-norm-gate-output-input");
    const ir::WeightRef weight_ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
    const auto weight = builder.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({dim})), weight_ref,
                                       "ready-norm-gate.weight");
    builder.SetJoined();
    const auto gate_input = builder.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})),
                                          "ready-norm-gate-gate-input");
    const auto gate = builder.Reshape(gate_input, ir::ShapeOf({dim, heads, rows, 1}));
    const auto normalized = builder.RmsNorm(out_input, 1e-6F);
    const auto scaled = builder.Mul(normalized, weight);
    const auto sigmoid = builder.Unary(gate, ir::UnaryFn::kSigmoid);
    const auto gated = builder.Mul(scaled, sigmoid);
    const auto output = builder.Reshape(gated, ir::ShapeOf({width, rows}));
    builder.Output(output);
    auto source_graph = Must(std::move(builder).Finish());

    const auto out_bytes = ir::StorageBytes(source_graph.tensor(out_input));
    const auto gate_bytes = ir::StorageBytes(source_graph.tensor(gate_input));
    const auto weight_parts = llmp::format::PartsOf(source_graph.tensor(weight).format);
    if (!out_bytes || !gate_bytes || *out_bytes != bytes || *gate_bytes != bytes || !weight_parts ||
        weight_parts->size() != 1)
      Fail("ready-norm-gate graph has incomplete typed storage facts");
    const auto weight_part = weight_parts->parts().front();

    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    auto funded_graph = source_graph.Clone(funder);
    if (!funded_graph) Fail("ready-norm-gate source graph admission was refused");
    const auto source_identity = funded_graph->identity();
    co::StepGraph source{.graph = std::move(*funded_graph),
                         .inputs = {{.tensor = out_input,
                                     .kind = co::InputKind::kFeatures,
                                     .segment = 0,
                                     .component = 17,
                                     .what = "ready-norm-gate-output"},
                                    {.tensor = gate_input,
                                     .kind = co::InputKind::kFeatures,
                                     .component = 17,
                                     .what = "ready-norm-gate-gate"}},
                         .weights = {{.tensor = weight,
                                      .ref = weight_ref,
                                      .parts = {{"ready-norm-gate.weight", weight_part.role}}}},
                         .state = {},
                         .logits = output,
                         .greedy = {},
                         .features = {},
                         .saves = {},
                         .owner_products = {}};
    pl::StepShape shape{.stage = 0,
                        .phase = ex::Phase::kDecode,
                        .segments = {{.slot = 0,
                                      .rows = static_cast<std::uint32_t>(rows),
                                      .outputs = static_cast<std::uint32_t>(rows)}}};

    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    const auto* pattern = pl::FindPattern("gdn_norm_gate");
    if (!pattern) Fail("ready-norm-gate planner pattern is unavailable");
    const auto io = co::GraphIO(source);
    const auto activation_base = Allocate(kActivations);
    const auto weight_address = Allocate(weight_part.readable);
    kn::Addresses addresses{
        .weight =
            [weight_ref, weight_address, weight_part](const ir::WeightRef& ref,
                                                      llmp::format::PartRole part) {
              return ref == weight_ref && part == weight_part.role ? weight_address : 0;
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation_base,
        .activation_bytes = kActivations,
        .activation_alignment = 256,
        .weight_alignment =
            [weight_ref, weight_part](const ir::WeightRef& ref, llmp::format::PartRole part) {
              return ref == weight_ref && part == weight_part.role ? std::uint64_t{256} : 0;
            }};
    addresses.weight_range = [&](const ir::WeightRef& ref,
                                 llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
      if (ref != weight_ref || part != weight_part.role) return {};
      return kn::BoundedAddress{
          .base = weight_address, .bytes = weight_part.readable, .alignment = 256};
    };
    const pl::Policy stage_policy{.active_stage = 0};
    const auto policy = Must(pl::ForStep(stage_policy, shape.rows(), shape.segments.size()));
    std::vector<ir::TensorId> staged_inputs;
    for (const auto& input : source.inputs) staged_inputs.push_back(input.tensor);
    pl::PlanOptions options{.row_invariant = shape.row_invariant,
                            .capturable = true,
                            .on_stream = table.on_stream,
                            .written = table.written,
                            .within = table.within};
    options.staged = staged_inputs;
    options.alignment_staged = staged_inputs;
    options.shape = &shape;
    options = Must(en::PrepareComponentPlanOptions(io, table, addresses, options));
    auto source_bindings = Must(en::ComparisonSourceBindings::Create(funder, source, 0));
    {
      auto initial =
          Must(pl::Plan(source.graph.Clone(), *table.registry, table.device, policy, options));
      (void)Must(source_bindings.Resolve(initial, addresses, table.written));
    }
    const auto canonical_source = source_bindings.boundary_view();
    if (!canonical_source) Fail("normal ready source declarations are unavailable");
    const auto source_view = *canonical_source;
    std::vector<co::ComparisonExternalBinding> external(source_view.external.begin(),
                                                        source_view.external.end());
    const auto region_external = external;
    const std::array filter{ir::Fused(ir::FusedKind::kNormGate)};
    pl::ComparisonSiteSelector selector;
    std::string discovered_pattern;
    {
      auto discovered = pl::DiscoverComparisonPlan(source_view, filter, false, funder,
                                                   *table.registry, table.device, policy, options);
      if (!discovered) Fail(discovered.error().detail);
      const auto& discovery = discovered->discovery;
      const pl::ComparisonDiscoveryTicket* selected_ticket = nullptr;
      for (const auto& ticket : discovery.tickets()) {
        if (ticket.pattern != "gdn_norm_gate" || ticket.registered_operation != filter.front() ||
            !ticket.boundary_representable || ticket.member_offsets.size() < 2)
          continue;
        if (selected_ticket != nullptr)
          Fail("ready-norm-gate discovery found multiple candidate sites");
        selected_ticket = &ticket;
      }
      if (discovery.source_identity() != source_identity || selected_ticket == nullptr)
        Fail("ready-norm-gate source has no exact representable planner-discovered site");
      selector.source_graph_identity = discovery.source_identity();
      selector.kind = selected_ticket->kind;
      selector.pattern = std::string(selected_ticket->pattern);
      selector.version = selected_ticket->version;
      selector.member_owner_segments = selected_ticket->owner_segments;
      selector.boundary_ports = selected_ticket->boundary_ports;
      if (selected_ticket->member_offsets.front() != 0 ||
          selected_ticket->member_offsets.back() != selected_ticket->member_origins.size())
        Fail("ready-norm-gate discovery returned malformed member grouping");
      selector.member_origins.reserve(selected_ticket->member_offsets.size() - 1);
      for (std::size_t group = 0; group + 1 < selected_ticket->member_offsets.size(); ++group) {
        const auto first = selected_ticket->member_offsets[group];
        const auto last = selected_ticket->member_offsets[group + 1];
        if (first >= last || last > selected_ticket->member_origins.size())
          Fail("ready-norm-gate discovery returned an empty or invalid member group");
        selector.member_origins.emplace_back(selected_ticket->member_origins.begin() + first,
                                             selected_ticket->member_origins.begin() + last);
      }
      discovered_pattern = std::string(selected_ticket->pattern);
    }
    auto made =
        en::ComparisonPair::Create(funder, std::move(source), std::move(shape), std::move(external),
                                   0, 0, en::ComparisonInputDomain::kTokenStep);
    if (!made) Fail(made.error().detail);
    auto pair = std::move(*made);
    options.shape = &pair->shape();
    auto planned = pair->Plan(selector, *table.registry, table.device, policy, options);
    if (!planned) Fail(planned.error().detail);
    const auto* incumbent_plan = pair->plan(en::ComparisonPairArm::kIncumbent);
    const auto* alternative_plan = pair->plan(en::ComparisonPairArm::kAlternative);
    if (!incumbent_plan || !alternative_plan ||
        std::ranges::none_of(
            incumbent_plan->plan.applied,
            [&](const pl::Applied& applied) { return applied.pattern == discovered_pattern; }) ||
        std::ranges::any_of(alternative_plan->plan.applied, [&](const pl::Applied& applied) {
          return applied.pattern == discovered_pattern;
        }))
      Fail("ready-norm-gate incumbent did not retain the discovered fusion");

    auto actual_bindings = Must(en::ComparisonSourceBindings::Create(funder, pair->source(), 0));
    std::array<std::optional<en::ComparisonResolvedBindings>, 2> binding_storage;
    for (std::size_t arm = 0; arm < binding_storage.size(); ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      const auto* plan = pair->plan(which);
      if (!plan) Fail("normal ready plan is absent");
      binding_storage[arm].emplace(
          Must(actual_bindings.Resolve(plan->plan, addresses, table.written)));
    }
    const std::array<std::span<const kn::ComparisonBindingRange>, 2> binding_views{
        binding_storage[0]->ranges(), binding_storage[1]->ranges()};
    auto installed = pair->InstallBindings(binding_views);
    if (!installed) Fail(installed.error().detail);
    const co::ComparisonBoundaryView region_view{.io = co::GraphIO(pair->source()),
                                                 .state = pair->source().state,
                                                 .active_stage = 0,
                                                 .external = region_external};
    std::vector<std::uint32_t> region_offsets{0};
    std::vector<ir::NodeId> region_members;
    for (const auto& group : pair->inventory()->site.accepted_member_origins) {
      region_members.insert(region_members.end(), group.begin(), group.end());
      region_offsets.push_back(static_cast<std::uint32_t>(region_members.size()));
    }
    auto normal_region = Must(
        co::MakeComparisonRegionBoundary({region_view, region_offsets, region_members}, funder));
    std::array<kn::ComparisonRegionRecipeSeal, 2> ready_descriptors;
    std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
    std::array<kn::StepProgram*, 2> live_programs{};
    for (std::size_t arm = 0; arm < programs.size(); ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      auto lower = pair->LoweringInput(which, addresses);
      if (!lower) Fail(lower.error().detail);
      auto lowered = table.lower(*lower);
      if (!lowered) Fail(lowered.error());
      if ((*lowered)->scratch_bytes() > kActivations || (*lowered)->pool_bytes() > kPool)
        Fail("ready-norm-gate program exceeds Screen storage");
      const auto certificate = (*lowered)->ProjectComparisonRegionCertificate(
          normal_region, kn::ComparisonCertificatePhase::kDescriptor);
      if (certificate.status != kn::ComparisonRegionStatus::kClosed)
        Fail("normal ready measured arm lacks its descriptor applicability certificate");
      ready_descriptors[arm] = certificate.seal;
      live_programs[arm] = lowered->get();
      programs[arm] = std::move(*lowered);
    }
    auto installed_programs =
        pair->InstallDescriptorPrograms(std::move(programs[0]), std::move(programs[1]));
    if (!installed_programs) Fail(installed_programs.error().detail);
    phase = "ready_norm_gate_descriptor";
    auto described = en::DescribeRecipeComparison(
        *pair, *table.registry, en::ComparisonRecipePreparation::kStatelessReadyBoundaryOperands);
    if (!described || !*described)
      Fail("ready-norm-gate pair did not produce a closed ready-boundary recipe key");
    const auto key = **described;
    const auto* ranked = table.registry->FindRecipeComparison(key);
    const auto preference =
        ranked ? ex::ClassifyRecipeLatency(ranked->incumbent_ns, ranked->alternative_ns)
               : ex::RecipePreference::kNeutral;
    const auto preference_name =
        !ranked                                            ? std::string_view("unmeasured")
        : preference == ex::RecipePreference::kAlternative ? std::string_view("alternative")
        : preference == ex::RecipePreference::kIncumbent   ? std::string_view("incumbent")
                                                           : std::string_view("neutral");
    Receipt("ready_norm_gate_rank_lookup",
            std::format("\"recipe_key_digest\":{},\"registered_rows\":{},"
                        "\"matched\":{},\"shape_class\":{},\"preference\":{}",
                        Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key))),
                        table.registry->ranks().recipe_comparisons.size(),
                        ranked == nullptr ? "false" : "true",
                        ranked == nullptr ? "null" : Quote(ranked->shape_class),
                        Quote(preference_name)));
    auto bound = pair->BindPrograms(*launch_, *launch_);
    if (!bound) Fail(bound.error().detail);
    RequireNormalReadyContext(*pair, normal_region, ready_descriptors, key, live_programs, table,
                              owned.owner, policy, addresses, options, true);
    auto closed = en::CloseComparisonPair(std::move(pair));
    if (!closed) Fail(closed.error().detail);
    pair = std::move(*closed);
    if (!pair->closed()) Fail("ready-norm-gate pair did not close after descriptor binding");

    std::array<std::size_t, 2> captured_nodes{};
    std::array<std::size_t, 2> capture_repetitions{};
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
      auto captured = pair->CaptureRegion(which, *launch_, native_, 1);
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
      if (!captured) Fail("ready-norm-gate complete body capture failed");
      const auto inventory = pair->captured_region_inventory(which, *launch_);
      if (!inventory || inventory->repetitions != 1 || inventory->recorded_nodes == 0)
        Fail("ready-norm-gate native body inventory is empty or not a single replay");
      captured_nodes[arm] = inventory->recorded_nodes;
      capture_repetitions[arm] = inventory->repetitions;
    }

    std::vector<float> host_weight(dim);
    for (std::size_t i = 0; i < host_weight.size(); ++i)
      host_weight[i] = 0.75F + static_cast<float>(i % 9) * 0.03125F;
    if (weight_part.bytes != host_weight.size() * sizeof(float))
      Fail("ready-norm-gate reference weight extent differs from its actual part");
    Check(cudaMemcpyAsync(reinterpret_cast<void*>(weight_address), host_weight.data(),
                          static_cast<std::size_t>(weight_part.bytes), cudaMemcpyHostToDevice,
                          CudaStream()));
    Finish();
    auto reference = ref::Executor::Create(source_graph);
    if (!reference) Fail(reference.error().detail);
    std::array<std::vector<float>, 2> first_frame_outputs;
    std::array<std::vector<float>, 2> previous_frame_inputs;
    std::array<double, 2> arm_max_error{};
    const auto run_arm = [&](std::size_t arm_index, std::span<const float> host_out,
                             std::span<const float> host_gate, std::span<const float> expected,
                             bool timed, std::vector<float>& actual) -> std::uint64_t {
      const auto arm =
          arm_index == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      const auto slots = pair->input_slots(arm);
      if (slots.size() != 2) Fail("ready-norm-gate missing original inputs");
      std::array<bool, 2> seen{};
      Check(cudaMemsetAsync(reinterpret_cast<void*>(activation_base), 0xA5,
                            static_cast<std::size_t>(kActivations), CudaStream()));
      for (const auto& slot : slots) {
        const auto index = slot.tensor == out_input ? 0U : slot.tensor == gate_input ? 1U : 2U;
        if (index >= 2 || seen[index] || slot.kind != kn::InputKind::kFeatures ||
            slot.bytes != bytes)
          Fail("ready-norm-gate invalid original input slot");
        seen[index] = true;
        const auto payload = index == 0 ? host_out : host_gate;
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), payload.data(),
                              static_cast<std::size_t>(slot.bytes), cudaMemcpyHostToDevice,
                              CudaStream()));
      }
      Finish();
      auto prepared = pair->PrepareFrame(arm, *launch_, native_);
      if (!prepared) Fail("ready-norm-gate empty-prefix preparation failed");
      Finish();
      auto replayed = pair->ReplayCapturedRegion(arm, *launch_, native_);
      if (!replayed) Fail("ready-norm-gate captured body replay failed");
      Finish();
      const double ns = Must(pair->captured_region_duration_ns(arm, *launch_));
      if (!std::isfinite(ns) || ns < 0.5 ||
          ns >= static_cast<double>(std::numeric_limits<long long>::max()))
        Fail("ready-norm-gate invalid captured timing interval");
      bool found_output = false;
      for (const auto& port : pair->value_ports(arm)) {
        if (port.kind != kn::ComparisonRegionValueKind::kOutput || port.source_tensor != gated)
          continue;
        if (found_output || port.bytes != bytes || port.role == ir::Role::kWeight)
          Fail("ready-norm-gate output port is duplicate or malformed");
        found_output = true;
        if (expected.size() != cells) Fail("ready-norm-gate reference extent changed");
        actual.resize(cells);
        Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(port.address),
                         static_cast<std::size_t>(port.bytes), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < cells; ++i) {
          if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            Fail("ready-norm-gate CPU/GPU comparison produced a nonfinite value");
          const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
          arm_max_error[arm_index] = std::max(arm_max_error[arm_index], error);
          if (error > 1e-5 * (1.0 + std::abs(expected[i])))
            Fail("ready-norm-gate arm exceeded the existing normalized RMS reference bound");
        }
      }
      if (!found_output) Fail("ready-norm-gate captured body has no output value port");
      std::vector<float> weight_after(dim);
      Check(cudaMemcpy(weight_after.data(), reinterpret_cast<const void*>(weight_address),
                       static_cast<std::size_t>(weight_part.bytes), cudaMemcpyDeviceToHost));
      if (weight_after != host_weight)
        Fail("ready-norm-gate execution modified immutable gamma storage");
      const auto inventory = pair->captured_region_inventory(arm, *launch_);
      if (!inventory || inventory->repetitions != 1 ||
          inventory->recorded_nodes != captured_nodes[arm_index])
        Fail("ready-norm-gate captured inventory changed");
      if (!timed) return 0;
      return static_cast<std::uint64_t>(std::llround(ns));
    };
    std::vector<float> timing_expected;
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      std::vector<float> host_out(cells), host_gate(cells);
      for (std::size_t i = 0; i < cells; ++i) {
        const auto a = static_cast<float>(static_cast<int>((i * 13U) % 97U) - 48);
        const auto b = static_cast<float>(static_cast<int>((i * 29U + 7U) % 89U) - 44);
        host_out[i] = a * (frame == 0 ? 0.0234375F : 0.015625F) +
                      (frame == 0 ? 0.125F : static_cast<float>(i % 7U) * 0.01171875F);
        host_gate[i] = b * (frame == 0 ? 0.017578125F : 0.02734375F) -
                       (frame == 0 ? 0.0625F : static_cast<float>(i % 5U) * 0.015625F);
      }
      if (std::ranges::all_of(host_out, [](float value) { return value == 0.0F; }) ||
          std::ranges::all_of(host_gate, [](float value) { return value == 0.0F; }))
        Fail("ready-norm-gate source frames must contain nonzero operands");
      if (frame != 0 &&
          (host_out == previous_frame_inputs[0] || host_gate == previous_frame_inputs[1]))
        Fail("ready-norm-gate source operands did not change between frames");
      previous_frame_inputs[0] = host_out;
      previous_frame_inputs[1] = host_gate;
      auto ref_out = reference->BindValues(out_input, host_out);
      auto ref_gate = reference->BindValues(gate_input, host_gate);
      auto ref_weight = reference->BindValues(weight, host_weight);
      if (!ref_out || !ref_gate || !ref_weight)
        Fail("ready-norm-gate original-IR input binding failed");
      auto ref_run = reference->Run();
      if (!ref_run) Fail(ref_run.error().detail);
      auto expected = reference->Values(gated);
      if (!expected || expected->size() != cells)
        Fail("ready-norm-gate original-IR output extent is invalid");

      for (std::size_t arm_index = 0; arm_index < 2; ++arm_index) {
        std::vector<float> actual;
        (void)run_arm(arm_index, host_out, host_gate, *expected, false, actual);
        if (frame == 0) {
          first_frame_outputs[arm_index] = actual;
        } else {
          if (first_frame_outputs[arm_index].size() != actual.size())
            Fail("ready-norm-gate output extent changed between frames");
          double changed = 0;
          for (std::size_t i = 0; i < cells; ++i)
            changed = std::max(changed, std::abs(static_cast<double>(actual[i]) -
                                                 first_frame_outputs[arm_index][i]));
          if (changed <= 1e-5)
            Fail("ready-norm-gate changed source inputs did not change the body output");
        }
      }
      if (frame == 1) timing_expected.assign(expected->begin(), expected->end());
    }
    Finish();
    const auto apply_names = [](const pl::OwnedComparisonPlan* plan) {
      std::string names = "[";
      if (plan != nullptr) {
        bool first = true;
        for (const auto& applied : plan->plan.applied) {
          if (!first) names.push_back(',');
          first = false;
          names += Quote(applied.pattern);
        }
      }
      names.push_back(']');
      return names;
    };
    Receipt("ready_norm_gate_validated",
            std::format("\"shape\":{{\"dim\":{},\"heads\":{},\"rows\":{}}},"
                        "\"component\":17,\"discovery_pattern\":{},\"preparation_domain\":{},"
                        "\"recipe_key_digest\":{},\"recipe_key\":{},"
                        "\"incumbent_identity\":{},\"alternative_identity\":{},"
                        "\"applied_patterns\":[{},{}],\"recorded_nodes\":[{},{}],"
                        "\"recording_repetitions\":[{},{}],\"frames\":2,\"max_error\":[{},{}],"
                        "\"capture_measurement\":\"gpu_duration\",\"duration_checked\":true,"
                        "\"ranking_samples_collected\":false,"
                        "\"production_applicability_claim\":false",
                        dim, heads, rows, Quote(discovered_pattern),
                        Quote(llmp::base::ToHex(key.preparation_domain)),
                        Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key))),
                        RecipeKeyJson(key), Quote(llmp::base::ToHex(key.incumbent.physical_recipe)),
                        Quote(llmp::base::ToHex(key.alternative.physical_recipe)),
                        apply_names(incumbent_plan), apply_names(alternative_plan),
                        captured_nodes[0], captured_nodes[1], capture_repetitions[0],
                        capture_repetitions[1], arm_max_error[0], arm_max_error[1]));
    if (timing) {
      int driver = 0, runtime = 0;
      Check(cudaDriverGetVersion(&driver));
      Check(cudaRuntimeGetVersion(&runtime));
      Receipt("ready_norm_gate_timing_environment",
              std::format("\"device\":{},\"driver\":{},\"runtime\":{},"
                          "\"dim\":{},\"heads\":{},\"rows\":{},\"fn\":\"sigmoid\","
                          "\"eps\":1e-6,\"dtypes\":\"f32,f32,f32->f32\","
                          "\"recorded_nodes\":[{},{}],\"recording_repetitions\":[1,1]",
                          Quote(backend_->device().device_model), driver, runtime, dim, heads, rows,
                          captured_nodes[0], captured_nodes[1]));
      const auto protocol_text = std::format(
          "llmp-ready-norm-gate-in-capture-single-body-timing-v3;prep={};"
          "frames=2-original-ir-validated-before-samples;"
          "input=poison-arena-then-restage-both-original-f32-features-before-each-replay;"
          "prefix=prepare-once-finish-before-timed-body;body=captured-complete-local-region;"
          "body-replays=1;fixed-warmup-pairs=4:incumbent-then-alternative;"
          "bookends=alternative-nine-before-and-nine-after:median-of-nine;"
          "alternating-warmup-pairs=4:start-incumbent-then-alternate;"
          "alternating-sample-pairs=9:start-incumbent-then-alternate;"
          "events=two-external-event-nodes-inside-original-body-capture;"
          "weights-and-cpu-output-check=after-every-arm;"
          "ordered-retained-records=52;filter=none;bookend-drift=10pct;retry=none",
          llmp::base::ToHex(key.preparation_domain));
      const auto shape_class =
          std::format("ready.norm_gate.f32.dim{}.heads{}.rows{}.sigmoid.eps1e-6", dim, heads, rows);
      MeasureReadyBody(
          key, protocol_text, shape_class,
          [&](en::ComparisonPairArm arm) {
            std::vector<float> actual;
            return run_arm(arm == en::ComparisonPairArm::kIncumbent ? 0U : 1U,
                           previous_frame_inputs[0], previous_frame_inputs[1], timing_expected,
                           true, actual);
          },
          true, 9);
    }
  }

  void RunReadyNormRopePair(std::uint64_t width, std::uint64_t rows, bool timing,
                            ir::RopeMode mode = ir::RopeMode::kNeox,
                            std::span<const std::uint32_t> section_dims = {},
                            bool bf16_gamma = false, std::uint32_t heads = 1) {
    const bool sectioned = !section_dims.empty();
    const auto position_count = sectioned ? section_dims.size() : std::size_t{1};
    const std::string_view event_prefix =
        sectioned ? "ready_norm_rope_sections" : "ready_norm_rope";
    // main() checks these products before Screen (and therefore before any
    // graph or device allocation) is constructed.
    const std::uint64_t cells = width * heads * rows;
    const std::uint64_t bytes = cells * sizeof(float);
    if (rows > std::numeric_limits<std::uint32_t>::max())
      Fail("ready-norm-rope rows exceed the planner segment range");

    ir::Builder builder;
    builder.SetComponent(17);
    builder.SetOwner(0);
    const auto out_input = builder.Input(
        ir::Dense(Encoding::kF32, ir::ShapeOf({width, heads, rows})), "ready-norm-rope-input");
    const ir::WeightRef weight_ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
    const auto weight = builder.Weight(
        ir::Dense(bf16_gamma ? Encoding::kBF16 : Encoding::kF32, ir::ShapeOf({width})), weight_ref,
        "ready-norm-rope.weight");
    if (!sectioned) builder.SetJoined();
    std::vector<ir::TensorId> position_inputs;
    for (std::size_t i = 0; i < position_count; ++i)
      position_inputs.push_back(builder.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})),
                                              "ready-norm-rope-positions"));
    builder.SetOwner(0);
    const auto normalized = builder.RmsNorm(out_input, 1e-6F);
    const auto wide_weight = bf16_gamma ? builder.Convert(weight, Encoding::kF32) : weight;
    const auto scaled = builder.Mul(normalized, wide_weight);
    std::vector<ir::TensorId> original_body{normalized};
    if (bf16_gamma) original_body.push_back(wide_weight);
    original_body.push_back(scaled);
    auto gated = scaled;
    std::uint32_t offset = 0;
    for (std::size_t i = 0; i < position_count; ++i) {
      ir::RopeParams params;
      params.mode = mode;
      params.dims = sectioned ? section_dims[i] : static_cast<std::uint32_t>(width);
      params.offset = offset;
      params.original_context = 4096;
      gated = builder.Rope(gated, position_inputs[i], {}, params);
      original_body.push_back(gated);
      offset += params.dims;
    }
    const auto output = builder.Reshape(gated, ir::ShapeOf({width, heads * rows}));
    builder.Output(output);
    auto source_graph = Must(std::move(builder).Finish());

    const auto out_bytes = ir::StorageBytes(source_graph.tensor(out_input));
    const auto gate_bytes = ir::StorageBytes(source_graph.tensor(position_inputs.front()));
    const auto weight_parts = llmp::format::PartsOf(source_graph.tensor(weight).format);
    if (!out_bytes || !gate_bytes || *out_bytes != bytes ||
        *gate_bytes != rows * sizeof(std::int32_t) || !weight_parts || weight_parts->size() != 1)
      Fail("ready-norm-rope graph has incomplete typed storage facts");
    const auto weight_part = weight_parts->parts().front();
    if (sectioned &&
        (weight_part.bytes != width * (bf16_gamma ? sizeof(std::uint16_t) : sizeof(float)) ||
         weight_part.readable < weight_part.bytes || weight_part.readable > kActivations))
      Fail("norm-rope sections gamma backing differs from its typed storage extent");

    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    auto funded_graph = source_graph.Clone(funder);
    if (!funded_graph) Fail("ready-norm-rope source graph admission was refused");
    const auto source_identity = funded_graph->identity();
    co::StepGraph source{
        .graph = std::move(*funded_graph),
        .inputs = {{.tensor = out_input,
                    .kind = sectioned ? co::InputKind::kStream : co::InputKind::kFeatures,
                    .segment = 0,
                    .component = 17,
                    .what = "ready-norm-rope-output"}},
        .weights = {{.tensor = weight,
                     .ref = weight_ref,
                     .parts = {{"ready-norm-rope.weight", weight_part.role}}}},
        .state = {},
        .logits = output,
        .greedy = {},
        .features = {},
        .saves = {},
        .owner_products = {}};
    for (const auto tensor : position_inputs)
      source.inputs.push_back({.tensor = tensor,
                               .kind = sectioned ? co::InputKind::kSide : co::InputKind::kPositions,
                               .segment = sectioned ? std::uint8_t{0} : ir::kJoinedSegment,
                               .component = 17,
                               .what = "ready-norm-rope-gate"});
    pl::StepShape shape{.stage = 0,
                        .phase = sectioned ? ex::Phase::kPrefill : ex::Phase::kDecode,
                        .segments = {{.slot = 0,
                                      .rows = static_cast<std::uint32_t>(rows),
                                      .outputs = static_cast<std::uint32_t>(rows)}}};

    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    const std::string_view pattern_name =
        sectioned ? "rms_norm_mul_rope_sections" : "interleaved_norm_rope";
    const auto* pattern = pl::FindPattern(pattern_name);
    if (!pattern) Fail("ready-norm-rope planner pattern is unavailable");
    const auto io = co::GraphIO(source);
    const auto activation_base = Allocate(kActivations);
    const auto weight_address = Allocate(weight_part.readable);
    kn::Addresses addresses{
        .weight =
            [weight_ref, weight_address, weight_part](const ir::WeightRef& ref,
                                                      llmp::format::PartRole part) {
              return ref == weight_ref && part == weight_part.role ? weight_address : 0;
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation_base,
        .activation_bytes = kActivations,
        .activation_alignment = 256,
        .weight_alignment =
            [weight_ref, weight_part](const ir::WeightRef& ref, llmp::format::PartRole part) {
              return ref == weight_ref && part == weight_part.role ? std::uint64_t{256} : 0;
            }};
    addresses.weight_range = [&](const ir::WeightRef& ref,
                                 llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
      if (ref != weight_ref || part != weight_part.role) return {};
      return kn::BoundedAddress{
          .base = weight_address, .bytes = weight_part.readable, .alignment = 256};
    };
    const pl::Policy stage_policy{.active_stage = 0};
    const auto policy = sectioned
                            ? stage_policy
                            : Must(pl::ForStep(stage_policy, shape.rows(), shape.segments.size()));
    std::vector<ir::TensorId> staged_inputs;
    for (const auto& input : source.inputs) staged_inputs.push_back(input.tensor);
    pl::PlanOptions options{.row_invariant = shape.row_invariant,
                            .capturable = true,
                            .on_stream = table.on_stream,
                            .written = table.written,
                            .within = table.within};
    options.staged = staged_inputs;
    options.alignment_staged = staged_inputs;
    if (!sectioned) options.shape = &shape;
    options = Must(en::PrepareComponentPlanOptions(io, table, addresses, options));
    auto source_bindings =
        Must(sectioned ? en::ComparisonSourceBindings::Create(funder, io, 0)
                       : en::ComparisonSourceBindings::Create(funder, source, 0));
    {
      auto initial =
          Must(pl::Plan(source.graph.Clone(), *table.registry, table.device, policy, options));
      (void)Must(source_bindings.Resolve(initial, addresses, table.written));
    }
    const auto canonical_source = source_bindings.boundary_view();
    if (!canonical_source) Fail("normal ready source declarations are unavailable");
    const auto source_view = *canonical_source;
    std::vector<co::ComparisonExternalBinding> external(source_view.external.begin(),
                                                        source_view.external.end());
    const auto region_external = external;
    const std::array filter{ir::Fused(sectioned ? ir::FusedKind::kRmsNormMulRopeSections
                                                : ir::FusedKind::kRmsNormMulRope)};
    pl::ComparisonSiteSelector selector;
    std::string discovered_pattern;
    {
      auto discovered = pl::DiscoverComparisonPlan(source_view, filter, false, funder,
                                                   *table.registry, table.device, policy, options);
      if (!discovered) Fail(discovered.error().detail);
      const auto& discovery = discovered->discovery;
      const pl::ComparisonDiscoveryTicket* selected_ticket = nullptr;
      for (const auto& ticket : discovery.tickets()) {
        if (ticket.pattern != pattern_name || ticket.registered_operation != filter.front() ||
            !ticket.boundary_representable || ticket.member_offsets.size() < 2)
          continue;
        if (sectioned) {
          std::vector<ir::NodeId> expected_members;
          for (const auto tensor : original_body)
            expected_members.push_back(source_graph.producer(tensor));
          std::vector<std::uint32_t> offsets(expected_members.size() + 1);
          for (std::size_t i = 0; i < offsets.size(); ++i)
            offsets[i] = static_cast<std::uint32_t>(i);
          if (ticket.kind != pl::ComparisonSiteKind::kRewrite || ticket.member_offsets != offsets ||
              ticket.member_origins != expected_members)
            continue;
        }
        if (selected_ticket != nullptr)
          Fail("ready-norm-rope discovery found multiple candidate sites");
        selected_ticket = &ticket;
      }
      if (discovery.source_identity() != source_identity || selected_ticket == nullptr)
        Fail("ready-norm-rope source has no exact representable planner-discovered site");
      selector.source_graph_identity = discovery.source_identity();
      selector.kind = selected_ticket->kind;
      selector.pattern = std::string(selected_ticket->pattern);
      selector.version = selected_ticket->version;
      selector.member_owner_segments = selected_ticket->owner_segments;
      selector.boundary_ports = selected_ticket->boundary_ports;
      if (selected_ticket->member_offsets.front() != 0 ||
          selected_ticket->member_offsets.back() != selected_ticket->member_origins.size())
        Fail("ready-norm-rope discovery returned malformed member grouping");
      selector.member_origins.reserve(selected_ticket->member_offsets.size() - 1);
      for (std::size_t group = 0; group + 1 < selected_ticket->member_offsets.size(); ++group) {
        const auto first = selected_ticket->member_offsets[group];
        const auto last = selected_ticket->member_offsets[group + 1];
        if (first >= last || last > selected_ticket->member_origins.size())
          Fail("ready-norm-rope discovery returned an empty or invalid member group");
        selector.member_origins.emplace_back(selected_ticket->member_origins.begin() + first,
                                             selected_ticket->member_origins.begin() + last);
      }
      discovered_pattern = std::string(selected_ticket->pattern);
    }
    auto made = en::ComparisonPair::Create(funder, std::move(source), std::move(shape),
                                           std::move(external), 0, 0,
                                           sectioned ? en::ComparisonInputDomain::kComponentPorts
                                                     : en::ComparisonInputDomain::kTokenStep);
    if (!made) Fail(made.error().detail);
    auto pair = std::move(*made);
    if (!sectioned) options.shape = &pair->shape();
    auto planned = pair->Plan(selector, *table.registry, table.device, policy, options);
    if (!planned) Fail(planned.error().detail);
    const auto* incumbent_plan = pair->plan(en::ComparisonPairArm::kIncumbent);
    const auto* alternative_plan = pair->plan(en::ComparisonPairArm::kAlternative);
    if (!incumbent_plan || !alternative_plan ||
        std::ranges::none_of(
            incumbent_plan->plan.applied,
            [&](const pl::Applied& applied) { return applied.pattern == discovered_pattern; }) ||
        std::ranges::any_of(alternative_plan->plan.applied, [&](const pl::Applied& applied) {
          return applied.pattern == discovered_pattern;
        }))
      Fail("ready-norm-rope incumbent did not retain the discovered fusion");

    if (!alternative_plan->alternative_owner)
      Fail("ready-norm-rope alternative lost its exact-site owner");
    const auto& excluded = alternative_plan->alternative_owner->site.excluded_competing_rewrites;
    const auto* competing = pl::FindPattern("rms_norm_mul_rope");
    if (!sectioned && mode == ir::RopeMode::kNeox &&
        (!competing || std::ranges::none_of(excluded, [&](const auto& record) {
          return record.pattern == competing->name && record.version == competing->version &&
                 record.site_identity ==
                     alternative_plan->alternative_owner->site.exact_site_identity;
        })))
      Fail("ready-norm-rope alternative omitted the eligible competing rewrite");
    Receipt(std::format("{}_plan", event_prefix),
            std::format("\"incumbent_applied\":{},\"alternative_applied\":{},\"excluded_competing_"
                        "rewrites\":{}",
                        AppliedSites(incumbent_plan->plan), AppliedSites(alternative_plan->plan),
                        excluded.size()));

    auto actual_bindings = Must(
        sectioned ? en::ComparisonSourceBindings::Create(funder, co::GraphIO(pair->source()), 0)
                  : en::ComparisonSourceBindings::Create(funder, pair->source(), 0));
    std::array<std::optional<en::ComparisonResolvedBindings>, 2> binding_storage;
    for (std::size_t arm = 0; arm < binding_storage.size(); ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      const auto* plan = pair->plan(which);
      if (!plan) Fail("normal ready plan is absent");
      binding_storage[arm].emplace(
          Must(actual_bindings.Resolve(plan->plan, addresses, table.written)));
    }
    const std::array<std::span<const kn::ComparisonBindingRange>, 2> binding_views{
        binding_storage[0]->ranges(), binding_storage[1]->ranges()};
    auto installed = pair->InstallBindings(binding_views);
    if (!installed) Fail(installed.error().detail);
    const co::ComparisonBoundaryView region_view{.io = co::GraphIO(pair->source()),
                                                 .state = pair->source().state,
                                                 .active_stage = 0,
                                                 .external = region_external};
    std::vector<std::uint32_t> region_offsets{0};
    std::vector<ir::NodeId> region_members;
    for (const auto& group : pair->inventory()->site.accepted_member_origins) {
      region_members.insert(region_members.end(), group.begin(), group.end());
      region_offsets.push_back(static_cast<std::uint32_t>(region_members.size()));
    }
    auto normal_region = Must(
        co::MakeComparisonRegionBoundary({region_view, region_offsets, region_members}, funder));
    std::array<kn::ComparisonRegionRecipeSeal, 2> ready_descriptors;
    std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
    std::array<kn::StepProgram*, 2> live_programs{};
    for (std::size_t arm = 0; arm < programs.size(); ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      auto lower = pair->LoweringInput(which, addresses);
      if (!lower) Fail(lower.error().detail);
      auto lowered = table.lower(*lower);
      if (!lowered) Fail(lowered.error());
      if ((*lowered)->scratch_bytes() > kActivations || (*lowered)->pool_bytes() > kPool)
        Fail("ready-norm-rope program exceeds Screen storage");
      const auto certificate = (*lowered)->ProjectComparisonRegionCertificate(
          normal_region, kn::ComparisonCertificatePhase::kDescriptor);
      if (certificate.status != kn::ComparisonRegionStatus::kClosed)
        Fail("normal ready measured arm lacks its descriptor applicability certificate");
      ready_descriptors[arm] = certificate.seal;
      programs[arm] = std::move(*lowered);
      live_programs[arm] = programs[arm].get();
    }
    auto installed_programs =
        pair->InstallDescriptorPrograms(std::move(programs[0]), std::move(programs[1]));
    if (!installed_programs) Fail(installed_programs.error().detail);
    phase = sectioned ? "ready_norm_rope_sections_descriptor" : "ready_norm_rope_descriptor";
    auto described = en::DescribeRecipeComparison(
        *pair, *table.registry, en::ComparisonRecipePreparation::kStatelessReadyBoundaryOperands);
    if (!described || !*described)
      Fail("ready-norm-rope pair did not produce a closed ready-boundary recipe key");
    const auto key = **described;
    const auto* ranked = table.registry->FindRecipeComparison(key);
    const auto preference =
        ranked ? ex::ClassifyRecipeLatency(ranked->incumbent_ns, ranked->alternative_ns)
               : ex::RecipePreference::kNeutral;
    const auto preference_name =
        !ranked                                            ? std::string_view("unmeasured")
        : preference == ex::RecipePreference::kAlternative ? std::string_view("alternative")
        : preference == ex::RecipePreference::kIncumbent   ? std::string_view("incumbent")
                                                           : std::string_view("neutral");
    Receipt(std::format("{}_rank_lookup", event_prefix),
            std::format("\"recipe_key_digest\":{},\"registered_rows\":{},"
                        "\"matched\":{},\"shape_class\":{},\"preference\":{}",
                        Quote(llmp::base::ToHex(ex::RecipeComparisonKeyDigest(key))),
                        table.registry->ranks().recipe_comparisons.size(),
                        ranked == nullptr ? "false" : "true",
                        ranked == nullptr ? "null" : Quote(ranked->shape_class),
                        Quote(preference_name)));
    auto bound = pair->BindPrograms(*launch_, *launch_);
    if (!bound) Fail(bound.error().detail);
    RequireNormalReadyContext(*pair, normal_region, ready_descriptors, key, live_programs, table,
                              owned.owner, policy, addresses, options, !sectioned);
    std::array<std::size_t, 2> body_kernels{1, 2};
    std::string actual_calls = "[";
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto* calls = live_programs[arm]->bound();
      if (!calls) Fail("norm-rope bound call inventory is absent");
      std::vector<std::string_view> names;
      for (std::size_t i = 0; i < calls->size(); ++i) names.push_back(calls->at(i).name);
      if (arm) actual_calls += ',';
      actual_calls += '[';
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (i) actual_calls += ',';
        actual_calls += Quote(names[i]);
      }
      actual_calls += ']';
      if (!sectioned) continue;
      const std::vector<std::string_view> fused{"llmp.image.rms_norm_mul_rope_sections"};
      std::vector<std::string_view> child;
      if (bf16_gamma) child.push_back("ggml.convert");
      child.push_back("ggml.rms_norm_mul.fused");
      for (std::size_t i = 0; i < position_count; ++i)
        child.push_back(i == 0 && mode == ir::RopeMode::kNeox ? "ggml.rope.neox" : "ggml.rope.ext");
      if (names != (arm == 0 ? fused : child))
        Fail(
            std::format("norm-rope sections actual calls differ from their checked "
                        "own-decomposition (arm {}, calls {})",
                        arm, actual_calls));
      body_kernels[arm] = names.size();
    }
    actual_calls += ']';
    auto closed = en::CloseComparisonPair(std::move(pair));
    if (!closed) Fail(closed.error().detail);
    pair = std::move(*closed);
    if (!pair->closed()) Fail("ready-norm-rope pair did not close after descriptor binding");

    std::array<std::size_t, 2> captured_nodes{};
    std::array<std::size_t, 2> capture_repetitions{};
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
      auto captured = pair->CaptureRegion(which, *launch_, native_, 1);
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
      if (!captured) Fail("ready-norm-rope complete body capture failed");
      const auto inventory = pair->captured_region_inventory(which, *launch_);
      if (!inventory || inventory->repetitions != 1 || inventory->recorded_nodes == 0)
        Fail("ready-norm-rope native body inventory is empty or not a single replay");
      if (inventory->recorded_nodes != body_kernels[arm] + 2)
        Fail("ready-norm-rope expected complete body plus optional two event nodes");
      captured_nodes[arm] = inventory->recorded_nodes;
      capture_repetitions[arm] = inventory->repetitions;
    }

    std::vector<float> host_weight(width);
    std::vector<std::byte> weight_bytes(static_cast<std::size_t>(weight_part.readable));
    for (std::size_t i = 0; i < host_weight.size(); ++i) {
      host_weight[i] = sectioned ? 0.73F + static_cast<float>(i % 11) * 0.02713F
                                 : 0.75F + static_cast<float>(i % 9) * 0.03125F;
      if (bf16_gamma) {
        const auto bits = std::bit_cast<std::uint32_t>(host_weight[i]);
        const auto rounded =
            static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
        std::memcpy(weight_bytes.data() + i * sizeof(rounded), &rounded, sizeof(rounded));
        host_weight[i] = std::bit_cast<float>(static_cast<std::uint32_t>(rounded) << 16U);
      } else {
        std::memcpy(weight_bytes.data() + i * sizeof(float), &host_weight[i], sizeof(float));
      }
    }
    Check(cudaMemcpyAsync(
        reinterpret_cast<void*>(weight_address), weight_bytes.data(),
        sectioned ? weight_bytes.size() : static_cast<std::size_t>(weight_part.bytes),
        cudaMemcpyHostToDevice, CudaStream()));
    Finish();
    auto reference = ref::Executor::Create(source_graph);
    if (!reference) Fail(reference.error().detail);
    std::array<double, 2> max_error{}, max_rms_relative{}, max_worst{};
    std::array<std::vector<float>, 2> first_outputs;
    const auto run_arm = [&](std::size_t arm_index, std::span<const float> input,
                             std::span<const std::int32_t> positions,
                             std::span<const float> expected, bool timed,
                             std::vector<float>& actual) -> std::uint64_t {
      if (input.size() != cells || positions.size() != rows * position_count ||
          expected.size() != cells)
        Fail("ready-norm-rope frame extent changed");
      const auto arm =
          arm_index == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      const auto slots = pair->input_slots(arm);
      if (slots.size() != 1 + position_count) Fail("ready-norm-rope missing original input slots");
      std::vector<bool> seen(1 + position_count, false);
      Check(cudaMemsetAsync(reinterpret_cast<void*>(activation_base), sectioned ? 0xFF : 0xA5,
                            static_cast<std::size_t>(kActivations), CudaStream()));
      for (const auto& slot : slots) {
        const auto position = std::ranges::find(position_inputs, slot.tensor);
        const auto index = slot.tensor == out_input ? std::size_t{0}
                           : position == position_inputs.end()
                               ? seen.size()
                               : static_cast<std::size_t>(position - position_inputs.begin()) + 1;
        if (index >= seen.size() || seen[index] ||
            slot.kind != (sectioned    ? kn::InputKind::kStepBytes
                          : index == 0 ? kn::InputKind::kFeatures
                                       : kn::InputKind::kPositions) ||
            slot.bytes != (index == 0 ? bytes : rows * sizeof(std::int32_t)))
          Fail("ready-norm-rope slot violates original F32/I32 contract");
        seen[index] = true;
        const void* payload = index == 0
                                  ? static_cast<const void*>(input.data())
                                  : static_cast<const void*>(positions.data() + (index - 1) * rows);
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), payload,
                              static_cast<std::size_t>(slot.bytes), cudaMemcpyHostToDevice,
                              CudaStream()));
      }
      Finish();
      auto prepared = pair->PrepareFrame(arm, *launch_, native_);
      if (!prepared) Fail(prepared.error().detail);
      Finish();
      auto replayed = pair->ReplayCapturedRegion(arm, *launch_, native_);
      if (!replayed) Fail(replayed.error().detail);
      Finish();
      const double nanoseconds = Must(pair->captured_region_duration_ns(arm, *launch_));
      if (sectioned && (!std::isfinite(nanoseconds) || nanoseconds <= 0))
        Fail("norm-rope sections completed duration is invalid");
      bool found = false;
      for (const auto& port : pair->value_ports(arm)) {
        if (port.kind != kn::ComparisonRegionValueKind::kOutput || port.source_tensor != gated)
          continue;
        if (found || port.bytes != bytes ||
            (sectioned && port.role != source_graph.tensor(gated).role))
          Fail("ready-norm-rope invalid output port");
        found = true;
        actual.resize(cells);
        Check(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(port.address),
                         static_cast<std::size_t>(port.bytes), cudaMemcpyDeviceToHost));
        double squared_error = 0, squared_reference = 0, worst = 0;
        for (std::size_t i = 0; i < cells; ++i) {
          const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
          if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
              (!sectioned && error > 1e-5 * (1.0 + std::abs(expected[i]))))
            Fail("ready-norm-rope CPU reference bound exceeded");
          max_error[arm_index] = std::max(max_error[arm_index], error);
          squared_error += error * error;
          squared_reference += static_cast<double>(expected[i]) * expected[i];
          worst =
              std::max(worst, error / std::max(std::abs(static_cast<double>(expected[i])), 1.0));
        }
        if (sectioned) {
          const double rms_relative = std::sqrt(squared_error / squared_reference);
          if (!std::isfinite(rms_relative) || rms_relative > 1e-5 || worst > 1e-4)
            Fail("norm-rope sections exceeded the existing image RMS-relative or worst bound");
          max_rms_relative[arm_index] = std::max(max_rms_relative[arm_index], rms_relative);
          max_worst[arm_index] = std::max(max_worst[arm_index], worst);
        }
      }
      if (!found) Fail("ready-norm-rope missing output port");
      if (sectioned) {
        std::vector<std::byte> after(weight_bytes.size());
        Check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(weight_address), after.size(),
                         cudaMemcpyDeviceToHost));
        if (after != weight_bytes)
          Fail("norm-rope sections modified immutable gamma bytes or readable tails");
      } else {
        std::vector<float> after(width);
        Check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(weight_address),
                         static_cast<std::size_t>(weight_part.bytes), cudaMemcpyDeviceToHost));
        if (after != host_weight) Fail("ready-norm-rope gamma modified");
      }
      for (std::size_t i = 0; i < position_count; ++i) {
        const auto slot = std::ranges::find(slots, position_inputs[i], &kn::InputSlot::tensor);
        if (slot == slots.end()) Fail("ready-norm-rope positions slot disappeared");
        std::vector<std::int32_t> after(rows);
        Check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(slot->device_address),
                         after.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        if (!std::ranges::equal(after, positions.subspan(i * rows, rows)))
          Fail("ready-norm-rope execution modified positions");
      }
      const auto inventory = pair->captured_region_inventory(arm, *launch_);
      if (!inventory || inventory->repetitions != 1 ||
          inventory->recorded_nodes != captured_nodes[arm_index])
        Fail("ready-norm-rope captured inventory changed");
      if (!timed) return 0;
      const double ns = nanoseconds;
      if (!std::isfinite(ns) || ns < 0.5 ||
          ns >= static_cast<double>(std::numeric_limits<long long>::max()))
        Fail("ready-norm-rope invalid timing interval");
      return static_cast<std::uint64_t>(std::llround(ns));
    };
    std::vector<float> timing_input, timing_expected;
    std::vector<std::int32_t> timing_positions;
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      std::vector<float> input(cells);
      std::vector<std::int32_t> positions(rows * position_count);
      std::vector<std::int64_t> reference_positions(rows * position_count);
      for (std::size_t i = 0; i < cells; ++i)
        input[i] =
            static_cast<float>(static_cast<int>((i * 13U + frame * 17U) % 97U) - 48) * 0.0234375F +
            0.125F;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto section = i / rows;
        positions[i] = static_cast<std::int32_t>(3 + (i % rows) * 7 + section * 11 +
                                                 frame * (19 + section * 3));
        reference_positions[i] = positions[i];
      }
      auto r_input = reference->BindValues(out_input, input);
      auto r_weight = reference->BindValues(weight, host_weight);
      if (!r_input || !r_weight) Fail("ready-norm-rope reference binding failed");
      for (std::size_t i = 0; i < position_count; ++i)
        Must(reference->BindInts(position_inputs[i],
                                 std::span(reference_positions).subspan(i * rows, rows)));
      std::vector<std::vector<float>> prior_positions_outputs;
      if (frame != 0) {
        for (std::size_t i = 0; i < position_count; ++i) {
          auto old_positions = reference_positions;
          for (std::size_t row = 0; row < rows; ++row)
            old_positions[i * rows + row] -= static_cast<std::int64_t>(19 + i * 3);
          Must(reference->BindInts(position_inputs[i],
                                   std::span(old_positions).subspan(i * rows, rows)));
          Must(reference->Run());
          auto values = reference->Values(output);
          if (!values || values->size() != cells)
            Fail("norm-rope counterfactual output extent invalid");
          prior_positions_outputs.emplace_back(values->begin(), values->end());
          Must(reference->BindInts(position_inputs[i],
                                   std::span(reference_positions).subspan(i * rows, rows)));
        }
      }
      auto ran = reference->Run();
      if (!ran) Fail(ran.error().detail);
      auto expected = reference->Values(output);
      if (!expected || expected->size() != cells) Fail("ready-norm-rope reference extent invalid");
      if (frame != 0) {
        for (const auto& prior : prior_positions_outputs) {
          double effect = 0;
          for (std::size_t i = 0; i < cells; ++i)
            effect = std::max(effect, std::abs(static_cast<double>((*expected)[i]) - prior[i]));
          if (effect <= 1e-5)
            Fail("norm-rope an independent changed position input did not affect its CPU result");
        }
      }
      for (std::size_t arm_index = 0; arm_index < 2; ++arm_index) {
        std::vector<float> actual;
        (void)run_arm(arm_index, input, positions, *expected, false, actual);
        if (frame == 0)
          first_outputs[arm_index] = actual;
        else {
          if (first_outputs[arm_index].size() != actual.size())
            Fail("ready-norm-rope output extent changed between frames");
          double changed = 0;
          for (std::size_t i = 0; i < cells; ++i)
            changed = std::max(
                changed, std::abs(static_cast<double>(actual[i]) - first_outputs[arm_index][i]));
          if (changed <= 1e-5) Fail("ready-norm-rope changed frame did not change output");
        }
      }
      if (frame == 1 && timing) {
        timing_input = input;
        timing_positions = positions;
        timing_expected.assign(expected->begin(), expected->end());
      }
    }
    std::string section_details, section_dims_json;
    if (sectioned) {
      section_dims_json = "[";
      for (std::size_t i = 0; i < section_dims.size(); ++i) {
        if (i) section_dims_json += ',';
        section_dims_json += std::to_string(section_dims[i]);
      }
      section_dims_json += ']';
      section_details = std::format(
          ",\"heads\":{},\"section_dims\":{},\"gamma_encoding\":{},\"positions_inputs\":{},"
          "\"actual_calls\":{},\"max_rms_relative\":[{},{}],\"max_worst\":[{},{}],\"rms_relative_"
          "limit\":1e-5,\"worst_limit\":1e-4,\"worst_metric\":\"abs(error)/"
          "max(abs(reference),1)\",\"gamma_bytes_and_readable_tails_unchanged\":true",
          heads, section_dims_json, Quote(bf16_gamma ? "bf16" : "f32"), position_count,
          actual_calls, max_rms_relative[0], max_rms_relative[1], max_worst[0], max_worst[1]);
    }
    Receipt(std::format("{}_validated", event_prefix),
            std::format("\"width\":{},\"rows\":{},\"rope_mode\":{},\"eps\":1e-6,"
                        "\"positions_encoding\":\"i32\",\"frames\":2,\"recipe_key\":{},"
                        "\"recorded_nodes\":[{},{}],\"recording_repetitions\":[{},{}],"
                        "\"max_error\":[{},{}],\"capture_measurement\":\"gpu_duration\",\"duration_"
                        "checked\":true,"
                        "\"ranking_samples_collected\":false,\"production_applicability_"
                        "claim\":false{}",
                        width, rows, Quote(mode == ir::RopeMode::kNeox ? "neox" : "normal"),
                        RecipeKeyJson(key), captured_nodes[0], captured_nodes[1],
                        capture_repetitions[0], capture_repetitions[1], max_error[0], max_error[1],
                        section_details));
    if (timing) {
      int driver = 0, runtime = 0;
      Check(cudaDriverGetVersion(&driver));
      Check(cudaRuntimeGetVersion(&runtime));
      Receipt("ready_norm_rope_timing_environment",
              std::format("\"device\":{},\"driver\":{},\"runtime\":{},"
                          "\"width\":{},\"rows\":{},\"rope_mode\":{},"
                          "\"eps\":1e-6,\"dtypes\":{},\"frequency\":null,\"heads\":{},"
                          "\"section_dims\":{},\"positions_inputs\":{},"
                          "\"recorded_nodes\":[{},{}],\"recording_repetitions\":[1,1]",
                          Quote(backend_->device().device_model), driver, runtime, width, rows,
                          Quote(mode == ir::RopeMode::kNeox ? "neox" : "normal"),
                          Quote(bf16_gamma ? "f32,bf16,i32->f32" : "f32,f32,i32->f32"), heads,
                          sectioned ? section_dims_json : "[]", position_count, captured_nodes[0],
                          captured_nodes[1]));
      const auto legacy_protocol_text = std::format(
          "llmp-ready-neox-norm-rope-in-capture-single-body-timing-v2;prep={};"
          "frames=2-original-ir-validated-before-samples;frequency=none;"
          "input=poison-arena-then-restage-original-f32-features-and-i32-positions-before-each-"
          "replay;"
          "prefix=prepare-once-finish-before-timed-body;body=captured-complete-local-region;"
          "body-replays=1;native-nodes=3,4;body-nodes=1,2;"
          "fixed-warmup-pairs=4:incumbent-then-alternative;"
          "bookends=alternative-before-and-after;"
          "alternating-warmup-pairs=4:start-incumbent-then-alternate;"
          "alternating-sample-pairs=9:start-incumbent-then-alternate;"
          "events=two-external-event-nodes-inside-original-body-capture;gamma-positions-and-cpu-"
          "output-check=after-every-"
          "arm;"
          "bookend-drift=10pct;retry=none",
          llmp::base::ToHex(key.preparation_domain));
      const auto protocol_text =
          sectioned
              ? std::format(
                    "llmp-ready-norm-rope-sections-in-capture-single-body-timing-v1;prep={};"
                    "mode={};head-width={};heads={};section-dims={};gamma={};"
                    "frames=2-original-ir-validated-before-samples;sample-input=second-original-"
                    "frame;"
                    "input=nan-poison-arena-then-restage-original-f32-stream-and-all-independent-"
                    "i32-positions;"
                    "prefix=fresh-prepare-and-finish-before-every-body;body=captured-complete-"
                    "local-region;"
                    "body-replays=1;native-nodes={},{};body-nodes={},{};"
                    "fixed-warmup-pairs=4:incumbent-then-alternative;"
                    "bookends=alternative-nine-before-and-nine-after:median-of-nine;"
                    "alternating-warmup-pairs=4:start-incumbent-then-alternate;"
                    "alternating-sample-pairs=9:start-incumbent-then-alternate;"
                    "events=two-external-event-nodes-inside-original-body-capture;"
                    "gamma-bytes-tails-positions-and-complete-original-cpu-output-check=every-arm;"
                    "ordered-retained-records=52;filter=none;bookend-drift=10pct;retry=none",
                    llmp::base::ToHex(key.preparation_domain),
                    mode == ir::RopeMode::kNeox ? "neox" : "normal", width, heads,
                    section_dims_json, bf16_gamma ? "bf16" : "f32", captured_nodes[0],
                    captured_nodes[1], captured_nodes[0] - 2, captured_nodes[1] - 2)
              : legacy_protocol_text;
      const auto shape_class =
          sectioned
              ? std::format(
                    "ready.rms_norm_mul_rope_sections.{}.head{}.heads{}.rows{}.dims{}.{}.eps1e-6",
                    bf16_gamma ? "bf16gamma" : "f32gamma", width, heads, rows, section_dims_json,
                    mode == ir::RopeMode::kNeox ? "neox" : "normal")
              : std::format("ready.rms_norm_mul_rope.f32.width{}.rows{}.neox.eps1e-6", width, rows);
      MeasureReadyBody(
          key, protocol_text, shape_class,
          [&](en::ComparisonPairArm arm) {
            std::vector<float> actual;
            return run_arm(arm == en::ComparisonPairArm::kIncumbent ? 0U : 1U, timing_input,
                           timing_positions, timing_expected, true, actual);
          },
          true, sectioned ? 9U : 1U);
    }
  }

  std::vector<co::StepWeight> SourceWeights(const ir::Graph& graph,
                                            std::span<const co::StepWeight> fixture_weights) {
    std::vector<co::StepWeight> result;
    result.reserve(fixture_weights.size());
    for (const auto& weight : fixture_weights) {
      const auto found = std::ranges::find_if(graph.nodes(), [&](const ir::Node& node) {
        if (node.key.op != ir::Op::kWeight || node.nout != 1) return false;
        const auto* ref = std::get_if<ir::WeightRef>(&graph.tensor(node.out[0]).binding);
        return ref != nullptr && *ref == weight.ref;
      });
      if (found == graph.nodes().end()) Fail("expanded fixture lost an immutable weight leaf");
      result.push_back({.tensor = found->out[0], .ref = weight.ref, .parts = weight.parts});
    }
    return result;
  }

  std::vector<co::ComparisonExternalBinding> SourceExternalBindings(const en::GraphBuildIO& io) {
    std::vector<co::ComparisonExternalBinding> result;
    std::uint32_t allocation_class = 1;
    for (const auto input : io.graph.inputs()) {
      const auto bytes = ir::StorageBytes(io.graph.tensor(input));
      if (!bytes) Fail("fixture input has no finite storage extent");
      result.push_back({.original_root = input,
                        .allocation_class = allocation_class++,
                        .allocation_bytes = *bytes,
                        .bytes = *bytes,
                        .alignment = 16});
    }
    for (const auto& weight : io.weights) {
      const auto parts = llmp::format::PartsOf(io.graph.tensor(weight.tensor).format);
      if (!parts || parts->size() != weight.parts.size())
        Fail("fixture weight has incomplete declared physical parts");
      for (const auto& [name, role] : weight.parts) {
        (void)name;
        const auto part = std::ranges::find(parts->parts(), role, &llmp::format::PartExtent::role);
        if (part == parts->parts().end()) Fail("fixture weight names an absent format part");
        result.push_back({.original_root = weight.tensor,
                          .part = role,
                          .allocation_class = allocation_class++,
                          .allocation_bytes = part->readable,
                          .bytes = part->readable,
                          .alignment = 16});
      }
    }
    return result;
  }

  pl::ComparisonSiteSelector SiteSelector(const ir::Graph& graph, const Fixture& fixture) {
    const auto* pattern = pl::FindPattern(fixture.patterns[0]);
    if (!pattern || pattern->kind != pl::PatternKind::kRewrite)
      Fail("fixture names no registered rewrite pattern");
    std::vector<ir::NodeId> origins;
    std::size_t norm = 0, mul = 0, unary = 0, rope = 0, matmul = 0, glu = 0;
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      const auto& node = graph.node(ir::NodeId{i});
      switch (node.key.op) {
        case ir::Op::kRmsNorm:
          ++norm;
          if (fixture.kind == FixtureKind::kNormMul || fixture.kind == FixtureKind::kNormGate ||
              fixture.kind == FixtureKind::kNormRope)
            origins.emplace_back(i);
          break;
        case ir::Op::kMul:
          ++mul;
          if (fixture.kind == FixtureKind::kNormMul || fixture.kind == FixtureKind::kNormGate ||
              fixture.kind == FixtureKind::kNormRope)
            origins.emplace_back(i);
          break;
        case ir::Op::kUnary:
          ++unary;
          if (fixture.kind == FixtureKind::kNormGate) origins.emplace_back(i);
          break;
        case ir::Op::kRope:
          ++rope;
          if (fixture.kind == FixtureKind::kNormRope) origins.emplace_back(i);
          break;
        case ir::Op::kMatMul:
          ++matmul;
          if (fixture.kind == FixtureKind::kMatMulGlu) origins.emplace_back(i);
          break;
        case ir::Op::kGlu:
          ++glu;
          if (fixture.kind == FixtureKind::kMatMulGlu) origins.emplace_back(i);
          break;
        default:
          break;
      }
    }
    const bool exact =
        (fixture.kind == FixtureKind::kNormMul && norm == 1 && mul == 1 && origins.size() == 2) ||
        (fixture.kind == FixtureKind::kNormGate && norm == 1 && mul == 2 && unary == 1 &&
         origins.size() == 4) ||
        (fixture.kind == FixtureKind::kNormRope && norm == 1 && mul == 1 && rope == 1 &&
         origins.size() == 3) ||
        (fixture.kind == FixtureKind::kMatMulGlu && matmul == 2 && glu == 1 && origins.size() == 3);
    if (!exact) Fail("source graph does not match the fixture's exact primitive site");
    std::vector<pl::ComparisonOriginSet> member_origins;
    member_origins.reserve(origins.size());
    for (const auto origin : origins) member_origins.push_back({origin});
    return {.source_graph_identity = graph.identity(),
            .kind = pl::ComparisonSiteKind::kRewrite,
            .pattern = std::string(fixture.patterns[0]),
            .version = pattern->version,
            .member_origins = std::move(member_origins),
            .member_owner_segments = {},
            .boundary_ports = {}};
  }

  void AssertAcceptedSite(const pl::ComparisonInventory& inventory, const Fixture& fixture) {
    if (inventory.site.selector.pattern != fixture.patterns[0] ||
        inventory.site.accepted_member_origins != inventory.site.selector.member_origins ||
        inventory.site.accepted_member_origins.empty())
      Fail("planner accepted a different exact source site");
    const auto found = std::ranges::find(inventory.incumbent.plan.applied, fixture.patterns[0],
                                         &pl::Applied::pattern);
    if (found == inventory.incumbent.plan.applied.end())
      Fail("default-policy incumbent did not apply the requested fixture pattern");
    std::vector<ir::NodeId> ordered_origins;
    ordered_origins.reserve(inventory.site.accepted_member_origins.size());
    for (const auto& member : inventory.site.accepted_member_origins) {
      if (member.size() != 1) Fail("exact fixture site has a non-singleton source origin");
      ordered_origins.push_back(member.front());
    }
    if (found->replaced != ordered_origins)
      Fail("incumbent rewrite changed the accepted ordered primitive origins");
    if (fixture.kind == FixtureKind::kNormRope && found->pattern != "interleaved_norm_rope")
      Fail("plain NEOX fixture did not take the default accepted interleaved rewrite");
  }

  void Run(const Fixture& fixture, std::uint64_t width, std::uint64_t rows, bool validation_only) {
    auto fixture_graph = BuildFixture(fixture, width, rows);
    auto source_graph = Must(ir::Expand(fixture_graph.graph, false));
    const auto source_weights = SourceWeights(source_graph, fixture_graph.weights);
    const en::GraphBuildIO source_io{source_graph, source_graph.inputs(), source_graph.outputs(),
                                     source_weights};
    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    pl::Policy policy{};
    const pl::PlanOptions plan_options{.capturable = true, .within = table.within};
    auto externals = SourceExternalBindings(source_io);
    auto funder = std::make_shared<ScreenMetadataFunder>(metadata_used_, metadata_capacity_);
    const pl::ComparisonSiteSelector selector = SiteSelector(source_graph, fixture);
    const co::ComparisonBoundaryView boundary_view{source_io, {}, 0, externals};
    phase = "comparison_capture";
    auto inventory = Must(pl::BoundComparison(boundary_view, selector, funder, *table.registry,
                                              table.device, policy, plan_options));
    auto alternative = Must(pl::MakeAlternative(inventory));
    auto primitive_plan = Must(pl::PlanAlternative(
        inventory, std::move(alternative), *table.registry, table.device, policy, plan_options));
    AssertAcceptedSite(inventory, fixture);
    Receipt("comparison_plan_sites",
            std::format(
                "\"accepted_pattern\":{},\"accepted_source_nodes\":{},"
                "\"incumbent_applied\":{},\"alternative_applied\":{}",
                Quote(inventory.site.selector.pattern),
                [&] {
                  std::string nodes = "[";
                  for (std::size_t i = 0; i < inventory.site.accepted_member_origins.size(); ++i) {
                    if (i != 0) nodes += ',';
                    if (inventory.site.accepted_member_origins[i].size() != 1)
                      Fail("accepted source member is not a singleton");
                    nodes += std::to_string(inventory.site.accepted_member_origins[i].front().v);
                  }
                  return nodes + ']';
                }(),
                AppliedSites(inventory.incumbent.plan), AppliedSites(primitive_plan.plan)));
    if (!primitive_plan.alternative_owner)
      Fail("the comparison alternative lost its paid exact-site owner");
    std::string exclusions = "[";
    const auto& excluded = primitive_plan.alternative_owner->site.excluded_competing_rewrites;
    for (std::size_t i = 0; i < excluded.size(); ++i) {
      if (i != 0) exclusions += ',';
      exclusions += std::format("{{\"pattern\":{},\"version\":{},\"site_identity\":{}}}",
                                Quote(excluded[i].pattern), excluded[i].version,
                                Quote(llmp::base::ToHex(excluded[i].site_identity)));
    }
    exclusions += ']';
    Receipt("comparison_exclusions", std::format("\"records\":{}", exclusions));
    if (fixture.kind == FixtureKind::kNormRope) {
      const auto* competing = pl::FindPattern("rms_norm_mul_rope");
      if (competing == nullptr || std::ranges::none_of(excluded, [&](const auto& record) {
            return record.pattern == competing->name && record.version == competing->version &&
                   record.site_identity ==
                       primitive_plan.alternative_owner->site.exact_site_identity;
          }))
        Fail(
            "the exact-site alternative did not record the eligible overlapping norm/rope rewrite");
    }
    if (std::ranges::any_of(primitive_plan.plan.applied, [&](const pl::Applied& applied) {
          return applied.pattern == fixture.patterns[0];
        }))
      Fail("the exact primitive arm reapplied its excluded comparison site");
    phase = "fused_prepare";
    Prepare(arms_[0], inventory.incumbent.plan, source_io, inventory.source_boundary, externals,
            table, fixture, "incumbent");
    phase = "decomposition_prepare";
    Prepare(arms_[1], primitive_plan.plan, source_io, inventory.source_boundary, externals, table,
            fixture, "alternative");
    if (fixture.kind == FixtureKind::kNormRope && arms_[1].selected_fusion == fixture.operation) {
      const bool same_query = arms_[0].final_query && arms_[1].final_query &&
                              arms_[0].final_query == arms_[1].final_query;
      const bool same_recipe =
          arms_[0].physical_recipes.size() == 1 && arms_[1].physical_recipes.size() == 1 &&
          arms_[0].physical_recipes[0].name == arms_[1].physical_recipes[0].name &&
          arms_[0].physical_recipes[0].identity == arms_[1].physical_recipes[0].identity;
      Receipt("comparison_unavailable",
              std::format("\"reason\":{},\"target_key_equal\":true,"
                          "\"selected_query_equal\":{},\"physical_recipe_equal\":{},"
                          "\"incumbent_recipe\":{},\"alternative_recipe\":{}",
                          Quote("exact-site alternative selected the target fused operation again"),
                          same_query, same_recipe, Identities(arms_[0].physical_recipes),
                          Identities(arms_[1].physical_recipes)));
      return;
    }
    if (fixture.kind == FixtureKind::kMatMulGlu) {
      if (source_graph.inputs().size() != 1 ||
          source_graph.tensor(source_graph.inputs().front()).format.encoding != Encoding::kF32 ||
          fixture_graph.weights.size() != 2 ||
          std::ranges::any_of(fixture_graph.weights, [&](const auto& weight) {
            return fixture_graph.graph.tensor(weight.tensor).format.encoding != Encoding::kF16;
          }))
        Fail("product reference contract requires one F32 activation and two F16 weights");
      if (arms_[0].host_inputs.size() != arms_[1].host_inputs.size() ||
          arms_[0].host_weights.size() != arms_[1].host_weights.size())
        Fail("product arms staged different fixture input or weight counts");
      for (const auto& input : arms_[0].host_inputs) {
        const auto other = std::ranges::find(arms_[1].host_inputs, input.logical_index,
                                             &CapturedInput::logical_index);
        if (other == arms_[1].host_inputs.end() || input.bytes != other->bytes)
          Fail("product arms staged different activation bytes");
      }
      for (const auto& weight : arms_[0].host_weights) {
        const auto other =
            std::ranges::find(arms_[1].host_weights, weight.ref, &CapturedWeight::ref);
        if (other == arms_[1].host_weights.end() || weight.bytes != other->bytes)
          Fail("product arms staged different immutable weight bytes");
      }
    }
    phase = "chain_proof";
    if (arms_[0].implementations.size() != 1 || !arms_[0].selected_fusion ||
        *arms_[0].selected_fusion != fixture.operation || arms_[1].implementations.empty())
      Fail("screen did not select the requested fusion and a nonempty decomposition chain");
    if (arms_[0].physical_recipes.empty() || arms_[1].physical_recipes.empty())
      Fail("screen did not bind physical recipes for both compared arms");
    if (fixture.kind == FixtureKind::kNormMul &&
        (arms_[0].implementations[0].name != "ggml.rms_norm_mul.fused" ||
         arms_[1].implementations.size() != 2 ||
         arms_[1].implementations[0].name != "ggml.rms_norm" ||
         arms_[1].implementations[1].name != "ggml.mul" || arms_[0].physical_recipes.size() != 1 ||
         arms_[0].physical_recipes[0].name != "ggml.rms_norm_mul.fused" ||
         arms_[1].physical_recipes.size() != 1 ||
         arms_[1].physical_recipes[0].name != "ggml.rms_norm_mul.unfused"))
      Fail("RMSNormMul no longer selects its audited one-versus-two launch recipes");
    if (fixture.kind == FixtureKind::kMatMulGlu &&
        (arms_[0].physical_recipes.size() != 1 ||
         arms_[0].physical_recipes[0].name != "ggml.mul_mat_glu.mmvf_fused" ||
         arms_[1].physical_recipes.size() != 3 ||
         arms_[1].physical_recipes[0].name != "ggml.mul_mat.mmvf" ||
         arms_[1].physical_recipes[1].name != "ggml.mul_mat.mmvf" ||
         arms_[1].physical_recipes[2].name != "ggml.swiglu"))
      Fail("MatMulGlu product reference requires exact fused-MMVF and MMVF/MMVF/SwiGLU recipes");
    const auto query = arms_[0].final_query;
    if (!query) Fail("default-policy incumbent has no actual selected-query identity");
    const bool comparator_proven =
        arms_[0].comparison_recipe && arms_[1].comparison_recipe &&
        arms_[0].comparison_recipe->state == kn::ComparisonProofState::kBoundVerified &&
        arms_[1].comparison_recipe->state == kn::ComparisonProofState::kBoundVerified;
    if (!validation_only && !comparator_proven)
      Fail("timing refused: both exact-site arms need bound comparison proof");
    int driver = 0, runtime = 0;
    Check(cudaDriverGetVersion(&driver));
    Check(cudaRuntimeGetVersion(&runtime));
    Receipt(
        "identity",
        std::format(
            "\"query\":{},\"selection_environment\":{},\"device\":{},\"driver\":{},\"runtime\":{}",
            Quote(llmp::base::ToHex(*query)),
            Quote(llmp::base::ToHex(table.registry->SelectionDigest())),
            Quote(table.device.device_model), driver, runtime));
    phase = "correctness";

    double sum_error = 0, sum_reference = 0, max_error = 0;
    std::size_t worst_index = 0;
    if (arms_[0].values.size() != arms_[1].values.size()) Fail("output size mismatch");
    for (std::size_t i = 0; i < arms_[0].values.size(); ++i) {
      const auto a = arms_[0].values[i], b = arms_[1].values[i];
      if (!std::isfinite(a) || !std::isfinite(b)) Fail("nonfinite output");
      const double error = static_cast<double>(a) - b;
      sum_error += error * error;
      sum_reference += static_cast<double>(b) * b;
      if (std::abs(error) > max_error) {
        max_error = std::abs(error);
        worst_index = i;
      }
    }
    const double relative_rms = std::sqrt(sum_error / std::max(sum_reference, 1e-30));
    const auto pairwise = std::format(
        "\"relative_rms\":{},\"max_absolute\":{},\"worst_index\":{},"
        "\"fused_value\":{},\"decomposition_value\":{}",
        relative_rms, max_error, worst_index, arms_[0].values[worst_index],
        arms_[1].values[worst_index]);
    if (fixture.kind == FixtureKind::kMatMulGlu) {
      Receipt("pairwise_difference", pairwise);
      const auto fused_reference = ProductReferenceError(arms_[0], width, rows);
      const auto decomposition_reference = ProductReferenceError(arms_[1], width, rows);
      constexpr double kFusedReferenceLimit = 1e-4;
      constexpr double kDecompositionReferenceLimit = 2e-2;
      Receipt(
          "product_reference",
          std::format("\"contract\":{},\"fused_max_relative\":{},"
                      "\"fused_max_absolute\":{},\"fused_limit\":{},"
                      "\"fused_worst_index\":{},\"fused_expected\":{},\"fused_actual\":{},"
                      "\"decomposition_max_relative\":{},"
                      "\"decomposition_max_absolute\":{},\"decomposition_limit\":{},"
                      "\"decomposition_worst_index\":{},\"decomposition_expected\":{},"
                      "\"decomposition_actual\":{}",
                      Quote("ggml_ops_test.cc:1094-1103:f16_mmvf_swiglu"), fused_reference.maximum,
                      fused_reference.max_absolute, kFusedReferenceLimit, fused_reference.index,
                      fused_reference.expected, fused_reference.actual,
                      decomposition_reference.maximum, decomposition_reference.max_absolute,
                      kDecompositionReferenceLimit, decomposition_reference.index,
                      decomposition_reference.expected, decomposition_reference.actual));
      if (fused_reference.maximum >= kFusedReferenceLimit ||
          decomposition_reference.maximum >= kDecompositionReferenceLimit)
        Fail("MatMulGlu arm exceeds its source-backed reference bound");
    } else {
      Receipt("correctness", pairwise);
      if (relative_rms > 1e-6 || max_error > 2e-5) Fail("own-decomposition correctness bound");
    }
    if (validation_only) {
      phase = "validated";
      Receipt("validated",
              std::format(
                  "\"fixture\":{},\"comparator_status\":{},"
                  "\"fused_proof_state\":{},\"fused_missing\":{},"
                  "\"primitive_proof_state\":{},\"primitive_missing\":{},"
                  "\"fusion_recipe\":{},\"decomposition_recipe\":{},"
                  "\"fusion_nodes\":{},\"decomposition_nodes\":{},"
                  "\"fusion_activation_bytes\":{},\"decomposition_activation_bytes\":{},"
                  "\"fusion_pool_bytes\":{},\"decomposition_pool_bytes\":{},"
                  "\"fusion_private_pool_bytes\":{},\"decomposition_private_pool_bytes\":{},"
                  "\"replayed_outputs_equal\":true,\"timing_performed\":false",
                  Quote(fixture.name), Quote(comparator_proven ? "proven" : "incomplete"),
                  Quote(ProofStateName(arms_[0].comparison_recipe->state)),
                  Quote(ComparisonMissingName(arms_[0].comparison_recipe->missing)),
                  Quote(ProofStateName(arms_[1].comparison_recipe->state)),
                  Quote(ComparisonMissingName(arms_[1].comparison_recipe->missing)),
                  Identities(arms_[0].physical_recipes), Identities(arms_[1].physical_recipes),
                  arms_[0].recorded->nodes(), arms_[1].recorded->nodes(),
                  arms_[0].program->scratch_bytes(), arms_[1].program->scratch_bytes(),
                  arms_[0].program->pool_bytes(), arms_[1].program->pool_bytes(),
                  arms_[0].program->private_pool_bytes(), arms_[1].program->private_pool_bytes()));
      return;
    }
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&stop_));
    phase = "warmup";
    for (int sample = 0; sample < kWarmups; ++sample)
      for (std::size_t arm = 0; arm < arms_.size(); ++arm)
        Receipt("sample", std::format("\"pair\":{},\"order\":{},\"arm\":{},\"ns\":{}", sample, arm,
                                      arm, Time(arms_[arm])));
    phase = "bookend_before";
    const double first = Time(arms_[1]);
    Receipt("bookend", std::format("\"arm\":1,\"ns\":{}", first));
    std::string pairs;
    for (int sample = -kWarmups; sample < kSamples; ++sample) {
      phase = sample < 0 ? "paired_warmup" : "paired_timing";
      std::array<double, 2> times{};
      const auto first_arm = static_cast<std::size_t>((sample + kWarmups) % 2);
      for (std::size_t order = 0; order < 2; ++order) {
        const auto index = (first_arm + order) % 2;
        const auto ns = Time(arms_[index]);
        times[index] = ns;
        // Flush each completed arm immediately, including a partial pair if
        // the next arm or a later bookend refuses the run.
        Receipt("sample", std::format("\"pair\":{},\"order\":{},\"arm\":{},\"ns\":{}", sample,
                                      order, index, ns));
        if (sample >= 0) arms_[index].samples.push_back(ns);
      }
      if (sample >= 0) {
        if (!pairs.empty()) pairs += ',';
        pairs +=
            std::format("{{\"pair\":{},\"first_arm\":{},\"fused_ns\":{},\"decomposition_ns\":{}}}",
                        sample, first_arm, times[0], times[1]);
      }
    }
    phase = "bookend_after";
    const double last = Time(arms_[1]);
    Receipt("bookend", std::format("\"arm\":1,\"ns\":{}", last));
    if (std::max(first, last) > 1.1 * std::min(first, last)) Fail("bookend drift exceeds 10%");
    const auto median = [](const std::vector<double>& raw) {
      auto ordered = raw;
      std::ranges::sort(ordered);
      return static_cast<std::uint64_t>(std::llround(ordered.at(kSamples / 2)));
    };
    const auto fused = median(arms_[0].samples), primitive_ns = median(arms_[1].samples);
    phase = "accepted";
    std::string parts;
    for (const auto& implementation : arms_[1].implementations) {
      if (!parts.empty()) parts += ',';
      parts += Identity(implementation);
    }
    const auto label = std::format("{}.width{}.rows{}", fixture.name, width, rows);
    std::string_view preference;
    switch (ex::ClassifyFusionLatency(fused, primitive_ns)) {
      case ex::FusionPreference::kDecomposition:
        preference = "decomposition";
        break;
      case ex::FusionPreference::kFusion:
        preference = "fusion";
        break;
      case ex::FusionPreference::kNeutral:
        preference = "neutral";
        break;
      default:
        Fail("timings do not qualify for latency classification");
    }
    // The measurement record can be copied verbatim into data.json only after
    // its provenance placeholder is replaced by the reviewed report receipt.
    std::println(
        "{{\"measurement\":{{\"shape_class\":{},\"query\":{},\"selection_environment\":{},"
        "\"fusion\":{},"
        "\"decomposition\":[{}],\"fused_ns\":{},\"decomposition_ns\":{},"
        "\"samples\":{},\"provenance\":\"UNREVIEWED\"}},"
        "\"device\":{},\"class\":{},\"arch\":{},\"multiprocessors\":{},"
        "\"driver\":{},\"runtime\":{},\"relative_rms\":{},\"max_absolute\":{},"
        "\"bookend_before_ns\":{},\"bookend_after_ns\":{},\"decision\":{},"
        "\"pairs\":[{}],\"fusion_recipe\":{},\"decomposition_recipe\":{},"
        "\"fusion_nodes\":{},\"decomposition_nodes\":{},"
        "\"fusion_activation_bytes\":{},\"decomposition_activation_bytes\":{}}}",
        Quote(label), Quote(llmp::base::ToHex(*query)),
        Quote(llmp::base::ToHex(table.registry->SelectionDigest())),
        Identity(arms_[0].implementations[0]), parts, fused, primitive_ns, kSamples,
        Quote(table.device.device_model), Quote(table.device.device_class), table.device.arch,
        table.device.multiprocessors, driver, runtime, relative_rms, max_error, first, last,
        Quote(preference), pairs, Identities(arms_[0].physical_recipes),
        Identities(arms_[1].physical_recipes), arms_[0].recorded->nodes(),
        arms_[1].recorded->nodes(), arms_[0].program->scratch_bytes(),
        arms_[1].program->scratch_bytes());
    std::fflush(stdout);
  }

 private:
  kn::ComparisonRecipeSeal WholeDescriptor(const kn::StepProgram& program,
                                           const pl::OwnedComparisonPlan& owned,
                                           const ir::Graph& source,
                                           const co::OwnedComparisonBoundary& boundary,
                                           const ex::Registry& registry) {
    const auto seal = program.comparison_recipe();
    if (!seal) Fail("whole program omitted its comparison recipe");
    if (seal->state != kn::ComparisonProofState::kDescriptorVerified ||
        seal->missing != kn::ComparisonMissing::kNone ||
        seal->scope != kn::ComparisonProofScope::kWholeProgramNative)
      Fail(std::format("whole descriptor unproved: state {}, missing {}, scope {}",
                       ProofStateName(seal->state), ComparisonMissingName(seal->missing),
                       static_cast<unsigned>(seal->scope)));
    const auto& plan = owned.plan;
    if (seal->policy != plan.policy_digest || seal->registry != registry.SelectionDigest() ||
        seal->subject != source.Digest() || seal->boundary != boundary.digest().combined ||
        seal->planned_graph != plan.graph_digest)
      Fail("whole descriptor differs from its actual source/plan/boundary/registry");
    const auto identity = program.comparison_plan_identity();
    if (!identity || identity->source_graph_identity != source.identity() ||
        identity->planned_graph_identity != plan.graph.identity() ||
        owned.final_node_origins.source_graph_identity != source.identity() ||
        owned.final_node_origins.planned_graph_identity != plan.graph.identity())
      Fail("whole source or final-plan identity differs");
    return *seal;
  }

  std::uint64_t Allocate(std::uint64_t bytes) {
    void* allocation = nullptr;
    Check(cudaMalloc(&allocation, bytes));
    allocations_.push_back(allocation);
    return reinterpret_cast<std::uintptr_t>(allocation);
  }
  cudaStream_t CudaStream() const { return static_cast<cudaStream_t>(native_.handle); }
  void Finish() {
    Check(cudaStreamSynchronize(CudaStream()));
    const auto fence = Must(execution_->Record(stream_));
    Check(cudaStreamSynchronize(CudaStream()));
    if (Must(execution_->Query(fence)) != llmp::providers::FenceState::kComplete)
      Fail("completion unproven");
    Must(execution_->Release(fence));
    if (launch_) Must(launch_->DrainGeneratedInputs());
  }
  void Prepare(Arm& arm, const pl::StepPlan& plan, const en::GraphBuildIO& io,
               const co::OwnedComparisonBoundary& boundary,
               std::span<const co::ComparisonExternalBinding> external,
               const kn::StepBackend& table, const Fixture& fixture, std::string_view arm_name) {
    std::vector<std::pair<ir::WeightRef, std::uint64_t>> weight_addresses;
    for (const auto& weight : io.weights) {
      const auto parts = llmp::format::PartsOf(io.graph.tensor(weight.tensor).format);
      if (!parts || parts->size() != 1 || weight.parts.size() != 1)
        Fail("fixture weight parts are invalid");
      for (const auto& part : parts->parts()) {
        if (weight.parts[0].second != part.role) Fail("fixture omitted a physical weight part");
        weight_addresses.emplace_back(weight.ref, Allocate(part.readable));
      }
    }
    const auto activation_base = Allocate(kActivations);
    kn::Addresses addresses{
        .weight =
            [&](const ir::WeightRef& ref, llmp::format::PartRole) {
              const auto found = std::ranges::find_if(
                  weight_addresses, [&](const auto& entry) { return entry.first == ref; });
              return found == weight_addresses.end() ? std::uint64_t{0} : found->second;
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation_base,
        .activation_bytes = kActivations,
        .activation_alignment = 256};
    Receipt("planned_storage", std::format("\"activation_bytes\":{},\"activation_capacity\":{}",
                                           plan.placement.extent, kActivations));
    if (plan.placement.extent > kActivations) Fail("planned activations exceed fixture storage");
    std::string logical_steps;
    for (const auto& step : plan.steps) {
      if (!logical_steps.empty()) logical_steps += ',';
      logical_steps += Quote(table.registry->at(step.implementation).name);
    }
    Receipt("logical_selection",
            std::format("\"fixture\":{},\"steps\":[{}]", Quote(fixture.name), logical_steps));
    const pl::Policy selection_policy{};
    pl::SelectionView final_view(
        *table.registry, table.device, selection_policy, false, true, {}, {},
        {.base = addresses.activation_alignment, .offset = 256, .within = table.within});
    for (const auto& step : plan.steps) {
      if (step.nodes.size() != 1) Fail("fixture selected a multi-node logical step");
      arm.grouped_or_batched |= !step.group.empty() || !step.batch.empty();
      arm.implementations.push_back({table.registry->at(step.implementation).name,
                                     table.registry->identity(step.implementation)});
      const auto& selected = plan.graph.node(step.nodes[0]);
      if (selected.key.op == ir::Op::kFused) {
        if (arm.selected_fusion) Fail("screen selected more than one fused operation");
        arm.selected_fusion = selected.key;
        if (arm.final_query) Fail("screen selected more than one fused query");
        final_view.Inspect(plan.graph, selected, [&](const ex::CapabilityQuery& q) {
          arm.final_query = ex::FusionQueryKey(q);
        });
      }
    }
    if (!table.lower) Fail("backend omitted its canonical lowerer");
    std::vector<kn::ComparisonBindingRange> bindings;
    bindings.reserve(external.size());
    for (const auto& fact : external) {
      const auto boundary_root = std::ranges::find(boundary.roots(), fact.original_root,
                                                   &co::ComparisonBoundaryRoot::original_root);
      if (boundary_root == boundary.roots().end())
        Fail("owned boundary omitted an external fixture root");
      const auto owned_fact = std::ranges::find_if(boundary.external(), [&](const auto& entry) {
        return entry.canonical_root == boundary_root->canonical && entry.part == fact.part;
      });
      if (owned_fact == boundary.external().end())
        Fail("owned boundary omitted an external fixture part");
      std::uint64_t base = 0;
      if (!fact.part) {
        if (fact.original_root.v >= plan.tensors.size())
          Fail("comparison input lacks its exact source-to-plan map");
        const auto mapped = plan.tensors[fact.original_root.v];
        if (!mapped.valid()) Fail("comparison input was removed by the selected alternative");
        const auto root = plan.graph.root(mapped);
        const auto placed = std::ranges::find_if(plan.placement.offsets, [&](const auto& entry) {
          return plan.graph.root(entry.first) == root;
        });
        if (placed == plan.placement.offsets.end())
          Fail("fixture input has no physical placement in the exact arm");
        base = addresses.activations + placed->second;
      } else {
        const auto weight = std::ranges::find_if(io.weights, [&](const co::StepWeight& entry) {
          return entry.tensor == fact.original_root;
        });
        if (weight == io.weights.end())
          Fail("comparison boundary weight is absent from GraphBuildIO");
        const auto address = std::ranges::find_if(
            weight_addresses, [&](const auto& entry) { return entry.first == weight->ref; });
        if (address == weight_addresses.end()) Fail("weight binding was not allocated");
        base = address->second;
      }
      bindings.push_back({.original_root = fact.original_root,
                          .part = fact.part,
                          .allocation_class = owned_fact->canonical_allocation_class,
                          .allocation_bytes = owned_fact->allocation_bytes,
                          .offset = owned_fact->offset,
                          .bytes = owned_fact->bytes,
                          .alignment = owned_fact->alignment,
                          .base = base,
                          .resource_part_identity = owned_fact->resource_part_identity});
    }
    kn::ComparisonRequest comparison{
        .mode = kn::ComparisonRunMode::kFull, .bindings = bindings, .expected_boundary = &boundary};
    const kn::LowerInput lower_input(plan, io, addresses, std::nullopt, &comparison);
    arm.program = Must(table.lower(lower_input));
    Receipt("lowered_storage", std::format("\"activation_bytes\":{},\"activation_capacity\":{},"
                                           "\"pool_bytes\":{},\"pool_capacity\":{}",
                                           arm.program->scratch_bytes(), kActivations,
                                           arm.program->pool_bytes(), kPool));
    if (arm.program->scratch_bytes() > kActivations || arm.program->pool_bytes() > kPool)
      Fail("lowered storage exceeds fixture capacity");
    if (const auto seal = arm.program->comparison_recipe())
      Receipt("lowered_comparison_seal",
              std::format("\"fixture\":{},\"plan_arm\":{},\"state\":{},\"missing\":{}",
                          Quote(fixture.name), Quote(arm_name), Quote(ProofStateName(seal->state)),
                          Quote(ComparisonMissingName(seal->missing))));
    Must(arm.program->Bind(*launch_));
    arm.comparison_recipe = arm.program->comparison_recipe();
    if (!arm.comparison_recipe) Fail("comparison request produced no immutable recipe seal");
    const auto* bound = arm.program->bound();
    if (!bound) Fail("program omitted bound implementation inventory");
    for (std::size_t i = 0; i < bound->size(); ++i) {
      const auto index = table.registry->Find(bound->at(i).name);
      if (!index) Fail("bound implementation absent from registry");
      arm.physical_recipes.push_back({bound->at(i).name, table.registry->identity(*index)});
    }
    const auto generated_value = [](std::size_t i, std::uint16_t seed, bool small) {
      const auto v = static_cast<float>(static_cast<int>((i * 37 + seed * 13) % 251) - 125);
      return small ? v / 8192.0F : 1.0F + v / 512.0F;
    };
    for (const auto& weight : io.weights) {
      const auto& desc = io.graph.tensor(weight.tensor);
      const auto parts = llmp::format::PartsOf(desc.format);
      if (!parts || parts->size() != 1) Fail("fixture weight has an unsupported physical format");
      const auto found = std::ranges::find_if(
          weight_addresses, [&](const auto& entry) { return entry.first == weight.ref; });
      if (found == weight_addresses.end()) Fail("fixture weight address was not allocated");
      std::vector<std::byte> bytes((*parts)[0].readable);
      if (desc.format.encoding == Encoding::kF32) {
        if ((*parts)[0].bytes % sizeof(float)) Fail("misaligned F32 fixture weight");
        for (std::size_t i = 0; i < (*parts)[0].bytes / sizeof(float); ++i) {
          const float v =
              generated_value(i, weight.ref.role, fixture.kind == FixtureKind::kMatMulGlu);
          std::memcpy(bytes.data() + i * sizeof(float), &v, sizeof(v));
        }
      } else if (desc.format.encoding == Encoding::kF16) {
        if ((*parts)[0].bytes % sizeof(__half)) Fail("misaligned F16 fixture weight");
        for (std::size_t i = 0; i < (*parts)[0].bytes / sizeof(__half); ++i) {
          const auto v = __float2half_rn(generated_value(i, weight.ref.role, true));
          std::memcpy(bytes.data() + i * sizeof(v), &v, sizeof(v));
        }
      } else {
        Fail("fixture weight generator does not support this encoding");
      }
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(found->second), bytes.data(), bytes.size(),
                            cudaMemcpyHostToDevice, CudaStream()));
      Finish();
      arm.host_weights.push_back({.ref = weight.ref, .bytes = bytes});
    }
    for (const auto& slot : arm.program->inputs()) {
      const auto input = std::ranges::find(io.graph.inputs(), slot.tensor);
      if (input == io.graph.inputs().end()) Fail("program input is not a graph input");
      const auto input_index = static_cast<std::size_t>(input - io.graph.inputs().begin());
      const auto encoding = io.graph.tensor(slot.tensor).format.encoding;
      const bool is_activation = input_index == 0;
      const bool is_position = fixture.kind == FixtureKind::kNormRope && input_index == 1;
      const auto value = [&](std::size_t i) {
        const std::uint16_t semantic_seed = is_activation ? 17 : 29;
        const auto v =
            static_cast<float>(static_cast<int>((i * 37 + semantic_seed * 13) % 251) - 125);
        return is_activation ? v / 67.0F : 1.0F + v / 512.0F;
      };
      std::vector<std::byte> bytes(slot.bytes);
      if (encoding == Encoding::kF32) {
        if (slot.bytes % sizeof(float)) Fail("misaligned F32 fixture input");
        for (std::size_t i = 0; i < slot.bytes / sizeof(float); ++i) {
          const float v = is_position ? static_cast<float>(i % 32) / 32.0F : value(i);
          std::memcpy(bytes.data() + i * sizeof(float), &v, sizeof(v));
        }
      } else if (encoding == Encoding::kF16) {
        if (slot.bytes % sizeof(__half)) Fail("misaligned F16 fixture input");
        for (std::size_t i = 0; i < slot.bytes / sizeof(__half); ++i) {
          const auto v = __float2half_rn(value(i));
          std::memcpy(bytes.data() + i * sizeof(v), &v, sizeof(v));
        }
      } else if (encoding == Encoding::kI32) {
        if (slot.bytes % sizeof(std::int32_t)) Fail("misaligned I32 fixture input");
        for (std::size_t i = 0; i < slot.bytes / sizeof(std::int32_t); ++i) {
          const std::int32_t v = is_position ? static_cast<std::int32_t>(3 + i % 32)
                                             : static_cast<std::int32_t>(i % 32);
          std::memcpy(bytes.data() + i * sizeof(v), &v, sizeof(v));
        }
      } else {
        Fail("fixture generator does not support this input encoding");
      }
      arm.host_inputs.push_back({.logical_index = input_index, .bytes = bytes});
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), bytes.data(), slot.bytes,
                            cudaMemcpyHostToDevice, CudaStream()));
      Finish();
    }
    Must(arm.program->Run(*launch_, native_));
    Finish();
    if (arm.program->outputs().size() != 1) Fail("unexpected output count");
    const auto& output = arm.program->outputs()[0];
    for (const auto& slot : arm.program->inputs()) {
      const auto input_end = slot.device_address + slot.bytes;
      const auto output_end = output.device_address + output.bytes;
      if (slot.device_address < output_end && output.device_address < input_end)
        Fail("a kept fixture input aliases the output storage");
    }
    arm.values.resize(output.bytes / sizeof(float));
    Check(cudaMemcpy(arm.values.data(), reinterpret_cast<void*>(output.device_address),
                     output.bytes, cudaMemcpyDeviceToHost));
    const auto verify_inputs_unchanged = [&](std::string_view stage) {
      for (const auto& slot : arm.program->inputs()) {
        const auto input = std::ranges::find(io.graph.inputs(), slot.tensor);
        if (input == io.graph.inputs().end()) Fail("program input is not a graph input");
        const auto input_index = static_cast<std::size_t>(input - io.graph.inputs().begin());
        const auto host =
            std::ranges::find(arm.host_inputs, input_index, &CapturedInput::logical_index);
        if (host == arm.host_inputs.end() || host->bytes.size() != slot.bytes)
          Fail("fixture input immutability receipt lacks original bytes");
        std::vector<std::byte> actual(slot.bytes);
        Check(cudaMemcpy(actual.data(), reinterpret_cast<void*>(slot.device_address), slot.bytes,
                         cudaMemcpyDeviceToHost));
        if (actual != host->bytes) {
          Receipt("input_mutation",
                  std::format("\"stage\":{},\"logical_input\":{}", Quote(stage), input_index));
          Fail("device execution modified a kept fixture input");
        }
      }
    };
    verify_inputs_unchanged("eager");
    arm.recorded =
        Must(arm.program->Record(*launch_, native_, [&]() -> std::expected<void, kn::StepFailure> {
          for (int i = 0; i < kReplays; ++i) {
            auto ran = arm.program->Run(*launch_, native_);
            if (!ran) return ran;
          }
          return {};
        }));
    Finish();
    Must(arm.program->Run(*launch_, native_));
    Finish();
    std::vector<float> repeated(arm.values.size());
    Check(cudaMemcpy(repeated.data(), reinterpret_cast<void*>(output.device_address), output.bytes,
                     cudaMemcpyDeviceToHost));
    if (repeated != arm.values) Fail("eager repeat differs from first checked output");
    std::size_t launches_per_replay = 0;
    for (const auto& recipe : arm.physical_recipes)
      launches_per_replay += recipe.name == "ggml.rms_norm_mul.unfused" ? 2 : 1;
    const auto expected_nodes = kReplays * launches_per_replay;
    if ((!arm.grouped_or_batched && arm.recorded->nodes() != expected_nodes) ||
        (arm.grouped_or_batched && arm.recorded->nodes() < expected_nodes))
      Fail("captured graph does not contain the complete repeated chain");
    Check(cudaMemsetAsync(reinterpret_cast<void*>(output.device_address), 0, output.bytes,
                          CudaStream()));
    Must(launch_->Launch(native_, *arm.recorded));
    Finish();
    std::vector<float> replayed(arm.values.size());
    Check(cudaMemcpy(replayed.data(), reinterpret_cast<void*>(output.device_address), output.bytes,
                     cudaMemcpyDeviceToHost));
    verify_inputs_unchanged("captured_replay");
    if (replayed != arm.values) {
      for (std::size_t i = 0; i < replayed.size(); ++i) {
        if (replayed[i] == arm.values[i]) continue;
        Receipt(
            "replay_mismatch",
            std::format("\"implementation\":{},\"index\":{},\"eager\":{},\"replayed\":{}",
                        Quote(arm.implementations.front().name), i, arm.values[i], replayed[i]));
        break;
      }
      Fail("captured replay differs from checked eager output");
    }
  }
  double Time(const Arm& arm) {
    Check(cudaEventRecord(start_, CudaStream()));
    Must(launch_->Launch(native_, *arm.recorded));
    Check(cudaEventRecord(stop_, CudaStream()));
    Check(cudaEventSynchronize(stop_));
    float ms = 0;
    Check(cudaEventElapsedTime(&ms, start_, stop_));
    if (!std::isfinite(ms) || ms <= 0) Fail("invalid elapsed time");
    Finish();
    return static_cast<double>(ms) * 1e6 / kReplays;
  }
  std::uint64_t metadata_used_ = 0;
  std::uint64_t metadata_capacity_ = kMetadata;
  std::vector<void*> allocations_;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_{};
  llmp::providers::NativeStream native_{};
  std::unique_ptr<kn::DeviceBackend> backend_;
  std::unique_ptr<kn::StepLaunch> launch_;
  std::array<Arm, 2> arms_;
  cudaEvent_t start_ = nullptr, stop_ = nullptr;
};
}  // namespace

int main(int argc, char** argv) {
  // First screen by default; expanded classes are individually supervised.
  std::uint64_t width = 1024, rows = 1;
  bool whole_q8 = false, whole_q8_timing = false;
  bool whole_exl3 = false, whole_exl3_timing = false, whole_exl3_f32 = false, whole_image = false,
       image_preflight = false, whole_image_timing = false;
  std::string artifact_path;
  std::string_view installed_selection;
  std::uint64_t gate_dim = 128, gate_heads = 48, gate_rows = 1;
  std::string_view fixture = "rms_norm_mul";
  bool validation_only = false, inventory = false, ready_norm = false, ready_norm_rope = false;
  bool ready_norm_timing = false, ready_norm_gate = false, ready_norm_gate_timing = false;
  bool ready_norm_rope_timing = false, ready_matmul_glu = false, bf16_gamma = false;
  std::uint32_t rope_sections = 0;
  bool rope_sections_timing = false;
  bool ready_norm_add = false, ready_matmul_add = false;
  bool ready_arithmetic_timing = false, explicit_validation = false;
  bool ready_gdn_gates = false, ready_hc = false, ready_hc_combine = false;
  bool ready_spatial = false, spatial_silu = false;
  std::optional<ReadyArithmetic> ready_split;
  std::uint64_t split_row = 12;
  const auto parse = [](const char* text, std::uint64_t& value) {
    const std::string_view s(text);
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    return ec == std::errc{} && end == s.data() + s.size() && value > 0;
  };
  int positional = 0;
  bool invalid = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--whole-image-block-preflight" || arg == "--whole-image-block") {
      if (whole_image || positional != 0) invalid = true;
      whole_image = true;
      image_preflight = arg == "--whole-image-block-preflight";
      validation_only = true;
      width = 256;
      rows = 10;
    } else if (arg == "--whole-exl3-pair" || arg == "--whole-exl3-products" ||
               arg == "--whole-q8-glu") {
      if (whole_exl3 || positional != 0) invalid = true;
      whole_exl3 = true;
      whole_q8 = arg == "--whole-q8-glu";
      whole_exl3_f32 = whole_q8 || arg == "--whole-exl3-products";
      validation_only = true;
      width = 896;
      rows = 1;
    } else if (arg == "--validate-installed" && i + 1 < argc) {
      if (!installed_selection.empty()) invalid = true;
      installed_selection = argv[++i];
      if (installed_selection != "hit" && installed_selection != "miss") invalid = true;
    } else if (arg == "--whole-image-time") {
      if (whole_image_timing) invalid = true;
      whole_image_timing = true;
    } else if (arg == "--whole-exl3-time" || arg == "--whole-q8-time") {
      if (whole_exl3_timing) invalid = true;
      whole_exl3_timing = true;
      whole_q8_timing = arg == "--whole-q8-time";
    } else if (arg == "--artifact" && i + 1 < argc) {
      if (!artifact_path.empty()) invalid = true;
      artifact_path = argv[++i];
    } else if (arg == "--validate-only") {
      validation_only = true;
      explicit_validation = true;
    } else if (arg == "--ready-norm-rope-sections-time") {
      if (rope_sections_timing) invalid = true;
      rope_sections_timing = true;
    } else if (arg == "--ready-arithmetic-time") {
      if (ready_arithmetic_timing) invalid = true;
      ready_arithmetic_timing = true;
    } else if (arg == "--ready-gdn-gates") {
      if (positional != 0 || ready_gdn_gates || ready_hc || ready_hc_combine || ready_spatial ||
          ready_split || ready_matmul_glu || ready_norm_add || ready_matmul_add)
        invalid = true;
      ready_gdn_gates = true;
      validation_only = true;
      width = 64;
      rows = 1;
    } else if (arg == "--ready-hc-combine") {
      if (positional != 0 || ready_gdn_gates || ready_hc || ready_hc_combine || ready_spatial ||
          ready_split || ready_matmul_glu || ready_norm_add || ready_matmul_add)
        invalid = true;
      ready_hc_combine = true;
      validation_only = true;
      width = 4096;
      rows = 1;
    } else if (arg == "--ready-hc-post-norm") {
      if (positional != 0 || ready_gdn_gates || ready_hc || ready_hc_combine || ready_spatial ||
          ready_split || ready_matmul_glu || ready_norm_add || ready_matmul_add)
        invalid = true;
      ready_hc = true;
      validation_only = true;
      width = 4096;
      rows = 1;
    } else if (arg == "--ready-spatial-channel-norm" ||
               arg == "--ready-spatial-channel-norm-silu") {
      if (positional != 0 || ready_gdn_gates || ready_hc || ready_hc_combine || ready_spatial ||
          ready_split || ready_matmul_glu || ready_norm_add || ready_matmul_add)
        invalid = true;
      ready_spatial = true;
      spatial_silu = arg == "--ready-spatial-channel-norm-silu";
      validation_only = true;
      width = 7;
      rows = 111;
    } else if (arg == "--ready-split-gated-residual" || arg == "--ready-split-modulate" ||
               arg == "--ready-split-gated-residual-modulate") {
      if (positional != 0 || ready_gdn_gates || ready_hc || ready_hc_combine || ready_spatial ||
          ready_split || ready_matmul_glu || ready_norm_add || ready_matmul_add)
        invalid = true;
      ready_split = arg == "--ready-split-gated-residual" ? ReadyArithmetic::kSplitResidual
                    : arg == "--ready-split-modulate"     ? ReadyArithmetic::kSplitModulate
                                                          : ReadyArithmetic::kSplitJoined;
      validation_only = true;
      width = 320;
      rows = 37;
    } else if (arg == "--ready-matmul-glu" || arg == "--ready-norm-add" ||
               arg == "--ready-matmul-add") {
      if (positional != 0 || ready_gdn_gates || ready_hc || ready_hc_combine || ready_spatial ||
          ready_split || ready_matmul_glu || ready_norm_add || ready_matmul_add)
        invalid = true;
      ready_matmul_glu = arg == "--ready-matmul-glu";
      ready_norm_add = arg == "--ready-norm-add";
      ready_matmul_add = arg == "--ready-matmul-add";
      validation_only = true;
      width = 256;
    } else if (arg == "--ready-norm") {
      ready_norm = true;
      validation_only = true;
    } else if (arg == "--ready-norm-rope-normal" || arg == "--ready-norm-rope-sections-normal" ||
               arg == "--ready-norm-rope-sections-neox") {
      if (positional != 0 || ready_norm_rope) invalid = true;
      ready_norm_rope = true;
      validation_only = true;
      rope_sections = arg == "--ready-norm-rope-normal"            ? 1U
                      : arg == "--ready-norm-rope-sections-normal" ? 2U
                                                                   : 3U;
      width = rope_sections == 3 ? 128 : 64;
      rows = 5;
    } else if (arg == "--bf16-gamma") {
      bf16_gamma = true;
    } else if (arg == "--ready-norm-rope" || arg == "--ready-norm-rope-time") {
      if (positional != 0 || rope_sections != 0) invalid = true;
      ready_norm_rope = true;
      if (arg == "--ready-norm-rope-time")
        ready_norm_rope_timing = true;
      else
        validation_only = true;
    } else if (arg == "--ready-norm-time") {
      ready_norm = true;
      ready_norm_timing = true;
    } else if (arg == "--ready-norm-gate" || arg == "--ready-norm-gate-time") {
      if (positional != 0) invalid = true;
      ready_norm_gate = true;
      if (arg == "--ready-norm-gate-time")
        ready_norm_gate_timing = true;
      else
        validation_only = true;
    } else if (arg == "--inventory")
      inventory = true;
    else if (arg == "--fixture" && i + 1 < argc)
      fixture = argv[++i];
    else {
      ++positional;
      if (whole_exl3 || whole_image) {
        if (positional == 1)
          invalid |= !parse(argv[i], rows);
        else
          invalid = true;
      } else if (ready_hc || ready_hc_combine || ready_gdn_gates) {
        if (positional == 1)
          invalid |= !parse(argv[i], rows);
        else
          invalid = true;
      } else if (ready_spatial) {
        if (positional == 1)
          invalid |= !parse(argv[i], width);
        else
          invalid = true;
      } else if (ready_norm_gate) {
        if (positional == 1)
          invalid |= !parse(argv[i], gate_dim);
        else if (positional == 2)
          invalid |= !parse(argv[i], gate_heads);
        else if (positional == 3)
          invalid |= !parse(argv[i], gate_rows);
        else
          invalid = true;
      } else if (ready_split && positional == 3) {
        invalid |= !parse(argv[i], split_row);
      } else if (positional == 1) {
        invalid |= !parse(argv[i], width);
      } else if (positional == 2) {
        invalid |= !parse(argv[i], rows);
      } else {
        invalid = true;
      }
    }
  }
  const auto max_dimension = std::numeric_limits<std::uint32_t>::max();
  const bool gate_geometry_invalid =
      gate_dim == 0 || gate_heads == 0 || gate_rows == 0 || gate_dim > max_dimension ||
      gate_heads > max_dimension || gate_rows > max_dimension ||
      gate_heads > std::numeric_limits<std::uint64_t>::max() / gate_dim ||
      gate_rows > std::numeric_limits<std::uint64_t>::max() / (gate_dim * gate_heads) ||
      gate_dim * gate_heads * gate_rows > kActivations / sizeof(float);
  const bool matmul_geometry_invalid = width == 0 || rows == 0 || width > max_dimension ||
                                       rows > max_dimension ||
                                       width > kActivations / (2 * sizeof(__half)) ||
                                       width > kActivations / (2 * sizeof(__half) * width);
  const bool ready_arithmetic = ready_matmul_glu || ready_norm_add || ready_matmul_add ||
                                ready_split || ready_spatial || ready_hc || ready_hc_combine ||
                                ready_gdn_gates;
  const bool gates_geometry_invalid =
      ready_gdn_gates &&
      (width != 64 || rows > max_dimension || rows > kActivations / (sizeof(float) * 64));
  const bool hc_geometry_invalid =
      (ready_hc || ready_hc_combine) && (width != 4096 || (ready_hc && rows > 65535) ||
                                         rows > kActivations / (4 * sizeof(float) * 4096));
  const bool split_geometry_invalid =
      ready_split && (split_row == 0 || split_row >= rows || split_row > max_dimension);
  const bool section_geometry_invalid =
      rope_sections != 0 && (width != (rope_sections == 3 ? 128U : 64U) || rows > max_dimension ||
                             width > kActivations / (3 * sizeof(float)) ||
                             rows > kActivations / (3 * sizeof(float) * width));
  if (invalid ||
      (!installed_selection.empty() &&
       !((whole_image && !image_preflight && !whole_image_timing) ||
         (whole_exl3 && !whole_q8 && rows == 1 && !whole_exl3_timing) ||
         (ready_norm && !ready_norm_timing) ||
         (ready_arithmetic && !ready_gdn_gates && !ready_arithmetic_timing))) ||
      (whole_exl3 &&
       (width != 896 || (whole_exl3_f32 && rows != 1) ||
        (rows != 1 && rows != 145 && rows != 1024) ||
        (whole_q8 ? !artifact_path.empty() : artifact_path.empty()) || ready_arithmetic ||
        ready_norm || ready_norm_gate || ready_norm_rope || inventory || ready_arithmetic_timing ||
        rope_sections_timing || bf16_gamma || fixture != "rms_norm_mul")) ||
      (!whole_exl3 && !whole_image && !artifact_path.empty()) ||
      (rope_sections_timing && (rope_sections == 0 || explicit_validation)) ||
      (ready_arithmetic_timing && (!ready_arithmetic || explicit_validation)) ||
      (bf16_gamma && rope_sections == 0) || section_geometry_invalid || hc_geometry_invalid ||
      gates_geometry_invalid ||
      (ready_arithmetic &&
       (ready_norm || ready_norm_gate || ready_norm_rope || inventory ||
        fixture != "rms_norm_mul" || width > max_dimension || rows > max_dimension ||
        (!ready_norm_add && !ready_split && !ready_spatial && !ready_hc && !ready_hc_combine &&
         !ready_gdn_gates && matmul_geometry_invalid) ||
        split_geometry_invalid)) ||
      (inventory && (validation_only || ready_norm || ready_norm_gate ||
                     fixture != "rms_norm_mul" || positional != 0)) ||
      ((ready_norm_timing || ready_norm_gate_timing || ready_norm_rope_timing) &&
       validation_only) ||
      ((ready_norm || ready_norm_gate) && fixture != "rms_norm_mul") ||
      (ready_norm && ready_norm_gate) ||
      (ready_norm_rope &&
       (ready_norm || ready_norm_gate || inventory || fixture != "rms_norm_mul" || width % 2 != 0 ||
        width > std::numeric_limits<std::uint32_t>::max() ||
        rows > std::numeric_limits<std::uint32_t>::max())) ||
      (ready_norm_gate && gate_geometry_invalid) || (ready_norm_gate && positional > 3) ||
      (!inventory && !FindFixture(fixture)) ||
      // Bound each F32 input before graph construction. Division and short-circuit
      // evaluation keep the extent product overflow-safe; the actual planner
      // and lowerer below account for simultaneous live tensors and scratch.
      (!ready_norm_gate &&
       (width > kActivations / sizeof(float) || rows > kActivations / (sizeof(float) * width)))) {
    Receipt("refused", "\"reason\":\"invalid arguments or fixture storage bounds\"");
    std::println(stderr,
                 "usage: llmp_fusion_ranks [--whole-image-block-preflight --artifact COMPOSITION "
                 "[10|17|77|264] | --whole-image-block [--validate-installed hit|miss] "
                 "--artifact COMPOSITION [10|11(miss)|17|77|264] | "
                 "--whole-q8-glu [--whole-q8-time] [1] | --whole-exl3-pair [--whole-exl3-time] "
                 "[--validate-installed hit|miss (R1)] --artifact PATH [1|145|1024] | "
                 "--whole-exl3-products [--whole-exl3-time] --artifact PATH [1] | "
                 "--inventory | --validate-only | --ready-norm [--validate-installed hit|miss] | "
                 "--ready-norm-time | --ready-norm-rope | --ready-norm-rope-normal | "
                 "--ready-norm-rope-sections-normal | --ready-norm-rope-sections-neox | "
                 "--ready-norm-rope-time | "
                 "--ready-norm-gate | --ready-norm-gate-time | --ready-matmul-glu | "
                 "--ready-norm-add | --ready-matmul-add | --ready-gdn-gates | --ready-hc-post-norm "
                 "| --ready-hc-combine | "
                 "--ready-spatial-channel-norm | "
                 "--ready-spatial-channel-norm-silu | --ready-split-gated-residual | "
                 "--ready-split-modulate | --ready-split-gated-residual-modulate] "
                 "[--ready-arithmetic-time | --ready-norm-rope-sections-time | "
                 "--validate-installed hit|miss (ordinary arithmetic)] [--fixture name] "
                 "[--bf16-gamma] [width [rows [split]]] "
                 "(gate modes: [dim [heads "
                 "[rows]]]; HC/GDN-gates modes: [rows])");
    return 2;
  }
  if (!inventory && !ready_norm && !ready_norm_gate && !validation_only &&
      fixture != "rms_norm_mul") {
    Receipt("refused", "\"reason\":\"new fixtures are validation-only until U8 source freeze\"");
    return 2;
  }
  if (whole_image_timing && (!whole_image || image_preflight || whole_exl3 || explicit_validation))
    Fail("image timing requires its complete whole pair without preflight or validation-only");
  if (whole_exl3_timing && (!whole_exl3 || whole_q8 != whole_q8_timing || whole_image ||
                            rows != 1 || explicit_validation))
    Fail("EXL3 timing requires its genuine R1 pair without validation-only or image modes");
  if (whole_image) {
    if (whole_exl3 || artifact_path.empty() || positional > 1 || inventory || ready_arithmetic ||
        ready_norm || ready_norm_rope || ready_norm_gate || ready_arithmetic_timing ||
        ready_norm_timing || ready_norm_rope_timing || ready_norm_gate_timing ||
        rope_sections_timing || rope_sections != 0 || fixture != "rms_norm_mul" ||
        (rows != 10 && rows != 17 && rows != 77 && rows != 264 &&
         !(rows == 11 && installed_selection == "miss")))
      Fail("image preflight requires only its authentic composition path");
    Receipt("begin", std::format("\"mode\":\"whole_image_block\",\"preflight_only\":{},"
                                 "\"validation_only\":{},\"timing_requested\":{}",
                                 image_preflight, !whole_image_timing, whole_image_timing));
    {
      Screen screen(/*defer_launch=*/true, image_preflight ? kMetadata : 128 * kMiB);
      if (image_preflight)
        screen.PreflightWholeImageBlock(artifact_path, static_cast<std::uint32_t>(rows));
      else
        screen.RunWholeImageBlock(artifact_path, static_cast<std::uint32_t>(rows),
                                  whole_image_timing, installed_selection);
    }
    Receipt("complete", std::format("\"preflight_only\":{},\"teardown_verified\":true,\"validation_"
                                    "only\":{},\"ranking_samples_collected\":{}",
                                    image_preflight, !whole_image_timing, whole_image_timing));
    return 0;
  }
  if (whole_exl3) {
    Receipt(
        "begin",
        std::format("\"mode\":{},\"rows\":{},"
                    "\"input_encoding\":{},\"validation_only\":{},\"timing_requested\":{}",
                    Quote(whole_q8 ? "whole_q8_glu" : "whole_exl3_pair"), rows,
                    Quote(whole_exl3_f32 ? "f32" : "f16"), !whole_exl3_timing, whole_exl3_timing));
    {
      Screen screen(/*defer_launch=*/true);
      screen.RunWholeGluProducts(static_cast<std::uint32_t>(rows), artifact_path, whole_exl3_timing,
                                 whole_exl3_f32, whole_q8, installed_selection);
    }
    Receipt(
        "complete",
        std::format(
            "\"teardown_verified\":true,\"validation_only\":{},\"ranking_samples_collected\":{}",
            !whole_exl3_timing, whole_exl3_timing));
    return 0;
  }
  if (inventory) {
    Screen screen;
    screen.Inventory();
    return 0;
  }
  if (ready_arithmetic) {
    Receipt("begin",
            std::format("\"mode\":{},\"width\":{},\"rows\":{},"
                        "\"validation_only\":{},\"timing_requested\":{}",
                        Quote(ready_gdn_gates    ? "ready_gdn_gates"
                              : ready_hc_combine ? "ready_hc_combine"
                              : ready_hc         ? "ready_hc_post_norm"
                              : ready_spatial    ? "ready_spatial_channel_norm"
                              : ready_split    ? (*ready_split == ReadyArithmetic::kSplitResidual
                                                      ? "ready_split_gated_residual"
                                                  : *ready_split == ReadyArithmetic::kSplitModulate
                                                      ? "ready_split_modulate"
                                                      : "ready_split_gated_residual_modulate")
                              : ready_norm_add ? "ready_norm_add"
                              : ready_matmul_add ? "ready_matmul_add"
                                                 : "ready_matmul_glu"),
                        width, rows, !ready_arithmetic_timing, ready_arithmetic_timing));
  } else if (rope_sections != 0) {
    Receipt("begin", std::format("\"mode\":\"ready_norm_rope_sections\",\"width\":{},\"heads\":3,"
                                 "\"rows\":{},\"sections\":{},\"gamma_encoding\":{},\"validation_"
                                 "only\":{},\"timing_requested\":{}",
                                 width, rows, rope_sections, Quote(bf16_gamma ? "bf16" : "f32"),
                                 !rope_sections_timing, rope_sections_timing));
  } else if (ready_norm_rope) {
    Receipt("begin", std::format("\"mode\":\"ready_norm_rope\",\"width\":{},\"rows\":{},"
                                 "\"validation_only\":{},\"timing_requested\":{}",
                                 width, rows, validation_only, ready_norm_rope_timing));
  } else if (ready_norm_gate) {
    Receipt("begin",
            std::format("\"mode\":\"ready_norm_gate\",\"dim\":{},"
                        "\"heads\":{},\"rows\":{},\"validation_only\":{},"
                        "\"timing_requested\":{}",
                        gate_dim, gate_heads, gate_rows, validation_only, ready_norm_gate_timing));
  } else {
    Receipt("begin", std::format("\"width\":{},\"rows\":{},\"samples\":{},\"replays\":{},"
                                 "\"validation_only\":{},\"ready_norm\":{},"
                                 "\"ready_norm_timing\":{}",
                                 width, rows, kSamples, kReplays, validation_only, ready_norm,
                                 ready_norm_timing));
  }
  {
    Screen screen;
    if (ready_arithmetic && !installed_selection.empty()) {
      const auto selected_fixture = ready_gdn_gates    ? ReadyArithmetic::kGdnGates
                                    : ready_hc_combine ? ReadyArithmetic::kHcCombine
                                    : ready_hc         ? ReadyArithmetic::kHcPostNorm
                                    : ready_spatial    ? ReadyArithmetic::kSpatialNorm
                                    : ready_split      ? *ready_split
                                    : ready_norm_add   ? ReadyArithmetic::kNormAdd
                                    : ready_matmul_add ? ReadyArithmetic::kMatMulAdd
                                                       : ReadyArithmetic::kMatMulGlu;
      screen.ValidateInstalledArithmetic(selected_fixture, width, rows,
                                         static_cast<std::uint32_t>(split_row), spatial_silu,
                                         installed_selection);
    } else if (ready_gdn_gates)
      screen.RunReadyGdnGates(rows, ready_arithmetic_timing);
    else if (ready_hc_combine)
      screen.RunReadyHcCombine(rows, ready_arithmetic_timing);
    else if (ready_hc)
      screen.RunReadyHcPostNorm(rows, ready_arithmetic_timing);
    else if (ready_spatial)
      screen.RunReadySpatialNorm(width, spatial_silu, ready_arithmetic_timing);
    else if (ready_split)
      screen.RunReadySplit(*ready_split, width, rows, static_cast<std::uint32_t>(split_row),
                           ready_arithmetic_timing);
    else if (ready_matmul_glu)
      screen.RunReadyMatMulGlu(width, rows, ready_arithmetic_timing);
    else if (ready_norm_add)
      screen.RunReadyNormAdd(width, rows, ready_arithmetic_timing);
    else if (ready_matmul_add)
      screen.RunReadyMatMulAdd(width, rows, ready_arithmetic_timing);
    else if (ready_norm_timing)
      screen.TimeReadyNorm(width, rows);
    else if (ready_norm_gate)
      screen.RunReadyNormGate(gate_dim, gate_heads, gate_rows, ready_norm_gate_timing);
    else if (rope_sections != 0)
      screen.RunReadyNormRopeSections(width, rows, rope_sections, bf16_gamma, rope_sections_timing);
    else if (ready_norm_rope)
      screen.RunReadyNormRope(width, rows, ready_norm_rope_timing);
    else if (ready_norm)
      screen.RunReadyNorm(width, rows, installed_selection);
    else
      screen.RunFixture(fixture, width, rows, validation_only);
  }
  phase = "retired";
  Receipt("complete",
          std::format("\"validation_only\":{}",
                      validation_only && !ready_arithmetic_timing && !rope_sections_timing));
}
