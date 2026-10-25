// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Fixed-shape actual shared-GDN ordinary-program control. The isolated Ready
// producer's physical recipes do not transfer to these retained programs.
#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "base/json.h"
#include "components/input_values.h"
#include "components/step.h"
#include "engine/comparison_pair.h"
#include "engine/step_inputs.h"
#include "execution/fusion_ranks.h"
#include "ir/builder.h"
#include "kernels/common/device_backend.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"
#include "planner/alternative.h"
#include "planner/pattern.h"
#include "providers/cuda/cuda_device_execution.h"
#include "runtime/component_admission.h"
#include "spec/mappings/dsv4.h"
#include "spec/spec.h"
#include "spec/stored.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace ir = llmp::ir;
namespace kn = llmp::kernels;
namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace planner = llmp::planner;
namespace reference = llmp::kernels::reference;
namespace spec = llmp::spec;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kMetadata = 64 * kMiB;

std::string Quote(std::string_view text) {
  std::string result;
  llmp::base::json::AppendQuoted(text, result);
  return result;
}

[[noreturn]] void Fail(std::string_view message) {
  std::string reason;
  llmp::base::json::AppendQuoted(message, reason);
  std::println("{{\"event\":\"refused\",\"reason\":{}}}", reason);
  std::fflush(stdout);
  std::println(stderr, "actual GDN screen refused: {}", message);
  std::fflush(stderr);
  std::abort();  // Unknown completion keeps every owner until process teardown.
}
template <class T>
void Require(const T& value, std::string_view message) {
  if (!value) Fail(message);
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) Fail(cudaGetErrorString(error));
}
template <class T, class E>
T Must(std::expected<T, E> result) {
  if (!result) {
    if constexpr (requires { result.error().detail; })
      Fail(result.error().detail);
    else if constexpr (requires { std::string_view(result.error()); })
      Fail(result.error());
    else
      Fail("provider operation refused");
  }
  if constexpr (!std::is_void_v<T>) return std::move(*result);
}
class MetadataFunder final : public ir::StorageFunder {
 public:
  MetadataFunder(std::uint64_t& used, std::uint64_t capacity) : used_(used), capacity_(capacity) {}
  bool Charge(std::uint64_t bytes) const override {
    if (bytes > capacity_ - used_) return false;
    used_ += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const override {
    if (bytes > used_) Fail("metadata accounting underflow");
    used_ -= bytes;
  }

 private:
  std::uint64_t& used_;
  std::uint64_t capacity_;
};
struct CudaBuffer {
  void* pointer = nullptr;
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  ~CudaBuffer() {
    if (pointer) Check(cudaFree(pointer));
  }
  bool Allocate(std::uint64_t bytes) {
    if (pointer) Fail("buffer allocated twice");
    if (bytes != 0) Check(cudaMalloc(&pointer, bytes));
    return true;
  }
  std::uint64_t address() const { return reinterpret_cast<std::uintptr_t>(pointer); }
};
struct TokenStateSource {
  spec::ModelSpec model;
  std::vector<co::BoundTensor> tensors;
  llmp::model::StateLayoutDerived layout;
  llmp::planner::StepShape shape;
  co::StepGraph step;
  std::optional<spec::Source> admitted_source{};
  std::shared_ptr<const llmp::artifact::Artifact> artifact{};
};

// Applied::replaced is an ordered one-to-one source attribution. A later
// round's accepted member can have several origins; require its attributed
// source to belong to that exact member and refuse ambiguous records.
const planner::Applied* AppliedForSite(const planner::StepPlan& plan, std::string_view pattern,
                                       std::span<const planner::ComparisonOriginSet> groups) {
  const planner::Applied* found = nullptr;
  for (const auto& applied : plan.applied) {
    if (applied.pattern != pattern || applied.replaced.size() != groups.size()) continue;
    bool same = true;
    for (std::size_t i = 0; i < groups.size(); ++i)
      same &= std::ranges::find(groups[i], applied.replaced[i]) != groups[i].end();
    if (!same) continue;
    if (found) return nullptr;
    found = &applied;
  }
  return found;
}

spec::Param SpecParam(std::string name, spec::ParamValue value) {
  return {.name = std::move(name), .value = std::move(value)};
}

spec::WeightBinding SpecWeight(std::string role, std::string tensor) {
  return {.role = std::move(role), .tensor = std::move(tensor)};
}

llmp::format::TensorFormat F32Format(std::initializer_list<std::uint64_t> dimensions) {
  return {.encoding = llmp::format::Encoding::kF32, .shape = ir::ShapeOf(dimensions)};
}

std::expected<TokenStateSource, std::string> MakeSmallGdnStateSource() {
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {SpecWeight("table", "token_embd")};
  spec::ComponentRef gdn{.kind = "attention.gdn"};
  gdn.params = {SpecParam("placement", std::string("pre")),
                SpecParam("eps", 1.0e-6),
                SpecParam("head_dim", std::int64_t{128}),
                SpecParam("k_heads", std::int64_t{16}),
                SpecParam("v_heads", std::int64_t{48}),
                SpecParam("conv", std::int64_t{4}),
                SpecParam("output_gate", std::string("sigmoid")),
                SpecParam("a_log", false),
                SpecParam("norm_zero_centered", false)};
  std::vector<co::BoundTensor> tensors;
  const auto add = [&](std::string role, std::initializer_list<std::uint64_t> dims) {
    const auto name = "gdn_" + role;
    gdn.weights.push_back(SpecWeight(std::move(role), name));
    tensors.push_back({.name = name, .format = F32Format(dims)});
  };
  add("qkv", {64, 10240});
  add("gate", {64, 6144});
  add("beta", {64, 48});
  add("alpha", {64, 48});
  add("dt_bias", {48});
  add("a", {48});
  add("conv", {4, 10240});
  add("norm", {128});
  add("output", {6144, 64});
  add("input_norm", {64});
  tensors.push_back({.name = "token_embd", .format = F32Format({64, 8})});
  tensors.push_back({.name = "output_head", .format = F32Format({64, 8})});
  spec::Stage stage{.name = "target", .context = 256, .embedding = std::move(embedding)};
  stage.stream.width = 64;
  stage.layers = {{.components = {std::move(gdn)}}};
  stage.folds = {"ssm.a_neg_exp", "gdn.value_heads_tiled"};
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {SpecWeight("weight", "output_head")};
  spec::ModelSpec model{.stages = {std::move(stage)}};
  spec::Canonicalize(model);
  auto valid = co::ValidateSpec(model, tensors);
  if (!valid) return std::unexpected(valid.error());
  auto layout = co::StageLayout(model, 0, 256, 1);
  if (!layout) return std::unexpected(layout.error());
  planner::StepShape shape{.stage = 0,
                           .phase = planner::Phase::kDecode,
                           .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  auto step = co::BuildStep(model, 0, shape, {.tensors = tensors, .state = &*layout});
  if (!step) return std::unexpected(step.error());
  return TokenStateSource{.model = std::move(model),
                          .tensors = std::move(tensors),
                          .layout = std::move(*layout),
                          .shape = std::move(shape),
                          .step = std::move(*step)};
}

// Existing component-reference contracts: Qwen's resident separate/scaled
// experts with their gated shared expert, and the parallel stacked Q8 case.
std::expected<TokenStateSource, std::string> MakeResidentMoeSource(bool qwen, std::uint32_t rows,
                                                                   bool f16_experts) {
  spec::ComponentRef component{.kind = "ffn.moe"};
  component.params = {SpecParam("placement", std::string(qwen ? "none" : "parallel")),
                      SpecParam("eps", 1.0e-6),
                      SpecParam("activation", std::string(qwen ? "silu" : "gelu_tanh")),
                      SpecParam("hidden", std::int64_t{qwen ? 24 : 64}),
                      SpecParam("experts", std::int64_t{8}),
                      SpecParam("used", std::int64_t{3}),
                      SpecParam("expert_hidden", std::int64_t{qwen ? 16 : 32}),
                      SpecParam("router", std::string("softmax_topk")),
                      SpecParam("router_input", std::string(qwen ? "stream" : "rms_scaled")),
                      SpecParam("normalize", true),
                      SpecParam("stacked", !qwen),
                      SpecParam("projection_scales", qwen),
                      SpecParam("expert_scales", !qwen),
                      SpecParam("shared_gate", qwen)};
  std::ranges::sort(component.params, {}, &spec::Param::name);
  auto instance = co::MakeInstance(component, {.width = 64, .context = 4096, .streams = 4},
                                   co::Site::kLayer, 0, 1);
  if (!instance) return std::unexpected(instance.error());
  auto roles = instance->def->shapes(*instance);
  if (!roles) return std::unexpected(roles.error());
  std::vector<co::BoundTensor> tensors;
  for (const auto& role : *roles) {
    const auto name = instance->def->roles[role.role].name;
    const bool product = name == "gate" || name == "up" || name == "down" ||
                         name == "gate_up_experts" || name == "gate_experts" ||
                         name == "up_experts" || name == "down_experts";
    const auto encoding =
        product && !qwen ? llmp::format::Encoding::kQ8_0
        : product && (f16_experts || name == "gate" || name == "up" || name == "down")
            ? llmp::format::Encoding::kF16
            : llmp::format::Encoding::kF32;
    auto format = ir::Dense(encoding, {.ne = role.ne, .rank = role.rank});
    if (encoding == llmp::format::Encoding::kQ8_0)
      format.layout = llmp::format::Layout::kGgmlBlocks;
    const auto tensor_name = std::format("moe_{}", name);
    const bool experts = name.ends_with("_experts");
    component.weights.push_back(
        {.role = std::string(name), .tensor = tensor_name, .experts = experts});
    tensors.push_back({.name = tensor_name, .format = format, .experts = experts});
  }
  planner::StepShape shape{.stage = 0, .segments = {}};
  if (rows == 1)
    shape.segments = {{.slot = 0, .rows = 1, .past = 3, .read_cells = 256, .outputs = 1}};
  else
    shape.segments = {{.slot = 0, .rows = 3, .read_cells = 256, .outputs = 1},
                      {.slot = 1, .rows = 2, .past = 7, .read_cells = 256, .outputs = 2}};
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {SpecWeight("table", "token_embd")};
  tensors.push_back({.name = "token_embd", .format = F32Format({64, 16})});
  tensors.push_back({.name = "output_head", .format = F32Format({64, 16})});
  spec::Stage stage{.name = "target", .context = 4096, .embedding = std::move(embedding)};
  stage.stream.width = 64;
  stage.layers = {{.components = {std::move(component)}}};
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {SpecWeight("weight", "output_head")};
  spec::ModelSpec model{.stages = {std::move(stage)}};
  spec::Canonicalize(model);
  auto valid = co::ValidateSpec(model, tensors);
  if (!valid) return std::unexpected(valid.error());
  auto layout = co::StageLayout(model, 0, 4096, rows);
  if (!layout) return std::unexpected(layout.error());
  auto step = co::BuildStep(model, 0, shape, {.tensors = tensors, .state = &*layout});
  if (!step) return std::unexpected(step.error());
  return TokenStateSource{.model = std::move(model),
                          .tensors = std::move(tensors),
                          .layout = std::move(*layout),
                          .shape = std::move(shape),
                          .step = std::move(*step)};
}

// Shape-cell constructor: copy actual component flags/roles/formats from an
// opened artifact, then change only declared synthetic dimensions and payloads.
// Its derived semantic digest is never reported as the published model digest.
std::expected<TokenStateSource, std::string> MakeAdmittedMoeSource(
    const std::string& artifact_path, std::uint32_t rows, std::string_view kind = "ffn.moe") {
  auto opened = llmp::artifact::Artifact::Open(artifact_path);
  if (!opened) return std::unexpected("cannot open actual numerical source artifact");
  auto artifact = std::make_shared<llmp::artifact::Artifact>(std::move(*opened));
  auto stored = spec::StoredSpec(*artifact);
  auto facts = spec::SourceOf(*artifact);
  if (!stored || !facts) return std::unexpected("actual artifact spec/source is invalid");
  const spec::Stage* actual_stage = nullptr;
  const spec::Layer* actual_layer = nullptr;
  std::size_t chosen = 0;
  for (const auto& stage : stored->stages)
    for (const auto& layer : stage.layers)
      for (std::size_t i = 0; i < layer.components.size(); ++i)
        if (!actual_stage && layer.components[i].kind == kind) {
          actual_stage = &stage;
          actual_layer = &layer;
          chosen = i;
        }
  if (!actual_stage) return std::unexpected("actual artifact lacks the requested source component");
  const bool hyper = kind != "ffn.moe";
  std::vector<spec::ComponentRef> components;
  std::optional<spec::ComponentRef> final_mix;
  if (hyper) {
    if (chosen == 0 || chosen + 1 >= actual_layer->components.size() ||
        actual_layer->components[chosen - 1].kind != "hc.mix" ||
        actual_layer->components[chosen + 1].kind != "hc.combine" ||
        !actual_stage->heads.final_norm || actual_stage->heads.final_norm->kind != "hc.mix")
      return std::unexpected("actual attention source lacks its complete HC assembly");
    components.assign(actual_layer->components.begin() + static_cast<std::ptrdiff_t>(chosen - 1),
                      actual_layer->components.begin() + static_cast<std::ptrdiff_t>(chosen + 2));
    final_mix = actual_stage->heads.final_norm;
  } else {
    components.push_back(actual_layer->components[chosen]);
  }
  std::uint64_t width = (kind == "attention.qsa" || kind == "attention.mqa")
                            ? actual_stage->stream.width
                        : kind == "attention.gdn" && rows > 16 ? 128
                                                               : 64;
  bool native_nvfp4_slab = false;
  if (!hyper) {
    for (const auto& binding : components.front().weights) {
      const auto tensor =
          std::ranges::find(facts->tensors, binding.tensor, &spec::SourceTensor::name);
      if (tensor == facts->tensors.end()) return std::unexpected("actual role source is absent");
      auto multiple =
          static_cast<std::uint64_t>(llmp::format::Info(tensor->format.encoding).block_elements);
      if (tensor->format.layout == llmp::format::Layout::kScaleAtoms) {
        multiple *= llmp::format::kScaleAtomBlocks;
        native_nvfp4_slab |=
            binding.experts && tensor->format.encoding == llmp::format::Encoding::kNvfp4;
      }
      width = std::max(width, multiple);
    }
    for (auto& param : components.front().params) {
      if (param.name == "hidden" || param.name == "expert_hidden")
        param.value = static_cast<std::int64_t>(width);
      if (param.name == "experts") {
        const auto actual = std::get<std::int64_t>(param.value);
        // Existing router helper's minimum eligible warp geometry, chosen
        // before probes. Keep the source router/gate/normalization flags.
        param.value = std::int64_t{actual >= 32 && actual % 32 == 0 ? 32 : 8};
      }
      if (param.name == "used") param.value = std::int64_t{3};
    }
  }
  // ExpertLayout's prepared CUTLASS slab requires width multiple128.
  if (native_nvfp4_slab) width = std::max(width, std::uint64_t{128});
  std::vector<co::BoundTensor> tensors;
  std::map<std::string, std::string> renamed;
  const auto bind_component = [&](spec::ComponentRef& component, std::uint16_t component_ordinal,
                                  co::Site site) -> std::expected<void, std::string> {
    auto instance = co::MakeInstance(component,
                                     {.width = width,
                                      .context = 8192,
                                      .streams = hyper ? actual_stage->stream.streams : 1,
                                      .folds = actual_stage->folds},
                                     site, 0, component_ordinal);
    if (!instance) return std::unexpected(instance.error());
    auto roles = instance->def->shapes(*instance);
    if (!roles) return std::unexpected(roles.error());
    for (auto& binding : component.weights) {
      if (!binding.stage.empty())
        return std::unexpected("shape cell cannot borrow foreign-stage weights");
      const auto original =
          std::ranges::find(facts->tensors, binding.tensor, &spec::SourceTensor::name);
      const auto role_ordinal = co::RoleIndex(*instance->def, binding.role);
      if (original == facts->tensors.end() || !role_ordinal)
        return std::unexpected("actual component role is absent from its source inventory");
      const auto shape = std::ranges::find(*roles, *role_ordinal, &co::RoleShape::role);
      if (shape == roles->end() || original->part != binding.part ||
          original->experts != binding.experts)
        return std::unexpected("actual component role/part/expert contract differs");
      auto tensor = *original;
      tensor.name = std::format("admitted_{}_{}_{}", component_ordinal, binding.role,
                                static_cast<int>(binding.part));
      tensor.storage.clear();
      tensor.format.shape = {.ne = shape->ne, .rank = shape->rank};
      renamed.emplace(binding.tensor, tensor.name);
      binding.tensor = tensor.name;
      tensors.push_back(std::move(tensor));
    }
    return {};
  };
  std::uint16_t ordinal = 1;
  for (auto& component : components) {
    if (auto bound = bind_component(component, ordinal++, co::Site::kLayer); !bound)
      return std::unexpected(bound.error());
  }
  if (final_mix)
    if (auto bound = bind_component(*final_mix, ordinal++, co::Site::kFinalNorm); !bound)
      return std::unexpected(bound.error());
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {SpecWeight("table", "token_embd")};
  tensors.push_back({.name = "token_embd", .format = F32Format({width, 16})});
  tensors.push_back({.name = "output_head", .format = F32Format({width, 16})});
  spec::Stage stage{.name = "target", .context = 8192, .embedding = std::move(embedding)};
  stage.stream.width = static_cast<std::uint32_t>(width);
  stage.layers = {{.components = std::move(components)}};
  if (hyper) {
    stage.stream.streams = actual_stage->stream.streams;
    stage.stream.mixing = actual_stage->stream.mixing;
    stage.heads.final_norm = std::move(final_mix);
  }
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {SpecWeight("weight", "output_head")};
  // Retain only actual preprocessing receipts touching this component, with
  // exact source names mapped to their explicitly derived shape-cell names.
  std::vector<spec::FoldReceipt> preprocessing;
  for (const auto& receipt : facts->preprocessing) {
    auto selected = receipt;
    selected.tensors.clear();
    for (const auto& name : receipt.tensors)
      if (const auto found = renamed.find(name); found != renamed.end())
        selected.tensors.push_back(found->second);
    if (!selected.tensors.empty()) {
      std::ranges::sort(selected.tensors);
      stage.folds.push_back(selected.name);
      preprocessing.push_back(std::move(selected));
    }
  }
  std::ranges::sort(stage.folds);
  stage.folds.erase(std::unique(stage.folds.begin(), stage.folds.end()), stage.folds.end());
  spec::ModelSpec model{.stages = {std::move(stage)}};
  spec::Canonicalize(model);
  facts->tensors = tensors;
  facts->folds = model.stages[0].folds;
  facts->preprocessing = std::move(preprocessing);
  if (const auto valid = co::ValidateFoldReceipts(model, 0, *facts); !valid)
    return std::unexpected(valid.error());
  if (const auto valid = co::ValidateSpec(model, tensors); !valid)
    return std::unexpected(valid.error());
  planner::StepShape shape{.stage = 0,
                           .phase = rows == 1 ? planner::Phase::kDecode : planner::Phase::kPrefill,
                           .segments = {{.slot = 0,
                                         .rows = rows,
                                         .past = kind == "attention.qsa"   ? 7000U
                                                 : kind == "attention.mqa" ? 256U
                                                                           : 7U,
                                         .read_cells = kind == "attention.qsa"   ? 7168U
                                                       : kind == "attention.mqa" ? 512U
                                                                                 : 256U,
                                         .outputs = rows}}};
  auto layout = co::StageLayout(model, 0, 8192, rows);
  if (!layout) return std::unexpected(layout.error());
  auto step = co::BuildStep(model, 0, shape, {.tensors = tensors, .state = &*layout});
  if (!step) return std::unexpected(step.error());
  return TokenStateSource{.model = std::move(model),
                          .tensors = std::move(tensors),
                          .layout = std::move(*layout),
                          .shape = std::move(shape),
                          .step = std::move(*step),
                          .admitted_source = std::move(*facts),
                          .artifact = std::move(artifact)};
}

// Real component dimensions from the existing DeepSeek-V4 Flash and
// Qwen3.8 Flash fixtures. Only unrelated embedding/head vocabulary is small.
// This preflight allocates no tensor payloads and grants no candidate numerics.
std::expected<TokenStateSource, std::string> MakeSpecializedStateSource(bool qsa,
                                                                        std::uint32_t rows) {
  const std::uint64_t width = qsa ? 2560 : 4096;
  constexpr std::uint32_t context = 8192;
  spec::ComponentRef mix{.kind = "hc.mix"};
  mix.params = {SpecParam("eps", 1.0e-6),
                SpecParam("mix", std::string(qsa ? "sigmoid" : "sinkhorn"))};
  if (qsa)
    mix.params.push_back(SpecParam("rank", std::int64_t{320}));
  else {
    mix.params.push_back(SpecParam("hc_eps", 1.0e-6));
    mix.params.push_back(SpecParam("iterations", std::int64_t{20}));
  }
  spec::ComponentRef attention{.kind = qsa ? "attention.qsa" : "attention.mqa"};
  attention.params = {SpecParam("placement", std::string("none")),
                      SpecParam("eps", 1.0e-6),
                      SpecParam("heads", std::int64_t{qsa ? 24 : 64}),
                      SpecParam("head_dim", std::int64_t{qsa ? 256 : 512}),
                      SpecParam("rope_dims", std::int64_t{64}),
                      SpecParam("rope_base", qsa ? 10000000.0 : 160000.0)};
  if (qsa) {
    attention.params.push_back(SpecParam("kv_heads", std::int64_t{2}));
    attention.params.push_back(
        SpecParam("rope_sections", std::vector<std::int64_t>{11, 11, 10, 0}));
    attention.params.push_back(SpecParam("indexer_heads", std::int64_t{4}));
    attention.params.push_back(SpecParam("indexer_dim", std::int64_t{128}));
    attention.params.push_back(SpecParam("ratio", std::int64_t{4}));
    attention.params.push_back(SpecParam("budget", std::int64_t{2048}));
    attention.params.push_back(SpecParam("indexer", std::string("fused")));
  } else {
    attention.params.push_back(SpecParam("q_rank", std::int64_t{1024}));
    attention.params.push_back(SpecParam("o_rank", std::int64_t{1024}));
    attention.params.push_back(SpecParam("o_groups", std::int64_t{8}));
    attention.params.push_back(SpecParam("window", std::int64_t{128}));
    attention.params.push_back(SpecParam("rope_factor", 16.0));
    attention.params.push_back(SpecParam("yarn_context", std::int64_t{65536}));
    attention.params.push_back(SpecParam("beta_fast", 32.0));
    attention.params.push_back(SpecParam("beta_slow", 1.0));
    attention.params.push_back(SpecParam("ratio", std::int64_t{4}));
    attention.params.push_back(SpecParam("overlap", true));
    attention.params.push_back(SpecParam("indexer", true));
    attention.params.push_back(SpecParam("indexer_heads", std::int64_t{64}));
    attention.params.push_back(SpecParam("index_dim", std::int64_t{128}));
    attention.params.push_back(SpecParam("top_k", std::int64_t{512}));
    attention.params.push_back(SpecParam("cache", std::string("ring")));
  }
  spec::ComponentRef combine{.kind = "hc.combine"};
  combine.params = {SpecParam("mix", std::string(qsa ? "sigmoid" : "sinkhorn"))};
  auto head_mix = mix;
  head_mix.params.push_back(SpecParam("inject", false));
  std::vector<co::BoundTensor> tensors;
  std::uint32_t ordinal = 0;
  const auto bind = [&](spec::ComponentRef& ref) -> std::expected<void, std::string> {
    std::ranges::sort(ref.params, {}, &spec::Param::name);
    const auto site = &ref == &head_mix ? co::Site::kFinalNorm : co::Site::kLayer;
    auto instance =
        co::MakeInstance(ref, {.width = width, .context = context, .streams = 4}, site, 0, 1);
    if (!instance) return std::unexpected(instance.error());
    auto roles = instance->def->shapes(*instance);
    if (!roles) return std::unexpected(roles.error());
    for (const auto& role : *roles) {
      const auto name = instance->def->roles[role.role].name;
      auto encoding = llmp::format::Encoding::kF32;
      const bool matrix = role.rank >= 2 && name != "comp_ape" && name != "idx_ape" && name != "fn";
      if (matrix) {
        if (qsa && ref.kind == "attention.qsa")
          encoding = llmp::format::Encoding::kMxfp8;
        else if (qsa)
          encoding = llmp::format::Encoding::kBF16;
        else if (name == "q_a" || name == "q_b" || name == "kv" || name == "out_a" ||
                 name == "out_b" || name == "idx_q")
          encoding = llmp::format::Encoding::kQ8_0;
        else
          encoding = llmp::format::Encoding::kF16;
      }
      auto format = ir::Dense(encoding, {.ne = role.ne, .rank = role.rank});
      if (encoding == llmp::format::Encoding::kQ8_0)
        format.layout = llmp::format::Layout::kGgmlBlocks;
      if (encoding == llmp::format::Encoding::kMxfp8) format.layout = llmp::format::Layout::kSplit;
      const auto tensor = std::format("specialized_{}_{}", ordinal, name);
      if (encoding == llmp::format::Encoding::kMxfp8) {
        for (const auto part :
             {llmp::format::PartRole::kCodes, llmp::format::PartRole::kBlockScales}) {
          const auto part_name =
              tensor + (part == llmp::format::PartRole::kCodes ? ".codes" : ".scales");
          ref.weights.push_back({.role = std::string(name), .tensor = part_name, .part = part});
          tensors.push_back({.name = part_name, .format = format, .part = part});
        }
      } else {
        ref.weights.push_back({.role = std::string(name), .tensor = tensor});
        tensors.push_back({.name = tensor, .format = format});
      }
    }
    ++ordinal;
    return {};
  };
  for (auto* ref : {&mix, &attention, &combine, &head_mix}) {
    auto bound = bind(*ref);
    if (!bound) return std::unexpected(bound.error());
  }
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {SpecWeight("table", "token_embd")};
  tensors.push_back({.name = "token_embd", .format = F32Format({width, 16})});
  tensors.push_back({.name = "output_head", .format = F32Format({width, 16})});
  spec::Stage stage{.name = "target", .context = context, .embedding = std::move(embedding)};
  stage.stream.width = width;
  stage.stream.streams = 4;
  stage.stream.mixing =
      spec::ComponentRef{.kind = "residual.hc",
                         .params = {SpecParam("mix", std::string(qsa ? "sigmoid" : "sinkhorn"))}};
  stage.layers = {{.components = {std::move(mix), std::move(attention), std::move(combine)}}};
  stage.heads.final_norm = std::move(head_mix);
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {SpecWeight("weight", "output_head")};
  spec::ModelSpec model{.stages = {std::move(stage)}};
  spec::Canonicalize(model);
  if (auto valid = co::ValidateSpec(model, tensors); !valid) return std::unexpected(valid.error());
  auto layout = co::StageLayout(model, 0, context, rows);
  if (!layout) return std::unexpected(layout.error());
  planner::StepShape shape{.stage = 0,
                           .phase = planner::Phase::kPrefill,
                           .segments = {{.slot = 0,
                                         .rows = rows,
                                         .past = qsa ? 7000U : 256U,
                                         .read_cells = qsa ? 7168U : 512U,
                                         .outputs = 1}}};
  auto step = co::BuildStep(model, 0, shape, {.tensors = tensors, .state = &*layout},
                            {.narrow = false, .rounded_index_mask = !qsa});
  if (!step) return std::unexpected(step.error());
  return TokenStateSource{.model = std::move(model),
                          .tensors = std::move(tensors),
                          .layout = std::move(*layout),
                          .shape = std::move(shape),
                          .step = std::move(*step)};
}

class Screen {
 public:
  Screen() {
    execution_ = Must(llmp::providers::cuda::OpenDeviceExecution(0));
    stream_ = Must(execution_->CreateStream());
    native_ = Must(execution_->Submission(stream_));
    kn::DeviceBackendOptions backend_options{
        .lower = {.expert_stride = [this](const ir::WeightRef& ref) {
          const auto found = expert_pitches_.find(ref);
          return found == expert_pitches_.end() ? std::uint64_t{0} : found->second;
        }}};
    backend_ = Must(kn::DeviceBackend::Create(0, *execution_, stream_, std::move(backend_options)));
    const auto charge = [this](std::uint64_t bytes) {
      if (bytes > kMetadata - metadata_used_) return false;
      metadata_used_ += bytes;
      return true;
    };
    const auto release = [this](std::uint64_t bytes) {
      if (bytes > metadata_used_) Fail("backend metadata accounting underflow");
      metadata_used_ -= bytes;
    };
    Must(backend_->FundConstruction(charge, release));
    Must(backend_->Declare({}, charge, release));
    library_.Allocate(backend_->library_workspace_bytes());
    persistent_.Allocate(backend_->persistent_bytes());
    std::vector<std::uint64_t> workspaces{library_.address()};
    if (backend_->persistent_bytes() != 0) workspaces.push_back(persistent_.address());
    Must(backend_->BindWorkspaces(workspaces));
  }
  ~Screen() {
    Finish();
    recorded_ = {};
    programs_ = {};
    launch_.reset();
    backend_->ReleaseWorkspaces();
    backend_.reset();
    Must(execution_->DestroyStream(stream_));
  }
  void PreflightSpecialized(bool qsa, std::uint32_t rows, const std::string& artifact_path = {},
                            std::string_view admitted_kind = "ffn.moe") {
    const bool admitted = !artifact_path.empty();
    auto source = admitted ? MakeAdmittedMoeSource(artifact_path, rows, admitted_kind)
                           : MakeSpecializedStateSource(qsa, rows);
    if (!source) Fail(std::format("specialized original source: {}", source.error()));
    auto funder = std::make_shared<MetadataFunder>(metadata_used_, kMetadata);
    auto owned = source->step.graph.Clone(funder);
    Require(owned, "specialized source graph was not funded");
    source->step.graph = std::move(*owned);
    const auto backend = backend_->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    planner::Policy policy{};
    if (!qsa && !admitted)
      policy = Must(planner::ParsePolicy(spec::mappings::kDeepSeekServingPolicy));
    if (admitted) {
      const auto rule = Must(llmp::runtime::AdmittedPolicy(source->model, *source->admitted_source,
                                                           nullptr, {.context = 8192, .slots = 1}));
      policy = rule.text.empty() ? planner::Policy{} : Must(planner::ParsePolicy(rule.text));
      std::println(
          "{{\"event\":\"admitted_policy\",\"text\":{},"
          "\"numerical_candidates\":{}}}",
          Quote(rule.text), policy.numerical_candidates.size());
      Require(policy.numerical_candidates.empty(), "serving policy contains diagnostic candidates");
      const auto semantic = source->model.Digest();
      policy.numerical = Must(llmp::runtime::AdmittedNumerics(
          source->model, *source->admitted_source, source->artifact->sources(), nullptr, semantic,
          {.context = 8192, .slots = 1}, 0, backend));
      policy.active_stage = 0;
      std::println(
          "{{\"event\":\"admitted_shape_cell\",\"artifact\":{},"
          "\"derived_semantic\":{},\"published_semantic_claim\":false,"
          "\"numerical_grants\":{},\"ranking_samples_collected\":false}}",
          Quote(source->artifact->id()), Quote(llmp::base::ToHex(semantic)),
          policy.numerical->entries().size());
    }
    auto active = planner::ForStep(policy, rows, source->shape.segments.size());
    if (!active) Fail(active.error());
    planner::PlanOptions options{.capturable = true,
                                 .written = recipe.written,
                                 .within = recipe.within,
                                 .shape = &source->shape};
    options.on_stream = recipe.on_stream;
    options.activation_alignment = 256;
    options.root_alignment = [](const ir::Graph& graph, ir::TensorId root) {
      const auto& tensor = graph.tensor(root);
      return std::holds_alternative<ir::WeightRef>(tensor.binding) ||
                     std::holds_alternative<ir::StateRef>(tensor.binding)
                 ? std::uint64_t{256}
                 : std::uint64_t{0};
    };
    std::vector<ir::TensorId> staged, aligned;
    for (const auto& input : source->step.inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
      if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
        aligned.push_back(input.tensor);
    }
    options.staged = staged;
    options.alignment_staged = aligned;
    // Logical preflight has all state addresses absent together. Derive
    // adjacency only from the same owned stage/owner layout used in serving.
    options.state_spans = [&](std::span<const planner::StateCellSlice> slices)
        -> std::optional<planner::StateSpanFact> {
      if (slices.empty()) return std::nullopt;
      const auto owner = slices.front().ref.segment;
      if (owner >= source->shape.segments.size()) return std::nullopt;
      std::vector<md::StateSlice> requested;
      for (const auto& slice : slices) {
        if (slice.ref.stage != source->shape.stage || slice.ref.segment != owner)
          return std::nullopt;
        requested.push_back({.layer = slice.ref.layer,
                             .tensor = slice.ref.tensor,
                             .first = slice.first,
                             .cells = slice.cells});
      }
      const auto bank = md::ContiguousStateSpan(source->layout, requested);
      if (!bank) return std::nullopt;
      return planner::StateSpanFact{.bank = *bank, .region_bytes = source->layout.bytes};
    };
    std::vector<std::string_view> patterns;
    std::vector<ir::OpKey> filter;
    for (const auto& pattern : planner::Patterns()) {
      patterns.push_back(pattern.name);
      if (std::ranges::find(filter, pattern.result) == filter.end())
        filter.push_back(pattern.result);
    }
    std::println(
        "{{\"event\":\"specialized_source\",\"source\":{},\"rows\":{},"
        "\"nodes\":{},\"state_tensors\":{},\"tensor_payload_allocations\":false,"
        "\"policy\":{},\"numerical_candidates\":0,\"numerical_grants\":{},"
        "\"proof_qualification\":false,\"ranking_samples_collected\":false}}",
        Quote(admitted ? "admitted-resident-moe-shape-cell"
              : qsa    ? "qwen-qsa-modelopt-component"
                       : "deepseek-csa-gguf-component"),
        rows, source->step.graph.nodes().size(), source->step.state.size(),
        Quote(admitted || qsa ? "default" : "DeepSeekServingPolicy"), admitted);
    auto plan = planner::Plan(source->step.graph.Clone(), *recipe.registry, recipe.device, *active,
                              options);
    if (!plan)
      Fail(std::format("specialized ordinary plan {}: {}",
                       planner::PlanErrorName(plan.error().error), plan.error().detail));
    for (const auto& call : plan->steps)
      std::println("{{\"event\":\"default_implementation\",\"name\":{}}}",
                   Quote(recipe.registry->at(call.implementation).name));
    std::println("{{\"event\":\"unqualified_numerical_sites\",\"count\":{}}}",
                 plan->missing_numerical.size());
    const auto io = co::GraphIO(source->step);
    const co::ComparisonBoundaryView boundary{
        .io = io, .state = source->step.state, .active_stage = 0};
    auto discovery = planner::DiscoverComparisonPlan(
        boundary, filter, false, funder, *recipe.registry, recipe.device, *active, options);
    if (!discovery)
      Fail(std::format("specialized discovery {}: {}",
                       planner::PlanErrorName(discovery.error().error), discovery.error().detail));
    for (const auto name : patterns) {
      std::size_t count = 0;
      for (const auto& ticket : discovery->discovery.tickets()) {
        if (ticket.kind != planner::ComparisonSiteKind::kRewrite || ticket.pattern != name)
          continue;
        ++count;
        std::string groups = "[";
        for (std::size_t group = 0; group + 1 < ticket.member_offsets.size(); ++group) {
          if (group) groups += ',';
          groups += '[';
          for (auto i = ticket.member_offsets[group]; i < ticket.member_offsets[group + 1]; ++i) {
            if (i != ticket.member_offsets[group]) groups += ',';
            const auto id = ticket.member_origins[i];
            groups += std::format("{{\"node\":{},\"op\":{}}}", id.v,
                                  Quote(ir::OpName(source->step.graph.node(id).key.op)));
          }
          groups += ']';
        }
        groups += ']';
        std::vector<planner::ComparisonOriginSet> applied_groups;
        for (std::size_t i = 0; i + 1 < ticket.member_offsets.size(); ++i)
          applied_groups.emplace_back(ticket.member_origins.begin() + ticket.member_offsets[i],
                                      ticket.member_origins.begin() + ticket.member_offsets[i + 1]);
        const auto* applied =
            AppliedForSite(discovery->incumbent.plan, ticket.pattern, applied_groups);
        std::println(
            "{{\"event\":\"committed_site\",\"pattern\":{},\"version\":{},"
            "\"boundary_representable\":{},\"numerical_change\":{},\"source_groups\":{}}}",
            Quote(name), ticket.version, ticket.boundary_representable,
            !applied ? Quote("unresolved-actual-applied")
            : applied->numerical_change
                ? Quote(planner::NumericalChangeName(*applied->numerical_change))
                : "null",
            groups);
      }
      std::println("{{\"event\":\"default_site_count\",\"pattern\":{},\"occurrences\":{}}}",
                   Quote(name), count);
    }
    std::println(
        "{{\"event\":\"specialized_preflight_complete\",\"ordinary_lower_performed\":false,"
        "\"whole_proof_qualified\":false,\"ranking_samples_collected\":false}}");
  }
  void Run(bool capture_validation, bool screen, std::string_view target_pattern,
           std::uint64_t target_occurrence, bool preflight_only, std::string_view contract = "gdn",
           std::uint32_t rows = 1, const std::string& artifact_path = {},
           bool quality_rejection = false, bool quality_attribution = false) {
    const bool quality_gdn = contract == "admitted-gdn" && rows == 17 &&
                             target_pattern == "mxfp8_inputs" && target_occurrence == 0;
    const bool quality_qsa = contract == "admitted-qsa" && rows == 17 &&
                             target_pattern == "mxfp8_inputs" && target_occurrence == 0;
    const bool quality_nvfp4 = contract == "admitted-moe" && rows == 17 &&
                               target_pattern == "moe_experts" && target_occurrence == 0;
    const bool quality_deepseek = contract == "deepseek-specialized" && rows == 64 &&
                                  target_pattern == "state_bank_attention" &&
                                  target_occurrence == 0;
    Require(
        !quality_rejection || (!screen && !preflight_only &&
                               (quality_gdn || quality_qsa || quality_nvfp4 || quality_deepseek)),
        "quality rejection requires a prospectively registered fixed source cell");
    const bool inverse_attribution = contract == "deepseek-specialized" && rows == 64 &&
                                     target_pattern == "inverse_rope_grouped_mul_mat" &&
                                     target_occurrence == 0;
    const bool qsa_attribution = contract == "admitted-qsa" && rows == 17 &&
                                 target_pattern == "mean_pool_norm_rope_store" &&
                                 target_occurrence == 0;
    Require(!quality_attribution ||
                (!quality_rejection && !capture_validation && !screen && !preflight_only &&
                 (inverse_attribution || qsa_attribution ||
                  (contract == "admitted-gdn" && rows == 17 && target_pattern == "bf16_rows" &&
                   (target_occurrence == 0 || target_occurrence == 1)))),
            "untimed attribution requires its registered fixed source occurrence");
    const bool defer_numeric_quality = quality_rejection || quality_attribution;
    const bool admitted = !artifact_path.empty();
    // Predeclared ComponentsQwen38 reference convention: max absolute gap /
    // max(1, max absolute reference value), independently for logits and each
    // complete F32 state tensor. This is correctness, not a speed measurement.
    const bool resident_moe = contract.starts_with("resident-") || contract == "admitted-moe";
    const bool source_qsa = contract == "admitted-qsa" || contract == "qwen-qsa-specialized";
    const bool specialized =
        contract == "deepseek-specialized" || contract == "qwen-qsa-specialized";
    const bool state_chain = !resident_moe && contract != "gdn";
    const bool nmse_chain = state_chain;
    const bool qwen_moe = contract.starts_with("resident-qwen-");
    const bool f16_moe = contract == "resident-qwen-f16-moe";
    const bool q8_moe = contract == "resident-q8-moe";
    // The F32 MoePieces 1e-5 oracle does not cover grouped F16 input rounding.
    // Reuse the established F16 product/GLU envelope (ggml_ops_test and
    // fusion_ranks reference_limits), with a separate stricter peer gate below.
    const double tolerance = f16_moe ? 2.0e-2 : (qwen_moe ? 1.0e-5 : 1.0e-4);
    std::vector<std::string_view> patterns;
    std::vector<ir::OpKey> discovery_filter;
    for (const auto& pattern : planner::Patterns()) {
      patterns.push_back(pattern.name);
      if (std::ranges::find(discovery_filter, pattern.result) == discovery_filter.end())
        discovery_filter.push_back(pattern.result);
    }
    auto source = admitted ? MakeAdmittedMoeSource(artifact_path, rows,
                                                   resident_moe                 ? "ffn.moe"
                                                   : source_qsa                 ? "attention.qsa"
                                                   : contract == "admitted-mqa" ? "attention.mqa"
                                                                                : "attention.gdn")
                  : specialized  ? MakeSpecializedStateSource(source_qsa, rows)
                  : resident_moe ? MakeResidentMoeSource(qwen_moe, rows, f16_moe)
                                 : MakeSmallGdnStateSource();
    Require(source, "check failed: source");
    const bool admitted_nvfp4 =
        admitted && std::ranges::any_of(source->tensors, [](const auto& t) {
          return t.experts && t.format.encoding == llmp::format::Encoding::kNvfp4;
        });
    const bool mxfp8_chain = state_chain && std::ranges::any_of(source->tensors, [](const auto& t) {
                               return t.format.encoding == llmp::format::Encoding::kMxfp8;
                             });
    const double chain_bound = mxfp8_chain ? 3.0e-3 : 5.0e-4;
    const bool reference_nmse = admitted || q8_moe || nmse_chain;
    const double reference_bound = nmse_chain ? chain_bound
                                   : admitted ? (admitted_nvfp4 ? 3.0e-2 : 5.0e-4)
                                   : q8_moe   ? 5.0e-4
                                              : tolerance;
    const std::string_view reference_metric = reference_nmse ? "nmse" : "normalized-max";
    auto& step = source->step;
    const auto route_node = std::ranges::find_if(step.graph.nodes(), [](const auto& node) {
      return node.key == ir::Primitive(ir::Op::kRoute);
    });
    const auto route_params = route_node == step.graph.nodes().end()
                                  ? ir::RouteParams{}
                                  : route_node->template Params<ir::RouteParams>();
    const auto expert_count = route_params.experts;
    const auto used_experts = route_params.used;
    if (resident_moe) {
      Require(expert_count > 0 && used_experts > 0, "original source lacks its expert route");
      std::println(
          "{{\"event\":\"actual_router_geometry\",\"experts\":{},\"used\":{},"
          "\"normalize\":{},\"scale\":{},\"kind\":{}}}",
          expert_count, used_experts, route_params.normalize, route_params.scale,
          static_cast<unsigned>(route_params.kind));
    }
    Require(!resident_moe || step.state.empty(), "resident MoE unexpectedly owns state");
    auto funder = std::make_shared<MetadataFunder>(metadata_used_, kMetadata);
    auto admitted_graph = step.graph.Clone(funder);
    Require(admitted_graph, "check failed: admitted_graph");
    step.graph = std::move(*admitted_graph);
    // Prepay a source-derived upper envelope: every original tensor plus
    // native call scratch headroom. Actual final placements are checked below.
    const auto add_capacity = [](std::uint64_t& total, std::uint64_t bytes) {
      Require(bytes <= UINT64_MAX - 255U && total <= UINT64_MAX - ((bytes + 255U) & ~255ULL),
              "source resource envelope overflow");
      total += (bytes + 255U) & ~255ULL;
    };
    std::uint64_t arena_bytes = 32 * kMiB, weight_capacity = 0;
    for (std::uint32_t id = 0; id < step.graph.tensor_count(); ++id) {
      const auto& tensor = step.graph.tensor(ir::TensorId{id});
      const auto parts = Must(llmp::format::PartsOf(tensor.format));
      if (tensor.role != ir::Role::kWeight && tensor.role != ir::Role::kState)
        for (const auto& part : parts.parts()) add_capacity(arena_bytes, part.readable);
    }
    for (const auto& weight : step.weights) {
      const auto& format = step.graph.tensor(weight.tensor).format;
      if (weight.ref.expert_array && format.encoding == llmp::format::Encoding::kNvfp4 &&
          format.layout == llmp::format::Layout::kScaleAtoms) {
        if (format.stacked == 2) {
          const kg::moe::ExpertLayout layout{.ffn = format.shape.ne[1] / 2,
                                             .width = format.shape.ne[0]};
          Require(layout.bytes() <= UINT64_MAX / format.shape.ne[2], "expert envelope overflow");
          add_capacity(weight_capacity, layout.bytes() * format.shape.ne[2]);
        }
      } else {
        for (const auto& part : Must(llmp::format::PartsOf(format)).parts())
          add_capacity(weight_capacity, part.readable);
      }
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    Check(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::uint64_t required_bytes = arena_bytes;
    add_capacity(required_bytes, weight_capacity);
    add_capacity(required_bytes, source->layout.bytes);
    Require(required_bytes <= free_bytes, "actual source payload exceeds fresh device memory");
    Require(activation_.Allocate(arena_bytes), "activation envelope allocation failed");
    Require(weights_.Allocate(weight_capacity), "native weight envelope allocation failed");
    Require(state_.Allocate(source->layout.bytes), "state image allocation failed");
    const auto backend = backend_->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    Require(backend.owner != nullptr, "actual screen requires its backend declaration owner");
    const auto declaration = Must(backend.owner->Ready());
    std::string device;
    llmp::base::json::AppendQuoted(recipe.device.device_model, device);
    int driver = 0, runtime = 0;
    Check(cudaDriverGetVersion(&driver));
    Check(cudaRuntimeGetVersion(&runtime));
    std::println(
        "{{\"event\":\"backend\",\"recipe\":\"{}\",\"build\":\"{}\",\"selection\":\"{}\","
        "\"device\":{},\"arch\":{},\"multiprocessors\":{},\"driver\":{},\"runtime\":{}}}",
        llmp::base::ToHex(declaration.recipe), llmp::base::ToHex(declaration.build),
        llmp::base::ToHex(declaration.selection), device, recipe.device.arch,
        recipe.device.multiprocessors, driver, runtime);
    if (resident_moe)
      std::println(
          "{{\"event\":\"source\",\"contract\":{},\"graph\":\"{}\","
          "\"selection_environment\":\"{}\",\"width\":{},\"rows\":{},\"experts\":{},\"used\":{},"
          "\"source_kind\":\"embedding-moe-head-token-step\",\"resident_experts\":true,\"paged_key_"
          "transfer\":false,\"cpu_metric\":\"{}\",\"cpu_tolerance\":{}}}",
          Quote(contract), llmp::base::ToHex(step.graph.Digest()),
          llmp::base::ToHex(recipe.registry->SelectionDigest()),
          source->model.stages[0].stream.width, rows, expert_count, used_experts,
          q8_moe ? "nmse" : "normalized-max", q8_moe ? 5.0e-4 : tolerance);
    else {
      std::println(
          "{{\"event\":\"source\",\"graph\":\"{}\",\"selection_environment\":\"{}\",\"width\":{},"
          "\"rows\":{},\"state_tensors\":{}}}",
          llmp::base::ToHex(step.graph.Digest()),
          llmp::base::ToHex(recipe.registry->SelectionDigest()),
          source->model.stages[0].stream.width, rows, step.state.size());
    }

    std::map<std::pair<ir::WeightRef, llmp::format::PartRole>, kn::BoundedAddress> weights;
    std::map<ir::TensorId, std::map<llmp::format::PartRole, std::vector<std::byte>>> weight_bytes;
    std::uint64_t weight_end = 0;
    struct FixtureSlab {
      std::uint64_t address;
      std::vector<std::byte> bytes;
    };
    std::vector<FixtureSlab> immutable_slabs;
    std::map<std::pair<std::uint16_t, std::string>, kn::ExpertPartBinding> expert_parts;
    std::map<std::pair<ir::WeightRef, llmp::format::PartRole>,
             std::pair<std::uint64_t, std::uint64_t>>
        strided_parts;
    ir::TensorId norm_weight{};
    bool initialized_folded_a = false;
    std::vector<ir::TensorId> negative_decay_leaves;
    for (const auto& weight : step.weights) {
      const auto& format = step.graph.tensor(weight.tensor).format;

      const auto parts = llmp::format::PartsOf(format);
      Require(parts, "check failed: parts");
      Require(parts->size() == weight.parts.size(), "native weight part inventory differs");
      const auto& name = weight.parts.front().first;
      std::string_view semantic_kind, semantic_role;
      const auto identify_role = [&](const spec::ComponentRef& component) {
        for (const auto& binding : component.weights)
          if (binding.tensor == name) {
            Require(semantic_role.empty() ||
                        (semantic_role == binding.role && semantic_kind == component.kind),
                    "source leaf has conflicting semantic roles");
            semantic_kind = component.kind;
            semantic_role = binding.role;
          }
      };
      if (state_chain) {
        for (const auto& layer : source->model.stages[0].layers)
          for (const auto& component : layer.components) identify_role(component);
        if (source->model.stages[0].heads.final_norm)
          identify_role(*source->model.stages[0].heads.final_norm);
      }
      if (name == "gdn_norm") norm_weight = weight.tensor;
      const bool needs_data = std::ranges::any_of(parts->parts(), [](const auto& part) {
        return part.role == llmp::format::PartRole::kData;
      });
      std::vector<float> values(needs_data ? ir::Elements(format.shape) : 0);
      for (std::size_t i = 0; i < values.size(); ++i) {
        const float fraction = static_cast<float>((i * 17U + 11U) % 97U) / 96.0F;
        // kGdnRanges: folded A strictly negative, dt_bias and convolution small.
        values[i] = resident_moe             ? (name == "token_embd"     ? -2.0F + 4.0F * fraction
                                                : name == "output_head"  ? -0.125F + 0.25F * fraction
                                                : format.shape.rank == 1 ? 0.9F + 0.2F * fraction
                                                                         : -0.3F + 0.6F * fraction)
                    : name == "gdn_a"        ? -1.2F + fraction
                    : name == "gdn_dt_bias"  ? -0.5F + fraction
                    : name == "gdn_conv"     ? -0.6F + 1.2F * fraction
                    : format.shape.rank == 1 ? 0.9F + 0.2F * fraction
                                             : -0.03F + 0.06F * fraction;
      }
      if (state_chain) {
        for (std::size_t i = 0; i < values.size(); ++i) {
          const float fraction = static_cast<float>((i * 17U + 11U) % 97U) / 96.0F;
          if (semantic_kind == "attention.gdn" && semantic_role == "a")
            values[i] = -1.2F + fraction;
          else if (semantic_kind == "attention.gdn" && semantic_role == "dt_bias")
            values[i] = -0.5F + fraction;
          else if (semantic_kind == "attention.gdn" && semantic_role == "conv")
            values[i] = -0.6F + 1.2F * fraction;
          else if (semantic_role.find("norm") != std::string_view::npos)
            values[i] = 0.9F + 0.2F * fraction;
        }
      }
      if (state_chain && semantic_kind == "attention.gdn" && semantic_role == "a") {
        Require(std::ranges::all_of(values, [](float x) { return x < 0; }),
                "semantic GDN decay payload must retain its negative folded domain");
        negative_decay_leaves.push_back(weight.tensor);
        std::println(
            "{{\"event\":\"semantic_weight_payload\",\"component\":{},\"role\":{},"
            "\"tensor\":{},\"negative_decay\":true}}",
            Quote(semantic_kind), Quote(semantic_role), weight.tensor.v);
      }
      if (name == "gdn_a") {
        Require(
            std::ranges::all_of(values, [](float value) { return value < 0.0F; }),
            "check failed: std::ranges::all_of(values, [](float value) { return value < 0.0F; })");
        initialized_folded_a = true;
      }
      for (const auto& part : parts->parts()) {
        std::vector<std::byte> encoded(part.readable, std::byte{0});
        if (part.role == llmp::format::PartRole::kData) {
          const auto matrix_elements = format.shape.ne[0] * format.shape.ne[1];
          for (std::uint64_t matrix = 0; matrix < llmp::format::Matrices(format); ++matrix) {
            const auto data = std::span(values).subspan(matrix * matrix_elements, matrix_elements);
            if (reference::EncodeRows(format, matrix, 0, data, encoded)) continue;
            // Existing dsv4_fast_test::Quantize fixture contract: IQ encoders
            // need an explicit importance vector. This creates real native
            // codes; CPU reference still decodes the exact resulting bytes.
            const auto& info = llmp::format::Info(format.encoding);
            Require(info.ggml.has_value(), "native weight encoding is unavailable");
            const auto type = static_cast<ggml_type>(info.ggml->type);
            Require(ggml_quantize_requires_imatrix(type), "unexpected single-part encoder refusal");
            std::vector<float> importance(format.shape.ne[0], 1.0F);
            ggml_quantize_init(type);
            const auto matrix_bytes = part.bytes / llmp::format::Matrices(format);
            const auto written = ggml_quantize_chunk(
                type, data.data(), encoded.data() + matrix * matrix_bytes, 0,
                static_cast<std::int64_t>(format.shape.ne[1]),
                static_cast<std::int64_t>(format.shape.ne[0]), importance.data());
            Require(written == matrix_bytes, "native importance-weighted encoding extent differs");
          }
        } else if (part.role == llmp::format::PartRole::kCodes) {
          Require(format.encoding == llmp::format::Encoding::kMxfp8 ||
                      format.encoding == llmp::format::Encoding::kNvfp4,
                  "synthetic native codes require the admitted MXFP8/NVFP4 format");
          const auto matrices = llmp::format::Matrices(format);
          const auto matrix_bytes = part.bytes / matrices;
          const auto row_bytes =
              format.shape.ne[0] * llmp::format::Info(format.encoding).element_bits / 8;
          std::uint64_t role_seed = 0;
          for (const unsigned char ch : name) role_seed = role_seed * 131U + ch;
          for (std::uint64_t matrix = 0; matrix < matrices; ++matrix) {
            for (std::uint64_t i = 0; i < matrix_bytes; ++i) {
              const auto row = i / row_bytes;
              const auto column = i % row_bytes;
              const auto seed =
                  role_seed + matrix * 101U + row * 37U + (column / 16U) * 19U + column * 7U;
              // Finite E4M3 or two independent signed E2M1 codes. Matrix,
              // row, block and role distinguish native expert payloads.
              const auto code =
                  format.encoding == llmp::format::Encoding::kMxfp8
                      ? ((seed & 1U) * 0x80U | (0x20U + seed % 24U))
                      : (((seed % 7U + 1U) | ((seed / 7U & 1U) << 3U)) |
                         (((seed / 13U % 7U + 1U) | ((seed / 17U & 1U) << 3U)) << 4U));
              encoded[matrix * matrix_bytes + i] = static_cast<std::byte>(code);
            }
          }
        } else if (part.role == llmp::format::PartRole::kBlockScales) {
          Require(format.encoding == llmp::format::Encoding::kMxfp8 ||
                      format.encoding == llmp::format::Encoding::kNvfp4,
                  "synthetic native scales require the admitted MXFP8/NVFP4 format");
          const auto matrices = llmp::format::Matrices(format);
          const auto matrix_bytes = part.bytes / matrices;
          const auto blocks =
              format.shape.ne[0] / llmp::format::Info(format.encoding).block_elements;
          std::uint64_t role_seed = 0;
          for (const unsigned char ch : name) role_seed = role_seed * 131U + ch;
          for (std::uint64_t matrix = 0; matrix < matrices; ++matrix) {
            for (std::uint64_t row = 0; row < format.shape.ne[1]; ++row) {
              for (std::uint64_t block = 0; block < blocks; ++block) {
                const auto offset = format.layout == llmp::format::Layout::kScaleAtoms
                                        ? llmp::format::ScaleAtomOffset(row, block, blocks)
                                        : row * blocks + block;
                Require(offset < matrix_bytes, "native scale atom exceeds matrix part");
                const auto seed = role_seed + matrix * 11U + row * 7U + block * 3U;
                // Finite positive scales; unused atom rows/readable tails stay zero.
                const auto scale = format.encoding == llmp::format::Encoding::kMxfp8
                                       ? 121U + seed % 3U
                                       : 0x08U + seed % 17U;
                encoded[matrix * matrix_bytes + offset] = static_cast<std::byte>(scale);
              }
            }
          }
        } else if (part.role == llmp::format::PartRole::kTensorScale) {
          Require(part.bytes % sizeof(float) == 0, "native tensor scale extent differs");
          const float scale = 1.0F;
          for (std::uint64_t i = 0; i < part.bytes; i += sizeof(scale))
            std::memcpy(encoded.data() + i, &scale, sizeof(scale));
        } else {
          Fail("unsupported admitted immutable part in fixed producer");
        }
        if (weight.ref.expert_array && format.encoding == llmp::format::Encoding::kNvfp4 &&
            format.layout == llmp::format::Layout::kScaleAtoms) {
          weight_bytes[weight.tensor].emplace(part.role, std::move(encoded));
          continue;
        }
        weight_end = (weight_end + 255U) & ~std::uint64_t{255U};
        Require(part.readable <= weight_capacity - weight_end, "native weight parts exceed arena");
        const kn::BoundedAddress bound{
            .base = weights_.address() + weight_end, .bytes = part.readable, .alignment = 256};
        Require(weights.emplace(std::pair{weight.ref, part.role}, bound).second,
                "native weight part is duplicated");
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(bound.base), encoded.data(), part.readable,
                              cudaMemcpyHostToDevice, static_cast<cudaStream_t>(native_.handle)));
        weight_bytes[weight.tensor].emplace(part.role, std::move(encoded));
        weight_end += part.readable;
      }
    }
    if (admitted_nvfp4) {
      const co::StepWeight* gate_up = nullptr;
      const co::StepWeight* down = nullptr;
      for (const auto& weight : step.weights) {
        const auto& format = step.graph.tensor(weight.tensor).format;
        if (!weight.ref.expert_array || format.encoding != llmp::format::Encoding::kNvfp4) continue;
        if (format.stacked == 2)
          gate_up = &weight;
        else if (format.stacked == 1)
          down = &weight;
      }
      Require(gate_up && down, "native prepared expert source requires gate/up and down arrays");
      const auto& gu = step.graph.tensor(gate_up->tensor).format;
      const auto& dn = step.graph.tensor(down->tensor).format;
      const kg::moe::ExpertLayout layout{.ffn = gu.shape.ne[1] / 2, .width = gu.shape.ne[0]};
      Require(layout.width % 128 == 0 && layout.ffn % 64 == 0 && dn.shape.ne[0] == layout.ffn &&
                  dn.shape.ne[1] == layout.width && dn.shape.ne[2] == gu.shape.ne[2],
              "native expert slab geometry differs");
      const auto experts = gu.shape.ne[2];
      const auto pitch = layout.bytes();
      weight_end = (weight_end + 255U) & ~std::uint64_t{255U};
      Require(pitch <= UINT64_MAX / experts && pitch * experts <= weight_capacity - weight_end,
              "native expert slab exceeds actual weight capacity");
      FixtureSlab slab{.address = weights_.address() + weight_end,
                       .bytes = std::vector<std::byte>(pitch * experts)};
      for (const auto* weight : {gate_up, down}) {
        expert_pitches_.emplace(weight->ref, pitch);
        const auto& format = step.graph.tensor(weight->tensor).format;
        auto one = format;
        one.shape.ne[2] = one.shape.ne[3] = 1;
        one.shape.rank = 2;
        const auto extents = Must(llmp::format::PartsOf(one));
        for (const auto& part : extents.parts()) {
          Require(part.role == llmp::format::PartRole::kCodes ||
                      part.role == llmp::format::PartRole::kBlockScales,
                  "native prepared expert slab has unexpected parts");
          const auto offset =
              weight == gate_up
                  ? (part.role == llmp::format::PartRole::kCodes ? layout.gate_up_codes()
                                                                 : layout.gate_up_scales())
                  : (part.role == llmp::format::PartRole::kCodes ? layout.down_codes()
                                                                 : layout.down_scales());
          const auto& bytes = weight_bytes.at(weight->tensor).at(part.role);
          Require(bytes.size() == experts * part.bytes && offset + part.readable <= pitch,
                  "native expert part does not fit its exact prepared slot");
          for (std::uint64_t e = 0; e < experts; ++e)
            std::memcpy(slab.bytes.data() + e * pitch + offset, bytes.data() + e * part.bytes,
                        part.bytes);
          const auto base = slab.address + offset;
          weights.emplace(
              std::pair{weight->ref, part.role},
              kn::BoundedAddress{.base = base,
                                 .bytes = (experts - 1) * pitch + part.readable,
                                 .alignment = 256,
                                 .allocation = kn::BackingAllocation{.base = slab.address,
                                                                     .bytes = slab.bytes.size()}});
          strided_parts.emplace(std::pair{weight->ref, part.role}, std::pair{pitch, part.bytes});
          const auto named =
              std::ranges::find(weight->parts, part.role, [](const auto& p) { return p.second; });
          Require(named != weight->parts.end(), "native expert source part name missing");
          expert_parts.emplace(std::pair{weight->ref.stage, named->first},
                               kn::ExpertPartBinding{.address = base,
                                                     .bytes = (experts - 1) * pitch + part.readable,
                                                     .pitch = pitch,
                                                     .format = one,
                                                     .role = part.role});
        }
      }
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(slab.address), slab.bytes.data(),
                            slab.bytes.size(), cudaMemcpyHostToDevice,
                            static_cast<cudaStream_t>(native_.handle)));
      weight_end += slab.bytes.size();
      immutable_slabs.push_back(std::move(slab));
    }
    Finish();
    if (contract == "gdn") {
      Require(norm_weight.valid(), "check failed: norm_weight.valid()");
      Require(initialized_folded_a, "check failed: initialized_folded_a");
    }
    std::map<ir::StateRef, kn::BoundedAddress> states;
    std::map<ir::TensorId, std::vector<float>> initial_values;
    std::vector<std::byte> initial_state(source->layout.bytes, std::byte{0});
    const auto encode_state = [&](const co::StepState& state, std::span<const float> values) {
      const auto& format = step.graph.tensor(state.tensor).format;
      const auto& item = source->layout.items.at(state.item);
      Require(
          item.offset <= initial_state.size() && item.bytes <= initial_state.size() - item.offset,
          "owned state item exceeds complete layout");
      auto bytes = std::span(initial_state).subspan(item.offset, item.bytes);
      const auto matrix_elements = format.shape.ne[0] * format.shape.ne[1];
      for (std::uint64_t matrix = 0; matrix < llmp::format::Matrices(format); ++matrix)
        Must(reference::EncodeRows(
            format, matrix, 0, values.subspan(matrix * matrix_elements, matrix_elements), bytes));
    };
    for (const auto& state : step.state) {
      Require(!state.borrowed, "fixture does not own borrowed state reset");
      const auto& item = source->layout.items.at(state.item);
      const auto& format = step.graph.tensor(state.tensor).format;
      const auto parts = Must(llmp::format::PartsOf(format));
      Require(parts.size() == 1 && parts[0].role == llmp::format::PartRole::kData &&
                  parts[0].bytes <= item.bytes,
              "complete source state must have its actual single encoded data part");
      std::vector<float> values(ir::Elements(format.shape), 0.0F);
      // QSA old raw history and already-completed pooled history share the
      // normative zero image (normalization/RoPE of zero is zero).
      if (!source_qsa)
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = static_cast<float>(static_cast<int>((i * 13U + 7U) % 61U) - 30) / 2048.0F;
      encode_state(state, values);
      initial_values.emplace(state.tensor, std::move(values));
      Require(states
                  .emplace(state.ref,
                           kn::BoundedAddress{
                               .base = state_.address() + item.offset,
                               .bytes = item.bytes,
                               .alignment = 256,
                               .allocation = kn::BackingAllocation{.base = state_.address(),
                                                                   .bytes = source->layout.bytes}})
                  .second,
              "duplicate owned state binding");
    }
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole part) {
      return weights.at({ref, part}).base;
    };
    addresses.weight_range = [&](const ir::WeightRef& ref,
                                 llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
      return weights.at({ref, part});
    };
    addresses.expert_part = [&](std::uint16_t stage,
                                std::string_view name) -> std::optional<kn::ExpertPartBinding> {
      const auto found = expert_parts.find({stage, std::string(name)});
      return found == expert_parts.end() ? std::nullopt : std::optional(found->second);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, llmp::format::PartRole) {
      return std::uint64_t{256};
    };
    addresses.state = [&](const ir::StateRef& ref) { return states.at(ref).base; };
    addresses.state_range = [&](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
      return states.at(ref);
    };
    addresses.state_alignment = [](const ir::StateRef&) { return std::uint64_t{256}; };
    addresses.activations = activation_.address();
    addresses.activation_bytes = arena_bytes;
    addresses.activation_alignment = 256;
    planner::PlanOptions options{
        .capturable = true, .within = recipe.within, .shape = &source->shape};
    options.written = recipe.written;
    options.on_stream = recipe.on_stream;
    std::vector<ir::TensorId> staged, aligned;
    for (const auto& input : step.inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
      if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
        aligned.push_back(input.tensor);
    }
    options.staged = staged;
    options.alignment_staged = aligned;
    options.activation_alignment = 256;
    options.root_alignment = [](const ir::Graph& graph, ir::TensorId root) {
      const auto& desc = graph.tensor(root);
      return std::holds_alternative<ir::WeightRef>(desc.binding) ||
                     std::holds_alternative<ir::StateRef>(desc.binding)
                 ? std::uint64_t{256}
                 : std::uint64_t{0};
    };
    options.state_spans = [&](std::span<const planner::StateCellSlice> slices)
        -> std::optional<planner::StateSpanFact> {
      if (slices.empty()) return std::nullopt;
      const auto owner = slices.front().ref.segment;
      if (owner >= source->shape.segments.size()) return std::nullopt;
      std::vector<md::StateSlice> requested;
      for (const auto& slice : slices) {
        if (slice.ref.stage != source->shape.stage || slice.ref.segment != owner)
          return std::nullopt;
        requested.push_back({.layer = slice.ref.layer,
                             .tensor = slice.ref.tensor,
                             .first = slice.first,
                             .cells = slice.cells});
      }
      const auto bank = md::ContiguousStateSpan(source->layout, requested);
      if (!bank) return std::nullopt;
      return planner::StateSpanFact{.bank = *bank, .region_bytes = source->layout.bytes};
    };
    planner::Policy base_policy{};
    if (specialized && !source_qsa)
      base_policy = Must(planner::ParsePolicy(spec::mappings::kDeepSeekServingPolicy));
    if (admitted) {
      const auto rule = Must(llmp::runtime::AdmittedPolicy(source->model, *source->admitted_source,
                                                           nullptr, {.context = 8192, .slots = 1}));
      base_policy = rule.text.empty() ? planner::Policy{} : Must(planner::ParsePolicy(rule.text));
      base_policy.numerical = Must(llmp::runtime::AdmittedNumerics(
          source->model, *source->admitted_source, source->artifact->sources(), nullptr,
          source->model.Digest(), {.context = 8192, .slots = 1}, 0, backend));
      base_policy.active_stage = 0;
      Require(base_policy.numerical_candidates.empty(), "serving policy has diagnostic candidates");
      std::println(
          "{{\"event\":\"admitted_execution\",\"artifact\":{},"
          "\"derived_semantic\":{},\"nvfp4_experts\":{},\"grants\":{},"
          "\"cpu_nmse_bound\":{},\"peer_gate_from_actual_applied\":true}}",
          Quote(source->artifact->id()), Quote(llmp::base::ToHex(source->model.Digest())),
          admitted_nvfp4, base_policy.numerical->entries().size(), reference_bound);
    }
    auto policy =
        planner::ForStep(base_policy, source->shape.rows(), source->shape.segments.size());
    Require(policy, "check failed: policy");
    auto preliminary =
        planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, *policy, options);
    if (!preliminary)
      Fail(std::format("normal {} rows {} planning {}: {}", contract, rows,
                       planner::PlanErrorName(preliminary.error().error),
                       preliminary.error().detail));
    auto bindings = en::ComparisonSourceBindings::Create(funder, step, 0);
    Require(bindings, "check failed: bindings");
    auto resolved = bindings->Resolve(*preliminary, addresses, recipe.written);
    if (!resolved) Fail(std::format("source binding resolution: {}", resolved.error().detail));
    const auto view = bindings->boundary_view();
    Require(view, "check failed: view");
    planner::ComparisonSiteSelector selector;
    bool checkpoint_only = false;
    {
      auto discovered =
          planner::DiscoverComparisonPlan(*view, discovery_filter, false, funder, *recipe.registry,
                                          recipe.device, *policy, options);
      if (!discovered)
        Fail(std::format("full-GDN discovery {}: {}",
                         planner::PlanErrorName(discovered.error().error),
                         discovered.error().detail));
      std::vector<std::uint64_t> occurrences(patterns.size());
      bool representable = true, selected = false;
      for (const auto& ticket : discovered->discovery.tickets()) {
        const auto family = std::ranges::find(patterns, ticket.pattern);
        if (family == patterns.end() || ticket.kind != planner::ComparisonSiteKind::kRewrite)
          continue;
        const auto index = static_cast<std::size_t>(family - patterns.begin());
        const auto occurrence = occurrences[index]++;
        Require(ticket.member_offsets.size() >= 2 && ticket.member_offsets.front() == 0 &&
                    ticket.member_offsets.back() == ticket.member_origins.size(),
                "committed full-GDN ticket has malformed source groups");
        bool convolution = false, history = false;
        std::string groups = "[";
        for (std::size_t group = 0; group + 1 < ticket.member_offsets.size(); ++group) {
          if (group) groups += ',';
          groups += '[';
          const auto first = ticket.member_offsets[group], last = ticket.member_offsets[group + 1];
          Require(first < last && last <= ticket.member_origins.size(),
                  "committed full-GDN ticket has an invalid source group");
          for (auto i = first; i < last; ++i) {
            if (i != first) groups += ',';
            const auto original = ticket.member_origins[i];
            Require(original.valid() && original.v < step.graph.nodes().size(),
                    "committed full-GDN ticket names an invalid original node");
            const auto& node = step.graph.node(original);
            convolution |= node.key == ir::Primitive(ir::Op::kSsmConv);
            history |= node.key == ir::Primitive(ir::Op::kCacheStore);
            groups += std::format("{{\"node\":{},\"op\":{}}}", original.v,
                                  Quote(ir::OpName(node.key.op)));
          }
          groups += ']';
        }
        groups += ']';
        const auto label = std::format("{}/{}/{}", ticket.pattern,
                                       resident_moe                   ? "resident-component"
                                       : ticket.pattern == "gdn_conv" ? (convolution ? "convolution"
                                                                         : history   ? "history"
                                                                                     : "other")
                                       : ticket.pattern == "gdn_step" ? "recurrence"
                                                                      : "norm-gate",
                                       occurrence);
        std::println(
            "{{\"event\":\"committed_site\",\"pattern\":{},\"occurrence\":{},\"label\":{},"
            "\"version\":{},\"ticket_ordinal\":{},\"boundary_representable\":{},"
            "\"registered_table_numerical_change\":{},\"source_groups\":{}}}",
            Quote(ticket.pattern), occurrence, Quote(label), ticket.version, ticket.ordinal,
            ticket.boundary_representable,
            planner::FindPattern(ticket.pattern)->numerical_change
                ? Quote(planner::NumericalChangeName(
                      *planner::FindPattern(ticket.pattern)->numerical_change))
                : "null",
            groups);
        if (ticket.pattern != target_pattern || occurrence != target_occurrence) continue;
        representable = ticket.boundary_representable;
        checkpoint_only = !representable;
        Require(!selected, "full-GDN committed occurrence is ambiguous");
        selected = true;
        selector.source_graph_identity = discovered->discovery.source_identity();
        selector.kind = ticket.kind;
        selector.pattern = std::string(ticket.pattern);
        selector.version = ticket.version;
        selector.member_owner_segments = ticket.owner_segments;
        selector.boundary_ports = ticket.boundary_ports;
        if (checkpoint_only) selector.checkpoint_match = ticket.checkpoint_match;
        for (std::size_t group = 0; group + 1 < ticket.member_offsets.size(); ++group)
          selector.member_origins.emplace_back(
              ticket.member_origins.begin() + ticket.member_offsets[group],
              ticket.member_origins.begin() + ticket.member_offsets[group + 1]);
      }
      for (std::size_t i = 0; i < patterns.size(); ++i)
        std::println("{{\"event\":\"default_site_count\",\"pattern\":{},\"occurrences\":{}}}",
                     Quote(patterns[i]), occurrences[i]);
      for (const auto& call : discovered->incumbent.plan.steps)
        std::println("{{\"event\":\"default_implementation\",\"name\":{},\"identity\":\"{}\"}}",
                     Quote(recipe.registry->at(call.implementation).name),
                     llmp::base::ToHex(recipe.registry->identity(call.implementation)));

      Require(representable || selector.kind == planner::ComparisonSiteKind::kRewrite,
              "checkpoint-only whole comparison requires an authentic rewrite ticket");
      if (preflight_only) {
        std::println(
            "{{\"event\":\"preflight_complete\",\"ordinary_lower_performed\":false,"
            "\"ready_key_transfer\":false,\"ranking_samples_collected\":false}}");
        return;
      }
      Require(selected, "requested full-GDN committed occurrence is absent");
    }
    auto inventory = checkpoint_only
                         ? planner::BoundWholeComparison(*view, selector, funder, *recipe.registry,
                                                         recipe.device, *policy, options)
                         : planner::BoundComparison(*view, selector, funder, *recipe.registry,
                                                    recipe.device, *policy, options);
    if (!inventory)
      Fail(std::format("full-GDN {} occurrence {} planning {}: {}", target_pattern,
                       target_occurrence, planner::PlanErrorName(inventory.error().error),
                       inventory.error().detail));
    const auto* actual_applied = AppliedForSite(inventory->incumbent.plan, target_pattern,
                                                inventory->site.accepted_member_origins);
    Require(actual_applied != nullptr,
            "accepted site has no unique retained actual Applied declaration");
    if (actual_applied->numerical_change) {
      Require(!actual_applied->numerical_candidate && !actual_applied->numerical_proofs.empty() &&
                  std::ranges::all_of(actual_applied->numerical_proofs,
                                      [&](const auto& proof) {
                                        return !proof.candidate &&
                                               proof.qualification.change ==
                                                   *actual_applied->numerical_change;
                                      }),
              "accepted numerical site lacks its genuine recorded production proof");
    }
    const bool numerical_pair = actual_applied->numerical_change.has_value();
    Require(!numerical_pair || (admitted || specialized),
            "new numerical fixture lacks its declared serving source contract");
    std::println(
        "{{\"event\":\"actual_site_numerics\",\"pattern\":{},\"change\":{},"
        "\"production_proofs\":{},\"diagnostic_candidate\":false}}",
        Quote(target_pattern),
        numerical_pair ? Quote(planner::NumericalChangeName(*actual_applied->numerical_change))
                       : "null",
        actual_applied->numerical_proofs.size());
    std::println(
        "{{\"event\":\"full_program_pair\",\"pattern\":{},\"occurrence\":{},"
        "\"ready_key_transfer\":false,\"adoptable\":false,"
        "\"checkpoint_only_whole\":{},\"local_boundary_representable\":{},"
        "\"checkpoint_subject\":\"{}\"}}",
        Quote(target_pattern), target_occurrence, checkpoint_only, !checkpoint_only,
        llmp::base::ToHex(inventory->site.whole_checkpoint.subject()));
    auto alternative = planner::MakeAlternative(*inventory);
    Require(alternative, "check failed: alternative");
    auto candidate = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                              recipe.device, *policy, options);
    if (!candidate)
      Fail(std::format("own alternative planning {}: {}",
                       planner::PlanErrorName(candidate.error().error), candidate.error().detail));
    // The accepted ticket's exact source groups name this occurrence inside
    // the full retained source. They grant no local Ready or remainder proof.
    std::vector<std::uint32_t> member_offsets{0};
    std::vector<ir::NodeId> members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      members.insert(members.end(), group.begin(), group.end());
      Require(members.size() <= UINT32_MAX, "full-GDN source group offset overflows");
      member_offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    auto site_boundary = co::MakeComparisonRegionBoundary(
        {.source = *view, .member_offsets = member_offsets, .members = members}, funder);
    if (!site_boundary) Fail(site_boundary.error().detail);
    std::array<kn::ComparisonRecipeSeal, 2> descriptor_seals;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto& owned_plan = arm == 0 ? inventory->incumbent : *candidate;
      const auto& plan = owned_plan.plan;
      Require((plan.placement.extent) <= (arena_bytes),
              "check failed: (plan.placement.extent) <= (arena_bytes)");
      for (const auto& selected : plan.steps) {
        const auto name = recipe.registry->at(selected.implementation).name;
        std::println(
            "{{\"event\":\"implementation\",\"arm\":{},\"name\":\"{}\",\"identity\":\"{}\"}}", arm,
            name, llmp::base::ToHex(recipe.registry->identity(selected.implementation)));
      }
      std::println(
          "{{\"event\":\"plan\",\"arm\":{},\"graph\":\"{}\",\"steps\":{},\"activation_bytes\":{},"
          "\"arena_capacity\":{},\"state_bytes\":{}}}",
          arm, llmp::base::ToHex(plan.graph_digest), plan.steps.size(), plan.placement.extent,
          arena_bytes, source->layout.bytes);
      auto arm_ranges = bindings->Resolve(plan, addresses, recipe.written);
      Require(arm_ranges, "check failed: arm_ranges");
      // Whole native proof follows the ordinary lower/validator over this
      // complete program and its actual source-scoped bound ranges.
      kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                    .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                    .bindings = arm_ranges->ranges(),
                                    .state = step.state,
                                    .active_stage = inventory->source_boundary.active_stage(),
                                    .expected_boundary = &inventory->source_boundary};
      kn::LowerInput input(plan, step, source->shape, addresses, source->model.stages[0].context,
                           options.state_spans, &request);
      auto program = recipe.lower(input);
      if (!program) Fail(std::format("full-GDN arm {} lowering: {}", arm, program.error()));
      auto& retained = programs_[arm];
      retained = std::move(*program);
      const auto descriptor = retained->comparison_recipe();
      if (!descriptor || descriptor->state != kn::ComparisonProofState::kDescriptorVerified ||
          descriptor->missing != kn::ComparisonMissing::kNone ||
          descriptor->scope != kn::ComparisonProofScope::kWholeProgramNative)
        Fail(std::format("full-GDN arm {} whole descriptor unproved: state={} missing={}", arm,
                         descriptor ? static_cast<int>(descriptor->state) : -1,
                         descriptor ? static_cast<int>(descriptor->missing) : -1));
      const auto identity = retained->comparison_plan_identity();
      Require(identity && identity->source_graph_identity == step.graph.identity() &&
                  identity->planned_graph_identity == plan.graph.identity() &&
                  owned_plan.final_node_origins.source_graph_identity ==
                      identity->source_graph_identity &&
                  owned_plan.final_node_origins.planned_graph_identity ==
                      identity->planned_graph_identity &&
                  descriptor->subject == inventory->source_boundary.source_digest() &&
                  descriptor->boundary == inventory->source_boundary.digest().combined &&
                  descriptor->planned_graph == plan.graph_digest &&
                  descriptor->policy == plan.policy_digest &&
                  descriptor->registry == recipe.registry->SelectionDigest(),
              "full-GDN whole descriptor differs from its retained source/plan");
      descriptor_seals[arm] = *descriptor;
    }
    // Construction declares the exact requirement without executing work.
    // Admit one shared pool only after both complete schedules are known.
    std::uint64_t pool_bytes = 0;
    for (const auto& program : programs_) pool_bytes = std::max(pool_bytes, program->pool_bytes());
    Check(cudaMemGetInfo(&free_bytes, &total_bytes));
    Require(pool_bytes <= free_bytes, "declared main-pool requirement exceeds fresh device memory");
    Require(pool_.Allocate(pool_bytes), "declared main-pool allocation failed");
    launch_ = Must(backend_->Launch({pool_.address(), llmp::base::Bytes(pool_bytes)}, false));
    std::println(
        "{{\"event\":\"shared_main_pool_admission\",\"pool_capacity\":{},"
        "\"incumbent_required\":{},\"alternative_required\":{}}}",
        pool_bytes, programs_[0]->pool_bytes(), programs_[1]->pool_bytes());
    for (std::size_t arm = 0; arm < 2; ++arm) {
      auto& retained = programs_[arm];
      const auto descriptor = retained->comparison_recipe();
      const auto identity = retained->comparison_plan_identity();
      Require(descriptor && identity && descriptor_seals[arm] == *descriptor,
              "whole descriptor changed before its shared scratch admission");
      auto bound = retained->Bind(*launch_);
      if (!bound) Fail(std::format("full-GDN arm {} binding: {}", arm, bound.error().detail));
      const auto bound_seal = retained->comparison_recipe();
      const auto bound_identity = retained->comparison_plan_identity();
      Require(bound_seal && en::SameWholeProgramRecipe(*descriptor, *bound_seal) &&
                  bound_identity &&
                  bound_identity->source_graph_identity == identity->source_graph_identity &&
                  bound_identity->planned_graph_identity == identity->planned_graph_identity,
              "full-GDN bound whole seal differs from its descriptor/source/plan");
      std::println(
          "{{\"event\":\"whole_program_arm_seal\",\"arm\":{},\"scope\":\"whole-program-native\","
          "\"state\":\"bound-verified\",\"missing\":\"none\",\"descriptor_bound_equal\":true,"
          "\"planned_graph\":\"{}\",\"physical_recipe\":\"{}\",\"policy\":\"{}\","
          "\"backend_recipe\":\"{}\",\"backend_build\":\"{}\",\"backend_selection\":\"{}\","
          "\"backend_device\":\"{}\",\"execution_method\":{},\"scheduled_nodes\":{}}}",
          arm, llmp::base::ToHex(bound_seal->planned_graph),
          llmp::base::ToHex(bound_seal->physical_recipe), llmp::base::ToHex(bound_seal->policy),
          llmp::base::ToHex(bound_seal->backend_recipe),
          llmp::base::ToHex(bound_seal->backend_build),
          llmp::base::ToHex(bound_seal->backend_selection),
          llmp::base::ToHex(bound_seal->backend_device),
          static_cast<int>(bound_seal->execution_method), bound_seal->scheduled_calls);
    }
    const auto whole_key = en::RecipeComparisonKeyForWholePrograms(
        inventory->source_boundary, *site_boundary, inventory->site, source->shape.phase,
        recipe.registry->SelectionDigest(), descriptor_seals[0], descriptor_seals[1]);
    Require(whole_key, "full-GDN authenticated whole programs did not produce a recipe key");
    const auto* measured = recipe.registry->FindRecipeComparison(*whole_key);
    std::println(
        "{{\"event\":\"whole_program_key\",\"recipe_key_digest\":\"{}\",\"version\":{},"
        "\"region_canonicalizer\":{},\"discovery_operation\":{},\"mode\":\"full\","
        "\"subject\":\"{}\",\"boundary\":\"{}\",\"policy\":\"{}\",\"device\":\"{}\","
        "\"selection_environment\":\"{}\",\"preparation_domain\":\"{}\","
        "\"preparation\":\"restored-state-single-invocation\",\"registered_rows\":{},"
        "\"matched\":{},\"ready_key_transfer\":false,\"ranking_samples_collected\":false}}",
        llmp::base::ToHex(llmp::execution::RecipeComparisonKeyDigest(*whole_key)),
        whole_key->version, whole_key->region_canonicalizer,
        Quote(whole_key->discovery_operation.op == ir::Op::kFused
                  ? std::format("fused.{}", ir::FusedKindName(whole_key->discovery_operation.fused))
                  : std::string(ir::OpName(whole_key->discovery_operation.op))),
        llmp::base::ToHex(whole_key->subject), llmp::base::ToHex(whole_key->boundary),
        llmp::base::ToHex(whole_key->policy), llmp::base::ToHex(whole_key->device),
        llmp::base::ToHex(whole_key->selection_environment),
        llmp::base::ToHex(whole_key->preparation_domain),
        recipe.registry->ranks().recipe_comparisons.size(), measured != nullptr);

    std::array<std::vector<std::int32_t>, 2> frame_tokens;
    for (std::size_t frame = 0; frame < frame_tokens.size(); ++frame) {
      frame_tokens[frame].resize(source->shape.rows());
      for (std::size_t row = 0; row < frame_tokens[frame].size(); ++row)
        frame_tokens[frame][row] = static_cast<std::int32_t>((3 + 2 * frame + row * (2 + frame)) %
                                                             (resident_moe ? 16 : 8));
    }
    std::array<std::vector<std::byte>, 2> reset_images;
    std::array<std::vector<float>, 2> expected_logits;
    std::array<std::vector<std::int64_t>, 2> expected_routes;
    std::array<std::map<ir::TensorId, std::vector<float>>, 2> expected_states;
    std::array<std::vector<std::uint8_t>, 2> written_bytes;
    std::array<std::map<std::string, std::vector<float>>, 2> incumbent_values;
    std::array<std::map<std::string, double>, 2> incumbent_cpu_nmse;
    const auto native = native_;
    const auto cuda_stream = static_cast<cudaStream_t>(native.handle);
    // QSA's native pool updates only newly completed block keys. Its
    // primitive expansion may also materialize unread incomplete/future keys.
    // Authenticate this distinction from the selected owned plan, never from
    // the requested pattern or from differences in the resulting bytes.
    struct CompletedBlockCache {
      ir::TensorId tensor;
      std::uint32_t ratio;
      std::uint8_t segment;
      std::array<bool, 2> native{};
    };
    std::vector<CompletedBlockCache> completed_caches;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto& plan = arm == 0 ? inventory->incumbent.plan : candidate->plan;
      for (const auto& selected : plan.steps) {
        if (recipe.registry->at(selected.implementation).name != "llmp.qsa.pool") continue;
        Require(selected.nodes.size() == 1, "QSA pool needs one exact selected producer");
        const auto id = selected.nodes.front();
        const auto& node = plan.graph.node(id);
        Require(node.key == ir::Fused(ir::FusedKind::kMeanPoolNormRopeStore) && node.nsrc == 6,
                "QSA cache contract lacks its typed selected pool");
        const auto params = node.Params<ir::BlockPoolParams>();
        const auto block_root = plan.graph.root(node.src[1]);
        const auto state = std::ranges::find_if(step.state, [&](const auto& value) {
          return value.tensor.v < plan.tensors.size() && plan.tensors[value.tensor.v].valid() &&
                 plan.graph.root(plan.tensors[value.tensor.v]) == block_root;
        });
        Require(state != step.state.end(), "QSA pool block root lacks its source state");
        const auto& item = source->layout.items[state->item];
        Require(item.contract.kind == llmp::model::StateKind::kBlockKeys &&
                    item.contract.encoding == llmp::format::Encoding::kBF16 &&
                    item.contract.cells == llmp::model::CellRule::kPerBlock &&
                    item.contract.basis == llmp::model::BlockBasis::kReadCells &&
                    params.ratio != 0 && item.contract.block_ratio == params.ratio,
                "QSA pool state is not the canonical typed BF16 block cache");
        const auto mapped_input = [&](std::size_t operand, co::InputKind kind) {
          return std::ranges::find_if(step.inputs, [&](const auto& input) {
            return input.kind == kind &&
                   input.segment == (kind == co::InputKind::kPositions ? ir::kJoinedSegment
                                                                       : state->ref.segment) &&
                   input.tensor.v < plan.tensors.size() && plan.tensors[input.tensor.v].valid() &&
                   plan.graph.root(plan.tensors[input.tensor.v]) ==
                       plan.graph.root(node.src[operand]);
          });
        };
        const auto positions = mapped_input(5, co::InputKind::kPositions);
        const auto block_positions = mapped_input(3, co::InputKind::kBlockPositions);
        const auto cells = mapped_input(4, co::InputKind::kStoreCells);
        Require(positions != step.inputs.end() && positions->sections == 4 &&
                    block_positions != step.inputs.end() && block_positions->sections == 4 &&
                    block_positions->cells.block == params.ratio && cells != step.inputs.end() &&
                    cells->cells.block == params.ratio && cells->cells.capacity == item.cells,
                "QSA pool positions/block cells lost their original owner recipe");
        const auto raw_root = plan.graph.root(node.src[0]);
        bool raw_store = false, block_store = false;
        for (const auto& original : step.graph.nodes()) {
          if (original.key != ir::Primitive(ir::Op::kCacheStore)) continue;
          const auto root = step.graph.root(original.src[0]);
          if (root == state->tensor) {
            Require(original.src[2] == cells->tensor, "QSA block store changed its index recipe");
            block_store = true;
          }
          const auto raw_state = std::ranges::find(step.state, root, &co::StepState::tensor);
          if (raw_state == step.state.end() || root.v >= plan.tensors.size() ||
              !plan.tensors[root.v].valid() || plan.graph.root(plan.tensors[root.v]) != raw_root)
            continue;
          const auto input =
              std::ranges::find(step.inputs, original.src[2], &co::StepInput::tensor);
          Require(raw_state->ref.segment == state->ref.segment && input != step.inputs.end() &&
                      input->kind == co::InputKind::kStoreCells &&
                      input->segment == state->ref.segment && input->cells.block == 0,
                  "QSA raw store lacks its canonical same-owner token cells");
          raw_store = true;
        }
        Require(raw_store && block_store, "QSA pool lacks original raw/block stores");
        for (std::size_t other = 0; other < plan.graph.nodes().size(); ++other) {
          if (other == id.v) continue;
          ir::ForEachWrite(plan.graph, plan.graph.nodes()[other], [&](ir::TensorId root) {
            Require(root != block_root, "QSA cache has another planned writer");
          });
        }
        auto existing =
            std::ranges::find(completed_caches, state->tensor, &CompletedBlockCache::tensor);
        if (existing == completed_caches.end()) {
          completed_caches.push_back({state->tensor, params.ratio, state->ref.segment, {}});
          existing = std::prev(completed_caches.end());
        }
        Require(existing->ratio == params.ratio && existing->segment == state->ref.segment &&
                    !existing->native[arm],
                "QSA cache pool authority is ambiguous");
        existing->native[arm] = true;
      }
    }
    const auto stage = [&](std::size_t arm, std::size_t frame) {
      auto& program = programs_[arm];
      // NaN-poison the complete arena so this arm cannot inherit valid logits
      // or temporaries from its peer if an expected write is missing.
      Check(cudaMemsetAsync(activation_.pointer, 0xff, arena_bytes, cuda_stream));
      // Every state format uses the same exact retained encoded image per arm.
      if (!reset_images[frame].empty())
        Check(cudaMemcpyAsync(state_.pointer, reset_images[frame].data(),
                              reset_images[frame].size(), cudaMemcpyHostToDevice, cuda_stream));
      std::vector<std::vector<std::byte>> payloads;
      payloads.reserve(program->inputs().size());
      for (const auto& slot : program->inputs()) {
        Require((slot.bytes) > (0U), "check failed: (slot.bytes) > (0U)");
        payloads.emplace_back(static_cast<std::size_t>(slot.bytes));
        auto& payload = payloads.back();
        if (slot.kind == kn::InputKind::kTokens || slot.kind == kn::InputKind::kPositions) {
          const auto input = std::ranges::find(step.inputs, slot.tensor, &co::StepInput::tensor);
          Require((input) != (step.inputs.end()), "check failed: (input) != (step.inputs.end())");
          const auto values = co::InputValues(*input, source->shape, frame_tokens[frame]);
          Require(values, "check failed: values");
          std::vector<std::int32_t> narrowed(values->size());
          std::ranges::transform(*values, narrowed.begin(), [](std::int64_t value) {
            return static_cast<std::int32_t>(value);
          });
          Require(en::step_inputs::WriteRows(payload, narrowed),
                  "check failed: en::step_inputs::WriteRows(payload, narrowed)");
        } else {
          auto written = en::step_inputs::WriteComputed(slot, source->shape, payload);
          Require(written, "check failed: written");
        }
        Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), payload.data(),
                              payload.size(), cudaMemcpyHostToDevice, cuda_stream));
      }
      // The owned stream orders every staging operation before the body;
      // completion also retains all host payloads until their copies finish.
      Finish();
    };
    std::array<bool, 2> cpu_quality{true, true};
    bool peer_quality = true;
    llmp::base::Sha256 quality_receipt;
    constexpr std::string_view gdn_quality_protocol =
        "llmp.u6.whole-quality-rejection.v1;fixed-source-payloads;two-changed-frames;"
        "ordinary-both-arms;complete-outputs-full-and-written-states;"
        "cpu-nmse=3e-3;peer-nmse=3e-3-reported-not-required;"
        "incumbent-cpu-fails;alternative-all-cpu-passes;"
        "finite-ownership-untouched-state-weights-immediate;"
        "passing-alternative-only-capture-when-authorized;no-timings";
    constexpr std::string_view qsa_quality_protocol =
        "llmp.u6.whole-quality-rejection.qsa-mxfp8-inputs.v1;fixed-admitted-qsa17-source-payloads;"
        "two-changed-frames;ordinary-both-arms;complete-outputs;"
        "cache-contract=completed-block-cache-v1;cpu-nmse=3e-3;"
        "peer-nmse=3e-3-reported-not-required;"
        "incumbent-cpu-fails;alternative-all-cpu-passes;"
        "finite-ownership-exact-unwritten-cache-state-weights-immediate;"
        "passing-alternative-only-capture-when-authorized;no-record-on-failure;no-timings";
    constexpr std::string_view nvfp4_quality_protocol =
        "llmp.u6.whole-quality-rejection.nvfp4-prefill.v1;fixed-source-payloads;"
        "two-changed-frames;ordinary-both-arms;complete-outputs-full-and-written-states;"
        "cpu-nmse=3e-2;peer-nmse=1e-3-and-cpu-nonregression=1.05-reported-not-required;"
        "incumbent-cpu-fails;alternative-all-cpu-passes;"
        "finite-ownership-untouched-state-weights-immediate;"
        "passing-alternative-only-capture-when-authorized;no-timings";
    constexpr std::string_view deepseek_quality_protocol =
        "llmp.u6.whole-quality-rejection.deepseek-statebank.v1;fixed-source-payloads;"
        "two-changed-frames;ordinary-both-arms;complete-outputs-full-and-written-states;"
        "cpu-nmse=5e-4;structural-peer-normalized-max=1e-5-reported-not-required;"
        "incumbent-cpu-fails;alternative-all-cpu-passes;"
        "finite-ownership-untouched-state-weights-immediate;"
        "passing-alternative-only-capture-when-authorized;no-timings";
    constexpr std::string_view bf16_row_attribution_protocol =
        "llmp.u6.whole-quality-attribution.bf16-rows.v1;fixed-gdn17-source-payloads;"
        "exact-default-occurrence-and-own-alternative;two-changed-frames;ordinary-both-arms;"
        "complete-outputs-full-and-written-states;cpu-nmse=3e-3;"
        "peer-nmse=3e-3-reported;finite-ownership-untouched-state-weights-immediate;"
        "no-capture;no-prior-whole-choice;no-adoption;no-record;no-timings";
    constexpr std::string_view inverse_attribution_protocol =
        "llmp.u6.whole-quality-attribution.inverse-rope-grouped.v1;fixed-deepseek64-source-"
        "payloads;"
        "exact-default-occurrence-and-own-alternative;two-changed-frames;ordinary-both-arms;"
        "complete-outputs-full-and-written-states;cpu-nmse=5e-4;"
        "structural-peer-normalized-max=1e-5-reported;finite-ownership-untouched-state-weights-"
        "immediate;"
        "no-capture;no-prior-whole-choice;no-adoption;no-record;no-timings";
    constexpr std::string_view qsa_attribution_protocol =
        "llmp.u6.whole-quality-attribution.qsa-mean-pool.v1;fixed-admitted-qsa17-source-payloads;"
        "exact-default-occurrence-and-own-alternative;two-changed-frames;ordinary-both-arms;"
        "complete-outputs;cache-contract=completed-block-cache-v1;cpu-nmse=3e-3;"
        "actual-numerical-peer-nmse=3e-3-or-structural-peer-normalized-max=1e-5-reported;"
        "finite-ownership-exact-unwritten-cache-state-weights-immediate;"
        "no-capture;no-prior-whole-choice;no-adoption;no-record;no-timings";
    const auto quality_protocol = quality_attribution && qsa_attribution ? qsa_attribution_protocol
                                  : quality_attribution && inverse_attribution
                                      ? inverse_attribution_protocol
                                  : quality_attribution ? bf16_row_attribution_protocol
                                  : quality_qsa         ? qsa_quality_protocol
                                  : quality_deepseek    ? deepseek_quality_protocol
                                  : quality_nvfp4       ? nvfp4_quality_protocol
                                                        : gdn_quality_protocol;
    if (quality_attribution)
      Require(
          (qsa_attribution && mxfp8_chain && reference_bound == 3.0e-3) ||
              (inverse_attribution && reference_bound == 5.0e-4 && !numerical_pair) ||
              (mxfp8_chain && reference_bound == 3.0e-3 &&
               actual_applied->numerical_change == ir::NumericalChange::kBf16ProductInputAndResult),
          "attribution lacks its exact registered source and actual numerical declaration");
    if (quality_rejection)
      Require(((quality_gdn || quality_qsa) && mxfp8_chain && reference_bound == 3.0e-3 &&
               actual_applied->numerical_change == ir::NumericalChange::kMxfp8ProductInput) ||
                  (quality_nvfp4 && admitted_nvfp4 && reference_bound == 3.0e-2 &&
                   actual_applied->numerical_change ==
                       ir::NumericalChange::kNvfp4ExpertIntermediates) ||
                  (quality_deepseek && reference_bound == 5.0e-4 && !numerical_pair),
              "quality rejection lacks its actual prospectively declared numerical proof");
    quality_receipt.Update(quality_protocol);
    quality_receipt.Update(
        llmp::base::ToHex(llmp::execution::RecipeComparisonKeyDigest(*whole_key)));
    const auto compare = [&](std::span<const float> actual, std::span<const float> expected,
                             std::size_t arm, std::size_t frame, std::string_view tensor, bool emit,
                             bool check_peer = true) {
      Require(!actual.empty() && actual.size() == expected.size(),
              "full reference array extent differs");
      double worst = 0.0, scale = 1.0, squared_error = 0.0, reference_energy = 0.0;
      std::size_t worst_index = 0;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        Require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
                "nonfinite full-program result");
        const auto gap = std::abs(static_cast<double>(actual[i]) - expected[i]);
        squared_error += gap * gap;
        reference_energy += static_cast<double>(expected[i]) * expected[i];
        if (gap > worst) {
          worst = gap;
          worst_index = i;
        }
        scale = std::max(scale, std::abs(static_cast<double>(expected[i])));
      }
      const double nmse = reference_energy > 0 ? squared_error / reference_energy : squared_error;
      // Prospective Q8 contract: ggml_ext_ops_test and qwen38_moe_test NMSE,
      // independently complemented by strict full-output production peer equality.
      const double cpu_bound = reference_bound;
      const bool accepted = reference_nmse ? nmse <= cpu_bound : worst / scale <= cpu_bound;
      if (emit || !accepted)
        std::println(
            "{{\"event\":\"validation\",\"frame\":{},\"arm\":{},\"tensor\":\"{}\",\"elements\":{},"
            "\"normalized_max_gap\":{},\"max_abs_gap\":{},\"reference_scale\":{},"
            "\"worst_index\":{},\"actual_at_worst\":{},\"reference_at_worst\":{},"
            "\"normalized_squared_error\":{},\"metric\":\"{}\",\"bound\":{},\"accepted\":{}}}",
            frame, arm, tensor, actual.size(), worst / scale, worst, scale, worst_index,
            actual[worst_index], expected[worst_index], nmse, reference_metric, cpu_bound,
            accepted);
      if (defer_numeric_quality) {
        cpu_quality[arm] &= accepted;
        quality_receipt.Update(
            std::format("{}:{}:{}:{}:{};", frame, arm, tensor, actual.size(), accepted));
        quality_receipt.Update(std::as_bytes(actual));
        quality_receipt.Update(std::as_bytes(expected));
      } else {
        Require(accepted, "full-program reference gap exceeds declared source bound");
      }
      if (check_peer && (f16_moe || q8_moe || admitted || state_chain)) {
        const std::string label(tensor);
        if (arm == 0) {
          incumbent_values[frame][label].assign(actual.begin(), actual.end());
          incumbent_cpu_nmse[frame][label] = nmse;
        } else {
          const auto& peer = incumbent_values[frame].at(label);
          Require(peer.size() == actual.size(), "complete peer extent differs");
          double squared = 0, energy = 0, max_gap = 0, max_ref = 1;
          for (std::size_t i = 0; i < actual.size(); ++i) {
            const double delta = static_cast<double>(actual[i]) - peer[i];
            squared += delta * delta;
            energy += static_cast<double>(peer[i]) * peer[i];
            max_gap = std::max(max_gap, std::abs(delta));
            max_ref = std::max(max_ref, std::abs(static_cast<double>(actual[i])));
          }
          const double peer_nmse = energy > 0 ? squared / energy : squared;
          const bool q8_change =
              actual_applied->numerical_change == ir::NumericalChange::kQ8VectorReduction;
          const double peer_bound = state_chain                    ? (q8_change ? 5.0e-4 : 3.0e-3)
                                    : admitted_nvfp4 && rows == 17 ? 1.0e-3
                                                                   : 5.0e-4;
          const bool quality_nonregression =
              !admitted_nvfp4 || rows != 17 || incumbent_cpu_nmse[frame].at(label) <= 1.05 * nmse;
          const bool peer_accepted =
              (numerical_pair ? peer_nmse <= peer_bound : max_gap / max_ref <= 1.0e-5) &&
              quality_nonregression;
          if (emit || !peer_accepted)
            std::println(
                "{{\"event\":\"production_peer_validation\",\"frame\":{},"
                "\"tensor\":{},\"normalized_max_gap\":{},\"nmse\":{},\"metric\":{},"
                "\"bound\":{},\"quality_nonregression\":{},\"accepted\":{}}}",
                frame, Quote(tensor), max_gap / max_ref, peer_nmse,
                Quote(numerical_pair ? "nmse" : "normalized-max"),
                numerical_pair ? peer_bound : 1.0e-5, quality_nonregression, peer_accepted);
          if (defer_numeric_quality)
            peer_quality &= peer_accepted;
          else
            Require(peer_accepted, "complete production peer exceeds prospective source contract");
        }
      }
    };
    const auto validate = [&](std::size_t arm, std::size_t frame, bool emit) {
      const auto& program = programs_[arm];
      const auto output_tensor = step.graph.outputs().front();
      const auto width = step.graph.tensor(output_tensor).ne(0);
      std::vector<float> actual(expected_logits[frame].size(),
                                std::numeric_limits<float>::quiet_NaN());
      std::vector<bool> covered(actual.size());
      Require(!program->outputs().empty(), "full source must return its complete output");
      for (const auto& output : program->outputs()) {
        Require(
            output.tensor == output_tensor && !output.features && output.bytes % sizeof(float) == 0,
            "normal output slot differs from its original logits tensor");
        std::uint64_t first = 0;
        if (output.segment != ir::kJoinedSegment) {
          Require(output.segment < source->shape.segments.size(),
                  "output slot names no source owner");
          for (std::size_t owner = 0; owner < output.segment; ++owner)
            first += source->shape.segments[owner].outputs;
          Require(output.bytes ==
                      width * source->shape.segments[output.segment].outputs * sizeof(float),
                  "owner output slot has a different source extent");
        } else {
          Require(output.bytes == actual.size() * sizeof(float), "joined output extent differs");
        }
        const auto offset = first * width;
        const auto count = output.bytes / sizeof(float);
        Require(offset <= actual.size() && count <= actual.size() - offset,
                "output slot exceeds its complete original array");
        for (std::uint64_t i = 0; i < count; ++i) {
          Require(!covered[offset + i], "normal output slots overlap in source order");
          covered[offset + i] = true;
        }
        Check(cudaMemcpy(actual.data() + offset,
                         reinterpret_cast<const void*>(output.device_address), output.bytes,
                         cudaMemcpyDeviceToHost));
      }
      Require(std::ranges::all_of(covered, [](bool value) { return value; }),
              "normal output slots omit original output values");
      if (emit)
        std::println(
            "{{\"event\":\"complete_output_inventory\",\"arm\":{},\"frame\":{},"
            "\"slots\":{},\"elements\":{},\"ordered_coverage_complete\":true}}",
            arm, frame, program->outputs().size(), actual.size());
      compare(actual, expected_logits[frame], arm, frame, "logits", emit);
      for (const auto& weight : step.weights) {
        for (const auto& [part, expected_weight] : weight_bytes.at(weight.tensor)) {
          std::vector<std::byte> observed(expected_weight.size());
          const auto base = weights.at({weight.ref, part}).base;
          const auto strided = strided_parts.find({weight.ref, part});
          if (strided == strided_parts.end()) {
            Check(cudaMemcpy(observed.data(), reinterpret_cast<const void*>(base), observed.size(),
                             cudaMemcpyDeviceToHost));
          } else {
            const auto [pitch, bytes] = strided->second;
            for (std::uint64_t i = 0; i < observed.size() / bytes; ++i)
              Check(cudaMemcpy(observed.data() + i * bytes,
                               reinterpret_cast<const void*>(base + i * pitch), bytes,
                               cudaMemcpyDeviceToHost));
          }
          Require(std::ranges::equal(observed, expected_weight),
                  "ordinary full program modified immutable native weight part bytes");
        }
      }
      for (const auto& slab : immutable_slabs) {
        std::vector<std::byte> actual_slab(slab.bytes.size());
        Check(cudaMemcpy(actual_slab.data(), reinterpret_cast<const void*>(slab.address),
                         actual_slab.size(), cudaMemcpyDeviceToHost));
        Require(std::ranges::equal(actual_slab, slab.bytes),
                "native expert slab or gaps were modified");
      }
      std::vector<std::byte> observed_state(source->layout.bytes);
      if (!observed_state.empty())
        Check(cudaMemcpy(observed_state.data(), state_.pointer, observed_state.size(),
                         cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < observed_state.size(); ++i)
        if (!written_bytes[frame][i])
          Require(observed_state[i] == reset_images[frame][i],
                  "program modified untouched state cell/padding/layout gap");
      for (const auto& state : step.state) {
        const auto& item = source->layout.items[state.item];
        const auto& format = step.graph.tensor(state.tensor).format;
        const auto raw =
            std::span<const std::byte>(observed_state).subspan(item.offset, item.bytes);
        const std::array parts{
            reference::PartView{.role = llmp::format::PartRole::kData, .bytes = raw}};
        actual.resize(ir::Elements(format.shape));
        const auto elements = format.shape.ne[0] * format.shape.ne[1];
        for (std::uint64_t matrix = 0; matrix < llmp::format::Matrices(format); ++matrix)
          Must(reference::DecodeRows(format, parts, matrix, 0, format.shape.ne[1],
                                     std::span(actual).subspan(matrix * elements, elements)));
        const auto label = std::format("state-{}", state.tensor.v);
        const auto cache =
            std::ranges::find(completed_caches, state.tensor, &CompletedBlockCache::tensor);
        if (cache != completed_caches.end()) {
          Require(cache->segment < source->shape.segments.size(),
                  "QSA cache owner frame is absent");
          const auto& segment = source->shape.segments[cache->segment];
          const std::uint64_t first = segment.past / cache->ratio;
          const std::uint64_t end = (std::uint64_t{segment.past} + segment.rows) / cache->ratio;
          const auto cache_width = format.shape.ne[0];
          Require(first <= end && end <= item.cells && end * cache_width <= actual.size(),
                  "QSA completed cache extent exceeds its owned state");
          const auto cache_row_bytes =
              Must(llmp::format::RowBytes(format, llmp::format::PartRole::kData));
          if (emit) {
            const auto original_written = std::ranges::count_if(
                std::span(written_bytes[frame]).subspan(item.offset, item.bytes),
                [](auto value) { return value != 0; });
            std::println(
                "{{\"event\":\"qsa_cache_obligation\",\"contract\":\"completed-block-cache-v1\","
                "\"arm\":{},\"frame\":{},\"tensor\":{},\"ratio\":{},\"first\":{},\"end\":{},"
                "\"native_pool\":{},\"completed_cpu_peer_elements\":{},"
                "\"new_written_cpu_peer_elements\":{},\"exact_unwritten_bytes\":{},"
                "\"primitive_full_cpu_elements\":{}}}",
                arm, frame, state.tensor.v, cache->ratio, first, end, cache->native[arm],
                end * cache_width, (end - first) * cache_width,
                cache->native[arm] ? item.bytes - (end - first) * cache_row_bytes
                                   : item.bytes - static_cast<std::uint64_t>(original_written),
                cache->native[arm] ? 0 : actual.size());
          }
          if (end != 0)
            compare(std::span(actual).first(end * cache_width),
                    std::span(expected_states[frame].at(state.tensor)).first(end * cache_width),
                    arm, frame, label + "-completed", emit);
          if (first != end)
            compare(std::span(actual).subspan(first * cache_width, (end - first) * cache_width),
                    std::span(expected_states[frame].at(state.tensor))
                        .subspan(first * cache_width, (end - first) * cache_width),
                    arm, frame, label + "-completed-written", emit);
          if (cache->native[arm]) {
            const auto row_bytes =
                Must(llmp::format::RowBytes(format, llmp::format::PartRole::kData));
            Require(row_bytes == cache_width * 2 && item.contract.cell_bytes == row_bytes,
                    "QSA cache rows are not exact packed BF16 cells");
            for (std::size_t byte = 0; byte < item.bytes; ++byte)
              if (byte < first * row_bytes || byte >= end * row_bytes)
                Require(raw[byte] == reset_images[frame][item.offset + byte],
                        "QSA native pool changed an unwritten cache byte");
            continue;
          }
        }
        compare(actual, expected_states[frame].at(state.tensor), arm, frame, label, emit,
                cache == completed_caches.end());
        std::vector<float> written_actual, written_expected;
        const auto row_bytes = Must(llmp::format::RowBytes(format, llmp::format::PartRole::kData));
        for (std::size_t row = 0; row < actual.size() / format.shape.ne[0]; ++row) {
          const auto byte = item.offset + row * row_bytes;
          if (!written_bytes[frame][byte]) continue;
          Require(std::ranges::all_of(std::span(written_bytes[frame]).subspan(byte, row_bytes),
                                      [](auto b) { return b != 0; }),
                  "logical state write cuts a root row");
          const auto first = row * format.shape.ne[0];
          const auto actual_row = std::span(actual).subspan(first, format.shape.ne[0]);
          written_actual.insert(written_actual.end(), actual_row.begin(), actual_row.end());
          const auto expected_row =
              std::span(expected_states[frame].at(state.tensor)).subspan(first, format.shape.ne[0]);
          written_expected.insert(written_expected.end(), expected_row.begin(), expected_row.end());
        }
        if (!written_actual.empty())
          compare(written_actual, written_expected, arm, frame, label + "-written", emit,
                  cache == completed_caches.end());
      }
    };
    for (std::size_t frame = 0; frame < 2; ++frame) {
      if (frame != 0) {
        for (auto& [tensor, values] : initial_values) {
          if (!source_qsa)
            for (float& value : values) value = value * -0.75F + 0.001F;
          const auto state = std::ranges::find(step.state, tensor, &co::StepState::tensor);
          Require(state != step.state.end(), "changed frame state tensor is absent");
          encode_state(*state, values);
        }
      }
      const auto& tokens = frame_tokens[frame];
      auto cpu = reference::Executor::Create(step.graph);
      Require(cpu, "check failed: cpu");
      for (const auto& weight : step.weights) {
        std::vector<reference::PartView> parts;
        for (const auto& [role, bytes] : weight_bytes.at(weight.tensor))
          parts.push_back({.role = role, .bytes = bytes});
        Require(cpu->Bind(weight.tensor, parts), "CPU must bind actual encoded weight bytes");
      }
      for (const auto tensor : negative_decay_leaves) {
        const auto values = Must(cpu->Values(tensor));
        Require(std::ranges::all_of(values, [](float x) { return std::isfinite(x) && x < 0; }),
                "actual encoded GDN decay leaf lost its negative folded domain");
      }
      for (const auto& state : step.state) {
        const auto& item = source->layout.items[state.item];
        const std::array part{reference::PartView{
            .role = llmp::format::PartRole::kData,
            .bytes = std::span<const std::byte>(initial_state).subspan(item.offset, item.bytes)}};
        Must(cpu->Bind(state.tensor, part));
      }
      for (const auto& input : step.inputs) {
        const auto values = co::InputValues(input, source->shape, tokens);
        if (values) {
          Must(cpu->BindInts(input.tensor, *values));
        } else if (input.kind == co::InputKind::kHadamard) {
          Must(cpu->BindValues(input.tensor, co::HadamardValues(input.read)));
        } else {
          const auto slot =
              std::ranges::find(programs_[0]->inputs(), input.tensor, &kn::InputSlot::tensor);
          Require(slot != programs_[0]->inputs().end(),
                  "original host input has no normal prepared slot");
          std::vector<std::byte> payload(slot->bytes);
          Must(en::step_inputs::WriteComputed(*slot, source->shape, payload));
          const std::array part{
              reference::PartView{.role = llmp::format::PartRole::kData, .bytes = payload}};
          Must(cpu->Bind(input.tensor, part));
        }
      }
      Require(cpu->Run(), "check failed: cpu->Run()");
      if (resident_moe) {
        const auto route = std::ranges::find_if(step.graph.nodes(), [](const ir::Node& node) {
          return node.key == ir::Primitive(ir::Op::kRoute);
        });
        Require(route != step.graph.nodes().end(), "resident MoE original route is absent");
        const auto ids = Must(cpu->Ints(route->out[0]));
        Require(ids.size() == ir::Elements(step.graph.tensor(route->out[0]).shape()) &&
                    std::ranges::all_of(ids,
                                        [&](auto id) {
                                          return id >= 0 &&
                                                 static_cast<std::uint64_t>(id) < expert_count;
                                        }),
                "CPU original route has invalid expert IDs or extent");
        expected_routes[frame].assign(ids.begin(), ids.end());
        if (frame != 0)
          Require(expected_routes[frame] != expected_routes[0],
                  "changed original stream did not change its actual CPU expert route");
        std::print("{{\"event\":\"reference_route\",\"frame\":{},\"ids\":[", frame);
        for (std::size_t i = 0; i < ids.size(); ++i) std::print("{}{}", i == 0 ? "" : ",", ids[i]);
        std::println("]}}");
      }
      if (f16_moe && frame == 0) {
        // A-priori local dot bound from decode_gpu_test.cc, independent of
        // observed GPU error. This is not a propagated full-network bound.
        for (const auto& node : step.graph.nodes()) {
          if (node.key != ir::Primitive(ir::Op::kMatMulId)) continue;
          const auto& a = step.graph.tensor(node.src[0]);
          const auto& b = step.graph.tensor(node.src[1]);
          const auto w = Must(cpu->Values(node.src[0]));
          const auto x = Must(cpu->Values(node.src[1]));
          const auto ids = Must(cpu->Ints(node.src[2]));
          const auto used = step.graph.tensor(node.src[2]).ne(0);
          double maximum_bound = 0.0;
          for (std::size_t ti = 0; ti < ids.size(); ++ti) {
            Require(ids[ti] >= 0 && static_cast<std::uint64_t>(ids[ti]) < a.ne(2),
                    "F16 dot diagnostic expert is invalid");
            for (std::uint64_t out = 0; out < a.ne(1); ++out) {
              double sum_products = 0.0, sum_weights = 0.0;
              for (std::uint64_t k = 0; k < a.ne(0); ++k) {
                const auto wi = (static_cast<std::uint64_t>(ids[ti]) * a.ne(1) + out) * a.ne(0) + k;
                const auto xi = ((ti / used) * b.ne(1) + (ti % used) % b.ne(1)) * b.ne(0) + k;
                Require(wi < w.size() && xi < x.size(), "F16 dot diagnostic extent differs");
                sum_products += std::abs(static_cast<double>(w[wi]) * x[xi]);
                sum_weights += std::abs(static_cast<double>(w[wi]));
              }
              maximum_bound = std::max(
                  maximum_bound, ((static_cast<double>(a.ne(0)) + 4.0) * std::ldexp(1.0, -24) +
                                  std::ldexp(1.0, -11)) *
                                         sum_products +
                                     std::ldexp(1.0, -24) * sum_weights);
            }
          }
          std::println(
              "{{\"event\":\"f16_grouped_dot_bound\",\"tensor\":{},"
              "\"k\":{},\"maximum_local_absolute_bound\":{},"
              "\"includes_upstream_error\":false}}",
              node.out[0].v, a.ne(0), maximum_bound);
        }
      }
      written_bytes[frame].assign(initial_state.size(), 0);
      for (const auto& node : step.graph.nodes()) {
        ir::ForEachWrittenSource(node, [&](std::size_t operand) {
          const auto destination = node.src[operand];
          const auto root = step.graph.root(destination);
          const auto state = std::ranges::find(step.state, root, &co::StepState::tensor);
          if (state == step.state.end()) return;
          Require(node.key == ir::Primitive(ir::Op::kCacheStore) && operand == 0,
                  "original state writer needs an explicit logical write-mask contract");
          const auto& dst = step.graph.tensor(destination);
          const auto& src = step.graph.tensor(node.src[1]);
          const auto& ids_desc = step.graph.tensor(node.src[2]);
          const auto ids = Must(cpu->Ints(node.src[2]));
          const auto& item = source->layout.items[state->item];
          const auto bytes =
              Must(llmp::format::RowBytes(dst.format, llmp::format::PartRole::kData));
          for (std::uint64_t i3 = 0; i3 < src.ne(3); ++i3)
            for (std::uint64_t i2 = 0; i2 < src.ne(2); ++i2)
              for (std::uint64_t i1 = 0; i1 < src.ne(1); ++i1) {
                const auto index = i1 + ids_desc.ne(0) * (i2 % ids_desc.ne(1) +
                                                          ids_desc.ne(1) * (i3 % ids_desc.ne(2)));
                Require(index < ids.size() && ids[index] >= 0 &&
                            static_cast<std::uint64_t>(ids[index]) < dst.ne(1),
                        "original logical state write index is invalid");
                const auto offset = item.offset + dst.view_offset +
                                    static_cast<std::uint64_t>(ids[index]) * dst.nb[1] +
                                    i2 * dst.nb[2] + i3 * dst.nb[3];
                Require(offset <= written_bytes[frame].size() &&
                            bytes <= written_bytes[frame].size() - offset,
                        "logical state write exceeds actual owned layout");
                std::ranges::fill(std::span(written_bytes[frame]).subspan(offset, bytes), 1);
              }
        });
      }
      reset_images[frame] = initial_state;
      if (defer_numeric_quality) {
        quality_receipt.Update(std::format("frame:{};", frame));
        quality_receipt.Update(reset_images[frame]);
        quality_receipt.Update(std::as_bytes(std::span(frame_tokens[frame])));
      }
      const auto logits = Must(cpu->Values(step.graph.outputs().front()));
      expected_logits[frame].assign(logits.begin(), logits.end());
      Require(!logits.empty() &&
                  std::ranges::all_of(logits, [](float x) { return std::isfinite(x); }) &&
                  std::ranges::any_of(logits, [](float x) { return std::abs(x) > 1.0e-8F; }),
              "complete original output must be finite and nontrivial");
      if (resident_moe) {
        double amplitude = 0;
        for (const auto value : logits) {
          Require(std::isfinite(value), "CPU complete resident output is nonfinite");
          amplitude = std::max(amplitude, std::abs(static_cast<double>(value)));
        }
        Require(amplitude >= 100 * tolerance,
                "CPU complete resident output amplitude is too small for its declared bound");
        std::println(
            "{{\"event\":\"reference_output\",\"frame\":{},\"elements\":{},"
            "\"max_abs\":{},\"minimum_amplitude\":{}}}",
            frame, logits.size(), amplitude, 100 * tolerance);
      }
      std::size_t changed_state_bytes = 0;
      for (const auto& state : step.state) {
        const auto& item = source->layout.items[state.item];
        const auto bytes = cpu->Bytes(state.tensor);
        Require(bytes.size() <= item.bytes, "CPU state data exceeds owned item");
        for (std::size_t i = 0; i < bytes.size(); ++i)
          changed_state_bytes += bytes[i] != initial_state[item.offset + i];
        const auto values = Must(cpu->Values(state.tensor));

        expected_states[frame][state.tensor].assign(values.begin(), values.end());
      }
      const auto written_count = std::ranges::count(written_bytes[frame], std::uint8_t{1});
      Require(written_count == 0 || changed_state_bytes != 0,
              "written original state fixture must exercise a nontrivial update");
      std::println(
          "{{\"event\":\"original_state_inventory\",\"frame\":{},\"complete_bytes\":{},"
          "\"logical_written_bytes\":{},\"changed_reference_bytes\":{},"
          "\"initial_encoded_bytes_decoded_by_reference\":true}}",
          frame, initial_state.size(), written_count, changed_state_bytes);
      if (frame != 0) {
        Require(expected_logits[frame] != expected_logits[0],
                "changed original frame did not change complete CPU logits");
      }
      for (std::size_t arm = 0; arm < 2; ++arm) {
        stage(arm, frame);
        Must(programs_[arm]->Run(*launch_, native));
        Finish();
        validate(arm, frame, true);
      }
    }
    if (quality_attribution) {
      const auto protocol = llmp::base::Sha256{}.Update(quality_protocol).Finish();
      std::println(
          "{{\"event\":\"quality_pair_attribution\",\"source_contract\":{},"
          "\"pattern\":{},\"occurrence\":{},\"incumbent_cpu_passed\":{},"
          "\"alternative_cpu_passed\":{},\"peer_passed\":{},"
          "\"ordinary_numeric_gates_passed\":{},\"recipe_key_digest\":{},"
          "\"protocol\":{},\"receipt\":{},\"metric\":{},\"bound\":{},"
          "\"frames\":2,\"arms\":2,\"ordinary_only_attribution\":true,"
          "\"prior_whole_choice_replayed\":false,\"record_emitted\":false,"
          "\"adoptable\":false,\"captured\":false,\"timing_samples\":0}}",
          Quote(contract), Quote(target_pattern), target_occurrence, cpu_quality[0], cpu_quality[1],
          peer_quality, cpu_quality[0] && cpu_quality[1] && peer_quality,
          Quote(llmp::base::ToHex(llmp::execution::RecipeComparisonKeyDigest(*whole_key))),
          Quote(llmp::base::ToHex(protocol)), Quote(llmp::base::ToHex(quality_receipt.Finish())),
          Quote(reference_metric), reference_bound);
      return;
    }
    if (quality_rejection) {
      const bool qualifies_rejection =
          (numerical_pair || quality_deepseek) && !cpu_quality[0] && cpu_quality[1];
      std::println(
          "{{\"event\":\"quality_rejection_qualification\",\"source_contract\":{},"
          "\"incumbent_cpu_passed\":{},\"alternative_cpu_passed\":{},\"peer_passed\":{},"
          "\"qualifies_rejection\":{},\"record_emitted\":false,\"timing_samples\":0}}",
          Quote(contract), cpu_quality[0], cpu_quality[1], peer_quality, qualifies_rejection);
      Require(qualifies_rejection,
              "quality rejection requires failing incumbent and fully passing alternative CPU "
              "gates; no record");
      const bool capture_alternative =
          descriptor_seals[1].execution_method != kn::ComparisonExecutionMethod::kOrdinaryOnly;
      if (capture_alternative) {
        stage(1, 0);
        recorded_[1] =
            Must(launch_->Capture(native, [&] { return programs_[1]->Run(*launch_, native); }));
        Require(recorded_[1]->nodes() > 0, "passing alternative capture contains no work");
        for (std::size_t frame = 0; frame < 2; ++frame) {
          stage(1, frame);
          Must(launch_->Launch(native, *recorded_[1]));
          Finish();
          validate(1, frame, true);
        }
        Require(cpu_quality[1], "captured own alternative failed a fixed CPU quality gate");
      }
      const auto protocol = llmp::base::Sha256{}.Update(quality_protocol).Finish();
      std::println(
          "{{\"event\":\"quality_rejection_result\",\"disposition\":\"reject_incumbent_quality\","
          "\"source_contract\":{},\"recipe_key_digest\":{},\"protocol\":{},\"receipt\":{},"
          "\"metric\":{},\"bound\":{},\"frames\":2,\"incumbent_cpu_passed\":false,"
          "\"alternative_cpu_passed\":true,\"peer_passed\":{},\"alternative_captured\":{},"
          "\"incumbent_ns\":0,\"alternative_ns\":0,\"samples\":0}}",
          Quote(contract),
          Quote(llmp::base::ToHex(llmp::execution::RecipeComparisonKeyDigest(*whole_key))),
          Quote(llmp::base::ToHex(protocol)), Quote(llmp::base::ToHex(quality_receipt.Finish())),
          Quote(reference_metric), reference_bound, peer_quality, capture_alternative);
      return;
    }
    std::println(
        "{{\"event\":\"validated\",\"mode\":\"ordinary\",\"frames\":2,\"arms\":2,"
        "\"source_contract\":{},\"metric\":{},\"bound\":{}}}",
        Quote(contract), Quote(reference_metric), reference_bound);
    if (!capture_validation && !screen) return;

    const bool pair_ordinary = std::ranges::any_of(descriptor_seals, [](const auto& seal) {
      return seal.execution_method == kn::ComparisonExecutionMethod::kOrdinaryOnly;
    });
    std::println("{{\"event\":\"pair_execution_method\",\"ordinary\":{}}}", pair_ordinary);
    // Only the ordinary complete program is captured. Restaging is outside
    // this body and occurs before every capture and every replay.
    for (std::size_t arm = 0; arm < 2; ++arm) {
      if (pair_ordinary) continue;
      stage(arm, 0);
      Must(launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
      auto captured =
          launch_->Capture(native, [&] { return programs_[arm]->Run(*launch_, native); });
      auto disabled = launch_->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled);
      Must(std::move(disabled));
      recorded_[arm] = Must(std::move(captured));
      Require(recorded_[arm]->nodes() > 2, "measured capture contains no ordinary program body");
    }
    struct OrdinaryEvents {
      cudaEvent_t start{}, end{};
      ~OrdinaryEvents() {
        if (end) (void)cudaEventDestroy(end);
        if (start) (void)cudaEventDestroy(start);
      }
    } events;
    if (pair_ordinary) {
      Check(cudaEventCreate(&events.start));
      Check(cudaEventCreate(&events.end));
    }
    const auto replay = [&](std::size_t arm, std::size_t frame, bool emit) {
      stage(arm, frame);
      double ns = 0.0;
      if (pair_ordinary) {
        Check(cudaEventRecord(events.start, cuda_stream));
        Must(programs_[arm]->Run(*launch_, native));
        Check(cudaEventRecord(events.end, cuda_stream));
        Finish();
        float milliseconds = 0.0F;
        Check(cudaEventElapsedTime(&milliseconds, events.start, events.end));
        ns = static_cast<double>(milliseconds) * 1.0e6;
      } else {
        Must(launch_->Launch(native, *recorded_[arm]));
        Finish();
        ns = Must(recorded_[arm]->gpu_duration_ns());
      }
      Require(std::isfinite(ns) && ns > 0, "completed capture duration is not positive and finite");
      validate(arm, frame, emit);
      return ns;
    };
    for (std::size_t frame = 0; frame < 2; ++frame)
      for (std::size_t arm = 0; arm < 2; ++arm) {
        (void)replay(arm, frame, true);
        std::println(
            "{{\"event\":\"execution_validated\",\"arm\":{},\"frame\":{},\"nodes\":{},\"body_"
            "repetitions\":1,\"execution_method\":{},\"positive_duration\":true}}",
            arm, frame, recorded_[arm] ? recorded_[arm]->nodes() : 0,
            static_cast<int>(pair_ordinary ? kn::ComparisonExecutionMethod::kOrdinaryOnly
                                           : kn::ComparisonExecutionMethod::kCaptured));
      }
    if (!screen) return;

    constexpr std::string_view protocol =
        "llmp.u6.whole-program.v2;invocations=1;"
        "duration=captured-internal-events-or-ordinary-owned-stream-events;"
        "restage=full-arena-nan-poison,full-identical-state,own-input-slots-before-every-replay;"
        "reference=declared-source-contract,complete-outputs-states-and-immutable-weights;"
        "cache-contract=completed-block-cache-v1;"
        "timed-frame=0;fixed-ab-warmups=4;alternative-bookend-before=median9;"
        "alternating-ab-ba-warmups=4;alternating-ab-ba-sample-pairs=9;"
        "alternative-bookend-after=median9;records=52;drift=10pct;retry=none;adoption=none";
    llmp::base::Sha256 protocol_hash;
    protocol_hash.Update(protocol);
    const auto protocol_digest = llmp::base::ToHex(protocol_hash.Finish());
    std::println("{{\"event\":\"protocol\",\"digest\":\"{}\",\"description\":\"{}\"}}",
                 protocol_digest, protocol);
    const auto time = [&](std::size_t arm, std::string_view phase, int pair, std::size_t order) {
      const auto ns = replay(arm, 0, false);
      std::println(
          "{{\"event\":\"sample\",\"phase\":\"{}\",\"pair\":{},\"order\":{},\"arm\":{},\"ns\":{}}}",
          phase, pair, order, arm, ns);
      std::fflush(stdout);
      return ns;
    };
    for (int pair = 0; pair < 4; ++pair)
      for (std::size_t arm = 0; arm < 2; ++arm) (void)time(arm, "fixed_ab_warmup", pair, arm);
    const auto median = [](std::vector<double> values) {
      Require(!values.empty() && values.size() % 2 == 1, "median requires odd observations");
      std::ranges::sort(values);
      return values[values.size() / 2];
    };
    std::vector<double> before_values, after_values;
    for (std::size_t order = 0; order < 9; ++order)
      before_values.push_back(time(1, "bookend_before", -1, order));
    const double before = median(before_values);
    std::array<std::vector<double>, 2> samples;
    for (int pair = -4; pair < 9; ++pair) {
      const auto first_arm = static_cast<std::size_t>((pair + 4) % 2);
      for (std::size_t order = 0; order < 2; ++order) {
        const auto arm = (first_arm + order) % 2;
        const auto ns = time(arm, pair < 0 ? "alternating_warmup" : "paired_timing", pair, order);
        if (pair >= 0) samples[arm].push_back(ns);
      }
    }
    for (std::size_t order = 0; order < 9; ++order)
      after_values.push_back(time(1, "bookend_after", -1, order));
    const double after = median(after_values);
    Require(std::max(before, after) <= 1.1 * std::min(before, after),
            "alternative bookend drift exceeds 10%; no retry");
    for (const auto& values : samples)
      Require(values.size() == 9, "first screen requires exactly nine pairs");
    std::println(
        "{{\"event\":\"whole_program_result\",\"scope\":\"whole-program\",\"source_contract\":{},"
        "\"metric\":{},\"bound\":{},"
        "\"protocol\":\"{}\",\"incumbent_median_ns\":{},\"alternative_median_ns\":{},\"pairs\":9,"
        "\"bookend_before_ns\":{},\"bookend_after_ns\":{},\"candidate_diagnostic\":true,\"ready_"
        "key_transfer\":false,\"adoptable\":false}}",
        Quote(contract), Quote(reference_metric), reference_bound, protocol_digest,
        median(samples[0]), median(samples[1]), before, after);
  }

 private:
  void Finish() {
    const auto fence = Must(execution_->Record(stream_));
    for (;;) {
      const auto state = Must(execution_->Query(fence));
      if (state == llmp::providers::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    Must(execution_->Release(fence));
  }
  std::uint64_t metadata_used_ = 0;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_{};
  llmp::providers::NativeStream native_{};
  std::map<ir::WeightRef, std::uint64_t> expert_pitches_;
  std::unique_ptr<kn::DeviceBackend> backend_;
  CudaBuffer activation_, weights_, state_, pool_, persistent_, library_;
  std::unique_ptr<kn::StepLaunch> launch_;
  std::array<std::unique_ptr<kn::StepProgram>, 2> programs_;
  std::array<std::unique_ptr<kn::Recorded>, 2> recorded_;
};
}  // namespace
int main(int argc, char** argv) {
  const std::string_view mode = argc >= 2 ? argv[1] : "";
  std::string_view pattern = "gdn_norm_gate";
  std::uint64_t occurrence = 0;
  std::string_view contract = "gdn";
  std::string artifact_path;
  std::uint32_t rows = 1;
  bool valid = true;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--pattern" && i + 1 < argc)
      pattern = argv[++i];
    else if (arg == "--source" && i + 1 < argc)
      contract = argv[++i];
    else if (arg == "--artifact" && i + 1 < argc)
      artifact_path = argv[++i];
    else if (arg == "--rows" && i + 1 < argc) {
      const std::string_view value(argv[++i]);
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), rows);
      valid &= error == std::errc{} && end == value.data() + value.size();
    } else if (arg == "--occurrence" && i + 1 < argc) {
      const std::string_view value(argv[++i]);
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), occurrence);
      valid &= error == std::errc{} && end == value.data() + value.size();
    } else
      valid = false;
  }
  const bool admitted = contract == "admitted-moe" || contract == "admitted-gdn" ||
                        contract == "admitted-qsa" || contract == "admitted-mqa";
  const bool specialized =
      admitted || contract == "deepseek-specialized" || contract == "qwen-qsa-specialized";
  valid &= admitted == !artifact_path.empty();
  const bool resident_moe = admitted || (contract != "gdn" && !specialized);
  valid &= specialized || contract == "gdn" || contract == "resident-qwen-moe" ||
           contract == "resident-q8-moe" || contract == "resident-qwen-f16-moe";
  valid &= specialized    ? (admitted ? (rows == 1 || rows == 5 || rows == 17)
                             : contract == "deepseek-specialized" ? (rows == 16 || rows == 64)
                                                                  : rows == 17)
           : resident_moe ? (rows == 1 || rows == 5)
                          : rows == 1;

  valid &=
      mode == "--preflight-only" ||
      std::ranges::any_of(planner::Patterns(), [&](const auto& p) { return p.name == pattern; });
  valid &= contract != "admitted-mqa" || rows == 1;
  valid &= mode != "--preflight-only" || specialized || resident_moe || argc == 2;
  valid &= mode == "--validation-only" || mode == "--capture-validation-only" ||
           mode == "--preflight-only" || mode == "--screen" ||
           mode == "--quality-rejection-validation" || mode == "--quality-pair-attribution";
  if (!valid) {
    std::println(
        stderr,
        "usage: llmp_fusion_gdn --validation-only|--capture-validation-only|--preflight-only|"
        "--screen|--quality-pair-attribution [--pattern name] [--occurrence index] "
        "[--source resident-qwen-moe|resident-qwen-f16-moe|resident-q8-moe --rows 1|5] "
        "[--source deepseek-specialized --rows 16|64 | qwen-qsa-specialized --rows 17]");
    return 2;
  }
  {
    Screen screen;
    if (specialized && mode == "--preflight-only")
      screen.PreflightSpecialized(contract == "qwen-qsa-specialized" || contract == "admitted-qsa",
                                  rows, artifact_path,
                                  contract == "admitted-gdn"   ? "attention.gdn"
                                  : contract == "admitted-mqa" ? "attention.mqa"
                                  : contract == "admitted-qsa" ? "attention.qsa"
                                                               : "ffn.moe");
    else
      screen.Run(mode == "--capture-validation-only", mode == "--screen", pattern, occurrence,
                 mode == "--preflight-only", contract, rows, artifact_path,
                 mode == "--quality-rejection-validation", mode == "--quality-pair-attribution");
  }
  std::println("{{\"event\":\"complete\",\"teardown_verified\":true}}");
  return 0;
}
