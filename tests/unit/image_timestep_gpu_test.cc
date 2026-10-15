// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Actual source conditioning, using the shared stateless graph and owned backend.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/sha256.h"
#include "components/spatial.h"
#include "engine/coverage.h"
#include "engine/execution_contract.h"
#include "engine/graph_plan.h"
#include "engine/paged_weights.h"
#include "engine/plan_cache.h"
#include "engine/runner_resources.h"
#include "engine/step_runs.h"
#include "engine/support.h"
#include "expected_error.h"
#include "kernels/ggml/step_program.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "spec/image_spatial.h"

namespace {
namespace en = llmp::engine;
namespace co = llmp::components;
namespace sp = llmp::spec;
namespace mo = llmp::model;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace ca = llmp::catalog;
namespace fm = llmp::format;
namespace sc = llmp::scheduler;
using llmp::base::Bytes;
using llmp::test_support::Failed;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kDomain = 512 * kMiB, kMetadata = 64 * kMiB;
constexpr std::uint64_t kCompilerAndOwner = 32 * kMiB, kProgram = 2 * kMiB;
constexpr std::uint64_t kActivation = 2 * kMiB, kPool = 2 * kMiB, kPinned = kMiB;
constexpr std::uint64_t kOutputOffset = 65536, kSavedOffset = 524288;
constexpr std::array<std::uint64_t, 2> kOutputBytes{131072, 32768};
constexpr std::uint64_t kOutputTotal = kOutputBytes[0] + kOutputBytes[1];
constexpr std::string_view kArtifact =
    "d1184efda09c12175b7bae6e9acf6cf3aeafb765f4f2ab6192cbf284c37eb3a4";
constexpr std::array<std::string_view, 4> kNames{
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight", "modulation.1.weight",
    "norm_out.linear.weight"};
constexpr std::array<std::uint64_t, 4> kWeightBytes{2097152, 33554432, 134217728, 33554432};
constexpr std::uint64_t kWeightTotal = 203423744;
// Fixed graph, source/clone/rewrite/native inventories, measuring+retained program,
// one <=67-node capture, and 16MiB source-index/owner/report margin. No payload oracle.
static_assert(12 * (37 * 480ULL + 74 * 192ULL) + 4096 * (37 + 74ULL) +
                  (74 + 4 + 31 * 37ULL + 4) * (512 + 1024 + 32) + 2 * kProgram +
                  67 * en::kGraphNodeHostBytes + 16 * kMiB <
              kCompilerAndOwner);
static_assert(kOutputOffset + kOutputTotal <= kSavedOffset &&
              kSavedOffset + kOutputTotal <= kPinned);

// Same owner and runner for the complete consumer. Limits are fixed before graph
// construction; actual measured programs must fit before node Start/admission.
struct Limits {
  std::uint64_t domain, metadata, compiler, program, activation, pool;
  std::uint64_t output_offset, nodes, tensors, graph_metadata, launches;
  std::size_t parts, selected_pages, reserved_pages;
  std::array<std::uint64_t, 2> outputs;
};
constexpr Limits kConditioning{kDomain,
                               kMetadata,
                               kCompilerAndOwner,
                               kProgram,
                               kActivation,
                               kPool,
                               kOutputOffset,
                               37,
                               74,
                               491520,
                               64,
                               4,
                               99,
                               130,
                               kOutputBytes};
constexpr Limits kBlock{1024 * kMiB, 96 * kMiB, 64 * kMiB, 4 * kMiB, 8 * kMiB,
                        8 * kMiB,    131072,    250,       500,      2912256,
                        256,         13,        308,       339,      {98304, 32768}};
constexpr std::array<std::string_view, 9> kBlockNames{
    "transformer_blocks.0.attn.to_q.weight",          "transformer_blocks.0.attn.to_k.weight",
    "transformer_blocks.0.attn.to_v.weight",          "transformer_blocks.0.attn.to_out.0.weight",
    "transformer_blocks.0.attn.norm_q.weight",        "transformer_blocks.0.attn.norm_k.weight",
    "transformer_blocks.0.img_mlp.gate_layer.weight", "transformer_blocks.0.img_mlp.proj.weight",
    "transformer_blocks.0.img_mlp.out.weight"};
constexpr std::array<std::uint64_t, 9> kBlockBytes{33554432, 33554432,  33554432,  33554432, 256,
                                                   256,      100663296, 100663296, 100663296};
static_assert(12 * (250 * 480ULL + 500 * 192ULL) + 4096 * (250 + 500ULL) +
                  (500 + 13 + 31 * 250ULL + 4) * (512 + 1024 + 32) + 2 * kBlock.program +
                  262 * en::kGraphNodeHostBytes + 16 * kMiB <
              kBlock.compiler);
static_assert(kBlock.output_offset + kBlock.outputs[0] + kBlock.outputs[1] <= kSavedOffset &&
              kSavedOffset + kBlock.outputs[0] + kBlock.outputs[1] <= kPinned);
static_assert(645922816ULL + kBlock.metadata + kBlock.activation + kBlock.pool + 32 * kMiB +
                  kPinned + en::kSlabSlotBytes <
              kBlock.domain);

constexpr Limits kFlow{16ULL << 30U, 1024 * kMiB, 768 * kMiB, 16 * kMiB, 32 * kMiB,
                       32 * kMiB,    65536,       5948,       11896,     66072576,
                       5948,         297,         6818,       6818,      {1024, 1024}};
constexpr std::uint64_t kFlowSourceBytes = 14230249472ULL;
constexpr std::uint64_t kFlowMappingBytes = 2675712;
static_assert(12 * (5948 * 480ULL + 11896 * 192ULL) + 4096 * (5948 + 11896ULL) +
                  (11896 + 297 + 31 * 5948ULL + 6) * (512 + 1024 + 32) + 2 * kFlow.program +
                  (5948 + 8) * en::kGraphNodeHostBytes + 16 * kMiB <
              kFlow.compiler);
static_assert(kFlow.compiler + kFlow.graph_metadata + kFlowMappingBytes + kProgram <
              kFlow.metadata);
static_assert(14298382336ULL + kFlow.metadata + kFlow.activation + kFlow.pool + 32 * kMiB +
                  kPinned + en::kSlabSlotBytes <
              kFlow.domain);
static_assert(kFlow.output_offset + 2048 < kSavedOffset && kSavedOffset + 2048 < kPinned);

en::Status Error(std::string what) { return std::unexpected(std::move(what)); }
std::string Hash(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  return llmp::base::ToHex(hash.Finish());
}
struct Lifetime final : en::PagedModel {
  explicit Lifetime(bool complete_block = false, bool whole_flow = false)
      : block(complete_block || whole_flow),
        flow(whole_flow),
        limits(flow ? kFlow : (block ? kBlock : kConditioning)) {}
  const bool block, flow;
  const Limits limits;
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  en::PagedWeights weights;
  en::RunnerResources resources{node, 0, 0};
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kn::StepLaunch> launch;
  std::optional<co::SpatialGraph> graph;
  std::map<std::pair<ir::WeightRef, fm::PartRole>, std::uint32_t> bindings;
  std::unique_ptr<kn::StepProgram> program;
  en::StepPlanRuns runs;
  std::array<std::uint32_t, 297> resource_ids{};
  std::array<std::uint64_t, 297> source_bytes{};
  std::array<float, 24576> block_input{};
  std::array<float, 256> latent{};
  std::array<float, 1> dt{-0.125F};
  std::array<std::int32_t, 18> positions{};
  std::array<float, 36> mask{};
  std::array<float, 512> input{};
  std::byte* host = nullptr;
  en::StepRuns step_runs{true};
  en::GraphStats stats;
  ca::Closure closure, empty;
  std::array<std::string, 2> original_sha{}, changed_sha{}, image_sha{};
  bool opened = false, retirement_attempted = false, retired = false;

  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return empty; }
  std::vector<ca::ExtentId> managed_extents() const override { return weights.extents(); }
  en::Status Release() override {
    runs.DropGraph();
    program.reset();
    graph.reset();
    step_runs.SetLaunch(nullptr);
    launch.reset();
    if (backend) backend->LaunchReleased();
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    if (auto r = weights.Release(node.memory()); !r) errors.push_back(r.error());
    return en::support::Joined(errors);
  }
  en::Status Retire() {
    if (!opened) return {};
    if (retirement_attempted)
      return retired ? en::Status{} : Error("the first retirement was not proven");
    retirement_attempted = true;
    const std::array<en::PagedModel*, 1> models{this};
    auto done = node.TearDown(models);
    retired = done.has_value();
    return done;
  }
};
struct Retire {
  void operator()(Lifetime* life) const {
    if (!life) return;
    auto done = life->Retire();
    EXPECT_TRUE(done) << Failed(done).value_or("");
    if (done) delete life;  // Unknown completion retains every borrower and its owner.
  }
};
kn::Addresses Addresses(Lifetime& life, bool measuring) {
  auto address = [&life](const ir::WeightRef& ref, fm::PartRole part) {
    const auto found = life.bindings.find({ref, part});
    return found == life.bindings.end() ? 0ULL : life.weights.resource_address(found->second);
  };
  kn::Addresses result;
  result.weight = address;
  result.activations = measuring ? 0 : life.node.activations().base;
  result.activation_bytes = life.limits.activation;
  result.activation_alignment = 256;
  result.weight_alignment = [address](const ir::WeightRef& ref, fm::PartRole part) {
    const auto actual = address(ref, part);
    return actual == 0 ? 0ULL : std::gcd(256ULL, actual);
  };
  return result;
}
en::Status CheckProgram(const Lifetime& life, const kn::StepProgram& program) {
  const auto& limit = life.limits;
  const std::array<std::uint64_t, 4> block_inputs{98304, 2048, 72, 144};
  const std::array<std::uint64_t, 6> flow_inputs{32768, 1024, 2048, 72, 144, 4};
  if (program.inputs().size() != (life.flow ? 6U : (life.block ? 4U : 1U)) ||
      program.outputs().size() != 2 || program.scratch_bytes() > limit.activation ||
      program.pool_bytes() > limit.pool || program.host_bytes() > limit.program ||
      program.launches() > limit.launches || !program.capturable() || program.routed())
    return Error("image consumer program ports or measured paid envelope differ");
  for (std::size_t i = 0; i < program.inputs().size(); ++i)
    if (program.inputs()[i].kind != kn::InputKind::kStepBytes || program.inputs()[i].segment != 0 ||
        program.inputs()[i].bytes !=
            (life.flow ? flow_inputs[i] : (life.block ? block_inputs[i] : sizeof(life.input))) ||
        program.inputs()[i].tensor != life.graph->graph().inputs()[i])
      return Error("image consumer input identity/owner/extent differs");
  for (std::size_t i = 0; i < 2; ++i)
    if (program.outputs()[i].segment != 0 || program.outputs()[i].bytes != limit.outputs[i] ||
        program.outputs()[i].tensor != life.graph->graph().outputs()[i])
      return Error("image consumer output identity, shape or owner differs");
  if (!program.bound()) return {};
  std::size_t products = 0, unary = 0, rope = 0, softmax = 0, glu = 0, norms = 0;
  if (program.bound()->size() > limit.launches) return Error("bounded selected plan inventory");
  for (std::size_t i = 0; i < program.bound()->size(); ++i) {
    const auto& name = program.bound()->at(i).name;
    products += name.starts_with("ggml.mul_mat.");
    unary += name == "ggml.unary";
    rope += name == "ggml.rope.ext";
    softmax += name == "ggml.soft_max";
    glu += name == "ggml.swiglu";
    norms += name == "ggml.norm";
  }
  if (products != (life.flow ? 296U : (life.block ? 13U : 4U)) ||
      unary != (life.flow ? 67U : (life.block ? 4U : 2U)) ||
      rope != (life.flow ? 192U : (life.block ? 6U : 0U)) ||
      softmax != (life.flow ? 32U : (life.block ? 1U : 0U)) ||
      glu != (life.flow ? 32U : (life.block ? 1U : 0U)) ||
      norms != (life.flow ? 65U : (life.block ? 2U : 0U)))
    return Error("complete selected conditioner/block operation inventory differs");
  return {};
}
en::Status SetupFixture(Lifetime& life, const std::filesystem::path& path) {
  auto available = llmp::platform::AvailableMemoryBytes();
  // Process/cache ceiling2GiB + fixed node domain + physical margin6GiB.
  if (!available || *available < (life.flow ? (24ULL << 30U) : (9ULL << 30U)))
    return Error("physical headroom");
  if (auto r = life.node.Open(); !r) return r;
  life.opened = true;
  life.node.SetHostFloor(0);
  auto paid = en::ExecutionMetadataAllowance::Admit(life.node.catalog(), life.node.domain(),
                                                    life.limits.metadata, *available, 6ULL << 30U);
  if (!paid) return Error(paid.error());
  life.paid = *paid;
  if (!life.paid->Charge(life.limits.compiler)) return Error("preconstruction owner payment");
  std::error_code ec;
  if (std::filesystem::file_size(path / "manifest.json", ec) != 1600 || ec ||
      std::filesystem::file_size(path / "index.json", ec) != 531405 || ec)
    return Error("fixed source metadata extents");
  if (auto r = life.weights.Open(path, {.expected_id = std::string(kArtifact)}); !r) return r;
  const auto& artifact = life.weights.artifact();
  if (artifact.resources().size() != 297 || artifact.groups().size() != 33 ||
      artifact.shards().size() != 4 || artifact.format_version() != 0)
    return Error("actual denoiser source inventory");
  const auto& profile = mo::QwenImage21().denoiser;
  const auto all_tensors = mo::DenoiserTensors(profile);
  std::vector<mo::QwenImageTensor> tensors;
  for (const auto name : kNames) {
    const auto found = std::ranges::find(all_tensors, name, &mo::QwenImageTensor::name);
    if (found == all_tensors.end()) return Error("fixed timestep source role");
    tensors.push_back(*found);
  }
  if (life.block)
    for (const auto name : kBlockNames) {
      const auto found = std::ranges::find(all_tensors, name, &mo::QwenImageTensor::name);
      if (found == all_tensors.end()) return Error("fixed block source role");
      tensors.push_back(*found);
    }
  if (life.flow) tensors = all_tensors;
  auto ids = mo::BindQwenImageComponent(tensors, "QwenImage21Transformer2DModel", artifact);
  if (!ids) return Error(ids.error());
  auto source = sp::SourceOfV0(artifact);
  if (!source) return Error(source.error());
  std::vector<co::BoundTensor> bound;
  std::vector<en::GroupPlace> groups(33, en::GroupPlace::kNone);
  for (std::size_t i = 0; i < life.limits.parts; ++i) {
    life.source_bytes[i] = i < 4 ? kWeightBytes[i] : (i < 13 ? kBlockBytes[i - 4] : 0);
    if (life.flow) {
      std::uint64_t bytes = 2;
      for (const auto n : tensors[i].shape)
        if (!n || __builtin_mul_overflow(bytes, n, &bytes)) return Error("source size overflow");
      life.source_bytes[i] = bytes;
    }
    life.resource_ids[i] = ids->at(i);
    const auto placement = artifact.ResourcePlacement(life.resource_ids[i]);
    if (!placement || placement->bytes.value() != life.source_bytes[i] ||
        (!life.flow && placement->group != (i < 4 ? 32U : 0U)))
      return Error("source role bytes or group differ");
    groups[placement->group] = en::GroupPlace::kDevice;
    const auto found = std::ranges::find(source->tensors, tensors[i].name, &co::BoundTensor::name);
    if (found == source->tensors.end() || found->format.encoding != fm::Encoding::kBF16)
      return Error("actual BF16 source format");
    bound.push_back(*found);
  }
  if (life.flow &&
      std::accumulate(life.source_bytes.begin(), life.source_bytes.end(), 0ULL) != kFlowSourceBytes)
    return Error("whole flow source bytes");
  if (auto r = life.weights.Reserve(life.node, groups, {}); !r) return r;
  if (life.weights.bytes() != life.limits.reserved_pages * 2 * kMiB ||
      life.weights.host_bytes() != 0 || life.weights.extents().size() != life.limits.reserved_pages)
    return Error("global group virtual reservation differs");
  if (life.flow) {
    auto mapping_bound = sp::ImageFlowSpecBound(profile, tensors);
    if (!mapping_bound || *mapping_bound != kFlowMappingBytes || !life.paid->Charge(*mapping_bound))
      return Error("whole flow mapping prepayment");
  }
  auto mapping = life.flow    ? sp::ImageFlowSpec(profile, 2, 4, tensors)
                 : life.block ? sp::ImageTransformerBlockSpec(profile, 0, 2, tensors)
                              : sp::ImageTimestepSpec(profile, tensors);
  if (!mapping) return Error(mapping.error());
  const std::array<std::span<const co::BoundTensor>, 1> tables{bound};
  const std::array owners{co::SpatialOwner{0, life.flow    ? co::SpatialShape{1, 2, 4096, 1}
                                              : life.block ? co::SpatialShape{1, 6, 4096, 1}
                                                           : co::SpatialShape{1, 2, 256, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {1, 2, 256, 1}},
                         co::SpatialInput{0, 1, {6, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 2, {6, 6, 1, 1}}};
  const std::array flow_ports{
      co::SpatialInput{0, 0, {1, 4, 64, 1}}, co::SpatialInput{0, 1, {1, 2, 256, 1}},
      co::SpatialInput{0, 2, {6, 3, 1, 1}, fm::Encoding::kI32},
      co::SpatialInput{0, 3, {6, 6, 1, 1}}, co::SpatialInput{0, 4, {1, 1, 1, 1}}};
  auto graph =
      co::BuildSpatialStep(mapping->spec, tables, owners,
                           {[paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
                            [paid = life.paid](std::uint64_t n) { paid->Release(n); }},
                           {.links = mapping->links,
                            .reference_buffers = false,
                            .exports = mapping->exports,
                            .inputs = life.flow    ? std::span<const co::SpatialInput>(flow_ports)
                                      : life.block ? std::span<const co::SpatialInput>(ports)
                                                   : std::span<const co::SpatialInput>{}});
  if (!graph) return Error(graph.error());
  if (graph->graph().nodes().size() > life.limits.nodes ||
      graph->graph().tensor_count() > life.limits.tensors ||
      graph->weights().size() != life.limits.parts ||
      graph->metadata_bytes() > life.limits.graph_metadata || graph->reference_bytes() != 0 ||
      graph->outputs().size() != 2)
    return Error(
        "fixed timestep graph inventory: nodes=" + std::to_string(graph->graph().nodes().size()) +
        "/" + std::to_string(life.limits.nodes) + " tensors=" +
        std::to_string(graph->graph().tensor_count()) + "/" + std::to_string(life.limits.tensors) +
        " parts=" + std::to_string(graph->weights().size()) + "/" +
        std::to_string(life.limits.parts) + " metadata=" + std::to_string(graph->metadata_bytes()) +
        "/" + std::to_string(life.limits.graph_metadata) +
        " reference=" + std::to_string(graph->reference_bytes()) +
        "/0 outputs=" + std::to_string(graph->outputs().size()) + "/2");
  life.graph.emplace(std::move(*graph));
  for (const auto& weight : life.graph->weights()) {
    if (weight.parts.size() != 1 ||
        life.graph->graph().tensor(weight.tensor).format.encoding != fm::Encoding::kBF16)
      return Error("source-resident encoded root");
    const auto& [name, part] = weight.parts[0];
    const auto id = artifact.FindResource(name);
    if (!id || !std::ranges::contains(std::span(life.resource_ids).first(life.limits.parts), *id) ||
        !life.bindings.emplace(std::pair{weight.ref, part}, *id).second)
      return Error("exact timestep role binding");
  }
  auto backend = kg::GgmlBackend::Create(0, life.node.execution(), life.node.stream(0), nullptr);
  if (!backend) return Error(backend.error());
  life.backend = std::move(*backend);
  const auto facts = llmp::providers::QueryDeviceFacts(0);
  if (!facts) return Error("actual device facts");
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  if (workspace.value() > 32 * kMiB) return Error("owned handle workspace");
  if (auto r = life.backend->Declare(
          {}, workspace.value(), [paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
          [paid = life.paid](std::uint64_t n) { paid->Release(n); });
      !r)
    return Error(r.error());
  if (auto r = life.resources.OpenCublas("timestep owned handle"); !r) return r;
  if (auto r = life.backend->LendCublas(&life.resources.cublas()); !r) return Error(r.error());
  const auto owned = life.backend->Backend();
  if (!owned.owner || !owned.owner->Ready()) return Error("owned backend readiness");
  const auto& table = owned.owner->backend();
  en::GraphBuildIO io{life.graph->graph(), life.graph->graph().inputs(),
                      life.graph->graph().outputs(), life.graph->weights()};
  const auto build = [&](bool measuring) {
    return en::BuildComponentProgram(
        io, table, {}, Addresses(life, measuring),
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = table.written,
         .within = table.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
  };
  {
    auto measured = build(true);
    if (!measured) return Error(measured.error());
    if (auto r = CheckProgram(life, **measured); !r) return r;
  }
  if (auto r = life.node.MapWorkspace(life.limits.activation, life.limits.pool); !r) return r;
  auto launch = life.backend->Launch({life.node.pool().base, Bytes(life.node.pool().bytes)});
  if (!launch) return Error(launch.error());
  life.launch = std::move(*launch);
  auto host = life.resources.Pinned(kPinned);
  if (!host) return Error(host.error());
  life.host = static_cast<std::byte*>(*host);
  life.step_runs.SetStaging(life.host, life.limits.output_offset);
  life.step_runs.SetLaunch(life.launch.get());
  auto program = build(false);
  if (!program) return Error(program.error());
  if (auto r = CheckProgram(life, **program); !r) return r;
  life.program = std::move(*program);
  if (auto r = life.program->Bind(*life.launch); !r) return Error(r.error().detail);
  if (!life.program->bound()) return Error("actual program not bound");
  if (auto r = CheckProgram(life, *life.program); !r) return r;
  if (auto r = life.node.Start(Bytes(life.limits.domain)); !r) return r;
  if (auto r = life.weights.Register(life.node, 0); !r) return r;
  auto extents = life.resources.extents();
  for (const auto* more : {&life.node.activations().extents, &life.node.pool().extents})
    extents.insert(extents.end(), more->begin(), more->end());
  std::vector<ca::ExtentId> selected;
  for (std::size_t i = 0; i < life.limits.parts; ++i) {
    // The128MiB modulation interval is offset0.5MiB into a page:65 pages.
    std::array<ca::ExtentId, 65> pages{};
    auto count = life.weights.ExtentsTouching(life.weights.resource_address(life.resource_ids[i]),
                                              life.source_bytes[i], pages);
    if (!count) return Error(count.error());
    selected.insert(selected.end(), pages.begin(),
                    pages.begin() + static_cast<std::ptrdiff_t>(*count));
  }
  std::ranges::sort(selected);
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  if (selected.size() != life.limits.selected_pages)
    return Error("complete selected source-page closure");
  extents.insert(extents.end(), selected.begin(), selected.end());
  auto closure = life.node.catalog().ClosureOfExtents(extents);
  if (!closure) return Error("complete timestep owned closure");
  life.closure = *closure;
  life.node.Run();
  return {};
}
en::Status Execute(Lifetime& life) {
  return life.node.WithRequest(0, life.closure, "actual timestep source", [&]() -> en::Status {
    auto coverage = life.node.Call(
        [&]() -> en::Status {
          en::Coverage result;
          en::CheckCoverage(life.node, 0, life.program->tensors(), "actual timestep", result);
          return result.violations ? Error(result.first_violation) : en::Status{};
        },
        "timestep catalog coverage");
    if (!coverage) return coverage;
    for (std::size_t repeat = 0; repeat < (life.block ? 7U : 5U); ++repeat) {
      const auto input =
          mo::QwenImageTimestepSinusoid(mo::QwenImage21().denoiser, repeat == 3 ? 500.F : 1000.F);
      if (input.size() != 512) return Error("source sinusoid size");
      bool unequal = false;
      for (std::size_t c = 0; c < 256; ++c) {
        // Exact source BF16 input boundary, WHCN [1,2,256,1]. Graph activations stay F32.
        life.input[2 * c] = mo::FromBf16(input[c]);
        life.input[2 * c + 1] = mo::FromBf16(input[256 + c]);
        unequal = unequal || input[c] != input[256 + c];
      }
      if (!unequal) return Error("two conditioning rows must differ");
      if (life.block) {
        for (std::size_t c = 0; c < 4096; ++c)
          for (std::size_t row = 0; row < 6; ++row)
            life.block_input[6 * c + row] =
                static_cast<float>(static_cast<int>((17 * c + 11 * row) % 101) - 50) / 100.F +
                (repeat == 5 && row == 5 ? .125F : 0.F);
        life.positions = {0, 1, 2, 2, 2, 2, 0, 1, -1, -1, 0, 0, 0, 1, -1, 0, -1, 0};
        if (repeat == 5) {
          life.positions[11] = -2;
          life.positions[17] = -3;
        }
        for (std::size_t query = 0; query < 6; ++query)
          for (std::size_t key = 0; key < 6; ++key)
            life.mask[6 * query + key] =
                query < 2 && key > query ? -std::numeric_limits<float>::infinity() : 0.F;
      }
      if (life.flow) {
        for (std::size_t c = 0; c < 4096; ++c)
          for (std::size_t row = 0; row < 2; ++row)
            life.block_input[2 * c + row] =
                static_cast<float>(static_cast<int>((17 * c + 11 * row) % 101) - 50) / 100.F;
        for (std::size_t i = 0; i < life.latent.size(); ++i)
          life.latent[i] = static_cast<float>(static_cast<int>((13 * i) % 101) - 50) / 100.F +
                           (repeat == 5 && i % 4 == 3 ? .125F : 0.F);
      }
      auto copies = life.step_runs.Stage(
          life.program->inputs(),
          [&](std::size_t slot, std::span<std::byte> out) -> en::Status {
            const auto input_bytes = std::as_bytes(std::span(life.input));
            const std::array<std::span<const std::byte>, 4> block_bytes{
                std::as_bytes(std::span(life.block_input)), input_bytes,
                std::as_bytes(std::span(life.positions)), std::as_bytes(std::span(life.mask))};
            const std::array<std::span<const std::byte>, 6> flow_bytes{
                std::as_bytes(std::span(life.block_input).first(8192)),
                std::as_bytes(std::span(life.latent)),
                input_bytes,
                std::as_bytes(std::span(life.positions)),
                std::as_bytes(std::span(life.mask)),
                std::as_bytes(std::span(life.dt))};
            if (slot >= (life.flow ? 6U : (life.block ? 4U : 1U))) return Error("input port count");
            const std::span<const std::byte> bytes = life.flow    ? flow_bytes[slot]
                                                     : life.block ? block_bytes[slot]
                                                                  : input_bytes;
            if (out.size() != bytes.size()) return Error("input port extent");
            std::memcpy(out.data(), bytes.data(), bytes.size());
            return {};
          },
          0);
      if (!copies) return Error(copies.error());
      std::array<en::RunCopy, 2> outputs{};
      std::uint64_t offset = 0;
      for (std::size_t branch = 0; branch < 2; ++branch) {
        const auto& output = life.program->outputs()[branch];
        outputs[branch] = {en::support::Address(life.host + life.limits.output_offset + offset),
                           output.device_address, output.bytes};
        offset += output.bytes;
      }
      en::StepQueued queued;
      auto ran = life.node.Job(
          life.closure,
          [&](llmp::providers::NativeStream native) {
            queued = life.step_runs.Queue(life.runs, *copies, {}, *life.program, outputs,
                                          life.runs.CaptureDue(true), life.stats, native);
            if (queued.result) return sc::JobResult::kQueued;
            return queued.before || queued.result.error().error == kn::StepError::kUnknown
                       ? sc::JobResult::kUnknown
                       : sc::JobResult::kNotStarted;
          },
          "timestep eager capture changed-input replay", 0);
      if (!ran) return ran;
      if (!queued.result) return Error(queued.result.error().detail);
      const auto expected = repeat == 0   ? en::RunPath::kEager
                            : repeat == 1 ? en::RunPath::kCaptured
                                          : en::RunPath::kReplayed;
      if (queued.path != expected) return Error("required retained capture path");
      en::Count(life.stats, queued.path);
      offset = 0;
      for (std::size_t branch = 0; branch < 2; ++branch) {
        const auto actual = std::span(
            reinterpret_cast<const float*>(life.host + life.limits.output_offset + offset),
            life.limits.outputs[branch] / 4);
        if (!std::ranges::all_of(actual, [](float n) { return std::isfinite(n); }))
          return Error("nonfinite full conditioning branch");
        const auto sha = Hash(std::as_bytes(actual));
        if (life.flow && branch == 1) {
          const auto* noise = reinterpret_cast<const float*>(life.host + life.limits.output_offset);
          for (std::size_t i = 0; i < actual.size(); ++i) {
            const float euler_expected = life.latent[i] + life.dt[0] * noise[i];
            if (std::abs(actual[i] - euler_expected) >
                1e-5F * std::max(1.F, std::abs(euler_expected)))
              return Error("complete Euler output lost noise or original latent contribution");
          }
        }
        if (repeat == 0) {
          life.original_sha[branch] = sha;
          std::memcpy(life.host + kSavedOffset + offset, actual.data(),
                      life.limits.outputs[branch]);
        } else if (repeat == 3) {
          const auto saved = reinterpret_cast<const float*>(life.host + kSavedOffset + offset);
          bool changed = false;
          if (life.flow) {
            for (std::size_t i = 0; i < actual.size(); ++i) changed |= actual[i] != saved[i];
          } else if (life.block && branch == 0) {
            for (std::size_t c = 0; c < 4096; ++c)
              for (std::size_t row = 0; row < 6; ++row) {
                const auto i = 6 * c + row;
                if (row < 2 && actual[i] != saved[i]) return Error("causal text row changed");
                if (row >= 2) changed = changed || actual[i] != saved[i];
              }
          } else {
            for (std::size_t c = 0; c < actual.size() / 2; ++c) {
              changed = changed || actual[2 * c] != saved[2 * c];
              if (actual[2 * c + 1] != saved[2 * c + 1])
                return Error("unchanged zero-time row was altered");
            }
          }
          if (!changed) return Error("new time did not change the actual branch");
          life.changed_sha[branch] = sha;
        } else if (life.block && repeat == 5) {
          if (!life.flow && branch == 1 && sha != life.original_sha[branch])
            return Error("stream/position change altered independent final scale");
          if (life.flow) {
            if (sha == life.original_sha[branch])
              return Error("changed image did not affect whole flow branch");
          } else if (branch == 0) {
            const auto saved = reinterpret_cast<const float*>(life.host + kSavedOffset + offset);
            bool indirect_change = false;
            for (std::size_t c = 0; c < 4096; ++c) {
              if (actual[6 * c] != saved[6 * c] || actual[6 * c + 1] != saved[6 * c + 1])
                return Error("image change escaped the causal text mask");
              indirect_change = indirect_change || actual[6 * c + 2] != saved[6 * c + 2];
            }
            if (!indirect_change)
              return Error("changed image key did not affect another image row");
          }
          life.image_sha[branch] = sha;
        } else if (sha != life.original_sha[branch]) {
          return Error("same source input replay changed its selected-program output");
        }
        offset += life.limits.outputs[branch];
      }
    }
    return {};
  });
}
TEST(ImageTimestepGpuTest, CompleteSourceConditioningBranchesReplayChangedInputs) {
  const auto* path = std::getenv("LLMP_TEST_DENOISER_ARTIFACT");
  if (path == nullptr)
    GTEST_SKIP() << "needs LLMP_TEST_DENOISER_ARTIFACT (an installed Qwen-Image denoiser artifact)";
  std::unique_ptr<Lifetime, Retire> life(new Lifetime);
  auto setup = SetupFixture(*life, path);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto ran = Execute(*life);
  ASSERT_TRUE(ran) << Failed(ran).value_or("");
  EXPECT_EQ(life->stats.eager, 1U);
  EXPECT_EQ(life->stats.captured, 1U);
  EXPECT_EQ(life->stats.replayed, 3U);
  EXPECT_EQ(life->stats.refused, 0U);
  RecordProperty("source_bytes", std::to_string(kWeightTotal));
  RecordProperty("source_parts", "4");
  RecordProperty("selected_pages", "99");
  RecordProperty("reserved_pages", "130");
  RecordProperty("selected_weight_page_bytes", "207618048");
  RecordProperty("reserved_weight_bytes", "272629760");
  RecordProperty("complete_output_values_per_step", "40960");
  RecordProperty("finite_values_checked", "204800");
  RecordProperty("changed_branches", "2");
  RecordProperty("unchanged_zero_time_rows", "2");
  RecordProperty("input_boundary", "source BF16 sinusoid widened exactly to F32");
  RecordProperty("metadata_bytes", std::to_string(life->graph->metadata_bytes()));
  RecordProperty("scratch_bytes", std::to_string(life->program->scratch_bytes()));
  RecordProperty("pool_bytes", std::to_string(life->program->pool_bytes()));
  RecordProperty("program_host_bytes", std::to_string(life->program->host_bytes()));
  RecordProperty("plan", llmp::base::ToHex(life->program->bound()->identity()));
  RecordProperty("eager", std::to_string(life->stats.eager));
  RecordProperty("captured", std::to_string(life->stats.captured));
  RecordProperty("replayed", std::to_string(life->stats.replayed));
  RecordProperty("capture_refused", std::to_string(life->stats.refused));
  for (std::size_t branch = 0; branch < 2; ++branch) {
    RecordProperty("branch" + std::to_string(branch) + "_original_sha", life->original_sha[branch]);
    RecordProperty("branch" + std::to_string(branch) + "_changed_sha", life->changed_sha[branch]);
  }
  std::string implementations;
  for (std::size_t i = 0; i < life->program->bound()->size(); ++i)
    implementations += life->program->bound()->at(i).name + ",";
  ASSERT_LE(implementations.size(), 4096U);
  RecordProperty("implementations", implementations);
  auto retired = life->Retire();
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
}
TEST(ImageTimestepGpuTest, CompleteSourceBlockConsumesConditioningAndReplaysChangedInputs) {
  const auto* path = std::getenv("LLMP_TEST_DENOISER_ARTIFACT");
  if (path == nullptr)
    GTEST_SKIP() << "needs LLMP_TEST_DENOISER_ARTIFACT (an installed Qwen-Image denoiser artifact)";
  std::unique_ptr<Lifetime, Retire> life(new Lifetime(true));
  auto setup = SetupFixture(*life, path);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto ran = Execute(*life);
  ASSERT_TRUE(ran) << Failed(ran).value_or("");
  EXPECT_EQ(life->stats.eager, 1U);
  EXPECT_EQ(life->stats.captured, 1U);
  EXPECT_EQ(life->stats.replayed, 5U);
  EXPECT_EQ(life->stats.refused, 0U);
  RecordProperty("source_bytes", "639631872");
  RecordProperty("source_parts", "13");
  RecordProperty("selected_pages", "308");
  RecordProperty("reserved_pages", "339");
  RecordProperty("selected_weight_page_bytes", "645922816");
  RecordProperty("reserved_weight_bytes", "710934528");
  RecordProperty("complete_output_values_per_step", "32768");
  RecordProperty("finite_values_checked", "229376");
  RecordProperty("completed_steps", "7");
  RecordProperty("changed_time_branches", "2");
  RecordProperty("unchanged_text_rows", "2");
  RecordProperty("unchanged_zero_time_scale_rows", "1");
  RecordProperty("changed_image_contribution", "1");
  RecordProperty("input_boundary", "source BF16 sinusoid widened exactly to F32; F32 stream");
  RecordProperty("metadata_bytes", std::to_string(life->graph->metadata_bytes()));
  RecordProperty("scratch_bytes", std::to_string(life->program->scratch_bytes()));
  RecordProperty("pool_bytes", std::to_string(life->program->pool_bytes()));
  RecordProperty("program_host_bytes", std::to_string(life->program->host_bytes()));
  RecordProperty("graph_nodes", std::to_string(life->graph->graph().nodes().size()));
  RecordProperty("graph_tensors", std::to_string(life->graph->graph().tensor_count()));
  RecordProperty("plan", llmp::base::ToHex(life->program->bound()->identity()));
  RecordProperty("eager", std::to_string(life->stats.eager));
  RecordProperty("captured", std::to_string(life->stats.captured));
  RecordProperty("replayed", std::to_string(life->stats.replayed));
  RecordProperty("capture_refused", std::to_string(life->stats.refused));
  for (std::size_t branch = 0; branch < 2; ++branch) {
    RecordProperty("branch" + std::to_string(branch) + "_original_sha", life->original_sha[branch]);
    RecordProperty("branch" + std::to_string(branch) + "_changed_sha", life->changed_sha[branch]);
    RecordProperty("branch" + std::to_string(branch) + "_image_sha", life->image_sha[branch]);
  }
  std::string implementations;
  for (std::size_t i = 0; i < life->program->bound()->size(); ++i)
    implementations += life->program->bound()->at(i).name + ",";
  ASSERT_LE(implementations.size(), 16384U);
  RecordProperty("implementations", implementations);
  auto retired = life->Retire();
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
}
TEST(ImageTimestepGpuTest, Complete32BlockSourceFlowAndEulerReplayChangedInputs) {
  const auto* path = std::getenv("LLMP_TEST_DENOISER_ARTIFACT");
  if (path == nullptr)
    GTEST_SKIP() << "needs LLMP_TEST_DENOISER_ARTIFACT (an installed Qwen-Image denoiser artifact)";
  std::unique_ptr<Lifetime, Retire> life(new Lifetime(false, true));
  auto setup = SetupFixture(*life, path);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto ran = Execute(*life);
  ASSERT_TRUE(ran) << Failed(ran).value_or("");
  EXPECT_EQ(life->stats.eager, 1U);
  EXPECT_EQ(life->stats.captured, 1U);
  EXPECT_EQ(life->stats.replayed, 5U);
  EXPECT_EQ(life->stats.refused, 0U);
  RecordProperty("source_bytes", std::to_string(kFlowSourceBytes));
  RecordProperty("source_parts", "297");
  RecordProperty("blocks", "32");
  RecordProperty("selected_pages", "6818");
  RecordProperty("reserved_pages", "6818");
  RecordProperty("selected_weight_page_bytes", "14298382336");
  RecordProperty("reserved_weight_bytes", "14298382336");
  RecordProperty("complete_output_values_per_step", "512");
  RecordProperty("finite_values_checked", "3584");
  RecordProperty("completed_steps", "7");
  RecordProperty("changed_time_branches", "2");
  RecordProperty("changed_image_branches", "2");
  RecordProperty("euler_values_checked", "1792");
  RecordProperty("input_boundary",
                 "source BF16 sinusoid widened exactly to F32; F32 text/latent/dt");
  RecordProperty("metadata_bytes", std::to_string(life->graph->metadata_bytes()));
  RecordProperty("scratch_bytes", std::to_string(life->program->scratch_bytes()));
  RecordProperty("pool_bytes", std::to_string(life->program->pool_bytes()));
  RecordProperty("program_host_bytes", std::to_string(life->program->host_bytes()));
  RecordProperty("graph_nodes", std::to_string(life->graph->graph().nodes().size()));
  RecordProperty("graph_tensors", std::to_string(life->graph->graph().tensor_count()));
  RecordProperty("plan", llmp::base::ToHex(life->program->bound()->identity()));
  RecordProperty("eager", std::to_string(life->stats.eager));
  RecordProperty("captured", std::to_string(life->stats.captured));
  RecordProperty("replayed", std::to_string(life->stats.replayed));
  RecordProperty("capture_refused", std::to_string(life->stats.refused));
  for (std::size_t branch = 0; branch < 2; ++branch) {
    RecordProperty("branch" + std::to_string(branch) + "_original_sha", life->original_sha[branch]);
    RecordProperty("branch" + std::to_string(branch) + "_changed_sha", life->changed_sha[branch]);
    RecordProperty("branch" + std::to_string(branch) + "_image_sha", life->image_sha[branch]);
  }
  std::map<std::string, std::uint64_t> counts;
  for (std::size_t i = 0; i < life->program->bound()->size(); ++i)
    ++counts[life->program->bound()->at(i).name];
  std::string implementations;
  for (const auto& [name, count] : counts)
    implementations += name + ":" + std::to_string(count) + ",";
  ASSERT_LE(implementations.size(), 16384U);
  RecordProperty("implementations", implementations);
  auto retired = life->Retire();
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
}
}  // namespace
