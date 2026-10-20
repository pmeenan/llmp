// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Actual installed F32 VAE parts, through the shared typed graph and StepRuns.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/sha256.h"
#include "compiler_inventory.h"
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
#include "image_family.h"
#include "image_published.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/step_program.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "spec/image_spatial.h"
#include "spec/stored.h"

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
constexpr std::uint64_t kDomain = 1024 * kMiB, kMetadata = 512 * kMiB;
constexpr std::uint64_t kCompilerAndOwner = 64 * kMiB, kProgram = 8 * kMiB;
constexpr std::uint64_t kActivation = 64 * kMiB, kPool = 2 * kMiB;
constexpr std::uint64_t kPinned = 32 * kMiB, kOutputOffset = 2 * kMiB;
constexpr std::uint64_t kWeightOffset = 26 * kMiB;
const std::string kArtifact = llmp::test_support::QwenImageVae();
constexpr std::array<std::string_view, 6> kNames{"post_quant_conv.weight",
                                                 "post_quant_conv.bias",
                                                 "decoder.conv_in.weight",
                                                 "decoder.conv_in.bias",
                                                 "decoder.up_blocks.3.upsampler.resample.1.weight",
                                                 "decoder.up_blocks.3.upsampler.resample.1.bias"};
constexpr std::array<std::uint64_t, 6> kWeightBytes{16384, 256, 2654208, 4608, 2985984, 1152};
constexpr std::uint64_t kTotalWeightBytes = 5662592;
// Four copies of tiny IR/rewrite/placement inventories and native descriptors,
// plus two retained programs and their captures, fit the fixed compiler owner.
// This prior native category becomes available to actual backend charges;
// it is no longer also held by the enclosing compiler payment.
constexpr std::uint64_t kNativeConstruction =
    llmp::test_support::compiler_inventory::NativeBytes((64 + 16 + 31 * 32ULL + 4), 64);
static_assert(llmp::test_support::compiler_inventory::GraphBytes(32, 64) + 4096 * (32 + 64ULL) +
                  llmp::test_support::compiler_inventory::NativeBytes((64 + 16 + 31 * 32ULL + 4),
                                                                      64) +
                  2 * kProgram + 32 * en::kGraphNodeHostBytes + 32 * kMiB <
              kCompilerAndOwner);

en::Status Error(std::string what) { return std::unexpected(std::move(what)); }
std::string Hash(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  return llmp::base::ToHex(hash.Finish());
}
struct SpanCase {
  sp::ModelSpec spec;
  std::optional<co::SpatialGraph> graph;
  std::optional<ir::Graph> direct;
  std::vector<co::StepWeight> direct_weights;
  bool pointwise = false, tiled = false;
  llmp::planner::Policy policy;
  std::map<std::pair<ir::WeightRef, fm::PartRole>, std::uint32_t> bindings;
  std::unique_ptr<kn::StepProgram> program;
  en::StepPlanRuns runs;
  co::SpatialShape shape;
  std::vector<float> input;
  std::vector<double> reference;
  double error_enclosure = 0;
  std::string eager_hash;
  std::array<std::uint64_t, 9> elapsed_ns{};
};

struct Lifetime final : en::PagedModel {
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  en::PagedWeights weights;
  en::RunnerResources resources{node, 0, 0};
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kn::StepLaunch> launch;
  std::array<SpanCase, 2> cases;
  std::array<std::uint32_t, 6> resource_ids{};
  std::array<std::uint64_t, 6> host_offsets{};
  std::byte* host = nullptr;
  en::StepRuns step_runs{true};
  en::GraphStats stats;
  ca::Closure closure, empty;
  std::uint64_t selected_pages = 0;
  bool tiled = false;
  std::uint64_t activation = kActivation, pool = kPool, pinned = kPinned;
  std::uint64_t output_offset = kOutputOffset, weight_offset = kWeightOffset;
  std::uint64_t guard = 0;
  bool opened = false, retirement_attempted = false, retired = false;

  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return empty; }
  std::vector<ca::ExtentId> managed_extents() const override { return weights.extents(); }
  en::Status Release() override {
    for (auto& item : cases) {
      item.runs.DropGraph();
      item.program.reset();
      item.direct.reset();
      item.direct_weights.clear();
      item.graph.reset();
    }
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
    const auto done = life->Retire();
    EXPECT_TRUE(done) << Failed(done).value_or("");
    if (done) delete life;  // Unknown completion retains the complete owner.
  }
};

kn::Addresses Addresses(Lifetime& life, SpanCase& item, bool measuring) {
  auto address = [&life, &item](const ir::WeightRef& ref, fm::PartRole role) {
    const auto found = item.bindings.find({ref, role});
    return found == item.bindings.end() ? 0ULL : life.weights.resource_address(found->second);
  };
  kn::Addresses result;
  result.weight = address;
  result.activations = measuring ? 0 : life.node.activations().base + life.guard;
  result.activation_bytes = life.activation;
  result.activation_alignment = 256;
  result.weight_alignment = [address](const ir::WeightRef& ref, fm::PartRole role) {
    const auto actual = address(ref, role);
    return actual == 0 ? 0ULL : std::gcd(256ULL, actual);
  };
  return result;
}

en::Status CheckProgram(const SpanCase& item, const kn::StepProgram& program, std::size_t index) {
  if (program.inputs().size() != 1 || program.outputs().size() != 1 ||
      program.inputs()[0].kind != kn::InputKind::kStepBytes || program.inputs()[0].segment != 0 ||
      program.outputs()[0].segment != 0 || program.inputs()[0].bytes != item.input.size() * 4 ||
      program.outputs()[0].bytes !=
          (item.tiled ? item.shape.width * item.shape.height * 1152 * 4
                      : (item.pointwise ? 1048576ULL : (index == 0 ? 18874368ULL : 73728ULL))) ||
      program.scratch_bytes() > (item.tiled ? 128 * kMiB : kActivation) ||
      program.pool_bytes() > (item.tiled ? 64 * kMiB : kPool) || program.host_bytes() > kProgram ||
      program.launches() > (item.tiled ? 13U : 8U) || !program.capturable() || program.routed())
    return Error("spatial program ownership, complete ports or paid envelope differ");
  // The 1x1 and 3x3 convolutions, both the direct one (the span's rank).
  const std::array<std::string_view, 4> prefix{"ggml.conv2d.f32", "ggml.add", "ggml.conv2d.f32",
                                               "ggml.add"};
  const std::array<std::string_view, 3> up{"ggml.upsample.nearest", "ggml.conv2d.f32", "ggml.add"};
  const std::array<std::string_view, 2> direct{"ggml.conv2d.f32", "ggml.add"};
  // The 1x1 convolution's own implementation; the tiled one adds its bias.
  const std::array<std::string_view, 1> product{kg::kConv1x1Name};
  const std::array<std::string_view, 1> tiled{kg::kConv2dTiledF32};
  const std::span<const std::string_view> expected =
      item.tiled       ? (index == 0 ? std::span<const std::string_view>(direct)
                                     : std::span<const std::string_view>(tiled))
      : item.pointwise ? (item.direct ? std::span<const std::string_view>(direct)
                                      : std::span<const std::string_view>(product))
                       : (index == 0 ? std::span<const std::string_view>(prefix)
                                     : std::span<const std::string_view>(up));
  if (item.tiled && index == 1) {
    const bool small = item.shape.width == 7;
    if (program.pool_bytes() != (small ? 403200ULL : 66355200ULL) ||
        program.launches() != (small ? 4U : 12U))
      return Error("actual tiled workspace/launch partition differs from checked stripes");
  }
  if (!program.bound())
    return {};  // Measurement cannot bind; the real program is checked after
                // Bind.
  if (program.bound()->size() != expected.size()) return Error("physical spatial step count");
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (program.bound()->at(i).name != expected[i]) return Error("physical spatial selection");
  return {};
}

en::GraphBuildIO GraphIO(const SpanCase& item) {
  const auto& graph = item.direct ? *item.direct : item.graph->graph();
  return {
      graph, graph.inputs(), graph.outputs(),
      item.direct ? std::span<const co::StepWeight>(item.direct_weights) : item.graph->weights()};
}

en::Status SetupFixture(Lifetime& life, const std::filesystem::path& path, bool pointwise = false,
                        std::uint64_t tile_width = 0, std::uint64_t tile_height = 0) {
  life.tiled = tile_width != 0;
  if (life.tiled) {
    if (!((tile_width == 7 && tile_height == 5) || (tile_width == 128 && tile_height == 128)))
      return Error("fixed diagnostic shape inventory");
    life.activation = 128 * kMiB;
    life.pool = 64 * kMiB;
    life.pinned = 128 * kMiB;
    life.output_offset = 8 * kMiB;
    life.weight_offset = 96 * kMiB;
    life.guard = 256;
  }
  auto available = llmp::platform::AvailableMemoryBytes();
  if (!available || *available < (4ULL << 30U) + (6ULL << 30U)) return Error("physical headroom");
  if (auto r = life.node.Open(); !r) return r;
  life.opened = true;
  life.node.SetHostFloor(0);
  auto paid = en::ExecutionMetadataAllowance::Admit(life.node.catalog(), life.node.domain(),
                                                    kMetadata, *available, 6ULL << 30U);
  if (!paid) return Error(paid.error());
  life.paid = *paid;
  if (!life.paid->Charge(kCompilerAndOwner - kNativeConstruction))
    return Error("preconstruction payment");
  std::error_code ec;
  if (std::filesystem::file_size(path / "manifest.json", ec) != 1312 || ec ||
      std::filesystem::file_size(path / "index.json", ec) != 68162 || ec)
    return Error("fixed metadata extent");
  if (auto r = life.weights.Open(path, {.expected_id = std::string(kArtifact)}); !r) return r;
  const auto& artifact = life.weights.artifact();
  if (artifact.resources().size() != 128 || artifact.groups().size() != 6 ||
      artifact.shards().size() != 1 || artifact.format_version() != 1)
    return Error("opened VAE inventory");
  const auto tensors = mo::VaeDecoderTensors(::llmp::test_support::QwenImageProfile().vae,
                                             ::llmp::test_support::QwenImageNames().vae);
  auto bound = mo::BindImageComponent(tensors, "AutoencoderKLQwenImage21", artifact);
  if (!bound) return Error(bound.error());
  auto source = sp::SourceOf(artifact);
  if (!source) return Error(source.error());
  if (source->tensors.size() != 128) return Error("fixed source tensor metadata");
  std::vector<en::GroupPlace> groups(6, en::GroupPlace::kNone);
  std::uint64_t offset = life.weight_offset;
  for (std::size_t i = 0; i < kNames.size(); ++i) {
    const auto id = artifact.FindResource(kNames[i]);
    if (!id) return Error("actual source role missing");
    auto placement = artifact.ResourcePlacement(*id);
    if (!placement || placement->bytes.value() != kWeightBytes[i])
      return Error("actual source part extent");
    life.resource_ids[i] = *id;
    groups.at(placement->group) = en::GroupPlace::kDevice;
    life.host_offsets[i] = offset;
    offset += kWeightBytes[i];
  }
  if (offset != life.weight_offset + kTotalWeightBytes || offset > life.pinned)
    return Error("weight host layout");
  if (auto r = life.weights.Reserve(life.node, groups, {}); !r) return r;
  if (life.weights.bytes() != 241172480ULL || life.weights.host_bytes() != 0 ||
      life.weights.extents().size() != 115)
    return Error(std::format("reserved source page geometry: {} bytes, {} extents",
                             life.weights.bytes(), life.weights.extents().size()));

  auto steps = mo::VaeDecoderPlan(::llmp::test_support::QwenImageProfile().vae, 64, 64);
  std::array<std::vector<mo::VaeStep>, 2> spans;
  spans[0].assign(steps.begin(), steps.begin() + 2);
  for (std::size_t i = 0; i + 1 < steps.size(); ++i) {
    if (steps[i].kind != mo::VaeStep::Kind::kUpsample || steps[i + 1].weight < 0 ||
        tensors[static_cast<std::size_t>(steps[i + 1].weight)].name != kNames[4])
      continue;
    spans[1] = {steps[i], steps[i + 1]};
  }
  if (spans[1].size() != 2) return Error("actual upsample span");
  // Reduced geometry is explicit; tensor identities/channels/kernel stay
  // actual.
  spans[1][0].height = spans[1][0].width = 4;
  spans[1][1].height = spans[1][1].width = 8;
  if (pointwise) {
    spans[0] = {steps[0]};
    spans[1] = spans[0];
  }
  if (life.tiled) {
    spans[0] = {steps[1]};
    spans[0][0].width = static_cast<std::uint32_t>(tile_width);
    spans[0][0].height = static_cast<std::uint32_t>(tile_height);
    spans[1] = spans[0];
  }
  const std::array<std::span<const co::BoundTensor>, 1> tables{source->tensors};
  for (std::size_t i = 0; i < 2; ++i) {
    auto& item = life.cases[i];
    item.pointwise = pointwise;
    item.tiled = life.tiled;
    if (life.tiled) {
      // Explicit diagnostic arm selection, not a measured production rank.
      item.policy.ranks.push_back(
          {"sm_121", i == 0 ? "ggml.conv2d.f32" : std::string(kg::kConv2dTiledF32), 0});
    } else if (i == 0) {
      // The direct convolution, the baseline the other span's selection
      // (the 1x1 convolution's own, or the planned default) is held to.
      item.policy.ranks.push_back({"sm_121", "ggml.conv2d.f32", 0});
    }
    auto spec = sp::VaeSpatialSpec(spans[i], tensors);
    if (!spec) return Error(spec.error());
    item.spec = std::move(*spec);
    item.shape = life.tiled ? co::SpatialShape{tile_width, tile_height, 64, 1}
                            : ((pointwise || i == 0) ? co::SpatialShape{64, 64, 64, 1}
                                                     : co::SpatialShape{4, 4, 288, 1});
    const std::array owners{co::SpatialOwner{0, item.shape}};
    auto graph =
        co::BuildSpatialStep(item.spec, tables, owners,
                             {[paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
                              [paid = life.paid](std::uint64_t n) { paid->Release(n); }},
                             {.reference_buffers = !life.tiled});
    if (!graph) return Error(graph.error());
    if (graph->graph().nodes().size() > 32 || graph->graph().tensor_count() > 64 ||
        graph->weights().size() != ((pointwise || life.tiled) ? 2U : (i == 0 ? 4U : 2U)))
      return Error("fixed spatial graph inventory");
    item.graph.emplace(std::move(*graph));
    for (const auto& weight : item.graph->weights()) {
      if (weight.parts.size() != 1 ||
          item.graph->graph().tensor(weight.tensor).format.encoding != fm::Encoding::kF32)
        return Error("resident source format");
      const auto& [name, role] = weight.parts[0];
      const auto id = artifact.FindResource(name);
      if (!id || std::ranges::find(life.resource_ids, *id) == life.resource_ids.end() ||
          !item.bindings.emplace(std::pair{weight.ref, role}, *id).second)
        return Error("complete component weight binding");
    }
    if (pointwise && i == 0) {
      // Same source-bound operation and ports; direct baseline is explicit IR,
      // not a production selector or a different source/precision policy.
      ir::Builder b(16);
      std::array<ir::TensorId, 2> parts{};
      for (const auto& weight : item.graph->weights()) {
        auto copy = weight;
        copy.tensor = b.Weight(item.graph->graph().tensor(weight.tensor).format, weight.ref);
        parts.at(weight.ref.role) = copy.tensor;
        item.direct_weights.push_back(std::move(copy));
      }
      b.SetOwner(0);
      const auto input = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({64, 64, 64, 1})));
      b.Output(b.Conv2d(parts[0], input, parts[1], {.stride = 1, .padding = 0, .dilation = 1}));
      auto direct = std::move(b).Finish();
      if (!direct) return Error("direct pointwise graph validation");
      item.direct.emplace(std::move(*direct));
    }
    item.input.resize(item.shape.width * item.shape.height * item.shape.channels);
    for (std::size_t n = 0; n < item.input.size(); ++n)
      item.input[n] = static_cast<float>(static_cast<int>((n * 17 + 3) % 257) - 128) / 128.F;
  }
  auto backend = kg::GgmlBackend::Create(0, life.node.execution(), life.node.stream(0), nullptr);
  if (!backend) return Error(backend.error());
  life.backend = std::move(*backend);
  if (auto r = life.resources.OpenSelectedWorkspace("selected kernel state",
                                                    life.backend->persistent_bytes());
      !r)
    return r;
  const auto facts = llmp::providers::QueryDeviceFacts(0);
  if (!facts) return Error("actual device facts");
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  if (workspace.value() > 64 * kMiB) return Error("handle workspace");
  std::vector<llmp::execution::TunedShape> tuning;
  if (life.tiled)
    tuning.push_back({.implementation = kg::kConv2dTiledF32,
                      .rows = tile_width * tile_height,
                      .k = 576,
                      .n = 1152,
                      .output_encoding = fm::Encoding::kF32});
  if (auto r = life.backend->FundConstruction(
          [paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
          [paid = life.paid](std::uint64_t n) { paid->Release(n); });
      !r)
    return r;
  if (auto r = life.backend->Declare(
          tuning, workspace.value(),
          [paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
          [paid = life.paid](std::uint64_t n) { paid->Release(n); });
      !r)
    return Error(r.error());
  if (auto r = life.resources.OpenCublas("spatial qualified handle"); !r) return r;
  if (auto r = life.backend->LendCublas(&life.resources.cublas()); !r) return Error(r.error());
  const auto owned = life.backend->Backend();
  if (!owned.owner || !owned.owner->Ready()) return Error("owned backend readiness");
  const auto& table = owned.owner->backend();
  for (std::size_t i = 0; i < 2; ++i) {
    auto& item = life.cases[i];
    auto io = GraphIO(item);
    const auto addresses = Addresses(life, item, true);
    auto measured = en::BuildComponentProgram(
        io, table, item.policy, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = table.written,
         .within = table.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    if (!measured) return Error(measured.error());
    if (auto r = CheckProgram(item, **measured, i); !r) return r;
  }
  if (auto r = life.node.MapWorkspace(life.activation + 2 * life.guard, life.pool + 2 * life.guard);
      !r)
    return r;
  auto launch = life.backend->Launch({life.node.pool().base + life.guard, Bytes(life.pool)},
                                     {.base = life.resources.selected_workspace().base,
                                      .bytes = life.resources.selected_workspace().bytes});
  if (!launch) return Error(launch.error());
  life.launch = std::move(*launch);
  auto host = life.resources.Pinned(life.pinned);
  if (!host) return Error(host.error());
  life.host = static_cast<std::byte*>(*host);
  life.step_runs.SetStaging(life.host, life.tiled ? 4 * kMiB : 2 * kMiB);
  life.step_runs.SetLaunch(life.launch.get());
  for (std::size_t i = 0; i < 2; ++i) {
    auto& item = life.cases[i];
    auto io = GraphIO(item);
    const auto addresses = Addresses(life, item, false);
    auto program = en::BuildComponentProgram(
        io, table, item.policy, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = table.written,
         .within = table.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    if (!program) return Error(program.error());
    if (auto r = CheckProgram(item, **program, i); !r) return r;
    // The enclosing compiler/owner charge already prepays two retained
    // kProgram slots. Transfer only after that real bound has been checked.
    (*program)->ConstructionPaid();
    item.program = std::move(*program);
    if (auto r = item.program->Bind(*life.launch); !r) return Error(r.error().detail);
    if (!item.program->bound()) return Error("actual program did not bind");
    if (auto r = CheckProgram(item, *item.program, i); !r) return r;
  }
  if (auto r = life.node.Start(Bytes(kDomain)); !r) return r;
  if (auto r = life.weights.Register(life.node, 0); !r) return r;
  // Closure construction remains before Run: no live catalog read off
  // scheduler.
  std::vector<ca::ExtentId> extents = life.resources.extents();
  const auto append = [&](const auto& more) {
    extents.insert(extents.end(), more.begin(), more.end());
  };
  append(life.node.activations().extents);
  append(life.node.pool().extents);
  std::vector<ca::ExtentId> selected;
  for (std::size_t i = 0; i < 6; ++i) {
    std::array<ca::ExtentId, 4> pages{};
    auto count = life.weights.ExtentsTouching(life.weights.resource_address(life.resource_ids[i]),
                                              kWeightBytes[i], pages);
    if (!count) return Error(count.error());
    selected.insert(selected.end(), pages.begin(),
                    pages.begin() + static_cast<std::ptrdiff_t>(*count));
  }
  std::ranges::sort(selected);
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  if (selected.size() != 5) return Error("selected source-page closure");
  life.selected_pages = selected.size();
  append(selected);
  auto closure = life.node.catalog().ClosureOfExtents(extents);
  if (!closure) return Error("complete runtime/source closure");
  life.closure = std::move(*closure);
  life.node.Run();
  return {};
}

std::span<const float> WeightValues(const Lifetime& life, std::string_view name) {
  const auto found = std::ranges::find(kNames, name);
  if (found == kNames.end()) return {};
  const auto i = static_cast<std::size_t>(found - kNames.begin());
  return {reinterpret_cast<const float*>(life.host + life.host_offsets[i]), kWeightBytes[i] / 4};
}

en::Status Reference(Lifetime& life, SpanCase& item) {
  auto instances = co::InstancesOf(item.spec, 0);
  if (!instances) return Error(instances.error());
  item.reference.assign(item.input.begin(), item.input.end());
  auto shape = item.shape;
  double error = 0;
  for (std::size_t i = 0; i < instances->layers[0].size(); ++i) {
    const auto& instance = instances->layers[0][i];
    const auto& source = item.spec.stages[0].layers[0].components[i];
    std::span<const float> kernel, bias;
    if (source.kind == "conv.spatial") {
      for (const auto& binding : source.weights) {
        if (binding.role == "kernel") kernel = WeightValues(life, binding.tensor);
        if (binding.role == "bias") bias = WeightValues(life, binding.tensor);
      }
      if (kernel.empty() || bias.empty()) return Error("reference source roles");
      const auto terms = kernel.size() / bias.size();
      double sum_abs = 0, max_bias = 0, max_input = 0, max_weight = 0;
      for (const auto value : item.reference) max_input = std::max(max_input, std::abs(value));
      for (std::size_t c = 0; c < bias.size(); ++c) {
        double sum = 0;
        for (std::size_t k = 0; k < terms; ++k) {
          const auto magnitude = std::abs(static_cast<double>(kernel[c * terms + k]));
          sum += magnitude;
          max_weight = std::max(max_weight, magnitude);
        }
        sum_abs = std::max(sum_abs, sum);
        max_bias = std::max(max_bias, std::abs(static_cast<double>(bias[c])));
      }
      // FP32 separate multiply/add is conservative for fused FMA too. Include
      // F64 oracle roundoff and subnormal flushing, no historical bit target.
      const double n = static_cast<double>(2 * terms + 2);
      const double gamma = (n * 0x1p-24) / (1 - n * 0x1p-24) * (1 + n * 0x1p-50);
      const bool pointwise =
          instance.params.Int("kernel") == 1 && instance.params.Int("padding") == 0;
      // Existing F32 cuBLAS may round both multiplicands to TF32. Bound that
      // source-relative error explicitly, without forcing intermediate casts.
      const double product_round =
          pointwise ? (2 * 0x1p-11 + 0x1p-22) * (max_input + error) * sum_abs : 0;
      error = error * sum_abs + product_round +
              gamma * ((max_input + error) * sum_abs + product_round + max_bias) +
              n * 0x1p-126 * (1 + max_input + error + max_weight) +
              n * 0x1p-52 * (max_input * sum_abs + max_bias);
    }
    co::SpatialRefIO ref{instance, shape, item.reference, kernel, bias, {}};
    if (auto r = co::RunSpatialReference(ref); !r) return Error(r.error());
    auto next = co::InferSpatialShape(instance, shape);
    if (!next) return Error(next.error());
    shape = *next;
    item.reference = std::move(ref.output);
  }
  if (!std::isfinite(error) || error < 0) return Error("nonfinite arithmetic enclosure");
  item.error_enclosure = error;
  return {};
}

en::Status Execute(Lifetime& life) {
  return life.node.WithRequest(0, life.closure, "actual spatial parts", [&]() -> en::Status {
    for (std::size_t i = 0; i < 6; ++i) {
      if (auto r =
              life.node.Copy(0, life.closure, life.weights.resource_address(life.resource_ids[i]),
                             life.host + life.host_offsets[i], kWeightBytes[i], true,
                             "completed source reference copy");
          !r)
        return r;
      const auto values = WeightValues(life, kNames[i]);
      if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); }))
        return Error("nonfinite source weights");
    }
    for (std::size_t i = 0; i < 2; ++i) {
      auto& item = life.cases[i];
      if (auto r = Reference(life, item); !r) return r;
      auto coverage = life.node.Call(
          [&]() -> en::Status {
            en::Coverage result;
            en::CheckCoverage(life.node, 0, item.program->tensors(), "actual spatial", result);
            return result.violations ? Error(result.first_violation) : en::Status{};
          },
          "spatial catalog coverage");
      if (!coverage) return coverage;
      for (std::size_t repeat = 0; repeat < (item.pointwise ? 7U : 3U); ++repeat) {
        auto copies = life.step_runs.Stage(
            item.program->inputs(),
            [&](std::size_t slot, std::span<std::byte> out) -> en::Status {
              if (slot != 0 || out.size() != item.input.size() * 4)
                return Error("input slot extent");
              std::memcpy(out.data(), item.input.data(), out.size());
              return {};
            },
            0);
        if (!copies) return Error(copies.error());
        const auto& output = item.program->outputs()[0];
        const std::array<en::RunCopy, 1> outputs{{{en::support::Address(life.host + kOutputOffset),
                                                   output.device_address, output.bytes}}};
        en::StepQueued queued;
        const auto begin = std::chrono::steady_clock::now();
        auto ran = life.node.Job(
            life.closure,
            [&](llmp::providers::NativeStream native) {
              queued = life.step_runs.Queue(item.runs, *copies, {}, *item.program, outputs,
                                            item.runs.CaptureDue(true), life.stats, native);
              if (queued.result) return sc::JobResult::kQueued;
              return queued.before || queued.result.error().error == kn::StepError::kUnknown
                         ? sc::JobResult::kUnknown
                         : sc::JobResult::kNotStarted;
            },
            "spatial eager capture replay", 0);
        item.elapsed_ns[repeat] =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - begin)
                                           .count());
        if (!ran) return ran;
        if (!queued.result) return Error(queued.result.error().detail);
        const auto expected = repeat == 0   ? en::RunPath::kEager
                              : repeat == 1 ? en::RunPath::kCaptured
                                            : en::RunPath::kReplayed;
        if (queued.path != expected) return Error("required actual capture/replay path");
        en::Count(life.stats, queued.path);
        const auto actual =
            std::span(reinterpret_cast<const float*>(life.host + kOutputOffset), output.bytes / 4);
        if (actual.size() != item.reference.size()) return Error("complete reference extent");
        double max_error = 0, squared = 0;
        for (std::size_t j = 0; j < actual.size(); ++j) {
          if (!std::isfinite(actual[j]) || !std::isfinite(item.reference[j]))
            return Error("nonfinite full output");
          const double difference = std::abs(static_cast<double>(actual[j]) - item.reference[j]);
          max_error = std::max(max_error, difference);
          squared += difference * difference;
        }
        if (max_error > item.error_enclosure)
          return Error("FP32 source arithmetic error enclosure");
        const auto sha = Hash(std::as_bytes(actual));
        if (repeat == 0)
          item.eager_hash = sha;
        else if (sha != item.eager_hash)
          return Error("same selected program replay differs");
        if (repeat == 0) {
          ::testing::Test::RecordProperty("span" + std::to_string(i) + "_max_error",
                                          std::format("{:.17g}", max_error));
          ::testing::Test::RecordProperty(
              "span" + std::to_string(i) + "_rms_error",
              std::format("{:.17g}", std::sqrt(squared / static_cast<double>(actual.size()))));
          ::testing::Test::RecordProperty("span" + std::to_string(i) + "_enclosure",
                                          std::format("{:.17g}", item.error_enclosure));
        }
      }
    }
    return {};
  });
}

void RunFixture(bool pointwise) {
  const auto path = llmp::test_support::QwenImagePublished(kArtifact);
  std::unique_ptr<Lifetime, Retire> life(new Lifetime);
  auto setup = SetupFixture(*life, path, pointwise);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto ran = Execute(*life);
  ASSERT_TRUE(ran) << Failed(ran).value_or("");
  EXPECT_EQ(life->stats.eager, 2U);
  EXPECT_EQ(life->stats.captured, 2U);
  EXPECT_EQ(life->stats.replayed, pointwise ? 10U : 2U);
  EXPECT_EQ(life->stats.refused, 0U);
  ::testing::Test::RecordProperty("actual_source_bytes", std::to_string(kTotalWeightBytes));
  ::testing::Test::RecordProperty("selected_pages", std::to_string(life->selected_pages));
  ::testing::Test::RecordProperty("complete_output_values", pointwise ? "524288" : "4737024");
  ::testing::Test::RecordProperty("eager", std::to_string(life->stats.eager));
  ::testing::Test::RecordProperty("captured", std::to_string(life->stats.captured));
  ::testing::Test::RecordProperty("replayed", std::to_string(life->stats.replayed));
  ::testing::Test::RecordProperty("capture_refused", std::to_string(life->stats.refused));
  for (std::size_t i = 0; i < kNames.size(); ++i) {
    const auto values = WeightValues(*life, kNames[i]);
    ::testing::Test::RecordProperty("source_part" + std::to_string(i) + "_sha",
                                    Hash(std::as_bytes(values)));
  }
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& item = life->cases[i];
    ::testing::Test::RecordProperty("span" + std::to_string(i) + "_plan",
                                    llmp::base::ToHex(item.program->bound()->identity()));
    std::string implementations;
    for (std::size_t op = 0; op < item.program->bound()->size(); ++op)
      implementations += item.program->bound()->at(op).name + ",";
    ASSERT_LE(implementations.size(), 1024U);
    ::testing::Test::RecordProperty("span" + std::to_string(i) + "_implementations",
                                    implementations);
    std::string parameters;
    for (const auto n : item.graph->parameters()) parameters += std::to_string(n) + ",";
    ASSERT_LE(parameters.size(), 1024U);
    ::testing::Test::RecordProperty("span" + std::to_string(i) + "_parameters", parameters);
    ::testing::Test::RecordProperty("span" + std::to_string(i) + "_input_sha",
                                    Hash(std::as_bytes(std::span(item.input))));
    ::testing::Test::RecordProperty("span" + std::to_string(i) + "_sha", life->cases[i].eager_hash);
    for (std::size_t j = 0; j < (pointwise ? 7U : 3U); ++j)
      ::testing::Test::RecordProperty(
          "span" + std::to_string(i) + "_step" + std::to_string(j) + "_ns",
          std::to_string(life->cases[i].elapsed_ns[j]));
  }
  auto retired = life->Retire();
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
}
TEST(ImageSpatialGpuTest, ActualF32VaePartsUseSharedTypedProgramsAndCompletedReplay) {
  RunFixture(false);
}
TEST(ImageSpatialGpuTest, PointwiseSourceDirectAndSharedProductCompleteReplay) { RunFixture(true); }

struct ConvSample {
  std::size_t index;
  double value, bound;
};
std::vector<ConvSample> ConvSamples(const Lifetime& life, const SpanCase& item, bool tensor_core,
                                    bool small) {
  const auto weight = WeightValues(life, "decoder.conv_in.weight");
  const auto bias = WeightValues(life, "decoder.conv_in.bias");
  const auto width = static_cast<std::size_t>(item.shape.width);
  const auto height = static_cast<std::size_t>(item.shape.height);
  const std::array<std::size_t, 7> boundary_y{0, 1, 44, 45, 89, 90, 127};
  const std::array<std::size_t, 4> boundary_x{0, 1, 126, 127};
  std::vector<ConvSample> samples;
  samples.reserve(small ? width * height * 1152 : boundary_y.size() * boundary_x.size() * 1152);
  const auto sample = [&](std::size_t channel, std::size_t y, std::size_t x) {
    double value = static_cast<double>(bias[channel]), magnitude = 0, max_weight = 0;
    for (std::size_t input_channel = 0; input_channel < 64; ++input_channel) {
      for (std::size_t ky = 0; ky < 3; ++ky) {
        for (std::size_t kx = 0; kx < 3; ++kx) {
          const auto iy = static_cast<std::int64_t>(y + ky) - 1;
          const auto ix = static_cast<std::int64_t>(x + kx) - 1;
          if (iy < 0 || ix < 0 || iy >= static_cast<std::int64_t>(height) ||
              ix >= static_cast<std::int64_t>(width))
            continue;
          const auto w =
              static_cast<double>(weight[kx + 3 * (ky + 3 * (input_channel + 64 * channel))]);
          const auto v = static_cast<double>(
              item.input[static_cast<std::size_t>(ix) +
                         width * (static_cast<std::size_t>(iy) + height * input_channel)]);
          value += w * v;
          magnitude += std::abs(w * v);
          max_weight = std::max(max_weight, std::abs(w));
        }
      }
    }
    // Independent scalar F64 summation, no candidate packing/kernel code.
    // Conservative one-ULP TF32 multiplicand envelope permits the library's
    // existing math choice without requesting or inserting any rounding.
    constexpr double n = 2 * 576 + 2;
    constexpr double gamma = n * 0x1p-24 / (1 - n * 0x1p-24);
    const double product_round = tensor_core ? (2 * 0x1p-10 + 0x1p-20) * magnitude : 0;
    const double bound =
        product_round +
        gamma * (magnitude + product_round + std::abs(static_cast<double>(bias[channel]))) +
        n * 0x1p-52 * (magnitude + std::abs(static_cast<double>(bias[channel]))) +
        n * 0x1p-126 * (3 + max_weight);
    samples.push_back({x + width * (y + height * channel), value, bound});
  };
  for (std::size_t channel = 0; channel < 1152; ++channel) {
    if (small) {
      for (std::size_t y = 0; y < height; ++y)
        for (std::size_t x = 0; x < width; ++x) sample(channel, y, x);
    } else {
      for (const auto y : boundary_y)
        for (const auto x : boundary_x) sample(channel, y, x);
    }
  }
  return samples;
}
en::Status TiledCanaries(Lifetime& life, bool initialize) {
  const std::array device{life.node.activations().base,
                          life.node.activations().base + life.guard + life.activation,
                          life.node.pool().base, life.node.pool().base + life.guard + life.pool};
  for (std::size_t i = 0; i < device.size(); ++i) {
    auto* host = life.host + 120 * kMiB + i * 256;
    if (initialize) std::memset(host, 0xa5, 256);
    if (auto done = life.node.Copy(0, life.closure, device[i], host, 256, !initialize,
                                   "completed activation/pool canary");
        !done)
      return done;
    if (!std::ranges::all_of(std::span(host, 256),
                             [](std::byte b) { return b == std::byte{0xa5}; }))
      return Error("tiled activation/pool boundary canary changed");
  }
  return {};
}
en::Status ExecuteTiled(Lifetime& life, bool small) {
  const std::string prefix = small ? "small_" : "representative_";
  return life.node.WithRequest(
      0, life.closure, "source F32 tiled convolution", [&]() -> en::Status {
        for (std::size_t i = 0; i < kNames.size(); ++i) {
          if (auto copied = life.node.Copy(0, life.closure,
                                           life.weights.resource_address(life.resource_ids[i]),
                                           life.host + life.host_offsets[i], kWeightBytes[i], true,
                                           "completed source F64 sample weights");
              !copied)
            return copied;
          if (!std::ranges::all_of(WeightValues(life, kNames[i]),
                                   [](float value) { return std::isfinite(value); }))
            return Error("nonfinite source F32 weights");
        }
        for (std::size_t arm = 0; arm < life.cases.size(); ++arm) {
          auto& item = life.cases[arm];
          const auto label = prefix + (arm == 0 ? "direct_" : "tiled_");
          if (auto initialized = TiledCanaries(life, true); !initialized) return initialized;
          auto coverage = life.node.Call(
              [&]() -> en::Status {
                en::Coverage result;
                en::CheckCoverage(life.node, 0, item.program->tensors(), "tiled source convolution",
                                  result);
                return result.violations ? Error(result.first_violation) : en::Status{};
              },
              "tiled catalog coverage");
          if (!coverage) return coverage;
          ::testing::Test::RecordProperty(label + "input_sha",
                                          Hash(std::as_bytes(std::span(item.input))));
          ::testing::Test::RecordProperty(label + "plan",
                                          llmp::base::ToHex(item.program->bound()->identity()));
          std::string implementations;
          for (std::size_t op = 0; op < item.program->bound()->size(); ++op)
            implementations += std::string(item.program->bound()->at(op).name) + ",";
          ::testing::Test::RecordProperty(label + "implementations", implementations);
          auto samples = ConvSamples(life, item, arm == 1, small);
          double maximum_error = 0, maximum_fraction = 0;
          std::string changed_hash;
          for (std::size_t repeat = 0; repeat < 9; ++repeat) {
            if (repeat == 7 || repeat == 8) {
              item.input[0] += repeat == 7 ? .5F : -.5F;
              samples = ConvSamples(life, item, arm == 1, small);
            }
            std::memset(life.host + life.output_offset - 256, 0x5a, 256);
            const auto& output = item.program->outputs()[0];
            std::memset(life.host + life.output_offset + output.bytes, 0x5a, 256);
            auto staged = life.step_runs.Stage(
                item.program->inputs(),
                [&](std::size_t slot, std::span<std::byte> out) -> en::Status {
                  if (slot != 0 || out.size() != item.input.size() * 4)
                    return Error("tiled input port");
                  std::memcpy(out.data(), item.input.data(), out.size());
                  return {};
                },
                0);
            if (!staged) return Error(staged.error());
            const std::array<en::RunCopy, 1> outputs{
                {{en::support::Address(life.host + life.output_offset), output.device_address,
                  output.bytes}}};
            en::StepQueued queued;
            const auto begin = std::chrono::steady_clock::now();
            auto ran = life.node.Job(
                life.closure,
                [&](llmp::providers::NativeStream native) {
                  queued = life.step_runs.Queue(item.runs, *staged, {}, *item.program, outputs,
                                                item.runs.CaptureDue(true), life.stats, native);
                  if (queued.result) return sc::JobResult::kQueued;
                  return queued.before || queued.result.error().error == kn::StepError::kUnknown
                             ? sc::JobResult::kUnknown
                             : sc::JobResult::kNotStarted;
                },
                "complete tiled/direct convolution and ports", 0);
            item.elapsed_ns[repeat] =
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - begin)
                                               .count());
            if (!ran) return ran;
            if (!queued.result) return Error(queued.result.error().detail);
            const auto expected = repeat == 0   ? en::RunPath::kEager
                                  : repeat == 1 ? en::RunPath::kCaptured
                                                : en::RunPath::kReplayed;
            if (queued.path != expected) return Error("required tiled capture/replay path");
            en::Count(life.stats, queued.path);
            const auto actual =
                std::span(reinterpret_cast<const float*>(life.host + life.output_offset),
                          static_cast<std::size_t>(output.bytes / 4));
            if (!std::ranges::all_of(actual, [](float v) { return std::isfinite(v); }))
              return Error("nonfinite complete tiled output");
            for (const auto& sample : samples) {
              const double error =
                  std::abs(static_cast<double>(actual[sample.index]) - sample.value);
              if (!std::isfinite(sample.value) || !std::isfinite(sample.bound) ||
                  error > sample.bound)
                return Error("independent F64 convolution enclosure");
              maximum_error = std::max(maximum_error, error);
              if (sample.bound > 0)
                maximum_fraction = std::max(maximum_fraction, error / sample.bound);
            }
            const auto digest = Hash(std::as_bytes(actual));
            if (repeat == 0)
              item.eager_hash = digest;
            else if (repeat == 7) {
              if (digest == item.eager_hash)
                return Error("changed staged input did not affect convolution");
              changed_hash = digest;
            } else if (digest != item.eager_hash)
              return Error("own-arm replay/restoration differs");
            for (auto* guard : {life.host + life.output_offset - 256,
                                life.host + life.output_offset + output.bytes}) {
              if (!std::ranges::all_of(std::span(guard, 256),
                                       [](std::byte b) { return b == std::byte{0x5a}; }))
                return Error("pinned output copy canary changed");
            }
            if (auto intact = TiledCanaries(life, false); !intact) return intact;
          }
          ::testing::Test::RecordProperty(label + "sha", item.eager_hash);
          ::testing::Test::RecordProperty(label + "changed_sha", changed_hash);
          ::testing::Test::RecordProperty(label + "samples", std::to_string(samples.size()));
          ::testing::Test::RecordProperty(label + "max_error",
                                          std::format("{:.17g}", maximum_error));
          ::testing::Test::RecordProperty(label + "max_enclosure_fraction",
                                          std::format("{:.17g}", maximum_fraction));
          ::testing::Test::RecordProperty(label + "activation",
                                          std::to_string(item.program->scratch_bytes()));
          ::testing::Test::RecordProperty(label + "pool",
                                          std::to_string(item.program->pool_bytes()));
          ::testing::Test::RecordProperty(label + "launches",
                                          std::to_string(item.program->launches()));
          ::testing::Test::RecordProperty(label + "program_host",
                                          std::to_string(item.program->host_bytes()));
          for (std::size_t repeat = 0; repeat < 9; ++repeat)
            ::testing::Test::RecordProperty(label + "step" + std::to_string(repeat) + "_ns",
                                            std::to_string(item.elapsed_ns[repeat]));
        }
        return {};
      });
}
void TiledSourceCase(bool small) {
  const auto path = llmp::test_support::QwenImagePublished(kArtifact);
  std::unique_ptr<Lifetime, Retire> life(new Lifetime);
  auto setup = SetupFixture(*life, path, false, small ? 7U : 128U, small ? 5U : 128U);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto ran = ExecuteTiled(*life, small);
  ASSERT_TRUE(ran) << Failed(ran).value_or("");
  EXPECT_EQ(life->stats.eager, 2U);
  EXPECT_EQ(life->stats.captured, 2U);
  EXPECT_EQ(life->stats.replayed, 14U);
  EXPECT_EQ(life->stats.refused, 0U);
  const auto prefix = small ? "small_" : "representative_";
  ::testing::Test::RecordProperty(std::string(prefix) + "eager", std::to_string(life->stats.eager));
  ::testing::Test::RecordProperty(std::string(prefix) + "captured",
                                  std::to_string(life->stats.captured));
  ::testing::Test::RecordProperty(std::string(prefix) + "replayed",
                                  std::to_string(life->stats.replayed));
  ::testing::Test::RecordProperty(std::string(prefix) + "capture_refused",
                                  std::to_string(life->stats.refused));
  ::testing::Test::RecordProperty(std::string(prefix) + "completed_canary_checks", "108");
  ::testing::Test::RecordProperty(std::string(prefix) + "finite_values",
                                  std::to_string((small ? 7ULL * 5 : 128ULL * 128) * 1152 * 18));
  ::testing::Test::RecordProperty(std::string(prefix) + "selected_pages",
                                  std::to_string(life->selected_pages));
  ::testing::Test::RecordProperty(std::string(prefix) + "source_bytes",
                                  std::to_string(kTotalWeightBytes));
  auto retired = life->Retire();
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
}
TEST(ImageSpatialGpuTest, TiledSourceThreeStripesPreserveHaloTailAndCompletedReplay) {
  TiledSourceCase(true);
  if (::testing::Test::HasFailure()) return;  // No new owner after unproven retirement.
  TiledSourceCase(false);
}
}  // namespace
