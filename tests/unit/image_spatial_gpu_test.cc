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
constexpr std::uint64_t kDomain = 1024 * kMiB, kMetadata = 512 * kMiB;
constexpr std::uint64_t kCompilerAndOwner = 64 * kMiB, kProgram = 8 * kMiB;
constexpr std::uint64_t kActivation = 64 * kMiB, kPool = 2 * kMiB;
constexpr std::uint64_t kPinned = 32 * kMiB, kOutputOffset = 2 * kMiB;
constexpr std::uint64_t kWeightOffset = 26 * kMiB;
constexpr std::string_view kArtifact =
    "44c1a20a202ce1078588cbd4b0c594bbadf365a01f26091c715fd16d5b2eb34d";
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
static_assert(12 * (32 * 480ULL + 64 * 192ULL) + 4096 * (32 + 64ULL) +
                  (64 + 16 + 31 * 32ULL + 4) * (512 + 1024 + 32) + 2 * kProgram +
                  32 * en::kGraphNodeHostBytes + 32 * kMiB <
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
  std::map<std::pair<ir::WeightRef, fm::PartRole>, std::uint32_t> bindings;
  std::unique_ptr<kn::StepProgram> program;
  en::StepPlanRuns runs;
  co::SpatialShape shape;
  std::vector<float> input;
  std::vector<double> reference;
  double error_enclosure = 0;
  std::string eager_hash;
  std::array<std::uint64_t, 3> elapsed_ns{};
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
  bool opened = false, retirement_attempted = false, retired = false;

  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return empty; }
  std::vector<ca::ExtentId> managed_extents() const override { return weights.extents(); }
  en::Status Release() override {
    for (auto& item : cases) {
      item.runs.DropGraph();
      item.program.reset();
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
  result.activations = measuring ? 0 : life.node.activations().base;
  result.activation_bytes = kActivation;
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
      program.outputs()[0].bytes != (index == 0 ? 18874368ULL : 73728ULL) ||
      program.scratch_bytes() > kActivation || program.pool_bytes() > kPool ||
      program.host_bytes() > kProgram || program.launches() > 8 || !program.capturable() ||
      program.routed())
    return Error("spatial program ownership, complete ports or paid envelope differ");
  const std::array<std::string_view, 4> prefix{"ggml.conv2d.f32", "ggml.add", "ggml.conv2d.f32",
                                               "ggml.add"};
  const std::array<std::string_view, 3> up{"ggml.upsample.nearest", "ggml.conv2d.f32", "ggml.add"};
  const std::span<const std::string_view> expected = index == 0
                                                         ? std::span<const std::string_view>(prefix)
                                                         : std::span<const std::string_view>(up);
  if (!program.bound())
    return {};  // Measurement cannot bind; the real program is checked after
                // Bind.
  if (program.bound()->size() != expected.size()) return Error("physical spatial step count");
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (program.bound()->at(i).name != expected[i]) return Error("physical spatial selection");
  return {};
}

en::Status SetupFixture(Lifetime& life, const std::filesystem::path& path) {
  auto available = llmp::platform::AvailableMemoryBytes();
  if (!available || *available < (4ULL << 30U) + (6ULL << 30U)) return Error("physical headroom");
  if (auto r = life.node.Open(); !r) return r;
  life.opened = true;
  life.node.SetHostFloor(0);
  auto paid = en::ExecutionMetadataAllowance::Admit(life.node.catalog(), life.node.domain(),
                                                    kMetadata, *available, 6ULL << 30U);
  if (!paid) return Error(paid.error());
  life.paid = *paid;
  if (!life.paid->Charge(kCompilerAndOwner)) return Error("preconstruction payment");
  std::error_code ec;
  if (std::filesystem::file_size(path / "manifest.json", ec) != 991 || ec ||
      std::filesystem::file_size(path / "index.json", ec) != 99639 || ec)
    return Error("fixed metadata extent");
  if (auto r = life.weights.Open(path, {.expected_id = std::string(kArtifact)}); !r) return r;
  const auto& artifact = life.weights.artifact();
  if (artifact.resources().size() != 238 || artifact.groups().size() != 6 ||
      artifact.shards().size() != 1 || artifact.format_version() != 0)
    return Error("opened VAE inventory");
  const auto tensors = mo::VaeDecoderTensors(mo::QwenImage21().vae);
  auto bound = mo::BindQwenImageComponent(tensors, "AutoencoderKLQwenImage21", artifact);
  if (!bound) return Error(bound.error());
  auto source = sp::SourceOfV0(artifact);
  if (!source) return Error(source.error());
  if (source->tensors.size() != 238) return Error("fixed source tensor metadata");
  std::vector<en::GroupPlace> groups(6, en::GroupPlace::kNone);
  std::uint64_t offset = kWeightOffset;
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
  if (offset != kWeightOffset + kTotalWeightBytes || offset > kPinned)
    return Error("weight host layout");
  if (auto r = life.weights.Reserve(life.node, groups, {}); !r) return r;
  if (life.weights.bytes() != 555745280ULL || life.weights.host_bytes() != 0 ||
      life.weights.extents().size() != 265)
    return Error("reserved source page geometry");

  auto steps = mo::VaeDecoderPlan(mo::QwenImage21().vae, 64, 64);
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
  const std::array<std::span<const co::BoundTensor>, 1> tables{source->tensors};
  for (std::size_t i = 0; i < 2; ++i) {
    auto& item = life.cases[i];
    auto spec = sp::VaeSpatialSpec(spans[i], tensors);
    if (!spec) return Error(spec.error());
    item.spec = std::move(*spec);
    item.shape = i == 0 ? co::SpatialShape{64, 64, 64, 1} : co::SpatialShape{4, 4, 288, 1};
    const std::array owners{co::SpatialOwner{0, item.shape}};
    auto graph =
        co::BuildSpatialStep(item.spec, tables, owners,
                             {[paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
                              [paid = life.paid](std::uint64_t n) { paid->Release(n); }});
    if (!graph) return Error(graph.error());
    if (graph->graph().nodes().size() > 32 || graph->graph().tensor_count() > 64 ||
        graph->weights().size() != (i == 0 ? 4U : 2U))
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
    item.input.resize(item.shape.width * item.shape.height * item.shape.channels);
    for (std::size_t n = 0; n < item.input.size(); ++n)
      item.input[n] = static_cast<float>(static_cast<int>((n * 17 + 3) % 257) - 128) / 128.F;
  }
  auto backend = kg::GgmlBackend::Create(0, life.node.execution(), life.node.stream(0), nullptr);
  if (!backend) return Error(backend.error());
  life.backend = std::move(*backend);
  const auto facts = llmp::providers::QueryDeviceFacts(0);
  if (!facts) return Error("actual device facts");
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  if (workspace.value() > 64 * kMiB) return Error("handle workspace");
  if (auto r = life.backend->Declare(
          {}, workspace.value(), [paid = life.paid](std::uint64_t n) { return paid->Charge(n); },
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
    en::GraphBuildIO io{item.graph->graph(), item.graph->graph().inputs(),
                        item.graph->graph().outputs(), item.graph->weights()};
    const auto addresses = Addresses(life, item, true);
    auto measured = en::BuildComponentProgram(
        io, table, {}, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = table.written,
         .within = table.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    if (!measured) return Error(measured.error());
    if (auto r = CheckProgram(item, **measured, i); !r) return r;
  }
  if (auto r = life.node.MapWorkspace(kActivation, kPool); !r) return r;
  auto launch = life.backend->Launch({life.node.pool().base, Bytes(life.node.pool().bytes)});
  if (!launch) return Error(launch.error());
  life.launch = std::move(*launch);
  auto host = life.resources.Pinned(kPinned);
  if (!host) return Error(host.error());
  life.host = static_cast<std::byte*>(*host);
  life.step_runs.SetStaging(life.host, 2 * kMiB);
  life.step_runs.SetLaunch(life.launch.get());
  for (std::size_t i = 0; i < 2; ++i) {
    auto& item = life.cases[i];
    en::GraphBuildIO io{item.graph->graph(), item.graph->graph().inputs(),
                        item.graph->graph().outputs(), item.graph->weights()};
    const auto addresses = Addresses(life, item, false);
    auto program = en::BuildComponentProgram(
        io, table, {}, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = table.written,
         .within = table.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    if (!program) return Error(program.error());
    if (auto r = CheckProgram(item, **program, i); !r) return r;
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
      error = error * sum_abs + gamma * ((max_input + error) * sum_abs + max_bias) +
              n * 0x1p-126 * (1 + max_input + error + max_weight) +
              n * 0x1p-52 * (max_input * sum_abs + max_bias);
    }
    co::SpatialRefIO ref{instance, shape, item.reference, kernel, bias, {}};
    if (auto r = instance.def->spatial->reference(ref); !r) return Error(r.error());
    auto next = instance.def->spatial->infer(instance, shape);
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
      for (std::size_t repeat = 0; repeat < 3; ++repeat) {
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

TEST(ImageSpatialGpuTest, ActualF32VaePartsUseSharedTypedProgramsAndCompletedReplay) {
  const auto* path = std::getenv("LLMP_TEST_VAE_ARTIFACT");
  ASSERT_NE(path, nullptr);
  std::unique_ptr<Lifetime, Retire> life(new Lifetime);
  auto setup = SetupFixture(*life, path);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto ran = Execute(*life);
  ASSERT_TRUE(ran) << Failed(ran).value_or("");
  EXPECT_EQ(life->stats.eager, 2U);
  EXPECT_EQ(life->stats.captured, 2U);
  EXPECT_EQ(life->stats.replayed, 2U);
  EXPECT_EQ(life->stats.refused, 0U);
  RecordProperty("actual_source_bytes", std::to_string(kTotalWeightBytes));
  RecordProperty("selected_pages", std::to_string(life->selected_pages));
  RecordProperty("complete_output_values", "4737024");
  RecordProperty("eager", std::to_string(life->stats.eager));
  RecordProperty("captured", std::to_string(life->stats.captured));
  RecordProperty("replayed", std::to_string(life->stats.replayed));
  RecordProperty("capture_refused", std::to_string(life->stats.refused));
  for (std::size_t i = 0; i < kNames.size(); ++i) {
    const auto values = WeightValues(*life, kNames[i]);
    RecordProperty("source_part" + std::to_string(i) + "_sha", Hash(std::as_bytes(values)));
  }
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& item = life->cases[i];
    RecordProperty("span" + std::to_string(i) + "_plan",
                   llmp::base::ToHex(item.program->bound()->identity()));
    std::string implementations;
    for (std::size_t op = 0; op < item.program->bound()->size(); ++op)
      implementations += item.program->bound()->at(op).name + ",";
    ASSERT_LE(implementations.size(), 1024U);
    RecordProperty("span" + std::to_string(i) + "_implementations", implementations);
    std::string parameters;
    for (const auto n : item.graph->parameters()) parameters += std::to_string(n) + ",";
    ASSERT_LE(parameters.size(), 1024U);
    RecordProperty("span" + std::to_string(i) + "_parameters", parameters);
    RecordProperty("span" + std::to_string(i) + "_input_sha",
                   Hash(std::as_bytes(std::span(item.input))));
    RecordProperty("span" + std::to_string(i) + "_sha", life->cases[i].eager_hash);
    for (std::size_t j = 0; j < 3; ++j)
      RecordProperty("span" + std::to_string(i) + "_step" + std::to_string(j) + "_ns",
                     std::to_string(life->cases[i].elapsed_ns[j]));
  }
  auto retired = life->Retire();
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
}
}  // namespace
