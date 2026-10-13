// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/component_image_build.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <map>
#include <utility>

#include "base/bytes.h"
#include "engine/support.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "providers/device_runtime.h"
#include "spec/image_spatial.h"

namespace llmp::runtime {
namespace {
namespace en = engine;
namespace co = components;
namespace md = model;
namespace kg = kernels::ggml;
using en::support::Error;
constexpr std::uint32_t kSize = 1024, kGrid = 64, kRows = kGrid * kGrid, kSteps = 2;
// Source descriptor/spec/input owner inventory, before any source-profile
// vectors or copied names exist: <=822 roles at4KiB, <=1435 components at8KiB,
// source BF16 noise, stable F32 copy, complete RGBA8 output and fixed
// schedule/control/registry objects (8MiB, separate from compiler inventory).
constexpr std::uint64_t kSourceOwnerBytes =
    822ULL * 4096 + (36ULL * 17 + 2 + 32ULL * 17 + 19 + 258) * 8192 + 8ULL * 1024 * 1024;
struct Paid {
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  std::uint64_t bytes = 0;
  ~Paid() {
    if (allowance) allowance->Release(bytes);
  }
  bool Add(std::uint64_t n) {
    if (n > UINT64_MAX - bytes || !allowance->Charge(n)) return false;
    bytes += n;
    return true;
  }
};
struct ImageGenerationOwner {
  Paid paid;
  std::shared_ptr<ImageGenerationSource> source;
  std::vector<std::int32_t> prompt;
  std::vector<std::uint16_t> noise;
  std::uint32_t drop = 0;
  std::array<std::vector<co::BoundTensor>, 3> tables;
  std::array<std::map<ir::WeightRef, std::uint32_t>, 3> resources;
  std::array<en::PagedWeights*, 3> weights{};
  std::array<std::shared_ptr<const co::SpatialGraph>, 3> graphs;
  std::vector<float> sigmas, timesteps;
  std::vector<std::byte> pixels;
  std::unique_ptr<kg::CublasHandle> cublas;
  std::unique_ptr<kg::GgmlBackend> backend;
};

// Same lower construction categories as the complete-flow proof, derived from
// this mapped stage's declared primitive/role/port inventory before graph emit.
// This includes two simultaneous program descriptors; retained capture nodes
// are separately measured/charged by EngineImage's normal PlanAccount.
std::expected<std::uint64_t, std::string> CompilerBound(const spec::SpatialComposition& mapping,
                                                        std::size_t roles, std::size_t ports) {
  std::uint64_t primitives = 0, components = 0;
  for (const auto& layer : mapping.spec.stages[0].layers)
    for (const auto& component : layer.components) {
      const auto* def = co::Find(component.kind, component.version);
      if (!def || !def->spatial) return Error("image compiler inventory has no spatial component");
      primitives += def->spatial->max_nodes;
      ++components;
    }
  // These inventories are already bounded by spec/value-ID representation.
  const auto nodes = primitives + 1 + ports + 2 * components;
  const auto tensors = 2 * nodes + roles;
  std::uint64_t result = 0;
  for (const auto term :
       {12 * (nodes * 480 + tensors * 192), 4096 * (nodes + tensors),
        (tensors + roles + 31 * nodes + ports) * (512 + 1024 + 32), 2 * (nodes + tensors) * 4096})
    if (__builtin_add_overflow(result, term, &result))
      return Error("image compiler bound overflow");
  return result;
}
std::expected<std::vector<co::BoundTensor>, std::string> Table(
    std::span<const md::QwenImageTensor> tensors) {
  std::vector<co::BoundTensor> out;
  for (const auto& tensor : tensors) {
    if (tensor.shape.empty() || tensor.shape.size() > 4) return Error("image source rank differs");
    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    const auto encoding = tensor.dtype == "BF16" ? format::Encoding::kBF16 : format::Encoding::kF32;
    if (tensor.dtype != "BF16" && tensor.dtype != "F32")
      return Error("image source encoding differs");
    auto format = ir::Dense(encoding, shape);
    if (!format::PartsOf(format)) return Error("image source geometry overflows");
    out.push_back({tensor.name, std::move(format)});
  }
  return out;
}
Status Fill(const ImageGenerationOwner& held, const ImageInvocation& invocation, std::size_t slot,
            std::span<std::byte> bytes) {
  const auto rows = static_cast<std::uint32_t>(held.prompt.size());
  const auto text = rows - held.drop;
  const auto& profile = md::QwenImage21();
  const auto put = [&]<typename T>(std::size_t at, T value) {
    std::memcpy(bytes.data() + at * sizeof(T), &value, sizeof(T));
  };
  const auto exact = [&](std::uint64_t count) { return bytes.size() == count * 4; };
  if (invocation.stage == 0) {
    if (slot < 2 && exact(rows)) {
      for (std::uint32_t row = 0; row < rows; ++row)
        put(row, slot == 0 ? held.prompt[row] : static_cast<std::int32_t>(row));
      return {};
    }
    if (slot == 2 && exact(std::uint64_t{rows} * rows)) {
      for (std::uint32_t q = 0; q < rows; ++q)
        for (std::uint32_t k = 0; k < rows; ++k)
          put(k + std::size_t{rows} * q, k <= q ? 0.F : -INFINITY);
      return {};
    }
  } else if (invocation.stage == 1 && invocation.iteration < kSteps) {
    const std::uint32_t joint = text + kRows;
    if (slot == 2 && exact(2 * profile.denoiser.timestep_dim)) {
      const auto sinusoid =
          md::QwenImageTimestepSinusoid(profile.denoiser, held.timesteps[invocation.iteration]);
      for (std::uint32_t c = 0; c < profile.denoiser.timestep_dim; ++c)
        for (std::uint32_t row = 0; row < 2; ++row)
          put(2 * c + row, md::FromBf16(sinusoid[row * profile.denoiser.timestep_dim + c]));
      return {};
    }
    if (slot == 3 && exact(3ULL * joint)) {
      for (std::uint32_t row = 0; row < joint; ++row)
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
          const auto image = row - std::min(row, text);
          const auto pos = row < text  ? static_cast<std::int32_t>(row)
                           : axis == 0 ? static_cast<std::int32_t>(text)
                           : axis == 1 ? static_cast<std::int32_t>(image / kGrid) - 32
                                       : static_cast<std::int32_t>(image % kGrid) - 32;
          put(row + std::size_t{joint} * axis, pos);
        }
      return {};
    }
    if (slot == 4 && exact(std::uint64_t{joint} * joint)) {
      for (std::uint32_t q = 0; q < joint; ++q)
        for (std::uint32_t k = 0; k < joint; ++k)
          put(k + std::size_t{joint} * q, q < text && k > q ? -INFINITY : 0.F);
      return {};
    }
    if (slot == 5 && exact(1)) {
      put(0, held.sigmas[invocation.iteration + 1] - held.sigmas[invocation.iteration]);
      return {};
    }
  } else if (invocation.stage == 2 && slot == 1 && exact(2 * profile.vae.z_dim)) {
    for (std::size_t c = 0; c < profile.vae.z_dim; ++c) {
      put(2 * c, profile.vae.latents_std[c]);
      put(2 * c + 1, profile.vae.latents_mean[c]);
    }
    return {};
  }
  return Error("image input does not match its immutable typed stage recipe");
}
}  // namespace

EngineImageBuild ComponentImageGenerationBuild(
    en::PagedNode& node, int owner, std::shared_ptr<ImageGenerationSource> source,
    std::shared_ptr<en::ExecutionMetadataAllowance> construction,
    std::shared_ptr<const planner::OwnedDeviceFacts> tuning) {
  return [&node, owner, source = std::move(source), construction = std::move(construction),
          tuning = std::move(tuning)](
             const ModelSettings&) -> std::expected<EngineImageParts, std::string> {
    const auto& p = md::QwenImage21();
    if (!source || !source->composition || !source->keep || !construction || owner < 0 ||
        source->composition->architecture() != p.pipeline_architecture ||
        source->prompt.size() != 39 || source->drop != 14 || source->noise.size() != kRows * 64ULL)
      return Error("image diagnostic requires the complete paid source/prompt/noise envelope");
    for (auto id : source->prompt)
      if (id < 0 || static_cast<std::uint32_t>(id) >= p.text.vocab)
        return Error("image text input ID lies outside the actual source vocabulary");
    for (const auto& weights : source->weights)
      if (!weights || !weights->opened()) return Error("image source artifact is not open");
    if (!construction->Charge(kSourceOwnerBytes))
      return Error("image source mapping is not prepaid");
    auto held = std::make_shared<ImageGenerationOwner>();
    held->paid.allowance = construction;
    held->paid.bytes = kSourceOwnerBytes;
    held->source = source;
    held->prompt = std::move(source->prompt);
    held->noise = std::move(source->noise);
    held->drop = source->drop;
    const std::array source_tensors{md::TextEncoderTensors(p.text), md::DenoiserTensors(p.denoiser),
                                    md::VaeDecoderTensors(p.vae)};
    const std::array<std::string_view, 3> role{"text_encoder", "transformer", "vae"};
    const std::array<std::string_view, 3> architecture{p.text_architecture, p.denoiser_architecture,
                                                       p.vae_architecture};
    std::array<std::vector<std::uint32_t>, 3> resource_ids;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto* component = source->composition->Find(role[i]);
      const auto& artifact = source->weights[i]->artifact();
      if (!component || component->artifact != artifact.id() ||
          component->architecture != architecture[i])
        return Error("image composition/source artifact identity differs");
      auto bindings = md::BindQwenImageComponent(source_tensors[i], architecture[i], artifact);
      if (!bindings) return Error(bindings.error());
      resource_ids[i] = std::move(*bindings);
      auto tensors = Table(source_tensors[i]);
      if (!tensors) return Error(tensors.error());
      held->tables[i] = std::move(*tensors);
      held->weights[i] = source->weights[i].get();
    }
    const std::uint32_t text_rows = static_cast<std::uint32_t>(held->prompt.size()) - held->drop;
    std::array mappings{spec::ImageTextSpec(p.text, 39, held->drop, source_tensors[0]),
                        spec::ImageFlowSpec(p.denoiser, text_rows, kRows, source_tensors[1]),
                        spec::ImageDecodeSpec(p.vae, kGrid, source_tensors[2])};
    const std::array<co::SpatialShape, 3> primary{
        co::SpatialShape{1, 39, 1, 1}, {1, text_rows, 4096, 1}, {1, kRows, 64, 1}};
    const auto joint = text_rows + kRows;
    const std::array<std::vector<co::SpatialInput>, 3> inputs{
        std::vector{co::SpatialInput{0, 0, {39, 1, 1, 1}, format::Encoding::kI32},
                    co::SpatialInput{0, 1, {39, 39, 1, 1}}},
        std::vector{
            co::SpatialInput{0, 0, {1, kRows, 64, 1}}, co::SpatialInput{0, 1, {1, 2, 256, 1}},
            co::SpatialInput{0, 2, {joint, 3, 1, 1}, format::Encoding::kI32},
            co::SpatialInput{0, 3, {joint, joint, 1, 1}}, co::SpatialInput{0, 4, {1, 1, 1, 1}}},
        std::vector{co::SpatialInput{0, 0, {1, 2, 64, 1}}}};
    for (std::size_t i = 0; i < 3; ++i) {
      if (!mappings[i]) return Error(mappings[i].error());
      auto compiler = CompilerBound(*mappings[i], source_tensors[i].size(), inputs[i].size() + 1);
      if (!compiler || !held->paid.Add(*compiler))
        return Error("image complete compiler owners are not prepaid");
      const std::array<std::span<const co::BoundTensor>, 1> tables{held->tables[i]};
      const std::array owners{co::SpatialOwner{0, primary[i]}};
      auto graph = co::BuildSpatialStep(
          mappings[i]->spec, tables, owners,
          {[construction](std::uint64_t bytes) { return construction->Charge(bytes); },
           [construction](std::uint64_t bytes) { construction->Release(bytes); }},
          {.links = mappings[i]->links,
           .reference_buffers = false,
           .input_encoding = i == 0 ? format::Encoding::kI32 : format::Encoding::kF32,
           .exports = mappings[i]->exports,
           .inputs = inputs[i]});
      if (!graph) return Error(std::format("image stage {}: {}", i, graph.error()));
      for (const auto& weight : graph->weights()) {
        if (weight.parts.size() != 1 || weight.parts[0].second != format::PartRole::kData)
          return Error("image source part contract changed");
        auto found =
            std::ranges::find(source_tensors[i], weight.parts[0].first, &md::QwenImageTensor::name);
        if (found == source_tensors[i].end())
          return Error("image graph names an unbound source role");
        const auto index = static_cast<std::size_t>(found - source_tensors[i].begin());
        held->resources[i].emplace(weight.ref, resource_ids[i][index]);
      }
      held->graphs[i] = std::make_shared<co::SpatialGraph>(std::move(*graph));
    }
    auto schedule = md::QwenImageSchedulerSigmas(p.scheduler, kSteps, kRows);
    if (!schedule) return Error(schedule.error());
    held->sigmas = std::move(schedule->sigmas);
    held->timesteps = std::move(schedule->timesteps);
    auto backend = kg::GgmlBackend::Create(0, node.execution(),
                                           node.stream(static_cast<std::uint32_t>(owner)), nullptr);
    if (!backend) return Error(backend.error());
    held->backend = std::move(*backend);
    const auto facts = providers::QueryDeviceFacts(0);
    if (!facts) return Error("image device facts unavailable");
    const auto blas_bytes =
        kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
    const auto tuples = tuning ? std::span<const execution::TunedShape>(tuning->facts().tuned)
                               : std::span<const execution::TunedShape>{};
    auto declared = held->backend->Declare(
        tuples, blas_bytes,
        [construction](std::uint64_t bytes) { return construction->Charge(bytes); },
        [construction](std::uint64_t bytes) { construction->Release(bytes); });
    if (!declared) return Error(declared.error());
    EngineImageParts out;
    out.keep = held;
    out.architecture = p.pipeline_architecture;
    out.backend = std::make_shared<kernels::StepBackend>(held->backend->Backend());
    base::Sha256 identity;
    identity.Update(source->composition->id());
    for (const auto& graph : held->graphs) identity.Update(base::ToHex(graph->graph().Digest()));
    identity.Update(std::as_bytes(std::span(held->prompt)));
    identity.Update(std::as_bytes(std::span(held->noise)));
    out.identity = identity.Finish();
    out.ports.push_back(
        {ir::Dense(format::Encoding::kF32, ir::ShapeOf({1, text_rows, 4096, 1})), 0, {}});
    auto& latent = out.ports.emplace_back(
        ImagePort{ir::Dense(format::Encoding::kF32, ir::ShapeOf({1, kRows, 64, 1})), 0, {}});
    latent.initial.resize(kRows * 64ULL * sizeof(float));
    for (std::size_t row = 0; row < kRows; ++row)
      for (std::size_t c = 0; c < 64; ++c) {
        const auto value = md::FromBf16(held->noise[row * 64 + c]);
        if (!std::isfinite(value)) return Error("image source noise is not finite");
        std::memcpy(latent.initial.data() + (row + kRows * c) * 4, &value, 4);
      }
    for (std::size_t i = 0; i < 3; ++i) {
      const auto count = source->weights[i]->artifact().layout().groups.size();
      out.weights.push_back({.weights = std::move(source->weights[i]),
                             .places = std::vector(count, en::GroupPlace::kDevice),
                             .slabs = {}});
      ImageStage stage;
      stage.graph = held->graphs[i];
      stage.weights = static_cast<std::uint16_t>(i);
      stage.addresses.weight = [held, i](const ir::WeightRef& ref, format::PartRole part) {
        const auto at = held->resources[i].find(ref);
        return part == format::PartRole::kData && at != held->resources[i].end()
                   ? held->weights[i]->resource_address(at->second)
                   : std::uint64_t{0};
      };
      stage.addresses.weight_alignment = [](const ir::WeightRef&, format::PartRole) { return 256; };
      if (i == 0) {
        stage.inputs = {{}, {}, {}};
        stage.outputs = {0};
      }
      if (i == 1) {
        stage.observation = 1;
        stage.inputs = {0, 1, {}, {}, {}, {}};
        stage.outputs = {{}, 1};
      }
      if (i == 2) {
        stage.inputs = {1, {}};
        stage.outputs = {{}};
      }
      out.stages.push_back(std::move(stage));
    }
    out.sequence = {{0, 0}, {1, 0}, {1, 1}, {2, 0}};
    out.first_output = 1;
    out.fill = [held](const ImageInvocation& invocation, std::size_t slot,
                      std::span<std::byte> bytes) { return Fill(*held, invocation, slot, bytes); };
    // Output formatting owns only four channels of interleaved pixel bytes,
    // paid in the source owner; no BF16 intermediate cast is imposed.
    if (!held->paid.Add(kSize * std::uint64_t{kSize} * 4))
      return Error("image pixel output is not prepaid");
    held->pixels.resize(kSize * std::size_t{kSize} * 4);
    out.finish =
        [held](std::span<const std::byte> bytes) -> std::expected<std::string, std::string> {
      if (bytes.size() != held->pixels.size() * sizeof(float))
        return Error("image final output extent differs");
      constexpr std::size_t pixels = kSize * std::size_t{kSize};
      for (std::size_t c = 0; c < 4; ++c)
        for (std::size_t i = 0; i < pixels; ++i) {
          float value = 0;
          std::memcpy(&value, bytes.data() + (c * pixels + i) * 4, 4);
          if (!std::isfinite(value)) return Error("image decoded output is not finite");
          const float scaled = (std::clamp(value, -1.F, 1.F) * .5F + .5F) * 255.F;
          const auto lo = static_cast<unsigned>(std::floor(scaled));
          const auto rounded =
              lo + static_cast<unsigned>(scaled - static_cast<float>(lo) > .5F ||
                                         (scaled - static_cast<float>(lo) == .5F && lo % 2));
          held->pixels[i * 4 + c] = static_cast<std::byte>(rounded);
        }
      return base::ToHex(base::Sha256().Update(held->pixels).Finish());
    };
    out.kernel_memory.push_back({.name = "image shared cuBLAS", .bytes = blas_bytes});
    out.prepare = [&node, owner, held, blas_bytes](std::span<const std::uint64_t> bases) -> Status {
      if (bases.size() != 1) return Error("image cuBLAS workspace is absent");
      auto blas = kg::CublasHandle::Create(0, node.execution(),
                                           node.stream(static_cast<std::uint32_t>(owner)),
                                           {.base = bases[0], .size = base::Bytes(blas_bytes)});
      if (!blas) return Error(blas.error().detail);
      held->cublas = std::move(*blas);
      return held->backend->LendCublas(held->cublas.get());
    };
    out.launch = [&node, held] {
      return held->backend->Launch(
          {.base = node.pool().base, .size = base::Bytes(node.pool().bytes)});
    };
    out.retire = [held] {
      held->backend->LaunchReleased();
      held->cublas.reset();
    };
    return out;
  };
}
}  // namespace llmp::runtime
