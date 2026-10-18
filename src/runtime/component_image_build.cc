// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/component_image_build.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <utility>

#include "base/bytes.h"
#include "base/json.h"
#include "chat/chat.h"
#include "engine/support.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "planner/policy.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "providers/device_runtime.h"
#include "runtime/component_admission.h"
#include "runtime/memory_guard.h"
#include "runtime/serving_support.h"
#include "spec/image_spatial.h"
#include "tokenizer/hf.h"

namespace llmp::runtime {
namespace {
namespace en = engine;
namespace co = components;
namespace md = model;
namespace kg = kernels::ggml;
using en::support::Error;
// Source descriptor/spec/input owner inventory, before any source-profile
// vectors or copied names exist: the family's tensors (roles) at 4 KiB, its
// components (17 a text layer and a flow block, the text encoder's 2 and
// the flow's 19 besides, the decoder's at most 258) at 8 KiB.
// Request-sized port, pixel and schedule vectors are separately paid below;
// incoming prompt/noise allocations retain their caller-owned payment.
std::uint64_t SourceMappingBytes(const md::ImageProfile& p) {
  const std::uint64_t roles =
      1 + std::uint64_t{p.text.layers} * static_cast<std::uint64_t>(md::TextTensor::kCount) +
      static_cast<std::uint64_t>(md::DenoiserGlobal::kCount) +
      std::uint64_t{p.denoiser.blocks} * static_cast<std::uint64_t>(md::BlockTensor::kCount) +
      md::VaeDecoderTensors(p.vae, {}).size();
  const std::uint64_t components =
      std::uint64_t{p.text.layers} * 17 + 2 + std::uint64_t{p.denoiser.blocks} * 17 + 19 + 258;
  return roles * 4096 + components * 8192;
}
using serving_support::ChargeExecutionMetadata;
struct Paid {
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  std::uint64_t bytes = 0;
  en::PagedNode* node = nullptr;
  std::uint64_t cap = 0;
  ~Paid() {
    if (allowance) allowance->Release(bytes);
  }
  bool Add(std::uint64_t n) {
    if (n > UINT64_MAX - bytes || !ChargeExecutionMetadata(allowance, n, node, cap)) return false;
    bytes += n;
    return true;
  }
};
struct RequestGeometry {
  std::uint32_t width = 0, height = 0, grid_width = 0, grid_height = 0;
  std::uint32_t rows = 0, prompt_rows = 0, text_rows = 0, steps = 0;
  std::uint64_t latent_elements = 0, pixels = 0, variable_bytes = 0;
};
std::expected<RequestGeometry, std::string> RequestOf(const ImageGenerationRequest& source,
                                                      const ModelSettings& settings,
                                                      const md::ImageProfile& profile) {
  RequestGeometry r;
  if ((source.width == 0) != (source.height == 0))
    return Error("image request requires both output dimensions or neither");
  r.width = source.width ? source.width : settings.image_size.value;
  r.height = source.height ? source.height : settings.image_size.value;
  r.steps = settings.image_steps.value;
  const auto factor = profile.vae.spatial_factor;
  if (!r.width || !r.height || !r.steps || !factor || r.width % factor || r.height % factor ||
      source.prompt.empty() || source.prompt.size() > INT32_MAX ||
      source.drop >= source.prompt.size())
    return Error("image request dimensions, scheduler count or prompt/drop are invalid");
  r.grid_width = r.width / factor;
  r.grid_height = r.height / factor;
  const auto rows = std::uint64_t{r.grid_width} * r.grid_height;
  r.prompt_rows = static_cast<std::uint32_t>(source.prompt.size());
  r.text_rows = r.prompt_rows - source.drop;
  if (rows > UINT32_MAX - r.text_rows || r.grid_width > INT32_MAX || r.grid_height > INT32_MAX)
    return Error("image request exceeds row/position representation");
  r.rows = static_cast<std::uint32_t>(rows);
  r.latent_elements = rows * profile.vae.z_dim;
  r.pixels = std::uint64_t{r.width} * r.height;
  if (profile.denoiser.in_channels != profile.vae.z_dim ||
      source.noise.size() != r.latent_elements ||
      r.pixels > std::uint64_t{PTRDIFF_MAX} / (4 * sizeof(float)))
    return Error("image noise or output extent differs from request geometry");
  // Initial F32 port, output RGBA8, sequence (text + steps*flow + decoder),
  // and scheduler's t/sigmas/timesteps. Double capacities cover each vector;
  // scheduler's push_back can retain twice its step count while t is live.
  const std::uint64_t schedule = (4 * std::uint64_t{r.steps} + 2) * sizeof(float);
  r.variable_bytes = 2 * r.latent_elements * sizeof(float);
  for (const auto term :
       {2 * r.pixels * 4, schedule, 2 * (std::uint64_t{r.steps} + 2) * sizeof(ImageInvocation)})
    if (__builtin_add_overflow(r.variable_bytes, term, &r.variable_bytes))
      return Error("image request construction bound overflows");
  return r;
}
struct ImageRequestOwner {
  Paid paid;
  std::shared_ptr<const ImageFamily> family;
  ImageGenerationRequest source;
  RequestGeometry request;
  std::vector<float> sigmas, timesteps;
  std::vector<std::byte> pixels;
};
constexpr std::uint64_t kRequestOwnerBytes =
    2 * sizeof(ImageRequestOwner) + 2 * sizeof(ImageRequest) + 4 * sizeof(ImagePort) + 4096;
}  // namespace
struct ImageGenerationModel {
  Paid paid;
  std::shared_ptr<ImageGenerationSource> source;
  std::shared_ptr<const ImageFamily> family;
  planner::Policy policy;
  std::array<std::vector<md::ImageTensor>, 3> source_tensors;
  std::array<std::vector<co::BoundTensor>, 3> tables;
  std::array<std::vector<std::uint32_t>, 3> resource_ids;
  std::array<en::PagedWeights*, 3> weights{};
  std::array<std::uint64_t, 3> compiler_limits{};
  kg::SelectedWorkspace selected_state{};
  std::unique_ptr<kg::CublasHandle> cublas;
  std::unique_ptr<kg::GgmlBackend> backend;
};
namespace {
struct ImageGeometryOwner {
  Paid paid;
  std::shared_ptr<ImageGenerationModel> model;
  std::array<std::map<ir::WeightRef, std::uint32_t>, 3> resources;
  std::array<std::shared_ptr<const co::SpatialGraph>, 3> graphs;
};

// Two complete owner/return representations, doubled vector capacities for
// three stages/eleven input ports, and fixed shared/control blocks.
// Kernel declaration storage is independently charged by Declare below.
std::uint64_t SourceOwnerBytes(const md::ImageProfile& p) {
  return SourceMappingBytes(p) + 2 * sizeof(ImageGenerationModel) + 2 * sizeof(EngineImageParts) +
         6 * sizeof(ImageStage) + 22 * sizeof(std::optional<std::uint32_t>) + 4096;
}

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
    std::span<const md::ImageTensor> tensors) {
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
Status Fill(const ImageRequestOwner& held, const ImageInvocation& invocation, std::size_t slot,
            std::span<std::byte> bytes) {
  const auto rows = static_cast<std::uint32_t>(held.source.prompt.size());
  const auto text = rows - held.source.drop;
  const auto& profile = held.family->profile;
  const auto put = [&]<typename T>(std::size_t at, T value) {
    std::memcpy(bytes.data() + at * sizeof(T), &value, sizeof(T));
  };
  const auto exact = [&](std::uint64_t count) {
    return count <= SIZE_MAX / 4 && bytes.size() == count * 4;
  };
  if (invocation.stage == 0) {
    if (slot < 2 && exact(rows)) {
      for (std::uint32_t row = 0; row < rows; ++row)
        put(row, slot == 0 ? held.source.prompt[row] : static_cast<std::int32_t>(row));
      return {};
    }
    if (slot == 2 && exact(std::uint64_t{rows} * rows)) {
      for (std::uint32_t q = 0; q < rows; ++q)
        for (std::uint32_t k = 0; k < rows; ++k)
          put(k + std::size_t{rows} * q, k <= q ? 0.F : -INFINITY);
      return {};
    }
  } else if (invocation.stage == 1 && invocation.iteration < held.request.steps) {
    const std::uint32_t joint = text + held.request.rows;
    if (slot == 2 && exact(2 * profile.denoiser.timestep_dim)) {
      const auto sinusoid =
          md::ImageTimestepSinusoid(profile.denoiser, held.timesteps[invocation.iteration]);
      for (std::uint32_t c = 0; c < profile.denoiser.timestep_dim; ++c)
        for (std::uint32_t row = 0; row < 2; ++row)
          put(2 * c + row, md::FromBf16(sinusoid[row * profile.denoiser.timestep_dim + c]));
      return {};
    }
    if (slot == 3 && exact(3ULL * joint)) {
      if (invocation.iteration > 0 && joint >= 2) {
        const auto pos0 =
            md::ImageDenoiserPosition(text, held.request.grid_height, held.request.grid_width, 0);
        const auto pos_last = md::ImageDenoiserPosition(text, held.request.grid_height,
                                                        held.request.grid_width, joint - 1);
        std::int32_t cur0 = -1, cur_last = -1;
        std::memcpy(&cur0, bytes.data(), sizeof(std::int32_t));
        std::memcpy(&cur_last, bytes.data() + (3ULL * joint - 1) * sizeof(std::int32_t),
                    sizeof(std::int32_t));
        if (cur0 == pos0[0] && cur_last == pos_last[2]) return {};
      }
      for (std::uint32_t row = 0; row < joint; ++row) {
        const auto positions =
            md::ImageDenoiserPosition(text, held.request.grid_height, held.request.grid_width, row);
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
          put(row + std::size_t{joint} * axis, positions[axis]);
        }
      }
      return {};
    }
    if (slot == 4 && exact(std::uint64_t{joint} * joint)) {
      if (invocation.iteration > 0 && joint >= 2 && text >= 1) {
        float v0 = 1.0f, v1 = 0.0f, vlast = 1.0f;
        std::memcpy(&v0, bytes.data(), sizeof(float));
        std::memcpy(&v1, bytes.data() + sizeof(float), sizeof(float));
        std::memcpy(&vlast, bytes.data() + (std::size_t{joint} * joint - 1) * sizeof(float),
                    sizeof(float));
        if (v0 == 0.0f && std::isinf(v1) && v1 < 0.0f && vlast == 0.0f) return {};
      }
      std::memset(bytes.data(), 0, bytes.size());
      const std::uint32_t masked_rows = std::min(text, joint);
      for (std::uint32_t q = 0; q < masked_rows; ++q) {
        for (std::uint32_t k = q + 1; k < joint; ++k) {
          put(k + std::size_t{joint} * q, -INFINITY);
        }
      }
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

std::expected<ImageRequest, std::string> ComponentImageGenerationRequest(
    const std::shared_ptr<const ImageFamily>& family, ImageGenerationRequest&& source,
    const ModelSettings& settings, std::shared_ptr<en::ExecutionMetadataAllowance> construction,
    en::PagedNode* construction_node, std::uint64_t construction_cap) {
  if (!family || !family->mapping) return Error("image request has no admitted family");
  const auto& p = family->profile;
  if (!source.keep || !construction) return Error("image request lacks its paid caller owner");
  auto request = RequestOf(source, settings, p);
  if (!request) return Error(request.error());
  const auto geometry = *request;
  for (auto id : source.prompt)
    if (id < 0 || static_cast<std::uint32_t>(id) >= p.text.vocab)
      return Error("image text input ID lies outside the actual source vocabulary");
  if (!ChargeExecutionMetadata(construction, kRequestOwnerBytes, construction_node,
                               construction_cap))
    return Error("image request owner is unpaid");
  auto held = std::make_shared<ImageRequestOwner>();
  held->paid.allowance = construction;
  held->paid.bytes = kRequestOwnerBytes;
  held->paid.node = construction_node;
  held->paid.cap = construction_cap;
  if (!held->paid.Add(geometry.variable_bytes)) return Error("image request vectors are unpaid");
  held->family = family;
  held->source = std::move(source);
  held->request = geometry;
  const auto image_rows = geometry.rows, text_rows = geometry.text_rows;
  const auto channels = p.vae.z_dim;
  auto schedule = md::ImageSchedulerSigmas(p.scheduler, geometry.steps, image_rows);
  if (!schedule) return Error(schedule.error());
  held->sigmas = std::move(schedule->sigmas);
  held->timesteps = std::move(schedule->timesteps);
  ImageRequest out;
  out.keep = held;
  base::Sha256 layout;
  layout.Update("llmp-image-request-layout-v1");
  layout.Update(family->mapping->architecture);
  const std::array shape{geometry.width, geometry.height, geometry.prompt_rows, held->source.drop};
  layout.Update(std::as_bytes(std::span(shape)));
  out.layout = layout.Finish();
  base::Sha256 identity;
  identity.Update(base::ToHex(out.layout));
  identity.Update(std::as_bytes(std::span(held->source.prompt)));
  identity.Update(std::as_bytes(std::span(held->source.noise)));
  identity.Update(std::as_bytes(std::span(held->sigmas)));
  identity.Update(std::as_bytes(std::span(held->timesteps)));
  out.identity = identity.Finish();
  out.ports.push_back(
      {ir::Dense(format::Encoding::kF32, ir::ShapeOf({1, text_rows, p.text.width, 1})), 0, {}});
  auto& latent = out.ports.emplace_back(ImagePort{
      ir::Dense(format::Encoding::kF32, ir::ShapeOf({1, image_rows, channels, 1})), 0, {}});
  latent.initial.resize(geometry.latent_elements * sizeof(float));
  for (std::size_t row = 0; row < image_rows; ++row)
    for (std::size_t c = 0; c < channels; ++c) {
      const auto value = md::FromBf16(held->source.noise[row * channels + c]);
      if (!std::isfinite(value)) return Error("image source noise is not finite");
      std::memcpy(latent.initial.data() + (row + std::size_t{image_rows} * c) * 4, &value, 4);
    }
  out.sequence.reserve(std::size_t{geometry.steps} + 2);
  out.sequence.push_back({0, 0});
  for (std::uint32_t iteration = 0; iteration < geometry.steps; ++iteration)
    out.sequence.push_back({1, iteration});
  out.sequence.push_back({2, 0});
  out.first_output = 1;
  out.fill = [held](const ImageInvocation& invocation, std::size_t slot,
                    std::span<std::byte> bytes) { return Fill(*held, invocation, slot, bytes); };
  // Interleaved pixel bytes are in the request-sized construction payment;
  // no BF16 intermediate cast is imposed.
  held->pixels.resize(geometry.pixels * 4);
  out.finish = [held](std::span<const std::byte> bytes) -> std::expected<std::string, std::string> {
    if (bytes.size() != held->pixels.size() * sizeof(float))
      return Error("image final output extent differs");
    const auto pixels = static_cast<std::size_t>(held->request.pixels);
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
    if (!held->source.output.empty()) {
      auto directory = platform::OpenPrivateDirectory(-1, held->source.output.c_str());
      if (!directory) return Error("image output directory: " + std::to_string(directory.error()));
      const artifact::FileDescriptor output_directory(*directory);
      const std::string_view rgba(reinterpret_cast<const char*>(held->pixels.data()),
                                  held->pixels.size());
      if (auto written = platform::ReplacePrivateFile(output_directory.get(), "image.rgba8", rgba);
          !written)
        return Error("image output write: " + std::to_string(written.error()));
    }
    return base::ToHex(base::Sha256().Update(held->pixels).Finish());
  };
  return out;
}

namespace {
// Whether the family's planning policy asks the graphs' builder to remove
// their layout round trips (planner/pattern.h option layout_sinking).
bool SinksLayouts(const planner::Policy& policy) {
  return !planner::Disabled(policy, "layout_sinking");
}
// The family's tensors, its mapping's names, in each stage's order.
std::array<std::vector<md::ImageTensor>, 3> SourceTensors(const ImageFamily& family) {
  const auto& p = family.profile;
  const auto& n = family.mapping->names;
  return {md::TextEncoderTensors(p.text, n.text), md::DenoiserTensors(p.denoiser, n.denoiser),
          md::VaeDecoderTensors(p.vae, n.vae)};
}
}  // namespace

std::expected<std::shared_ptr<const ImageFamily>, std::string> ReadImageFamily(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components) {
  auto admitted = ReadImageAdmission(composition, components);
  if (!admitted) return std::unexpected(admitted.error());
  auto family = std::make_shared<ImageFamily>();
  family->mapping = admitted->mapping;
  family->profile = std::move(admitted->profile);
  family->spec = std::move(admitted->family);
  return family;
}

std::expected<ImageGeometry, std::string> ComponentImageGenerationGeometry(
    const std::shared_ptr<ImageGenerationSource>& source, ImageGenerationRequest&& request_source,
    const ModelSettings& settings, std::shared_ptr<en::ExecutionMetadataAllowance> construction) {
  auto model = source ? source->prepared.lock() : nullptr;
  if (!model || !construction || !model->family)
    return Error("image geometry requires the retained model source");
  const auto& p = model->family->profile;
  auto checked = RequestOf(request_source, settings, p);
  if (!checked) return Error(checked.error());
  const auto geometry = *checked;
  const auto drop = request_source.drop;
  const auto image_rows = geometry.rows, prompt_rows = geometry.prompt_rows;
  const auto channels = p.vae.z_dim;
  const auto geometry_bytes = SourceMappingBytes(p) + 2 * sizeof(ImageGeometryOwner) +
                              2 * sizeof(ImageGeometry) + 6 * sizeof(ImageStage) + 4096;
  if (!ChargeExecutionMetadata(construction, geometry_bytes, source->construction_node,
                               source->construction_cap))
    return Error("image geometry mapping is not prepaid");
  auto held = std::make_shared<ImageGeometryOwner>();
  held->paid.allowance = construction;
  held->paid.bytes = geometry_bytes;
  held->paid.node = source->construction_node;
  held->paid.cap = source->construction_cap;
  held->model = model;
  auto prepared_request = ComponentImageGenerationRequest(
      model->family, std::move(request_source), settings, construction, source->construction_node,
      source->construction_cap);
  if (!prepared_request) return Error(prepared_request.error());
  const auto& source_tensors = model->source_tensors;
  const auto text_rows = geometry.text_rows;
  std::array mappings{
      spec::ImageTextSpec(p.text, prompt_rows, drop, source_tensors[0]),
      spec::ImageFlowSpec(p.denoiser, text_rows, image_rows, source_tensors[1]),
      spec::ImageDecodeSpec(p.vae, geometry.grid_height, geometry.grid_width, source_tensors[2])};
  const std::array<co::SpatialShape, 3> primary{co::SpatialShape{1, prompt_rows, 1, 1},
                                                {1, text_rows, p.text.width, 1},
                                                {1, image_rows, channels, 1}};
  const auto joint = text_rows + image_rows;
  const std::array<std::vector<co::SpatialInput>, 3> inputs{
      std::vector{co::SpatialInput{0, 0, {prompt_rows, 1, 1, 1}, format::Encoding::kI32},
                  co::SpatialInput{0, 1, {prompt_rows, prompt_rows, 1, 1}}},
      std::vector{co::SpatialInput{0, 0, {1, image_rows, channels, 1}},
                  co::SpatialInput{0, 1, {1, 2, p.denoiser.timestep_dim, 1}},
                  co::SpatialInput{0, 2, {joint, 3, 1, 1}, format::Encoding::kI32},
                  co::SpatialInput{0, 3, {joint, joint, 1, 1}},
                  co::SpatialInput{0, 4, {1, 1, 1, 1}}},
      std::vector{co::SpatialInput{0, 0, {1, 2, channels, 1}}}};
  for (std::size_t i = 0; i < 3; ++i) {
    if (!mappings[i]) return Error(mappings[i].error());
    auto compiler = CompilerBound(*mappings[i], source_tensors[i].size(), inputs[i].size() + 1);
    if (!compiler) return Error(compiler.error());
    // Compilation is serial on the model owner. Retained old/candidate graph
    // descriptors each have their own BuildSpatialStep payment; this common
    // two-program compiler scratch ceiling is held once at its high-water.
    if (*compiler > model->compiler_limits[i]) {
      if (!model->paid.Add(*compiler - model->compiler_limits[i]))
        return Error("image complete compiler owners are not prepaid");
      model->compiler_limits[i] = *compiler;
    }
    const std::array<std::span<const co::BoundTensor>, 1> tables{model->tables[i]};
    const std::array owners{co::SpatialOwner{0, primary[i]}};
    auto graph = co::BuildSpatialStep(
        mappings[i]->spec, tables, owners,
        {[construction, source](std::uint64_t bytes) {
           return ChargeExecutionMetadata(construction, bytes, source->construction_node,
                                          source->construction_cap);
         },
         [construction](std::uint64_t bytes) { construction->Release(bytes); }},
        {.links = mappings[i]->links,
         .reference_buffers = false,
         .input_encoding = i == 0 ? format::Encoding::kI32 : format::Encoding::kF32,
         .exports = mappings[i]->exports,
         .inputs = inputs[i],
         .sink_layouts = SinksLayouts(model->policy)});
    if (!graph) return Error(std::format("image stage {}: {}", i, graph.error()));
    for (const auto& weight : graph->weights()) {
      if (weight.parts.size() != 1 || weight.parts[0].second != format::PartRole::kData)
        return Error("image source part contract changed");
      auto found =
          std::ranges::find(source_tensors[i], weight.parts[0].first, &md::ImageTensor::name);
      if (found == source_tensors[i].end())
        return Error("image graph names an unbound source role");
      const auto index = static_cast<std::size_t>(found - source_tensors[i].begin());
      held->resources[i].emplace(weight.ref, model->resource_ids[i][index]);
    }
    held->graphs[i] = std::make_shared<co::SpatialGraph>(std::move(*graph));
  }
  ImageGeometry out;
  out.keep = held;
  base::Sha256 identity;
  identity.Update(source->composition->id());
  for (const auto& graph : held->graphs) identity.Update(base::ToHex(graph->graph().Digest()));
  out.identity = identity.Finish();
  out.request = std::move(*prepared_request);
  for (std::size_t i = 0; i < 3; ++i) {
    ImageStage stage;
    stage.graph = held->graphs[i];
    stage.weights = static_cast<std::uint16_t>(i);
    stage.addresses.weight = [held, i](const ir::WeightRef& ref, format::PartRole part) {
      const auto at = held->resources[i].find(ref);
      return part == format::PartRole::kData && at != held->resources[i].end()
                 ? held->model->weights[i]->resource_address(at->second)
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
      // The text rows (the encoder's, written once a generation) and the
      // joint rows' positions and mask (Fill) stay the same over a
      // generation's steps.
      stage.invariant = {0, 3, 4};
    }
    if (i == 2) {
      stage.inputs = {1, {}};
      stage.outputs = {{}};
    }
    out.stages.push_back(std::move(stage));
  }
  return out;
}

EngineImageBuild ComponentImageGenerationBuild(
    en::PagedNode& node, int owner, std::shared_ptr<ImageGenerationSource> source,
    std::shared_ptr<en::ExecutionMetadataAllowance> construction,
    std::shared_ptr<const planner::OwnedDeviceFacts> tuning) {
  return [&node, owner, source = std::move(source), construction = std::move(construction),
          tuning = std::move(tuning)](
             const ModelSettings& settings) -> std::expected<EngineImageParts, std::string> {
    if (!source || !source->composition || !source->keep || !construction || owner < 0)
      return Error("image generation requires the complete paid source envelope");
    for (const auto& weights : source->weights)
      if (!weights || !weights->opened()) return Error("image source artifact is not open");
    if (!source->family) {
      std::array<const artifact::Artifact*, 3> opened{};
      for (std::size_t i = 0; i < opened.size(); ++i) opened[i] = &source->weights[i]->artifact();
      auto family = ReadImageFamily(*source->composition, opened);
      if (!family) return Error(family.error());
      source->family = std::move(*family);
    }
    const auto& family = *source->family;
    const auto& p = family.profile;
    if (source->composition->architecture() != family.mapping->architecture ||
        family.mapping->components.size() != 3)
      return Error("image generation requires the complete paid source envelope");
    auto request = RequestOf(*source, settings, p);
    if (!request) return Error(request.error());
    for (auto id : source->prompt)
      if (id < 0 || static_cast<std::uint32_t>(id) >= p.text.vocab)
        return Error("image text input ID lies outside the actual source vocabulary");
    const auto owner_bytes = SourceOwnerBytes(p);
    if (!ChargeExecutionMetadata(construction, owner_bytes, source->construction_node,
                                 source->construction_cap))
      return Error("image source mapping is not prepaid");
    auto held = std::make_shared<ImageGenerationModel>();
    held->paid.allowance = construction;
    held->paid.bytes = owner_bytes;
    held->paid.node = source->construction_node;
    held->paid.cap = source->construction_cap;
    held->source = source;
    held->family = source->family;
    auto policy = planner::ParsePolicy(family.mapping->serving->serving_rule->text);
    if (!policy) return Error(std::format("the image planning policy: {}", policy.error()));
    held->policy = std::move(*policy);
    held->source_tensors = SourceTensors(family);
    const auto& source_tensors = held->source_tensors;

    for (std::size_t i = 0; i < 3; ++i) {
      const auto& rule = family.mapping->components[i];
      const auto* component = source->composition->Find(rule.role);
      const auto& artifact = source->weights[i]->artifact();
      if (!component || component->artifact != artifact.id() ||
          !std::ranges::contains(rule.architectures, component->architecture))
        return Error("image composition/source artifact identity differs");
      auto bindings = md::BindImageComponent(source_tensors[i], component->architecture, artifact);
      if (!bindings) return Error(bindings.error());
      held->resource_ids[i] = std::move(*bindings);
      auto tensors = Table(source_tensors[i]);
      if (!tensors) return Error(tensors.error());
      held->tables[i] = std::move(*tensors);
      held->weights[i] = source->weights[i].get();
    }
    auto backend = kg::GgmlBackend::Create(0, node.execution(),
                                           node.stream(static_cast<std::uint32_t>(owner)), nullptr);
    if (!backend) return Error(backend.error());
    held->backend = std::move(*backend);
    const auto facts = providers::QueryDeviceFacts(0);
    if (!facts) return Error("image device facts unavailable");
    const auto blas_bytes =
        kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
    // The denoiser's F32 attention, tuned by default on the GB10 at the
    // mapping's qualified joint rows, and this request's where its image
    // rows are the qualified ones (its heads' width).
    const auto joint = std::uint64_t{request->text_rows} + request->rows;
    std::vector<execution::TunedShape> default_tuned;
    if (!tuning && facts->architecture / 10 == 121) {
      std::vector<std::uint64_t> rows(family.mapping->attention_rows.begin(),
                                      family.mapping->attention_rows.end());
      // A request at the qualified image rows, whatever its prompt's.
      if (std::ranges::contains(family.mapping->attention_image_rows, request->rows) &&
          !std::ranges::contains(rows, joint))
        rows.push_back(joint);
      for (const auto r : rows)
        default_tuned.push_back({.implementation = kg::kAttentionF32,
                                 .rows = r,
                                 .k = p.denoiser.head_dim,
                                 .n = p.denoiser.head_dim,
                                 .output_encoding = format::Encoding::kF32});
    }
    const auto tuples = tuning ? std::span<const execution::TunedShape>(tuning->facts().tuned)
                               : std::span<const execution::TunedShape>(default_tuned);
    auto declared = held->backend->Declare(
        tuples, blas_bytes,
        [construction, source](std::uint64_t bytes) {
          return ChargeExecutionMetadata(construction, bytes, source->construction_node,
                                         source->construction_cap);
        },
        [construction](std::uint64_t bytes) { construction->Release(bytes); });
    if (!declared) return Error(declared.error());
    source->prepared = held;
    auto geometry_parts = ComponentImageGenerationGeometry(source,
                                                           {.prompt = std::move(source->prompt),
                                                            .drop = source->drop,
                                                            .width = source->width,
                                                            .height = source->height,
                                                            .noise = std::move(source->noise),
                                                            .keep = source->keep,
                                                            .output = source->output},
                                                           settings, construction);
    if (!geometry_parts) return Error(geometry_parts.error());
    EngineImageParts out;
    out.policy = held->policy;
    out.keep = held;
    out.construction = construction;
    out.architecture = family.mapping->architecture;
    out.backend = std::make_shared<kernels::StepBackend>(held->backend->Backend());
    out.geometry_keep = geometry_parts->keep;
    out.identity = geometry_parts->identity;
    out.request = std::move(geometry_parts->request);
    out.stages = std::move(geometry_parts->stages);
    for (std::size_t i = 0; i < 3; ++i) {
      const auto count = source->weights[i]->artifact().layout().groups.size();
      out.weights.push_back({.weights = std::move(source->weights[i]),
                             .places = std::vector(count, en::GroupPlace::kDevice),
                             .slabs = {}});
    }
    out.kernel_memory.push_back({.name = "image shared cuBLAS", .bytes = blas_bytes});
    const auto selected_bytes = held->backend->persistent_bytes();
    if (selected_bytes)
      out.kernel_memory.push_back({.name = "image selected kernel state", .bytes = selected_bytes});
    out.prepare = [&node, owner, held, blas_bytes,
                   selected_bytes](std::span<const std::uint64_t> bases) -> Status {
      if (bases.size() != 1U + static_cast<std::size_t>(selected_bytes != 0))
        return Error("image kernel workspaces are absent");
      held->selected_state = {selected_bytes ? bases[1] : 0, selected_bytes};
      auto blas = kg::CublasHandle::Create(0, node.execution(),
                                           node.stream(static_cast<std::uint32_t>(owner)),
                                           {.base = bases[0], .size = base::Bytes(blas_bytes)});
      if (!blas) return Error(blas.error().detail);
      held->cublas = std::move(*blas);
      return held->backend->LendCublas(held->cublas.get());
    };
    // The launch spans the pool's virtual capacity because one launch
    // serves every geometry the model takes (a growth needs no new launch,
    // and the program lifecycle binds one per model). What a step draws is
    // bounded per operation by its planned scratch (LaunchContext::Begin
    // limits the pool to it), within the stage's measured envelope, which
    // EngineImage checks at bind against its geometry's pool prefix: no
    // draw passes the prefix the image's closures name. It never runs
    // concurrent lanes: their pools would come from the end of that virtual
    // capacity, past anything the image's closures name (component_build.cc
    // configures lanes for LLM recipes only, from each model's own prefix).
    out.launch = [&node,
                  held]() -> std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> {
      auto made = held->backend->Launch(
          {.base = node.pool().base, .size = base::Bytes(node.pool().capacity_bytes)},
          held->selected_state);
      if (made && static_cast<kg::GgmlStepLaunch*>(made->get())->context().lanes() != 0)
        return std::unexpected("an image launch has concurrent lanes");
      return made;
    };
    out.retire = [held] {
      held->backend->LaunchReleased();
      held->selected_state = {};
      held->cublas.reset();
    };
    return out;
  };
}
namespace {
struct InstalledImageOwner {
  Paid paid;
  std::shared_ptr<const ImageFamily> family;
  chat::ImagePromptRenderer render = nullptr;  // its mapping's prompt renderer
  std::optional<tokenizer::Tokenizer> tokenizer;
  std::filesystem::path noise;
  ModelSettings settings;
};
struct ImageInputOwner {
  Paid paid;
};
std::expected<ImageGenerationRequest, std::string> ReadImageInput(
    const std::shared_ptr<InstalledImageOwner>& owner, const ImageInput& input,
    ModelSettings& settings) {
  const auto& profile = owner->family->profile;
  settings = owner->settings;
  if (input.steps) settings.image_steps.value = input.steps;
  if ((input.width == 0) != (input.height == 0))
    return Error("image request needs both dimensions or neither");
  const auto width = input.width ? input.width : settings.image_size.value;
  const auto height = input.height ? input.height : settings.image_size.value;
  if (!width || !height || width % profile.vae.spatial_factor ||
      height % profile.vae.spatial_factor || !settings.image_steps.value)
    return Error("image dimensions or schedule do not match the source geometry");
  const auto rows =
      std::uint64_t{width / profile.vae.spatial_factor} * (height / profile.vae.spatial_factor);
  if (rows > UINT32_MAX || rows > std::uint64_t{PTRDIFF_MAX} / (4 * profile.vae.z_dim))
    return Error("image noise extent overflows its representation");
  const auto elements = rows * profile.vae.z_dim;
  if (input.prompt.size() > (UINT64_MAX - 65536) / 16)
    return Error("image prompt working bound overflows");
  const auto rendered_bound = chat::ImagePromptBytesBound(input.prompt.size());
  if (rendered_bound > SIZE_MAX) return Error("image rendering bound overflows");
  const auto token_bound =
      owner->tokenizer->OutputTokensBound(static_cast<std::size_t>(rendered_bound));
  const auto work = owner->tokenizer->WorkingBytes(static_cast<std::size_t>(rendered_bound));
  // Fund rendering, its complete tokenization and the marked-span token
  // indexes before the renderer or encoder allocates any of them.
  std::uint64_t initial = 16 * input.prompt.size() + 65536;
  for (const auto& [count, unit] :
       {std::pair{work, std::uint64_t{1}},
        std::pair{token_bound, std::uint64_t{2} * sizeof(tokenizer::TokenId)},
        std::pair{rendered_bound, std::uint64_t{2} * sizeof(std::size_t)}}) {
    std::uint64_t term = 0;
    if (__builtin_mul_overflow(count, unit, &term) ||
        __builtin_add_overflow(initial, term, &initial))
      return Error("image tokenization working bound overflows");
  }
  for (const auto term : {8 * elements, std::uint64_t{input.output.native().size()},
                          std::uint64_t{input.noise.native().size()}}) {
    std::uint64_t doubled = 0;
    if (__builtin_mul_overflow(term, std::uint64_t{2}, &doubled) ||
        __builtin_add_overflow(initial, doubled, &initial))
      return Error("image input working bound overflows");
  }
  if (!ChargeExecutionMetadata(owner->paid.allowance, initial, owner->paid.node, owner->paid.cap))
    return Error("image input construction exceeds available memory");
  auto keep = std::make_shared<ImageInputOwner>();
  keep->paid.allowance = owner->paid.allowance;
  keep->paid.node = owner->paid.node;
  keep->paid.cap = owner->paid.cap;
  keep->paid.bytes = initial;
  auto rendered = owner->render(input.prompt);
  if (!rendered) return Error(rendered.error().ToString());
  if (rendered->rendered.text.size() > rendered_bound)
    return Error("image renderer exceeded its paid framing bound");
  ImageGenerationRequest request;
  request.keep = keep;
  request.width = width;
  request.height = height;
  request.output = input.output;
  std::vector<std::size_t> span_tokens;
  span_tokens.reserve(rendered->rendered.specials.size());
  if (auto encoded =
          owner->tokenizer->EncodeMarked(rendered->rendered.text, rendered->rendered.specials,
                                         {.special = tokenizer::SpecialTokens::kUserDefinedOnly,
                                          .add_bos_eos = false,
                                          .max_bytes = static_cast<std::size_t>(rendered_bound),
                                          .max_tokens = static_cast<std::size_t>(token_bound)},
                                         request.prompt, &span_tokens);
      !encoded)
    return Error(encoded.error().ToString());
  if (rendered->drop_span >= span_tokens.size())
    return Error("image dropped prefix span is absent");
  const auto drop = span_tokens[rendered->drop_span];
  if (drop > UINT32_MAX) return Error("image dropped prefix is unrepresentable");
  request.drop = static_cast<std::uint32_t>(drop);
  const auto& noise_path = input.noise.empty() ? owner->noise : input.noise;
  auto noise = platform::ReadSmallFile(noise_path, elements * sizeof(std::uint16_t));
  if (!noise || noise->size() != elements * sizeof(std::uint16_t))
    return Error("image noise file differs from the exact requested BF16 extent");
  request.noise.resize(elements);
  std::memcpy(request.noise.data(), noise->data(), noise->size());
  return request;
}
std::expected<std::uint64_t, std::string> RetainedImageArtifactBytes(const artifact::Artifact& a) {
  // Documents bound retained names/string capacities. Index-derived records,
  // source placements and future registration descriptors each have distinct
  // inventory terms; no decoded weight payload belongs to this payment.
  std::uint64_t bytes = 8 * artifact::kMaxManifestBytes;
  for (const auto& file : a.files())
    if (!file.path.starts_with("data/")) {
      std::uint64_t term = 0;
      if (__builtin_mul_overflow(file.bytes.value(), std::uint64_t{8}, &term) ||
          __builtin_add_overflow(bytes, term, &bytes))
        return Error("image source metadata bound overflows");
    }
  for (const auto term : {4096ULL * a.resources().size(), 4096ULL * a.expert_arrays().size(),
                          1024ULL * a.chunk_sha256().size(), 4096ULL * a.groups().size(),
                          1024ULL * a.shards().size()})
    if (__builtin_add_overflow(bytes, term, &bytes))
      return Error("image source descriptor bound overflows");
  return bytes;
}
}  // namespace

EngineImageBuild ComponentImageServingBuild(en::PagedNode& node, std::filesystem::path store,
                                            std::string composition_id, int owner,
                                            std::string prompt, std::filesystem::path noise,
                                            std::uint64_t parser_loan, std::uint64_t cap) {
  return [&node, store = std::move(store), composition_id = std::move(composition_id), owner,
          prompt = std::move(prompt), noise = std::move(noise), parser_loan, cap,
          retained = std::shared_ptr<InstalledImageOwner>{},
          source = std::shared_ptr<ImageGenerationSource>{}](const ModelSettings& settings) mutable
             -> std::expected<EngineImageParts, std::string> {
    // These installed readers run under the Server's still-live parser loan;
    // retained allocations acquire their own payment before the loan retires.
    const auto required = StartupBootstrapBytes(1);
    if (!required || parser_loan < *required || !node.open_for_setup() || cap == 0 || source)
      return Error("ordinary image source requires its held startup parser loan and stopped node");
    auto allowance = serving_support::AdmitStartup(node, 65536, cap);
    if (!allowance) return Error(allowance.error());
    retained = std::make_shared<InstalledImageOwner>();
    retained->paid.allowance = *allowance;
    retained->paid.bytes = 65536;
    retained->paid.node = &node;
    retained->paid.cap = cap;
    retained->noise = noise;
    retained->settings = settings;
    source = std::make_shared<ImageGenerationSource>();
    source->keep = retained;
    source->construction_node = &node;
    source->construction_cap = cap;
    auto composition = artifact::OpenComposition(store / composition_id, composition_id);
    if (!composition) return Error(composition.error().ToString());
    const auto* mapping = spec::FindImageMapping(composition->architecture());
    if (mapping == nullptr || mapping->components.size() != source->weights.size())
      return Error("installed composition has no qualified shared image mapping");
    if (!retained->paid.Add(2 * artifact::kMaxMetadataTotal + 8 * artifact::kMaxManifestBytes))
      return Error("retained image composition metadata is unpaid");
    source->composition = std::make_shared<const artifact::Composition>(std::move(*composition));
    for (std::size_t i = 0; i < mapping->components.size(); ++i) {
      const auto* component = source->composition->Find(mapping->components[i].role);
      if (!component) return Error("image composition is missing a required component");
      auto weights = std::make_unique<en::PagedWeights>();
      artifact::OpenOptions options;
      options.expected_id = component->artifact;
      if (auto opened = weights->Open(store / component->artifact, options); !opened)
        return Error(opened.error());
      if (weights->artifact().model().architecture != component->architecture)
        return Error("image artifact architecture differs from its composition");
      auto bytes = RetainedImageArtifactBytes(weights->artifact());
      if (!bytes || !retained->paid.Add(*bytes))
        return Error(bytes ? "retained image source metadata is unpaid" : bytes.error());
      source->weights[i] = std::move(weights);
    }
    std::array<const artifact::Artifact*, 3> opened{};
    for (std::size_t i = 0; i < opened.size(); ++i) opened[i] = &source->weights[i]->artifact();
    auto family = ReadImageFamily(*source->composition, opened);
    if (!family) return Error(family.error());
    source->family = *family;
    retained->family = std::move(*family);
    // The pipeline's prompt renderer, its mapping's (by name).
    retained->render = chat::FindImagePromptRenderer(mapping->prompt);
    if (retained->render == nullptr)
      return Error("image pipeline has no prompt renderer of its mapping's name");
    const auto tokenizer_file = source->composition->format_version() >= 2
                                    ? std::string(mapping->tokenizer)
                                    : std::filesystem::path(mapping->tokenizer).filename().string();
    const auto json = source->composition->Metadata(tokenizer_file);
    if (!json) return Error("image composition has no native tokenizer metadata");
    const auto parsing = base::json::ParseWorkingBytes(*json);
    const auto expanded = 32 * std::uint64_t{json->size()} + parsing;
    if (expanded > parser_loan) return Error("image tokenizer exceeds the held parser loan");
    auto spec = tokenizer::ReadHfTokenizer(*json);
    if (!spec) return Error(spec.error().ToString());
    auto tokenizer_bytes = tokenizer::Tokenizer::ConstructionBytes(*spec);
    if (!tokenizer_bytes || !retained->paid.Add(*tokenizer_bytes))
      return Error("image tokenizer construction is unpaid");
    auto tokenizer = tokenizer::Tokenizer::Create(std::move(*spec));
    if (!tokenizer) return Error(tokenizer.error().ToString());
    retained->tokenizer.emplace(std::move(*tokenizer));
    ModelSettings request_settings;
    auto initial = ReadImageInput(retained, {.prompt = prompt}, request_settings);
    if (!initial) return Error(initial.error());
    static_cast<ImageGenerationRequest&>(*source) = std::move(*initial);
    // Retain the parsed model source independently of the consumable input.
    auto build = ComponentImageGenerationBuild(node, owner, source, *allowance);
    auto parts = build(request_settings);
    if (!parts) return Error(parts.error());
    parts->prepare_request = [retained, source, construction = *allowance](
                                 const ImageInput& input, const base::Sha256Digest& layout)
        -> std::expected<std::variant<ImageRequest, ImageGeometry>, std::string> {
      ModelSettings next_settings;
      auto request = ReadImageInput(retained, input, next_settings);
      if (!request) return Error(request.error());
      auto checked = RequestOf(*request, next_settings, retained->family->profile);
      if (!checked) return Error(checked.error());
      base::Sha256 hash;
      hash.Update("llmp-image-request-layout-v1");
      hash.Update(retained->family->mapping->architecture);
      const std::array shape{checked->width, checked->height, checked->prompt_rows, request->drop};
      hash.Update(std::as_bytes(std::span(shape)));
      if (hash.Finish() == layout) {
        auto content =
            ComponentImageGenerationRequest(retained->family, std::move(*request), next_settings,
                                            construction, retained->paid.node, retained->paid.cap);
        if (!content) return Error(content.error());
        return std::variant<ImageRequest, ImageGeometry>(std::move(*content));
      }
      auto geometry = ComponentImageGenerationGeometry(source, std::move(*request), next_settings,
                                                       construction);
      if (!geometry) return Error(geometry.error());
      return std::variant<ImageRequest, ImageGeometry>(std::move(*geometry));
    };
    return parts;
  };
}

}  // namespace llmp::runtime
