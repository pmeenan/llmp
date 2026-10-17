// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/image_pipeline.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "components/step.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/files.h"
#include "import/readers/diffusers.h"
#include "import/readers/safetensors.h"
#include "ir/graph.h"
#include "model/image_pipeline.h"
#include "spec/image_family.h"
#include "spec/image_spatial.h"

namespace llmp::import {
namespace {

std::unexpected<Error> Bad(std::string_view reason, Rule rule = Rule::kConfig) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}

const Checkpoint* FindComponent(const Checkpoint& source, std::string_view role) {
  for (const auto& comp : source.components) {
    if (comp.role == role) return &comp;
  }
  return nullptr;
}

// A kept pipeline file's members as source values (flattened as a
// config.json's).
std::expected<std::vector<spec::SourceValue>, Error> FileValues(const Checkpoint& source,
                                                                std::string_view name) {
  const auto found = std::ranges::find(source.files, name, &SourceFile::name);
  if (found == source.files.end()) return Bad("the pipeline lacks a file its mapping reads");
  const auto index = static_cast<std::uint64_t>(found - source.files.begin());
  auto opened = OpenCheckpointFile(found->path, index);
  if (!opened) return std::unexpected(opened.error());
  auto text = ReadDocument(*opened, kMaxPipelineFileBytes, index);
  if (!text) return std::unexpected(text.error());
  auto doc = base::json::Parse(*text, {.max_bytes = text->size(),
                                       .max_depth = 8,
                                       .max_values = std::size_t{1} << 16U,
                                       .max_string_bytes = text->size()});
  if (!doc) return Bad("a pipeline file its mapping reads is not valid JSON");
  auto metadata = FlattenMetadata(doc->root());
  if (!metadata) return std::unexpected(metadata.error());
  Checkpoint flat;
  flat.metadata = std::move(*metadata);
  return ValuesOf(flat);
}

// The family spec of the pipeline (the mapping's), from its components'
// configs and the kept files the mapping reads.
std::expected<spec::ModelSpec, Error> Family(const Checkpoint& source,
                                             const spec::ImageMapping& mapping) {
  std::vector<std::vector<spec::SourceValue>> values;
  std::vector<spec::ImageComponentSource> components;
  values.reserve(mapping.components.size());
  for (const auto& rule : mapping.components) {
    const auto* comp = FindComponent(source, rule.role);
    if (comp == nullptr) return Bad("the pipeline lacks a component its mapping names");
    values.push_back(ValuesOf(*comp));
  }
  for (std::size_t i = 0; i < mapping.components.size(); ++i) {
    const auto* comp = FindComponent(source, mapping.components[i].role);
    components.push_back(
        {.role = comp->role, .architecture = comp->architecture, .values = values[i]});
  }
  std::vector<std::string_view> names;
  for (const auto& r : mapping.params)
    if (!r.file.empty() && !std::ranges::contains(names, r.file)) names.push_back(r.file);
  for (const auto& r : mapping.requirements)
    if (!r.file.empty() && !std::ranges::contains(names, r.file)) names.push_back(r.file);
  std::vector<std::vector<spec::SourceValue>> file_values;
  file_values.reserve(names.size());
  std::vector<spec::ImageSource> files;
  for (const auto name : names) {
    auto read = FileValues(source, name);
    if (!read) return std::unexpected(read.error());
    file_values.push_back(std::move(*read));
    files.push_back({.name = name, .values = file_values.back()});
  }
  auto family = spec::SynthesizeImageFamily(mapping, components, files);
  if (!family) return Bad("the pipeline's configs do not give its image mapping's family");
  return std::move(*family);
}

// A component's plan: the tensors its family names (at the family's
// dtype and shape), each copied whole; the checkpoint's others (a text
// encoder's vision tower and output head, a VAE's encoder: what text to
// image generation does not run) are not imported, as a target's import
// leaves its drafter's tensors.
// Each tensor's group, as the component's own structure divides it (its
// spec binds every tensor in one spatial layer): the text encoder's token
// table and one group a layer; the denoiser's globals and one group a
// block; the VAE's up blocks one group each, the rest of its decoder one.
std::vector<ComponentGroup> GroupsOf(std::string_view kind,
                                     std::span<const model::ImageTensor> tensors,
                                     const model::ImageProfile& p, const model::ImageNames& names) {
  using Kind = ComponentGroup::Kind;
  std::vector<ComponentGroup> out(tensors.size());
  if (kind == spec::kImageTextKind) {
    constexpr auto kPerLayer = static_cast<std::size_t>(model::TextTensor::kCount);
    for (std::size_t i = 0; i < tensors.size(); ++i)
      out[i] = i == 0 ? ComponentGroup{.kind = Kind::kTable}
                      : ComponentGroup{.kind = Kind::kLayer,
                                       .layer = static_cast<std::uint32_t>((i - 1) / kPerLayer)};
  } else if (kind == spec::kImageFlowKind) {
    constexpr auto kGlobals = static_cast<std::size_t>(model::DenoiserGlobal::kCount);
    constexpr auto kPerBlock = static_cast<std::size_t>(model::BlockTensor::kCount);
    for (std::size_t i = kGlobals; i < tensors.size(); ++i)
      out[i] = {.kind = Kind::kLayer,
                .layer = static_cast<std::uint32_t>((i - kGlobals) / kPerBlock)};
  } else {
    const auto pattern = names.vae.up_block;
    const auto at = pattern.find("{layer}");
    for (std::size_t i = 0; i < tensors.size(); ++i)
      for (std::uint32_t b = 0; b < p.vae.dim_mult.size() && at != std::string_view::npos; ++b) {
        const auto prefix = std::string(pattern.substr(0, at)) + std::to_string(b) +
                            std::string(pattern.substr(at + 7)) + ".";
        if (tensors[i].name.starts_with(prefix)) out[i] = {.kind = Kind::kLayer, .layer = b};
      }
  }
  return out;
}

std::expected<ComponentPlan, Error> PlanComponent(
    const Checkpoint& comp, std::span<const model::ImageTensor> expected_tensors,
    const spec::ModelSpec& spec, std::vector<ComponentGroup> groups) {
  if (comp.container != "safetensors") {
    return Bad("component container must be safetensors", Rule::kContainer);
  }
  spec::Source canonical{.architecture = comp.architecture, .container = comp.container};
  canonical.tensors.reserve(expected_tensors.size());
  std::vector<CopyRoute> routes;
  routes.reserve(expected_tensors.size());
  std::uint64_t payload = 0;

  for (const auto& need : expected_tensors) {
    const Tensor* found = comp.FindTensor(need.name);
    if (!found) return Bad("a component lacks a tensor its mapping names");
    if (found->dtype != need.dtype || found->dims != need.shape) {
      return Bad("a component tensor's format or shape is not its family's");
    }
    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(need.shape.size());
    for (std::size_t d = 0; d < need.shape.size(); ++d) {
      shape.ne[d] = need.shape[need.shape.size() - 1 - d];
    }
    const auto encoding = need.dtype == "BF16" ? format::Encoding::kBF16 : format::Encoding::kF32;
    canonical.tensors.push_back({
        .name = need.name,
        .format = ir::Dense(encoding, shape),
        .part = format::PartRole::kData,
        .experts = false,
    });
    const auto si = static_cast<std::uint32_t>(found - comp.tensors.data());
    routes.push_back({
        .source_tensor = si,
        .extent = found->extent,
        .readable = 0,
    });
    payload += found->extent.bytes;
  }
  if (auto valid = components::ValidateSpec(spec, canonical.tensors); !valid) {
    return Bad("a component's spec does not bind its tensors");
  }
  return ComponentPlan{
      .checkpoint = comp,
      .canonical = std::move(canonical),
      .spec = spec,
      .routes = std::move(routes),
      .payload_bytes = payload,
      .groups = std::move(groups),
  };
}

}  // namespace

bool IsImagePipeline(const Checkpoint& source) {
  return source.container == "diffusers" && spec::FindImageMapping(source.architecture) != nullptr;
}

std::expected<NormalizationCost, Error> ImagePipelineBound(const Checkpoint& source) {
  if (!IsImagePipeline(source)) {
    return Bad("not a diffusers pipeline with an image mapping", Rule::kContainer);
  }
  std::uint64_t total_payload = 0;
  std::uint64_t total_tensors = 0;
  for (const auto& comp : source.components) {
    for (const auto& t : comp.tensors) {
      total_payload += t.extent.bytes;
      ++total_tensors;
    }
  }
  return NormalizationCost{
      .metadata_bytes = std::uint64_t{512} << 20U,
      .work_units = total_tensors,
      .payload_bytes = total_payload,
  };
}

std::expected<ImagePipelinePlan, Error> PlanImagePipeline(const Checkpoint& source) {
  if (!IsImagePipeline(source)) {
    return Bad("not a diffusers pipeline with an image mapping", Rule::kContainer);
  }
  const auto& mapping = *spec::FindImageMapping(source.architecture);
  auto family = Family(source, mapping);
  if (!family) return std::unexpected(family.error());
  auto profile = spec::ImageProfileOf(*family, mapping);
  if (!profile) return Bad("the family spec does not read back");
  const auto& p = *profile;
  const auto& names = mapping.names;

  // Each component's tensors and a spec of it at a minimal geometry (one
  // prompt row; one latent row; a one-row VAE latent that its spatial
  // factor makes whole): what binds its tensors into groups.
  const auto text_tensors = model::TextEncoderTensors(p.text, names.text);
  auto text_spec = spec::ImageTextSpec(p.text, 1, 0, text_tensors);
  if (!text_spec) return Bad("cannot build the text encoder's spec");
  const auto denoiser_tensors = model::DenoiserTensors(p.denoiser, names.denoiser);
  auto denoiser_spec = spec::ImageFlowSpec(p.denoiser, 1, 1, denoiser_tensors);
  if (!denoiser_spec) return Bad("cannot build the denoiser's spec");
  const auto vae_tensors = model::VaeDecoderTensors(p.vae, names.vae);
  auto vae_spec = spec::ImageDecodeSpec(p.vae, 64, 64, vae_tensors);
  if (!vae_spec) return Bad("cannot build the VAE's spec");

  ImagePipelinePlan out;
  out.components.reserve(mapping.components.size());
  const std::array<std::pair<const std::vector<model::ImageTensor>*, const spec::ModelSpec*>, 3>
      by_kind{{{&text_tensors, &text_spec->spec},
               {&denoiser_tensors, &denoiser_spec->spec},
               {&vae_tensors, &vae_spec->spec}}};
  for (const auto& rule : mapping.components) {
    const auto* comp = FindComponent(source, rule.role);
    const auto kind = rule.kinds.front();
    const std::size_t at = kind == spec::kImageTextKind ? 0 : kind == spec::kImageFlowKind ? 1 : 2;
    auto plan = PlanComponent(*comp, *by_kind[at].first, *by_kind[at].second,
                              GroupsOf(kind, *by_kind[at].first, p, names));
    if (!plan) return std::unexpected(plan.error());
    out.payload_bytes += plan->payload_bytes;
    out.components.push_back(std::move(*plan));
  }
  out.family = std::move(*family);
  return out;
}

}  // namespace llmp::import
