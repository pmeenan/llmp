// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/image_pipeline.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <numeric>
#include <set>
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
#include "import/source_mapping.h"
#include "import/stage_recipe.h"
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
// config.json's), re-checking its identity and recording its parsed receipt.
std::expected<std::vector<spec::SourceValue>, Error> FileValues(
    std::vector<SourceFile>& pipeline_files, std::string_view name) {
  const auto found = std::ranges::find(pipeline_files, name, &SourceFile::name);
  if (found == pipeline_files.end()) return Bad("the pipeline lacks a file its mapping reads");
  const auto index = static_cast<std::uint64_t>(found - pipeline_files.begin());
  auto opened = OpenCheckpointFile(found->path, index);
  if (!opened) return std::unexpected(opened.error());
  if (opened->bytes != found->bytes || opened->device != found->device ||
      opened->inode != found->inode) {
    return std::unexpected(Error{.rule = Rule::kIo,
                                 .reason = "a pipeline file changed identity or size after listing",
                                 .file = index});
  }
  auto text = ReadDocument(*opened, kMaxPipelineFileBytes, index);
  if (!text) return std::unexpected(text.error());
  const auto digest = base::Sha256().Update(*text).Finish();
  if (found->parsed.bytes != 0 &&
      (found->parsed.bytes != opened->bytes || digest != found->parsed.sha256)) {
    return std::unexpected(Error{.rule = Rule::kIo,
                                 .reason = "a pipeline file changed content after parsing",
                                 .file = index});
  }
  found->parsed = {.bytes = opened->bytes, .sha256 = digest};
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
                                             const spec::ImageMapping& mapping,
                                             std::vector<SourceFile>& pipeline_files) {
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
    auto read = FileValues(pipeline_files, name);
    if (!read) return std::unexpected(read.error());
    file_values.push_back(std::move(*read));
    files.push_back({.name = name, .values = file_values.back()});
  }
  auto family = spec::SynthesizeImageFamily(mapping, components, files);
  if (!family) return Bad("the pipeline's configs do not give its image mapping's family");
  return std::move(*family);
}

// A component's plan: the tensors its family names (at the family's
// dtype and shape), each copied whole; the checkpoint's others its mapping
// names as left out (a text encoder's vision tower and output head, a
// VAE's encoder: what text to image generation does not run) are not
// imported, and any other is refused.
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

}  // namespace

bool IsImagePipeline(const Checkpoint& source) {
  return source.container == "diffusers" && spec::FindImageMapping(source.architecture) != nullptr;
}

namespace image_pipeline_detail {
std::expected<ComponentPlan, Error> PlanComponent(
    const Checkpoint& comp, std::span<const model::ImageTensor> expected_tensors,
    std::span<const std::string_view> ignored, const spec::ModelSpec& spec,
    std::vector<ComponentGroup> groups, std::span<const std::string_view> ignored_prefixes,
    std::span<const spec::RecipeExpansion::Tensor> recipe_inventory) {
  if (comp.container != "safetensors" && comp.container != "gguf")
    return Bad("image component uses an unsupported source container", Rule::kContainer);
  struct PhysicalExtent {
    std::uint32_t file = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<std::uint32_t> physical_files(comp.files.size());
  std::iota(physical_files.begin(), physical_files.end(), 0U);
  for (std::size_t i = 0; i < comp.files.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      const auto& a = comp.files[i];
      const auto& b = comp.files[j];
      const bool aliases = (!a.path.empty() && a.path == b.path) ||
                           (a.inode != 0 && a.device == b.device && a.inode == b.inode);
      if (!aliases) continue;
      if (a.bytes != b.bytes || a.device != b.device || a.inode != b.inode)
        return Bad("aliased image source files disagree", Rule::kConfig);
      physical_files[i] = physical_files[j];
      break;
    }
  }
  std::vector<PhysicalExtent> source_extents;
  source_extents.reserve(comp.tensors.size());
  std::set<std::string_view> source_names;
  for (std::size_t i = 0; i < comp.tensors.size(); ++i) {
    const auto& tensor = comp.tensors[i];
    if (tensor.name.empty() || !NoControlCharacters(tensor.name) ||
        !source_names.insert(tensor.name).second || tensor.extent.file >= comp.files.size())
      return std::unexpected(
          Error{.rule = Rule::kBounds,
                .reason = "an image source tensor has an invalid or duplicate name/file",
                .file = tensor.extent.file,
                .kind = ItemKind::kTensor,
                .item = i});
    if (tensor.extent.offset > UINT64_MAX - tensor.extent.bytes ||
        tensor.extent.offset + tensor.extent.bytes > comp.files[tensor.extent.file].bytes)
      return std::unexpected(Error{.rule = Rule::kBounds,
                                   .reason = "an image source tensor extent exceeds its file",
                                   .file = tensor.extent.file,
                                   .kind = ItemKind::kTensor,
                                   .item = i});
    const auto source_bytes = SourcePartBytes(tensor);
    if (!source_bytes || *source_bytes != tensor.extent.bytes)
      return std::unexpected(
          Error{.rule = Rule::kContainer,
                .reason = "an image source tensor format disagrees with its extent",
                .file = tensor.extent.file,
                .kind = ItemKind::kTensor,
                .item = i});
    source_extents.push_back({.file = physical_files[tensor.extent.file],
                              .begin = tensor.extent.offset,
                              .end = tensor.extent.offset + tensor.extent.bytes});
  }
  std::ranges::sort(source_extents, [](const PhysicalExtent& a, const PhysicalExtent& b) {
    return a.file != b.file ? a.file < b.file : a.begin < b.begin;
  });
  for (std::size_t i = 1; i < source_extents.size(); ++i)
    if (source_extents[i].file == source_extents[i - 1].file &&
        source_extents[i].begin < source_extents[i - 1].end)
      return Bad("image source tensor extents overlap", Rule::kContainer);

  // Every tensor is one the stage reads or one its mapping leaves out by
  // name: another (an extra norm or bias of a variant) would run with
  // different math than the checkpoint's, as the LLM route refuses.
  std::set<std::string_view> read;
  for (const auto& need : expected_tensors)
    if (!read.insert(need.name).second)
      return Bad("an image stage binds one physical source tensor more than once");
  for (std::size_t i = 0; i < comp.tensors.size(); ++i) {
    const std::string_view name = comp.tensors[i].name;
    const bool consumed = read.contains(name);
    const bool omitted =
        std::ranges::any_of(
            ignored, [&](std::string_view rule) { return spec::IgnoredImageTensor(rule, name); }) ||
        std::ranges::any_of(ignored_prefixes,
                            [&](std::string_view prefix) { return name.starts_with(prefix); });
    if (consumed && omitted)
      return std::unexpected(
          Error{.rule = Rule::kContainer,
                .reason = "an image mapping both reads and omits a source tensor",
                .file = comp.tensors[i].extent.file,
                .kind = ItemKind::kTensor,
                .item = i});
    if (consumed || omitted) continue;
    return std::unexpected(Error{.rule = Rule::kContainer,
                                 .reason = "a component tensor its image mapping neither reads "
                                           "nor leaves out",
                                 .file = comp.tensors[i].extent.file,
                                 .kind = ItemKind::kTensor,
                                 .item = i});
  }
  spec::Source canonical{.architecture = comp.architecture, .container = comp.container};
  canonical.tensors.reserve(expected_tensors.size());
  std::vector<CopyRoute> routes;
  routes.reserve(expected_tensors.size());
  std::uint64_t payload = 0;

  for (const auto& need : expected_tensors) {
    const Tensor* found = comp.FindTensor(need.name);
    if (!found) return Bad("a component lacks a tensor its mapping names");
    const auto shape = found->format.shape;
    if (shape.rank != need.shape.size())
      return Bad("a component tensor's rank is not its family's");
    for (std::size_t d = 0; d < need.shape.size(); ++d)
      if (shape.ne[d] != need.shape[need.shape.size() - 1 - d])
        return Bad("a component tensor's shape is not its family's");
    const auto parts = format::PartsOf(found->format);
    if (!parts || parts->size() != 1 || found->part != format::PartRole::kData ||
        found->format.params != format::EncodingParams{} || found->format.stacked != 1 ||
        found->extent.bytes != (*parts)[0].bytes)
      return Bad("a component tensor has an inconsistent source format");
    canonical.tensors.push_back({.name = need.name,
                                 .format = found->format,
                                 .part = format::PartRole::kData,
                                 .experts = false});
    const auto si = static_cast<std::uint32_t>(found - comp.tensors.data());
    routes.push_back({
        .source_tensor = si,
        .extent = found->extent,
        .readable = 0,
    });
    if (__builtin_add_overflow(payload, found->extent.bytes, &payload)) {
      return Bad("a component's payload exceeds 64-bit bounds", Rule::kBounds);
    }
  }
  std::set<std::pair<std::string_view, format::PartRole>> checked_source_shapes;
  for (const auto& contract : recipe_inventory) {
    if (contract.source_dims.empty()) continue;
    if (!checked_source_shapes.emplace(contract.name, contract.part).second)
      return Bad("a recipe repeats an exact raw source-dimension contract");
    const auto expected_count =
        std::ranges::count_if(expected_tensors, [&](const model::ImageTensor& tensor) {
          return tensor.name == contract.name && tensor.part == contract.part;
        });
    if (expected_count != 1)
      return Bad("a raw source-dimension contract has no unique physical tensor binding");
    const auto found = std::ranges::find_if(comp.tensors, [&](const Tensor& tensor) {
      return tensor.name == contract.name && tensor.part == contract.part;
    });
    if (found == comp.tensors.end())
      return Bad("a raw source-dimension contract has no source tensor");
    if (std::ranges::count_if(std::ranges::subrange(found, comp.tensors.end()),
                              [&](const Tensor& tensor) {
                                return tensor.name == contract.name && tensor.part == contract.part;
                              }) != 1)
      return Bad("a raw source-dimension contract has ambiguous source tensors");
    if (found->dims != contract.source_dims)
      return Bad("a source tensor's raw header dimensions differ from its recipe contract");
  }
  if (auto valid =
          components::ValidateSpec(spec, canonical.tensors, components::ValidationPurpose::kImport);
      !valid) {
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

}  // namespace image_pipeline_detail

std::expected<ComponentPlan, Error> PlanSpatialComponent(const Checkpoint& component,
                                                         const spec::Mapping& mapping,
                                                         std::uint64_t metadata_bytes) {
  if (mapping.stage_recipe == nullptr || mapping.stage.empty() ||
      mapping.role != mapping.stage_recipe->role || mapping.stage != mapping.stage_recipe->name)
    return Bad("configured spatial component mapping has no matching stage recipe",
               Rule::kUnsupported);
  if (auto valid = spec::ValidateMapping(mapping); !valid)
    return Bad("configured spatial component mapping is invalid", Rule::kConfig);
  if (!spec::SupportsSource(mapping, component.architecture, component.container))
    return Bad("configured spatial component mapping does not accept this source",
               Rule::kContainer);
  if (metadata_bytes == 0)
    return Bad("configured spatial component has no metadata budget", Rule::kBounds);

  const spec::ModelSpec family{};
  auto expanded =
      ExpandStageRecipe(*mapping.stage_recipe, family, metadata_bytes, component.metadata);
  if (!expanded) return Bad("cannot expand configured spatial component recipe");
  std::vector<model::ImageTensor> tensors;
  std::vector<ComponentGroup> groups;
  tensors.reserve(expanded->tensors.size());
  groups.reserve(expanded->tensors.size());
  for (const auto& tensor : expanded->tensors) {
    tensors.push_back(
        {.name = tensor.name, .dtype = {}, .shape = tensor.shape, .part = tensor.part});
    const auto group_kind =
        tensor.group == spec::RecipeGroupKind::kTable   ? ComponentGroup::Kind::kTable
        : tensor.group == spec::RecipeGroupKind::kLayer ? ComponentGroup::Kind::kLayer
                                                        : ComponentGroup::Kind::kGlobal;
    groups.push_back({.kind = group_kind, .layer = tensor.layer});
  }
  spec::ModelSpec component_spec{.pipeline = spec::PipelineKind::kImage,
                                 .stages = {std::move(expanded->stage)}};
  spec::Canonicalize(component_spec);
  return image_pipeline_detail::PlanComponent(component, tensors, mapping.ignored_tensors,
                                              component_spec, std::move(groups),
                                              mapping.ignored_tensor_prefixes, expanded->tensors);
}

std::expected<NormalizationCost, Error> ImagePipelineBound(const Checkpoint& source) {
  if (!IsImagePipeline(source)) {
    return Bad("not a diffusers pipeline with an image mapping", Rule::kContainer);
  }
  std::uint64_t total_payload = 0;
  std::uint64_t total_tensors = 0;
  for (const auto& comp : source.components) {
    for (const auto& t : comp.tensors) {
      if (__builtin_add_overflow(total_payload, t.extent.bytes, &total_payload) ||
          __builtin_add_overflow(total_tensors, std::uint64_t{1}, &total_tensors)) {
        return Bad("image pipeline payload exceeds 64-bit bounds", Rule::kBounds);
      }
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
  std::set<std::string_view> mapped_roles;
  for (const auto& component : source.components) {
    if (!mapped_roles.insert(component.role).second)
      return Bad("an image source repeats a physical component role");
    if (!std::ranges::contains(mapping.components, component.role, &spec::ImageComponentRule::role))
      return Bad("the image source contains a component role its mapping does not account for");
  }
  std::vector<SourceFile> pipeline_files = source.files;
  auto family = Family(source, mapping, pipeline_files);
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
  const auto vae_tensors = model::VaeDecoderTensors(p.vae, names.vae);
  auto vae_spec = spec::ImageDecodeTemplate(p.vae, vae_tensors);
  if (!vae_spec) return Bad("cannot build the VAE's spec");

  ImagePipelinePlan out;
  out.files = std::move(pipeline_files);
  out.components.reserve(mapping.components.size());
  for (const auto& rule : mapping.components) {
    const auto* comp = FindComponent(source, rule.role);
    if (comp == nullptr) return Bad("the pipeline lacks a component its mapping names");
    if (!std::ranges::contains(rule.containers, comp->container))
      return Bad("an image component uses a container its mapping does not allow",
                 Rule::kContainer);
    std::vector<model::ImageTensor> expected_tensors;
    std::vector<ComponentGroup> groups;
    std::vector<spec::RecipeExpansion::Tensor> recipe_inventory;
    spec::ModelSpec component_spec;
    const auto kind = rule.kinds.front();
    if (rule.recipe != nullptr) {
      auto expanded = ExpandStageRecipe(*rule.recipe, *family, std::uint64_t{512} << 20U);
      if (!expanded) return Bad("cannot expand the mapping's component stage recipe");
      expected_tensors.reserve(expanded->tensors.size());
      groups.reserve(expanded->tensors.size());
      for (auto& tensor : expanded->tensors) {
        expected_tensors.push_back(
            {.name = tensor.name, .dtype = {}, .shape = tensor.shape, .part = tensor.part});
        const auto group_kind =
            tensor.group == spec::RecipeGroupKind::kTable   ? ComponentGroup::Kind::kTable
            : tensor.group == spec::RecipeGroupKind::kLayer ? ComponentGroup::Kind::kLayer
                                                            : ComponentGroup::Kind::kGlobal;
        groups.push_back({.kind = group_kind, .layer = tensor.layer});
      }
      recipe_inventory = std::move(expanded->tensors);
      component_spec = {.pipeline = spec::PipelineKind::kImage,
                        .stages = {std::move(expanded->stage)}};
      spec::Canonicalize(component_spec);
    } else if (kind == spec::kImageTextKind) {
      expected_tensors = text_tensors;
      component_spec = text_spec->spec;
      groups = GroupsOf(kind, expected_tensors, p, names);
    } else if (kind == spec::kImageVaeKind) {
      expected_tensors = vae_tensors;
      component_spec = *vae_spec;
      groups = GroupsOf(kind, expected_tensors, p, names);
    } else {
      return Bad("an image component has no mapping-owned stage recipe");
    }
    auto plan = image_pipeline_detail::PlanComponent(*comp, expected_tensors, rule.ignored_tensors,
                                                     component_spec, std::move(groups), {},
                                                     recipe_inventory);
    if (!plan) return std::unexpected(plan.error());
    if (__builtin_add_overflow(out.payload_bytes, plan->payload_bytes, &out.payload_bytes)) {
      return Bad("image pipeline payload exceeds 64-bit bounds", Rule::kBounds);
    }
    out.components.push_back(std::move(*plan));
  }
  out.family = std::move(*family);
  return out;
}

}  // namespace llmp::import
