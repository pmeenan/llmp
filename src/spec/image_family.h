// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Image pipeline mappings and the family spec a composition stores (D-089,
// D-109, D-113; docs/engine-components.md#image-pipelines).
//
// An image pipeline (a diffusers pipeline: a text encoder, a flow
// transformer and a VAE, joined by a composition) is not built from
// per-layer component specs yet (M3.5): each request's geometry makes its
// stages' graphs (spec/image_spatial.h) from the family's hyperparameters.
// Those are data. An ImageMapping, compiled in and validated like a
// mapping (mapping.h), names a pipeline architecture's components (each a
// role, a stored stage and its accepted architectures), where each family
// parameter comes from (a key of a component's config.json, flattened as
// the importer reads it, or of a kept pipeline file such as the
// scheduler's, or a constant the source library defines in code), what
// the source must say besides (requirements), its checkpoints' tensor
// names and its serving recipe with the planning policy it plans with.
//
// SynthesizeImageFamily applies a mapping at import: the family spec, a
// pipeline spec of one stage a component (in the composition's component
// order, which is by role), each stage one layer of family components
// whose parameters are the family's (the kinds and parameters this file's
// schema defines: image.text_transformer, image.flow_transformer,
// image.flow_schedule, image.vae_decoder). The composition keeps it as
// its spec.json, and its approval pins its digest. ImageProfileOf reads a
// stored family spec back into the model's profile, checking every
// parameter against the schema's ranges; the runtime builds every graph
// from it, so no family's numbers are compiled in.

#ifndef LLMP_SPEC_IMAGE_FAMILY_H_
#define LLMP_SPEC_IMAGE_FAMILY_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

#include "model/image_pipeline.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::spec {

// The schema's component kinds.
inline constexpr std::string_view kImageTextKind = "image.text_transformer";
inline constexpr std::string_view kImageFlowKind = "image.flow_transformer";
inline constexpr std::string_view kImageScheduleKind = "image.flow_schedule";
inline constexpr std::string_view kImageVaeKind = "image.vae_decoder";

// A component of the pipeline: its composition role (model_index.json's),
// its stored stage, and the architectures (a config.json's model_type or
// _class_name) it accepts, the first the one its approval records.
struct ImageComponentRule {
  std::string_view role{};
  std::string_view stage{};
  StageRole stage_role = StageRole::kTextEncoder;
  std::span<const std::string_view> kinds{};  // the family components its stage holds
  std::span<const std::string_view> architectures{};
};

// How a family parameter is read: from a component's config (by role) or
// a kept pipeline file (by its name in the pipeline, "scheduler/..."),
// under `key`, as the parameter's type (an integer key read as a real, a
// list of flags as integers 0 and 1, `reverse` read back to front), or
// with no key, the constant `value` (a real's `value.real`, an integer's
// `value.integer`; a list's constants are not supported).
struct ImageParamRule {
  std::string_view kind{};
  std::string_view param{};
  std::string_view role{};
  std::string_view file{};
  std::string_view key{};
  Literal value{};
  bool reverse = false;
};

// What the source must say besides: `key` (of a role's config or a kept
// file) holds `value` of `type` (kInt, kReal, kBool or kString).
struct ImageRequirement {
  std::string_view role{};
  std::string_view file{};
  std::string_view key{};
  ValueType type = ValueType::kString;
  Literal value{};
};

struct ImageMapping {
  std::string_view architecture{};  // the pipeline's (model_index.json's _class_name)
  std::uint16_t version = 1;
  std::span<const ImageComponentRule> components{};  // sorted by role
  std::span<const ImageParamRule> params{};
  std::span<const ImageRequirement> requirements{};
  model::ImageNames names{};
  // How serving runs it: its settings' names and, as the recipe's serving
  // rule, the planning policy every stage plans with (planner/policy.h
  // text).
  const ServingRecipe* serving = nullptr;
  // Joint rows the denoiser's F32 attention is tuned for by default (the
  // qualified geometry's), beside a request's own.
  std::span<const std::uint64_t> attention_rows{};
  std::span<const std::uint64_t> attention_image_rows{};  // a request's own rows too, at these
  // The tokenizer the composition keeps (a kept file's name) and the
  // pipeline's prompt renderer (chat/chat.h's, by name).
  std::string_view tokenizer{};
  std::string_view prompt{};
};

std::span<const ImageMapping* const> ImageMappings();
const ImageMapping* FindImageMapping(std::string_view architecture);
const ImageMapping* ImageFamilyMapping(std::string_view architecture);
const ImageMapping& QwenImage21Mapping();
// Its own rules: components unique and sorted, every schema parameter of
// its components' kinds read by exactly one rule naming one of them, keys
// source keys, names present, one policy.
std::expected<void, std::string> ValidateImageMapping(const ImageMapping& mapping);

// A component's or a kept file's values, by its role or file name.
struct ImageSource {
  std::string_view name{};
  std::span<const SourceValue> values{};
};

// The family spec of a pipeline whose components (by role, with their
// architectures) and kept files have these values; refused where a
// component is missing or of another architecture, a requirement does not
// hold, or a parameter is absent, of another type or out of the schema's
// range.
struct ImageComponentSource {
  std::string_view role{};
  std::string_view architecture{};
  std::span<const SourceValue> values{};
};
std::expected<ModelSpec, std::string> SynthesizeImageFamily(
    const ImageMapping& mapping, std::span<const ImageComponentSource> components,
    std::span<const ImageSource> files);

// The profile a stored family spec gives, its parameters checked against
// the schema again (stored input is untrusted): stage for stage the
// mapping's, each with exactly its kinds.
std::expected<model::ImageProfile, std::string> ImageProfileOf(const ModelSpec& spec,
                                                               const ImageMapping& mapping);

}  // namespace llmp::spec

#endif  // LLMP_SPEC_IMAGE_FAMILY_H_
