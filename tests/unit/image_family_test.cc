// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Image pipeline mappings and family specs (spec/image_family.h): every
// compiled mapping's own rules; a family synthesized from a small
// pipeline's source values as the importer reads them, and refused where a
// requirement, component, architecture or parameter is not the mapping's;
// the profile a stored family spec reads back as, and its refusals of a
// spec the schema does not admit; the frozen family (the published
// composition's spec.json) pinned by the pipeline's approval.

#include "spec/image_family.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/sha256.h"
#include "image_family.h"
#include "image_published.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "spec/mappings/qwen_image.h"
#include "spec/spec.h"

namespace {
namespace sp = llmp::spec;
namespace md = llmp::model;

using Values = std::vector<sp::SourceValue>;

// A small pipeline under the Qwen-Image mapping's keys.
struct Toy {
  Values text{{"text_config.num_hidden_layers", std::int64_t{2}},
              {"text_config.hidden_size", std::int64_t{64}},
              {"text_config.num_attention_heads", std::int64_t{4}},
              {"text_config.num_key_value_heads", std::int64_t{2}},
              {"text_config.head_dim", std::int64_t{16}},
              {"text_config.intermediate_size", std::int64_t{128}},
              {"text_config.vocab_size", std::int64_t{1000}},
              {"text_config.rms_norm_eps", 1e-06},
              {"text_config.rope_theta", std::int64_t{5000000}},
              {"text_config.hidden_act", std::string("silu")},
              {"text_config.attention_bias", false}};
  Values flow{{"num_layers", std::int64_t{2}},
              {"num_attention_heads", std::int64_t{2}},
              {"attention_head_dim", std::int64_t{32}},
              {"mlp_ratio", std::int64_t{3}},
              {"in_channels", std::int64_t{16}},
              {"out_channels", std::int64_t{16}},
              {"context_in_dim", std::int64_t{64}},
              {"axes_dims_rope", std::vector<std::int64_t>{8, 12, 12}},
              {"eps", 1e-06},
              {"patch_size", std::int64_t{1}},
              {"causal_condition", true}};
  Values vae{{"z_dim", std::int64_t{4}},
             {"decoder_base_dim", std::int64_t{8}},
             {"dim_mult", std::vector<std::int64_t>{1, 2, 4}},
             {"num_res_blocks", std::int64_t{1}},
             {"temperal_downsample", std::vector<std::int64_t>{0, 1}},
             {"out_channels", std::int64_t{3}},
             {"scale_factor_spatial", std::int64_t{4}},
             {"latents_mean", std::vector<double>{0.5, -0.25, 0.0, 1.0}},
             {"latents_std", std::vector<double>{1.0, 2.0, 0.5, 3.0}},
             {"is_residual", true}};
  Values scheduler{{"num_train_timesteps", std::int64_t{1000}},
                   {"base_image_seq_len", std::int64_t{256}},
                   {"max_image_seq_len", std::int64_t{8192}},
                   {"base_shift", 0.5},
                   {"max_shift", 0.9},
                   {"shift_terminal", 0.02},
                   {"time_shift_type", std::string("exponential")},
                   {"use_dynamic_shifting", true},
                   {"invert_sigmas", false},
                   {"stochastic_sampling", false}};
  std::string text_arch = "qwen3_vl";
  std::string flow_arch = "QwenImage21Transformer2DModel";
  std::string vae_arch = "AutoencoderKLQwenImage";

  std::expected<sp::ModelSpec, std::string> Synthesize() const {
    const std::vector<sp::ImageComponentSource> components{
        {.role = "text_encoder", .architecture = text_arch, .values = text},
        {.role = "transformer", .architecture = flow_arch, .values = flow},
        {.role = "vae", .architecture = vae_arch, .values = vae}};
    const std::vector<sp::ImageSource> files{
        {.name = "scheduler/scheduler_config.json", .values = scheduler}};
    return sp::SynthesizeImageFamily(sp::mappings::QwenImage21Mapping(), components, files);
  }
};

void Set(Values& values, std::string_view key, sp::ParamValue value) {
  for (auto& v : values)
    if (v.key == key) {
      v.value = std::move(value);
      return;
    }
  values.push_back({std::string(key), std::move(value)});
}
void Drop(Values& values, std::string_view key) {
  std::erase_if(values, [&](const sp::SourceValue& v) { return v.key == key; });
}

TEST(ImageFamily, EveryMappingIsValidAndFoundByItsArchitecture) {
  ASSERT_FALSE(sp::ImageMappings().empty());
  for (const sp::ImageMapping* m : sp::ImageMappings()) {
    SCOPED_TRACE(m->architecture);
    const auto valid = sp::ValidateImageMapping(*m);
    EXPECT_TRUE(valid.has_value()) << valid.error();
    EXPECT_EQ(sp::FindImageMapping(m->architecture), m);
  }
  EXPECT_EQ(sp::FindImageMapping("FluxPipeline"), nullptr);
}

TEST(ImageFamily, AMappingRefusesWhatItCannotSay) {
  const auto& good = sp::mappings::QwenImage21Mapping();
  constexpr sp::ServingRecipe kNoRule{.family = "Qwen-Image", .name = "QwenImage21"};
  auto no_policy = good;
  no_policy.serving = &kNoRule;
  EXPECT_FALSE(sp::ValidateImageMapping(no_policy));
  auto unsupported_requirement = good;
  std::vector<sp::ImageRequirement> requirements(good.requirements.begin(),
                                                 good.requirements.end());
  ASSERT_FALSE(requirements.empty());
  requirements.front().type = sp::ValueType::kInts;
  unsupported_requirement.requirements = requirements;
  EXPECT_FALSE(sp::ValidateImageMapping(unsupported_requirement));
  auto no_serving = good;
  no_serving.serving = nullptr;
  EXPECT_FALSE(sp::ValidateImageMapping(no_serving));
  auto no_name = good;
  no_name.names.denoiser.globals[0] = {};
  EXPECT_FALSE(sp::ValidateImageMapping(no_name));
  // Names that would bind one tensor to two roles.
  auto same_layer = good;
  same_layer.names.text.layer[2] = same_layer.names.text.layer[1];
  EXPECT_FALSE(sp::ValidateImageMapping(same_layer));
  auto same_global = good;
  same_global.names.denoiser.globals[1] = same_global.names.denoiser.globals[0];
  EXPECT_FALSE(sp::ValidateImageMapping(same_global));
  auto same_block = good;
  same_block.names.denoiser.block[8] = same_block.names.denoiser.block[7];
  EXPECT_FALSE(sp::ValidateImageMapping(same_block));
  auto same_suffix = good;
  same_suffix.names.vae.norm2 = same_suffix.names.vae.norm1;
  EXPECT_FALSE(sp::ValidateImageMapping(same_suffix));
  auto same_part = good;
  same_part.names.vae.bias = same_part.names.vae.weight;
  EXPECT_FALSE(sp::ValidateImageMapping(same_part));
  auto no_layer = good;
  no_layer.names.text.layer[3] = "model.layers.v_proj.weight";
  EXPECT_FALSE(sp::ValidateImageMapping(no_layer));
  // A parameter rule dropped: its kind's parameter is read by none.
  std::vector<sp::ImageParamRule> fewer(good.params.begin() + 1, good.params.end());
  auto missing = good;
  missing.params = fewer;
  EXPECT_FALSE(sp::ValidateImageMapping(missing));
  // Components out of role order.
  std::vector<sp::ImageComponentRule> swapped(good.components.begin(), good.components.end());
  std::swap(swapped[0], swapped[1]);
  auto unsorted = good;
  unsorted.components = swapped;
  EXPECT_FALSE(sp::ValidateImageMapping(unsorted));
}

TEST(ImageFamily, ASynthesizedFamilyReadsBackAsItsSourceSaid) {
  const Toy toy;
  auto spec = toy.Synthesize();
  ASSERT_TRUE(spec.has_value()) << spec.error();
  EXPECT_EQ(spec->pipeline, sp::PipelineKind::kImage);
  ASSERT_EQ(spec->stages.size(), 3U);
  EXPECT_EQ(spec->stages[0].name, "text_encoder");
  EXPECT_EQ(spec->stages[1].name, "denoiser");
  EXPECT_EQ(spec->stages[2].name, "vae");
  for (std::uint32_t i = 0; i < 3; ++i) EXPECT_EQ(spec->stages[i].artifact, i);
  // Canonical: its JSON reads back to itself.
  const auto json = sp::ToJson(*spec);
  auto reread = sp::FromJson(json);
  ASSERT_TRUE(reread.has_value()) << reread.error();
  EXPECT_EQ(*reread, *spec);

  auto p = sp::ImageProfileOf(*spec, sp::mappings::QwenImage21Mapping());
  ASSERT_TRUE(p.has_value()) << p.error();
  EXPECT_EQ(p->text.layers, 2U);
  EXPECT_EQ(p->text.width, 64U);
  EXPECT_EQ(p->text.kv_heads, 2U);
  // A real is the float its shortest decimal names, as a literal would be.
  EXPECT_EQ(p->text.rms_eps, 1e-06F);
  EXPECT_EQ(p->text.rope_theta, 5000000.0F);
  EXPECT_EQ(p->denoiser.width, 64U);
  EXPECT_EQ(p->denoiser.mlp, 192U);
  EXPECT_EQ(p->denoiser.rope_axes, (std::array<std::uint32_t, 3>{8, 12, 12}));
  EXPECT_EQ(p->denoiser.eps, 1e-06F);
  // The source library's constants.
  EXPECT_EQ(p->denoiser.timestep_dim, 256U);
  EXPECT_EQ(p->denoiser.timestep_divisor, 1000.0F);
  EXPECT_EQ(p->denoiser.timestep_scale, 1000.0F);
  EXPECT_EQ(p->denoiser.timestep_period, 10000.0);
  EXPECT_EQ(p->denoiser.rope_theta, 10000.0F);
  EXPECT_EQ(p->scheduler.max_shift, 0.9);
  EXPECT_EQ(p->scheduler.shift_terminal, 0.02);
  EXPECT_EQ(p->vae.dim_mult, (std::vector<std::uint32_t>{1, 2, 4}));
  // The encoder's temporal downsampling, reversed: the decoder's upsampling.
  EXPECT_EQ(p->vae.temporal_upsample, (std::vector<bool>{true, false}));
  EXPECT_EQ(p->vae.latents_mean, (std::vector<float>{0.5F, -0.25F, 0.0F, 1.0F}));
  EXPECT_EQ(p->vae.spatial_factor, 4U);
  // The stages' widths are the family's.
  EXPECT_EQ(spec->stages[0].stream.width, 64U);
  EXPECT_EQ(spec->stages[1].stream.width, 64U);
  EXPECT_EQ(spec->stages[2].stream.width, 4U);
}

TEST(ImageFamily, SynthesisRefusesWhatTheMappingDoesNotAdmit) {
  {
    Toy toy;  // a requirement that does not hold
    Set(toy.scheduler, "time_shift_type", std::string("linear"));
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // a requirement absent
    Drop(toy.flow, "causal_condition");
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // a parameter absent
    Drop(toy.text, "text_config.head_dim");
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // out of the schema's range
    Set(toy.flow, "num_attention_heads", std::int64_t{0});
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // of another type
    Set(toy.text, "text_config.hidden_size", std::string("64"));
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // a list of another length
    Set(toy.flow, "axes_dims_rope", std::vector<std::int64_t>{8, 12});
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // latent statistics that are not the latent's
    Set(toy.vae, "latents_std", std::vector<double>{1.0, 2.0, 0.5});
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // an architecture the mapping does not accept
    toy.flow_arch = "FluxTransformer2DModel";
    EXPECT_FALSE(toy.Synthesize());
  }
  {
    Toy toy;  // a pipeline file absent
    const std::vector<sp::ImageComponentSource> components{
        {.role = "text_encoder", .architecture = toy.text_arch, .values = toy.text},
        {.role = "transformer", .architecture = toy.flow_arch, .values = toy.flow},
        {.role = "vae", .architecture = toy.vae_arch, .values = toy.vae}};
    EXPECT_FALSE(sp::SynthesizeImageFamily(sp::mappings::QwenImage21Mapping(), components, {}));
    // A component absent.
    const std::vector<sp::ImageSource> files{
        {.name = "scheduler/scheduler_config.json", .values = toy.scheduler}};
    EXPECT_FALSE(sp::SynthesizeImageFamily(sp::mappings::QwenImage21Mapping(),
                                           std::span(components).first(2), files));
  }
}

TEST(ImageFamily, AStoredSpecReadsBackOnlyAsTheSchemaAdmits) {
  const auto& mapping = sp::mappings::QwenImage21Mapping();
  const auto spec = Toy{}.Synthesize();
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const auto param = [](sp::ModelSpec& s, std::size_t stage, std::string_view name) {
    for (auto& p : s.stages[stage].layers[0].components[0].params)
      if (p.name == name) return &p;
    return static_cast<sp::Param*>(nullptr);
  };
  {
    auto s = *spec;  // a parameter out of range
    param(s, 1, "heads")->value = std::int64_t{0};
    EXPECT_FALSE(sp::ImageProfileOf(s, mapping));
  }
  {
    auto s = *spec;  // an integer stored as a real
    param(s, 0, "layers")->value = 2.0;
    EXPECT_FALSE(sp::ImageProfileOf(s, mapping));
  }
  {
    auto s = *spec;  // a parameter the schema does not have
    s.stages[0].layers[0].components[0].params.push_back({.name = "zz", .value = std::int64_t{1}});
    EXPECT_FALSE(sp::ImageProfileOf(s, mapping));
  }
  {
    auto s = *spec;  // a width that is not the family's
    s.stages[1].stream.width = 128;
    EXPECT_FALSE(sp::ImageProfileOf(s, mapping));
  }
  {
    auto s = *spec;  // a stage of another kind
    s.stages[2].layers[0].components[0].kind = std::string(sp::kImageTextKind);
    EXPECT_FALSE(sp::ImageProfileOf(s, mapping));
  }
  {
    auto s = *spec;  // a stage missing
    s.stages.pop_back();
    EXPECT_FALSE(sp::ImageProfileOf(s, mapping));
  }
}

TEST(ImageFamily, TheFrozenFamilyIsTheApprovedOne) {
  const auto& family = llmp::test_support::QwenImage();
  const auto& approvals = llmp::runtime::ImageApprovals();
  ASSERT_EQ(approvals.size(), 1U);
  EXPECT_EQ(approvals[0].architecture, family.mapping->architecture);
  EXPECT_EQ(llmp::base::ToHex(family.spec.Digest()), approvals[0].spec);
  // Its tensor lists are its components' approved inventories' sizes.
  ASSERT_EQ(approvals[0].components.size(), 3U);
  EXPECT_EQ(llmp::test_support::QwenImageTextTensors().size(), approvals[0].components[0].tensors);
  EXPECT_EQ(llmp::test_support::QwenImageDenoiserTensors().size(),
            approvals[0].components[1].tensors);
  EXPECT_EQ(llmp::test_support::QwenImageVaeTensors().size(), approvals[0].components[2].tensors);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& c = approvals[0].components[i];
    SCOPED_TRACE(c.name);
    EXPECT_TRUE(std::ranges::contains(family.mapping->components[i].architectures, c.architecture));
    EXPECT_EQ(c.policy, llmp::runtime::ApprovedPolicy::kNone);
    EXPECT_EQ(c.inventory.size(), 64U);
    EXPECT_EQ(c.spec.size(), 64U);
    EXPECT_FALSE(c.sources.empty());
  }
  EXPECT_FALSE(approvals[0].sources.empty());
}

// The published v1 composition and its components (a Spark's v1 store;
// skipped where the store does not hold them) are admitted, and the
// family they make is the frozen one.
TEST(ImageFamily, ThePublishedPipelineIsAdmitted) {
  namespace ts = llmp::test_support;
  const auto root = ts::QwenImagePublished(ts::QwenImageComposition());
  std::error_code error;
  if (!std::filesystem::is_directory(root, error))
    GTEST_SKIP() << "no published Qwen-Image composition at " << root;
  auto composition = llmp::artifact::OpenComposition(root, std::string(ts::QwenImageComposition()));
  ASSERT_TRUE(composition.has_value()) << composition.error().ToString();
  std::vector<llmp::artifact::Artifact> opened;
  opened.reserve(composition->components().size());
  for (const auto& c : composition->components()) {
    auto a = llmp::artifact::Artifact::Open(ts::QwenImagePublished(c.artifact));
    ASSERT_TRUE(a.has_value()) << c.role << ": " << a.error().ToString();
    opened.push_back(std::move(*a));
  }
  std::vector<const llmp::artifact::Artifact*> components;
  for (const auto& a : opened) components.push_back(&a);
  const std::array<std::string, 3> ids{ts::QwenImageText(), ts::QwenImageTransformer(),
                                       ts::QwenImageVae()};
  ASSERT_EQ(components.size(), ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) EXPECT_EQ(components[i]->id(), ids[i]);
  auto admitted = llmp::runtime::ReadImageAdmission(*composition, components);
  ASSERT_TRUE(admitted.has_value()) << admitted.error();
  EXPECT_EQ(admitted->family, ts::QwenImage().spec);
  EXPECT_EQ(admitted->mapping, ts::QwenImage().mapping);
}

}  // namespace
