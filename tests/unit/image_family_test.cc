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
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "artifact_writer.h"
#include "base/sha256.h"
#include "image_family.h"
#include "image_published.h"
#include "image_recipes.h"
#include "import/image_pipeline.h"
#include "import/stage_recipe.h"
#include "ir/graph.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "spec/mappings/qwen_image.h"
#include "spec/spec.h"
#include "spec/stored.h"

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

sp::ModelSpec TinyImageComponent(sp::StageRole role) {
  sp::Stage stage{.name = "component", .role = role};
  stage.stream.width = 64;
  stage.layers.push_back({.components = {{.kind = "projection.spatial",
                                          .params = {{"in_channels", std::int64_t{64}},
                                                     {"out_channels", std::int64_t{64}},
                                                     {"silu", true}},
                                          .weights = {{.role = "weight", .tensor = "weight"}}}}});
  sp::ModelSpec model{.pipeline = sp::PipelineKind::kImage, .stages = {std::move(stage)}};
  sp::Canonicalize(model);
  return model;
}

struct SyntheticImage {
  std::filesystem::path root;
  std::vector<llmp::artifact::Artifact> opened;
  std::unique_ptr<llmp::artifact::Composition> composition;
  std::vector<const llmp::artifact::Artifact*> components;

  static std::expected<SyntheticImage, std::string> Make(
      const std::filesystem::path& root, Toy toy = {},
      std::string_view transformer_container = "gguf",
      std::optional<sp::StageRole> transformer_role = std::nullopt,
      std::vector<std::pair<std::string, std::vector<std::string>>> transformer_folds = {},
      std::string_view transformer_source_bytes = "source bytes", bool transformer_q4 = false,
      bool bad_transformer_fold = false) {
    auto family = toy.Synthesize();
    if (!family) return std::unexpected(family.error());
    SyntheticImage result{.root = root, .opened = {}, .composition = nullptr, .components = {}};
    const auto& mapping = sp::mappings::QwenImage21Mapping();
    std::array<std::string, 3> ids;
    for (std::size_t i = 0; i < mapping.components.size(); ++i) {
      const auto& rule = mapping.components[i];
      const std::string container = i == 1 ? std::string(transformer_container) : "safetensors";
      auto component =
          TinyImageComponent(i == 1 && transformer_role ? *transformer_role : rule.stage_role);
      if (i == 1 && transformer_q4) {
        component.stages.front().layers.clear();
        sp::Canonicalize(component);
      }
      if (i == 1 && bad_transformer_fold) component.stages.front().folds = {"unapplied-fold"};
      llmp::test::V1Plan plan;
      plan.architecture = std::string(rule.architectures.front());
      plan.container = container;
      plan.sources = {{std::format("synthetic-{}.{}", rule.role, container),
                       i == 1 ? std::string(transformer_source_bytes) : "source bytes"}};
      plan.spec = sp::ToJson(component);
      auto format = llmp::ir::Dense(llmp::format::Encoding::kF32, llmp::ir::ShapeOf({64, 64}));
      if (i == 1 && transformer_q4) {
        format = llmp::ir::Dense(llmp::format::Encoding::kQ4_K, llmp::ir::ShapeOf({256, 64}));
        format.layout = llmp::format::Layout::kGgmlBlocks;
      }
      plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                      .resources = {{.name = "weight", .format = format}}}};
      if (i == 1) plan.folds = std::move(transformer_folds);
      const auto artifact_path = llmp::test::Publish(root, llmp::test::Render(plan));
      auto artifact = llmp::artifact::Artifact::Open(artifact_path);
      if (!artifact) return std::unexpected(artifact.error().ToString());
      ids[i] = artifact->id();
      result.opened.push_back(std::move(*artifact));
    }
    llmp::test::CompositionPlan composition;
    composition.architecture = "QwenImage21Pipeline";
    composition.meta = {{"model_index.json", R"({"_class_name":"QwenImage21Pipeline"})"}};
    composition.spec = sp::ToJson(*family);
    for (std::size_t i = 0; i < mapping.components.size(); ++i)
      composition.components.emplace_back(mapping.components[i].role, ids[i],
                                          mapping.components[i].architectures.front());
    const auto composition_path = llmp::test::Publish(root, llmp::test::Render(composition));
    auto opened_composition = llmp::artifact::OpenComposition(composition_path);
    if (!opened_composition) return std::unexpected(opened_composition.error().ToString());
    result.composition =
        std::make_unique<llmp::artifact::Composition>(std::move(*opened_composition));
    for (const auto& artifact : result.opened) result.components.push_back(&artifact);
    return result;
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

TEST(ImageFamily, PipelineBoundFundsRetainedMetadataBeforeFamilyConstruction) {
  namespace im = llmp::import;
  const auto temp =
      std::filesystem::temp_directory_path() / std::format("llmp-image-bound-{}", ::getpid());
  std::error_code ignored;
  std::filesystem::remove_all(temp, ignored);
  std::filesystem::create_directories(temp, ignored);
  const std::string scheduler = R"({"num_train_timesteps":1000,"base_image_seq_len":256,
    "max_image_seq_len":8192,"base_shift":0.5,"max_shift":0.9,"shift_terminal":0.02,
    "time_shift_type":"exponential","use_dynamic_shifting":true,"invert_sigmas":false,
    "stochastic_sampling":false})";
  const auto scheduler_path = temp / "scheduler.json";
  {
    std::ofstream out(scheduler_path);
    out << scheduler;
  }
  im::Checkpoint source{.container = "diffusers", .architecture = "QwenImage21Pipeline"};
  source.files.push_back({.name = "scheduler/scheduler_config.json",
                          .path = scheduler_path,
                          .bytes = scheduler.size()});
  const Toy toy;
  auto family = toy.Synthesize();
  ASSERT_TRUE(family) << family.error();
  const auto& mapping = sp::mappings::QwenImage21Mapping();
  const auto profile = sp::ImageProfileOf(*family, mapping);
  ASSERT_TRUE(profile);
  const std::array values{toy.text, toy.flow, toy.vae};
  for (std::size_t i = 0; i < mapping.components.size(); ++i) {
    const auto& rule = mapping.components[i];
    im::Checkpoint component{.role = std::string(rule.role),
                             .container = "safetensors",
                             .architecture = std::string(rule.architectures.front())};
    for (const auto& entry : values[i])
      component.metadata.push_back(
          {.key = entry.key,
           .value = std::visit([](const auto& v) -> im::MetaValue { return v; }, entry.value)});
    std::ranges::sort(component.metadata, {}, &im::MetaEntry::key);
    std::vector<md::ImageTensor> tensors;
    if (i == 0) {
      tensors = md::TextEncoderTensors(profile->text, mapping.names.text);
    } else {
      const auto expansion = llmp::test_support::QwenImageStage(*family, *profile, rule.role);
      ASSERT_TRUE(expansion) << expansion.error();
      for (const auto& tensor : expansion->tensors)
        tensors.push_back(
            {.name = tensor.name, .dtype = {}, .shape = tensor.shape, .part = tensor.part});
    }
    component.files.push_back({.name = "model.safetensors"});
    for (const auto& tensor : tensors) {
      std::vector<std::uint64_t> dims(tensor.shape.rbegin(), tensor.shape.rend());
      const auto format =
          llmp::ir::Dense(llmp::format::Encoding::kF32, *llmp::format::ShapeOf(dims));
      const auto bytes = llmp::format::PartsOf(format)->Find(tensor.part)->bytes;
      component.tensors.push_back({.name = tensor.name,
                                   .format = format,
                                   .part = tensor.part,
                                   .extent = {.offset = component.files[0].bytes, .bytes = bytes},
                                   .dims = tensor.shape});
      component.files[0].bytes += bytes;
    }
    if (i == 2) {
      const auto header = llmp::test_support::QwenImageVaeSourceOf(profile->vae);
      const auto conv =
          std::ranges::find(component.tensors, "decoder.conv_in.weight", &im::Tensor::name);
      ASSERT_NE(conv, component.tensors.end());
      conv->dims = header.tensors[0].dims;
    }
    source.components.push_back(std::move(component));
  }
  // File identity must match the descriptor (the in-memory components need
  // no payload reads for planning).
  const auto opened = im::OpenCheckpointFile(scheduler_path, 0);
  ASSERT_TRUE(opened);
  source.files[0].device = opened->device;
  source.files[0].inode = opened->inode;
  const auto ordinary = im::ImagePipelineBound(source);
  ASSERT_TRUE(ordinary) << ordinary.error().ToString();
  ASSERT_TRUE(im::PlanImagePipeline(source, {ordinary->metadata_bytes, ordinary->work_units}));

  // The text inventory is derived from config, independent of how few
  // tensors a malformed source supplies. Refuse representation and budget
  // violations before allocating those derived descriptors.
  for (const auto layers : {300, 4096}) {
    auto sparse = source;
    sparse.components[0].tensors.clear();
    const auto entry = std::ranges::find(sparse.components[0].metadata,
                                         "text_config.num_hidden_layers", &im::MetaEntry::key);
    ASSERT_NE(entry, sparse.components[0].metadata.end());
    entry->value = std::int64_t{layers};
    const auto refused = im::PlanImagePipeline(sparse, {ordinary->metadata_bytes, UINT64_MAX});
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().rule, im::Rule::kBounds);
    EXPECT_EQ(refused.error().reason, layers == 300
                                          ? "image pipeline text spec exceeds normalization limits"
                                          : "cannot bound the text encoder's spec");
  }
  EXPECT_FALSE(sp::ImageTextSpecBound(md::ImageTextProfile{.layers = 4096}));
  EXPECT_EQ(sp::ImageTextSpecBound(profile->text),
            sp::ImageTextSpecBound(profile->text,
                                   md::TextEncoderTensors(profile->text, mapping.names.text)));

  constexpr std::size_t kPayload = 8U << 20U;
  const std::array<im::MetaValue, 4> metadata{
      std::string(kPayload, 'x'), std::vector<std::int64_t>(kPayload / 8, 7),
      std::vector<double>(kPayload / 8, 0.5), std::vector<std::string>{std::string(kPayload, 'x')}};
  for (const auto& value : metadata) {
    source.components[0].metadata.push_back({.key = "_name_or_path", .value = value});
    std::ranges::sort(source.components[0].metadata, {}, &im::MetaEntry::key);
    const auto cost = im::ImagePipelineBound(source);
    ASSERT_TRUE(cost) << cost.error().ToString();
    EXPECT_GE(cost->metadata_bytes, ordinary->metadata_bytes + kPayload);
    const auto refused = im::PlanImagePipeline(source, {ordinary->metadata_bytes, UINT64_MAX});
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().rule, im::Rule::kBounds);
    const auto planned = im::PlanImagePipeline(source, {cost->metadata_bytes, cost->work_units});
    ASSERT_TRUE(planned) << planned.error().ToString();
    const auto* retained = planned->components[0].checkpoint.FindValue("_name_or_path");
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(*retained, value);
    ASSERT_EQ(planned->files.size(), 1U);
    EXPECT_EQ(planned->files[0].parsed.bytes, scheduler.size());
    EXPECT_EQ(planned->files[0].parsed.sha256, llmp::base::Sha256().Update(scheduler).Finish());
    std::erase_if(source.components[0].metadata,
                  [](const auto& entry) { return entry.key == "_name_or_path"; });
  }
  // Dotted-key flattening repeats a shared object prefix for every leaf.
  // Its output can be much larger than both input and JSON parse tables.
  std::string prefixed = scheduler.substr(0, scheduler.rfind('}'));
  prefixed += ",\"" + std::string(512, 'p') + "\":{";
  for (unsigned i = 0; i < 1024; ++i) {
    if (i != 0) prefixed += ',';
    prefixed += std::format("\"k{}\":1", i);
  }
  prefixed += "}}";
  {
    std::ofstream out(scheduler_path);
    out << prefixed;
  }
  source.files[0].bytes = prefixed.size();
  const auto flattened = im::ImagePipelineBound(source);
  ASSERT_TRUE(flattened) << flattened.error().ToString();
  EXPECT_GT(flattened->metadata_bytes, ordinary->metadata_bytes + 1024U * 512U);
  EXPECT_FALSE(im::PlanImagePipeline(source, {ordinary->metadata_bytes, UINT64_MAX}));
  const auto large_file_plan =
      im::PlanImagePipeline(source, {flattened->metadata_bytes, flattened->work_units});
  ASSERT_TRUE(large_file_plan) << large_file_plan.error().ToString();
  EXPECT_EQ(large_file_plan->files[0].parsed.sha256,
            llmp::base::Sha256().Update(prefixed).Finish());
  std::filesystem::remove_all(temp, ignored);
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

// A stage's digest and its tensor inventory's (each tensor's name, its
// role-derived shape, its group, layer and part).
std::pair<std::string, std::string> StageDigests(const llmp::spec::RecipeExpansion& expanded) {
  std::string inventory;
  for (const auto& t : expanded.tensors) {
    inventory += t.name;
    for (const auto d : t.shape) inventory += std::format(",{}", d);
    inventory +=
        std::format(";{}:{}:{}|", static_cast<int>(t.group), t.layer, static_cast<int>(t.part));
  }
  sp::ModelSpec stored{.pipeline = sp::PipelineKind::kImage, .stages = {expanded.stage}};
  sp::Canonicalize(stored);
  return {llmp::base::ToHex(stored.Digest()),
          llmp::base::ToHex(llmp::base::Sha256().Update(inventory).Finish())};
}

// The components' recipes over the frozen family (the published
// composition's) are the published components' stored specs, byte for byte:
// the digests their approvals pin. A recipe change that would change what a
// re-import stores fails here, before any model file is read.
TEST(ImageFamily, RecipesExpandToTheApprovedComponentSpecs) {
  const auto& family = llmp::test_support::QwenImage();
  const auto& approvals = llmp::runtime::ImageApprovals();
  ASSERT_EQ(approvals.size(), 1U);
  for (const std::size_t i : {1U, 2U}) {
    const auto& rule = family.mapping->components[i];
    SCOPED_TRACE(rule.role);
    auto expanded = llmp::test_support::QwenImageStage(family.spec, family.profile, rule.role);
    ASSERT_TRUE(expanded) << expanded.error();
    EXPECT_EQ(StageDigests(*expanded).first, approvals[0].components[i].spec);
  }
}

// A small family's stages, frozen before the mappings' value languages were
// unified (2026-10-26): the flow transformer and the VAE decoder, their
// tensor inventories too.
TEST(ImageFamily, ToyRecipesKeepTheirFrozenExpansions) {
  Toy toy;
  auto family = toy.Synthesize();
  ASSERT_TRUE(family) << family.error();
  EXPECT_EQ(llmp::base::ToHex(family->Digest()),
            "1dc3de6cdd598c87f599d13c4ef3b3712402a551ef4016d42d836e249c619753");
  const auto& mapping = sp::mappings::QwenImage21Mapping();
  const auto flow =
      std::ranges::find(mapping.components, "transformer", &sp::ImageComponentRule::role);
  ASSERT_NE(flow, mapping.components.end());
  EXPECT_TRUE(std::ranges::contains(flow->containers, "gguf"));
  const auto text =
      std::ranges::find(mapping.components, "text_encoder", &sp::ImageComponentRule::role);
  ASSERT_NE(text, mapping.components.end());
  EXPECT_FALSE(std::ranges::contains(text->containers, "gguf"));
  auto profile = sp::ImageProfileOf(*family, mapping);
  ASSERT_TRUE(profile) << profile.error();
  auto flow_stage = llmp::test_support::QwenImageStage(*family, *profile, "transformer");
  ASSERT_TRUE(flow_stage) << flow_stage.error();
  EXPECT_EQ(
      StageDigests(*flow_stage),
      std::pair(std::string("d2613a01e1d87d98890da4961043e95b8925943a09cfc5133f0e22dd74c6b46b"),
                std::string("5559396b2b82190e70b58db45c49afac43128c8f6f9117bad38253e60c3e1b00")));
  auto vae_stage = llmp::test_support::QwenImageStage(*family, *profile, "vae");
  ASSERT_TRUE(vae_stage) << vae_stage.error();
  EXPECT_EQ(
      StageDigests(*vae_stage),
      std::pair(std::string("861590f6c183f8c4e0c7926bf14b0f00a3b3c96768c7f9830e14e459d28fb79c"),
                std::string("0d9aabb6d3ec5e6d98a3267a48a677956380783f7f9e04f8c723c7d97a5096e5")));
  // The same stages from a family made of the profile alone (the tests'
  // graph source, image_recipes.h).
  auto from_profile = llmp::test_support::QwenImageStage(
      llmp::test_support::QwenImageFamilyOf(*profile), *profile, "vae");
  ASSERT_TRUE(from_profile) << from_profile.error();
  EXPECT_EQ(StageDigests(*from_profile), StageDigests(*vae_stage));
  from_profile = llmp::test_support::QwenImageStage(llmp::test_support::QwenImageFamilyOf(*profile),
                                                    *profile, "transformer");
  ASSERT_TRUE(from_profile) << from_profile.error();
  EXPECT_EQ(StageDigests(*from_profile), StageDigests(*flow_stage));
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
  auto no_inputs = good;
  no_inputs.host_inputs = {};
  EXPECT_FALSE(sp::ValidateImageMapping(no_inputs));
  std::vector<sp::ImageHostInput> inputs(good.host_inputs.begin(), good.host_inputs.end());
  ASSERT_GE(inputs.size(), 2U);
  auto duplicate_input = good;
  duplicate_input.host_inputs = inputs;
  inputs[1] = inputs[0];
  EXPECT_FALSE(sp::ValidateImageMapping(duplicate_input));
  inputs.assign(good.host_inputs.begin(), good.host_inputs.end());
  inputs[0].value = static_cast<sp::ImageInputValue>(255);
  EXPECT_FALSE(sp::ValidateImageMapping(duplicate_input));
  inputs.assign(good.host_inputs.begin(), good.host_inputs.end());
  inputs[0].stage = static_cast<std::uint16_t>(good.components.size());
  EXPECT_FALSE(sp::ValidateImageMapping(duplicate_input));
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

TEST(ImageFamily, RuntimeAdmissionValidatesSyntheticClosureBeforeHistoryLookup) {
  const auto temp =
      std::filesystem::temp_directory_path() / std::format("llmp-image-admission-{}", ::getpid());
  std::error_code ignored;
  std::filesystem::remove_all(temp, ignored);
  auto valid = SyntheticImage::Make(temp / "valid");
  ASSERT_TRUE(valid) << (valid ? "" : valid.error());
  const auto admitted = llmp::runtime::ReadImageAdmission(*valid->composition, valid->components);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_EQ(admitted->approval, nullptr);

  // The source receipts at the outer boundary are unchanged while the
  // transformer's authentic GGUF child and ID change. Complete closure
  // admission still succeeds and does not borrow the historical BF16 row.
  auto q4 = SyntheticImage::Make(temp / "q4", {}, "gguf", std::nullopt, {},
                                 "different authentic transformer bytes", true);
  ASSERT_TRUE(q4) << (q4 ? "" : q4.error());
  auto q4_source = sp::SourceOf(*q4->components[1]);
  ASSERT_TRUE(q4_source) << (q4_source ? "" : q4_source.error());
  ASSERT_EQ(q4_source->container, "gguf");
  ASSERT_EQ(q4_source->tensors.size(), 1U);
  EXPECT_EQ(q4_source->tensors[0].format.encoding, llmp::format::Encoding::kQ4_K);
  ASSERT_EQ(valid->composition->sources().size(), q4->composition->sources().size());
  for (std::size_t i = 0; i < valid->composition->sources().size(); ++i) {
    EXPECT_EQ(valid->composition->sources()[i].name, q4->composition->sources()[i].name);
    EXPECT_EQ(valid->composition->sources()[i].bytes, q4->composition->sources()[i].bytes);
    EXPECT_EQ(valid->composition->sources()[i].sha256, q4->composition->sources()[i].sha256);
  }
  EXPECT_NE(valid->components[1]->id(), q4->components[1]->id());
  auto q4_admitted = llmp::runtime::ReadImageAdmission(*q4->composition, q4->components);
  ASSERT_TRUE(q4_admitted) << (q4_admitted ? "" : q4_admitted.error());
  EXPECT_EQ(q4_admitted->approval, nullptr);

  const auto publish_variant = [&](llmp::test::CompositionPlan plan) {
    const auto path = llmp::test::Publish(temp / "valid", llmp::test::Render(plan));
    return llmp::artifact::OpenComposition(path);
  };
  llmp::test::CompositionPlan plan;
  plan.architecture = valid->composition->architecture();
  plan.meta = {
      {"model_index.json", std::string(*valid->composition->Metadata("model_index.json"))}};
  plan.spec = std::string(*valid->composition->spec_json());
  for (const auto& component : valid->composition->components())
    plan.components.emplace_back(component.role, component.artifact, component.architecture);

  auto wrong_role_plan = plan;
  std::get<0>(wrong_role_plan.components[0]) = "transf0rmer";
  auto wrong_role = publish_variant(std::move(wrong_role_plan));
  ASSERT_TRUE(wrong_role) << wrong_role.error().ToString();
  EXPECT_FALSE(llmp::runtime::ReadImageAdmission(*wrong_role, valid->components));

  auto wrong_order_plan = plan;
  std::swap(std::get<1>(wrong_order_plan.components[0]),
            std::get<1>(wrong_order_plan.components[1]));
  std::swap(std::get<2>(wrong_order_plan.components[0]),
            std::get<2>(wrong_order_plan.components[1]));
  auto wrong_order = publish_variant(std::move(wrong_order_plan));
  ASSERT_TRUE(wrong_order) << wrong_order.error().ToString();
  auto reversed = valid->components;
  std::swap(reversed[0], reversed[1]);
  EXPECT_FALSE(llmp::runtime::ReadImageAdmission(*wrong_order, reversed));

  auto wrong_id_plan = plan;
  std::get<1>(wrong_id_plan.components[1]) = std::string(64, '0');
  auto wrong_id = publish_variant(std::move(wrong_id_plan));
  ASSERT_TRUE(wrong_id) << wrong_id.error().ToString();
  EXPECT_FALSE(llmp::runtime::ReadImageAdmission(*wrong_id, valid->components));

  auto null_child = valid->components;
  null_child[1] = nullptr;
  EXPECT_FALSE(llmp::runtime::ReadImageAdmission(*valid->composition, null_child));

  auto unsupported = SyntheticImage::Make(temp / "unsupported", {}, "mlx");
  ASSERT_TRUE(unsupported) << (unsupported ? "" : unsupported.error());
  EXPECT_FALSE(
      llmp::runtime::ReadImageAdmission(*unsupported->composition, unsupported->components));

  auto bad_role = SyntheticImage::Make(temp / "bad-role", {}, "gguf", sp::StageRole::kVae);
  ASSERT_TRUE(bad_role) << (bad_role ? "" : bad_role.error());
  EXPECT_FALSE(llmp::runtime::ReadImageAdmission(*bad_role->composition, bad_role->components));

  auto bad_fold = SyntheticImage::Make(temp / "bad-fold", {}, "gguf", std::nullopt, {},
                                       "source bytes", false, true);
  ASSERT_TRUE(bad_fold) << (bad_fold ? "" : bad_fold.error());
  EXPECT_FALSE(llmp::runtime::ReadImageAdmission(*bad_fold->composition, bad_fold->components));
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
  ASSERT_NE(admitted->approval, nullptr);
  EXPECT_EQ(admitted->family, ts::QwenImage().spec);
  EXPECT_EQ(admitted->mapping, ts::QwenImage().mapping);

  // The historical source closure still matches, but a different valid
  // stored family is authoritative and therefore loses the old overlay.
  auto changed_family = admitted->family;
  auto& text_params = changed_family.stages[0].layers[0].components[0].params;
  const auto layers = std::ranges::find(text_params, std::string_view("layers"), &sp::Param::name);
  ASSERT_NE(layers, text_params.end());
  const auto old_layers = std::get<std::int64_t>(layers->value);
  layers->value = std::int64_t{old_layers > 1 ? old_layers - 1 : old_layers + 1};
  sp::Canonicalize(changed_family);
  ASSERT_NE(changed_family.Digest(), admitted->family.Digest());

  llmp::test::CompositionPlan altered;
  altered.architecture = composition->architecture();
  for (const auto& source : composition->sources()) {
    const auto bytes = composition->Metadata(source.name);
    ASSERT_TRUE(bytes.has_value()) << source.name;
    altered.meta.emplace_back(source.name, std::string(*bytes));
  }
  altered.spec = sp::ToJson(changed_family);
  for (const auto& component : composition->components())
    altered.components.emplace_back(component.role, component.artifact, component.architecture);
  const auto altered_root = std::filesystem::temp_directory_path() /
                            std::format("llmp-image-family-overlay-{}", ::getpid());
  auto altered_files = llmp::test::Render(altered);
  llmp::test::PublishHooks version_two;
  version_two.manifest = [](std::string& manifest) {
    constexpr std::string_view v1 = "\"format_version\":1";
    const auto position = manifest.find(v1);
    if (position != std::string::npos)
      manifest.replace(position, v1.size(), "\"format_version\":2");
  };
  const auto altered_path = llmp::test::Publish(altered_root, altered_files, version_two);
  for (const auto& [name, bytes] : altered.meta) {
    const auto path = altered_path / "meta" / name;
    std::error_code metadata_error;
    std::filesystem::create_directories(path.parent_path(), metadata_error);
    ASSERT_FALSE(metadata_error) << metadata_error.message();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file) << path;
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(file) << path;
  }
  auto altered_composition = llmp::artifact::OpenComposition(altered_path);
  ASSERT_TRUE(altered_composition) << altered_composition.error().ToString();
  ASSERT_EQ(altered_composition->format_version(), 2U);
  ASSERT_EQ(altered_composition->sources().size(), composition->sources().size());
  for (std::size_t i = 0; i < composition->sources().size(); ++i) {
    EXPECT_EQ(altered_composition->sources()[i].name, composition->sources()[i].name);
    EXPECT_EQ(altered_composition->sources()[i].bytes, composition->sources()[i].bytes);
    EXPECT_EQ(altered_composition->sources()[i].sha256, composition->sources()[i].sha256);
  }
  const auto* historical = llmp::runtime::FindImageApproval(
      altered_composition->architecture(), altered_composition->sources(), components,
      altered_composition->components());
  ASSERT_NE(historical, nullptr);
  auto altered_admission = llmp::runtime::ReadImageAdmission(*altered_composition, components);
  ASSERT_TRUE(altered_admission) << altered_admission.error();
  EXPECT_EQ(altered_admission->family, changed_family);
  EXPECT_EQ(altered_admission->approval, nullptr);
}

}  // namespace
