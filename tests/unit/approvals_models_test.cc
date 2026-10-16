// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The approved checkpoints as data (runtime/approvals.h, D-109) against the
// published v1 store (LLMP_TEST_V1_STORE; by default the models directory's
// m36-v1-artifacts, whose absence fails rather than skips):
//
//   - every pinned spec digest is the one the retired compiled profiles'
//     values synthesize (old admission's oracle) and the stored one, and every
//     drafter pin the composition the profiles' values made, which the
//     drafter's kept metadata now gives;
//   - every artifact and drafter inventory is admitted, and no drafter pairs
//     with another approved target;
//   - every served family's default configuration (with and without its
//     drafter) resolves to settings its approval admits.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "config/node_config.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "model/qwen38.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "runtime/model_settings.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {
namespace rt = llmp::runtime;
namespace sp = llmp::spec;
namespace md = llmp::model;
using Values = std::expected<std::vector<sp::SourceValue>, std::string>;

std::string Store() {
  if (auto named = llmp::test_support::V1Store(); !named.empty()) return named;
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return std::string(home != nullptr ? home : "") + "/.local/share/llmp/m36-v1-artifacts";
}

// A target approval's values from its retired compiled profile, as the
// admission before approval-as-data synthesized its expected spec.
Values ProfileValues(const rt::Approval& a) {
  if (a.name == "gemma-2-2b-it-q8_0") return sp::mappings::Gemma2V0Values(md::Gemma2_2B());
  if (a.name == "gemma-3-4b-it-qat-q4_0") return sp::mappings::Gemma3V0Values(md::Gemma3_4BQat());
  if (a.name == "gemma-4-26b-a4b-it-ud-q4_k_m")
    return sp::mappings::Gemma4V0Values(md::Gemma4_26BA4B());
  if (a.name == "gemma-4-31b-it-ud-q4_k_xl") return sp::mappings::Gemma4V0Values(md::Gemma4_31B());
  if (a.name == "qwen2.5-0.5b-instruct-fp16")
    return sp::mappings::Qwen2V0Values(md::Qwen25Instruct05B());
  if (a.name.starts_with("qwen2.5-0.5b-instruct-exl3"))
    return sp::mappings::Qwen2V0Values(md::Qwen25Instruct05BExl3());
  if (a.name == "deepseek-v4-flash-0731-ud-q2_k_xl")
    return sp::mappings::Dsv4V0Values(md::Dsv4Flash());
  if (a.name == "qwen3.8-flash-next-ud-iq3_xxs") {
    const md::Qwen38PleHash hash{
        .multipliers = {23703573157769, 20109073645365, 8052911324071},
        .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                    300001275},
        .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                  20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}};
    return sp::mappings::Qwen38GgufValues(md::Qwen38Flash(), hash);
  }
  // Qwen3.8's native imports: the GGUF values with the importer's own keys
  // (MLX's affine n-gram rows of 160; ModelOpt's relaid-out experts, their
  // tensor-scale vectors and activation scales).
  if (a.name == "qwen3.8-flash-next-modelopt" || a.name == "qwen3.8-flash-next-mlx") {
    const bool modelopt = a.name == "qwen3.8-flash-next-modelopt";
    const md::Qwen38PleHash hash{
        .multipliers = {23703573157769, 20109073645365, 8052911324071},
        .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                    300001275},
        .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                  20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}};
    md::Qwen38Profile p = md::Qwen38Flash();
    if (!modelopt) p.ple_row = 160;
    auto values = sp::mappings::Qwen38GgufValues(p, hash);
    if (!values) return values;
    values->push_back({.key = "llmp.qwen4exp.indexer", .value = std::string("fused")});
    values->push_back({.key = "llmp.qwen4exp.experts_stacked", .value = modelopt});
    values->push_back({.key = "llmp.qwen4exp.expert_scales", .value = modelopt});
    if (modelopt)
      values->push_back({.key = "llmp.qwen4exp.expert_activation_scales", .value = true});
    return values;
  }
  return std::unexpected("no compiled profile for " + std::string(a.name));
}

// A drafter approval's values from its retired compiled profile.
Values DrafterProfileValues(const rt::Approval& a) {
  if (a.name == "mtp-gemma-4-26b-a4b-it")
    return sp::mappings::Gemma4AssistantV0Values(md::Gemma4Assistant26(), md::Gemma4_26BA4B());
  if (a.name == "mtp-gemma-4-31b-it")
    return sp::mappings::Gemma4AssistantV0Values(md::Gemma4Assistant31(), md::Gemma4_31B());
  if (a.name == "dspark-deepseek-v4-flash-0731-q8_0")
    return sp::mappings::DsparkV0Values(md::DsparkDeepSeekV4Flash());
  if (a.name == "qwen3.8-flash-next-modelopt-mtp-mia47172")
    return sp::mappings::Qwen38MtpNativeValues(md::Qwen38Flash(), 47172);
  return std::unexpected("no compiled profile for " + std::string(a.name));
}

// A model entry with its defaults: the artifact and drafter alone.
llmp::config::ModelEntry DefaultEntry(std::string id, std::string drafter = {}) {
  llmp::config::ModelEntry entry;
  entry.name = "model";
  entry.artifact = std::move(id);
  if (!drafter.empty()) entry.drafter = std::move(drafter);
  return entry;
}

TEST(ApprovalsModels, PublishedArtifactsAndPairsMatchTheirPinnedSpecs) {
  const auto store = Store();
  ASSERT_TRUE(std::filesystem::exists(store + "/v1-ids.json"))
      << "no published v1 store at " << store << " (LLMP_TEST_V1_STORE)";
  const auto names = llmp::test_support::ReadStringMap(store + "/v1-ids.json", "artifacts");
  ASSERT_FALSE(names.empty()) << store;
  std::size_t targets = 0, drafters = 0, served = 0;
  for (const auto& [v0, v1] : names) {
    SCOPED_TRACE(v1);
    auto artifact = rt::OpenInstalled(store, v1);
    ASSERT_TRUE(artifact) << artifact.error();
    const rt::Approval* approval =
        rt::FindApproval(artifact->model().architecture, artifact->sources());
    ASSERT_NE(approval, nullptr);
    if (approval->drafter) {
      EXPECT_TRUE(rt::CheckComponentArtifactAdmission(*artifact));
      const auto* mapping = rt::ComposedDrafterMapping(*artifact);
      ASSERT_NE(mapping, nullptr);
      auto source = sp::SourceOf(*artifact, mapping);
      ASSERT_TRUE(source) << source.error();
      auto inventory = rt::CheckComponentDrafterAdmission(*source, artifact->sources());
      EXPECT_TRUE(inventory) << (inventory ? "" : inventory.error());
      ++drafters;
      continue;
    }
    auto admitted = rt::ReadComponentAdmission(*artifact);
    if (!admitted) {
      ADD_FAILURE() << admitted.error();  // and on to the next artifact
      continue;
    }
    EXPECT_EQ(admitted->approval, approval);
    EXPECT_EQ(llmp::base::ToHex(admitted->model.Digest()), approval->spec);
    // The pin is what the retired compiled profile's values synthesize.
    auto values = ProfileValues(*approval);
    ASSERT_TRUE(values) << values.error();
    const sp::Mapping& mapping = approval->mapping != nullptr
                                     ? approval->mapping()
                                     : *sp::FindMapping(approval->architecture);
    auto expected = sp::Synthesize(mapping, admitted->source, *values);
    ASSERT_TRUE(expected) << expected.error();
    EXPECT_EQ(llmp::base::ToHex(expected->Digest()), approval->spec);
    ++targets;
    // Its default configuration starts within its approval.
    if (rt::RecipeOf(approval->architecture) != nullptr) {
      auto facts = rt::ArtifactFactsOf(*artifact, nullptr, "f16");
      ASSERT_TRUE(facts) << facts.error();
      auto settings = rt::ResolveSettings(DefaultEntry(v1), *facts, nullptr, false);
      ASSERT_TRUE(settings) << settings.error();
      auto policy = rt::ComponentAdmissionPolicy(
          admitted->model, admitted->source, artifact->sources(),
          {.context = settings->context.value, .slots = settings->max_slots.value});
      EXPECT_TRUE(policy) << (policy ? "" : policy.error()) << " at context "
                          << settings->context.value << " and " << settings->max_slots.value
                          << " owners";
      ++served;
    }
  }
  EXPECT_GE(targets, 1U);
  EXPECT_GE(drafters, 1U);
  EXPECT_GE(served, 1U);

  // The pairs: "compositions" holds {name: {target, drafter}}.
  const auto ids = llmp::test_support::ReadWhole(store + "/v1-ids.json");
  ASSERT_TRUE(ids);
  auto parsed = llmp::base::json::Parse(*ids);
  ASSERT_TRUE(parsed);
  const auto pairs = parsed->root().find("compositions");
  ASSERT_TRUE(pairs && pairs->is_object());
  ASSERT_GE(pairs->size(), 1U);
  for (std::size_t i = 0; i < pairs->size(); ++i) {
    SCOPED_TRACE(pairs->key(i));
    const auto pair = pairs->member(i);
    const auto target_id = pair.find("target");
    const auto drafter_id = pair.find("drafter");
    ASSERT_TRUE(target_id && drafter_id);
    auto target = rt::OpenInstalled(store, std::string(target_id->string()));
    auto drafter = rt::OpenInstalled(store, std::string(drafter_id->string()));
    ASSERT_TRUE(target && drafter);
    auto facts = rt::ArtifactFactsOf(*target, &*drafter, "f16");
    ASSERT_TRUE(facts) << facts.error();
    EXPECT_TRUE(facts->composed_spec.has_value()) << facts->composed_spec_error;
    EXPECT_TRUE(facts->composed_spec_error.empty()) << facts->composed_spec_error;
    // Its default configuration, speculating, starts within both approvals.
    auto settings = rt::ResolveSettings(
        DefaultEntry(std::string(target_id->string()), std::string(drafter_id->string())), *facts,
        nullptr, false);
    ASSERT_TRUE(settings) << settings.error();
    EXPECT_TRUE(settings->speculation.value);
    auto admitted = rt::ReadComponentAdmission(*target);
    ASSERT_TRUE(admitted) << admitted.error();
    const llmp::spec::Envelope envelope{.context = settings->context.value,
                                        .slots = settings->max_slots.value};
    auto policy = rt::ComponentAdmissionPolicy(admitted->model, admitted->source, target->sources(),
                                               envelope);
    EXPECT_TRUE(policy) << (policy ? "" : policy.error());
    auto drafter_envelope = rt::ComponentDrafterEnvelope(*drafter, envelope);
    EXPECT_TRUE(drafter_envelope) << (drafter_envelope ? "" : drafter_envelope.error());
    // The kept values compose what the compiled profiles' values composed.
    const sp::Mapping* draft_mapping = rt::ComposedDrafterMapping(*drafter);
    const rt::Approval* target_approval =
        rt::FindApproval(target->model().architecture, target->sources());
    ASSERT_NE(target_approval, nullptr);
    const sp::Mapping* target_mapping = target_approval->mapping != nullptr
                                            ? &target_approval->mapping()
                                            : sp::FindMapping(target->model().architecture);
    ASSERT_TRUE(draft_mapping && target_mapping);
    auto source = sp::SourceOf(*drafter, draft_mapping);
    ASSERT_TRUE(source) << source.error();
    const rt::Approval* approval =
        rt::FindApproval(drafter->model().architecture, drafter->sources());
    ASSERT_NE(approval, nullptr);
    auto profile_values = DrafterProfileValues(*approval);
    ASSERT_TRUE(profile_values) << profile_values.error();
    auto kept_values = rt::ComponentDrafterValues(*target, *drafter);
    ASSERT_TRUE(kept_values) << kept_values.error();
    auto from_profile = sp::ComposeDrafter(admitted->model, *target_mapping, *draft_mapping,
                                           *source, *profile_values);
    auto from_kept =
        sp::ComposeDrafter(admitted->model, *target_mapping, *draft_mapping, *source, *kept_values);
    ASSERT_TRUE(from_profile && from_kept);
    EXPECT_EQ(llmp::base::ToHex(from_kept->Digest()), llmp::base::ToHex(from_profile->Digest()));
    EXPECT_EQ(llmp::base::ToHex(from_kept->Digest()), approval->spec);
    // A drafter never pairs with another approved target.
    for (const auto& [v0, v1] : names) {
      if (v1 == target_id->string() || v1 == drafter_id->string()) continue;
      auto other = rt::OpenInstalled(store, v1);
      ASSERT_TRUE(other);
      const rt::Approval* a = rt::FindApproval(other->model().architecture, other->sources());
      if (a == nullptr || a->drafter) continue;
      EXPECT_FALSE(rt::ComponentDrafterValues(*other, *drafter)) << v1;
    }
  }
}

}  // namespace
