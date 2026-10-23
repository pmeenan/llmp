// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The approved checkpoints as data (runtime/approvals.h, D-109) against the
// published v1 store (LLMP_TEST_V1_STORE; by default the models directory's
// `artifacts`, falling back to `m36-v1-artifacts`, whose absence fails: a
// models run needs it):
//
//   - every pinned spec digest is the stored one, and every drafter pin the
//     composition the drafter's kept metadata gives;
//   - every artifact and drafter inventory is admitted; other targets
//     pair only when the declared architecture and actual tensor geometry agree;
//   - every image pipeline's composition is admitted with its components
//     (their sources, inventories and specs, its family spec the pinned one);
//   - every served family's default configuration (with and without its
//     drafter) resolves to settings its approval admits.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/composition.h"
#include "base/json.h"
#include "base/sha256.h"
#include "components/step.h"
#include "config/node_config.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "runtime/model_settings.h"
#include "spec/mapping.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {
namespace rt = llmp::runtime;
namespace sp = llmp::spec;

// A model entry with its defaults: the artifact and drafter alone.
llmp::config::ModelEntry DefaultEntry(std::string id, std::string drafter = {}) {
  llmp::config::ModelEntry entry;
  entry.name = "model";
  entry.artifact = std::move(id);
  if (!drafter.empty()) entry.drafter = std::move(drafter);
  return entry;
}

TEST(ApprovalsModels, PublishedArtifactsAndPairsMatchTheirPinnedSpecs) {
  const auto store = llmp::test_support::V1Store();
  // The IDs and pairs are the pinned in-tree map's; only the artifacts come
  // from the store.
  const auto pinned = llmp::test_support::V1IdsPath();
  const auto names = llmp::test_support::ReadStringMap(pinned, "artifacts");
  ASSERT_FALSE(names.empty()) << "no pinned v1 IDs at " << pinned;
  ASSERT_TRUE(std::filesystem::is_directory(store))
      << "no published v1 store at " << store << " (LLMP_TEST_V1_STORE)";
  std::size_t targets = 0, drafters = 0, served = 0, pipelines = 0, image_components = 0;
  for (const auto& [v0, v1] : names) {
    SCOPED_TRACE(v1);
    // An image pipeline's composition: it and its components admitted
    // together, its stored family the pinned one.
    if (auto composition = llmp::artifact::OpenComposition(std::filesystem::path(store) / v1, v1)) {
      std::vector<llmp::artifact::Artifact> opened;
      for (const auto& c : composition->components()) {
        auto a = rt::OpenInstalled(store, c.artifact);
        ASSERT_TRUE(a) << a.error();
        opened.push_back(std::move(*a));
      }
      std::vector<const llmp::artifact::Artifact*> components;
      for (const auto& a : opened) components.push_back(&a);
      auto admitted = rt::ReadImageAdmission(*composition, components);
      ASSERT_TRUE(admitted) << admitted.error();
      ASSERT_NE(admitted->approval, nullptr);
      EXPECT_EQ(admitted->approval, &rt::ImageApprovals().front());
      EXPECT_EQ(llmp::base::ToHex(admitted->family.Digest()), admitted->approval->spec);
      ++pipelines;
      continue;
    }
    auto artifact = rt::OpenInstalled(store, v1);
    ASSERT_TRUE(artifact) << artifact.error();
    const rt::Approval* approval =
        rt::FindApproval(artifact->model().architecture, artifact->sources());
    if (approval == nullptr) {
      // An image pipeline's component, admitted with its composition above.
      const bool component = std::ranges::any_of(rt::ImageApprovals(), [&](const auto& p) {
        return std::ranges::any_of(p.components, [&](const rt::Approval& c) {
          return c.architecture == artifact->model().architecture &&
                 rt::SourcesMatch(c.sources, artifact->sources());
        });
      });
      EXPECT_TRUE(component) << "no approval names it";
      image_components += component ? 1 : 0;
      continue;
    }
    if (approval->drafter) {
      EXPECT_TRUE(rt::CheckComponentArtifactAdmission(*artifact));
      auto stored = sp::StoredSpec(*artifact);
      ASSERT_TRUE(stored) << stored.error();
      auto source = sp::SourceOf(*artifact);
      ASSERT_TRUE(source) << source.error();
      auto inventory = rt::CheckComponentDrafterAdmission(*stored, *source, artifact->sources());
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
  EXPECT_GE(pipelines, 1U);
  EXPECT_EQ(image_components, 3 * pipelines);

  // The pairs: "compositions" holds {name: {target, drafter}}.
  const auto ids = llmp::test_support::ReadWhole(pinned);
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
    auto drafter_envelope = rt::ComponentDrafterEnvelope(envelope);
    EXPECT_TRUE(drafter_envelope) << (drafter_envelope ? "" : drafter_envelope.error());
    auto stored = sp::StoredSpec(*drafter);
    ASSERT_TRUE(stored) << stored.error();
    auto source = sp::SourceOf(*drafter);
    ASSERT_TRUE(source) << source.error();
    const rt::Approval* approval =
        rt::FindApproval(drafter->model().architecture, drafter->sources());
    ASSERT_NE(approval, nullptr);
    auto from_stored = sp::ComposeStoredDrafter(admitted->model, *stored);
    ASSERT_TRUE(from_stored) << from_stored.error();
    EXPECT_EQ(llmp::base::ToHex(from_stored->Digest()), approval->spec) << sp::ToJson(*from_stored);
    // Pair names qualify a measured composition; architecture and tensor
    // geometry decide whether another target can compose the same drafter.
    for (const auto& [v0, v1] : names) {
      SCOPED_TRACE(v1);
      if (v1 == target_id->string() || v1 == drafter_id->string()) continue;
      // An image pipeline's composition is no target.
      if (llmp::artifact::OpenComposition(std::filesystem::path(store) / v1, v1)) continue;
      auto other = rt::OpenInstalled(store, v1);
      ASSERT_TRUE(other);
      const rt::Approval* a = rt::FindApproval(other->model().architecture, other->sources());
      if (a == nullptr || a->drafter) continue;
      auto other_model = sp::StoredSpec(*other);
      auto other_source = sp::SourceOf(*other);
      ASSERT_TRUE(other_model && other_source);
      auto composed = sp::ComposeStoredDrafter(*other_model, *stored);
      if (other_model->stages.front().stream.width != admitted->model.stages.front().stream.width) {
        if (composed) {
          const std::array<std::span<const sp::SourceTensor>, 2> tables{other_source->tensors,
                                                                        source->tensors};
          EXPECT_FALSE(llmp::components::ValidateSpec(*composed, tables))
              << "different target widths must not silently compose: " << v1;
        }
        continue;
      }
      // Qwen's converted and native targets have the same semantic geometry.
      // They exercise the actual stored target + existing native MTP artifact.
      if (other->model().architecture == "qwen4exp") {
        ASSERT_TRUE(composed) << composed.error();
        const std::array<std::span<const sp::SourceTensor>, 2> tables{other_source->tensors,
                                                                      source->tensors};
        const auto validated = llmp::components::ValidateSpec(*composed, tables);
        EXPECT_TRUE(validated) << (validated ? "" : validated.error());
        const auto admission = rt::CheckComposedAdmission(*composed, *other, *drafter);
        EXPECT_TRUE(admission) << (admission ? "" : admission.error());
      }
    }
  }
}

}  // namespace
