// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 mappings on the published v1 artifacts (label `models`;
// skipped when the store or the target is absent): DeepSeek V4 Flash 0731's
// UD-Q2_K_XL (d2046349) and the DSpark drafter (6b1b9dca) composed with it.
// The target's stored spec is validated against its tensors and equals the
// spec of the unit tests' fixture (dsv4_fixture.h, synthesized with the
// family's values, tests/support/families/dsv4.h), so the unit tests'
// structure and layout checks hold for the real artifact; the drafter
// composes (ComposeDrafter) from the values it keeps, as serving composes
// it, equal to the fixture's composition; the stage layouts are the M3
// runners' state (Dsv4State in both window modes, DsparkState); and each
// step kind builds: the target's prefill with its drafter's injection, a
// state-only chunk, joined greedy decodes, a verify, and the drafter's
// blocks. (The community IQ2_XXS import, a4fb10cc…, has its own frozen
// serving screen in dsv4_engine_target_gpu_test.)
//
// These build the steps, they do not run them (the reference executor
// binds whole weights: tens of GB a target); DeepSeek V4's numerical gate
// is the lowering's on the GPU against today's runner (slice G4b).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "components/step.h"
#include "dsv4_fixture.h"
#include "families/dsv4.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "state_layouts.h"
#include "v1_frozen.h"

namespace {

namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace fm = llmp::test_support::families;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace rt = llmp::runtime;
namespace sp = llmp::spec;
namespace fx = llmp::test_support::dsv4;
namespace fs = std::filesystem;

struct Opened {
  ar::Artifact artifact;
  sp::Source source;
};

// The v1 artifact the store names `name`, with its source; nullopt when
// absent.
std::optional<Opened> Open(std::string_view name, const sp::Mapping& mapping) {
  const auto id = llmp::test_support::V1IdNamed(name);
  if (!id) return std::nullopt;
  auto artifact =
      ar::Artifact::Open(fs::path(llmp::test_support::V1Store()) / *id, {.expected_id = *id});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return std::nullopt;
  EXPECT_EQ(artifact->format_version(), 1U);
  EXPECT_EQ(sp::FindMapping(artifact->model().architecture), &mapping);
  // As the runtime reads it (spec/stored.h).
  auto source = sp::SourceOf(*artifact);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  if (!source) return std::nullopt;
  return Opened{.artifact = std::move(*artifact), .source = std::move(*source)};
}

std::size_t Count(const llmp::ir::Graph& g, llmp::ir::Op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes(), [op](const llmp::ir::Node& n) { return n.key.op == op; }));
}

pl::Segment Seg(std::uint8_t slot, std::uint32_t rows, std::uint32_t past, std::uint32_t read,
                std::uint32_t outputs, bool greedy = false) {
  return {.slot = slot,
          .rows = rows,
          .past = past,
          .read_cells = read,
          .outputs = outputs,
          .greedy = greedy};
}

TEST(ComponentsDsv4Models, EveryInstalledTargetAndItsDrafterMakeTheirSpecs) {
  const fm::Dsv4Profile& p = fm::Dsv4Flash();
  const fm::DsparkProfile& d = fm::DsparkDeepSeekV4Flash();
  const auto values = fm::Dsv4Values(p);
  const auto drafter_values = fm::DsparkValues(d);
  ASSERT_TRUE(values.has_value() && drafter_values.has_value());
  const auto& mapping = sp::mappings::Dsv4Mapping();
  const auto& drafter_mapping = sp::mappings::DsparkMapping();
  auto target = Open("deepseek", mapping);
  if (!target) {
    GTEST_FAIL() << "no DeepSeek V4 v1 artifact in " << llmp::test_support::V1Store();
  }
  auto drafter = Open("dspark", drafter_mapping);
  auto spec = sp::StoredSpec(target->artifact);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  ASSERT_TRUE(co::ValidateSpec(*spec, target->source.tensors));
  EXPECT_TRUE(spec->stages.front().folds.empty());
  // The unit tests' fixture is this artifact's tensors.
  const sp::Source mine = fx::SourceOf("deepseek4", fx::Resources(p), mapping);
  auto fixture = sp::Synthesize(mapping, mine, *values);
  ASSERT_TRUE(fixture.has_value()) << fixture.error();
  EXPECT_EQ(sp::ToJson(*spec), sp::ToJson(*fixture));
  EXPECT_EQ(target->source.tensors.size(), mine.tensors.size());
  // The M3 runner's state, in both window modes.
  for (const auto window : {fm::Dsv4Window::kRing, fm::Dsv4Window::kFull}) {
    sp::ModelSpec form = *spec;
    if (window == fm::Dsv4Window::kFull) {
      for (sp::Layer& l : form.stages[0].layers) {
        for (sp::ComponentRef& c : l.components) {
          for (sp::Param& q : c.params) {
            if (c.kind == "attention.mqa" && q.name == "cache") q.value = std::string("full");
          }
        }
      }
    }
    const auto layout = co::StageLayoutUniform(form, 0, 65536, 512, 256);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    const auto want = fm::Dsv4State(p, 65536, 512, window);
    ASSERT_TRUE(want.has_value());
    ASSERT_EQ(layout->items.size(), want->tensors.size());
    for (std::size_t i = 0; i < want->tensors.size(); ++i) {
      EXPECT_EQ(layout->items[i].offset, want->tensors[i].offset) << i;
    }
    EXPECT_EQ(layout->bytes, want->bytes);
  }
  if (!drafter) {
    GTEST_FAIL() << "no DSpark v1 artifact in " << llmp::test_support::V1Store();
  }
  // Its drafter composed as serving composes it: by the values it keeps.
  const sp::Mapping* composed_mapping = rt::ComposedDrafterMapping(drafter->artifact);
  ASSERT_EQ(composed_mapping, &drafter_mapping);
  auto kept = rt::ComponentDrafterValues(target->artifact, drafter->artifact);
  ASSERT_TRUE(kept.has_value()) << kept.error();
  auto composed = sp::ComposeDrafter(*spec, mapping, drafter_mapping, drafter->source, *kept);
  ASSERT_TRUE(composed.has_value()) << composed.error();
  const std::array<std::span<const co::BoundTensor>, 2> artifacts{target->source.tensors,
                                                                  drafter->source.tensors};
  ASSERT_TRUE(co::ValidateSpec(*composed, artifacts));
  // The fixture's composition from the family's values.
  const sp::Source drafted = fx::SourceOf("dflash", fx::DrafterResources(d), drafter_mapping);
  EXPECT_EQ(drafter->source.tensors.size(), drafted.tensors.size());
  auto fixture_composed =
      sp::ComposeDrafter(*fixture, mapping, drafter_mapping, drafted, *drafter_values);
  ASSERT_TRUE(fixture_composed.has_value()) << fixture_composed.error();
  EXPECT_EQ(sp::ToJson(*composed), sp::ToJson(*fixture_composed));
  const auto t_layout = co::StageLayoutUniform(*composed, 0, 65536, 512, 256);
  const auto d_layout = co::StageLayoutUniform(*composed, 1, 65536, 16, 256);
  ASSERT_TRUE(t_layout.has_value() && d_layout.has_value());
  const auto ring = fm::DsparkState(d, 16);
  ASSERT_TRUE(ring.has_value());
  EXPECT_EQ(d_layout->bytes, ring->bytes);
  const std::array<const md::StateLayoutDerived*, 2> layouts{&*t_layout, &*d_layout};
  const co::Bindings t_bindings{.tensors = target->source.tensors,
                                .state = &*t_layout,
                                .stages = layouts,
                                .artifacts = artifacts};
  const co::Bindings d_bindings{.tensors = drafter->source.tensors,
                                .state = &*d_layout,
                                .stages = layouts,
                                .artifacts = artifacts};
  const std::uint32_t read = static_cast<std::uint32_t>(t_layout->items.front().cells);
  // Every step kind builds.
  const auto prefill =
      co::BuildStep(*composed, 0, {.segments = {Seg(0, 512, 0, read, 1)}}, t_bindings);
  ASSERT_TRUE(prefill.has_value()) << prefill.error();
  EXPECT_EQ(Count(prefill->graph, llmp::ir::Op::kAttention), p.layers);
  EXPECT_EQ(Count(prefill->graph, llmp::ir::Op::kIndexScore), 21U);
  // The injection: each drafter layer's ring written.
  EXPECT_EQ(std::ranges::count_if(
                prefill->state,
                [](const co::StepState& st) { return st.ref.stage == 1 && st.tap && st.written; }),
            static_cast<std::ptrdiff_t>(d.blocks.layers));
  const auto state_only =
      co::BuildStep(*composed, 0, {.segments = {Seg(0, 300, 512, read, 0)}}, t_bindings);
  ASSERT_TRUE(state_only.has_value()) << state_only.error();
  const auto decodes =
      co::BuildStep(*composed, 0,
                    {.phase = pl::Phase::kDecode,
                     .segments = {Seg(0, 1, 812, read, 1, true), Seg(1, 1, 40000, read, 1, true)}},
                    t_bindings);
  ASSERT_TRUE(decodes.has_value()) << decodes.error();
  EXPECT_TRUE(decodes->greedy.valid());
  const auto verify = co::BuildStep(
      *composed, 0, {.phase = pl::Phase::kVerify, .segments = {Seg(0, 6, 813, read, 6)}},
      t_bindings);
  ASSERT_TRUE(verify.has_value()) << verify.error();
  const auto block = co::BuildStep(
      *composed, 1,
      {.stage = 1, .phase = pl::Phase::kDraft, .segments = {Seg(0, 5, 819, 256, 5, true)}},
      d_bindings);
  ASSERT_TRUE(block.has_value()) << block.error();
  EXPECT_TRUE(block->greedy.valid());
}

// Serving's layouts (components::StageLayout) of the composed target and
// drafter, at the speed gate's context and chunk and at the C2 capacity
// screen's (docs/experiments/m36-dsv4-target/dspark.md): every compressed
// bank contiguous, and a slot's state (its region, a prompt's first demand)
// near the packed layout's, which C2's capacity needs (padding each small
// tensor to an extent cost 506 MiB a slot, not 56): at the C2 screen's
// context the same, at the gate's within an extent and a straddle for
// each growing tensor of an extent or more.
TEST(ComponentsDsv4Models, ServingStateLayoutsKeepBanksAndPackedCapacity) {
  const auto& mapping = sp::mappings::Dsv4Mapping();
  const auto& drafter_mapping = sp::mappings::DsparkMapping();
  auto target = Open("deepseek", mapping);
  auto drafter = Open("dspark", drafter_mapping);
  if (!target || !drafter) {
    GTEST_FAIL() << "no DeepSeek V4 and DSpark v1 artifacts in " << llmp::test_support::V1Store();
  }
  auto spec = sp::StoredSpec(target->artifact);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  auto kept = rt::ComponentDrafterValues(target->artifact, drafter->artifact);
  ASSERT_TRUE(kept.has_value()) << kept.error();
  auto composed = sp::ComposeDrafter(*spec, mapping, drafter_mapping, drafter->source, *kept);
  ASSERT_TRUE(composed.has_value()) << composed.error();
  for (const auto& [context, rows] : {std::pair{262144U, 4096U}, {8448U, 256U}}) {
    for (std::size_t stage = 0; stage < 2; ++stage) {
      const auto f =
          llmp::test_support::ExpectServingLayout("deepseek", *composed, stage, context, rows);
      if (stage == 0) EXPECT_GT(f.banks, 0U);
      if (context == 8448) EXPECT_EQ(f.rule.initial, f.packed.initial);
      // Each tensor on its own extents costs at most that extent and one
      // its neighbours straddle.
      EXPECT_LE(f.rule.region, f.packed.region + f.big);
      EXPECT_LE(f.rule.initial, f.packed.initial + 2 * f.big);
      EXPECT_LE(f.rule.growths, f.packed.growths);
    }
  }
}

}  // namespace
