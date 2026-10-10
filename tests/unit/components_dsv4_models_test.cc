// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 mappings on the installed v0 artifacts (label `models`;
// each case skips when its artifact is absent): DeepSeek V4 Flash 0731's
// UD-Q2_K_XL (8a355bfb), the community IQ2_XXS (cd39d504) and the DSpark
// drafter (dd2d3f9c) composed with each. Each spec is synthesized
// (SynthesizeV0, ComposeDrafterV0) from its artifact's tensors and the
// values today's profiles give (Dsv4V0Values, DsparkV0Values) and
// validated against its tensors; it equals the spec of the unit tests'
// fixture (dsv4_fixture.h), so the unit tests' structure and layout checks
// hold for the real artifacts; the stage layouts are today's state
// (Dsv4State in both window modes, DsparkState); and each step kind
// builds: the target's prefill with its drafter's injection, a state-only
// chunk, joined greedy decodes, a verify, and the drafter's blocks.
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
#include <system_error>
#include <vector>

#include "artifact/artifact.h"
#include "components/step.h"
#include "dsv4_fixture.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"

namespace {

namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace sp = llmp::spec;
namespace fx = llmp::test_support::dsv4;
namespace fs = std::filesystem;

constexpr std::string_view kTarget =
    "m3-artifacts/8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
constexpr std::string_view kCommunity =
    "m3-artifacts/cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac";
constexpr std::string_view kDspark =
    "m3-artifacts/dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5";

std::optional<fs::path> Installed(std::string_view id) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / id;
  if (std::error_code error; !fs::exists(root, error)) return std::nullopt;
  return root;
}

struct Opened {
  ar::Artifact artifact;
  sp::Source source;
};

std::optional<Opened> Open(std::string_view id, const sp::Mapping& mapping) {
  const auto root = Installed(id);
  if (!root) return std::nullopt;
  auto artifact = ar::Artifact::Open(*root, {.expected_id = {}});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return std::nullopt;
  EXPECT_EQ(sp::FindMapping(artifact->model().architecture), &mapping);
  // As the runtime reads any artifact (spec/stored.h).
  auto source = sp::SourceOf(*artifact, &mapping);
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
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  const auto values = sp::mappings::Dsv4V0Values(p);
  const auto drafter_values = sp::mappings::DsparkV0Values(d);
  ASSERT_TRUE(values.has_value() && drafter_values.has_value());
  const auto& mapping = sp::mappings::Dsv4Mapping();
  const auto& drafter_mapping = sp::mappings::DsparkMapping();
  auto drafter = Open(kDspark, drafter_mapping);
  std::size_t opened = 0;
  for (const std::string_view id : {kTarget, kCommunity}) {
    SCOPED_TRACE(id);
    auto target = Open(id, mapping);
    if (!target) continue;
    ++opened;
    // Today's binding takes it.
    auto binding = md::BindDsv4(p, target->artifact);
    ASSERT_TRUE(binding.has_value()) << binding.error();
    auto spec = sp::SynthesizeV0(target->artifact, mapping, *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    ASSERT_TRUE(co::ValidateSpec(*spec, target->source.tensors));
    EXPECT_TRUE(spec->stages.front().folds.empty());
    // The unit tests' fixture is this artifact's tensors.
    const sp::Source mine = fx::SourceOf("deepseek4", fx::Resources(p), mapping);
    auto fixture = sp::Synthesize(mapping, mine, *values);
    ASSERT_TRUE(fixture.has_value()) << fixture.error();
    EXPECT_EQ(sp::ToJson(*spec), sp::ToJson(*fixture));
    EXPECT_EQ(target->source.tensors.size(), mine.tensors.size());
    // Today's state, in both window modes.
    for (const auto window : {md::Dsv4Window::kRing, md::Dsv4Window::kFull}) {
      sp::ModelSpec form = *spec;
      if (window == md::Dsv4Window::kFull) {
        for (sp::Layer& l : form.stages[0].layers) {
          for (sp::ComponentRef& c : l.components) {
            for (sp::Param& q : c.params) {
              if (c.kind == "attention.mqa" && q.name == "cache") q.value = std::string("full");
            }
          }
        }
      }
      const auto layout = co::StageLayout(form, 0, 65536, 512);
      ASSERT_TRUE(layout.has_value()) << layout.error();
      const auto want = md::Dsv4State(p, 65536, 512, window);
      ASSERT_TRUE(want.has_value());
      ASSERT_EQ(layout->items.size(), want->tensors.size());
      for (std::size_t i = 0; i < want->tensors.size(); ++i) {
        EXPECT_EQ(layout->items[i].offset, want->tensors[i].offset) << i;
      }
      EXPECT_EQ(layout->bytes, want->bytes);
    }
    if (!drafter) continue;
    // Its drafter composed.
    auto drafter_binding = md::BindDspark(d, drafter->artifact, p, *binding);
    ASSERT_TRUE(drafter_binding.has_value()) << drafter_binding.error();
    auto composed =
        sp::ComposeDrafterV0(*spec, mapping, drafter->artifact, drafter_mapping, *drafter_values);
    ASSERT_TRUE(composed.has_value()) << composed.error();
    const std::array<std::span<const co::BoundTensor>, 2> artifacts{target->source.tensors,
                                                                    drafter->source.tensors};
    ASSERT_TRUE(co::ValidateSpec(*composed, artifacts));
    const sp::Source drafted = fx::SourceOf("dflash", fx::DrafterResources(d), drafter_mapping);
    EXPECT_EQ(drafter->source.tensors.size(), drafted.tensors.size());
    const auto t_layout = co::StageLayout(*composed, 0, 65536, 512);
    const auto d_layout = co::StageLayout(*composed, 1, 65536, 16);
    ASSERT_TRUE(t_layout.has_value() && d_layout.has_value());
    const auto ring = md::DsparkState(d, 16);
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
    EXPECT_EQ(std::ranges::count_if(prefill->state,
                                    [](const co::StepState& st) {
                                      return st.ref.stage == 1 && st.tap && st.written;
                                    }),
              static_cast<std::ptrdiff_t>(d.blocks.layers));
    const auto state_only =
        co::BuildStep(*composed, 0, {.segments = {Seg(0, 300, 512, read, 0)}}, t_bindings);
    ASSERT_TRUE(state_only.has_value()) << state_only.error();
    const auto decodes = co::BuildStep(
        *composed, 0,
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
  if (opened == 0) GTEST_SKIP() << "no DeepSeek V4 artifact installed";
}

}  // namespace
