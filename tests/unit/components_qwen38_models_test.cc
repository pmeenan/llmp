// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 mappings on the published v1 artifacts (label `models`; each
// case fails when the store or its artifact is absent): the native
// ModelOpt import (d145900e), the GGUF quantization's (18a67153) and the
// ModelOpt MTP drafter (316caa59) composed with its approved ModelOpt
// target. Each target's stored spec is validated against its tensors; the
// ModelOpt one equals the spec of the unit tests' native fixture
// (qwen38_fixture.h NativeModelOpt, the importer's own form) and the GGUF
// one the fixture's GGUF form synthesized with the family's values
// (tests/support/families/qwen.h), the n-gram hash's constants pinned;
// both imports' layers are the same components; the stage layouts are the
// M3 runner's state; the drafter composes from the values it keeps, as
// serving composes it, to its approved composition; and each step kind
// builds: a prefill, a greedy decode, a verify with its commit, and the
// drafter's pass. (The v0 CUTLASS-layout and GGML-experts ModelOpt imports,
// c4fb47a9 and 67617f87, have no v1 counterparts but the native one.)
//
// These build the steps, they do not run them: the reference executor
// binds whole weights (97 GB for a ModelOpt artifact, its n-gram table
// 28.8 GB alone), more than a Spark holds beside it, so Qwen3.8's
// numerical gate is the lowering's on the GPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "components/step.h"
#include "families/qwen.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "qwen38_fixture.h"
#include "qwen38_vocab.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
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
namespace fx = llmp::test_support::qwen38;
namespace fs = std::filesystem;

// Qwen3.8 Flash's n-gram hash constants, as its GGUF's qwen4exp.ple.* keys
// and the native importer give them.
fm::Qwen38PleHash FlashHash() {
  return {
      .multipliers = {23703573157769, 20109073645365, 8052911324071},
      .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                  160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                  300001275},
      .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}};
}

struct Opened {
  ar::Artifact artifact;
  sp::Source source;
};

// The v1 artifact the store names `name`, with its source; nullopt when
// absent.
std::optional<Opened> OpenNamed(std::string_view name) {
  const auto id = llmp::test_support::V1IdNamed(name);
  if (!id) return std::nullopt;
  auto artifact =
      ar::Artifact::Open(fs::path(llmp::test_support::V1Store()) / *id, {.expected_id = *id});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return std::nullopt;
  EXPECT_EQ(artifact->format_version(), 1U);
  auto source = sp::SourceOf(*artifact);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  if (!source) return std::nullopt;
  return Opened{.artifact = std::move(*artifact), .source = std::move(*source)};
}

struct Target {
  Opened opened;
  sp::ModelSpec spec;
};

// A target with its stored spec.
std::optional<Target> Open(std::string_view name) {
  auto opened = OpenNamed(name);
  if (!opened) return std::nullopt;
  EXPECT_EQ(sp::FindMapping(opened->artifact.model().architecture), &sp::mappings::Qwen38Mapping());
  auto spec = sp::StoredSpec(opened->artifact);
  EXPECT_TRUE(spec.has_value()) << (spec ? "" : spec.error());
  if (!spec) return std::nullopt;
  return Target{.opened = std::move(*opened), .spec = std::move(*spec)};
}

std::vector<std::vector<std::string>> Kinds(const sp::Stage& s) {
  std::vector<std::vector<std::string>> out;
  for (const sp::Layer& l : s.layers) {
    auto& kinds = out.emplace_back();
    for (const sp::ComponentRef& c : l.components) kinds.push_back(c.kind);
  }
  return out;
}

std::size_t Count(const llmp::ir::Graph& g, llmp::ir::Op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes(), [op](const llmp::ir::Node& n) { return n.key.op == op; }));
}

std::expected<sp::SamplingDefaults, std::string> GgufSamplingDefaults(
    const ar::Artifact& artifact) {
  constexpr std::array<std::string_view, 3> wanted{
      "general.sampling.temp", "general.sampling.top_k", "general.sampling.top_p"};
  for (const ar::ListedFile& file : artifact.files()) {
    if (file.role != ar::FileRole::kSourceMetadata || !file.path.ends_with(".kv.gguf")) continue;
    const auto raw = artifact.ReadMetadata(std::string_view(file.path).substr(5));
    if (!raw) return std::unexpected(raw.error().ToString());
    const auto metadata = ar::ReadGgufMetadata(
        std::as_bytes(std::span<const char>(raw->data(), raw->size())), wanted);
    if (!metadata) return std::unexpected(metadata.error().ToString());
    const auto temperature = metadata->find(wanted[0]);
    const auto top_k = metadata->find(wanted[1]);
    const auto top_p = metadata->find(wanted[2]);
    if (temperature == metadata->end() || top_k == metadata->end() || top_p == metadata->end())
      continue;
    if (temperature->second.kind != ar::GgufValue::Kind::kFloat ||
        top_k->second.kind != ar::GgufValue::Kind::kInteger ||
        top_p->second.kind != ar::GgufValue::Kind::kFloat)
      return std::unexpected("GGUF sampling metadata has unexpected value types");
    return sp::SamplingDefaults{.temperature = temperature->second.real,
                                .top_p = top_p->second.real,
                                .top_k = top_k->second.integer};
  }
  return std::unexpected("GGUF artifact has no complete raw sampling metadata");
}

pl::Segment Seg(std::uint32_t rows, std::uint32_t past, std::uint32_t outputs,
                bool greedy = false) {
  return {.slot = 0,
          .rows = rows,
          .past = past,
          .read_cells = 256,
          .outputs = outputs,
          .greedy = greedy};
}

TEST(ComponentsQwen38Models, EveryInstalledImportMakesItsSpec) {
  const auto& p = fm::Qwen38Flash();
  std::optional<std::vector<std::vector<std::string>>> kinds;
  std::size_t opened = 0;
  for (const std::string_view name : {"qwen38-modelopt", "qwen38-gguf"}) {
    SCOPED_TRACE(name);
    auto t = Open(name);
    if (!t) continue;
    ++opened;
    const auto& tensors = t->opened.source.tensors;
    const auto valid = co::ValidateSpec(t->spec, tensors);
    ASSERT_TRUE(valid.has_value()) << valid.error();
    EXPECT_EQ(
        t->spec.stages.front().folds,
        (std::vector<std::string>{"gdn.value_heads_tiled", "norm.plus_one", "ssm.a_neg_exp"}));
    if (!kinds) {
      kinds = Kinds(t->spec.stages.front());
    } else {
      EXPECT_EQ(Kinds(t->spec.stages.front()), *kinds);
    }
    if (name == "qwen38-modelopt") {
      // The unit tests' native fixture is this artifact's tensors.
      const auto fixture = fx::NativeModelOpt();
      auto published_fixture = fixture.model;
      // The native source dialect represents the three expert-scale tensors
      // individually. This older published artifact uses the equivalent
      // converted representation: one projection_scales flag. The component
      // contract accepts either form (projection OR per-projection flags).
      std::size_t moe_components = 0;
      for (auto& stage : published_fixture.stages)
        for (auto& layer : stage.layers)
          for (auto& component : layer.components) {
            if (component.kind != "ffn.moe") continue;
            ++moe_components;
            const auto projection =
                std::ranges::find(component.params, "projection_scales", &sp::Param::name);
            ASSERT_NE(projection, component.params.end());
            ASSERT_TRUE(std::holds_alternative<bool>(projection->value));
            EXPECT_FALSE(std::get<bool>(projection->value));
            projection->value = true;
            for (const std::string_view scale_name : {"down_scale", "gate_scale", "up_scale"}) {
              const auto scale = std::ranges::find(component.params, scale_name, &sp::Param::name);
              ASSERT_NE(scale, component.params.end()) << scale_name;
              ASSERT_TRUE(std::holds_alternative<bool>(scale->value)) << scale_name;
              EXPECT_TRUE(std::get<bool>(scale->value)) << scale_name;
              component.params.erase(scale);
            }
          }
      ASSERT_EQ(moe_components, 48U);
      EXPECT_EQ(sp::ToJson(t->spec), sp::ToJson(published_fixture));
      ASSERT_EQ(tensors.size(), fixture.source.tensors.size());
      for (const sp::SourceTensor& s : tensors) {
        const auto at = std::ranges::find(fixture.source.tensors, s.name, &sp::SourceTensor::name);
        ASSERT_NE(at, fixture.source.tensors.end()) << s.name;
        EXPECT_EQ(at->format, s.format) << s.name;
        EXPECT_EQ(at->part, s.part) << s.name;
      }
    } else {
      // The fixture's GGUF form with the family's values.
      const auto values = fm::Qwen38GgufValues(p, FlashHash());
      ASSERT_TRUE(values.has_value()) << values.error();
      auto fixture = sp::Synthesize(sp::mappings::Qwen38Mapping(),
                                    fx::TargetSource(p, fx::Form::kGguf), *values);
      ASSERT_TRUE(fixture.has_value()) << fixture.error();
      const auto sampling = GgufSamplingDefaults(t->opened.artifact);
      ASSERT_TRUE(sampling.has_value()) << sampling.error();
      fixture->stages.front().sampling = *sampling;
      EXPECT_EQ(sp::ToJson(t->spec), sp::ToJson(*fixture));
    }
    // The M3 runner's state, and every target step kind builds.
    const auto layout = co::StageLayoutUniform(t->spec, 0, 8192, 512, 256);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    EXPECT_EQ(layout->bytes, fm::Qwen38StateBytes(p, 8192, 512));
    const std::array<const md::StateLayoutDerived*, 1> layouts{&*layout};
    const std::array<std::span<const co::BoundTensor>, 1> artifacts{tensors};
    const co::Bindings bindings{
        .tensors = tensors, .state = &*layout, .stages = layouts, .artifacts = artifacts};
    const auto prefill = co::BuildStep(t->spec, 0, {.segments = {Seg(64, 0, 1)}}, bindings);
    ASSERT_TRUE(prefill.has_value()) << prefill.error();
    EXPECT_EQ(Count(prefill->graph, llmp::ir::Op::kGatedDeltaNet), 36U);
    EXPECT_EQ(Count(prefill->graph, llmp::ir::Op::kAttention), 12U);
    const auto decode = co::BuildStep(
        t->spec, 0, {.phase = pl::Phase::kDecode, .segments = {Seg(1, 64, 1, true)}}, bindings);
    ASSERT_TRUE(decode.has_value()) << decode.error();
    EXPECT_TRUE(decode->greedy.valid());
    const auto verify = co::BuildStep(
        t->spec, 0, {.phase = pl::Phase::kVerify, .segments = {Seg(4, 65, 4)}}, bindings);
    ASSERT_TRUE(verify.has_value()) << verify.error();
    // Each linear-attention layer's four saves and the n-gram block's.
    EXPECT_EQ(verify->saves.size(), (36U * 4U) + 1U);
    const auto commit = co::BuildCommit(t->spec, 0, {.segments = {Seg(2, 65, 0)}}, bindings);
    ASSERT_TRUE(commit.has_value()) << commit.error();
    EXPECT_EQ(Count(commit->graph, llmp::ir::Op::kGatedDeltaNet), 36U);
  }
  if (opened == 0) GTEST_FAIL() << "no Qwen3.8 v1 artifact in " << llmp::test_support::V1Store();
  ASSERT_TRUE(kinds.has_value());
  EXPECT_EQ(kinds->size(), p.layers);
}

TEST(ComponentsQwen38Models, TheMtpDrafterComposesWithItsTargets) {
  const auto& p = fm::Qwen38Flash();
  auto drafter = OpenNamed("qwen38-modelopt-mtp");
  if (!drafter) GTEST_FAIL() << "no Qwen3.8 MTP v1 drafter in " << llmp::test_support::V1Store();
  auto t = Open("qwen38-modelopt");
  if (!t) GTEST_FAIL() << "no ModelOpt Qwen3.8 v1 target in " << llmp::test_support::V1Store();
  auto stored = sp::StoredSpec(drafter->artifact);
  ASSERT_TRUE(stored) << stored.error();
  const auto& source = drafter->source;
  auto spec = sp::ComposeStoredDrafter(t->spec, *stored);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  // Its approval pins this composition.
  const auto approved = rt::CheckComposedAdmission(*spec, t->opened.artifact, drafter->artifact);
  EXPECT_TRUE(approved.has_value()) << (approved ? "" : approved.error());
  const auto& tensors = t->opened.source.tensors;
  const std::array<std::span<const co::BoundTensor>, 2> artifacts{tensors, source.tensors};
  const auto valid = co::ValidateSpec(*spec, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  // Its own import's folds, never the target's.
  EXPECT_EQ(spec->stages.at(1).folds, source.folds);
  // Its head: its own selected rows (the artifact keeps them).
  const bool selected = std::ranges::any_of(
      source.tensors, [](const sp::SourceTensor& s) { return s.name == "draft_output.ids"; });
  EXPECT_TRUE(selected);
  const auto& head = spec->stages.at(1).heads.output;
  ASSERT_TRUE(head.has_value());
  EXPECT_EQ(std::ranges::any_of(head->weights,
                                [](const sp::WeightBinding& w) { return w.role == "ids"; }),
            selected);
  // State layouts.
  const auto t_layout = co::StageLayoutUniform(*spec, 0, 8192, 512, 256);
  const auto d_layout = co::StageLayoutUniform(*spec, 1, 8192, 512, 256);
  ASSERT_TRUE(t_layout.has_value() && d_layout.has_value());
  EXPECT_EQ(t_layout->bytes, fm::Qwen38StateBytes(p, 8192, 512));
  EXPECT_EQ(d_layout->bytes, fm::Qwen38MtpStateBytes(p, 8192, 512));
  // The draft-vocabulary study's prepared manifests record these layouts'
  // bytes at its context and 8,192-row chunks (benchmarks/qwen38_vocab.h).
  {
    namespace dv = llmp::benchmarks::draft_vocab;
    const auto study_target = co::StageLayoutUniform(*spec, 0, dv::kContext, 8192, 256);
    const auto study_drafter = co::StageLayoutUniform(*spec, 1, dv::kContext, 8192, 256);
    ASSERT_TRUE(study_target.has_value()) << study_target.error();
    ASSERT_TRUE(study_drafter.has_value()) << study_drafter.error();
    EXPECT_EQ(study_target->bytes, dv::kTargetStateBytes);
    EXPECT_EQ(study_drafter->bytes, dv::kMtpStateBytes);
  }
  const std::array<const md::StateLayoutDerived*, 2> layouts{&*t_layout, &*d_layout};
  // The target writes the drafter's streams; the drafter's pass and its
  // catch-up build.
  const auto prefill = co::BuildStep(
      *spec, 0, {.segments = {Seg(16, 0, 1)}},
      {.tensors = tensors, .state = &*t_layout, .stages = layouts, .artifacts = artifacts});
  ASSERT_TRUE(prefill.has_value()) << prefill.error();
  EXPECT_TRUE(std::ranges::any_of(prefill->state,
                                  [](const co::StepState& st) { return st.tap && st.written; }));
  const co::Bindings d{
      .tensors = source.tensors, .state = &*d_layout, .stages = layouts, .artifacts = artifacts};
  const auto pass = co::BuildStep(
      *spec, 1, {.stage = 1, .phase = pl::Phase::kDraft, .segments = {Seg(1, 16, 1, true)}}, d);
  ASSERT_TRUE(pass.has_value()) << pass.error();
  ASSERT_TRUE(pass->features.valid());
  EXPECT_EQ(pass->graph.tensor(pass->features).ne(0), p.hc_width());
  EXPECT_TRUE(pass->greedy.valid());
  const auto catch_up = co::BuildStep(*spec, 1, {.stage = 1, .segments = {Seg(3, 16, 0)}}, d);
  ASSERT_TRUE(catch_up.has_value()) << catch_up.error();
}

// Serving's layouts (components::StageLayout) at the speed gate's context
// and chunk: the ModelOpt target with its MTP drafter, and the GGUF target.
// Their small fixed tensors (recurrent and convolution state) pack; each
// tensor of an extent or more starts on its own.
TEST(ComponentsQwen38Models, ServingStateLayouts) {
  auto drafter = OpenNamed("qwen38-modelopt-mtp");
  auto t = Open("qwen38-modelopt");
  auto gguf = Open("qwen38-gguf");
  if (!drafter || !t || !gguf) {
    GTEST_FAIL() << "no Qwen3.8 v1 targets and MTP drafter in " << llmp::test_support::V1Store();
  }
  auto stored = sp::StoredSpec(drafter->artifact);
  ASSERT_TRUE(stored) << stored.error();
  auto spec = sp::ComposeStoredDrafter(t->spec, *stored);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  for (std::size_t stage = 0; stage < 2; ++stage) {
    const auto f =
        llmp::test_support::ExpectServingLayout("qwen38-modelopt", *spec, stage, 33792, 4096);
    EXPECT_EQ(f.banks, 0U);
    EXPECT_LE(f.rule.region, f.spread.region);
    EXPECT_LE(f.rule.region, f.packed.region + f.big);
  }
  const auto f = llmp::test_support::ExpectServingLayout("qwen38-gguf", gguf->spec, 0, 33792, 4096);
  EXPECT_LE(f.rule.region, f.spread.region);
}

}  // namespace
