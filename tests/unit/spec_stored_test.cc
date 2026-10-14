// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The spec a v1 artifact or composition stores (spec/stored.h, D-113) on
// artifacts the test-only writer makes (tests/support/artifact_writer.h):
// the stored spec read back with its digest, the artifact's tensors as a
// source, the applied folds checked against the catalog and each stage's
// folds, the stages' artifact indexes, a composition of a target and its
// drafter, v0 artifacts (the prototype's goldens) as before, and the M3
// per-family binders refusing v1 artifacts.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "artifact/format_map.h"
#include "artifact_writer.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "model/qwen38.h"
#include "model/qwen_image.h"
#include "spec/mapping.h"
#include "spec/spec.h"
#include "spec/stored.h"

namespace {

namespace fs = std::filesystem;
namespace artifact = llmp::artifact;
namespace format = llmp::format;
namespace spec = llmp::spec;
using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;
using llmp::test::V1Group;
using llmp::test::V1Plan;

std::string Env(const char* name) {
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value == nullptr ? std::string() : std::string(value);
}

fs::path Scratch(std::string_view name) {
  const std::string base = Env("LLMP_TEST_SCRATCH");
  const fs::path root = base.empty() ? fs::path(::testing::TempDir()) : fs::path(base);
  const fs::path dir = root / "spec-stored" / name;
  std::error_code error;
  fs::remove_all(dir, error);
  fs::create_directories(dir, error);
  return dir;
}

fs::path Corpus() {
  const std::string configured = Env("LLMP_TEST_ARTIFACT_CORPUS");
  return configured.empty() ? fs::path("artifact-corpus") : fs::path(configured);
}

TensorFormat Format(Encoding e, Layout l, std::vector<std::uint64_t> ne) {
  return {.encoding = e, .layout = l, .shape = *format::ShapeOf(ne)};
}

spec::ComponentRef C(std::string kind, std::vector<spec::WeightBinding> weights) {
  return {.kind = std::move(kind), .version = 1, .params = {}, .weights = std::move(weights)};
}

spec::WeightBinding W(std::string role, std::string tensor, PartRole part = PartRole::kData) {
  return {.role = std::move(role), .tensor = std::move(tensor), .part = part};
}

// A one-layer target over the artifact's tensors.
spec::ModelSpec Target(std::uint32_t artifact) {
  spec::Stage s{.name = "target", .role = spec::StageRole::kTarget, .artifact = artifact};
  s.embedding = C("embedding.token", {W("table", "token_embd.weight")});
  s.stream = {.streams = 1, .width = 64, .mixing = std::nullopt};
  s.layers.push_back(spec::Layer{
      .components = {C("norm.rms", {W("weight", "blk.0.attn_norm.weight")}),
                     C("ffn.moe", {spec::WeightBinding{.role = "gate",
                                                       .tensor = "blk.0.ffn_exps.codes",
                                                       .part = PartRole::kCodes,
                                                       .experts = true},
                                   spec::WeightBinding{.role = "gate",
                                                       .tensor = "blk.0.ffn_exps.scales",
                                                       .part = PartRole::kBlockScales,
                                                       .experts = true}})}});
  s.taps = {spec::FeatureTap{.name = "features", .layer = 0, .point = spec::TapPoint::kLayerOut}};
  s.heads.output = C("head.linear", {W("weight", "output.weight")});
  s.folds = {"norm.plus_one"};
  spec::ModelSpec m{.pipeline = spec::PipelineKind::kLlm, .stages = {s}};
  spec::Canonicalize(m);
  return m;
}

TensorFormat Mxfp8(std::uint64_t k, std::uint64_t n) {
  return Format(Encoding::kMxfp8, Layout::kSplit, {k, n});
}

V1Plan Plan(std::optional<std::string> stored) {
  V1Plan p;
  p.architecture = "toy";
  p.expert_count = 2;
  p.spec = std::move(stored);
  p.groups = {
      V1Group{.kind = artifact::GroupKind::kTable,
              .resources = {{.name = "token_embd.weight",
                             .aliases = {"output.weight"},
                             .format = Format(Encoding::kQ8_0, Layout::kGgmlBlocks, {64, 100}),
                             .rows = true}}},
      V1Group{.kind = artifact::GroupKind::kLayer,
              .layer = 0,
              .resources = {{.name = "blk.0.attn_norm.weight",
                             .format = Format(Encoding::kF32, Layout::kDense, {64})}}},
      V1Group{.kind = artifact::GroupKind::kExpert,
              .layer = 0,
              .arrays = {{.name = "blk.0.ffn_exps.codes",
                          .format = Mxfp8(64, 32),
                          .part = PartRole::kCodes},
                         {.name = "blk.0.ffn_exps.scales",
                          .format = Mxfp8(64, 32),
                          .part = PartRole::kBlockScales}}},
  };
  p.folds = {{"norm.plus_one", {"blk.0.attn_norm.weight"}}};
  return p;
}

artifact::Artifact Open(const fs::path& root) {
  auto opened = artifact::Artifact::Open(root);
  if (!opened) {
    ADD_FAILURE() << opened.error().ToString();
    std::abort();
  }
  return std::move(*opened);
}

TEST(SpecStoredTest, ReadsTheStoredSpecWithItsDigest) {
  const spec::ModelSpec want = Target(0);
  ASSERT_TRUE(spec::Validate(want).has_value());
  const std::string text = spec::ToJson(want);
  const artifact::Artifact a = Open(Publish(Scratch("stored"), Render(Plan(text))));
  const auto got = spec::StoredSpec(a);
  ASSERT_TRUE(got.has_value()) << got.error();
  EXPECT_EQ(*got, want);
  // The listed file's digest is the spec's.
  const auto file = std::ranges::find(a.files(), "spec.json", &artifact::ListedFile::path);
  ASSERT_NE(file, a.files().end());
  EXPECT_EQ(file->sha256, got->Digest());
}

TEST(SpecStoredTest, TheArtifactsTensorsAsASource) {
  const artifact::Artifact a = Open(Publish(Scratch("source"), Render(Plan(std::nullopt))));
  const auto source = spec::SourceOf(a);
  ASSERT_TRUE(source.has_value()) << source.error();
  EXPECT_EQ(source->architecture, "toy");
  EXPECT_EQ(source->container, "gguf");
  const auto find = [&](std::string_view name) -> const spec::SourceTensor* {
    const auto it = std::ranges::find(source->tensors, name, &spec::SourceTensor::name);
    return it == source->tensors.end() ? nullptr : &*it;
  };
  const spec::SourceTensor* table = find("token_embd.weight");
  const spec::SourceTensor* head = find("output.weight");
  ASSERT_NE(table, nullptr);
  ASSERT_NE(head, nullptr);
  EXPECT_EQ(table->format, head->format);
  EXPECT_EQ(table->format, Format(Encoding::kQ8_0, Layout::kGgmlBlocks, {64, 100}));
  const spec::SourceTensor* scales = find("blk.0.ffn_exps.scales");
  ASSERT_NE(scales, nullptr);
  EXPECT_TRUE(scales->experts);
  EXPECT_EQ(scales->part, PartRole::kBlockScales);
  EXPECT_EQ(scales->format, Format(Encoding::kMxfp8, Layout::kSplit, {64, 32, 2}));
  EXPECT_EQ(source->tensors.size(), 5U);
  // The importer's folds, so that a spec synthesized from the source lists
  // them and no component applies one again.
  EXPECT_EQ(source->folds, std::vector<std::string>{"norm.plus_one"});
}

TEST(SpecStoredTest, AppliedFoldsMustBeTheCatalogsAndTheStages) {
  {
    V1Plan p = Plan(spec::ToJson(Target(0)));
    p.folds = {{"norm.minus_one", {"blk.0.attn_norm.weight"}}};
    const artifact::Artifact a = Open(Publish(Scratch("unknown-fold"), Render(p)));
    const auto stored = spec::StoredSpec(a);
    ASSERT_FALSE(stored.has_value());
    EXPECT_NE(stored.error().find("not in the fold catalog"), std::string::npos) << stored.error();
    const auto source = spec::SourceOf(a);
    ASSERT_FALSE(source.has_value());
    EXPECT_NE(source.error().find("not in the fold catalog"), std::string::npos);
  }
  {
    V1Plan p = Plan(spec::ToJson(Target(0)));
    p.folds = {{"norm.plus_one", {"blk.0.attn_norm.weight"}},
               {"ssm.a_neg_exp", {"blk.0.attn_norm.weight"}}};
    const artifact::Artifact a = Open(Publish(Scratch("unlisted-fold"), Render(p)));
    const auto stored = spec::StoredSpec(a);
    ASSERT_FALSE(stored.has_value());
    EXPECT_NE(stored.error().find("not among stage target's folds"), std::string::npos)
        << stored.error();
    EXPECT_TRUE(spec::SourceOf(a).has_value());
  }
  {
    // A fold record must name a tensor the stages bind (by resource name;
    // here only the head binds it, by the table's tied alias).
    spec::ModelSpec tied = Target(0);
    tied.stages[0].embedding.reset();
    V1Plan p = Plan(spec::ToJson(tied));
    p.folds = {{"norm.plus_one", {"token_embd.weight"}}};
    const artifact::Artifact a = Open(Publish(Scratch("fold-of-alias"), Render(p)));
    const auto stored = spec::StoredSpec(a);
    EXPECT_TRUE(stored.has_value()) << stored.error();
  }
  {
    // A stage without the head: nothing of the folded tensor is bound.
    spec::ModelSpec headless = Target(0);
    headless.stages[0].heads.output.reset();
    headless.stages[0].embedding.reset();
    V1Plan p = Plan(spec::ToJson(headless));
    p.folds = {{"norm.plus_one", {"token_embd.weight"}}};
    const artifact::Artifact a = Open(Publish(Scratch("fold-of-nothing"), Render(p)));
    const auto stored = spec::StoredSpec(a);
    ASSERT_FALSE(stored.has_value());
    EXPECT_NE(stored.error().find("names no tensor"), std::string::npos) << stored.error();
  }
}

TEST(SpecStoredTest, RefusesWhatTheArtifactCannotHold) {
  {
    const artifact::Artifact a =
        Open(Publish(Scratch("other-artifact"), Render(Plan(spec::ToJson(Target(1))))));
    const auto stored = spec::StoredSpec(a);
    ASSERT_FALSE(stored.has_value());
    EXPECT_NE(stored.error().find("not this one"), std::string::npos) << stored.error();
  }
  {
    const artifact::Artifact a = Open(Publish(Scratch("no-spec"), Render(Plan(std::nullopt))));
    EXPECT_FALSE(spec::StoredSpec(a).has_value());
  }
  {
    // Canonical JSON, but not a spec this build reads.
    const artifact::Artifact a =
        Open(Publish(Scratch("not-a-spec"),
                     Render(Plan(std::string(R"({"pipeline":"llm","stages":[],"version":2})"
                                             "\n")))));
    const auto stored = spec::StoredSpec(a);
    ASSERT_FALSE(stored.has_value());
    EXPECT_NE(stored.error().find("version"), std::string::npos) << stored.error();
  }
}

TEST(SpecStoredTest, V0ArtifactsAsBefore) {
  std::error_code error;
  fs::path root;
  for (const auto& entry : fs::directory_iterator(Corpus() / "golden" / "tiny", error)) {
    root = entry.path();
  }
  ASSERT_FALSE(root.empty()) << "no corpus at " << Corpus();
  const artifact::Artifact a = Open(root);
  EXPECT_FALSE(spec::StoredSpec(a).has_value());
  const auto source = spec::SourceOf(a);
  const auto v0 = spec::SourceOfV0(a);
  ASSERT_TRUE(source.has_value() && v0.has_value());
  EXPECT_EQ(source->container, v0->container);
  ASSERT_EQ(source->tensors.size(), v0->tensors.size());
  for (std::size_t i = 0; i < v0->tensors.size(); ++i) {
    EXPECT_EQ(source->tensors[i].name, v0->tensors[i].name);
    EXPECT_EQ(source->tensors[i].format, v0->tensors[i].format);
    EXPECT_EQ(source->tensors[i].part, v0->tensors[i].part);
  }
  // Open reads each v0 resource's format as format_map.h does.
  for (const artifact::Resource& r : a.resources()) {
    const auto f = artifact::FormatOf(r.repr);
    ASSERT_TRUE(f.has_value()) << r.name;
    EXPECT_EQ(r.format, *f) << r.name;
  }
}

// The M3 per-family binders read v0's representation descriptors, which a
// v1 artifact leaves empty: each refuses it, naming its version, rather
// than misreading it (D-113).
TEST(SpecStoredTest, LegacyFamiliesRefuseV1Artifacts) {
  namespace model = llmp::model;
  const artifact::Artifact a = Open(Publish(Scratch("legacy"), Render(Plan(std::nullopt))));
  ASSERT_EQ(a.format_version(), 1U);
  const auto refused = [](const auto& result, std::string_view what) {
    ASSERT_FALSE(result.has_value()) << what;
    EXPECT_NE(result.error().find("format version 1"), std::string::npos)
        << what << ": " << result.error();
  };
  refused(model::BindApprovedGemma2(a), "Gemma2 approved");
  refused(model::BindGemma2(model::Gemma2_2B(), a), "Gemma2");
  refused(model::BindApprovedGemma3(a), "Gemma3 approved");
  refused(model::BindGemma3(model::Gemma3_4BQat(), a), "Gemma3");
  refused(model::BindGemma4(model::Gemma4_26BA4B(), a), "Gemma4");
  refused(model::BindGemma4Assistant(model::Gemma4Assistant26(), a), "Gemma4 assistant");
  refused(model::BindDsv4(model::Dsv4Flash(), a), "DeepSeek V4");
  refused(model::BindDspark(model::DsparkDeepSeekV4Flash(), a, model::Dsv4Flash(),
                            model::Dsv4Binding{}),
          "DSpark");
  refused(model::BindQwen38(model::Qwen38Flash(), a), "Qwen3.8");
  refused(model::BindQwen38Mtp(model::Qwen38Flash(), a), "Qwen3.8 MTP");
  refused(model::BindQwen2(model::Qwen25Instruct05B(), a), "Qwen2");
  refused(model::BindQwen2Exl3(model::Qwen25Instruct05BExl3(), a), "Qwen2 EXL3");
  // The v0 source reader too; SourceOf reads both versions.
  refused(spec::SourceOfV0(a), "SourceOfV0");
  EXPECT_TRUE(spec::SourceOf(a).has_value());
}

// A target and its drafter: two artifacts, the drafter carrying no spec of
// its own, and a composition whose spec binds both (components in role
// order: the assistant is artifact 0, the target artifact 1).
TEST(SpecStoredTest, ACompositionsSpecBindsItsComponents) {
  const fs::path store = Scratch("composition");
  const artifact::Artifact target = Open(Publish(store, Render(Plan(std::nullopt))));
  V1Plan dp = Plan(std::nullopt);
  dp.architecture = "toy-assistant";
  dp.folds = {{"norm.plus_one", {"blk.0.attn_norm.weight"}}};
  const artifact::Artifact drafter = Open(Publish(store, Render(dp)));
  // A target whose importer applied no fold (a safetensors checkpoint's).
  V1Plan up = Plan(std::nullopt);
  up.architecture = "toy";
  up.container = "safetensors";
  up.folds = {};
  const artifact::Artifact plain = Open(Publish(store, Render(up)));

  spec::ModelSpec m = Target(1);
  spec::Stage d{.name = "assistant", .role = spec::StageRole::kDrafter, .artifact = 0};
  d.stream = {.streams = 1, .width = 32, .mixing = std::nullopt};
  d.layers.push_back(
      spec::Layer{.components = {C("norm.rms", {W("weight", "blk.0.attn_norm.weight")})}});
  d.reads = {spec::Read{.stage = "target", .kind = spec::ReadKind::kTap, .name = "features"}};
  d.folds = {"norm.plus_one"};
  m.stages.push_back(d);
  spec::Canonicalize(m);
  ASSERT_TRUE(spec::Validate(m).has_value()) << spec::Validate(m).error();

  const auto compose = [&](const spec::ModelSpec& s, const artifact::Artifact& t) {
    llmp::test::CompositionPlan plan;
    plan.architecture = "toy";
    plan.components = {{"assistant", drafter.id(), "toy-assistant"}, {"target", t.id(), "toy"}};
    plan.spec = spec::ToJson(s);
    auto opened = artifact::OpenComposition(Publish(store, Render(plan)));
    if (!opened) {
      ADD_FAILURE() << opened.error().ToString();
      std::abort();
    }
    return std::move(*opened);
  };
  const artifact::Composition composition = compose(m, target);
  const std::vector<const artifact::Artifact*> ordered = {&drafter, &target};
  const auto got = spec::StoredSpec(composition, ordered);
  ASSERT_TRUE(got.has_value()) << got.error();
  EXPECT_EQ(*got, m);
  // Its bound digest: the spec's with the components' IDs in order.
  const std::vector<std::string_view> ids = {drafter.id(), target.id()};
  EXPECT_TRUE(spec::BoundDigest(*got, ids).has_value());

  const std::vector<const artifact::Artifact*> swapped = {&target, &drafter};
  EXPECT_FALSE(spec::StoredSpec(composition, swapped).has_value());
  const std::vector<const artifact::Artifact*> short_list = {&drafter};
  EXPECT_FALSE(spec::StoredSpec(composition, short_list).has_value());

  spec::ModelSpec past = m;
  for (spec::Stage& s : past.stages) {
    if (s.role == spec::StageRole::kDrafter) {
      s.artifact = 2;
    }
  }
  const auto refused = spec::StoredSpec(compose(past, target), ordered);
  ASSERT_FALSE(refused.has_value());
  EXPECT_NE(refused.error().find("of 2"), std::string::npos) << refused.error();

  // Each component's applied folds against its own stages' folds: the
  // drafter's fold listed only on the target is refused.
  spec::ModelSpec unfolded = m;
  for (spec::Stage& s : unfolded.stages) {
    if (s.role == spec::StageRole::kDrafter) {
      s.folds.clear();
    }
  }
  const auto unlisted = spec::StoredSpec(compose(unfolded, target), ordered);
  ASSERT_FALSE(unlisted.has_value());
  EXPECT_NE(unlisted.error().find("not among stage assistant's folds"), std::string::npos)
      << unlisted.error();
  // A target of a container whose stage lists no fold beside a folded
  // drafter: the folds are per stage, so the drafter's never cover the
  // target's tensors.
  spec::ModelSpec mixed = m;
  for (spec::Stage& s : mixed.stages) {
    if (s.role == spec::StageRole::kTarget) {
      s.folds.clear();
    }
  }
  const std::vector<const artifact::Artifact*> mixed_order = {&drafter, &plain};
  const auto per_stage = spec::StoredSpec(compose(mixed, plain), mixed_order);
  ASSERT_TRUE(per_stage.has_value()) << per_stage.error();
  ASSERT_EQ(per_stage->stages.size(), 2U);
  EXPECT_TRUE(per_stage->stages[0].folds.empty());  // the target
  EXPECT_EQ(per_stage->stages[1].folds, std::vector<std::string>{"norm.plus_one"});
  // The folded target under that spec is refused: its fold is not its stage's.
  const std::vector<const artifact::Artifact*> folded_order = {&drafter, &target};
  EXPECT_FALSE(spec::StoredSpec(compose(mixed, target), folded_order).has_value());

  // A drafter's binding of the target's tensor (WeightBinding::stage) is
  // the target artifact's: here it alone binds the norm the target's fold
  // names, which the drafter's own folds never cover.
  spec::ModelSpec borrowed = m;
  for (spec::Stage& s : borrowed.stages) {
    if (s.role == spec::StageRole::kTarget) {
      std::erase_if(s.layers[0].components,
                    [](const spec::ComponentRef& c) { return c.kind == "norm.rms"; });
    } else {
      spec::WeightBinding w = W("weight", "blk.0.attn_norm.weight");
      w.stage = "target";
      s.layers[0].components.push_back(C("norm.rms", {w}));
    }
  }
  spec::Canonicalize(borrowed);
  ASSERT_TRUE(spec::Validate(borrowed).has_value()) << spec::Validate(borrowed).error();
  const auto through = spec::StoredSpec(compose(borrowed, target), ordered);
  EXPECT_TRUE(through.has_value()) << through.error();
  // Without that binding nothing binds the folded norm from the target.
  spec::ModelSpec unbound = borrowed;
  for (spec::Stage& s : unbound.stages) {
    if (s.role == spec::StageRole::kDrafter) {
      s.layers[0].components.pop_back();
    }
  }
  const auto nothing = spec::StoredSpec(compose(unbound, target), ordered);
  ASSERT_FALSE(nothing.has_value());
  EXPECT_NE(nothing.error().find("names no tensor"), std::string::npos) << nothing.error();
}

}  // namespace
