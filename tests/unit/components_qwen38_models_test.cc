// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 mappings on the installed v0 artifacts (label `models`;
// each case skips when its artifact is absent): the ModelOpt import with
// its experts in the CUTLASS layout (c4fb47a9) and in GGML's NVFP4 blocks
// (67617f87), a GGUF quantization's (5356b5b0) and the MTP drafter
// (056a750e) composed with the ModelOpt targets. Each spec is synthesized
// (SynthesizeV0, ComposeDrafterV0) from its artifact's resources and the
// values its binding and n-gram hash give (Qwen38V0Values, read as today's
// runner reads them) and validated against its tensors; the ModelOpt ones
// equal the specs of the unit tests' fixture (qwen38_fixture.h) with those
// values, so the unit tests' structure and layout checks hold for the
// real artifacts; every import's layers are the same components; the
// stage layouts are today's state (model/qwen38.h); and each step kind
// builds: a prefill, a greedy decode, a verify with its commit, and the
// drafter's pass.
//
// These build the steps, they do not run them: the reference executor
// binds whole weights (97 GB for a ModelOpt artifact, its n-gram table
// 28.8 GB alone), more than a Spark holds beside it, so Qwen3.8's
// numerical gate is the lowering's on the GPU against today's runner
// (slice G3b), as component_runner_gpu_test's is for the earlier models.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "artifact/artifact.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/qwen38.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "qwen38_fixture.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"

namespace {

namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace sp = llmp::spec;
namespace fx = llmp::test_support::qwen38;
namespace fs = std::filesystem;

constexpr std::string_view kCutlass =
    "m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
constexpr std::string_view kGgmlExperts =
    "m3-artifacts/67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6";
constexpr std::string_view kGguf =
    "qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78";
constexpr std::string_view kMtp =
    "m3-artifacts/056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea";

std::optional<fs::path> Installed(std::string_view id) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / id;
  if (std::error_code error; !fs::exists(root, error)) return std::nullopt;
  return root;
}

// A resource's I64 values, read from its shard.
std::vector<std::int64_t> ReadI64(const fs::path& root, const ar::Artifact& a,
                                  const md::Qwen38Tensor& t) {
  const auto placement = a.ResourcePlacement(t.index);
  EXPECT_TRUE(placement.has_value());
  if (!placement) return {};
  const ar::Group& g = a.groups()[placement->group];
  const ar::Shard& shard = a.shards()[g.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() +
                                       placement->offset.value()));
  std::vector<std::int64_t> out(t.ne.at(0));
  in.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(out.size() * sizeof(std::int64_t)));
  EXPECT_TRUE(in.good());
  return out;
}

struct Target {
  fs::path root;
  ar::Artifact artifact;
  md::Qwen38Binding binding;
  std::vector<sp::SourceValue> values;
  sp::Source source;
  sp::ModelSpec spec;
};

// The artifact's spec as today's runner reads its binding and hash.
std::optional<Target> Open(std::string_view id) {
  const auto root = Installed(id);
  if (!root) return std::nullopt;
  auto artifact = ar::Artifact::Open(*root, {.expected_id = {}});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return std::nullopt;
  const auto& p = md::Qwen38Flash();
  auto binding = md::BindQwen38(p, *artifact);
  EXPECT_TRUE(binding.has_value()) << (binding ? "" : binding.error());
  if (!binding) return std::nullopt;
  const std::uint64_t rows = binding->ple_table.ne.back();
  std::expected<md::Qwen38PleHash, std::string> hash;
  if (binding->gguf()) {
    hash = md::ReadQwen38GgufHash(p, *artifact, rows);
  } else {
    const md::Qwen38Layer& l = binding->layers.at(p.ple_layer);
    hash = md::CheckQwen38PleHash(p, ReadI64(*root, *artifact, l.ple_multipliers),
                                  ReadI64(*root, *artifact, l.ple_head_offsets),
                                  ReadI64(*root, *artifact, l.ple_head_vocab), rows);
  }
  EXPECT_TRUE(hash.has_value()) << (hash ? "" : hash.error());
  if (!hash) return std::nullopt;
  auto values = sp::mappings::Qwen38V0Values(p, *binding, *hash);
  EXPECT_TRUE(values.has_value()) << (values ? "" : values.error());
  if (!values) return std::nullopt;
  const auto& mapping = sp::mappings::Qwen38Mapping();
  EXPECT_EQ(sp::FindMapping(artifact->model().architecture), &mapping);
  // As the runtime reads any artifact (spec/stored.h), the mapping's v0
  // pairs assembled.
  auto source = sp::SourceOf(*artifact, &mapping);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  auto spec = sp::SynthesizeV0(*artifact, mapping, *values);
  EXPECT_TRUE(spec.has_value()) << (spec ? "" : spec.error());
  if (!source || !spec) return std::nullopt;
  return Target{.root = *root,
                .artifact = std::move(*artifact),
                .binding = std::move(*binding),
                .values = std::move(*values),
                .source = std::move(*source),
                .spec = std::move(*spec)};
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
  const auto& p = md::Qwen38Flash();
  struct Case {
    std::string_view id;
    std::optional<fx::Form> fixture;  // the ModelOpt imports' fixture form
  };
  std::optional<std::vector<std::vector<std::string>>> kinds;
  std::size_t opened = 0;
  for (const Case& c : {Case{kCutlass, fx::Form::kCutlass},
                        Case{kGgmlExperts, fx::Form::kGgmlExperts}, Case{kGguf, std::nullopt}}) {
    SCOPED_TRACE(c.id);
    auto t = Open(c.id);
    if (!t) continue;
    ++opened;
    const auto valid = co::ValidateSpec(t->spec, t->source.tensors);
    ASSERT_TRUE(valid.has_value()) << valid.error();
    EXPECT_EQ(
        t->spec.stages.front().folds,
        (std::vector<std::string>{"gdn.value_heads_tiled", "norm.plus_one", "ssm.a_neg_exp"}));
    if (!kinds) {
      kinds = Kinds(t->spec.stages.front());
    } else {
      EXPECT_EQ(Kinds(t->spec.stages.front()), *kinds);
    }
    if (c.fixture) {
      // The unit tests' fixture is this artifact's tensors (its converter
      // one version of the ModelOpt importer's).
      const auto& mapping = sp::mappings::Qwen38Mapping();
      const sp::Source mine =
          fx::SourceOf("qwen4exp", "safetensors", std::string(fx::kModelOptConverter),
                       fx::Resources(p, *c.fixture), mapping);
      auto fixture = sp::Synthesize(mapping, mine, t->values);
      ASSERT_TRUE(fixture.has_value()) << fixture.error();
      EXPECT_EQ(sp::ToJson(t->spec), sp::ToJson(*fixture));
      ASSERT_EQ(t->source.tensors.size(), mine.tensors.size());
      for (const sp::SourceTensor& s : t->source.tensors) {
        const auto at = std::ranges::find(mine.tensors, s.name, &sp::SourceTensor::name);
        ASSERT_NE(at, mine.tensors.end()) << s.name;
        EXPECT_EQ(at->format, s.format) << s.name;
        EXPECT_EQ(at->part, s.part) << s.name;
      }
    }
    // Today's state, and every target step kind builds.
    const auto layout = co::StageLayout(t->spec, 0, 8192, 512);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    EXPECT_EQ(layout->bytes, md::Qwen38StateBytes(p, 8192, 512));
    const std::array<const md::StateLayoutDerived*, 1> layouts{&*layout};
    const std::array<std::span<const co::BoundTensor>, 1> artifacts{t->source.tensors};
    const co::Bindings bindings{
        .tensors = t->source.tensors, .state = &*layout, .stages = layouts, .artifacts = artifacts};
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
  if (opened == 0) GTEST_SKIP() << "no Qwen3.8 artifact installed";
  ASSERT_TRUE(kinds.has_value());
  EXPECT_EQ(kinds->size(), p.layers);
}

TEST(ComponentsQwen38Models, TheMtpDrafterComposesWithItsTargets) {
  const auto& p = md::Qwen38Flash();
  const auto mtp_root = Installed(kMtp);
  if (!mtp_root) GTEST_SKIP() << "no Qwen3.8 MTP drafter installed";
  auto drafter = ar::Artifact::Open(*mtp_root, {.expected_id = {}});
  ASSERT_TRUE(drafter.has_value());
  auto mb = md::BindQwen38Mtp(p, *drafter);
  ASSERT_TRUE(mb.has_value()) << mb.error();
  auto values = sp::mappings::Qwen38MtpV0Values(p, *mb);
  ASSERT_TRUE(values.has_value()) << values.error();
  const auto& mapping = sp::mappings::Qwen38MtpMapping();
  EXPECT_EQ(sp::FindMapping(drafter->model().architecture), &mapping);
  auto source = sp::SourceOf(*drafter, &mapping);
  ASSERT_TRUE(source.has_value()) << source.error();
  std::size_t composed = 0;
  for (const std::string_view id : {kCutlass, kGgmlExperts}) {
    SCOPED_TRACE(id);
    auto t = Open(id);
    if (!t) continue;
    ++composed;
    auto spec =
        sp::ComposeDrafterV0(t->spec, sp::mappings::Qwen38Mapping(), *drafter, mapping, *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    const std::array<std::span<const co::BoundTensor>, 2> artifacts{t->source.tensors,
                                                                    source->tensors};
    const auto valid = co::ValidateSpec(*spec, artifacts);
    ASSERT_TRUE(valid.has_value()) << valid.error();
    // Its own import's folds, the target's.
    EXPECT_EQ(spec->stages.at(1).folds, spec->stages.at(0).folds);
    // Its head: its own selected rows where the artifact has them.
    const auto& head = spec->stages.at(1).heads.output;
    ASSERT_TRUE(head.has_value());
    EXPECT_EQ(std::ranges::any_of(head->weights,
                                  [](const sp::WeightBinding& w) { return w.role == "ids"; }),
              mb->selected_head());
    // State layouts.
    const auto t_layout = co::StageLayout(*spec, 0, 8192, 512);
    const auto d_layout = co::StageLayout(*spec, 1, 8192, 512);
    ASSERT_TRUE(t_layout.has_value() && d_layout.has_value());
    EXPECT_EQ(t_layout->bytes, md::Qwen38StateBytes(p, 8192, 512));
    EXPECT_EQ(d_layout->bytes, md::Qwen38MtpStateBytes(p, 8192, 512));
    const std::array<const md::StateLayoutDerived*, 2> layouts{&*t_layout, &*d_layout};
    // The target writes the drafter's streams; the drafter's pass and its
    // catch-up build.
    const auto prefill = co::BuildStep(*spec, 0, {.segments = {Seg(16, 0, 1)}},
                                       {.tensors = t->source.tensors,
                                        .state = &*t_layout,
                                        .stages = layouts,
                                        .artifacts = artifacts});
    ASSERT_TRUE(prefill.has_value()) << prefill.error();
    EXPECT_TRUE(std::ranges::any_of(prefill->state,
                                    [](const co::StepState& st) { return st.tap && st.written; }));
    const co::Bindings d{
        .tensors = source->tensors, .state = &*d_layout, .stages = layouts, .artifacts = artifacts};
    const auto pass = co::BuildStep(
        *spec, 1, {.stage = 1, .phase = pl::Phase::kDraft, .segments = {Seg(1, 16, 1, true)}}, d);
    ASSERT_TRUE(pass.has_value()) << pass.error();
    ASSERT_TRUE(pass->features.valid());
    EXPECT_EQ(pass->graph.tensor(pass->features).ne(0), p.hc_width());
    EXPECT_TRUE(pass->greedy.valid());
    const auto catch_up = co::BuildStep(*spec, 1, {.stage = 1, .segments = {Seg(3, 16, 0)}}, d);
    ASSERT_TRUE(catch_up.has_value()) << catch_up.error();
  }
  if (composed == 0) GTEST_SKIP() << "no ModelOpt Qwen3.8 target installed";
}

}  // namespace
