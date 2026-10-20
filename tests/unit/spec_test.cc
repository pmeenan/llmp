// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model spec (spec/spec.h): its canonical JSON round trip and digest,
// its refusals of untrusted JSON, the bound digest; the architecture
// mappings (spec/mapping.h) validated as data and applied to the Gemma 2
// and Gemma 3 tensor tables (tests/unit/data) with today's profiles.

#include "spec/spec.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/format_map.h"
#include "artifact/json.h"
#include "artifact/representation.h"
#include "base/sha256.h"
#include "components/step.h"
#include "families/dsv4.h"
#include "families/gemma.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"
#include "spec_fixtures.h"

namespace llmp::spec {
namespace {

namespace fam = llmp::test_support::families;

ComponentRef C(std::string kind, std::vector<Param> params, std::vector<WeightBinding> weights) {
  return {.kind = std::move(kind),
          .version = 1,
          .params = std::move(params),
          .weights = std::move(weights)};
}

WeightBinding W(std::string role, std::string tensor) {
  return {.role = std::move(role), .tensor = std::move(tensor)};
}

// A small spec using every part of the format: a target with two
// hyper-connection streams, a side input, taps, an MTP head and a softcap,
// and a drafter reading the target's tap and borrowing its state.
ModelSpec Fixture() {
  Stage target{.name = "target", .role = StageRole::kTarget, .artifact = 0, .context = 4096};
  target.embedding = C("embedding.token", {{"scale", 8.0}}, {W("table", "token_embd.weight")});
  target.stream = {.streams = 2,
                   .width = 64,
                   .mixing = C("residual.hyper", {{"kind", std::string("sigmoid")}}, {})};
  target.side = {SideInput{
      .name = "ngram", .source = C("side.table", {}, {W("table", "ngram.weight")}), .layers = {0}}};
  for (std::uint32_t il = 0; il < 2; ++il) {
    target.layers.push_back(Layer{
        .components = {
            C("attention.dense",
              {{"eps", 1e-6},
               {"heads", std::int64_t{4}},
               {"qk_norm", false},
               {"rope_sections", std::vector<std::int64_t>{3, 3, 2}},
               {"rotation", std::string("neox")},
               {"window", std::int64_t{il == 0 ? 128 : 0}}},
              {W("k", std::format("blk.{}.attn_k.weight", il)),
               WeightBinding{.role = "q",
                             .tensor = std::format("blk.{}.attn_q.weight", il),
                             .part = format::PartRole::kBlockScales},
               W("q", std::format("blk.{}.attn_q.weight", il))}),
            C("ffn.moe", {{"factors", std::vector<double>{0.5, -2.25}}, {"used", std::int64_t{2}}},
              {WeightBinding{.role = "gate",
                             .tensor = std::format("blk.{}.ffn_gate_exps.weight", il),
                             .experts = true}}),
        }});
  }
  target.taps = {FeatureTap{.name = "features", .layer = 1, .point = TapPoint::kLayerOut},
                 FeatureTap{.name = "post_norm", .layer = 0, .point = TapPoint::kPostFinalNorm},
                 FeatureTap{.name = "streams", .layer = 0, .point = TapPoint::kStreams}};
  target.heads.final_norm = C("norm.rms", {{"eps", 1e-6}}, {W("weight", "output_norm.weight")});
  target.heads.output = C("head.linear", {}, {W("weight", "token_embd.weight")});
  target.heads.softcap = 30.0;
  target.heads.mtp = {
      C("head.mtp", {{"depth", std::int64_t{1}}}, {W("proj", "mtp.0.proj.weight")})};

  Stage drafter{.name = "assistant", .role = StageRole::kDrafter, .artifact = 1};
  drafter.stream = {.width = 32};
  drafter.layers.push_back(
      Layer{.components = {C("attention.query_only", {{"kv_layer", std::int64_t{1}}}, {})}});
  drafter.reads = {Read{.stage = "target", .kind = ReadKind::kTap, .name = "post_norm"},
                   Read{.stage = "target", .kind = ReadKind::kState, .name = "kv"}};

  target.folds = {"norm.plus_one"};

  ModelSpec spec{.pipeline = PipelineKind::kLlm, .stages = {target, drafter}};
  Canonicalize(spec);
  return spec;
}

std::string Replace(std::string text, std::string_view from, std::string_view to) {
  const std::size_t at = text.find(from);
  EXPECT_NE(at, std::string::npos) << from;
  if (at != std::string::npos) {
    text.replace(at, from.size(), to);
  }
  return text;
}

TEST(SpecTest, CanonicalJsonRoundTrips) {
  const ModelSpec spec = Fixture();
  ASSERT_TRUE(Validate(spec).has_value()) << Validate(spec).error();
  const std::string text = ToJson(spec);
  EXPECT_TRUE(text.ends_with("}\n"));
  // Reals in their shortest round-trip form, flags, names and both lists.
  EXPECT_NE(text.find(R"("eps":"1e-06")"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("factors":["0.5","-2.25"])"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("rope_sections":[3,3,2])"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("qk_norm":false)"), std::string::npos) << text;
  EXPECT_NE(text.find(R"({"experts":true,"part":"data","role":"gate")"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("softcap":"30")"), std::string::npos) << text;
  // The artifact JSON reader calls the writer's bytes canonical.
  const auto doc = artifact::json::Parse(text);
  ASSERT_TRUE(doc.has_value());
  EXPECT_TRUE(doc->canonical());

  auto back = FromJson(text);
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, spec);
  EXPECT_EQ(ToJson(*back), text);
  EXPECT_EQ(spec.Digest(), base::Sha256().Update(text).Finish());
  // Order of construction does not matter once canonical.
  ModelSpec shuffled = spec;
  std::ranges::reverse(shuffled.stages[0].taps);
  std::ranges::reverse(shuffled.stages[0].layers[0].components[0].params);
  Canonicalize(shuffled);
  EXPECT_EQ(shuffled.Digest(), spec.Digest());
  // Any change is another digest.
  ModelSpec other = spec;
  other.stages[0].layers[1].components[0].params[0].value = 1e-5;
  EXPECT_NE(other.Digest(), spec.Digest());
  // Image stage links and exports round-trip canonically and validate.
  Stage vae{.name = "vae", .role = StageRole::kVae, .artifact = 0};
  vae.stream = {.width = 16};
  vae.layers.push_back(Layer{.components = {C("norm.rms", {{"eps", 1e-6}}, {W("weight", "w")}),
                                            C("residual.spatial", {}, {})}});
  vae.links = {GraphLink{.input = 0}, GraphLink{.input = 1, .auxiliary = 0}};
  vae.exports = {2};
  ModelSpec img{.pipeline = PipelineKind::kImage, .stages = {vae}};
  Canonicalize(img);
  ASSERT_TRUE(Validate(img).has_value()) << Validate(img).error();
  const std::string img_json = ToJson(img);
  auto img_back = FromJson(img_json);
  ASSERT_TRUE(img_back.has_value()) << img_back.error();
  EXPECT_EQ(*img_back, img);
  EXPECT_EQ(ToJson(*img_back), img_json);
  ModelSpec bad_links = img;
  bad_links.stages[0].links.pop_back();
  EXPECT_FALSE(Validate(bad_links).has_value());
  ModelSpec bad_exports = img;
  bad_exports.stages[0].exports = {0};
  EXPECT_FALSE(Validate(bad_exports).has_value());
}

TEST(SpecTest, RefusesWhatIsNotCanonicalOrValid) {
  const std::string good = ToJson(Fixture());
  ASSERT_TRUE(FromJson(good).has_value());
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"whitespace", Replace(good, R"("version":1)", R"("version": 1)")},
      {"no newline", good.substr(0, good.size() - 1)},
      {"unknown key", Replace(good, R"("folds":)", R"("extra":1,"folds":)")},
      {"duplicate key",
       Replace(good, R"("pipeline":"llm")", R"("pipeline":"llm","pipeline":"llm")")},
      {"newer version", Replace(good, R"("version":1})", R"("version":2})")},
      {"unknown fold", Replace(good, R"("norm.plus_one")", R"("norm.times_two")")},
      {"fold twice", Replace(good, R"("folds":["norm.plus_one"])",
                             R"("folds":["norm.plus_one","norm.plus_one"])")},
      // Folds are a stage's, never the model's.
      {"model-wide folds", Replace(good, R"({"pipeline":)", R"({"folds":[],"pipeline":)")},
      {"stage without folds", Replace(good, R"("folds":[],)", "")},
      {"unknown pipeline", Replace(good, R"("pipeline":"llm")", R"("pipeline":"video")")},
      {"artifact index", Replace(good, R"("artifact":1)", R"("artifact":16)")},
      {"component version", Replace(good, R"("kind":"head.linear","params":{},"version":1)",
                                    R"("kind":"head.linear","params":{},"version":65536)")},
      {"version 0", Replace(good, R"("kind":"head.linear","params":{},"version":1)",
                            R"("kind":"head.linear","params":{},"version":0)")},
      {"integer past 64 bits", Replace(good, R"("used":2)", R"("used":99999999999999999999)")},
      {"real not shortest", Replace(good, R"("scale":"8")", R"("scale":"8.0")")},
      {"real out of range", Replace(good, R"("scale":"8")", R"("scale":"1e400")")},
      {"mixed list", Replace(good, R"([3,3,2])", R"([3,"3",2])")},
      {"string not a name", Replace(good, R"("neox")", R"("Neox")")},
      {"unknown part", Replace(good, R"("part":"block_scales")", R"("part":"scales")")},
      {"experts false", Replace(good, R"({"experts":true,)", R"({"experts":false,)")},
      {"weights unsorted", Replace(good, R"("role":"k","tensor":"blk.0.attn_k.weight")",
                                   R"("role":"r","tensor":"blk.0.attn_k.weight")")},
      {"tap past the layers",
       Replace(good, R"("layer":1,"name":"features")", R"("layer":2,"name":"features")")},
      {"read of no tap", Replace(good, R"("name":"post_norm","stage":"target")",
                                 R"("name":"posts","stage":"target")")},
      {"read of itself",
       Replace(good, R"("name":"kv","stage":"target")", R"("name":"kv","stage":"assistant")")},
      {"streams without mixing",
       Replace(good, R"("streams":1,"width":32)", R"("streams":2,"width":32)")},
      {"too many streams",
       Replace(good, R"("streams":2,"width":64)", R"("streams":17,"width":64)")},
      {"two targets", Replace(good, R"("role":"drafter")", R"("role":"target")")},
      {"stage name twice", Replace(good, R"("name":"assistant")", R"("name":"target")")},
      {"softcap zero", Replace(good, R"("softcap":"30")", R"("softcap":"0")")},
      {"side layer past",
       Replace(good, R"("layers":[0],"name":"ngram")", R"("layers":[2],"name":"ngram")")},
      {"escape", Replace(good, R"("token_embd.weight")", R"("token\"embd.weight")")},
      // -0 compares equal to 0 but would be a second encoding and digest.
      {"negative zero", Replace(good, R"("scale":"8")", R"("scale":"-0")")},
      {"negative zero listed", Replace(good, R"(["0.5","-2.25"])", R"(["-0","-2.25"])")},
      {"integer -0", Replace(good, R"("used":2)", R"("used":-0)")},
      {"not a number", Replace(good, R"(["0.5","-2.25"])", R"(["nan","-2.25"])")},
      {"negative infinity", Replace(good, R"("scale":"8")", R"("scale":"-inf")")},
      {"exponent not shortest", Replace(good, R"("eps":"1e-06")", R"("eps":"1e-6")")},
      {"leading zero", Replace(good, R"("scale":"8")", R"("scale":"08")")},
      {"plus sign", Replace(good, R"("scale":"8")", R"("scale":"+8")")},
      {"hexadecimal", Replace(good, R"("scale":"8")", R"("scale":"0x8")")},
      {"empty string", Replace(good, R"("neox")", R"("")")},
      {"key not a name", Replace(good, R"("qk_norm":false)", R"("qk-norm":false)")},
      // A ninth level: a list inside a list parameter.
      {"nesting past eight", Replace(good, R"([3,3,2])", R"([[3],3,2])")},
      {"final norm tap at another layer",
       Replace(good, R"("layer":0,"name":"post_norm")", R"("layer":1,"name":"post_norm")")},
      {"reads unsorted", Replace(good,
                                 R"({"kind":"tap","name":"post_norm","stage":"target"},)"
                                 R"({"kind":"state","name":"kv","stage":"target"})",
                                 R"({"kind":"state","name":"kv","stage":"target"},)"
                                 R"({"kind":"tap","name":"post_norm","stage":"target"})")},
      {"kind not dotted names",
       Replace(good, R"("kind":"head.linear")", R"("kind":"head..linear")")},
      {"stage name not a name", Replace(good, R"("name":"assistant")", R"("name":"Assistant")")},
      {"data after", good + "{}\n"},
  };
  for (const auto& [what, text] : bad) {
    const auto parsed = FromJson(text);
    EXPECT_FALSE(parsed.has_value()) << what;
  }
  EXPECT_FALSE(FromJson(std::string(kMaxSpecBytes + 1, ' ')).has_value());
  // A newer spec is refused for its version, whatever keys it holds.
  const auto newer = FromJson(Replace(Replace(good, R"({"pipeline":)", R"({"extra":1,"pipeline":)"),
                                      R"("version":1})", R"("version":2})"));
  ASSERT_FALSE(newer.has_value());
  EXPECT_TRUE(newer.error().starts_with("version")) << newer.error();
  // Validate's own rules on built specs.
  ModelSpec empty_reals = Fixture();
  empty_reals.stages[0].layers[0].components[1].params[0].value = std::vector<double>{};
  EXPECT_FALSE(Validate(empty_reals).has_value());
  ModelSpec nan = Fixture();
  nan.stages[0].heads.softcap = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(Validate(nan).has_value());
  ModelSpec unsorted = Fixture();
  std::ranges::reverse(unsorted.stages[0].taps);
  EXPECT_FALSE(Validate(unsorted).has_value());
  ModelSpec unsorted_folds = Fixture();
  unsorted_folds.stages[1].folds = {"ssm.a_neg_exp", "norm.plus_one"};
  EXPECT_FALSE(Validate(unsorted_folds).has_value());
  Canonicalize(unsorted_folds);
  EXPECT_TRUE(Validate(unsorted_folds).has_value());
  ModelSpec uncatalogued = Fixture();
  uncatalogued.stages[1].folds = {"norm.times_two"};
  EXPECT_FALSE(Validate(uncatalogued).has_value());
  // Stages of one artifact carry that artifact's folds: the same list.
  ModelSpec one_artifact = Fixture();
  one_artifact.stages[1].artifact = one_artifact.stages[0].artifact;
  const auto differing = Validate(one_artifact);
  ASSERT_FALSE(differing.has_value());
  EXPECT_NE(differing.error().find("runs from the same artifact"), std::string::npos)
      << differing.error();
  one_artifact.stages[1].folds = one_artifact.stages[0].folds;
  EXPECT_TRUE(Validate(one_artifact).has_value()) << Validate(one_artifact).error();
  ModelSpec negative_zero = Fixture();
  negative_zero.stages[0].embedding->params[0].value = -0.0;
  EXPECT_FALSE(Validate(negative_zero).has_value());
  ModelSpec negative_zero_listed = Fixture();
  negative_zero_listed.stages[0].layers[0].components[1].params[0].value =
      std::vector<double>{0.5, -0.0};
  EXPECT_FALSE(Validate(negative_zero_listed).has_value());
}

// A binding's tensor is its holding stage's: the reading stage's own, or
// the stage WeightBinding::stage names, whose artifact and folds it has.
TEST(SpecTest, ABindingsTensorIsItsHoldingStages) {
  const ModelSpec spec = Fixture();
  const Stage& target = spec.stages[0];
  const Stage& drafter = spec.stages[1];
  ASSERT_EQ(target.name, "target");
  ASSERT_EQ(drafter.name, "assistant");
  const WeightBinding own = W("weight", "blk.0.attn_norm.weight");
  WeightBinding shared = W("table", "token_embd.weight");
  shared.stage = "target";
  EXPECT_EQ(HoldingStage(spec, drafter, own), &drafter);
  const Stage* holder = HoldingStage(spec, drafter, shared);
  ASSERT_EQ(holder, &target);
  EXPECT_EQ(holder->artifact, 0U);
  EXPECT_EQ(holder->folds, std::vector<std::string>{"norm.plus_one"});
  EXPECT_TRUE(drafter.folds.empty());
  WeightBinding dangling = shared;
  dangling.stage = "nowhere";
  EXPECT_EQ(HoldingStage(spec, drafter, dangling), nullptr);
}

TEST(SpecTest, CanonicalEdgesRoundTrip) {
  // Reals at the edges of their text, a name that reads like a number's
  // spelling, and the integer extremes: each one encoding, read back.
  ModelSpec spec = Fixture();
  std::vector<Param>& params = spec.stages[0].layers[0].components[0].params;
  params.push_back({"big", std::numeric_limits<double>::max()});
  params.push_back({"huge", std::numeric_limits<std::int64_t>::min()});
  params.push_back({"name_nan", std::string("nan")});
  params.push_back({"subnormal", std::numeric_limits<double>::denorm_min()});
  params.push_back({"tenth", 0.1});
  params.push_back({"zero", 0.0});
  params.push_back({"empty", std::vector<std::int64_t>{}});
  Canonicalize(spec);
  ASSERT_TRUE(Validate(spec).has_value()) << Validate(spec).error();
  const std::string text = ToJson(spec);
  EXPECT_NE(text.find(R"("big":"1.7976931348623157e+308")"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("huge":-9223372036854775808)"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("subnormal":"5e-324")"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("tenth":"0.1")"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("zero":"0")"), std::string::npos) << text;
  EXPECT_NE(text.find(R"("empty":[])"), std::string::npos) << text;
  auto back = FromJson(text);
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, spec);
  EXPECT_EQ(ToJson(*back), text);
  // "nan" is a name, not a real.
  EXPECT_EQ(back->stages[0].layers[0].components[0].Find("name_nan")->value,
            ParamValue(std::string("nan")));
}

TEST(SpecTest, BoundDigestCarriesTheArtifacts) {
  const ModelSpec spec = Fixture();
  const std::array<std::string_view, 2> a = {"aaaa", "bbbb"};
  const std::array<std::string_view, 2> b = {"aaaa", "cccc"};
  const auto da = BoundDigest(spec, a);
  const auto db = BoundDigest(spec, b);
  ASSERT_TRUE(da.has_value() && db.has_value());
  EXPECT_NE(*da, *db);
  EXPECT_NE(*da, spec.Digest());
  EXPECT_EQ(*da, *BoundDigest(spec, a));
  // The drafter's artifact 1 must be bound, and IDs are not empty.
  const std::array<std::string_view, 1> one = {"aaaa"};
  EXPECT_FALSE(BoundDigest(spec, one).has_value());
  const std::array<std::string_view, 2> blank = {"aaaa", ""};
  EXPECT_FALSE(BoundDigest(spec, blank).has_value());
  // Length-prefixed: moving bytes between IDs is another binding.
  const std::array<std::string_view, 2> shifted = {"aaaab", "bbb"};
  EXPECT_NE(*BoundDigest(spec, shifted), *da);
  // A spec Validate refuses (here unsorted, so not its canonical twin's
  // digest) binds nothing.
  ModelSpec unsorted = spec;
  std::ranges::reverse(unsorted.stages[0].taps);
  EXPECT_FALSE(BoundDigest(unsorted, a).has_value());
}

// ---------------------------------------------------------------- mappings

TEST(MappingTest, CompiledMappingsAreValid) {
  ASSERT_FALSE(Mappings().empty());
  for (const Mapping* m : Mappings()) {
    const auto ok = ValidateMapping(*m);
    EXPECT_TRUE(ok.has_value()) << m->architecture << ": " << (ok ? "" : ok.error());
    for (const auto container : m->containers)
      EXPECT_EQ(FindMapping(m->architecture, container, m->version), m);
  }
  EXPECT_EQ(FindMapping("gemma9"), nullptr);
  EXPECT_EQ(FindMapping("gemma4-assistant")->role, StageRole::kDrafter);
  EXPECT_EQ(FindFold("norm.plus_one")->name, "norm.plus_one");
  EXPECT_EQ(FindFold("norm.times_two"), nullptr);
}

TEST(MappingTest, InvalidMappingsAreRefused) {
  const Mapping& good = mappings::Gemma2Mapping();
  // Each policy rule's text: printable lines (the planner's own rules are
  // its), not empty, at most kMaxPolicyText bytes.
  const std::string long_text(kMaxPolicyText + 1, '\n');
  for (const std::string_view policy :
       {std::string_view("max_decode_rows 16"), std::string_view("disable x\n\t\n"),
        std::string_view(), std::string_view(long_text)}) {
    Mapping m = good;
    const std::array rules{PolicyRule{.text = "max_decode_rows 16\n"}, PolicyRule{.text = policy}};
    m.policies = rules;
    EXPECT_FALSE(ValidateMapping(m).has_value()) << policy;
  }
  const auto with_tensor = [&](TensorRule rule) {
    std::vector<TensorRule> rules(good.tensors.begin(), good.tensors.end());
    rules.push_back(rule);
    Mapping m = good;
    m.tensors = rules;
    return ValidateMapping(m).has_value();
  };
  EXPECT_FALSE(with_tensor({.pattern = "blk.{layer}.{layer}.x", .slot = "ffn", .role = "x"}));
  EXPECT_FALSE(with_tensor({.pattern = "token_{layer}.x", .slot = "embedding", .role = "x"}));
  EXPECT_FALSE(with_tensor({.pattern = "blk.{layer}.x", .slot = "nothing", .role = "x"}));
  EXPECT_FALSE(with_tensor({.pattern = "blk.{layer}.ffn_up.weight", .slot = "ffn", .role = "x"}));
  EXPECT_FALSE(with_tensor({.pattern = "blk.{layer}.y", .slot = "ffn", .role = "up"}));
  EXPECT_FALSE(with_tensor({.pattern = "blk.{layer}.y z", .slot = "ffn", .role = "y"}));
  const auto with_param = [&](ParamRule rule) {
    std::vector<ParamRule> rules(good.params.begin(), good.params.end());
    rules.push_back(rule);
    Mapping m = good;
    m.params = rules;
    return ValidateMapping(m).has_value();
  };
  EXPECT_TRUE(with_param({.slot = "ffn", .param = "extra", .key = "k", .type = ValueType::kInt}));
  EXPECT_FALSE(with_param({.slot = "ffn",
                           .param = "x",
                           .key = "k",
                           .type = ValueType::kInt,
                           .transform = Transform::kSqrt}));
  EXPECT_FALSE(with_param({.slot = "ffn", .param = "x", .layer_types = "z", .key = "k"}));
  EXPECT_FALSE(with_param({.slot = "final_norm", .param = "x", .layer_types = "l", .key = "k"}));
  EXPECT_FALSE(with_param({.slot = "stage", .param = "width", .key = "k"}));
  EXPECT_FALSE(with_param(
      {.slot = "ffn", .param = "x", .type = ValueType::kString, .value = {.text = "Bad"}}));
  EXPECT_FALSE(with_param(
      {.slot = "", .param = "x", .key = "k", .type = ValueType::kString, .value = {.text = "a"}}));
  Mapping folds = good;
  const std::array unknown = {FoldRule{.fold = "norm.times_two", .container = "gguf"}};
  folds.folds = unknown;
  EXPECT_FALSE(ValidateMapping(folds).has_value());
  Mapping types = good;
  types.layer_types = "lG";
  EXPECT_FALSE(ValidateMapping(types).has_value());
  const std::array slots = {SlotRule{.slot = "stage", .kind = "a.b"}};
  Mapping pseudo = good;
  pseudo.slots = slots;
  EXPECT_FALSE(ValidateMapping(pseudo).has_value());
  // Rules for one role and part split the layer types; none may bind a
  // layer type twice (no types: every type).
  const Mapping& g4 = mappings::Gemma4Mapping();
  const auto with_g4_tensor = [&](TensorRule rule) {
    std::vector<TensorRule> rules(g4.tensors.begin(), g4.tensors.end());
    rules.push_back(rule);
    Mapping m = g4;
    m.tensors = rules;
    return ValidateMapping(m).has_value();
  };
  EXPECT_TRUE(with_g4_tensor({.pattern = "blk.{layer}.attn_v_g.weight",
                              .slot = "attention",
                              .role = "v",
                              .layer_types = "g"}));
  EXPECT_FALSE(with_g4_tensor({.pattern = "blk.{layer}.attn_v_g.weight",
                               .slot = "attention",
                               .role = "v",
                               .layer_types = "lg"}));
  EXPECT_FALSE(
      with_g4_tensor({.pattern = "blk.{layer}.attn_v_g.weight", .slot = "attention", .role = "v"}));
  EXPECT_FALSE(with_tensor(
      {.pattern = "blk.{layer}.ffn_up_l.weight", .slot = "ffn", .role = "up", .layer_types = "l"}));
  // A second embedding, final norm or output only as the other gate's
  // alternative of a gated one.
  const auto with_slot = [&](const Mapping& base, std::vector<SlotRule> more) {
    std::vector<SlotRule> all(base.slots.begin(), base.slots.end());
    all.insert(all.end(), more.begin(), more.end());
    Mapping m = base;
    m.slots = all;
    return ValidateMapping(m).has_value();
  };
  EXPECT_FALSE(with_slot(
      good, {{.slot = "norm2", .site = Site::kFinalNorm, .kind = "norm.rms", .gate = "k"}}));
  Mapping gated = good;
  std::vector<SlotRule> gated_slots;
  for (SlotRule s : good.slots) {
    if (s.site == Site::kFinalNorm) s.gate = "k";
    gated_slots.push_back(s);
  }
  gated.slots = gated_slots;
  ASSERT_TRUE(ValidateMapping(gated).has_value());
  EXPECT_TRUE(with_slot(gated, {{.slot = "norm2",
                                 .site = Site::kFinalNorm,
                                 .kind = "norm.rms",
                                 .gate = "k",
                                 .gate_absent = true}}));
  EXPECT_FALSE(with_slot(
      gated, {{.slot = "norm2", .site = Site::kFinalNorm, .kind = "norm.rms", .gate = "k"}}));
  EXPECT_FALSE(with_slot(gated, {{.slot = "norm2", .site = Site::kFinalNorm, .kind = "norm.rms"}}));
}

const Param& P(const ComponentRef& c, std::string_view name) {
  const Param* p = c.Find(name);
  EXPECT_NE(p, nullptr) << c.kind << "." << name;
  static const Param none{};
  return p != nullptr ? *p : none;
}

std::string Tensor(const ComponentRef& c, std::string_view role) {
  const auto found = std::ranges::find(c.weights, role, &WeightBinding::role);
  return found == c.weights.end() ? std::string() : found->tensor;
}

TEST(MappingTest, Gemma2SpecFromItsTensorsAndProfile) {
  const auto values = fam::Gemma2Values(fam::Gemma2_2B());
  ASSERT_TRUE(values.has_value()) << values.error();
  const auto spec = Synthesize(mappings::Gemma2Mapping(), test_support::Gemma2Source(), *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  const Stage& s = spec->stages[0];
  EXPECT_EQ(s.folds, std::vector<std::string>{"norm.plus_one"});
  EXPECT_EQ(s.context, 8192U);
  EXPECT_EQ(s.stream.width, 2304U);
  EXPECT_EQ(s.stream.streams, 1U);
  EXPECT_EQ(s.heads.softcap, 30.0);
  ASSERT_TRUE(s.embedding && s.heads.final_norm && s.heads.output);
  EXPECT_EQ(P(*s.embedding, "scale").value, ParamValue(48.0));
  // The tied head reads the embedding.
  EXPECT_EQ(Tensor(*s.heads.output, "weight"), "token_embd.weight");
  ASSERT_EQ(s.layers.size(), 26U);
  for (std::uint32_t il = 0; il < 26; ++il) {
    ASSERT_EQ(s.layers[il].components.size(), 2U);
    const ComponentRef& a = s.layers[il].components[0];
    const ComponentRef& f = s.layers[il].components[1];
    EXPECT_EQ(a.kind, "attention.dense");
    EXPECT_EQ(f.kind, "ffn.glu");
    EXPECT_EQ(P(a, "window").value, ParamValue(std::int64_t{il % 2 == 0 ? 4096 : 0})) << il;
    EXPECT_EQ(P(a, "scale").value, ParamValue(0.0625));
    EXPECT_EQ(P(a, "softcap").value, ParamValue(50.0));
    EXPECT_EQ(P(a, "rope_base").value, ParamValue(10000.0));
    EXPECT_EQ(P(a, "rope_dims").value, ParamValue(std::int64_t{256}));
    EXPECT_EQ(P(a, "eps").value, ParamValue(static_cast<double>(1.0e-6F)));
    EXPECT_EQ(a.weights.size(), 6U);
    EXPECT_EQ(Tensor(a, "q"), std::format("blk.{}.attn_q.weight", il));
    EXPECT_EQ(P(f, "hidden").value, ParamValue(std::int64_t{9216}));
    EXPECT_EQ(P(f, "activation").value, ParamValue(std::string("gelu_tanh")));
    EXPECT_EQ(f.weights.size(), 5U);
  }
  // Canonical and stable: it round-trips and a second synthesis agrees.
  auto back = FromJson(ToJson(*spec));
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, *spec);
  const auto again = Synthesize(mappings::Gemma2Mapping(), test_support::Gemma2Source(), *values);
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(again->Digest(), spec->Digest());
  // A source of another container carries no GGUF fold.
  Source other = test_support::Gemma2Source();
  other.container = "safetensors";
  Mapping synthetic = mappings::Gemma2Mapping();
  synthetic.containers = {};  // exercise fold semantics in an explicitly neutral test mapping
  const auto unfolded = Synthesize(synthetic, other, *values);
  ASSERT_TRUE(unfolded.has_value());
  EXPECT_TRUE(unfolded->stages[0].folds.empty());
  EXPECT_NE(unfolded->Digest(), spec->Digest());
  // One whose importer applied the fold (a v1 artifact's record) carries
  // it as GGUF's converter's: the same spec.
  other.folds = {"norm.plus_one"};
  const auto imported = Synthesize(synthetic, other, *values);
  ASSERT_TRUE(imported.has_value()) << imported.error();
  EXPECT_EQ(imported->Digest(), spec->Digest());
  // A fold outside the catalog is refused, never ignored.
  other.folds = {"norm.times_two"};
  EXPECT_FALSE(Synthesize(mappings::Gemma2Mapping(), other, *values).has_value());
}

TEST(MappingTest, Gemma3SpecFromItsTensorsAndProfile) {
  const auto values = fam::Gemma3Values(fam::Gemma3_4BQat());
  ASSERT_TRUE(values.has_value()) << values.error();
  const auto spec = Synthesize(mappings::Gemma3Mapping(), test_support::Gemma3Source(), *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const Stage& s = spec->stages[0];
  EXPECT_EQ(s.context, 131072U);
  EXPECT_EQ(s.stream.width, 2560U);
  EXPECT_FALSE(s.heads.softcap.has_value());
  EXPECT_EQ(P(*s.embedding, "scale").value, ParamValue(std::sqrt(2560.0)));
  ASSERT_EQ(s.layers.size(), 34U);
  for (std::uint32_t il = 0; il < 34; ++il) {
    const ComponentRef& a = s.layers[il].components.at(0);
    const bool local = (il + 1) % 6 != 0;
    EXPECT_EQ(P(a, "window").value, ParamValue(std::int64_t{local ? 1024 : 0})) << il;
    EXPECT_EQ(P(a, "rope_base").value, ParamValue(local ? 10000.0 : 1000000.0)) << il;
    EXPECT_EQ(P(a, "rope_scale").value, ParamValue(local ? 1.0 : 0.125)) << il;
    EXPECT_EQ(P(a, "qk_norm").value, ParamValue(true));
    EXPECT_EQ(P(a, "softcap").value, ParamValue(0.0));
    EXPECT_EQ(Tensor(a, "k_norm"), std::format("blk.{}.attn_k_norm.weight", il));
    EXPECT_EQ(a.weights.size(), 8U);
  }
  EXPECT_EQ(Tensor(*s.heads.output, "weight"), "token_embd.weight");
}

TEST(MappingTest, OptionalCheckpointKeysAreRead) {
  // Keys llama.cpp reads when a checkpoint has them: a sliding-window
  // rotation base, Gemma 3's final softcap, and a layer pattern that must
  // agree with the mapping's.
  const auto with = [](std::vector<SourceValue> v, std::string_view key, ParamValue value) {
    v.push_back({.key = std::string(key), .value = std::move(value)});
    return v;
  };
  const auto g2 = *fam::Gemma2Values(fam::Gemma2_2B());
  const Mapping& m2 = mappings::Gemma2Mapping();
  const auto swa2 =
      Synthesize(m2, test_support::Gemma2Source(), with(g2, "gemma2.rope.freq_base_swa", 500.0));
  ASSERT_TRUE(swa2.has_value()) << swa2.error();
  EXPECT_EQ(P(swa2->stages[0].layers[0].components[0], "rope_base").value, ParamValue(500.0));
  EXPECT_EQ(P(swa2->stages[0].layers[1].components[0], "rope_base").value, ParamValue(10000.0));
  EXPECT_TRUE(Synthesize(m2, test_support::Gemma2Source(),
                         with(g2, "gemma2.attention.sliding_window_pattern", std::int64_t{2}))
                  .has_value());
  EXPECT_FALSE(Synthesize(m2, test_support::Gemma2Source(),
                          with(g2, "gemma2.attention.sliding_window_pattern", std::int64_t{3}))
                   .has_value());

  const auto g3 = *fam::Gemma3Values(fam::Gemma3_4BQat());
  const Mapping& m3 = mappings::Gemma3Mapping();
  auto v3 = with(g3, "gemma3.final_logit_softcapping", 30.0);
  v3 = with(v3, "gemma3.rope.freq_base_swa", 20000.0);
  const auto s3 = Synthesize(m3, test_support::Gemma3Source(), v3);
  ASSERT_TRUE(s3.has_value()) << s3.error();
  EXPECT_EQ(s3->stages[0].heads.softcap, 30.0);
  EXPECT_EQ(P(s3->stages[0].layers[0].components[0], "rope_base").value, ParamValue(20000.0));
  EXPECT_EQ(P(s3->stages[0].layers[5].components[0], "rope_base").value, ParamValue(1000000.0));
  // A 0 softcap is refused rather than read as none.
  EXPECT_FALSE(
      Synthesize(m3, test_support::Gemma3Source(), with(g3, "gemma3.final_logit_softcapping", 0.0))
          .has_value());
  EXPECT_FALSE(Synthesize(m3, test_support::Gemma3Source(),
                          with(g3, "gemma3.attention.sliding_window_pattern", std::int64_t{2}))
                   .has_value());
}

TEST(MappingTest, SynthesisRefusesWhatTheMappingDoesNotCover) {
  const auto values = *fam::Gemma2Values(fam::Gemma2_2B());
  const Mapping& m = mappings::Gemma2Mapping();
  {
    Source s = test_support::Gemma2Source();
    s.tensors.push_back({.name = "blk.0.extra.weight"});
    EXPECT_FALSE(Synthesize(m, s, values).has_value());
  }
  {
    Source s = test_support::Gemma2Source();
    std::erase_if(s.tensors, [](const SourceTensor& t) { return t.name == "blk.3.ffn_up.weight"; });
    const auto r = Synthesize(m, s, values);
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("blk.3.ffn_up.weight"), std::string::npos) << r.error();
  }
  {
    // A layer past the count matches no rule.
    Source s = test_support::Gemma2Source();
    s.tensors.push_back({.name = "blk.26.attn_q.weight"});
    EXPECT_FALSE(Synthesize(m, s, values).has_value());
  }
  {
    Source s = test_support::Gemma2Source();
    s.tensors[0].part = format::PartRole::kCodes;
    EXPECT_FALSE(Synthesize(m, s, values).has_value());
  }
  {
    Source s = test_support::Gemma2Source();
    s.architecture = "gemma3";
    EXPECT_FALSE(Synthesize(m, s, values).has_value());
  }
  {
    auto v = values;
    std::erase_if(v, [](const SourceValue& x) { return x.key == "gemma2.attention.head_count"; });
    EXPECT_FALSE(Synthesize(m, test_support::Gemma2Source(), v).has_value());
    auto twice = values;
    twice.push_back(values.front());
    EXPECT_FALSE(Synthesize(m, test_support::Gemma2Source(), twice).has_value());
    auto wrong = values;
    wrong.push_back({.key = "gemma2.rope.scaling.type", .value = std::string("yarn")});
    EXPECT_FALSE(Synthesize(m, test_support::Gemma2Source(), wrong).has_value());
    auto typed = values;
    std::ranges::find(typed, std::string_view("gemma2.attention.head_count"), &SourceValue::key)
        ->value = 8.0;
    EXPECT_FALSE(Synthesize(m, test_support::Gemma2Source(), typed).has_value());
  }
  {
    // Gemma 3's global layers scale linearly, only.
    auto v = *fam::Gemma3Values(fam::Gemma3_4BQat());
    std::ranges::find(v, std::string_view("gemma3.rope.scaling.type"), &SourceValue::key)->value =
        std::string("yarn");
    EXPECT_FALSE(
        Synthesize(mappings::Gemma3Mapping(), test_support::Gemma3Source(), v).has_value());
  }
}

// ------------------------------------------------------------------ Gemma 4

const fam::Gemma4Profile& G4(std::uint32_t size) {
  return size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
}

// The spec from the profile's values is the spec from the checkpoint's
// own GGUF metadata (the fixture's), and holds what today's builder reads.
TEST(MappingTest, Gemma4SpecsFromTheirTablesAndProfiles) {
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    const auto& p = G4(size);
    const auto values = fam::Gemma4Values(p);
    ASSERT_TRUE(values.has_value()) << values.error();
    const Source source = test_support::Gemma4Source(size);
    const auto spec = Synthesize(mappings::Gemma4Mapping(), source, *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    const auto from_gguf =
        Synthesize(mappings::Gemma4Mapping(), source, test_support::Gemma4FixtureValues(size));
    ASSERT_TRUE(from_gguf.has_value()) << from_gguf.error();
    EXPECT_EQ(*from_gguf, *spec);
    // No norm fold: the converter keeps Gemma 4's norm weights as they are.
    EXPECT_TRUE(spec->stages[0].folds.empty());
    const Stage& s = spec->stages[0];
    EXPECT_EQ(s.context, p.context);
    EXPECT_EQ(s.stream.width, p.width);
    EXPECT_EQ(s.heads.softcap, 30.0);
    EXPECT_EQ(Tensor(*s.heads.output, "weight"), "token_embd.weight");
    ASSERT_EQ(s.layers.size(), p.layers);
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      SCOPED_TRACE(il);
      ASSERT_EQ(s.layers[il].components.size(), 3U);
      const ComponentRef& a = s.layers[il].components[0];
      const ComponentRef& f = s.layers[il].components[1];
      const ComponentRef& o = s.layers[il].components[2];
      const bool local = p.local(il);
      EXPECT_EQ(a.kind, "attention.dense");
      EXPECT_EQ(P(a, "kv_heads").value, ParamValue(std::int64_t{p.kv_heads(il)}));
      EXPECT_EQ(P(a, "key_dim").value, ParamValue(std::int64_t{p.head_dim(il)}));
      EXPECT_EQ(P(a, "rope_dims").value,
                ParamValue(std::int64_t{local ? p.local_rope_dims : p.global_rope_dims}));
      EXPECT_EQ(P(a, "rope_base").value,
                ParamValue(static_cast<double>(local ? p.local_rope_base : p.global_rope_base)));
      EXPECT_EQ(P(a, "window").value, ParamValue(std::int64_t{local ? p.window : 0}));
      EXPECT_EQ(P(a, "kv_tie").value, ParamValue(!local));
      EXPECT_EQ(P(a, "rope_freqs").value, ParamValue(!local));
      EXPECT_EQ(P(a, "scale").value, ParamValue(1.0));
      // A global layer: no V of its own, the shared rotation factors.
      EXPECT_EQ(Tensor(a, "v").empty(), !local);
      EXPECT_EQ(Tensor(a, "rope_freqs"), local ? "" : "rope_freqs.weight");
      EXPECT_EQ(f.kind, p.experts != 0 ? "ffn.moe" : "ffn.glu");
      EXPECT_EQ(P(f, "hidden").value, ParamValue(std::int64_t{p.ffn}));
      if (p.experts != 0) {
        EXPECT_EQ(P(f, "experts").value, ParamValue(std::int64_t{p.experts}));
        EXPECT_EQ(P(f, "used").value, ParamValue(std::int64_t{p.experts_used}));
        EXPECT_EQ(P(f, "expert_hidden").value, ParamValue(std::int64_t{p.expert_ffn}));
        const auto array = std::ranges::find(f.weights, "gate_up_experts", &WeightBinding::role);
        ASSERT_NE(array, f.weights.end());
        EXPECT_TRUE(array->experts);
      }
      EXPECT_EQ(o.kind, "scale.learned");
      EXPECT_EQ(Tensor(o, "weight"), std::format("blk.{}.layer_output_scale.weight", il));
    }
    auto back = FromJson(ToJson(*spec));
    ASSERT_TRUE(back.has_value()) << back.error();
    EXPECT_EQ(*back, *spec);
    // Structural eligibility is shared across envelopes, with
    // owner_padded_reads disabled on measured Gemma 4 26B-A4B C4 decode
    // regression; scheduling limits remain envelope-specific below.
    const auto bounded =
        DefaultPolicy(mappings::Gemma4Mapping(), *values, {.context = 4096, .slots = 2});
    const auto envelope =
        DefaultPolicy(mappings::Gemma4Mapping(), *values, {.context = 8192, .slots = 4});
    const auto outside =
        DefaultPolicy(mappings::Gemma4Mapping(), *values, {.context = 8448, .slots = 2});
    const auto unknown = DefaultPolicy(mappings::Gemma4Mapping(), *values);
    ASSERT_TRUE(bounded && envelope && outside && unknown);
    for (const auto& policy : {*bounded, *envelope, *outside, *unknown})
      EXPECT_EQ(policy, "max_decode_rows 16\ndisable owner_padded_reads\n");
    // How serving composes them (M3's Gemma 4 serving): prompts alone with
    // one future planned and none captured; outside, every owner alone.
    for (const Envelope e :
         {Envelope{.context = 4096, .slots = 2}, Envelope{.context = 8192, .slots = 4},
          Envelope{.context = 8448, .slots = 2}, Envelope{}}) {
      const auto rule = DefaultRule(mappings::Gemma4Mapping(), *values, e);
      ASSERT_TRUE(rule.has_value() && *rule != nullptr);
      EXPECT_TRUE((*rule)->prompts_alone);
      EXPECT_EQ((*rule)->owners_alone, e.context == 0 || e.context > 8192);
      EXPECT_EQ((*rule)->prefill_lookahead, 1U);
      EXPECT_FALSE((*rule)->capture_ahead);
    }
  }
}

TEST(MappingTest, Gemma4MappingRefusesWhatItDoesNotCover) {
  const Mapping& m = mappings::Gemma4Mapping();
  const auto values26 = *fam::Gemma4Values(G4(26));
  const auto values31 = *fam::Gemma4Values(G4(31));
  const auto value = [](std::vector<SourceValue>& v, std::string_view key) -> ParamValue& {
    return std::ranges::find(v, key, &SourceValue::key)->value;
  };
  {
    // A per-layer list of another length.
    auto v = values26;
    std::get<std::vector<std::int64_t>>(value(v, "gemma4.attention.head_count_kv")).pop_back();
    EXPECT_FALSE(Synthesize(m, test_support::Gemma4Source(26), v));
  }
  {
    // A sliding-window pattern other than the mapping's layer types.
    auto v = values26;
    std::get<std::vector<std::int64_t>>(value(v, "gemma4.attention.sliding_window_pattern"))[5] = 1;
    EXPECT_FALSE(Synthesize(m, test_support::Gemma4Source(26), v));
  }
  {
    // A V weight in a global layer (its V is K's).
    Source s = test_support::Gemma4Source(31);
    SourceTensor v = *std::ranges::find(s.tensors, "blk.0.attn_v.weight", &SourceTensor::name);
    v.name = "blk.5.attn_v.weight";
    s.tensors.push_back(v);
    EXPECT_FALSE(Synthesize(m, s, values31));
  }
  {
    // The routed tensors of a checkpoint without experts.
    Source s = test_support::Gemma4Source(31);
    const Source moe = test_support::Gemma4Source(26);
    s.tensors.push_back(
        *std::ranges::find(moe.tensors, "blk.0.ffn_gate_inp.weight", &SourceTensor::name));
    EXPECT_FALSE(Synthesize(m, s, values31));
    // The 26B's tensors with no expert count: its routed tensors match no slot.
    auto v = values26;
    std::erase_if(v, [](const SourceValue& x) { return x.key == "gemma4.expert_count"; });
    EXPECT_FALSE(Synthesize(m, test_support::Gemma4Source(26), v));
  }
  {
    // Shared KV layers, which the components do not implement.
    auto v = values31;
    v.push_back({.key = "gemma4.attention.shared_kv_layers", .value = std::int64_t{4}});
    EXPECT_FALSE(Synthesize(m, test_support::Gemma4Source(31), v));
  }
  // A drafter's mapping does not synthesize alone.
  const auto assistant = *fam::Gemma4AssistantValues(fam::Gemma4Assistant26(), G4(26));
  EXPECT_FALSE(Synthesize(mappings::Gemma4AssistantMapping(),
                          test_support::Gemma4AssistantSource(26), assistant));
}

// The assistant composed with its target: its stage reads the target's K
// and V and its final-norm tap, binds the target's token table, and each
// layer reads its target's last layer of its type.
TEST(MappingTest, Gemma4AssistantComposesWithItsTarget) {
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    const auto& p = G4(size);
    const auto& a = size == 26 ? fam::Gemma4Assistant26() : fam::Gemma4Assistant31();
    const auto target = Synthesize(mappings::Gemma4Mapping(), test_support::Gemma4Source(size),
                                   *fam::Gemma4Values(p));
    ASSERT_TRUE(target.has_value()) << target.error();
    const auto values = fam::Gemma4AssistantValues(a, p);
    ASSERT_TRUE(values.has_value()) << values.error();
    const Source source = test_support::Gemma4AssistantSource(size);
    const auto spec = ComposeDrafter(*target, mappings::Gemma4Mapping(),
                                     mappings::Gemma4AssistantMapping(), source, *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    const auto from_gguf =
        ComposeDrafter(*target, mappings::Gemma4Mapping(), mappings::Gemma4AssistantMapping(),
                       source, test_support::Gemma4AssistantFixtureValues(size));
    ASSERT_TRUE(from_gguf.has_value()) << from_gguf.error();
    EXPECT_EQ(*from_gguf, *spec);
    ASSERT_EQ(spec->stages.size(), 2U);
    const Stage& t = spec->stages[0];
    const Stage& d = spec->stages[1];
    EXPECT_EQ(d.name, "assistant");
    EXPECT_EQ(d.role, StageRole::kDrafter);
    EXPECT_EQ(d.artifact, 1U);
    EXPECT_EQ(d.stream.width, fam::kGemma4AssistantWidth);
    EXPECT_FALSE(d.heads.softcap.has_value());
    ASSERT_EQ(t.taps.size(), 1U);
    EXPECT_EQ(t.taps[0].name, "features");
    ASSERT_EQ(d.reads.size(), 3U);
    // Its input rows: the target's table, by stage.
    const auto table = std::ranges::find(d.embedding->weights, "table", &WeightBinding::role);
    ASSERT_NE(table, d.embedding->weights.end());
    EXPECT_EQ(table->stage, "target");
    EXPECT_EQ(table->tensor, "token_embd.weight");
    EXPECT_EQ(P(*d.embedding, "scale").value, ParamValue(std::sqrt(static_cast<double>(p.width))));
    ASSERT_EQ(d.layers.size(), fam::kGemma4AssistantLayers);
    for (std::uint32_t il = 0; il < d.layers.size(); ++il) {
      const ComponentRef& at = d.layers[il].components[0];
      EXPECT_EQ(at.kind, "attention.borrowed");
      EXPECT_EQ(P(at, "target_layer").value,
                ParamValue(std::int64_t{il == 3 ? a.global_target_layer : a.local_target_layer}));
      EXPECT_EQ(P(at, "heads").value, ParamValue(std::int64_t{a.heads}));
    }
    EXPECT_EQ(Tensor(*d.heads.output, "projection"), "nextn.post_projection.weight");
    // The drafter's bindings survive the canonical JSON.
    auto back = FromJson(ToJson(*spec));
    ASSERT_TRUE(back.has_value()) << back.error();
    EXPECT_EQ(*back, *spec);
    EXPECT_TRUE(ToJson(*spec).contains(R"("stage":"target")"));
    // Folds are each stage's own artifact's, never merged: a target whose
    // importer folded its norms leaves the drafter's unfolded, and the
    // other way round.
    {
      Source folded_target = test_support::Gemma4Source(size);
      folded_target.folds = {"norm.plus_one"};
      const auto ft = Synthesize(mappings::Gemma4Mapping(), folded_target, *fam::Gemma4Values(p));
      ASSERT_TRUE(ft.has_value()) << ft.error();
      const auto c = ComposeDrafter(*ft, mappings::Gemma4Mapping(),
                                    mappings::Gemma4AssistantMapping(), source, *values);
      ASSERT_TRUE(c.has_value()) << c.error();
      EXPECT_EQ(c->stages[0].folds, std::vector<std::string>{"norm.plus_one"});
      EXPECT_TRUE(c->stages[1].folds.empty());
      EXPECT_NE(c->Digest(), spec->Digest());
      Source folded_drafter = source;
      folded_drafter.folds = {"norm.plus_one"};
      const auto d2 = ComposeDrafter(*target, mappings::Gemma4Mapping(),
                                     mappings::Gemma4AssistantMapping(), folded_drafter, *values);
      ASSERT_TRUE(d2.has_value()) << d2.error();
      EXPECT_TRUE(d2->stages[0].folds.empty());
      EXPECT_EQ(d2->stages[1].folds, std::vector<std::string>{"norm.plus_one"});
      EXPECT_NE(d2->Digest(), c->Digest());
      auto rt = FromJson(ToJson(*d2));
      ASSERT_TRUE(rt.has_value()) << rt.error();
      EXPECT_EQ(*rt, *d2);
    }
    const auto policy = DefaultPolicy(mappings::Gemma4AssistantMapping(), *values);
    ASSERT_TRUE(policy.has_value());
    EXPECT_EQ(*policy, "max_decode_rows 16\n");
    // A second assistant of the same name is refused, and so is a weight
    // of a stage the spec has not.
    EXPECT_FALSE(ComposeDrafter(*spec, mappings::Gemma4Mapping(),
                                mappings::Gemma4AssistantMapping(), source, *values));
    ModelSpec bad = *spec;
    for (WeightBinding& w : bad.stages[1].embedding->weights) {
      if (!w.stage.empty()) w.stage = "nowhere";
    }
    EXPECT_FALSE(Validate(bad));
    for (WeightBinding& w : bad.stages[1].embedding->weights) {
      if (!w.stage.empty()) w.stage = "assistant";
    }
    EXPECT_FALSE(Validate(bad));
  }
}

TEST(MappingTest, DrafterTargetsNameArchitecturesWithoutCheckpointProfiles) {
  EXPECT_EQ(mappings::Gemma4AssistantMapping().draft->target_architecture, "gemma4");
  EXPECT_EQ(mappings::DsparkMapping().draft->target_architecture, "deepseek4");
}

TEST(MappingTest, VocabularyUsesTheEmittedHeadRowsWithinItsPhysicalWeight) {
  auto source = test_support::Gemma2Source();
  auto values = fam::Gemma2Values(fam::Gemma2_2B());
  ASSERT_TRUE(values);
  auto model = Synthesize(mappings::Gemma2Mapping(), source, *values);
  ASSERT_TRUE(model) << model.error();
  auto full = Vocabulary(*model, source);
  ASSERT_TRUE(full) << full.error();
  auto& head = *model->stages.front().heads.output;
  head.params.push_back({.name = "rows", .value = std::int64_t{123}});
  Canonicalize(*model);
  EXPECT_EQ(*Vocabulary(*model, source), 123U);
  auto selected = std::ranges::find(head.params, "rows", &Param::name);
  ASSERT_NE(selected, head.params.end());
  selected->value = std::int64_t{0};
  EXPECT_EQ(*Vocabulary(*model, source), *full);
  selected->value = std::int64_t{*full} + 1;
  EXPECT_FALSE(Vocabulary(*model, source));
}

Source Qwen2GgufSource(std::uint32_t layers = 1) {
  Source s{.architecture = "qwen2", .container = "gguf"};
  const auto add = [&](std::string name) { s.tensors.push_back({.name = std::move(name)}); };
  add("token_embd.weight");
  add("output_norm.weight");
  for (std::uint32_t il = 0; il < layers; ++il) {
    add(std::format("blk.{}.attn_norm.weight", il));
    add(std::format("blk.{}.attn_q.weight", il));
    add(std::format("blk.{}.attn_k.weight", il));
    add(std::format("blk.{}.attn_v.weight", il));
    add(std::format("blk.{}.attn_q.bias", il));
    add(std::format("blk.{}.attn_k.bias", il));
    add(std::format("blk.{}.attn_v.bias", il));
    add(std::format("blk.{}.attn_output.weight", il));
    add(std::format("blk.{}.ffn_norm.weight", il));
    add(std::format("blk.{}.ffn_gate.weight", il));
    add(std::format("blk.{}.ffn_up.weight", il));
    add(std::format("blk.{}.ffn_down.weight", il));
  }
  return s;
}

TEST(MappingTest, FromOrRatioEvaluatesMissingKeyOnNonHalfBillionQwen) {
  const Source source = Qwen2GgufSource(1);
  const std::vector<SourceValue> values{
      {.key = "qwen2.context_length", .value = std::int64_t{32768}},
      {.key = "qwen2.embedding_length", .value = std::int64_t{1536}},
      {.key = "qwen2.block_count", .value = std::int64_t{1}},
      {.key = "qwen2.feed_forward_length", .value = std::int64_t{4096}},
      {.key = "qwen2.attention.head_count", .value = std::int64_t{12}},
      {.key = "qwen2.attention.head_count_kv", .value = std::int64_t{2}},
      {.key = "qwen2.attention.layer_norm_rms_epsilon", .value = 1e-6},
      {.key = "qwen2.rope.freq_base", .value = 10000.0},
  };
  const auto spec = Synthesize(mappings::Qwen2Mapping(), source, values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const ComponentRef& attn = spec->stages[0].layers[0].components[0];
  EXPECT_EQ(attn.kind, "attention.dense");
  EXPECT_EQ(P(attn, "key_dim").value, ParamValue(std::int64_t{128}));
  EXPECT_EQ(P(attn, "value_dim").value, ParamValue(std::int64_t{128}));
  EXPECT_EQ(P(attn, "rope_dims").value, ParamValue(std::int64_t{128}));
  const double want_scale = 1.0 / std::sqrt(128.0);
  EXPECT_DOUBLE_EQ(std::get<double>(P(attn, "scale").value), want_scale);
}

TEST(MappingTest, GenericDenseDecoderSynthesizesLlamaAndMistralWithRopeFreqsAndYarn) {
  Source llama_source{.architecture = "llama", .container = "gguf"};
  const auto add = [&](Source& s, std::string name) {
    s.tensors.push_back({.name = std::move(name)});
  };
  add(llama_source, "token_embd.weight");
  add(llama_source, "output_norm.weight");
  add(llama_source, "rope_freqs.weight");
  add(llama_source, "blk.0.attn_norm.weight");
  add(llama_source, "blk.0.attn_q.weight");
  add(llama_source, "blk.0.attn_k.weight");
  add(llama_source, "blk.0.attn_v.weight");
  add(llama_source, "blk.0.attn_output.weight");
  add(llama_source, "blk.0.ffn_norm.weight");
  add(llama_source, "blk.0.ffn_gate.weight");
  add(llama_source, "blk.0.ffn_up.weight");
  add(llama_source, "blk.0.ffn_down.weight");

  const std::vector<SourceValue> llama_values{
      {.key = "llama.context_length", .value = std::int64_t{131072}},
      {.key = "llama.embedding_length", .value = std::int64_t{2048}},
      {.key = "llama.block_count", .value = std::int64_t{1}},
      {.key = "llama.feed_forward_length", .value = std::int64_t{8192}},
      {.key = "llama.attention.head_count", .value = std::int64_t{32}},
      {.key = "llama.attention.head_count_kv", .value = std::int64_t{8}},
      {.key = "llama.attention.key_length", .value = std::int64_t{64}},
      {.key = "llama.attention.value_length", .value = std::int64_t{64}},
      {.key = "llama.attention.layer_norm_rms_epsilon", .value = 1e-5},
      {.key = "llama.rope.dimension_count", .value = std::int64_t{64}},
      {.key = "llama.rope.freq_base", .value = 500000.0},
  };
  const auto llama_spec = Synthesize(mappings::LlamaGgufMapping(), llama_source, llama_values);
  ASSERT_TRUE(llama_spec.has_value()) << llama_spec.error();
  const ComponentRef& llama_attn = llama_spec->stages[0].layers[0].components[0];
  EXPECT_EQ(P(llama_attn, "rotation").value, ParamValue(std::string("normal")));
  EXPECT_EQ(P(llama_attn, "qkv_bias").value, ParamValue(false));
  EXPECT_EQ(P(llama_attn, "rope_freqs").value, ParamValue(true));
  EXPECT_EQ(Tensor(llama_attn, "rope_freqs"), "rope_freqs.weight");
  EXPECT_EQ(llama_spec->stages[0].heads.output->weights[0].tensor, "token_embd.weight");

  Source mistral_source = llama_source;
  mistral_source.architecture = "mistral";
  mistral_source.container = "safetensors";
  std::erase_if(mistral_source.tensors,
                [](const SourceTensor& t) { return t.name == "rope_freqs.weight"; });
  add(mistral_source, "output.weight");
  const std::vector<SourceValue> mistral_values{
      {.key = "mistral.context_length", .value = std::int64_t{32768}},
      {.key = "mistral.embedding_length", .value = std::int64_t{4096}},
      {.key = "mistral.block_count", .value = std::int64_t{1}},
      {.key = "mistral.feed_forward_length", .value = std::int64_t{14336}},
      {.key = "mistral.attention.head_count", .value = std::int64_t{32}},
      {.key = "mistral.attention.head_count_kv", .value = std::int64_t{8}},
      {.key = "mistral.attention.sliding_window", .value = std::int64_t{4096}},
      {.key = "mistral.attention.layer_norm_rms_epsilon", .value = 1e-5},
      {.key = "mistral.rope.freq_base", .value = 10000.0},
      {.key = "mistral.rope.scaling.type", .value = std::string("yarn")},
      {.key = "mistral.rope.scaling.factor", .value = 4.0},
      {.key = "mistral.rope.scaling.original_context_length", .value = std::int64_t{8192}},
      {.key = "mistral.rope.scaling.yarn_beta_fast", .value = 32.0},
      {.key = "mistral.rope.scaling.yarn_beta_slow", .value = 1.0},
  };
  const auto mistral_spec =
      Synthesize(mappings::MistralSafetensorsMapping(), mistral_source, mistral_values);
  ASSERT_TRUE(mistral_spec.has_value()) << mistral_spec.error();
  const ComponentRef& mistral_attn = mistral_spec->stages[0].layers[0].components[0];
  EXPECT_EQ(P(mistral_attn, "rotation").value, ParamValue(std::string("neox")));
  EXPECT_EQ(P(mistral_attn, "window").value, ParamValue(std::int64_t{4096}));
  EXPECT_EQ(P(mistral_attn, "yarn").value, ParamValue(true));
  EXPECT_DOUBLE_EQ(std::get<double>(P(mistral_attn, "rope_factor").value), 4.0);
  EXPECT_EQ(P(mistral_attn, "yarn_context").value, ParamValue(std::int64_t{8192}));
  EXPECT_EQ(mistral_spec->stages[0].heads.output->weights[0].tensor, "output.weight");
}

TEST(MappingTest, CheckLayerFlagsRefusesFlagWhenScheduleResolvesToAllA) {
  Mapping m = mappings::Gemma4Mapping();
  m.layer_flag_type = 'g';
  auto source = test_support::Gemma4Source(26);
  auto values = *fam::Gemma4Values(fam::Gemma4_26BA4B());
  const auto s = Synthesize(m, source, values);
  ASSERT_FALSE(s.has_value());
  EXPECT_NE(s.error().find("not of the mapping's layer types"), std::string::npos) << s.error();
}

TEST(MappingTest, ContainerLookupIsExactAndUnknownVariantsRefuse) {
  for (const Mapping* mapping : Mappings())
    for (const auto container : mapping->containers)
      EXPECT_EQ(FindMapping(mapping->architecture, container, mapping->version), mapping);
  EXPECT_EQ(FindMapping("gemma2", "safetensors"), &mappings::Gemma2SafetensorsMapping());
  EXPECT_EQ(FindMapping("gemma3_text", "safetensors"), &mappings::Gemma3SafetensorsMapping());
  EXPECT_EQ(FindMapping("gemma3", "safetensors"), &mappings::Gemma3ConditionalSafetensorsMapping());
  EXPECT_EQ(FindMapping("qwen4exp", "exl3"), nullptr);
  EXPECT_EQ(FindMapping("qwen4exp", ""), nullptr);
  EXPECT_EQ(FindMapping("qwen4exp-mtp", "gguf"), nullptr);
  EXPECT_EQ(FindMapping("qwen4exp-mtp", "safetensors"), &mappings::Qwen38MtpNativeMapping());
  auto values = fam::Gemma2Values(fam::Gemma2_2B());
  ASSERT_TRUE(values);
  auto source = test_support::Gemma2Source();
  source.container = "safetensors";
  EXPECT_FALSE(Synthesize(mappings::Gemma2Mapping(), source, *values));
}

TEST(MappingTest, SignedZeroCanonicalizesAndContainerFoldsSuppressRules) {
  auto values = *fam::Gemma2Values(fam::Gemma2_2B());
  values.push_back({.key = "test.zero", .value = -0.0});
  Mapping mapping = mappings::Gemma2Mapping();
  std::vector<ParamRule> params(mapping.params.begin(), mapping.params.end());
  params.push_back(
      {.slot = "embedding", .param = "zero", .key = "test.zero", .type = ValueType::kReal});
  params.push_back({.slot = "embedding",
                    .param = "again",
                    .type = ValueType::kBool,
                    .value = {.flag = true},
                    .unless_fold = "norm.plus_one"});
  mapping.params = params;
  for (const auto transform : {Transform::kNone, Transform::kSqrt}) {
    params[params.size() - 2].transform = transform;
    auto made = Synthesize(mapping, test_support::Gemma2Source(), values);
    ASSERT_TRUE(made) << made.error();
    const auto& embedding = *made->stages.front().embedding;
    ASSERT_NE(embedding.Find("zero"), nullptr);
    EXPECT_FALSE(std::signbit(std::get<double>(embedding.Find("zero")->value)));
    EXPECT_EQ(embedding.Find("again"), nullptr);
  }
}

TEST(MappingTest, FallbackPartsAndUnusedSharedTensorsAreRefused) {
  auto values = *fam::Gemma2Values(fam::Gemma2_2B());
  Mapping mapping = mappings::Gemma2Mapping();
  std::vector<TensorRule> rules(mapping.tensors.begin(), mapping.tensors.end());
  auto output = std::ranges::find(rules, "output", &TensorRule::slot);
  ASSERT_NE(output, rules.end());
  output->part = format::PartRole::kBlockScales;
  mapping.tensors = rules;
  const auto wrong_part = Synthesize(mapping, test_support::Gemma2Source(), values);
  ASSERT_FALSE(wrong_part);
  EXPECT_NE(wrong_part.error().find("fallback tensor"), std::string::npos);

  mapping = mappings::Gemma2Mapping();
  rules.assign(mapping.tensors.begin(), mapping.tensors.end());
  rules.push_back(
      {.pattern = "shared_global", .slot = "attention", .role = "extra", .layer_types = "g"});
  mapping.tensors = rules;
  auto source = test_support::Gemma2Source();
  std::erase_if(source.tensors, [](const SourceTensor& tensor) {
    return tensor.name.starts_with("blk.") && !tensor.name.starts_with("blk.0.");
  });
  source.tensors.push_back({.name = "shared_global"});
  std::ranges::find(values, "gemma2.block_count", &SourceValue::key)->value = std::int64_t{1};
  const auto unused = Synthesize(mapping, source, values);
  ASSERT_FALSE(unused);
  EXPECT_NE(unused.error().find("no component instance"), std::string::npos);
}

TEST(MappingTest, IntegerListRequirementsCannotEnterAnUnsupportedSchema) {
  Mapping mapping = mappings::Gemma2Mapping();
  std::vector<ParamRule> params(mapping.params.begin(), mapping.params.end());
  params.push_back({.key = "test.list", .type = ValueType::kInts});
  mapping.params = params;
  EXPECT_FALSE(ValidateMapping(mapping));
}

TEST(MappingTest, StringValuedLayerTypeKey) {
  Mapping m = mappings::Gemma2Mapping();
  m.layer_types = "lg";
  m.layer_type_key = "test.layer_types";
  m.layer_type_codes = {};
  ASSERT_TRUE(ValidateMapping(m));

  // Synthesize with 4 layers: "lglg"
  std::vector<SourceValue> values = *fam::Gemma2Values(fam::Gemma2_2B());
  std::ranges::find(values, "gemma2.block_count", &SourceValue::key)->value = std::int64_t{4};
  values.push_back({.key = "test.layer_types", .value = ParamValue(std::string("lglg"))});

  auto source = test_support::Gemma2Source();
  std::erase_if(source.tensors, [](const SourceTensor& t) {
    if (!t.name.starts_with("blk.")) return false;
    for (int i = 0; i < 4; ++i) {
      if (t.name.starts_with(std::format("blk.{}.", i))) return false;
    }
    return true;
  });
  auto spec = Synthesize(m, source, values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  EXPECT_EQ(spec->stages.front().layers.size(), 4U);

  // Too short string is refused
  values.back().value = ParamValue(std::string("lgl"));
  EXPECT_FALSE(Synthesize(m, source, values));

  // Invalid character is refused
  values.back().value = ParamValue(std::string("lglx"));
  EXPECT_FALSE(Synthesize(m, source, values));
}

TEST(MappingTest, DeclarativeCompanionSynthesisViaSynthesize) {
  const std::uint32_t size = 26;
  const auto& p = G4(size);
  const auto& a = fam::Gemma4Assistant26();
  auto target_values = *fam::Gemma4Values(p);
  auto assistant_values = *fam::Gemma4AssistantValues(a, p);

  std::vector<SourceValue> combined_values = target_values;
  for (const auto& v : assistant_values) {
    if (!std::ranges::contains(combined_values, v.key, &SourceValue::key)) {
      combined_values.push_back(v);
    }
  }

  const auto assistant_m = mappings::Gemma4AssistantMapping();
  Mapping drafter_m = assistant_m;
  drafter_m.architecture = "gemma4-drafter";
  std::vector<std::string> pattern_strings;
  std::vector<TensorRule> drafter_rules;
  for (const TensorRule& r : assistant_m.tensors) {
    auto rule = r;
    if (!rule.target) {
      pattern_strings.push_back("drafter." + std::string(rule.pattern));
      rule.pattern = pattern_strings.back();
    }
    drafter_rules.push_back(rule);
  }
  drafter_m.tensors = drafter_rules;

  Source combined_source = test_support::Gemma4Source(size);
  Source as_source = test_support::Gemma4AssistantSource(size);
  for (SourceTensor t : as_source.tensors) {
    t.name = "drafter." + t.name;
    combined_source.tensors.push_back(std::move(t));
  }

  Mapping m = mappings::Gemma4Mapping();
  m.companion = &drafter_m;

  auto synthesized = Synthesize(m, combined_source, combined_values);
  ASSERT_TRUE(synthesized.has_value()) << synthesized.error();
  ASSERT_EQ(synthesized->stages.size(), 2U);
  EXPECT_EQ(synthesized->stages[0].name, "target");
  EXPECT_EQ(synthesized->stages[1].name, "assistant");
  EXPECT_EQ(synthesized->stages[1].role, StageRole::kDrafter);
  EXPECT_EQ(synthesized->stages[1].artifact, 0U);
  EXPECT_TRUE(components::ValidateSpec(*synthesized, combined_source.tensors).has_value());
}

TEST(SpecTest, ForwardReadsCyclesAndTargetAfterDrafterAreRefused) {
  auto spec = Fixture();
  spec.stages.front().reads.push_back(
      {.stage = "assistant", .kind = ReadKind::kState, .name = "kv"});
  Canonicalize(spec);
  EXPECT_FALSE(Validate(spec));
  spec = Fixture();
  std::ranges::reverse(spec.stages);
  EXPECT_FALSE(Validate(spec));
}

}  // namespace
}  // namespace llmp::spec
