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
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace llmp::spec {
namespace {

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

  ModelSpec spec{
      .pipeline = PipelineKind::kLlm, .stages = {target, drafter}, .folds = {"norm.plus_one"}};
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
  ModelSpec negative_zero = Fixture();
  negative_zero.stages[0].embedding->params[0].value = -0.0;
  EXPECT_FALSE(Validate(negative_zero).has_value());
  ModelSpec negative_zero_listed = Fixture();
  negative_zero_listed.stages[0].layers[0].components[1].params[0].value =
      std::vector<double>{0.5, -0.0};
  EXPECT_FALSE(Validate(negative_zero_listed).has_value());
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
    EXPECT_EQ(FindMapping(m->architecture), m);
  }
  EXPECT_EQ(FindMapping("gemma4"), nullptr);
  EXPECT_EQ(FindFold("norm.plus_one")->name, "norm.plus_one");
  EXPECT_EQ(FindFold("norm.times_two"), nullptr);
}

TEST(MappingTest, InvalidMappingsAreRefused) {
  const Mapping& good = mappings::Gemma2Mapping();
  // Its policy text: printable lines (the planner's own rules are its).
  for (const std::string_view policy :
       {std::string_view("max_decode_rows 16"), std::string_view("disable x\n\t\n")}) {
    Mapping m = good;
    m.policy = policy;
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
  const auto values = mappings::Gemma2V0Values(model::Gemma2_2B());
  ASSERT_TRUE(values.has_value()) << values.error();
  const auto spec = Synthesize(mappings::Gemma2Mapping(), test_support::Gemma2Source(), *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  const Stage& s = spec->stages[0];
  EXPECT_EQ(spec->folds, std::vector<std::string>{"norm.plus_one"});
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
  const auto unfolded = Synthesize(mappings::Gemma2Mapping(), other, *values);
  ASSERT_TRUE(unfolded.has_value());
  EXPECT_TRUE(unfolded->folds.empty());
  EXPECT_NE(unfolded->Digest(), spec->Digest());
}

TEST(MappingTest, Gemma3SpecFromItsTensorsAndProfile) {
  const auto values = mappings::Gemma3V0Values(model::Gemma3_4BQat());
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
  const auto g2 = *mappings::Gemma2V0Values(model::Gemma2_2B());
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

  const auto g3 = *mappings::Gemma3V0Values(model::Gemma3_4BQat());
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
  const auto values = *mappings::Gemma2V0Values(model::Gemma2_2B());
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
    auto v = *mappings::Gemma3V0Values(model::Gemma3_4BQat());
    std::ranges::find(v, std::string_view("gemma3.rope.scaling.type"), &SourceValue::key)->value =
        std::string("yarn");
    EXPECT_FALSE(
        Synthesize(mappings::Gemma3Mapping(), test_support::Gemma3Source(), v).has_value());
  }
}

}  // namespace
}  // namespace llmp::spec
