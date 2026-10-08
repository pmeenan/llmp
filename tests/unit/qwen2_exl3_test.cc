// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The EXL3 Qwen2 adapter and its native operation plan (model/qwen2_exl3.h),
// on synthetic resource lists and launch tables: binding refuses anything
// the plan does not read or reads differently; the plan refuses an
// unrecorded phase kind and a linear case the table lacks or that is not
// upstream's path; its operations follow the record's order
// (docs/experiments/backend-proof-p0/exl3-op-plan.json); its placement
// never lets two live tensors share a byte; its launch digest covers the
// launch data. The regions it places are pinned: they are the itemized
// buffer plan the EXL3 phase memory limits are pre-registered against
// (docs/backend-proof.md, "Memory and workspace").

#include "model/qwen2_exl3.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "model/qwen2.h"

namespace {

using llmp::model::Exl3Arm;
using llmp::model::Exl3LaunchTable;
using llmp::model::Exl3LinearPlan;
using llmp::model::Exl3Path;
using llmp::model::Exl3Phase;
using llmp::model::Exl3PhasePlan;
using llmp::model::Exl3Pin;
using llmp::model::Exl3Resource;
using llmp::test_support::Failed;

const llmp::model::Qwen2Profile& Profile() { return llmp::model::Qwen25Instruct05BExl3(); }

Exl3Resource Resource(std::string name, std::string family, std::string dtype,
                      std::vector<std::uint64_t> shape, std::string role = "") {
  Exl3Resource r;
  r.name = std::move(name);
  r.family = std::move(family);
  r.dtype = std::move(dtype);
  r.shape = std::move(shape);
  r.role = std::move(role);
  return r;
}

Exl3LinearPlan Packed(Exl3Path path, int shape, int blocks, int concurrency = 0, int config = 0) {
  Exl3LinearPlan plan;
  plan.path = path;
  plan.shape = shape;
  plan.blocks = blocks;
  plan.concurrency = concurrency;
  plan.config = config;
  return plan;
}

Exl3LinearPlan Reconstructed(Exl3Path path, std::vector<Exl3Pin> pins) {
  Exl3LinearPlan plan;
  plan.path = path;
  plan.pins = std::move(pins);
  return plan;
}

Exl3Pin Pin(int m, int k, int n, int ldc, bool f32) {
  Exl3Pin pin;
  pin.m = m;
  pin.k = k;
  pin.n = n;
  pin.ldc = ldc;
  pin.f32 = f32;
  return pin;
}

void AddLinear(std::vector<Exl3Resource>& out, const std::string& name, std::uint64_t k,
               std::uint64_t n, std::uint32_t bits, bool bias) {
  Exl3Resource trellis =
      Resource(name + ".trellis", "exl3", "I16", {k / 16, n / 16, 16ULL * bits}, "trellis");
  trellis.k_bits = bits;
  trellis.in_features = k;
  trellis.out_features = n;
  trellis.codebook = "mcg";
  out.push_back(trellis);
  out.push_back(Resource(name + ".suh", "exl3", "F16", {k}, "suh"));
  out.push_back(Resource(name + ".svh", "exl3", "F16", {n}, "svh"));
  out.push_back(Resource(name + ".mcg", "exl3", "I32", {}, "mcg"));
  if (bias) {
    out.push_back(Resource(name + ".bias", "plain", "F16", {n}));
  }
}

std::vector<Exl3Resource> Resources() {
  const auto& p = Profile();
  std::vector<Exl3Resource> out;
  out.push_back(Resource("model.embed_tokens.weight", "plain", "BF16", {p.vocab, p.width}));
  out.push_back(Resource("model.norm.weight", "plain", "BF16", {p.width}));
  AddLinear(out, "lm_head", p.width, p.vocab, 8, false);
  for (std::uint32_t l = 0; l < p.layers; ++l) {
    const std::string b = std::format("model.layers.{}.", l);
    out.push_back(Resource(b + "input_layernorm.weight", "plain", "BF16", {p.width}));
    out.push_back(Resource(b + "post_attention_layernorm.weight", "plain", "BF16", {p.width}));
    const std::uint32_t bits = l < 10 ? 4 : 5;
    AddLinear(out, b + "self_attn.q_proj", p.width, p.width, bits, true);
    AddLinear(out, b + "self_attn.k_proj", p.width, p.kv_width(), bits, true);
    AddLinear(out, b + "self_attn.v_proj", p.width, p.kv_width(), bits, true);
    AddLinear(out, b + "self_attn.o_proj", p.width, p.width, bits, false);
    AddLinear(out, b + "mlp.gate_proj", p.width, p.ffn, bits, false);
    AddLinear(out, b + "mlp.up_proj", p.width, p.ffn, bits, false);
    AddLinear(out, b + "mlp.down_proj", p.ffn, p.width, bits, false);
  }
  return out;
}

llmp::model::Exl3Binding Binding() {
  const std::vector<Exl3Resource> resources = Resources();
  return llmp::model::BindQwen2Exl3(Profile(), "qwen2", resources).value();
}

// A table with every case the trajectories need, each on upstream's path
// (tile shapes and grids arbitrary but valid in form).
Exl3LaunchTable Table(Exl3Arm arm, std::string source = "test") {
  const llmp::model::Exl3Binding b = Binding();
  Exl3LaunchTable table(std::move(source));
  const auto add = [&](const llmp::model::Exl3LinearBinding& l, int rows, bool f32) {
    Exl3LinearPlan plan;
    if (rows <= 144) {
      plan =
          Packed(arm == Exl3Arm::kO && rows == 1 && l.bits == 4 ? Exl3Path::kGemv : Exl3Path::kGemm,
                 1, 16);
    } else {
      std::vector<Exl3Pin> pins;
      for (const int n : llmp::model::Exl3Slices(l.n)) {
        pins.push_back(Pin(rows, l.k, n, l.n, f32));
      }
      plan = Reconstructed(rows >= 1024 ? Exl3Path::kReconstructFused : Exl3Path::kReconstruct,
                           std::move(pins));
    }
    ASSERT_TRUE(table.Add(l.name, rows, std::move(plan)).has_value());
  };
  for (const int rows : {1, 32, 144, 145, 1023, 1024}) {
    add(b.lm_head, rows, false);
    for (const auto& layer : b.layers) {
      add(layer.q, rows, false);
      add(layer.k, rows, false);
      add(layer.v, rows, false);
      add(layer.o, rows, true);
      add(layer.down, rows, true);
      if (rows <= 32) {
        EXPECT_TRUE(
            table.Add(layer.gate.name, rows, Packed(Exl3Path::kMulti, 3, 24, 2), layer.up.name)
                .has_value());
      } else {
        add(layer.gate, rows, true);
        add(layer.up, rows, true);
      }
    }
  }
  return table;
}

TEST(Qwen2Exl3Test, BindsEveryTensorAndRefusesAnythingElse) {
  const auto binding = Binding();
  ASSERT_EQ(binding.layers.size(), 24U);
  EXPECT_EQ(binding.lm_head.n, 151936);
  EXPECT_EQ(binding.lm_head.bits, 8);
  EXPECT_TRUE(binding.layers[3].q.bias.has_value());
  EXPECT_FALSE(binding.layers[3].o.bias.has_value());
  EXPECT_EQ(binding.layers[0].down.k, 4864);

  const auto refused = [](std::vector<Exl3Resource> resources,
                          std::string_view architecture = "qwen2") {
    return Failed(llmp::model::BindQwen2Exl3(Profile(), architecture, resources));
  };
  EXPECT_TRUE(refused(Resources(), "llama").has_value());
  {
    auto r = Resources();
    r.push_back(Resource("model.layers.0.self_attn.o_proj.bias", "plain", "F16", {896}));
    EXPECT_NE(refused(r).value_or("").find("does not read"), std::string::npos);
  }
  for (const auto& change : std::vector<std::pair<std::string, std::function<void(Exl3Resource&)>>>{
           {"model.layers.2.mlp.up_proj.trellis", [](Exl3Resource& r) { r.k_bits = 3; }},
           {"model.layers.2.mlp.up_proj.trellis", [](Exl3Resource& r) { r.codebook = "3inst"; }},
           {"model.layers.2.mlp.up_proj.svh", [](Exl3Resource& r) { r.dtype = "F32"; }},
           {"model.layers.2.mlp.up_proj.mcg", [](Exl3Resource& r) { r.role = "mul1"; }},
           {"model.norm.weight", [](Exl3Resource& r) { r.dtype = "F32"; }},
           {"model.embed_tokens.weight", [](Exl3Resource& r) { r.shape = {151936, 1024}; }},
       }) {
    auto r = Resources();
    auto found = std::ranges::find(r, change.first, &Exl3Resource::name);
    ASSERT_NE(found, r.end());
    change.second(*found);
    EXPECT_TRUE(refused(r).has_value()) << change.first;
  }
  {
    auto r = Resources();
    std::erase_if(
        r, [](const Exl3Resource& x) { return x.name == "model.layers.7.self_attn.k_proj.bias"; });
    EXPECT_NE(refused(r).value_or("").find("k_proj.bias"), std::string::npos);
  }
}

// A profile with a zero count is refused with an error, never divided by.
TEST(Qwen2Exl3Test, RefusesAProfileWithAZeroCount) {
  const std::vector<Exl3Resource> resources = Resources();
  using Field = std::uint32_t llmp::model::Qwen2Profile::*;
  for (const auto& [name, field] : std::vector<std::pair<std::string, Field>>{
           {"kv_heads", &llmp::model::Qwen2Profile::kv_heads},
           {"heads", &llmp::model::Qwen2Profile::heads},
           {"head_dim", &llmp::model::Qwen2Profile::head_dim},
           {"width", &llmp::model::Qwen2Profile::width},
           {"ffn", &llmp::model::Qwen2Profile::ffn},
           {"vocab", &llmp::model::Qwen2Profile::vocab},
           {"layers", &llmp::model::Qwen2Profile::layers}}) {
    llmp::model::Qwen2Profile p = Profile();
    p.*field = 0;
    EXPECT_EQ(Failed(llmp::model::BindQwen2Exl3(p, "qwen2", resources)),
              "a profile the EXL3 plan cannot run")
        << name;
  }
}

TEST(Qwen2Exl3Test, OnlyRecordedPhaseKindsArePlanned) {
  using llmp::model::RecordedPhase;
  for (const int rows : {32, 144, 145, 1023, 1024}) {
    EXPECT_TRUE(RecordedPhase({.rows = rows, .past = 0}));
    EXPECT_FALSE(RecordedPhase({.rows = rows, .past = 1}));
  }
  EXPECT_FALSE(RecordedPhase({.rows = 64, .past = 0}));
  EXPECT_FALSE(RecordedPhase({.rows = 512, .past = 0}));
  // Single-token steps by padded K length: 256, 1,024 (the P3 addendum) and 1,280.
  EXPECT_TRUE(RecordedPhase({.rows = 1, .past = 32}));
  EXPECT_TRUE(RecordedPhase({.rows = 1, .past = 255}));
  EXPECT_FALSE(RecordedPhase({.rows = 1, .past = 256}));  // padded 512
  EXPECT_TRUE(RecordedPhase({.rows = 1, .past = 1023}));  // padded 1,024
  EXPECT_TRUE(RecordedPhase({.rows = 1, .past = 1039}));  // padded 1,280
  EXPECT_FALSE(RecordedPhase({.rows = 1, .past = 1280}));

  const auto binding = Binding();
  const auto table = Table(Exl3Arm::kG);
  const auto refused = Failed(
      llmp::model::PlanPhase(Profile(), binding, table, Exl3Arm::kG, {.rows = 64, .past = 0}));
  EXPECT_NE(refused.value_or("").find("record it first"), std::string::npos);
}

TEST(Qwen2Exl3Test, OperationsFollowTheRecord) {
  const auto binding = Binding();
  const auto table = Table(Exl3Arm::kG);
  const std::vector<std::string> small = {
      "attn_norm",       "attn_norm.cast", "q_proj",          "q_proj.bias_add",   "k_proj",
      "k_proj.bias_add", "v_proj",         "v_proj.bias_add", "rope_q.cast",       "rope_q",
      "rope_k.cast",     "rope_k",         "rope_k.cast_out", "kv_write.k",        "kv_write.v",
      "attention",       "attention.cast", "o_proj",          "attn_residual_add", "mlp_norm",
      "mlp_norm.cast",   "gate_up",        "swiglu",          "swiglu.cast",       "down_proj",
      "mlp_residual_add"};
  for (const int rows : {32, 145}) {
    const auto plan =
        llmp::model::PlanPhase(Profile(), binding, table, Exl3Arm::kG, {.rows = rows, .past = 0});
    ASSERT_TRUE(plan.has_value()) << Failed(plan).value_or("");
    std::vector<std::string> expected = {"inputs", "embed"};
    for (int l = 0; l < 24; ++l) {
      for (const std::string& name : small) {
        if (name == "gate_up" && rows > 32) {
          expected.emplace_back("gate_proj");
          expected.emplace_back("up_proj");
        } else {
          expected.push_back(name);
        }
      }
    }
    for (const char* name : {"final_norm", "final_norm.cast", "lm_head"}) {
      expected.emplace_back(name);
    }
    std::vector<std::string> got;
    for (const auto& op : plan->ops) {
      got.push_back(op.name);
    }
    EXPECT_EQ(got, expected);
    const auto& q =
        *std::ranges::find(plan->ops, std::string("q_proj"), &llmp::model::Exl3Op::name);
    EXPECT_EQ(q.implementation, rows == 32 ? "exl3.linear.gemm" : "exl3.linear.reconstruct");
    EXPECT_EQ(plan->tensors.at("gate").dtype, llmp::model::Exl3Dtype::kF32);
    EXPECT_EQ(plan->tensors.at("attn.out").dtype, llmp::model::Exl3Dtype::kF16);
  }
  const auto fused =
      llmp::model::PlanPhase(Profile(), binding, table, Exl3Arm::kG, {.rows = 1024, .past = 0});
  ASSERT_TRUE(fused.has_value());
  EXPECT_EQ(std::ranges::find(fused->ops, std::string("lm_head"), &llmp::model::Exl3Op::name)
                ->implementation,
            "exl3.linear.reconstruct_fused");
  EXPECT_EQ(fused->slots.count("lm_head.xh"), 0U);  // the fused path transforms no input
}

TEST(Qwen2Exl3Test, RefusesACaseTheTableLacksOrThatIsNotUpstreamsPath) {
  const auto binding = Binding();
  const std::string q = "model.layers.5.self_attn.q_proj";
  // A missing case.
  {
    Exl3LaunchTable table("partial");
    const auto refused = Failed(
        llmp::model::PlanPhase(Profile(), binding, table, Exl3Arm::kG, {.rows = 32, .past = 0}));
    EXPECT_NE(refused.value_or("").find("no plan"), std::string::npos);
  }
  // The full table with one entry replaced.
  const auto with = [&](Exl3Arm arm, int rows, const std::string& linear, Exl3LinearPlan plan) {
    const Exl3LaunchTable table = Table(arm);
    const auto b = Binding();
    Exl3LaunchTable out("changed");
    for (const int r : {1, 32, 144, 145, 1023, 1024}) {
      const auto copy = [&](const llmp::model::Exl3LinearBinding& l) {
        const Exl3LinearPlan* found = table.Find(l.name, r);
        if (found != nullptr) {
          Exl3LinearPlan p = l.name == linear && r == rows ? plan : *found;
          std::ignore = out.Add(l.name, r, p, std::string(table.Second(l.name, r)));
        }
      };
      copy(b.lm_head);
      for (const auto& layer : b.layers) {
        for (const auto* l :
             {&layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down}) {
          copy(*l);
        }
      }
    }
    return Failed(llmp::model::PlanPhase(
        Profile(), binding, out, arm,
        rows == 1 ? Exl3Phase{.rows = 1, .past = 32} : Exl3Phase{.rows = rows, .past = 0}));
  };
  // The GEMV in EXL3-G, and in EXL3-O above eight rows.
  const Exl3LinearPlan gemv = Packed(Exl3Path::kGemv, 0, 8);
  EXPECT_TRUE(with(Exl3Arm::kG, 1, q, gemv).has_value());
  EXPECT_FALSE(with(Exl3Arm::kO, 1, q, gemv).has_value());
  EXPECT_TRUE(with(Exl3Arm::kO, 32, q, gemv).has_value());
  // A packed plan where upstream reconstructs, and the unfused
  // reconstruction where it fuses.
  const Exl3LinearPlan gemm = Packed(Exl3Path::kGemm, 1, 16);
  EXPECT_TRUE(with(Exl3Arm::kG, 145, q, gemm).has_value());
  EXPECT_TRUE(with(Exl3Arm::kG, 1024, q,
                   Reconstructed(Exl3Path::kReconstruct, {Pin(1024, 896, 896, 896, false)}))
                  .has_value());
  // A pin recorded for another GEMM: another row count, output or width.
  for (const Exl3Pin& pin : {Pin(1024, 896, 896, 896, false), Pin(145, 896, 896, 896, true),
                             Pin(145, 896, 128, 896, false)}) {
    EXPECT_TRUE(
        with(Exl3Arm::kG, 145, q, Reconstructed(Exl3Path::kReconstruct, {pin})).has_value());
  }
  // The head's five slices need five pins.
  EXPECT_TRUE(with(Exl3Arm::kG, 145, "lm_head",
                   Reconstructed(Exl3Path::kReconstruct, {Pin(145, 896, 32768, 151936, false)}))
                  .has_value());
}

// Every tensor's slot, and every linear's scratch, is disjoint from every
// other one live at the same time in a layer (or the phase).
TEST(Qwen2Exl3Test, PlacementNeverOverlapsLiveTensors) {
  const auto binding = Binding();
  for (const Exl3Arm arm : {Exl3Arm::kG, Exl3Arm::kO}) {
    const auto table = Table(arm);
    for (const Exl3Phase phase :
         {Exl3Phase{.rows = 32, .past = 0}, Exl3Phase{.rows = 144, .past = 0},
          Exl3Phase{.rows = 145, .past = 0}, Exl3Phase{.rows = 1023, .past = 0},
          Exl3Phase{.rows = 1024, .past = 0}, Exl3Phase{.rows = 1, .past = 32},
          Exl3Phase{.rows = 1, .past = 1023}, Exl3Phase{.rows = 1, .past = 1030}}) {
      const auto plan = llmp::model::PlanPhase(Profile(), binding, table, arm, phase);
      ASSERT_TRUE(plan.has_value()) << Failed(plan).value_or("");
      // Lifetimes over the embedding, layer 0 and the output, read from
      // the operations: from a tensor's first use to its last, and through
      // the whole layer span for one live when the layers begin (every
      // layer reads it).
      std::map<std::string, std::pair<int, int>> life;
      int step = 0;
      int layer_first = -1;
      int layer_last = -1;
      for (const auto& op : plan->ops) {
        if (op.layer > 0) {
          continue;
        }
        if (op.layer == 0) {
          layer_first = layer_first < 0 ? step : layer_first;
          layer_last = step;
        }
        std::vector<std::string> names = op.inputs;
        names.insert(names.end(), op.outputs.begin(), op.outputs.end());
        for (const char* part : {".a_had", ".xh", ".w"}) {
          names.push_back(op.name + part);
        }
        for (const std::string& name : names) {
          if (!plan->slots.contains(name)) {
            continue;
          }
          auto [it, fresh] = life.try_emplace(name, step, step);
          it->second.second = step;
        }
        ++step;
      }
      // The residual stream is one buffer under three names.
      for (const char* name : {"embed.out", "resid.in", "resid.out"}) {
        life[name] = {
            std::min(life["embed.out"].first, life[name].first),
            std::max({life["resid.out"].second, life["resid.in"].second, life[name].second})};
      }
      for (auto& [name, span] : life) {
        if (span.first < layer_first && span.second >= layer_first) {
          span.second = std::max(span.second, layer_last);
        }
      }
      EXPECT_EQ(life.at("logits").first, step - 1);  // the head writes it last
      for (const auto& [a, la] : life) {
        const auto& sa = plan->slots.at(a);
        EXPECT_LE(sa.offset + sa.bytes, plan->region) << a;
        EXPECT_EQ(sa.offset % 256, 0U) << a;
        for (const auto& [b, lb] : life) {
          if (a >= b || la.first > lb.second || lb.first > la.second) {
            continue;
          }
          const auto& sb = plan->slots.at(b);
          const bool alias =
              sa.offset == sb.offset && ((a.starts_with("resid") || a == "embed.out") &&
                                         (b.starts_with("resid") || b == "embed.out") &&
                                         a != "resid.mid" && b != "resid.mid");
          if (alias) {
            continue;  // the residual stream's one buffer
          }
          EXPECT_TRUE(sa.offset + sa.bytes <= sb.offset || sb.offset + sb.bytes <= sa.offset)
              << a << " and " << b << " overlap at " << phase.rows << " rows";
        }
      }
    }
  }
}

TEST(Qwen2Exl3Test, TheLaunchDigestCoversTheLaunchData) {
  const auto binding = Binding();
  const Exl3Phase phase{.rows = 32, .past = 0};
  const auto digest = [&](const Exl3LaunchTable& table, Exl3Arm arm) {
    return llmp::model::PlanPhase(Profile(), binding, table, arm, phase).value().launch_digest;
  };
  const auto base = digest(Table(Exl3Arm::kG), Exl3Arm::kG);
  EXPECT_EQ(base, digest(Table(Exl3Arm::kG), Exl3Arm::kG));
  EXPECT_NE(base, digest(Table(Exl3Arm::kG, "another tuning cache"), Exl3Arm::kG));
  EXPECT_NE(base, digest(Table(Exl3Arm::kG), Exl3Arm::kO));
  const auto step = [&](int past) {
    return llmp::model::PlanPhase(Profile(), binding, Table(Exl3Arm::kG), Exl3Arm::kG,
                                  {.rows = 1, .past = past})
        .value()
        .launch_digest;
  };
  EXPECT_NE(step(32), step(33));  // the position is part of the phase
}

// Everything a phase plan holds but its position: operations, launches,
// tensors, placement and region.
std::string Shape(const Exl3PhasePlan& plan) {
  std::string out =
      std::format("rows {} padded {} region {}\n", plan.phase.rows, plan.padded, plan.region);
  for (const auto& o : plan.ops) {
    out += std::format("{} {} {} {}", o.name, o.layer, o.implementation, o.linear + "/" + o.second);
    for (const auto& t : o.inputs) {
      out += " <" + t;
    }
    for (const auto& t : o.outputs) {
      out += " >" + t;
    }
    const auto& l = o.launch;
    out += std::format(" [{} {} {} {} {}", static_cast<int>(l.path), l.shape, l.blocks,
                       l.concurrency, l.config);
    for (const auto& pin : l.pins) {
      out += std::format(" {} {} {} {} {}", pin.m, pin.k, pin.n, pin.ldc, pin.f32);
      for (const auto word : pin.config) {
        out += std::format(",{}", word);
      }
    }
    out += "]\n";
  }
  for (const auto& [name, t] : plan.tensors) {
    out += std::format("{} {}", name, llmp::model::Exl3DtypeName(t.dtype));
    for (const auto d : t.shape) {
      out += std::format(",{}", d);
    }
    out += "\n";
  }
  for (const auto& [name, slot] : plan.slots) {
    out += std::format("{} @{} +{}\n", name, slot.offset, slot.bytes);
  }
  return out;
}

// A single-token step's kind is its padded K length: the planner accepts a
// step at any position whose padded length is recorded, and the record,
// the plan gate and Tier E key it by that length alone. So every position
// of one padded length must plan the same operations, launches, tensors,
// placement and region (the buffer needs its limit `E` was derived from):
// only the host inputs' contents and the KV write's cells move with the
// position, and the attention's parallel blocks and pool scratch depend
// only on the rows and padded length (ops.h PlanFlashAttnVec).
TEST(Qwen2Exl3Test, EveryPositionOfAStepKindPlansTheSameBuffers) {
  const auto binding = Binding();
  for (const Exl3Arm arm : {Exl3Arm::kG, Exl3Arm::kO}) {
    const auto table = Table(arm);
    for (const auto& [first, last] :
         {std::pair{1, 255}, std::pair{768, 1023}, std::pair{1024, 1279}}) {
      const auto base =
          llmp::model::PlanPhase(Profile(), binding, table, arm, {.rows = 1, .past = first});
      ASSERT_TRUE(base.has_value()) << base.error();
      const std::string want = Shape(*base);
      for (int past = first + 1; past <= last; ++past) {
        const auto plan =
            llmp::model::PlanPhase(Profile(), binding, table, arm, {.rows = 1, .past = past});
        ASSERT_TRUE(plan.has_value()) << plan.error();
        ASSERT_EQ(Shape(*plan), want) << "step at " << past << " against " << first;
      }
    }
  }
}

// The itemized buffer plan: each recorded phase kind's region (the
// activations, the linears' scratch and the F16 logits, placed by
// lifetime), which the EXL3 phase memory limits pre-register (backend-
// proof.md, "Memory and workspace"). A change here changes those limits'
// basis, so it is a test failure until the limits are re-derived.
TEST(Qwen2Exl3Test, RegionsAreThePreRegisteredBufferPlan) {
  const auto binding = Binding();
  const std::map<std::pair<int, int>, std::uint64_t> expected = {
      {{32, 0}, 9'838'592},     {{144, 0}, 44'273'664},   {{145, 0}, 103'301'376},
      {{1023, 0}, 373'247'744}, {{1024, 0}, 371'720'192}, {{1, 32}, 307'456},
      {{1, 1023}, 307'456},     {{1, 1030}, 307'456},
  };
  for (const Exl3Arm arm : {Exl3Arm::kG, Exl3Arm::kO}) {
    for (const auto& [key, bytes] : expected) {
      const auto plan = llmp::model::PlanPhase(Profile(), binding, Table(arm), arm,
                                               {.rows = key.first, .past = key.second});
      ASSERT_TRUE(plan.has_value());
      EXPECT_EQ(plan->region, bytes) << key.first << " rows at " << key.second;
    }
  }
}

}  // namespace
