// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Unit tests for Slice K1: Lossy KV cache encoding (kv_cache_encoding = "q8_0")
// support across migrated LLMs:
// - Head dimension D = 512 support for Q8_0 in attention.dense (Gemma 4 global layers)
// - Q8_0 KV cache support in attention.qsa (Qwen3.8 + MTP)
// - ModelSpec transformations via KvCacheEncodingBytes and ApplyKvCacheEncoding
// - Rejection of specialized MQA (DeepSeek V4)
// - Kernel predicate checks (AcceptsQuantAttention, CheckFlashAttnVecQuant)

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/state_encoding.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/reference/exec.h"
#include "model/state_contract.h"
#include "spec/spec.h"

namespace llmp::components {
namespace {

namespace ref = llmp::kernels::reference;
namespace kg = llmp::kernels::ggml;
using format::Encoding;
using spec::ComponentRef;
using spec::Param;
using spec::ParamValue;

Param P(std::string name, ParamValue value) {
  return Param{.name = std::move(name), .value = std::move(value)};
}
ParamValue I(std::int64_t v) { return ParamValue(v); }
ParamValue R(double v) { return ParamValue(v); }
ParamValue B(bool v) { return ParamValue(v); }
ParamValue N(std::string v) { return ParamValue(std::move(v)); }

const StageFacts kFacts{.width = 256, .context = 4096, .streams = 1};
const StageFacts kFactsQwen38{.width = 256, .context = 4096, .streams = 4};

ComponentRef MakeDenseAttention(std::int64_t head_dim, std::string_view kv_encoding,
                                double softcap = 0.0) {
  std::vector<Param> p{
      P("placement", N("pre")),
      P("eps", R(1e-6)),
      P("heads", I(4)),
      P("kv_heads", I(2)),
      P("key_dim", I(head_dim)),
      P("value_dim", I(head_dim)),
      P("rope_dims", I(head_dim)),
      P("rope_base", R(10000.0)),
      P("rope_scale", R(1.0)),
      P("rotation", N("neox")),
      P("window", I(0)),
      P("softcap", R(softcap)),
      P("scale", R(0.125)),
      P("kv_encoding", N(std::string(kv_encoding))),
  };
  std::ranges::sort(p, {}, &Param::name);
  return ComponentRef{.kind = "attention.dense", .version = 1, .params = std::move(p)};
}

ComponentRef MakeQsaAttention(std::int64_t head_dim, std::string_view kv_encoding) {
  std::vector<Param> p{
      P("norm_zero_centered", B(false)),
      P("eps", R(1e-6)),
      P("heads", I(4)),
      P("kv_heads", I(1)),
      P("head_dim", I(head_dim)),
      P("rope_dims", I(head_dim <= 64 ? head_dim : 64)),
      P("rope_base", R(10000.0)),
      P("rope_sections", ParamValue(std::vector<std::int64_t>{16, 16, 16, 16})),
      P("indexer_heads", I(2)),
      P("indexer_dim", I(64)),
      P("ratio", I(16)),
      P("budget", I(128)),
      P("indexer", N("split")),
      P("kv_encoding", N(std::string(kv_encoding))),
  };
  std::ranges::sort(p, {}, &Param::name);
  return ComponentRef{.kind = "attention.qsa", .version = 1, .params = std::move(p)};
}

ComponentRef MakeMqaAttention(std::int64_t head_dim, std::string_view kv_encoding,
                              bool with_indexer = false) {
  std::vector<Param> p{
      P("eps", R(1e-6)),          P("heads", I(4)),
      P("head_dim", I(head_dim)), P("rope_dims", I(head_dim <= 64 ? head_dim : 64)),
      P("q_rank", I(64)),         P("o_rank", I(64)),
      P("o_groups", I(1)),        P("window", I(64)),
      P("rope_base", R(10000.0)), P("kv_encoding", N(std::string(kv_encoding))),
  };
  if (with_indexer) {
    p.push_back(P("ratio", I(16)));
    p.push_back(P("overlap", B(true)));
    p.push_back(P("indexer", B(true)));
    p.push_back(P("indexer_heads", I(2)));
    p.push_back(P("index_dim", I(64)));
    p.push_back(P("top_k", I(32)));
  }
  std::ranges::sort(p, {}, &Param::name);
  return ComponentRef{.kind = "attention.mqa", .version = 1, .params = std::move(p)};
}

}  // namespace

// ------------------------------------------------------------- D=512 attention.dense

TEST(Q8StateTest, DenseAttentionD512ShapesAndState) {
  // Valid D=512 with Q8_0 and no softcap (Gemma 4 global layers).
  {
    auto comp = MakeDenseAttention(512, "q8_0", 0.0);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(inst.has_value()) << inst.error();
    auto state = inst->def->state(*inst);
    ASSERT_EQ(state.tensors.size(), 2U);
    EXPECT_EQ(state.tensors[0].name, "k");
    EXPECT_EQ(state.tensors[0].encoding, Encoding::kQ8_0);
    // 2 KV heads, head_dim 512: (2 * 512 / 32) * 34 = 32 * 34 = 1088 bytes.
    EXPECT_EQ(state.tensors[0].cell_bytes, 1088U);
    EXPECT_EQ(state.tensors[1].name, "v");
    EXPECT_EQ(state.tensors[1].encoding, Encoding::kQ8_0);
    EXPECT_EQ(state.tensors[1].cell_bytes, 1088U);
  }

  // Refused: D=512 with softcap != 0.0 (softcap is only valid at D256).
  {
    auto comp = MakeDenseAttention(512, "q8_0", 50.0);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    EXPECT_FALSE(inst.has_value());
  }

  // Refused: D=512 with unequal key_dim and value_dim.
  {
    auto comp = MakeDenseAttention(512, "q8_0", 0.0);
    for (auto& p : comp.params) {
      if (p.name == "value_dim") p.value = I(256);
    }
    std::ranges::sort(comp.params, {}, &Param::name);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    EXPECT_FALSE(inst.has_value());
  }

  // Refused: unsupported head dimension D=128 for Q8_0.
  {
    auto comp = MakeDenseAttention(128, "q8_0", 0.0);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    EXPECT_FALSE(inst.has_value());
  }
}

TEST(Q8StateTest, DenseAttentionD512ReferenceRequiresAndUsesStoreRow) {
  auto comp = MakeDenseAttention(512, "q8_0", 0.0);
  auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(inst.has_value()) << inst.error();

  RefIO io{.instance = &*inst};
  io.segments = {{.rows = 1, .past = 0, .read_cells = 1, .outputs = 1}};
  io.stream.assign(kFacts.width, 0.5);
  io.state_cells = {4, 4};
  io.state = {{{std::vector<double>(4 * 1024, 0.0), std::vector<double>(4 * 1024, 0.0)}}};
  // Populate weights for required roles (norm, q, k, v, output).
  const auto shapes = inst->def->shapes(*inst);
  ASSERT_TRUE(shapes.has_value());
  io.weights.resize(inst->def->roles.size());
  for (const auto& s : *shapes) {
    std::uint64_t count = 1;
    for (std::size_t i = 0; i < s.rank; ++i) count *= s.ne[i];
    io.weights[s.role].assign(count, 0.01f);
  }

  // Without store_row, reference fails for Q8_0.
  io.store_row = nullptr;
  EXPECT_FALSE(inst->def->reference(io).has_value());

  // With store_row, reference succeeds and stores state.
  io.store_row = &ref::StoreStateRow;
  auto res = inst->def->reference(io);
  ASSERT_TRUE(res.has_value()) << res.error();
  EXPECT_EQ(io.stream.size(), kFacts.width);
}

// ------------------------------------------------------------- attention.qsa

TEST(Q8StateTest, QsaQ8ShapesAndState) {
  // Valid Q8_0 at head_dim = 256 (Qwen3.8).
  {
    auto comp = MakeQsaAttention(256, "q8_0");
    auto inst = MakeInstance(comp, kFactsQwen38, Site::kLayer, 0, 1);
    ASSERT_TRUE(inst.has_value()) << inst.error();
    auto state = inst->def->state(*inst);
    ASSERT_EQ(state.tensors.size(), 4U);

    // K and V: Q8_0 with 1 KV head of 256 elements = (256 / 32) * 34 = 272 bytes.
    EXPECT_EQ(state.tensors[0].name, "k");
    EXPECT_EQ(state.tensors[0].kind, model::StateKind::kKv);
    EXPECT_EQ(state.tensors[0].encoding, Encoding::kQ8_0);
    EXPECT_EQ(state.tensors[0].cell_bytes, 272U);

    EXPECT_EQ(state.tensors[1].name, "v");
    EXPECT_EQ(state.tensors[1].kind, model::StateKind::kKv);
    EXPECT_EQ(state.tensors[1].encoding, Encoding::kQ8_0);
    EXPECT_EQ(state.tensors[1].cell_bytes, 272U);

    // indexer_k: F32 with idim = 64 -> 64 * 4 = 256 bytes.
    EXPECT_EQ(state.tensors[2].name, "indexer_k");
    EXPECT_EQ(state.tensors[2].kind, model::StateKind::kIndex);
    EXPECT_EQ(state.tensors[2].encoding, Encoding::kF32);
    EXPECT_EQ(state.tensors[2].cell_bytes, 256U);

    // indexer_blocks: BF16 with idim = 64 -> 64 * 2 = 128 bytes.
    EXPECT_EQ(state.tensors[3].name, "indexer_blocks");
    EXPECT_EQ(state.tensors[3].kind, model::StateKind::kBlockKeys);
    EXPECT_EQ(state.tensors[3].encoding, Encoding::kBF16);
    EXPECT_EQ(state.tensors[3].cell_bytes, 128U);
  }

  // Refused: unsupported head dimension D=128 for Q8_0 in QSA.
  {
    auto comp = MakeQsaAttention(128, "q8_0");
    auto inst = MakeInstance(comp, kFactsQwen38, Site::kLayer, 0, 1);
    EXPECT_FALSE(inst.has_value());
  }

  // F16 default remains unchanged.
  {
    auto comp = MakeQsaAttention(256, "f16");
    auto inst = MakeInstance(comp, kFactsQwen38, Site::kLayer, 0, 1);
    ASSERT_TRUE(inst.has_value()) << inst.error();
    auto state = inst->def->state(*inst);
    EXPECT_EQ(state.tensors[0].encoding, Encoding::kF16);
    EXPECT_EQ(state.tensors[0].cell_bytes, 512U);  // 256 * 2
  }
}

TEST(Q8StateTest, QsaQ8ReferenceRequiresAndUsesStoreRow) {
  auto comp = MakeQsaAttention(256, "q8_0");
  auto inst = MakeInstance(comp, kFactsQwen38, Site::kLayer, 0, 1);
  ASSERT_TRUE(inst.has_value()) << inst.error();

  RefIO io{.instance = &*inst};
  io.segments = {{.rows = 1, .past = 0, .read_cells = 1, .outputs = 1}};
  io.stream.assign(kFactsQwen38.width, 0.25);
  io.state_cells = {64, 64, 64, 16};
  io.state = {{{std::vector<double>(64 * 256, 0.0), std::vector<double>(64 * 256, 0.0),
                std::vector<double>(64 * 64, 0.0), std::vector<double>(16 * 64, 0.0)}}};

  const auto shapes = inst->def->shapes(*inst);
  ASSERT_TRUE(shapes.has_value());
  io.weights.resize(inst->def->roles.size());
  for (const auto& s : *shapes) {
    std::uint64_t count = 1;
    for (std::size_t i = 0; i < s.rank; ++i) count *= s.ne[i];
    io.weights[s.role].assign(count, 0.01f);
  }

  // Without store_row, reference fails.
  io.store_row = nullptr;
  EXPECT_FALSE(inst->def->reference(io).has_value());

  // With store_row, reference succeeds.
  io.store_row = &ref::StoreStateRow;
  auto res = inst->def->reference(io);
  ASSERT_TRUE(res.has_value()) << res.error();
  EXPECT_EQ(io.stream.size(), kFactsQwen38.width);
}

// ------------------------------------------------------------- ApplyKvCacheEncoding

TEST(Q8StateTest, ApplyKvCacheEncodingGemma4MultiLayer) {
  // Construct a model spec mimicking Gemma 4 + assistant drafter:
  // Target:
  //   Layer 0: SWA attention.dense (D=256)
  //   Layer 1: Global attention.dense (D=512, softcap=0.0)
  // Drafter:
  //   embedding.projected + Layer 0: attention.borrowed (D=512, target_layer=1)
  auto borrowed = MakeDenseAttention(512, "f16", 0.0);
  borrowed.kind = "attention.borrowed";
  borrowed.params.push_back(P("target_layer", I(1)));
  std::ranges::sort(borrowed.params, {}, &Param::name);
  std::vector<Param> proj_params{P("eps", R(1e-6)), P("target_width", I(256)), P("vocab", I(1024))};
  std::ranges::sort(proj_params, {}, &Param::name);

  spec::ModelSpec model{
      .pipeline = spec::PipelineKind::kLlm,
      .stages =
          {
              spec::Stage{
                  .name = "gemma4",
                  .role = spec::StageRole::kTarget,
                  .context = 4096,
                  .stream = {.streams = 1, .width = 256},
                  .layers =
                      {
                          spec::Layer{.components = {MakeDenseAttention(256, "f16", 0.0)}},
                          spec::Layer{.components = {MakeDenseAttention(512, "f16", 0.0)}},
                      },
                  .taps = {{.name = "features", .layer = 1, .point = spec::TapPoint::kStream}},
              },
              spec::Stage{
                  .name = "assistant",
                  .role = spec::StageRole::kDrafter,
                  .context = 4096,
                  .stream = {.streams = 1, .width = 256},
                  .embedding =
                      ComponentRef{
                          .kind = "embedding.projected", .version = 1, .params = proj_params},
                  .layers =
                      {
                          spec::Layer{.components = {borrowed}},
                      },
                  .reads = {{.stage = "gemma4", .kind = spec::ReadKind::kTap, .name = "features"},
                            {.stage = "gemma4", .kind = spec::ReadKind::kState, .name = "k"},
                            {.stage = "gemma4", .kind = spec::ReadKind::kState, .name = "v"}},
              },
          },
  };

  const auto old_digest = model.Digest();
  auto bytes = KvCacheEncodingBytes(model, Encoding::kQ8_0);
  ASSERT_TRUE(bytes.has_value()) << bytes.error();
  ASSERT_GT(*bytes, 0U);

  // Insufficient bytes rejected.
  EXPECT_FALSE(ApplyKvCacheEncoding(model, Encoding::kQ8_0, *bytes - 1).has_value());
  EXPECT_EQ(model.Digest(), old_digest);

  // Sufficient bytes admitted: both D=256 and D=512 layers (plus borrowed) quantized.
  ASSERT_TRUE(ApplyKvCacheEncoding(model, Encoding::kQ8_0, *bytes).has_value());
  EXPECT_NE(model.Digest(), old_digest);

  for (const auto& stage : model.stages) {
    for (const auto& layer : stage.layers) {
      for (const auto& comp : layer.components) {
        const auto* p = comp.Find("kv_encoding");
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(p->value, ParamValue(std::string("q8_0")));
      }
    }
  }
}

TEST(Q8StateTest, ApplyKvCacheEncodingQwen38WithMtpAndExemptComponents) {
  // Construct a model spec mimicking Qwen3.8:
  // Target stage (4 streams):
  //   Layer 0: attention.gdn (recurrent, unquantized)
  //   Layer 1: attention.qsa (D=256, full attention with indexer)
  // Drafter stage (MTP, 1 stream):
  //   embedding.mtp + Layer 0: attention.qsa (D=256)
  std::vector<Param> gdn_params{
      P("head_dim", I(128)),
      P("heads", I(2)),
      P("width", I(256)),
  };
  std::ranges::sort(gdn_params, {}, &Param::name);
  std::vector<Param> mtp_emb_params{
      P("eps", R(1e-6)),
      P("norm_zero_centered", B(true)),
      P("vocab", I(1024)),
  };
  std::ranges::sort(mtp_emb_params, {}, &Param::name);

  spec::ModelSpec model{
      .pipeline = spec::PipelineKind::kLlm,
      .stages =
          {
              spec::Stage{
                  .name = "qwen38",
                  .role = spec::StageRole::kTarget,
                  .context = 4096,
                  .stream = {.streams = 4, .width = 256},
                  .layers =
                      {
                          spec::Layer{
                              .components =
                                  {
                                      ComponentRef{.kind = "attention.gdn",
                                                   .version = 1,
                                                   .params = gdn_params},
                                      MakeQsaAttention(256, "f16"),
                                  },
                          },
                      },
                  .taps = {{.name = "streams", .layer = 0, .point = spec::TapPoint::kStreams}},
              },
              spec::Stage{
                  .name = "mtp",
                  .role = spec::StageRole::kDrafter,
                  .context = 4096,
                  .stream = {.streams = 1, .width = 256},
                  .embedding =
                      ComponentRef{.kind = "embedding.mtp", .version = 1, .params = mtp_emb_params},
                  .layers =
                      {
                          spec::Layer{
                              .components = {MakeQsaAttention(256, "f16")},
                          },
                      },
                  .reads = {{.stage = "qwen38", .kind = spec::ReadKind::kTap, .name = "streams"}},
              },
          },
  };

  const auto old_digest = model.Digest();
  auto bytes = KvCacheEncodingBytes(model, Encoding::kQ8_0);
  ASSERT_TRUE(bytes.has_value()) << bytes.error();
  ASSERT_GT(*bytes, 0U);

  ASSERT_TRUE(ApplyKvCacheEncoding(model, Encoding::kQ8_0, *bytes).has_value());
  EXPECT_NE(model.Digest(), old_digest);

  // Target stage checks:
  // attention.gdn remains unquantized (no kv_encoding param).
  EXPECT_EQ(model.stages[0].layers[0].components[0].Find("kv_encoding"), nullptr);
  // attention.qsa is quantized to q8_0.
  const auto* target_qsa_p = model.stages[0].layers[0].components[1].Find("kv_encoding");
  ASSERT_NE(target_qsa_p, nullptr);
  EXPECT_EQ(target_qsa_p->value, ParamValue(std::string("q8_0")));

  // Drafter stage checks:
  // attention.qsa is quantized to q8_0.
  const auto* mtp_qsa_p = model.stages[1].layers[0].components[0].Find("kv_encoding");
  ASSERT_NE(mtp_qsa_p, nullptr);
  EXPECT_EQ(mtp_qsa_p->value, ParamValue(std::string("q8_0")));
}

TEST(Q8StateTest, ApplyKvCacheEncodingSupportsMqa) {
  std::vector<Param> inject_params{
      P("eps", R(1e-6)),
      P("taps", I(1)),
      P("target_width", I(256)),
      P("vocab", I(1024)),
  };
  std::ranges::sort(inject_params, {}, &Param::name);
  spec::ModelSpec model{
      .pipeline = spec::PipelineKind::kLlm,
      .stages =
          {
              spec::Stage{
                  .name = "dsv4",
                  .role = spec::StageRole::kTarget,
                  .context = 4096,
                  .stream = {.streams = 1, .width = 256},
                  .layers =
                      {
                          spec::Layer{
                              .components =
                                  {
                                      MakeMqaAttention(512, "f16", true),
                                  },
                          },
                      },
                  .taps = {{.name = "tap0", .layer = 0, .point = spec::TapPoint::kStream}},
              },
              spec::Stage{
                  .name = "dflash",
                  .role = spec::StageRole::kDrafter,
                  .context = 4096,
                  .stream = {.streams = 1, .width = 256},
                  .embedding =
                      ComponentRef{
                          .kind = "embedding.inject", .version = 1, .params = inject_params},
                  .layers =
                      {
                          spec::Layer{
                              .components =
                                  {
                                      MakeMqaAttention(256, "f16", false),
                                  },
                          },
                      },
                  .reads = {{.stage = "dsv4", .kind = spec::ReadKind::kTap, .name = "tap0"}},
              },
          },
  };

  auto bytes = KvCacheEncodingBytes(model, Encoding::kQ8_0);
  ASSERT_TRUE(bytes.has_value()) << bytes.error();
  EXPECT_GT(*bytes, 0U);
  auto applied = ApplyKvCacheEncoding(model, Encoding::kQ8_0, *bytes);
  ASSERT_TRUE(applied.has_value()) << applied.error();
  const auto* mqa_p = model.stages[0].layers[0].components[0].Find("kv_encoding");
  ASSERT_NE(mqa_p, nullptr);
  EXPECT_EQ(mqa_p->value, ParamValue(std::string("q8_0")));
  const auto* dflash_p = model.stages[1].layers[0].components[0].Find("kv_encoding");
  ASSERT_NE(dflash_p, nullptr);
  EXPECT_EQ(dflash_p->value, ParamValue(std::string("q8_0")));
}

TEST(Q8StateTest, MqaAttentionD512AndD256ShapesAndState) {
  // DeepSeek V4 D=512 with indexer:
  {
    auto comp = MakeMqaAttention(512, "q8_0", true);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(inst.has_value()) << inst.error();
    auto state = inst->def->state(*inst);
    ASSERT_GE(state.tensors.size(), 5U);
    EXPECT_EQ(state.tensors[0].name, "kv");
    EXPECT_EQ(state.tensors[0].encoding, Encoding::kQ8_0);
    // 1 KV head, head_dim 512: (512 / 32) * 34 = 544 bytes.
    EXPECT_EQ(state.tensors[0].cell_bytes, 544U);

    EXPECT_EQ(state.tensors[1].name, "compressed");
    EXPECT_EQ(state.tensors[1].encoding, Encoding::kQ8_0);
    EXPECT_EQ(state.tensors[1].cell_bytes, 544U);

    EXPECT_EQ(state.tensors[2].name, "compressed_kv");
    EXPECT_EQ(state.tensors[2].encoding, Encoding::kF32);

    EXPECT_EQ(state.tensors[3].name, "compressed_score");
    EXPECT_EQ(state.tensors[3].encoding, Encoding::kF32);

    EXPECT_EQ(state.tensors[4].name, "index");
    EXPECT_EQ(state.tensors[4].encoding, Encoding::kF16);
    EXPECT_EQ(state.tensors[4].cell_bytes, 64U * 2U);
  }

  // DSpark D=256:
  {
    auto comp = MakeMqaAttention(256, "q8_0", false);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(inst.has_value()) << inst.error();
    auto state = inst->def->state(*inst);
    ASSERT_EQ(state.tensors.size(), 1U);
    EXPECT_EQ(state.tensors[0].name, "kv");
    EXPECT_EQ(state.tensors[0].encoding, Encoding::kQ8_0);
    // 1 KV head, head_dim 256: (256 / 32) * 34 = 272 bytes.
    EXPECT_EQ(state.tensors[0].cell_bytes, 272U);
  }

  // D=128 rejected for Q8_0:
  {
    auto comp = MakeMqaAttention(128, "q8_0", false);
    auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
    EXPECT_FALSE(inst.has_value());
  }
}

TEST(Q8StateTest, MqaAttentionReferenceQ8RequiresStoreRow) {
  auto comp = MakeMqaAttention(256, "q8_0", false);
  auto inst = MakeInstance(comp, kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(inst.has_value()) << inst.error();

  RefIO io;
  io.instance = &*inst;
  io.segments = {{.rows = 1, .past = 0, .read_cells = 1}};
  io.stream = std::vector<double>(256, 1.0);
  io.weights.resize(inst->def->roles.size());
  io.weights[0] = std::vector<float>(256 * 64, 0.1f);      // q_a
  io.weights[1] = std::vector<float>(64, 1.0f);            // q_norm
  io.weights[2] = std::vector<float>(64 * 4 * 256, 0.1f);  // q_b
  io.weights[3] = std::vector<float>(256 * 256, 0.1f);     // kv
  io.weights[4] = std::vector<float>(256, 1.0f);           // kv_norm
  io.weights[5] = std::vector<float>(4, 0.0f);             // sinks
  io.weights[6] = std::vector<float>(4 * 256 * 64, 0.1f);  // out_a
  io.weights[7] = std::vector<float>(64 * 256, 0.1f);      // out_b
  io.state = {{std::vector<double>(64 * 256, 0.0)}};
  io.state_cells = {64};

  // Without store_row -> Refused.
  EXPECT_FALSE(inst->def->reference(io).has_value());

  // With store_row -> Accepted.
  io.store_row = [](Encoding enc, std::span<const double> src,
                    std::span<double> dst) -> std::expected<void, std::string> {
    EXPECT_EQ(enc, Encoding::kQ8_0);
    EXPECT_EQ(src.size(), dst.size());
    std::ranges::copy(src, dst.begin());
    return {};
  };
  EXPECT_TRUE(inst->def->reference(io).has_value());
}

// ------------------------------------------------------------- Kernel Checks

TEST(Q8StateTest, AcceptsQuantAttentionPredicate) {
  const auto* cap = kg::CapabilityOf(kg::kFlashAttnVecQ8Name);
  ASSERT_NE(cap, nullptr);
  ASSERT_NE(cap->accepts, nullptr);

  // Accepted head dimensions: 64, 256, 512.
  for (const std::uint64_t head : {64U, 256U, 512U}) {
    execution::CapabilityQuery q{
        .k = head,
        .n = head,
        .arch = 121,
    };
    EXPECT_TRUE(cap->accepts(q)) << "head dim: " << head;
  }

  // Rejected head dimension 128.
  {
    execution::CapabilityQuery q{
        .k = 128,
        .n = 128,
        .arch = 121,
    };
    EXPECT_FALSE(cap->accepts(q));
  }

  // Accepted when F32 dense sinks operand is present (operand index 4).
  {
    std::array<execution::OperandFormat, 5> ops{};
    ops[4] = execution::OperandFormat{
        .operand = 4, .encoding = Encoding::kF32, .layout = format::Layout::kDense};
    execution::CapabilityQuery q{
        .operands = ops,
        .k = 256,
        .n = 256,
        .arch = 121,
    };
    EXPECT_TRUE(cap->accepts(q));
  }

  // Rejected when sinks operand is not F32.
  {
    std::array<execution::OperandFormat, 5> ops{};
    ops[4] = execution::OperandFormat{
        .operand = 4, .encoding = Encoding::kF16, .layout = format::Layout::kDense};
    execution::CapabilityQuery q{
        .operands = ops,
        .k = 256,
        .n = 256,
        .arch = 121,
    };
    EXPECT_FALSE(cap->accepts(q));
  }
}

TEST(Q8StateTest, CheckFlashAttnVecQuantHead512) {
  struct Context {
    ggml_context* ctx = nullptr;
    std::vector<std::uint8_t> buffer;
    Context() : buffer(4 * 1024 * 1024) {
      struct ggml_init_params params{
          .mem_size = buffer.size(),
          .mem_buffer = buffer.data(),
          .no_alloc = false,
      };
      ctx = ggml_init(params);
    }
    ~Context() {
      if (ctx) ggml_free(ctx);
    }
  };

  Context c;
  alignas(64) std::vector<std::uint8_t> storage(16 * 1024 * 1024, 0);
  std::size_t offset = 0;
  const auto bind = [&](ggml_tensor* t) {
    constexpr std::size_t kAlign = 64;
    offset = (offset + kAlign - 1) & ~(kAlign - 1);
    t->data = storage.data() + offset;
    offset += ggml_nbytes(t);
    return t;
  };

  const auto make_attention = [&](ggml_type k_type, ggml_type v_type, std::int64_t d,
                                  std::int64_t cells, float softcap, ggml_tensor* sinks = nullptr) {
    auto* q = bind(ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, d, 1, 4));
    auto* k = bind(ggml_new_tensor_3d(c.ctx, k_type, d, cells, 2));
    auto* v = bind(ggml_new_tensor_3d(c.ctx, v_type, d, cells, 2));
    auto* m = bind(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, cells, 1));
    auto* node = ggml_flash_attn_ext(c.ctx, q, k, v, m, 0.125f, 0.0f, softcap);
    if (sinks) node->src[4] = sinks;
    bind(node);
    ggml_prec_set_acc(node, GGML_PREC_F32);
    return node;
  };

  // D=512, Q8_0, softcap = 0.0 -> Accepted!
  EXPECT_TRUE(
      kg::CheckFlashAttnVecQuant(make_attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 512, 256, 0.0f))
          .has_value());

  // D=512, Q4_0 -> Rejected (Q4_0 is only compiled at 64 and 256).
  EXPECT_FALSE(
      kg::CheckFlashAttnVecQuant(make_attention(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 512, 256, 0.0f))
          .has_value());

  // D=512, softcap = 5.0 -> Rejected (softcap is only valid at D256).
  EXPECT_FALSE(
      kg::CheckFlashAttnVecQuant(make_attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 512, 256, 5.0f))
          .has_value());

  // D=256, Q8_0, softcap = 5.0 -> Accepted.
  EXPECT_TRUE(
      kg::CheckFlashAttnVecQuant(make_attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 256, 5.0f))
          .has_value());

  // With F32 sinks (4 query heads) -> Accepted!
  auto* sinks = bind(ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 4));
  EXPECT_TRUE(kg::CheckFlashAttnVecQuant(
                  make_attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 256, 0.0f, sinks))
                  .has_value());

  // With mismatched sinks count (8 instead of 4) -> Rejected.
  auto* bad_sinks = bind(ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 8));
  EXPECT_FALSE(kg::CheckFlashAttnVecQuant(
                   make_attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 256, 0.0f, bad_sinks))
                   .has_value());
}

}  // namespace llmp::components
