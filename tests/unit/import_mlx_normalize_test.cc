// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "components/step.h"
#include "import/mlx_qwen38.h"
#include "import/reader.h"
#include "import/writer.h"
#if defined(LLMP_HAS_CUDA)
#include "kernels/decode/selected_source.h"
#endif
#include "qwen38_fixture.h"
#include "runtime/component_admission.h"
#include "spec/mappings/qwen38.h"
#include "tokenizer_fixtures.h"

namespace llmp::import {
namespace {
namespace fx = llmp::test_support::qwen38;
using format::Encoding;
using format::PartRole;

TEST(ImportMlxNormalizeTest, NativeSplitPartsValidateAcrossCompleteTarget) {
  auto f = fx::NativeMlx();
  EXPECT_EQ(f.source.architecture, "qwen4exp");
  EXPECT_EQ(f.source.container, "mlx");
  EXPECT_EQ(f.source.folds, std::vector<std::string>{"gdn.value_heads_tiled"});
  EXPECT_EQ(f.source.tensors.size(), 2954U);

  // Verify inventory hash computation
  std::vector<const spec::SourceTensor*> sorted;
  sorted.reserve(f.source.tensors.size());
  for (const auto& t : f.source.tensors) sorted.push_back(&t);
  std::ranges::sort(sorted, {}, [](const auto* tensor) { return tensor->name; });
  base::Sha256 hash;
  for (const auto* tensor : sorted) {
    hash.Update(tensor->name)
        .Update("\n")
        .Update(format::Info(tensor->format.encoding).name)
        .Update("\n");
    for (std::uint8_t i = 0; i < tensor->format.shape.rank; ++i)
      hash.Update(std::to_string(tensor->format.shape.ne[i])).Update(",");
    hash.Update("\n");
  }
  const auto hex = base::ToHex(hash.Finish());
  EXPECT_EQ(hex, "2d9281bb7d13656bccac99f285180bb96ce6922d7156252e5bf8c8816452a7cd");

  const auto& mapping = spec::mappings::Qwen38NativeMapping();
  EXPECT_TRUE(mapping.folds.empty());
  EXPECT_TRUE(mapping.policies.empty());
  EXPECT_TRUE(mapping.v0_assembly.empty());

  auto spec = spec::Synthesize(mapping, f.source, f.values);
  ASSERT_TRUE(spec) << spec.error();
  auto valid = components::ValidateSpec(*spec, f.source.tensors);
  ASSERT_TRUE(valid) << valid.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  const auto& stage = spec->stages[0];
  ASSERT_EQ(stage.layers.size(), 48U);
  EXPECT_EQ(stage.folds, std::vector<std::string>{"gdn.value_heads_tiled"});

  // Embedding and output head must NOT be tied.
  ASSERT_TRUE(stage.embedding);
  ASSERT_TRUE(stage.heads.output);
  EXPECT_NE(stage.embedding->weights[0].tensor, stage.heads.output->weights[0].tensor);

  unsigned linears = 0, attention = 0, moes = 0;
  for (const auto& layer : stage.layers) {
    for (const auto& c : layer.components) {
      if (c.kind == "attention.gdn") {
        ++linears;
        ASSERT_NE(c.Find("a_log"), nullptr);
        EXPECT_EQ(std::get<bool>(c.Find("a_log")->value), true);
      }
      if (c.kind == "attention.qsa") ++attention;
      if (c.kind == "ffn.moe") {
        ++moes;
        ASSERT_NE(c.Find("projection_scales"), nullptr);
        EXPECT_EQ(std::get<bool>(c.Find("projection_scales")->value), false);
        for (const auto role : {"gate_experts", "up_experts", "down_experts"}) {
          EXPECT_EQ(std::ranges::count_if(c.weights, [&](const auto& w) { return w.role == role; }),
                    3);
          EXPECT_EQ(std::ranges::count_if(c.weights,
                                          [&](const auto& w) {
                                            return w.role == role && w.part == PartRole::kBiases &&
                                                   w.experts;
                                          }),
                    1);
          EXPECT_EQ(std::ranges::count_if(c.weights,
                                          [&](const auto& w) {
                                            return w.role == role &&
                                                   w.part == PartRole::kBlockScales && w.experts;
                                          }),
                    1);
          EXPECT_EQ(std::ranges::count_if(c.weights,
                                          [&](const auto& w) {
                                            return w.role == role && w.part == PartRole::kCodes &&
                                                   w.experts;
                                          }),
                    1);
        }
      }
    }
  }
  EXPECT_EQ(linears, 36U);
  EXPECT_EQ(attention, 12U);
  EXPECT_EQ(moes, 48U);
  ASSERT_EQ(stage.side.size(), 1U);
  EXPECT_EQ(stage.side[0].source.weights.size(), 3U);
}

TEST(ImportMlxNormalizeTest, NativeMappingRejectsMissingPartAndAppliedRawFolds) {
  auto f = fx::NativeMlx();
  auto before = f.source.tensors;
  std::erase_if(f.source.tensors,
                [](const auto& t) { return t.name == "blk.0.ffn_gate_exps.weight_bias"; });
  auto missing = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
  ASSERT_TRUE(missing);
  EXPECT_FALSE(components::ValidateSpec(*missing, f.source.tensors));

  f.source.tensors = std::move(before);
  for (const auto fold : {"norm.plus_one", "ssm.a_neg_exp"}) {
    f.source.folds = {"gdn.value_heads_tiled", fold};
    auto contradictory =
        spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
    ASSERT_TRUE(contradictory);
    EXPECT_FALSE(components::ValidateSpec(*contradictory, f.source.tensors));
  }
  f.source.folds.clear();
  auto unpermuted = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
  ASSERT_TRUE(unpermuted);
  EXPECT_FALSE(components::ValidateSpec(*unpermuted, f.source.tensors));
}

TEST(ImportMlxNormalizeTest, UnknownSourceRefusalKeepsSourceOwnership) {
  Checkpoint source;
  source.container = "safetensors";
  source.architecture = "qwen4_exp";
  source.quantization = "mlx affine";
  source.tensors.push_back({.name = "owned source"});
  const auto* owned = source.tensors.data();
  EXPECT_FALSE(MlxTargetBound(source));
  EXPECT_FALSE(NormalizeMlxTarget(std::move(source), {}));
  EXPECT_EQ(source.tensors.data(), owned);
  ASSERT_EQ(source.tensors.size(), 1U);
  EXPECT_EQ(source.tensors[0].name, "owned source");
}

TEST(ImportMlxNormalizeModelsTest,
     RealStagedCheckpointNormalizesPublishesAdmitsAndPlansAffineSplitWeights) {
  const char* env_path = std::getenv("LLMP_MLX_QWEN38_SOURCE");
  const std::filesystem::path path =
      env_path != nullptr ? std::filesystem::path(env_path)
                          : std::filesystem::path(test_support::ModelsDir()) /
                                "models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80";
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    GTEST_SKIP() << "no staged MLX Qwen3.8 source: " << path;
  }
  auto checkpoint = ReadCheckpoint(path);
  ASSERT_TRUE(checkpoint) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->container, "mlx");
  EXPECT_EQ(checkpoint->architecture, "qwen4_exp");
  EXPECT_EQ(checkpoint->quantization, "mlx affine");
  EXPECT_EQ(checkpoint->files.size(), 24U);

  auto bound = MlxTargetBound(*checkpoint);
  ASSERT_TRUE(bound) << bound.error().ToString();

  NormalizationLimits limits{bound->metadata_bytes, bound->work_units};
  auto plan = NormalizeMlxTarget(std::move(*checkpoint), limits);
  ASSERT_TRUE(plan) << plan.error().ToString();
  EXPECT_EQ(plan->canonical().tensors.size(), 2954U);

#if defined(LLMP_HAS_CUDA)
  // Verify DecodeSelectedSource supports all Affine split decoders
  auto decode_source = kernels::decode::DecodeSelectedSource(format::Encoding::kAffine);
  ASSERT_TRUE(decode_source) << (decode_source ? "" : decode_source.error());
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.grouped.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.rows.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.vector.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.vector_glu.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.matrix.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.matrix_fma.affine.split"));
#endif

  // Verify component spec validation and stage layout for admitted spec
  const std::array<std::span<const spec::SourceTensor>, 1> tables{plan->canonical().tensors};
  auto valid_spec = components::ValidateSpec(plan->spec(), tables);
  ASSERT_TRUE(valid_spec) << (valid_spec ? "" : valid_spec.error());

  auto stage_layout = components::StageLayout(plan->spec(), 0, 8192, 512);
  ASSERT_TRUE(stage_layout) << (stage_layout ? "" : stage_layout.error());
  EXPECT_GT(stage_layout->bytes, 0U);
  RecordProperty("stage_layout_bytes", std::to_string(stage_layout->bytes));

  // NOLINTNEXTLINE(concurrency-mt-unsafe): read-only test environment.
  const char* store_root = std::getenv("LLMP_TEST_MLX_PUBLICATION_STORE");
  if (store_root == nullptr || *store_root == '\0')
    GTEST_SKIP() << "normalization checked; publication and admission need "
                    "LLMP_TEST_MLX_PUBLICATION_STORE (a scratch store for a v1 artifact)";
  const std::filesystem::path scratch(store_root);
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  WriteOptions options{
      .store = scratch / "mlx-qwen38-v1",
      .build = "test",
      .metadata_bytes = std::max(bound->metadata_bytes, std::uint64_t{1} << 30U),
  };
  auto published = WriteArtifact(*plan, options);
  ASSERT_TRUE(published) << published.error().ToString();
  RecordProperty("artifact_id", published->id);
  RecordProperty("artifact_path", published->path.string());

  auto artifact = artifact::Artifact::Open(published->path, {.expected_id = published->id});
  ASSERT_TRUE(artifact) << artifact.error().ToString();

  // Verify admission
  auto artifact_admitted = runtime::CheckComponentArtifactAdmission(*artifact);
  ASSERT_TRUE(artifact_admitted) << (artifact_admitted ? "" : artifact_admitted.error());

  auto admission = runtime::ReadComponentAdmission(*artifact);
  ASSERT_TRUE(admission) << (admission ? "" : admission.error());
  EXPECT_EQ(admission->model.pipeline, spec::PipelineKind::kLlm);
  EXPECT_EQ(admission->model.stages.size(), 1U);
  EXPECT_EQ(admission->model.stages[0].layers.size(), 48U);
  EXPECT_EQ(admission->model.stages[0].folds, std::vector<std::string>{"gdn.value_heads_tiled"});
  EXPECT_EQ(admission->source.container, "mlx");
  EXPECT_EQ(admission->source.architecture, "qwen4exp");

  spec::Envelope envelope{.context = 8192, .slots = 1};
  auto policy = runtime::ComponentAdmissionPolicy(admission->model, admission->source,
                                                  artifact->sources(), envelope);
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());

  RecordProperty("published_tensors", std::to_string(admission->source.tensors.size()));

  std::filesystem::remove_all(scratch, ec);
}

}  // namespace
}  // namespace llmp::import
