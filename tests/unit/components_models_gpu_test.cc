// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The components end to end on the real Gemma 2 2B and Gemma 3 4B QAT
// artifacts (label `models`): the spec synthesized from each v0 artifact
// (spec/mapping.h SynthesizeV0), validated against its tensors, built into
// a prefill step and three decode steps (components/step.h BuildStep) and
// run by the reference executor (double arithmetic over the decoded
// weights, F16 KV state), against today's production runner on the GPU
// (engine/gemma{2,3}_runner.h with the serving recipe) for the same prompt
// and the tokens it picks. The production path quantizes activations for
// its quantized products (Q8_1), rounds attention's operands to F16 and
// fuses norms, so the logits agree to a tolerance, not bit for bit: the same
// top token at every step (or two tokens within kTieMargin of each other on
// both sides), and every logit within kLogitTolerance of the reference's.
//
// What the tolerance can see: the weights' binding and decoding, the
// executor's operations against GGML's, the rotation's pairing, the norms'
// epsilon (Gemma 3) and a window that binds. What it cannot: at this
// prompt's positions a 2% query scale, Gemma 2's attention softcap or a
// local layer's rotation base move the logits about as far as the
// difference between the two arithmetics does, or less (measured on
// spark-b, 2026-10-09: 0.49 to 1.03 against 0.46 to 0.70; with the
// reference's products fed Q8_1-rounded activations and F16 queries the
// difference stays 0.35 to 0.71, so it is not the activations'
// quantization). Those parameters are pinned elsewhere: the spec this test
// synthesizes from the artifact is the tensor table's (spec_models_test),
// whose steps components_order_test holds to today's builders node for
// node, every arithmetic parameter bit for bit.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/gemma2_runner.h"
#include "engine/gemma3_runner.h"
#include "engine/paged_node.h"
#include "engine/support.h"
#include "kernels/reference/exec.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "tokenizer_fixtures.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace ref = llmp::kernels::reference;
namespace fs = std::filesystem;

// The largest logit difference allowed between the production runner and
// the reference (Gemma 2's logits are capped at 30, Gemma 3's are not).
// Measured on spark-b (2026-10-09; each case prints its report): the
// largest difference 0.46 to 0.70 per step, the mean 0.05 to 0.11, over
// the four steps of each model; the bound leaves headroom over 0.70.
constexpr double kLogitTolerance = 1.0;
// Two tokens whose logits are this close on both sides are a tie the two
// arithmetics may break differently (Gemma 2's third decode: 0.027 and
// 0.015).
constexpr double kTieMargin = 0.25;
constexpr std::array<std::int32_t, 6> kPrompt{2, 818, 5279, 529, 7001, 563};
constexpr std::uint32_t kDecodes = 3;

std::size_t Top(std::span<const float> row) {
  return static_cast<std::size_t>(std::ranges::max_element(row) - row.begin());
}

// Today's runner: the prompt's frontier logits, then each greedy decode's.
template <typename Runner, typename Options>
std::vector<std::vector<float>> Production(const Options& options) {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto runner = std::make_unique<Runner>(node, options, 0, 0);
  std::vector<en::PagedModel*> entered;
  std::vector<std::vector<float>> rows;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    entered.push_back(runner.get());
    if (auto x = runner->Setup(); !x) return x;
    if (auto x = node.MapWorkspace(runner->activations_needed(), runner->pool_needed()); !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + (16ULL << 20U));
    if (auto x = node.Start(llmp::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                              4 * node.StateCapacity()));
        !x)
      return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    if (auto x = runner->SelectSlots(std::array<std::uint32_t, 1>{0}); !x) return x;
    return node.WithRequest(0, runner->closure(), "components control", [&]() -> en::Status {
      std::vector<float> logits;
      const typename Runner::Work prefill{0, 0, kPrompt, &logits};
      if (auto x = runner->Wave(std::span(&prefill, 1)); !x) return x;
      rows.push_back(logits);
      for (std::uint32_t i = 0; i < kDecodes; ++i) {
        const std::array<std::int32_t, 1> token{static_cast<std::int32_t>(Top(rows.back()))};
        const typename Runner::Work decode{0, static_cast<std::uint32_t>(kPrompt.size() + i), token,
                                           &logits};
        if (auto x = runner->Wave(std::span(&decode, 1)); !x) return x;
        rows.push_back(logits);
      }
      return {};
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = runner.release();  // its memory may still be in use
  return rows;
}

std::vector<std::byte> ReadResource(const fs::path& root, const ar::Artifact& a,
                                    std::uint32_t resource) {
  const ar::Placement p = a.ResourcePlacement(resource).value();
  const ar::Group& g = a.groups()[p.group];
  const ar::Shard& shard = a.shards()[g.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(
      static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() + p.offset.value()));
  std::vector<std::byte> out(p.bytes.value());
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  EXPECT_TRUE(in.good());
  return out;
}

// The components' logits for the same steps: the prompt, then `tokens`.
std::vector<std::vector<float>> Reference(const fs::path& root, const ar::Artifact& a,
                                          const llmp::spec::ModelSpec& spec,
                                          std::span<const std::int32_t> tokens) {
  const auto source = llmp::spec::SourceOfV0(a);
  EXPECT_TRUE(source.has_value());
  if (!source) return {};
  const auto valid = co::ValidateSpec(spec, source->tensors);
  EXPECT_TRUE(valid.has_value()) << valid.error();
  constexpr std::uint32_t kContext = 512;
  const auto layout = co::StageLayout(spec, 0, kContext, 64);
  EXPECT_TRUE(layout.has_value());
  if (!layout) return {};
  std::map<std::string, std::vector<std::byte>> weights;
  std::map<std::pair<std::uint32_t, std::uint16_t>, std::vector<std::byte>> state;
  std::vector<std::vector<float>> rows;
  for (std::uint32_t step = 0; step <= tokens.size(); ++step) {
    const bool prefill = step == 0;
    const std::uint32_t past = prefill ? 0 : static_cast<std::uint32_t>(kPrompt.size()) + step - 1;
    const std::uint32_t n = prefill ? static_cast<std::uint32_t>(kPrompt.size()) : 1;
    const llmp::planner::StepShape shape{
        .segments = {{.slot = 0, .rows = n, .past = past, .read_cells = 256, .outputs = 1}}};
    auto graph = co::BuildStep(spec, 0, shape, {.tensors = source->tensors, .state = &*layout});
    EXPECT_TRUE(graph.has_value()) << graph.error();
    if (!graph) return {};
    auto exec = ref::Executor::Create(graph->graph);
    EXPECT_TRUE(exec.has_value()) << exec.error().detail;
    if (!exec) return {};
    for (const co::StepWeight& w : graph->weights) {
      EXPECT_EQ(w.parts.size(), 1U);
      const std::string& name = w.parts.front().first;
      auto& bytes = weights[name];
      if (bytes.empty()) bytes = ReadResource(root, a, a.FindResource(name).value());
      const ref::PartView part{.role = w.parts.front().second, .bytes = bytes};
      const auto bound = exec->Bind(w.tensor, std::span(&part, 1));
      EXPECT_TRUE(bound.has_value()) << name << ": " << bound.error().detail;
    }
    for (const co::StepState& s : graph->state) {
      auto& bytes = state[{s.ref.layer, s.ref.tensor}];
      if (bytes.empty()) {
        const std::vector<float> zeros(llmp::ir::Elements(graph->graph.tensor(s.tensor).shape()),
                                       0.0F);
        EXPECT_TRUE(exec->BindValues(s.tensor, zeros));
      } else {
        const ref::PartView part{.role = llmp::format::PartRole::kData, .bytes = bytes};
        EXPECT_TRUE(exec->Bind(s.tensor, std::span(&part, 1)));
      }
    }
    std::vector<std::int32_t> ids(kPrompt.begin(), kPrompt.end());
    if (!prefill) ids = {tokens[step - 1]};
    for (const co::StepInput& input : graph->inputs) {
      const auto values = co::InputValues(input, shape, ids);
      EXPECT_TRUE(values.has_value()) << values.error();
      EXPECT_TRUE(exec->BindInts(input.tensor, *values));
    }
    const auto ran = exec->Run();
    EXPECT_TRUE(ran.has_value()) << ran.error().detail;
    if (!ran) return {};
    auto logits = exec->Values(graph->logits);
    EXPECT_TRUE(logits.has_value());
    rows.push_back(std::move(*logits));
    for (const co::StepState& s : graph->state) {
      const auto bytes = exec->Bytes(s.tensor);
      state[{s.ref.layer, s.ref.tensor}].assign(bytes.begin(), bytes.end());
    }
  }
  return rows;
}

struct Case {
  std::string name;
  std::string path;
};

class ComponentsModelsGpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentsModelsGpu, StepsMatchTheProductionRunner) {
  const Case& c = GetParam();
  const fs::path root = llmp::test_support::ModelsDir() + "/" + c.path;
  if (std::error_code error; !fs::exists(root, error)) GTEST_SKIP() << "no artifact in " << root;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "components";
  std::vector<std::vector<float>> production;
  std::vector<llmp::spec::SourceValue> values;
  if (c.name == "gemma2") {
    production = Production<en::Gemma2Runner>(en::Gemma2Options{.artifact = root,
                                                                .out = out,
                                                                .context = 8192,
                                                                .max_rows = 128,
                                                                .slots = 2,
                                                                .max_wave_rows = 256,
                                                                .max_head_rows = 2,
                                                                .owner_decode = true,
                                                                .packed_prefill = true,
                                                                .bounded_roots = true,
                                                                .fuse_norms = true,
                                                                .fuse_quant_glu = true,
                                                                .fuse_norm_rope = false,
                                                                .fuse_norm_add = true});
    values = *llmp::spec::mappings::Gemma2V0Values(llmp::model::Gemma2_2B());
  } else {
    production = Production<en::Gemma3Runner>(en::Gemma3Options{.artifact = root,
                                                                .out = out,
                                                                .context = 8192,
                                                                .max_rows = 128,
                                                                .slots = 2,
                                                                .max_wave_rows = 256,
                                                                .max_head_rows = 2,
                                                                .owner_decode = true,
                                                                .packed_prefill = true,
                                                                .bounded_roots = true,
                                                                .device_masks = true,
                                                                .fuse_norms = true,
                                                                .fuse_quant_glu = true,
                                                                .fuse_norm_rope = true,
                                                                .fuse_norm_add = true});
    values = *llmp::spec::mappings::Gemma3V0Values(llmp::model::Gemma3_4BQat());
  }
  ASSERT_EQ(production.size(), kDecodes + 1);
  std::vector<std::int32_t> picked;
  for (std::uint32_t i = 0; i < kDecodes; ++i) {
    picked.push_back(static_cast<std::int32_t>(Top(production[i])));
  }
  auto artifact = ar::Artifact::Open(root, {.expected_id = {}, .trusted_owner = ::geteuid()});
  ASSERT_TRUE(artifact.has_value()) << artifact.error().ToString();
  const auto* mapping = llmp::spec::FindMapping(artifact->model().architecture);
  ASSERT_NE(mapping, nullptr);
  const auto spec = llmp::spec::SynthesizeV0(*artifact, *mapping, values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const auto reference = Reference(root, *artifact, *spec, picked);
  ASSERT_EQ(reference.size(), production.size());
  for (std::size_t i = 0; i < reference.size(); ++i) {
    ASSERT_EQ(reference[i].size(), production[i].size());
    double worst = 0.0;
    double sum = 0.0;
    for (std::size_t j = 0; j < reference[i].size(); ++j) {
      const double d = std::fabs(static_cast<double>(reference[i][j]) - production[i][j]);
      worst = std::max(worst, d);
      sum += d;
    }
    const std::size_t top = Top(reference[i]);
    const std::size_t chosen = Top(production[i]);
    // How far the production's choice is below the reference's top, by
    // each side's logits (0 when they agree).
    const double reference_margin = static_cast<double>(reference[i][top] - reference[i][chosen]);
    const double production_margin =
        static_cast<double>(production[i][chosen] - production[i][top]);
    std::printf(
        "%s step %zu: top %zu (production %zu, margins %.4f and %.4f), max |logit difference| "
        "%.4f, mean %.5f\n",
        c.name.c_str(), i, top, chosen, reference_margin, production_margin, worst,
        sum / static_cast<double>(reference[i].size()));
    // The same top token, or a near tie: the two tokens within kTieMargin
    // of each other on both sides.
    EXPECT_TRUE(top == chosen ||
                (reference_margin <= kTieMargin && production_margin <= kTieMargin))
        << "step " << i;
    EXPECT_LE(worst, kLogitTolerance) << "step " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Gemma, ComponentsModelsGpu,
    testing::Values(
        Case{.name = "gemma2",
             .path = "gemma2-import-20261007/artifacts/"
                     "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870"},
        Case{.name = "gemma3",
             .path = "gemma3-import-20261007/artifacts/"
                     "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb"}),
    [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
