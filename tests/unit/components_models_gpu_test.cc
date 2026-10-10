// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The components end to end on the real Gemma 2 2B and Gemma 3 4B QAT
// artifacts (label `models`): the spec synthesized from each v0 artifact
// (spec/mapping.h SynthesizeV0), validated against its tensors, built into
// a prefill step and three decode steps (components/step.h BuildStep) and
// run by the reference executor (double arithmetic over the decoded
// weights, F16 KV state), against the served model on the GPU (the
// runtime's server over the shared engine, runtime/engine_llm.h) for the
// same prompt and the tokens it picks. The served path quantizes
// activations for its quantized products (Q8_1), rounds attention's
// operands to F16 and fuses norms, so the logits agree to a tolerance, not
// bit for bit: the same top token at every step (or two tokens within
// kTieMargin of each other on both sides), and every logit within
// kLogitTolerance of the reference's.
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
// whose steps were once held to the M3 builders node for node, every
// arithmetic parameter bit for bit (retired with them after the
// migration's gate).

#include <gtest/gtest.h>
#include <sys/stat.h>
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
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "kernels/reference/exec.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/step_shape.h"
#include "runtime/serving.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "tokenizer_fixtures.h"

namespace {

namespace co = llmp::components;
namespace rt = llmp::runtime;
namespace cfg = llmp::config;
namespace ar = llmp::artifact;
namespace ref = llmp::kernels::reference;
namespace fs = std::filesystem;

// The largest logit difference allowed between the served model and the
// reference (Gemma 2's logits are capped at 30, Gemma 3's are not).
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

// The served model (the runtime's server over the shared engine, one owner
// at a 4,096 context): the prompt's frontier logits, then each greedy
// decode's.
std::vector<std::vector<float>> Production(const fs::path& root, const std::string& name) {
  std::vector<std::vector<float>> rows;
  const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const fs::path scratch_root = base != nullptr ? base : testing::TempDir();
  std::error_code error;
  fs::create_directories(scratch_root, error);
  std::string dir = (scratch_root / "components-XXXXXX").string();
  if (mkdtemp(dir.data()) == nullptr) {
    ADD_FAILURE() << "no scratch directory";
    return rows;
  }
  const fs::path scratch = dir;
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  auto life = std::make_unique<Lifetime>();
  life->roles.installed = root.parent_path();
  life->roles.spill = scratch / "spill";
  life->roles.state = scratch / "state";
  for (const auto& path : {life->roles.spill, life->roles.state}) {
    if (!fs::create_directory(path) || chmod(path.c_str(), 0700) != 0) {
      ADD_FAILURE() << "no private directory " << path;
      return rows;
    }
  }
  cfg::ModelEntry entry;
  entry.name = name;
  entry.artifact = root.filename().string();
  entry.overrides["context"] = std::int64_t{4096};
  entry.overrides["prefill_chunk"] = std::int64_t{128};
  entry.overrides["max_slots"] = std::int64_t{1};
  life->config.models.push_back(entry);
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, stderr);
  const auto run = [&]() -> rt::Status {
    if (auto r = life->server->Start(true); !r) return r;
    auto* model = dynamic_cast<rt::Llm*>(life->server->Find(name));
    if (model == nullptr) return std::unexpected("the configured model is missing");
    rt::SwapParts parts;
    if (auto r = life->server->Activate(*model, parts); !r) return r;
    auto branch = model->branch(0);
    if (!branch) return std::unexpected(branch.error());
    const std::array<rt::Llm::Branch*, 1> selected{*branch};
    if (auto r = life->server->SelectRequestBranches(*model, selected); !r) return r;
    std::vector<float> logits;
    if (auto r = (*branch)->Prefill(kPrompt, logits); !r) return r;
    rows.push_back(logits);
    for (std::uint32_t i = 0; i < kDecodes; ++i) {
      const std::array<std::int32_t, 1> token{static_cast<std::int32_t>(Top(rows.back()))};
      if (auto r = (*branch)->Prefill(token, logits); !r) return r;
      rows.push_back(logits);
    }
    return {};
  }();
  EXPECT_TRUE(run) << (run ? "" : run.error());
  const auto retired = life->server->TearDown();
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) {
    std::ignore = life.release();  // its memory may still be in use
    return rows;
  }
  life.reset();
  fs::remove_all(scratch, error);
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

TEST_P(ComponentsModelsGpu, StepsMatchTheServedModel) {
  const Case& c = GetParam();
  const fs::path root = llmp::test_support::ModelsDir() + "/" + c.path;
  if (std::error_code error; !fs::exists(root, error)) GTEST_SKIP() << "no artifact in " << root;
  const std::vector<std::vector<float>> production = Production(root, c.name);
  const std::vector<llmp::spec::SourceValue> values =
      c.name == "gemma2" ? *llmp::spec::mappings::Gemma2V0Values(llmp::model::Gemma2_2B())
                         : *llmp::spec::mappings::Gemma3V0Values(llmp::model::Gemma3_4BQat());
  ASSERT_EQ(production.size(), kDecodes + 1);
  std::vector<std::int32_t> picked;
  for (std::uint32_t i = 0; i < kDecodes; ++i) {
    picked.push_back(static_cast<std::int32_t>(Top(production[i])));
  }
  auto artifact = ar::Artifact::Open(root, {.expected_id = {}});
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
