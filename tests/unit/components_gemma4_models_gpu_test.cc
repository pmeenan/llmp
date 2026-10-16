// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 4 components end to end on the real artifacts (label
// `models`): the specs the published v1 artifacts store (spec/stored.h
// StoredSpec), validated against their tensors (SourceOf), built into steps
// (components/step.h BuildStep) and run by the reference executor (double
// arithmetic over the decoded weights, F16 KV state), against the served
// model on the GPU (the runtime's server over the shared engine,
// runtime/engine_llm.h) for the same prompt and the tokens it picks:
//
//   - 26B-A4B: a two-token prompt and one decode;
//   - 31B: the two-token prompt.
//
// The served path quantizes activations for its quantized products,
// rounds attention's operands to F16 and fuses norms, so the logits agree
// to a tolerance, not bit for bit: the same top token at every step (or two
// tokens within kTieMargin of each other on both sides) and every logit
// within its tolerance of the reference's. Each case prints its measured
// differences. (The assistant's draft passes were once compared the same
// way against the M3 assistant, retired with it after the migration's
// gate; the engine's are held bit for bit by component_runner_gemma4_gpu_test
// run to run and by the served assistant's acceptance in
// gemma4_serving_gpu_test.)

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "planner/step_shape.h"
#include "runtime/serving.h"
#include "spec/mapping.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {

namespace co = llmp::components;
namespace rt = llmp::runtime;
namespace cfg = llmp::config;
namespace ar = llmp::artifact;
namespace ref = llmp::kernels::reference;
namespace fs = std::filesystem;
namespace pl = llmp::planner;

// The largest logit difference allowed between the served model and the
// reference (the targets' logits are capped at 30), set over the differences measured on spark-b
// (2026-10-09; each case prints its report) with headroom, as components_models_gpu_test's. The
// dense 31B's prompt: 2.11 at most, 0.29 on average. The routed 26B-A4B's prompt 6.20 and 1.91, its
// decode 2.97 and 0.73. Where the 26B's differ (review, 2026-10-09, spark-b, the reference run
// again with each row's expert ids forced to the production's): routing explains the decode's
// excess (forced, 2.31 and 0.33: the production chose otherwise at near ties, probabilities 1.002
// to 1.011 apart, each case printing its closest), not the prompt's (forced, 6.24 and 1.95). The
// prompt's is the production's arithmetic (quantized activations, F16 operands) gathered over the
// layers (0.3 to 2.5% of layer 0's output, the 31B's 0.25 to 0.37%; 6% by the last layer's
// attention), then grown by that layer's cancellation in the frontier row (its output's norm 38
// from an input of 92: 13% of it). Every top token agrees.
// Against the served engine (the mapping's production recipe: fused norms and, on the 31B, the
// quantized GLU; spark, 2026-10-10) the same prompts give the dense 31B 3.48 and 0.40 and the
// routed 26B-A4B 5.14 and 1.45 (prompt), 2.45 and 0.55 (decode). The figures above were taken
// against the retired M3 runner at its default, unfused options (31B 2.11 and 0.29; 26B 6.20 and
// 1.91, 2.97 and 0.73), so the dense bound is raised with the same ~30% headroom.
constexpr double kDenseTolerance = 4.5;
constexpr double kRoutedTolerance = 8.0;
// Two tokens whose logits are this close on both sides are a tie the two
// arithmetics may break differently.
constexpr double kTieMargin = 0.25;
constexpr std::array<std::int32_t, 2> kPrompt{2, 818};
constexpr std::uint32_t kContext = 512;
constexpr std::uint32_t kRows = 64;

std::int32_t Top(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

// ---------------------------------------------------------------- production

// What the served model gives (the runtime's server over the shared engine,
// one owner at a 4,096 context, no drafter): the prompt's frontier logits,
// then the decode's of the token it picks.
struct Production {
  std::vector<float> prompt, decode;
};

Production RunProduction(const fs::path& target, bool decode) {
  Production out;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* base = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path scratch_root = base != nullptr ? base : testing::TempDir();
  std::error_code error;
  fs::create_directories(scratch_root, error);
  std::string dir = (scratch_root / "components4-XXXXXX").string();
  if (mkdtemp(dir.data()) == nullptr) {
    ADD_FAILURE() << "no scratch directory";
    return out;
  }
  const fs::path scratch = dir;
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  auto life = std::make_unique<Lifetime>();
  life->roles.installed = target.parent_path();
  life->roles.spill = scratch / "spill";
  life->roles.state = scratch / "state";
  for (const auto& path : {life->roles.spill, life->roles.state}) {
    if (!fs::create_directory(path) || chmod(path.c_str(), 0700) != 0) {
      ADD_FAILURE() << "no private directory " << path;
      return out;
    }
  }
  cfg::ModelEntry entry;
  entry.name = "gemma";
  entry.artifact = target.filename().string();
  entry.overrides["context"] = std::int64_t{4096};
  entry.overrides["prefill_chunk"] = std::int64_t{128};
  entry.overrides["max_slots"] = std::int64_t{1};
  life->config.models.push_back(entry);
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, stderr);
  const auto run = [&]() -> rt::Status {
    if (auto r = life->server->Start(true); !r) return r;
    auto* model = dynamic_cast<rt::Llm*>(life->server->Find("gemma"));
    if (model == nullptr) return std::unexpected("the configured model is missing");
    rt::SwapParts parts;
    if (auto r = life->server->Activate(*model, parts); !r) return r;
    auto branch = model->branch(0);
    if (!branch) return std::unexpected(branch.error());
    const std::array<rt::Llm::Branch*, 1> selected{*branch};
    if (auto r = life->server->SelectRequestBranches(*model, selected); !r) return r;
    if (auto r = (*branch)->Prefill(kPrompt, out.prompt); !r) return r;
    if (!decode) return {};
    const std::array<std::int32_t, 1> token{Top(out.prompt)};
    return (*branch)->Prefill(token, out.decode);
  }();
  EXPECT_TRUE(run) << (run ? "" : run.error());
  const auto retired = life->server->TearDown();
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) {
    std::ignore = life.release();  // its memory may still be in use
    return out;
  }
  life.reset();
  fs::remove_all(scratch, error);
  return out;
}

// ----------------------------------------------------------------- reference

// One weight's bytes: a resource, or an expert array's slices one after
// the other (the IR's [K, N, experts]).
std::vector<std::byte> ReadWeight(const fs::path& root, const ar::Artifact& a,
                                  const std::string& name) {
  const auto read = [&](const ar::Placement& p, std::vector<std::byte>& out) {
    const ar::Group& g = a.groups()[p.group];
    const ar::Shard& shard = a.shards()[g.shard];
    std::ifstream in(root / shard.path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() +
                                         p.offset.value()));
    const std::size_t at = out.size();
    out.resize(at + p.bytes.value());
    in.read(reinterpret_cast<char*>(out.data() + at),
            static_cast<std::streamsize>(p.bytes.value()));
    EXPECT_TRUE(in.good()) << name;
  };
  std::vector<std::byte> out;
  if (const auto r = a.FindResource(name); r) {
    read(a.ResourcePlacement(*r).value(), out);
    return out;
  }
  const auto arrays = a.expert_arrays();
  const auto at = std::ranges::find(arrays, name, &ar::ExpertArray::name);
  EXPECT_NE(at, arrays.end()) << name;
  if (at == arrays.end()) return out;
  const auto index = static_cast<std::uint32_t>(at - arrays.begin());
  for (std::uint32_t e = 0; e < at->count; ++e) read(a.SlicePlacement(index, e).value(), out);
  return out;
}

struct Model {
  fs::path root;
  ar::Artifact* artifact;
};

// The reference's runs over one composed spec: its weights by (stage,
// name), each stage's state by (stage, layer, tensor) for slot 0.
class Reference {
 public:
  Reference(const llmp::spec::ModelSpec& spec, std::vector<Model> models,
            std::vector<llmp::spec::Source> sources)
      : spec_(spec), models_(std::move(models)), sources_(std::move(sources)) {
    for (const auto& s : sources_) tables_.emplace_back(s.tensors);
    for (std::size_t i = 0; i < spec_.stages.size(); ++i) {
      auto layout = co::StageLayout(spec_, i, kContext, kRows);
      EXPECT_TRUE(layout.has_value()) << (layout ? "" : layout.error());
      layouts_.push_back(std::move(*layout));
    }
    for (const auto& l : layouts_) stage_layouts_.push_back(&l);
  }

  const llmp::model::StateLayoutDerived& layout(std::size_t stage) const {
    return layouts_.at(stage);
  }
  // One step: its logits.
  std::vector<float> Run(const pl::StepShape& shape, std::span<const std::int32_t> tokens) {
    const auto stage = shape.stage;
    const llmp::spec::Stage& s = spec_.stages.at(stage);
    auto graph = co::BuildStep(spec_, stage, shape,
                               {.tensors = tables_.at(s.artifact),
                                .state = &layouts_.at(stage),
                                .stages = stage_layouts_,
                                .artifacts = tables_});
    EXPECT_TRUE(graph.has_value()) << (graph ? "" : graph.error());
    if (!graph) return {};
    auto exec = ref::Executor::Create(graph->graph);
    EXPECT_TRUE(exec.has_value()) << (exec ? "" : exec.error().detail);
    if (!exec) return {};
    for (const co::StepWeight& w : graph->weights) {
      const std::string& name = w.parts.front().first;
      auto& bytes = weights_[{w.ref.stage, name}];
      if (bytes.empty()) {
        const Model& m = models_.at(spec_.stages.at(w.ref.stage).artifact);
        bytes = ReadWeight(m.root, *m.artifact, name);
      }
      const ref::PartView part{.role = w.parts.front().second, .bytes = bytes};
      const auto bound = exec->Bind(w.tensor, std::span(&part, 1));
      EXPECT_TRUE(bound.has_value()) << name << ": " << (bound ? "" : bound.error().detail);
    }
    for (const co::StepState& st : graph->state) {
      auto& bytes = state_[{st.ref.stage, st.ref.layer, st.ref.tensor}];
      if (bytes.empty()) {
        const std::vector<float> zeros(llmp::ir::Elements(graph->graph.tensor(st.tensor).shape()),
                                       0.0F);
        EXPECT_TRUE(exec->BindValues(st.tensor, zeros));
      } else {
        const ref::PartView part{.role = llmp::format::PartRole::kData, .bytes = bytes};
        EXPECT_TRUE(exec->Bind(st.tensor, std::span(&part, 1)));
      }
    }
    for (const co::StepInput& input : graph->inputs) {
      const auto values = co::InputValues(input, shape, tokens);
      EXPECT_TRUE(values.has_value()) << (values ? "" : values.error());
      EXPECT_TRUE(exec->BindInts(input.tensor, *values));
    }
    const auto ran = exec->Run();
    EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error().detail);
    if (!ran) return {};
    // How close each routed selection came to another (the probability of
    // its last chosen expert over the first not chosen): a production whose
    // arithmetic differs may choose otherwise where it is near 1.
    double closest = INFINITY;
    std::size_t selections = 0;
    for (const llmp::ir::Node& n : graph->graph.nodes()) {
      if (n.key.op != llmp::ir::Op::kRoute) continue;
      const auto p = n.Params<llmp::ir::RouteParams>();
      const auto logits = exec->Values(n.src[0]);
      if (!logits) continue;
      for (std::size_t t = 0; t + p.experts <= logits->size(); t += p.experts) {
        std::vector<float> row(logits->begin() + static_cast<std::ptrdiff_t>(t),
                               logits->begin() + static_cast<std::ptrdiff_t>(t + p.experts));
        std::ranges::sort(row, std::greater<>());
        closest = std::min(closest, std::exp(static_cast<double>(row[p.used - 1] - row[p.used])));
        ++selections;
      }
    }
    if (selections != 0) {
      std::printf("routing: %zu selections, the closest a ratio of %.6f\n", selections, closest);
    }
    std::vector<float> out = *exec->Values(graph->logits);
    for (const co::StepState& st : graph->state) {
      const auto bytes = exec->Bytes(st.tensor);
      state_[{st.ref.stage, st.ref.layer, st.ref.tensor}].assign(bytes.begin(), bytes.end());
    }
    return out;
  }

 private:
  using Key = std::tuple<std::uint16_t, std::uint32_t, std::uint16_t>;
  const llmp::spec::ModelSpec& spec_;
  std::vector<Model> models_;
  std::vector<llmp::spec::Source> sources_;
  std::vector<std::span<const co::BoundTensor>> tables_;
  std::vector<llmp::model::StateLayoutDerived> layouts_;
  std::vector<const llmp::model::StateLayoutDerived*> stage_layouts_;
  std::map<std::pair<std::uint16_t, std::string>, std::vector<std::byte>> weights_;
  std::map<Key, std::vector<std::byte>> state_;
};

// The served row against the reference's: the same top token (or a
// near tie) and every logit within `tolerance`; prints the differences.
void ExpectClose(std::string_view what, std::span<const float> reference,
                 std::span<const float> production, double tolerance) {
  ASSERT_EQ(reference.size(), production.size()) << what;
  double worst = 0.0;
  double sum = 0.0;
  for (std::size_t j = 0; j < reference.size(); ++j) {
    const double d = std::fabs(static_cast<double>(reference[j]) - production[j]);
    worst = std::max(worst, d);
    sum += d;
  }
  const auto top = static_cast<std::size_t>(Top(reference));
  const auto chosen = static_cast<std::size_t>(Top(production));
  const double reference_margin = static_cast<double>(reference[top] - reference[chosen]);
  const double production_margin = static_cast<double>(production[chosen] - production[top]);
  std::printf(
      "%.*s: top %zu (production %zu, margins %.4f and %.4f), max |logit difference| %.4f, "
      "mean %.5f\n",
      static_cast<int>(what.size()), what.data(), top, chosen, reference_margin, production_margin,
      worst, sum / static_cast<double>(reference.size()));
  EXPECT_TRUE(top == chosen || (reference_margin <= kTieMargin && production_margin <= kTieMargin))
      << what;
  EXPECT_LE(worst, tolerance) << what;
}

// The published v1 artifact the store's v1-ids.json names `name`.
std::optional<fs::path> Installed(std::string_view name) {
  const auto id = llmp::test_support::V1IdNamed(name);
  if (!id) return std::nullopt;
  return fs::path(llmp::test_support::V1Store()) / *id;
}

TEST(ComponentsGemma4ModelsGpu, Gemma26MatchesTheServedModel) {
  const auto target = Installed("gemma26");
  if (!target) GTEST_FAIL() << "no published v1 Gemma 4 26B in " << llmp::test_support::V1Store();
  const Production production = RunProduction(*target, true);
  ASSERT_FALSE(production.prompt.empty());
  ASSERT_FALSE(production.decode.empty());
  auto ta = ar::Artifact::Open(*target, {.expected_id = {}});
  ASSERT_TRUE(ta.has_value());
  const auto spec = llmp::spec::StoredSpec(*ta);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  auto ts = llmp::spec::SourceOf(*ta);
  ASSERT_TRUE(ts.has_value()) << ts.error();
  const auto valid = co::ValidateSpec(*spec, ts->tensors);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  Reference reference(*spec, {{*target, &*ta}}, {*ts});
  const auto past = static_cast<std::uint32_t>(kPrompt.size());
  const auto prompt = reference.Run(
      {.stage = 0,
       .phase = pl::Phase::kPrefill,
       .segments = {{.slot = 0, .rows = past, .past = 0, .read_cells = 256, .outputs = 1}}},
      kPrompt);
  ExpectClose("gemma26 prompt", prompt, production.prompt, kRoutedTolerance);
  // The decode of the served model's pick.
  const std::array<std::int32_t, 1> next{Top(production.prompt)};
  const auto decode = reference.Run(
      {.stage = 0,
       .phase = pl::Phase::kDecode,
       .segments = {{.slot = 0, .rows = 1, .past = past, .read_cells = 256, .outputs = 1}}},
      next);
  ExpectClose("gemma26 decode", decode, production.decode, kRoutedTolerance);
}

TEST(ComponentsGemma4ModelsGpu, Gemma31MatchesTheServedModel) {
  const auto target = Installed("gemma31");
  if (!target) GTEST_FAIL() << "no published v1 Gemma 4 31B in " << llmp::test_support::V1Store();
  const Production production = RunProduction(*target, false);
  ASSERT_FALSE(production.prompt.empty());
  auto ta = ar::Artifact::Open(*target, {.expected_id = {}});
  ASSERT_TRUE(ta.has_value());
  const auto spec = llmp::spec::StoredSpec(*ta);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  auto ts = llmp::spec::SourceOf(*ta);
  ASSERT_TRUE(ts.has_value()) << ts.error();
  const auto valid = co::ValidateSpec(*spec, ts->tensors);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  Reference reference(*spec, {{*target, &*ta}}, {*ts});
  const auto prompt =
      reference.Run({.stage = 0,
                     .phase = pl::Phase::kPrefill,
                     .segments = {{.slot = 0,
                                   .rows = static_cast<std::uint32_t>(kPrompt.size()),
                                   .past = 0,
                                   .read_cells = 256,
                                   .outputs = 1}}},
                    kPrompt);
  ExpectClose("gemma31 prompt", prompt, production.prompt, kDenseTolerance);
}

}  // namespace
