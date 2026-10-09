// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h) over the component step
// source (engine/component_source.h) and the GGML module's lowering
// (kernels/ggml/step_program.h), on the real Gemma 2 2B and Gemma 3 4B QAT
// artifacts (label `models`), against today's runners (engine/gemma{2,3}_
// runner.h) configured to plan what the lowering reproduces: the serving
// recipe's fused norms, quantized GeGLU, shared Q8 preparation, device
// masks, Gemma 3's fused norm-rope and grouped KV stores, with per-owner
// attention and the head's rows gathered after the final norm (the owner
// attention forms and the narrowed final layer, `frontier_head`, are slice
// G1b's). For the same prompt and then three decodes of the tokens today's
// runner picks, each step's logits are bit for bit today's: the same
// kernels in the same order over the same values, with the masks built on
// the device and with them built on the host and staged by the runner. The decode shape repeats,
// so its program is captured on its second run and replayed after
// (D-090), and the replays' logits are compared too.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
#include "engine/component_source.h"
#include "engine/gemma2_runner.h"
#include "engine/gemma3_runner.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/support.h"
#include "kernels/ggml/step_program.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "tokenizer_fixtures.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace fs = std::filesystem;

constexpr std::array<std::int32_t, 6> kPrompt{2, 818, 5279, 529, 7001, 563};
constexpr std::uint32_t kDecodes = 3;
constexpr std::uint32_t kContext = 8192;
constexpr std::uint32_t kMaxRows = 128;

std::size_t Top(std::span<const float> row) {
  return static_cast<std::size_t>(std::ranges::max_element(row) - row.begin());
}

// Today's runner: the prompt's frontier logits, then each decode's of
// `tokens` (picked by the host from today's own logits when empty).
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
    return node.WithRequest(0, runner->closure(), "lowering control", [&]() -> en::Status {
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

pl::StepShape StepOf(pl::Phase phase, std::uint32_t past, std::uint32_t rows) {
  return {.stage = 0,
          .phase = phase,
          .segments = {{.slot = 0,
                        .rows = rows,
                        .past = past,
                        .read_cells = (past + rows + 255) / 256 * 256,
                        .outputs = 1}},
          .row_invariant = false,
          .read_align = 256};
}

struct Component {
  std::vector<std::vector<float>> rows;
  std::uint64_t replayed = 0;
};

// The model runner over the component source and the GGML lowering, for
// the prompt and then `tokens`, with masks built on the device or on the
// host (staged by the runner).
Component Lowered(const fs::path& root, const llmp::spec::ModelSpec& spec,
                  std::string_view policy_text, std::span<const std::int32_t> tokens,
                  const fs::path& scratch, bool device_masks) {
  Component out;
  auto artifact = ar::Artifact::Open(root, {.expected_id = {}, .trusted_owner = ::geteuid()});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return out;
  const auto source = llmp::spec::SourceOfV0(*artifact);
  EXPECT_TRUE(source.has_value());
  if (!source) return out;
  const auto valid = co::ValidateSpec(spec, source->tensors);
  EXPECT_TRUE(valid.has_value()) << valid.error();
  auto layout = co::StageLayout(spec, 0, kContext, kMaxRows);
  EXPECT_TRUE(layout.has_value()) << layout.error();
  if (!layout) return out;
  const pl::StepShape prefill = StepOf(pl::Phase::kPrefill, 0, kPrompt.size());
  // Each weight's resource, by what the steps bind (the head's included).
  std::map<llmp::ir::WeightRef, std::uint32_t> resource;
  {
    auto step = co::BuildStep(spec, 0, prefill, {.tensors = source->tensors, .state = &*layout});
    EXPECT_TRUE(step.has_value()) << step.error();
    if (!step) return out;
    for (const co::StepWeight& w : step->weights) {
      const auto found = artifact->FindResource(w.parts.front().first);
      EXPECT_TRUE(found.has_value()) << w.parts.front().first;
      if (found) resource[w.ref] = *found;
    }
  }

  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::unique_ptr<en::ModelRunner> runner;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::vector<en::PagedModel*> entered;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    auto made = kg::GgmlBackend::Create(0, node.execution(), node.stream(0), nullptr,
                                        {.lower = {.device_masks = device_masks}});
    if (!made) return std::unexpected(made.error());
    backend = std::move(*made);
    auto weights = std::make_unique<en::PagedWeights>();
    if (auto x = weights->Open(root); !x) return x;
    en::PagedWeights* paged = weights.get();
    auto policy = pl::ParsePolicy(policy_text);
    if (!policy) return std::unexpected(policy.error());
    auto component = en::ComponentStepSource::Create(
        {.spec = &spec,
         .stage = 0,
         .bindings = {.tensors = source->tensors, .state = &*layout},
         .bound_digest = spec.Digest(),
         .policy = *policy,
         .backend = backend->Backend()});
    if (!component) return std::unexpected(component.error());
    en::RunnerConfig config;
    config.slots = {.model = "lowered",
                    .slots = 1,
                    .stages = {{.name = "target", .layout = *layout}},
                    .spec = spec.Digest(),
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    std::vector<en::GroupPlace> places(paged->artifact().groups().size(), en::GroupPlace::kDevice);
    config.weights.push_back(
        {.weights = std::move(weights), .places = std::move(places), .slabs = {}});
    config.setup_shapes = {prefill, StepOf(pl::Phase::kDecode, kPrompt.size(), 1)};
    config.launch = [&]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)});
    };
    config.weight_address = [&resource, paged](const llmp::ir::WeightRef& ref,
                                               llmp::format::PartRole) -> std::uint64_t {
      const auto at = resource.find(ref);
      return at == resource.end() ? 0 : paged->resource_address(at->second);
    };
    auto created = en::ModelRunner::Create(node, std::move(config), std::move(*component), 0, 0);
    if (!created) return std::unexpected(created.error());
    runner = std::move(*created);
    entered.push_back(runner.get());
    if (auto x = runner->Setup(); !x) return x;
    if (auto x =
            node.MapWorkspace(runner->activations_needed(),
                              std::max<std::uint64_t>(runner->pool_needed(), en::kPagedExtent));
        !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + (16ULL << 20U));
    if (auto x = node.Start(llmp::base::Bytes(fixed + paged->extents().size() * en::kPagedExtent +
                                              4 * node.StateCapacity()));
        !x)
      return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    if (auto x = runner->slots().SelectSlots(std::vector<std::uint32_t>{0}); !x) return x;
    return node.WithRequest(0, runner->slots().execution(), "lowered steps", [&]() -> en::Status {
      const auto step = [&](const pl::StepShape& shape,
                            std::span<const std::int32_t> ids) -> en::Status {
        const std::array<en::SegmentWork, 1> work{{{.tokens = ids, .bytes = {}}}};
        auto ran = runner->RunStep(
            {.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}});
        if (!ran) return std::unexpected(ran.error().what);
        if (ran->path == en::RunPath::kReplayed) ++out.replayed;
        const auto& bytes = ran->outputs.at(0);
        std::vector<float> logits(bytes.size() / sizeof(float));
        std::memcpy(logits.data(), bytes.data(), logits.size() * sizeof(float));
        out.rows.push_back(std::move(logits));
        return {};
      };
      if (auto x = step(prefill, kPrompt); !x) return x;
      for (std::uint32_t i = 0; i < tokens.size(); ++i) {
        const std::array<std::int32_t, 1> id{tokens[i]};
        if (auto x = step(StepOf(pl::Phase::kDecode, kPrompt.size() + i, 1), id); !x) return x;
      }
      return {};
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) {
    std::ignore = runner.release();  // its memory may still be in use
    std::ignore = backend.release();
  }
  runner.reset();
  return out;
}

struct Case {
  std::string name;
  std::string path;
};

class ComponentRunnerGpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentRunnerGpu, LoweredStepsAreTodaysBitForBit) {
  const Case& c = GetParam();
  const fs::path root = llmp::test_support::ModelsDir() + "/" + c.path;
  if (std::error_code error; !fs::exists(root, error)) GTEST_SKIP() << "no artifact in " << root;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "lowered";
  fs::create_directories(out);
  std::vector<std::vector<float>> production;
  std::vector<llmp::spec::SourceValue> values;
  std::string_view policy;
  if (c.name == "gemma2") {
    production = Production<en::Gemma2Runner>(en::Gemma2Options{.artifact = root,
                                                                .out = out,
                                                                .context = kContext,
                                                                .max_rows = kMaxRows,
                                                                .slots = 1,
                                                                .frontier_head = false,
                                                                .owner_decode = false,
                                                                .packed_prefill = false,
                                                                .shared_q8 = true,
                                                                .owner_prefill = false,
                                                                .flexible_owner_prefill = false,
                                                                .device_masks = true,
                                                                .bounded_roots = false,
                                                                .fuse_norms = true,
                                                                .fuse_quant_glu = true,
                                                                .fuse_norm_rope = false,
                                                                .fuse_norm_add = true});
    values = *llmp::spec::mappings::Gemma2V0Values(llmp::model::Gemma2_2B());
    policy =
        "max_decode_rows 16\ndisable grouped_stores\ndisable rms_norm_mul_rope\ndisable "
        "rope_store\n";
  } else {
    production = Production<en::Gemma3Runner>(en::Gemma3Options{.artifact = root,
                                                                .out = out,
                                                                .context = kContext,
                                                                .max_rows = kMaxRows,
                                                                .slots = 1,
                                                                .frontier_head = false,
                                                                .owner_decode = false,
                                                                .packed_prefill = false,
                                                                .shared_q8 = true,
                                                                .owner_prefill = false,
                                                                .flexible_owner_prefill = false,
                                                                .bounded_roots = false,
                                                                .device_masks = true,
                                                                .fuse_norms = true,
                                                                .fuse_quant_glu = true,
                                                                .fuse_norm_rope = true,
                                                                .fuse_norm_add = true,
                                                                .group_kv_stores = true});
    values = *llmp::spec::mappings::Gemma3V0Values(llmp::model::Gemma3_4BQat());
    policy = "max_decode_rows 16\ndisable rope_store\n";
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
  for (const bool device_masks : {true, false}) {
    const Component lowered = Lowered(root, *spec, policy, picked, out, device_masks);
    ASSERT_EQ(lowered.rows.size(), production.size());
    // The repeated decode shape: run once, captured beside its second run,
    // replayed on its third.
    EXPECT_GE(lowered.replayed, 1U);
    for (std::size_t i = 0; i < production.size(); ++i) {
      ASSERT_EQ(lowered.rows[i].size(), production[i].size()) << "step " << i;
      double worst = 0.0;
      std::size_t differ = 0;
      for (std::size_t j = 0; j < production[i].size(); ++j) {
        const double d = std::fabs(static_cast<double>(lowered.rows[i][j]) - production[i][j]);
        worst = std::max(worst, d);
        differ += std::memcmp(&lowered.rows[i][j], &production[i][j], sizeof(float)) != 0;
      }
      std::printf(
          "%s (%s masks) step %zu: %zu of %zu logits differ in their bits, the largest "
          "by %.6g\n",
          c.name.c_str(), device_masks ? "device" : "host", i, differ, production[i].size(), worst);
      EXPECT_EQ(differ, 0U) << "step " << i << (device_masks ? "" : ", host masks");
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Gemma, ComponentRunnerGpu,
    testing::Values(
        Case{.name = "gemma2",
             .path = "gemma2-import-20261007/artifacts/"
                     "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870"},
        Case{.name = "gemma3",
             .path = "gemma3-import-20261007/artifacts/"
                     "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb"}),
    [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
