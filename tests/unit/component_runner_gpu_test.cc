// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h) over the component step
// source (engine/component_source.h) and the GGML module's lowering
// (kernels/ggml/step_program.h), on the real Gemma 2 2B and Gemma 3 4B QAT
// published v1 artifacts (label `models`: their stored specs and tensors),
// under the serving recipe (the component source planning under the
// architecture mapping's policy, gated as the model's approval gates it). Two owners run
// one script of waves: joined prefill chunks of equal rows (128, 37 and 2:
// owner prefill and its flexible tiles; 20 over equal reads at different
// pasts, each owner's own mask), ragged and single-owner chunks (per-owner
// attention), joined one-row decodes returning logits and tokens chosen on
// the device (owner decode, bounded roots over unequal reads), state-only
// chunks, and one owner's chunks past its sliding layers' ring until it
// wraps, then joined decodes again. The repeated decode shape is captured
// on its second run and replayed after (D-090). Every output is finite,
// and a second runner over the same script gives every output bit for bit
// (the replays included). (The script was once compared output for output
// with the M3 runners' under their serving options, bit for bit, retired
// with them after the migration's gate.)

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
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/execution_contract.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/support.h"
#include "kernels/ggml/step_program.h"
#include "planner/step_shape.h"
#include "platform/host_probe.h"
#include "runtime/approvals.h"
#include "spec/mapping.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace fs = std::filesystem;

constexpr std::uint32_t kMaxRows = 128;
constexpr std::uint32_t kOwners = 2;

// ------------------------------------------------------------ the script

enum class Head : std::uint8_t { kLogits, kGreedy, kNone };
struct Unit {
  std::uint32_t slot = 0;
  std::uint32_t rows = 0;
};
struct Wave {
  std::vector<Unit> units;
  Head head = Head::kLogits;
};
Wave W(std::vector<Unit> units, Head head = Head::kLogits) {
  return {.units = std::move(units), .head = head};
}
// A wave as the runner takes it: each owner's past and tokens.
struct Planned {
  Wave wave;
  std::vector<std::uint32_t> past;
  std::vector<std::vector<std::int32_t>> tokens;
};

// Each owner's own token stream (teacher-forced: the runner's picks are
// compared, not fed back).
std::int32_t TokenAt(std::uint32_t slot, std::uint32_t position) {
  return static_cast<std::int32_t>(100 + (((position * 7919U) + (slot * 104729U) + 13U) % 20000U));
}

// `local_cells`: the sliding layers' ring, which owner 0 then wraps.
std::vector<Planned> Script(std::uint32_t local_cells) {
  std::vector<Wave> waves = {
      W({{0, 128}, {1, 128}}),  // owner prefill, 128 rows
      W({{0, 37}, {1, 37}}),    // flexible owner prefill
      W({{0, 2}, {1, 2}}),      // its smallest tile
      W({{0, 100}, {1, 5}}),    // ragged: per owner
      W({{0, 1}, {1, 1}}),      // owner decode, bounded roots
      W({{0, 1}, {1, 1}}, Head::kGreedy),
      W({{0, 1}}),
      W({{1, 3}}, Head::kNone),  // state-only
      W({{0, 1}, {1, 1}}, Head::kGreedy),
      W({{0, 1}, {1, 1}}),
      // Owner 1 near owner 0 (259 and 272), then owner prefill over equal
      // reads (512 cells each) with each owner's own mask.
      W({{1, 80}}, Head::kNone),
      W({{0, 20}, {1, 20}}),
  };
  std::array<std::uint32_t, kOwners> past{};
  for (const Wave& w : waves) {
    for (const Unit& u : w.units) past.at(u.slot) += u.rows;
  }
  // Owner 0 past its ring (a state-only chunk at a time), then a chunk
  // returning its logits.
  while (past[0] + kMaxRows <= local_cells + kMaxRows + 64) {
    waves.push_back(W({{0, kMaxRows}}, Head::kNone));
    past[0] += kMaxRows;
  }
  waves.push_back(W({{0, kMaxRows}}));
  // Joined decodes over one wrapped ring and one short cache, each head's
  // shape three times: captured beside its second run, replayed on its
  // third.
  for (const Head head : {Head::kLogits, Head::kGreedy}) {
    for (int i = 0; i < 3; ++i) waves.push_back(W({{0, 1}, {1, 1}}, head));
  }
  waves.push_back(W({{0, 9}, {1, 9}}));  // equal rows, unequal reads: per owner
  std::vector<Planned> out;
  past = {};
  for (const Wave& w : waves) {
    Planned p;
    p.wave = w;
    for (const Unit& u : w.units) {
      p.past.push_back(past.at(u.slot));
      std::vector<std::int32_t> ids;
      for (std::uint32_t i = 0; i < u.rows; ++i)
        ids.push_back(TokenAt(u.slot, past.at(u.slot) + i));
      p.tokens.push_back(std::move(ids));
      past.at(u.slot) += u.rows;
    }
    out.push_back(std::move(p));
  }
  return out;
}

pl::StepShape ShapeOf(const Planned& p, std::uint32_t context) {
  pl::StepShape shape;
  shape.phase = pl::Phase::kDecode;
  shape.read_align = 256;
  for (std::size_t i = 0; i < p.wave.units.size(); ++i) {
    const Unit& u = p.wave.units[i];
    if (u.rows > 1) shape.phase = pl::Phase::kPrefill;
    const std::uint32_t end = p.past[i] + u.rows;
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(u.slot),
                              .rows = u.rows,
                              .past = p.past[i],
                              .read_cells = std::min(context, (end + 255) / 256 * 256),
                              .outputs = p.wave.head == Head::kNone ? 0U : 1U,
                              .group = 0,
                              .want_head = p.wave.head != Head::kNone,
                              .greedy = p.wave.head == Head::kGreedy});
  }
  return shape;
}

// Each wave's outputs, by unit: a logits row's or a token's bytes (none
// for a state-only wave).
using Outputs = std::vector<std::vector<std::vector<std::byte>>>;

template <typename T>
std::vector<std::byte> Bytes(std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(), bytes.end()};
}

// ------------------------------------------------------------ the components

struct Component {
  Outputs outputs;
  std::uint64_t replayed = 0;
};

// The model runner over the component source (the mapping's policy) and
// the GGML lowering, for the same script.
Component Lowered(const fs::path& root, const llmp::spec::ModelSpec& spec,
                  const llmp::spec::Mapping& mapping,
                  std::span<const llmp::spec::SourceValue> values, std::uint32_t context,
                  std::span<const Planned> script, const fs::path& scratch) {
  Component out;
  auto artifact = ar::Artifact::Open(root, {.expected_id = {}});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return out;
  const auto source = llmp::spec::SourceOf(*artifact);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  if (!source) return out;
  const auto valid = co::ValidateSpec(spec, source->tensors);
  EXPECT_TRUE(valid.has_value()) << valid.error();
  auto layout = co::StageLayout(spec, 0, context, kMaxRows);
  EXPECT_TRUE(layout.has_value()) << layout.error();
  if (!layout) return out;
  std::vector<pl::StepShape> shapes;
  for (const Planned& p : script) {
    const pl::StepShape shape = ShapeOf(p, context);
    if (std::ranges::find(shapes, shape) == shapes.end()) shapes.push_back(shape);
  }
  // Each weight's resource, by what the steps bind (the head's included).
  std::map<llmp::ir::WeightRef, std::uint32_t> resource;
  {
    auto step =
        co::BuildStep(spec, 0, shapes.front(), {.tensors = source->tensors, .state = &*layout});
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
  kg::SelectedWorkspace selected_state{};
  std::vector<en::PagedModel*> entered;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    auto made = kg::GgmlBackend::Create(0, node.execution(), node.stream(0), nullptr, {});
    if (!made) return std::unexpected(made.error());
    backend = std::move(*made);
    // Native construction has a separate finite fixture allowance. Retained
    // plans still acquire the runner's ordinary paid ownership before transfer.
    const auto available = llmp::platform::AvailableMemoryBytes();
    if (!available) return std::unexpected("native construction physical headroom is unknown");
    auto paid = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), 128ULL << 20U,
                                                      *available, 6ULL << 30U);
    if (!paid) return std::unexpected(paid.error());
    if (auto funded =
            backend->FundConstruction([paid = *paid](std::uint64_t n) { return paid->Charge(n); },
                                      [paid = *paid](std::uint64_t n) { paid->Release(n); });
        !funded)
      return std::unexpected(funded.error());
    auto weights = std::make_unique<en::PagedWeights>();
    if (auto x = weights->Open(root); !x) return x;
    en::PagedWeights* paged = weights.get();
    auto component = en::ComponentStepSource::Create(
        {.spec = &spec,
         .stage = 0,
         .bindings = {.tensors = source->tensors, .state = &*layout},
         .bound_digest = spec.Digest(),
         .mapping = &mapping,
         .values = values,
         .policy = std::nullopt,
         .backend = backend->Backend()});
    if (!component) return std::unexpected(component.error());
    en::RunnerConfig config;
    config.slots = {.model = "lowered",
                    .slots = kOwners,
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
    config.setup_shapes = shapes;
    const auto selected_bytes = backend->persistent_bytes();
    if (selected_bytes)
      config.kernel_memory.push_back({.name = "selected kernel state", .bytes = selected_bytes});
    config.prepare = [&selected_state,
                      selected_bytes](std::span<const std::uint64_t> bases) -> en::Status {
      if (bases.size() != static_cast<std::size_t>(selected_bytes != 0))
        return std::unexpected("selected kernel state is not mapped");
      selected_state = {selected_bytes ? bases[0] : 0, selected_bytes};
      return {};
    };
    config.retire = [&] {
      backend->LaunchReleased();
      selected_state = {};
    };
    config.launch =
        [&](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)}, selected_state);
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
    if (auto x = runner->slots().SelectSlots(std::vector<std::uint32_t>{0, 1}); !x) return x;
    return node.WithRequest(0, runner->slots().execution(), "lowered steps", [&]() -> en::Status {
      for (const Planned& p : script) {
        std::vector<en::SegmentWork> work;
        for (const auto& ids : p.tokens) work.push_back({.tokens = ids, .bytes = {}});
        auto ran = runner->RunStep({.shape = ShapeOf(p, context),
                                    .work = work,
                                    .step_bytes = {},
                                    .advance = true,
                                    .ahead = {}});
        if (!ran) return std::unexpected(ran.error().what);
        if (ran->path == en::RunPath::kReplayed) ++out.replayed;
        auto& row = out.outputs.emplace_back();
        if (p.wave.head == Head::kNone) continue;
        for (const auto& bytes : ran->outputs) row.push_back(bytes);
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
  std::string name;  // the store's v1-ids.json name
};

// The sliding layers' window: the largest an attention component of the
// stored spec names.
std::uint32_t Window(const llmp::spec::ModelSpec& spec) {
  std::int64_t window = 0;
  for (const llmp::spec::Layer& layer : spec.stages.front().layers) {
    for (const llmp::spec::ComponentRef& c : layer.components) {
      const llmp::spec::Param* p = c.Find("window");
      if (p == nullptr) continue;
      if (const auto* v = std::get_if<std::int64_t>(&p->value)) window = std::max(window, *v);
    }
  }
  return static_cast<std::uint32_t>(window);
}

// The values serving plans the mapping's policy with: the gates the
// model's approval holds, as flags (runtime/component_admission.cc
// ApprovedRule).
std::vector<llmp::spec::SourceValue> ApprovedGates(const ar::Artifact& artifact) {
  std::vector<llmp::spec::SourceValue> gates;
  const auto* approval =
      llmp::runtime::FindApproval(artifact.model().architecture, artifact.sources());
  EXPECT_NE(approval, nullptr) << "no approval for the published artifact";
  if (approval == nullptr) return gates;
  for (const std::string_view gate : approval->gates)
    gates.push_back({.key = std::string(gate), .value = true});
  return gates;
}

class ComponentRunnerGpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentRunnerGpu, ServingRecipeScriptIsFiniteAndTheSameRunToRun) {
  const Case& c = GetParam();
  const auto id = llmp::test_support::V1IdNamed(c.name);
  if (!id) GTEST_FAIL() << "no published v1 " << c.name << " in " << llmp::test_support::V1Store();
  const fs::path root = fs::path(llmp::test_support::V1Store()) / *id;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "lowered";
  fs::create_directories(out);
  const bool gemma2 = c.name == "gemma2";
  // Two owners: Gemma 2 at its 8,192 context, Gemma 3 at its two-owner
  // serving bound (4,096). The sliding layers' ring: the window and a
  // chunk, padded to 256 cells.
  const std::uint32_t context = gemma2 ? 8192U : 4096U;
  auto artifact = ar::Artifact::Open(root, {.expected_id = {}});
  ASSERT_TRUE(artifact.has_value()) << artifact.error().ToString();
  const auto* mapping = llmp::spec::FindMapping(artifact->model().architecture);
  ASSERT_NE(mapping, nullptr);
  const auto spec = llmp::spec::StoredSpec(*artifact);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const std::vector<llmp::spec::SourceValue> values = ApprovedGates(*artifact);
  const std::uint32_t window = Window(*spec);
  ASSERT_GT(window, 0U);
  const std::uint32_t local_cells = (std::min(context, window + kMaxRows) + 255U) / 256U * 256U;
  const std::vector<Planned> script = Script(local_cells);
  const Component first = Lowered(root, *spec, *mapping, values, context, script, out);
  const Component second = Lowered(root, *spec, *mapping, values, context, script, out);
  ASSERT_EQ(first.outputs.size(), script.size());
  ASSERT_EQ(second.outputs.size(), script.size());
  // The repeated joined decodes: run once, captured beside the second
  // run, replayed on the third (each head's).
  EXPECT_GE(first.replayed, 2U);
  EXPECT_EQ(second.replayed, first.replayed);
  std::size_t compared = 0;
  for (std::size_t w = 0; w < script.size(); ++w) {
    const std::size_t outputs =
        script[w].wave.head == Head::kNone ? 0 : script[w].wave.units.size();
    ASSERT_EQ(first.outputs[w].size(), outputs) << "wave " << w;
    ASSERT_EQ(second.outputs[w].size(), outputs) << "wave " << w;
    for (std::size_t u = 0; u < outputs; ++u) {
      const auto& got = first.outputs[w][u];
      const auto& again = second.outputs[w][u];
      ASSERT_EQ(got.size(), again.size()) << "wave " << w << " owner " << u;
      EXPECT_EQ(std::memcmp(got.data(), again.data(), got.size()), 0)
          << "wave " << w << " owner " << u << ": another "
          << (script[w].wave.head == Head::kGreedy ? "token" : "row");
      if (script[w].wave.head == Head::kGreedy) {
        ASSERT_EQ(got.size(), sizeof(std::int32_t));
        std::int32_t token = -1;
        std::memcpy(&token, got.data(), sizeof(token));
        EXPECT_GE(token, 0) << "wave " << w << " owner " << u;
      } else {
        ASSERT_FALSE(got.empty());
        ASSERT_EQ(got.size() % sizeof(float), 0U);
        std::size_t finite = 0;
        for (std::size_t j = 0; j < got.size() / sizeof(float); ++j) {
          float v = 0;
          std::memcpy(&v, got.data() + (j * sizeof(float)), sizeof(float));
          finite += std::isfinite(v) ? 1 : 0;
        }
        EXPECT_EQ(finite, got.size() / sizeof(float)) << "wave " << w << " owner " << u;
      }
      ++compared;
    }
  }
  EXPECT_GT(compared, 0U);
  std::printf("%s: %zu waves, %zu owner outputs checked, %llu replayed\n", c.name.c_str(),
              script.size(), compared, static_cast<unsigned long long>(first.replayed));
}

INSTANTIATE_TEST_SUITE_P(Gemma, ComponentRunnerGpu,
                         testing::Values(Case{.name = "gemma2"}, Case{.name = "gemma3"}),
                         [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
